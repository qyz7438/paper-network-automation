"""Extend the frozen pilot with full-document cue windows and fresh local models.

Existing PDF parsing, initial entity extraction, model evidence and passage
vectors are reused. New candidates and rankings remain unreviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sqlite3
import time

from .automatic_pilot import (EXCEPTIONS, MAX_SENTENCE_CHARS, MODEL_TRIGGER, OUTPUT, RELATION_SCHEMA,
                              _load_extractor, _render, _safe_value, interpret_paper, validate_draft,
                              validate_model_evidence)
from .card_enrichment import SOURCE, source_sentence, validate_quote
from .core import Jobs, check_inputs, contained, object_hash, output_root, read_json, sha, write_json
from .fulltext_cues import cue_windows
from .local_models import DEFAULT_MODELS, LABELS, LocalModels
from .local_pipeline import extract_paper, retrieve, stage, validate_result, write_vectors
from .reclassify_fresh import load_completed
from .relation_trial import span_issues


FRESH = OUTPUT / "fresh-pdf-run"
FULLTEXT = OUTPUT / "fulltext-cue-run"
VERSION = "fulltext-cue-automatic-pilot-v1"


def merge_records(base: dict, extra: dict) -> dict:
    if base["paper_id"] != extra["paper_id"] or base["source_sha256"] != extra["source_sha256"]:
        raise ValueError("fulltext and original candidates belong to different sources")
    passages = list(base["selection"]["passages"])
    known_passages = {p["id"] for p in passages}
    for passage in extra["selection"]["passages"]:
        if passage["id"] in known_passages:
            raise ValueError("duplicate source passage identity")
        known_passages.add(passage["id"])
        passages.append(passage)
    occurrences = {item["id"]: item for item in base["occurrences"]}
    for item in extra["occurrences"]:
        if item["id"] in occurrences:
            old = occurrences[item["id"]]
            if any(old[key] != item[key] for key in ("paper_id", "type", "text", "evidence")):
                raise ValueError("duplicate mention identity has different source content")
            continue
        occurrences[item["id"]] = item
    return {"schema": "combined-original-and-fulltext-candidates-v1", "paper_id": base["paper_id"],
            "source_sha256": base["source_sha256"], "selection": {"passages": passages,
            "scope": "original_selected_passages_plus_additional_cue_windows_not_full_semantic_read"},
            "occurrences": list(occurrences.values()),
            "rejected_occurrences": base.get("rejected_occurrences", []) + extra.get("rejected_occurrences", [])}


def expand_model_evidence(paper: dict, parsed: dict, base_record: dict, combined: dict,
                          base_evidence: dict, models: LocalModels) -> dict:
    validate_model_evidence(base_evidence, paper, parsed, base_record)
    validate_result(combined, paper, parsed, combined["selection"])
    pages = {page["page"]: page for page in parsed["pages"]}
    original = {item["id"]: item for item in base_record["occurrences"]}
    groups = {}
    for item in deepcopy(base_evidence["sentences"]):
        ev = item["evidence"]
        groups[(ev["page"], ev["start"], ev["end"])] = item
    for mention in combined["occurrences"]:
        if mention["id"] in original:
            continue
        ev = mention["evidence"]
        sentence = source_sentence(pages[ev["page"]], ev["start"], ev["end"])
        key = (sentence["page"], sentence["start"], sentence["end"])
        if key not in groups:
            sentence_evidence = {**sentence, "pdf": paper["pdf"], "pdf_sha256": paper["pdf_sha256"]}
            validate_quote(paper, pages, sentence_evidence)
            groups[key] = {"evidence": sentence_evidence, "mention_ids": [], "raw_relation_output": None}
        groups[key]["mention_ids"].append(mention["id"])
    by_id = {item["id"]: item for item in combined["occurrences"]}
    rejected, calls, skipped_long = list(base_evidence["rejected_relation_outputs"]), 0, 0
    for group in groups.values():
        text = group["evidence"]["quote"]
        eligible = any(by_id[mid]["type"] == "method" for mid in group["mention_ids"]) and bool(MODEL_TRIGGER.search(text))
        originally_eligible = (any(mid in original and original[mid]["type"] == "method" for mid in group["mention_ids"])
                               and bool(MODEL_TRIGGER.search(text)) and len(text) <= MAX_SENTENCE_CHARS)
        if not eligible or originally_eligible:
            continue
        if len(text) > MAX_SENTENCE_CHARS:
            skipped_long += 1
            continue
        with models.torch.inference_mode():
            raw = _load_extractor(models).extract_relations(text, RELATION_SCHEMA, threshold=0.5,
                                                           include_confidence=True, include_spans=True)
        issues = span_issues(raw, text)
        if issues:
            rejected.append({"sentence": group["evidence"], "raw": _safe_value(raw), "issues": _safe_value(issues),
                             "state": "model_output_rejected_not_absent_relation"})
        else:
            group["raw_relation_output"] = raw
        calls += 1
    result = {**base_evidence, "sentences": list(groups.values()), "rejected_relation_outputs": rejected,
              "stats": {**base_evidence["stats"], "additional_relation_model_calls": calls,
                        "additional_long_sentences_not_sent": skipped_long}}
    validate_model_evidence(result, paper, parsed, combined)
    return result


def _fresh_retrieval_record(db, fresh: Path) -> dict:
    rows = db.execute("SELECT result,result_hash,input_hash FROM jobs WHERE paper_id=? AND stage=? AND state='done'",
                      ("cohort", "fresh_retrieval")).fetchall()
    if len(rows) != 1:
        raise ValueError("exactly one completed fresh retrieval record required")
    path = contained(fresh, rows[0][0])
    if sha(path) != rows[0][1]:
        raise ValueError("fresh retrieval receipt changed")
    return read_json(path), rows[0][2]


def run(vault: Path, models: LocalModels, retry_failed: bool = False) -> dict:
    import numpy as np
    started = time.perf_counter()
    vault = vault.resolve()
    output, fresh, source = output_root(vault, vault / FULLTEXT), output_root(vault, vault / FRESH), vault / SOURCE
    if output != (vault / FULLTEXT).resolve() or fresh == output:
        raise ValueError("dedicated fulltext output required")
    manifest, original = read_json(source / "manifest.json"), read_json(fresh / "fresh-drafts.json")
    if len(manifest["papers"]) != 100 or len({p["id"] for p in manifest["papers"]}) != 100:
        raise ValueError("frozen 100-paper cohort required")
    if (original["inputs"]["manifest_sha256"] != sha(source / "manifest.json")
            or original["inputs"]["exceptions_sha256"] != sha(vault / EXCEPTIONS)
            or models.identity != original["models"]):
        raise ValueError("fresh source or model identity changed")
    check_inputs(vault, manifest)
    papers = {p["id"]: p for p in manifest["papers"]}
    old_rows = {r["paper_id"]: r for r in original["papers"]}
    if set(old_rows) != set(papers):
        raise ValueError("fresh source cohort changed")
    queue = Jobs(output / "fulltext-state.sqlite", output)
    rows, parsed_map, base_docs, extra_docs, errors = [], {}, [], [], []
    counts = Counter()
    queries = [{"id": q["id"], "text": q["explicit_english"]} for q in read_json(source / "evaluation-plan.json")["retrieval"]]
    try:
        with closing(sqlite3.connect((fresh / "fresh-state.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
            fresh_index, fresh_query_key = _fresh_retrieval_record(db, fresh)
            old_vector_path = contained(fresh, fresh_index["vectors"])
            if sha(old_vector_path) != fresh_index["vectors_sha256"]:
                raise ValueError("fresh source vectors changed")
            old_vectors = np.load(old_vector_path, allow_pickle=False)
            for paper in manifest["papers"]:
                pid = paper["id"]
                old = old_rows[pid]
                if old["state"] != "automatic_draft_needs_review":
                    rows.append(old)
                    counts[old["state"]] += 1
                    continue
                try:
                    parsed = load_completed(db, fresh, pid, "fresh_parse")
                    base = load_completed(db, fresh, pid, "fresh_entities")
                    base_evidence = load_completed(db, fresh, pid, "fresh_role_model")
                    validate_result(base, paper, parsed, base["selection"])
                    validate_model_evidence(base_evidence, paper, parsed, base)
                    extra_selection = cue_windows(paper, parsed, base["selection"])
                    ekey = object_hash([VERSION, paper, extra_selection, models.identity, LABELS, 0.45,
                                        sha(Path(__file__).with_name("fulltext_cues.py")), inspect.getsource(extract_paper)])
                    extra, cached = stage(queue, output, pid, "fulltext_entities", ekey,
                        lambda p=paper, s=extra_selection: extract_paper(models, p, s, 0.45),
                        lambda value, p=paper, d=parsed, s=extra_selection: validate_result(value, p, d, s), retry_failed)
                    counts["extra_entities_cached" if cached else "extra_entities_computed"] += 1
                    combined = merge_records(base, extra)
                    validate_result(combined, paper, parsed, combined["selection"])
                    mkey = object_hash([VERSION, ekey, object_hash(base_evidence), models.identity,
                                        MODEL_TRIGGER.pattern, MODEL_TRIGGER.flags, RELATION_SCHEMA,
                                        inspect.getsource(expand_model_evidence)])
                    evidence, cached = stage(queue, output, pid, "fulltext_role_model", mkey,
                        lambda p=paper, d=parsed, b=base, c=combined, e=base_evidence: expand_model_evidence(p, d, b, c, e, models),
                        lambda value, p=paper, d=parsed, c=combined: validate_model_evidence(value, p, d, c), retry_failed)
                    counts["role_model_cached" if cached else "role_model_computed"] += 1
                    if not cached:
                        counts["additional_relation_model_calls"] += evidence["stats"]["additional_relation_model_calls"]
                    dkey = object_hash([VERSION, mkey, sha(Path(__file__).with_name("automatic_roles.py")), inspect.getsource(interpret_paper)])
                    row, cached = stage(queue, output, pid, "fulltext_draft", dkey,
                        lambda p=paper, c=combined, e=evidence: interpret_paper(p, c, e),
                        lambda value, p=paper, d=parsed, c=combined: validate_draft(value, p, d, c), retry_failed)
                    counts["draft_cached" if cached else "draft_computed"] += 1
                    rows.append(row)
                    parsed_map[pid] = parsed
                    base_docs.extend(base["selection"]["passages"])
                    extra_docs.extend(extra_selection["passages"])
                    counts["additional_cue_windows"] += len(extra_selection["passages"])
                    counts["additional_entity_mentions"] += len(extra["occurrences"])
                    counts["rejected_extra_entity_spans"] += len(extra.get("rejected_occurrences", []))
                    for mention in row["mentions"]:
                        counts["role_" + mention["role"]["role"]] += 1
                except Exception as error:
                    failure = {"paper_id": pid, "error_type": type(error).__name__, "error": str(error),
                               "state": "tool_or_runtime_failure_not_missing_paper"}
                    errors.append(failure)
                    rows.append({"paper_id": pid, "title": paper["title"], "card": paper["card"],
                                 "source_sha256": paper["pdf_sha256"], "state": "pipeline_error", "failure": failure,
                                 "formal_graph_publication": False})
                if len(rows) % 10 == 0:
                    print(json.dumps({"fulltext_processed": len(rows), "total": 100, "errors": len(errors)}, ensure_ascii=True), flush=True)
        expected_fresh_key = object_hash([original["schema"], base_docs, queries, models.identity,
                                          inspect.getsource(LocalModels.encode), inspect.getsource(retrieve)])
        if (len(base_docs) != fresh_index["documents"] or old_vectors.shape != (len(base_docs), 384)
                or expected_fresh_key != fresh_query_key):
            write_json(output / "last-error.json", {"state": "retrieval_withheld_source_order_or_paper_failure",
                       "errors": errors, "completed_rows": len(rows), "full_corpus_unlocked": False}, output)
            raise ValueError("fresh passage vector identity/order differs or some paper failed; retrieval withheld")
        qkey = object_hash([VERSION, fresh_index["vectors_sha256"], extra_docs, queries, models.identity,
                            inspect.getsource(LocalModels.encode), inspect.getsource(retrieve)])
        vector_path = output / "cache" / "expanded-vectors" / (qkey + ".npy")

        def make_queries():
            extra_vectors = models.encode([d["text"] for d in extra_docs], "passage")
            vectors = np.concatenate([old_vectors, extra_vectors], axis=0)
            query_vectors = models.encode([q["text"] for q in queries], "query")
            write_vectors(vector_path, vectors, output)
            docs = base_docs + extra_docs
            result = [{"id": q["id"], **retrieve(q["text"], docs, vectors, vector, papers)}
                      for q, vector in zip(queries, query_vectors, strict=True)]
            return {"queries": result, "vectors": vector_path.relative_to(output).as_posix(),
                    "vectors_sha256": sha(vector_path), "documents": len(docs)}

        def validate_queries(value):
            path = contained(output, value["vectors"])
            if sha(path) != value["vectors_sha256"] or np.load(path, allow_pickle=False).shape != (len(base_docs) + len(extra_docs), 384):
                raise ValueError("expanded vector index changed")
            if [q["id"] for q in value["queries"]] != [q["id"] for q in queries]:
                raise ValueError("expanded query set changed")
            for query in value["queries"]:
                for row in query["rows"]:
                    pages = {page["page"]: page for page in parsed_map[row["paper_id"]]["pages"]}
                    for passage in row["passages"]:
                        validate_quote(papers[row["paper_id"]], pages, {**passage, "quote": passage["text"]})

        query_result, qcached = stage(queue, output, "cohort", "fulltext_retrieval", qkey, make_queries, validate_queries, retry_failed)
        check_inputs(vault, manifest)
    finally:
        queue.close()
    value = {"schema": VERSION, "models": models.identity, "inputs": {
        "manifest_sha256": sha(source / "manifest.json"), "exceptions_sha256": sha(vault / EXCEPTIONS),
        "fresh_drafts_sha256": sha(fresh / "fresh-drafts.json"), "fulltext_pilot_sha256": sha(Path(__file__)),
        "fulltext_cues_sha256": sha(Path(__file__).with_name("fulltext_cues.py")),
        "automatic_roles_sha256": sha(Path(__file__).with_name("automatic_roles.py"))},
        "papers": rows, "queries": query_result["queries"],
        "boundary": "Original PDF plus additional cue-selected windows, local models and existing source vectors; candidates only."}
    changed = write_json(output / "fulltext-drafts.json", value, output)
    stable = {k: v for k, v in counts.items() if not k.endswith("_cached") and not k.endswith("_computed") and k != "additional_relation_model_calls"}
    markdown_changed = _render(output, rows, query_result["queries"], stable) if not errors else False
    summary = {"schema": VERSION, "papers_in_frozen_cohort": len(rows), "counts": dict(counts), "errors": errors,
               "new_entity_model_calls": models.extraction_calls, "new_embedding_batches": models.embedding_calls,
               "retrieval_cached": qcached, "elapsed_seconds": round(time.perf_counter() - started, 3),
               "drafts_changed": changed, "markdown_changed": markdown_changed,
               "formal_graph_publication": False, "all100_usable": False,
               "boundary": "Incremental full-text cue work; fresh PDF parse and initial model/index timing are in the preceding receipt."}
    write_json(output / "fulltext-run.json", summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.vault, LocalModels(args.models, 4), args.retry_failed), ensure_ascii=True))


if __name__ == "__main__":
    main()
