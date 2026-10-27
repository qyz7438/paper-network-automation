"""Fresh offline PDF-to-candidate benchmark for the frozen paper pilot.

Each stage has a durable cache; model/source failures remain explicit and are
not interpreted as absent papers or negative claims. No graph is published.
"""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import time

from .automatic_pilot import (EXCEPTIONS, OUTPUT, _render, collect_model_evidence, interpret_paper,
                              validate_draft, validate_model_evidence)
from .card_enrichment import SOURCE, BASELINE, validate_quote, verified_parses
from .core import Jobs, check_inputs, contained, object_hash, output_root, parse_key, parse_pdf, read_json, sha, write_json
from .local_models import DEFAULT_MODELS, LABELS, LocalModels
from .local_pipeline import extract_paper, retrieve, select_passages, stage, validate_result, write_vectors


COLD_OUTPUT = OUTPUT / "fresh-pdf-run"
VERSION = "fresh-pdf-automatic-pilot-v1"


def timed(produce):
    started = time.perf_counter()
    value = produce()
    return {"value": value, "stage_seconds": round(time.perf_counter() - started, 6)}


def validate_parsed(value: dict, paper: dict) -> None:
    parsed = value["value"]
    if parsed["paper_id"] != paper["id"] or parsed["pdf_sha256"] != paper["pdf_sha256"] or not parsed["pages"]:
        raise ValueError("fresh parse belongs to another source")
    for page in parsed["pages"]:
        for block in page["blocks"]:
            if page["text"][block["start"]:block["end"]] != block["text"]:
                raise ValueError("fresh parse block does not match page")


def _query(vault: Path, output: Path, source: Path, manifest: dict, rows: list[dict],
           parsed: dict, docs: list[dict], models: LocalModels, queue: Jobs, retry_failed: bool) -> tuple[list[dict], bool]:
    import numpy as np
    queries = [{"id": q["id"], "text": q["explicit_english"]}
               for q in read_json(source / "evaluation-plan.json")["retrieval"]]
    paper_map = {p["id"]: p for p in manifest["papers"]}
    if not docs:
        raise ValueError("no completed source passages available for fresh retrieval")
    key = object_hash([VERSION, docs, queries, models.identity, inspect.getsource(LocalModels.encode), inspect.getsource(retrieve)])
    vectors_path = output / "cache" / "fresh-vectors" / (key + ".npy")

    def produce():
        started = time.perf_counter()
        vectors = models.encode([doc["text"] for doc in docs], "passage")
        query_vectors = models.encode([q["text"] for q in queries], "query")
        write_vectors(vectors_path, vectors, output)
        result = [{"id": q["id"], **retrieve(q["text"], docs, vectors, vector, paper_map)}
                  for q, vector in zip(queries, query_vectors, strict=True)]
        return {"queries": result, "vectors": vectors_path.relative_to(output).as_posix(),
                "vectors_sha256": sha(vectors_path), "documents": len(docs),
                "stage_seconds": round(time.perf_counter() - started, 6)}

    def validate(value):
        path = contained(output, value["vectors"])
        if sha(path) != value["vectors_sha256"] or np.load(path, allow_pickle=False).shape != (len(docs), 384):
            raise ValueError("fresh retrieval vector cache changed")
        if [q["id"] for q in value["queries"]] != [q["id"] for q in queries]:
            raise ValueError("fresh query set changed")
        valid_ids = {r["paper_id"] for r in rows if r["state"] == "automatic_draft_needs_review"}
        for query in value["queries"]:
            for row in query["rows"]:
                if row["paper_id"] not in valid_ids:
                    raise ValueError("query returned failed or quarantined paper")
                pages = {p["page"]: p for p in parsed[row["paper_id"]]["pages"]}
                for passage in row["passages"]:
                    validate_quote(paper_map[row["paper_id"]], pages, {**passage, "quote": passage["text"]})

    value, cached = stage(queue, output, "cohort", "fresh_retrieval", key, produce, validate, retry_failed)
    return value, cached


def run(vault: Path, models: LocalModels, retry_failed: bool = False) -> dict:
    started = time.perf_counter()
    vault, output = vault.resolve(), output_root(vault, vault / COLD_OUTPUT)
    if output != (vault / COLD_OUTPUT).resolve():
        raise ValueError("dedicated fresh PDF output required")
    source, baseline = vault / SOURCE, vault / BASELINE
    manifest = read_json(source / "manifest.json")
    if len(manifest["papers"]) != 100 or len({p["id"] for p in manifest["papers"]}) != 100:
        raise ValueError("frozen 100-paper cohort required")
    if manifest != read_json(baseline / "manifest.json") or models.identity != read_json(source / "models.json"):
        raise ValueError("frozen source or model identity changed")
    check_inputs(vault, manifest)
    paper_map = {p["id"]: p for p in manifest["papers"]}
    exceptions = read_json(vault / EXCEPTIONS)["decisions"]
    quarantine = {}
    for item in exceptions:
        pid = item["paper_id"]
        if pid not in paper_map or pid in quarantine or item["state"] != "source_identity_mismatch" or item["source_sha256"] != paper_map[pid]["pdf_sha256"]:
            raise ValueError("invalid source exception")
        quarantine[pid] = item
    # Recheck the existing identity adjudication before excluding the known mismatched source.
    exception_parses, _ = verified_parses(baseline, [paper_map[pid] for pid in quarantine])
    for pid, item in quarantine.items():
        validate_quote(paper_map[pid], {p["page"]: p for p in exception_parses[pid]["pages"]}, item["evidence"])
    preflight_seconds = time.perf_counter() - started
    queue = Jobs(output / "fresh-state.sqlite", output)
    rows, parsed_by_id, docs, failures = [], {}, [], []
    counts, stage_seconds = Counter(), Counter()
    try:
        for paper in manifest["papers"]:
            pid = paper["id"]
            if pid in quarantine:
                rows.append({"paper_id": pid, "title": paper["title"], "card": paper["card"],
                             "source_sha256": paper["pdf_sha256"], "state": "source_quarantined",
                             "source_exception": quarantine[pid], "formal_graph_publication": False})
                counts["papers_quarantined"] += 1
                continue
            try:
                pkey = object_hash([VERSION, parse_key(paper), paper["id"], paper["title"]])
                parsed_result, cached = stage(queue, output, pid, "fresh_parse", pkey,
                    lambda p=paper: timed(lambda: parse_pdf(str(contained(vault, p["pdf"])), p["pdf_sha256"], p["title"], p["id"])),
                    lambda value, p=paper: validate_parsed(value, p), retry_failed)
                counts["parse_cached" if cached else "parse_computed"] += 1
                stage_seconds["parse"] += parsed_result["stage_seconds"]
                parsed = parsed_result["value"]
                selected = select_passages(paper, parsed, limit=12)
                ekey = object_hash([VERSION, pkey, selected, models.identity, 0.45, LABELS,
                                    inspect.getsource(extract_paper), sha(Path(__file__).with_name("local_models.py"))])
                record_result, cached = stage(queue, output, pid, "fresh_entities", ekey,
                    lambda p=paper, s=selected: timed(lambda: extract_paper(models, p, s, 0.45)),
                    lambda value, p=paper, d=parsed, s=selected: validate_result(value["value"], p, d, s), retry_failed)
                counts["entities_cached" if cached else "entities_computed"] += 1
                stage_seconds["entities"] += record_result["stage_seconds"]
                record = record_result["value"]
                mkey = object_hash([VERSION, ekey, models.identity, sha(Path(__file__).with_name("automatic_pilot.py")),
                                    sha(Path(__file__).with_name("automatic_roles.py"))])
                evidence_result, cached = stage(queue, output, pid, "fresh_role_model", mkey,
                    lambda p=paper, d=parsed, r=record: timed(lambda: collect_model_evidence(p, d, r, models)),
                    lambda value, p=paper, d=parsed, r=record: validate_model_evidence(value["value"], p, d, r), retry_failed)
                counts["role_model_cached" if cached else "role_model_computed"] += 1
                stage_seconds["role_model"] += evidence_result["stage_seconds"]
                dkey = object_hash([VERSION, mkey, sha(Path(__file__).with_name("automatic_roles.py")), inspect.getsource(interpret_paper)])
                draft_result, cached = stage(queue, output, pid, "fresh_draft", dkey,
                    lambda p=paper, r=record, e=evidence_result["value"]: timed(lambda: interpret_paper(p, r, e)),
                    lambda value, p=paper, d=parsed, r=record: validate_draft(value["value"], p, d, r), retry_failed)
                counts["draft_cached" if cached else "draft_computed"] += 1
                stage_seconds["draft"] += draft_result["stage_seconds"]
                row = draft_result["value"]
                rows.append(row)
                parsed_by_id[pid] = parsed
                docs.extend(selected["passages"])
                counts["person_candidates"] += len(row["person_candidates"])
                counts["valid_mention_candidates"] += len(row["mentions"])
                counts["rejected_entity_spans"] += len(record["rejected_occurrences"])
                counts["person_model_calls_total"] += row["stats"]["person_model_calls"]
                counts["relation_model_calls_total"] += row["stats"]["relation_model_calls"]
                for mention in row["mentions"]:
                    counts["role_" + mention["role"]["role"]] += 1
            except Exception as error:
                failure = {"paper_id": pid, "error_type": type(error).__name__, "error": str(error),
                           "state": "tool_or_runtime_failure_not_missing_paper"}
                failures.append(failure)
                rows.append({"paper_id": pid, "title": paper["title"], "card": paper["card"],
                             "source_sha256": paper["pdf_sha256"], "state": "pipeline_error", "failure": failure,
                             "formal_graph_publication": False})
                counts["paper_runtime_failures"] += 1
            if len(rows) % 10 == 0:
                print(json.dumps({"fresh_processed": len(rows), "total": 100, "failures": len(failures)}, ensure_ascii=True), flush=True)
        retrieval, retrieval_cached = _query(vault, output, source, manifest, rows, parsed_by_id, docs, models, queue, retry_failed)
        check_inputs(vault, manifest)
    finally:
        queue.close()
    counts["fresh_retrieval_documents"] = len(docs)
    value = {"schema": VERSION, "models": models.identity, "inputs": {
        "manifest_sha256": sha(source / "manifest.json"), "exceptions_sha256": sha(vault / EXCEPTIONS),
        "cold_pilot_sha256": sha(Path(__file__)), "automatic_pilot_sha256": sha(Path(__file__).with_name("automatic_pilot.py")),
        "automatic_roles_sha256": sha(Path(__file__).with_name("automatic_roles.py"))},
        "papers": rows, "queries": retrieval["queries"],
        "boundary": "Fresh original-PDF parsing, local entity/role models and retrieval; no manual source addenda or formal graph writes."}
    changed = write_json(output / "fresh-drafts.json", value, output)
    stable_counts = {key: value for key, value in counts.items() if not key.endswith("_cached") and not key.endswith("_computed")}
    if not failures:
        markdown_changed = _render(output, rows, retrieval["queries"], stable_counts)
    else:
        markdown_changed = False
    summary = {"schema": VERSION, "papers_in_frozen_cohort": len(rows), "counts": dict(counts), "failures": failures,
               "preflight_seconds": round(preflight_seconds, 3), "summed_per_paper_stage_seconds": dict(stage_seconds),
               "fresh_retrieval_stage_seconds": retrieval["stage_seconds"], "retrieval_cached": retrieval_cached,
               "new_entity_model_calls": models.extraction_calls, "new_embedding_batches": models.embedding_calls,
               "elapsed_seconds": round(time.perf_counter() - started, 3), "drafts_changed": changed,
               "markdown_changed": markdown_changed, "formal_graph_publication": False, "all100_usable": False,
               "boundary": "Fresh source and model work for eligible papers; one known identity mismatch remains quarantined."}
    write_json(output / "fresh-run.json", summary, output)
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
