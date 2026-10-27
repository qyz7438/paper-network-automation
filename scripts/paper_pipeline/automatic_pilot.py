"""Build source-bound automatic drafts for the frozen 100-paper pilot.

No reviewed author/content addenda are read by this prediction path. The
output is a review workspace, not a graph publisher or a complete-card claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import math
from pathlib import Path
import time

from .automatic_roles import MODEL_TRIGGER, RELATION_SCHEMA, classify_mention
from .card_enrichment import BASELINE, SOURCE, author_evidence, source_sentence, validate_quote, verified_parses
from .core import Jobs, check_inputs, contained, object_hash, output_root, read_json, sha, write_json
from .local_models import DEFAULT_MODELS, LocalModels
from .local_pipeline import retrieve, stage, validate_result
from .relation_trial import span_issues
from .render import md_text, write_text


OUTPUT = Path("_working/paper-network-automation-v1/automatic-repair-v1")
EXCEPTIONS = Path("_working/paper-network-automation-v1/pilot100-card-enrichment-v1/source-exceptions.json")
VERSION = "automatic-source-drafts-v1"
MAX_SENTENCE_CHARS = 1200


def _load_extractor(models: LocalModels):
    if models.extractor is None:
        from gliner2 import AutoExtractor
        path = contained(models.root, models.manifest["models"]["extractor"]["directory"])
        models.extractor = AutoExtractor.from_pretrained(str(path), map_location="cpu", local_files_only=True).eval()
        if getattr(models.extractor.config, "architecture", None) != "boundary":
            raise ValueError("frozen boundary extractor required")
    return models.extractor


def _safe_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return value


def _person_candidates(paper: dict, page: dict, header_end: int, models: LocalModels) -> tuple[list[dict], list[dict]]:
    header = page["text"][:header_end]
    if not header.strip():
        return [], []
    extractor = _load_extractor(models)
    with models.torch.inference_mode():
        raw = extractor.extract_entities(header, ["person"], threshold=0.5, include_spans=True, include_confidence=True)
    entities = raw.get("entities")
    if not isinstance(entities, dict) or set(entities) != {"person"} or not isinstance(entities["person"], list):
        raise ValueError("unexpected person extraction result")
    result, rejected = [], []
    for item in entities["person"]:
        if not isinstance(item, dict) or not all(key in item for key in ("text", "start", "end", "confidence")) or span_issues(item, header):
            rejected.append({"raw": _safe_value(item), "reason": "invalid_person_candidate_span_or_score",
                             "state": "model_output_rejected_not_missing_author"})
            continue
        evidence = {"pdf": paper["pdf"], "pdf_sha256": paper["pdf_sha256"], "page": page["page"],
                    "start": item["start"], "end": item["end"], "quote": item["text"]}
        validate_quote(paper, {page["page"]: page}, evidence)
        result.append({"name_candidate": item["text"], "confidence": item["confidence"], "evidence": evidence,
                       "flags": (["multiline_or_fragment"] if "\n" in item["text"] else []) +
                                (["possible_email"] if "@" in item["text"] else []),
                       "state": "person_mention_not_verified_author_or_byline_order"})
    return result, rejected


def collect_model_evidence(paper: dict, parsed: dict, record: dict, models: LocalModels) -> dict:
    validate_result(record, paper, parsed, record["selection"])
    pages = {page["page"]: page for page in parsed["pages"]}
    if 1 not in pages:
        raise ValueError("first source page unavailable")
    crosscheck = author_evidence(paper, pages[1])
    people, rejected_people = _person_candidates(paper, pages[1], crosscheck["header_end"], models)
    grouped = {}
    for mention in record["occurrences"]:
        evidence = mention["evidence"]
        sentence = source_sentence(pages[evidence["page"]], evidence["start"], evidence["end"])
        key = (sentence["page"], sentence["start"], sentence["end"])
        grouped.setdefault(key, {"sentence": sentence, "mentions": []})["mentions"].append(mention)
    sentences, rejected_relations, calls, skipped_long = [], [], 0, 0
    for group in grouped.values():
        sentence = group["sentence"]
        text = sentence["quote"]
        raw = None
        eligible = any(m["type"] == "method" for m in group["mentions"]) and bool(MODEL_TRIGGER.search(text))
        if eligible and len(text) <= MAX_SENTENCE_CHARS:
            with models.torch.inference_mode():
                raw = _load_extractor(models).extract_relations(text, RELATION_SCHEMA, threshold=0.5,
                                                               include_confidence=True, include_spans=True)
            issues = span_issues(raw, text)
            if issues:
                rejected_relations.append({"sentence": sentence, "raw": _safe_value(raw),
                                           "issues": _safe_value(issues),
                                           "state": "model_output_rejected_not_absent_relation"})
                raw = None
            calls += 1
        elif eligible:
            skipped_long += 1
        sentence_evidence = {**sentence, "pdf": paper["pdf"], "pdf_sha256": paper["pdf_sha256"]}
        validate_quote(paper, pages, sentence_evidence)
        sentences.append({"evidence": sentence_evidence, "mention_ids": [m["id"] for m in group["mentions"]],
                          "raw_relation_output": raw})
    return {"paper_id": paper["id"], "title": paper["title"], "card": paper["card"],
            "source_sha256": paper["pdf_sha256"], "source_state": "no_known_identity_exception",
            "author_crosscheck": crosscheck, "person_candidates": people, "rejected_person_candidates": rejected_people,
            "sentences": sentences, "rejected_relation_outputs": rejected_relations,
            "stats": {"relation_model_calls": calls, "person_model_calls": int(bool(pages[1]["text"][:crosscheck["header_end"]].strip())),
                      "long_sentences_not_sent_to_relation_model": skipped_long}}


def validate_model_evidence(value: dict, paper: dict, parsed: dict, record: dict) -> None:
    if value["paper_id"] != paper["id"] or value["source_sha256"] != paper["pdf_sha256"]:
        raise ValueError("model evidence source identity mismatch")
    pages = {page["page"]: page for page in parsed["pages"]}
    validate_result(record, paper, parsed, record["selection"])
    by_id = {item["id"]: item for item in record["occurrences"]}
    ids = []
    for sentence in value["sentences"]:
        ev = sentence["evidence"]
        validate_quote(paper, pages, ev)
        if sentence["raw_relation_output"] is not None and span_issues(sentence["raw_relation_output"], ev["quote"]):
            raise ValueError("cached relation output differs from source sentence")
        for identifier in sentence["mention_ids"]:
            mention = by_id[identifier]["evidence"]
            if mention["page"] != ev["page"] or not ev["start"] <= mention["start"] < mention["end"] <= ev["end"]:
                raise ValueError("cached sentence does not contain its original mention")
        ids.extend(sentence["mention_ids"])
    expected = {m["id"] for m in record["occurrences"]}
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("model evidence lost or duplicated original mentions")
    for item in value["person_candidates"]:
        validate_quote(paper, pages, item["evidence"])
    for item in value["author_crosscheck"]["names"]:
        if item["evidence"]:
            validate_quote(paper, pages, item["evidence"])


def interpret_paper(paper: dict, record: dict, evidence: dict) -> dict:
    mentions = {item["id"]: item for item in record["occurrences"]}
    predictions = []
    for sentence in evidence["sentences"]:
        source = sentence["evidence"]
        for identifier in sentence["mention_ids"]:
            mention = mentions[identifier]
            ev = mention["evidence"]
            role = classify_mention(mention["type"], source["quote"], ev["start"] - source["start"],
                                    ev["end"] - source["start"], sentence["raw_relation_output"])
            predictions.append({"mention_id": identifier, "type": mention["type"], "text": mention["text"],
                                "mention_evidence": ev, "sentence_evidence": source,
                                "role": role, "identity_state": "unresolved", "canonical_assignment": None,
                                "state": "candidate_not_semantically_accepted"})
    return {key: value for key, value in evidence.items() if key != "sentences"} | {
            "mentions": predictions,
            "state": "automatic_draft_needs_review", "formal_graph_publication": False}


def predict_paper(paper: dict, parsed: dict, record: dict, models: LocalModels) -> dict:
    evidence = collect_model_evidence(paper, parsed, record, models)
    validate_model_evidence(evidence, paper, parsed, record)
    return interpret_paper(paper, record, evidence)


def validate_draft(row: dict, paper: dict, parsed: dict, record: dict) -> None:
    if row["paper_id"] != paper["id"] or row["source_sha256"] != paper["pdf_sha256"]:
        raise ValueError("draft/source identity mismatch")
    pages = {page["page"]: page for page in parsed["pages"]}
    expected = {m["id"] for m in record["occurrences"]}
    by_id = {item["id"]: item for item in record["occurrences"]}
    mentions = row["mentions"]
    if len(mentions) != len(expected) or {m["mention_id"] for m in mentions} != expected:
        raise ValueError("draft lost or duplicated original mentions")
    for item in mentions:
        original = by_id[item["mention_id"]]
        if (item["type"] != original["type"] or item["text"] != original["text"]
                or item["mention_evidence"] != original["evidence"]):
            raise ValueError("draft mention differs from the original candidate")
        validate_quote(paper, pages, item["mention_evidence"])
        validate_quote(paper, pages, item["sentence_evidence"])
        if item["canonical_assignment"] is not None or item["state"] != "candidate_not_semantically_accepted":
            raise ValueError("candidate cannot establish an identity or accepted graph edge")
    for item in row["person_candidates"]:
        validate_quote(paper, pages, item["evidence"])
    for item in row["author_crosscheck"]["names"]:
        if item["evidence"]:
            validate_quote(paper, pages, item["evidence"])


def _make_queries(source: Path, output: Path, manifest: dict, parsed: dict, quarantine: set[str],
                  models: LocalModels, queue: Jobs) -> tuple[list[dict], bool]:
    import numpy as np
    index, plan = read_json(source / "index.json"), read_json(source / "evaluation-plan.json")
    path = contained(source, index["vectors"])
    if sha(path) != index["sha256"]:
        raise ValueError("frozen retrieval vector index changed")
    vectors = np.load(path, allow_pickle=False)
    if vectors.shape != (index["rows"], index["dimension"]) or not np.isfinite(vectors).all():
        raise ValueError("invalid retrieval vector index")
    docs = index["documents"]
    included = [i for i, doc in enumerate(docs) if doc["paper_id"] not in quarantine]
    queries = [{"id": q["id"], "text": q["explicit_english"]} for q in plan["retrieval"]]
    papers = {paper["id"]: paper for paper in manifest["papers"]}
    key = object_hash([VERSION, queries, index["sha256"], models.identity, sorted(quarantine),
                       inspect.getsource(retrieve), inspect.getsource(LocalModels.encode)])

    def produce():
        encoded = models.encode([q["text"] for q in queries], "query")
        rows = []
        for q, vector in zip(queries, encoded, strict=True):
            result = retrieve(q["text"], [docs[i] for i in included], vectors[included], vector, papers)
            rows.append({"id": q["id"], **result})
        return {"queries": rows}

    def validate(value):
        if [q["id"] for q in value["queries"]] != [q["id"] for q in queries]:
            raise ValueError("cached retrieval queries changed")
        for query in value["queries"]:
            for row in query["rows"]:
                if row["paper_id"] in quarantine:
                    raise ValueError("quarantined source leaked into retrieval")
                pages = {page["page"]: page for page in parsed[row["paper_id"]]["pages"]}
                for passage in row["passages"]:
                    validate_quote(papers[row["paper_id"]], pages, {**passage, "quote": passage["text"]})

    value, cached = stage(queue, output, "cohort", "automatic_retrieval", key, produce, validate)
    return value["queries"], cached


def _render(output: Path, rows: list[dict], queries: list[dict], counts: dict) -> bool:
    lines = ["# 自动论文候选：冻结 100 篇试验", "",
             "这些是程序候选，不是已核验作者列表、正式卡片或图谱关系。未解决内容保留待审。", "",
             "| 状态 | 数量 |", "| --- | --- |"]
    lines.extend(f"| {md_text(str(key))} | {value} |" for key, value in sorted(counts.items()))
    for query in queries:
        lines += ["", "## " + md_text(query["query"]), "", "| 论文 | 原文段落 |", "| --- | --- |"]
        for item in query["rows"]:
            passage = item["passages"][0]
            lines.append("| [[" + item["card"] + "|" + md_text(item["title"]) + "]] | [[" + passage["pdf"] +
                         "#page=" + str(passage["page"]) + "|原文 p." + str(passage["page"]) + "]] " +
                         md_text(passage["text"]) + " |")
    for row in rows:
        lines += ["", "## " + md_text(row["title"]), "", "[[" + row["card"] + "|原卡]]", ""]
        if row["state"] == "source_quarantined":
            lines += ["来源身份冲突，未将所附原文内容归给此卡。"]
            continue
        lines += ["### 作者候选", "", "已有姓名仅做首文字面回指；人物模型输出不代表完整署名或顺序。"]
        lines += ["- " + md_text(p["name_candidate"]) + "（待核验）" for p in row["person_candidates"]]
        lines += ["", "### 方法、问题、任务与数据集候选", ""]
        for item in row["mentions"]:
            if item["role"]["role"] == "unresolved":
                continue
            ev = item["mention_evidence"]
            lines.append("- " + md_text(item["type"]) + " / " + md_text(item["role"]["role"]) + ": " +
                         md_text(item["text"]) + " — [[" + ev["pdf"] + "#page=" + str(ev["page"]) + "|原文]]（待审）")
        lines += ["", "未展示的提及及角色冲突保存在 automatic-drafts.json 中，均未自动入图。"]
    return write_text(output / "automatic-drafts.md", "\n".join(lines) + "\n", output)


def run(vault: Path, output: Path, models: LocalModels) -> dict:
    started = time.perf_counter()
    vault, output = vault.resolve(), output_root(vault, output)
    if output != (vault / OUTPUT).resolve():
        raise ValueError("dedicated automatic pilot output required")
    source, baseline = vault / SOURCE, vault / BASELINE
    manifest, records = read_json(source / "manifest.json"), read_json(source / "records.json")
    if manifest != read_json(baseline / "manifest.json"):
        raise ValueError("frozen source and parser manifests differ")
    papers = {p["id"]: p for p in manifest["papers"]}
    record_map = {r["paper_id"]: r for r in records}
    if len(papers) != 100 or len(records) != 100 or set(papers) != set(record_map):
        raise ValueError("exactly the frozen 100 distinct papers are required")
    if models.identity != read_json(source / "models.json"):
        raise ValueError("local model identity differs from frozen trial")
    check_inputs(vault, manifest)
    parsed, parse_hashes = verified_parses(baseline, manifest["papers"])
    exceptions = read_json(vault / EXCEPTIONS)["decisions"]
    quarantine = {}
    for item in exceptions:
        pid = item["paper_id"]
        if pid in quarantine or pid not in papers or item["state"] != "source_identity_mismatch" or item["source_sha256"] != papers[pid]["pdf_sha256"]:
            raise ValueError("invalid source identity quarantine")
        validate_quote(papers[pid], {p["page"]: p for p in parsed[pid]["pages"]}, item["evidence"])
        quarantine[pid] = item
    preflight_seconds = time.perf_counter() - started
    queue = Jobs(output / "automatic-state.sqlite", output)
    rows, counts = [], Counter()
    try:
        for paper in manifest["papers"]:
            pid = paper["id"]
            if pid in quarantine:
                rows.append({"paper_id": pid, "title": paper["title"], "card": paper["card"],
                             "source_sha256": paper["pdf_sha256"], "state": "source_quarantined",
                             "source_exception": quarantine[pid], "formal_graph_publication": False})
                counts["papers_quarantined"] += 1
                continue
            record = record_map[pid]
            model_key = object_hash([VERSION, paper, record, parse_hashes[pid], models.identity, RELATION_SCHEMA,
                                     MODEL_TRIGGER.pattern, MODEL_TRIGGER.flags, MAX_SENTENCE_CHARS,
                                     sha(Path(__file__).with_name("card_enrichment.py")),
                                     *[inspect.getsource(f) for f in (collect_model_evidence, _person_candidates,
                                                                       _load_extractor, _safe_value, source_sentence, span_issues)]])
            evidence, model_cached = stage(queue, output, pid, "automatic_model_evidence", model_key,
                                           lambda p=paper, r=record: collect_model_evidence(p, parsed[p["id"]], r, models),
                                           lambda value, p=paper, r=record: validate_model_evidence(value, p, parsed[p["id"]], r))
            key = object_hash([VERSION, model_key, sha(Path(__file__).with_name("automatic_roles.py")),
                               sha(Path(__file__).with_name("card_enrichment.py")), inspect.getsource(interpret_paper)])
            row, cached = stage(queue, output, pid, "automatic_draft", key,
                                lambda p=paper, r=record, e=evidence: interpret_paper(p, r, e),
                                lambda value, p=paper, r=record: validate_draft(value, p, parsed[p["id"]], r))
            rows.append(row)
            counts["drafts_cached" if cached else "drafts_computed"] += 1
            counts["model_evidence_cached" if model_cached else "model_evidence_computed"] += 1
            if not model_cached:
                counts["relation_model_calls"] += row["stats"]["relation_model_calls"]
                counts["person_model_calls"] += row["stats"]["person_model_calls"]
            counts["person_candidates"] += len(row["person_candidates"])
            counts["retained_author_names_crosschecked"] += row["author_crosscheck"]["header_matched"]
            for mention in row["mentions"]:
                counts["role_" + mention["role"]["role"]] += 1
            if len(rows) % 10 == 0:
                print(json.dumps({"processed": len(rows), "total": 100}, ensure_ascii=True), flush=True)
        draft_seconds = time.perf_counter() - started - preflight_seconds
        queries, query_cached = _make_queries(source, output, manifest, parsed, set(quarantine), models, queue)
        check_inputs(vault, manifest)
    finally:
        queue.close()
    summary = {"schema": VERSION, "papers_in_frozen_cohort": len(rows), "counts": dict(counts),
               "preflight_seconds": round(preflight_seconds, 3), "draft_seconds": round(draft_seconds, 3),
               "query_cached": query_cached, "source_entity_extraction": "reused_existing_100_paper_cache",
               "formal_graph_publication": False, "all100_usable": False,
               "boundary": "Predictions are candidates; 100% source spans do not establish semantic correctness."}
    value = {"schema": VERSION, "models": models.identity, "inputs": {
        "manifest_sha256": sha(source / "manifest.json"), "records_sha256": sha(source / "records.json"),
        "exceptions_sha256": sha(vault / EXCEPTIONS), "parse_hashes": parse_hashes,
        "automatic_pilot_sha256": sha(Path(__file__)), "automatic_roles_sha256": sha(Path(__file__).with_name("automatic_roles.py"))},
        "papers": rows, "queries": queries,
        "boundary": "Working candidates only; manually reviewed author/content addenda were not prediction inputs."}
    changed = write_json(output / "automatic-drafts.json", value, output)
    stable_counts = {key: value for key, value in counts.items()
                     if key not in {"drafts_cached", "drafts_computed", "model_evidence_cached", "model_evidence_computed",
                                    "relation_model_calls", "person_model_calls"}}
    stable_counts["paper_drafts"] = len(rows) - len(quarantine)
    markdown_changed = _render(output, rows, queries, stable_counts)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    summary["outputs_changed"] = {"json": changed, "markdown": markdown_changed}
    write_json(output / "automatic-run.json", summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    args = parser.parse_args()
    print(json.dumps(run(args.vault, args.vault / OUTPUT, LocalModels(args.models, 4)), ensure_ascii=True))


if __name__ == "__main__":
    main()
