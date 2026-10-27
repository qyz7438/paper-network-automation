"""Audit the automatic pilot against frozen, evaluation-only source records.

Exact-name matching is a bounded retrieval check, not semantic ground truth for
unmatched candidates. The audit cannot publish cards or approve the 100.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import unicodedata

from .automatic_pilot import OUTPUT
from .core import output_root, read_json, sha, write_json


SOURCE_REVIEW = Path("pilot100-english-v1/retrieval-content-review.json")
GOLD_CONTENT = Path("pilot100-card-enrichment-v1/main-reviewed-content.json")
GOLD_AUTHORS = Path("pilot100-card-enrichment-v1/main-reviewed-authors.json")
GOLD_ROLES = Path("pilot100-card-enrichment-v1/main-role-decisions.json")
ROLE_MAP = {"paper_proposes_method": "proposed", "paper_uses_method": "adopted",
            "paper_uses_optimizer": "adopted", "paper_compares_method": "compared",
            "paper_analyzes_method": "analyzed", "paper_evaluates_on_dataset": "evaluation_data",
            "paper_trains_on_dataset": "training_data", "paper_studies_problem": "studied_problem"}


def normalize(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum())


def author_matches(row: dict, gold: dict) -> dict:
    truth = {normalize(item["name"]) for item in gold["authors"] if normalize(item["name"])}
    retained = {normalize(item["name"]) for item in row["author_crosscheck"]["names"] if item["evidence"]}
    extracted = {normalize(item["name_candidate"]) for item in row["person_candidates"]
                 if normalize(item["name_candidate"])}
    metadata = {normalize(item["name"]) for item in row.get("metadata_author_candidates", {}).get("parts", [])
                if item.get("evidence") and normalize(item["name"])}
    candidates = retained | extracted | metadata
    return {"gold_author_names": len(truth), "candidate_name_strings": len(candidates),
            "exact_name_matches": len(truth & candidates), "candidate_strings_not_exact_gold": len(candidates - truth),
            "gold_names_not_exact_candidate": len(truth - candidates),
            "retained_metadata_exact_matches": len(truth & retained),
            "person_model_exact_matches": len(truth & extracted),
            "pdf_metadata_header_exact_matches": len(truth & metadata),
            "additional_matches_from_pdf_metadata": len((truth & metadata) - retained - extracted),
            "boundary": "Candidate person mentions may include nonauthors or text fragments; no inferred byline order."}


def claim_matches(row: dict, gold: dict) -> dict:
    mapping = {"methods": {"method"}, "datasets": {"dataset"}, "problem": {"question", "task"}}
    result = {}
    for field, kinds in mapping.items():
        claims = gold["fields"][field]["claims"]
        literal = overlap = 0
        for claim in claims:
            name = normalize(claim.get("name", ""))
            if name and any(item["type"] in kinds and normalize(item["text"]) == name for item in row["mentions"]):
                literal += 1
            if any(item["type"] in kinds and "mention_evidence" in item
                   and any(item["mention_evidence"]["pdf_sha256"] == ev["pdf_sha256"]
                           and item["mention_evidence"]["page"] == ev["page"]
                           and max(item["mention_evidence"]["start"], ev["start"]) < min(item["mention_evidence"]["end"], ev["end"])
                           for ev in claim.get("evidence", [])) for item in row["mentions"]):
                overlap += 1
        result[field] = {"reviewed_claims": len(claims), "exact_name_occurrence_matches": literal,
                         "not_exact_name_matched": len(claims) - literal,
                         "same_type_mention_overlaps_reference_evidence": overlap,
                         "gold_state": gold["fields"][field]["state"]}
    return result


def sample_roles(rows: list[dict], selected: set[str], plan: dict) -> tuple[list[dict], dict]:
    result, population = [], Counter()
    by_role = {role: [] for role in plan["categories"]}
    for row in rows:
        if row["paper_id"] not in selected or row["state"] != "automatic_draft_needs_review":
            continue
        for mention in row["mentions"]:
            role = mention["role"]["role"]
            if role in by_role:
                population[role] += 1
                by_role[role].append((row, mention))
    for role, candidates in by_role.items():
        candidates.sort(key=lambda pair: hashlib.sha256((plan["seed"] + pair[1]["mention_id"]).encode("utf-8")).hexdigest())
        chosen, per_paper = [], Counter()
        for row, mention in candidates:
            if per_paper[row["paper_id"]] >= plan.get("max_per_paper_per_role", plan["per_category_max"]):
                continue
            chosen.append((row, mention))
            per_paper[row["paper_id"]] += 1
            if len(chosen) == plan["per_category_max"]:
                break
        for row, mention in chosen:
            result.append({"paper_id": row["paper_id"], "title": row["title"], "mention_id": mention["mention_id"],
                           "type": mention["type"], "text": mention["text"], "predicted_role": role,
                           "role_details": mention["role"], "sentence_evidence": mention["sentence_evidence"],
                           "mention_evidence": mention["mention_evidence"], "review_state": "pending_source_judgment"})
    return result, dict(population)


def retrieval_review(queries: list[dict], previous: dict) -> dict:
    prior = {(row["query"], row["paper_id"]): row for row in previous["rows"]}
    categories, pending = [], []
    for query in queries:
        supported = unsupported = unreviewed = 0
        for row in query["rows"]:
            review = prior.get((query["query"], row["paper_id"]))
            ids = [passage["id"] for passage in row["passages"]]
            if review is None or ids != review["passage_ids"]:
                unreviewed += 1
                pending.append({"query": query["query"], "paper_id": row["paper_id"], "title": row["title"],
                                "passages": row["passages"], "review_state": "pending_source_judgment"})
            elif review["topic_supported_in_retrieved_passages"]:
                supported += 1
            else:
                unsupported += 1
        categories.append({"query": query["query"], "rows": len(query["rows"]),
                           "previous_source_review_reusable": supported + unsupported,
                           "topic_supported": supported, "topic_unsupported": unsupported, "unreviewed": unreviewed})
    return {"queries": categories, "unreviewed_rows": pending,
            "boundary": "Reuses only judgments for identical query, paper and ordered passage IDs; development evidence, not held-out recall."}


def run(vault: Path, version: int = 2, mode: str = "cached") -> dict:
    started = time.perf_counter()
    vault = vault.resolve()
    output = output_root(vault, vault / OUTPUT)
    base = output.parent
    if version not in (1, 2, 3, 4):
        raise ValueError("unsupported frozen audit version")
    if mode not in ("cached", "fresh", "fresh-reclassified", "fulltext", "fulltext-author"):
        raise ValueError("unsupported automatic draft source")
    plan_path = output / f"evaluation-plan-v{version}.json"
    sampling_path = output / ("audit-sampling-plan-v3-diverse.json" if version == 3 else f"audit-sampling-plan-v{version}.json")
    plan, sampling = read_json(plan_path), read_json(sampling_path)
    for relative, expected in plan["input_sha256"].items():
        if sha(base / relative) != expected:
            raise ValueError("frozen evaluation input changed: " + relative)
    drafts_path = output / ({"cached": "automatic-drafts.json", "fresh": "fresh-pdf-run/fresh-drafts.json",
                             "fresh-reclassified": "fresh-pdf-run/fresh-reclassified-v3.json",
                             "fulltext": "fulltext-cue-run/fulltext-drafts.json",
                             "fulltext-author": "fulltext-cue-run/fulltext-author-drafts.json"}[mode])
    drafts = read_json(drafts_path)
    if (drafts["inputs"]["manifest_sha256"] != plan["input_sha256"]["pilot100-english-v1/manifest.json"]
            or (mode == "cached" and drafts["inputs"].get("records_sha256") != plan["input_sha256"]["pilot100-english-v1/records.json"])
            or drafts["inputs"]["exceptions_sha256"] != plan["input_sha256"]["pilot100-card-enrichment-v1/source-exceptions.json"]):
        raise ValueError("draft inputs do not match frozen evaluation")
    if mode in ("fresh", "fresh-reclassified") and drafts.get("schema") != "fresh-pdf-automatic-pilot-v1":
        raise ValueError("fresh audit requires fresh PDF prediction artifacts")
    if mode in ("fulltext", "fulltext-author") and drafts.get("schema") != "fulltext-cue-automatic-pilot-v1":
        raise ValueError("fulltext audit requires cue-expanded prediction artifacts")
    rows = drafts["papers"]
    if len(rows) != plan["corpus_papers"] or len({r["paper_id"] for r in rows}) != plan["corpus_papers"]:
        raise ValueError("draft cohort is not the frozen 100")
    row_map = {r["paper_id"]: r for r in rows}
    selected = set(plan["gold_sample"]["paper_ids"])
    gold = {r["paper_id"]: r for r in read_json(base / GOLD_CONTENT)["papers"]}
    authors = {r["paper_id"]: r for r in read_json(base / GOLD_AUTHORS)["papers"]}
    if len(selected) != plan["gold_sample"]["sample_size"] or not selected <= row_map.keys():
        raise ValueError("frozen holdout sample differs from draft")
    field_rows, unassessable = [], []
    for pid in sorted(selected):
        if row_map[pid]["state"] != "automatic_draft_needs_review":
            unassessable.append({"paper_id": pid, "state": row_map[pid]["state"]})
            continue
        if pid not in gold or pid not in authors:
            unassessable.append({"paper_id": pid, "state": "no_existing_reviewed_field_reference"})
            continue
        field_rows.append({"paper_id": pid, "authors": author_matches(row_map[pid], authors[pid]),
                           "fields": claim_matches(row_map[pid], gold[pid])})
    packet, population = sample_roles(rows, selected, sampling)
    gold_role_rows = []
    mentions = {m["mention_id"]: m for row in rows if row["state"] == "automatic_draft_needs_review" for m in row["mentions"]}
    for decision in read_json(base / GOLD_ROLES)["decisions"]:
        if decision["paper_id"] not in selected:
            continue
        mention = mentions.get(decision["mention_id"])
        if mention is None:
            gold_role_rows.append({"mention_id": decision["mention_id"], "state": "source_mention_not_found"})
            continue
        expected = ROLE_MAP.get(decision["accepted_role"]) if decision["decision"] == "accepted" else None
        actual = mention["role"]["role"]
        gold_role_rows.append({"mention_id": decision["mention_id"], "gold_decision": decision["decision"],
                               "gold_role": expected, "automatic_role": actual,
                               "state": "unmapped_reference_role" if decision["decision"] == "accepted" and expected is None else
                                        "match" if expected and actual == expected else
                                        "correct_abstention" if expected is None and actual == "unresolved" else
                                        "mismatch_or_unresolved"})
    retrieval = retrieval_review(drafts["queries"], read_json(base / SOURCE_REVIEW))
    retrieval_packet = retrieval.pop("unreviewed_rows")
    fields = {}
    for field in ("methods", "datasets", "problem"):
        fields[field] = {key: sum(row["fields"][field][key] for row in field_rows)
                         for key in ("reviewed_claims", "exact_name_occurrence_matches", "not_exact_name_matched",
                                     "same_type_mention_overlaps_reference_evidence")}
    author_totals = {key: sum(row["authors"][key] for row in field_rows)
                     for key in ("gold_author_names", "candidate_name_strings", "exact_name_matches",
                                 "candidate_strings_not_exact_gold", "gold_names_not_exact_candidate",
                                 "retained_metadata_exact_matches", "person_model_exact_matches",
                                 "pdf_metadata_header_exact_matches", "additional_matches_from_pdf_metadata")}
    summary = {"schema": "automatic-pilot-audit-v1", "state": "sampled_audit_pending_semantic_judgment",
               "drafts_sha256": sha(drafts_path), "plan_sha256": sha(plan_path),
               "sampling_plan_sha256": sha(sampling_path), "sample_papers": len(selected),
               "sample_papers_assessable": len(field_rows), "sample_papers_unassessable": unassessable,
               "author_exact_name_counts": author_totals, "field_exact_name_counts": fields,
               "role_population": population, "role_review_packet_count": len(packet),
               "previous_role_decisions_on_holdout": dict(Counter(r["state"] for r in gold_role_rows)),
               "retrieval": retrieval, "retrieval_reference_sha256": sha(base / SOURCE_REVIEW),
               "formal_graph_publication": False, "all100_usable": False,
               "boundary": "All matches are source/name checks; complete semantic audit and Obsidian acceptance remain pending."}
    suffix = f"{mode}-v{version}"
    write_json(output / f"role-review-packet-{suffix}.json", {"sampling_plan_sha256": sha(sampling_path),
                                                      "rows": packet}, output)
    write_json(output / f"retrieval-review-packet-{suffix}.json", {"drafts_sha256": sha(drafts_path),
                                                                     "rows": retrieval_packet}, output)
    write_json(output / f"automatic-audit-detail-{suffix}.json", {"field_rows": field_rows, "role_gold_rows": gold_role_rows}, output)
    summary["audit_process_seconds"] = round(time.perf_counter() - started, 3)
    write_json(output / f"automatic-audit-summary-{suffix}.json", summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, type=Path)
    parser.add_argument("--version", choices=(1, 2, 3, 4), default=2, type=int)
    parser.add_argument("--mode", choices=("cached", "fresh", "fresh-reclassified", "fulltext", "fulltext-author"), default="cached")
    args = parser.parse_args()
    print(json.dumps(run(args.vault, args.version, args.mode), ensure_ascii=True))


if __name__ == "__main__":
    main()
