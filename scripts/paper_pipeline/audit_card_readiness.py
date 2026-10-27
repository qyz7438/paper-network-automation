"""Audit source-bound extraction separately from accepted paper-card content."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json

from .core import cache_path, check_inputs, output_root, read_json, sha, write_json
from .local_pipeline import validate_result
from .semantic import match_text

SUPPORTED = {"original_supported", "official_bibliography", "not_applicable"}
GENERIC = {"model", "method", "approach", "framework", "algorithm", "learning", "training", "data", "features",
           "generalization", "classification", "segmentation", "detection", "attention", "fusion", "token", "loss"}


def field_state(profile: dict, field: str) -> str:
    value = profile.get(field)
    return value.get("state", "unresolved") if isinstance(value, dict) else "unresolved"


def audit(vault: Path, output: Path, baseline: Path) -> dict:
    output = output_root(vault, vault / output)
    baseline = output_root(vault, vault / baseline)
    manifest, records = read_json(output / "manifest.json"), read_json(output / "records.json")
    check_inputs(vault, manifest)
    if len(records) != len(manifest["papers"]) or len({p["pdf_sha256"] for p in manifest["papers"]}) != len(records):
        raise ValueError("incomplete or duplicate source cohort")
    by_paper = {r["paper_id"]: r for r in records}
    profiles = read_json(vault / "_wiki/network/paper-profiles.json")
    retained = {}
    for group_id, group in profiles["groups"].items():
        profile = group["profile"]
        for paper_id in profile.get("paper_ids", []):
            retained.setdefault(paper_id, []).append((group_id, profile))
    rows, totals = [], Counter()
    for paper in manifest["papers"]:
        record = by_paper[paper["id"]]
        parsed = read_json(cache_path(baseline, paper))
        validate_result(record, paper, parsed, record["selection"])
        candidates = retained.get(paper["id"], [])
        # Multiple source profiles are not silently resolved by choosing the best.
        profile = candidates[0][1] if len(candidates) == 1 else {}
        states = {f: field_state(profile, f) for f in ("authors", "methods", "problem", "datasets")}
        gaps = [f for f, state in states.items() if state not in SUPPORTED]
        counts = Counter(x["type"] for x in record["occurrences"])
        suspicious = [x["id"] for x in record["occurrences"] if x["type"] == "method"
                      and (len(match_text(x["text"]).strip()) < 3 or match_text(x["text"]).casefold() in GENERIC)]
        exact = sum(bool(x.get("exact_name_candidate_ids")) for x in record["occurrences"])
        row = {"paper_id": paper["id"], "title": paper["title"], "card": paper["card"], "pdf": paper["pdf"],
               "source_sha256": paper["pdf_sha256"], "source_spans_valid": True, "candidate_counts": dict(counts),
               "retained_profile_groups": [g for g, _ in candidates], "retained_field_states": states,
               "retained_complete_fieldset": not gaps and len(candidates) == 1,
               "fields_requiring_verification": gaps, "recorded_graph_authors": len(paper.get("recorded_authors", [])),
               "recorded_author_boundary": paper.get("author_state"),
               "suspicious_generic_or_fragment_method_mentions": suspicious,
               "exact_name_candidates": exact, "model_role_unresolved_count": sum(x["role"] == "unresolved_model_mention" for x in record["occurrences"]),
               "rejected_model_spans": len(record.get("rejected_occurrences", [])),
               "new_card_semantic_acceptance": "pending_main_source_and_role_review"}
        rows.append(row)
        totals["papers_with_retained_profile"] += bool(candidates)
        totals["papers_with_retained_complete_fieldset"] += row["retained_complete_fieldset"]
        totals["valid_mentions"] += len(record["occurrences"])
        totals["suspicious_method_mentions"] += len(suspicious)
        totals["exact_name_candidates"] += exact
        totals["unresolved_model_roles"] += row["model_role_unresolved_count"]
        totals["rejected_model_spans"] += row["rejected_model_spans"]
        for field in gaps:
            totals["papers_requiring_" + field + "_verification"] += 1
    value = {"schema": "paper-content-readiness-audit-v1", "state": "source_extraction_complete_semantic_acceptance_pending",
             "papers": len(rows), "all_source_spans_valid": True, "counts": dict(totals), "rows": rows,
             "all100_cards_usable": False, "full_corpus_unlocked": False,
             "boundary": "Legacy field states are retained evidence, not new verification. Model mentions are not adopted-method or dataset-use facts; generic flags are review candidates, not automatic deletion.",
             "inputs": {"manifest_sha256": sha(output / "manifest.json"), "records_sha256": sha(output / "records.json"),
                        "profiles_sha256": sha(vault / "_wiki/network/paper-profiles.json"), "audit_code_sha256": sha(Path(__file__))}}
    write_json(output / "pilot-content-audit.json", value, output)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("_working/paper-network-automation-v1/pilot100-english-v1"))
    parser.add_argument("--baseline", type=Path, default=Path("_working/paper-network-automation-v1/pilot100-working-v1"))
    args = parser.parse_args()
    value = audit(args.vault.resolve(), args.output, args.baseline)
    print(json.dumps({k: value[k] for k in ("state", "papers", "counts", "full_corpus_unlocked")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
