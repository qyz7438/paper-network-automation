"""Reinterpret verified fulltext model outputs after a role-policy change."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from .automatic_pilot import EXCEPTIONS, OUTPUT, interpret_paper, validate_draft, validate_model_evidence
from .card_enrichment import SOURCE
from .core import check_inputs, contained, output_root, read_json, sha, write_json
from .fulltext_pilot import merge_records
from .reclassify_fresh import load_completed


def load_direct(db, output: Path, paper_id: str, stage_name: str) -> dict:
    entries = db.execute("SELECT result,result_hash FROM jobs WHERE paper_id=? AND stage=? AND state='done'",
                         (paper_id, stage_name)).fetchall()
    if len(entries) != 1:
        raise ValueError("exactly one fulltext cache job required: " + stage_name)
    path = contained(output, entries[0][0])
    if sha(path) != entries[0][1]:
        raise ValueError("fulltext cache hash changed: " + stage_name)
    return read_json(path)


def run(vault: Path) -> dict:
    started = time.perf_counter()
    vault = vault.resolve()
    full = output_root(vault, vault / OUTPUT / "fulltext-cue-run")
    fresh = output_root(vault, vault / OUTPUT / "fresh-pdf-run")
    source = vault / SOURCE
    manifest = read_json(source / "manifest.json")
    original_path = full / "fulltext-drafts.json"
    original = read_json(original_path)
    if (original.get("schema") != "fulltext-cue-automatic-pilot-v1"
            or original["inputs"]["manifest_sha256"] != sha(source / "manifest.json")
            or original["inputs"]["exceptions_sha256"] != sha(vault / EXCEPTIONS)):
        raise ValueError("fulltext source identity changed")
    check_inputs(vault, manifest)
    papers = {paper["id"]: paper for paper in manifest["papers"]}
    if len(papers) != 100 or {row["paper_id"] for row in original["papers"]} != set(papers):
        raise ValueError("frozen pilot cohort changed")
    rows, counts = [], Counter()
    with closing(sqlite3.connect((fresh / "fresh-state.sqlite").as_uri() + "?mode=ro", uri=True)) as first, \
         closing(sqlite3.connect((full / "fulltext-state.sqlite").as_uri() + "?mode=ro", uri=True)) as second:
        for old in original["papers"]:
            pid = old["paper_id"]
            if old["state"] != "automatic_draft_needs_review":
                rows.append(old)
                counts[old["state"]] += 1
                continue
            paper = papers[pid]
            parsed = load_completed(first, fresh, pid, "fresh_parse")
            base = load_completed(first, fresh, pid, "fresh_entities")
            extra = load_direct(second, full, pid, "fulltext_entities")
            evidence = load_direct(second, full, pid, "fulltext_role_model")
            combined = merge_records(base, extra)
            validate_model_evidence(evidence, paper, parsed, combined)
            row = interpret_paper(paper, combined, evidence)
            validate_draft(row, paper, parsed, combined)
            rows.append(row)
            counts["papers_reclassified"] += 1
            for mention in row["mentions"]:
                counts["role_" + mention["role"]["role"]] += 1
    check_inputs(vault, manifest)
    value = {**original, "papers": rows, "inputs": {**original["inputs"],
             "original_fulltext_drafts_sha256": sha(original_path), "reclassifier_sha256": sha(Path(__file__)),
             "current_role_code_sha256": sha(Path(__file__).with_name("automatic_roles.py"))},
             "boundary": "Fulltext source/model evidence was reclassified from verified caches; no model calls or formal graph writes."}
    changed = write_json(full / "fulltext-reclassified-v4.json", value, full)
    result = {"state": "fulltext_evidence_reclassified_needs_review", "counts": dict(counts), "model_calls": 0,
              "elapsed_seconds": round(time.perf_counter() - started, 3), "output_changed": changed,
              "all100_usable": False, "formal_graph_publication": False}
    write_json(full / "fulltext-reclassification-v4-receipt.json", result, full)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.vault), ensure_ascii=True))


if __name__ == "__main__":
    main()
