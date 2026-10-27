"""Reclassify persisted fresh-PDF model evidence without rerunning the model."""
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


def load_completed(db, output: Path, paper_id: str, stage_name: str) -> dict:
    entries = db.execute("SELECT result,result_hash FROM jobs WHERE paper_id=? AND stage=? AND state='done'",
                         (paper_id, stage_name)).fetchall()
    if len(entries) != 1:
        raise ValueError("fresh model cache must have exactly one completed " + stage_name + " job")
    path = contained(output, entries[0][0])
    if sha(path) != entries[0][1]:
        raise ValueError("fresh model cache hash changed: " + stage_name)
    return read_json(path)["value"]


def run(vault: Path) -> dict:
    started = time.perf_counter()
    vault = vault.resolve()
    output = output_root(vault, vault / OUTPUT / "fresh-pdf-run")
    source = vault / SOURCE
    manifest = read_json(source / "manifest.json")
    original_path = output / "fresh-drafts.json"
    original = read_json(original_path)
    if (original["schema"] != "fresh-pdf-automatic-pilot-v1"
            or original["inputs"]["manifest_sha256"] != sha(source / "manifest.json")
            or original["inputs"]["exceptions_sha256"] != sha(vault / EXCEPTIONS)):
        raise ValueError("fresh prediction inputs changed")
    check_inputs(vault, manifest)
    papers = {p["id"]: p for p in manifest["papers"]}
    if len(papers) != 100 or {r["paper_id"] for r in original["papers"]} != set(papers):
        raise ValueError("fresh prediction cohort differs from the frozen 100")
    rows, counts = [], Counter()
    with closing(sqlite3.connect((output / "fresh-state.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        for old in original["papers"]:
            pid = old["paper_id"]
            if old["state"] != "automatic_draft_needs_review":
                rows.append(old)
                counts[old["state"]] += 1
                continue
            paper = papers[pid]
            parsed = load_completed(db, output, pid, "fresh_parse")
            record = load_completed(db, output, pid, "fresh_entities")
            evidence = load_completed(db, output, pid, "fresh_role_model")
            validate_model_evidence(evidence, paper, parsed, record)
            row = interpret_paper(paper, record, evidence)
            validate_draft(row, paper, parsed, record)
            rows.append(row)
            counts["papers_reclassified"] += 1
            for mention in row["mentions"]:
                counts["role_" + mention["role"]["role"]] += 1
    check_inputs(vault, manifest)
    value = {**original, "papers": rows, "inputs": {**original["inputs"],
             "original_fresh_drafts_sha256": sha(original_path),
             "reclassifier_sha256": sha(Path(__file__)),
             "current_role_code_sha256": sha(Path(__file__).with_name("automatic_roles.py"))},
             "boundary": "Fresh PDF/model evidence was reclassified from verified caches; no model calls or formal graph writes."}
    changed = write_json(output / "fresh-reclassified-v3.json", value, output)
    result = {"state": "fresh_evidence_reclassified_needs_review", "counts": dict(counts),
              "model_calls": 0, "elapsed_seconds": round(time.perf_counter() - started, 3),
              "output_changed": changed, "all100_usable": False, "formal_graph_publication": False}
    write_json(output / "fresh-reclassification-v3-receipt.json", result, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.vault), ensure_ascii=True))


if __name__ == "__main__":
    main()
