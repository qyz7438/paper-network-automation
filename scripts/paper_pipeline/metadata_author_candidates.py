"""Add PDF metadata names as header-matched author candidates, not byline facts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3
from contextlib import closing
import time

from .automatic_pilot import EXCEPTIONS, OUTPUT
from .card_enrichment import SOURCE, author_evidence, validate_quote
from .core import check_inputs, contained, output_root, read_json, sha, write_json
from .reclassify_fresh import load_completed
from .render import md_text, write_text


def split_author_metadata(raw: str) -> list[str]:
    if not isinstance(raw, str):
        raise ValueError("PDF author metadata must be text")
    return [part.strip() for part in re.split(r"\s*;\s*|\s+and\s+|\s*,\s*", raw, flags=re.I) if part.strip()]


def metadata_candidates(paper: dict, page: dict, raw: str) -> dict:
    parts = split_author_metadata(raw)
    # Keep all parts for inspection, including single-token names and groups;
    # a literal header match is not proof of a complete or ordered byline.
    checked = author_evidence({**paper, "recorded_authors": parts}, page)
    for candidate in checked["names"]:
        if candidate["evidence"]:
            validate_quote(paper, {page["page"]: page}, candidate["evidence"])
        candidate["state"] = ("metadata_name_header_match_needs_byline_review" if candidate["evidence"]
                              else "metadata_name_unmatched_or_ambiguous")
    return {"metadata_field": "pdf_document_author", "raw_metadata": raw, "parts": checked["names"],
            "header_matched": checked["header_matched"], "complete_byline_asserted": False,
            "order_asserted": False, "boundary": "PDF metadata and page text only; person identity and author order remain unverified."}


def run(vault: Path, reclassified: bool = False) -> dict:
    import pymupdf
    started = time.perf_counter()
    vault = vault.resolve()
    full = output_root(vault, vault / OUTPUT / "fulltext-cue-run")
    fresh = output_root(vault, vault / OUTPUT / "fresh-pdf-run")
    source = vault / SOURCE
    source_path = full / ("fulltext-reclassified-v4.json" if reclassified else "fulltext-drafts.json")
    manifest, original = read_json(source / "manifest.json"), read_json(source_path)
    if (original.get("schema") != "fulltext-cue-automatic-pilot-v1"
            or original["inputs"]["manifest_sha256"] != sha(source / "manifest.json")
            or original["inputs"]["exceptions_sha256"] != sha(vault / EXCEPTIONS)):
        raise ValueError("fulltext source identity changed")
    check_inputs(vault, manifest)
    papers = {paper["id"]: paper for paper in manifest["papers"]}
    if len(papers) != 100 or {row["paper_id"] for row in original["papers"]} != set(papers):
        raise ValueError("frozen pilot cohort changed")
    rows, matched_names, metadata_nonempty = [], 0, 0
    with closing(sqlite3.connect((fresh / "fresh-state.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        for old in original["papers"]:
            if old["state"] != "automatic_draft_needs_review":
                rows.append(old)
                continue
            paper = papers[old["paper_id"]]
            parsed = load_completed(db, fresh, paper["id"], "fresh_parse")
            page = next((page for page in parsed["pages"] if page["page"] == 1), None)
            if page is None:
                raise ValueError("first source page unavailable")
            path = contained(vault, paper["pdf"])
            with pymupdf.open(path) as document:
                raw = document.metadata.get("author") or ""
            candidate = metadata_candidates(paper, page, raw)
            matched_names += candidate["header_matched"]
            metadata_nonempty += bool(raw.strip())
            rows.append({**old, "metadata_author_candidates": candidate})
    check_inputs(vault, manifest)
    value = {**original, "papers": rows, "inputs": {**original["inputs"],
             "fulltext_drafts_sha256": sha(source_path), "metadata_code_sha256": sha(Path(__file__))},
             "boundary": "Fulltext candidates plus PDF metadata/header matches; no complete-byline claim or graph publication."}
    changed = write_json(full / "fulltext-author-drafts.json", value, full)
    lines = ["# PDF 元数据作者候选", "", "仅显示能与第一页文字相符的 PDF 元数据姓名；是否为作者、完整顺序与团组仍需核验。", ""]
    for row in rows:
        if row["state"] != "automatic_draft_needs_review":
            continue
        candidates = row["metadata_author_candidates"]["parts"]
        names = [item for item in candidates if item["evidence"]]
        if not names:
            continue
        lines += ["## " + md_text(row["title"]), "", "[[" + row["card"] + "|原卡]]", ""]
        for item in names:
            ev = item["evidence"]
            lines.append("- " + md_text(item["name"]) + " — [[" + ev["pdf"] + "#page=1|首页原文]]（待审）")
        lines.append("")
    markdown_changed = write_text(full / "metadata-author-candidates.md", "\n".join(lines), full)
    result = {"schema": "pdf-metadata-author-candidates-v1", "papers_with_metadata": metadata_nonempty,
              "header_matched_name_occurrences": matched_names, "elapsed_seconds": round(time.perf_counter() - started, 3),
              "drafts_changed": changed, "markdown_changed": markdown_changed, "complete_byline_asserted": False,
              "formal_graph_publication": False, "all100_usable": False}
    write_json(full / "metadata-author-receipt.json", result, full)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--reclassified", action="store_true", help="read verified v4 role reclassification")
    args = parser.parse_args()
    print(json.dumps(run(args.vault, args.reclassified), ensure_ascii=True))


if __name__ == "__main__":
    main()
