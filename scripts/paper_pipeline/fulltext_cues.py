"""Select source-located experimental and method windows across the PDF body.

The cues prioritize *where to look*. They do not assert the meaning, identity,
or role of a candidate and do not replace exact source/semantic validation.
"""
from __future__ import annotations

import re

from .core import object_hash
from .semantic import match_text


DATA_CUE = re.compile(r"\b(?:datasets?|data\s+sets?|benchmarks?|training|pretrain\w*|train(?:ed|ing)?|"
                      r"evaluat\w*|test(?:ed|ing)?|validation|experiments?)\b", re.I)
METHOD_CUE = re.compile(r"\b(?:we|our|this\s+(?:paper|work|study|method|model))\b[\s\S]{0,500}?"
                        r"\b(?:propos\w*|introduc\w*|present\w*|develop\w*|design\w*|"
                        r"us(?:e|es|ed|ing)|employ\w*|adopt\w*|appl\w*|compar\w*)\b", re.I)
PROBLEM_CUE = re.compile(r"\b(?:to\s+address|we\s+(?:study|address|investigate|examine)|"
                         r"our\s+(?:goal|objective)|the\s+(?:problem|challenge))\b", re.I)
REF_HEADER = re.compile(r"^(?:(?:\d+|[IVXLCDM]+)[.)]?\s+)?(?:references|bibliography)\s*$", re.I)
APPENDIX_HEADER = re.compile(r"^(?:appendix|appendices)\b", re.I)


def cue_windows(paper: dict, parsed: dict, selected: dict, words_per_window: int = 160, stride: int = 120) -> dict:
    if not 1 <= stride <= words_per_window <= 512:
        raise ValueError("invalid cue window size or stride")
    if parsed["paper_id"] != paper["id"] or parsed["pdf_sha256"] != paper["pdf_sha256"]:
        raise ValueError("cue windows belong to another paper")
    prior = selected["passages"]
    windows, seen, in_references = [], set(), False
    for page in parsed["pages"]:
        for block in page["blocks"]:
            if page["text"][block["start"]:block["end"]] != block["text"]:
                raise ValueError("cue source block does not match page")
            first = block["text"].splitlines()[0].strip() if block["text"].splitlines() else ""
            if REF_HEADER.fullmatch(first):
                in_references = True
                continue
            if APPENDIX_HEADER.match(first):
                in_references = False
            if in_references:
                continue
            words = list(re.finditer(r"\S+", block["text"]))
            for offset in range(0, len(words), stride):
                if offset and offset + words_per_window - stride >= len(words):
                    break
                chunk = words[offset:offset + words_per_window]
                if not chunk:
                    continue
                start, end = block["start"] + chunk[0].start(), block["start"] + chunk[-1].end()
                if any(p["page"] == page["page"] and p["start"] <= start < end <= p["end"] for p in prior):
                    continue
                literal = page["text"][start:end]
                if len(literal) < 35:
                    continue
                normalized = match_text(literal)
                tags = [name for name, pattern in (("dataset_or_experiment", DATA_CUE),
                                                    ("self_method", METHOD_CUE),
                                                    ("problem_or_goal", PROBLEM_CUE)) if pattern.search(normalized)]
                key = (page["page"], start, end)
                if not tags or key in seen:
                    continue
                seen.add(key)
                windows.append({"id": "span:" + object_hash([paper["id"], page["page"], start, end])[:24],
                                "paper_id": paper["id"], "page": page["page"], "start": start, "end": end,
                                "text": literal, "section": "fulltext_cue_not_semantically_classified",
                                "pdf": paper["pdf"], "pdf_sha256": paper["pdf_sha256"], "cue_types": tags})
    return {"passages": windows, "scope": "additional_cue_windows_not_exhaustive_full_document"}
