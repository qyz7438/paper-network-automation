"""Source-bounded, unreviewed role candidates for a paper mention.

Neither a model edge nor a lexical cue establishes a publishable graph edge.
"""
from __future__ import annotations

import re
import unicodedata

from .audit_card_readiness import GENERIC
from .card_enrichment import role_candidate
from .relation_trial import span_issues


RELATION_SCHEMA = ["proposes", "uses", "compares"]
SELF_ACTOR = re.compile(
    r"^(?:we|our\s+(?:(?:proposed|own|new)\s+)?(?:work|method|model|approach|framework|system)|"
    r"this\s+(?:paper|work|study|method|model|approach))$", re.I,
)
MODEL_TRIGGER = re.compile(
    r"\b(?:we|our|this\s+(?:paper|work|study|method|model))\b"
    r"[\s\S]{0,600}?\b(?:propos\w*|introduc\w*|present\w*|develop\w*|design\w*|"
    r"us(?:e|es|ed|ing)|employ\w*|adopt\w*|appl\w*|compar\w*)\b", re.I,
)
OWN_COMPARISON = re.compile(r"\bwe\s+(?:(?:also|further|directly)\s+)?compar\w*\b", re.I)
BACKGROUND = re.compile(r"\b(?:prior|previous|earlier|existing|related)\s+(?:work|works|method|methods|approach|approaches|studies|study)\b", re.I)
TERM = re.compile(r"\b(?:the\s+)?(?:name|term|word|notation|label)\s*[\"'“‘]*\s*$", re.I)
DEPENDENT = re.compile(r"\b(?:within|based\s+on|inspired\s+by|same\s+as|variant\s+of|"
                       r"instead\s+of|rather\s+than|following)\b", re.I)
NEGATION = re.compile(r"\b(?:not|never|without|rather\s+than|instead\s+of)\b", re.I)
CONDITIONAL = re.compile(r"\b(?:if|unless|suppose|assuming|hypothetically)\b", re.I)
ANALYTICAL_FRAME = re.compile(r"\b(?:in|from)\s+the\s+(?:language|context|perspective|terms|form)\s+of\b", re.I)
DATA_PREDICATE = re.compile(r"\b(?:train|pretrain|evaluat|test|benchmark)\w*\b", re.I)
PROPOSED_USE = re.compile(r"\b(?:propose|plan|intend|aim)\w*\s+to\s+(?:use|apply|employ|adopt)\b", re.I)
ANALYSIS_REDIRECT = re.compile(r"\b(?:to\s+address|they\s+(?:(?:successfully|directly)\s+)?address)\b", re.I)
REFERENCE = re.compile(r"^(?:\[[\d,\s\-–]+\]|(?:fig(?:ure)?|table|tab|algorithm|alg|appendix|section|sec|equation|eq)\.?\s*[\dA-Z][\w.\-]*)$", re.I)
DATA_REGIME = re.compile(r"^(?:(?:online|offline|llm|open[ -]?web|closed[ -]?web)(?:\s+(?:rl|reinforcement\s+learning))?)$", re.I)
ADVERB_FRAGMENT = re.compile(r"^[a-z]{3,}ly$", re.I)
PERFORMANCE_FRAGMENT = re.compile(r"\bperformance$", re.I)
TERMINOLOGY_USE = re.compile(r"\b(?:we|they)\s+use\b", re.I)
TERMINOLOGY_DEFINITION = re.compile(r"\bto\s+refer\s+to\b", re.I)
NONMETHOD_APPOSITIVE = re.compile(
    r"^[\s,()\u2014\u2013-]*(?:a|an)\s+family\s+of\s+[^.;]{0,120}\b(?:manifolds|datasets|benchmarks)\b", re.I,
)


def _non_entity(kind: str, value: str) -> bool:
    normalized = re.sub(r"[-\u00ad]\s*\n\s*", "", unicodedata.normalize("NFKC", value))
    normalized = " ".join(normalized.split()).strip()
    if REFERENCE.fullmatch(normalized):
        return True
    if kind == "method" and normalized.casefold() in GENERIC | {"employment", "application", "usage"}:
        return True
    if kind == "method" and (ADVERB_FRAGMENT.fullmatch(normalized) or PERFORMANCE_FRAGMENT.search(normalized)):
        return True
    return kind == "dataset" and bool(DATA_REGIME.fullmatch(normalized))


def _background_for_mention(sentence: str, start: int) -> bool:
    cues = list(BACKGROUND.finditer(sentence, 0, start))
    if not cues:
        return False
    between = sentence[cues[-1].end():start]
    return not re.search(r"\b(?:we|our|this\s+(?:paper|work|study|method|model))\b", between, re.I)


def _covered(span: dict, start: int, end: int) -> bool:
    return span["start"] <= start < end <= span["end"]


def _self_actor(span: dict) -> bool:
    return bool(SELF_ACTOR.fullmatch(" ".join(span["text"].split()).strip(" ,.;:")))


def _model_signals(sentence: str, start: int, end: int, kind: str, raw: dict) -> list[dict]:
    if span_issues(raw, sentence):
        raise ValueError("relation model span or confidence differs from source")
    relations = raw.get("relation_extraction")
    if not isinstance(relations, dict) or any(key not in RELATION_SCHEMA for key in relations):
        raise ValueError("unexpected relation schema")
    signals = []
    for label, edges in relations.items():
        if not isinstance(edges, list):
            raise ValueError("malformed relation list")
        for edge in edges:
            head, tail = edge.get("head"), edge.get("tail")
            if not isinstance(head, dict) or not isinstance(tail, dict):
                raise ValueError("relation endpoints must have source spans")
            if not all(key in endpoint for endpoint in (head, tail) for key in ("start", "end", "text")):
                raise ValueError("relation endpoint missing source span")
            if label == "compares":
                if kind == "method" and OWN_COMPARISON.search(sentence) and (_covered(head, start, end) or _covered(tail, start, end)):
                    signals.append({"role": "compared", "basis": "relation_model", "edge": edge})
                continue
            if not _covered(tail, start, end):
                continue
            prefix = sentence[:start]
            if not _self_actor(head):
                if _background_for_mention(sentence, start) and (BACKGROUND.search(head["text"]) or not MODEL_TRIGGER.search(sentence)):
                    signals.append({"role": "background", "basis": "relation_model_other_actor", "edge": edge})
                continue
            if TERM.search(prefix) or DEPENDENT.search(sentence[head["end"]:start]) or NEGATION.search(sentence[head["end"]:start]):
                continue
            if label == "proposes" and kind == "method":
                signals.append({"role": "proposed", "basis": "relation_model", "edge": edge})
            elif label == "uses" and kind == "method":
                signals.append({"role": "adopted", "basis": "relation_model", "edge": edge})
    return signals


def classify_mention(kind: str, sentence: str, start: int, end: int, raw: dict | None = None) -> dict:
    """Return an explicitly unreviewed candidate or abstain on conflict.

    Offsets are relative to the *exact* sentence supplied to the local model.
    """
    if not 0 <= start < end <= len(sentence):
        raise ValueError("mention is outside source sentence")
    if _non_entity(kind, sentence[start:end]):
        return {"role": "unresolved", "reason": "generic_fragment_or_cross_reference", "signals": []}
    if kind == "method" and TERMINOLOGY_USE.search(sentence[:start]) and TERMINOLOGY_DEFINITION.search(sentence[end:]):
        return {"role": "unresolved", "reason": "terminology_not_method", "signals": []}
    if kind == "method" and NONMETHOD_APPOSITIVE.search(sentence[end:]):
        return {"role": "unresolved", "reason": "entity_type_not_established", "signals": []}
    rule = role_candidate(kind, sentence, start, end)
    model = _model_signals(sentence, start, end, kind, raw) if raw is not None else []
    mapping = {"proposed_candidate": "proposed", "adopted_candidate": "adopted",
               "compared_candidate": "compared", "analyzed_candidate": "analyzed",
               "training_data_candidate": "training_data", "evaluated_candidate": "evaluation_data",
               "studied_problem_candidate": "studied_problem"}
    signals = list(model)
    rule_role = mapping.get(rule["role"])
    prefix = sentence[:start]
    if CONDITIONAL.search(sentence):
        return {"role": "unresolved", "reason": "conditional_statement", "signals": signals}
    if PROPOSED_USE.search(prefix):
        return {"role": "unresolved", "reason": "proposed_use_not_confirmed_execution", "signals": signals}
    if rule_role == "compared" and kind != "method":
        rule_role = None
    if rule_role == "analyzed" and (ANALYTICAL_FRAME.search(prefix) or ANALYSIS_REDIRECT.search(prefix)):
        rule_role = None
    if kind == "dataset" and rule_role in {"training_data", "evaluation_data"}:
        predicates = list(DATA_PREDICATE.finditer(prefix))
        if len(predicates) > 1:
            return {"role": "unresolved", "reason": "multiple_dataset_predicates", "signals": signals}
    # A model name can be a term, or the object of a more specific proposal.
    if kind == "method" and (TERM.search(prefix) or DEPENDENT.search(prefix)):
        if rule_role in {"proposed", "adopted"}:
            rule_role = None
    if rule_role:
        signals.append({"role": rule_role, "basis": "source_predicate_rule", "predicate": rule})
    if (kind in {"method", "dataset", "task", "question"} and _background_for_mention(sentence, start)
            and not MODEL_TRIGGER.search(sentence) and not signals):
        signals.append({"role": "background", "basis": "explicit_prior_work_cue"})
    roles = {signal["role"] for signal in signals}
    if rule["role"] == "negated_candidate":
        return {"role": "unresolved", "reason": "negated_statement", "signals": signals}
    if not roles:
        return {"role": "unresolved", "reason": "no_supported_local_role_signal", "signals": signals}
    if len(roles) != 1:
        return {"role": "unresolved", "reason": "role_signals_conflict", "signals": signals}
    return {"role": next(iter(roles)), "state": "unreviewed_candidate", "signals": signals}
