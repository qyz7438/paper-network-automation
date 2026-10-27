"""A frozen query-translation gate followed by a source-validated 100-paper run."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

from .core import (Jobs, cache_path, check_inputs, contained, freeze, object_hash,
                   parse_pdf, read_json, sha, write_json)
from .core import output_root
from .local_models import LocalModels, DEFAULT_MODELS
from .local_pipeline import retrieve, run as run_local
from .query_translation import LocalTranslator, SPECS
from .semantic import registry

OUTPUT = Path("_working/paper-network-automation-v1/pilot100-local-v1")
BASELINE = Path("_working/paper-network-automation-v1/pilot100-working-v1")
OLD = Path("_working/paper-network-automation-v1/pilot20-v1")
OLD_LOCAL = Path("_working/paper-network-automation-v1/pilot20-local-models-v1")
ENGLISH_OUTPUT = Path("_working/paper-network-automation-v1/pilot100-english-v1")
PATTERNS = {"fusion": r"fusion|multimodal|multi-modal|cross.modal|multi.sensor",
            "credit": r"credit assignment|long.horizon|sparse.reward|hierarchical reinforcement|delayed.reward",
            "attention": r"attention.sink|register|outlier.token|token.artifact",
            "equilibrium": r"equilibrium|fixed.point|convergen|implicit.layer"}


def precision(rows: list[dict], relevant: list[str], k: int = 5) -> float:
    return sum(r["paper_id"] in relevant for r in rows[:k]) / k


def check_translator_identity(plan: dict, identity: dict, translations: list[dict] | None = None) -> None:
    expected = plan.get("candidate")
    if not expected or "repo" not in expected:
        return
    checkpoint = identity["checkpoint"]
    engines = {v["engine"] for v in SPECS.values() if v["repo"] == expected["repo"] and v["revision"] == expected["revision"]}
    if (checkpoint["repo"] != expected["repo"] or checkpoint["revision"] != expected["revision"]
            or identity["engine"] not in engines):
        raise ValueError("translator does not match the frozen plan")
    if translations is not None and any(t["engine"] != identity["engine"] for t in translations):
        raise ValueError("mixed translator outputs in gate")


def gate20(vault: Path, output: Path | None = None, model_key: str = "qwen17") -> dict:
    import numpy as np
    output = output_root(vault, vault / (output or OUTPUT))
    plan = read_json(output / "evaluation-plan.json")
    manifest = read_json(vault / OLD_LOCAL / "manifest.json")
    check_inputs(vault, manifest)
    started = time.perf_counter()
    all_cases = plan["cases"] + plan.get("confirmation_cases", [])
    if len({c["id"] for c in all_cases}) != len(all_cases):
        raise ValueError("query case IDs must be unique")
    with LocalTranslator(output / "translation-cache", model_key=model_key) as translator:
        check_translator_identity(plan, translator.identity)
        translations = [{"case_id": c["id"], **translator.translate(c["query"], c["protected"], c["required"])}
                        for c in all_cases]
        translation_receipt = {"identity": translator.identity, "calls": translator.calls, "cache_hits": translator.hits,
                               "server_pid": None if translator.process is None else translator.process.pid}
    by_id = {x["case_id"]: x for x in translations}
    index = read_json(vault / OLD_LOCAL / "index.json")
    vector_path = contained(vault / OLD_LOCAL, index["vectors"])
    if sha(vector_path) != index["sha256"]:
        raise ValueError("development vector index changed")
    if set(plan["judged_ids"]) != {p["id"] for p in manifest["papers"]}:
        raise ValueError("development judgments differ from source cohort")
    docs = index["documents"]
    vectors = np.load(vector_path, allow_pickle=False)[:len(docs)]
    models = LocalModels(DEFAULT_MODELS, 4)
    if models.identity != read_json(vault / OLD_LOCAL / "models.json"):
        raise ValueError("development embedding identity changed")
    paper_map = {p["id"]: p for p in manifest["papers"]}
    cases = {c["id"]: c for c in plan["cases"]}
    inputs = [(q, kind, text) for q in plan["retrieval"] for kind, text in
              [("chinese", cases[q["case_id"]]["query"]), ("translated", by_id[q["case_id"]]["english"]),
               ("explicit_english", q["explicit_english"])]]
    query_vectors = models.encode([x[2] for x in inputs], "query")
    results = []
    for (query, kind, text), vector in zip(inputs, query_vectors, strict=True):
        result = retrieve(text, docs, vectors, vector, paper_map)
        results.append({"query_id": query["id"], "kind": kind, "precision_at_5": precision(result["rows"], query["relevant_ids"]),
                        "relevant_ids": query["relevant_ids"], **result})
    means = {kind: round(sum(r["precision_at_5"] for r in results if r["kind"] == kind) / len(plan["retrieval"]), 6)
             for kind in ("chinese", "translated", "explicit_english")}
    value = {"schema": "query-translation-gate-v1", "plan_sha256": sha(output / "evaluation-plan.json"),
             "translations": translations, "translation_runtime": translation_receipt, "retrieval": results,
             "mean_precision_at_5": means, "judgment_scope": plan["judgment_scope"],
             "requires_meaning_review": True, "seconds": round(time.perf_counter() - started, 3),
             "protected_inputs_unchanged": True, "index_sha256": sha(vault / OLD_LOCAL / "index.json")}
    check_inputs(vault, manifest)
    write_json(output / "gate20.json", value, output)
    return value


def accepted_gate(output: Path) -> dict:
    plan = read_json(output / "evaluation-plan.json")
    gate = read_json(output / "gate20.json")
    review = read_json(output / "meaning-review.json")
    if plan.get("candidate"):
        check_translator_identity(plan, gate["translation_runtime"]["identity"], gate["translations"])
    if gate["plan_sha256"] != sha(output / "evaluation-plan.json") or review["gate_sha256"] != sha(output / "gate20.json"):
        raise ValueError("review does not bind current gate and plan")
    all_cases = plan["cases"] + plan.get("confirmation_cases", [])
    if review["reviewer"] != "llm-wiki-main" or set(review["cases"]) != {c["id"] for c in all_cases}:
        raise ValueError("complete main-owner meaning review required")
    if any(type(x.get("meaning_preserved")) is not bool for x in review["cases"].values()):
        raise ValueError("meaning_preserved must be a boolean")
    passed = sum(review["cases"][c["id"]]["meaning_preserved"] is True for c in plan["cases"])
    if passed < plan["gate"]["minimum_meaning_pass"]:
        raise ValueError("translation meaning gate failed")
    confirmation_passed = sum(review["cases"][c["id"]]["meaning_preserved"] is True for c in plan.get("confirmation_cases", []))
    if confirmation_passed < plan["gate"].get("minimum_confirmation_pass", 0):
        raise ValueError("new translation cases failed")
    translations = {x["case_id"]: x for x in gate["translations"]}
    if any(translations[q["case_id"]]["errors"] or not review["cases"][q["case_id"]]["meaning_preserved"] for q in plan["retrieval"]):
        raise ValueError("a retrieval query translation is rejected")
    if any(not x["meaning_preserved"] and not translations[c]["errors"] for c, x in review["cases"].items()):
        raise ValueError("undetected meaning drift prevents expansion")
    scores = gate["mean_precision_at_5"]
    if scores["translated"] + plan["gate"]["max_precision_drop"] + 1e-9 < scores["chinese"]:
        raise ValueError("retrieval non-regression gate failed")
    return {"meaning_passed": passed, "cases": len(plan["cases"]), "confirmation_passed": confirmation_passed,
            "confirmation_cases": len(plan.get("confirmation_cases", [])), "precision": scores,
            "gate_sha256": sha(output / "gate20.json"), "review_sha256": sha(output / "meaning-review.json")}


def prepare100(vault: Path, plan: dict) -> dict:
    baseline = vault / BASELINE
    if (baseline / "manifest.json").exists():
        manifest = read_json(baseline / "manifest.json")
        check_inputs(vault, manifest)
        selection = read_json(baseline / "selection.json")
        if len(manifest["papers"]) != 100 or selection["selection_plan_sha256"] != object_hash(plan["selection"]):
            raise ValueError("existing 100-paper cohort differs")
        if not (baseline / "registry.json").is_file() or any(not cache_path(baseline, p).is_file() for p in manifest["papers"]):
            raise RuntimeError("frozen source import incomplete; retain cohort and inspect source import jobs")
        return manifest
    started = time.perf_counter()
    old = read_json(vault / OLD / "manifest.json")
    check_inputs(vault, old)
    graph = read_json(vault / "_wiki/network/graph.json")
    old_ids = {p["id"] for p in old["papers"]}
    selected = [p["id"] for p in old["papers"]]
    used_sha = {p["pdf_sha256"] for p in old["papers"]}
    parsed = {p["id"]: read_json(cache_path(vault / OLD, p)) for p in old["papers"]}
    candidates = [n for n in graph["nodes"] if n["type"] == "paper" and n["id"] not in old_ids]
    candidates.sort(key=lambda n: object_hash([plan["selection"]["seed"], n["id"]]))
    excluded, groups = [], {}
    considered = set()
    for group, quota in plan["selection"]["strata"] + [["general_fill", 80]]:
        count = 0
        for node in candidates:
            if len(selected) >= 100 or count >= quota:
                break
            if node["id"] in considered:
                continue
            if group in PATTERNS and not re.search(PATTERNS[group], node["label"], re.I):
                continue
            considered.add(node["id"])
            choices = node.get("metadata", {}).get("pdf_en")
            if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], str)
                    or any(t in choices[0].casefold() for t in ("镜像", "mirror")) or not isinstance(node.get("card"), str)):
                excluded.append({"id": node["id"], "reason": "source_metadata_not_eligible"})
                continue
            try:
                path = contained(vault, choices[0])
                if not path.is_file() or not contained(vault, node["card"]).is_file():
                    excluded.append({"id": node["id"], "reason": "explicit_source_or_card_path_absent"})
                    continue
                digest = sha(path)
                if digest in used_sha:
                    excluded.append({"id": node["id"], "reason": "duplicate_physical_publication"})
                    continue
                result = parse_pdf(str(path), digest, node["label"], node["id"])
            except (OSError, ValueError, RuntimeError) as error:
                excluded.append({"id": node["id"], "reason": "source_preflight_error", "error": type(error).__name__ + ": " + str(error)})
                continue
            selected.append(node["id"])
            used_sha.add(digest)
            parsed[node["id"]] = result
            groups[node["id"]] = group
            count += 1
        if len(selected) >= 100:
            break
    selection = {"old_ids": sorted(old_ids), "new_groups": groups, "excluded": excluded, "selected_ids": selected,
                 "group_counts": dict(Counter(groups.values())), "preflight_seconds": round(time.perf_counter() - started, 3),
                 "selection_plan_sha256": object_hash(plan["selection"]),
                 "selection_basis": "metadata_and_source_eligibility_before_model_inference_not_random_sample",
                 "state": "eligible_sources_ready" if len(selected) == 100 else "insufficient_eligible_sources"}
    write_json(baseline / "selection.json", selection, baseline)
    if len(selected) != 100:
        raise ValueError("fewer than 100 unique readable original sources")
    manifest = freeze(vault, baseline, selected)
    jobs = Jobs(baseline / "state.sqlite", baseline)
    try:
        from .core import parse_key
        for paper in manifest["papers"]:
            result = parsed[paper["id"]]
            path = cache_path(baseline, paper)
            job = jobs.enqueue(paper["id"], "parse", parse_key(paper))
            if jobs.get(job)["state"] != "done":
                if not jobs.claim(job):
                    raise RuntimeError("source import job is busy")
                write_json(path, result, baseline)
                jobs.finish(job, path.relative_to(baseline).as_posix(), sha(path))
    finally:
        jobs.close()
    write_json(baseline / "registry.json", registry(manifest["existing_entities"]), baseline)
    return manifest


def run100(vault: Path, output: Path | None = None) -> dict:
    output = output_root(vault, vault / (output or OUTPUT))
    approval = accepted_gate(output)
    plan = read_json(output / "evaluation-plan.json")
    manifest = prepare100(vault, plan)
    gate = read_json(output / "gate20.json")
    translations = {x["case_id"]: x for x in gate["translations"]}
    cases = {c["id"]: c for c in plan["cases"]}
    query_inputs = [{"label": cases[q["case_id"]]["query"], "text": translations[q["case_id"]]["english"]} for q in plan["retrieval"]]
    receipt = run_local(vault, vault / BASELINE, output, LocalModels(DEFAULT_MODELS, 4), query_inputs=query_inputs)
    receipt["translation_gate"] = approval
    receipt["cohort"] = {"old": 20, "new": 80, "total": len(manifest["papers"])}
    write_json(output / "workflow-run.json", receipt, output)
    return receipt


def diagnose100(vault: Path, output: Path) -> dict:
    """Evaluate original-English processing independently of disabled translation.

    This creates working candidates, never an all-card or publication approval.
    """
    output = output_root(vault, vault / output)
    if output != (vault / ENGLISH_OUTPUT).resolve() or (output / "gate20.json").exists():
        raise ValueError("diagnostic requires its dedicated English output, disjoint from translation gates")
    plan = read_json(output / "evaluation-plan.json")
    if plan.get("query_policy") != "curated_english_control" or not plan.get("full_corpus_requires_pilot_acceptance"):
        raise ValueError("explicit diagnostic policy and separate full-corpus gate required")
    check_inputs(vault, read_json(vault / OLD / "manifest.json"))
    original_plan = read_json(vault / "_working/paper-network-automation-v1/pilot100-local-v1/evaluation-plan-unassisted-v1.json")
    if plan["selection"] != original_plan["selection"] or plan["retrieval"] != original_plan["retrieval"]:
        raise ValueError("diagnostic cohort/English controls differ from frozen baseline")
    manifest = prepare100(vault, plan)
    cases = {c["id"]: c for c in plan["cases"]}
    queries = [{"label": cases[q["case_id"]]["query"], "text": q["explicit_english"]} for q in plan["retrieval"]]
    receipt = run_local(vault, vault / BASELINE, output, LocalModels(DEFAULT_MODELS, 4), query_inputs=queries)
    receipt.update({"query_policy": "curated_english_control", "automatic_translation": "disabled_failed_quality_gate",
                    "cohort": {"old": 20, "new": 80, "total": len(manifest["papers"])}, "full_corpus_unlocked": False,
                    "semantic_acceptance": "pending_main_pilot_content_audit"})
    write_json(output / "workflow-run.json", receipt, output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["gate", "run", "diagnose100"])
    parser.add_argument("--vault", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--translator", choices=["qwen17", "hymt2"], default="qwen17")
    args = parser.parse_args()
    if args.command == "gate":
        value = gate20(args.vault.resolve(), args.output, args.translator)
    elif args.command == "run":
        value = run100(args.vault.resolve(), args.output)
    else:
        if args.output is None:
            parser.error("diagnose100 requires a dedicated --output")
        value = diagnose100(args.vault.resolve(), args.output)
    print(json.dumps({k: value[k] for k in ("seconds", "mean_precision_at_5", "coverage", "cache", "cohort", "translation_gate") if k in value}, ensure_ascii=True))


if __name__ == "__main__":
    main()
