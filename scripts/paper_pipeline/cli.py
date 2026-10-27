"""Run with python -m scripts.paper_pipeline.cli --help."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time
import uuid

from .core import Jobs, cache_path, check_inputs, contained, freeze, object_hash, output_root, parse_manifest, read_json, sha, write_json
from .render import reports
from . import semantic
from .semantic import ENGINE, alternative_query, extract, make_graph, problem_query, registry, validate_spans

COHORT = ["2203.11496", "2205.13542", "2210.10163", "2306.11029", "2509.03961", "2603.17528", "2309.16588", "2602.22394", "2607.19139", "2312.01072", "2201.13259", "2505.11821", "2509.08755", "2602.03719", "2602.16165", "2006.08656", "2308.09564", "2604.21160", "2509.24458", "2606.29093"]


def extraction_key(paper: dict, parsed: dict, vocabulary: dict) -> str:
    return object_hash([paper["pdf_sha256"], object_hash(parsed), vocabulary, ENGINE, sha(Path(semantic.__file__))])


def run(vault: Path, output: Path, workers: int = 2) -> dict:
    output = output_root(vault, output)
    manifest = read_json(output / "manifest.json")
    started = time.perf_counter()
    parsing = parse_manifest(vault, output, manifest, workers)
    if parsing["failed"] or parsing["busy"]:
        receipt = {"state": "parse_incomplete", "parse": parsing, "formal_publication": False}
        write_json(output / "last-run.json", receipt, output)
        return receipt
    vocabulary = registry(manifest["existing_entities"])
    write_json(output / "registry.json", vocabulary, output)
    queue = Jobs(output / "state.sqlite", output)
    results = []
    created = 0
    cached = 0
    semantic_started = time.perf_counter()
    try:
        for paper in manifest["papers"]:
            parsed = read_json(cache_path(output, paper))
            key = extraction_key(paper, parsed, vocabulary)
            job = queue.enqueue(paper["id"], "extract", key)
            path = output / "candidates" / (key + ".json")
            state = queue.get(job)
            valid = False
            if state["state"] == "done":
                try:
                    result = read_json(path)
                    validate_spans(result, parsed)
                    valid = result["registry_hash"] == object_hash(vocabulary) and sha(path) == state["result_hash"]
                except (OSError, ValueError, KeyError, TypeError):
                    valid = False
                if not valid:
                    queue.invalidate(job)
            if valid:
                cached += 1
            else:
                if not queue.claim(job):
                    raise RuntimeError("candidate job busy or failed: " + paper["id"])
                try:
                    result = extract(paper, parsed, vocabulary)
                    validate_spans(result, parsed)
                    write_json(path, result, output)
                    queue.finish(job, path.relative_to(output).as_posix(), sha(path))
                    created += 1
                except Exception as error:
                    queue.fail(job, type(error).__name__ + ": " + str(error))
                    raise
            results.append(result)
    finally:
        queue.close()
    semantic_seconds = round(time.perf_counter() - semantic_started, 3)
    graph = make_graph(results, vocabulary)
    nodes = {n["id"]: n for n in graph["nodes"]}
    for edge in graph["edges"]:
        if edge["directed"] != (nodes[edge["source"]]["type"] == nodes[edge["target"]]["type"]):
            raise ValueError("candidate graph direction/type mismatch")
    queries = {"problems": {e["key"]: problem_query(results, vocabulary, e["key"])
                            for e in vocabulary["entities"] if e["type"] == "question"},
               "alternatives": {key: alternative_query(results, vocabulary, key)
                                for key in ["cross-attention", "contrastive-alignment", "trajectory-balance", "register-tokens"]}}
    coverage = {"cohort_size": len(manifest["papers"]), "processed": len(results),
                "engine": ENGINE, "formal_publication": False,
                "occurrences": sum(len(r["occurrences"]) for r in results),
                "evidence_span_validation": "all_passed_against_frozen_page_cache",
                "semantic_acceptance": "unreviewed_candidates_not_formal_relations",
                "papers_with_problem_candidate": sum(any(o["type"] == "question" and o["role"] != "mention_only" for o in r["occurrences"]) for r in results),
                "papers_with_method_candidate": sum(any(o["type"] == "method" and o["role"] != "mention_only" for o in r["occurrences"]) for r in results),
                "occurrence_roles": dict(Counter(o["role"] for r in results for o in r["occurrences"])),
                "identity_states": dict(Counter(read_json(cache_path(output, p))["identity"]["state"] for p in manifest["papers"])),
                "problem_counts": {k: len(q["rows"]) for k, q in queries["problems"].items()},
                "remaining_gaps": ["no newly verified author list", "dataset usage requires role verification",
                                   "lexical registry does not discover arbitrary unseen mechanisms",
                                   "no GROBID/Docling/GLiNER or LLM adapter executed in this pilot"]}
    changed = {}
    for filename, data in [("candidate-graph.json", graph), ("queries.json", queries), ("coverage.json", coverage)]:
        changed[filename] = write_json(output / filename, data, output)
    changed.update(reports(vault, output, manifest, results, queries, coverage))
    check_inputs(vault, manifest)
    receipt = {"state": "candidate_pipeline_complete", "formal_publication": False,
               "parse_new": parsing["parsed"], "parse_cached": parsing["cached"],
               "extract_new": created, "extract_cached": cached, "parse_seconds": parsing["seconds"],
               "extract_seconds": semantic_seconds, "total_seconds": round(time.perf_counter() - started, 3),
               "protected_inputs_unchanged": True, "artifacts_changed": changed,
               "code_hashes": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
               "artifact_hashes": {name: sha(output / name) for name in ["manifest.json", "registry.json", "candidate-graph.json", "queries.json", "coverage.json", "pilot.md", "index.html", *sorted(name for name in changed if name.endswith(".svg"))]},
               "coverage": coverage}
    write_json(output / "runs" / (uuid.uuid4().hex + ".json"), receipt, output)
    write_json(output / "last-run.json", receipt, output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run", "query"])
    parser.add_argument("--vault", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ids", help="comma-separated existing graph paper IDs; default fixed pilot20 cohort")
    parser.add_argument("--problem")
    parser.add_argument("--method")
    parser.add_argument("--domain")
    args = parser.parse_args()
    vault = args.vault.resolve()
    output = output_root(vault, args.output or vault / "_working/paper-network-automation-v1/pilot20-v1")
    if args.command == "freeze":
        ids = args.ids.split(",") if args.ids else ["paper:arxiv:" + x for x in COHORT]
        manifest = freeze(vault, output, ids)
        summary = {"state": "frozen", "papers": len(manifest["papers"])}
    elif args.command == "run":
        receipt = run(vault, output, args.workers)
        summary = {k: v for k, v in receipt.items() if k not in {"coverage", "artifact_hashes", "parse"}}
        if receipt["state"] != "candidate_pipeline_complete":
            import json
            print(json.dumps(summary, ensure_ascii=True))
            return 2
    else:
        if bool(args.problem) == bool(args.method):
            parser.error("query requires exactly one of --problem or --method")
        manifest = read_json(output / "manifest.json")
        check_inputs(vault, manifest)
        vocabulary = read_json(output / "registry.json")
        queue = Jobs(output / "state.sqlite", output)
        try:
            results = []
            for paper in manifest["papers"]:
                # Bind the selected cache to current source + registry, not an arbitrary historical job.
                parsed = read_json(cache_path(output, paper))
                key = extraction_key(paper, parsed, vocabulary)
                job = queue.get(object_hash([paper["id"], "extract", key]))
                if job["state"] != "done":
                    raise ValueError("current extraction missing")
                path = contained(output, job["result"])
                if sha(path) != job["result_hash"]:
                    raise ValueError("candidate artifact changed")
                result = read_json(path)
                validate_spans(result, parsed)
                results.append(result)
        finally:
            queue.close()
        data = problem_query(results, vocabulary, args.problem, args.domain) if args.problem else alternative_query(results, vocabulary, args.method)
        write_json(output / "last-query.json", data, output)
        summary = {"rows": len(data["rows"]), "query_artifact": str(output / "last-query.json")}
    import json
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
