"""Local-model extraction, existing-entity suggestions and passage retrieval."""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from .core import Jobs, cache_path, check_inputs, checked_output, contained, object_hash, output_root, read_json, sha, write_json
from .local_models import DEFAULT_MODELS, LABELS, LocalModels
from .render import md_text, write_text
from .semantic import AUTHOR_CUE, HEADING, match_text

ENGINE = "gliner25-e5-local-v1"
DEFAULT_BASELINE = "_working/paper-network-automation-v1/pilot20-v1"
DEFAULT_OUTPUT = "_working/paper-network-automation-v1/pilot20-local-models-v1"
QUERIES = [{"label": "跨模态特征融合", "text": "multimodal feature fusion"},
           {"label": "长时序决策的信用分配与稀疏奖励", "text": "temporal credit assignment sparse rewards in long horizon decision making"},
           {"label": "Attention sink 的机制分析与处理", "text": "attention sink mechanisms and mitigation"},
           {"label": "深度平衡模型的固定点求解", "text": "deep equilibrium fixed point solver"}]


def select_passages(paper: dict, parsed: dict, limit: int = 12) -> dict:
    if not 1 <= limit <= 32:
        raise ValueError("passages per paper must be between 1 and 32")
    if parsed["paper_id"] != paper["id"] or parsed["pdf_sha256"] != paper["pdf_sha256"]:
        raise ValueError("parsed source identity mismatch")
    candidates = []
    section = "front"
    for page in parsed["pages"]:
        for block in page["blocks"]:
            if page["text"][block["start"]:block["end"]] != block["text"]:
                raise ValueError("source cache span mismatch")
            first = match_text(block["text"].splitlines()[0]) if block["text"].splitlines() else ""
            heading = HEADING.match(first) if len(first) < 100 else None
            if heading:
                section = heading[1].casefold()
            if section.startswith("related") or section in {"references", "bibliography"}:
                continue
            words = list(re.finditer(r"\S+", block["text"]))
            for offset in range(0, len(words), 160):
                chunk = words[offset:offset + 160]
                start, end = block["start"] + chunk[0].start(), block["start"] + chunk[-1].end()
                text = page["text"][start:end]
                if len(text) < 90 or sum(c.isalpha() for c in text) / len(text) < 0.45:
                    continue
                priority = (0 if section == "abstract" else 1 if section.startswith(("method", "approach"))
                            else 2 if AUTHOR_CUE.search(match_text(text)) else 3 if page["page"] == 1 else 4)
                item = {"id": "span:" + object_hash([paper["id"], page["page"], start, end])[:24],
                        "paper_id": paper["id"], "page": page["page"], "start": start, "end": end,
                        "text": text, "section": section, "pdf": paper["pdf"], "pdf_sha256": paper["pdf_sha256"]}
                candidates.append((priority, page["page"], start, item))
    chosen = sorted(candidates, key=lambda x: x[:3])[:limit]
    return {"passages": [x[3] for x in sorted(chosen, key=lambda x: x[1:3])],
            "eligible_windows": len(candidates), "unselected_windows": max(0, len(candidates) - limit),
            "scope": "selected_passages_not_exhaustive_full_paper"}


def bind_occurrences(paper: dict, passage: dict, raw: dict) -> list[dict]:
    occurrences = []
    for kind, values in raw.get("entities", {}).items():
        if kind not in LABELS:
            raise ValueError("unexpected extraction type")
        for value in values:
            start, end = value["start"], value["end"]
            score = float(value["confidence"])
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(passage["text"])
                    or passage["text"][start:end] != value["text"] or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("local model span/score does not match source")
            absolute = passage["start"] + start
            evidence = {"pdf": paper["pdf"], "pdf_sha256": paper["pdf_sha256"], "page": passage["page"],
                        "start": absolute, "end": passage["start"] + end, "quote": value["text"]}
            occurrences.append({"id": "mention:" + object_hash([paper["id"], kind, evidence])[:24],
                                "paper_id": paper["id"], "type": kind, "text": value["text"],
                                "model_confidence": score, "role": "unresolved_model_mention",
                                "interpretation_state": "unreviewed_candidate", "context_span_id": passage["id"],
                                "evidence": evidence})
    return occurrences


def validate_result(result: dict, paper: dict, parsed: dict, selected: dict) -> None:
    if result["paper_id"] != paper["id"] or result["source_sha256"] != paper["pdf_sha256"]:
        raise ValueError("local result belongs to different paper")
    pages = {p["page"]: p["text"] for p in parsed["pages"]}
    contexts = {p["id"]: p for p in selected["passages"]}
    for item in result["occurrences"]:
        evidence = item["evidence"]
        context = contexts[item["context_span_id"]]
        if (item["paper_id"] != paper["id"] or evidence["pdf"] != paper["pdf"]
                or evidence["pdf_sha256"] != paper["pdf_sha256"] or evidence["page"] != context["page"]
                or not context["start"] <= evidence["start"] < evidence["end"] <= context["end"]
                or pages[evidence["page"]][evidence["start"]:evidence["end"]] != evidence["quote"]):
            raise ValueError("local result evidence mismatch")
    for rejected in result.get("rejected_occurrences", []):
        context = contexts[rejected["context_span_id"]]
        if (rejected["state"] != "rejected_model_span" or rejected["paper_id"] != paper["id"]
                or rejected["source_context"] != context
                or pages[context["page"]][context["start"]:context["end"]] != context["text"]):
            raise ValueError("rejected model span lost its original context")


def extract_paper(models, paper: dict, selected: dict, threshold: float) -> dict:
    occurrences, rejected = {}, []
    for passage in selected["passages"]:
        raw = models.extract(passage["text"], threshold)
        for kind, values in raw.get("entities", {}).items():
            for value in values:
                try:
                    items = bind_occurrences(paper, passage, {"entities": {kind: [value]}})
                except (ValueError, TypeError, KeyError) as error:
                    model_value = {k: value.get(k) for k in ("text", "start", "end", "confidence")}
                    model_value = {k: str(v) if isinstance(v, float) and not math.isfinite(v) else v for k, v in model_value.items()}
                    rejected.append({"paper_id": paper["id"], "state": "rejected_model_span", "type": kind,
                                     "context_span_id": passage["id"], "source_context": passage,
                                     "model_value": model_value, "reason": type(error).__name__ + ": " + str(error)})
                    continue
                for item in items:
                    occurrences[item["id"]] = item
    return {"schema": "paper-local-candidates-v1", "paper_id": paper["id"], "source_sha256": paper["pdf_sha256"],
            "selection": selected, "occurrences": list(occurrences.values()), "rejected_occurrences": rejected}


def extraction_key(paper: dict, selected: dict, identity: dict, threshold: float) -> str:
    return object_hash([ENGINE, selected, paper, identity, threshold,
                        sha(Path(__file__).with_name("local_models.py")),
                        *[inspect.getsource(f) for f in (extract_paper, bind_occurrences, validate_result)]])


def entity_documents(manifest: dict, registry: dict) -> list[dict]:
    entities = {e["id"]: dict(e) for e in manifest["existing_entities"]}
    for entity in registry["entities"]:
        entities[entity["id"]] = dict(entity)
    rows = []
    for key, entity in sorted(entities.items()):
        label = entity["label"]
        if entity["type"] not in LABELS or not isinstance(label, str) or not label.strip():
            continue
        if re.fullmatch(r"(?:method|task|question|dataset)[-:]canonical[-:]v\d+[-:].*", label, re.I):
            continue
        rows.append({"id": key, "type": entity["type"], "label": label, "card": entity.get("card") or entity.get("existing_card"),
                     "text": entity["type"] + ": " + label + ". " + entity.get("definition", "")})
    return rows


def nearest_entities(occurrences: list[dict], entities: list[dict], entity_vectors, mention_vectors, top: int = 3) -> list[dict]:
    import numpy as np
    rows = []
    for item, vector in zip(occurrences, mention_vectors, strict=True):
        eligible = [i for i, entity in enumerate(entities) if entity["type"] == item["type"]]
        scores = entity_vectors[eligible] @ vector
        order = np.argsort(-scores, kind="stable")[:top]
        normalized = match_text(item["text"]).casefold()
        exact = [entities[i]["id"] for i in eligible if match_text(entities[i]["label"]).casefold() == normalized]
        suggestions = [{"entity_id": entities[eligible[j]]["id"], "label": entities[eligible[j]]["label"],
                        "cosine_similarity": round(float(scores[j]), 6)} for j in order]
        rows.append({**item, "exact_name_candidate_ids": exact,
                     "identity_state": "exact_name_candidate" if len(exact) == 1 else "unresolved",
                     "suggestions": suggestions, "canonical_assignment": None,
                     "match_basis": "embedding_retrieval_not_identity_or_adoption_proof"})
    return rows


def retrieve(query: str, documents: list[dict], vectors, query_vector, paper_map: dict, top: int = 5) -> dict:
    import numpy as np
    scores = vectors @ query_vector
    grouped = {}
    for index in np.argsort(-scores, kind="stable"):
        passage = documents[int(index)]
        identifier = passage["paper_id"]
        if identifier not in grouped:
            paper = paper_map[identifier]
            grouped[identifier] = {"paper_id": identifier, "title": paper["title"], "card": paper["card"],
                                   "score": round(float(scores[index]), 6), "passages": []}
        row = grouped[identifier]
        if len(row["passages"]) < 2:
            row["passages"].append({**passage, "cosine_similarity": round(float(scores[index]), 6)})
    return {"query": query, "rows": list(grouped.values())[:top],
            "state": "ranked_passage_candidates_not_semantic_graph_edges"}


def context_methods(row: dict, mentions: list[dict]) -> list[str]:
    contexts = {p["id"] for p in row["passages"]}
    names = {}
    for item in mentions:
        if item["type"] == "method" and item["context_span_id"] in contexts:
            names.setdefault(match_text(item["text"]).casefold(), item["text"])
    return list(names.values())[:5]


def write_vectors(path: Path, values, boundary: Path) -> None:
    import numpy as np
    checked_output(path, boundary)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".local-vectors-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stage(queue: Jobs, output: Path, paper_id: str, name: str, key: str, produce,
          validate=lambda value: None, retry_failed: bool = False) -> tuple[dict, bool]:
    identifier = queue.enqueue(paper_id, name, key)
    path = output / "cache" / name / (key + ".json")
    job = queue.get(identifier)
    if job["state"] == "failed" and retry_failed:
        write_json(output / "failures" / (uuid.uuid4().hex + ".json"),
                   {"job_id": identifier, "attempts": job["attempts"], "error": job["error"]}, output)
        queue.connection.execute("""UPDATE jobs SET state='pending', error=NULL, result=NULL, result_hash=NULL,
            lease_until=NULL, lease_token=NULL WHERE id=? AND state='failed' AND attempts=?""", (identifier, job["attempts"]))
        job = queue.get(identifier)
    if job["state"] == "done":
        try:
            if sha(path) != job["result_hash"]:
                raise ValueError("cached artifact hash changed")
            data = read_json(path)
            validate(data)
            return data, True
        except (OSError, ValueError, KeyError, TypeError):
            queue.invalidate(identifier)
    if not queue.claim(identifier, lease_seconds=3600):
        raise RuntimeError("local stage is busy or failed; inspect state.sqlite: " + name)
    try:
        data = produce()
        validate(data)
        write_json(path, data, output)
        queue.finish(identifier, path.relative_to(output).as_posix(), sha(path))
        return data, False
    except Exception as error:
        queue.fail(identifier, type(error).__name__ + ": " + str(error))
        raise


def render_report(output: Path, manifest: dict, records: list[dict], queries: list[dict], summary: dict) -> bool:
    kinds = {"method": "方法", "question": "问题表述", "task": "任务", "dataset": "数据集"}
    lines = [f"# 本地小模型：{len(manifest['papers'])} 篇论文试验", "", "> 原文候选与语义检索；采用角色、实体身份和适用条件仍待核定。", "",
             f"处理 {summary['processed']} 篇；抽取 {summary['occurrences']} 条原文表述；现有实体索引 {summary['indexed_entities']} 个。", "",
             "GLiNER2.5 Small 提取英文片段，multilingual-e5-small 检索段落与已有实体。分数仅用于排序，不自动合并节点或增加语义边。", "",
             "## 按共享问题找论文", ""]
    if summary.get("rejected_model_spans", 0):
        lines[6:6] = [f"有 {summary['rejected_model_spans']} 条模型词段未通过原文位置校验，已隔离在 records.json 的 rejected_occurrences；没有用来建立关系。", ""]
    by_paper = {p["id"]: p for p in manifest["papers"]}
    mentions = {r["paper_id"]: r["occurrences"] for r in records}
    for query in queries:
        lines.extend(["### " + md_text(query.get("label", query["query"])), "", "实际检索词：" + md_text(query["query"]), "",
                      "| 论文 | 命中段落的方法表述候选 | 英文来源 |", "| --- | --- | --- |"])
        for row in query["rows"]:
            methods = context_methods(row, mentions[row["paper_id"]])
            evidence = row["passages"][0]
            lines.append("| [[" + row["card"] + "\\|" + md_text(row["title"]) + "]] | " + md_text("；".join(methods) or "当前片段未抽取到方法")
                         + " | [[" + evidence["pdf"] + "#page=" + str(evidence["page"]) + "\\|物理页 " + str(evidence["page"]) + "]] |")
        lines.extend(["", "排序依据是段落相似度；需打开原文区分提出、采用、比较、分析和仅提及。", ""])
    lines.extend(["## 每篇的候选与实体匹配", "", "下表每篇展示前 16 条；完整记录、分数与字符位置保存在 records.json。", ""])
    for record in records:
        paper = by_paper[record["paper_id"]]
        lines.extend(["### " + md_text(paper["title"]), "", "[[" + paper["card"] + "|打开原卡]]", "",
                      "| 原文表述 | 类型 | 已有实体候选 | 来源 |", "| --- | --- | --- | --- |"])
        for item in record["occurrences"][:16]:
            names = "；".join(x["label"] for x in item["suggestions"][:2]) or "无同类实体候选"
            lines.append("| " + md_text(item["text"]) + " | " + kinds[item["type"]] + " | " + md_text(names) + " | [[" + paper["pdf"]
                         + "#page=" + str(item["evidence"]["page"]) + "\\|原文]] |")
        if not record["occurrences"]:
            lines.append("当前所选片段未抽取到候选；保留缺口。")
        lines.append("")
    lines.extend(["## 范围", "", "每篇最多选取固定数量的英文片段；不是完整科学审读。作者未重新抽取；数据集提及不代表训练或评价采用。", "",
                  "查询结果和已有实体建议没有自动写入正式图谱；新表述保留为论文内记录。两个负对照照常参加检索，用于检查误召回。", ""])
    return write_text(output / "report.md", "\n".join(lines), output)


def run(vault: Path, baseline: Path, output: Path, models: LocalModels, limit: int = 12,
        threshold: float = 0.45, retry_failed: bool = False, query_inputs: list[dict] | None = None) -> dict:
    import numpy as np
    query_inputs = QUERIES if query_inputs is None else query_inputs
    if not query_inputs or any(not q.get("label") or not q.get("text") for q in query_inputs):
        raise ValueError("explicit query labels and search wording are required")
    started = time.perf_counter()
    baseline, output = output_root(vault, baseline), output_root(vault, output)
    if output == baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError("local output must be separate from baseline")
    manifest = read_json(baseline / "manifest.json")
    check_inputs(vault, manifest)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("local frozen sources differ")
    write_json(manifest_path, manifest, output)
    write_json(output / "models.json", models.identity, output)
    queue = Jobs(output / "state.sqlite", output)
    records, documents, current_keys = [], [], []
    counts = Counter()
    try:
        for paper in manifest["papers"]:
            parsed = read_json(cache_path(baseline, paper))
            selected = select_passages(paper, parsed, limit)
            documents.extend(selected["passages"])
            key = extraction_key(paper, selected, models.identity, threshold)
            result, cached = stage(queue, output, paper["id"], "local_extract", key,
                                   lambda p=paper, s=selected: extract_paper(models, p, s, threshold),
                                   lambda value, p=paper, d=parsed, s=selected: validate_result(value, p, d, s), retry_failed)
            records.append(result)
            current_keys.append(key)
            counts["extraction_cached" if cached else "extraction_new"] += 1
            print(json.dumps({"paper_id": paper["id"], "state": "cached" if cached else "extracted", "occurrences": len(result["occurrences"])}, ensure_ascii=True), flush=True)
        entities = entity_documents(manifest, read_json(baseline / "registry.json"))
        occurrences = [item for record in records for item in record["occurrences"]]
        passage_count, entity_count = len(documents), len(entities)
        texts = [p["text"] for p in documents] + [e["text"] for e in entities]
        embed_key = object_hash([ENGINE, documents, entities, models.identity, inspect.getsource(LocalModels.encode)])
        vector_path = output / "cache" / (embed_key + ".npy")
        def make_index():
            values = models.encode(texts, "passage")
            write_vectors(vector_path, values, output)
            return {"vectors": vector_path.relative_to(output).as_posix(), "sha256": sha(vector_path),
                    "rows": len(texts), "dimension": 384, "documents": documents, "entities": entities}
        def validate_index(value):
            if (value["rows"] != len(texts) or value["documents"] != documents or value["entities"] != entities
                    or sha(contained(output, value["vectors"])) != value["sha256"]):
                raise ValueError("embedding index cache mismatch")
        index, cached = stage(queue, output, "cohort", "embedding_index", embed_key, make_index, validate_index, retry_failed)
        counts["index_cached" if cached else "index_new"] += 1
        vectors = np.load(contained(output, index["vectors"]), allow_pickle=False)
        if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
            raise ValueError("invalid persisted embeddings")
        link_key = object_hash([current_keys, embed_key, index["sha256"], inspect.getsource(nearest_entities),
                                inspect.getsource(match_text), inspect.getsource(LocalModels.encode)])
        def make_links():
            mention_vectors = models.encode([x["type"] + ": " + x["text"] for x in occurrences], "query")
            return {"items": nearest_entities(occurrences, entities, vectors[passage_count:], mention_vectors)}
        linked, cached = stage(queue, output, "cohort", "entity_suggestions", link_key, make_links, retry_failed=retry_failed)
        counts["links_cached" if cached else "links_new"] += 1
        by_id = {item["id"]: item for item in linked["items"]}
        records = [{**record, "occurrences": [by_id[o["id"]] for o in record["occurrences"]]} for record in records]
        paper_map = {p["id"]: p for p in manifest["papers"]}
        query_key = object_hash([query_inputs, embed_key, index["sha256"], inspect.getsource(retrieve), inspect.getsource(LocalModels.encode)])
        def make_queries():
            queries = models.encode([q["text"] for q in query_inputs], "query")
            return {"queries": [{**retrieve(q["text"], documents, vectors[:passage_count], v, paper_map), "label": q["label"]}
                                for q, v in zip(query_inputs, queries, strict=True)]}
        queries, cached = stage(queue, output, "cohort", "retrieval", query_key, make_queries, retry_failed=retry_failed)
        counts["queries_cached" if cached else "queries_new"] += 1
    finally:
        queue.close()
    summary = {"engine": ENGINE, "processed": len(records), "selected_passages": passage_count, "indexed_entities": entity_count,
               "occurrences": len(occurrences), "papers_with_method_mentions": sum(any(o["type"] == "method" for o in r["occurrences"]) for r in records),
               "occurrence_types": dict(Counter(o["type"] for o in occurrences)), "selection_limit": limit, "extraction_threshold": threshold,
               "rejected_model_spans": sum(len(r.get("rejected_occurrences", [])) for r in records),
               "selection_uncovered_windows": sum(r["selection"]["unselected_windows"] for r in records),
               "semantic_acceptance": "unreviewed_mentions_and_retrieval_suggestions", "formal_publication": False}
    changed = {name: write_json(output / name, value, output) for name, value in
               [("records.json", records), ("queries.json", queries), ("coverage.json", summary), ("index.json", index)]}
    changed["report.md"] = render_report(output, manifest, records, queries["queries"], summary)
    check_inputs(vault, manifest)
    receipt = {"state": "local_model_pipeline_complete", "coverage": summary, "cache": dict(counts),
               "model_calls": {"extraction": models.extraction_calls, "embedding_batches": models.embedding_calls},
               "runtime_models": models.runtime_evidence() if hasattr(models, "runtime_evidence") else {},
               "model_identity": models.identity, "protected_inputs_unchanged": True, "formal_publication": False,
               "artifacts_changed": changed, "artifact_hashes": {name: sha(output / name) for name in changed},
               "code_hashes": {name: sha(Path(__file__).with_name(name)) for name in ("local_models.py", "local_pipeline.py")},
               "seconds": round(time.perf_counter() - started, 3)}
    write_json(output / "runs" / (uuid.uuid4().hex + ".json"), receipt, output)
    write_json(output / "last-run.json", receipt, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "query"])
    parser.add_argument("--vault", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--passages", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--query")
    parser.add_argument("--search-query", help="optional explicit English retrieval wording; original query remains the label")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--retry-failed", action="store_true", help="retry failed stages and retain their previous errors")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or not 1 <= args.top <= 20:
        parser.error("invalid extraction threshold or result count")
    vault = args.vault.resolve()
    output = output_root(vault, args.output or vault / DEFAULT_OUTPUT)
    models = LocalModels(args.models, args.threads)
    if args.command == "run":
        receipt = run(vault, args.baseline or vault / DEFAULT_BASELINE, output, models, args.passages, args.threshold, args.retry_failed)
        summary = {k: receipt[k] for k in ("state", "coverage", "cache", "model_calls", "seconds", "protected_inputs_unchanged")}
    else:
        if not args.query:
            parser.error("query text is required")
        import numpy as np
        manifest = read_json(output / "manifest.json")
        check_inputs(vault, manifest)
        if read_json(output / "models.json") != models.identity:
            raise ValueError("query model identity differs from frozen index")
        index = read_json(output / "index.json")
        path = contained(output, index["vectors"])
        if sha(path) != index["sha256"]:
            raise ValueError("query index changed")
        vectors = np.load(path, allow_pickle=False)[:len(index["documents"])]
        wording = args.search_query or args.query
        data = {**retrieve(wording, index["documents"], vectors, models.encode([wording], "query")[0],
                           {p["id"]: p for p in manifest["papers"]}, args.top), "label": args.query}
        write_json(output / "last-query.json", data, output)
        summary = {"rows": len(data["rows"]), "result": str(output / "last-query.json")}
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
