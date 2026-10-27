"""Transparent candidate extraction and problem-oriented graph queries.

Rules are an inspectable pilot registry, not proof of scientific equivalence.
No rule can publish to the formal network or infer experimental dataset use.
"""
from __future__ import annotations

from collections import defaultdict
import re
import unicodedata
from typing import Any

from .core import object_hash

ENGINE = "source-rules-v4"

METHOD_SCOPES = {
    "cross-attention": {"feature-fusion", "representation-alignment"},
    "bev-representation": {"feature-fusion"},
    "contrastive-alignment": {"representation-alignment"},
    "text-difference": {"feature-fusion"},
    "gated-fusion": {"feature-fusion"},
    "register-tokens": {"attention-sinks", "activation-artifacts"},
    "causal-token-interventions": {"attention-sinks", "activation-artifacts"},
    "hierarchical-credit": {"temporal-credit"},
    "branch-sampling": {"temporal-credit"},
    "trajectory-balance": {"temporal-credit"},
    "reward-shaping": {"temporal-credit"},
    "fine-grained-rewards": {"temporal-credit"},
    "policy-optimization": {"temporal-credit"},
    "broyden": {"fixed-point-convergence"},
    "anderson": {"fixed-point-convergence"},
}


def match_text(literal: str) -> str:
    normalized = unicodedata.normalize("NFKC", literal)
    normalized = re.sub(r"(?<=\w)[\-\u2010\u2011\u00ad]\s*\n\s*(?=\w)", "", normalized)
    return " ".join(normalized.split())


def registry(existing: list[dict[str, Any]]) -> dict[str, Any]:
    definitions = [
        ("question", "multimodal-information", "跨模态信息联合利用", "融合与表征对齐的上层检索入口；两者不等同。", None, []),
        ("question", "feature-fusion", "跨模态特征融合", "整合不同模态的互补特征；排除 diffusion 的词面匹配。", "multimodal-information", [r"\bfusion\b", r"\b(?:lidar|camera|radar|sar|optical|multi.modal|multimodal|text|language)\b"]),
        ("question", "representation-alignment", "跨模态表征对齐", "图文或其他模态表征对应；不自动认定为联合特征融合。", "multimodal-information", [r"\b(?:contrastive|alignment|aligning|align)\b", r"\b(?:text|language)\b", r"\b(?:images?|visual|vision)\b", r"(?:\b(?:vision.language|image.text|text.image|cross.modal)\b.{0,70}\b(?:align\w*|contrastive)\b|\b(?:align\w*|contrastive)\b.{0,70}\b(?:vision.language|image.text|text.image|cross.modal)\b|\balign\w*.{0,50}\b(?:images?|vision|visual)\b.{0,20}\b(?:and|with|to)\b.{0,20}\b(?:text|language)\b|\bcontrastive\b.{0,80}\b(?:images?|visual|vision)\b.{0,40}\b(?:text|language)\b)"]),
        ("question", "temporal-credit", "长期信用分配", "长期或多步决策中的贡献归因；不等同于优化或固定点收敛。", None, [r"\bcredit\s+assignment\b"]),
        ("question", "fixed-point-convergence", "隐式模型的固定点求解", "隐式网络固定点及求解问题；排除一般数学收敛。", None, [r"\b(?:fixed.point|equilibrium)\b", r"\b(?:network|model|layer|solver)\b"]),
        ("question", "attention-sinks", "Attention sink 的处理", "对 attention sink 的直接讨论；不将所有 register 现象归为 sink。", None, [r"\battention\s+sinks?\b"]),
        ("question", "activation-artifacts", "高范数 token 与表征伪影", "视觉/生成模型内部 token 伪影；与 sink 分别检索。", None, [r"\b(?:high.norm|large.norm|artifacts?|artefacts?)\b", r"\b(?:token|tokens|feature|features|activation|activations)\b"]),
        ("method", "cross-attention", "Cross-attention", "跨输入注意力交互机制。", None, [r"\bcross[\s-]*attention\b"]),
        ("method", "bev-representation", "Unified BEV representation", "将不同模态变换到共享 BEV 表征空间；单一 LiDAR 的 BEV 特征不构成该方法。", None, [r"\b(?:bird.s.eye.view|bev)\b", r"\b(?:unified|unifies|shared|common)\b", r"\b(?:representations?|space|features?)\b"]),
        ("method", "contrastive-alignment", "Contrastive representation alignment", "对比式表征对齐方法族。", None, [r"\bcontrastive\b", r"\b(?:images?|visual|vision)\b", r"\b(?:text|language)\b"]),
        ("method", "text-difference", "Text-difference-guided fusion", "文本差异参与融合的候选机制。", None, [r"\btext\s+difference\b"]),
        ("method", "gated-fusion", "Gated fusion", "门控融合方法族。", None, [r"\b(?:gated\s+fusion|gating\s+mechanism|gated\s+mechanism)\b"]),
        ("method", "register-tokens", "Learnable register tokens", "新增可学习 register token 的候选机制；不将已存在的隐式寄存现象视为同一方法。", None, [r"\bregister\s+tokens?\b|\bregisters\b", r"\b(?:learnable|trainable|additional)\b"]),
        ("method", "causal-token-interventions", "Causal token interventions (analysis)", "因果 token 干预的分析方法；不宣称其为 sink 抑制方案。", None, [r"\bcausal\b", r"\b(?:interventions?|interpretability)\b", r"\b(?:tokens?|attention)\b"]),
        ("concept", "implicit-semantic-registers", "Implicit semantic registers", "已存在 token 的功能解释；并不等于显式新增可学习 register token。", None, [r"\bimplicit\s+semantic\s+registers?\b"]),
        ("method", "hierarchical-credit", "Hierarchical credit assignment", "分层决策或分层信用分配机制的候选。", None, [r"\bhierarch(?:ical|y)\b", r"\b(?:credit\s+assignment|reinforcement\s+learning)\b"]),
        ("method", "branch-sampling", "Contrastive branch sampling", "通过分支采样构造对比的候选机制。", None, [r"\bbranch\s+sampling\b"]),
        ("method", "trajectory-balance", "Trajectory balance", "轨迹平衡方法族。", None, [r"\btrajectory\s+balance\b"]),
        ("method", "reward-shaping", "Reward shaping", "奖励塑形方法族。", None, [r"\breward\s+shaping\b"]),
        ("method", "fine-grained-rewards", "Fine-grained reward construction", "细粒度奖励构造方法族。", None, [r"\breward\s+decomposition\b|\bfine.grained\s+reward\b"]),
        ("method", "policy-optimization", "Policy optimization", "策略优化上层方法族。", None, [r"\bpolicy\s+optimi[sz]ation\b|\bppo\b"]),
        ("method", "broyden", "Broyden solver", "Broyden 固定点/求根求解器。", None, [r"\bbroyden"]),
        ("method", "anderson", "Anderson acceleration", "Anderson 求解加速。", None, [r"\banderson\b"]),
        ("task", "3d-detection", "3D object detection", "三维目标检测。", None, [r"\b3d\s+object\s+detection\b"]),
        ("task", "segmentation", "Semantic segmentation", "语义分割任务族。", None, [r"\bsemantic\s+segmentation\b"]),
        ("task", "image-classification", "Image classification", "图像分类任务族。", None, [r"\bimage\s+classification\b"]),
        ("task", "change-detection", "Change detection", "变化检测任务。", None, [r"\bchange\s+detection\b"]),
        ("task", "retrieval", "Cross-modal retrieval", "跨模态检索任务族。", None, [r"\b(?:image.text|text.image|cross.modal|image.to.text|text.to.image)\s+retrieval\b"]),
        ("task", "agent-decisions", "Multi-turn agent decision making", "多轮 agent 决策设置。", None, [r"\b(?:multi.turn|long.horizon)\b", r"\b(?:agents?|decision|reinforcement)\b"]),
        ("dataset", "nuscenes", "nuScenes", "仅记录原文提及；用途另行核对。", None, [r"\bnuscenes\b"]),
        ("dataset", "kitti", "KITTI", "仅记录原文提及；用途另行核对。", None, [r"\bkitti\b"]),
        ("dataset", "imagenet", "ImageNet", "ImageNet 家族名称；不合并具体版本。", None, [r"\bimagenet(?:.1k|.1000)?\b"]),
        ("dataset", "cifar10", "CIFAR-10", "CIFAR-10，保留与 CIFAR-100 的区别。", None, [r"\bcifar[\s-]?10\b"]),
        ("dataset", "cifar100", "CIFAR-100", "CIFAR-100。", None, [r"\bcifar[\s-]?100\b"]),
        ("dataset", "alfworld", "ALFWorld", "仅记录提及。", None, [r"\balfworld\b"]),
        ("dataset", "webshop", "WebShop", "仅记录提及。", None, [r"\bwebshop\b"]),
    ]
    index = defaultdict(list)
    for node in existing:
        index[(node["type"], " ".join(node["label"].casefold().split()))].append(node)
    entities = []
    for kind, slug, label, definition, parent, patterns in definitions:
        matches = index[(kind, label.casefold())]
        reuse = matches[0] if len(matches) == 1 else None
        entities.append({"id": reuse["id"] if reuse else "pilot:" + kind + ":" + slug,
                         "key": slug, "type": kind, "label": label, "definition": definition,
                         "parent_key": parent, "patterns": patterns,
                         "query_problem_scope": sorted(METHOD_SCOPES.get(slug, set())),
                         "identity_basis": "exact_existing_label_candidate" if reuse else "pilot_only_definition",
                         "existing_card": reuse.get("card") if reuse else None,
                         "publication_state": "unpublished_candidate"})
    return {"schema": "paper-pipeline-registry-v1", "engine": ENGINE, "entities": entities}


AUTHOR_CUE = re.compile(r"\b(?:we\s+(?:propose|introduce|present|develop|design|use|employ)|our\s+(?:method|model|approach|framework)|this\s+(?:paper|work))\b", re.I)
HEADING = re.compile(r"^(?:(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+[.)]?)\s*)?(related\s+works?|references|bibliography|introduction|abstract|method(?:s|ology)?|approach|experiments?|conclusions?)\b", re.I)


def extract(paper: dict[str, Any], parsed: dict[str, Any], vocabulary: dict[str, Any]) -> dict[str, Any]:
    if parsed["paper_id"] != paper["id"] or parsed["pdf_sha256"] != paper["pdf_sha256"]:
        raise ValueError("parsed source belongs to another paper")
    survey = bool(re.search(r"\bsurvey\b", paper["title"], re.I))
    entities = vocabulary["entities"]
    compiled = {e["id"]: [re.compile(p, re.I) for p in e["patterns"]] for e in entities}
    occurrences = []
    section = "front"
    domains = set()
    for domain, pattern in {"remote-sensing": r"remote\s+sensing|satellite|optical.sar",
                            "3d-perception": r"lidar|3d\s+(?:object|understanding)|point.vlm|point\s+cloud",
                            "medical": r"medical|radiograph|chest\s+x.ray",
                            "language-agents": r"\bagents?\b|llm",
                            "vision": r"vision\s+transformer|visual\s+feature|image\s+classification|diffusion\s+transformer"}.items():
        if re.search(pattern, paper["title"], re.I):
            domains.add(domain)
    for page in parsed["pages"]:
        for block in page["blocks"]:
            literal = block["text"]
            if page["text"][block["start"]:block["end"]] != literal:
                raise ValueError("literal PDF cache span corrupted")
            text = match_text(literal)
            heading_text = text if len(text) < 100 else match_text(literal.splitlines()[0])
            heading = HEADING.match(heading_text) if len(heading_text) < 100 else None
            if heading:
                section = heading[1].casefold()
                if section.startswith("related"):
                    section = "related work"
            excluded = section in {"references", "bibliography", "related work"}
            if excluded:
                continue
            cue = bool(AUTHOR_CUE.search(text))
            if survey:
                role = "survey_taxonomy_candidate"
            elif cue:
                role = "author_claim_candidate"
            elif page["page"] == 1 and len(text) > 140:
                role = "front_page_context_candidate"
            else:
                role = "mention_only"
            for entity in entities:
                patterns = compiled[entity["id"]]
                if not patterns or not all(p.search(text) for p in patterns):
                    continue
                evidence = {"pdf": paper["pdf"], "pdf_sha256": paper["pdf_sha256"],
                            "page": page["page"], "start": block["start"], "end": block["end"],
                            "quote": literal, "basis": "literal_original_pdf_block",
                            "section": section}
                occurrences.append({"id": "occ:" + object_hash([paper["id"], entity["id"], evidence])[:24],
                                    "paper_id": paper["id"], "entity_id": entity["id"],
                                    "type": entity["type"], "role": role,
                                    "interpretation_state": "unreviewed_candidate", "evidence": evidence})
    by_entity = defaultdict(list)
    priority = {"author_claim_candidate": 0, "front_page_context_candidate": 1,
                "survey_taxonomy_candidate": 2, "mention_only": 3}
    for occurrence in occurrences:
        by_entity[occurrence["entity_id"]].append(occurrence)
    occurrences = [o for values in by_entity.values()
                   for o in sorted(values, key=lambda item: (priority[item["role"]], item["evidence"]["page"], item["evidence"]["start"]))[:4]]
    return {"schema": "paper-pipeline-candidates-v1", "engine": ENGINE,
            "registry_hash": object_hash(vocabulary), "paper_id": paper["id"],
            "title": paper["title"], "card": paper["card"], "pdf": paper["pdf"],
            "source_sha256": paper["pdf_sha256"], "survey": survey,
            "domains": sorted(domains), "occurrences": occurrences,
            "authors": {"state": paper.get("author_state"), "recorded": paper.get("recorded_authors", [])},
            "gaps": ["author ordering/identity not newly extracted", "dataset use is not established by a mention",
                     "semantic mappings require review; no formal graph changes"]}


def validate_spans(result: dict[str, Any], parsed: dict[str, Any]) -> None:
    if result["paper_id"] != parsed["paper_id"] or result["source_sha256"] != parsed["pdf_sha256"]:
        raise ValueError("candidate/source mismatch")
    pages = {page["page"]: page for page in parsed["pages"]}
    for occurrence in result["occurrences"]:
        evidence = occurrence["evidence"]
        page = pages.get(evidence["page"])
        if (occurrence["paper_id"] != result["paper_id"] or evidence["pdf"] != result["pdf"]
                or not isinstance(evidence["start"], int) or isinstance(evidence["start"], bool)
                or not isinstance(evidence["end"], int) or isinstance(evidence["end"], bool)
                or evidence["start"] < 0 or evidence["end"] <= evidence["start"]
                or page is None or evidence["pdf_sha256"] != parsed["pdf_sha256"]
                or page["text"][evidence["start"]:evidence["end"]] != evidence["quote"]):
            raise ValueError("candidate evidence does not match frozen physical page")


def eligible_candidate(occurrence: dict[str, Any]) -> bool:
    """Keep weak first-page method context in extraction, outside default discovery."""
    return occurrence["role"] != "mention_only" and (
        occurrence["type"] != "method" or occurrence["role"] in {
            "author_claim_candidate", "survey_taxonomy_candidate"})


def make_graph(results: list[dict[str, Any]], vocabulary: dict[str, Any]) -> dict[str, Any]:
    nodes = {entity["id"]: {k: entity[k] for k in ["id", "type", "label", "key", "definition"]}
             for entity in vocabulary["entities"]}
    edges = []
    by_key = {entity["key"]: entity for entity in vocabulary["entities"]}
    for entity in vocabulary["entities"]:
        if entity["parent_key"]:
            edges.append({"source": entity["id"], "target": by_key[entity["parent_key"]]["id"],
                          "relation": "subproblem_of", "directed": True, "basis": "pilot_registry_definition"})
    for result in results:
        nodes[result["paper_id"]] = {"id": result["paper_id"], "type": "paper", "label": result["title"],
                                     "card": result["card"], "domains": result["domains"]}
        grouped = defaultdict(list)
        for occurrence in result["occurrences"]:
            if eligible_candidate(occurrence):
                grouped[occurrence["entity_id"]].append(occurrence)
        for identifier, occurrences in grouped.items():
            edges.append({"source": result["paper_id"], "target": identifier,
                          "relation": "source_describes_candidate", "directed": False,
                          "status": "candidate", "occurrence_ids": [o["id"] for o in occurrences]})
    connected = {e[k] for e in edges for k in ("source", "target")}
    connected.update(result["paper_id"] for result in results)
    return {"schema": "paper-pipeline-candidate-graph-v1", "publication_state": "not_formal_graph",
            "nodes": [node for key, node in nodes.items() if key in connected], "edges": edges}


def problem_query(results: list[dict[str, Any]], vocabulary: dict[str, Any], key: str,
                  domain: str | None = None) -> dict[str, Any]:
    by_key = {entity["key"]: entity for entity in vocabulary["entities"]}
    if key not in by_key or by_key[key]["type"] != "question":
        raise ValueError("unknown problem key")
    descendants = {key}
    while True:
        expanded = descendants | {entity["key"] for entity in vocabulary["entities"]
                                  if entity["type"] == "question" and entity["parent_key"] in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    problem_ids = {by_key[k]["id"] for k in descendants}
    entities = {entity["id"]: entity for entity in vocabulary["entities"]}
    rows = []
    for result in results:
        if domain and domain not in result["domains"]:
            continue
        strong = [o for o in result["occurrences"] if eligible_candidate(o)]
        problems = [o for o in strong if o["type"] == "question" and o["entity_id"] in problem_ids]
        if not problems:
            continue
        specific_keys = {entities[o["entity_id"]]["key"] for o in problems}
        methods = [o for o in strong if o["type"] == "method"
                   and specific_keys.intersection(entities[o["entity_id"]]["query_problem_scope"])]
        unique_methods = sorted({o["entity_id"] for o in methods})
        rows.append({"paper_id": result["paper_id"], "title": result["title"], "card": result["card"],
                     "pdf": result["pdf"], "domains": result["domains"], "survey": result["survey"],
                     "tasks": sorted({entities[o["entity_id"]]["label"] for o in strong if o["type"] == "task"}),
                     "method_families": [{"id": i, "key": entities[i]["key"], "label": entities[i]["label"]} for i in unique_methods],
                     "problem_matches": sorted({entities[o["entity_id"]]["key"] for o in problems}),
                     "paths": [[by_key[key]["id"], o["entity_id"], result["paper_id"]] for o in problems],
                     "evidence": [o for o in problems + methods],
                     "match_basis": "survey_taxonomy_candidate" if result["survey"] else "source_bound_problem_method_candidate",
                     "interpretation_state": "unreviewed_candidate_not_adoption_or_transfer_proof"})
    return {"problem": key, "label": by_key[key]["label"], "rows": rows,
            "method_family_count": len({m["id"] for row in rows for m in row["method_families"]}),
            "domain_count": len({d for row in rows for d in row["domains"]})}


def alternative_query(results: list[dict[str, Any]], vocabulary: dict[str, Any], method_key: str) -> dict[str, Any]:
    by_key = {entity["key"]: entity for entity in vocabulary["entities"]}
    if method_key not in by_key or by_key[method_key]["type"] != "method":
        raise ValueError("unknown method key")
    method_id = by_key[method_key]["id"]
    scoped_keys = by_key[method_key]["query_problem_scope"]
    shared = {by_key[key]["id"] for key in scoped_keys
              if any(not row["survey"] and any(m["id"] == method_id for m in row["method_families"])
                     for row in problem_query(results, vocabulary, key)["rows"])}
    query_rows = {}
    for key, entity in by_key.items():
        if entity["type"] != "question" or entity["id"] not in shared:
            continue
        for row in problem_query(results, vocabulary, key)["rows"]:
            if any(m["id"] == method_id for m in row["method_families"]):
                continue
            other = [m for m in row["method_families"] if m["id"] != method_id]
            if not other or row["survey"]:
                continue
            row = dict(row, alternative_families=other, via_problem=key,
                       match_basis="candidate_alternative_via_shared_specific_problem")
            query_rows[(row["paper_id"], key)] = row
    return {"method": method_key, "label": by_key[method_key]["label"],
            "rows": list(query_rows.values()), "state": "candidate_discovery_not_validated_transfer"}
