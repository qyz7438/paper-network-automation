"""Readable, offline candidate reports; no JavaScript or formal note edits."""
from __future__ import annotations

from html import escape
from pathlib import Path
from textwrap import wrap
from urllib.parse import quote

from .core import checked_output


def uri(vault: Path, relative: str) -> str:
    return "obsidian://open?vault=" + quote(vault.name, safe="") + "&file=" + quote(relative, safe="")


def md_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value.replace("|", "\\|").replace("\n", " ").replace("[", "\\[").replace("]", "\\]")


def diagram(query: dict) -> list[str]:
    """Draw actual query candidates; arrows are not invented method/problem facts."""
    def label(value: str, limit: int = 65) -> str:
        value = " ".join(value.split())
        value = value if len(value) <= limit else value[:limit - 1] + "…"
        return escape(value, quote=True).replace("`", "&#96;").replace("\\", "&#92;")
    lines = ["```mermaid", "flowchart LR", 'q{"' + label(query["label"]) + '"}']
    methods, tasks = {}, {}
    for index, row in enumerate(query["rows"]):
        paper = "p" + str(index)
        lines.extend([paper + '["' + label(row["title"]) + '"]', "q --- " + paper, "class " + paper + " paper;"])
        for method in row["method_families"]:
            identifier = methods.get(method["id"])
            if identifier is None:
                identifier = "m" + str(len(methods))
                methods[method["id"]] = identifier
                lines.extend([identifier + '{{"' + label(method["label"], 40) + '"}}', "class " + identifier + " method;"])
            lines.append(identifier + " --- " + paper)
        for task in row.get("tasks", []):
            identifier = tasks.get(task)
            if identifier is None:
                identifier = "t" + str(len(tasks))
                tasks[task] = identifier
                lines.extend([identifier + '(["' + label(task, 40) + '"])', "class " + identifier + " task;"])
            lines.append(identifier + " --- " + paper)
    lines.extend(["class q problem;", "classDef problem fill:#8664ca,color:#fff,stroke:#bda5e5;",
                  "classDef method fill:#258d82,color:#fff,stroke:#88c7c0;",
                  "classDef paper fill:#4375b2,color:#fff,stroke:#a4c0e4;",
                  "classDef task fill:#d39325,color:#fff,stroke:#f1cd83;", "```", ""])
    return lines


def svg_diagram(vault: Path, query: dict) -> str:
    """Static, source-bound graph that Obsidian can render without Mermaid."""
    def xml(value: str) -> str:
        clean = "".join(c for c in value if c in "\t\n\r" or 0x20 <= ord(c) <= 0xD7FF
                        or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF)
        return escape(clean, quote=True)
    rows = query["rows"]
    method_labels = {m["id"]: m["label"] for r in rows for m in r["method_families"]}
    task_labels = list(dict.fromkeys(t for r in rows for t in r.get("tasks", [])))
    height = max(420, 100 + 110 * max(len(rows), len(method_labels), len(task_labels), 1))
    nodes = [("q", "problem", query["label"], 120, height / 2, None)]
    methods, tasks, edges = {}, {}, []
    def positions(count: int) -> list[float]:
        return [height * (i + 1) / (count + 1) for i in range(count)]
    for i, ((key, name), y) in enumerate(zip(method_labels.items(), positions(len(method_labels)))):
        methods[key] = "m" + str(i)
        nodes.append((methods[key], "method", name, 440, y, None))
    for i, (name, y) in enumerate(zip(task_labels, positions(len(task_labels)))):
        tasks[name] = "t" + str(i)
        nodes.append((tasks[name], "task", name, 1250, y, None))
    for i, (row, y) in enumerate(zip(rows, positions(len(rows)))):
        key = "p" + str(i)
        nodes.append((key, "paper", row["title"], 865, y, uri(vault, row["card"])))
        edges.append(("q", key))
        edges.extend((methods[m["id"]], key) for m in row["method_families"])
        edges.extend((tasks[t], key) for t in row.get("tasks", []))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 {height}" role="img">',
             '<title>' + xml(query["label"]) + ' — source-bound candidates</title>',
             '<rect width="100%" height="100%" fill="#17212b"/>',
             '<g stroke="#8799ad" stroke-width="1.4" opacity="0.65">']
    centers = {n[0]: (n[3], n[4]) for n in nodes}
    for a, b in edges:
        x1, y1 = centers[a]
        x2, y2 = centers[b]
        parts.append(f'<line data-from="{a}" data-to="{b}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    parts.append('</g>')
    fills = {"problem": "#7750ad", "method": "#176f67", "task": "#925f13", "paper": "#285989"}
    for key, kind, name, x, y, link in nodes:
        if link:
            parts.append('<a href="' + xml(link) + '">')
        parts.append(f'<g id="{key}" data-kind="{kind}" fill="{fills[kind]}" stroke="#bccbda"><title>' + xml(name) + '</title>')
        if kind == "problem":
            parts.append(f'<polygon points="{x-100},{y} {x},{y-65} {x+100},{y} {x},{y+65}"/>')
        elif kind == "method":
            parts.append(f'<polygon points="{x-135},{y} {x-105},{y-45} {x+105},{y-45} {x+135},{y} {x+105},{y+45} {x-105},{y+45}"/>')
        else:
            width, node_height, radius = (390, 84, 3) if kind == "paper" else (230, 78, 39)
            parts.append(f'<rect x="{x-width/2}" y="{y-node_height/2}" width="{width}" height="{node_height}" rx="{radius}"/>')
        words = " ".join(name.split())
        lines = wrap(words, width=48 if kind == "paper" else 26) or [""]
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1][:-1] + "…"
        parts.append('<text fill="#fff" stroke="none" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14">')
        for i, line in enumerate(lines):
            parts.append(f'<tspan x="{x}" y="{y+(i-(len(lines)-1)/2)*18+5}">' + xml(line) + '</tspan>')
        parts.append('</text></g>')
        if link:
            parts.append('</a>')
    if not rows:
        parts.append(f'<text x="700" y="{height/2}" fill="#fff" text-anchor="middle" font-family="system-ui,sans-serif" font-size="20">当前样本没有候选论文</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def write_text(path: Path, text: str, boundary: Path) -> bool:
    checked_output(path, boundary)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def reports(vault: Path, output: Path, manifest: dict, results: list[dict], queries: dict, coverage: dict) -> dict:
    count = len(manifest["papers"])
    changed = {}
    lines = [f"# {count} 篇论文自动解析与跨领域问题检索试验", "",
             "> 独立候选产物：尚未发布到正式图谱。原文摘录命中不等于语义关系已确认。", "",
             f"输入 {len(manifest['papers'])} 篇；处理结果 {len(results)} 篇；规则引擎 {coverage['engine']}。", "",
             "本试验保留不同含义的问题：特征融合、表征对齐、长期信用分配、固定点求解、attention sink、高范数 token 伪影。", "",
             "## 从问题找不同思路", ""]
    html_parts = [f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>论文网络 {count} 篇试验</title>",
                  "<style>body{background:#f4f6f8;color:#17212b;font:16px/1.65 system-ui;max-width:1240px;margin:auto;padding:28px}h1{font-size:28px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #dde3eb;text-align:left}summary{cursor:pointer;font-size:20px;font-weight:650;padding:14px;background:#e8eff9}details{margin:20px 0}a{color:#1955a5}small{color:#596578}.badge{background:#fff1ce;border-radius:5px;padding:3px 8px}blockquote{border-left:4px solid #a8bdd5;margin:10px;padding:10px;background:white;white-space:pre-wrap}svg{max-width:100%;height:auto;background:white;border-radius:8px}</style>",
                  f"<h1>论文网络：{count} 篇自动化试验</h1><p class='badge'>候选检索 / 未写入正式图谱 / 规则引擎</p>",
                  f"<p>冻结 {len(manifest['papers'])} 篇，输出 {len(results)} 篇。点击论文打开 Obsidian 原卡，展开证据核对英文原文。</p>",
                  "<svg viewBox='0 0 1050 125' role='img' aria-label='问题、方法、任务与论文采用不同图形'><path d='M120 60H300M400 60H600M700 60H900' stroke='#9aabc1' stroke-width='2'/><polygon points='25,60 80,15 135,60 80,105' fill='#8664ca'/><text x='80' y='65' fill='white' text-anchor='middle'>问题</text><polygon points='300,25 375,25 405,60 375,95 300,95 275,60' fill='#258d82'/><text x='340' y='65' fill='white' text-anchor='middle'>方法</text><polygon points='610,15 665,105 555,105' fill='#d39325'/><text x='610' y='78' fill='white' text-anchor='middle'>任务</text><rect x='880' y='25' width='130' height='70' rx='8' fill='#4375b2'/><text x='945' y='65' fill='white' text-anchor='middle'>论文</text></svg>"]
    for key, query in queries["problems"].items():
        lines.extend(["### " + query["label"], "", f"论文 {len(query['rows'])}；方法族 {query['method_family_count']}；候选领域 {query['domain_count']}。", ""])
        if key in {"feature-fusion", "temporal-credit"}:
            lines.extend(["候选关联预览：菱形为问题、六边形为方法、圆角节点为任务、矩形为论文；表格保留完整名称与原卡链接。", ""])
            graph_name = key + ".svg"
            changed[graph_name] = write_text(output / graph_name, svg_diagram(vault, query), output)
            graph_path = (output / graph_name).relative_to(vault).as_posix()
            lines.extend(["[[" + graph_path + "|打开大图]]；下方表格可打开原论文卡。", "", "![[" + graph_path + "]]", ""])
        lines.extend(["| 论文 | 方法族候选 | 任务／领域线索 | 依据类型 |", "| --- | --- | --- | --- |"])
        html_parts.append("<details open><summary>" + escape(query["label"]) + " · " + str(len(query["rows"])) + " 篇</summary><table><tr><th>论文</th><th>不同思路候选</th><th>领域线索</th><th>来源</th></tr>")
        for row in query["rows"]:
            names = "；".join(m["label"] for m in row["method_families"]) or "具体方法待核对"
            facets = ", ".join(row.get("tasks", []) + row["domains"])
            basis = "综述分类候选" if row["survey"] else "原文映射候选"
            lines.append("| [[" + row["card"] + "\\|" + md_text(row["title"]) + "]] | " + md_text(names) + " | " + md_text(facets) + " | " + basis + " |")
            html_parts.append("<tr><td><a href='" + escape(uri(vault, row["card"]), quote=True) + "'>" + escape(row["title"]) + "</a></td><td>" + escape(names) + "</td><td>" + escape(facets) + "</td><td>" + basis + "</td></tr>")
        lines.append("")
        html_parts.append("</table></details>")
    lines.extend(["## 从方法找其他解法", "", "下列结果经共享问题连接，排除仍使用输入方法族的论文；结果是候选发现，不证明可直接迁移。", ""])
    html_parts.append("<h2>从方法找其他解法</h2><p>以共同问题为桥梁，按其他方法族展开；候选不等于已验证替代。</p>")
    for key, query in queries["alternatives"].items():
        lines.extend(["### " + query["label"], ""])
        html_parts.append("<details><summary>" + escape(query["label"]) + " · " + str(len(query["rows"])) + " 条候选路径</summary><ul>")
        for row in query["rows"]:
            families = "、".join(m["label"] for m in row["alternative_families"])
            lines.append("- [[" + row["card"] + "|" + md_text(row["title"]) + "]]；经 `" + row["via_problem"] + "` → " + families)
            html_parts.append("<li><a href='" + escape(uri(vault, row["card"]), quote=True) + "'>" + escape(row["title"]) + "</a> — " + escape(row["via_problem"] + " → " + families) + "</li>")
        if not query["rows"]:
            lines.append("- 当前样本未检索到有据的不同方法族候选，保留缺口。")
        lines.append("")
        html_parts.append("</ul></details>")
    lines.extend(["## 各论文的原文证据", "", "下面展示每篇首个问题候选的完整原文块。所有候选及逐条证据见 candidates/ 与 queries.json。", ""])
    html_parts.append(f"<h2>{count} 篇来源与证据</h2>")
    by_id = {result["paper_id"]: result for result in results}
    for paper in manifest["papers"]:
        result = by_id.get(paper["id"])
        lines.extend(["### " + paper["title"], "", "- 原卡：[[" + paper["card"] + "|打开论文卡]]", "- 英文原文：[[" + paper["pdf"] + "|打开 PDF]]"])
        html_parts.append("<details><summary>" + escape(paper["title"]) + "</summary><p><a href='" + escape(uri(vault, paper["card"]), quote=True) + "'>Obsidian 原卡</a> · <a href='" + escape((vault / paper["pdf"]).as_uri(), quote=True) + "'>英文 PDF</a></p>")
        problems = [o for o in result["occurrences"] if o["type"] == "question" and o["role"] != "mention_only"] if result else []
        if problems:
            evidence = problems[0]["evidence"]
            lines.extend(["- 证据：[[" + paper["pdf"] + "#page=" + str(evidence["page"]) + "|物理页 " + str(evidence["page"]) + "]]；状态：未审候选", ""])
            lines.extend("> " + part for part in evidence["quote"].splitlines())
            page_url = (vault / paper["pdf"]).as_uri() + "#page=" + str(evidence["page"])
            html_parts.append("<a href='" + escape(page_url, quote=True) + "'>核对英文原文物理页 " + str(evidence["page"]) + "</a><small>；未审候选</small><blockquote>" + escape(evidence["quote"]) + "</blockquote>")
        else:
            lines.append("- 未形成当前词表支持的问题关联；不据此断言论文没有贡献。")
            html_parts.append("<p>未形成当前词表支持的问题关联。</p>")
        lines.extend(["", "- 作者沿用既有记录，未重新核定完整名单；数据集提及不等于实验采用。", ""])
        html_parts.append("</details>")
    lines.extend(["## 检查边界", "", "- 正式图谱、注册表、原卡和 PDF 必须保持冻结哈希。", "- 不用原文字符串命中替代语义复核，不按连接数量凑边。", "- 该样本按问题选取，不能作为全库准确率或加速倍数估计。", "- 机器校验、运行计时、异常和覆盖率见 coverage.json 与运行回执。", ""])
    html_parts.append("<p><small>冻结原件保留；候选未进行正式发布。完整逐条证据、运行回执和覆盖率见本目录 JSON。</small></p></html>")
    changed.update({"markdown_changed": write_text(output / "pilot.md", "\n".join(lines), output),
                    "html_changed": write_text(output / "index.html", "\n".join(html_parts), output)})
    return changed
