"""Behavioral and negative tests for working-only literature automation."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.paper_pipeline.core import Jobs, cache_path, checked_output, contained, freeze, object_hash, output_root, parse_key, read_json, sha, write_json
from scripts.paper_pipeline.semantic import alternative_query, extract, make_graph, problem_query, registry, validate_spans
from scripts.paper_pipeline.render import diagram, md_text, reports, svg_diagram, uri, write_text


def source(blocks: list[str], identifier: str = "paper:test") -> tuple[dict, dict]:
    items = []
    text = ""
    for block in blocks:
        start = len(text)
        text += block + "\n\n"
        items.append({"text": block, "start": start, "end": start + len(block), "bbox": [0, 0, 1, 1]})
    paper = {"id": identifier, "title": "A local research paper", "card": "paper.md", "pdf": "pdfs/source.pdf", "pdf_sha256": "a" * 64}
    parsed = {"paper_id": identifier, "pdf_sha256": "a" * 64, "pages": [{"page": 1, "text": text, "blocks": items}]}
    return paper, parsed


class PipelineTests(unittest.TestCase):
    def test_pdf_control_newlines_keep_table_and_repeat_write_stable(self):
        quote = "PDF formula\r\nfirst\r\rsecond\nwith | [symbols]"
        table = "| source | " + md_text(quote) + " |\n"
        self.assertNotIn("\r", table)
        self.assertEqual(table.count("\n"), 1)
        self.assertIn("\\| \\[symbols\\]", table)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            path = root / "source-table.md"
            self.assertTrue(write_text(path, table, root))
            before = path.read_bytes()
            self.assertFalse(write_text(path, table, root))
            self.assertEqual(path.read_bytes(), before)

    def setUp(self):
        self.vocabulary = registry([])

    def keys(self, result, kind):
        mapping = {e["id"]: e["key"] for e in self.vocabulary["entities"]}
        return {mapping[o["entity_id"]] for o in result["occurrences"] if o["type"] == kind}

    def test_diffusion_is_not_fusion(self):
        paper, parsed = source(["We propose a diffusion model for visual generation using text guidance."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertNotIn("feature-fusion", self.keys(result, "question"))

    def test_pixel_registration_is_not_register_tokens(self):
        paper, parsed = source(["We present pixel-registered imaging and synthetic fine-tuning for defogging."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertNotIn("register-tokens", self.keys(result, "method"))
        self.assertNotIn("attention-sinks", self.keys(result, "question"))

    def test_contrastive_rl_is_not_visual_language_alignment(self):
        paper, parsed = source(["We propose contrastive branch sampling for hierarchical credit assignment."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertNotIn("contrastive-alignment", self.keys(result, "method"))

    def test_hierarchical_convolution_is_not_hierarchical_credit(self):
        paper, parsed = source(["We propose hierarchical equilibrium neural network layers with a Broyden solver."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertNotIn("hierarchical-credit", self.keys(result, "method"))

    def test_lidar_bev_carrier_is_not_shared_multimodal_bev_fusion(self):
        paper, parsed = source(["We fuse a LiDAR BEV feature map with image features using cross-attention."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertNotIn("bev-representation", self.keys(result, "method"))

    def test_first_page_background_method_is_queued_but_not_default_discovery(self):
        background = ("Existing fusion methods use a shared BEV representation for proposals. "
                      "Previous methods project features into this space, and compare different sensors. ")
        paper, parsed = source([background, "We propose cross-attention fusion for camera and LiDAR."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertIn("bev-representation", self.keys(result, "method"))
        query = problem_query([result], self.vocabulary, "feature-fusion")
        self.assertEqual({m["key"] for m in query["rows"][0]["method_families"]}, {"cross-attention"})
        graph = make_graph([result], self.vocabulary)
        self.assertNotIn("pilot:method:bev-representation", {n["id"] for n in graph["nodes"]})

    def test_unused_registry_entities_do_not_become_isolated_graph_nodes(self):
        paper, parsed = source(["We use trajectory balance for credit assignment."])
        result = extract(paper, parsed, self.vocabulary)
        graph = make_graph([result], self.vocabulary)
        incident = {e[k] for e in graph["edges"] for k in ("source", "target")}
        self.assertTrue(all(n["id"] in incident or n["type"] == "paper" for n in graph["nodes"]))

    def test_implicit_register_function_is_not_added_learnable_tokens(self):
        paper, parsed = source(["We show that template tokens act as implicit semantic registers and attention sinks."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertNotIn("register-tokens", self.keys(result, "method"))
        self.assertIn("implicit-semantic-registers", self.keys(result, "concept"))

    def test_rl_structural_alignment_is_not_cross_modal_representation_alignment(self):
        paper, parsed = source(["Point-Vision-Language Models have geometric hallucination. We identify the cause not as a representation bottleneck but as a structural misalignment in reinforcement learning. Our reward credit assignment yields targeted structural alignment and a verifier."])
        self.assertNotIn("representation-alignment", self.keys(extract(paper, parsed, self.vocabulary), "question"))

    def test_pdf_line_break_and_ligatures_preserve_literal_evidence(self):
        paper, parsed = source(["We propose cross-\nattention for LiDAR camera fu\uFB01sion. We solve credit assign-\nment."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertIn("cross-attention", self.keys(result, "method"))
        self.assertIn("temporal-credit", self.keys(result, "question"))
        validate_spans(result, parsed)

    def test_early_mentions_cannot_hide_later_author_claim(self):
        blocks = ["Trajectory balance is discussed in previous work."] * 5
        blocks.append("We propose trajectory balance for credit assignment.")
        paper, parsed = source(blocks)
        result = extract(paper, parsed, self.vocabulary)
        entity = next(e for e in self.vocabulary["entities"] if e["key"] == "trajectory-balance")
        self.assertTrue(any(o["entity_id"] == entity["id"] and o["role"] == "author_claim_candidate" for o in result["occurrences"]))

    def test_background_domains_do_not_become_application_facets(self):
        paper, parsed = source(["We study vision transformers. Prior applications include medical images and remote sensing."])
        paper["title"] = "Vision Transformers Need Registers"
        result = extract(paper, parsed, self.vocabulary)
        self.assertEqual(result["domains"], ["vision"])

    def test_generic_convergence_is_not_learning_or_fixed_point(self):
        paper, parsed = source(["We prove convergence of graph Dirichlet energies on intersecting manifolds."])
        self.assertEqual(self.keys(extract(paper, parsed, self.vocabulary), "question"), set())

    def test_credit_and_fixed_point_are_distinct(self):
        paper, parsed = source(["We propose trajectory balance to improve credit assignment."])
        keys = self.keys(extract(paper, parsed, self.vocabulary), "question")
        self.assertIn("temporal-credit", keys)
        self.assertNotIn("fixed-point-convergence", keys)

    def test_representation_alignment_does_not_imply_feature_fusion(self):
        paper, parsed = source(["We use contrastive learning to align image and text representations."])
        keys = self.keys(extract(paper, parsed, self.vocabulary), "question")
        self.assertIn("representation-alignment", keys)
        self.assertNotIn("feature-fusion", keys)

    def test_reference_section_does_not_create_scientific_relation(self):
        paper, parsed = source(["References", "We propose cross-attention fusion for camera and LiDAR."])
        self.assertEqual(extract(paper, parsed, self.vocabulary)["occurrences"], [])

    def test_related_work_does_not_become_our_method(self):
        paper, parsed = source(["2 Related Work", "We propose cross-attention fusion for camera and LiDAR.", "3 Method", "We use trajectory balance for credit assignment."])
        keys = self.keys(extract(paper, parsed, self.vocabulary), "method")
        self.assertNotIn("cross-attention", keys)
        self.assertIn("trajectory-balance", keys)

    def test_roman_numbered_related_works_with_author_cue_is_excluded(self):
        paper, parsed = source(["II. RELATED WORKS", "We propose cross-attention fusion for camera and LiDAR.",
                                "III. METHOD", "We use trajectory balance for credit assignment."])
        keys = self.keys(extract(paper, parsed, self.vocabulary), "method")
        self.assertNotIn("cross-attention", keys)
        self.assertIn("trajectory-balance", keys)

    def test_survey_is_taxonomy_not_adoption(self):
        paper, parsed = source(["We survey hierarchical credit assignment and trajectory balance."])
        paper["title"] = "A Survey of Credit Assignment"
        result = extract(paper, parsed, self.vocabulary)
        self.assertTrue(result["occurrences"])
        self.assertEqual({o["role"] for o in result["occurrences"]}, {"survey_taxonomy_candidate"})

    def test_exact_literal_span_binding_and_tamper_detection(self):
        paper, parsed = source(["We propose cross-attention fusion for camera and LiDAR."])
        result = extract(paper, parsed, self.vocabulary)
        validate_spans(result, parsed)
        result["occurrences"][0]["evidence"]["quote"] += " invented"
        with self.assertRaises(ValueError):
            validate_spans(result, parsed)

    def test_evidence_cannot_link_a_different_pdf(self):
        paper, parsed = source(["We address credit assignment with trajectory balance."])
        result = extract(paper, parsed, self.vocabulary)
        result["occurrences"][0]["evidence"]["pdf"] = "pdfs/another.pdf"
        with self.assertRaises(ValueError):
            validate_spans(result, parsed)

    def test_different_paper_binding_is_rejected(self):
        paper, parsed = source(["We address credit assignment."])
        parsed["paper_id"] = "paper:another"
        with self.assertRaises(ValueError):
            extract(paper, parsed, self.vocabulary)

    def test_problem_rollup_keeps_alignment_and_fusion_distinct(self):
        p1, s1 = source(["We propose cross-attention fusion for LiDAR and camera."], "paper:a")
        p2, s2 = source(["We use contrastive learning to align image and text."], "paper:b")
        results = [extract(p1, s1, self.vocabulary), extract(p2, s2, self.vocabulary)]
        self.assertEqual(len(problem_query(results, self.vocabulary, "multimodal-information")["rows"]), 2)
        self.assertEqual([r["paper_id"] for r in problem_query(results, self.vocabulary, "feature-fusion")["rows"]], ["paper:a"])

    def test_alternatives_use_shared_problem_and_different_paper_method(self):
        p1, s1 = source(["We use trajectory balance for credit assignment."], "paper:a")
        p2, s2 = source(["Our approach uses hierarchical credit assignment."], "paper:b")
        p3, s3 = source(["We propose cross-attention fusion for LiDAR and camera."], "paper:c")
        results = [extract(p, s, self.vocabulary) for p, s in [(p1, s1), (p2, s2), (p3, s3)]]
        rows = alternative_query(results, self.vocabulary, "trajectory-balance")["rows"]
        self.assertEqual({r["paper_id"] for r in rows}, {"paper:b"})

    def test_method_component_cannot_become_unrelated_problem_solution(self):
        paper, parsed = source(["We propose cross-attention image text alignment. We also discuss credit assignment."])
        result = extract(paper, parsed, self.vocabulary)
        query = problem_query([result], self.vocabulary, "temporal-credit")
        self.assertEqual(query["rows"][0]["method_families"], [])

    def test_class_direction_is_preserved(self):
        paper, parsed = source(["We propose cross-attention fusion for LiDAR and camera."])
        graph = make_graph([extract(paper, parsed, self.vocabulary)], self.vocabulary)
        nodes = {n["id"]: n for n in graph["nodes"]}
        for edge in graph["edges"]:
            self.assertEqual(edge["directed"], nodes[edge["source"]]["type"] == nodes[edge["target"]]["type"])

    def test_unknown_query_is_explicit(self):
        with self.assertRaises(ValueError):
            problem_query([], self.vocabulary, "not-registered")

    def test_dataset_mentions_do_not_acquire_training_role(self):
        paper, parsed = source(["We evaluate our approach on KITTI and compare related work."])
        result = extract(paper, parsed, self.vocabulary)
        self.assertIn("kitti", self.keys(result, "dataset"))
        self.assertNotIn("training", str(result["occurrences"]))

    def test_exact_existing_entity_can_be_candidate_reuse(self):
        vocabulary = registry([{"id": "method:existing", "label": "Cross-attention", "type": "method", "card": "existing.md"}])
        entity = next(e for e in vocabulary["entities"] if e["key"] == "cross-attention")
        self.assertEqual(entity["id"], "method:existing")
        self.assertEqual(entity["publication_state"], "unpublished_candidate")

    def test_ambiguous_existing_labels_not_auto_merged(self):
        vocabulary = registry([{"id": i, "label": "Cross-attention", "type": "method"} for i in ["m:1", "m:2"]])
        self.assertTrue(next(e for e in vocabulary["entities"] if e["key"] == "cross-attention")["id"].startswith("pilot:"))

    def test_source_path_containment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                contained(root, "../secret")
            with self.assertRaises(ValueError):
                contained(root, str(root.parent.resolve()))

    def test_output_cannot_write_formal_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                output_root(root, root / "_wiki")

    def test_json_repeat_preserves_bytes_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.json"
            value = {"items": ["source" * 100 for _ in range(50)]}
            self.assertTrue(write_json(path, value))
            before = (sha(path), path.stat().st_mtime_ns)
            self.assertFalse(write_json(path, value))
            self.assertEqual((sha(path), path.stat().st_mtime_ns), before)
            self.assertEqual(read_json(path), value)

    def test_sqlite_claims_are_exclusive_and_expired_lease_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jobs.sqlite"
            first, second = Jobs(path), Jobs(path)
            try:
                job = first.enqueue("p", "parse", "source-hash")
                self.assertTrue(first.claim(job))
                self.assertFalse(second.claim(job))
                first.connection.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job,))
                self.assertTrue(second.claim(job))
                with self.assertRaises(ValueError):
                    first.finish(job, "stale.json", "stale-hash")
                second.finish(job, "cache.json", "hash")
                self.assertEqual(second.get(job)["attempts"], 2)
                self.assertFalse(first.claim(job))
            finally:
                first.close(); second.close()

    def test_failed_job_is_not_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = Jobs(Path(directory) / "jobs.sqlite")
            try:
                job = queue.enqueue("p", "parse", "hash")
                queue.claim(job)
                queue.fail(job, "ValueError: OCR required")
                row = queue.get(job)
                self.assertEqual(row["state"], "failed")
                self.assertIn("OCR", row["error"])
            finally:
                queue.close()

    def test_symlink_cannot_redirect_database_or_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            target = root / "outside.sqlite"
            target.write_bytes(b"preserve")
            link = output / "state.sqlite"
            try:
                link.symlink_to(target)
            except OSError as error:
                self.skipTest("symlink permission unavailable: " + str(error))
            with self.assertRaises(ValueError):
                Jobs(link, output)
            with self.assertRaises(ValueError):
                write_json(link, {"changed": True}, output)
            self.assertEqual(target.read_bytes(), b"preserve")

    def test_report_source_page_links_escape_and_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports"
            output.mkdir()
            paper, parsed = source(["We propose cross-attention fusion for LiDAR and camera. <script>bad()</script>"])
            paper["title"] = "Hostile <script>title</script>"
            paper["pdf"] = "pdfs/space and \u4e2d\u6587.pdf"
            result = extract(paper, parsed, self.vocabulary)
            query = problem_query([result], self.vocabulary, "feature-fusion")
            queries = {"problems": {"feature-fusion": query}, "alternatives": {}}
            coverage = {"engine": "test"}
            first = reports(root, output, {"papers": [paper]}, [result], queries, coverage)
            self.assertTrue(all(first.values()))
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn((root / paper["pdf"]).as_uri() + "#page=1", html)
            self.assertIn("&amp;file=paper.md", html)
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("1 篇来源与证据", html)
            self.assertNotIn("20 篇", html)
            markdown = (output / "pilot.md").read_text(encoding="utf-8")
            self.assertIn("| [[paper.md\\|", markdown)
            self.assertNotIn("| [[paper.md|", markdown)
            self.assertIn("原文映射候选", markdown)
            self.assertNotIn("source_bound_problem_method", markdown)
            second = reports(root, output, {"papers": [paper]}, [result], queries, coverage)
            self.assertFalse(any(second.values()))
            empty_query = {**query, "rows": [], "method_family_count": 0, "domain_count": 0}
            empty = {"problems": {"feature-fusion": empty_query}, "alternatives": {}}
            updated = reports(root, output, {"papers": [paper]}, [result], empty, coverage)
            self.assertTrue(updated["feature-fusion.svg"])
            empty_svg = (output / "feature-fusion.svg").read_text(encoding="utf-8")
            self.assertNotIn('data-kind="paper"', empty_svg)
            self.assertIn("当前样本没有候选论文", empty_svg)
            self.assertFalse(any(reports(root, output, {"papers": [paper]}, [result], empty, coverage).values()))

    def test_mermaid_uses_actual_query_nodes_and_cannot_close_code_fence(self):
        q = {"label": "Problem", "rows": [{"title": 'paper ``` <script> "', "tasks": ["Task"],
                                             "method_families": [{"id": "m", "label": "Method"}]}]}
        value = "\n".join(diagram(q))
        self.assertEqual(value.count("```"), 2)
        self.assertIn("m0 --- p0", value)
        self.assertIn("t0 --- p0", value)
        self.assertNotIn("<script>", value)

    def test_mermaid_source_labels_cannot_insert_statement_lines(self):
        title = "Paper\r\nclassDef injected fill:red;\tend"
        q = {"label": "Problem\r\nTitle", "rows": [{"title": title, "tasks": ["Task\nname"],
                "method_families": [{"id": "m", "label": "Method\rname"}]}]}
        lines = diagram(q)
        self.assertTrue(all("\n" not in line and "\r" not in line and "\t" not in line for line in lines))
        self.assertIn('p0["Paper classDef injected fill:red; end"]', lines)
        self.assertFalse(any(line.startswith("classDef injected") for line in lines))

    def test_svg_uses_query_edges_shapes_and_escaped_source_links(self):
        import xml.etree.ElementTree as ET
        q = {"label": "Problem & issue", "rows": [{"title": '<script>Paper</script>', "card": "folder/a b.md",
             "tasks": ["Task"], "method_families": [{"id": "m", "label": "Method"}]}]}
        value = svg_diagram(Path("E:/vault"), q)
        root = ET.fromstring(value)
        ns = {"s": "http://www.w3.org/2000/svg"}
        self.assertEqual({(x.attrib["data-from"], x.attrib["data-to"]) for x in root.findall(".//s:line", ns)},
                         {("q", "p0"), ("m0", "p0"), ("t0", "p0")})
        self.assertEqual(len(root.findall(".//s:polygon", ns)), 2)
        self.assertEqual(len(root.findall(".//s:rect", ns)), 3)
        self.assertIn("folder%2Fa%20b.md", root.find(".//s:a", ns).attrib["href"])
        self.assertNotIn("<script>", value)

    def test_svg_rejects_xml_invalid_controls_in_source_labels(self):
        import xml.etree.ElementTree as ET
        q = {"label": "Problem\x01", "rows": [{"title": "Paper\x00\x0b\ufffe", "card": "p.md",
             "tasks": ["Task\x02"], "method_families": [{"id": "m", "label": "Method\x03"}]}]}
        value = svg_diagram(Path("E:/vault"), q)
        ET.fromstring(value)
        self.assertTrue(all(c not in value for c in ["\x00", "\x01", "\x02", "\x03", "\x0b", "\ufffe"]))

    def test_parser_dependency_version_invalidates_cache(self):
        paper = {"pdf_sha256": "same-source"}
        with patch("pymupdf.VersionBind", "version-A"):
            first = parse_key(paper)
        with patch("pymupdf.VersionBind", "version-B"):
            second = parse_key(paper)
        self.assertNotEqual(first, second)

    def test_real_pdf_roundtrip_cache_recovery_and_protected_inputs(self):
        import pymupdf
        from scripts.paper_pipeline.cli import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "_working/paper-network-automation-v1/test"
            (root / "pdfs").mkdir()
            (root / "_wiki/network").mkdir(parents=True)
            nodes = []
            for index, text in enumerate(["Sensor Fusion Example\nWe propose cross-attention fusion for LiDAR and camera.",
                                          "Trajectory Learning Example\nWe propose trajectory balance for credit assignment."]):
                pdf = root / "pdfs" / (str(index) + ".pdf")
                with pymupdf.open() as doc:
                    page = doc.new_page()
                    page.insert_text((50, 50), text)
                    doc.save(pdf)
                card = "_wiki/" + str(index) + ".md"
                (root / card).write_text("preserved card " + str(index), encoding="utf-8")
                nodes.append({"id": "paper:" + str(index), "type": "paper", "label": text.splitlines()[0], "card": card,
                              "metadata": {"pdf_en": [pdf.relative_to(root).as_posix()]}})
            write_json(root / "_wiki/network/graph.json", {"nodes": nodes, "edges": []})
            manifest = freeze(root, output, [node["id"] for node in nodes])
            first = run(root, output, workers=1)
            self.assertEqual(first["parse_new"], 2)
            self.assertTrue(first["protected_inputs_unchanged"])
            second = run(root, output, workers=1)
            self.assertEqual((second["parse_new"], second["extract_new"]), (0, 0))
            self.assertFalse(any(second["artifacts_changed"].values()))
            path = cache_path(output, manifest["papers"][0])
            corrupted = read_json(path)
            corrupted["pages"][0]["text"] += " invented"
            write_json(path, corrupted, output)
            recovered = run(root, output, workers=1)
            self.assertEqual(recovered["parse_new"], 1)
            self.assertEqual(recovered["extract_new"], 0)
            (root / nodes[0]["card"]).write_text("changed input", encoding="utf-8")
            with self.assertRaises(ValueError):
                run(root, output, workers=1)


if __name__ == "__main__":
    unittest.main()
