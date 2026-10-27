"""Behavioral checks without model download or model inference."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.paper_pipeline.core import Jobs, cache_path, object_hash, read_json, sha, write_json
from scripts.paper_pipeline.local_pipeline import bind_occurrences, context_methods, entity_documents, extraction_key, extract_paper, nearest_entities, retrieve, run, select_passages, stage, validate_result
from scripts.paper_pipeline.semantic import registry
from tests.test_paper_pipeline import source


class FakeModels:
    def __init__(self):
        self.identity = {"test_only": True}
        self.extraction_calls = self.embedding_calls = 0

    def extract(self, text, threshold):
        self.extraction_calls += 1
        word = "cross-attention"
        start = text.index(word)
        return {"entities": {"method": [{"text": word, "start": start, "end": start + len(word), "confidence": 0.9}]}}

    def encode(self, texts, prefix):
        self.embedding_calls += 1
        values = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            values[i, hashlib.sha256((prefix + text).encode()).digest()[0]] = 1
        return values


class LocalPipelineTests(unittest.TestCase):
    def sample(self):
        return source(["We propose cross-attention fusion to combine complementary camera and LiDAR features for robust object detection. Our method addresses inaccurate calibration in multimodal detection."])

    def test_selection_preserves_offsets_and_respects_budget(self):
        paper, parsed = self.sample()
        selected = select_passages(paper, parsed, 1)
        self.assertEqual(len(selected["passages"]), 1)
        p = selected["passages"][0]
        self.assertEqual(parsed["pages"][0]["text"][p["start"]:p["end"]], p["text"])
        with self.assertRaises(ValueError):
            select_passages(paper, parsed, 0)

    def test_references_not_selected(self):
        paper, parsed = source(["References", self.sample()[1]["pages"][0]["text"]])
        self.assertEqual(select_passages(paper, parsed)["passages"], [])

    def test_span_binding_and_tampering(self):
        paper, parsed = self.sample()
        selected = select_passages(paper, parsed)
        p = selected["passages"][0]
        raw = FakeModels().extract(p["text"], 0.45)
        items = bind_occurrences(paper, p, raw)
        result = {"paper_id": paper["id"], "source_sha256": paper["pdf_sha256"], "occurrences": items}
        validate_result(result, paper, parsed, selected)
        items[0]["evidence"]["end"] += 1
        with self.assertRaises(ValueError):
            validate_result(result, paper, parsed, selected)

    def test_invented_model_span_and_nan_rejected(self):
        paper, parsed = self.sample()
        p = select_passages(paper, parsed)["passages"][0]
        raw = FakeModels().extract(p["text"], 0.45)
        wrong = copy.deepcopy(raw)
        wrong["entities"]["method"][0]["text"] = "invented"
        with self.assertRaises(ValueError):
            bind_occurrences(paper, p, wrong)
        raw["entities"]["method"][0]["confidence"] = float("nan")
        with self.assertRaises(ValueError):
            bind_occurrences(paper, p, raw)

    def test_bad_model_span_is_quarantined_without_losing_valid_span(self):
        paper, parsed = self.sample()
        selected = select_passages(paper, parsed)
        raw = FakeModels().extract(selected["passages"][0]["text"], 0.45)
        bad = copy.deepcopy(raw["entities"]["method"][0])
        bad["text"] = "invented text"
        raw["entities"]["method"].append(bad)
        with patch.object(FakeModels, "extract", return_value=raw):
            result = extract_paper(FakeModels(), paper, selected, 0.45)
        self.assertEqual(len(result["occurrences"]), 1)
        self.assertEqual(len(result["rejected_occurrences"]), 1)
        self.assertEqual(result["rejected_occurrences"][0]["model_value"]["text"], "invented text")
        validate_result(result, paper, parsed, selected)
        result["rejected_occurrences"][0]["source_context"]["text"] = "changed"
        with self.assertRaises(ValueError):
            validate_result(result, paper, parsed, selected)

    def test_nearest_neighbors_do_not_assign_canonical_identity(self):
        items = [{"type": "method", "text": "Fusion", "id": "x"}]
        entities = [{"id": "m1", "type": "method", "label": "Fusion"}, {"id": "q", "type": "question", "label": "Fusion"},
                    {"id": "m2", "type": "method", "label": "Fusion"}]
        rows = nearest_entities(items, entities, np.array([[1, 0], [1, 0], [0.9, 0.1]]), np.array([[1, 0]]))
        self.assertEqual(rows[0]["exact_name_candidate_ids"], ["m1", "m2"])
        self.assertEqual(rows[0]["identity_state"], "unresolved")
        self.assertIsNone(rows[0]["canonical_assignment"])
        self.assertNotIn("q", [s["entity_id"] for s in rows[0]["suggestions"]])

    def test_retrieval_groups_papers_and_preserves_evidence(self):
        documents = [{"paper_id": "a", "id": "s1"}, {"paper_id": "a", "id": "s2"}, {"paper_id": "b", "id": "s3"}]
        papers = {k: {"title": k, "card": k + ".md"} for k in ("a", "b")}
        data = retrieve("query", documents, np.array([[1, 0], [0.9, 0.1], [0, 1]]), np.array([1, 0]), papers)
        self.assertEqual([r["paper_id"] for r in data["rows"]], ["a", "b"])
        self.assertEqual(len(data["rows"][0]["passages"]), 2)

    def test_displayed_methods_belong_to_retrieved_context_and_are_deduplicated(self):
        row = {"passages": [{"id": "a"}]}
        mentions = [{"type": "method", "context_span_id": "a", "text": "Cross-attention"},
                    {"type": "method", "context_span_id": "a", "text": "cross-attention"},
                    {"type": "method", "context_span_id": "b", "text": "Unrelated background method"}]
        self.assertEqual(context_methods(row, mentions), ["Cross-attention"])

    def test_stage_corrupt_cache_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            queue = Jobs(out / "state.sqlite", out)
            try:
                value, cached = stage(queue, out, "p", "test", "k", lambda: {"n": 1})
                self.assertFalse(cached)
                write_json(out / "cache/test/k.json", {"n": 2}, out)
                value, cached = stage(queue, out, "p", "test", "k", lambda: {"n": 1})
                self.assertFalse(cached)
                self.assertEqual(value, {"n": 1})
            finally:
                queue.close()

    def test_failed_stage_requires_explicit_retry_and_retains_error(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            queue = Jobs(out / "state.sqlite", out)
            try:
                def fail():
                    raise RuntimeError("transient test failure")
                with self.assertRaisesRegex(RuntimeError, "transient"):
                    stage(queue, out, "p", "test", "k", fail)
                with self.assertRaisesRegex(RuntimeError, "busy or failed"):
                    stage(queue, out, "p", "test", "k", lambda: {"ok": True})
                data, cached = stage(queue, out, "p", "test", "k", lambda: {"ok": True}, retry_failed=True)
                self.assertFalse(cached)
                self.assertTrue(data["ok"])
                error = read_json(next((out / "failures").glob("*.json")))
                self.assertIn("transient test failure", error["error"])
            finally:
                queue.close()

    def test_extraction_key_binds_producer_validator_and_tokenizer_identity(self):
        paper, parsed = self.sample()
        selected = select_passages(paper, parsed)
        first = extraction_key(paper, selected, {"tokenizers": "a"}, 0.45)
        second = extraction_key(paper, selected, {"tokenizers": "b"}, 0.45)
        self.assertNotEqual(first, second)
        with patch("scripts.paper_pipeline.local_pipeline.inspect.getsource", return_value="changed producer"):
            third = extraction_key(paper, selected, {"tokenizers": "a"}, 0.45)
        self.assertNotEqual(first, third)

    def test_end_to_end_cache_noop_and_baseline_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            baseline = vault / "_working/paper-network-automation-v1/baseline"
            out = vault / "_working/paper-network-automation-v1/models"
            paper, parsed = self.sample()
            manifest = {"papers": [paper], "protected": {}, "existing_entities": []}
            write_json(baseline / "manifest.json", manifest)
            write_json(baseline / "registry.json", registry([]))
            write_json(cache_path(baseline, paper), parsed)
            baseline_hash = sha(baseline / "manifest.json")
            first = run(vault, baseline, out, FakeModels())
            self.assertGreater(first["model_calls"]["extraction"], 0)
            second = run(vault, baseline, out, FakeModels())
            self.assertEqual(second["model_calls"], {"extraction": 0, "embedding_batches": 0})
            self.assertFalse(any(second["artifacts_changed"].values()))
            self.assertEqual(sha(baseline / "manifest.json"), baseline_hash)
            self.assertIn("\\|", (out / "report.md").read_text(encoding="utf-8"))
            self.assertIn("1 篇论文试验", (out / "report.md").read_text(encoding="utf-8"))
            self.assertNotIn("20 篇论文试验", (out / "report.md").read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                run(vault, baseline, baseline, FakeModels())


if __name__ == "__main__":
    unittest.main()
