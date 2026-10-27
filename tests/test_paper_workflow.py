"""Behavioral checks for local query rejection and conditional batch expansion."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.paper_pipeline.core import read_json, sha, write_json
from scripts.paper_pipeline.query_translation import LocalTranslator, constraint_errors, query_glossary, build_messages
from scripts.paper_pipeline.batch_workflow import accepted_gate, precision, check_translator_identity, diagnose100, OUTPUT, BASELINE, OLD, OLD_LOCAL


class TranslationTests(unittest.TestCase):
    def test_missing_constraint_is_visible(self):
        self.assertTrue(constraint_errors("q", "multimodal learning", ["SAR"], [["without", "unpaired"]]))
        self.assertFalse(constraint_errors("q", "SAR fusion without paired data", ["SAR"], [["without", "unpaired"]]))

    def test_reasoning_chinese_and_truncation_rejected(self):
        self.assertTrue(constraint_errors("q", "<think>test</think> 中文", [], []))
        self.assertTrue(constraint_errors("q", "valid English", [], [], "length"))

    def test_cached_translation_reuses_no_process_and_detects_change(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.paper_pipeline.query_translation.verify", return_value={}):
            cache = Path(folder)
            with LocalTranslator(cache) as translator:
                with patch.object(translator, "_start"), patch.object(translator, "_request", return_value={
                        "choices": [{"message": {"content": "SAR fusion without paired data"}, "finish_reason": "stop"}], "usage": {}}):
                    original = translator.translate("q", ["SAR"], [["without"]])
                self.assertEqual(translator.translate("q", ["SAR"], [["without"]]), original)
                self.assertEqual((translator.calls, translator.hits), (1, 1))
                path = next(cache.glob("*.json"))
                value = read_json(path)
                value["english"] = "SAR fusion without labeled data"
                write_json(path, value, cache)
                with self.assertRaisesRegex(ValueError, "cache validation"):
                    translator.translate("q", ["SAR"], [["without"]])

    def test_rejected_translation_is_cached_with_its_failure(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.paper_pipeline.query_translation.verify", return_value={}):
            with LocalTranslator(Path(folder)) as translator:
                with patch.object(translator, "_start"), patch.object(translator, "_request", return_value={
                        "choices": [{"message": {"content": "SAR fusion"}, "finish_reason": "length"}]}):
                    value = translator.translate("q", ["SAR"])
                self.assertEqual(value["state"], "rejected")
                self.assertEqual(translator.translate("q", ["SAR"]), value)

    def test_phase_sensitive_glossary_does_not_change_reasoning_task(self):
        self.assertIn(("推理", "inference"), query_glossary("只推理，不微调参数"))
        self.assertNotIn(("推理", "inference"), query_glossary("多步推理的信用分配"))
        self.assertNotIn(("推理", "inference"), query_glossary("不微调模型的多步推理"))
        self.assertTrue(constraint_errors("q", "depth balanced model", ["deep equilibrium model"], []))

    def test_specialized_model_uses_original_inside_author_template(self):
        original = "只推理，不更新参数"
        messages = build_messages(original, ["inference"], [("推理", "inference")], "hymt2")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].endswith(original))
        self.assertIn("推理 翻译成 inference", messages[0]["content"])

    def test_request_failure_is_retained_and_not_retried(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.paper_pipeline.query_translation.verify", return_value={}):
            cache = Path(folder)
            with LocalTranslator(cache) as translator:
                with patch.object(translator, "_start"), patch.object(translator, "_request", side_effect=TimeoutError("local timeout")):
                    with self.assertRaises(TimeoutError):
                        translator.translate("q")
                self.assertEqual(len(list((cache / "failures").glob("*.json"))), 1)
                with self.assertRaisesRegex(RuntimeError, "previous translation failure"):
                    translator.translate("q")


class GateTests(unittest.TestCase):
    def test_diagnostic_cannot_mix_with_gate_or_source_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            vault = Path(folder)
            for relative in (OUTPUT, BASELINE, OLD, OLD_LOCAL, Path("_working/paper-network-automation-v1/pilot100-hymt2-v1")):
                with self.assertRaisesRegex(ValueError, "dedicated English output"):
                    diagnose100(vault, relative)

    def test_different_translator_cannot_enter_specialized_plan(self):
        plan = {"candidate": {"repo": "tencent/Hy-MT2-1.8B-GGUF", "revision": "a0c709d9fac510f2c807aa3af52872340dc37a4a"}}
        wrong = {"checkpoint": {"repo": "Qwen/Qwen3-1.7B-GGUF", "revision": "90862c4b9d2787eaed51d12237eafdfe7c5f6077"}, "engine": "qwen17-query-translation-v2-glossary"}
        with self.assertRaisesRegex(ValueError, "does not match"):
            check_translator_identity(plan, wrong)
        right = {"checkpoint": plan["candidate"], "engine": "hymt2-query-translation-v1"}
        check_translator_identity(plan, right, [{"engine": "hymt2-query-translation-v1"}])
        with self.assertRaisesRegex(ValueError, "mixed"):
            check_translator_identity(plan, right, [{"engine": "qwen17-query-translation-v2-glossary"}])

    def test_precision_uses_fixed_cutoff(self):
        self.assertEqual(precision([{"paper_id": "p"}], ["p"]), 0.2)

    def test_review_cannot_approve_a_different_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "evaluation-plan.json", {"cases": [{"id": "q"}], "retrieval": [{"case_id": "q"}],
                       "gate": {"minimum_meaning_pass": 1, "max_precision_drop": 0.1}}, root)
            write_json(root / "gate20.json", {"plan_sha256": sha(root / "evaluation-plan.json"),
                       "translations": [{"case_id": "q", "errors": []}],
                       "mean_precision_at_5": {"chinese": 0.8, "translated": 0.8}}, root)
            review = {"reviewer": "llm-wiki-main", "gate_sha256": sha(root / "gate20.json"),
                      "cases": {"q": {"meaning_preserved": True}}}
            write_json(root / "meaning-review.json", review, root)
            self.assertEqual(accepted_gate(root)["meaning_passed"], 1)
            review["cases"]["q"]["meaning_preserved"] = "false"
            write_json(root / "meaning-review.json", review, root)
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                accepted_gate(root)
            review["cases"]["q"]["meaning_preserved"] = True
            write_json(root / "meaning-review.json", review, root)
            gate = read_json(root / "gate20.json")
            gate["mean_precision_at_5"]["translated"] = 0.0
            write_json(root / "gate20.json", gate, root)
            with self.assertRaisesRegex(ValueError, "does not bind"):
                accepted_gate(root)

    def test_unflagged_semantic_drift_blocks_expansion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {"cases": [{"id": "q"}, {"id": "neg"}], "retrieval": [{"case_id": "q"}],
                    "gate": {"minimum_meaning_pass": 1, "max_precision_drop": 0.1}}
            write_json(root / "evaluation-plan.json", plan, root)
            write_json(root / "gate20.json", {"plan_sha256": sha(root / "evaluation-plan.json"),
                       "translations": [{"case_id": "q", "errors": []}, {"case_id": "neg", "errors": []}],
                       "mean_precision_at_5": {"chinese": 0.8, "translated": 0.8}}, root)
            write_json(root / "meaning-review.json", {"reviewer": "llm-wiki-main", "gate_sha256": sha(root / "gate20.json"),
                       "cases": {"q": {"meaning_preserved": True}, "neg": {"meaning_preserved": False}}}, root)
            with self.assertRaisesRegex(ValueError, "undetected meaning drift"):
                accepted_gate(root)


if __name__ == "__main__":
    unittest.main()
