from contextlib import nullcontext
import unittest

from scripts.paper_pipeline.fulltext_pilot import expand_model_evidence, merge_records


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract_relations(self, text, *args, **kwargs):
        self.calls += 1
        start = text.index("RedNet")
        return {"relation_extraction": {"proposes": [], "uses": [{
            "head": {"text": "We", "start": 0, "end": 2, "confidence": 0.9},
            "tail": {"text": "RedNet", "start": start, "end": start + 6, "confidence": 0.9},
            "confidence": 0.8}], "compares": []}}


class FakeModels:
    def __init__(self):
        self.extractor = FakeExtractor()
        self.torch = FakeTorch()


class FullTextPilotTests(unittest.TestCase):
    def setUp(self):
        first, second = "We propose BlueNet.", "We use RedNet."
        self.text = first + "\n\n" + second
        offset = len(first) + 2
        self.paper = {"id": "p", "title": "T", "card": "c.md", "pdf": "a.pdf", "pdf_sha256": "abc"}
        page = {"page": 1, "text": self.text, "blocks": [{"start": 0, "end": len(first), "text": first},
                {"start": offset, "end": len(self.text), "text": second}]}
        self.parsed = {"paper_id": "p", "pdf_sha256": "abc", "pages": [page]}
        def item(mid, word, span):
            start = self.text.index(word)
            ev = {"pdf": "a.pdf", "pdf_sha256": "abc", "page": 1, "start": start, "end": start + len(word), "quote": word}
            return {"id": mid, "paper_id": "p", "type": "method", "text": word, "evidence": ev, "context_span_id": span}
        self.base = {"paper_id": "p", "source_sha256": "abc", "selection": {"passages": [
            {"id": "s1", "page": 1, "start": 0, "end": len(first)}]}, "occurrences": [item("m1", "BlueNet", "s1")], "rejected_occurrences": []}
        self.extra = {"paper_id": "p", "source_sha256": "abc", "selection": {"passages": [
            {"id": "s2", "page": 1, "start": offset, "end": len(self.text)}]}, "occurrences": [item("m2", "RedNet", "s2")], "rejected_occurrences": []}
        sentence = {"page": 1, "start": 0, "end": len(first), "quote": first, "pdf": "a.pdf", "pdf_sha256": "abc"}
        self.evidence = {"paper_id": "p", "source_sha256": "abc", "sentences": [
            {"evidence": sentence, "mention_ids": ["m1"], "raw_relation_output": None}],
            "person_candidates": [], "author_crosscheck": {"names": []}, "rejected_relation_outputs": [],
            "stats": {"relation_model_calls": 1, "person_model_calls": 0}}

    def test_merge_preserves_both_source_mentions(self):
        combined = merge_records(self.base, self.extra)
        self.assertEqual({m["id"] for m in combined["occurrences"]}, {"m1", "m2"})
        self.assertEqual(len(combined["selection"]["passages"]), 2)

    def test_new_sentence_runs_model_without_repeating_old_sentence(self):
        combined = merge_records(self.base, self.extra)
        models = FakeModels()
        result = expand_model_evidence(self.paper, self.parsed, self.base, combined, self.evidence, models)
        self.assertEqual(models.extractor.calls, 1)
        self.assertEqual(result["stats"]["additional_relation_model_calls"], 1)
        self.assertEqual({mid for group in result["sentences"] for mid in group["mention_ids"]}, {"m1", "m2"})

    def test_duplicate_identity_with_different_content_is_rejected(self):
        self.extra["occurrences"][0]["id"] = "m1"
        with self.assertRaises(ValueError):
            merge_records(self.base, self.extra)


if __name__ == "__main__":
    unittest.main()
