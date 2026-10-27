from contextlib import nullcontext
import unittest

from scripts.paper_pipeline.automatic_pilot import (_person_candidates, collect_model_evidence, interpret_paper,
                                                    predict_paper, validate_draft, validate_model_evidence)


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeExtractor:
    def __init__(self, bad_person=False):
        self.bad_person = bad_person

    def extract_entities(self, text, *args, **kwargs):
        start = text.index("Alice Example")
        name = "Other Name" if self.bad_person else "Alice Example"
        return {"entities": {"person": [{"text": name, "start": start,
                                            "end": start + len("Alice Example"), "confidence": 0.9}]}}

    def extract_relations(self, text, *args, **kwargs):
        start = text.index("BlueNet")
        return {"relation_extraction": {"proposes": [{
            "head": {"text": "We", "start": 0, "end": 2, "confidence": 0.9},
            "tail": {"text": "BlueNet", "start": start, "end": start + 7, "confidence": 0.9},
            "confidence": 0.8,
        }], "uses": [], "compares": []}}


class FakeModels:
    def __init__(self, bad_person=False):
        self.extractor = FakeExtractor(bad_person)
        self.torch = FakeTorch()


class AutomaticPilotTests(unittest.TestCase):
    def setUp(self):
        header = "A Test Paper\nAlice Example\nAbstract"
        sentence = "We propose BlueNet for classification."
        text = header + "\n\n" + sentence
        sentence_start = len(header) + 2
        start = text.index("BlueNet")
        self.paper = {"id": "paper:test", "title": "A Test Paper", "card": "card.md",
                      "pdf": "a.pdf", "pdf_sha256": "abc", "recorded_authors": ["Alice Example"]}
        self.page = {"page": 1, "text": text, "blocks": [
            {"start": 0, "end": len(header), "text": header},
            {"start": sentence_start, "end": len(text), "text": sentence},
        ]}
        self.parsed = {"paper_id": "paper:test", "pdf_sha256": "abc", "pages": [self.page]}
        ev = {"pdf": "a.pdf", "pdf_sha256": "abc", "page": 1, "start": start,
              "end": start + 7, "quote": "BlueNet"}
        passage = {"id": "s1", "page": 1, "start": sentence_start, "end": len(text)}
        self.record = {"paper_id": "paper:test", "source_sha256": "abc", "selection": {"passages": [passage]},
                       "occurrences": [{"id": "m1", "paper_id": "paper:test", "type": "method", "text": "BlueNet",
                                        "evidence": ev, "context_span_id": "s1"}], "rejected_occurrences": []}

    def test_draft_retains_source_and_keeps_role_unreviewed(self):
        row = predict_paper(self.paper, self.parsed, self.record, FakeModels())
        validate_draft(row, self.paper, self.parsed, self.record)
        self.assertEqual(row["mentions"][0]["role"]["role"], "proposed")
        self.assertIsNone(row["mentions"][0]["canonical_assignment"])
        self.assertFalse(row["formal_graph_publication"])
        self.assertEqual(row["person_candidates"][0]["name_candidate"], "Alice Example")
        self.assertEqual(row["author_crosscheck"]["header_matched"], 1)

    def test_bad_model_person_span_is_recorded_as_rejected(self):
        people, rejected = _person_candidates(self.paper, self.page, self.page["text"].index("Abstract"), FakeModels(True))
        self.assertEqual(people, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["state"], "model_output_rejected_not_missing_author")

    def test_validator_refuses_lost_mention(self):
        row = predict_paper(self.paper, self.parsed, self.record, FakeModels())
        row["mentions"] = []
        with self.assertRaises(ValueError):
            validate_draft(row, self.paper, self.parsed, self.record)

    def test_cached_model_evidence_can_be_reinterpreted_without_model_calls(self):
        models = FakeModels()
        evidence = collect_model_evidence(self.paper, self.parsed, self.record, models)
        validate_model_evidence(evidence, self.paper, self.parsed, self.record)
        models.extractor = None
        first = interpret_paper(self.paper, self.record, evidence)
        second = interpret_paper(self.paper, self.record, evidence)
        self.assertEqual(first, second)
        self.assertEqual(first["mentions"][0]["role"]["role"], "proposed")

    def test_cached_sentence_cannot_be_reassigned_to_another_mention(self):
        evidence = collect_model_evidence(self.paper, self.parsed, self.record, FakeModels())
        evidence["sentences"][0]["evidence"]["start"] = 0
        evidence["sentences"][0]["evidence"]["end"] = 1
        evidence["sentences"][0]["evidence"]["quote"] = "A"
        with self.assertRaises(ValueError):
            validate_model_evidence(evidence, self.paper, self.parsed, self.record)


if __name__ == "__main__":
    unittest.main()
