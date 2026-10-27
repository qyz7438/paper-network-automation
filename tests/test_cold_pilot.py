import unittest

from scripts.paper_pipeline.cold_pilot import timed, validate_parsed


class FreshPilotTests(unittest.TestCase):
    def test_parser_cache_is_bound_to_the_paper_and_literal_blocks(self):
        paper = {"id": "paper:test", "pdf_sha256": "abc"}
        parsed = {"paper_id": "paper:test", "pdf_sha256": "abc", "pages": [
            {"text": "some text", "blocks": [{"start": 0, "end": 4, "text": "some"}]}
        ]}
        value = timed(lambda: parsed)
        validate_parsed(value, paper)
        self.assertGreaterEqual(value["stage_seconds"], 0)
        parsed["pages"][0]["blocks"][0]["text"] = "else"
        with self.assertRaises(ValueError):
            validate_parsed(value, paper)

    def test_wrong_source_identity_is_not_accepted(self):
        value = {"value": {"paper_id": "paper:other", "pdf_sha256": "abc", "pages": [{}]}}
        with self.assertRaises(ValueError):
            validate_parsed(value, {"id": "paper:test", "pdf_sha256": "abc"})


if __name__ == "__main__":
    unittest.main()
