import unittest

from scripts.paper_pipeline.metadata_author_candidates import metadata_candidates, split_author_metadata


class MetadataAuthorTests(unittest.TestCase):
    def test_metadata_names_are_only_candidates_with_literal_header_evidence(self):
        text = "A Study\nAlice Example1\nAbstract\nSome text."
        paper = {"id": "p", "pdf": "a.pdf", "pdf_sha256": "abc"}
        result = metadata_candidates(paper, {"page": 1, "text": text}, "Alice Example; Bob Example")
        self.assertEqual(result["header_matched"], 1)
        self.assertEqual(result["parts"][0]["evidence"]["quote"], "Alice Example")
        self.assertIsNone(result["parts"][1]["evidence"])
        self.assertFalse(result["complete_byline_asserted"])
        self.assertFalse(result["order_asserted"])

    def test_metadata_separators_and_groups_are_retained(self):
        self.assertEqual(split_author_metadata("Alice Example and Bob Example; SomeGroup"),
                         ["Alice Example", "Bob Example", "SomeGroup"])

    def test_email_occurrence_is_not_header_author_evidence(self):
        text = "A Study\nalice@example.org\nAbstract"
        paper = {"id": "p", "pdf": "a.pdf", "pdf_sha256": "abc"}
        result = metadata_candidates(paper, {"page": 1, "text": text}, "alice example")
        self.assertEqual(result["header_matched"], 0)


if __name__ == "__main__":
    unittest.main()
