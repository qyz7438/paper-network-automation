import unittest

from scripts.paper_pipeline.audit_automatic_pilot import author_matches, claim_matches, normalize, retrieval_review, sample_roles


class AutomaticAuditTests(unittest.TestCase):
    def test_name_normalization_preserves_model_numbers(self):
        self.assertNotEqual(normalize("ResNet-50"), normalize("ResNet-101"))
        self.assertEqual(normalize("ResNet-50"), normalize("resnet 50"))

    def test_author_comparison_keeps_unmatched_candidates_visible(self):
        row = {"author_crosscheck": {"names": [{"name": "Alice Example", "evidence": {"page": 1}}]},
               "person_candidates": [{"name_candidate": "False Person"}]}
        gold = {"authors": [{"name": "Alice Example"}, {"name": "Bob Example"}]}
        result = author_matches(row, gold)
        self.assertEqual(result["exact_name_matches"], 1)
        self.assertEqual(result["candidate_strings_not_exact_gold"], 1)
        self.assertEqual(result["gold_names_not_exact_candidate"], 1)
        self.assertEqual(result["retained_metadata_exact_matches"], 1)
        self.assertEqual(result["person_model_exact_matches"], 0)

    def test_pdf_metadata_match_is_counted_separately_from_existing_names(self):
        row = {"author_crosscheck": {"names": [{"name": "Alice Example", "evidence": {"page": 1}}]},
               "person_candidates": [], "metadata_author_candidates": {"parts": [
                   {"name": "Bob Example", "evidence": {"page": 1}}]}}
        gold = {"authors": [{"name": "Alice Example"}, {"name": "Bob Example"}]}
        result = author_matches(row, gold)
        self.assertEqual(result["exact_name_matches"], 2)
        self.assertEqual(result["additional_matches_from_pdf_metadata"], 1)

    def test_literal_claim_match_is_not_a_role_acceptance(self):
        ev = {"pdf_sha256": "abc", "page": 1, "start": 10, "end": 17}
        row = {"mentions": [{"type": "method", "text": "BlueNet", "role": {"role": "background"}, "mention_evidence": ev}]}
        gold = {"fields": {"methods": {"state": "supported", "claims": [{"name": "BlueNet", "evidence": [ev]}]},
                           "datasets": {"state": "unknown", "claims": []},
                           "problem": {"state": "unknown", "claims": []}}}
        result = claim_matches(row, gold)
        self.assertEqual(result["methods"]["exact_name_occurrence_matches"], 1)
        self.assertEqual(result["methods"]["same_type_mention_overlaps_reference_evidence"], 1)
        self.assertNotIn("semantic_acceptance", result["methods"])

    def test_retrieval_judgment_requires_same_passages(self):
        old = {"rows": [{"query": "fusion", "paper_id": "p1", "passage_ids": ["s1"],
                         "topic_supported_in_retrieved_passages": True}]}
        queries = [{"query": "fusion", "rows": [{"paper_id": "p1", "title": "example", "passages": [{"id": "s2"}]}]}]
        self.assertEqual(retrieval_review(queries, old)["queries"][0]["unreviewed"], 1)
        queries[0]["rows"][0]["passages"][0]["id"] = "s1"
        self.assertEqual(retrieval_review(queries, old)["queries"][0]["topic_supported"], 1)

    def test_role_sample_is_deterministic_and_scoped(self):
        def mention(mid):
            return {"mention_id": mid, "type": "method", "text": mid, "role": {"role": "proposed"},
                    "sentence_evidence": {}, "mention_evidence": {}}
        rows = [{"paper_id": "p1", "title": "one", "state": "automatic_draft_needs_review",
                 "mentions": [mention("a"), mention("b"), mention("c")]},
                {"paper_id": "p2", "title": "two", "state": "automatic_draft_needs_review",
                 "mentions": [mention("d")]}]
        plan = {"categories": ["proposed"], "seed": "frozen:", "per_category_max": 2}
        first, population = sample_roles(rows, {"p1"}, plan)
        second, _ = sample_roles(list(reversed(rows)), {"p1"}, plan)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(population["proposed"], 3)
        self.assertTrue(all(row["paper_id"] == "p1" for row in first))

    def test_role_sample_can_limit_each_paper(self):
        def mention(mid):
            return {"mention_id": mid, "type": "method", "text": mid, "role": {"role": "proposed"},
                    "sentence_evidence": {}, "mention_evidence": {}}
        rows = [{"paper_id": "p1", "title": "one", "state": "automatic_draft_needs_review",
                 "mentions": [mention("a"), mention("b")]},
                {"paper_id": "p2", "title": "two", "state": "automatic_draft_needs_review",
                 "mentions": [mention("c")]}]
        plan = {"categories": ["proposed"], "seed": "frozen:", "per_category_max": 3, "max_per_paper_per_role": 1}
        sample, _ = sample_roles(rows, {"p1", "p2"}, plan)
        self.assertEqual({item["paper_id"] for item in sample}, {"p1", "p2"})
        self.assertEqual(len(sample), 2)


if __name__ == "__main__":
    unittest.main()
