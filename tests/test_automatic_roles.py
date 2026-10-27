import unittest

from scripts.paper_pipeline.automatic_roles import classify_mention


def edge(text, head, tail, label):
    h, t = text.index(head), text.index(tail)
    return {"relation_extraction": {label: [{
        "head": {"text": head, "start": h, "end": h + len(head), "confidence": 0.9},
        "tail": {"text": tail, "start": t, "end": t + len(tail), "confidence": 0.9},
        "confidence": 0.8,
    }]}}


class AutomaticRoleTests(unittest.TestCase):
    def classify(self, text, mention, raw=None, kind="method"):
        start = text.index(mention)
        return classify_mention(kind, text, start, start + len(mention), raw)

    def test_self_proposal_with_exact_source_object(self):
        text = "We propose BlueNet for image classification."
        result = self.classify(text, "BlueNet", edge(text, "We", "BlueNet", "proposes"))
        self.assertEqual(result["role"], "proposed")
        self.assertEqual({x["basis"] for x in result["signals"]}, {"relation_model", "source_predicate_rule"})
        self.assertEqual(result["state"], "unreviewed_candidate")

    def test_previous_work_is_not_our_proposal(self):
        text = "Previous work proposed OldNet for image classification."
        result = self.classify(text, "OldNet", edge(text, "Previous work", "OldNet", "proposes"))
        self.assertEqual(result["role"], "background")

    def test_mentioned_framework_is_not_the_proposed_strategy(self):
        text = "We propose a reward strategy within GAE."
        result = self.classify(text, "GAE", edge(text, "We", "GAE", "proposes"))
        self.assertEqual(result["role"], "unresolved")

    def test_adopting_a_name_is_not_using_a_method(self):
        text = "We adopt the name KMD to describe our metric."
        result = self.classify(text, "KMD", edge(text, "We", "KMD", "uses"))
        self.assertEqual(result["role"], "unresolved")

    def test_baseline_use_conflict_is_not_silently_adopted(self):
        text = "We use RedNet as a baseline."
        result = self.classify(text, "RedNet", edge(text, "We", "RedNet", "uses"))
        self.assertEqual(result["role"], "unresolved")
        self.assertEqual(result["reason"], "role_signals_conflict")

    def test_own_comparison_targets_baseline(self):
        text = "We compare BlueNet with RedNet."
        result = self.classify(text, "RedNet", edge(text, "BlueNet", "RedNet", "compares"))
        self.assertEqual(result["role"], "compared")

    def test_negation_never_becomes_adoption(self):
        text = "We do not use RedNet."
        result = self.classify(text, "RedNet", edge(text, "We", "RedNet", "uses"))
        self.assertEqual(result["role"], "unresolved")

    def test_model_output_must_match_exact_source(self):
        text = "We use BlueNet."
        raw = edge(text, "We", "BlueNet", "uses")
        raw["relation_extraction"]["uses"][0]["tail"]["text"] = "RedNet"
        with self.assertRaises(ValueError):
            self.classify(text, "BlueNet", raw)

    def test_hypothetical_use_does_not_establish_adoption(self):
        text = "If we apply a shallow model, it may fail to converge."
        result = self.classify(text, "shallow model")
        self.assertEqual(result["reason"], "conditional_statement")

    def test_training_and_evaluation_predicates_are_not_conflated(self):
        text = "We train on synthetic questions and evaluate on CIFAR-10."
        result = self.classify(text, "CIFAR-10", kind="dataset")
        self.assertEqual(result["role"], "unresolved")
        self.assertEqual(result["reason"], "multiple_dataset_predicates")

    def test_nonmethod_mention_is_not_assigned_method_comparison(self):
        text = "We compare the two methods described in Appendix D."
        result = self.classify(text, "Appendix D", kind="dataset")
        self.assertEqual(result["role"], "unresolved")

    def test_analysis_language_is_not_the_analyzed_method(self):
        text = "We present an analysis of this phenomenon in the language of lossy compression."
        result = self.classify(text, "lossy compression")
        self.assertEqual(result["role"], "unresolved")

    def test_background_cue_does_not_leak_into_current_paper_clause(self):
        text = "Unlike existing work on credit assignment, this paper makes credit assignment the central topic."
        first = text.index("credit assignment")
        second = text.rindex("credit assignment")
        self.assertEqual(classify_mention("task", text, first, first + len("credit assignment"))["role"], "background")
        self.assertEqual(classify_mention("task", text, second, second + len("credit assignment"))["role"], "unresolved")

    def test_cross_reference_is_not_a_method_or_dataset(self):
        text = "We compare BlueNet (Al-\ngorithm 2) and prior results [1, 19]."
        self.assertEqual(self.classify(text, "Al-\ngorithm 2")["reason"], "generic_fragment_or_cross_reference")
        self.assertEqual(self.classify(text, "[1, 19]", kind="dataset")["reason"], "generic_fragment_or_cross_reference")

    def test_research_setting_is_not_a_named_dataset(self):
        text = "Existing work studies online RL."
        self.assertEqual(self.classify(text, "online RL", kind="dataset")["reason"], "generic_fragment_or_cross_reference")

    def test_proposed_future_use_does_not_assert_completed_adoption(self):
        text = "We propose to use BlueNet for classification."
        result = self.classify(text, "BlueNet", edge(text, "We", "BlueNet", "uses"))
        self.assertEqual(result["reason"], "proposed_use_not_confirmed_execution")

    def test_purpose_is_not_the_object_of_the_analysis(self):
        text = "We analyze perceptual losses to address unrealistic artifacts."
        self.assertEqual(self.classify(text, "unrealistic artifacts", kind="question")["role"], "unresolved")

    def test_generic_study_topic_is_not_a_method_entity(self):
        text = "We study gen-\neralization in neural networks."
        self.assertEqual(self.classify(text, "gen-\neralization")["reason"], "generic_fragment_or_cross_reference")

    def test_adverb_and_performance_fragment_are_not_automatically_methods(self):
        self.assertEqual(self.classify("We heavily use prior work.", "heavily")["role"], "unresolved")
        self.assertEqual(self.classify("We investigate IDE performance.", "IDE performance")["role"], "unresolved")

    def test_terminology_definition_does_not_establish_method_use(self):
        text = "We use (text/visual) HCA to refer to model performance."
        self.assertEqual(self.classify(text, "text/visual")["reason"], "terminology_not_method")

    def test_named_manifold_family_does_not_establish_method_type(self):
        text = "We propose BlueSet\u2014 a family of topologically non-trivial manifolds for evaluation."
        self.assertEqual(self.classify(text, "BlueSet")["reason"], "entity_type_not_established")
        algorithm = "We propose BlueNet, a family of algorithms for classification."
        self.assertEqual(self.classify(algorithm, "BlueNet")["role"], "proposed")


if __name__ == "__main__":
    unittest.main()
