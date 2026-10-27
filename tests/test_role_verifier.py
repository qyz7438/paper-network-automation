import unittest

from scripts.paper_pipeline.role_verifier import parse_verdict, request_input


class RoleVerifierTests(unittest.TestCase):
    def test_strict_output(self):
        self.assertEqual(parse_verdict('{"judgment":"uncertain"}', "stop"), "uncertain")
        for raw, finish in [('{"judgment":"supported","other":1}', "stop"),
                            ('{"judgment":"true"}', "stop"), ('not json', "stop"),
                            ('{"judgment":"supported"}', "length")]:
            with self.assertRaises(ValueError):
                parse_verdict(raw, finish)

    def test_source_bounding(self):
        row = {"type":"method", "predicted_role":"adopted", "text":"BlueNet",
               "sentence_evidence":{"quote":"We use BlueNet."}}
        self.assertEqual(request_input(row)["candidate"], "BlueNet")
        row["text"] = "RedNet"
        with self.assertRaises(ValueError):
            request_input(row)


if __name__ == "__main__":
    unittest.main()
