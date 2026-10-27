import unittest

from scripts.paper_pipeline.relation_trial import span_issues


class RelationSpanTests(unittest.TestCase):
    def test_nested_head_and_tail_must_be_literal(self):
        text = 'We use AdamW.'
        raw = {'relations': [{'head': {'text': 'We', 'start': 0, 'end': 2},
                             'tail': {'text': 'AdamW', 'start': 7, 'end': 12}, 'confidence': 0.9}]}
        self.assertEqual(span_issues(raw, text), [])
        raw['relations'][0]['tail']['text'] = 'Adam'
        self.assertEqual(len(span_issues(raw, text)), 1)

    def test_invalid_span_is_recorded_without_offset_repair(self):
        raw = {'text': 'GAE', 'start': True, 'end': 4, 'confidence': float('nan')}
        errors = span_issues(raw, 'GAE ')
        self.assertEqual(len(errors), 2)
        self.assertIs(raw['start'], True)


if __name__ == '__main__':
    unittest.main()
