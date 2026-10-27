import unittest
from copy import deepcopy
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile

from scripts.paper_pipeline.card_enrichment import BASELINE, OUTPUT, SOURCE, author_evidence, reviewed_content, role_candidate, run, source_sentence, scope_match, validate_quote, verified_parses
from scripts.paper_pipeline.core import PARSER_VERSION, cache_path, parse_key, sha, write_json


class CardEnrichmentTests(unittest.TestCase):
    def test_author_superscripts_accents_and_exact_physical_quote(self):
        text = 'A source title\nDejan Slep\u010dev1  Daniela2 Rus3\nAbstract\nWe study solvers.'
        paper = {'pdf': 'a.pdf', 'pdf_sha256': 'abc', 'recorded_authors': [
            {'label': 'Dejan Slep\u010dev', 'status': 'verified'}, {'label': 'Daniela Rus', 'status': 'verified'}]}
        result = author_evidence(paper, {'page': 1, 'text': text})
        self.assertEqual(result['header_matched'], 2)
        for row in result['names']:
            e = row['evidence']
            self.assertEqual(text[e['start']:e['end']], e['quote'])
        self.assertFalse(result['automatically_complete_author_list'])

    def test_author_body_mention_does_not_repair_missing_header_name(self):
        text = 'Paper\nAlice Smith\nAbstract\nWe compare with Bob Jones.'
        paper = {'pdf': 'a.pdf', 'pdf_sha256': 'abc', 'recorded_authors': ['Alice Smith', 'Bob Jones']}
        result = author_evidence(paper, {'page': 1, 'text': text})
        self.assertEqual(result['header_matched'], 1)
        self.assertIsNone(result['names'][1]['evidence'])
        self.assertFalse(result['all_retained_names_header_matched'])

    def test_name_substring_is_not_a_person_match(self):
        paper = {'pdf': 'a.pdf', 'pdf_sha256': 'abc', 'recorded_authors': ['Li']}
        self.assertEqual(author_evidence(paper, {'page': 1, 'text': 'Title\nLiDAR Models\nAbstract'})['header_matched'], 0)

    def test_each_method_has_its_own_predicate_scope(self):
        text = 'Prior work employs OldNet; we propose NewNet based on BaseNet.'
        self.assertEqual(role_candidate('method', text, text.index('OldNet'))['role'], 'unresolved')
        self.assertEqual(role_candidate('method', text, text.index('NewNet'))['role'], 'proposed_candidate')
        self.assertEqual(role_candidate('method', text, text.index('BaseNet'))['role'], 'unresolved')

    def test_negation_and_comparison_are_not_adoption(self):
        text = 'We do not use ResNet.'
        self.assertEqual(role_candidate('method', text, text.index('ResNet'))['role'], 'negated_candidate')
        text = 'We compare our system with ResNet.'
        self.assertEqual(role_candidate('method', text, text.index('ResNet'))['role'], 'compared_candidate')
        text = 'We use ResNet for prediction.'
        self.assertEqual(role_candidate('method', text, text.index('ResNet'))['role'], 'adopted_candidate')

    def test_analysis_is_distinct_from_proposal(self):
        text = 'We present an analysis of attention sinks.'
        self.assertEqual(role_candidate('method', text, text.index('attention'))['role'], 'analyzed_candidate')

    def test_dataset_used_in_own_evaluation(self):
        text = 'We evaluate our method on KITTI and nuScenes.'
        self.assertEqual(role_candidate('dataset', text, text.index('nuScenes'))['role'], 'evaluated_candidate')
        text = 'Earlier work evaluates on KITTI.'
        self.assertEqual(role_candidate('dataset', text, text.index('KITTI'))['role'], 'unresolved')

    def test_source_sentence_is_a_literal_original_range(self):
        text = 'Background.\nWe propose NewNet for images.\nOther work uses OldNet.'
        page = {'page': 2, 'text': text, 'blocks': [{'start': 0, 'end': len(text), 'text': text}]}
        start = text.index('NewNet')
        quote = source_sentence(page, start, start + 6)
        self.assertEqual(text[quote['start']:quote['end']], quote['quote'])
        self.assertEqual(quote['quote'], 'We propose NewNet for images.')
        with self.assertRaises(ValueError):
            source_sentence(page, -1, 2)

    def test_shared_scope_requires_specific_evidence_and_preserves_hyphens(self):
        entity = {'patterns': [r'\battention\s+sinks?\b']}
        self.assertTrue(scope_match('We analyze an attention-sink pattern.', entity))
        self.assertFalse(scope_match('We use ordinary attention for feature fusion.', entity))
        with self.assertRaises(ValueError):
            scope_match('Anything', {'patterns': []})

    def test_subordinate_method_is_not_a_new_contribution(self):
        text = 'We propose NewNet, which uses OldNet for classification.'
        self.assertEqual(role_candidate('method', text, text.index('OldNet'))['role'], 'unresolved')
        text = 'We use a model that compares against OldNet.'
        self.assertEqual(role_candidate('method', text, text.index('OldNet'))['role'], 'unresolved')

    def test_tampered_or_foreign_source_is_rejected(self):
        paper = {'pdf': 'original.pdf', 'pdf_sha256': 'abc'}
        evidence = {'pdf': 'original.pdf', 'pdf_sha256': 'abc', 'page': 1, 'start': 0, 'end': 3, 'quote': 'abc'}
        validate_quote(paper, {1: {'text': 'abcd'}}, evidence)
        with self.assertRaises(ValueError):
            validate_quote(paper, {1: {'text': 'axcd'}}, evidence)
        with self.assertRaises(ValueError):
            validate_quote(paper, {1: {'text': 'abcd'}}, {**evidence, 'pdf': 'other.pdf'})

    def test_use_as_baseline_does_not_become_method_adoption(self):
        text = 'We use ResNet as a strong baseline.'
        self.assertEqual(role_candidate('method', text, text.index('ResNet'))['role'], 'compared_candidate')
        text = 'We use ResNet as backbone and CNN as a baseline.'
        self.assertEqual(role_candidate('method', text, text.index('ResNet'))['role'], 'adopted_candidate')
        self.assertEqual(role_candidate('method', text, text.index('CNN'))['role'], 'compared_candidate')

    def test_base_method_of_a_new_variant_is_not_newly_proposed(self):
        text = 'We propose a variant of ResNet.'
        self.assertEqual(role_candidate('method', text, text.index('ResNet'))['role'], 'unresolved')

    def test_lowercase_new_sentence_cannot_inherit_proposal(self):
        text = 'We propose NewNet. our baseline is OldNet.'
        start = text.index('OldNet')
        self.assertEqual(role_candidate('method', text, start)['role'], 'unresolved')
        page = {'page': 1, 'text': text, 'blocks': [{'start': 0, 'end': len(text), 'text': text}]}
        self.assertEqual(source_sentence(page, start, start + 6)['quote'], 'our baseline is OldNet.')

    def test_author_email_is_not_a_second_byline_name(self):
        text = 'Title\nAlice Example2\nalice@example.org\nAbstract'
        self.assertIn('\nalice@example.org\n', text)
        paper = {'pdf': 'a.pdf', 'pdf_sha256': 'abc', 'recorded_authors': ['Alice Example']}
        result = author_evidence(paper, {'page': 1, 'text': text})
        self.assertEqual(result['header_matched'], 1)
        self.assertEqual(result['names'][0]['evidence']['quote'], 'Alice Example')

    def test_manifest_card_metadata_divergence_is_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as folder:
            vault = Path(folder).resolve()
            manifest = {'papers': [{'id': 'paper:a', 'card': 'correct.md'}]}
            write_json(vault / SOURCE / 'manifest.json', manifest)
            write_json(vault / SOURCE / 'records.json', [])
            write_json(vault / BASELINE / 'manifest.json', {'papers': [{'id': 'paper:a', 'card': 'wrong.md'}]})
            with self.assertRaisesRegex(ValueError, 'manifests differ'):
                run(vault, OUTPUT)

    def test_parser_cache_is_bound_to_completed_hash_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline = Path(folder).resolve()
            paper = {'id': 'paper:a', 'pdf_sha256': 'abc'}
            path = cache_path(baseline, paper)
            data = {'paper_id': 'paper:a', 'pdf_sha256': 'abc', 'parser': PARSER_VERSION, 'pages': []}
            write_json(path, data)
            with closing(sqlite3.connect(baseline / 'state.sqlite')) as db, db:
                db.execute('CREATE TABLE jobs (state TEXT,result TEXT,result_hash TEXT,paper_id TEXT,stage TEXT,input_hash TEXT)')
                db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)', ('done', path.relative_to(baseline).as_posix(), sha(path), 'paper:a', 'parse', parse_key(paper)))
            self.assertEqual(verified_parses(baseline, [paper])[0]['paper:a'], data)
            write_json(path, {**data, 'pages': [{'text': 'tampered'}]})
            with self.assertRaisesRegex(ValueError, 'completed job'):
                verified_parses(baseline, [paper])
            write_json(path, {**data, 'paper_id': 'paper:other'})
            with closing(sqlite3.connect(baseline / 'state.sqlite')) as db, db:
                db.execute('UPDATE jobs SET result_hash=?', (sha(path),))
            with self.assertRaisesRegex(ValueError, 'another source'):
                verified_parses(baseline, [paper])


class ReviewedContentTests(unittest.TestCase):
    def setUp(self):
        self.paper = {'id': 'paper:a', 'pdf': 'a.pdf', 'pdf_sha256': 'source-a'}
        self.text = 'We propose NetA for robust fusion. We evaluate on SetA.'
        self.pages = {1: {'text': self.text}}
        evidence = {'pdf': 'a.pdf', 'pdf_sha256': 'source-a', 'page': 1,
                    'start': 0, 'end': len(self.text), 'quote': self.text}
        self.content = {'paper_id': 'paper:a', 'source_sha256': 'source-a',
                        'main_owner': '/root', 'state': 'main_reviewed_source_fields', 'gaps': [],
                        'fields': {
                            'methods': {'state': 'supported', 'claims': [
                                {'name': 'NetA', 'role': 'proposed', 'text': 'A fusion method.', 'evidence': [evidence]}]},
                            'problem': {'state': 'supported', 'claims': [
                                {'name': 'robust fusion', 'text': 'The target problem.', 'evidence': [evidence]}]},
                            'datasets': {'state': 'supported', 'claims': [
                                {'name': 'SetA', 'role': 'evaluated_on', 'text': 'Evaluation data.', 'evidence': [evidence]}]}}}

    def test_source_reviewed_fields_preserve_all100_acceptance_boundary(self):
        result = reviewed_content(self.paper, self.pages, self.content)
        self.assertEqual(result, self.content)
        self.assertNotIn('all100_cards_usable', result)

    def test_foreign_source_or_unreviewed_owner_cannot_be_admitted(self):
        for key, value in [('paper_id', 'paper:b'), ('source_sha256', 'wrong'),
                           ('main_owner', '/worker'), ('state', 'draft')]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'source binding'):
                reviewed_content(self.paper, self.pages, {**self.content, key: value})

    def test_unknown_does_not_become_supported_and_needs_concrete_gap(self):
        content = deepcopy(self.content)
        content['fields']['datasets'] = {'state': 'unknown', 'claims': []}
        with self.assertRaisesRegex(ValueError, 'unknown fields'):
            reviewed_content(self.paper, self.pages, content)
        content['gaps'] = ['The original does not identify the evaluation split.']
        self.assertEqual(reviewed_content(self.paper, self.pages, content)['fields']['datasets']['state'], 'unknown')
        content['fields']['datasets']['claims'] = self.content['fields']['datasets']['claims']
        with self.assertRaises(ValueError):
            reviewed_content(self.paper, self.pages, content)

    def test_wrong_role_or_field_schema_is_rejected(self):
        for field, role in [('methods', 'evaluated_on'), ('datasets', 'proposed')]:
            content = deepcopy(self.content)
            content['fields'][field]['claims'][0]['role'] = role
            with self.subTest(field=field), self.assertRaises(ValueError):
                reviewed_content(self.paper, self.pages, content)
        content = deepcopy(self.content)
        content['fields']['problem']['claims'][0]['role'] = 'proposed'
        with self.assertRaises(ValueError):
            reviewed_content(self.paper, self.pages, content)

    def test_source_span_mutation_and_boolean_page_are_rejected(self):
        for key, value in [('quote', 'Not the original.'), ('pdf', 'foreign.pdf'), ('page', True)]:
            content = deepcopy(self.content)
            content['fields']['methods']['claims'][0]['evidence'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                reviewed_content(self.paper, self.pages, content)

    def test_missing_evidence_and_duplicate_claim_are_rejected(self):
        content = deepcopy(self.content)
        content['fields']['methods']['claims'][0]['evidence'] = []
        with self.assertRaises(ValueError):
            reviewed_content(self.paper, self.pages, content)
        content = deepcopy(self.content)
        content['fields']['methods']['claims'] *= 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reviewed_content(self.paper, self.pages, content)


if __name__ == '__main__':
    unittest.main()
