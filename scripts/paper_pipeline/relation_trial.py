"""Run one frozen local learned-relation diagnostic on original paper passages."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .card_enrichment import BASELINE, SOURCE, validate_quote, verified_parses
from .core import Jobs, check_inputs, contained, object_hash, output_root, read_json, sha, write_json
from .local_models import DEFAULT_MODELS, LocalModels
from .local_pipeline import stage

OUTPUT = Path('_working/paper-network-automation-v1/pilot100-relations-v1')


def span_issues(value, source: str, path: str = '$') -> list[dict]:
    """Record malformed model spans without guessing a replacement offset."""
    errors = []
    if isinstance(value, dict):
        if {'text', 'start', 'end'}.issubset(value):
            start, end = value['start'], value['end']
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source)
                    or source[start:end] != value['text']):
                errors.append({'path': path, 'reason': 'model_span_differs_from_input', 'raw': value})
        if 'confidence' in value:
            confidence = value['confidence']
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                errors.append({'path': path, 'reason': 'invalid_model_confidence'})
        for key, item in value.items():
            errors.extend(span_issues(item, source, path + '/' + key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            errors.extend(span_issues(item, source, path + '/' + str(i)))
    return errors


def run(vault: Path) -> dict:
    started = time.perf_counter()
    output = output_root(vault, vault / OUTPUT)
    plan = read_json(output / 'evaluation-plan.json')
    if plan['state'] != 'frozen_before_inference' or len(plan['cases']) != 18:
        raise ValueError('frozen 18-case plan required')
    cases = plan['cases']
    if len({c['case_id'] for c in cases}) != len(cases) or any(type(c['expected_support']) is not bool for c in cases):
        raise ValueError('unique cases and explicit source judgments required')
    source = vault / SOURCE
    for field, name in [('source_manifest_sha256', 'manifest.json'), ('source_records_sha256', 'records.json'),
                        ('model_manifest_sha256', 'models.json')]:
        if plan[field] != sha(source / name):
            raise ValueError('frozen trial input changed: ' + name)
    manifest = read_json(source / 'manifest.json')
    if manifest != read_json(vault / BASELINE / 'manifest.json'):
        raise ValueError('source/parser manifests differ')
    check_inputs(vault, manifest)
    paper_map = {p['id']: p for p in manifest['papers']}
    identifiers = {c['paper_id'] for c in cases}
    parsed, parse_hashes = verified_parses(vault / BASELINE, [paper_map[i] for i in sorted(identifiers)])
    if parse_hashes != plan['parse_cache_hashes']:
        raise ValueError('frozen source cache hashes changed')
    for case in cases:
        validate_quote(paper_map[case['paper_id']], {p['page']: p for p in parsed[case['paper_id']]['pages']}, case['evidence'])
    local = LocalModels(DEFAULT_MODELS, plan['threads'])
    if local.identity != read_json(source / 'models.json'):
        raise ValueError('model/runtime identity differs from frozen paper index')
    identity = {'plan_sha256': sha(output / 'evaluation-plan.json'), 'model': local.identity,
                'code_sha256': sha(Path(__file__)),
                'dependency_sha256': {name: sha(Path(__file__).with_name(name))
                                      for name in ('core.py', 'local_models.py', 'local_pipeline.py', 'card_enrichment.py')}}
    queue = Jobs(output / 'state.sqlite', output)
    extractor, results = None, []
    calls = hits = 0
    try:
        for case in cases:
            key = object_hash([case, identity])
            def produce(case=case):
                nonlocal extractor, calls
                if extractor is None:
                    from gliner2 import AutoExtractor
                    path = contained(local.root, local.manifest['models']['extractor']['directory'])
                    extractor = AutoExtractor.from_pretrained(str(path), map_location='cpu', local_files_only=True).eval()
                    if getattr(extractor.config, 'architecture', None) != 'boundary':
                        raise ValueError('frozen GLiNER boundary architecture required')
                calls += 1
                with local.torch.inference_mode():
                    raw = extractor.extract_relations(case['evidence']['quote'], plan['relation_types'],
                                                      threshold=plan['threshold'], include_confidence=True, include_spans=True)
                if not isinstance(raw, dict):
                    raise ValueError('relation API did not return a structured result')
                return {'case_id': case['case_id'], 'case_hash': object_hash(case), 'raw_model_output': raw,
                        'span_issues': span_issues(raw, case['evidence']['quote']),
                        'semantic_acceptance': 'pending_main_actor_object_judgment'}
            def validate(value, case=case):
                if (value['case_id'] != case['case_id'] or value['case_hash'] != object_hash(case)
                        or value['span_issues'] != span_issues(value['raw_model_output'], case['evidence']['quote'])):
                    raise ValueError('cached relation result differs from its source case')
            value, cached = stage(queue, output, case['paper_id'], 'relation_inference', key, produce, validate)
            hits += cached
            results.append(value)
            print(json.dumps({'case': case['case_id'], 'state': 'cached' if cached else 'inferred',
                              'invalid_spans': len(value['span_issues'])}, ensure_ascii=True), flush=True)
    except Exception as error:
        write_json(output / 'last-error.json', {'state': 'tool_or_runtime_error_not_a_negative_prediction',
                   'identity': identity, 'completed_cases': len(results), 'model_calls': calls,
                   'error_type': type(error).__name__, 'error': str(error), 'full_corpus_unlocked': False}, output)
        raise
    finally:
        queue.close()
    check_inputs(vault, manifest)
    value = {'schema': 'local-relation-trial-results-v1', 'identity': identity, 'cases': results,
             'semantic_acceptance': 'pending_main_actor_object_judgment', 'full_corpus_unlocked': False,
             'formal_publication': False}
    changed = write_json(output / 'results.json', value, output)
    receipt = {'state': 'local_relation_diagnostic_complete_not_semantic_acceptance', 'cases': len(results),
               'model_calls': calls, 'cache_hits': hits, 'invalid_span_cases': sum(bool(r['span_issues']) for r in results),
               'results_changed': changed, 'seconds': round(time.perf_counter() - started, 3),
               'protected_inputs_unchanged': True, 'full_corpus_unlocked': False,
               'runtime_model': {'class': type(extractor).__name__, 'parameters': sum(p.numel() for p in extractor.parameters()),
                                 'training': extractor.training, 'device': str(next(extractor.parameters()).device)} if extractor is not None else None}
    write_json(output / 'last-run.json', receipt, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vault', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.vault.resolve()), ensure_ascii=True))


if __name__ == '__main__':
    main()
