"""Crosscheck retained names and bind model mentions to literal author statements.

This writes working drafts only. A sentence predicate or retrieval scope match
is review evidence, not a method-identity merge or a publication decision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import inspect
import json
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

from .core import PARSER_VERSION, cache_path, check_inputs, contained, object_hash, output_root, parse_key, read_json, sha, write_json
from .local_models import DEFAULT_MODELS, LocalModels
from .local_pipeline import retrieve, validate_result
from .render import md_text, write_text
from .semantic import match_text
from .audit_card_readiness import GENERIC, field_state

SOURCE = Path('_working/paper-network-automation-v1/pilot100-english-v1')
BASELINE = Path('_working/paper-network-automation-v1/pilot100-working-v1')
OUTPUT = Path('_working/paper-network-automation-v1/pilot100-card-enrichment-v1')
SCOPE_KEYS = {'fusion': 'feature-fusion', 'credit': 'temporal-credit',
              'sink': 'attention-sinks', 'equilibrium': 'fixed-point-convergence'}
BOUNDARY = re.compile(r'[.!?]\s+|;\s*|\n\s*\n')
ABBREVIATION = re.compile(r'\b(?:e\.g|i\.e|et\s+al|figs?|eqs?|secs?|dr|mr|mrs|prof|vs|cf)\.$', re.I)
SUBJECT = r'\b(?:we|our\s+(?:method|model|framework|approach|system))\s+'
CUE = re.compile(SUBJECT + r'(?:(?:do|does)\s+not\s+|never\s+)?'
                 r'(?:further\s+|also\s+|instead\s+|then\s+)?'
                 r'(?P<verb>propose|introduce|present|develop|design|use|employ|adopt|apply|'
                 r'analyze|analyse|study|investigate|examine|explore|compare|evaluate|test|benchmark|'
                 r'pretrain|train|address|consider|focus)(?:s|d)?\b', re.I)
NEGATION = re.compile(r'\b(?:not|never|without|rather\s+than|instead\s+of)\b', re.I)
DEPENDENT = re.compile(r'\b(?:based\s+on|inspired\s+by|building\s+on|using|via|with|for|to|of|from|following|which|who|that|while)\b', re.I)
ANALYSIS = re.compile(r'\b(?:analys[ei]s|analysis|results?|comparisons?|evaluations?|surveys?|reviews?|evidence)\b', re.I)


def letters(text: str) -> tuple[str, list[int]]:
    chars, offsets = [], []
    for i, char in enumerate(text):
        for folded in unicodedata.normalize('NFKD', char).casefold():
            if folded.isalpha():
                chars.append(folded)
                offsets.append(i)
    return ''.join(chars), offsets


def literal_evidence(paper: dict, page: int, start: int, end: int, text: str) -> dict:
    return {'pdf': paper['pdf'], 'pdf_sha256': paper['pdf_sha256'], 'page': page,
            'start': start, 'end': end, 'quote': text[start:end]}


def author_evidence(paper: dict, page: dict) -> dict:
    text = page['text']
    stop = re.search(r'\babstract\b', text, re.I)
    end = stop.start() if stop else min(len(text), 2500)
    header = text[:end]
    email_ranges = [m.span() for m in re.finditer(r'\{[^{}]{1,300}\}\s*@\S+|\S+@\S+', header)]
    normal, offsets = letters(header)
    rows = []
    for order, author in enumerate(paper.get('recorded_authors', [])):
        name = author.get('label', '') if isinstance(author, dict) else author
        if not isinstance(name, str) or not name.strip():
            raise ValueError('malformed retained author name')
        needle = letters(name)[0]
        placeholder = bool(re.search(r'\b(?:unknown|et\s+al)\b|相关团队', name, re.I))
        matches = []
        if not placeholder and len(needle) >= 3:
            for found in re.finditer(re.escape(needle), normal):
                start, finish = offsets[found.start()], offsets[found.end() - 1] + 1
                before = start > 0 and header[start - 1].isalpha()
                after = finish < len(header) and header[finish].isalpha()
                email = any(lo <= start < finish <= hi for lo, hi in email_ranges)
                if not before and not after and not email:
                    matches.append(literal_evidence(paper, page['page'], start, finish, text))
        rows.append({'name': name, 'retained_order': order, 'metadata_verified': isinstance(author, dict)
                     and author.get('status') == 'verified', 'retained_metadata': author,
                     'state': 'header_name_crosschecked' if len(matches) == 1 else
                     'placeholder_not_a_name' if placeholder else 'unmatched_or_ambiguous',
                     'evidence': matches[0] if len(matches) == 1 else None})
    count = sum(r['evidence'] is not None for r in rows)
    return {'state': 'retained_names_crosschecked' if rows and count == len(rows) else 'partial_or_missing',
            'names': rows, 'header_matched': count, 'header_end': end,
            'all_retained_names_header_matched': bool(rows) and count == len(rows),
            'automatically_complete_author_list': False,
            'boundary': 'Name strings only; retained order/provenance preserved, not a new person merge or complete-byline assertion.'}


def sentence_boundaries(text: str) -> list[int]:
    return [m.end() for m in BOUNDARY.finditer(text)
            if not (text[m.start()] == '.' and ABBREVIATION.search(text[max(0, m.start() - 12):m.start() + 1]))]


def source_sentence(page: dict, start: int, end: int) -> dict:
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(page['text']):
        raise ValueError('mention range outside original page')
    blocks = [b for b in page['blocks'] if b['start'] <= start < end <= b['end']]
    if len(blocks) != 1:
        raise ValueError('mention must belong to one original block')
    block = blocks[0]
    if page['text'][block['start']:block['end']] != block['text']:
        raise ValueError('original block mismatch')
    cuts = [block['start']] + [block['start'] + end for end in sentence_boundaries(block['text'])] + [block['end']]
    lo = max(x for x in cuts if x <= start)
    hi = min(x for x in cuts if x >= end)
    while lo < start and page['text'][lo].isspace():
        lo += 1
    while hi > end and page['text'][hi - 1].isspace():
        hi -= 1
    return {'page': page['page'], 'start': lo, 'end': hi, 'quote': page['text'][lo:hi]}


def role_candidate(kind: str, sentence: str, mention_start: int, mention_end: int | None = None) -> dict:
    if not 0 <= mention_start < len(sentence):
        raise ValueError('mention outside sentence')
    # A predicate after the entity cannot make it the predicate's object.
    cues = [m for m in CUE.finditer(sentence, 0, mention_start)]
    if not cues:
        return {'role': 'unresolved', 'reason': 'no_preceding_author_predicate'}
    cue = cues[-1]
    prefix = sentence[cue.end():mention_start]
    if len(prefix) > 180 or sentence_boundaries(prefix):
        return {'role': 'unresolved', 'reason': 'predicate_outside_local_clause'}
    verb = cue['verb'].casefold()
    evidence = {'predicate': cue.group(), 'predicate_start': cue.start(), 'predicate_end': cue.end()}
    if NEGATION.search(cue.group() + prefix):
        return {'role': 'negated_candidate', **evidence}
    if verb == 'compare':
        return {'role': 'compared_candidate', **evidence}
    end = mention_end if mention_end is not None else mention_start + len(re.match(r'\S+', sentence[mention_start:])[0])
    if not mention_start < end <= len(sentence):
        raise ValueError('invalid mention end')
    suffix = re.split(r'[,;.!?]|\b(?:and|but|while|whereas)\b', sentence[end:], maxsplit=1, flags=re.I)[0][:100]
    if kind == 'method' and verb in {'use', 'employ', 'adopt', 'apply'} and (
            re.search(r'\bbaselines?\b', prefix, re.I)
            or re.search(r'\b(?:as\s+(?:(?:a|an|the|our|strong|additional|reference)\s+)*baselines?|for\s+comparison)\b', suffix, re.I)):
        return {'role': 'compared_candidate', **evidence}
    if kind in {'method', 'question', 'task'} and (verb in {'analyze', 'analyse', 'study', 'investigate', 'examine', 'explore'}
                                                or verb == 'present' and ANALYSIS.search(prefix)):
        return {'role': 'analyzed_candidate', **evidence}
    if kind == 'method' and not DEPENDENT.search(prefix):
        if verb in {'propose', 'introduce', 'present', 'develop', 'design'} and not ANALYSIS.search(prefix):
            return {'role': 'proposed_candidate', **evidence}
        if verb in {'use', 'employ', 'adopt', 'apply'}:
            return {'role': 'adopted_candidate', **evidence}
    if kind == 'dataset' and verb in {'evaluate', 'test', 'benchmark', 'train', 'pretrain'} and re.search(r'\bon\b', prefix, re.I):
        return {'role': 'training_data_candidate' if verb in {'train', 'pretrain'} else 'evaluated_candidate', **evidence}
    if kind in {'question', 'task'} and verb in {'address', 'consider', 'focus'} and not DEPENDENT.search(prefix):
        return {'role': 'studied_problem_candidate', **evidence}
    return {'role': 'unresolved', 'reason': 'object_or_relation_not_established'}


def scope_match(text: str, entity: dict) -> bool:
    patterns = entity.get('patterns')
    if not patterns:
        raise ValueError('explicit pre-existing question scope required')
    normalized = re.sub(r'[-\u2010-\u2015]', ' ', match_text(text))
    return all(re.search(pattern, normalized, re.I) for pattern in patterns)


def validate_quote(paper: dict, pages: dict, evidence: dict) -> None:
    page = pages.get(evidence['page'])
    if (type(evidence['page']) is not int or page is None
            or type(evidence['start']) is not int or type(evidence['end']) is not int
            or not 0 <= evidence['start'] < evidence['end'] <= len(page['text'])
            or page['text'][evidence['start']:evidence['end']] != evidence['quote']
            or evidence['pdf'] != paper['pdf'] or evidence['pdf_sha256'] != paper['pdf_sha256']):
        raise ValueError('enrichment quote differs from frozen original')


def reviewed_content(paper: dict, pages: dict, addendum: dict) -> dict:
    """Check a main-curated supplement; source syntax is not semantic review."""
    if (addendum.get('paper_id') != paper['id']
            or addendum.get('source_sha256') != paper['pdf_sha256']
            or addendum.get('main_owner') != '/root'
            or addendum.get('state') != 'main_reviewed_source_fields'):
        raise ValueError('content supplement lacks main source binding')
    fields, gaps = addendum.get('fields'), addendum.get('gaps')
    if (not isinstance(fields, dict) or set(fields) != {'methods', 'problem', 'datasets'}
            or not isinstance(gaps, list)
            or any(not isinstance(gap, str) or not gap.strip() for gap in gaps)):
        raise ValueError('explicit content fields and gaps required')
    roles = {'methods': {'proposed', 'used', 'analyzed'},
             'datasets': {'trained_on', 'evaluated_on', 'constructed'}}
    for field, value in fields.items():
        if not isinstance(value, dict) or set(value) != {'state', 'claims'}:
            raise ValueError('explicit field state and claims required')
        claims = value['claims']
        if (value['state'] not in {'supported', 'unknown'} or not isinstance(claims, list)
                or (value['state'] == 'supported') != bool(claims)
                or value['state'] == 'unknown' and not gaps):
            raise ValueError('unknown fields require a gap and no claims')
        seen = set()
        for claim in claims:
            keys = {'name', 'text', 'evidence'} | ({'role'} if field in roles else set())
            if (not isinstance(claim, dict) or set(claim) != keys
                    or any(not isinstance(claim.get(k), str) or not claim[k].strip()
                           for k in ('name', 'text'))
                    or field in roles and claim.get('role') not in roles[field]
                    or not isinstance(claim['evidence'], list) or not claim['evidence']):
                raise ValueError('named typed content claim and source evidence required')
            key = (claim['name'], claim.get('role'), claim['text'])
            if key in seen:
                raise ValueError('duplicate reviewed field claim')
            seen.add(key)
            for evidence in claim['evidence']:
                validate_quote(paper, pages, evidence)
    return addendum


def verified_parses(baseline: Path, papers: list[dict]) -> tuple[dict, dict]:
    """Use the existing parse-job receipt without mutating its owner's queue."""
    parsed, hashes = {}, {}
    with closing(sqlite3.connect((baseline / 'state.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
        for paper in papers:
            path, key = cache_path(baseline, paper), parse_key(paper)
            jobs = db.execute('SELECT state,result,result_hash FROM jobs WHERE paper_id=? AND stage=? AND input_hash=?',
                              (paper['id'], 'parse', key)).fetchall()
            if len(jobs) != 1 or jobs[0][0] != 'done' or jobs[0][1] != path.relative_to(baseline).as_posix() or sha(path) != jobs[0][2]:
                raise ValueError('original parser cache differs from its completed job: ' + paper['id'])
            value = read_json(path)
            if value['paper_id'] != paper['id'] or value['pdf_sha256'] != paper['pdf_sha256'] or value['parser'] != PARSER_VERSION:
                raise ValueError('parser cache belongs to another source or parser')
            parsed[paper['id']], hashes[paper['id']] = value, jobs[0][2]
    return parsed, hashes


def render(output: Path, rows: list[dict], queries: list[dict], summary: dict) -> bool:
    lines = ['# 100 篇来源与关系补充', '',
             '本页是原卡的补充草稿。已核对的作者署名与待审关系分开；没有写入正式图谱。', '',
             '自动句子角色已发现真实误判，尚未获准用于自动写入。仅“主线程接受”条目属于已策展的来源陈述；其余保持待审或已否决。', '',
             '| 状态 | 数量 |', '| --- | --- |']
    lines += ['| ' + md_text(str(k)) + ' | ' + str(v) + ' |' for k, v in summary.items() if isinstance(v, int)]
    for query in queries:
        lines += ['', '## 共享问题：' + md_text(query['label']), '',
                  '按已有问题定义检查原文片段后排序；片段支持讨论该主题，不自动证明本文采用某方法。', '',
                  '| 论文 | 原文主题证据 |', '| --- | --- |']
        for row in query['rows']:
            e = row['passages'][0]
            lines += ['| [[' + row['card'] + '\\|' + md_text(row['title']) + ']] | [[' + e['pdf'] + '#page='
                      + str(e['page']) + '\\|原文 p.' + str(e['page']) + ']] ' + md_text(e['text']) + ' |']
    for row in rows:
        lines += ['', '## ' + md_text(row['title']), '', '[[' + row['card'] + '|原卡]]', '']
        if row['source_state'] == 'source_identity_mismatch':
            lines += ['来源隔离：所附 PDF 题名属于另一篇论文。没有把该原文的作者或方法归给此卡。', '']
            continue
        lines += ['### 作者', '']
        addendum = row.get('main_reviewed_author_addendum')
        if addendum:
            names = addendum['authors']
            lines += ['主线程已核对原文可视署名与顺序：' + '、'.join(md_text(a['name']) for a in names) + '。', '']
        else:
            names = row['author_crosscheck']['names']
            lines += ['已有名单顺序保留；首页姓名匹配不等于全名单或人物身份已完成复核。', '']
        for author in names:
            evidence = author.get('evidence')
            lines += ['- ' + md_text(author['name']) + ('：[[' + evidence['pdf'] + '#page=1|首页原文]] `'
                       + md_text(evidence['quote']).replace('`', '') + '`' if evidence else '：待核对，不补猜。')]
        if not names:
            lines += ['作者缺口仍保留。']
        content = row.get('main_reviewed_content')
        if content:
            lines += ['', '### 主线程核对的内容字段', '',
                      '以下是原文约束的字段补充；完整卡片、规范实体及图谱发布仍待验收。', '',
                      '| 字段 | 对象及用途 | 原文证据 |', '| --- | --- | --- |']
            for field, value in content['fields'].items():
                field_label = {'methods': '方法', 'problem': '问题/任务', 'datasets': '数据集'}[field]
                if value['state'] == 'unknown':
                    lines += ['| ' + field_label + ' | 待核对 | 见下方具体缺口 |']
                for claim in value['claims']:
                    sources = ['[[' + e['pdf'] + '#page=' + str(e['page'])
                               + '\\|原文 p.' + str(e['page']) + ']] ' + md_text(e['quote'])
                               for e in claim['evidence']]
                    role_label = {'proposed': '本文提出', 'used': '本文采用', 'analyzed': '本文分析',
                                  'trained_on': '训练来源', 'evaluated_on': '评测来源',
                                  'constructed': '本文构建'}.get(claim.get('role'))
                    label = claim['name'] + (' (' + role_label + ')' if role_label else '')
                    lines += ['| ' + field_label + ' | ' + md_text(label) + '：' + md_text(claim['text'])
                              + ' | ' + '<br>'.join(sources) + ' |']
            lines += ['', '待补缺口：' + ('；'.join(md_text(gap) for gap in content['gaps'])
                                       if content['gaps'] else '未另列缺口；不代表已完成实体解析或整卡验收。')]
        lines += ['', '### 句子级关系证据（待主线程语义接受）', '',
                  '| 原文实体 | 类型 | 候选角色 | 支持句及原文 |', '| --- | --- | --- | --- |']
        for claim in row['role_candidates']:
            e = claim['sentence_evidence']
            review = claim.get('main_review')
            role = '待审' if review is None else '主线程接受：' + review['accepted_role'] if review['decision'] == 'accepted' else '主线程否决：' + review['reason']
            lines += ['| ' + md_text(claim['text']) + ' | ' + claim['type'] + ' | ' + md_text(role) + ' | '
                      + md_text(e['quote']) + ' [[' + e['pdf'] + '#page=' + str(e['page']) + '\\|原文]] |']
        lines += ['', '未解决的提及数：' + str(len(row['unresolved_mention_ids']))
                  + '。已有策展字段保持在原档案中，本页不覆盖它们。', '']
    return write_text(output / 'cards-and-discovery.md', '\n'.join(lines), output)


def run(vault: Path, output: Path, models=None) -> dict:
    import numpy as np
    started = time.perf_counter()
    output = output_root(vault, vault / output)
    if output != (vault / OUTPUT).resolve():
        raise ValueError('dedicated enrichment output required; prior trials are immutable inputs')
    source, baseline = vault / SOURCE, vault / BASELINE
    manifest, records = read_json(source / 'manifest.json'), read_json(source / 'records.json')
    if manifest != read_json(baseline / 'manifest.json'):
        raise ValueError('source and parser-baseline manifests differ')
    check_inputs(vault, manifest)
    paper_map = {p['id']: p for p in manifest['papers']}
    record_map = {r['paper_id']: r for r in records}
    if len(paper_map) != 100 or set(paper_map) != set(record_map) or len(records) != 100:
        raise ValueError('frozen 100-paper source/record cohort differs')
    exceptions = read_json(output / 'source-exceptions.json')['decisions']
    addenda = read_json(output / 'main-reviewed-authors.json')['papers']
    reviews = read_json(output / 'main-role-decisions.json')['decisions']
    content_file = read_json(output / 'main-reviewed-content.json')
    if content_file.get('schema') != 'main-reviewed-paper-content-v1':
        raise ValueError('main-reviewed content schema required')
    content_rows = content_file['papers']
    content_map = {r['paper_id']: r for r in content_rows}
    if len(content_map) != len(content_rows) or not set(content_map).issubset(paper_map):
        raise ValueError('duplicate or foreign content supplement')
    quarantine = {r['paper_id']: r for r in exceptions}
    additions = {r['paper_id']: r for r in addenda}
    if len(quarantine) != len(exceptions) or len(additions) != len(addenda) or not set(quarantine).issubset(paper_map) or not set(additions).issubset(paper_map):
        raise ValueError('ambiguous or foreign source decisions')
    if set(quarantine) & set(additions):
        raise ValueError('cannot supplement authors of a quarantined binding')
    if set(quarantine) & set(content_map):
        raise ValueError('cannot supplement content of a quarantined binding')
    parsed_sources, parse_hashes = verified_parses(baseline, manifest['papers'])
    profiles = read_json(vault / '_wiki/network/paper-profiles.json')
    retained = defaultdict(list)
    for group_id, group in profiles['groups'].items():
        for pid in group['profile'].get('paper_ids', []):
            retained[pid].append((group_id, group['profile']))
    rows, counts, roles, all_pages = [], Counter(), Counter(), {}
    for paper in manifest['papers']:
        record = record_map[paper['id']]
        parsed = parsed_sources[paper['id']]
        validate_result(record, paper, parsed, record['selection'])
        pages = {p['page']: p for p in parsed['pages']}
        all_pages[paper['id']] = pages
        row = {'paper_id': paper['id'], 'title': paper['title'], 'card': paper['card'], 'pdf': paper['pdf'],
               'source_sha256': paper['pdf_sha256'], 'source_state': 'frozen_source_with_no_known_identity_exception',
               'retained_profiles': [{'group_id': gid, 'field_states': {f: field_state(profile, f)
                                    for f in ('authors', 'methods', 'problem', 'datasets')}} for gid, profile in retained[paper['id']]],
               'role_candidates': [], 'unresolved_mention_ids': [], 'card_semantic_acceptance': 'pending_main_review'}
        if paper['id'] in quarantine:
            decision = quarantine[paper['id']]
            if decision['source_sha256'] != paper['pdf_sha256'] or decision['state'] != 'source_identity_mismatch':
                raise ValueError('source quarantine must bind the observed wrong identity')
            validate_quote(paper, pages, decision['evidence'])
            row.update(source_state='source_identity_mismatch', source_exception=decision)
            counts['quarantined_source_bindings'] += 1
            rows.append(row)
            continue
        row['author_crosscheck'] = author_evidence(paper, pages[1])
        counts['papers_with_all_retained_names_in_header'] += row['author_crosscheck']['all_retained_names_header_matched']
        counts['author_header_name_matches'] += row['author_crosscheck']['header_matched']
        if paper['id'] in additions:
            addendum = additions[paper['id']]
            if addendum['source_sha256'] != paper['pdf_sha256'] or addendum['state'] != 'main_reviewed_original_byline':
                raise ValueError('author supplement must be main-reviewed and source-bound')
            if sha(contained(output, addendum['header_image'])) != addendum['header_image_sha256']:
                raise ValueError('reviewed header image changed')
            names = [a['name'] for a in addendum['authors']]
            if not names or len(names) != len(set(names)):
                raise ValueError('nonempty ordered author names required')
            for order, author in enumerate(addendum['authors']):
                validate_quote(paper, pages, author['evidence'])
                if (author['evidence']['page'] != 1
                        or not letters(author['name'])[0]
                        or letters(author['evidence']['quote'])[0] != letters(author['name'])[0]
                        or author['evidence']['end'] > row['author_crosscheck']['header_end']
                        or type(author['byline_order']) is not int or author['byline_order'] != order):
                    raise ValueError('author supplement differs from original byline')
            row['main_reviewed_author_addendum'] = addendum
            counts['papers_with_main_reviewed_author_supplement'] += 1
        if paper['id'] in content_map:
            row['main_reviewed_content'] = reviewed_content(paper, pages, content_map[paper['id']])
            counts['papers_with_main_reviewed_content_fields'] += 1
            counts['main_reviewed_content_claims'] += sum(len(v['claims']) for v in content_map[paper['id']]['fields'].values())
            counts['papers_with_main_reviewed_four_field_evidence'] += bool(
                paper['id'] in additions
                and all(value['state'] == 'supported' for value in content_map[paper['id']]['fields'].values()))
        for item in record['occurrences']:
            quote = source_sentence(pages[item['evidence']['page']], item['evidence']['start'], item['evidence']['end'])
            rule = role_candidate(item['type'], quote['quote'], item['evidence']['start'] - quote['start'], item['evidence']['end'] - quote['start'])
            generic = item['type'] == 'method' and (len(match_text(item['text'])) < 3 or match_text(item['text']).casefold() in GENERIC)
            if rule['role'] == 'unresolved' or generic:
                row['unresolved_mention_ids'].append(item['id'])
                counts['generic_method_mentions_not_promoted'] += generic
                continue
            evidence = {**quote, 'pdf': paper['pdf'], 'pdf_sha256': paper['pdf_sha256']}
            validate_quote(paper, pages, evidence)
            row['role_candidates'].append({'mention_id': item['id'], 'text': item['text'], 'type': item['type'],
                                          **rule, 'sentence_evidence': evidence, 'mention_evidence': item['evidence'],
                                          'scope_boundary': 'Predicate binds this source occurrence, including sentence modifiers; it does not claim global originality of the shortened entity name.',
                                          'state': 'source_predicate_candidate_not_semantic_acceptance'})
            roles[rule['role']] += 1
        counts['papers_with_role_candidates'] += bool(row['role_candidates'])
        counts['unresolved_mentions'] += len(row['unresolved_mention_ids'])
        rows.append(row)
    all_claims = {c['mention_id']: (r['paper_id'], c) for r in rows for c in r['role_candidates']}
    reviewed = set()
    allowed_roles = {'paper_proposes_method', 'paper_uses_method', 'paper_uses_optimizer', 'paper_evaluates_on_dataset',
                     'paper_trains_on_dataset', 'paper_studies_problem', 'paper_compares_method', 'paper_analyzes_method'}
    for review in reviews:
        mid = review['mention_id']
        if mid in reviewed or mid not in all_claims:
            raise ValueError('duplicate or stale main role decision')
        pid, claim = all_claims[mid]
        if (review['paper_id'] != pid or review['sentence_evidence'] != claim['sentence_evidence']
                or review['decision'] not in {'accepted', 'rejected'}
                or review['decision'] == 'accepted' and review['accepted_role'] not in allowed_roles
                or review['decision'] == 'rejected' and review.get('accepted_role') is not None):
            raise ValueError('main decision differs from its source occurrence')
        claim['main_review'] = review
        reviewed.add(mid)
        counts['role_statements_main_' + review['decision']] += 1
    index = read_json(source / 'index.json')
    vector_path = contained(source, index['vectors'])
    if sha(vector_path) != index['sha256']:
        raise ValueError('frozen document vector index changed')
    vectors = np.load(vector_path, allow_pickle=False)
    if vectors.shape != (index['rows'], 384) or not np.isfinite(vectors).all():
        raise ValueError('invalid source document vectors')
    plan, registry = read_json(source / 'evaluation-plan.json'), read_json(baseline / 'registry.json')
    entities = {e['key']: e for e in registry['entities']}
    cases = {c['id']: c for c in plan['cases']}
    query_texts = [q['explicit_english'] for q in plan['retrieval']]
    models = models or LocalModels(DEFAULT_MODELS, 4)
    if models.identity != read_json(source / 'models.json'):
        raise ValueError('query encoder differs from frozen source index')
    embedding_key = object_hash([query_texts, models.identity, inspect.getsource(LocalModels.encode)])
    existing = read_json(output / 'enrichment.json') if (output / 'enrichment.json').exists() else {}
    cached = existing.get('query_embedding_cache', {})
    reused = cached.get('key') == embedding_key
    query_vectors = np.asarray(cached['vectors'], dtype=np.float32) if reused else models.encode(query_texts, 'query')
    if query_vectors.shape != (len(query_texts), 384) or not np.isfinite(query_vectors).all() or not np.allclose(np.linalg.norm(query_vectors, axis=1), 1, atol=1e-4):
        raise ValueError('invalid query embedding cache')
    queries = []
    for q, query_vector in zip(plan['retrieval'], query_vectors, strict=True):
        entity = entities[SCOPE_KEYS[q['case_id']]]
        eligible = [i for i, doc in enumerate(index['documents']) if doc['paper_id'] not in quarantine and scope_match(doc['text'], entity)]
        docs = [index['documents'][i] for i in eligible]
        result = retrieve(q['explicit_english'], docs, vectors[eligible], query_vector, paper_map)
        result.update(label=cases[q['case_id']]['query'], shared_question=entity['key'], source_scope=entity['patterns'],
                      admitted_passages=len(docs), boundary='source_discussion_candidate_not_adoption_or_problem_solution_edge')
        for returned in result['rows']:
            for doc in returned['passages']:
                validate_quote(paper_map[doc['paper_id']], all_pages[doc['paper_id']], {**doc, 'quote': doc['text']})
        queries.append(result)
    counts.update(paper_drafts=len(rows), role_candidates=sum(roles.values()), source_eligible_paper_drafts=len(rows) - len(quarantine))
    inputs = {name: sha(path) for name, path in {'manifest': source / 'manifest.json', 'records': source / 'records.json',
              'index': source / 'index.json', 'models': source / 'models.json', 'registry': baseline / 'registry.json',
              'profiles': vault / '_wiki/network/paper-profiles.json', 'source_exceptions': output / 'source-exceptions.json',
              'author_addenda': output / 'main-reviewed-authors.json', 'main_role_decisions': output / 'main-role-decisions.json',
              'main_reviewed_content': output / 'main-reviewed-content.json',
              'code': Path(__file__)}.items()}
    value = {'schema': 'source-bound-card-enrichment-v1', 'inputs_sha256': inputs, 'parse_cache_hashes': parse_hashes, 'summary': dict(counts),
             'role_counts': dict(roles), 'rows': rows, 'queries': queries,
             'query_embedding_cache': {'key': embedding_key, 'vectors': query_vectors.tolist()},
             'all100_cards_usable': False, 'full_corpus_unlocked': False, 'formal_publication': False,
             'boundary': 'Main-reviewed byline additions are content evidence. Name matches, sentence predicates, scope definitions and embedding ranks do not automatically accept cards or create shared identity edges.'}
    changed = write_json(output / 'enrichment.json', value, output)
    rendered = render(output, rows, queries, dict(counts))
    check_inputs(vault, manifest)
    receipt = {'state': 'source_enrichment_complete_content_review_pending', 'summary': dict(counts), 'role_counts': dict(roles),
               'query_embedding_reused': reused, 'model_calls': {'extraction': models.extraction_calls, 'embedding_batches': models.embedding_calls},
               'artifacts_changed': {'enrichment.json': changed, 'cards-and-discovery.md': rendered},
               'protected_inputs_unchanged': True, 'full_corpus_unlocked': False, 'formal_publication': False,
               'seconds': round(time.perf_counter() - started, 3)}
    write_json(output / 'last-run.json', receipt, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vault', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.vault.resolve(), OUTPUT), ensure_ascii=True))


if __name__ == '__main__':
    main()
