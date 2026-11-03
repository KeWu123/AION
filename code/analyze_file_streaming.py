"""Verify and analyze a sealed exp_file_streaming study without rerunning it.

Usage (paths are relative to the repository, or absolute)::
  python code/analyze_file_streaming.py --input experiments/... --out reviews/.../analysis

Every input byte is checked against the COMPLETE manifest before CSVs are read.
The output directory must not exist. All failures, censors and truth exclusions
are retained. Timings and byte counters concern this serial FileIO experiment;
they do not measure physical-device I/O or a controlled cold cache.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('forward_wr', 'forward_wor', 'atomic_wr', 'atomic_wor', 'indexed_wr',
        'indexed_wor', 'rebase', 'exact_fused', 'coordinated_wor', 'cohort_wor',
        'known_wr_oracle')
ORACLES = ('exact_fused', 'known_wr_oracle')
CELL = ('dataset', 'function', 'fraction')
SESSION = (*CELL, 'replicate', 'method')
TABLES = ('sessions', 'queries', 'paths', 'specs', 'source_preparation',
          'prefix_preparation', 'selection_preparation')
BOOTSTRAP_SEED = 202610170617
BOOTSTRAP_DRAWS = 10000
# Exact sealed legacy driver identities; no implicit old/new factor default.
# The first is pilot_v1, the second the reviewed pre-factor replay fixture.
LEGACY_PRECISION_FACTORS = {
    '2757a8fb980dc75b8b53e0294ea83def9097589f6127c0e6de12664585a6a3e3': (.25, .5, 1., 2.),
    'a117cd2c916407f2ab79ee2707815b38eff601c27f035604f8e7365ffbc4e504': (.25, .5, 1., 2.),
}
RATIO_METRICS = ('continuation_seconds', 'contribution_evaluations',
                 'actual_requested_bytes', 'actual_returned_bytes',
                 'actual_read_calls', 'logical_new_rows',
                 'peak_query_map_entries', 'retained_cache_payload_bytes')


class InvalidStudy(ValueError):
    """The study does not satisfy its sealed data contract."""


def require(condition, message):
    if not condition:
        raise InvalidStudy(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def safe_file(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path != root, f'unsafe manifest path: {relative}')
    require(path.is_file(), f'missing sealed file: {relative}')
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def number(value):
    """Parse CSV scalars; leave JSON payloads, names and hashes untouched."""
    if value == 'True':
        return True
    if value == 'False':
        return False
    if value == '':
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_csv(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return [{k: number(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def write_csv(path, rows):
    require(bool(rows), f'empty output table: {path}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n', encoding='utf8')


def key(row, names):
    return tuple(row[n] for n in names)


def index_unique(rows, names, label):
    result = {}
    for row in rows:
        k = key(row, names)
        require(k not in result, f'duplicate {label}: {k}')
        result[k] = row
    return result


def assert_grid(index, expected, label):
    actual = set(index)
    require(actual == expected,
            f'{label} grid mismatch: missing={list(expected - actual)[:5]}, extra={list(actual - expected)[:5]}')


def check_counter(counter, label):
    require(counter['read_accounting_complete'] is True, f'incomplete read accounting: {label}')
    for name in ('read_exception_calls', 'read_consumed_on_error_bytes', 'read_error_position_unknown'):
        require(counter[name] == 0, f'read failure in {label}: {name}')
    require(counter['read_requested_bytes'] == counter['read_returned_bytes'], f'short read: {label}')
    for name, value in counter.items():
        require(isinstance(value, (int, float, bool)) and math.isfinite(value) and value >= 0,
                f'invalid counter {label}.{name}')


def verify_seal(study):
    require(not (study / 'INCOMPLETE').exists(), 'study is INCOMPLETE')
    require((study / 'manifest.json').is_file(), 'missing manifest.json')
    manifest = read_json(study / 'manifest.json')
    require(manifest.get('status') == 'COMPLETE', 'manifest status is not COMPLETE')
    require(not (study / 'failure.json').exists(), 'failed study cannot be analyzed as COMPLETE')
    hashes = manifest['output_sha256']
    files = {p.relative_to(study).as_posix() for p in study.rglob('*') if p.is_file() and p != study / 'manifest.json'}
    require(files == set(hashes), f'manifest file inventory mismatch: missing={set(hashes)-files}, extra={files-set(hashes)}')
    verified = {}
    for name, expected in hashes.items():
        actual = sha(safe_file(study, name))
        require(actual == expected, f'output hash mismatch: {name}')
        verified[name] = actual
    before, after = (read_json(study / name) for name in ('sources_before.json', 'sources_after.json'))
    require(before == after == manifest['source_sha256'] == manifest['source_after_sha256'],
            'source before/after hashes disagree')
    for name, expected in before.items():
        require(verified.get('source/' + name) == expected, f'source snapshot mismatch: {name}')
    require(verified['protocol.json'] == manifest['protocol_sha256'], 'protocol hash mismatch')
    for name, expected in manifest['physical_source_after_sha256'].items():
        require(verified.get('physical_sources/' + name) == expected, f'physical after hash mismatch: {name}')
    require({n for n in files if n.startswith('physical_sources/')} ==
            {'physical_sources/' + n for n in manifest['physical_source_after_sha256']},
            'physical source inventory mismatch')
    return manifest, verified


def precision_contract(study, protocol, driver_sha256):
    """Cross-check sealed factors against source AST, without executing it.

    Current protocols must declare factors and the source must both define a
    literal tuple and use it cyclically in query_contract's target expression.
    Missing metadata is accepted only for exact known legacy source hashes,
    additionally checking their inline target expression. Arbitrary metadata,
    an unused constant, and an unknown legacy source are not defaults.
    """
    source = study / 'source' / 'code' / 'exp_file_streaming.py'
    require(sha(source) == driver_sha256, 'precision source hash mismatch')
    try:
        tree = ast.parse(source.read_text(encoding='utf8'))
    except (SyntaxError, UnicodeError) as error:
        raise InvalidStudy('cannot parse sealed precision source') from error
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'query_contract']
    require(len(definitions) == 1, 'missing/duplicate source query_contract')
    target_assignments = [node for node in definitions[0].body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'targets' for target in node.targets)]
    require(len(target_assignments) == 1, 'missing/duplicate source target expression')
    actual_expression = ast.dump(target_assignments[0].value, include_attributes=False)
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'PRECISION_FACTORS' for target in node.targets)]
    declared = protocol.get('heterogeneous_precision_factors')
    if 'heterogeneous_precision_factors' in protocol:
        require(isinstance(declared, list) and declared and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
            for value in declared), 'invalid protocol precision factors')
        require(len(constants) == 1, 'protocol precision factors lack unique source constant')
        require(sum(isinstance(node, ast.Name) and node.id == 'PRECISION_FACTORS' and isinstance(node.ctx, ast.Store)
                    for node in ast.walk(tree)) == 1, 'source precision constant is reassigned')
        try:
            literal = ast.literal_eval(constants[0].value)
        except (ValueError, TypeError, SyntaxError) as error:
            raise InvalidStudy('source precision constant is not literal') from error
        require(isinstance(literal, tuple) and literal and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
            for value in literal), 'invalid source precision factors')
        factors = tuple(float(value) for value in literal)
        require(tuple(declared) == factors, 'protocol/source precision factors mismatch')
        expected = '[args.target * (PRECISION_FACTORS[k % len(PRECISION_FACTORS)] if profile == "heterogeneous" else 1.) for k in range(K)]'
        mode = 'protocol factors cross-checked with literal source constant and cyclic target AST'
    else:
        require(not constants, 'source declares precision factors but protocol metadata is missing')
        require(driver_sha256 in LEGACY_PRECISION_FACTORS, 'unknown legacy precision source without factor metadata')
        factors = LEGACY_PRECISION_FACTORS[driver_sha256]
        expected = '[args.target * ((.25, .5, 1., 2.)[k % 4] if profile == "heterogeneous" else 1.) for k in range(K)]'
        mode = 'exact known legacy source hash and inline cyclic target AST'
    require(actual_expression == ast.dump(ast.parse(expected, mode='eval').body, include_attributes=False),
            'source target expression does not use validated precision factors')
    require(protocol['arguments']['precision_profile'] in ('uniform', 'heterogeneous'), 'unknown precision profile')
    return dict(heterogeneous_precision_factors=list(factors), driver_sha256=driver_sha256, validation=mode)


def check_events(events, queries, reports, session, args):
    K = args['K']
    if session['method'] == 'exact_fused':
        require(events and events[0]['kind'] == 'oracle_all_known'
                and events[0]['queries'] == list(range(K))
                and events[0]['information'] == 'all-functions-known oracle', 'invalid exact oracle event')
        if len(events) > 1:
            N = session['visible_records']
            expected = [(first, min(N, first + args['exact_chunk'])) for first in range(0, N, args['exact_chunk'])]
            require([(e.get('first'), e.get('last')) for e in events[1:]] == expected
                    and all(e['kind'] == 'fused_chunk' and e['queries'] == list(range(K)) for e in events[1:]),
                    'lost/invalid fused chunk event')
        return
    admitted, finishes, commits, releases = {}, {}, {}, {}
    released = None
    for event in events:
        kind = event['kind']
        if kind == 'admit_wave':
            require(released is None, 'admission before epoch release')
            for q in event['queries']:
                require(q not in admitted, 'query admitted twice')
                admitted[q] = event
        elif kind == 'epoch_commit':
            epoch = event['epoch']
            require(released is None and epoch not in commits and event['count'] > 0,
                    'invalid or duplicate commit')
            require(len(set(event['participants'])) == len(event['participants']), 'duplicate participant')
            for q in event['participants']:
                require(q in admitted and q not in finishes, 'commit before admission or after finish')
            commits[epoch] = event
            released = epoch
        elif kind == 'epoch_joint_release':
            epoch = event['epoch']
            require(released == epoch and epoch not in releases, 'release without matching commit')
            for field in ('participants', 'count'):
                require(event[field] == commits[epoch][field], 'committed/released batch mismatch')
            releases[epoch] = event
            released = None
        elif kind == 'finish':
            q = event['query']
            require(released is None and q in admitted and q not in finishes, 'invalid finish event')
            finishes[q] = event
    require(released is None and set(admitted) == set(finishes) == set(range(K)), 'lost admission or finish')
    require(set(commits) == set(releases) == set(range(session['epoch_count'])), 'lost committed/released epoch')
    require(len({e['wave'] for e in admitted.values()}) == session['admission_waves'], 'admission wave count mismatch')
    for q, row in queries.items():
        admission = admitted[q]
        require(row['registration_epoch'] == admission['epoch'] and row['admission_wave'] == admission['wave']
                and row['registration_physical'] == admission['source_h'], 'registration metadata mismatch')
        require(row['n'] == finishes[q]['n'] and row['finish_reason'] == finishes[q]['reason'], 'finish metadata mismatch')
        ordinary = [e for e in commits.values() if q in e['participants']]
        expected_n = 0
        require(len(reports[q]) == 1 + len(ordinary), 'lost registration or ordinary report')
        require(reports[q][0]['stage'] == 'registration' and reports[q][0]['n'] == 0, 'missing registration report')
        for report, commit in zip(reports[q][1:], ordinary):
            j = commit['participants'].index(q)
            require(commit['progress'][j] == expected_n, 'committed progress mismatch')
            expected_n += commit['count']
            require(report['stage'] == 'ordinary' and report['epoch'] == commit['epoch']
                    and report['n'] == expected_n, 'committed batch/report mismatch')
        require(expected_n == row['n'], 'lost query observations')
    if session['method'] == 'known_wr_oracle':
        require(all(e['epoch'] == 0 and e['wave'] == 0 for e in admitted.values()),
                'known oracle did not activate all functions initially')


def verify_study(study):
    manifest, verified = verify_seal(study)
    protocol = read_json(study / 'protocol.json')
    args = protocol['arguments']
    factors = precision_contract(study, protocol, manifest['source_sha256']['code/exp_file_streaming.py'])['heterogeneous_precision_factors']
    require(len(set(args['arms'])) == len(args['arms']) and set(args['arms']) <= set(ARMS), 'unknown/duplicate arms')
    for axis in ('datasets', 'functions', 'fractions'):
        require(len(set(args[axis])) == len(args[axis]) and bool(args[axis]), f'duplicate/empty {axis}')
    require(args['reps'] > 0 and args['K'] > 0, 'nonpositive replicate/query count')
    tables = {name: read_csv(study / (name + '.csv')) for name in TABLES}
    cells = list(itertools.product(args['datasets'], args['functions'], args['fractions']))
    bases = {(*cell, rep) for cell in cells for rep in range(args['reps'])}
    sessions = index_unique(tables['sessions'], SESSION, 'session')
    expected_sessions = {(*base, arm) for base in bases for arm in args['arms']}
    assert_grid(sessions, expected_sessions, 'session')
    queries = index_unique(tables['queries'], (*SESSION, 'query'), 'query')
    assert_grid(queries, {(*s, q) for s in expected_sessions for q in range(args['K'])}, 'query')
    specs = index_unique(tables['specs'], (*CELL, 'replicate', 'query'), 'spec')
    assert_grid(specs, {(*b, q) for b in bases for q in range(args['K'])}, 'spec')
    paths = index_unique(tables['paths'], (*SESSION, 'query', 'report'), 'report')
    require(len(sessions) == manifest['sessions'] and len(queries) == manifest['queries']
            and len(paths) == manifest['all_reports'], 'manifest table count mismatch')
    source_preps = index_unique(tables['source_preparation'], ('dataset', 'replicate'), 'source preparation')
    assert_grid(source_preps, set(itertools.product(args['datasets'], range(args['reps']))), 'source preparation')
    prefix_preps = index_unique(tables['prefix_preparation'], ('dataset', 'replicate', 'fraction'), 'prefix preparation')
    assert_grid(prefix_preps, set(itertools.product(args['datasets'], range(args['reps']), args['fractions'])), 'prefix preparation')
    selection_preps = index_unique(tables['selection_preparation'], (*CELL, 'replicate'), 'selection preparation')
    assert_grid(selection_preps, bases, 'selection preparation')
    caps = args['query_caps'] if args.get('query_caps') is not None else [args['cap']] * args['K']
    targets = [args['target'] * (factors[q % len(factors)] if args['precision_profile'] == 'heterogeneous' else 1.)
               for q in range(args['K'])]
    require(len(caps) == args['K'], 'invalid cap vector')
    alpha = args['alpha'] / args['K']
    report_groups = defaultdict(list)
    for row in tables['paths']:
        require(key(row, (*SESSION, 'query')) in queries and isinstance(row['report'], int)
                and not isinstance(row['report'], bool) and row['report'] > 0, 'report outside declared query grid')
        report_groups[key(row, (*SESSION, 'query'))].append(row)
    exact_truth = {}
    for dataset, audit in manifest['audit'].items():
        require(audit['truth_functions'] == len([s for s in specs if s[0] == dataset]), 'truth function count mismatch')
        for row in audit.get('exact_truth_refinements', []):
            exact_truth[(dataset, row['function'], row['fraction'], row['replicate'], row['query'])] = Fraction(
                int(row['numerator']), int(row['denominator']))
    truth_values = {}
    for row in [*tables['queries'], *tables['paths']]:
        ident = key(row, (*CELL, 'replicate', 'query'))
        q = row['query']
        require(row['target_relative'] == targets[q] and row['query_cap'] == caps[q] and row['alpha'] == alpha,
                f'query contract mismatch: {ident}')
        require(row['lower'] >= 0 and row['lower'] <= row['upper'] and math.isfinite(row['lower']), 'invalid interval')
        require(math.isfinite(row['truth']) and row['truth_roundoff_bound'] >= 0, 'invalid audited truth')
        value = (row['truth'], row['truth_roundoff_bound'])
        require(ident not in truth_values or truth_values[ident] == value, 'truth differs between arms/reports')
        truth_values[ident] = value
        if row['truth_classification'] == 'exact rational sum of float64 pairs':
            require(ident in exact_truth, 'missing exact truth refinement')
            theta = exact_truth[ident]
            covered = Fraction.from_float(float(row['lower'])) <= theta and (
                not math.isfinite(row['upper']) or theta <= Fraction.from_float(float(row['upper'])))
        else:
            require(row['truth_classification'] == 'compensated envelope', 'unknown truth classification')
            low, high = row['truth'] - row['truth_roundoff_bound'], row['truth'] + row['truth_roundoff_bound']
            contained = row['lower'] <= low and high <= row['upper']
            disjoint = row['upper'] < low or high < row['lower']
            require(contained or disjoint, 'ambiguous truth needs exact refinement')
            covered = contained
        require(row['covered'] is bool(covered), 'truth coverage classification mismatch')
        target = row['lower'] > 0 and math.isfinite(row['upper']) and (row['upper'] - row['lower']) / row['lower'] <= targets[q]
        require(row['target'] is bool(target), 'target flag differs from interval')
        expected_information = 'all-functions-known oracle' if row['method'] in ORACLES else 'streaming frozen-workload'
        require(row['information'] == expected_information, 'stronger oracle information metadata missing')
    for ident, row in specs.items():
        q = ident[-1]
        spec = read_json_string(row['spec_json'])
        require(spec['query'] == q and spec['function'] == ident[1] and row['target_relative'] == targets[q]
                and row['query_cap'] == caps[q] and row['alpha'] == alpha, 'spec/contract mismatch')
    for ident, row in source_preps.items():
        dataset, rep = ident
        N = manifest['input'][dataset]['population']
        name = f'{dataset}_{rep}.bin'
        require(row['source_sha256'] == verified['physical_sources/' + name]
                == manifest['physical_source_after_sha256'][name], 'source preparation hash mismatch')
        require(row['source_bytes'] == 32 * N == (study / 'physical_sources' / name).stat().st_size
                and row['record_bytes'] == 32 and row['source_seed'] == args['seed'] + rep, 'physical source metadata mismatch')
    metrics = []
    for ident, session in sessions.items():
        dataset, function, fraction, rep, arm = ident
        N = manifest['input'][dataset]['population']
        h = int(N * fraction)
        qq = {q: queries[(*ident, q)] for q in range(args['K'])}
        pp = {q: sorted(report_groups[(*ident, q)], key=lambda r: r['report']) for q in qq}
        require(session['queries'] == args['K'] and session['initial_h'] == h
                and session['source_seed'] == args['seed'] + rep, 'session dimensions mismatch')
        require(read_json_string(session['target_vector_json']) == targets
                and read_json_string(session['cap_vector_json']) == caps, 'session targets/caps differ across arms')
        spec_list = [read_json_string(specs[(*ident[:-1], q)]['spec_json']) for q in qq]
        require(session['all_specs_sha256'] == hashlib.sha256(json.dumps(spec_list, sort_keys=True).encode()).hexdigest(),
                'session function definitions differ from frozen specs')
        information = 'all-functions-known oracle' if arm in ORACLES else 'streaming frozen-workload'
        require(session['information'] == information and session['active_window'] ==
                (args['K'] if arm in ORACLES else min(args['K'], args['active_window'])), 'session information/admission mismatch')
        stem = f'{dataset}_{function}_{fraction}_{rep}_{arm}'
        event_name = f'traces/{stem}.events.json'
        require(session['events_sha256'] == verified[event_name], 'event hash mismatch')
        events = read_json(study / event_name)
        check_events(events, qq, pp, session, args)
        stats = read_json_string(session['backend_stats_json'])
        require(stats['N'] == N and stats['initial_h'] == h and stats['failed'] is False
                and stats['prefix_setup_ownership'] == 'shared', 'backend contract mismatch')
        for section in ('setup', 'continuation', 'prefix_setup'):
            check_counter(stats[section], f'{ident}.{section}')
        prefix = prefix_preps[(dataset, rep, fraction)]
        require(prefix['initial_h'] == h and stats['prefix_setup'] == read_json_string(prefix['common_prefix_stats_json']),
                'repeated shared prefix counters differ from unique setup row')
        require(stats['physical_count'] == session['final_source_prefix'] and stats['seen_count'] == session['visible_records']
                and h <= stats['physical_count'] <= stats['seen_count'] <= N, 'source exposure ledger mismatch')
        for output, counter in [('actual_read_calls', 'read_calls'), ('actual_requested_bytes', 'read_requested_bytes'),
                                ('actual_returned_bytes', 'read_returned_bytes'), ('actual_read_seconds', 'read_seconds')]:
            require(session[output] == stats['continuation'][counter], 'session/read API counters disagree')
        require(session['retained_cache_payload_bytes'] == stats['retained_cache_payload_bytes'], 'cache counters disagree')
        trace = None
        if arm != 'exact_fused':
            trace_name = f'traces/{stem}.npz'
            require(session['trace_sha256'] == verified[trace_name], 'session trace hash mismatch')
            trace = np.load(study / trace_name, allow_pickle=False)
            require(set(trace.files) == {f'q{q}_{field}' for q in qq for field in ('positions', 'fresh', 'u', 'v')}, 'lost trace arrays')
        else:
            require(not session['trace_sha256'], 'exact fused unexpectedly claims a draw trace')
        try:
            for q, row in qq.items():
                require(row['censored'] is (not row['target']), 'censor flag mismatch')
                require(len(pp[q]) == row['reports'] and [r['report'] for r in pp[q]] == list(range(1, row['reports'] + 1)),
                        'lost/duplicate report number')
                require(all(pp[q][-1][f] == row[f] for f in ('n', 'lower', 'upper', 'target', 'covered')), 'final query/report differs')
                require(row['suffix_cap_applies'] is (arm != 'exact_fused'), 'oracle suffix-cap metadata missing')
                require(row['events_sha256'] == session['events_sha256'], 'query events hash mismatch') if arm != 'exact_fused' else None
                require(row['contribution_evaluations'] == row['n'] + row['prefix_rows'], 'query evaluations mismatch')
                if arm == 'exact_fused':
                    require(row['n'] == N and row['finish_reason'] == 'exact_census' and row['unique_positions'] == N,
                            'invalid exact census')
                    continue
                require(row['n'] <= caps[q] and row['trace_sha256'] == session['trace_sha256'], 'cap/trace mismatch')
                arrays = {f: trace[f'q{q}_{f}'] for f in ('positions', 'fresh', 'u', 'v')}
                require(all(a.ndim == 1 and len(a) == row['n'] for a in arrays.values()), 'trace observation count mismatch')
                positions = arrays['positions']
                require(positions.dtype == np.dtype('int64') and arrays['fresh'].dtype == np.dtype('bool'), 'trace position/fresh dtype mismatch')
                require(np.all((0 <= positions) & (positions < N)), 'out-of-population position')
                require(all(a.dtype == np.dtype('float64') and np.all(np.isfinite(a)) and np.all((a >= 0) & (a <= 1))
                            for a in (arrays['u'], arrays['v'])), 'invalid float64 pair trace')
                require(row['draw_positions_sha256'] == array_sha(positions)
                        and row['pairs_sha256'] == hashlib.sha256(arrays['u'].tobytes() + arrays['v'].tobytes()).hexdigest(),
                        'query array hash mismatch')
                require(row['fresh_count'] == int(arrays['fresh'].sum()) and row['unique_positions'] == len(np.unique(positions)),
                        'trace freshness/uniqueness mismatch')
                replacement = arm.endswith('_wr') or arm == 'known_wr_oracle'
                if not replacement:
                    require(row['unique_positions'] == row['n'] and row['suffix_remaining'] == N - row['prefix_rows'] - row['n'],
                            'WoR depletion mismatch')
                if arm.startswith('indexed'):
                    require(row['fresh_count'] == 0, 'indexed virtual cursor incorrectly marked fresh')
                if arm == 'rebase':
                    require(row['prefix_rows'] == row['registration_physical']
                            and np.array_equal(positions, np.arange(row['prefix_rows'], row['prefix_rows'] + row['n'])), 'Rebase prefix/positions mismatch')
                else:
                    require(row['prefix_rows'] == 0, 'unexpected generic prefix observations')
                moments = read_json_string(row['prefix_moments_json'])
                require(moments['n'] == row['prefix_rows'], 'prefix moment count mismatch')
                reason = 'target' if row['target'] else 'census' if not replacement and row['n'] + row['prefix_rows'] == N else 'cap'
                require(row['finish_reason'] == reason and (reason != 'cap' or row['n'] == caps[q]), 'invalid retirement reason')
        finally:
            if trace is not None:
                trace.close()
        for field, total in [('targets', sum(r['target'] for r in qq.values())),
                             ('censored', sum(r['censored'] for r in qq.values())),
                             ('contribution_evaluations', sum(r['contribution_evaluations'] for r in qq.values()))]:
            require(session[field] == total, f'session aggregate mismatch: {field}')
        require(session['scalar_exp_evaluations'] == session['contribution_evaluations'] * len(spec_list[0]['centers']), 'scalar evaluation mismatch')
        numeric = {k: v for k, v in session.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for k, value in numeric.items():
            require(math.isfinite(value) and value >= 0, f'invalid session metric: {k}')
        components = sum(session[k] for k in ('init_seconds', 'sampling_and_cache_seconds', 'function_seconds', 'prefix_prepare_seconds', 'report_seconds'))
        require(components <= session['continuation_seconds'] + 1e-6, 'component times exceed full continuation')
        row = {k: session[k] for k in SESSION}
        row.update({k: v for k, v in numeric.items() if k not in row})
        row.update(population=N, all_reports=sum(len(p) for p in pp.values()),
                   queries_finalized=len(qq), total_query_observations=sum(r['n'] for r in qq.values()),
                   total_fresh_flags=sum(r['fresh_count'] for r in qq.values()),
                   total_query_unique_positions=sum(r['unique_positions'] for r in qq.values()),
                   total_compiled_prefix_rows=sum(r['prefix_rows'] for r in qq.values()),
                   final_query_exclusions=sum(not r['covered'] for r in qq.values()),
                   all_report_exclusions=sum(not r['covered'] for p in pp.values() for r in p),
                   target_completions=sum(r['target'] for r in qq.values()),
                   completed_and_covered_queries=sum(r['target'] and r['covered'] for r in qq.values()),
                   logical_new_rows=stats['seen_count'] - h,
                   contiguous_prefix_new_rows=stats['physical_count'] - h,
                   continuation_other_seconds=max(0., session['continuation_seconds'] - components),
                   information=information, quality_complete=all(r['target'] for r in qq.values()),
                   final_coverage=all(r['covered'] for r in qq.values()),
                   anytime_coverage=all(r['covered'] for p in pp.values() for r in p))
        for k, v in stats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row['backend_' + k] = v
        for section in ('setup', 'continuation'):
            row.update({f'backend_{section}_{k}': v for k, v in stats[section].items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        metrics.append(row)
    for dataset, audit in manifest['audit'].items():
        require(audit['final_query_exclusions'] == sum(not r['covered'] for r in tables['queries'] if r['dataset'] == dataset)
                and audit['all_report_exclusions'] == sum(not r['covered'] for r in tables['paths'] if r['dataset'] == dataset),
                'manifest exclusion totals disagree with retained rows')
    return manifest, verified, protocol, tables, metrics


def read_json_string(value):
    return json.loads(value)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and np.all(np.isfinite(values)), 'nonfinite/empty summary input')
    p10, median, p90 = np.quantile(values, [.1, .5, .9], method='linear')
    return dict(median=float(median), p10=float(p10), p90=float(p90))


def summarize(tables, metrics):
    grouped, qgrouped = defaultdict(list), defaultdict(list)
    for row in metrics:
        grouped[key(row, (*CELL, 'method'))].append(row)
    for row in tables['queries']:
        qgrouped[key(row, (*CELL, 'method'))].append(row)
    rows = []
    for ident, group in grouped.items():
        qq = qgrouped[ident]
        row = dict(zip((*CELL, 'method'), ident))
        row.update(replicates=len(group), queries=len(qq), target_completions=sum(q['target'] for q in qq),
                   censored_queries=sum(q['censored'] for q in qq), final_query_exclusions=sum(not q['covered'] for q in qq),
                   all_reports=sum(g['all_reports'] for g in group), all_report_exclusions=sum(g['all_report_exclusions'] for g in group),
                   information=group[0]['information'], quality_complete=all(g['quality_complete'] for g in group),
                   final_coverage=all(g['final_coverage'] for g in group), anytime_coverage=all(g['anytime_coverage'] for g in group))
        row.update(target_completion_rate=row['target_completions']/len(qq),
                   final_query_coverage_rate=1-row['final_query_exclusions']/len(qq),
                   all_report_coverage_rate=1-row['all_report_exclusions']/row['all_reports'])
        for name in group[0]:
            if name not in (*SESSION, 'source_seed') and all(isinstance(g[name], (int, float)) and not isinstance(g[name], bool) for g in group):
                row.update({name + '_' + stat: value for stat, value in quantiles([g[name] for g in group]).items()})
        for name in ('n', 'reports', 'contribution_evaluations', 'fresh_count', 'unique_positions', 'prefix_rows'):
            row.update({'query_' + name + '_' + stat: value for stat, value in quantiles([q[name] for q in qq]).items()})
        rows.append(row)
    return rows


def summarize_queries(tables):
    grouped, report_groups = defaultdict(list), defaultdict(list)
    names = (*CELL, 'method', 'query')
    for row in tables['queries']:
        grouped[key(row, names)].append(row)
    for row in tables['paths']:
        report_groups[key(row, names)].append(row)
    result = []
    for ident, rows in grouped.items():
        reports = report_groups[ident]
        row = dict(zip(names, ident))
        row.update(replicates=len(rows), target_relative=rows[0]['target_relative'],
                   query_cap=rows[0]['query_cap'], nominal_alpha=rows[0]['alpha'],
                   target_completions=sum(r['target'] for r in rows),
                   censored=sum(r['censored'] for r in rows),
                   final_truth_exclusions=sum(not r['covered'] for r in rows),
                   retained_reports=len(reports), all_report_truth_exclusions=sum(not r['covered'] for r in reports),
                   finish_reason_counts_json=json.dumps({reason: sum(r['finish_reason'] == reason for r in rows)
                       for reason in sorted({r['finish_reason'] for r in rows})}, sort_keys=True))
        for metric in ('n', 'contribution_evaluations', 'reports', 'fresh_count', 'unique_positions', 'prefix_rows'):
            row.update({metric + '_' + stat: value for stat, value in quantiles([r[metric] for r in rows]).items()})
        result.append(row)
    return result


def paired_ratios(metrics, args, map_peak_status):
    """A single joint replicate resample is shared by all cells/arms/metrics.

    Ratios are median(numerator arm)/median(denominator arm), never the median
    of paired ratios. Pairing uses replicate IDs, retaining every replicate.
    """
    groups = defaultdict(dict)
    for row in metrics:
        groups[key(row, CELL)][(row['method'], row['replicate'])] = row
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    reps = list(range(args['reps']))
    resamples = rng.integers(0, len(reps), size=(BOOTSTRAP_DRAWS, len(reps)))
    result = []
    for cell in sorted(groups):
        group = groups[cell]
        contrasts = [('forward_wr_vs_all', arm, 'forward_wr') for arm in args['arms'] if arm != 'forward_wr' and 'forward_wr' in args['arms']]
        contrasts += [('sequential_wor_vs_shared', arm, 'forward_wor') for arm in ('atomic_wor', 'coordinated_wor', 'cohort_wor')
                      if arm in args['arms'] and 'forward_wor' in args['arms']]
        contrasts += [('exact_vs_all', 'exact_fused', arm) for arm in args['arms'] if arm != 'exact_fused' and 'exact_fused' in args['arms']]
        for family, numerator, denominator in contrasts:
            arows = [group[(numerator, r)] for r in reps]
            brows = [group[(denominator, r)] for r in reps]
            censored = sum(r['censored'] for r in [*arows, *brows])
            exclusions = sum(r['all_report_exclusions'] for r in [*arows, *brows])
            quality = censored == 0
            for metric in RATIO_METRICS:
                a, b = np.asarray([r[metric] for r in arows]), np.asarray([r[metric] for r in brows])
                ma, mb = float(np.median(a)), float(np.median(b))
                am, bm = np.median(a[resamples], axis=1), np.median(b[resamples], axis=1)
                valid = bm > 0
                boot = am[valid] / bm[valid]
                # Zero denominator is explicitly undefined; no epsilon or hidden draw removal.
                defined = mb > 0 and np.all(valid)
                ci = np.quantile(boot, [.025, .975], method='linear') if defined else (None, None)
                result.append(dict(zip(CELL, cell), contrast=family, metric=metric,
                    numerator_arm=numerator, denominator_arm=denominator,
                    orientation='median(numerator_arm)/median(denominator_arm); >1 means denominator uses less',
                    numerator_median=ma, denominator_median=mb,
                    ratio_of_medians=ma / mb if mb > 0 else None,
                    bootstrap_ci025=ci[0], bootstrap_ci975=ci[1],
                    paired_replicates=len(reps), bootstrap_draws=BOOTSTRAP_DRAWS,
                    zero_denominator_bootstrap_draws=int((~valid).sum()), ratio_ci_defined=defined,
                    censored_queries_both_arms=censored, all_report_exclusions_both_arms=exclusions,
                    comparable_quality=quality, comparable_quality_speedup_claim_permitted=quality and metric == 'continuation_seconds',
                    quality_label='all queries reach the same per-query precision targets at the same nominal alpha' if quality else 'diagnostic cost ratio only: at least one query is censored before target',
                    observed_truth_miss=exclusions > 0,
                    map_peak_measurement_status=map_peak_status if metric == 'peak_query_map_entries' else 'not applicable',
                    metric_claim_permitted=metric != 'peak_query_map_entries' or map_peak_status == 'exact replay',
                    stronger_information_oracle_in_contrast=numerator in ORACLES or denominator in ORACLES,
                    bootstrap_informative=len(reps) > 1))
    return result


def setup_accounting(tables, manifest):
    rows = []
    for prep in tables['source_preparation']:
        row = {k: prep[k] for k in ('dataset', 'replicate')}
        row.update(component='source', deduplication_key='dataset,replicate',
                   seconds=sum(prep[k] for k in ('source_validation_seconds', 'permutation_seconds', 'materialize_and_write_seconds')),
                   source_bytes_written=prep['source_bytes'], **{k: prep[k] for k in ('source_validation_seconds', 'permutation_seconds', 'materialize_and_write_seconds')})
        rows.append(row)
    for prep in tables['prefix_preparation']:
        for component, time_field, stats_field in (
                ('prefix', 'common_prefix_seconds', 'common_prefix_stats_json'),
                ('context', 'common_context_seconds', 'common_context_stats_json')):
            row = {k: prep[k] for k in ('dataset', 'replicate', 'fraction')}
            row.update(component=component, deduplication_key='dataset,replicate,fraction', seconds=prep[time_field])
            stats = read_json_string(prep[stats_field])
            counters = stats if component == 'prefix' else {k: stats['setup'][k] + stats['continuation'][k]
                for k in ('read_calls', 'read_requested_bytes', 'read_returned_bytes', 'read_seconds', 'decode_seconds', 'copy_seconds')}
            if component == 'prefix':
                check_counter(counters, 'common prefix')
            else:
                for section in ('setup', 'continuation'):
                    check_counter(stats[section], 'common context.' + section)
            row.update({k: counters[k] for k in ('read_calls', 'read_requested_bytes', 'read_returned_bytes', 'read_seconds', 'decode_seconds', 'copy_seconds')})
            rows.append(row)
    for prep in tables['selection_preparation']:
        row = {k: prep[k] for k in (*CELL, 'replicate')}
        row.update(component='selection', deduplication_key='dataset,function,fraction,replicate', seconds=prep['common_selection_seconds'])
        rows.append(row)
    totals = []
    for component in ('source', 'prefix', 'context', 'selection'):
        group = [r for r in rows if r['component'] == component]
        totals.append(dict(component=component, unique_cost_rows=len(group), total_seconds=sum(r['seconds'] for r in group),
                           **{f'total_{k}': sum(r.get(k, 0) for r in group) for k in
                              ('read_calls', 'read_requested_bytes', 'read_returned_bytes', 'source_bytes_written')}))
    totals.append(dict(component='jit_warmup', unique_cost_rows=1, total_seconds=manifest['jit_preparation_seconds']))
    totals.append(dict(component='deferred_truth_audit', unique_cost_rows=len(manifest['audit']),
                       total_seconds=sum(a['seconds'] for a in manifest['audit'].values())))
    return rows, totals


PLOT_SCRIPT = r'''"""Editable figures from verified analysis CSVs; no experiment execution."""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
BASE = Path(__file__).resolve().parents[1]
FONT = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8.2,
 'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
 'axes.labelcolor':'#262E38','text.color':'#262E38','xtick.color':'#262E38',
 'ytick.color':'#262E38','legend.frameon':False,'savefig.dpi':300})
COLORS={'forward_wr':'#3E5871','forward_wor':'#0072B2','coordinated_wor':'#009E73',
 'rebase':'#66717D','exact_fused':'#262E38'}
LABELS={'forward_wr':'Sequential WR','forward_wor':'Sequential WoR',
 'coordinated_wor':'Coordinated WoR','rebase':'Rebase','exact_fused':'Exact (all known)'}
SELECTED=tuple(COLORS)
with (BASE/'cell_summary.csv').open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
REPS=sorted({int(r['replicates']) for r in rows})
REPLICATE_NOTE='R=1: descriptive pilot; no replicate uncertainty.' if REPS==[1] else 'Median and 10–90% replicate range; R='+','.join(map(str,REPS))+'.'
cells=sorted({(r['dataset'],r['function'],float(r['fraction'])) for r in rows})
index={(r['dataset'],r['function'],float(r['fraction']),r['method']):r for r in rows}
arms=[a for a in SELECTED if any(r['method']==a for r in rows)]
checks={'font':FONT,'selected_arms':arms,'all_arms_table':'cell_summary.csv',
 'replicates':REPS,'replicate_note':REPLICATE_NOTE,
 'whiskers':'replicate p10 and p90; points are medians',
 'scope':'Full continuation includes policy, evaluation, all reports and teardown; FileIO API bytes, uncontrolled OS cache; no device-I/O measurement.',
 'quality_mark':'x indicates any censored query in that cell/arm; truth misses are reported independently in CSVs',
 'layout_checks':{}}
def export(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer(); overlaps=[]
    for axnum,ax in enumerate(fig.axes):
        tickboxes=[(t.get_text(),t.get_window_extent(renderer)) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        for j,(a,ba) in enumerate(tickboxes):
            for b,bb in tickboxes[j+1:]:
                if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>2 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>2: overlaps.append({'axis':axnum,'labels':[a,b]})
    checks['layout_checks'][name]={'adjacent_xtick_bounding_box_overlaps':overlaps,
        'save_bbox':'tight includes all artists; rotated bounding-box overlap is conservative'}
    for ext in ('pdf','svg','png'): fig.savefig(BASE/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
def panel(metric,ylabel,name):
    fig,ax=plt.subplots(figsize=(7,3.25))
    xs=np.arange(len(cells)); offsets=np.linspace(-.3,.3,max(1,len(arms)))
    positive=[]
    for j,arm in enumerate(arms):
        for i,cell in enumerate(cells):
            if (*cell,arm) not in index: continue
            r=index[(*cell,arm)]
            mid,lo,hi=[float(r[f'{metric}_{s}']) for s in ('median','p10','p90')]
            x=i+offsets[j]; positive.extend(v for v in (mid,lo,hi) if v>0)
            good=r['quality_complete']=='True'
            ax.errorbar(x,mid,yerr=[[max(0,mid-lo)],[max(0,hi-mid)]],fmt='o' if good else 'x',
               color=COLORS[arm],markersize=4.3,capsize=2,elinewidth=.9,label=LABELS[arm] if i==0 else None)
    if positive and max(positive)/min(positive)>50 and all(float(index[(*c,a)][metric+'_p10'])>0 for c in cells for a in arms if (*c,a) in index): ax.set_yscale('log')
    ax.set_xticks(xs); ax.set_xticklabels([f'{d.upper()}\n{f.replace("weighted_price","weighted price")}\n$h/N={h:g}$' for d,f,h in cells],fontsize=7.9)
    ax.set_ylabel(ylabel); ax.grid(axis='y',alpha=.17,lw=.6)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=min(len(arms),3),fontsize=7.9,columnspacing=1.5)
    ax.text(0,-.36,REPLICATE_NOTE+' ×: target censor. Truth misses and all arms: CSV.',transform=ax.transAxes,fontsize=7.9)
    fig.subplots_adjust(left=.10,right=.99,bottom=.29,top=.80)
    export(fig,name)
panel('continuation_seconds','Full continuation (s)','file_streaming_continuation')
panel('actual_returned_bytes','Continuation FileIO bytes returned','file_streaming_read_api')
# Additive components are from the same session. Plot the median of each
# component, explicitly not a decomposition of the median total.
parts=[('init_seconds','#AFBECC','Initialize'),('sampling_and_cache_seconds','#3E5871','Sampling + cache'),
 ('function_seconds','#0072B2','Function'),('prefix_prepare_seconds','#009E73','Prefix moments'),
 ('report_seconds','#66717D','Reports'),('continuation_other_seconds','#D9DEE3','Other + teardown')]
NCOLS=min(4,max(1,len(cells))); NROWS=(len(cells)+NCOLS-1)//NCOLS
fig,axes=plt.subplots(NROWS,NCOLS,figsize=(7,2.1*NROWS+1.05),squeeze=False,sharey=True)
SHORT={'forward_wr':'WR','forward_wor':'WoR','coordinated_wor':'Coord.','rebase':'Rebase','exact_fused':'Exact*'}
for ax,cell in zip(axes.flat,cells):
    bottom=np.zeros(len(arms))
    for metric,color,label in parts:
        vals=np.array([float(index[(*cell,a)][metric+'_median']) for a in arms])
        ax.bar(np.arange(len(arms)),vals,bottom=bottom,color=color,width=.72,label=label); bottom+=vals
    ax.set_xticks(np.arange(len(arms))); ax.set_xticklabels([SHORT[a] for a in arms],rotation=35,ha='right',fontsize=7.9)
    ax.set_title(f'{cell[0].upper()}\n{cell[1].replace("weighted_price","weighted price")}\n$h/N={cell[2]:g}$',fontsize=8.2)
    ax.grid(axis='y',alpha=.15,lw=.5)
for ax in axes[:,0]: ax.set_ylabel('Component median (s)')
for ax in list(axes.flat)[len(cells):]: ax.axis('off')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=7.9)
fig.text(.1,.01,'Sum of component medians. Exact*: all functions known. Full continuation is shown separately.',fontsize=7.9)
fig.subplots_adjust(left=.10,right=.99,bottom=.17 if NROWS==1 else .10,top=.74 if NROWS==1 else .82,wspace=.25,hspace=.95)
export(fig,'file_streaming_components')
(BASE/'figures'/'figure_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
'''


def analyze(study, out):
    require(study.is_dir(), f'missing study: {study}')
    require(not out.exists(), f'output exists (write-once): {out}')
    require(not out.is_relative_to(study), 'analysis output must be outside sealed input')
    manifest, verified, protocol, tables, metrics = verify_study(study)
    summaries = summarize(tables, metrics)
    # Older sealed pilots measured maps only after whole batches, which can
    # miss within-batch peaks. Preserve that diagnostic, forbid a peak claim.
    replay_source = (study / 'source' / 'code' / 'exp_file_streaming.py').read_text(encoding='utf8')
    map_peak_status = 'exact replay' if 'peak_map_replay_seconds' in replay_source and all(
        'peak_map_replay_seconds' in r for r in metrics) else 'known-incomplete post-batch diagnostic'
    for row in metrics:
        row['map_peak_measurement_status'] = map_peak_status
    for row in summaries:
        row['map_peak_measurement_status'] = map_peak_status
    ratios = paired_ratios(metrics, protocol['arguments'], map_peak_status)
    common, totals = setup_accounting(tables, manifest)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE').write_text('Analysis not complete until analysis_manifest.json.\n', encoding='utf8')
    try:
        for name in ('source', 'figures'):
            (out / name).mkdir()
        shutil.copyfile(Path(__file__), out / 'source' / 'analyze_file_streaming.py')
        for name in TABLES:
            shutil.copyfile(study / (name + '.csv'), out / (name + '_retained.csv'))
        write_csv(out / 'session_metrics.csv', metrics)
        write_csv(out / 'cell_summary.csv', summaries)
        write_csv(out / 'query_summary.csv', summarize_queries(tables))
        if ratios:
            write_csv(out / 'paired_ratios.csv', ratios)
        write_csv(out / 'common_setup_unique.csv', common)
        write_csv(out / 'common_setup_totals.csv', totals)
        bootstrap = dict(seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS, interval=[.025, .975],
            quantile_method='numpy linear', estimator='median(numerator arm values) / median(denominator arm values)',
            resampling='Draw R replicate IDs with replacement for each bootstrap draw. A single 10000 x R index matrix is shared by all cells, arms and metrics, preserving within-replicate pairing across the whole study. R=protocol reps (20 in the formal study). All paired replicates, including censors and observed truth exclusions, remain eligible for every draw. PCG64 default_rng with the stated seed; no filtering.',
            zero_denominator='Point ratio undefined if denominator median is zero. If any bootstrap denominator is zero, CI is undefined and zero-draw count is reported. No epsilon and no conditional CI.',
            scope='Pointwise 95% percentile intervals, not simultaneous or bias-corrected; R=1 intervals are degenerate and marked uninformative.',
            quality='A continuation ratio is eligible for a comparable-precision speedup statement only if every query in both complete arms reaches the same per-query target at the same nominal alpha. Observed truth misses are reported independently, never filtered or used for selecting paired replicates. Nominal confidence allows nonzero misses. Oracle contrasts retain stronger-information labels.')
        report = dict(status='VERIFIED', input=str(study), input_manifest_sha256=sha(study / 'manifest.json'),
            input_output_hashes_verified=verified, source_snapshot_sha256=manifest['source_sha256'],
            analyzer_source_sha256=sha(Path(__file__)), protocol_arguments=protocol['arguments'],
            precision_contract=precision_contract(study, protocol, manifest['source_sha256']['code/exp_file_streaming.py']),
            integrity_checks=['exact output inventory and all manifest hashes', 'all source snapshots and before/after source hashes',
                'physical source after hashes and byte lengths', 'exact session/query/spec/setup grids',
                'identical frozen function definitions, targets, caps, alpha across arms',
                'all query/report sequences, final coverage and manifest exclusion counts',
                'all generic positions/fresh/float64 pair hashes and observation counts',
                'committed/released/admitted/finished event accounting', 'stronger oracle information and suffix-cap exceptions',
                'backend read accounting and deduplicated common-prefix counters'],
            sessions=len(metrics), queries=len(tables['queries']), all_reports=len(tables['paths']),
            censored_queries=sum(q['censored'] for q in tables['queries']),
            final_query_exclusions=sum(not q['covered'] for q in tables['queries']),
            all_report_exclusions=sum(not r['covered'] for r in tables['paths']),
            summary_quantiles='median and empirical p10/p90, numpy linear quantiles; no rows excluded',
            map_peak_measurement_status=map_peak_status,
            map_peak_limitation='Old post-batch map counters can miss within-batch peaks (including before a census clears state). They remain retained diagnostics; no map-peak ratio claim is permitted unless sealed source records exact replay.',
            nominal_confidence=dict(family_alpha=protocol['arguments']['alpha'], per_query_alpha=protocol['arguments']['alpha']/protocol['arguments']['K'],
                interpretation='Observed coverage is descriptive; nonzero truth misses are possible under nominal confidence. All misses stay in summaries and resamples.'),
            bootstrap=bootstrap, common_setup_accounting='source once per dataset/replicate; prefix and context separately once per dataset/replicate/fraction; function selection once per dataset/function/fraction/replicate. Repeated backend prefix_setup rows are checked, never added again. Common costs are not added to continuation or multiplied by arms/functions.',
            measurement_scope='Actual unbuffered FileIO API calls and logical exposure ledger, serial jobs, warm/uncontrolled OS cache. No physical-device or cold-cache claim. Map entries are counts; cache bytes are retained record payload, not Python RSS.',
            truth_scope='Use sealed offline audit of evaluated float64 pairs; validate retained classification against compensated envelopes or exact rational refinements. Does not rerun kernels or claim a formal transcendental/IEEE proof.',
            timing_scope='Full continuation includes initialization, sampling/cache policy, function evaluation, prefix moments, all reports, teardown. Hashing/serialization/uniqueness and deferred truth are outside. Backend read/decode/copy times are nested subcounters, never added to top-level components.',
            bootstrap_intervals_informative=protocol['arguments']['reps'] > 1)
        write_json(out / 'analysis.json', report)
        write_json(out / 'bootstrap_definition.json', bootstrap)
        write_json(out / 'cell_summary.json', summaries)
        write_json(out / 'paired_ratios.json', ratios)
        (out / 'figures' / 'gen_file_streaming.py').write_text(PLOT_SCRIPT, encoding='utf8')
        exec(compile(PLOT_SCRIPT, str(out / 'figures' / 'gen_file_streaming.py'), 'exec'),
             {'__file__': str(out / 'figures' / 'gen_file_streaming.py'), '__name__': '__main__'})
        (out / 'README.txt').write_text(
            'Verified file-streaming study analysis. Every retained query and report, including censors and truth exclusions, is in *_retained.csv.\n'
            'cell_summary.csv contains all declared arms and median/p10/p90 metrics. session_metrics.csv preserves replicate pairing.\n'
            'paired_ratios.csv states numerator and denominator for each ratio and its quality eligibility. A censored contrast is diagnostic only.\n'
            'Common source, prefix, context, selection costs are separate and deduplicated in common_setup_unique.csv.\n'
            'Editable native matplotlib figures: figures/gen_file_streaming.py; rerun only on a copy to preserve this write-once output seal.\n'
            'analysis.json documents checks, source hashes, scope and bootstrap definition. No physical-device I/O claim.\n', encoding='utf8')
        seal = dict(status='COMPLETE', input_manifest_sha256=sha(study / 'manifest.json'),
                    analyzer_source_sha256=sha(Path(__file__)), environment=dict(python=platform.python_version(), numpy=np.__version__),
                    output_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
                                   if p.is_file() and p.name != 'INCOMPLETE'})
        write_json(out / 'analysis_manifest.json', seal)
        (out / 'INCOMPLETE').unlink()
    except BaseException as error:
        write_json(out / 'failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    study, out = ((ROOT / p).resolve() for p in (args.input, args.out))
    try:
        report = analyze(study, out)
    except (InvalidStudy, FileExistsError) as error:
        parser.exit(2, f'Analysis refused: {error}\n')
    print(json.dumps({k: report[k] for k in ('status', 'sessions', 'queries', 'all_reports', 'censored_queries',
                                            'final_query_exclusions', 'all_report_exclusions')}, indent=2))
    print('COMPLETE: ' + str(out).encode('ascii', 'backslashreplace').decode('ascii'))


if __name__ == '__main__':
    main()
