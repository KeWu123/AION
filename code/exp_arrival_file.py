"""Matched actual-file experiments with scripted round-based query arrivals.

All arms run serially. Arrival gaps count committed service rounds, not
wall-clock milliseconds. Empty rounds do no work. Query latency is measured
from actual admission to its final interval report. Deferred truth never
enters timed execution. See the frozen protocol in the output directory.
"""
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import argparse
import hashlib
import json
import math
import platform
import shutil
import time

import numpy as np
import exp_file_streaming as driver
import exp_file_controlled as base
import exp_history_file as history
import file_workload as workload
from anq.file_store import RECORD_DTYPE, ForwardFileStore, prepare_prefix

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('forward_wor', 'cohort_wor', 'coordinated_wor')

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')

def audit(data, sessions, queries, reports, out):
    """Audit matched definitions, arrivals, every report and actual bounded pairs."""
    records = np.empty(len(data[0]), dtype=RECORD_DTYPE)
    records['id'] = np.arange(len(records), dtype=np.uint64)
    for name, column in zip(('p', 'r', 'q'), data):
        records[name] = column
    truths, exact_truths = {}, {}
    by_id = {s['session_id']: s for s in sessions}
    for s in sessions:
        events = json.loads((out / s['events_file']).read_text())
        specs = json.loads(s['specs_json'])
        epochs = json.loads(s['admission_epochs_json'])
        admitted = {}
        for event in events:
            if event['kind'] == 'admit_wave':
                for k in event['queries']:
                    assert event['epoch'] == epochs[k], (s['session_id'], k, event)
                    admitted[k] = event['epoch']
        assert len(admitted) == len(specs)
        traces = np.load(out / s['trace_file'])
        physical = np.memmap(ROOT / s['source_file'], dtype=RECORD_DTYPE, mode='r')
        for k, spec in enumerate(specs):
            pos = traces[f'q{k}_positions']
            assert len(pos) == len(np.unique(pos))
            payload = physical[pos]
            for name in ('p', 'r', 'q'):
                assert np.array_equal(payload[name], records[payload['id']][name])
            u, v = workload.evaluate(payload, spec, validate=False)
            assert np.array_equal(u, traces[f'q{k}_u']) and np.array_equal(v, traces[f'q{k}_v'])
            key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            if key not in truths:
                a, b = workload.evaluate(records, spec, validate=False)
                su, sv = float(workload._neumaier(a)), float(workload._neumaier(b))
                theta = su / sv
                error = 8 * (math.ulp(su) + abs(theta) * math.ulp(sv)) / sv + 8 * math.ulp(theta)
                truths[key] = (theta, error)
        traces.close()
        del physical
    for row in [*queries, *reports]:
        s = by_id[row['session_id']]
        spec = json.loads(s['specs_json'])[row['query']]
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        theta, error = truths[key]
        contained = row['lower'] <= theta-error and theta+error <= row['upper']
        disjoint = row['upper'] < theta-error or theta+error < row['lower']
        covered = bool(contained)
        if not (contained or disjoint):
            if key not in exact_truths:
                a, b = workload.evaluate(records, spec, validate=False)
                exact_truths[key] = (sum(map(Fraction.from_float, map(float, a)), Fraction()) /
                                    sum(map(Fraction.from_float, map(float, b)), Fraction()))
            exact = exact_truths[key]
            covered = (Fraction.from_float(row['lower']) <= exact and
                       (not math.isfinite(row['upper']) or exact <= Fraction.from_float(row['upper'])))
        row.update(truth=theta, truth_roundoff_bound=error, covered=bool(covered))
    return dict(unique_full_population_truths=len(truths),
                all_report_exclusions=sum(not x['covered'] for x in reports),
                final_exclusions=sum(not x['covered'] for x in queries),
                arrivals_replayed=True, bounded_pairs_replayed=True,
                exact_truth_refinements=len(exact_truths))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--datasets', nargs='+', default=['tpch', 'tpcds'])
    parser.add_argument('--synthetic-n', type=int, default=0)
    parser.add_argument('--target', type=float, default=.10)
    parser.add_argument('--source-study', default='experiments/innovation_20261102/history_file_formal_v1')
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    for name in ('traces', 'source', 'warmup'):
        (out / name).mkdir()
    (out / 'INCOMPLETE').write_text('Final manifest required.\n')
    names = sorted(set(driver.SOURCE_NAMES + ('code/exp_history_file.py', 'code/exp_arrival_file.py')))
    hashes = {n: base.sha(ROOT / n) for n in names}
    for n in names:
        dest = out / 'source' / n
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / n, dest)
    protocol = dict(status='FROZEN_BEFORE_TIMING', arguments=vars(args),
        datasets=args.datasets, fraction=.9, queries=8, arrival_gaps=[0, 1, 2],
        precision_profiles=['uniform', 'heterogeneous'], target=args.target,
        precision_factors=list(driver.PRECISION_FACTORS), cap='own census (population size)', block=4096,
        alpha=.05, per_query_alpha='.05/8', methods=list(ARMS),
        arrival_clock='Query k arrives at committed service round k*gap. All active queries receive one block per round. Empty rounds are skipped; there is no modeled wall-clock idle time.',
        definitions='Eight weighted-price functions from the same initial 4096-row context. Only evaluated on admission. Same functions, targets, caps and arrivals across all arms.',
        source='Reuse the 20 sealed random-order files and seeds from the history-file study; no resampling or preparation benefit is claimed.',
        latency='Measured elapsed seconds from admission of the wave to the query final report, including intervening query work. Session continuation includes initialization, control, cache/file access, evaluations, inference and close.',
        accounting='Warm OS cache. New source records and FileIO API bytes are distinct. JIT, trace serialization, map replay and deferred truth excluded from continuation.',
        analysis='All cells and replicates retained. Ratio of medians, pointwise 95% paired whole-source bootstrap, 10000 resamples. Compare Epoch shared with both CohortWoR and SeqWoR.',
        sources=hashes, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    dump(out / 'protocol.json', protocol)
    source_manifest = None
    if not args.synthetic_n:
        source_manifest = json.loads((ROOT / args.source_study / 'manifest.json').read_text())
        assert source_manifest['status'] == 'COMPLETE'
    jit = history.warmup(out / 'warmup')
    sessions, queries, reports, inputs, audits = [], [], [], {}, {}
    def save():
        for name, rows in (('sessions', sessions), ('queries', queries), ('reports', reports)):
            driver.write_csv(out / (name + '.csv'), rows)
    base.original_loader.DATA_ROOT = ROOT / 'data'
    for dataset in args.datasets:
        if args.synthetic_n:
            data = tuple(np.random.default_rng(j+30).random(args.synthetic_n) for j in range(3))
            inputs[dataset] = dict(population=args.synthetic_n, synthetic=True)
        else:
            data, inputs[dataset] = base.original_loader.load(dataset)
        N = len(data[0])
        for rep in range(args.reps):
            seed = 20261102000 + rep
            if args.synthetic_n:
                source = out / f'{dataset}_{rep}.bin'
                base.physical_file(source, data, seed)
            else:
                source = ROOT / args.source_study / 'physical_sources' / f'{dataset}_{rep}.bin'
                assert source.stat().st_size == N * RECORD_DTYPE.itemsize
                assert base.sha(source) == source_manifest['files_sha256'][
                    f'physical_sources/{dataset}_{rep}.bin']
            h = int(.9*N)
            prefix = prepare_prefix(source, N, h)
            with ForwardFileStore(source, N, h, prefix=prefix) as store:
                context = store.read_range(max(0, h-4096), h)
            specs = workload.select_specs(context, 8, 8, 'weighted_price')
            for gap in (0, 1, 2):
                for profile_index, profile in enumerate(('uniform', 'heterogeneous')):
                    cfg = SimpleNamespace(block=4096, cap=N, target=args.target, alpha=.05,
                        active_window=8, precision_profile=profile,
                        admission_epochs=[k*gap for k in range(8)], measure_query_latency=True)
                    shift = (rep + gap + profile_index) % len(ARMS)
                    for arm in np.roll(np.array(ARMS), shift):
                        arm = str(arm)
                        identity = f'{dataset}_r{rep}_g{gap}_{profile}_{arm}'
                        trace = out / 'traces' / (identity + '.npz')
                        try:
                            result, qq, pp = driver.run_session(source, N, h, prefix,
                                [dict(s) for s in specs], arm, seed, cfg, trace)
                        except BaseException:
                            save()
                            raise
                        common = dict(session_id=identity, dataset=dataset, replicate=rep,
                            arrival_gap=gap, precision_profile=profile, method=arm, fraction=.9)
                        for row in (result, *qq, *pp):
                            row.update(common)
                        result.update(specs_json=json.dumps(specs, sort_keys=True),
                            source_seed=seed, source_file=source.relative_to(ROOT).as_posix(),
                            source_sha256='', trace_file=trace.relative_to(out).as_posix(),
                            events_file=trace.with_suffix('.events.json').relative_to(out).as_posix(),
                            admission_epochs_json=json.dumps(cfg.admission_epochs),
                            mean_response_seconds=float(np.mean([q['response_seconds'] for q in qq])),
                            p95_response_seconds=float(np.quantile([q['response_seconds'] for q in qq], .95)),
                            new_source_rows=result['final_source_prefix']-h)
                        sessions.append(result)
                        queries.extend(qq)
                        reports.extend(pp)
                        save()
            print('TIMED', dataset, rep, 'sessions', len(sessions), flush=True)
            del prefix
        # All timed arms for this population finish before full-population auditing.
        local = [s for s in sessions if s['dataset'] == dataset]
        audits[dataset] = audit(data, local, [q for q in queries if q['dataset'] == dataset],
            [r for r in reports if r['dataset'] == dataset], out)
        for source_name in set(s['source_file'] for s in local):
            digest = base.sha(ROOT / source_name)
            for s in local:
                if s['source_file'] == source_name:
                    s['source_sha256'] = digest
        save()
        dump(out / 'audit.json', audits)
        print('AUDITED', dataset, audits[dataset], flush=True)
        del data
    assert hashes == {n: base.sha(ROOT / n) for n in names}
    manifest = dict(status='COMPLETE', sessions=len(sessions), queries=len(queries),
        reports=len(reports), inputs=inputs, audit=audits, jit_seconds=jit,
        source_sha256=hashes, environment=dict(python=platform.python_version(),
            numpy=np.__version__, platform=platform.platform()),
        files_sha256={p.relative_to(out).as_posix(): base.sha(p) for p in out.rglob('*')
            if p.is_file() and p.name not in ('INCOMPLETE', 'manifest.json')})
    dump(out / 'manifest.json', manifest)
    (out / 'INCOMPLETE').unlink()
    print('COMPLETE', len(sessions), len(queries), len(reports), flush=True)

if __name__ == '__main__':
    main()
