"""Summarize every scripted-arrival cell with source-paired uncertainty."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('dataset', 'precision_profile', 'arrival_gap')
METRICS = ('continuation_seconds', 'mean_response_seconds', 'p95_response_seconds',
           'new_source_rows', 'contribution_evaluations', 'actual_returned_bytes')
ARMS = ('forward_wor', 'cohort_wor', 'coordinated_wor')

def rows(path):
    with path.open(newline='', encoding='utf8') as f:
        return list(csv.DictReader(f))

def write(path, data):
    if not data:
        return
    with path.open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    study, out = ROOT / args.study, ROOT / args.out
    meta = json.loads((study / 'manifest.json').read_text())
    protocol = json.loads((study / 'protocol.json').read_text())
    assert meta['status'] == 'COMPLETE' and not (study / 'INCOMPLETE').exists()
    sessions, queries, reports = (rows(study / (n + '.csv')) for n in ('sessions', 'queries', 'reports'))
    assert (len(sessions), len(queries), len(reports)) == (meta['sessions'], meta['queries'], meta['reports'])
    out.mkdir(parents=True, exist_ok=False)
    groups = {}
    for s in sessions:
        groups.setdefault(tuple(s[k] for k in KEYS), []).append(s)
    cells, contrasts = [], []
    ordered = sorted(groups, key=lambda k: (0 if k[0] == 'tpch' else 1,
                     0 if k[1] == 'uniform' else 1, int(k[2])))
    for key in ordered:
        group = groups[key]
        n = len(group) // len(ARMS)
        assert len(group) == n * len(ARMS)
        indices = np.random.default_rng(2026110401).integers(0, n, (10000, n))
        arms = {arm: sorted((s for s in group if s['method'] == arm), key=lambda s: int(s['replicate'])) for arm in ARMS}
        reference_ids = [s['replicate'] for s in arms[ARMS[0]]]
        assert all([s['replicate'] for s in arms[a]] == reference_ids for a in ARMS)
        for i in range(n):
            for field in ('all_specs_sha256', 'target_vector_json', 'cap_vector_json', 'admission_epochs_json', 'source_sha256'):
                assert len({arms[a][i][field] for a in ARMS}) == 1, (key, field)
        for arm, records in arms.items():
            cell = dict(zip(KEYS, key), method=arm, replicates=n,
                        censored=sum(int(s['censored']) for s in records))
            for metric in METRICS:
                cell[metric] = float(np.median([float(s[metric]) for s in records]))
            cells.append(cell)
        own = arms['coordinated_wor']
        for reference in ('forward_wor', 'cohort_wor'):
            for metric in METRICS:
                a = np.array([float(s[metric]) for s in own])
                b = np.array([float(s[metric]) for s in arms[reference]])
                ratio = float(np.median(a) / np.median(b))
                boot = np.median(a[indices], axis=1) / np.median(b[indices], axis=1)
                lo, hi = map(float, np.quantile(boot, [.025, .975]))
                contrasts.append(dict(zip(KEYS, key), method='coordinated_wor', reference=reference,
                    metric=metric, ratio=ratio, ci_low=lo, ci_high=hi, replicates=n))
    write(out / 'cell_summary.csv', cells)
    write(out / 'paired_ratios.csv', contrasts)
    def row_for(key, reference, metric):
        return next(c for c in contrasts if tuple(c[k] for k in KEYS) == key and
                    c['reference'] == reference and c['metric'] == metric)
    def point(key, metric):
        return f"{row_for(key, 'cohort_wor', metric)['ratio']:.2f}"
    def interval(key, reference):
        c = row_for(key, reference, 'continuation_seconds')
        return f"{c['ratio']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}]"
    table = [r'\begin{table*}[t]', r'\centering',
        r'\caption{Matched queries with scripted arrivals at $h/T=0.9$ (' + str(protocol['arguments']['reps']) + r' paired sources). Epoch (shared) relative to CohortWoR and SeqWoR. Time is complete continuation; response is the per-session mean admission-to-final-report latency. Entries are ratios of medians, with pointwise 95\% paired-bootstrap intervals for time.}',
        r'\label{tab:arrivals}', r'\begingroup', r'\small', r'\setlength{\tabcolsep}{5pt}',
        r'\renewcommand{\arraystretch}{1.10}', r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrrr@{}}',
        r'\toprule', r'\textbf{Data} & \textbf{Precision} & \textbf{Gap} & \textbf{Time / Cohort} & \textbf{Time / Seq} & \textbf{Response / Cohort} & \textbf{Rows / Cohort} & \textbf{Evals / Cohort} \\',
        r'\midrule']
    for index, key in enumerate(ordered):
        dataset, profile, gap = key
        if index and index % 3 == 0:
            table.append(r'\addlinespace[2pt]')
        table.append(' & '.join(('TPC-H' if dataset == 'tpch' else 'TPC-DS',
            'Uniform' if profile == 'uniform' else 'Mixed', gap,
            interval(key, 'cohort_wor'), interval(key, 'forward_wor'),
            point(key, 'mean_response_seconds'), point(key, 'new_source_rows'),
            point(key, 'contribution_evaluations'))) + r' \\')
    target = protocol['target'] * 100
    mixed = ', '.join(f'{target*x:g}'+r'\%' for x in protocol['precision_factors'])
    table += [r'\bottomrule', r'\end{tabular*}', r'\par\smallskip', r'\footnotesize\raggedright',
        r'Gap counts committed rounds of 4096 observations per active query; query $k$ arrives at round $k\cdot\mathrm{gap}$. Uniform precision is ' + f'{target:g}' + r'\%; mixed targets cycle through ' + mixed + r'. Queries can continue to their own census; each spends $0.05/8$. Empty rounds are skipped. Response times measure actual execution after round-driven admission.',
        r'\par\endgroup', r'\end{table*}']
    (out / 'arrival_file_table.tex').write_text('\n'.join(table) + '\n', encoding='utf8')
    exclusions = lambda data: sum(s['covered'].lower() != 'true' for s in data)
    summary = dict(status='COMPLETE', sessions=len(sessions), queries=len(queries), reports=len(reports),
        cells=len(cells), contrasts=len(contrasts), censored=sum(int(s['censored']) for s in sessions),
        final_exclusions=exclusions(queries), all_report_exclusions=exclusions(reports),
        input_manifest_sha256=hashlib.sha256((study / 'manifest.json').read_bytes()).hexdigest(),
        scope=f"{len(meta['inputs'])} datasets, h/T=.9, 3 round-driven arrival gaps, 2 precision profiles, 3 methods, {protocol['arguments']['reps']} source-paired replicates. Query definitions, quality levels and arrival scripts match across methods.",
        interval='Pointwise 95% paired whole-source percentile bootstrap, ratio of medians, 10000 resamples.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    async_time = [c for c in contrasts if int(c['arrival_gap']) > 0 and
                  c['reference'] == 'cohort_wor' and c['metric'] == 'continuation_seconds']
    numbers = {'ArrivalSessions': len(sessions), 'ArrivalQueries': len(queries),
        'ArrivalReports': len(reports), 'ArrivalCensors': summary['censored'],
        'ArrivalFinalMisses': summary['final_exclusions'], 'ArrivalAllMisses': summary['all_report_exclusions'],
        'ArrivalAsyncTimeLo': f"{min(c['ratio'] for c in async_time):.2f}",
        'ArrivalAsyncTimeHi': f"{max(c['ratio'] for c in async_time):.2f}"}
    for dataset, prefix in (('tpch', 'ArrivalTpch'), ('tpcds', 'ArrivalTpcds')):
        key = (dataset, 'heterogeneous', '1')
        if key not in groups:
            continue
        for metric, name in (('continuation_seconds', 'Time'), ('new_source_rows', 'Rows'),
                             ('mean_response_seconds', 'Response'), ('contribution_evaluations', 'Evals')):
            c = row_for(key, 'cohort_wor', metric)
            numbers[prefix+name] = f"{c['ratio']:.2f}"
            numbers[prefix+name+'Low'] = f"{c['ci_low']:.2f}"
            numbers[prefix+name+'High'] = f"{c['ci_high']:.2f}"
    lifecycle_path = ROOT / 'results/history_file/lifecycle.csv'
    if lifecycle_path.is_file():
        for s in rows(lifecycle_path):
            if s['dataset'] == 'tpch' and s['fraction'] == '0.9' and s['method'] == 'coordinated_wor':
                name = {'1':'One', '16':'Sixteen', '64':'SixtyFour'}.get(s['sessions'])
                if name:
                    numbers['Lifecycle'+name] = f"{float(s['lifecycle_median']):.3f}"
                    numbers['LifecycleSeq'+name] = f"{float(s['reference_lifecycle_median']):.3f}"
                    numbers['LifecyclePrep'+name] = f"{float(s['amortized_preparation_median']):.3f}"
    (out / 'arrival_file_numbers.tex').write_text(
        '% Generated from sealed arrival and lifecycle results.\n' +
        ''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in numbers.items()), encoding='utf8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
