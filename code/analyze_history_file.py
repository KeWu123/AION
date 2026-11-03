"""Analyze the complete history-file study, with explicit deployment amortization."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NAMES={'forward_wor':'SeqWoR','cohort_wor':'CohortWoR','atomic_wor':'Epoch (independent)',
       'coordinated_wor':'Epoch (shared)','rebase':'Rebase','indexed_wor':'IndexWoR'}
ARMS=tuple(NAMES)
KEYS=['study','dataset','fraction','active_window']


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def paired_ratio(a,b,indices):
    denominator=np.median(b)
    boot_denominator=np.median(b[indices],axis=1)
    if denominator<=0 or np.any(boot_denominator<=0):
        return float('nan'),float('nan'),float('nan')
    ratios=np.median(a[indices],axis=1)/boot_denominator
    low,high=np.quantile(ratios,[.025,.975])
    return float(np.median(a)/np.median(b)),float(low),float(high)


def analyze(study,out):
    meta=json.loads((study/'manifest.json').read_text(encoding='utf8'))
    assert meta['status']=='COMPLETE' and not (study/'INCOMPLETE').exists()
    out.mkdir(parents=True,exist_ok=False)
    s=pd.read_csv(study/'sessions.csv');q=pd.read_csv(study/'queries.csv');p=pd.read_csv(study/'reports.csv')
    assert len(s)==meta['sessions'] and len(q)==meta['queries'] and len(p)==meta['reports']
    s['new_ids']=s.visible_records-s.initial_h
    s['new_source_rows']=s.final_source_prefix-s.initial_h
    for frame in (q,p):
        if frame.covered.dtype!=bool:frame['covered']=frame.covered.astype(str).str.lower().eq('true')
    sweep=s[s.study=='concurrency']
    assert sweep.groupby(['dataset','replicate']).all_specs_sha256.nunique().max()==1
    assert sweep.groupby(['dataset','replicate']).target_vector_json.nunique().max()==1
    cells=[];contrasts=[];rng=np.random.default_rng(20261102071)
    for key,group in s.groupby(KEYS):
        counts=group.groupby('method').size()
        assert set(counts.index)==set(ARMS) and counts.nunique()==1
        n=int(counts.iloc[0]);indices=rng.integers(0,n,(10000,n))
        reference=group[group.method=='forward_wor'].sort_values('replicate')
        if key[0]=='concurrency':
            # Query definitions and quality targets match at each replicate.
            assert group.groupby('replicate').all_specs_sha256.nunique().max()==1
            assert group.groupby('replicate').target_vector_json.nunique().max()==1
        for arm in ARMS:
            arm_rows=group[group.method==arm].sort_values('replicate')
            assert np.array_equal(arm_rows.replicate,reference.replicate)
            ids=set(arm_rows.session_id)
            qq=q[q.session_id.isin(ids)]; pp=p[p.session_id.isin(ids)]
            row=dict(zip(KEYS,key));row.update(method=arm,replicates=n,
                continuation_median=float(arm_rows.continuation_seconds.median()),
                api_mib_median=float(arm_rows.actual_returned_bytes.median()/2**20),
                new_source_rows_median=float(arm_rows.new_source_rows.median()),
                new_ids_median=float(arm_rows.new_ids.median()),
                evaluations_median=float(arm_rows.contribution_evaluations.median()),
                selection_median=float(arm_rows.selection_seconds.median()),
                censored_queries=int(arm_rows.censored.sum()),queries=len(qq),reports=len(pp),
                final_exclusions=int((~qq.covered).sum()),all_report_exclusions=int((~pp.covered).sum()),
                sessions_with_any_exclusion=int(pp.assign(miss=~pp.covered).groupby('session_id')['miss'].max().sum()))
            cells.append(row)
            for metric in ['continuation_seconds','actual_returned_bytes','new_ids','new_source_rows']:
                if metric=='new_source_rows' and arm=='indexed_wor':continue
                estimate,low,high=paired_ratio(arm_rows[metric].to_numpy(),reference[metric].to_numpy(),indices)
                contrast=dict(zip(KEYS,key));contrast.update(method=arm,reference='forward_wor',metric=metric,
                    ratio=estimate,ci_low=low,ci_high=high,replicates=n,
                    defined=bool(np.isfinite(estimate)),
                    undefined_reason='' if np.isfinite(estimate) else 'zero reference median in point or bootstrap sample',
                    interpretation='same-policy execution' if key[0]=='history' else 'matched query definitions')
                contrasts.append(contrast)
    write_csv(out/'cell_summary.csv',cells);write_csv(out/'paired_ratios.csv',contrasts)
    # M counts application sessions sharing one preparation, never benchmark arms.
    # C/M is setup-only accounting. C/M + T(h) is a conditional warm-state model,
    # not an empirical replay that resets and forgets a previously exposed suffix.
    src=pd.read_csv(study/'source_preparation.csv');pre=pd.read_csv(study/'prefix_preparation.csv')
    c=pre.merge(src,on=['dataset','replicate'],validate='many_to_one')
    c['preparation_seconds']=(c.permutation_seconds+c.materialize_and_write_seconds+
                             c.source_validation_seconds+c.cache_seconds)
    write_csv(out/'preparation_by_replicate.csv',c.to_dict('records'))
    lifecycles=[];budgets=[]
    for key,group in s[s.study=='history'].groupby(['dataset','fraction']):
        prep=c[(c.dataset==key[0]) & (c.fraction==key[1])].sort_values('replicate')
        reference=group[group.method=='forward_wor'].sort_values('replicate')
        for arm in ARMS:
            rows=group[group.method==arm].sort_values('replicate')
            assert np.array_equal(rows.replicate,prep.replicate)
            delta=reference.continuation_seconds.to_numpy()-rows.continuation_seconds.to_numpy()
            saved=float(np.median(delta));setup=float(np.median(prep.preparation_seconds))
            budgets.append(dict(dataset=key[0],fraction=key[1],method=arm,
                preparation_median=setup,paired_saving_median=saved,
                setup_budget_recovery_sessions=math.ceil(setup/saved) if saved>0 else '',
                meaning='setup budget divided by paired continuation saving; common setup does not change matched lifecycle ordering'))
            for M in (1,2,4,8,16,32,64,128,256):
                own=prep.preparation_seconds.to_numpy()/M+rows.continuation_seconds.to_numpy()
                ref=prep.preparation_seconds.to_numpy()/M+reference.continuation_seconds.to_numpy()
                lifecycles.append(dict(dataset=key[0],fraction=key[1],method=arm,sessions=M,
                    amortized_preparation_median=float(np.median(prep.preparation_seconds/M)),
                    lifecycle_median=float(np.median(own)),reference_lifecycle_median=float(np.median(ref)),
                    lifecycle_ratio=float(np.median(own)/np.median(ref))))
    write_csv(out/'lifecycle.csv',lifecycles);write_csv(out/'setup_budget_recovery.csv',budgets)
    summary=dict(status='COMPLETE',sessions=len(s),primary_sessions=int((s.study=='history').sum()),
        controlled_sessions=int((s.study=='concurrency').sum()),queries=len(q),reports=len(p),
        censored=int(s.censored.sum()),final_exclusions=int((~q.covered).sum()),
        all_report_exclusions=int((~p.covered).sum()),input=meta['input'],audit=meta['audit'],
        interval='Pointwise 95% paired whole-source percentile bootstrap; ratio of medians; 10000 resamples.',
        lifecycle='Source validation, random permutation, materialization/write and prefix-cache preparation are common to all arms. M counts application sessions sharing snapshot preparation, not independent new permutations. C/M is setup-only accounting; C/M+T(h) models the measured warm state, not an executed multi-session trajectory. Input normalization/loading, JIT and integrity hashing are separately recorded.',
        input_manifest_sha256=hashlib.sha256((study/'manifest.json').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    # Generated manuscript values are derived only from the completed tables.
    df=pd.DataFrame(cells); rr=pd.DataFrame(contrasts)
    history=df[df.study=='history']; shared=rr[(rr.study=='history') & (rr.method=='coordinated_wor') & (rr.metric=='continuation_seconds')]
    numbers={'HistoryFileSessions':int((s.study=='history').sum()),'HistoryFileQueries':int(len(q[q.study=='history'])),
        'HistoryFileReports':int(len(p[p.study=='history'])),'HistoryFileCensors':int(s[s.study=='history'].censored.sum()),
        'HistoryFileFinalMisses':int((~q[q.study=='history'].covered).sum()),
        'HistoryFileAllMisses':int((~p[p.study=='history'].covered).sum()),
        'HistoryFileSharedRatioLo':f'{shared.ratio.min():.2f}',
        'HistoryFileSharedRatioHi':f'{shared.ratio.max():.2f}',
        'NewPrepLo':f'{c.preparation_seconds.min():.2f}',
        'NewPrepHi':f'{c.preparation_seconds.max():.2f}'}
    prep_medians=c.groupby(['dataset','fraction']).preparation_seconds.median()
    numbers.update(NewPrepMedianLo=f'{prep_medians.min():.2f}',NewPrepMedianHi=f'{prep_medians.max():.2f}')
    for window,name in [(1,'One'),(8,'Eight')]:
        row=rr[(rr.study=='concurrency')&(rr.method=='coordinated_wor')&
               (rr.metric=='continuation_seconds')&(rr.active_window==window)].iloc[0]
        numbers['HistoryConcShared'+name]=f'{row.ratio:.2f}'
        numbers['HistoryConcShared'+name+'Low']=f'{row.ci_low:.2f}'
        numbers['HistoryConcShared'+name+'High']=f'{row.ci_high:.2f}'
    recoveries=[x['setup_budget_recovery_sessions'] for x in budgets if
                x['method']=='coordinated_wor' and x['setup_budget_recovery_sessions']!='']
    numbers.update(HistoryBudgetLo=min(recoveries),HistoryBudgetHi=max(recoveries))
    (out/'history_file_numbers.tex').write_text('% Generated by analyze_history_file.py; no hand-entered outcomes.\n'+''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in numbers.items()),encoding='utf8')
    lines=[r'\begin{table*}[t]',r'\centering',
        r"\caption{File-backed sessions with queries selected from each method's evolving history (20 paired source permutations). Median full continuation time and FileIO API bytes.}",
        r'\label{tab:history-file}',r'\begingroup\small',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.12}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}',r'\toprule',r'\tableheaderrow',
        r' & \multicolumn{4}{c}{\tablehead{\textbf{TPC-H}}} & \multicolumn{4}{c}{\tablehead{\textbf{TPC-DS}}} \\',
        r'\cmidrule(lr){2-5}\cmidrule(l){6-9}',
        r' & \multicolumn{2}{c}{$h/T=0.1$} & \multicolumn{2}{c}{$h/T=0.9$} & \multicolumn{2}{c}{$h/T=0.1$} & \multicolumn{2}{c}{$h/T=0.9$} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}',
        r'\textbf{Method} & Time (s) & API MiB & Time (s) & API MiB & Time (s) & API MiB & Time (s) & API MiB \\',r'\midrule']
    for arm in ARMS:
        if arm=='indexed_wor':lines.append(r'\midrule')
        label=NAMES[arm];label=r'\textbf{'+label+'}' if arm in ('atomic_wor','coordinated_wor') else label
        values=[]
        for dataset in ('tpch','tpcds'):
            for fraction in (.1,.9):
                row=history[(history.method==arm)&(history.dataset==dataset)&(history.fraction==fraction)].iloc[0]
                values.extend([f'{row.continuation_median:.3f}',f'{row.api_mib_median:.3f}'])
        lines.append(label+' & '+' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular*}',r'\par\smallskip\footnotesize\raggedright',
        r'All methods receive the same initial context, policy and quality targets; later query functions can differ.',
        r'IndexWoR permits unrestricted access; the other methods share one forward source.',
        r'Continuation includes query selection; the OS cache is warm. Matched-definition comparisons appear in Figure~\ref{fig:lifecycle-concurrency}.',
        r'\par\endgroup',r'\end{table*}']
    (out/'history_file_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({k:summary[k] for k in ['sessions','primary_sessions','controlled_sessions','queries','reports','censored','final_exclusions','all_report_exclusions']},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();analyze(ROOT/a.study,ROOT/a.out)


if __name__=='__main__':main()
