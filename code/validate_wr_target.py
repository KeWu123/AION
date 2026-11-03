"""Independent protocol/raw-data validation for the generated WR target study.

The first run recomputes all contexts, audit ratios, and reported intervals.
After an isolated rerun, --final-from-pilot validates raw non-time equality and
all current source/CSV/data hashes without repeating the expensive computation.
"""
from pathlib import Path
import argparse, ast, csv, hashlib, json, math, platform, time
import numpy as np
from anq.ratio import Prefix, RatioTrace

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'
REPORT=ROOT/'reviews/current_audit/wr_target_validation.md'
LOOKS=[4096,8192,16384,32768,65536]
METHODS=['rebase_fixed4096','fresh_wr_to_target','exact_full_scan']
PILOT_REBASE_LITERAL='4096 unseen rows, any target failure retained; late-prefix comparisons match target only when both arms attain it'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_csv(name):return list(csv.DictReader((OUT/name).open(encoding='utf8')))
def key(row):return float(row['fraction']),int(row['replicate'])
def criterion(lo,hi):return bool(lo>0 and (hi-lo)/lo<=.1)


def prove_metadata_only_transition(old_hash,new_hash):
    path=ROOT/'code/exp_wr_target.py';source=path.read_bytes();text=source.decode('utf8')
    tree=ast.parse(text)
    nodes=[n.value for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='rebase'
           and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
    assert len(nodes)==1
    segment=ast.get_source_segment(text,nodes[0]).encode('utf8')
    assert source.count(segment)==1
    restored=source.replace(segment,repr(PILOT_REBASE_LITERAL).encode('utf8'),1)
    restored_hash=hashlib.sha256(restored).hexdigest()
    assert restored_hash==old_hash and hashlib.sha256(source).hexdigest()==new_hash
    return dict(pilot_driver_sha256=old_hash,final_driver_sha256=new_hash,
        old_protocol_rebase=PILOT_REBASE_LITERAL,new_protocol_rebase=nodes[0].value,
        reverse_single_literal_replacement_sha256=restored_hash,
        metadata_only_change_proven_by_whole_file_hash=True)


def validate_records(record,rows,paths):
    assert len(rows)==240
    assert record['protocol']['contexts']==80 and record['protocol']['replications_per_fraction']==20
    assert record['protocol']['checkpoints']==LOOKS and record['protocol']['alpha']==.05
    assert len({(key(r),r['method']) for r in rows})==240
    for p,h in record['source_sha256'].items():assert sha(ROOT/p)==h,('stale source',p)
    for p,h in record['csv_sha256'].items():assert sha(OUT/p)==h,('stale CSV',p)
    assert sha(ROOT/'data/lo_data.npz')==record['data_sha256']
    groups={}
    for p in paths:groups.setdefault(key(p),[]).append(p)
    for r in rows:
        lo,hi=float(r['lower']),float(r['upper']);target=criterion(lo,hi)
        assert target==(r['target']=='True') and (r['censored']=='True')==(not target)
        assert (lo<=float(r['truth'])<=hi)==(r['covered']=='True')
        if r['method']=='fresh_wr_to_target':
            pp=groups[key(r)];n=int(r['new_rows'])
            assert [int(p['n']) for p in pp]==LOOKS[:len(pp)] and int(pp[-1]['n'])==n
            assert int(r['report_count'])==len(pp)
            assert all(p['target']=='False' for p in pp[:-1])
            assert (pp[-1]['target']=='True')==target
            assert target or n==LOOKS[-1]
            assert float(r['postselection_seconds'])>=float(pp[-1]['cumulative_seconds'])>0
            assert all(float(b['cumulative_seconds'])>float(a['cumulative_seconds']) for a,b in zip(pp,pp[1:]))
            assert (float(pp[-1]['lower']),float(pp[-1]['upper']))==(lo,hi)
            assert int(r['contribution_pair_evaluations'])==n
        elif r['method']=='rebase_fixed4096':
            assert int(r['new_rows'])==4096 and int(r['report_count'])==1
            assert int(r['contribution_pair_evaluations'])==int(r['prefix_rows'])+4096
        else:
            assert r['method']=='exact_full_scan' and int(r['new_rows'])==record['protocol']['population']
            assert int(r['report_count'])==1 and int(r['contribution_pair_evaluations'])==record['protocol']['population']
    for p in paths:
        lo,hi=float(p['lower']),float(p['upper'])
        assert criterion(lo,hi)==(p['target']=='True')
        assert float(p['observable_width'])==(hi-lo)/lo
    for s in record['summary']:
        selected=[r for r in rows if float(r['fraction'])==s['fraction'] and r['method']==s['method']]
        assert len(selected)==s['replications']==20
        assert sum(r['target']=='True' for r in selected)==s['target_successes']
        assert sum(r['censored']=='True' for r in selected)==s['censored']
        assert sum(r['covered']=='True' for r in selected)==s['coverage_count']
        assert np.median([float(r['postselection_seconds']) for r in selected])==s['postselection_seconds_median']
        np.testing.assert_array_equal(np.quantile([float(r['postselection_seconds']) for r in selected],[.1,.9]),s['postselection_seconds_q10_q90'])
        assert np.median([int(r['new_rows']) for r in selected])==s['new_rows_median']
        assert {str(n):sum(int(r['new_rows'])==n for r in selected) for n in sorted({int(r['new_rows']) for r in selected})}==s['new_rows_distribution']
        def wilson95(count):
            z=1.959963984540054;size=len(selected);p=count/size;d=1+z*z/size
            center=(p+z*z/(2*size))/d;half=z*np.sqrt(p*(1-p)/size+z*z/(4*size*size))/d
            return [max(0.,center-half),min(1.,center+half)]
        np.testing.assert_allclose(wilson95(s['target_successes']),s['target_wilson95'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(wilson95(s['coverage_count']),s['coverage_wilson95'],rtol=0,atol=1e-12)
    return dict(query_rows=len(rows),checkpoint_rows=len(paths),source_csv_data_hashes_current=True,
        target_censoring_coverage_and_summary_checks=True,cumulative_timer_checks=True,
        first_success_or_cap_stopping_checks=True)


def write_report(result,record):
    fully_reconstructed=result['status']=='current_records_fully_reconstructed'
    final=fully_reconstructed or result['status']=='final_isolated_records_validated'
    lines=['# WR observable-target experiment: independent validation','',
        'Status: '+('current records validated by full raw-context reconstruction.' if fully_reconstructed else 'final records validated against the independently reconstructed snapshot.' if final else 'protocol reconstruction completed.'),'',
        'The validator reconstructs all 80 retained generated queries from raw price/discount columns, the original seed, and the original `choice(T, size=t0+4096, replace=False)` call. It verifies each observed 75% revenue quantile, all 246 WR checkpoints, and every Rebase report. Audit ratios use independently compensated full-relation sums after the threshold is fixed. Fresh WR uses the original independent seed offset, and incremental chunks match the corresponding uninterrupted integer stream exactly.','',
        'All source, input-data, and CSV SHA-256 values match. Query identities, target/censor flags, coverage, report counts, new-row distributions, timing medians/quantiles, and summary counts reconcile with the raw records. WR stops at the first observable `L>0 and (U-L)/L<=0.10` success or the declared 65,536-row cap. No truth-normalized stopping criterion is used.','',
        '## Scope and timing audit','',
        '- Common observed context, selected threshold, input loading, and reconstruction of the existing hidden WoR order are excluded from every postselection arm. Rebase still pays actual selected-prefix function evaluation/aggregation and its future raw gathers; WR pays its own new sampling.','- The WR timer begins before RNG creation and wraps incremental sampling, raw gathers/revenue/mask evaluation, every concatenation, full trace reconstruction, and every interval inversion. Each retained checkpoint time is cumulative. Final elapsed time also includes report bookkeeping. No required operation was found outside that timer.','- Rebase and exact raw-scan costs are remeasured in the same process, with arm order deterministically rotated across contexts. Rebase is fixed at 4,096 future rows; exact reevaluates all 6,001,215 raw pairs. Rotation is balanced to within one first-position count per fraction, rather than independently randomized.','- Retained clocks permit cumulative/summary consistency checks; they do not independently prove hardware elapsed time. Arrays, allocations, caches, and garbage collection are shared by the same-process design. The pilot timing window overlapped other agent computation and must not support final latency claims.','',
        '## Quality matching and limitations','',
        'WR reaches the observable target in all 80 contexts. Rebase reaches it in all 40 contexts at 50%/90% exposure and in none of the 40 contexts at 1%/10% exposure; the latter are retained as censored failures. Thus only the late-exposure Rebase-versus-WR comparison is matched to the same target. Exact full scans reach the target throughout and return substantially greater precision. Calling the entire 80-context experiment universally quality matched would be incorrect.','',
        'This uses one fixed threshold-generation policy and only 20 contexts per exposure fraction. All arms cover audit truth at their retained final reports in this sample; 20/20 coverage has a broad Wilson interval. The finite experiment does not prove a universal speed ordering or an exact empirical error ceiling.','']
    lines += ['Excluding reconstruction of the reserved WoR continuation order favors Rebase; WR pays for its new independent draws.','']
    if final:
        lines += ['## Current measurements','',
            '| Exposed | Arm | Target | Censored | Median new rows | Median postselection seconds |',
            '|---:|---|---:|---:|---:|---:|']
        for r in record['summary']:
            lines.append(f"| {100*r['fraction']:g}% | {r['method']} | {r['target_successes']}/20 | {r['censored']} | {int(r['new_rows_median']):,} | {r['postselection_seconds_median']:.6f} |")
        lines += ['', ('All current query and checkpoint records have been independently reconstructed from raw inputs.' if fully_reconstructed else 'All non-time fields in both final CSVs exactly match the fully recomputed snapshot.')+' Current source, CSV, data, stopping, coverage, and timing-summary checks passed. Clock isolation is an execution condition, not something this validator proves.','']
        if 'pilot_to_final_metadata_only_transition' in result:
            transition=result['pilot_to_final_metadata_only_transition']
            lines += ['## Pilot-to-final source provenance','',
                'Pilot driver SHA-256: `'+transition['pilot_driver_sha256']+'`.',
                'Final driver SHA-256: `'+transition['final_driver_sha256']+'`.',
                'Replacing only the final `protocol.rebase` string literal with its original text reproduces the entire pilot source hash exactly. This proves the change is metadata only; the pilot is retained as executed under its original hash. Final CSV hashes and the independently recomputed pilot input hashes are recorded separately in the validation JSON.','']
    lines += ['Validation JSON: `experiments/results/wr_target_validation.json`. Reproducer: `python code/validate_wr_target.py` (full current reconstruction); `--final-from-pilot` reconciles later timing records with a previously reconstructed snapshot.','']
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text('\n'.join(lines),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final-from-pilot',action='store_true');args=parser.parse_args()
    started=time.perf_counter();record=json.loads((OUT/'wr_target.json').read_text(encoding='utf8'))
    rows=load_csv('wr_target_queries.csv');paths=load_csv('wr_target_checkpoints.csv')
    checks=validate_records(record,rows,paths)
    snapshot=OUT/'wr_target_pilot_protocol_snapshot.json'
    if args.final_from_pilot:
        pilot=json.loads(snapshot.read_text(encoding='utf8'))
        expected=pilot['source_sha256'].copy();driver='code/exp_wr_target.py'
        transition=None
        if record['source_sha256'][driver]!=expected[driver]:
            transition=prove_metadata_only_transition(expected[driver],record['source_sha256'][driver])
            expected[driver]=record['source_sha256'][driver]
        assert record['source_sha256']==expected
        def nontime(rr):return [{k:v for k,v in r.items() if 'seconds' not in k} for r in rr]
        assert nontime(rows)==pilot['query_non_time_records']
        assert nontime(paths)==pilot['checkpoint_non_time_records']
        result=pilot['validation']|dict(status='final_isolated_records_validated',
            final_record_checks=checks,final_non_time_records_match_independently_verified_pilot=True)
        result['independently_recomputed_pilot_input_hashes']={
            'record_sha256':pilot['pilot_record_sha256'],'csv_sha256':pilot['pilot_csv_sha256'],
            'source_sha256':pilot['source_sha256'],'protocol_snapshot_sha256':sha(snapshot)}
        result['limitations'][0]='pilot timing overlap excluded from latency claims; final isolated records validated separately'
        if transition:result['pilot_to_final_metadata_only_transition']=transition
    else:
        original=load_csv('valid_reuse_generated_queries.csv');contexts=[r for r in original if r['method']=='rebase']
        original_wr={key(r):r for r in original if r['method']=='fresh_wr_no_prefix'}
        z=np.load(ROOT/'data/lo_data.npz');price=z['l_extendedprice'];discount=z['l_discount']
        T=len(price);normalizer=float(price.max());revenue=price*(1-discount)/normalizer
        denominator=math.fsum(revenue);audits=[];max_truth_error=0.;interval_checks=0;wr_checks=0
        def evaluate(ids,threshold):
            v=price[ids]*(1-discount[ids])/normalizer
            return v*(v>=threshold),v
        for index,old in enumerate(contexts):
            identity=key(old);seed=int(old['seed']);t0=int(old['prefix_rows']);threshold=float(old['threshold'])
            assert t0==int(T*float(old['fraction']))
            ids=np.random.default_rng(seed).choice(T,size=t0+4096,replace=False)
            cv=price[ids[:t0]]*(1-discount[ids[:t0]])/normalizer
            assert float(np.quantile(cv,.75))==threshold
            actual_truth=math.fsum(revenue[revenue>=threshold])/denominator
            max_truth_error=max(max_truth_error,abs(actual_truth-float(old['audit_truth'])))
            assert abs(actual_truth-float(old['audit_truth']))<1e-11
            selected=[r for r in rows if key(r)==identity]
            assert [r['method'] for r in selected]==list(np.roll(np.array(METHODS),index%3))
            assert all(float(r['threshold'])==threshold and int(r['seed'])==seed and int(r['prefix_rows'])==t0 for r in selected)
            assert all(float(r['truth'])==float(old['audit_truth']) for r in selected)
            reb=next(r for r in selected if r['method']=='rebase_fixed4096')
            pu,pv=evaluate(ids[:t0],threshold);u,v=evaluate(ids[t0:],threshold)
            tr=RatioTrace(u,v,T,Prefix.from_rows(pu,pv),.05)
            np.testing.assert_array_equal(tr.interval(iterations=28),[float(reb['lower']),float(reb['upper'])]);interval_checks+=1
            assert (float(reb['lower'])<=actual_truth<=float(reb['upper']))==(reb['covered']=='True')
            pp=[p for p in paths if key(p)==identity];last=int(pp[-1]['n'])
            full=np.random.default_rng(seed+10000000).integers(0,T,size=last)
            chunk_rng=np.random.default_rng(seed+10000000);chunks=[];n=0
            for p in pp:
                checkpoint=int(p['n']);chunks.append(chunk_rng.integers(0,T,size=checkpoint-n));n=checkpoint
                np.testing.assert_array_equal(np.concatenate(chunks),full[:n]);wr_checks+=1
                u,v=evaluate(full[:n],threshold);tr=RatioTrace(u,v,None,Prefix(),.05)
                lo,hi=tr.interval(iterations=28)
                np.testing.assert_array_equal([lo,hi],[float(p['lower']),float(p['upper'])]);interval_checks+=1
                assert criterion(lo,hi)==(p['target']=='True')
                assert (lo<=actual_truth<=hi)==(p['covered']=='True')
                if n==4096:
                    prior=original_wr[identity]
                    np.testing.assert_array_equal([lo,hi],[float(prior['lower']),float(prior['upper'])])
            exact=next(r for r in selected if r['method']=='exact_full_scan')
            assert float(exact['lower'])<=actual_truth<=float(exact['upper'])
            audits.append(dict(fraction=identity[0],replicate=identity[1],threshold=threshold,
                independently_compensated_audit_truth=actual_truth,
                exposed_id_sha256=hashlib.sha256(ids[:t0].tobytes()).hexdigest(),
                rebase_suffix_id_sha256=hashlib.sha256(ids[t0:].tobytes()).hexdigest(),
                fresh_wr_first4096_id_sha256=hashlib.sha256(full[:4096].tobytes()).hexdigest(),
                wr_stop_rows=last,wr_checkpoints=len(pp)))
            if (index+1)%20==0:print('independently validated contexts',index+1,'seconds',round(time.perf_counter()-started,2),flush=True)
        result=dict(status='current_records_fully_reconstructed',checks=checks,
            independently_reconstructed_contexts=len(audits),independently_recomputed_intervals=interval_checks,
            uninterrupted_vs_incremental_rng_prefix_checks=wr_checks,
            original_wr_first4096_intervals_matched=80,independent_compensated_truth_max_abs_difference=max_truth_error,
            contexts=audits,limitations=['hardware timing/isolation cannot be independently established by reconstructing statistical records',
                'low-exposure Rebase failures are censored, not matched-precision successes',
                'WoR order creation is excluded as existing shared context; WR new sampling is included',
                'timer nesting verified by source inspection; no independent hardware clock calibration'])
        snapshot.write_text(json.dumps(dict(source_sha256=record['source_sha256'],
            query_non_time_records=[{k:v for k,v in r.items() if 'seconds' not in k} for r in rows],
            checkpoint_non_time_records=[{k:v for k,v in r.items() if 'seconds' not in k} for r in paths],
            pilot_csv_sha256=record['csv_sha256'],pilot_record_sha256=sha(OUT/'wr_target.json'),validation=result),indent=2),encoding='utf8')
    result['validator_source_sha256']=sha(Path(__file__))
    result['current_input_sha256']={name:sha(OUT/name) for name in ['wr_target.json','wr_target_queries.csv','wr_target_checkpoints.csv']}
    result['elapsed_seconds']=time.perf_counter()-started;result['all_checks_passed']=True
    (OUT/'wr_target_validation.json').write_text(json.dumps(result,indent=2),encoding='utf8');write_report(result,record)
    print(json.dumps({k:v for k,v in result.items() if k not in ['contexts','checks','final_record_checks']},indent=2),flush=True)


if __name__=='__main__':main()
