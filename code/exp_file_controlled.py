"""Matched complete-query study on actual read-only file/cache backends.

Separate from the loaded-array studies. Frozen initial-context functions are
identical across every arm. All actual timing jobs must run serially. Byte
counters describe FileIO calls, including OS-cached reads, not device I/O.
"""
from pathlib import Path
import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import platform
import time
import numpy as np
import exp_interleaved_complete as original_loader
from anq.file_store import RECORD_DTYPE, ForwardFileStore, IndexedFileStore, prepare_prefix
from anq.position_scan import SequentialPositionScan, AtomicPositionScan
from anq.ratio import Prefix, RatioTrace
import file_workload as workload

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('forward_wr','forward_wor','atomic_wr','atomic_wor',
        'indexed_wr','indexed_wor','rebase','exact_fused')
SOURCE_NAMES = ('code/exp_file_controlled.py','code/file_workload.py',
                'code/anq/file_store.py','code/anq/position_scan.py',
                'code/anq/ratio.py','code/exp_interleaved_complete.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',encoding='utf8',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_enclosure(su,sv,N):
    if sv <= 0:
        raise ValueError('population ratio denominator must be positive')
    theta = float(su/sv)
    gamma = (N+16)*np.finfo(float).eps/(1-(N+16)*np.finfo(float).eps)
    return (max(0.,float(np.nextafter(theta*(1-gamma)/(1+gamma),-np.inf))),
            float(np.nextafter(theta*(1+gamma)/(1-gamma),np.inf)))


def is_target(lo,hi,target):
    return bool(lo > 0 and math.isfinite(hi) and (hi-lo)/lo <= target)


def physical_file(path, data, seed):
    """Build the uniform materialized source outside every continuation timer."""
    N = len(data[0])
    validation_started=time.perf_counter()
    for column in data:
        if (column.ndim != 1 or len(column) != N or np.any(~np.isfinite(column))
                or np.any(column < 0) or np.any(column > 1)):
            raise ValueError('source builder requires matching finite normalized columns')
    source_validation_seconds=time.perf_counter()-validation_started
    start = time.perf_counter()
    order = np.random.default_rng(seed).permutation(N)
    order_seconds = time.perf_counter()-start
    start = time.perf_counter()
    record = np.empty(N,dtype=RECORD_DTYPE)
    record['id'] = order.astype(np.uint64)
    for name,column in zip(('p','r','q'),data):
        record[name] = column[order]
    record.tofile(path)
    build_seconds = time.perf_counter()-start
    info = dict(source_seed=seed,source_sha256=sha(path),source_bytes=path.stat().st_size,
                permutation_seconds=order_seconds,materialize_and_write_seconds=build_seconds,
                source_validation_seconds=source_validation_seconds,
                order_sha256=array_sha(order),record_bytes=32)
    assert path.stat().st_size == 32*N
    del record,order
    return info


def run_session(path,N,h,prefix,specs,arm,seed,args,trace_path):
    K = len(specs)
    start = time.perf_counter()
    Store = IndexedFileStore if arm.startswith('indexed') else ForwardFileStore
    store = Store(path,N,h,prefix=prefix)
    scan = None
    state = [dict(us=[],vs=[],positions=[],n=0,reports=0,evals=0,
                  sampling_seconds=0.,function_seconds=0.,report_seconds=0.,
                  prefix_prepare_seconds=0.,peak_map_entries=0,prefix=Prefix()) for _ in specs]
    active = list(range(K))
    done,paths = [],[]
    sample_seconds = function_seconds = report_seconds = prefix_seconds = 0.
    peak_map = 0
    proposal_count = 0
    if arm not in ('rebase','exact_fused'):
        cls = AtomicPositionScan if arm.startswith('atomic') else SequentialPositionScan
        scan = cls(N,seed=seed+700,replacement=arm.endswith('_wr'))
        # For unrestricted indexing, this is an internal virtual-address pool.
        # Only store.gather() changes the actual visible-ID ledger; no full
        # source prefix is exposed to the query policy by the virtual cursor.
        scan.expose(N if arm.startswith('indexed') else h,return_positions=False)
        for s in state:
            s['qid'] = scan.register()
    init_seconds = time.perf_counter()-start
    try:
        if arm == 'exact_fused':
            t = time.perf_counter()
            store.expose_to(N,return_positions=False)
            su,sv = np.zeros(K),np.zeros(K)
            su_parts,sv_parts = [[] for _ in specs],[[] for _ in specs]
            centers = np.asarray([s['centers'] for s in specs])
            scales = np.asarray([1/(2*s['bandwidth']**2) for s in specs])
            sample_seconds += time.perf_counter()-t
            # Streaming fixed-size contiguous chunks; no full-N address list
            # or full-N output arrays for this strong complete comparator.
            for first in range(0,N,args.exact_chunk):
                t = time.perf_counter()
                records = store.read_range(first,min(N,first+args.exact_chunk))
                sample_seconds += time.perf_counter()-t
                t = time.perf_counter()
                a,b = workload.fused_first_moments(records['p'],records['r'],records['q'],
                    centers,scales,specs[0]['function']=='weighted_price')
                for k in range(K):
                    su_parts[k].append(float(a[k]));sv_parts[k].append(float(b[k]))
                function_seconds += time.perf_counter()-t
            t = time.perf_counter()
            su = np.asarray([math.fsum(x) for x in su_parts])
            sv = np.asarray([math.fsum(x) for x in sv_parts])
            for k in range(K):
                lo,hi = exact_enclosure(su[k],sv[k],N)
                state[k].update(n=N,evals=N,reports=1)
                paths.append(dict(query=k,report=1,n=N,physical=N,lower=lo,upper=hi,
                                  target=is_target(lo,hi,args.target),stage='exact_fused'))
                done.append(dict(query=k,n=N,contribution_evaluations=N,lower=lo,upper=hi,
                                 target=is_target(lo,hi,args.target),censored=not is_target(lo,hi,args.target),
                                 reports=1,registration_physical=h,prefix_rows=0,
                                 prefix_moments_json='',draw_positions_sha256='',pairs_sha256=''))
            report_seconds += time.perf_counter()-t
            active=[]
        elif arm == 'rebase':
            for k,s in enumerate(state):
                t = time.perf_counter()
                records = store.read_range(0,h)
                sample_seconds += time.perf_counter()-t
                t = time.perf_counter()
                u,v = workload.evaluate(records,specs[k],validate=False)
                function_seconds += time.perf_counter()-t
                t = time.perf_counter()
                s['prefix'] = workload.prefix_from_pairs(u,v)
                elapsed = time.perf_counter()-t
                prefix_seconds += elapsed
                s['prefix_prepare_seconds'] += elapsed
                s['evals'] += h
        while active:
            if arm == 'rebase':
                count = min(args.block,N-store.physical_rows,
                            min(args.cap-state[k]['n'] for k in active))
                if count <= 0:
                    raise AssertionError('unhandled rebase census/cap')
                t = time.perf_counter()
                old_h = store.physical_rows
                store.expose_to(old_h+count,return_positions=False)
                records = store.read_range(old_h,old_h+count)
                sample_seconds += time.perf_counter()-t
                batches = {k:records for k in active}
                addresses = {k:np.arange(old_h,old_h+count,dtype=np.int64) for k in active}
            else:
                count = min(args.block,min(args.cap-state[k]['n'] for k in active))
                if not arm.endswith('_wr'):
                    count = min(count,min(N-state[k]['n'] for k in active))
                if count <= 0:
                    raise AssertionError('unhandled query census/cap')
                t = time.perf_counter()
                qs = [state[k]['qid'] for k in active]
                if arm.startswith('atomic'):
                    pos,_,trials = scan.draw_epochs(qs,count)
                    proposal_count += int(trials.sum())
                else:
                    pos = np.stack([scan.draw(qid,count)[0] for qid in qs])
                if not arm.startswith('indexed'):
                    store.expose_to(scan.physical,return_positions=False)
                flat_records = store.gather(pos.ravel())
                batches = {k:flat_records[i*count:(i+1)*count] for i,k in enumerate(active)}
                addresses = {k:pos[i] for i,k in enumerate(active)}
                sample_seconds += time.perf_counter()-t
                peak_map = max(peak_map,sum(scan.query_entries(q) for q in qs))
                for k in active:
                    state[k]['peak_map_entries'] = max(state[k]['peak_map_entries'],scan.query_entries(state[k]['qid']))
            finished=[]
            for k in active:
                s = state[k]
                t = time.perf_counter()
                u,v = workload.evaluate(batches[k],specs[k],validate=False)
                s['us'].append(u);s['vs'].append(v)
                s['positions'].append(addresses[k].copy())
                s['n'] += count;s['evals'] += count
                elapsed = time.perf_counter()-t
                s['function_seconds'] += elapsed;function_seconds += elapsed
                t = time.perf_counter()
                us,vs = np.concatenate(s['us']),np.concatenate(s['vs'])
                population = None if arm.endswith('_wr') else N
                tr = RatioTrace(us,vs,population,s['prefix'],args.alpha/K)
                lo,hi = tr.interval(iterations=24)
                s['reports'] += 1
                target = is_target(lo,hi,args.target)
                paths.append(dict(query=k,report=s['reports'],n=s['n'],physical=store.physical_rows,
                                  lower=lo,upper=hi,target=target,stage='ordinary'))
                elapsed = time.perf_counter()-t
                s['report_seconds'] += elapsed;report_seconds += elapsed
                census = s['n']+s['prefix'].n == N and population is not None
                if target or s['n'] >= args.cap or census:
                    done.append(dict(query=k,n=s['n'],contribution_evaluations=s['evals'],
                        lower=lo,upper=hi,target=target,censored=not target,reports=s['reports'],
                        registration_physical=h,prefix_rows=s['prefix'].n,
                        prefix_moments_json=json.dumps(vars(s['prefix']),sort_keys=True),
                        draw_positions_sha256='',pairs_sha256=''))
                    if scan is not None:
                        scan.stop(s['qid'])
                    finished.append(k)
            active = [k for k in active if k not in finished]
        # Complete policy and cache bookkeeping, including store teardown.
        # Reproduction hashes/serialization and offline truth are separate.
        stats = store.stats
        assert not stats['failed']
        for counter in (stats['setup'],stats['continuation'],stats['prefix_setup']):
            assert counter['read_accounting_complete']
            assert counter['read_exception_calls']==0 and counter['read_error_position_unknown']==0
    finally:
        store.close()
    continuation = time.perf_counter()-start
    saved = {}
    for k,s in enumerate(state):
        if s['positions']:
            saved[f'q{k}_positions'] = np.concatenate(s['positions'])
    # Fixed source bytes plus selected-function definitions replay Rebase's
    # prefix; the entire prefix is not duplicated in every trace archive.
    if saved:
        np.savez_compressed(trace_path,**saved)
    trace_digest = sha(trace_path) if saved else ''
    for row in done:
        row['trace_sha256'] = trace_digest
        s = state[row['query']]
        if s['positions']:
            row['draw_positions_sha256'] = array_sha(saved[f"q{row['query']}_positions"])
            row['pairs_sha256'] = hashlib.sha256(np.concatenate(s['us']).tobytes()+
                                                np.concatenate(s['vs']).tobytes()).hexdigest()
    result = dict(method=arm,queries=K,targets=sum(r['target'] for r in done),
        censored=sum(r['censored'] for r in done),continuation_seconds=continuation,
        init_seconds=init_seconds,sampling_and_cache_seconds=sample_seconds,
        function_seconds=function_seconds,prefix_prepare_seconds=prefix_seconds,
        report_seconds=report_seconds,contribution_evaluations=sum(r['contribution_evaluations'] for r in done),
        scalar_exp_evaluations=sum(r['contribution_evaluations'] for r in done)*len(specs[0]['centers']),
        peak_query_map_entries=peak_map,atomic_pool_proposals=proposal_count,
        actual_read_calls=stats['continuation']['read_calls'],
        actual_requested_bytes=stats['continuation']['read_requested_bytes'],
        actual_returned_bytes=stats['continuation']['read_returned_bytes'],
        actual_read_seconds=stats['continuation']['read_seconds'],
        initial_h=h,final_source_prefix=stats['physical_count'],
        visible_records=stats['seen_count'],retained_cache_payload_bytes=stats['retained_cache_payload_bytes'],
        trace_sha256=trace_digest,backend_stats_json=json.dumps(stats,sort_keys=True))
    assert len(done)==K and len({r['query'] for r in done})==K
    return result,done,paths


def truth_audit(data,all_specs,queries,paths,physical_sources,dataset):
    """Private deferred truth and compiled-prefix checks; no timed trajectory access."""
    lookup = {}
    started = time.perf_counter()
    max_reference_error = 0.
    max_prefix_ulps=0.
    prefix_cases=0
    rec = np.empty(len(data[0]),dtype=RECORD_DTYPE)
    rec['id'] = np.arange(len(rec),dtype=np.uint64)
    for name,column in zip(('p','r','q'),data):
        rec[name]=column
    for key,spec in all_specs.items():
        u,v=workload.evaluate(rec,spec,validate=False)
        a,b=math.fsum(u),math.fsum(v)
        assert b > 0
        theta=a/b
        # Guard for truth aggregation/division of evaluated float64 pairs.
        error = 8*(math.ulp(a)+abs(theta)*math.ulp(b))/b + 8*math.ulp(theta)
        lookup[key]=(theta,error)
        ids=np.linspace(0,len(rec)-1,min(129,len(rec))).astype(np.int64)
        for i in ids:
            xyz=[float(rec[i][name]) for name in ('p','r','q')]
            weight=math.fsum(math.exp(-math.fsum((xyz[d]-c[d])**2 for d in range(3)) /
                (2*spec['bandwidth']**2)) for c in spec['centers'])/len(spec['centers'])
            ru,rv=(xyz[0]*weight,weight) if spec['function']=='weighted_price' else (weight,1.)
            max_reference_error=max(max_reference_error,abs(float(u[i])-ru),abs(float(v[i])-rv))
    exact_lookup={}
    for row in [*queries,*paths]:
        key=(row['function'],row['fraction'],row['replicate'],row['query'])
        theta,error=lookup[key]
        row['truth'],row['truth_roundoff_bound']=theta,error
        row['covered']=bool(row['lower']<=theta<=row['upper'])
        contained=row['lower']<=theta-error and theta+error<=row['upper']
        disjoint=row['upper']<theta-error or theta+error<row['lower']
        row['truth_classification']='compensated envelope'
        if not (contained or disjoint):
            # Query census can produce an enclosure narrower than a generic
            # truth roundoff bound. Resolve the actual float-pair ratio exactly;
            # do not drop, relabel or guess ambiguous reports.
            if key not in exact_lookup:
                u,v=workload.evaluate(rec,all_specs[key],validate=False)
                a=sum((Fraction.from_float(float(x)) for x in u),Fraction())
                b=sum((Fraction.from_float(float(x)) for x in v),Fraction())
                assert b>0
                exact_lookup[key]=a/b
            truth=exact_lookup[key]
            lower=Fraction.from_float(float(row['lower']))
            upper=Fraction.from_float(float(row['upper'])) if math.isfinite(row['upper']) else None
            row['covered']=bool(lower<=truth and (upper is None or truth<=upper))
            row['truth_classification']='exact rational sum of float64 pairs'
    # Recompute every actually used Rebase prefix from its exact physical file.
    # This is deferred until ALL timed sessions for the population have ended.
    # Checking the actual prefix totals is distinct from a kernel unit test.
    for row in queries:
        if row['method'] != 'rebase':
            continue
        key=(row['function'],row['fraction'],row['replicate'],row['query'])
        source_path=physical_sources/f"{dataset}_{row['replicate']}.bin"
        source=np.memmap(source_path,mode='r',dtype=RECORD_DTYPE)
        h=int(row['prefix_rows'])
        u,v=workload.evaluate(source[:h],all_specs[key],validate=False)
        ref=[math.fsum(x) for x in (u,v,u*u,v*v,u*v)]
        stored=json.loads(row['prefix_moments_json'])
        assert stored['n']==h
        for name,value in zip(('su','sv','suu','svv','suv'),ref):
            ulps=abs(stored[name]-value)/math.ulp(value)
            max_prefix_ulps=max(max_prefix_ulps,ulps)
            assert ulps<=2,('compiled prefix differs from per-value fsum',key,name,ulps)
        prefix_cases+=1
        del source,u,v
    assert max_reference_error < 8e-16
    return dict(truth_functions=len(lookup),max_scalar_reference_error=max_reference_error,
                actual_prefix_cases=prefix_cases,max_compensated_prefix_ulps=max_prefix_ulps,
                exact_truth_refinements=[dict(function=k[0],fraction=k[1],replicate=k[2],query=k[3],
                    numerator=str(value.numerator),denominator=str(value.denominator))
                    for k,value in sorted(exact_lookup.items())],
                all_report_exclusions=sum(not r['covered'] for r in paths),
                final_query_exclusions=sum(not r['covered'] for r in queries),
                seconds=time.perf_counter()-started,
                scope='Full retained-population per-value compensated truth of evaluated float64 pairs; '
                      'scalar exp spot checks; not a formal transcendental/IEEE proof')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--data-root',type=Path,default=ROOT/'artifacts/exact_inputs_20261017/data')
    ap.add_argument('--datasets',nargs='+',default=['tpch','tpcds'])
    ap.add_argument('--fractions',nargs='+',type=float,default=[.1,.9])
    ap.add_argument('--functions',nargs='+',choices=['affinity','weighted_price'],default=['affinity','weighted_price'])
    ap.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    ap.add_argument('--reps',type=int,default=20)
    ap.add_argument('--seed',type=int,default=202610172000)
    ap.add_argument('--K',type=int,default=8)
    ap.add_argument('--centers',type=int,default=8)
    ap.add_argument('--block',type=int,default=4096)
    ap.add_argument('--cap',type=int,default=65536)
    ap.add_argument('--target',type=float,default=.10)
    ap.add_argument('--alpha',type=float,default=.05)
    ap.add_argument('--exact-chunk',type=int,default=65536)
    ap.add_argument('--synthetic-n',type=int,default=0)
    args=ap.parse_args()
    assert all(0<f<1 for f in args.fractions)
    assert args.reps>0 and args.K>0 and args.centers>0 and args.block>0 and args.cap>0
    assert 0<args.alpha<1 and args.target>0 and args.exact_chunk>0
    out=ROOT/args.out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out/'INCOMPLETE').write_text('Study is not complete until manifest.json.\n',encoding='utf8')
    for name in ('source','physical_sources','traces'):
        (out/name).mkdir()
    sources={name:sha(ROOT/name) for name in SOURCE_NAMES}
    for name in SOURCE_NAMES:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    protocol=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        contract='AllK functions frozen from common initial last4096 prefix before arms. '
                 'AllK active initially; predictable common batches, retire after own certified target/cap/census.',
        time='Complete per-arm continuation, common actual prefix preparation/function selection/source construction '
             'reported separately. Prefix setup reusable equally across arms. No simultaneous timing jobs.',
        io='Actual unbuffered FileIO reads with warm/uncontrolled OS page cache; no device I/O or cold-cache claim.',
        baselines='Grouped missing-page indexed sampling, forward WR/WoR, atomic WR/WoR, '
                  'compiled compensated-prefix Rebase, one-row-pass fused Exact across identical functions.',
        fixed_h_source='Headerless32byte(id uint64,p,r,q float64), one seeded uniform permutation per population/replicate.',
        caveat='Controlled initial-prefix-selected functions, not later adaptive function admissions; '
               'atomic cross-query dependence differs but all use same own-query CS/alpha and target. '
               'No product of query capitals or independent-query error claim.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
    start=time.perf_counter();workload.warmup()
    for cls in (SequentialPositionScan,AtomicPositionScan):
        for wr in (False,True):
            s=cls(16,seed=33,replacement=wr);s.expose(4,return_positions=False)
            qs=[s.register() for _ in range(2)]
            if cls is AtomicPositionScan:
                s.draw_epochs(qs,2)
            else:
                s.draw(qs[0],2)
            s.query_entries(qs[0])
    RatioTrace(np.full(8,.5),np.ones(8),None,alpha=args.alpha/args.K).interval(iterations=24)
    jit_seconds=time.perf_counter()-start
    sessions,queries,paths,preparations,spec_rows=[],[],[],[],[]
    loads,audits={},{}
    original_loader.DATA_ROOT=args.data_root
    for dataset in args.datasets:
        if args.synthetic_n:
            N=args.synthetic_n
            assert N>1
            rng=np.random.default_rng(831)
            data=tuple(rng.random(N) for _ in range(3))
            loads[dataset]=dict(population=N,synthetic=True,seed=831,
                                normalized_arrays_sha256=[array_sha(a) for a in data])
        else:
            data,loads[dataset]=original_loader.load(dataset)
            N=len(data[0])
        all_specs={}
        for rep in range(args.reps):
            seed=args.seed+rep
            file_path=out/'physical_sources'/f'{dataset}_{rep}.bin'
            prep=physical_file(file_path,data,seed)
            prep.update(dataset=dataset,replicate=rep)
            for fraction in args.fractions:
                h=int(N*fraction)
                assert h>0
                t=time.perf_counter();prefix=prepare_prefix(file_path,N,h)
                prefix_seconds=time.perf_counter()-t
                t=time.perf_counter()
                with ForwardFileStore(file_path,N,h,prefix=prefix) as store:
                    context=store.read_range(max(0,h-4096),h)
                context_seconds=time.perf_counter()-t
                for function in args.functions:
                    t=time.perf_counter();specs=workload.select_specs(context,args.K,args.centers,function)
                    selection_seconds=time.perf_counter()-t
                    preparations.append(dict(**prep,function=function,fraction=fraction,initial_h=h,
                        common_prefix_seconds=prefix_seconds,common_prefix_stats_json=json.dumps(prefix.stats,sort_keys=True),
                        common_context_seconds=context_seconds,common_selection_seconds=selection_seconds))
                    for spec in specs:
                        all_specs[(function,fraction,rep,spec['query'])]=spec
                        spec_rows.append(dict(dataset=dataset,function=function,fraction=fraction,
                            replicate=rep,query=spec['query'],spec_json=json.dumps(spec,sort_keys=True)))
                    for arm in np.roll(np.asarray(args.arms),(rep+int(fraction*10))%len(args.arms)):
                        trace_path=out/'traces'/f'{dataset}_{function}_{fraction}_{rep}_{arm}.npz'
                        s,qq,pp=run_session(file_path,N,h,prefix,specs,str(arm),seed,args,trace_path)
                        for row in [s,*qq,*pp]:
                            row.update(dataset=dataset,function=function,fraction=fraction,replicate=rep,source_seed=seed,method=str(arm))
                        sessions.append(s);queries.extend(qq);paths.extend(pp)
                    print(dataset,rep,fraction,function,'timed arms complete',flush=True)
                    write_csv(out/'sessions.csv',sessions);write_csv(out/'queries.csv',queries)
                    write_csv(out/'paths.csv',paths);write_csv(out/'preparation.csv',preparations)
                    write_csv(out/'specs.csv',spec_rows)
                del prefix,context
        dq=[r for r in queries if r['dataset']==dataset]
        dp=[r for r in paths if r['dataset']==dataset]
        audits[dataset]=truth_audit(data,all_specs,dq,dp,out/'physical_sources',dataset)
        write_csv(out/'queries.csv',queries);write_csv(out/'paths.csv',paths)
        del data
    assert all(sha(ROOT/name)==digest for name,digest in sources.items())
    meta=dict(status='COMPLETE',protocol_sha256=sha(out/'protocol.json'),
        source_sha256=sources,input=loads,audit=audits,jit_preparation_seconds=jit_seconds,
        sessions=len(sessions),queries=len(queries),all_reports=len(paths),
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        output_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
                       if p.is_file() and p.name!='INCOMPLETE'},
        note='Physical source files and all draw-position traces retained; checksums alone are not replication.')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    (out/'INCOMPLETE').unlink()
    print('COMPLETE',len(sessions),len(queries),len(paths),flush=True)


if __name__=='__main__':
    main()
