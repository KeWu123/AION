"""Actual-file own-history policies and matched-definition concurrency study.

Every timing arm runs serially. Truth and replay verification are deferred
until every timed arm for a population has finished. Existing studies are
never overwritten. See the frozen protocol in each new output directory.
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
import file_workload as workload
import file_streaming_diagnostics as diagnostics
from anq.file_store import RECORD_DTYPE, prepare_prefix, ForwardFileStore
from anq.position_scan import SequentialPositionScan, AtomicPositionScan
from anq.coordinated_scan import CoordinatedPositionScan

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('forward_wor','cohort_wor','atomic_wor','coordinated_wor','rebase','indexed_wor')


def history_spec(context, k, reports, *, function='weighted_price', centers=8):
    """Only released context and previously committed interval reports select a query."""
    spec = workload.select_specs(context, (k % 8)+1, centers, function)[k % 8]
    spec['query'] = k
    if reports:
        last = reports[-1]
        lo, hi = last['lower'], last['upper']
        midpoint = (lo+hi)/2 if math.isfinite(hi) else .5
        spec['bandwidth'] *= .85 + .3*float(np.clip(midpoint,0,1))
    return spec


def json_write(path, value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def warmup(out):
    t = time.perf_counter()
    workload.warmup(); diagnostics.warmup()
    for cls in (SequentialPositionScan,AtomicPositionScan,CoordinatedPositionScan):
        s=cls(32,seed=13,replacement=False); s.expose(8,return_positions=False)
        qs=[s.register(),s.register()]
        if isinstance(s,AtomicPositionScan): s.draw_epochs(qs,4)
        else:
            for q in qs:s.draw(q,4)
    data=tuple(np.random.default_rng(j).random(256) for j in range(3))
    path=out/'warmup.bin';base.physical_file(path,data,3)
    prefix=prepare_prefix(path,256,64)
    args=SimpleNamespace(block=32,cap=64,target=.2,alpha=.05,active_window=2,
                         precision_profile='heterogeneous')
    for arm in ARMS:
        driver.run_session(path,256,64,prefix,[None]*4,arm,9,args,out/(arm+'_warm.npz'),
                           selection_policy=history_spec)
    return time.perf_counter()-t


def audit(data, sessions, queries, reports, out):
    """Verify each selection from retained public history; audit every reported interval."""
    started=time.perf_counter(); N=len(data[0])
    rec=np.empty(N,dtype=RECORD_DTYPE);rec['id']=np.arange(N,dtype=np.uint64)
    for name,column in zip(('p','r','q'),data):rec[name]=column
    truth_cache={}; selected=changed=late=0; max_ref=0.; exact_count=0
    for session in sessions:
        identity=session['session_id']
        pp=[x for x in reports if x['session_id']==identity]
        qq=[x for x in queries if x['session_id']==identity]
        specs=json.loads(session['specs_json'])
        events=json.loads((out/session['events_file']).read_text(encoding='utf8'))
        traces=np.load(out/session['trace_file'])
        physical=np.memmap(out/session['source_file'],mode='r',dtype=RECORD_DTYPE)
        h=session['initial_h']; history=physical[max(0,h-4096):h].copy()
        offsets={k:0 for k in range(session['queries'])}
        for event in events:
            if event['kind']=='history_selection':
                assert event['context_ids']==history['id'].tolist()
                count=event['policy_report_count']
                assert 0<=count<=len(pp)
                assert all(x['epoch']<=event['epoch'] for x in pp[:count])
                expected=history_spec(history,event['query'],pp[:count])
                assert expected==event['spec']==specs[event['query']]
                selected+=1
                if event['query']>=session['active_window']:
                    late+=1
                    initial=history_spec(physical[max(0,h-4096):h],event['query'],[])
                    changed+=expected!=initial
            elif event['kind']=='epoch_joint_release':
                for k in event['participants']:
                    first=offsets[k]; last=first+event['count']
                    pos=traces[f'q{k}_positions'][first:last]
                    history=np.concatenate((history,physical[pos]))[-4096:].copy()
                    offsets[k]=last
        for row in [*qq,*pp]:
            spec=specs[row['query']]
            key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
            if key not in truth_cache:
                u,v=workload.evaluate(rec,spec,validate=False)
                su,sv=float(workload._neumaier(u)),float(workload._neumaier(v))
                assert sv>0
                theta=su/sv
                error=8*(math.ulp(su)+abs(theta)*math.ulp(sv))/sv+8*math.ulp(theta)
                for i in np.linspace(0,N-1,9).astype(int):
                    xyz=[float(rec[i][c]) for c in ('p','r','q')]
                    weight=math.fsum(math.exp(-math.fsum((xyz[d]-c[d])**2 for d in range(3))/(2*spec['bandwidth']**2)) for c in spec['centers'])/len(spec['centers'])
                    ru,rv=(xyz[0]*weight,weight) if spec['function']=='weighted_price' else (weight,1.)
                    max_ref=max(max_ref,abs(float(u[i])-ru),abs(float(v[i])-rv))
                truth_cache[key]=(theta,error)
            theta,error=truth_cache[key]
            row.update(truth=theta,truth_roundoff_bound=error,
                covered=bool(row['lower']<=theta<=row['upper']))
            contained=row['lower']<=theta-error and theta+error<=row['upper']
            disjoint=row['upper']<theta-error or theta+error<row['lower']
            if not (contained or disjoint):
                u,v=workload.evaluate(rec,spec,validate=False)
                exact=sum(map(Fraction.from_float,map(float,u)),Fraction())/sum(map(Fraction.from_float,map(float,v)),Fraction())
                row['covered']=bool(Fraction.from_float(row['lower'])<=exact and
                    (not math.isfinite(row['upper']) or exact<=Fraction.from_float(row['upper'])))
                exact_count+=1
        del physical
        if len(truth_cache)%40<8:print('AUDIT',identity,'unique truths',len(truth_cache),flush=True)
    assert max_ref<8e-16,max_ref
    return dict(unique_truth_functions=len(truth_cache),history_selections_verified=selected,
        later_selections=late,later_changed_from_initial_context=changed,
        max_scalar_kernel_error=max_ref,exact_truth_refinements=exact_count,
        all_report_exclusions=sum(not x['covered'] for x in reports),
        final_exclusions=sum(not x['covered'] for x in queries),
        seconds=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--reps',type=int,default=20);p.add_argument('--seed',type=int,default=20261102000)
    p.add_argument('--datasets',nargs='+',default=['tpch','tpcds'])
    p.add_argument('--synthetic-n',type=int,default=0)
    p.add_argument('--skip-sweep',action='store_true')
    args=p.parse_args();out=ROOT/args.out
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    for name in ('traces','physical_sources','source','warmup'):(out/name).mkdir()
    (out/'INCOMPLETE').write_text('Not complete until the final manifest.\n')
    source_names=sorted(set(driver.SOURCE_NAMES+('code/exp_history_file.py',)))
    hashes={n:base.sha(ROOT/n) for n in source_names}
    for n in source_names:
        dst=out/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    protocol=dict(arguments=vars(args),primary=dict(datasets=args.datasets,fractions=[.1,.9],
        function='weighted_price',K=8,active_window=4,arms=ARMS,reps=args.reps),
        sweep=dict(dataset='tpch',fraction=.9,function='weighted_price',K=16,
                   active_windows=[1,2,4,8],definitions='identical initial-context functions across all arms and windows'),
        quality=dict(alpha=.05,target=.10,precision_factors=[.5,1,1.5,2],cap=131072,block=4096),
        selection='At each admission use the most recent 4096 already released payloads in deterministic query-tag order and the last committed interval. First wave uses common initial prefix tail. Freeze each selected function. No truth is available.',
        comparison='Primary arms follow the same policy, not necessarily the same later estimands. Sweep matches functions, information, alpha, targets and caps.',
        preparation='Report source permutation, materialization and prefix-cache work once. Deployment reuse counts are explicit independent sessions; never divide by the number of competing benchmark arms.',
        timing='All arms serial, method order rotated by replicate and cell. Warm OS cache; bytes are actual FileIO API work. Full continuation includes selection, initialization, inference and close; excludes JIT, serialization, exact map replay, and deferred truth.',
        sources=hashes,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    json_write(out/'protocol.json',protocol)
    jit=warmup(out/'warmup')
    sessions=[];queries=[];reports=[];sources=[];prefixes=[];loads={};audits={}
    tables=dict(sessions=sessions,queries=queries,reports=reports,source_preparation=sources,prefix_preparation=prefixes)
    def save():
        for name,rows in tables.items():driver.write_csv(out/(name+'.csv'),rows)
    base.original_loader.DATA_ROOT=ROOT/'data'
    for dataset in args.datasets:
        if args.synthetic_n:
            data=tuple(np.random.default_rng(j+30).random(args.synthetic_n) for j in range(3))
            loads[dataset]=dict(population=args.synthetic_n,synthetic=True)
        else:data,loads[dataset]=base.original_loader.load(dataset)
        N=len(data[0])
        for rep in range(args.reps):
            seed=args.seed+rep;path=out/'physical_sources'/f'{dataset}_{rep}.bin'
            sources.append(dict(dataset=dataset,replicate=rep,**base.physical_file(path,data,seed)))
            for fraction in (.1,.9):
                h=int(N*fraction);t=time.perf_counter();prefix=prepare_prefix(path,N,h)
                prefixes.append(dict(dataset=dataset,replicate=rep,fraction=fraction,
                    cache_seconds=time.perf_counter()-t,cache_stats_json=json.dumps(prefix.stats)))
                with ForwardFileStore(path,N,h,prefix=prefix) as store:context=store.read_range(max(0,h-4096),h)
                cases=[('history',8,4)]
                if dataset=='tpch' and fraction==.9 and not args.skip_sweep:
                    cases += [('concurrency',16,w) for w in (1,2,4,8)]
                for study,K,window in cases:
                    cfg=SimpleNamespace(block=4096,cap=131072,target=.10,alpha=.05,
                        active_window=window,precision_profile='heterogeneous')
                    for arm in np.roll(np.array(ARMS),(rep+int(fraction*10)+window)%len(ARMS)):
                        arm=str(arm);identity=f'{dataset}_{study}_{fraction}_{rep}_w{window}_{arm}'
                        trace=out/'traces'/(identity+'.npz')
                        specs=([None]*K if study=='history' else workload.select_specs(context,K,8,'weighted_price'))
                        try:
                            result,qq,pp=driver.run_session(path,N,h,prefix,specs,arm,seed,cfg,trace,
                                selection_policy=history_spec if study=='history' else None)
                        except BaseException:save();raise
                        common=dict(dataset=dataset,study=study,function='weighted_price',fraction=fraction,
                            replicate=rep,source_seed=seed,session_id=identity,method=arm,active_window=window)
                        for row in [result,*qq,*pp]:row.update(common)
                        result.update(specs_json=json.dumps(specs,sort_keys=True),
                            source_file=path.relative_to(out).as_posix(),trace_file=trace.relative_to(out).as_posix(),
                            events_file=trace.with_suffix('.events.json').relative_to(out).as_posix())
                        sessions.append(result);queries.extend(qq);reports.extend(pp);save()
                    print('TIMED',dataset,rep,fraction,study,window,'sessions',len(sessions),flush=True)
                del prefix
        local_s=[x for x in sessions if x['dataset']==dataset]
        audits[dataset]=audit(data,local_s,[x for x in queries if x['dataset']==dataset],
                              [x for x in reports if x['dataset']==dataset],out)
        save();json_write(out/'audit.json',audits);del data
    assert hashes=={n:base.sha(ROOT/n) for n in source_names}
    manifest=dict(status='COMPLETE',sessions=len(sessions),queries=len(queries),reports=len(reports),
        input=loads,audit=audits,jit_seconds=jit,source_sha256=hashes,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        files_sha256={p.relative_to(out).as_posix():base.sha(p) for p in out.rglob('*')
            if p.is_file() and p.name not in ('INCOMPLETE','manifest.json')})
    json_write(out/'manifest.json',manifest);(out/'INCOMPLETE').unlink()
    print('COMPLETE',len(sessions),len(queries),len(reports),flush=True)


if __name__=='__main__':main()
