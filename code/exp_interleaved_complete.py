"""Complete adaptive eight-query sessions, common target and cost accounting.

Loaded-array engine emulation; new identities are NOT measured disk reads.
Each arm runs its own history-dependent policy from the same initial context.
All contribution evaluation, selection, sampling and reports are timed. Input
loading, uniform-order construction and JIT are separate costs. Renewal source
validation/copy is charged in its session initialization.
"""
from pathlib import Path
import argparse,csv,hashlib,json,platform,sys,time
import numpy as np
from numba import types
from numba.typed import Dict
from anq.scan_pool import SparsePoolScan,indexed_batch,warmup
from anq.forward_wr import ForwardReplacementScan,warmup as warmup_wr
from anq.ratio import Prefix,RatioTrace
from anq.joint_family import JointFamilySummary

ROOT=Path(__file__).resolve().parents[1]
DATA_ROOT=ROOT/'data'
K=8;WINDOW=4;BLOCK=4096;CAP=65536;ALPHA=.05;TARGET=.10;ITERS=24
ARMS=['renewal','forward_wr','prefix_catchup','indexed_wor','fresh_wr','exact','joint_threshold']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load(name):
    t=time.perf_counter()
    if name=='tpch':
        z=np.load(DATA_ROOT/'lo_data.npz')
        price=z['l_extendedprice'].astype(float);d=z['l_discount'].astype(float)
        qty=z['l_quantity'].astype(float);revenue=price*(1-d)
        paths=['data/lo_data.npz']
    else:
        sys.path.insert(0,str(ROOT/'.runtime'));import duckdb
        c=duckdb.connect(str(DATA_ROOT/'tpcds_sf1.duckdb'),read_only=True)
        a=c.execute('SELECT ss_ext_sales_price::DOUBLE p, ss_net_paid::DOUBLE r, ss_quantity::DOUBLE q FROM ss WHERE ss_ext_sales_price IS NOT NULL AND ss_net_paid IS NOT NULL AND ss_quantity IS NOT NULL ORDER BY rowid').fetchnumpy();c.close()
        price=a['p'];revenue=a['r'];qty=a['q'];paths=['data/tpcds_sf1.duckdb']
    # Public domain bounds are extracted once and disclosed as preprocessing.
    for a in (price,revenue,qty):
        assert not np.ma.isMaskedArray(a), 'NULL rows must be resolved by declared population SQL'
        assert np.all(np.isfinite(a)) and a.min()>=0
    scales=[float(price.max()),float(revenue.max()),float(qty.max())]
    data=(price/scales[0],revenue/scales[1],qty/scales[2])
    hash_begin=time.perf_counter()
    file_hashes={p:sha(DATA_ROOT/Path(p).name) for p in paths}
    array_hashes=[hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest() for a in data]
    hash_seconds=time.perf_counter()-hash_begin
    return data,dict(input_seconds=time.perf_counter()-t,population=len(price),
        normalizers=scales,array_payload_bytes=sum(x.nbytes for x in data),
        file_sha256=file_hashes,normalized_arrays_sha256=array_hashes,
        hash_seconds_included=hash_seconds,population_order='TPC-DS ORDER BY rowid; TPC-H retained archive array order')


def select_query(data,context,k):
    # Only the latest 4096 visible identities select thresholds. No truth access.
    ids=context[-4096:];p,r,q=(a[ids] for a in data);kind=k%4
    key=r if kind==0 else (q if kind==2 else p)
    if kind in (0,2):low=float(np.quantile(key,.75));high=1.
    elif kind==1:low=float(np.quantile(key,.50));high=1.
    else:low,high=map(float,np.quantile(key,[.25,.75]))
    return dict(kind=kind,low=low,high=high)


def evaluate(data,ids,spec):
    p,r,q=(a[ids] for a in data);kind=spec['kind']
    key=r if kind==0 else (q if kind==2 else p)
    selected=(key>=spec['low']) & (key<=spec['high'])
    if kind in (0,2):return r*selected,r
    return q*selected,selected.astype(float)


class TruthIndex:
    """Private offline audit; never given to selection/sampling/stopping policy."""
    def __init__(self,data):
        p,r,q=data;self.tables={};self.revenue=float(np.sum(r))
        for kind,key,weight in [(0,r,r),(1,p,q),(2,q,r)]:
            idx=np.argsort(key,kind='stable')
            self.tables[kind]=(key[idx],np.r_[0.,np.cumsum(weight[idx])])
    def query(self,spec):
        kind=1 if spec['kind']==3 else spec['kind'];key,totals=self.tables[kind]
        left=np.searchsorted(key,spec['low'],side='left')
        right=np.searchsorted(key,spec['high'],side='right')
        numerator=totals[right]-totals[left]
        denominator=self.revenue if kind in (0,2) else right-left
        return float(numerator/denominator)


def audit(data,queries,sessions):
    """Deferred truth eliminates inline full scans from all timed sessions."""
    begin=time.perf_counter();index=TruthIndex(data);build=time.perf_counter()-begin
    # Independent vector evaluation cross-check, including ties and endpoints.
    rng=np.random.default_rng(714863);maximum=0.;cases=0
    candidates=[queries[0],queries[-1]]
    for kind in range(4):
        for _ in range(6):
            a,b=sorted(rng.random(2));candidates.append(dict(kind=kind,low=a,high=b))
    for spec in candidates:
        u,v=evaluate(data,slice(None),spec)
        if v.sum()==0:continue
        ref=float(np.sum(u)/np.sum(v));error=abs(index.query(spec)-ref)
        assert error<1e-9,(spec,error)
        maximum=max(maximum,error);cases+=1
    for row in queries:
        row['truth']=index.query(row)
        row['covered']=bool(row['lower']<=row['truth']<=row['upper'])
    for s in sessions:
        selected=[q for q in queries if q['method']==s['method'] and q['replicate']==s['replicate'] and q['fraction']==s['fraction']]
        assert len(selected)==K
        s['covered_queries']=sum(q['covered'] for q in selected)
        s['any_miss']=any(not q['covered'] for q in selected)
    return dict(index_build_seconds=build,total_audit_seconds=time.perf_counter()-begin,
                independent_full_vector_cases=cases,max_truth_discrepancy=maximum,
                role='offline only, unavailable to policy; no inline audit cache effects')


def run_session(data,order,inverse_order,fraction,rep,arm,seed,physical_data):
    N=len(order);h0=int(N*fraction);physical=h0;context=order[max(0,h0-4096):h0].copy()
    # Query-local sampling IDs and cache membership are distinct from row values.
    start=time.perf_counter();scan=None;summary=None
    visible=np.zeros(N,dtype=bool);visible[order[:h0]]=True
    if arm in ('renewal','forward_wr'):
        cls=SparsePoolScan if arm=='renewal' else ForwardReplacementScan
        scan=cls(order,seed+700);scan.expose(h0)
    elif arm=='joint_threshold':summary=JointFamilySummary(data,order[:h0],N,alpha=ALPHA)
    common_init=time.perf_counter()-start
    active={};done=[];next_k=0;paths=[];peak_entries=0;steps=0
    # Exact address footprint of newly accessed payload records; no timing or
    # physical disk-fault claim. Prefix pages are already logically cached.
    page_bytes=4096;record_bytes=24;first_uncached_page=(h0*record_bytes)//page_bytes
    touched=np.zeros((N*record_bytes+page_bytes-1)//page_bytes,dtype=bool)

    def register():
        nonlocal next_k
        begin=time.perf_counter();k=next_k;next_k+=1;spec=select_query(data,context,k)
        state=dict(k=k,spec=spec,reg_h=physical,evals=0,n=0,reports=0,
                   sample_seconds=0.,function_seconds=0.,prefix_prepare_seconds=0.,summary_exact_fallback=False,report_seconds=0.,selection_seconds=time.perf_counter()-begin,
                   us=[],vs=[],rng=np.random.default_rng(seed+10000+13*k),remaining=N)
        t=time.perf_counter()
        if arm in ('renewal','forward_wr'):state['qid']=scan.register()
        if arm=='indexed_wor':state['map']=Dict.empty(types.int64,types.int64)
        if arm=='prefix_catchup':
            u,v=evaluate(physical_data,slice(0,physical),spec);state['evals']+=physical
            prep_begin=time.perf_counter();state['prefix']=Prefix.from_rows(u,v)
            state['prefix_prepare_seconds']+=time.perf_counter()-prep_begin;state['seen']=physical
        state['function_seconds']+=time.perf_counter()-t
        active[k]=state

    while next_k<min(K,WINDOW):register()
    while active:
        # Round robin is fixed from currently active IDs; completion/admissions
        # depend only on reported intervals, never audit truth.
        for k in sorted(list(active)):
            s=active[k];step_start=time.perf_counter();t=time.perf_counter()
            ids=np.empty(0,dtype=np.int64)
            if arm in ('renewal','forward_wr'):
                n=min(BLOCK,CAP-s['n'],N-s['n']) if arm=='renewal' else min(BLOCK,CAP-s['n'])
                ids,new=scan.draw(s['qid'],n)
                oldH=physical;physical=scan.physical_rows
                if physical>oldH:context=order[max(0,physical-4096):physical].copy()
                peak_entries=max(peak_entries,scan.query_entries(s['qid']))
            elif arm=='indexed_wor':
                n=min(BLOCK,CAP-s['n'],s['remaining']);ids=np.empty(n,dtype=order.dtype)
                s['remaining']=indexed_batch(order,s['remaining'],s['map'],s['rng'],ids)
                peak_entries=max(peak_entries,len(s['map']))
                fresh=ids[~visible[ids]];visible[ids]=True
                positions=inverse_order[fresh]
                touched[(positions*record_bytes)//page_bytes]=True
                touched[((positions+1)*record_bytes-1)//page_bytes]=True
                if len(fresh):context=np.r_[context,fresh][-4096:]
            elif arm=='fresh_wr':
                n=min(BLOCK,CAP-s['n']);ids=s['rng'].integers(0,N,size=n)
                fresh=ids[~visible[ids]];visible[ids]=True
                positions=inverse_order[fresh]
                touched[(positions*record_bytes)//page_bytes]=True
                touched[((positions+1)*record_bytes-1)//page_bytes]=True
                if len(fresh):
                    first=np.unique(fresh,return_index=True)[1]
                    context=np.r_[context,fresh[np.sort(first)]][-4096:]
            elif arm=='prefix_catchup':
                # Every intervening globally exposed row must be evaluated.
                # Reuse another query's already-read batch before reading more.
                if physical==s['seen'] and physical<N and s['n']<CAP:
                    physical+=min(BLOCK,N-physical,CAP-s['n'])
                    context=order[max(0,physical-4096):physical].copy()
                end=min(physical,s['reg_h']+CAP)
                ids=slice(s['seen'],end);s['seen']=end
            elif arm=='exact':
                ids=slice(None);physical=N;visible[:]=True
                context=order[-4096:].copy()
            s['sample_seconds']+=time.perf_counter()-t
            t=time.perf_counter()
            if arm not in ('joint_threshold',):
                u,v=evaluate(physical_data if arm=='prefix_catchup' else data,ids,s['spec'])
                count=N if arm=='exact' else (ids.stop-ids.start if isinstance(ids,slice) else len(ids))
                s['evals']+=count;s['n']+=count
                if arm!='exact':s['us'].append(u);s['vs'].append(v)
            s['function_seconds']+=time.perf_counter()-t
            t=time.perf_counter();s['reports']+=1
            if arm=='exact':
                su,sv=float(np.sum(u)),float(np.sum(v));theta=su/sv
                gamma=N*np.finfo(float).eps/(1-N*np.finfo(float).eps)
                lo=max(0.,np.nextafter(theta*(1-gamma)/(1+gamma),-np.inf))
                hi=np.nextafter(theta*(1+gamma)/(1-gamma),np.inf)
            elif arm=='joint_threshold':
                lo,hi=summary.interval(s['spec'])
                if not (lo>0 and (hi-lo)/lo<=TARGET):
                    # Complete the family strategy with a deterministic exact
                    # fallback. No extra stochastic level is spent. The fixed
                    # initial-family event still covers later selected queries.
                    s['report_seconds']+=time.perf_counter()-t
                    paths.append(dict(method=arm,replicate=rep,fraction=fraction,query=k,
                        n=s['n'],physical=physical,lower=lo,upper=hi,target=False,
                        report=s['reports'],stage='summary_trial'))
                    s['reports']+=1
                    t=time.perf_counter();u,v=evaluate(data,slice(None),s['spec'])
                    s['function_seconds']+=time.perf_counter()-t
                    s['evals']+=N;s['n']+=N;s['summary_exact_fallback']=True
                    physical=N;visible[:]=True;context=order[-4096:].copy()
                    t=time.perf_counter();su,sv=float(np.sum(u)),float(np.sum(v));theta=su/sv
                    gamma=N*np.finfo(float).eps/(1-N*np.finfo(float).eps)
                    lo=max(0.,np.nextafter(theta*(1-gamma)/(1+gamma),-np.inf))
                    hi=np.nextafter(theta*(1+gamma)/(1-gamma),np.inf)
            else:
                tr=RatioTrace(np.concatenate(s['us']),np.concatenate(s['vs']),
                    None if arm in ('fresh_wr','forward_wr') else N,
                    s.get('prefix',Prefix()),ALPHA/K)
                lo,hi=tr.interval(iterations=ITERS)
            s['report_seconds']+=time.perf_counter()-t
            target=bool(lo>0 and (hi-lo)/lo<=TARGET)
            paths.append(dict(method=arm,replicate=rep,fraction=fraction,query=k,n=s['n'],physical=physical,
                              lower=lo,upper=hi,target=target,report=s['reports'],
                              stage=('exact_fallback' if s['summary_exact_fallback'] else 'summary') if arm=='joint_threshold' else 'ordinary'))
            finish=target or s['n']>=CAP or arm in ('exact','joint_threshold') or (arm=='prefix_catchup' and physical==N) or (arm in ('renewal','indexed_wor') and s['n']==N)
            if finish:
                if arm in ('renewal','forward_wr'):scan.stop(s['qid'])
                row=dict(method=arm,replicate=rep,fraction=fraction,query=k,seed=seed,
                    **s['spec'],registration_physical=s['reg_h'],query_draws=s['n'],
                    contribution_evaluations=s['evals'],lower=lo,upper=hi,truth=None,
                    covered=None,target=target,censored=not target,
                    sampling_seconds=s['sample_seconds'],function_seconds=s['function_seconds'],
                    prefix_prepare_seconds=s['prefix_prepare_seconds'],summary_exact_fallback=s['summary_exact_fallback'],
                    report_seconds=s['report_seconds'],selection_seconds=s['selection_seconds'],
                    reports=s['reports'],audit_seconds=0.)
                done.append(row);del active[k]
                if next_k<K:register()
            steps+=1
    query_total=sum(r['sampling_seconds']+r['function_seconds']+r['report_seconds']+r['selection_seconds'] for r in done)
    wall_without_audit=time.perf_counter()-start
    if arm not in ('fresh_wr','indexed_wor'):
        end=N if arm=='exact' else physical
        if end>h0:touched[first_uncached_page:((end*record_bytes-1)//page_bytes)+1]=True
    result=dict(method=arm,replicate=rep,fraction=fraction,seed=seed,
        targets=sum(r['target'] for r in done),covered_queries=None,queries=K,
        any_miss=None,censored=sum(r['censored'] for r in done),
        init_seconds=common_init,postselection_seconds=wall_without_audit,
        query_component_seconds=query_total,
        prefix_prepare_seconds=sum(r['prefix_prepare_seconds'] for r in done),
        summary_exact_fallbacks=sum(r['summary_exact_fallback'] for r in done),
        retained_summary_array_bytes=summary.retained_array_bytes if summary else 0,
        contribution_evaluations=sum(r['contribution_evaluations'] for r in done)+(summary.rows if summary else 0),
        initial_physical=h0,new_identities=(int(visible.sum())-h0 if arm in ('fresh_wr','indexed_wor') else (N-h0 if arm=='exact' else physical-h0)),
        logical_cache_payload_bytes=(physical*24 if arm in ('renewal','forward_wr','prefix_catchup','joint_threshold') else int(visible.sum())*24),
        peak_query_map_entries=peak_entries,steps=steps)
    result.update(new_payload_4k_pages=int(touched.sum()),payload_record_bytes=record_bytes,
                  page_metric='logical address footprint; already cached full prefix pages excluded except boundary')
    return result,done,paths


def write_csv(path,rows):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    global DATA_ROOT
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=20)
    ap.add_argument('--seed',type=int,default=2026101800)
    ap.add_argument('--out',default='experiments/innovation_20261016/interleaved_complete')
    ap.add_argument('--data-root',type=Path,default=DATA_ROOT)
    ap.add_argument('--datasets',nargs='+',default=['tpch','tpcds']);ap.add_argument('--fractions',nargs='+',type=float,default=[.1,.5,.9])
    args=ap.parse_args();out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Refusing to overwrite completed run')
    DATA_ROOT=args.data_root
    sources=[Path(__file__),ROOT/'code/anq/scan_pool.py',ROOT/'code/anq/ratio.py',ROOT/'code/anq/forward_wr.py',ROOT/'code/anq/joint_family.py']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    for p in sources:
        dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    begin=time.perf_counter();warmup();warmup_wr()
    # All int64 production signatures are warmed and charged before timing.
    s=SparsePoolScan(np.arange(4,dtype=np.int64));s.expose(2);q=s.register();s.draw(q,4)
    s.query_entries(q)
    d=Dict.empty(types.int64,types.int64);indexed_batch(np.arange(4,dtype=np.int64),4,d,np.random.default_rng(1),np.empty(4,dtype=np.int64))
    jit=time.perf_counter()-begin
    prep=[];sessions=[];queries=[];paths=[];loads={};audits={}
    for dataset in args.datasets:
        data,loads[dataset]=load(dataset);N=len(data[0])
        for rep in range(args.reps):
            seed=args.seed+rep;start=time.perf_counter();order=np.random.default_rng(seed).permutation(N).astype(np.int64)
            prep.append(dict(dataset=dataset,replicate=rep,seed=seed,permutation_seconds=time.perf_counter()-start,permutation_bytes=order.nbytes))
            t=time.perf_counter();inverse_order=np.empty(N,dtype=np.int64);inverse_order[order]=np.arange(N,dtype=np.int64)
            prep[-1].update(instrumentation_inverse_seconds=time.perf_counter()-t,instrumentation_inverse_bytes=inverse_order.nbytes)
            t=time.perf_counter();physical_data=tuple(a[order] for a in data)
            prep[-1].update(shuffled_payload_preparation_seconds=time.perf_counter()-t,
                shuffled_payload_bytes=sum(a.nbytes for a in physical_data))
            for fraction in args.fractions:
                for arm in np.roll(np.array(ARMS),(rep+int(fraction*10))%len(ARMS)):
                    a,b,c=run_session(data,order,inverse_order,fraction,rep,str(arm),seed,physical_data)
                    for r in [a,*b,*c]:r['dataset']=dataset
                    sessions.append(a);queries.extend(b);paths.extend(c)
                print(dataset,'rep',rep,'fraction',fraction,'elapsed',round(time.perf_counter()-begin,1),flush=True)
                write_csv(out/'sessions.csv',sessions);write_csv(out/'queries.csv',queries);write_csv(out/'paths.csv',paths)
        audits[dataset]=audit(data,[r for r in queries if r['dataset']==dataset],
                             [r for r in sessions if r['dataset']==dataset])
        write_csv(out/'sessions.csv',sessions);write_csv(out/'queries.csv',queries)
    write_csv(out/'preparation.csv',prep)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    record=dict(protocol=dict(reps=args.reps,queries=K,active_window=WINDOW,block=BLOCK,cap=CAP,
        alpha=ALPHA,query_alpha=ALPHA/K,target='L>0 and (U-L)/L <= .10',iterations=ITERS,
        arms=ARMS,fractions=args.fractions,policy='latest4096 visible identities; thresholds .75/.5/[.25,.75]; round robin; admit on completion',
        joint='fixed-H simultaneous HS/EBS-Serfling intersection (half alpha each;5delta empirical constant), two first/second-moment sorted summaries and quantity bins; exact deterministic per-query fallback when target not met; full source exposure/context update; complete strategy',
        scope='loaded-array logical engine; no disk I/O claim; arbitrary indexed access allowed to oracle arms',
        time='all selection, evaluations, sampling/maps/cache membership, concatenation, CS preparation/inversion, registration and method initialization; prefix aggregation isolated as included subtotal; offline truth/input/order/shuffled-payload/JIT separate; Rebase prefix and catch-up use contiguous physical payload slices; Exact contiguous original arrays',
        caveat='each method follows own adaptive trajectory; shared initial context, not identical later selected queries'),
        input=loads,audit=audits,jit_preparation_seconds=jit,source_sha256=hashes,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        output_sha256={p.name:sha(p) for p in out.glob('*.csv')},wall_seconds=time.perf_counter()-begin)
    (out/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print('complete',len(sessions),'sessions',len(queries),'queries',flush=True)

if __name__=='__main__':main()
