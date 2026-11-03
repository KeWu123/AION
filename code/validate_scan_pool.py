"""Cross-check compiled sparse swap-delete against a materialized array."""
from pathlib import Path
import copy,hashlib,itertools,json,sys
import numpy as np
from numba import types
from numba.typed import Dict
from anq.scan_pool import SparsePoolScan,_delete,indexed_batch,warmup
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reviews/innovation_20261016/scan_pool_checks'

def maps_match(fwd,inv,arr):
    assert [fwd[i] if i in fwd else i for i in range(len(arr))]==arr
    for i,x in enumerate(arr):assert (inv[x] if x in inv else x)==i
    assert all(0<=k<len(arr) and k!=v for k,v in fwd.items())
    assert all(v<len(arr) and k!=v for k,v in inv.items())

def main():
    warmup();deletions=0;events=0;dtype_cases=0
    for N in range(1,8):
        for order in itertools.permutations(range(N)):
            f=Dict.empty(types.int64,types.int64);iv=Dict.empty(types.int64,types.int64);arr=list(range(N))
            for x in order:
                _delete(f,iv,len(arr),x)
                at=arr.index(x);arr[at]=arr[-1];arr.pop();maps_match(f,iv,arr);deletions+=1
    rng=np.random.default_rng(91063)
    for N in [1,2,3,7,16,63,512,10000]:
        for rep in range(12):
            order=rng.permutation(N);scan=SparsePoolScan(order,rep+100)
            scan.expose(int(rng.integers(0,N+1)));qs=[scan.register() for _ in range(4)]
            arrays={q:list(range(N)) for q in qs};refH=scan.physical_rows
            for step in range(80):
                active=[q for q in qs if arrays[q]]
                if not active:break
                q=int(rng.choice(active));n=min(int(rng.integers(0,20)),len(arrays[q]))
                state=scan._queries[q];gen=np.random.default_rng();gen.bit_generator.state=copy.deepcopy(state['rng'].bit_generator.state)
                expected=[];fresh=[];arr=arrays[q]
                for _ in range(n):
                    selected=arr[int(gen.integers(0,len(arr)))];new=selected>=refH
                    x=refH if new else selected;refH+=int(new)
                    expected.append(int(order[x]));fresh.append(new)
                    at=arr.index(x);arr[at]=arr[-1];arr.pop()
                actual,origin=scan.draw(q,n)
                assert actual.tolist()==expected and origin.tolist()==fresh and scan.physical_rows==refH
                maps_match(state['forward'],state['inverse'],arr)
                assert scan.query_entries(q)<=2*(N-len(arr))
                if refH<N and rng.random()<.3:
                    gap=min(int(rng.integers(0,1000)),N-refH);scan.expose(gap);refH+=gap
                events+=1
    for dtype,labels in [(np.int8,[0,1,127]),(np.int64,[0,1,2**63-1]),(np.uint64,[0,2**63,2**64-1])]:
        for h in range(4):
            order=np.asarray(labels,dtype=dtype);s=SparsePoolScan(order);s.expose(h);q=s.register()
            before=s._queries[q]['remaining'];zero,_=s.draw(q,0)
            assert before==s._queries[q]['remaining'] and zero.dtype==order.dtype
            ids,_=s.draw(q,3);assert set(ids.tolist())==set(labels) and ids.dtype==order.dtype
            try:s.draw(q,1);raise AssertionError('overflow accepted')
            except ValueError:pass
            s.stop(q);dtype_cases+=1
    for N in range(1,65):
        f=Dict.empty(types.int64,types.int64);out=np.empty(N,dtype=np.int64)
        assert indexed_batch(np.arange(N,dtype=np.int64),N,f,np.random.default_rng(N),out)==0
        assert sorted(out.tolist())==list(range(N))
    result=dict(arbitrary_swap_deletions=deletions,reference_interleaved_events=events,
        dtype_and_census_cases=dtype_cases,indexed_census_cases=64,
        production_sha256=hashlib.sha256((ROOT/'code/anq/scan_pool.py').read_bytes()).hexdigest(),
        checked_compiled=True,status='PASS')
    OUT.mkdir(exist_ok=True);(OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
