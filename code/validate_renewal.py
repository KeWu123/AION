"""Independent explicit-set checks for the sparse interleaved access primitive."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from anq.renewal import RenewalScan,InterleavedRenewalScan,_GrowingComplement,_take_rank

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reviews/innovation_20261016/interleaved_code_checks'

def inspect_tree(node):
    if node is None:return [],0
    left,lm=inspect_tree(node.left);right,rm=inspect_tree(node.right)
    assert node.lo<node.hi
    if node.left:assert node.priority<=node.left.priority
    if node.right:assert node.priority<=node.right.priority
    values=left+list(range(node.lo,node.hi))+right
    assert len(values)==len(set(values)) and values==sorted(values)
    assert node.mass==lm+node.hi-node.lo+rm
    return values,node.mass

def main():
    rng=np.random.default_rng(202610167);pool_ops=0;scheduled=0
    for rep in range(200):
        H=int(rng.integers(0,50));p=_GrowingComplement(H,rng);ref=list(range(H))
        for step in range(200):
            if not ref or rng.random()<.35:
                grow=int(rng.integers(0,15));p.sync(H+grow);ref+=list(range(H,H+grow));H+=grow
            else:
                rank=int(rng.integers(len(ref)));value,p.root=_take_rank(p.root,rank,rng)
                assert value==ref.pop(rank)
            values,mass=inspect_tree(p.root)
            assert values==ref and mass==len(ref);pool_ops+=1
    for N in range(1,65):
        for rep in range(8):
            order=rng.permutation(N);s=InterleavedRenewalScan(order,int(rng.integers(2**31)))
            s.expose(int(rng.integers(N+1)));qs=[s.register() for _ in range(4)]
            used={q:set() for q in qs}
            for step in range(4*N):
                eligible=[q for q in qs if len(used[q])<N]
                if not eligible:break
                q=int(rng.choice(eligible));k=min(int(rng.integers(1,8)),N-len(used[q]))
                ids,new=s.draw(q,k)
                assert len(ids)==k and len(set(ids))==k and not set(ids)&used[q]
                used[q].update(ids)
                assert used[q]<=set(s.exposed)
                if s.physical_rows<N and rng.random()<.1:s.expose(1)
                # No-op reports/draws must not change or duplicate state.
                before=s._queries[q]['pool'].seen;a,b=s.draw(q,0)
                assert len(a)==len(b)==0 and s._queries[q]['pool'].seen==before
                scheduled+=1
            assert all(len(x)==N for x in used.values()) and s.physical_rows==N
            for q in qs:s.stop(q)
    dtype_cases=0
    for dtype,labels in [(np.int8,[0,1,127]),(np.int64,[0,1,2**63-1]),
                         (np.uint64,[0,2**63,2**64-1])]:
        order=np.asarray(labels,dtype=dtype)
        for cls in (RenewalScan,InterleavedRenewalScan):
            s=cls(order);s.expose(1)
            if cls is RenewalScan:
                s.register();zero,_=s.draw(0);ids,_=s.draw(3);s.stop()
            else:
                q=s.register();zero,_=s.draw(q,0);ids,_=s.draw(q,3);s.stop(q)
            assert ids.dtype==zero.dtype==s.exposed.dtype==order.dtype
            assert set(ids.tolist())==set(labels) and s.physical_rows==3
            dtype_cases+=1
    result=dict(explicit_set_operations=pool_ops,interleaved_draw_events=scheduled,
                integer_identity_boundary_cases=dtype_cases,
                four_query_configurations=512,interval_order_mass_heap='PASS',
                unique_query_samples_source_boundaries_and_empty_draws='PASS',
                source_sha256=hashlib.sha256((ROOT/'code/anq/renewal.py').read_bytes()).hexdigest())
    OUT.mkdir(exist_ok=True);(OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
