"""Independent scalar functions and compensated-moment checks, no timing study."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from anq.file_store import RECORD_DTYPE
import file_workload as w

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reviews/innovation_20261017/file_workload_validation.json'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args()
    out=args.out
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    rng = np.random.default_rng(173881)
    values = 0
    max_error = 0.
    for n in (1,2,7,128,4097):
        rec = np.zeros(n, dtype=RECORD_DTYPE)
        for name in ('p','r','q'):
            rec[name] = rng.random(n)
        for function in ('affinity','weighted_price'):
            for centers in (1,8,17):
                specs = w.select_specs(rec, K=3, centers=centers, function=function)
                su, sv = w.fused_first_moments(rec['p'],rec['r'],rec['q'],
                    np.asarray([s['centers'] for s in specs]),
                    np.asarray([1/(2*s['bandwidth']**2) for s in specs]),
                    function == 'weighted_price')
                for k,spec in enumerate(specs):
                    u,v = w.evaluate(rec,spec)
                    assert np.all(0 <= u) and np.all(u <= v) and np.all(v <= 1)
                    refs = []
                    for row in rec:
                        weight = math.fsum(math.exp(-math.fsum((float(row[name])-c[d])**2
                            for d,name in enumerate(('p','r','q')))/(2*spec['bandwidth']**2))
                            for c in spec['centers']) / len(spec['centers'])
                        refs.append((float(row['p'])*weight, weight) if function=='weighted_price'
                                    else (weight,1.))
                    ref = np.asarray(refs)
                    error = max(float(np.max(abs(u-ref[:,0]))), float(np.max(abs(v-ref[:,1]))))
                    max_error = max(max_error,error)
                    assert error < 6e-16, (n,function,centers,error)
                    moments = w.prefix_from_pairs(u,v)
                    expected = [math.fsum(x) for x in (u,v,u*u,v*v,u*v)]
                    actual = [moments.su,moments.sv,moments.suu,moments.svv,moments.suv]
                    assert all(abs(a-b) <= 2*math.ulp(b) for a,b in zip(actual,expected))
                    assert abs(su[k]-expected[0]) <= 2*math.ulp(expected[0])
                    assert abs(sv[k]-expected[1]) <= 2*math.ulp(expected[1])
                    values += n
    moment_cases = 0
    for n in (0,1,2,31,4096,1000000):
        for mode in range(4):
            if mode == 0:
                u,v = rng.random(n),rng.random(n)
            elif mode == 1:
                u = np.where(np.arange(n)%2,1.,np.nextafter(0.,1.))
                v = np.full(n,np.finfo(float).tiny)
            elif mode == 2:
                u = np.exp2(-rng.integers(0,1075,size=n).astype(float))
                v = u[::-1].copy()
            else:
                u,v = np.full(n,.1),np.ones(n)
            got = w.prefix_from_pairs(u,v)
            expected = [math.fsum(x) for x in (u,v,u*u,v*v,u*v)]
            assert all(abs(a-b) <= 2*math.ulp(b) for a,b in
                       zip((got.su,got.sv,got.suu,got.svv,got.suv),expected))
            moment_cases += 1
    invalid_cases=0
    rec=np.zeros(1,dtype=RECORD_DTYPE)
    rec['p']=rec['r']=rec['q']=.5
    spec=dict(function='weighted_price',bandwidth=.2,centers=[[.5,.5,.5]])
    for bw in (0.,-1.,1e-160,1e-320,1e308,float('nan'),float('inf')):
        try:
            w.evaluate(rec,dict(spec,bandwidth=bw))
        except ValueError:
            invalid_cases+=1
        else:
            raise AssertionError(('invalid bandwidth accepted',bw))
    for value in (-.01,2.,float('nan'),float('inf')):
        broken=rec.copy();broken['p']=value
        try:
            w.evaluate(broken,spec)
        except ValueError:
            invalid_cases+=1
        else:
            raise AssertionError(('invalid normalized coordinate accepted',value))
    record = dict(status='PASS',scalar_pair_values=values,max_scalar_absolute_error=max_error,
                  compensated_moment_cases=moment_cases,max_case_rows=1000000,
                  invalid_domain_cases=invalid_cases,
                  comparison='Scalar math.exp+math.fsum; compiled sixmoments vs per-value math.fsum '
                             'within2ulps on actual checked cases, not universal bitwise equivalence or full IEEE proof.',
                  seconds=time.perf_counter()-started,
                  source_sha256=hashlib.sha256((ROOT/'code/file_workload.py').read_bytes()).hexdigest(),
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
