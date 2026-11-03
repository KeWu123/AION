"""Behavioral/reference checks for automatic shared-scan prefix accounting."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from anq.session import SharedScan
from anq.ratio import RatioTrace,Prefix
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'experiments/results').mkdir(parents=True, exist_ok=True)
checks=0
for seed in range(40):
    population=np.array([[0,1],[1,.5],[.5,.25],[1,0],[0,.5],[.25,.125]],float)
    s=SharedScan(len(population),seed=seed)
    prefix=s.expose(2)
    # Data-dependent choice genuinely uses the exposed raw prefix.
    col=int(population[prefix].sum(axis=0).argmax())
    u=lambda ids:population[ids,col]
    v=lambda ids:population[ids,1-col]
    s.register(u,v,.5)
    try:s.expose(1);raise AssertionError('active external reads accepted')
    except RuntimeError:pass
    for block in (1,2):
        report=s.consume(block,iterations=35)
        ids=s.exposed
        direct=RatioTrace(u(ids[2:]),v(ids[2:]),len(population),Prefix.from_rows(u(ids[:2]),v(ids[:2])),report['level'])
        expect=direct.interval(iterations=35)
        assert np.array_equal(np.array(expect),np.array([report['lower'],report['upper']]))
        checks+=1
    s.stop()
    s.register(v,u,2.)
    census=s.consume(1)
    truth=float(v(np.arange(6)).sum()/u(np.arange(6)).sum())
    assert census['lower']<=truth<=census['upper']
    assert census['start']==5 and census['end']==6
    assert s.spent <= s.alpha
    checks+=1
# Callback failure consumes exposure and invalidates its old registration.
s=SharedScan(10,capacity=10)
s.expose(2)
calls=[0]
def failure(ids):
    calls[0]+=1
    if calls[0]>1:raise ValueError('callback failure')
    return np.ones(len(ids))
s.register(failure,lambda ids:np.ones(len(ids)),1.)
try:s.consume(3);raise AssertionError('failure not surfaced')
except ValueError:pass
assert len(s.exposed)==5
s.register(lambda ids:np.zeros(len(ids)),lambda ids:np.ones(len(ids)),0.)
assert s.events[-1]['prefix']['n']==5
# Exact accounting of representable levels, and no silent replenishment.
s=SharedScan(10,alpha=.1)
s.register(lambda ids:np.zeros(len(ids)),lambda ids:np.ones(len(ids)),0.,level=.1)
s.stop()
try:s.register(lambda ids:np.zeros(len(ids)),lambda ids:np.ones(len(ids)),0.,level=1e-20);raise AssertionError('overspending allowed')
except ValueError:pass
# No-read reports must work immediately at registration, including census.
no_read_checks=0
for size,expose in [(2,0),(2,1),(2,2)]:
    for level in (0.,.01):
        s=SharedScan(size,seed=1)
        s.expose(expose)
        u=lambda ids:np.full(len(ids),.5)
        v=lambda ids:np.ones(len(ids))
        s.register(u,v,.5,level=level)
        before=s.exposed.copy();spent=s.spent
        report=s.report()
        expected=RatioTrace([],[],size,Prefix.from_rows(u(before),v(before)),level).interval(iterations=40)
        assert (report['lower'],report['upper'])==expected
        assert np.array_equal(s.exposed,before) and s.spent==spent
        assert report['suffix_rows']==0 and not report['certified']
        if expose==size:
            s.stop();s.register(u,v,0.,level=0.)
            assert s.report()['certified']
        no_read_checks+=1
s=SharedScan(2);s.expose(2)
s.register(lambda ids:np.ones(len(ids)),lambda ids:np.zeros(len(ids)),1.)
try:s.report();raise AssertionError('undefined census ratio reported')
except ValueError:pass
result=dict(checks=checks,callback_failure_accounted=True,overspending_refused=True,
            no_read_registration_cases=no_read_checks,undefined_census_refused=True,
            files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'code/anq/session.py',ROOT/'code/anq/ratio.py',Path(__file__)]})
p=ROOT/'experiments/results/session_controller_checks.json';p.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
