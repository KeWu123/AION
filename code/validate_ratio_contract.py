"""Behavioral regressions for current closed reports and census registration."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anq.ratio import Prefix, RatioTrace

root = Path(__file__).resolve().parents[1]
(root/'experiments/results').mkdir(parents=True, exist_ok=True)
u = np.r_[np.ones(10), np.zeros(10)]
census = RatioTrace(u, np.ones(20), 20)
assert census.capital_crossed(.5)
assert not census.rejects(.5)
assert census.interval()[0] <= .5 <= census.interval()[1]
# A positive-probability non-census path can reaccept after an old crossing.
# Five unexposed pairs are (u,v)=(0,0), so the population truth is still .5.
partial = RatioTrace(u, np.ones(20), 25)
assert partial.capital_crossed(.5)
assert not partial.rejects(.5)
empty = RatioTrace([], [], 2, Prefix.from_rows([.2, .4], [.5, .5]))
assert empty.interval()[0] <= .6 <= empty.interval()[1]
assert not empty.rejects(.6)
assert not empty.capital_crossed(.6)
before = RatioTrace([], [], 3, Prefix.from_rows([0, 0], [0, 0]))
assert before.interval() == (0., np.inf)
try:
    RatioTrace([], [], 2, Prefix.from_rows([0, 0], [0, 0])).interval()
    raise AssertionError('undefined denominator accepted')
except ValueError:
    pass
subnormal = RatioTrace([1.], [np.nextafter(0., 1.)], 1)
assert np.isinf(subnormal.interval()[1])
record = dict(census=census.interval(), noncensus=partial.interval(),
              empty_census=empty.interval(), zero_denominator_refused=True,
              historical_and_current_decisions_distinguished=True,
              source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), root/'code/anq/ratio.py']})
(root/'experiments/results/ratio_contract_checks.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record,indent=2))
