"""Serial shared-scan controller implementing registration and exposure accounting.

Contribution callbacks are trusted pure functions of supplied row identities.
The controller is an engine component, not a sandbox for malicious callbacks or
an information-flow proof about reads performed outside its interface.
"""
from dataclasses import dataclass, field
import hashlib
from fractions import Fraction
import numpy as np
from .ratio import Prefix, RatioTrace


@dataclass
class _Registered:
    u: object
    v: object
    reference: float
    level: float
    start: int
    prefix: Prefix
    us: list = field(default_factory=list)
    vs: list = field(default_factory=list)


class SharedScan:
    """One immutable snapshot, hidden uniform permutation, and serial queries.

    A finite `capacity` is a uniform prefix of the population permutation;
    exhausting it stops observation rather than silently generating fresh rows.
    Population size in every center remains the full size, not capacity.
    """
    def __init__(self, size, alpha=.05, seed=20261016, capacity=None,
                 snapshot_id='fixed-population'):
        if int(size) != size or size < 1 or not 0 <= alpha < 1:
            raise ValueError('positive population size and 0<=alpha<1 required')
        capacity = size if capacity is None else capacity
        if int(capacity) != capacity or not 0 <= capacity <= size:
            raise ValueError('invalid scan capacity')
        self.size, self.alpha = int(size), float(alpha)
        self.snapshot_id = str(snapshot_id)
        self._order = np.random.default_rng(seed).choice(size,size=capacity,replace=False)
        self._t = 0
        self._active = None
        self._spent = 0.
        self._spent_exact = Fraction(0)
        self._k = 0
        self.events = []

    @property
    def exposed(self):
        """Copy of exposed identities only; the remaining order stays private."""
        return self._order[:self._t].copy()

    @property
    def spent(self):
        return self._spent

    def expose(self, n):
        """Warm-up/exploration read; forbidden while a query is active."""
        if self._active is not None:
            raise RuntimeError('active query must consume or stop before an external read')
        ids=self._take(n)
        self.events.append(dict(event='expose',end=self._t,rows=len(ids)))
        return ids.copy()

    def _take(self, n):
        if int(n) != n or n < 0 or self._t+n > len(self._order):
            raise ValueError('read exceeds reserved unseen suffix')
        ids=self._order[self._t:self._t+n]
        self._t += int(n)
        return ids

    @staticmethod
    def _values(fn, ids):
        x=np.asarray(fn(ids.copy()),float)
        if x.shape != ids.shape:
            raise ValueError('contribution callback must return one value per supplied row')
        if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError('contributions must be finite in [0,1]')
        return x

    def register(self, u, v, reference, level=None):
        if self._active is not None:
            raise RuntimeError('only one registered query may be active')
        if not np.isfinite(reference) or reference < 0:
            raise ValueError('finite nonnegative reference required')
        k=self._k+1
        proposed=self.alpha*6/(np.pi*np.pi*k*k) if level is None else float(level)
        if not np.isfinite(proposed) or proposed < 0 or proposed >= 1:
            raise ValueError('invalid query level')
        # Compare against remaining budget before addition; no tolerance can
        # authorize spending past the declared budget.
        amount=Fraction.from_float(proposed)
        if amount > Fraction.from_float(self.alpha)-self._spent_exact:
            raise ValueError('session error budget exhausted')
        ids=self.exposed
        pu,pv=self._values(u,ids),self._values(v,ids)
        prefix=Prefix.from_rows(pu,pv)
        self._active=_Registered(u,v,float(reference),proposed,self._t,prefix)
        self._spent_exact += amount
        self._spent = float(self._spent_exact)
        self._k=k
        self.events.append(dict(event='register',query=k,start=self._t,
                                prefix=prefix.__dict__,level=proposed,
                                snapshot=self.snapshot_id,capital_initial=1.,
                                exposed_id_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))

    def consume(self, n, iterations=40):
        q=self._active
        if q is None:
            raise RuntimeError('register a query first')
        if n <= 0:
            raise ValueError('consume at least one new row')
        # A failing callback cannot consume hidden rows and then let a
        # subsequent registration ignore that read.
        if int(n) != n or self._t+n > len(self._order):
            raise ValueError('read exceeds reserved unseen suffix')
        ids=self._order[self._t:self._t+int(n)].copy()
        # Callback execution is an exposure even when it raises. Mark it first
        # and invalidate this registration on failure; the next registration
        # must rebase all identities already supplied to a callback.
        self._t += int(n)
        try:
            u,v=self._values(q.u,ids),self._values(q.v,ids)
        except Exception:
            self.events.append(dict(event='failed_callback',query=self._k,end=self._t))
            self._active=None
            raise
        q.us.append(u);q.vs.append(v)
        return self.report(iterations=iterations)

    def report(self, iterations=40):
        """Report the current registered query without exposing another row.

        This also supports registration at census or at an exhausted reserved
        scan prefix. A zero population denominator remains undefined.
        """
        q=self._active
        if q is None:
            raise RuntimeError('register a query first')
        u=np.concatenate(q.us) if q.us else np.empty(0)
        v=np.concatenate(q.vs) if q.vs else np.empty(0)
        trace=RatioTrace(u,v,self.size,q.prefix,q.level)
        lower,upper=trace.interval(iterations=iterations)
        report=dict(query=self._k,start=q.start,end=self._t,suffix_rows=self._t-q.start,
                    level=q.level,lower=lower,upper=upper,
                    certified=bool(q.reference < lower or q.reference > upper))
        self.events.append(dict(event='report',**report))
        return report

    def stop(self):
        if self._active is None:
            raise RuntimeError('no query is active')
        self.events.append(dict(event='stop',query=self._k,end=self._t))
        self._active=None
