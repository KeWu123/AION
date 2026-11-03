"""Cached outer brackets for the monotone, directional RatioTrace capitals.

Candidates are fixed using only the registration prefix.  New rows are visited
once in bounded-size chunks; reports inspect O(G) accumulated states, never the
past suffix. Rejected candidates bound the *outside* of the continuous accepted
set. Empty-set containment is vacuous: this monitor need not copy RatioTrace's
particular deterministic-hull fallback when that accepted set is empty.
The arithmetic guards are conservative engineering enclosures, not a replacement
for the exact-arithmetic monotonicity argument.
"""
from __future__ import annotations
import math
import numpy as np


class RatioGrid:
    """Fixed registration grid, fed a prepared finite-population RatioTrace.

    Preparation of predictable arrays is charged separately by the benchmark.
    Only ``advance`` reads suffix rows; ``interval`` uses cached state.  A grid
    point stops permanently when it leaves deterministic feasibility.  Capital
    from a stopped point is never used as a bracket.
    """
    def __init__(self, trace, points=64, chunk_rows=512, candidates=None):
        if trace.population is None:
            raise ValueError('this cached monitor requires a known finite population')
        if int(points) != points or points < 2 or int(chunk_rows) != chunk_rows or chunk_rows < 1:
            raise ValueError('at least two grid points and positive chunk size required')
        self.trace, self.chunk_rows, self.n = trace, int(chunk_rows), 0
        estimate = trace.prefix.su/trace.prefix.sv if trace.prefix.sv > 0 else 0.
        if candidates is None:
            lo, hi = (.5*estimate, 1.5*estimate) if estimate > 0 and np.isfinite(1.5*estimate) else (0., 1.)
            candidates = np.linspace(max(0., lo), hi, int(points))
            # A positive subnormal estimate can collapse adjacent floating-point
            # points. A broad prefix-independent fallback preserves usability;
            # explicitly supplied invalid grids still raise below.
            if np.any(~np.isfinite(candidates)) or np.any(np.diff(candidates) <= 0):
                candidates = np.linspace(0., 1., int(points))
        self.grid = np.asarray(candidates, float).copy()
        if self.grid.ndim != 1 or len(self.grid) < 2 or np.any(~np.isfinite(self.grid)) or np.any(self.grid < 0) or np.any(np.diff(self.grid) <= 0):
            raise ValueError('grid must be finite, nonnegative, and strictly increasing')
        self.registration_estimate = float(estimate)
        self.log_plus = np.zeros(len(self.grid)); self.log_minus = np.zeros(len(self.grid))
        self.abs_plus = np.zeros(len(self.grid)); self.abs_minus = np.zeros(len(self.grid))
        self.active = np.ones(len(self.grid), dtype=bool)
        self.failed = False; self.bad_factor_count = 0
        self._su = [trace.prefix.su, trace.prefix.su]
        self._sv = [trace.prefix.sv, trace.prefix.sv]
        self._positive_v = trace.prefix.sv > 0
        self._cache = None
        self._cache_feasible()
        self.active &= (self.grid >= self._inner[0]) & (self.grid <= self._inner[1])

    @staticmethod
    def _add_interval(state, x):
        if not len(x):
            return
        s = math.fsum(x)
        sl, sh = np.nextafter(s, -np.inf), np.nextafter(s, np.inf)
        state[0] = max(0., float(np.nextafter(state[0]+sl, -np.inf)))
        state[1] = float(np.nextafter(state[1]+sh, np.inf))

    def _cache_feasible(self):
        """Compensated chunk sums plus outward error; no past-row replay."""
        p = self.trace.prefix
        r = self.trace.population-p.n-self.n
        al0, ah0 = self._su; bl0, bh0 = self._sv
        ea = 8*(math.ulp(ah0)+math.ulp(p.su)); eb = 8*(math.ulp(bh0)+math.ulp(p.sv))
        al, ah = max(0., al0-ea), ah0+ea
        bl, bh = max(0., bl0-eb), bh0+eb
        if r == 0 and not self._positive_v:
            raise ValueError('ratio undefined: population denominator is zero')
        lo = max(0., float(np.nextafter(al/(bh+r), -np.inf))) if bh+r > 0 else 0.
        hi = float(np.nextafter((ah+r)/bl, np.inf)) if bl > 0 else np.inf
        self._feasible = (lo, hi)
        # Inner bounds only filter grid points. Over-filtering loses precision.
        ilo = ah0/(bl0+r) if bl0+r > 0 else np.inf
        ihi = (al0+r)/bh0 if bh0 > 0 else np.inf
        self._inner = (float(np.nextafter(ilo, np.inf)), float(np.nextafter(ihi, -np.inf)))

    def advance(self, n=None):
        """Consume each newly exposed row exactly once, in forward chunks."""
        tr = self.trace
        n = len(tr.u) if n is None else int(n)
        if not self.n <= n <= len(tr.u):
            raise ValueError('checkpoint must advance monotonically within the suffix')
        self._cache = None
        while self.n < n:
            start, stop = self.n, min(n, self.n+self.chunk_rows)
            if not self.failed and tr.alpha > 0 and np.any(self.active):
                g = self.grid[None, :]
                a, b, rem = tr.a[start:stop, None], tr.b[start:stop, None], tr.remaining[start:stop, None]
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    # Stop before the first infeasible row. Deterministic sets
                    # shrink, so a stopped candidate can never return.
                    prelo = a/(b+rem)
                    prehi = np.divide(a+rem, b, out=np.full_like(a, np.inf), where=b > 0)
                    admissible = (g >= prelo) & (g <= prehi)
                    alive = np.logical_and.accumulate(admissible, axis=0) & self.active[None, :]
                    m = -(a-g*b)/rem
                    d = tr.u[start:stop, None]-g*tr.v[start:stop, None]-m
                    denp, denm = g+m, 1-m
                    bp = np.divide(tr.cap, denp, out=np.full_like(denp, np.inf), where=denp > 0)
                    bm = np.divide(tr.cap, denm, out=np.full_like(denm, np.inf), where=denm > 0)
                    xp = np.minimum(tr.scale[start:stop, None], bp)*d
                    xm = -np.minimum(tr.scale[start:stop, None], bm)*d
                    bad = alive & ((~np.isfinite(xp)) | (~np.isfinite(xm)) | (xp <= -1) | (xm <= -1))
                    if np.any(bad):
                        self.failed = True
                        self.bad_factor_count += int(np.count_nonzero(bad))
                    else:
                        xp = np.where(alive, xp, 0.); xm = np.where(alive, xm, 0.)
                        lp, lm = np.log1p(xp), np.log1p(xm)
                        self.log_plus += lp.sum(axis=0); self.log_minus += lm.sum(axis=0)
                        self.abs_plus += np.abs(lp).sum(axis=0); self.abs_minus += np.abs(lm).sum(axis=0)
                        self.active = alive[-1].copy()
            self._add_interval(self._su, tr.u[start:stop]); self._add_interval(self._sv, tr.v[start:stop])
            self._positive_v = self._positive_v or bool(np.any(tr.v[start:stop] > 0))
            self.n = stop
        self._cache_feasible()
        self.active &= (self.grid >= self._inner[0]) & (self.grid <= self._inner[1])
        return self

    def interval(self):
        """Return cached conservative outer endpoints at the consumed prefix."""
        if self._cache is not None:
            return self._cache
        lo, hi = self._feasible
        tr = self.trace
        if self.n == 0 or self.failed or tr.alpha == 0 or tr.prefix.n+self.n == tr.population:
            self._cache = (lo, hi)
            return self._cache
        target = math.log(2/tr.alpha)
        # This also dominates observed differences between chunked and scalar
        # accumulation in the independent validation. Borderline states retain D.
        eps = np.finfo(float).eps
        ep = 128*eps*(1+self.n+self.abs_plus)
        em = 128*eps*(1+self.n+self.abs_minus)
        shared=tr.log_error_bound(self.grid,self.n)
        if np.any(~np.isfinite(shared[self.active])):
            self._cache=(lo,hi)
            return self._cache
        ep += shared; em += shared
        plus = self.active & (self.log_plus-ep >= target)
        minus = self.active & (self.log_minus-em >= target)
        # Also enclose ordinary resolved, nonempty 48-step reference bisection
        # endpoints. No claim is made to copy its D-on-empty/failure fallback.
        pad = ((hi-lo)*2**-47 if np.isfinite(hi) else 0.)+1e-12*(1+max(self.grid))
        lower = max(lo, float(self.grid[plus][-1]-pad)) if np.any(plus) else lo
        upper = min(hi, float(self.grid[minus][0]+pad)) if np.any(minus) else hi
        self._cache = (lower, upper) if lower <= upper else (lo, hi)
        return self._cache

    @property
    def persistent_bytes(self):
        return sum(x.nbytes for x in (self.grid, self.log_plus, self.log_minus, self.abs_plus, self.abs_minus, self.active))

    @property
    def chunk_working_set_upper_bytes(self):
        """Conservative array allocation estimate, excluding input trace arrays."""
        return int(20*self.chunk_rows*len(self.grid)*8+8*self.chunk_rows*8)
