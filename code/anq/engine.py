#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive AQP engine.

A *query* is an aggregate over a table, specified by a per-row contribution
function f(row_index) -> value in [lo, hi] (e.g. 1{phi} * value).  The engine draws
rows i.i.d. uniformly with replacement and maintains either a fixed-sample-size CI
or an anytime-valid confidence sequence.

Certification task: given a reference value v, certify "aggregate != v" as soon as
the current interval excludes v, at an arbitrary data-dependent time.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from .bounds import (BernoulliCS, MixtureHoeffdingCS, CSConfig,
                     fixed_n_hoeffding_halfwidth)


# ---------------------------------------------------------------------------
# Exact certification thresholds for the sharp sequence, at a fixed reference
# ---------------------------------------------------------------------------
# A query asks whether theta != v.  The sharp sequence certifies at time n exactly
# when v falls outside its interval, which is exactly the event M_n(v) >= 1/alpha.
# Solving M_n(v) = 1/alpha for the *observed sum* S_n instead of for a half-width
# removes every dependence on the running mean: for fixed (v, alpha) the boundary is
# a pair of thresholds on S_n alone, hence on the running mean alone, and it can be
# tabulated once and reused by every query that shares the reference value.  That is
# what makes the sharp boundary as cheap per report as the selection-blind Hoeffding
# mixture while being strictly tighter.
#
# M_n(v; S) is a sum of exponentials in S, hence convex in S, so its sublevel set
# {S : M_n(v;S) < 1/alpha} is an interval: two roots, one on each side of the
# minimiser, and both are found by bisection.  At S = n*v the value is at most 1
# (Gibbs/Jensen, see Corollary 1), so v is always inside the interval at the running
# mean and both brackets are valid.
DEFAULT_ALPHAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                  0.01, 0.03, 0.05, 0.1, 0.2)


def grid_index(alphas, level: float) -> int:
    """The table column to read for a requested level: the largest grid alpha <= level.

    That is the widest interval the grid can offer, hence the one whose guarantee covers
    the requested level.  Reading the *upper* endpoint from the grid alpha above the level
    would narrow the upper arm below what the level permits --- the interval would certify
    more readily than it is charged, breaking anytime validity.

    This single function is the one place the rule is implemented.  ``CertThreshold.arrays``
    and both experiment drivers (``exp_selection.sharp_pair``, ``exp_e2e.gidx``) call it,
    and ``code/validate_tabulation.py`` asserts they all resolve to this same object, so a
    driver cannot drift away from the verified path again.
    """
    a = np.asarray(alphas)
    if not len(a) or level < a[0]:
        raise ValueError('level is below the available alpha grid; disable certification')
    return max(int(np.searchsorted(a, level, side="right")) - 1, 0)


def _minimiser(cs: BernoulliCS, n: int, v: float) -> int:
    """Integer S in [0,n] minimising the convex function log M_n(v; S)."""
    a, b = 0, n
    while a < b:
        m = (a + b) // 2
        if cs.log_M(n, m + 1, v) >= cs.log_M(n, m, v):
            b = m
        else:
            a = m + 1
    return a


def mean_thresholds(cs: BernoulliCS, n: int, v: float, alpha: float):
    """(lo, hi) such that the sharp sequence at level alpha certifies 'theta < v'
    iff the running mean is <= lo, and 'theta > v' iff it is >= hi.

    lo = -inf means no certification is possible below v within n samples;
    hi = +inf means none above.  Both are legitimate answers.

    log M_n(v; S) is convex in S with minimiser S*, and M_n(v; v*n) <= 1 < 1/alpha,
    so f = log M_n(v; .) - log(1/alpha) satisfies f(S*) < 0 and the two roots are the
    last index with f >= 0 below S* and the first index with f >= 0 above it.  The two
    bisections therefore have OPPOSITE polarities: on [0, S*] the predicate f>=0 is
    true then false, on [S*, n] it is false then true.
    """
    target = np.log(1.0 / alpha)
    if n <= 0:
        return -np.inf, np.inf
    star = _minimiser(cs, n, v)
    if cs.log_M(n, star, v) >= target:      # cannot happen for alpha < 1; be safe
        return -np.inf, np.inf

    if cs.log_M(n, 0, v) >= target:                    # lower root exists
        a, b = 0, star                                 # f(a) >= 0 > f(b)
        while a + 1 < b:
            m = (a + b) // 2
            if cs.log_M(n, m, v) >= target:
                a = m
            else:
                b = m
        lo = a / n
    else:
        lo = -np.inf

    if cs.log_M(n, n, v) >= target:                    # upper root exists
        a, b = star, n                                 # f(a) < 0 <= f(b)
        while a + 1 < b:
            m = (a + b) // 2
            if cs.log_M(n, m, v) >= target:
                b = m
            else:
                a = m
        hi = b / n
    else:
        hi = np.inf
    return lo, hi


class CertThreshold:
    """Tabulated sharp certification thresholds for one reference value.

    The thresholds are precomputed on a grid of (n, alpha) and read back with two
    conservative roundings:
      * n     -> the largest grid n not exceeding the requested n.  LO is increasing in
                 n and HI decreasing, both towards the reference, so a smaller n always
                 gives the harder test.
      * alpha -> the largest grid alpha not exceeding the requested level, used for
                 *both* endpoints.  The interval widens as alpha shrinks (LO falls, HI
                 rises), so the largest grid alpha at or below the requested level is the
                 one that makes the interval widest and therefore certification hardest.
    Both roundings only make certification harder, so validity is preserved and the cost
    is a few percent of the sample count.

    Note on the alpha rounding: rounding the upper endpoint to the nearest grid alpha
    *above* the level would shrink HI and make certification *easier*, i.e. it would be
    anti-conservative.  The two endpoints therefore share one index.  This was a genuine
    defect in an earlier revision: it made the effective level up to one grid step (a
    factor of 3 on this grid) larger than the level the procedure was charged.
    """

    def __init__(self, cfg: CSConfig | None = None, reference: float = 0.5,
                 nmax: int = 20000, alphas=DEFAULT_ALPHAS, use_sharp: bool = True):
        self.reference = reference
        self.alphas = np.array(sorted(alphas))
        self.ns = np.unique(np.round(np.geomspace(1, nmax, 160)).astype(int))
        self.nmax = nmax
        self.use_sharp = use_sharp
        self.LO = np.full((self.ns.size, self.alphas.size), -np.inf)
        self.HI = np.full((self.ns.size, self.alphas.size), np.inf)
        if use_sharp:
            cs = BernoulliCS(cfg)
            for i, n in enumerate(self.ns):
                for j, a in enumerate(self.alphas):
                    self.LO[i, j], self.HI[i, j] = mean_thresholds(
                        cs, int(n), reference, float(a))
        else:
            hoc = MixtureHoeffdingCS(cfg)
            for i, n in enumerate(self.ns):
                for j, a in enumerate(self.alphas):
                    h = hoc.halfwidth(int(n), float(a))
                    self.LO[i, j], self.HI[i, j] = reference - h, reference + h

    def arrays(self, level: float, nmax: int | None = None):
        """Per-n conservative (lo, hi) threshold arrays for n = 1..nmax."""
        nmax = self.nmax if nmax is None else nmax
        if level <= self.alphas[0]:
            return np.full(nmax, -np.inf), np.full(nmax, np.inf)
        j = grid_index(self.alphas, level)
        idx = np.searchsorted(self.ns, np.arange(1, nmax + 1), side="left")
        idx = np.maximum(idx - 1, 0)
        return self.LO[idx, j], self.HI[idx, j]

    def crosses(self, mhat: np.ndarray, level: float):
        """Mask over n = 1..len(mhat) of the certification events."""
        lo, hi = self.arrays(level, mhat.size)
        return (mhat <= lo) | (mhat >= hi)


# --- CS boundary cache: h(n, level) depends on n and level only (data-independent)
_CS_TABLE: dict = {}


def cs_table(max_n: int, level: float, cfg: CSConfig | None = None) -> np.ndarray:
    key = (max_n, float(level), repr(cfg))
    if key in _CS_TABLE:
        return _CS_TABLE[key]
    cs = MixtureHoeffdingCS(cfg)
    h = np.empty(max_n + 1)
    h[0] = np.inf
    for n in range(1, max_n + 1):
        h[n] = cs.halfwidth(n, level)
    _CS_TABLE[key] = h
    return h


class CSInterp:
    """Conservative, cheap half-width oracle for *data-dependent* levels.

    The CS half-width h(n, alpha) is decreasing in n and decreasing in alpha (a smaller
    alpha is a stronger guarantee and a wider interval).  On a
    grid of (log n, log alpha) we therefore get a guaranteed *upper* bound on any
    interior point by taking the corner with the smallest bracketing n and the
    largest bracketing alpha.  Using an upper bound only widens intervals, so the
    anytime guarantee (Theorem 2) is preserved.
    """

    def __init__(self, cfg: CSConfig | None = None, nmax: int = 20000,
                 alphas: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1, 0.2)):
        self.cfg = cfg
        self.nmax = nmax
        self.alphas = np.array(sorted(alphas))
        self.ns = np.unique(np.round(np.geomspace(1, nmax, 160)).astype(int))
        cs = MixtureHoeffdingCS(cfg)
        self.H = np.empty((self.ns.size, self.alphas.size))
        for i, n in enumerate(self.ns):
            for j, a in enumerate(self.alphas):
                self.H[i, j] = cs.halfwidth(int(n), float(a))

    def hw(self, n: int, alpha: float) -> float:
        if n <= 0:
            return np.inf
        n = min(max(int(n), 1), self.nmax)
        # smallest bracketing n
        i1 = int(np.searchsorted(self.ns, n, side="left"))
        i0 = max(i1 - 1, 0)
        # largest bracketing alpha
        j0 = int(np.searchsorted(self.alphas, alpha, side="right")) - 1
        if j0 < 0:
            return np.inf
        j1 = min(j0 + 1, self.alphas.size - 1)
        return float(max(self.H[i0, j0], self.H[i0, j1]))


@dataclass
class QueryResult:
    n_used: int
    estimate: float
    lo: float
    hi: float
    certified: bool
    trace_n: list = field(default_factory=list)
    trace_est: list = field(default_factory=list)
    trace_lo: list = field(default_factory=list)
    trace_hi: list = field(default_factory=list)


class AnytimeEngine:
    def __init__(self, table: dict, alpha: float, method: str = "cs",
                 cs_cfg: CSConfig | None = None, lo: float = 0.0, hi: float = 1.0,
                 seed: int | None = None):
        self.table = table
        self.n_table = len(next(iter(table.values())))
        self.alpha = alpha
        self.method = method            # 'cs' | 'hoeffding' | 'clt' | 'eb'
        self.lo, self.hi = lo, hi
        self.R = hi - lo
        self.cs_cfg = cs_cfg
        self.rng = np.random.default_rng(seed)

    # ---- half-width vector for a whole trajectory ------------------------
    def _halfwidths(self, vals: np.ndarray, level: float) -> np.ndarray:
        n = vals.size
        if self.method == "cs":
            return self.R * cs_table(n, level, self.cs_cfg)[1:]
        ns = np.arange(1, n + 1, dtype=float)
        if self.method == "hoeffding":
            return fixed_n_hoeffding_halfwidth(n, level, self.R) * np.ones(n) if False else \
                self.R * np.sqrt(np.log(2.0 / level) / (2.0 * ns))
        # running mean / variance (vectorised)
        from scipy.stats import norm
        c1 = np.cumsum(vals)
        c2 = np.cumsum(vals ** 2)
        mean = c1 / ns
        var = np.zeros(n)
        if n > 1:
            v = (c2[1:] - c1[1:] ** 2 / ns[1:]) / (ns[1:] - 1.0)
            var[1:] = np.maximum(v, 0.0)
        if self.method == "clt":
            return norm.ppf(1.0 - level / 2.0) * np.sqrt(var / ns)
        # empirical Bernstein (Maurer & Pontil 2009)
        L = np.log(2.0 / level)
        denom = np.maximum(ns - 1.0, 1.0)
        return np.sqrt(2.0 * var * L / ns) + 7.0 * self.R * L / (3.0 * denom)

    def run(self, contrib_fn, reference: float | None, max_n: int,
            level: float | None = None, record: bool = False,
            stop_rule: str = "exclude") -> QueryResult:
        """Draw max_n samples.  'exclude' => stop when interval excludes reference.
        'fixed' => never stop early (use all max_n samples)."""
        level = self.alpha if level is None else level
        idx = self.rng.integers(0, self.n_table, size=max_n)
        vals = np.asarray(contrib_fn(idx), dtype=np.float64)
        csum = np.cumsum(vals)
        ns = np.arange(1, max_n + 1)
        mhat = csum / ns
        hw = self._halfwidths(vals, level)
        lo = mhat - hw
        hi = mhat + hw
        if reference is not None and stop_rule == "exclude":
            mask = (lo > reference) | (hi < reference)
            certified = bool(mask.any())
            n_used = int(np.argmax(mask) + 1) if certified else max_n
        else:
            certified = bool(reference is not None and (lo[-1] > reference or hi[-1] < reference))
            n_used = max_n
        return QueryResult(
            n_used=n_used, estimate=float(mhat[n_used - 1]),
            lo=float(lo[n_used - 1]), hi=float(hi[n_used - 1]), certified=certified,
            trace_n=(list(map(int, ns)) if record else []),
            trace_est=(list(map(float, mhat)) if record else []),
            trace_lo=(list(map(float, lo)) if record else []),
            trace_hi=(list(map(float, hi)) if record else []))
