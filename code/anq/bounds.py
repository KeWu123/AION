#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical bounds for interactive AQP.

Fixed-sample-size bounds (classical baselines):
  - FixedNCI(kind='clt'|'hoeffding'|'eb')  : valid only for a *pre-declared* n.

Anytime-valid confidence sequence (our core; self-contained and provable):
  - MixtureHoeffdingCS

Derivation of MixtureHoeffdingCS
--------------------------------
Let X_1..X_n in [a,b], R = b-a, mu = E[X], S_n = sum X_i, mu_hat = S_n/n.
By Hoeffding's lemma, for any rho in R:  E[exp(rho (X-mu))] <= exp(rho^2 R^2 / 8).
Hence for each rho,  M_n^(rho) := exp( rho (S_n - n mu) - n rho^2 R^2 / 8 )
is a nonnegative supermartingale with M_0^(rho)=1.
Any mixture  M_n(mu) = sum_k w_k * (1/2)[ M_n^(rho_k)(mu) + M_n^(-rho_k)(mu) ]
(with sum_k w_k = 1, rho_k = kappa_k / R) is again a nonnegative supermartingale
with M_0 = 1.  Ville's inequality gives

    P( exists n>=1 : M_n(mu) >= 1/alpha ) <= alpha .

Therefore  CS_n = { mu : M_n(mu) < 1/alpha }  satisfies
    P( for all n>=1 : mu in CS_n ) >= 1 - alpha ,
and, because the property is uniform in n, it also holds at ANY stopping time tau
(optional stopping for the underlying supermartingale).  With zeta = (S_n - n mu)/R,

    log M_n = logsumexp_k [ log(w_k/2) + log(2 cosh(kappa_k zeta)) - n kappa_k^2 / 8 ] .

M_n is even and strictly increasing in |mu - mu_hat|, so the CS is the symmetric
interval mu_hat +- h where h solves M_n(mu_hat +- h) = 1/alpha.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Fixed-sample-size confidence intervals (baselines)
# --------------------------------------------------------------------------
def fixed_n_hoeffding_halfwidth(n: int, alpha: float, R: float = 1.0) -> float:
    """Two-sided Hoeffding half-width for a pre-declared sample size n."""
    if n <= 0:
        return np.inf
    return R * np.sqrt(np.log(2.0 / alpha) / (2.0 * n))


class FixedNCI:
    """Classical fixed-sample-size confidence interval.  Valid ONLY if n is fixed
    in advance (no peeking)."""

    def __init__(self, kind: str = "hoeffding", lo: float = 0.0, hi: float = 1.0):
        assert kind in ("clt", "hoeffding", "eb")
        self.kind = kind
        self.lo, self.hi = lo, hi

    def __call__(self, x: np.ndarray, alpha: float):
        n = x.size
        m = float(np.mean(x)) if n else np.nan
        if n == 0:
            return (np.nan, np.nan)
        R = self.hi - self.lo
        if self.kind == "clt":
            s = float(np.std(x, ddof=1)) if n > 1 else 0.0
            hw = norm.ppf(1.0 - alpha / 2.0) * s / np.sqrt(n)
        elif self.kind == "hoeffding":
            hw = fixed_n_hoeffding_halfwidth(n, alpha, R)
        else:  # empirical Bernstein (Maurer & Pontil 2009)
            L = np.log(2.0 / alpha)
            s2 = float(np.var(x, ddof=1)) if n > 1 else 0.0
            hw = np.sqrt(2.0 * s2 * L / n) + 7.0 * R * L / (3.0 * max(n - 1, 1))
        return (m - hw, m + hw)


# --------------------------------------------------------------------------
# Anytime-valid confidence sequence
# --------------------------------------------------------------------------
@dataclass
class CSConfig:
    kappa0: float = 0.05
    growth: float = 2.0
    nk: int = 13
    weight_pow: float = 1.1   # w_k proportional to (k+1)^(-weight_pow)

    def kappas(self):
        return self.kappa0 * self.growth ** np.arange(self.nk)

    def weights(self):
        k = np.arange(self.nk, dtype=float) + 1.0
        w = k ** (-self.weight_pow)
        return w / w.sum()


# The shipped configuration: the tightest valid point of the 108-configuration
# sweep in code/tune_cs.py (kappa0=0.01, growth=1.4, nk=16, weight_pow=0.4).
# Defined here, next to CSConfig, so that a class can use it as its own default
# without a circular import; anq/config.py re-exports it as the single import site.
CANON = CSConfig(kappa0=0.01, growth=1.4, nk=16, weight_pow=0.4)


def _log_M(n: int, d_over_R: float, kappas: np.ndarray, log_w_half: np.ndarray) -> float:
    """log M_n(mu_hat + d) with d_over_R = d / R  (>= 0)."""
    if n <= 0:
        return 0.0
    if d_over_R == 0.0:
        # at d=0: terms are 2*exp(-n k^2/8) ; log(w_k/2)+log(2)+... = log w_k - n k^2/8
        return float(np.logaddexp.reduce(log_w_half + np.log(2.0)
                                         - n * kappas ** 2 / 8.0))
    a = kappas * n * d_over_R                      # >= 0
    # log(w_k/2) + log(e^a + e^-a) - n k^2/8
    log_cosh2 = np.logaddexp(a, -a)
    return float(np.logaddexp.reduce(log_w_half + log_cosh2 - n * kappas ** 2 / 8.0))


class MixtureHoeffdingCS:
    """Anytime-valid confidence sequence for the mean of a [lo,hi]-bounded variable."""

    def __init__(self, cfg: CSConfig | None = None, lo: float = 0.0, hi: float = 1.0):
        self.cfg = cfg or CSConfig()
        self.lo, self.hi = lo, hi
        self.kappas = self.cfg.kappas()
        self.log_w_half = np.log(self.cfg.weights() / 2.0)

    def halfwidth(self, n: int, alpha: float, R: float | None = None) -> float:
        """Half-width h at sample size n.  alpha should be the *time-uniform* level."""
        if n <= 0:
            return np.inf
        R = self.hi - self.lo if R is None else R
        target = np.log(1.0 / alpha)

        def f(d_over_R):
            return _log_M(n, max(d_over_R, 0.0), self.kappas, self.log_w_half) - target

        # f(0) <= 0 (validity), f -> +inf ; bracket then bisect.
        if f(0.0) >= 0.0:                     # alpha so large that even d=0 is rejected
            return 0.0
        hi = 1e-6
        while f(hi) < 0.0 and hi < 1e6:
            hi *= 1.6
        d = brentq(f, 0.0, hi, xtol=1e-12, rtol=1e-12)
        return d * R

    def __call__(self, x: np.ndarray, alpha: float):
        n = x.size
        if n == 0:
            return (np.nan, np.nan)
        m = float(np.mean(x))
        h = self.halfwidth(n, alpha)
        return (m - h, m + h)


def cs_halfwidth(n: int, alpha: float, R: float = 1.0, cfg: CSConfig | None = None) -> float:
    """Convenience: mixture-Hoeffding CS half-width, scaled to range R.

    Because the construction only sees d/R, scaling by R is exact."""
    cs = MixtureHoeffdingCS(cfg, lo=0.0, hi=1.0)
    return R * cs.halfwidth(n, alpha, R=1.0)


# --------------------------------------------------------------------------
# Sharp [0,1] confidence sequence ("Bernoulli" / mgf-exact)
# --------------------------------------------------------------------------
# For ANY X in [0,1] with mean mu, the mgf is maximised by a two-point law on
# {0,1}:   E[e^{rho X}] <= (1-mu) + mu e^{rho}  =:  e^{L(mu,rho)} .
# Hence  M_n^(rho)(mu) := exp( rho S_n - n L(mu,rho) )  is a supermartingale with
# M_0 = 1, and any mixture over rho (symmetrised) is too, so Ville applies.
# This bound uses the *true* curvature of the [0,1] domain and is strictly sharper
# than the Hoeffding bound exp(rho^2/8) whenever mu(1-mu) < 1/4.
def _L(mu: float, rho: float) -> float:
    """log( (1-mu) + mu e^{rho} ) in a numerically stable way."""
    if mu <= 0.0:
        return 0.0
    if mu >= 1.0:
        return rho
    return float(np.logaddexp(np.log1p(-mu), np.log(mu) + rho))


class BernoulliCS:
    """Sharp anytime-valid CS for means of variables supported in [lo, hi]."""

    def __init__(self, cfg: CSConfig | None = None, lo: float = 0.0, hi: float = 1.0):
        # Default to the shipped CANON configuration, so a caller who omits cfg gets
        # the same sequence every experiment reports; the earlier ad-hoc default
        # (kappa0=0.05, growth=2.0, nk=13) silently differed from CANON.
        self.cfg = cfg or CANON
        self.lo, self.hi = lo, hi
        self.rhos = self.cfg.kappas()
        self.log_w_half = np.log(self.cfg.weights() / 2.0)

    def log_M(self, n: int, S: float, mu: float) -> float:
        """log M_n(mu) given n samples with sum S (all in the normalised [0,1] scale)."""
        if n <= 0:
            return 0.0
        if not 0 <= mu <= 1:
            raise ValueError('candidate mean must lie in [0,1]')
        if mu == 0:
            Lp = Ln = np.zeros_like(self.rhos)
        elif mu == 1:
            Lp, Ln = self.rhos, -self.rhos
        else:
            l1m = np.log1p(-mu)
            lm = np.log(mu)
            Lp = np.logaddexp(l1m, lm + self.rhos)
            Ln = np.logaddexp(l1m, lm - self.rhos)
        pos = self.rhos * S - n * Lp
        neg = -self.rhos * S - n * Ln
        return float(np.logaddexp.reduce(self.log_w_half + np.logaddexp(pos, neg)))

    def _edge(self, n: int, S: float, m: float, target: float, direction: int) -> float:
        """Find the CS boundary on one side of m (direction = +1 or -1)."""
        def f(mu):
            return self.log_M(n, S, mu) - target
        a = m                                  # f(a) < 0 by assumption
        step = 0.5 / np.sqrt(n)
        for _ in range(120):
            b = a + direction * step
            if direction > 0 and b >= 1.0:
                b = 1.0
            if direction < 0 and b <= 0.0:
                b = 0.0
            if f(b) >= 0.0:
                return brentq(f, min(a, b), max(a, b), xtol=1e-13, rtol=1e-13)
            if b == 0.0 or b == 1.0:
                return b
            a = b
            step *= 1.5
        return a

    def interval(self, x: np.ndarray, alpha: float) -> tuple:
        """CS interval for the sample x (values in [lo,hi])."""
        n = x.size
        if n == 0:
            return (self.lo, self.hi)
        R = self.hi - self.lo
        xz = (np.asarray(x, dtype=float) - self.lo) / R
        S = float(xz.sum())
        m = min(max(S / n, 0.0), 1.0)
        target = np.log(1.0 / alpha)
        if self.log_M(n, S, m) >= target:
            return (self.lo + m * R, self.lo + m * R)
        lo = self._edge(n, S, m, target, -1)
        hi = self._edge(n, S, m, target, +1)
        return (self.lo + lo * R, self.lo + hi * R)

    def halfwidth(self, n: int, mu_hat: float, alpha: float) -> float:
        """Half-width at sample size n and observed mean mu_hat (normalised [0,1]).

        Convention note: this returns the UPPER arm only, which is the shorter or
        equal arm for mu_hat >= 1/2 and the arm the cost tables quote.  The betting
        comparison in exp_betting.py instead reports max(upper, lower); at the
        studied means (mu <= 0.510) the two conventions coincide to the printed
        precision, but they are not identical in general."""
        S = n * mu_hat
        m = min(max(mu_hat, 0.0), 1.0)
        target = np.log(1.0 / alpha)
        if self.log_M(n, S, m) >= target:
            return 0.0
        hi = self._edge(n, S, m, target, +1)
        return abs(hi - m)
