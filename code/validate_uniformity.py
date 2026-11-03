#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X8 — The price of uniformity in n, decomposed.

The paper's cost tables report what the sequence costs in samples.  This experiment
says *where the cost comes from*, by comparing, at the same (n, mu_hat, alpha), four
half-widths for the same data:

  wald          the fixed-sample Wald half-width z_{1-alpha/2} sqrt(v/n), the best a
                procedure can do at one pre-declared n (given the variance);
  oracle_rho    the best half-width achievable by ANY single exponential
                supermartingale at this n.  This is still a valid anytime bound; the
                point is that its rho must be chosen, and the right rho depends on n;
  mixture       our tabulated sharp sequence, which mixes over a *fixed* grid of rho so
                that one construction is good at every n;
  chernoff_ref  the asymptotic rate sqrt(2 v (log log(2n) + log(1/alpha)) / n) that a
                time-uniform Chernoff bound attains, as a reference for what a
                path-adaptive construction would reach.

Two ratios are what matter and they are different claims:

  mixture / oracle_rho   the price of using one fixed mixture at every n instead of the
                         n-optimal rho.  This is the part of the cost that is an
                         implementation choice, and it is what a stitched or
                         path-adaptive construction would recover.
  oracle_rho / wald      the price of being valid at every n at all.  This one is not an
                         implementation choice: a statement that holds simultaneously at
                         all n is strictly stronger than one that holds at a single n.

CONVENTION (this is where an earlier version of this script was wrong)
--------------------------------------------------------------------
The canonical implementation `bounds.BernoulliCS.log_M` evaluates the *symmetrised*
mixture

    M_n(mu) = sum_k w_k * ( M_n^(rho_k)(mu) + M_n^(-rho_k)(mu) ) / 2 ,      E[M_n] = 1 .

The factor 1/2 is not cosmetic: it is what makes E[M_n(mu)] = 1 under H_0, and dropping
it makes the statistic exactly twice as large, so the level set {M_n < 1/alpha} comes out
narrower by roughly 0.026 in half-width at n = 500.  A single-rho oracle evaluated
without the 1/2 is therefore a *narrower* object than any legitimate procedure, and
dividing our mixture by it inflates the "price of a fixed prior" by about 10%
(reported 1.435 instead of 1.280 at n = 500).

Every construction in this file is evaluated with the same convention, and
`_selfcheck()` asserts that the generic mixture with the CANON grid and CANON weights
reproduces `BernoulliCS.halfwidth` to 1e-9 on a grid of (n, mu_hat).  The assertion is
the guard: if the two conventions ever drift apart again, this script fails loudly
instead of publishing a ratio.

A second question this file answers: how much of the fixed-prior gap is recoverable by
choosing a *better* prior, as opposed to by abandoning a fixed prior altogether?  A
Gaussian prior on log rho is fitted on a subset of the points and evaluated on the
held-out rest, so the reported gain is out-of-sample.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anq.bounds import BernoulliCS
from anq.config import CANON

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "results")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
POINTS = [(500, 0.09), (1000, 0.30), (2656, 0.09), (4000, 0.50),
          (6921, 0.45), (20000, 0.50)]
RHOS = np.exp(np.linspace(np.log(1e-5), np.log(60.0), 12000))
TARGET = np.log(1.0 / ALPHA)


# ---------------------------------------------------------------------------
# Generic single-rho and generic mixture half-widths, in the CANONICAL convention.
# ---------------------------------------------------------------------------
def log_M_rho(rho, n, S, mu):
    """log[ (M_n^(rho)(mu) + M_n^(-rho)(mu)) / 2 ]  -- symmetrised, E = 1."""
    mu = min(max(mu, 1e-12), 1 - 1e-12)
    Lp = np.logaddexp(np.log1p(-mu), np.log(mu) + rho)
    Ln = np.logaddexp(np.log1p(-mu), np.log(mu) - rho)
    return float(np.logaddexp(rho * S - n * Lp, -rho * S - n * Ln) - np.log(2.0))


def log_M_mix(rhos, log_w, n, S, mu):
    """log sum_k w_k exp( log_M_rho(rho_k, n, S, mu) )."""
    mu = min(max(mu, 1e-12), 1 - 1e-12)
    Lp = np.logaddexp(np.log1p(-mu), np.log(mu) + rhos)
    Ln = np.logaddexp(np.log1p(-mu), np.log(mu) - rhos)
    pos = rhos * S - n * Lp
    neg = -rhos * S - n * Ln
    return float(np.logaddexp.reduce(log_w + np.logaddexp(pos, neg) - np.log(2.0)))


def _hw(f, m, n, target=TARGET):
    """Smallest c >= 0 with f(m + c) >= target.

    Mirrors bounds.BernoulliCS._edge: an outward geometric scan from m, then brentq on the
    bracketing pair, clamping at the upper end of [0,1].  The clamp matters at small n,
    where the interval can reach the boundary; the half-width is then 1-m, not the value
    at which f would have crossed.

    Note that the root is of (f - target), not of f: f returns log M_n, and the crossing
    we want is log M_n = log(1/alpha).  Solving f = 0 instead returns the point where
    M_n = 1, which is a much narrower "interval" and is the bug the self-check caught.
    """
    if f(m) >= target:
        return 0.0
    g = lambda x: f(x) - target
    a, step = m, 0.5 / np.sqrt(n)
    for _ in range(160):
        b = min(a + step, 1.0)
        if g(b) >= 0.0:
            return abs(brentq(g, a, b, xtol=1e-13, rtol=1e-13) - m)
        if b >= 1.0:
            return 1.0 - m
        a, step = b, step * 1.5
    return a - m


def hw_rho(rho, n, S, mu):
    return _hw(lambda x: log_M_rho(rho, n, S, x), mu, n)


def hw_mix(rhos, log_w, n, S, mu):
    return _hw(lambda x: log_M_mix(rhos, log_w, n, S, x), mu, n)


def oracle_rho(n, S, mu, rhos=RHOS):
    """Best rho for THIS n, i.e. an oracle that is allowed to know n."""
    best, best_r = np.inf, None
    for r in rhos:
        h = hw_rho(r, n, S, mu)
        if h < best:
            best, best_r = h, r
    return best, float(best_r)


def canon_logw(rhos):
    idx = np.arange(len(rhos), dtype=float)
    w = (idx + 1.0) ** (-CANON.weight_pow)
    return np.log(w / w.sum())


def gauss_prior(m, s, nk=41, lo=1e-4, hi=100.0):
    """A Gaussian prior on log rho, discretised on a geometric grid."""
    r = np.exp(np.linspace(np.log(lo), np.log(hi), nk))
    lw = -0.5 * ((np.log(r) - m) / s) ** 2
    lw = lw - np.log(np.sum(np.exp(lw)))
    return r, lw


# ---------------------------------------------------------------------------
def _selfcheck():
    """The generic mixture must reproduce the canonical implementation exactly."""
    cs = BernoulliCS(CANON)
    r = CANON.kappas()
    lw = canon_logw(r)
    worst = 0.0
    for n in (37, 500, 1000, 2656, 6921, 20000):
        for muh in (0.02, 0.09, 0.30, 0.45, 0.50, 0.91):
            h_canon = cs.halfwidth(n, muh, ALPHA)
            h_mine = hw_mix(r, lw, n, n * muh, muh)
            worst = max(worst, abs(h_canon - h_mine))
    return worst


def main():
    t0 = time.time()
    err = _selfcheck()
    print("selfcheck: max |generic - BernoulliCS| = %.3e   %s"
          % (err, "OK" if err < 1e-9 else "*** MISMATCH ***"), flush=True)
    if err >= 1e-9:
        raise SystemExit("convention drift: refusing to publish ratios")

    cs = BernoulliCS(CANON)
    canon_r = CANON.kappas()
    canon_lw = canon_logw(canon_r)

    rows = []
    for n, muhat in POINTS:
        S = n * muhat
        v = muhat * (1.0 - muhat)
        h_mix = cs.halfwidth(n, muhat, ALPHA)
        h_or, rho_star = oracle_rho(n, S, muhat)
        h_wald = norm.ppf(1 - ALPHA / 2) * np.sqrt(v / n)
        h_cher = np.sqrt(2 * v * (np.log(np.log(2 * n)) + np.log(1 / ALPHA)) / n)
        rows.append({
            "n": n, "mu_hat": round(muhat, 6), "variance": round(v, 6),
            "wald": round(h_wald, 6),
            "oracle_rho": round(h_or, 6), "oracle_rho_value": round(rho_star, 5),
            "mixture": round(h_mix, 6), "chernoff_ref": round(float(h_cher), 6),
            "mixture_over_oracle": round(h_mix / h_or, 3),
            "oracle_over_wald": round(h_or / h_wald, 3),
            "mixture_over_wald": round(h_mix / h_wald, 3)})
        print(rows[-1], flush=True)

    # ---- How much of the fixed-prior gap is recoverable with a better fixed prior?
    # Fit on a subset, evaluate on the held-out rest, so the gain is out-of-sample.
    FIT = [(500, 0.09), (4000, 0.50), (20000, 0.50)]
    TEST = [(1000, 0.30), (2656, 0.09), (6921, 0.45)]

    def ratios(points, r, lw):
        rs = [hw_mix(r, lw, n, n * m, m) / oracle_rho(n, n * m, m)[0]
              for n, m in points]
        return float(np.mean(rs)), float(np.max(rs))

    best = None
    for m in np.linspace(-3.2, -0.4, 15):
        for s in np.linspace(0.3, 1.6, 14):
            r, lw = gauss_prior(m, s)
            mr, _ = ratios(FIT, r, lw)
            if best is None or mr < best[0]:
                best = (mr, m, s)
    _, m_fit, s_fit = best
    r_g, lw_g = gauss_prior(m_fit, s_fit)

    prior = {}
    for tag, (r, lw) in (("canon", (canon_r, canon_lw)),
                         ("gauss_fit", (r_g, lw_g))):
        a, b = ratios(FIT, r, lw)
        c, d = ratios(TEST, r, lw)
        e, f = ratios(POINTS, r, lw)
        prior[tag] = {"fit_mean": round(a, 3), "fit_worst": round(b, 3),
                      "test_mean": round(c, 3), "test_worst": round(d, 3),
                      "all_mean": round(e, 3), "all_worst": round(f, 3)}
        print("%-10s FIT %.3f/%.3f  TEST %.3f/%.3f  ALL %.3f/%.3f"
              % (tag, a, b, c, d, e, f), flush=True)
    prior["gauss_fit"]["log_rho_center"] = round(m_fit, 4)
    prior["gauss_fit"]["log_rho_sd"] = round(s_fit, 4)
    prior["gauss_fit"]["rho_center"] = round(float(np.exp(m_fit)), 4)

    res = {"alpha": ALPHA, "cs_cfg": CANON.__dict__, "n_rho_grid": RHOS.size,
           "selfcheck_max_abs_err": float(err),
           "convention": "symmetrised, E[M_n]=1, matching bounds.BernoulliCS.log_M",
           "rows": rows,
           "mixture_over_oracle_max": round(max(r["mixture_over_oracle"] for r in rows), 3),
           "oracle_over_wald_max": round(max(r["oracle_over_wald"] for r in rows), 3),
           "oracle_over_wald_min": round(min(r["oracle_over_wald"] for r in rows), 3),
           "mixture_over_wald_max": round(max(r["mixture_over_wald"] for r in rows), 3),
           "prior_family": prior,
           "wall_seconds": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(OUT, "x8_uniformity.json"), "w"), indent=1)
    print("wrote x8_uniformity.json %.1fs" % res["wall_seconds"])


if __name__ == "__main__":
    main()
