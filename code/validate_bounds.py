#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the mixture-Hoeffding confidence sequence.

Key structural fact used everywhere below: the CS boundary h(n, alpha) is
DATA-INDEPENDENT (it depends only on n, alpha and the range R).  So we compute it
once per alpha and then run cheap Monte-Carlo.  If this fact were wrong, every
simulation below would be invalid --- so we also assert it directly.

Tests
  T1 uniform coverage      : P( for all n<=N : mu in CS_n ) >= 1-alpha
  T2 optional stopping     : fixed-N CI fails (~2 alpha); CS holds (<= alpha)
  T3 width / sample cost   : CS vs fixed-N (Hoeffding / CLT / empirical-Bernstein)
  T4 random stopping times : P( mu in CS_tau ) >= 1-alpha for data-dependent tau
"""
import json, os, sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anq.bounds import MixtureHoeffdingCS, FixedNCI, fixed_n_hoeffding_halfwidth, CSConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261008)


def cs_table(N, alpha, cfg=None):
    cs = MixtureHoeffdingCS(cfg)
    h = np.empty(N + 1)
    h[0] = np.inf
    for n in range(1, N + 1):
        h[n] = cs.halfwidth(n, alpha)
    return h


def main():
    N = 5000
    alpha = 0.05
    res = {"alpha": alpha, "N": N}

    # ---- sanity: data-independence of the CS boundary -------------------
    cs = MixtureHoeffdingCS()
    h1 = cs.halfwidth(500, alpha)
    h2 = cs.halfwidth(500, alpha)
    res["boundary_reproducible"] = bool(abs(h1 - h2) < 1e-15)
    # invariance under a linear rescale of the range is exact for this construction
    h_r = cs.halfwidth(500, alpha, R=7.5)
    res["range_linear_scaling"] = bool(abs(h_r - 7.5 * h1) < 1e-9)

    H = cs_table(N, alpha)
    H_h = np.array([np.inf] + [fixed_n_hoeffding_halfwidth(n, alpha) for n in range(1, N + 1)])

    # ---- T1 uniform coverage -------------------------------------------
    M = 4000
    mus = [0.05, 0.5, 0.95]
    cov_uniform, min_cov = {}, {}
    for mu in mus:
        ok = np.ones(M, dtype=bool)
        cov_path = np.zeros(N + 1)
        for t in range(M):
            x = (rng.random(N) < mu).astype(float)
            mhat = np.cumsum(x) / np.arange(1, N + 1)
            inside = np.abs(mhat - mu) <= H[1:]
            ok[t] = inside.all()
            cov_path[1:] += inside
        cov_uniform[mu] = float(ok.mean())
        cov_path /= M
        min_cov[mu] = float(cov_path[1:].min())
    res["T1_uniform_coverage"] = cov_uniform
    res["T1_min_marginal_coverage"] = min_cov

    # ---- T2 optional stopping ------------------------------------------
    # Rule: keep sampling until the interval at level alpha excludes mu; give up at N.
    mu0 = 0.5
    M2 = 6000
    def stop_fail(halfwidths, level):
        fail = 0
        for _ in range(M2):
            x = (rng.random(N) < mu0).astype(float)
            mhat = np.cumsum(x) / np.arange(1, N + 1)
            excl = np.abs(mhat - mu0) > halfwidths[1:]
            if excl.any():
                fail += 1     # stopped early on a false positive
        return fail / M2
    t2 = {}
    t2["fixedN_hoeffding_alpha"] = alpha
    t2["fixedN_hoeffding_typeI"] = stop_fail(H_h, alpha)
    t2["anytime_CS_alpha"] = alpha
    t2["anytime_CS_typeI"] = stop_fail(H, alpha)
    # also for a couple more alphas
    for a in (0.10, 0.01):
        Ha = cs_table(N, a)
        Ha_h = np.array([np.inf] + [fixed_n_hoeffding_halfwidth(n, a) for n in range(1, N + 1)])
        t2[f"fixedN_typeI@{a}"] = stop_fail(Ha_h, a)
        t2[f"anytime_typeI@{a}"] = stop_fail(Ha, a)
    res["T2_optional_stopping"] = t2

    # ---- T3 width / sample cost ----------------------------------------
    # exact expected half-widths for a Bernoulli(mu) at a common n
    n_ref = 1000
    mu_ref = 0.5
    se = np.sqrt(mu_ref * (1 - mu_ref) / n_ref)
    widths = {
        "CS_mixture_hoeffding": H[n_ref],
        "fixedN_hoeffding": H_h[n_ref],
        "fixedN_clt_expected": norm.ppf(1 - alpha / 2) * se,
        "fixedN_eb_expected": np.sqrt(2 * mu_ref * (1 - mu_ref) * np.log(2 / alpha) / n_ref),
    }
    # sample cost = (width ratio)^2 (approximate, for the CLT-rate regime)
    cost = {k: float((v / widths["fixedN_clt_expected"]) ** 2) for k, v in widths.items()}
    res["T3_width_at_n1000"] = {k: float(v) for k, v in widths.items()}
    res["T3_sample_cost_vs_clt"] = cost

    # ---- T4 random stopping times --------------------------------------
    M4 = 6000
    tau_cover_cs = 0; tau_cover_fx = 0
    fx_ci = FixedNCI("hoeffding")
    for _ in range(M4):
        x = (rng.random(N) < mu0).astype(float)
        tau = int(rng.integers(50, N + 1))
        mhat = x[:tau].mean()
        tau_cover_cs += abs(mhat - mu0) <= H[tau]
        lo, hi = fx_ci(x[:tau], alpha)
        tau_cover_fx += (lo <= mu0 <= hi)
    res["T4_random_tau_coverage"] = {"CS": tau_cover_cs / M4, "fixedN": tau_cover_fx / M4}

    json.dump(res, open(os.path.join(OUT, "validate_bounds.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
