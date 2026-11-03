#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the betting CS: uniform coverage, optional stopping, tightness."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anq.betting import betting_cs
from anq.bounds import MixtureHoeffdingCS, fixed_n_hoeffding_halfwidth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "results")
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
N = 2000
M = 3000


def betting_table(N, alpha, c=0.5, d=0.95, grid=401):
    """Data-independent? No: betting CS depends on the path.  So we must simulate."""
    return None


def main():
    rng = np.random.default_rng(2026)
    res = {"alpha": ALPHA, "N": N, "M": M}

    # --- width at a reference n (per-path) --------------------------------
    nref = 500
    widths_bet, widths_cs = [], []
    cs = MixtureHoeffdingCS()
    for _ in range(600):
        x = (rng.random(nref) < 0.5).astype(float)
        lo, hi = betting_cs(x, ALPHA)
        widths_bet.append(hi - lo)
        widths_cs.append(2 * cs.halfwidth(nref, ALPHA))
    res["median_width_n500"] = {
        "betting": float(np.median(widths_bet)),
        "mixture_hoeffding": float(np.median(widths_cs)),
        "fixedN_hoeffding": float(2 * fixed_n_hoeffding_halfwidth(nref, ALPHA)),
    }

    # --- uniform coverage over all n (path-wise) --------------------------
    # betting CS is not data-independent -> recompute at a geometric subgrid of n
    ns_check = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    mus = [0.2, 0.5, 0.8]
    cov = {}
    for mu in mus:
        bad = 0
        for _ in range(M):
            x = (rng.random(N) < mu).astype(float)
            ok = True
            for n in ns_check:
                lo, hi = betting_cs(x[:n], ALPHA, grid=201)
                if not (lo <= mu <= hi):
                    ok = False
                    break
            if not ok:
                bad += 1
        cov[mu] = 1.0 - bad / M
    res["uniform_coverage_at_checkpoints"] = cov

    # --- optional stopping type-I error -----------------------------------
    # rule: stop the first time the interval excludes mu (null true)
    mu0 = 0.5
    fail_bet = 0
    for _ in range(M):
        x = (rng.random(N) < mu0).astype(float)
        hit = False
        for n in ns_check:
            lo, hi = betting_cs(x[:n], ALPHA, grid=201)
            if hi < mu0 or lo > mu0:
                hit = True
                break
        if hit:
            fail_bet += 1
    res["optional_stopping_typeI_betting"] = fail_bet / M
    res["optional_stopping_nominal"] = ALPHA

    json.dump(res, open(os.path.join(OUT, "validate_betting.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
