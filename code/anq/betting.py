#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Betting (predictable-plug-in) confidence sequence for bounded random variables.

For X_t in [0,1] with mean mu, and any F_{t-1}-measurable lambda_t >= 0 with
lambda_t <= 1/mu, the capital

    K_n(mu) = prod_{t=1..n} ( 1 + lambda_t(mu) (X_t - mu) )

is a nonnegative martingale with K_0 = 1.  By Ville's inequality,
P( exists n : K_n(mu) >= 1/alpha ) <= alpha, so

    CS_n = { mu in [0,1] : K_n(mu) < 1/alpha }

is an anytime-valid confidence sequence.  We use a "predictable plug-in" betting
rule lambda_t(mu) = min( c / (1 - muhat_{t-1}), d / mu ) with muhat the running
mean (muhat_0 = 1/2), which keeps the capital nonnegative for every X_t in [0,1]
and adapts to the observed variance.  (cf. Waudby-Smith & Ramdas, JRSS-B 2024.)
"""
from __future__ import annotations
import numpy as np


def betting_logK_grid(x: np.ndarray, mu_grid: np.ndarray, c: float = 0.5,
                      d: float = 0.95) -> np.ndarray:
    """log K_n(mu) for every mu in mu_grid, vectorised over (mu, t)."""
    n = x.size
    if n == 0:
        return np.zeros_like(mu_grid)
    mhat_prev = np.empty(n)
    mhat_prev[0] = 0.5
    if n > 1:
        mhat_prev[1:] = np.cumsum(x)[:-1] / np.arange(1, n)
    mhat_prev = np.clip(mhat_prev, 1e-9, 1 - 1e-9)
    lam_plug = c / (1.0 - mhat_prev)                     # (n,)
    lam = np.minimum(lam_plug[None, :], d / np.maximum(mu_grid[:, None], 1e-9))
    arg = lam * (x[None, :] - mu_grid[:, None])          # (G, n)
    arg = np.maximum(arg, -1.0 + 1e-12)                  # keep factor >= 0
    logK = np.log1p(arg).sum(axis=1)
    return logK


def betting_cs(x: np.ndarray, alpha: float, c: float = 0.5, d: float = 0.95,
               grid: int = 401) -> tuple:
    """Return (lo, hi) of the betting CS; lo/hi clipped to [0,1]."""
    n = x.size
    if n == 0:
        return (0.0, 1.0)
    mus = np.linspace(0.0, 1.0, grid)
    logK = betting_logK_grid(x, mus, c, d)
    thresh = np.log(1.0 / alpha)
    inside = logK < thresh
    if not inside.any():
        m = float(np.mean(x))
        return (m, m)
    # largest contiguous block containing the running mean
    m = float(np.mean(x))
    j = int(np.argmin(np.abs(mus - m)))
    lo = j
    while lo - 1 >= 0 and inside[lo - 1]:
        lo -= 1
    hi = j
    while hi + 1 < grid and inside[hi + 1]:
        hi += 1
    return (float(mus[lo]), float(mus[hi]))
