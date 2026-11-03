#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selection-aware error control for a sequence of *adaptively chosen* AQP queries.

Set-up
------
An analyst issues queries Q_1, Q_2, ... one at a time.  Before seeing Q_k's data
she may use everything observed so far (including the outcomes of Q_1..Q_{k-1}) to
decide (i) which query to run and (ii) how much sample budget to give it.  For each
query she may declare a *certified finding*: "the aggregate of Q_k differs from the
reference value v_k by more than the half-width" (i.e. the anytime-valid CS excludes
v_k).

Goal : online family-wise error rate  FWER = P( exists k : H_k true and H_k certified )
       <= alpha, uniformly over all such adaptive strategies.

Design
------
  * SpendingSequence : alpha_k = alpha * 6 / (pi^2 k^2).  Deterministic, so
    F_{k-1}-measurable, with sum_k alpha_k = alpha.  Provably controls FWER
    (Corollary in the paper).  No power benefit from rejections.
  * AlphaInvesting: a legacy capped proportional-wealth policy, pointwise
    dominated by its deterministic gamma spending envelope. It is not the
    Foster--Stine update and is not a contribution of the current paper.

"""
from __future__ import annotations
import numpy as np
from scipy.special import zeta
from dataclasses import dataclass, field

GAMMA_POW_DEFAULT = 1.2


def zeta_em(s: float, N: int = 500000) -> float:
    """zeta(s) for s > 1 by Euler-Maclaurin.

    zeta(1.2) converges slowly (the tail integral alone is ~0.43), so the plain
    partial sum understates it and would make sum_k gamma_k exceed 1; the tail
    integral plus the first correction terms fix the fourth decimal.
    """
    k = np.arange(1, N + 1, dtype=float)
    return float(np.sum(k ** (-s)) + N ** (1.0 - s) / (s - 1.0)
                 + 0.5 * N ** (-s) + s * N ** (-s - 1.0) / 12.0)


GAMMA0_DEFAULT = 1.0 / zeta_em(GAMMA_POW_DEFAULT)     # ~0.1789, sum_k gamma_k = 1


def spending_levels(K: int, alpha: float, exponent: float = 2.0) -> np.ndarray:
    """alpha_k = alpha * Zeta(exponent)^{-1} * k^{-exponent}; sum_{k=1..inf} = alpha."""
    if int(K) != K or K < 0 or not 0 <= alpha < 1 or exponent <= 1:
        raise ValueError('require K>=0, 0<=alpha<1, and exponent>1')
    k = np.arange(1, K + 1, dtype=float)
    w = k ** (-exponent) / np.nextafter(float(zeta(exponent,1)),np.inf)
    return alpha * w


class SpendingSequence:
    """Provably FWER-controlling deterministic spending.

    The level alpha_k is fixed before query k is chosen, so it does not depend on
    which query the analyst picks or on any data.  Hence the union bound over the
    (adaptive) set of true hypotheses applies.
    """

    def __init__(self, alpha: float, exponent: float = 2.0, K: int = 100000):
        self.alpha = alpha
        self.levels = spending_levels(K, alpha, exponent)

    def level(self, k: int) -> float:
        """1-indexed query number."""
        if k <= len(self.levels):
            return float(self.levels[k - 1])
        return 0.0


class AlphaInvesting:
    """Legacy capped proportional-wealth policy, not Foster--Stine investing.

    This API is retained to reproduce historical ablations. With cap=True,
    W<=alpha and summable gamma imply total charged levels<=alpha, provided
    each query has a conditionally valid certificate. It is pointwise
    dominated by deterministic alpha*gamma spending. The net implemented
    rejection update is W <- W+(omega-2)*level, followed by the optional cap;
    nonrejection costs level. Uncapped execution has no asserted FWER or
    mFDR guarantee. The current ratio paper does not use this policy.
    """

    def __init__(self, alpha: float, gamma0: float = None, omega: float = 3.0,
                 gamma_pow: float = GAMMA_POW_DEFAULT, cap: bool = True):
        self.alpha = alpha
        if gamma_pow <= 1 or not 0 <= alpha < 1 or omega < 1:
            raise ValueError('require gamma_pow>1, 0<=alpha<1, omega>=1')
        safe_max = 1.0 / np.nextafter(float(zeta(gamma_pow,1)),np.inf)
        self.gamma0 = safe_max if gamma0 is None else float(gamma0)
        if not 0 <= self.gamma0 <= safe_max:
            raise ValueError('gamma0 must satisfy the infinite spending budget')
        self.omega = omega
        self.gamma_pow = gamma_pow
        self.cap = bool(cap)
        self.W = alpha
        self.k = 0

    def level(self, k: int) -> float:
        g = self.gamma0 * (k ** (-self.gamma_pow))
        return float(min(g * self.W, self.W))

    def update(self, k: int, level_k: float, certified: bool):
        gain = (self.omega - 1.0) * level_k if certified else 0.0
        self.W = self.W - level_k + gain
        if self.cap:
            self.W = min(self.W, self.alpha)
        self.W = max(self.W, 0.0)

    def gamma_sum(self, K: int = 100000) -> float:
        """sum_{k<=K} gamma_k, as a sanity hook for the theorem's hypothesis."""
        return float(np.sum(np.arange(1, K + 1, dtype=float) ** (-self.gamma_pow))
                     * self.gamma0)
