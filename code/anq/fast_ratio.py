"""Compiled evaluation of the unchanged directional ratio factors.

No fastmath; prepared centers, scales, numerical guard and closed inversion
remain those of RatioTrace. This is an optimization, not a new certificate.
"""
import math
import numpy as np
from numba import njit
from .ratio import RatioTrace


@njit(cache=False)
def _paths(theta,n,u,v,a,b,scale,remaining,finite,cap):
    positive=np.empty(n);negative=np.empty(n);p=0.;m=0.
    for i in range(n):
        center=-(a[i]-theta*b[i])/remaining[i] if finite else 0.
        d=u[i]-theta*v[i]-center
        dp=theta+center;dm=1-center
        lp=min(scale[i],cap/dp) if dp>0 else scale[i]
        lm=min(scale[i],cap/dm) if dm>0 else scale[i]
        fp=1+lp*d;fm=1-lm*d
        if not math.isfinite(fp) or not math.isfinite(fm) or fp<=0 or fm<=0:
            raise ValueError('candidate infeasible or numerical factor failure')
        p+=math.log1p(lp*d);m+=math.log1p(-lm*d)
        positive[i]=p;negative[i]=m
    return positive,negative


class FastRatioTrace(RatioTrace):
    def log_paths(self,theta,n=None):
        if not np.isfinite(theta) or theta<0:
            raise ValueError('finite nonnegative ratio candidate required')
        n=len(self.u) if n is None else int(n)
        if not 0<=n<=len(self.u):raise ValueError('invalid checkpoint')
        remaining=np.empty(0) if self.population is None else self.remaining
        return _paths(float(theta),n,self.u,self.v,self.a,self.b,self.scale,
                      remaining,self.population is not None,self.cap)


def warmup():
    for N in [None,8]:
        tr=FastRatioTrace(np.array([.3,.7]),np.ones(2),N)
        tr.interval(iterations=2)
