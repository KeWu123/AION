"""Common bounded functions and aggregation for the real-file controlled study."""
import math
import numpy as np
from numba import njit
from anq.ratio import Prefix


@njit(cache=False)
def gaussian_pairs(p, r, q, centers, scale, weighted):
    # Compiled low-level kernel: normalized finite coordinates, nonempty
    # normalized centers and a positive finite scale are preconditions.
    u = np.empty(len(p), dtype=np.float64)
    v = np.empty(len(p), dtype=np.float64)
    for i in range(len(p)):
        total = 0.
        for j in range(len(centers)):
            a = p[i]-centers[j,0]
            b = r[i]-centers[j,1]
            c = q[i]-centers[j,2]
            total += math.exp(-(a*a+b*b+c*c)*scale)
        weight = total / len(centers)
        if weighted:
            # A Gaussian-weighted mean of normalized price; denominator varies.
            u[i], v[i] = p[i]*weight, weight
        else:
            u[i], v[i] = weight, 1.
    return u, v


@njit(cache=False)
def _neumaier(values):
    total, correction = 0., 0.
    for value in values:
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
    return total + correction


@njit(cache=False)
def prefix_moments(u, v):
    # No fastmath/reassociation. Positive compensated first moments determine
    # centering; the other predictable moments only choose the next stake.
    return (_neumaier(u), _neumaier(v), _neumaier(u*u),
            _neumaier(v*v), _neumaier(u*v))


def prefix_from_pairs(u, v):
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if u.ndim != 1 or u.shape != v.shape:
        raise ValueError('matching one-dimensional pair vectors required')
    if (np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)) or
            np.any(u < 0) or np.any(u > 1) or np.any(v < 0) or np.any(v > 1)):
        raise ValueError('bounded finite pairs required')
    return Prefix(len(u), *map(float, prefix_moments(u, v)))


def evaluate(records, spec, *, validate=True):
    """Bounded pairs for normalized coordinates; validate public calls by default.

    The controlled driver can omit repeated domain scans only after its source
    builder has validated every coordinate and frozen the function parameters.
    """
    if spec['function'] not in ('affinity', 'weighted_price'):
        raise ValueError('unknown function')
    centers = np.asarray(spec['centers'], dtype=np.float64)
    bandwidth = float(spec['bandwidth'])
    with np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        scale = float(np.divide(.5,np.float64(bandwidth)*np.float64(bandwidth)))
    if validate:
        if (centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0
                or np.any(~np.isfinite(centers)) or np.any(centers < 0) or np.any(centers > 1)):
            raise ValueError('nonempty normalized finite three-dimensional centers required')
        if not math.isfinite(bandwidth) or bandwidth <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError('positive bandwidth with a finite inverse squared scale required')
        for name in ('p','r','q'):
            x = records[name]
            if x.ndim != 1 or np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x > 1):
                raise ValueError('normalized finite record coordinates required')
    return gaussian_pairs(records['p'], records['r'], records['q'],
                          centers,scale, spec['function'] == 'weighted_price')


@njit(cache=False)
def fused_first_moments(p, r, q, centers, scales, weighted):
    """One row pass on prevalidated normalized arrays/frozen function parameters."""
    K = len(scales)
    su, sv = np.zeros(K), np.zeros(K)
    cu, cv = np.zeros(K), np.zeros(K)
    for i in range(len(p)):
        for k in range(K):
            total = 0.
            for j in range(centers.shape[1]):
                a = p[i]-centers[k,j,0]
                b = r[i]-centers[k,j,1]
                c = q[i]-centers[k,j,2]
                total += math.exp(-(a*a+b*b+c*c)*scales[k])
            weight = total / centers.shape[1]
            u, v = (p[i]*weight, weight) if weighted else (weight, 1.)
            updated = su[k] + u
            cu[k] += ((su[k]-updated)+u if abs(su[k]) >= abs(u)
                      else (u-updated)+su[k])
            su[k] = updated
            updated = sv[k] + v
            cv[k] += ((sv[k]-updated)+v if abs(sv[k]) >= abs(v)
                      else (v-updated)+sv[k])
            sv[k] = updated
    return su+cu, sv+cv


def select_specs(context, K=8, centers=8, function='affinity'):
    """Freeze every function from the same initial visible context."""
    if len(context) == 0 or centers < 1 or K < 1:
        raise ValueError('nonempty context and positive query/center counts required')
    if function not in ('affinity', 'weighted_price'):
        raise ValueError('unknown function')
    xyz = np.column_stack([context[name] for name in ('p', 'r', 'q')])
    specs = []
    for k in range(K):
        order = np.argsort(xyz[:,k % 3], kind='stable')
        # Rotate the chosen cohort within the public initial context, keeping
        # functions distinct even when the same attribute selects the centers.
        ranks = np.rint(np.linspace(.05,.95,centers)*(len(xyz)-1)).astype(np.int64)
        selected = order[(ranks + (k//3)*max(1,len(xyz)//(3*centers))) % len(xyz)]
        specs.append(dict(query=k, function=function, bandwidth=.18+.02*k,
                          centers=xyz[selected].tolist()))
    return specs


def warmup():
    from anq.file_store import RECORD_DTYPE
    rec = np.zeros(5, dtype=RECORD_DTYPE)
    rec['p'] = np.linspace(0,1,5)
    rec['r'], rec['q'] = rec['p'], rec['p']
    for function in ('affinity','weighted_price'):
        spec = select_specs(rec, K=1, function=function)[0]
        u, v = evaluate(rec, spec)
        prefix_from_pairs(u, v)
        fused_first_moments(rec['p'], rec['r'], rec['q'],
                            np.asarray([spec['centers']]),
                            np.asarray([1/(2*spec['bandwidth']**2)]),
                            function == 'weighted_price')
