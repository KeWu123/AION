"""Guard-free hedged ratio certificates, including adaptive prefix rebasing.

The population is fixed, pairs lie in [0,1]^2, and sum(v)>0. A prefix may
select the query, but earns no capital: its totals enter only the remaining
population centering. `prepare` never inspects future values to set a stake.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Prefix:
    n: int = 0
    su: float = 0.0
    sv: float = 0.0
    suu: float = 0.0
    svv: float = 0.0
    suv: float = 0.0

    @classmethod
    def from_rows(cls, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        if u.shape != v.shape or u.ndim != 1:
            raise ValueError('prefix vectors must have the same one-dimensional shape')
        if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)) or np.any((u < 0) | (u > 1) | (v < 0) | (v > 1)):
            raise ValueError('bounded finite pairs required')
        return cls(len(u), math.fsum(u), math.fsum(v), float(u @ u),
                   float(v @ v), float(u @ v))


class RatioTrace:
    """Prepared predictable state for one registered query and unseen suffix.

    `population` is None for uniform WR draws; otherwise the known population
    size. Prefix totals MUST cover every row exposed before registration.
    Alpha is fixed at registration; a zero level never rejects.
    """
    def __init__(self, u, v, population=None, prefix=Prefix(), alpha=0.05,
                 cap=0.5, variance_floor=1e-6):
        self.u, self.v = np.asarray(u, float), np.asarray(v, float)
        if self.u.ndim != 1 or self.u.shape != self.v.shape:
            raise ValueError('matching one-dimensional suffix vectors required')
        if np.any(~np.isfinite(self.u)) or np.any(~np.isfinite(self.v)) or np.any((self.u < 0) | (self.u > 1) | (self.v < 0) | (self.v > 1)):
            raise ValueError('bounded finite pairs required')
        if not 0 <= alpha < 1 or not 0 < cap <= .9 or variance_floor <= 0 or not np.isfinite(variance_floor):
            raise ValueError('invalid alpha, cap, or variance floor')
        if int(prefix.n) != prefix.n or prefix.n < 0 or any(not np.isfinite(x) or x < 0 for x in (prefix.su,prefix.sv,prefix.suu,prefix.svv,prefix.suv)):
            raise ValueError('invalid prefix state')
        if population is not None and (int(population) != population or population < prefix.n + len(self.u)):
            raise ValueError('suffix exceeds remaining population')
        self.prefix, self.population, self.alpha, self.cap = prefix, population, alpha, cap
        if not len(self.u):
            self.a = self.b = self.scale = np.empty(0)
            self.remaining = None if population is None else np.empty(0,dtype=int)
            self._error_a = self._error_b = self._log_scale_prefix = np.empty(0)
            return
        self.a = prefix.su + np.r_[0.0, np.cumsum(self.u)[:-1]]
        self.b = prefix.sv + np.r_[0.0, np.cumsum(self.v)[:-1]]
        count = np.maximum(prefix.n + np.arange(len(self.u)), 1)
        a2 = prefix.suu + np.r_[0.0, np.cumsum(self.u**2)[:-1]]
        b2 = prefix.svv + np.r_[0.0, np.cumsum(self.v**2)[:-1]]
        ab = prefix.suv + np.r_[0.0, np.cumsum(self.u*self.v)[:-1]]
        # A tiny denominator can make an otherwise harmless stake estimate
        # overflow. A bounded, arbitrary predictor is still predictable; this
        # changes stakes only, never the candidate's exact conditional center.
        guess = np.divide(self.a, self.b, out=np.ones_like(self.a), where=self.b >= 1e-12)
        guess = np.minimum(guess, 1e6)
        var = (a2 - 2*guess*ab + guess**2*b2)/count - ((self.a-guess*self.b)/count)**2
        var = np.maximum(var, variance_floor)
        # Candidate-free, past-only proposed magnitude. The floor changes power,
        # not centering or the martingale proof. Query-local clock starts at one.
        self.scale = np.sqrt(2*np.log(2/max(alpha, 1e-300)) /
                             (np.maximum(np.arange(1, len(self.u)+1), 10)*var))
        self.remaining = None if population is None else population-prefix.n-np.arange(len(self.u))
        self._prepare_error_budget()

    def _prepare_error_budget(self):
        """Bound upstream centering error; preserve raw diagnostic paths.

        Nonnegative-sum gamma bounds use epsilon, twice unit roundoff. Prefix
        storage/addition and underflow receive extra padding. Each directional
        log factor is globally Lipschitz in its center, with constant at most
        scale/min(cap,1-cap). The resulting bound is affine in theta, so cached
        reports need O(1) work to retrieve it. Local cap/log arithmetic retains
        a separate engineering guard; this is not a full IEEE-754 proof.
        """
        eps=np.finfo(float).eps; tiny=np.nextafter(0.,1.)
        j=np.arange(len(self.u),dtype=float)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            gamma=(j+2)*eps/(1-(j+2)*eps)
            # Convert a textbook error relative to exact sum into a bound
            # relative to its nonnegative stored approximation.
            relative=gamma/(1-gamma)+8*eps
            inflate=1+relative
            ea=relative*self.a+8*math.ulp(self.prefix.su)+8*(j+1)*tiny
            eb=relative*self.b+8*math.ulp(self.prefix.sv)+8*(j+1)*tiny
            lip=self.scale/min(self.cap,1-self.cap)
            if self.remaining is None:
                self._error_a=np.zeros(len(self.u));self._error_b=np.zeros(len(self.u))
            else:
                # theta*b multiplication, subtraction from a and division by
                # remaining add <=16 eps*(a+theta*b)/remaining, plus underflow.
                self._error_a=np.nextafter(np.cumsum(lip*((ea+16*eps*self.a)/self.remaining+8*tiny))*inflate,np.inf)
                self._error_b=np.nextafter(np.cumsum(lip*(eb+16*eps*self.b)/self.remaining)*inflate,np.inf)
            self._log_scale_prefix=np.nextafter(np.cumsum(np.log1p(self.scale))*inflate,np.inf)

    def log_error_bound(self, theta, n=None):
        """Cumulative centering bound plus a local floating-operation guard.

        A finite lower capital can certify a rejected endpoint. This numerical
        lower approximation need not itself be monotone in the candidate.
        """
        theta=np.asarray(theta,float)
        if np.any(~np.isfinite(theta)) or np.any(theta < 0):
            raise ValueError('finite nonnegative ratio candidate required')
        n=len(self.u) if n is None else int(n)
        if not 0 <= n <= len(self.u):
            raise ValueError('invalid checkpoint')
        if not n:
            return float(0.) if theta.ndim==0 else np.zeros_like(theta)
        with np.errstate(over='ignore',invalid='ignore'):
            center=self._error_a[n-1]+theta*self._error_b[n-1]
            # |log factor| <= -log(1-cap)+log(1+scale)+log(1+theta).
            magnitude=n*(-math.log1p(-self.cap)+np.log1p(theta))+self._log_scale_prefix[n-1]
            result=(1+16*np.finfo(float).eps)*(center+128*np.finfo(float).eps*(1+n+magnitude))
            result=np.where(np.isfinite(result),np.nextafter(result,np.inf),np.inf)
        return float(result) if theta.ndim==0 else result

    def feasible(self, n=None):
        n = len(self.u) if n is None else int(n)
        if not 0 <= n <= len(self.u):
            raise ValueError('invalid checkpoint')
        if self.population is None:
            return 0.0, np.inf
        a = math.fsum([self.prefix.su, *self.u[:n]])
        b = math.fsum([self.prefix.sv, *self.v[:n]])
        # Outer arithmetic enclosure. Four ulps cover the two compensated
        # aggregations and the stored prefix rounding; use outward division.
        ea = 4*(math.ulp(a)+math.ulp(self.prefix.su))
        eb = 4*(math.ulp(b)+math.ulp(self.prefix.sv))
        al, ah = max(0., a-ea), a+ea
        bl, bh = max(0., b-eb), b+eb
        r = self.population-self.prefix.n-n
        if r == 0:
            if b == 0:
                raise ValueError('ratio undefined: population denominator is zero')
            return max(0., np.nextafter(al/bh, -np.inf)), (np.nextafter(ah/bl, np.inf) if bl > 0 else np.inf)
        return max(0., np.nextafter(al/(bh+r), -np.inf)), (np.nextafter((ah+r)/bl, np.inf) if bl > 0 else np.inf)

    def log_paths(self, theta, n=None):
        if not np.isfinite(theta) or theta < 0:
            raise ValueError('finite nonnegative ratio candidate required')
        n = len(self.u) if n is None else int(n)
        if not 0 <= n <= len(self.u):
            raise ValueError('invalid checkpoint')
        m = np.zeros(n) if self.remaining is None else -(self.a[:n]-theta*self.b[:n])/self.remaining[:n]
        d = self.u[:n]-theta*self.v[:n]-m
        # No drift clipping, and no division by a zero support endpoint.
        den_plus, den_minus = theta+m, 1-m
        bound_plus = np.divide(self.cap, den_plus, out=np.full(n, np.inf), where=den_plus > 0)
        bound_minus = np.divide(self.cap, den_minus, out=np.full(n, np.inf), where=den_minus > 0)
        lp = np.minimum(self.scale[:n], bound_plus)
        lm = np.minimum(self.scale[:n], bound_minus)
        fp, fm = 1+lp*d, 1-lm*d
        # Infeasible candidates may have arbitrary signs; callers invert only
        # inside deterministic feasibility. Never silently repair a bad factor.
        if np.any(~np.isfinite(fp)) or np.any(~np.isfinite(fm)) or np.any(fp <= 0) or np.any(fm <= 0):
            raise ValueError('candidate infeasible or numerical factor failure')
        return np.cumsum(np.log1p(lp*d)), np.cumsum(np.log1p(-lm*d))

    def capital_crossed(self, theta):
        """Diagnostic historical stochastic crossing, not a closed-report verdict."""
        if self.alpha == 0:
            return False
        try:
            p, m = self.log_paths(theta)
        except (ValueError, FloatingPointError):
            return False  # arithmetic failure cannot earn a certificate
        return bool(np.any(np.maximum(p, m) >= np.log(2/self.alpha)))

    def rejects(self, theta, n=None):
        """Reject exactly when the reference is outside the current closed report.

        Earlier crossings remain available via `capital_crossed`; they do not
        override census, reacceptance, or a conservative inversion fallback.
        """
        if not np.isfinite(theta) or theta < 0:
            raise ValueError('finite nonnegative reference required')
        lower, upper = self.interval(n)
        return bool(theta < lower or theta > upper)

    def interval(self, n=None, iterations=48):
        """Conservative outer endpoints of the directional accepted set.

        The returned closed hull includes equality and never rounds inward.
        Center-error-adjusted capital must certify each narrowing endpoint;
        uncertain evaluations may reduce precision. Detected empty brackets
        retain the deterministic hull. Full enumeration returns that hull.
        Unbounded
        upper endpoints remain infinity if no finite rejection is located.
        """
        n = len(self.u) if n is None else int(n)
        lo, hi = self.feasible(n)
        if n == 0 or lo == hi or self.alpha == 0 or (self.population is not None and self.prefix.n+n == self.population):
            return lo, hi
        threshold = np.log(2/self.alpha)
        arithmetic_failed = [False]
        def f(th, arm):
            try:
                p, m = self.log_paths(th, n)
            except (ValueError, FloatingPointError):
                # An unresolved evaluation must retain candidates. NaN makes
                # every comparison false, returning an outer feasible bound.
                arithmetic_failed[0] = True
                return np.nan
            error=self.log_error_bound(th,n)
            if not np.isfinite(error):
                arithmetic_failed[0]=True
                return np.nan
            return (p if arm == 0 else m)[-1] - threshold-error if n else -threshold
        a, b = lo, hi
        if not np.isfinite(b):
            b = max(1.0, 2*a)
            for _ in range(100):
                if f(b, 1) >= 0:
                    break
                b *= 2
            else:
                b = np.inf
        # K+ decreases: lower accepted boundary. K- increases: upper one.
        lower = lo
        if f(lo, 0) >= 0:
            right = b if np.isfinite(b) else max(1.0, 2*lo)
            while not np.isfinite(hi) and f(right, 0) >= 0 and right < 1e100:
                right *= 2
            if f(right, 0) >= 0:
                return lo, hi  # unresolved: conservative fallback
            left = lo
            for _ in range(iterations):
                mid = (left+right)/2
                if f(mid, 0) >= 0:
                    left = mid
                else:
                    right = mid
            lower = left
        upper = hi
        if np.isfinite(b) and f(b, 1) >= 0:
            left, right = lo, b
            if f(left, 1) >= 0:
                return lo, hi  # empty stochastic set: retain deterministic hull
            for _ in range(iterations):
                mid = (left+right)/2
                if f(mid, 1) >= 0:
                    right = mid
                else:
                    left = mid
            upper = right
        return (lower, upper) if lower <= upper and not arithmetic_failed[0] else (lo, hi)
