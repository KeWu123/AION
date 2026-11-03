"""A strengthened established finite-family comparator, not a new CS method.

At one fixed prefix size, intersect Hoeffding--Serfling and empirical
Bernstein--Serfling mean bounds with half the session budget each. Primary:
Bardenet & Maillard (2015), Corollary2.5 / Theorem4.3. The empirical bound's
one-sided failure is5delta, accounted for below. The query family is exactly
the four threshold/interval templates of exp_interleaved_v3, with at most
4(N+1)^2 functions and two bounded means per function.
"""
import numpy as np

class JointFamilySummary:
    def __init__(self,data,ids,N,alpha=.05):
        self.h=len(ids);self.N=int(N);self.alpha=float(alpha);self.tables={}
        if self.N!=N or not 0<=self.h<=self.N or self.N<1 or not 0<self.alpha<1:
            raise ValueError('positive N/level and fixed prefix size in [0,N] required')
        if len(data)!=3 or any(len(a)!=self.N for a in data):
            raise ValueError('three full population columns of declared length required')
        self.rows=3*self.h
        raw=[a[ids] for a in data]
        if any(np.ma.isMaskedArray(a) for a in raw):raise ValueError('resolve NULL rows explicitly')
        # Accumulation guards below are for float64, including callers that
        # supplied float32 or integer columns. Cast only exposed rows.
        p,r,q=(np.asarray(a,dtype=np.float64) for a in raw)
        if any(not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in (p,r,q)):
            raise ValueError('finite normalized exposed columns in [0,1] required')
        self.total_r=float(np.sum(r));self.total_r2=float(np.sum(r*r))
        for kind,key,weight in [(0,r,r),(1,p,q)]:
            order=np.argsort(key,kind='stable');w=weight[order]
            self.tables[kind]=(key[order],np.r_[0.,np.cumsum(w)],np.r_[0.,np.cumsum(w*w)])
        bins=np.rint(q*100).astype(int)
        if not np.array_equal(q,bins/100):
            raise ValueError('this comparator uses the declared 101-point quantity grid')
        self.tables[2]=(np.arange(101)/100,
            np.r_[0.,np.cumsum(np.bincount(bins,weights=r,minlength=101))],
            np.r_[0.,np.cumsum(np.bincount(bins,weights=r*r,minlength=101))])
        self.retained_array_bytes=sum(x.nbytes for table in self.tables.values() for x in table)

    def moments(self,spec):
        if not self.h:raise ValueError('no empirical moments at an empty prefix')
        kind=1 if spec['kind']==3 else spec['kind'];key,sums,squares=self.tables[kind]
        a=int(np.searchsorted(key,spec['low'],'left'));b=int(np.searchsorted(key,spec['high'],'right'))
        mu=(sums[b]-sums[a])/self.h;mu2=(squares[b]-squares[a])/self.h
        if kind in (0,2):mv=self.total_r/self.h;mv2=self.total_r2/self.h
        else:mv=(b-a)/self.h;mv2=mv
        # Enclose variance errors from positive sums, subtraction and squaring;
        # eps is twice IEEE unit roundoff. This is a conservative guard for
        # these evaluated arrays, not a universal floating-point proof.
        eps=np.finfo(float).eps;pad=64*eps*self.h/(1-self.h*eps)
        vu=min(.25,max(0.,mu2-mu*mu)+pad);vv=min(.25,max(0.,mv2-mv*mv)+pad)
        return float(mu),float(mv),float(vu),float(vv)

    def radii(self,vu,vv):
        h=self.h;N=self.N
        if not h:return np.inf,np.inf
        if h==N:return 0.,0.
        rho=1-(h-1)/N if h<=N/2 else (1-h/N)*(1+1/h)
        # Eight scalar means per (N+1)^2, two tails each. Split alpha equally
        # between the two bound types. Empirical Bernstein has5delta per tail.
        lh=np.log(32*(N+1.)**2/self.alpha)
        le=np.log(160*(N+1.)**2/self.alpha)
        eh=np.sqrt(rho*lh/(2*h));kappa=7/3+3/np.sqrt(2)
        eu=np.sqrt(2*vu*rho*le/h)+kappa*le/h
        ev=np.sqrt(2*vv*rho*le/h)+kappa*le/h
        return float(min(eh,eu)),float(min(eh,ev))

    def interval(self,spec):
        if not self.h:return 0.,1.
        mu,mv,vu,vv=self.moments(spec);eu,ev=self.radii(vu,vv)
        eps=np.finfo(float).eps;pad=32*eps*self.h/(1-self.h*eps)
        eu+=pad;ev+=pad
        if self.h==self.N and mv==0:raise ValueError('undefined zero-denominator census ratio')
        lo=max(0.,(mu-eu)/(mv+ev));hi=min(1.,(mu+eu)/(mv-ev)) if mv>ev else 1.
        return float(np.nextafter(lo,-np.inf)) if lo else 0.,float(np.nextafter(hi,np.inf))
