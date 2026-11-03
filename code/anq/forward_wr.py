"""A minimal replacement-mode renewal on the same forward source.

This is the important simpler alternative to query-local exclusion pools.
It redirects an unexposed uniform physical address to the next source row.
"""
import numpy as np
from numba import njit
from .scan_pool import SparsePoolScan


@njit(cache=False)
def _wr_batch(order,physical,rng,out,fresh):
    for j in range(len(out)):
        position=rng.integers(0,len(order))
        new=position>=physical
        if new:
            position=physical;physical+=1
        out[j]=order[position];fresh[j]=new
    return physical


class ForwardReplacementScan(SparsePoolScan):
    def register(self):
        q=self._next;self._next+=1
        self._queries[q]=dict(remaining=self.size,
            rng=np.random.default_rng(int(self._rng.integers(0,2**63-1))))
        self.events.append(dict(event='register_wr',query=q,physical=self._physical,
                                registration_row_evaluations=0))
        return q

    def draw(self,q,n=1):
        if q not in self._queries:raise RuntimeError('unknown/stopped query')
        if int(n)!=n or n<0 or (n and not self.size):raise ValueError('invalid WR draw count')
        state=self._queries[q];out=np.empty(int(n),dtype=self._order.dtype)
        fresh=np.empty(int(n),dtype=bool)
        if not n:return out,fresh
        self._physical=_wr_batch(self._order,self._physical,state['rng'],out,fresh)
        state['remaining']-=int(n)  # bookkeeping only: can be negative under WR
        self.events.append(dict(event='draw_wr',query=q,count=int(n),physical=self._physical,
                                new_physical=int(fresh.sum()),query_used=self.size-state['remaining']))
        return out,fresh

    def query_entries(self,q):
        if q not in self._queries:raise RuntimeError('unknown/stopped query')
        return 0


def warmup():
    for dtype in [np.int32,np.int64]:
        s=ForwardReplacementScan(np.arange(4,dtype=dtype),1)
        s.expose(2);q=s.register();s.draw(q,8);s.stop(q)
