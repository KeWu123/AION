"""Expected-O(1) query-local renewal over one hidden physical order.

Each query's implicit array contains ALL unused physical positions. A sampled
unexposed position is redirected to the next physical position, and the actual
returned position is deleted. Sparse forward/inverse maps implement arbitrary
swap-delete without allocating or initializing an N-element array per query.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict


@njit(cache=False)
def _delete(forward, inverse, remaining, value):
    slot = inverse[value] if value in inverse else value
    last = remaining-1
    replacement = forward[last] if last in forward else last
    if last in forward:
        del forward[last]
    if value in inverse:
        del inverse[value]
    if slot != last:
        if replacement == slot:
            if slot in forward: del forward[slot]
        else:
            forward[slot] = replacement
        if replacement == slot:
            if replacement in inverse: del inverse[replacement]
        else:
            inverse[replacement] = slot


@njit(cache=False)
def _batch(order, physical, remaining, forward, inverse, rng, out, fresh):
    for j in range(len(out)):
        slot = rng.integers(0, remaining)
        position = forward[slot] if slot in forward else slot
        new = position >= physical
        if new:
            position = physical
            physical += 1
        out[j] = order[position]
        fresh[j] = new
        _delete(forward, inverse, remaining, position)
        remaining -= 1
    return physical, remaining


@njit(cache=False)
def indexed_batch(order, remaining, forward, rng, out):
    """Unrestricted indexed SRS comparator, sparse Fisher--Yates only."""
    for j in range(len(out)):
        slot=rng.integers(0,remaining)
        position=forward[slot] if slot in forward else slot
        remaining-=1
        replacement=forward[remaining] if remaining in forward else remaining
        if remaining in forward:del forward[remaining]
        if slot!=remaining:forward[slot]=replacement
        out[j]=order[position]
    return remaining


class SparsePoolScan:
    """Serialized logical queries, complete retained prefix, private RNG.

    O(1) query registration, expected O(1) sampling work, O(j) sparse query
    state after j draws (hash-table expectation). Counters are row identities,
    not measured I/O. The order must be a hidden uniform source permutation.
    """
    def __init__(self, order, seed=None):
        order=np.asarray(order)
        if order.ndim!=1 or not np.issubdtype(order.dtype,np.integer):
            raise ValueError('one-dimensional integer physical order required')
        if len(np.unique(order))!=len(order) or np.any(order<0):
            raise ValueError('unique nonnegative physical identities required')
        # Numba accepts native-endian integers. Normalize values before any
        # query state is created, preserving signedness and integer width.
        self._order=order.astype(order.dtype.newbyteorder('='),copy=True)
        self.size=len(order);self._physical=0
        self._rng=np.random.default_rng(seed);self._queries={};self._next=0;self.events=[]

    @property
    def physical_rows(self):return self._physical

    @property
    def exposed(self):return self._order[:self._physical].copy()

    def expose(self,n):
        if int(n)!=n or n<0 or self._physical+n>self.size:
            raise ValueError('invalid physical count')
        ids=self._order[self._physical:self._physical+int(n)].copy()
        self._physical+=int(n)
        self.events.append(dict(event='expose',count=int(n),physical=self._physical))
        return ids

    def register(self):
        q=self._next;self._next+=1
        self._queries[q]=dict(remaining=self.size,
            forward=Dict.empty(types.int64,types.int64),inverse=Dict.empty(types.int64,types.int64),
            rng=np.random.default_rng(int(self._rng.integers(0,2**63-1))))
        self.events.append(dict(event='register',query=q,physical=self._physical,
                                registration_row_evaluations=0))
        return q

    def draw(self,q,n=1):
        if q not in self._queries:raise RuntimeError('unknown/stopped query')
        state=self._queries[q]
        if int(n)!=n or n<0 or n>state['remaining']:
            raise ValueError('invalid query draw count')
        out=np.empty(int(n),dtype=self._order.dtype);fresh=np.empty(int(n),dtype=bool)
        if not n:return out,fresh
        self._physical,state['remaining']=_batch(self._order,self._physical,state['remaining'],
            state['forward'],state['inverse'],state['rng'],out,fresh)
        self.events.append(dict(event='draw',query=q,count=int(n),physical=self._physical,
                                new_physical=int(fresh.sum()),query_used=self.size-state['remaining']))
        return out,fresh

    def query_entries(self,q):
        s=self._queries[q]
        return len(s['forward'])+len(s['inverse'])

    def stop(self,q):
        if q not in self._queries:raise RuntimeError('unknown/stopped query')
        state=self._queries.pop(q)
        self.events.append(dict(event='stop',query=q,used=self.size-state['remaining'],physical=self._physical))


def warmup():
    s=SparsePoolScan(np.arange(4));s.expose(2);q=s.register();s.draw(q,4);s.stop(q)
    d=Dict.empty(types.int64,types.int64)
    indexed_batch(np.arange(4),4,d,np.random.default_rng(1),np.empty(4,dtype=np.int32))
