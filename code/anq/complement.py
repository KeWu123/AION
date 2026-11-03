"""Query-local complement ranks over a shared, hidden physical permutation.

The treap contains USED physical positions only. An absent subtree represents
all unused positions in its key interval, so no global-prefix catch-up is needed.
Numba compiles the same bounded-ID kernel; preparation is charged separately.
Logical events are serialized. This module does not implement disk I/O.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _size(nodes, root):
    return 0 if root < 0 else nodes[root, 3]


@njit(cache=True)
def _refresh(nodes, root):
    nodes[root, 3] = 1 + _size(nodes, nodes[root, 1]) + _size(nodes, nodes[root, 2])


@njit(cache=True)
def _insert(nodes, priorities, root, new):
    if root < 0:
        return new
    if nodes[new, 0] < nodes[root, 0]:
        child = _insert(nodes, priorities, nodes[root, 1], new)
        nodes[root, 1] = child
        if priorities[child] < priorities[root]:
            nodes[root, 1] = nodes[child, 2]
            nodes[child, 2] = root
            _refresh(nodes, root); _refresh(nodes, child)
            return child
    else:
        child = _insert(nodes, priorities, nodes[root, 2], new)
        nodes[root, 2] = child
        if priorities[child] < priorities[root]:
            nodes[root, 2] = nodes[child, 1]
            nodes[child, 1] = root
            _refresh(nodes, root); _refresh(nodes, child)
            return child
    _refresh(nodes, root)
    return root


@njit(cache=True)
def _complement_select(nodes, root, rank):
    """Zero-based rank among nonnegative positions absent from the tree."""
    lower = 0
    while root >= 0:
        left = nodes[root, 1]
        available_left = nodes[root, 0] - lower - _size(nodes, left)
        if rank < available_left:
            root = left
        else:
            rank -= available_left
            lower = nodes[root, 0] + 1
            root = nodes[root, 2]
    return lower + rank


@njit(cache=True)
def _draw(order, physical, used, root, nodes, priorities, rng, out, fresh):
    for offset in range(len(out)):
        j = used + offset
        cached = physical - j
        old = rng.integers(0, len(order) - j) < cached
        if old:
            position = _complement_select(nodes, root, rng.integers(0, cached))
        else:
            position = physical
            physical += 1
        out[offset] = order[position]
        fresh[offset] = not old
        nodes[j, 0] = position
        nodes[j, 1] = nodes[j, 2] = -1
        nodes[j, 3] = 1
        priorities[j] = rng.integers(0, 2**63-1)
        root = _insert(nodes, priorities, root, j)
    return physical, root


class ComplementRenewalScan:
    """One complete cache, O(1) registration, O(j) per-query used-row state.

    Every row remains available in the underlying engine. Each query's private
    treap indexes its own used positions, never the entire shared cache. Uniform
    source order and independent private coins are statistical assumptions;
    disclosed experiment seeds give reproducibility, not cryptographic secrecy.
    """
    def __init__(self, order, seed=20261016):
        order = np.asarray(order)
        if order.ndim != 1 or not np.issubdtype(order.dtype, np.integer):
            raise ValueError('one-dimensional integer physical order required')
        if len(np.unique(order)) != len(order) or np.any(order < 0):
            raise ValueError('unique nonnegative physical identities required')
        self._order = order.astype(order.dtype.newbyteorder('='),copy=True)
        self.size = len(order)
        self._physical = 0
        self._rng = np.random.default_rng(seed)
        self._queries = {}
        self._next = 0
        self.events = []

    @property
    def physical_rows(self):
        return self._physical

    @property
    def exposed(self):
        return self._order[:self._physical].copy()

    def expose(self, n):
        if int(n) != n or n < 0 or self._physical + n > self.size:
            raise ValueError('invalid physical count')
        ids = self._order[self._physical:self._physical+int(n)].copy()
        self._physical += int(n)
        self.events.append(dict(event='expose', count=int(n), physical=self._physical))
        return ids

    def register(self):
        q = self._next
        self._next += 1
        self._queries[q] = dict(used=0, root=-1,
            nodes=np.empty((16,4), dtype=np.int64), priorities=np.empty(16,dtype=np.int64),
            rng=np.random.default_rng(int(self._rng.integers(0,2**63-1))))
        self.events.append(dict(event='register', query=q, physical=self._physical,
                                registration_row_evaluations=0))
        return q

    def draw(self, q, n=1):
        if q not in self._queries:
            raise RuntimeError('unknown/stopped query')
        state = self._queries[q]
        if int(n) != n or n < 0 or n > self.size-state['used']:
            raise ValueError('invalid query draw count')
        n = int(n)
        out = np.empty(n,dtype=self._order.dtype)
        fresh = np.empty(n,dtype=bool)
        if not n:
            return out, fresh
        need = state['used'] + n
        if need > len(state['priorities']):
            cap = max(need, 2*len(state['priorities']))
            nodes = np.empty((cap,4),dtype=np.int64)
            priorities = np.empty(cap,dtype=np.int64)
            nodes[:state['used']] = state['nodes'][:state['used']]
            priorities[:state['used']] = state['priorities'][:state['used']]
            state['nodes'],state['priorities'] = nodes,priorities
        self._physical,state['root'] = _draw(self._order,self._physical,state['used'],
            state['root'],state['nodes'],state['priorities'],state['rng'],out,fresh)
        state['used'] += n
        self.events.append(dict(event='draw',query=q,count=n,new_physical=int(fresh.sum()),
                                physical=self._physical,query_used=state['used']))
        return out,fresh

    def query_state_bytes(self,q):
        s = self._queries[q]
        return s['nodes'].nbytes+s['priorities'].nbytes

    def stop(self,q):
        if q not in self._queries:
            raise RuntimeError('unknown/stopped query')
        state=self._queries.pop(q)
        self.events.append(dict(event='stop',query=q,used=state['used'],physical=self._physical))


def warmup():
    s=ComplementRenewalScan(np.arange(4));s.expose(2)
    q=s.register();s.draw(q,4);s.stop(q)
