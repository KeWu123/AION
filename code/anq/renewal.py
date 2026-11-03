"""Query-local uniform WoR order from a complete cache and a hidden suffix.

This is an ID-level access primitive. Its counters measure row identities, not
disk IO or UDF cost. It assumes the retained cache contains every exposed row.
The supplied physical permutation is trusted to be uniform and hidden from the
policy; Python privacy is not an information-flow sandbox.
"""
import numpy as np


class _LazyPool:
    """Partial Fisher--Yates: O(draws) state, O(1) expected work per draw."""
    def __init__(self, size, rng):
        self.left=int(size)
        self.swaps={}
        self.rng=rng

    def take(self):
        if not self.left:
            raise RuntimeError('empty cached-row pool')
        at=int(self.rng.integers(self.left))
        value=self.swaps.get(at,at)
        self.left-=1
        last=self.swaps.pop(self.left,self.left)
        if at!=self.left:
            self.swaps[at]=last
        return value


class RenewalScan:
    """Serial adaptive registrations; each receives a fresh conditional order.

    `expose` and `draw` reveal IDs only through their return values. Stop before
    exploration or the next registration. Rows newly read during a query enter
    the persistent cache but are excluded from that query's initial-cache pool.
    The old-row pool is reset lazily, without copying/shuffling the full cache.
    """
    def __init__(self, order, seed=20261016):
        order=np.asarray(order)
        if order.ndim!=1 or not np.issubdtype(order.dtype,np.integer):
            raise ValueError('one-dimensional integer physical order required')
        if len(np.unique(order))!=len(order) or np.any(order<0):
            raise ValueError('physical row identities must be unique and nonnegative')
        self._order=order.copy()
        self.size=len(order)
        self._physical=0
        self._cache=[]
        self._rng=np.random.default_rng(seed)
        self._pool=None
        self._used=0
        self._new_remaining=0
        self.queries=0
        self.events=[]

    @property
    def exposed(self):
        return np.asarray(self._cache,dtype=self._order.dtype).copy()

    @property
    def physical_rows(self):
        return self._physical

    def expose(self,n):
        if self._pool is not None:
            raise RuntimeError('stop active query before exploration')
        if int(n)!=n or n<0 or self._physical+n>self.size:
            raise ValueError('invalid physical read count')
        ids=self._order[self._physical:self._physical+int(n)].copy()
        self._physical+=int(n);self._cache.extend(ids.tolist())
        self.events.append(dict(event='expose',count=int(n),physical=self._physical))
        return ids

    def register(self):
        if self._pool is not None:
            raise RuntimeError('stop current query first')
        self._pool=_LazyPool(len(self._cache),self._rng)
        self._new_remaining=self.size-self._physical
        self._used=0;self.queries+=1
        self.events.append(dict(event='register',query=self.queries,
                                cached=self._physical,registration_row_evaluations=0))

    def draw(self,n):
        if self._pool is None:
            raise RuntimeError('register before drawing')
        if int(n)!=n or n<0 or n>self.size-self._used:
            raise ValueError('invalid query-local draw count')
        out=np.empty(int(n),dtype=self._order.dtype);origin=np.empty(int(n),dtype=bool)
        before=self._physical
        for j in range(int(n)):
            remaining=self.size-self._used
            # The integer draw gives exact count-proportional origin choice.
            old=int(self._rng.integers(remaining))<self._pool.left
            if old:
                out[j]=self._cache[self._pool.take()]
            else:
                out[j]=self._order[self._physical]
                self._physical+=1;self._new_remaining-=1
                self._cache.append(int(out[j]))
            origin[j]=not old
            self._used+=1
            assert self._pool.left+self._new_remaining==self.size-self._used
        self.events.append(dict(event='draw',query=self.queries,count=int(n),
                                new_physical=self._physical-before,used=self._used))
        return out,origin

    def stop(self):
        if self._pool is None:
            raise RuntimeError('no active query')
        self.events.append(dict(event='stop',query=self.queries,used=self._used,
                                physical=self._physical))
        self._pool=None


class _Range:
    __slots__=('lo','hi','priority','left','right','mass')
    def __init__(self,lo,hi,rng):
        self.lo,self.hi=int(lo),int(hi)
        self.priority=int(rng.integers(2**63))
        self.left=self.right=None
        self.mass=self.hi-self.lo


def _mass(node):
    return node.mass if node else 0


def _refresh(node):
    node.mass=_mass(node.left)+node.hi-node.lo+_mass(node.right)
    return node


def _merge(a,b):
    """Join ordered interval treaps; all keys in a precede all keys in b."""
    if a is None:return b
    if b is None:return a
    if a.priority<b.priority:
        a.right=_merge(a.right,b);return _refresh(a)
    b.left=_merge(a,b.left);return _refresh(b)


def _split(node,key):
    if node is None:return None,None
    if node.lo<key:
        a,b=_split(node.right,key);node.right=a
        return _refresh(node),b
    a,b=_split(node.left,key);node.left=b
    return a,_refresh(node)


def _insert(root,node):
    if root is None:return node
    if node.priority<root.priority:
        node.left,node.right=_split(root,node.lo)
        return _refresh(node)
    if node.lo<root.lo:root.left=_insert(root.left,node)
    else:root.right=_insert(root.right,node)
    return _refresh(root)


def _erase(root,key):
    if key==root.lo:return _merge(root.left,root.right)
    if key<root.lo:root.left=_erase(root.left,key)
    else:root.right=_erase(root.right,key)
    return _refresh(root)


def _take_rank(root,rank,rng):
    node=root
    while True:
        left=_mass(node.left);length=node.hi-node.lo
        if rank<left:node=node.left
        elif rank>=left+length:rank-=left+length;node=node.right
        else:break
    value=node.lo+rank-left;lo,hi=node.lo,node.hi
    # Erase and reinsert at the top level. Returning freshly prioritized split
    # nodes directly into a recursive child can violate an ancestor's heap order.
    root=_erase(root,lo)
    if value>lo:root=_insert(root,_Range(lo,value,rng))
    if value+1<hi:root=_insert(root,_Range(value+1,hi,rng))
    return value,root


class _GrowingComplement:
    """Unused positions in a growing global prefix, compressed into intervals.

    Expansions are lazy: only a scheduled query sees newly cached intervals.
    Removing a sampled point splits at most one interval. Thus the query has
    O(draws+1) interval nodes, not a bitmap or copied array of the global cache.
    """
    def __init__(self,exposed,rng):
        self.rng=rng;self.seen=int(exposed)
        self.root=_Range(0,exposed,rng) if exposed else None

    def sync(self,exposed):
        if exposed<self.seen:raise RuntimeError('physical cursor cannot move backwards')
        if exposed>self.seen:
            self.root=_merge(self.root,_Range(self.seen,exposed,self.rng))
            self.seen=int(exposed)

    def take(self):
        if not self.root:raise RuntimeError('no unused cached row')
        rank=int(self.rng.integers(self.root.mass))
        value,self.root=_take_rank(self.root,rank,self.rng)
        return value

    def own_new_row(self,position):
        if position!=self.seen:raise RuntimeError('sync cache before physical draw')
        # This newly exposed position is already consumed by this query.
        self.seen+=1

    @property
    def count(self):return _mass(self.root)


class InterleavedRenewalScan:
    """One source, multiple logical queries, serialized atomic sampling events.

    Other queries may advance the physical cursor between this query's draws.
    Each query gets uniform remaining identities conditional on full global
    history. No catch-up evaluations are required. This implements sampling
    only; query functions and statistical monitors are attached by the engine.
    It is deliberately not a claim of thread-safe parallel execution.
    """
    def __init__(self,order,seed=20261016):
        order=np.asarray(order)
        if order.ndim!=1 or not np.issubdtype(order.dtype,np.integer):
            raise ValueError('one-dimensional integer physical order required')
        if len(np.unique(order))!=len(order) or np.any(order<0):
            raise ValueError('unique nonnegative physical identities required')
        self._order=order.copy();self.size=len(order);self._physical=0
        self._rng=np.random.default_rng(seed)
        self._queries={};self._next=0;self.events=[]

    @property
    def physical_rows(self):return self._physical

    @property
    def exposed(self):return self._order[:self._physical].copy()

    def expose(self,n):
        if int(n)!=n or n<0 or self._physical+n>self.size:
            raise ValueError('invalid physical count')
        ids=self._order[self._physical:self._physical+int(n)].copy()
        self._physical+=int(n)
        self.events.append(dict(event='external_exposure',count=int(n),physical=self._physical))
        return ids

    def register(self):
        q=self._next;self._next+=1
        child=np.random.default_rng(int(self._rng.integers(2**63)))
        self._queries[q]=dict(used=0,pool=_GrowingComplement(self._physical,child))
        self.events.append(dict(event='register',query=q,physical=self._physical,
                                registration_row_evaluations=0))
        return q

    def draw(self,q,n=1):
        if q not in self._queries:raise RuntimeError('unknown/stopped query')
        state=self._queries[q]
        if int(n)!=n or n<0 or n>self.size-state['used']:
            raise ValueError('invalid query draw count')
        if not n:
            # A no-op must not accumulate lazy intervals as other queries read.
            return np.empty(0,dtype=self._order.dtype),np.empty(0,dtype=bool)
        pool=state['pool'];pool.sync(self._physical)
        out=np.empty(int(n),dtype=self._order.dtype);origin=np.empty(int(n),dtype=bool)
        for j in range(int(n)):
            unused=self.size-state['used']
            assert pool.count==self._physical-state['used']
            old=int(pool.rng.integers(unused))<pool.count
            if old:position=pool.take()
            else:
                position=self._physical;pool.own_new_row(position);self._physical+=1
            out[j]=self._order[position];origin[j]=not old;state['used']+=1
        self.events.append(dict(event='draw',query=q,count=int(n),
                                new_physical=int(origin.sum()),physical=self._physical,
                                query_used=state['used']))
        return out,origin

    def stop(self,q):
        if q not in self._queries:raise RuntimeError('unknown/stopped query')
        state=self._queries.pop(q)
        self.events.append(dict(event='stop',query=q,used=state['used'],physical=self._physical))
