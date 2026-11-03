"""Candidate atomic-epoch renewal; not part of the frozen formal studies.

An epoch commits its participating queries before any new record is returned.
Each query gets one WoR observation. Marginals hold given the epoch's past;
observations within the epoch can be dependent. No within-epoch adaptivity is
offered by this API. Newly consumed source records are counted once, separately
from how many queries consume the shared record.
"""
import numpy as np
from numba import njit
from .scan_pool import SparsePoolScan, _delete


@njit(cache=False)
def _propose(forward, remaining, rng):
    slot = rng.integers(0, remaining)
    return forward[slot] if slot in forward else slot


@njit(cache=False)
def _cached(forward, remaining, physical_before, rng):
    proposals = 0
    while True:
        position = _propose(forward, remaining, rng)
        proposals += 1
        if position < physical_before:
            return position, proposals


@njit(cache=False)
def _follower(forward, remaining, leader_remaining, source_new,
              physical_before, rng):
    if source_new and rng.integers(0, remaining) < leader_remaining:
        return physical_before, True, 0
    position, proposals = _cached(forward, remaining, physical_before, rng)
    return position, False, proposals


class AtomicPoolScan(SparsePoolScan):
    """Sparse WoR query pools coupled only at committed epoch boundaries.

    Persistent query maps and singleton draw() come from SparsePoolScan. This
    reference wrapper deliberately makes no compiled-throughput claim. Its new
    kernel helpers are compiled, but epoch coordination is a Python loop.
    Invalid public requests are rejected before mutation. An unexpected epoch
    runtime failure disables further writes to this instance; recovery requires
    reconstruction. Atomic refers to information release, not rollback.
    """
    def __init__(self, order, seed=None):
        self._failed = False
        super().__init__(order, seed)

    @property
    def failed(self):
        return self._failed

    def _ensure_healthy(self):
        if self._failed:
            raise RuntimeError('failed epoch: discard and reconstruct the scan instance')

    def register(self):
        self._ensure_healthy()
        return super().register()

    def expose(self, n):
        self._ensure_healthy()
        return super().expose(n)

    def draw(self, q, n=1):
        self._ensure_healthy()
        return super().draw(q, n)

    def stop(self, q):
        self._ensure_healthy()
        return super().stop(q)

    def draw_epoch(self, queries):
        self._ensure_healthy()
        queries = tuple(queries)
        if len(set(queries)) != len(queries):
            raise ValueError('duplicate participating query')
        for q in queries:
            if q not in self._queries:
                raise RuntimeError('unknown/stopped query')
            if self._queries[q]['remaining'] <= 0:
                raise ValueError('exhausted query')
        try:
            return self._run_epoch(queries)
        except BaseException as error:
            self._failed = True
            # Interrupts can occur after a pool has changed. Preserve their
            # control-flow semantics while making that partial state unusable.
            if not isinstance(error, Exception):
                raise
            raise RuntimeError('epoch failed; discard and reconstruct the scan instance') from error

    def _run_epoch(self, queries):
        out = np.empty(len(queries), dtype=self._order.dtype)
        fresh = np.zeros(len(queries), dtype=bool)
        trials = np.zeros(len(queries), dtype=np.int64)
        if not queries:
            return out, fresh, trials
        before = self._physical
        leader = min(range(len(queries)),
                     key=lambda i: self._queries[queries[i]]['remaining'])
        lead = self._queries[queries[leader]]
        leader_remaining = lead['remaining']
        proposed = _propose(lead['forward'], leader_remaining, lead['rng'])
        source_new = bool(proposed >= before)
        positions = [None] * len(queries)
        positions[leader] = before if source_new else int(proposed)
        fresh[leader] = source_new
        trials[leader] = 1
        # All cached tests use the PRE-epoch cursor. No participant sees the
        # newly fetched row as an ordinary cached draw during this epoch.
        for i, q in enumerate(queries):
            if i == leader:
                continue
            state = self._queries[q]
            positions[i], fresh[i], trials[i] = _follower(
                state['forward'], state['remaining'], leader_remaining,
                source_new, before, state['rng'])
        for i, q in enumerate(queries):
            state = self._queries[q]
            position = int(positions[i])
            out[i] = self._order[position]
            _delete(state['forward'], state['inverse'], state['remaining'], position)
            state['remaining'] -= 1
        self._physical = before + int(source_new)
        self.events.append(dict(event='atomic_epoch', queries=list(queries),
                                leader=queries[leader], physical_before=before,
                                physical=self._physical,
                                new_physical=int(source_new),
                                new_consumers=int(fresh.sum()),
                                pool_proposals=trials.tolist()))
        return out, fresh, trials


def warmup():
    scan = AtomicPoolScan(np.arange(5), seed=1)
    scan.expose(2)
    a, b = scan.register(), scan.register()
    scan.draw(a, 1)
    scan.draw_epoch([a, b])
