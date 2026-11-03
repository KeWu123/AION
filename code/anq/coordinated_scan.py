"""Candidate atomic renewal with one shared cache-proposal tape.

The candidate uses no record payloads, contribution values, source-order array,
or population-size allocation. Equal virtual pools with equal progress consume
the same physical-position sequence when participating in the same microepochs.
Different pools keep their own swap-delete state while using a common random
tape for cached draws. Marginals refer to the ideal independent-uniform-word
model; finite spawned PCG64 streams are its deterministic implementation.

All position/fresh information is private to the file layer. Serialized use and
the inherited fail-stop contract apply; there is no rollback or thread safety.
"""

import numpy as np
from numba import njit
from numba.np.random.generator_core import next_uint64
from numba.typed import List

from .position_scan import (
    AtomicPositionScan, _MAX, _DICT_TYPE,
    _integer, _check_output_shape, _randbelow, _threshold, _delete,
)


@njit(cache=False)
def _coordinated_kernel(size, cursor, remaining, forwards, inverses,
                        shared_rng, replacement, out, fresh, trials):
    q_count, rounds = out.shape
    unresolved = np.empty(q_count, dtype=np.int64)
    thresholds = np.empty(q_count, dtype=np.uint64)
    for step in range(rounds):
        before = cursor[0]
        if replacement:
            position = _randbelow(shared_rng, size)
            source_new = position >= before
            if source_new:
                position = before
                cursor[0] += 1
            for i in range(q_count):
                out[i, step] = position
                fresh[i, step] = source_new
                trials[i, step] = 1
            continue

        d_star = remaining[0]
        for i in range(1, q_count):
            d_star = min(d_star, remaining[i])
        # Source, shared thinning-gate, and cache tapes use disjoint successive
        # random words. Every query begins each shared tape at the same word.
        source_new = _randbelow(shared_rng, d_star) < size - before
        unresolved_count = 0
        for i in range(q_count):
            fresh[i, step] = source_new and remaining[i] == d_star
            trials[i, step] = 0
            if fresh[i, step]:
                out[i, step] = before
            elif source_new:
                unresolved[unresolved_count] = i
                unresolved_count += 1
                thresholds[i] = _threshold(remaining[i])

        # Equal bounds receive the same thinning decision, including cohorts
        # whose progress is behind a different query's d_star.
        while unresolved_count:
            word = next_uint64(shared_rng.bit_generator)
            still_unresolved = 0
            for k in range(unresolved_count):
                i = unresolved[k]
                if word >= thresholds[i]:
                    accepts_new = word % np.uint64(remaining[i]) < np.uint64(d_star)
                    fresh[i, step] = accepts_new
                    if accepts_new:
                        out[i, step] = before
                else:
                    unresolved[still_unresolved] = i
                    still_unresolved += 1
            unresolved_count = still_unresolved

        # Start a NEW cache tape after every gate has finished. Gate words are
        # never reused as cache proposals, which would condition these draws.
        unresolved_count = 0
        for i in range(q_count):
            if not fresh[i, step]:
                unresolved[unresolved_count] = i
                unresolved_count += 1
                thresholds[i] = _threshold(remaining[i])

        # Every cached participant reads the same tape from its first word.
        # Compact resolved queries out in place; the tape is never retained.
        while unresolved_count:
            word = next_uint64(shared_rng.bit_generator)
            still_unresolved = 0
            for k in range(unresolved_count):
                i = unresolved[k]
                resolved = False
                if word >= thresholds[i]:
                    if trials[i, step] == _MAX:
                        raise OverflowError("cache proposal counter overflow")
                    trials[i, step] += 1
                    slot = np.int64(word % np.uint64(remaining[i]))
                    forward = forwards[i]
                    position = forward[slot] if slot in forward else slot
                    if position < before:
                        out[i, step] = position
                        resolved = True
                if not resolved:
                    unresolved[still_unresolved] = i
                    still_unresolved += 1
            unresolved_count = still_unresolved

        for i in range(q_count):
            _delete(forwards[i], inverses[i], remaining[i], out[i, step])
            remaining[i] -= 1
        cursor[0] = before + int(source_new)


class CoordinatedPositionScan(AtomicPositionScan):
    """Same public API as AtomicPositionScan, with coordinated cached choices.

    draw_epochs(queries, n=1) returns (positions, fresh, trials), shape (Q, n).
    No intermediate information is returned. For WoR, trials counts actual pool
    proposals: accepting a fresh record costs zero; a word passing the bound's
    uint64 threshold maps to one slot and counts once. Bound-rejection words
    and source/acceptance gates do not count as pool proposals. WR shares one
    proposal and records one per participant, as in AtomicPositionScan.

    Construction spawns the shared PCG64 child first. Query registration spawns
    distinct later children. The shared stream provides the source gate, a
    common thinning-gate tape, and a separate cache tape after all gates finish.
    Equal bounds see identical thinning decisions. Query streams are used by
    inherited draw(), which remains the sequential rule;
    it does not promise coordinated choices between separate draw() calls.

    Persistent state is O(j) per WoR query after j observations; temporary state
    is O(Q), apart from requested O(Q*n) return arrays. Cache rejection has no
    deterministic work bound. Source order, file I/O and payloads stay external.
    """

    def __init__(self, size, seed=None, replacement=False):
        super().__init__(size, seed, replacement)
        child = self._seeds.spawn(1)[0]
        self._shared_rng = np.random.Generator(np.random.PCG64(child))

    def draw_epochs(self, queries, n=1):
        self._healthy()
        count = _integer(n, "epoch count")
        try:
            participants = tuple(queries)
        except TypeError as error:
            raise ValueError("queries must be an iterable of query IDs") from error
        states = []
        seen = set()
        for query in participants:
            q, state = self._query(query)
            if q in seen:
                raise ValueError("duplicate participating query")
            seen.add(q)
            if not self._replacement and count > state["remaining"]:
                raise ValueError("epoch count exceeds a query's remaining positions")
            states.append(state)
        if states and count and self._size == 0:
            raise ValueError("cannot draw from an empty source")
        _check_output_shape(len(states), count)
        try:
            out = np.empty((len(states), count), dtype=np.int64)
            fresh = np.empty(out.shape, dtype=np.bool_)
            trials = np.empty(out.shape, dtype=np.int64)
            if not states or not count:
                return out, fresh, trials
            forwards = List.empty_list(_DICT_TYPE)
            inverses = List.empty_list(_DICT_TYPE)
            remaining = np.empty(len(states), dtype=np.int64)
            for i, state in enumerate(states):
                forwards.append(state["forward"])
                inverses.append(state["inverse"])
                remaining[i] = state["remaining"]
            _coordinated_kernel(self._size, self._cursor, remaining, forwards,
                                inverses, self._shared_rng, self._replacement,
                                out, fresh, trials)
            for i, state in enumerate(states):
                state["remaining"] = int(remaining[i])
            return out, fresh, trials
        except BaseException:
            self._failed = True
            raise


def warmup():
    """Compile the candidate with small private-position examples only."""
    for replacement in (False, True):
        scan = CoordinatedPositionScan(7, seed=3, replacement=replacement)
        a, b = scan.register(), scan.register()
        scan.expose(3, return_positions=False)
        scan.draw(a, 1)
        scan.draw_epochs([a, b], 2)
