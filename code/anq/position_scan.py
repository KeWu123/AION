"""Private physical-position sampling over an externally hidden file order.

The engine knows N and a retained prefix length H; it neither loads a source
permutation nor decodes record IDs. The file layer must own the hidden uniform
permutation and keep returned offsets/fresh flags out of policy-visible context.

Uniform-marginal arguments use ideal independent uniform random words. This
implementation uses finite deterministic PCG64 streams, independently derived
with SeedSequence.spawn; it makes no exact-distribution claim for finite seeds.
Atomic means a precommitted information-release boundary, not transaction
rollback or thread safety. Instances must be used by one serialized caller.
"""

from numbers import Integral

import numpy as np
from numba import njit, types, typeof
from numba.np.random.generator_core import next_uint64
from numba.typed import Dict, List


_MAX = (1 << 63) - 1
_DICT_TYPE = types.DictType(types.int64, types.int64)
_RNG_TYPE = typeof(np.random.Generator(np.random.PCG64(0)))


@njit(cache=False)
def _threshold(bound):
    """2**64 % bound, computed entirely in wrapping uint64 arithmetic."""
    d = np.uint64(bound)
    return (np.uint64(0) - d) % d


@njit(cache=False)
def _randbelow(rng, bound):
    if bound == 1:
        return np.int64(0)
    d = np.uint64(bound)
    threshold = _threshold(bound)
    while True:
        word = next_uint64(rng.bit_generator)
        if word >= threshold:
            return np.int64(word % d)


@njit(cache=False)
def _delete(forward, inverse, remaining, value):
    slot = inverse[value] if value in inverse else value
    last = remaining - 1
    replacement = forward[last] if last in forward else last
    if last in forward:
        del forward[last]
    if value in inverse:
        del inverse[value]
    if slot != last:
        if replacement == slot:
            if slot in forward:
                del forward[slot]
            if replacement in inverse:
                del inverse[replacement]
        else:
            forward[slot] = replacement
            inverse[replacement] = slot


@njit(cache=False)
def _propose(forward, remaining, rng):
    slot = _randbelow(rng, remaining)
    return forward[slot] if slot in forward else slot


@njit(cache=False)
def _cached(forward, remaining, before, rng):
    trials = np.int64(0)
    while True:
        if trials == _MAX:
            raise OverflowError("cache proposal counter overflow")
        position = _propose(forward, remaining, rng)
        trials += 1
        if position < before:
            return position, trials


@njit(cache=False)
def _sequential_kernel(size, cursor, remaining, forward, inverse, rng,
                       replacement, out, fresh):
    for step in range(len(out)):
        if replacement:
            position = _randbelow(rng, size)
        else:
            position = _propose(forward, remaining, rng)
        new = position >= cursor[0]
        if new:
            position = cursor[0]
            cursor[0] += 1
        out[step] = position
        fresh[step] = new
        if not replacement:
            _delete(forward, inverse, remaining, position)
            remaining -= 1
    return remaining


@njit(cache=False)
def _atomic_kernel(size, cursor, remaining, forwards, inverses, rngs,
                   replacement, out, fresh, trials):
    """Entire n-round coordination is compiled; Python runs only per batch."""
    q_count, rounds = out.shape
    for step in range(rounds):
        before = cursor[0]
        if replacement:
            position = _randbelow(rngs[0], size)
            new = position >= before
            if new:
                position = before
                cursor[0] += 1
            for i in range(q_count):
                out[i, step] = position
                fresh[i, step] = new
                trials[i, step] = 1
            continue

        leader = 0
        for i in range(1, q_count):
            if remaining[i] < remaining[leader]:
                leader = i
        d_star = remaining[leader]
        proposed = _propose(forwards[leader], d_star, rngs[leader])
        source_new = proposed >= before
        out[leader, step] = before if source_new else proposed
        fresh[leader, step] = source_new
        trials[leader, step] = 1
        for i in range(q_count):
            if i == leader:
                continue
            if source_new and _randbelow(rngs[i], remaining[i]) < d_star:
                out[i, step] = before
                fresh[i, step] = True
                trials[i, step] = 0
            else:
                position, count = _cached(forwards[i], remaining[i], before, rngs[i])
                out[i, step] = position
                fresh[i, step] = False
                trials[i, step] = count
        for i in range(q_count):
            _delete(forwards[i], inverses[i], remaining[i], out[i, step])
            remaining[i] -= 1
        cursor[0] = before + int(source_new)


def _integer(value, name, maximum=_MAX):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)  # Validate unsigned values before any int64 conversion.
    if result < 0 or result > maximum:
        raise ValueError(f"{name} outside [0, {maximum}]")
    return result


def _check_output_shape(rows, columns=1):
    # NumPy arrays use signed pointer-sized byte counts, not just item counts.
    maximum = np.iinfo(np.intp).max // np.dtype(np.int64).itemsize
    if max(rows, columns) > maximum or rows * columns > maximum:
        raise ValueError("requested output exceeds platform array size")


class SequentialPositionScan:
    """Sparse position-only renewal; no O(N) source or per-query arrays.

    Inputs are non-bool Python/NumPy integers. size <= 2**63-1; seed is None or
    an integer in [0, 2**64-1]. expose(n) advances by n additional records and
    returns their private offsets. expose(n, return_positions=False) returns
    None and updates only H, allowing huge prefixes without an output array.
    draw() returns private offsets/fresh flags.
    Empty draws are allowed, including at census. query_entries() counts sparse
    map entries; stopped and unknown query IDs raise RuntimeError.

    Invalid requests do not change state. Any BaseException during the mutation
    phase fail-stops every later writing API, without rollback. Read properties
    remain diagnostic only after failure. The original exception is re-raised.
    """

    def __init__(self, size, seed=None, replacement=False):
        self._size = _integer(size, "size")
        if not isinstance(replacement, (bool, np.bool_)):
            raise ValueError("replacement must be boolean")
        self._replacement = bool(replacement)
        if seed is not None:
            seed = _integer(seed, "seed", (1 << 64) - 1)
        self._seeds = np.random.SeedSequence(seed)
        self._cursor = np.zeros(1, dtype=np.int64)
        self._queries = {}
        self._next = 0
        self._failed = False

    @property
    def size(self):
        return self._size

    @property
    def physical(self):
        return int(self._cursor[0])

    @property
    def replacement(self):
        return self._replacement

    @property
    def failed(self):
        return self._failed

    def _healthy(self):
        if self._failed:
            raise RuntimeError("scan failed; discard and reconstruct this instance")

    def _query(self, query):
        q = _integer(query, "query")
        if q not in self._queries:
            raise RuntimeError("unknown/stopped query")
        return q, self._queries[q]

    def register(self):
        self._healthy()
        if self._next > _MAX:
            raise ValueError("query ID space exhausted")
        try:
            child_seed = self._seeds.spawn(1)[0]
            state = dict(remaining=self._size,
                         forward=Dict.empty(types.int64, types.int64),
                         inverse=Dict.empty(types.int64, types.int64),
                         rng=np.random.Generator(np.random.PCG64(child_seed)))
            q = self._next
            self._queries[q] = state
            self._next += 1
            return q
        except BaseException:
            self._failed = True
            raise

    def expose(self, n, *, return_positions=True):
        self._healthy()
        count = _integer(n, "expose count")
        if not isinstance(return_positions, (bool, np.bool_)):
            raise ValueError("return_positions must be boolean")
        before = self.physical
        if count > self._size - before:
            raise ValueError("exposure exceeds source size")
        if return_positions:
            _check_output_shape(count)
        try:
            positions = (np.arange(before, before + count, dtype=np.int64)
                         if return_positions else None)
            self._cursor[0] = before + count
            return positions
        except BaseException:
            self._failed = True
            raise

    def draw(self, q, n=1):
        self._healthy()
        _, state = self._query(q)
        count = _integer(n, "draw count")
        if self._replacement:
            if count and self._size == 0:
                raise ValueError("cannot draw from an empty source")
        elif count > state["remaining"]:
            raise ValueError("draw exceeds query's remaining positions")
        _check_output_shape(count)
        try:
            out = np.empty(count, dtype=np.int64)
            fresh = np.empty(count, dtype=np.bool_)
            if count:
                state["remaining"] = int(_sequential_kernel(
                    self._size, self._cursor, state["remaining"],
                    state["forward"], state["inverse"], state["rng"],
                    self._replacement, out, fresh))
            return out, fresh
        except BaseException:
            self._failed = True
            raise

    def query_entries(self, q):
        _, state = self._query(q)
        return len(state["forward"]) + len(state["inverse"])

    def stop(self, q):
        self._healthy()
        query, _ = self._query(q)
        try:
            del self._queries[query]
        except BaseException:
            self._failed = True
            raise


class AtomicPositionScan(SequentialPositionScan):
    """Precommit Q participants for n microepochs, with no intermediate release.

    draw_epochs returns arrays shaped (Q, n), in supplied query order. Every
    participant gets one observation per microepoch. At most one source record
    is exposed in each microepoch; fresh marks each consumer of that record.
    A leader is the first participant with the fewest remaining positions.
    trials counts pool proposals (leader 1; follower accepting new row 0;
    cached follower counts all rejection attempts). WR shares the first
    participant's proposal; each trial entry is 1 for the shared proposal.

    Marginals concern each participant given the pre-microepoch history;
    within-microepoch observations can be dependent. Rejection sampling of
    cached positions can be expensive when a cache is very sparse. Persistent
    query storage is O(j) map entries after j WoR draws; work per call includes
    O(Q) packaging and O(Q*n) outputs, never an array of source size.
    """

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
            rngs = List.empty_list(_RNG_TYPE)
            remaining = np.empty(len(states), dtype=np.int64)
            for i, state in enumerate(states):
                forwards.append(state["forward"])
                inverses.append(state["inverse"])
                rngs.append(state["rng"])
                remaining[i] = state["remaining"]
            _atomic_kernel(self._size, self._cursor, remaining, forwards,
                           inverses, rngs, self._replacement, out, fresh, trials)
            for i, state in enumerate(states):
                state["remaining"] = int(remaining[i])
            return out, fresh, trials
        except BaseException:
            self._failed = True
            raise

    def draw_epoch(self, queries):
        out, fresh, trials = self.draw_epochs(queries, 1)
        return out[:, 0], fresh[:, 0], trials[:, 0]


def warmup():
    """Compile both kernels without accessing external files."""
    for replacement in (False, True):
        scan = AtomicPositionScan(5, seed=1, replacement=replacement)
        a, b = scan.register(), scan.register()
        scan.expose(2)
        scan.draw(a, 1)
        scan.draw_epochs([a, b], 2)
