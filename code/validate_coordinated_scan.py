"""Synthetic candidate validation; no formal data, driver, or timing runs.

The independent reference retains tapes and uses dense Python lists; production
compacts unresolved queries and retains no tape. Huge integer fixtures use a
slow replay-of-overwrites oracle without an inverse map or population array.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

import anq.coordinated_scan as candidate
import anq.position_scan as inherited
from anq.coordinated_scan import CoordinatedPositionScan


MAX = (1 << 63) - 1
MASK = (1 << 64) - 1


def bounded(rng, bound, trace=None):
    if bound == 1:
        return 0
    threshold = (1 << 64) % bound
    while True:
        word = int(rng.bit_generator.random_raw())
        if trace is not None:
            trace.append(word)
        if word >= threshold:
            return word % bound


class DensePool:
    def __init__(self, size):
        self.values = list(range(size))
    def __len__(self):
        return len(self.values)
    def __getitem__(self, slot):
        return self.values[slot]
    def delete(self, value):
        slot = self.values.index(value)
        self.values[slot] = self.values[-1]
        self.values.pop()


class ReplayPool:
    """Slow independent huge-N oracle: a history of overwritten slots."""
    def __init__(self, size):
        self.remaining = size
        self.overwrites = []
    def __len__(self):
        return self.remaining
    def __getitem__(self, slot):
        for changed_slot, value in reversed(self.overwrites):
            if changed_slot == slot:
                return value
        return slot
    def delete(self, value):
        possible = {value, *(slot for slot, _ in self.overwrites)}
        slot = next(slot for slot in possible
                    if slot < self.remaining and self[slot] == value)
        last_value = self[self.remaining - 1]
        self.remaining -= 1
        if slot != self.remaining:
            self.overwrites.append((slot, last_value))


class IndependentReference:
    def __init__(self, size, seed, replacement=False, pool_type=DensePool):
        self.size, self.replacement, self.physical = size, replacement, 0
        self.seeds = np.random.SeedSequence(seed)
        self.shared = np.random.Generator(np.random.PCG64(self.seeds.spawn(1)[0]))
        self.queries, self.next = {}, 0
        self.pool_type = pool_type
        self.last_raw_gate_words = self.last_raw_cache_words = 0
        self.last_uint_rejections = 0
        self.last_trace = []
    def register(self):
        q = self.next
        self.next += 1
        self.queries[q] = (self.pool_type(self.size),
                           np.random.Generator(np.random.PCG64(self.seeds.spawn(1)[0])))
        return q
    def expose(self, n, return_positions=True):
        before = self.physical
        self.physical += n
        return np.arange(before, self.physical, dtype=np.int64) if return_positions else None
    def stop(self, q):
        del self.queries[q]
    def draw(self, q, n=1):
        pool, rng = self.queries[q]
        out, fresh = [], []
        for _ in range(n):
            position = bounded(rng, self.size) if self.replacement else pool[bounded(rng, len(pool))]
            new = position >= self.physical
            if new:
                position = self.physical
                self.physical += 1
            out.append(position); fresh.append(new)
            if not self.replacement:
                pool.delete(position)
        return np.asarray(out, np.int64), np.asarray(fresh, bool)
    def draw_epochs(self, queries, n=1):
        out = np.empty((len(queries), n), np.int64)
        fresh = np.empty(out.shape, bool)
        trials = np.zeros(out.shape, np.int64)
        self.last_raw_gate_words = self.last_raw_cache_words = 0
        self.last_uint_rejections = 0
        self.last_trace = []
        if not queries:
            return out, fresh, trials
        for step in range(n):
            h = self.physical
            if self.replacement:
                proposed = bounded(self.shared, self.size)
                new = proposed >= h
                out[:, step] = h if new else proposed
                fresh[:, step] = new
                trials[:, step] = 1
                self.physical += int(new)
                continue
            remaining = [len(self.queries[q][0]) for q in queries]
            d_star = min(remaining)
            source_words = []
            source_new = bounded(self.shared, d_star, source_words) < self.size - h
            gate_tape = []
            # Reference implementation deliberately gives each query a fresh
            # tape cursor, retaining shared words generated for earlier queries.
            for i, d in enumerate(remaining):
                if not source_new:
                    fresh[i, step] = False
                elif d == d_star:
                    fresh[i, step] = True
                else:
                    cursor = 0
                    while True:
                        if cursor == len(gate_tape):
                            gate_tape.append(int(self.shared.bit_generator.random_raw()))
                        word = gate_tape[cursor]
                        cursor += 1
                        if word >= (1 << 64) % d:
                            fresh[i, step] = word % d < d_star
                            break
                        self.last_uint_rejections += 1
            self.last_raw_gate_words += len(gate_tape)
            cache_tape = []  # Never reuse gate words here.
            for i, q in enumerate(queries):
                if fresh[i, step]:
                    out[i, step] = h
                    continue
                pool = self.queries[q][0]
                d = len(pool)
                cursor = 0
                while True:
                    if cursor == len(cache_tape):
                        cache_tape.append(int(self.shared.bit_generator.random_raw()))
                    word = cache_tape[cursor]
                    cursor += 1
                    if word < (1 << 64) % d:
                        self.last_uint_rejections += 1
                        continue
                    trials[i, step] += 1
                    position = pool[word % d]
                    if position < h:
                        out[i, step] = position
                        break
            self.last_raw_cache_words += len(cache_tape)
            for i, q in enumerate(queries):
                self.queries[q][0].delete(int(out[i, step]))
            self.physical += int(source_new)
            self.last_trace.append(dict(prefix_before=h, remaining=remaining, d_star=d_star,
                                        source_new=source_new,
                                        source_words=[f"0x{w:016x}" for w in source_words],
                                        gate_words=[f"0x{w:016x}" for w in gate_tape],
                                        cache_words=[f"0x{w:016x}" for w in cache_tape],
                                        positions=out[:, step].tolist(), fresh=fresh[:, step].tolist(),
                                        trials=trials[:, step].tolist()))
        return out, fresh, trials


def results_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.dtype == b.dtype and np.array_equal(a, b), (a, b)


def states_equal(scan, reference, huge=False):
    assert scan.physical == reference.physical
    assert scan._shared_rng.bit_generator.state == reference.shared.bit_generator.state
    assert scan._seeds.state == reference.seeds.state
    assert set(scan._queries) == set(reference.queries)
    for q, (pool, rng) in reference.queries.items():
        state = scan._queries[q]
        assert state["rng"].bit_generator.state == rng.bit_generator.state
        if scan.replacement:
            assert scan.query_entries(q) == 0
            continue
        assert state["remaining"] == len(pool)
        assert scan.query_entries(q) <= 2 * (scan.size - len(pool))
        slots = set(map(int, state["forward"]))
        if huge:
            slots |= {0, max(0, len(pool) - 1)}
            slots |= {slot for slot, _ in pool.overwrites if slot < len(pool)}
        else:
            slots = set(range(len(pool)))
        for slot in slots:
            if slot < len(pool):
                assert int(state["forward"].get(slot, slot)) == pool[slot]
        for value, slot in state["inverse"].items():
            assert pool[int(slot)] == int(value)


def pair(size, seed, replacement=False, huge=False):
    return (CoordinatedPositionScan(size, seed, replacement),
            IndependentReference(size, seed, replacement, ReplayPool if huge else DensePool))


def check_mixed_dense():
    operations = batches = 0
    for replacement in (False, True):
        for size in (0, 1, 2, 3, 7, 19, 67):
            for seed in range(15):
                scan, ref = pair(size, seed, replacement)
                scheduler = np.random.default_rng(seed + size * 103)
                for step in range(100):
                    active = list(ref.queries)
                    choice = int(scheduler.integers(0, 7))
                    if not active or choice == 0:
                        assert scan.register() == ref.register()
                    elif choice == 1:
                        n = int(scheduler.integers(0, size - ref.physical + 1))
                        positions = bool(scheduler.integers(0, 2))
                        actual = scan.expose(n, return_positions=positions)
                        expected = ref.expose(n, positions)
                        assert actual is expected is None or np.array_equal(actual, expected)
                    elif choice == 2 and len(active) > 1:
                        q = int(scheduler.choice(active))
                        scan.stop(q); ref.stop(q)
                    elif choice <= 4:
                        q = int(scheduler.choice(active))
                        cap = (5 if size else 0) if replacement else len(ref.queries[q][0])
                        n = int(scheduler.integers(0, min(cap, 5) + 1))
                        results_equal(scan.draw(q, n), ref.draw(q, n))
                    else:
                        qs = list(map(int, scheduler.choice(active, size=int(scheduler.integers(0, len(active) + 1)), replace=False)))
                        cap = (5 if size else 0) if replacement else min([len(ref.queries[q][0]) for q in qs] + [5])
                        n = int(scheduler.integers(0, min(cap, 5) + 1))
                        results_equal(scan.draw_epochs(qs, n), ref.draw_epochs(qs, n))
                        batches += 1
                    states_equal(scan, ref)
                    operations += 1
    return dict(operations=operations, atomic_batches=batches, modes=["WoR", "WR"],
                sizes=[0, 1, 2, 3, 7, 19, 67], seeds_per_size_mode=15,
                checks="all outputs, sparse pools, inverse maps, shared/private PCG64 states")


def install_pool(scan, ref, q, used):
    """Synthetic valid prefix fixture, built by independent dense deletion."""
    pool = ref.queries[q][0]
    for value in used:
        pool.delete(value)
    state = scan._queries[q]
    state["remaining"] = len(pool)
    state["forward"].clear(); state["inverse"].clear()
    for slot in range(len(pool)):
        value = pool[slot]
        if value != slot:
            state["forward"][slot] = value
            state["inverse"][value] = slot


def check_synchrony():
    cases = 0
    for replacement in (False, True):
        for size in (1, 2, 7, 41):
            for prefix in (0, size // 2, size):
                scan, ref = pair(size, 127 + size + prefix, replacement)
                qs = [scan.register() for _ in range(4)]
                assert qs == [ref.register() for _ in range(4)]
                scan.expose(prefix, return_positions=False); ref.expose(prefix, False)
                n = 3 * size if replacement else size
                actual = scan.draw_epochs(qs, n)
                results_equal(actual, ref.draw_epochs(qs, n))
                for array in actual:
                    assert np.all(array == array[0])
                if not replacement:
                    assert len(set(map(int, actual[0][0]))) == size
                states_equal(scan, ref)
                cases += 1
    # Identical pools A/B stay coupled even when C has a different, faster
    # history, during background exposure, pauses, and late registrations.
    scan, ref = pair(37, 800)
    qs = [scan.register() for _ in range(3)]
    assert qs == [ref.register() for _ in range(3)]
    scan.expose(12, return_positions=False); ref.expose(12, False)
    results_equal(scan.draw(qs[2], 9), ref.draw(qs[2], 9))
    for n in (4, 0, 6):
        actual = scan.draw_epochs(qs, n)
        results_equal(actual, ref.draw_epochs(qs, n))
        for array in actual:
            assert np.array_equal(array[0], array[1])
    results_equal(scan.draw(qs[2], 5), ref.draw(qs[2], 5))
    delta = min(3, scan.size - scan.physical)
    scan.expose(delta, return_positions=False); ref.expose(delta, False)
    late = scan.register(); assert late == ref.register()
    for cohort in ([late, qs[2]], [qs[1], late, qs[0], qs[2]], [qs[0], qs[1]]):
        actual = scan.draw_epochs(cohort, 3)
        results_equal(actual, ref.draw_epochs(cohort, 3))
        if qs[0] in cohort and qs[1] in cohort:
            for array in actual:
                assert np.array_equal(array[cohort.index(qs[0])], array[cohort.index(qs[1])])
        states_equal(scan, ref)
    # Minimal counterexample: E is certain; independent Bernoulli gates can
    # split A/B, while the shared gate keeps their entire outcome identical.
    split_seed = None
    split_flags = None
    for seed in range(100):
        seeds = np.random.SeedSequence(seed)
        seeds.spawn(1)
        rngs = [np.random.default_rng(c) for c in seeds.spawn(3)]
        flags = [bounded(rngs[i], 2) < 1 for i in (0, 1)]
        if flags[0] != flags[1]:
            split_seed = seed
            split_flags = flags
            break
    assert split_seed is not None
    counterexample_trace = None
    separate_tapes_trace = None
    for seed in range(100):
        scan, ref = pair(2, seed)
        qs = [scan.register() for _ in range(3)]
        assert qs == [ref.register() for _ in range(3)]
        scan.expose(1, return_positions=False); ref.expose(1, False)
        install_pool(scan, ref, qs[2], [0])
        actual = scan.draw_epochs(qs, 1)
        results_equal(actual, ref.draw_epochs(qs, 1))
        if seed == split_seed:
            counterexample_trace = ref.last_trace
        if separate_tapes_trace is None and ref.last_trace[0]["gate_words"] and ref.last_trace[0]["cache_words"]:
            separate_tapes_trace = dict(seed=seed, trace=ref.last_trace)
        assert actual[1][2, 0]
        for array in actual:
            assert array[0, 0] == array[1, 0]
    assert separate_tapes_trace is not None
    # Fixed participants and no observations: batching must preserve trajectory.
    for replacement in (False, True):
        a, _ = pair(23, 52, replacement)
        b, _ = pair(23, 52, replacement)
        qa, qb = [a.register() for _ in range(3)], [b.register() for _ in range(3)]
        a.expose(12, return_positions=False); b.expose(12, return_positions=False)
        actual = a.draw_epochs(qa, 20)
        chunks = [b.draw_epoch(qb) for _ in range(20)]
        results_equal(actual, tuple(np.stack([x[i] for x in chunks], axis=1) for i in range(3)))
        assert a._shared_rng.bit_generator.state == b._shared_rng.bit_generator.state
    # The common tapes make an unchanged participant set invariant to input
    # ordering, including unequal query progress and different cache contents.
    a, ra = pair(37, 77)
    b, rb = pair(37, 77)
    qa, qb = [a.register() for _ in range(4)], [b.register() for _ in range(4)]
    a.expose(13, return_positions=False); b.expose(13, return_positions=False)
    for i, n in enumerate((3, 7, 0, 5)):
        results_equal(a.draw(qa[i], n), b.draw(qb[i], n))
    actual = a.draw_epochs(qa, 10)
    permutation = [2, 0, 3, 1]
    reordered = b.draw_epochs([qb[i] for i in permutation], 10)
    results_equal(actual, tuple(array[np.argsort(permutation)] for array in reordered))
    assert a._shared_rng.bit_generator.state == b._shared_rng.bit_generator.state
    return dict(synchronized_complete_stream_cases=cases, heterogeneous_leader_pause_background_late_queries=True,
                counterexample=dict(size=2, prefix=1, used=[[], [], [0]],
                                    independent_gates_split_seed=split_seed,
                                    independent_gate_flags=split_flags,
                                    shared_gate_seeds_checked=100,
                                    independent_reference_raw_trace=counterexample_trace,
                                    separate_gate_and_cache_tapes_trace=separate_tapes_trace),
                same_pool_fresh_positions_and_trials_identical=True,
                batched_and_single_microepoch_trajectory_equal=True,
                participant_order_invariance=True)


def check_sparse_cache():
    maximum_trials = total_raw_cache_words = rounds = 0
    # One or two cached positions among 50+ unused slots, over many rounds and
    # seeds. These are small synthetic valid states, not real-file workloads.
    for seed in range(60):
        scan, ref = pair(127, seed + 30000)
        qs = [scan.register() for _ in range(3)]
        assert qs == [ref.register() for _ in range(3)]
        scan.expose(61, return_positions=False); ref.expose(61, False)
        install_pool(scan, ref, qs[0], list(range(60)))
        install_pool(scan, ref, qs[1], list(range(1, 61)))
        install_pool(scan, ref, qs[2], list(range(59)))
        for _ in range(20):
            actual = scan.draw_epochs(qs, 1)
            expected = ref.draw_epochs(qs, 1)
            results_equal(actual, expected)
            maximum_trials = max(maximum_trials, int(actual[2].max()))
            total_raw_cache_words += ref.last_raw_cache_words
            states_equal(scan, ref)
            rounds += 1
    assert maximum_trials > 20
    return dict(size=127, initial_prefix=61, seeds=60, rounds=rounds,
                initial_cached_counts=[1, 1, 2], initial_remaining=[67, 67, 68],
                maximum_observed_pool_trials=maximum_trials,
                reference_raw_cache_words=total_raw_cache_words,
                distinct_cache_sets_equal_and_unequal_bounds=True)


def check_uint_boundaries():
    sizes = sorted({1, 2, MAX, *[d for k in range(1, 63)
                               for d in ((1 << k) - 1, 1 << k, (1 << k) + 1)
                               if 1 <= d <= MAX]})
    uint_rejections = 0
    calls = 0
    for size in sizes:
        for replacement in (False, True):
            scan, ref = pair(size, MASK, replacement, huge=True)
            qs = [scan.register() for _ in range(3)]
            assert qs == [ref.register() for _ in range(3)]
            scan.expose(np.uint64(size), return_positions=False); ref.expose(size, False)
            for i, q in enumerate(qs):
                n = i if replacement else min(i, max(0, size - 1))
                results_equal(scan.draw(np.uint64(q), np.uint64(n)), ref.draw(q, n))
            count = 12 if replacement else min(12, min(len(p) for p, _ in ref.queries.values()))
            actual = scan.draw_epochs(np.asarray(qs, np.uint64), np.uint64(count))
            results_equal(actual, ref.draw_epochs(qs, count))
            uint_rejections += ref.last_uint_rejections
            assert not actual[1].any()
            states_equal(scan, ref, huge=True)
            calls += 1
    # Gate tapes also exercise huge integer rejection, including the slowest
    # acceptance region just above a half-range power of two.
    for size in ((1 << 62) + 1, (1 << 62) + 2, MAX):
        for seed in range(40):
            scan, ref = pair(size, seed + 40000, huge=True)
            qs = [scan.register() for _ in range(4)]
            assert qs == [ref.register() for _ in range(4)]
            scan.expose(1, return_positions=False); ref.expose(1, False)
            results_equal(scan.draw(qs[2], 1), ref.draw(qs[2], 1))
            actual = scan.draw_epochs(qs, 8)
            results_equal(actual, ref.draw_epochs(qs, 8))
            uint_rejections += ref.last_uint_rejections
            states_equal(scan, ref, huge=True)
            calls += 1
    assert uint_rejections > 0
    return dict(population_bounds=len(sizes), cases=calls,
                powers_of_two_and_neighbors=True, maximum=MAX,
                uint_threshold_rejections_exercised=uint_rejections,
                exact_shared_and_private_raw_pcg64_state=True,
                no_population_allocation=True)


def snapshot(scan):
    return (scan.physical, scan.failed, scan._next, repr(scan._seeds.state),
            repr(scan._shared_rng.bit_generator.state),
            {q: (s["remaining"], dict(s["forward"]), dict(s["inverse"]),
                 repr(s["rng"].bit_generator.state)) for q, s in scan._queries.items()})


def expect_error(call, kinds=(ValueError, RuntimeError)):
    try:
        call()
    except kinds:
        return
    raise AssertionError("request did not fail")


def check_public_inputs():
    invalid = 0
    bad_ints = [-1, 1.0, True, np.bool_(False), None, float("nan"), "1", np.uint64(1 << 63)]
    for bad in bad_ints:
        expect_error(lambda bad=bad: CoordinatedPositionScan(bad))
        invalid += 1
    for seed in (-1, True, 1.0, 1 << 64):
        expect_error(lambda seed=seed: CoordinatedPositionScan(1, seed))
        invalid += 1
    expect_error(lambda: CoordinatedPositionScan(1, replacement=1))
    invalid += 1
    for replacement in (False, True):
        scan = CoordinatedPositionScan(7, np.uint64(MASK), np.bool_(replacement))
        q, stopped = scan.register(), scan.register()
        scan.expose(np.uint64(2)); scan.draw(np.uint64(q), np.uint64(2)); scan.stop(stopped)
        calls = []
        for bad in bad_ints:
            calls += [lambda bad=bad: scan.expose(bad), lambda bad=bad: scan.draw(q, bad),
                      lambda bad=bad: scan.draw_epochs([q], bad), lambda bad=bad: scan.draw(bad, 0),
                      lambda bad=bad: scan.stop(bad), lambda bad=bad: scan.query_entries(bad)]
        calls += [lambda: scan.expose(8), lambda: scan.draw(1234), lambda: scan.draw(stopped, 0),
                  lambda: scan.stop(stopped), lambda: scan.draw_epochs([q, q], 0),
                  lambda: scan.draw_epochs([q, 1234], 0), lambda: scan.draw_epochs(None),
                  lambda: scan.draw_epochs(np.array([[q]]), 0), lambda: scan.draw(q, MAX),
                  lambda: scan.draw_epochs([q], MAX), lambda: scan.draw_epochs([], MAX),
                  lambda: scan.expose(1, return_positions=1),
                  lambda: scan.expose(-1, return_positions=False),
                  lambda: scan.expose(8, return_positions=False)]
        if not replacement:
            calls += [lambda: scan.draw(q, 6), lambda: scan.draw_epochs([q], 6)]
        before = snapshot(scan)
        for call in calls:
            expect_error(call)
            assert snapshot(scan) == before
            invalid += 1
        # Zero-size batch paths cannot advance either random stream or cursor.
        for call in (lambda: scan.draw(q, 0), lambda: scan.draw_epochs([q], 0),
                     lambda: scan.draw_epochs([], 3), lambda: scan.draw_epoch([])):
            call()
            assert snapshot(scan) == before
        scan.expose(scan.size - scan.physical, return_positions=False)
        actual = scan.draw_epochs([q], 30 if replacement else 5)
        assert not actual[1].any()
        if not replacement:
            expect_error(lambda: scan.draw_epochs([q], 1))
            assert not scan.failed
    for replacement in (False, True):
        scan = CoordinatedPositionScan(0, 1, replacement)
        q = scan.register()
        before = snapshot(scan)
        assert scan.expose(0, return_positions=False) is None
        assert scan.draw(q, 0)[0].shape == (0,)
        assert scan.draw_epochs([], 3)[0].shape == (0, 3)
        expect_error(lambda: scan.draw_epochs([q], 1))
        assert snapshot(scan) == before
    return dict(invalid_requests_without_mutation=invalid,
                zero_batches_preserve_all_rng_states=True,
                empty_census_duplicates_unsigned_expose_and_stopped_queries=True)


def check_fail_stop():
    cases = 0
    for operation in ("coordinated", "sequential", "register", "expose", "expose_no_output", "stop"):
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit, BaseException):
            scan = CoordinatedPositionScan(8, 12)
            q = scan.register()
            error = error_type("injected after partial mutation")
            restore = lambda: None
            if operation in ("coordinated", "sequential"):
                module = candidate if operation == "coordinated" else inherited
                name = "_coordinated_kernel" if operation == "coordinated" else "_sequential_kernel"
                original = getattr(module, name)
                def broken(*args, **kwargs):
                    args[1][0] += 1
                    scan._queries[q]["forward"][0] = 7
                    scan._shared_rng.bit_generator.random_raw()
                    scan._queries[q]["rng"].bit_generator.random_raw()
                    raise error
                setattr(module, name, broken)
                restore = lambda module=module, name=name, original=original: setattr(module, name, original)
                call = lambda: scan.draw_epochs([q], 1) if operation == "coordinated" else scan.draw(q, 1)
            elif operation == "register":
                original_seeds = scan._seeds
                class BrokenSeeds:
                    @property
                    def state(self):
                        return original_seeds.state
                    def spawn(self, n):
                        original_seeds.spawn(n)
                        raise error
                scan._seeds = BrokenSeeds()
                call = scan.register
            elif operation == "expose":
                original_arange = inherited.np.arange
                def broken_arange(*args, **kwargs):
                    scan._cursor[0] += 1
                    raise error
                inherited.np.arange = broken_arange
                restore = lambda: setattr(inherited.np, "arange", original_arange)
                call = lambda: scan.expose(1)
            elif operation == "expose_no_output":
                original_cursor = scan._cursor
                class BrokenCursor:
                    def __getitem__(self, key):
                        return original_cursor[key]
                    def __setitem__(self, key, value):
                        original_cursor[key] = value
                        raise error
                scan._cursor = BrokenCursor()
                call = lambda: scan.expose(1, return_positions=False)
            else:
                class BrokenQueries(dict):
                    def __delitem__(self, key):
                        super().__delitem__(key)
                        raise error
                scan._queries = BrokenQueries(scan._queries)
                call = lambda: scan.stop(q)
            try:
                try:
                    call()
                except BaseException as caught:
                    assert caught is error
                else:
                    raise AssertionError("injected failure swallowed")
                assert scan.failed
                before = snapshot(scan)
                for call in (scan.register, lambda: scan.expose(0), lambda: scan.draw(q, 0),
                             lambda: scan.stop(q), lambda: scan.draw_epochs([], 0),
                             lambda: scan.draw_epoch([])):
                    expect_error(call, (RuntimeError,))
                    assert snapshot(scan) == before
                cases += 1
            finally:
                restore()
    return dict(partially_mutating_failures=cases, original_exception_identity_preserved=True,
                keyboard_interrupt_system_exit_base_exception=True,
                future_write_apis_disabled=True, rollback=False)


def check_statistics(repetitions):
    results = []
    size = 7
    for replacement in (False, True):
        observed = np.zeros((3, size), np.int64)
        expected = np.zeros((3, size), float)
        variance = np.zeros((3, size), float)
        for seed in range(repetitions):
            scan = CoordinatedPositionScan(size, seed + 50000, replacement)
            qs = [scan.register() for _ in range(3)]
            scan.expose(3, return_positions=False)
            prior = set(map(int, scan.draw(qs[2], 2)[0]))
            h = scan.physical
            for i in range(3):
                used = prior if i == 2 and not replacement else set()
                d = size - len(used)
                probabilities = np.zeros(size, float)
                for p in set(range(h)) - used:
                    probabilities[p] = 1 / d
                if h < size:
                    probabilities[h] = (size - h) / d
                expected[i] += probabilities
                variance[i] += probabilities * (1 - probabilities)
            out, fresh, trials = scan.draw_epochs(qs, 1)
            assert out[0, 0] == out[1, 0] and fresh[0, 0] == fresh[1, 0]
            for i in range(3):
                observed[i, out[i, 0]] += 1
        z = np.divide(observed - expected, np.sqrt(variance),
                      out=np.zeros_like(expected), where=variance > 0)
        assert float(np.abs(z).max()) < 7
        results.append(dict(mode="WR" if replacement else "WoR", repetitions=repetitions,
                            observed=observed.tolist(), conditional_expected=expected.tolist(),
                            maximum_absolute_position_z=float(np.abs(z).max()), threshold=7))
    return results


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    protected = [root / "code/anq/position_scan.py", root / "code/validate_position_scan.py"]
    protected += sorted((root / "reviews/innovation_20261017/position_scan").glob("*"))
    protected = [p for p in protected if p.is_file()]
    before_hashes = {str(p.relative_to(root)): digest(p) for p in protected}
    report = dict(environment=dict(python=sys.version, numpy=np.__version__, numba=numba.__version__,
                                   platform=platform.platform()),
                  scope="synthetic candidate validation only; no formal data, driver, or timing benchmark",
                  api="AtomicPositionScan-compatible; unchanged return shapes; WoR fresh consumers have trials=0",
                  limits=["Finite PCG64 is not ideal independent coins.",
                          "Uses internal Numba next_uint64 and inherited private helpers.",
                          "Shared-gate and cache tapes use separate successive random words.",
                          "Equal virtual arrays remain synchronized only when their progress and participation match.",
                          "Reject loops have no deterministic bound; no lifecycle-optimality claim.",
                          "Serialized caller only; failure requires discard, without rollback.",
                          "No source-order, file-I/O, contribution evaluation or policy isolation validation here."])
    for name, check in [("dense_reference", check_mixed_dense), ("synchrony", check_synchrony),
                        ("sparse_cache", check_sparse_cache), ("uint_boundaries", check_uint_boundaries),
                        ("public_inputs", check_public_inputs), ("fail_stop", check_fail_stop),
                        ("statistics", lambda: check_statistics(args.repetitions))]:
        report[name] = check()
        print(f"PASS {name}", flush=True)
    assert candidate._coordinated_kernel.nopython_signatures
    report["nopython_signatures"] = [str(s) for s in candidate._coordinated_kernel.nopython_signatures]
    after_hashes = {str(p.relative_to(root)): digest(p) for p in protected}
    assert after_hashes == before_hashes
    report["protected_unchanged_sha256"] = after_hashes
    report["source_sha256"] = {"code/anq/coordinated_scan.py": digest(root / "code/anq/coordinated_scan.py"),
                                "code/validate_coordinated_scan.py": digest(__file__)}
    report["passed"] = True
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
