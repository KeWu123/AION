"""Reproducible validation for the isolated position-only sampling candidate.

Run: python -B code/validate_position_scan.py --out reviews/.../position_scan
Dense arrays and exhaustive source permutations occur in THIS validator only.
"""

import argparse
import itertools
import json
import math
import platform
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numba
import numpy as np
from numba import njit

import anq.position_scan as engine
from anq.position_scan import AtomicPositionScan, SequentialPositionScan


MASK = (1 << 64) - 1
MAX = (1 << 63) - 1


def bounded(rng, bound):
    """Independent Python arbitrary-precision reduction of NumPy raw words."""
    if bound == 1:
        return 0
    threshold = (1 << 64) % bound
    while True:
        word = int(rng.bit_generator.random_raw())
        if word >= threshold:
            return word % bound


class DenseReference:
    def __init__(self, size, seed, replacement=False):
        self.size = size
        self.replacement = replacement
        self.physical = 0
        self.seeds = np.random.SeedSequence(seed)
        self.queries = {}
        self.next = 0

    def register(self):
        q = self.next
        self.next += 1
        self.queries[q] = [list(range(self.size)),
                           np.random.Generator(np.random.PCG64(self.seeds.spawn(1)[0]))]
        return q

    def expose(self, count):
        before = self.physical
        self.physical += count
        return np.arange(before, self.physical, dtype=np.int64)

    @staticmethod
    def delete(pool, position):
        index = pool.index(position)
        pool[index] = pool[-1]
        pool.pop()

    def draw(self, q, n=1):
        pool, rng = self.queries[q]
        out, fresh = [], []
        for _ in range(n):
            position = bounded(rng, self.size) if self.replacement else pool[bounded(rng, len(pool))]
            new = position >= self.physical
            if new:
                position = self.physical
                self.physical += 1
            out.append(position)
            fresh.append(new)
            if not self.replacement:
                self.delete(pool, position)
        return np.asarray(out, dtype=np.int64), np.asarray(fresh, dtype=bool)

    def draw_epochs(self, queries, n=1):
        out = np.empty((len(queries), n), dtype=np.int64)
        fresh = np.empty(out.shape, dtype=bool)
        trials = np.empty(out.shape, dtype=np.int64)
        if not queries:
            return out, fresh, trials
        for step in range(n):
            before = self.physical
            if self.replacement:
                proposed = bounded(self.queries[queries[0]][1], self.size)
                new = proposed >= before
                out[:, step] = before if new else proposed
                fresh[:, step] = new
                trials[:, step] = 1
                self.physical += int(new)
                continue
            leader = min(range(len(queries)), key=lambda i: len(self.queries[queries[i]][0]))
            d_star = len(self.queries[queries[leader]][0])
            pool, rng = self.queries[queries[leader]]
            proposed = pool[bounded(rng, len(pool))]
            source_new = proposed >= before
            out[leader, step] = before if source_new else proposed
            fresh[leader, step] = source_new
            trials[leader, step] = 1
            for i, q in enumerate(queries):
                if i == leader:
                    continue
                pool, rng = self.queries[q]
                if source_new and bounded(rng, len(pool)) < d_star:
                    out[i, step], fresh[i, step], trials[i, step] = before, True, 0
                else:
                    attempts = 0
                    while True:
                        position = pool[bounded(rng, len(pool))]
                        attempts += 1
                        if position < before:
                            break
                    out[i, step], fresh[i, step], trials[i, step] = position, False, attempts
            for i, q in enumerate(queries):
                self.delete(self.queries[q][0], int(out[i, step]))
            self.physical += int(source_new)
        return out, fresh, trials

    def stop(self, q):
        del self.queries[q]


def equal_result(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert np.array_equal(a, b), (a, b)
        assert a.dtype == b.dtype


def equal_state(actual, dense):
    assert actual.physical == dense.physical
    assert set(actual._queries) == set(dense.queries)
    for q, (pool, rng) in dense.queries.items():
        state = actual._queries[q]
        assert state["rng"].bit_generator.state == rng.bit_generator.state
        if actual.replacement:
            assert actual.query_entries(q) == 0
            continue
        assert state["remaining"] == len(pool)
        expanded = [int(state["forward"].get(i, i)) for i in range(len(pool))]
        assert expanded == pool
        assert len(set(expanded)) == len(expanded)
        assert actual.query_entries(q) <= 2 * (actual.size - len(pool))
        for value, slot in state["inverse"].items():
            assert pool[int(slot)] == int(value)
        assert all(value < actual.physical for value in set(range(actual.size)) - set(pool))


@njit(cache=False)
def compiled_words(rng, bounds, count):
    values = np.empty((len(bounds), count), dtype=np.int64)
    thresholds = np.empty(len(bounds), dtype=np.uint64)
    for i in range(len(bounds)):
        thresholds[i] = engine._threshold(bounds[i])
        for j in range(count):
            values[i, j] = engine._randbelow(rng, bounds[i])
    return values, thresholds


def check_rng():
    bounds = sorted({1, 2, MAX, *[x for k in range(1, 63)
                                for x in ((1 << k) - 1, 1 << k, (1 << k) + 1)
                                if 1 <= x <= MAX]})
    words = 256
    a_rng, b_rng = np.random.default_rng(413), np.random.default_rng(413)
    actual, thresholds = compiled_words(a_rng, np.asarray(bounds, dtype=np.int64), words)
    for i, d in enumerate(bounds):
        assert int(thresholds[i]) == (1 << 64) % d, d
        expected = np.array([bounded(b_rng, d) for _ in range(words)], dtype=np.int64)
        assert np.array_equal(actual[i], expected), d
        assert np.all((actual[i] >= 0) & (actual[i] < d))
    assert a_rng.bit_generator.state == b_rng.bit_generator.state
    return dict(bounds=len(bounds), samples_per_bound=words,
                exact_raw_word_trajectory=True, maximum_bound=MAX)


def check_dense():
    operations = 0
    batches = 0
    for replacement in (False, True):
        for size in (0, 1, 2, 3, 7, 19, 64):
            for seed in range(12):
                actual = AtomicPositionScan(size, seed, replacement)
                dense = DenseReference(size, seed, replacement)
                scheduler = np.random.default_rng(seed + size * 101)
                for step in range(100):
                    active = list(dense.queries)
                    choice = int(scheduler.integers(0, 7))
                    if not active or choice == 0:
                        assert actual.register() == dense.register()
                    elif choice == 1:
                        count = int(scheduler.integers(0, size - dense.physical + 1))
                        assert np.array_equal(actual.expose(count), dense.expose(count))
                    elif choice == 2 and len(active) > 1:
                        q = int(scheduler.choice(active))
                        actual.stop(q)
                        dense.stop(q)
                    elif choice <= 4:
                        q = int(scheduler.choice(active))
                        capacity = (5 if size else 0) if replacement else len(dense.queries[q][0])
                        count = int(scheduler.integers(0, min(capacity, 5) + 1))
                        equal_result(actual.draw(q, count), dense.draw(q, count))
                    else:
                        queries = list(map(int, scheduler.choice(active, size=int(scheduler.integers(0, len(active) + 1)), replace=False)))
                        capacity = (5 if size else 0) if replacement else min([len(dense.queries[q][0]) for q in queries] + [5])
                        count = int(scheduler.integers(0, min(capacity, 5) + 1))
                        equal_result(actual.draw_epochs(queries, count), dense.draw_epochs(queries, count))
                        batches += 1
                    equal_state(actual, dense)
                    operations += 1
    # Singleton atomic batches must exactly match sequential calls and preserve
    # their RNG trajectories, including pause/expose changes between calls.
    for replacement in (False, True):
        for size in (1, 2, 17):
            a = AtomicPositionScan(size, 44, replacement)
            b = SequentialPositionScan(size, 44, replacement)
            qa, qb = a.register(), b.register()
            for n in (0, min(1, size), size - min(1, size)):
                ao, af, at = a.draw_epochs([qa], n)
                bo, bf = b.draw(qb, n)
                equal_result((ao[0], af[0]), (bo, bf))
                assert np.all(at == 1)
                assert a.physical == b.physical
    # A fixed membership n-round call equals n single-round calls with no
    # intervening observations; all internal random draws must match.
    for replacement in (False, True):
        a = AtomicPositionScan(37, 71, replacement)
        b = AtomicPositionScan(37, 71, replacement)
        qa = [a.register() for _ in range(4)]
        qb = [b.register() for _ in range(4)]
        a.expose(15); b.expose(15)
        a.draw(qa[2], 5); b.draw(qb[2], 5)
        actual = a.draw_epochs(qa[::-1], 20)
        chunks = [b.draw_epoch(qb[::-1]) for _ in range(20)]
        equal_result(actual, tuple(np.stack([x[i] for x in chunks], axis=1) for i in range(3)))
        assert a.physical == b.physical
    return dict(scripted_operations=operations, atomic_batches=batches,
                sizes=[0, 1, 2, 3, 7, 19, 64], seeds_per_size_mode=12,
                modes=["WoR", "WR"], exact_outputs_rng_and_dense_pools=True)


def snapshot(scan):
    return (scan.physical, scan.failed, scan._next, repr(scan._seeds.state),
            {q: (s["remaining"], dict(s["forward"]), dict(s["inverse"]),
                 repr(s["rng"].bit_generator.state)) for q, s in scan._queries.items()})


def expect_error(call, kinds=(ValueError, RuntimeError)):
    try:
        call()
    except kinds:
        return
    raise AssertionError("request did not fail")


def check_boundaries():
    invalid = 0
    bad_ints = [-1, 1.0, True, np.bool_(False), None, float("nan"), "1", np.uint64(1 << 63)]
    for bad in bad_ints:
        expect_error(lambda bad=bad: SequentialPositionScan(bad))
        invalid += 1
    for seed in (-1, True, 1.0, 1 << 64):
        expect_error(lambda seed=seed: SequentialPositionScan(1, seed))
        invalid += 1
    expect_error(lambda: SequentialPositionScan(1, replacement=1))
    invalid += 1
    for replacement in (False, True):
        s = AtomicPositionScan(7, seed=np.uint64(MASK), replacement=np.bool_(replacement))
        q, other = s.register(), s.register()
        s.expose(np.uint64(2)); s.draw(np.uint64(q), np.uint64(2))
        s.stop(other)
        calls = []
        for bad in bad_ints:
            calls.extend([lambda bad=bad: s.expose(bad), lambda bad=bad: s.draw(q, bad),
                          lambda bad=bad: s.draw_epochs([q], bad),
                          lambda bad=bad: s.draw(bad, 0), lambda bad=bad: s.stop(bad),
                          lambda bad=bad: s.query_entries(bad)])
        calls.extend([lambda: s.expose(8), lambda: s.draw(1234), lambda: s.draw(other, 0),
                      lambda: s.stop(other), lambda: s.draw_epochs([q, q], 0),
                      lambda: s.draw_epochs([q, 1234], 0), lambda: s.draw_epochs(None),
                      lambda: s.draw_epochs(np.array([[q]]), 0),
                      lambda: s.draw(q, MAX), lambda: s.draw_epochs([q], MAX),
                      lambda: s.draw_epochs([], MAX),
                      lambda: s.expose(1, return_positions=1),
                      lambda: s.expose(1, return_positions=None),
                      lambda: s.expose(-1, return_positions=False),
                      lambda: s.expose(8, return_positions=False)])
        if not replacement:
            calls += [lambda: s.draw(q, 6), lambda: s.draw_epochs([q], 6)]
        before = snapshot(s)
        for call in calls:
            expect_error(call)
            assert snapshot(s) == before
            invalid += 1
        s.expose(s.size - s.physical)
        if replacement:
            out, fresh = s.draw(q, 100)
            assert len(out) == 100 and not fresh.any()
        else:
            out, fresh = s.draw(q, 5)
            assert len(set(map(int, out))) == 5 and not fresh.any()
            expect_error(lambda: s.draw(q, 1))
        assert s.draw(q, 0)[0].shape == (0,)
        assert s.draw_epochs([], 4)[0].shape == (0, 4)
        assert s.draw_epochs([q], 0)[0].shape == (1, 0)
        s.stop(q)
    for cls in (SequentialPositionScan, AtomicPositionScan):
        for replacement in (False, True):
            s = cls(0, 1, replacement)
            q = s.register()
            assert s.expose(0).size == 0
            assert s.expose(0, return_positions=False) is None
            assert s.draw(q, 0)[0].size == 0
            expect_error(lambda: s.draw(q, 1))
            if isinstance(s, AtomicPositionScan):
                assert s.draw_epochs([], 7)[0].shape == (0, 7)
                assert s.draw_epoch([])[0].shape == (0,)
                expect_error(lambda: s.draw_epochs([q], 1))
            assert not s.failed
    # Huge-N construction/registration/draws allocate only tiny sparse state.
    huge = AtomicPositionScan(np.uint64(MAX), np.uint64(MASK))
    queries = [huge.register() for _ in range(3)]
    huge.expose(np.uint64(3))
    out, fresh, trials = huge.draw_epochs(np.asarray(queries, dtype=np.uint64), 30)
    assert out.dtype == np.int64 and huge.physical <= 33
    assert all(huge.query_entries(q) <= 60 for q in queries)
    assert not hasattr(huge, "_order")
    # No-output exposure must not create an H-sized address array, including
    # a census-sized virtual address pool for an external indexed file arm.
    no_output_entries = []
    original_arange = engine.np.arange
    def forbidden_arange(*args, **kwargs):
        raise AssertionError("no-output exposure called arange")
    try:
        engine.np.arange = forbidden_arange
        for replacement in (False, True):
            s = AtomicPositionScan(np.uint64(MAX), seed=8, replacement=replacement)
            q = s.register()
            assert s.expose(np.uint64(MAX), return_positions=np.bool_(False)) is None
            assert s.physical == MAX and not s.failed
            out, fresh = s.draw(q, 30)
            assert np.all((0 <= out) & (out < MAX)) and not fresh.any()
            if not replacement:
                assert len(set(map(int, out))) == 30
            no_output_entries.append(s.query_entries(q))
    finally:
        engine.np.arange = original_arange
    for prop in ("physical", "size", "replacement", "failed"):
        expect_error(lambda prop=prop: setattr(huge, prop, 4), (AttributeError,))
    return dict(invalid_requests_without_mutation=invalid,
                empty_census_unsigned_pause_expose_and_batch=True,
                huge_size=MAX, huge_queries=3, huge_epochs=30,
                huge_map_entries=[huge.query_entries(q) for q in queries],
                no_output_exposure_to_census=True,
                no_output_census_map_entries_wor_wr=no_output_entries)


def check_fail_stop():
    cases = 0
    def disabled(s, q):
        state = snapshot(s)
        for call in (s.register, lambda: s.expose(0), lambda: s.draw(q, 0),
                     lambda: s.stop(q), lambda: s.draw_epochs([], 0),
                     lambda: s.draw_epoch([])):
            expect_error(call, (RuntimeError,))
            assert snapshot(s) == state

    for kernel_name in ("_sequential_kernel", "_atomic_kernel"):
        original = getattr(engine, kernel_name)
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit, BaseException):
            s = AtomicPositionScan(8, 12)
            q = s.register()
            error = error_type("injected after partial mutation")
            def broken(*args, _error=error, **kwargs):
                args[1][0] += 1
                s._queries[q]["forward"][0] = 7
                s._queries[q]["rng"].bit_generator.random_raw()
                raise _error
            setattr(engine, kernel_name, broken)
            try:
                try:
                    s.draw(q, 1) if kernel_name == "_sequential_kernel" else s.draw_epochs([q], 1)
                except BaseException as caught:
                    assert caught is error
                else:
                    raise AssertionError("injected exception was swallowed")
                assert s.failed and s.physical == 1
                assert s.query_entries(q) == 1
                disabled(s, q)
                cases += 1
            finally:
                setattr(engine, kernel_name, original)
    for operation in ("register", "expose", "expose_no_output", "stop"):
        for error_type in (RuntimeError, KeyboardInterrupt, SystemExit, BaseException):
            s = AtomicPositionScan(8, 12)
            q = s.register()
            error = error_type("injected after partial mutation")
            if operation == "register":
                original_seeds = s._seeds
                class BrokenSeeds:
                    @property
                    def state(self):
                        return original_seeds.state
                    def spawn(self, n):
                        original_seeds.spawn(n)
                        raise error
                s._seeds = BrokenSeeds()
                call = s.register
            elif operation == "expose":
                original_arange = engine.np.arange
                def broken_arange(*args, **kwargs):
                    s._cursor[0] += 1
                    raise error
                engine.np.arange = broken_arange
                call = lambda: s.expose(1)
            elif operation == "expose_no_output":
                original_cursor = s._cursor
                class BrokenCursor:
                    def __getitem__(self, key):
                        return original_cursor[key]
                    def __setitem__(self, key, value):
                        original_cursor[key] = value
                        raise error
                s._cursor = BrokenCursor()
                call = lambda: s.expose(1, return_positions=False)
            else:
                class BrokenQueries(dict):
                    def __delitem__(self, key):
                        super().__delitem__(key)
                        raise error
                s._queries = BrokenQueries(s._queries)
                call = lambda: s.stop(q)
            try:
                try:
                    call()
                except BaseException as caught:
                    assert caught is error
                else:
                    raise AssertionError("injected exception was swallowed")
                assert s.failed
                disabled(s, q)
                cases += 1
            finally:
                if operation == "expose":
                    engine.np.arange = original_arange
    return dict(partially_mutating_failure_cases=cases,
                exception_identity_preserved=True,
                keyboard_interrupt_system_exit_and_base_exception=True,
                future_write_apis_disabled=True, rollback_claimed=False)


def powerset(values):
    values = tuple(values)
    return [set(c) for k in range(len(values) + 1) for c in itertools.combinations(values, k)]


def check_exact_epochs():
    states = 0
    for q_count, sizes in ((2, range(1, 6)), (3, range(1, 4))):
        for size in sizes:
            for h in range(size + 1):
                possible = [s for s in powerset(range(h)) if len(s) < size]
                for used in itertools.product(possible, repeat=q_count):
                    ds = [size - len(s) for s in used]
                    leader = min(range(q_count), key=lambda i: ds[i])
                    caches = [set(range(h)) - s for s in used]
                    u = size - h
                    lead_options = [(p, False, Fraction(1, ds[leader])) for p in caches[leader]]
                    if u:
                        lead_options.append((h, True, Fraction(u, ds[leader])))
                    marginals = [Counter() for _ in range(q_count)]
                    total = Fraction(0)
                    for lp, new, lead_prob in lead_options:
                        options = []
                        for i in range(q_count):
                            if i == leader:
                                options.append([(lp, Fraction(1))])
                                continue
                            accept = Fraction(ds[leader], ds[i]) if new else Fraction(0)
                            choices = [(h, accept)] if accept else []
                            if accept < 1:
                                assert caches[i], (size, h, used, i)
                                choices.extend((p, (1 - accept) / len(caches[i])) for p in caches[i])
                            options.append(choices)
                        for outcomes in itertools.product(*options):
                            probability = lead_prob * math.prod(item[1] for item in outcomes)
                            total += probability
                            for i, item in enumerate(outcomes):
                                marginals[i][item[0]] += probability
                    assert total == 1
                    for i in range(q_count):
                        expected = Counter({p: Fraction(1, ds[i]) for p in caches[i]})
                        if u:
                            expected[h] = Fraction(u, ds[i])
                        assert marginals[i] == expected, (size, h, used, marginals, expected)
                    states += 1
    return dict(ideal_rational_states=states, pair_sizes_through=5,
                triple_sizes_through=3, all_prefixes_and_used_subsets=True)


def check_exact_source_sequences():
    configurations = 0
    for replacement in (False, True):
        for size in range(1, 5):
            length = 3 if replacement else size
            counts = Counter()
            def visit(order, pool, h, sequence, probability):
                if len(sequence) == length:
                    counts[tuple(sequence)] += probability
                    return
                for virtual in pool:
                    position = h if virtual >= h else virtual
                    new_h = h + int(virtual >= h)
                    next_pool = pool if replacement else [p for p in pool if p != position]
                    visit(order, next_pool, new_h, sequence + [order[position]], probability / len(pool))
            for order in itertools.permutations(range(size)):
                visit(order, list(range(size)), 0, [], Fraction(1, math.factorial(size)))
            expected_count = size ** length if replacement else math.factorial(size)
            assert len(counts) == expected_count
            assert set(counts.values()) == {Fraction(1, expected_count)}
            configurations += 1
    return dict(configurations=configurations, sizes_through=4,
                wor_complete_sequences=True, wr_sequence_length=3,
                hidden_uniform_source_permutations_enumerated=True)


def check_statistics(repetitions):
    results = []
    size = 5
    for replacement in (False, True):
        counts = np.zeros((2, size), dtype=np.int64)
        pairs = Counter()
        for seed in range(repetitions):
            s = AtomicPositionScan(size, seed + 700000, replacement)
            a, b = s.register(), s.register()
            s.expose(size)
            first = int(s.draw(a, 1)[0][0])
            out, fresh, _ = s.draw_epochs([a, b], 1)
            assert not fresh.any()
            counts[0, out[0, 0]] += 1
            counts[1, out[1, 0]] += 1
            pairs[first, int(out[0, 0])] += 1
        expected = repetitions / size
        chi = [float(np.square(row - expected).sum() / expected) for row in counts]
        assert max(chi) < 40  # conservative df=4 smoke threshold
        categories = size * size if replacement else size * (size - 1)
        expected_pair = repetitions / categories
        pair_chi = sum((pairs[i, j] - expected_pair) ** 2 / expected_pair
                       for i in range(size) for j in range(size) if replacement or i != j)
        assert pair_chi < 100
        if not replacement:
            assert not any(pairs[i, i] for i in range(size))
        results.append(dict(mode="WR" if replacement else "WoR", repetitions=repetitions,
                            marginal_counts=counts.tolist(), marginal_chi_square=chi,
                            query_a_ordered_pair_chi_square=pair_chi,
                            thresholds=dict(marginal=40, ordered_pair=100)))
    return results


def check_partial_prefix_statistics(repetitions):
    results = []
    size = 7
    for replacement in (False, True):
        observed = np.zeros((2, size), dtype=np.int64)
        expected = np.zeros((2, size), dtype=float)
        variance = np.zeros((2, size), dtype=float)
        observed_fresh = np.zeros(2, dtype=np.int64)
        expected_fresh = np.zeros(2, dtype=float)
        variance_fresh = np.zeros(2, dtype=float)
        for seed in range(repetitions):
            s = AtomicPositionScan(size, seed + 900000, replacement)
            a, b = s.register(), s.register()
            s.expose(3)
            earlier = set(map(int, s.draw(a, 2)[0]))
            h = s.physical
            for i in range(2):
                used = earlier if i == 0 and not replacement else set()
                d = size - len(used)
                probabilities = np.zeros(size, dtype=float)
                for p in set(range(h)) - used:
                    probabilities[p] = 1 / d
                if h < size:
                    probabilities[h] = (size - h) / d
                expected[i] += probabilities
                variance[i] += probabilities * (1 - probabilities)
                p_new = (size - h) / d
                expected_fresh[i] += p_new
                variance_fresh[i] += p_new * (1 - p_new)
            out, fresh, _ = s.draw_epochs([a, b], 1)
            for i in range(2):
                observed[i, out[i, 0]] += 1
                observed_fresh[i] += int(fresh[i, 0])
        z = np.divide(observed - expected, np.sqrt(variance),
                      out=np.zeros_like(expected), where=variance > 0)
        fresh_z = (observed_fresh - expected_fresh) / np.sqrt(variance_fresh)
        assert np.max(np.abs(z)) < 7
        assert np.max(np.abs(fresh_z)) < 7
        results.append(dict(mode="WR" if replacement else "WoR", repetitions=repetitions,
                            initial_prefix=3, query_a_prior_draws=2,
                            observed_position_counts=observed.tolist(),
                            conditional_expected_position_counts=expected.tolist(),
                            maximum_absolute_position_z=float(np.max(np.abs(z))),
                            observed_fresh=observed_fresh.tolist(),
                            conditional_expected_fresh=expected_fresh.tolist(),
                            fresh_z=fresh_z.tolist(), threshold_absolute_z=7))
    return results


def check_compilation_and_benchmark():
    engine.warmup()
    assert engine._atomic_kernel.nopython_signatures
    assert engine._sequential_kernel.nopython_signatures
    size, n, q_count = 1_000_000, 25000, 4
    results = []
    for replacement in (False, True):
        s = AtomicPositionScan(size, 99, replacement)
        queries = [s.register() for _ in range(q_count)]
        s.expose(1000)
        started = time.perf_counter()
        out, fresh, trials = s.draw_epochs(queries, n)
        elapsed = time.perf_counter() - started
        assert out.shape == (q_count, n)
        assert fresh.sum(axis=0).max() <= q_count
        assert s.physical <= 1000 + n
        results.append(dict(mode="WR" if replacement else "WoR", size=size,
                            participants=q_count, epochs=n, seconds=elapsed,
                            observations_per_second=q_count * n / elapsed,
                            physical=s.physical, maximum_trials=int(trials.max()),
                            map_entries=[s.query_entries(q) for q in queries]))
    return dict(nopython_atomic_signatures=[str(x) for x in engine._atomic_kernel.nopython_signatures],
                nopython_sequential_signatures=[str(x) for x in engine._sequential_kernel.nopython_signatures],
                warmed_single_run_benchmarks=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    report = dict(environment=dict(python=sys.version, numpy=np.__version__,
                                   numba=numba.__version__, platform=platform.platform()),
                  limits=["Finite PCG64 seeds are not ideal independent uniform coins.",
                          "Numba generator_core.next_uint64 is an internal version dependency.",
                          "Atomic sharing introduces allowed within-epoch dependence.",
                          "Cache rejection can require arbitrarily many proposals.",
                          "Serialized use only; fail-stop provides no rollback.",
                          "This validates the position engine, not source shuffling, file I/O, ID decoding, policy isolation, or OS I/O savings."])
    started = time.perf_counter()
    for name, check in [("rng_boundaries", check_rng), ("dense_reference", check_dense),
                        ("public_boundaries", check_boundaries), ("fail_stop", check_fail_stop),
                        ("exact_epoch_enumeration", check_exact_epochs),
                        ("exact_source_sequences", check_exact_source_sequences),
                        ("statistics", lambda: check_statistics(args.repetitions)),
                        ("partial_prefix_statistics", lambda: check_partial_prefix_statistics(args.repetitions)),
                        ("compilation_and_benchmark", check_compilation_and_benchmark)]:
        before = time.perf_counter()
        report[name] = check()
        print(f"PASS {name} ({time.perf_counter() - before:.3f}s)", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["passed"] = True
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
