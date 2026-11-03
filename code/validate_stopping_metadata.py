"""Small, exact checks for stopping_metadata_coupling.py; no heavy study.

This checks actual finite RNG ticket paths with Fraction, invalid inputs,
source exposure, and immutable sampler state. Seeded cost summaries are only
smoke observations, never a proof of the general value or an empirical test
with a probability tolerance. Run: python code/validate_stopping_metadata.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import FrozenInstanceError
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import random
from typing import Callable, Iterable

from stopping_metadata_coupling import (
    FiniteStoppingCoupling,
    FullRecord,
    IntegerAlias,
    MeteredOneUnreadSource,
    execute_two_observation,
)


class Scripted:
    def __init__(self, values: Iterable[int]) -> None:
        self.values = iter(values)
        self.bounds: list[int] = []

    def __call__(self, bound: int) -> int:
        self.bounds.append(bound)
        return next(self.values)

    def exhausted(self) -> bool:
        sentinel = object()
        return next(self.values, sentinel) is sentinel


def expect_error(error: type[Exception], action: Callable[[], object]) -> None:
    try:
        action()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def exact_alias(weights: tuple[int, ...]) -> int:
    table = IntegerAlias.from_weights(weights)
    counts = [0] * len(weights)
    for column in range(len(weights)):
        for coin in range(table.total):
            rng = Scripted((column, coin))
            counts[table.draw(rng)] += 1
            assert rng.bounds == [len(weights), sum(weights)]
            assert rng.exhausted()
    assert counts == [len(weights) * value for value in weights]
    return len(weights) * table.total


def exact_cached_law(sampler: FiniteStoppingCoupling) -> tuple[dict, int]:
    """Enumerate actual draw_cached paths in the selected tiny test cases.

    All residual mixture tickets execute the identical code. Verify one
    representative ticket for every combination of alias column/coin inputs,
    then multiply its mass by the number W of identical tickets.
    """
    law: dict[tuple[int, ...], Fraction] = defaultdict(Fraction)
    paths = 0
    for ticket, target in enumerate(sampler.targets):
        rng = Scripted((ticket,))
        actual = sampler.draw_cached(rng)
        assert actual == target and rng.exhausted()
        assert rng.bounds == [sampler.choice_count]
        law[actual] += Fraction(1, sampler.choice_count)
        paths += 1
    w = sampler.residual_choices
    if w:
        per_coordinate = tuple(product(range(sampler.m), range(w)))
        mass = Fraction(w, sampler.choice_count * (sampler.m * w) ** sampler.k)
        for pairs in product(per_coordinate, repeat=sampler.k):
            flat = [value for pair in pairs for value in pair]
            rng = Scripted((len(sampler.targets), *flat))
            actual = sampler.draw_cached(rng)
            assert len(actual) == sampler.k
            assert rng.bounds == [sampler.choice_count] + [sampler.m, w] * sampler.k
            assert rng.exhausted()
            law[actual] += mass
            paths += 1
        assert all(table.total == w for table in sampler.residual_tables)
    else:
        assert sampler.residual_tables == ()
    return dict(law), paths


def check_family(m: int, family: Iterable[Iterable[int]]) -> dict:
    sampler = FiniteStoppingCoupling(m, family)
    law, paths = exact_cached_law(sampler)
    assert sum(law.values()) == 1
    for q in range(sampler.k):
        for a in range(m):
            assert sum(p for row, p in law.items() if row[q] == a) == Fraction(1, m)
            degree = sum(target[q] == a for target in sampler.targets)
            assert sampler.degrees[q][a] == degree
    assert sampler.delta == max(
        sum(target[q] == a for target in sampler.targets)
        for q in range(sampler.k) for a in range(m)
    )
    assert min(law.get(target, Fraction(0)) for target in sampler.targets) == Fraction(
        1, m * sampler.delta
    )
    costs = {
        target: (2 - law.get(target, Fraction(0))) / (m + 1)
        for target in sampler.targets
    }
    assert max(costs.values()) == sampler.worst_total_cost
    assert sampler.oracle_total_cost == (2 - Fraction(1, m)) / (m + 1)

    # Verify actual first-plan tickets separately; only read ticket zero reads.
    read_rng = Scripted((0,))
    read_plan = sampler.draw_first(read_rng)
    assert read_plan.read_required is True and read_plan.indices == (m,) * sampler.k
    assert read_rng.bounds == [m + 1] and read_rng.exhausted()
    for first_ticket in range(1, m + 1):
        rng = Scripted((first_ticket, 0))
        cached_plan = sampler.draw_first(rng)
        assert cached_plan.read_required is False
        assert cached_plan.indices == sampler.targets[0]
        assert rng.bounds == [m + 1, sampler.choice_count] and rng.exhausted()
    if len(sampler.targets) == m ** sampler.k:
        assert len(law) == m ** sampler.k
        assert set(law.values()) == {Fraction(1, m ** sampler.k)}
    if sampler.delta == 1:
        assert set(costs.values()) == {sampler.oracle_total_cost}
    return {
        "m": m, "k": sampler.k, "family_size": len(sampler.targets),
        "delta": sampler.delta, "residual_choices": sampler.residual_choices,
        "exact_cached_paths": paths, "worst_total_cost": str(max(costs.values())),
    }


def boundary_checks() -> int:
    checks = 0
    invalid_families = (
        (ValueError, 0, [(0,)]), (ValueError, -1, [(0,)]),
        (TypeError, True, [(0,)]), (TypeError, 2.0, [(0,)]),
        (ValueError, 2, []), (ValueError, 2, [()]),
        (ValueError, 2, [(0,), (0, 1)]),
        (ValueError, 2, [(0,), (0,)]),
        (ValueError, 2, [(-1,)]), (ValueError, 2, [(2,)]),
        (TypeError, 2, [(True,)]), (TypeError, 2, [(0.0,)]),
        (TypeError, 2, [("0",)]), (TypeError, 2, [1]),
        (TypeError, 2, None),
    )
    for error, m, family in invalid_families:
        expect_error(error, lambda m=m, family=family: FiniteStoppingCoupling(m, family))
        checks += 1
    for weights, error in (((), ValueError), ((0, 0), ValueError),
                           ((-1, 1), ValueError), ((True,), TypeError),
                           ((1.0,), TypeError)):
        expect_error(error, lambda weights=weights: IntegerAlias.from_weights(weights))
        checks += 1

    # Public dataclass construction must preserve the same structural invariant.
    invalid_tables = (
        (ValueError, 0, (0,), (0,)), (ValueError, -1, (0,), (0,)),
        (TypeError, True, (0,), (0,)), (TypeError, 1.0, (0,), (0,)),
        (ValueError, 1, (), ()), (ValueError, 1, (0, 1), (0,)),
        (ValueError, 1, (0,), (7,)),  # Original reviewer counterexample.
        (ValueError, 1, (-1,), (0,)), (ValueError, 1, (2,), (0,)),
        (TypeError, 1, (True,), (0,)), (TypeError, 1, (0.0,), (0,)),
        (ValueError, 1, (0,), (-1,)), (ValueError, 1, (0,), (1,)),
        (TypeError, 1, (0,), (True,)), (TypeError, 1, (0,), (0.0,)),
        (TypeError, 1, None, (0,)), (TypeError, 1, (0,), None),
    )
    for error, total, cutoffs, aliases in invalid_tables:
        expect_error(error, lambda total=total, cutoffs=cutoffs, aliases=aliases:
                     IntegerAlias(total, cutoffs, aliases))
        checks += 1
    external_cutoffs, external_aliases = [0, 2], [1, 0]
    direct = IntegerAlias(2, external_cutoffs, external_aliases)
    external_cutoffs[0] = 99
    external_cutoffs.append(3)
    external_aliases.clear()
    assert direct.cutoffs == (0, 2) and direct.aliases == (1, 0)
    assert direct.draw(Scripted((0, 0))) == 1
    expect_error(FrozenInstanceError, lambda: setattr(direct, "cutoffs", (2, 2)))
    checks += 2

    # Single-pass iterables are accepted and copied, not retained as mutable input.
    mutable = [[0, 0]]
    sampler = FiniteStoppingCoupling(3, (iter(row) for row in mutable))
    mutable[0][0] = 2
    assert sampler.targets == ((0, 0),)
    snapshot = repr(sampler)
    expect_error(FrozenInstanceError, lambda: setattr(sampler, "delta", 99))
    checks += 2

    for bad, error in ((True, TypeError), (0.0, TypeError), (-1, ValueError),
                       (sampler.population_size, ValueError)):
        expect_error(error, lambda bad=bad: sampler.draw_first(lambda _: bad))
        assert repr(sampler) == snapshot
        checks += 1
    # A bad residual-branch RNG return is also rejected after earlier valid draws.
    expect_error(TypeError, lambda: sampler.draw_cached(Scripted((1, 0, True))))
    assert repr(sampler) == snapshot
    checks += 1

    class RNGFailure(Exception):
        pass

    class FailAfter:
        def __init__(self, prefix: Iterable[int]) -> None:
            self.prefix = iter(prefix)

        def __call__(self, _: int) -> int:
            try:
                return next(self.prefix)
            except StopIteration:
                raise RNGFailure("intentional RNG failure")

    for prefix in ((), (1,), (1, 1), (1, 1, 0)):
        expect_error(RNGFailure, lambda prefix=prefix: sampler.draw_first(FailAfter(prefix)))
        assert repr(sampler) == snapshot
        checks += 1
    # A successful later draw needs no reset and matches a fresh sampler.
    assert sampler.draw_first(Scripted((1, 0))) == FiniteStoppingCoupling(
        3, [(0, 0)]
    ).draw_first(Scripted((1, 0)))
    checks += 1
    return checks


def fixture_records(m: int) -> tuple[tuple[FullRecord, ...], FullRecord]:
    cache = tuple(FullRecord(f"cache-{a}", f"actual cached payload {a}".encode())
                  for a in range(m))
    unread = FullRecord("z", b"actual next-source full payload, unavailable before next")
    return cache, unread


def full_record_checks() -> int:
    checks = 0
    sampler = FiniteStoppingCoupling(2, [(0,)])
    cache, unread = fixture_records(2)

    class SpyIterator:
        def __init__(self, record: FullRecord) -> None:
            self.record = record
            self.calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls != 1:
                raise AssertionError("unexpected second next-source call")
            return self.record

    spy = SpyIterator(unread)
    source = MeteredOneUnreadSource(spy)
    result = execute_two_observation(sampler, cache, source, (0,), Scripted((0, 1)))
    assert result.first[0] is unread and result.second[0] is cache[1]
    assert spy.calls == source.next_calls == source.source_reads == result.source_reads == 1
    expect_error(RuntimeError, lambda: next(source))
    assert spy.calls == 1
    checks += 2

    # Cached hit: neither source iteration nor a second observation is needed.
    source = MeteredOneUnreadSource(SpyIterator(unread))
    result = execute_two_observation(sampler, cache, source, (0,), Scripted((1, 0)))
    assert result.first[0] is cache[0] and result.second == (None,)
    assert source.next_calls == source.source_reads == 0
    checks += 1

    # Cached miss: residual alias returns label 1; aligned-z continuation reads.
    source = MeteredOneUnreadSource((unread,))
    result = execute_two_observation(
        sampler, cache, source, (0,), Scripted((1, 1, 1, 0, 0))
    )
    assert result.first[0] is cache[1] and result.second[0] is unread
    assert result.first_plan.read_required is False and result.source_reads == 1
    checks += 1

    # The other continuation branch uses the unused cached record without reading.
    source = MeteredOneUnreadSource((unread,))
    result = execute_two_observation(
        sampler, cache, source, (0,), Scripted((1, 1, 1, 0, 1, 0))
    )
    assert result.first[0] is cache[1] and result.second[0] is cache[0]
    assert result.source_reads == source.next_calls == 0
    checks += 1

    # An ID-only source is rejected after its actual yielded item is charged.
    for forged in ("z", 2, {"hidden_id": "z"}):
        source = MeteredOneUnreadSource((forged,))
        expect_error(TypeError, lambda: execute_two_observation(
            sampler, cache, source, (0,), Scripted((0,))
        ))
        assert source.next_calls == source.source_reads == 1
        checks += 1
    source = MeteredOneUnreadSource(())
    expect_error(StopIteration, lambda: execute_two_observation(
        sampler, cache, source, (0,), Scripted((0,))
    ))
    assert source.next_calls == 1 and source.source_reads == 0
    checks += 1
    source = MeteredOneUnreadSource((cache[0],))
    expect_error(ValueError, lambda: execute_two_observation(
        sampler, cache, source, (0,), Scripted((0,))
    ))
    assert source.next_calls == source.source_reads == 1
    checks += 1
    for invalid_cache, error in ((("a", "b"), TypeError), ((cache[0],), ValueError),
                                 ((cache[0], cache[0]), ValueError)):
        source = MeteredOneUnreadSource((unread,))
        expect_error(error, lambda invalid_cache=invalid_cache: execute_two_observation(
            sampler, invalid_cache, source, (0,), Scripted((0,))
        ))
        assert source.next_calls == 0
        checks += 1
    source = MeteredOneUnreadSource((unread,))
    expect_error(TypeError, lambda: execute_two_observation(
        sampler, cache, source, (True,), Scripted((0,))
    ))
    assert source.next_calls == 0
    checks += 1
    expect_error(TypeError, lambda: FullRecord("z", "ID-only, missing bytes payload"))
    checks += 1
    source = MeteredOneUnreadSource((unread,))
    next(source)
    expect_error(ValueError, lambda: execute_two_observation(
        sampler, cache, source, (0,), Scripted((0,))
    ))
    assert source.next_calls == source.source_reads == 1
    checks += 1
    return checks


def exact_execution_checks() -> list[dict]:
    """Enumerate small actual full-record executions, including continuation."""
    class NeedDraw(Exception):
        def __init__(self, bound: int) -> None:
            self.bound = bound

    cases = (
        (1, [(0,)], (0,)),
        (2, [(0,)], (0,)),
        (2, [(0, 0), (1, 1)], (0, 0)),
        (2, list(product(range(2), repeat=2)), (0, 1)),
        (3, [(0,)], (0,)),
    )
    summaries = []
    for m, family, target in cases:
        sampler = FiniteStoppingCoupling(m, family)
        cache, unread = fixture_records(m)
        label_by_identity = {id(record): a for a, record in enumerate((*cache, unread))}
        joint: dict[tuple, Fraction] = defaultdict(Fraction)
        total_cost = Fraction(0)
        leaves = 0

        def visit(prefix: tuple[int, ...], probability: Fraction) -> None:
            nonlocal total_cost, leaves
            consumed = 0

            def draw(bound: int) -> int:
                nonlocal consumed
                if consumed == len(prefix):
                    raise NeedDraw(bound)
                value = prefix[consumed]
                consumed += 1
                return value

            source = MeteredOneUnreadSource((unread,))
            try:
                result = execute_two_observation(sampler, cache, source, target, draw)
            except NeedDraw as request:
                for value in range(request.bound):
                    visit((*prefix, value), probability / request.bound)
                return
            assert consumed == len(prefix)
            assert source.next_calls == source.source_reads == result.source_reads
            first = tuple(label_by_identity[id(record)] for record in result.first)
            second = tuple(None if record is None else label_by_identity[id(record)]
                           for record in result.second)
            joint[(first, second)] += probability
            total_cost += probability * result.source_reads
            leaves += 1

        visit((), Fraction(1))
        assert sum(joint.values()) == 1
        first_law: dict[tuple, Fraction] = defaultdict(Fraction)
        for (first, _), probability in joint.items():
            first_law[first] += probability
        for q in range(sampler.k):
            for a in range(m + 1):
                assert sum(p for row, p in first_law.items() if row[q] == a) == Fraction(
                    1, m + 1
                )
        for first, first_probability in first_law.items():
            for q in range(sampler.k):
                if first[q] == target[q]:
                    assert all(second[q] is None for row, second in joint if row == first)
                else:
                    for unused in set(range(m + 1)) - {first[q]}:
                        assert sum(p for (row, second), p in joint.items()
                                   if row == first and second[q] == unused) == first_probability / m
        cached_law, _ = exact_cached_law(sampler)
        assert total_cost == (2 - cached_law.get(target, Fraction(0))) / (m + 1)
        summaries.append({
            "m": m, "k": sampler.k, "family_size": len(sampler.targets),
            "complete_rng_paths": leaves, "exact_actual_source_cost": str(total_cost),
            "checked": "actual next, first marginals, conditional second marginals, total cost",
        })
    return summaries


def seeded_cost_smoke() -> list[dict]:
    summaries = []
    cases = (
        (1, [(0, 0, 0)], (0, 0, 0), 1601),
        (2, [(0,)], (0,), 1602),
        (2, list(product(range(2), repeat=2)), (0, 1), 1603),
    )
    for m, family, target, seed in cases:
        sampler = FiniteStoppingCoupling(m, family)
        cache, unread = fixture_records(m)
        rng = random.Random(seed)
        reads = 0
        trials = 256
        for _ in range(trials):
            source = MeteredOneUnreadSource((unread,))
            result = execute_two_observation(sampler, cache, source, target, rng.randrange)
            assert result.source_reads in (0, 1)
            assert source.next_calls == source.source_reads == result.source_reads
            population = (*cache, unread)
            for q, first in enumerate(result.first):
                assert any(first is record for record in population)
                second = result.second[q]
                if first is cache[target[q]]:
                    assert second is None
                else:
                    assert second is not None and second is not first
                    assert any(second is record for record in population)
            if result.source_reads == 0:
                assert all(record is not unread for record in result.first)
                assert all(record is not unread for record in result.second)
            reads += result.source_reads
        summaries.append({
            "m": m, "k": sampler.k, "family_size": len(family), "seed": seed,
            "trials": trials, "actual_next_source_reads": reads,
            "observed_mean": str(Fraction(reads, trials)),
            "exact_task_cost": str(sampler.worst_total_cost),
            "interpretation": "seeded fixture smoke only; no statistical tolerance or proof",
        })
    return summaries


def run_checks() -> dict:
    alias_paths = sum(exact_alias(weights) for weights in (
        (1,), (0, 2, 1), (0, 0, 5), (2, 2), (4, 0, 1, 3),
    ))
    families: list[tuple[int, Iterable[Iterable[int]]]] = [
        (1, [(0,)]), (1, [(0, 0, 0)]),
        (2, [(0, 0, 0)]),
        (2, [(0, 0, 0), (1, 1, 1)]),
        (2, list(product(range(2), repeat=3))),
        (3, [(0, 0)]),
        (3, [(0, 0), (1, 1), (2, 2)]),
        (3, list(product(range(3), repeat=2))),
        (3, [(0, 0), (0, 1), (1, 2), (2, 2)]),
        (3, [(0, 0), (0, 1), (0, 2)]),
    ]
    # All 15 nonempty subsets for m=2,K=2, plus all K=1 subsets for m=2,3.
    for m, k in ((2, 2), (2, 1), (3, 1)):
        universe = tuple(product(range(m), repeat=k))
        for size in range(1, len(universe) + 1):
            families.extend((m, subset) for subset in combinations(universe, size))
    exact_cases = [check_family(m, family) for m, family in families]
    boundaries = boundary_checks()
    full_records = full_record_checks()
    exact_executions = exact_execution_checks()
    smoke = seeded_cost_smoke()
    return {
        "status": "PASS",
        "scope": "standalone finite-family two-observation reference algorithm only",
        "exact_family_cases": len(exact_cases),
        "exact_cached_paths": sum(case["exact_cached_paths"] for case in exact_cases),
        "exact_standalone_alias_paths": alias_paths,
        "boundary_and_rng_checks": boundaries,
        "full_record_interface_checks": full_records,
        "exact_full_record_executions": exact_executions,
        "cases": exact_cases,
        "seeded_cost_smoke": smoke,
        "limitations": [
            "small exact implementation checks do not prove the general theorem",
            "no full-AQP, multi-unread-record, longer-horizon, or production-driver claim",
            "no heavy Monte Carlo, Numba, large-file I/O, or existing driver was run",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional small JSON results file")
    args = parser.parse_args()
    result = run_checks()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
