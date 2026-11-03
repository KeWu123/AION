"""Exact reference sampler for one finite two-observation stopping family.

This is a standalone reference algorithm, not a full AQP implementation.
Cached records have index labels 0,...,m-1; the sole unread record has the
*planning symbol* m. A FirstPlan does not expose or return that unread record.
Only the small full-record execution fixture below calls next(source) and
can turn that symbol into a returned full record.

For a nonempty family F of distinct K-tuples, define Delta=max(degree),
D=m*Delta and W=D-|F|. Conditional on the cached branch, draw a uniform
integer in range(D). Its first |F| choices select the corresponding F tuple;
the remaining W choices select independent residual coordinate labels with
integer weights Delta-degree(q,a). Integer alias tables make each residual
coordinate require exactly two randbelow calls. There are no floating-point
probabilities, CDFs, rejection loops, candidate evaluations, or source reads
in the planner.

For explicitly supplied F, preprocessing takes O(K*|F| + K*m) integer
operations and storage O(K*|F| + K*m), including retained input tuples.
A first draw takes at most 2+2*K random-integer operations and O(K) output
work after preprocessing. Integer bit costs and the RNG's own bit cost are
additional. The exact worst total extra-source cost is
(2 - 1/(m*Delta))/(m+1), only for this stated stopping family and interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Iterator


RandBelow = Callable[[int], int]


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    """Require a built-in Python integer; in particular reject booleans."""
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, not {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _draw(randbelow: RandBelow, bound: int) -> int:
    """Check an external RNG result without storing or changing sampler state."""
    value = randbelow(bound)
    if type(value) is not int:
        raise TypeError("randbelow must return a Python int (not bool)")
    if not 0 <= value < bound:
        raise ValueError(f"randbelow({bound}) returned an out-of-range integer")
    return value


def _target_tuple(target: Iterable[int], m: int, k: int | None = None) -> tuple[int, ...]:
    try:
        row = tuple(target)
    except TypeError as exc:
        raise TypeError("each target must be an iterable of integer cache labels") from exc
    if not row:
        raise ValueError("target tuples must have K >= 1")
    if k is not None and len(row) != k:
        raise ValueError("all targets must have the same length K")
    for label in row:
        _integer(label, "target label")
        if label >= m:
            raise ValueError("target labels must lie in range(m)")
    return row


@dataclass(frozen=True, slots=True)
class IntegerAlias:
    """Immutable exact alias table for a nonzero vector of integer weights.

    Choose column uniformly from range(len(cutoffs)) and coin uniformly from
    range(total). Return the column iff coin < cutoffs[column], else its alias.
    from_weights returns label a with probability weights[a]/total exactly.
    Direct construction checks the same structural bounds and copies sequence
    inputs to tuples, so a valid table cannot retain externally mutable lists.
    """

    total: int
    cutoffs: tuple[int, ...]
    aliases: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.total, "alias total", minimum=1)
        try:
            cutoffs, aliases = tuple(self.cutoffs), tuple(self.aliases)
        except TypeError as exc:
            raise TypeError("alias cutoffs and aliases must be integer sequences") from exc
        if not cutoffs or len(cutoffs) != len(aliases):
            raise ValueError("alias cutoffs and aliases must have the same nonzero length")
        for cutoff in cutoffs:
            _integer(cutoff, "alias cutoff")
            if cutoff > self.total:
                raise ValueError("alias cutoffs must not exceed total")
        for alias in aliases:
            _integer(alias, "alias label")
            if alias >= len(cutoffs):
                raise ValueError("alias labels must index the table")
        object.__setattr__(self, "cutoffs", cutoffs)
        object.__setattr__(self, "aliases", aliases)

    @classmethod
    def from_weights(cls, weights: Iterable[int]) -> IntegerAlias:
        values = tuple(weights)
        if not values:
            raise ValueError("alias weights must be nonempty")
        for value in values:
            _integer(value, "alias weight")
        total = sum(values)
        if total == 0:
            raise ValueError("alias weights must have positive total")
        size = len(values)
        scaled = [size * value for value in values]
        small = [a for a, value in enumerate(scaled) if value < total]
        large = [a for a, value in enumerate(scaled) if value >= total]
        cutoffs = [total] * size
        aliases = list(range(size))
        while small and large:
            low, high = small.pop(), large.pop()
            cutoffs[low] = scaled[low]
            aliases[low] = high
            scaled[high] -= total - scaled[low]
            if scaled[high] < total:
                small.append(high)
            else:
                large.append(high)
        # Exact integer arithmetic leaves only columns of scaled mass total.
        # Their initialized full cutoffs and self aliases are already correct.
        return cls(total, tuple(cutoffs), tuple(aliases))

    def draw(self, randbelow: RandBelow) -> int:
        column = _draw(randbelow, len(self.cutoffs))
        coin = _draw(randbelow, self.total)
        return column if coin < self.cutoffs[column] else self.aliases[column]


@dataclass(frozen=True, slots=True)
class FirstPlan:
    """Index-only first-epoch plan; not a collection of returned full records.

    read_required=True means an executor must actually expose the next source
    record before returning it. In this planner that branch has all indices m.
    Otherwise every index is cached and no source access is needed.
    """

    indices: tuple[int, ...]
    read_required: bool


@dataclass(frozen=True, slots=True, init=False)
class FiniteStoppingCoupling:
    """Preprocessed exact common first law for a known distinct finite family.

    The target family is copied once and remains immutable. No method changes
    sampler state, including when an RNG raises or returns an invalid value.
    The caller owns the RNG; its advancement is not rolled back on exceptions.
    """

    m: int
    k: int
    targets: tuple[tuple[int, ...], ...]
    degrees: tuple[tuple[int, ...], ...]
    delta: int
    choice_count: int
    residual_choices: int
    residual_tables: tuple[IntegerAlias, ...]

    def __init__(self, m: int, targets: Iterable[Iterable[int]]) -> None:
        _integer(m, "m", minimum=1)
        rows: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        k: int | None = None
        for target in targets:
            row = _target_tuple(target, m, k)
            if row in seen:
                raise ValueError("F must contain distinct target tuples; duplicate found")
            if k is None:
                k = len(row)
            seen.add(row)
            rows.append(row)
        if k is None:
            raise ValueError("F must be nonempty")
        degrees = [[0] * m for _ in range(k)]
        for row in rows:
            for q, label in enumerate(row):
                degrees[q][label] += 1
        frozen_degrees = tuple(tuple(column) for column in degrees)
        delta = max(max(column) for column in frozen_degrees)
        choice_count = m * delta
        residual_choices = choice_count - len(rows)
        if residual_choices:
            tables = tuple(
                IntegerAlias.from_weights(delta - count for count in column)
                for column in frozen_degrees
            )
        else:
            # No all-zero residual distribution is constructed or normalized.
            tables = ()
        for name, value in (
            ("m", m), ("k", k), ("targets", tuple(rows)),
            ("degrees", frozen_degrees), ("delta", delta),
            ("choice_count", choice_count), ("residual_choices", residual_choices),
            ("residual_tables", tables),
        ):
            object.__setattr__(self, name, value)

    @property
    def population_size(self) -> int:
        return self.m + 1

    @property
    def worst_total_cost(self) -> Fraction:
        return (2 - Fraction(1, self.choice_count)) / self.population_size

    @property
    def oracle_total_cost(self) -> Fraction:
        return (2 - Fraction(1, self.m)) / self.population_size

    def draw_cached(self, randbelow: RandBelow) -> tuple[int, ...]:
        """Draw a cached tuple with uniform coordinate marginals 1/m."""
        ticket = _draw(randbelow, self.choice_count)
        if ticket < len(self.targets):
            return self.targets[ticket]
        return tuple(table.draw(randbelow) for table in self.residual_tables)

    def draw_first(self, randbelow: RandBelow) -> FirstPlan:
        """Plan all-z with probability 1/(m+1), else the cached coupling.

        These are symbolic index labels only: this method never reads a source
        or claims that an unread full record has already been returned.
        """
        if _draw(randbelow, self.population_size) == 0:
            return FirstPlan((self.m,) * self.k, True)
        return FirstPlan(self.draw_cached(randbelow), False)


@dataclass(frozen=True, slots=True)
class FullRecord:
    """Tiny complete-record fixture; an ID or planning symbol is insufficient."""

    record_id: str
    payload: bytes

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not self.record_id:
            raise TypeError("record_id must be a nonempty str")
        if type(self.payload) is not bytes:
            raise TypeError("a full-record fixture requires an actual bytes payload")


class MeteredOneUnreadSource(Iterator[FullRecord]):
    """Expose at most one actual iterator item through next(), with accounting.

    No hidden-ID map, synthetic record constructor, or uncounted lookup exists.
    A yielded item is charged even if rejected for not being a FullRecord.
    next_calls counts actual calls to the wrapped iterator; source_reads counts
    successful yields from it. An exhausted or invalid source cannot be retried.
    This fixture is deliberately mutable because actual exposure is irreversible.
    """

    def __init__(self, records: Iterable[FullRecord]) -> None:
        self._records = iter(records)
        self._attempted = False
        self._next_calls = 0
        self._source_reads = 0

    @property
    def next_calls(self) -> int:
        return self._next_calls

    @property
    def source_reads(self) -> int:
        return self._source_reads

    def __iter__(self) -> MeteredOneUnreadSource:
        return self

    def __next__(self) -> FullRecord:
        if self._attempted:
            raise RuntimeError("the sole unread source item was already attempted")
        self._attempted = True
        self._next_calls += 1
        record = next(self._records)
        self._source_reads += 1
        if not isinstance(record, FullRecord):
            raise TypeError("source must yield an actual FullRecord, not a hidden ID")
        return record


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Actual full records after both logical service epochs of the fixture."""

    first_plan: FirstPlan
    first: tuple[FullRecord, ...]
    second: tuple[FullRecord | None, ...]
    source_reads: int


def execute_two_observation(
    coupling: FiniteStoppingCoupling,
    cached_records: Iterable[FullRecord],
    source: MeteredOneUnreadSource,
    target: Iterable[int],
    randbelow: RandBelow,
) -> ExecutionResult:
    """Run a first draw plus an optimal two-observation continuation.

    This is a tiny full-record demonstration, not a production source adapter.
    The source must be fresh. The task tuple may be any element of C^K; the
    claimed minimax value applies to the preprocessed family. Calls which fail
    after an actual source exposure leave that exposure charged; neither source
    side effects nor the caller's RNG are rolled back.
    """
    cache = tuple(cached_records)
    if len(cache) != coupling.m:
        raise ValueError("cache must contain exactly m full records")
    if any(not isinstance(record, FullRecord) for record in cache):
        raise TypeError("cache entries must be actual FullRecord objects")
    cache_ids = {record.record_id for record in cache}
    if len(cache_ids) != coupling.m:
        raise ValueError("cached full-record IDs must be distinct")
    task = _target_tuple(target, coupling.m, coupling.k)
    if source.next_calls or source.source_reads:
        raise ValueError("execution requires a fresh one-unread source")

    def read_unread() -> FullRecord:
        record = next(source)  # Required charged next-source exposure.
        if record.record_id in cache_ids:
            raise ValueError("the unread record must be distinct from cached records")
        return record

    plan = coupling.draw_first(randbelow)
    second: list[FullRecord | None] = [None] * coupling.k
    if plan.read_required:
        unread = read_unread()
        first = (unread,) * coupling.k
        # Every task target is cached, so all queries survive the all-z branch.
        for q in range(coupling.k):
            second[q] = cache[_draw(randbelow, coupling.m)]
    else:
        first = tuple(cache[label] for label in plan.indices)
        survivors = tuple(q for q in range(coupling.k) if plan.indices[q] != task[q])
        if survivors:
            # This branch is impossible for m=1, so m-1 is never normalized there.
            if _draw(randbelow, coupling.m) == 0:
                unread = read_unread()
                for q in survivors:
                    second[q] = unread
            else:
                for q in survivors:
                    label = _draw(randbelow, coupling.m - 1)
                    if label >= plan.indices[q]:
                        label += 1
                    second[q] = cache[label]
    return ExecutionResult(plan, first, tuple(second), source.source_reads)
