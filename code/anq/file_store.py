"""Read-only file backends with an explicit logical exposure ledger.

The headerless source consists of 32-byte little-endian ``RECORD_DTYPE``
records, already in its physical permutation. Positions are physical row
addresses, never the uint64 labels in the ``id`` field. No constructor scans
the population: only pages intersecting the initial prefix are read.

``prepare_prefix`` optionally pays for the exact same immutable prefix cache
once before constructing both backends. ``stats`` separates that preparation,
local setup, and continuation. Byte counters measure actual unbuffered Python
``FileIO.read`` calls; they make no claim about OS cache misses or device I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers
import os
from pathlib import Path
import sys
from time import perf_counter
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


RECORD_DTYPE = np.dtype([("id", "<u8"), ("p", "<f8"),
                         ("r", "<f8"), ("q", "<f8")])
RECORD_BYTES = 32


def _integer(value: object, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, not a boolean")
    value = int(value)
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be in [0, {maximum}]")
    return value


def _parameters(N: object, initial_h: object, page_bytes: object) -> tuple[int, int, int]:
    N = _integer(N, "N", sys.maxsize // RECORD_BYTES)
    initial_h = _integer(initial_h, "initial_h", N)
    page_bytes = _integer(page_bytes, "page_bytes", sys.maxsize)
    if not page_bytes or page_bytes % RECORD_BYTES:
        raise ValueError("page_bytes must be a positive multiple of 32")
    return N, initial_h, page_bytes


def _signature(handle, expected_bytes: int) -> tuple[int, int, int, int]:
    stat = os.fstat(handle.fileno())
    if stat.st_size != expected_bytes:
        raise ValueError(f"file length {stat.st_size} does not match N * 32 ({expected_bytes})")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass
class _Counters:
    read_calls: int = 0
    read_requested_bytes: int = 0
    read_returned_bytes: int = 0
    read_exception_calls: int = 0
    read_consumed_on_error_bytes: int = 0
    read_error_position_unknown: int = 0
    read_seconds: float = 0.0
    # False means a signal/secondary failure interrupted byte accounting.
    # Counts then remain diagnostics, never an asserted complete read total.
    read_accounting_complete: bool = True
    decode_seconds: float = 0.0
    copy_seconds: float = 0.0
    copied_output_bytes: int = 0
    range_copy_calls: int = 0
    range_copy_segments: int = 0
    range_output_bytes: int = 0
    wall_seconds: float = 0.0

    def snapshot(self) -> dict:
        result = vars(self).copy()
        result["decode_copy_seconds"] = self.decode_seconds + self.copy_seconds
        result["observed_api_read_bytes"] = (self.read_returned_bytes
                                              + self.read_consumed_on_error_bytes)
        return result


def _read_exact(handle, byte_count: int, counters: _Counters) -> bytes:
    """Count every real read, including a short read if the source changes."""
    chunks = []
    remaining = byte_count
    while remaining:
        read_start = handle.tell()
        chunk = None
        counters.read_accounting_complete = False
        counters.read_calls += 1
        counters.read_requested_bytes += remaining
        started = perf_counter()
        try:
            chunk = handle.read(remaining)
        except BaseException:
            counters.read_exception_calls += 1
            if chunk is None:
                # A signal or wrapper may raise after the file cursor advanced
                # but before Python received bytes. If chunk is already assigned
                # the finally block counts its return, never both categories.
                try:
                    counters.read_consumed_on_error_bytes += max(0, handle.tell() - read_start)
                except BaseException:
                    counters.read_error_position_unknown += 1
                else:
                    counters.read_accounting_complete = True
            else:
                counters.read_returned_bytes += len(chunk)
                counters.read_accounting_complete = True
            raise
        else:
            # Account before calling a timer: a timer-side interrupt must not
            # silently turn a completed read into a zero-byte measurement.
            counters.read_returned_bytes += len(chunk)
            counters.read_accounting_complete = True
        finally:
            counters.read_seconds += perf_counter() - started
        if not chunk:
            raise OSError("source ended early; source files must remain unchanged while open")
        chunks.append(chunk)
        remaining -= len(chunk)
    if len(chunks) == 1:
        return chunks[0]
    started = perf_counter()
    result = b"".join(chunks)
    counters.copy_seconds += perf_counter() - started
    return result


def _decode_pages(raw: bytes, start_page: int, rows_per_page: int,
                  counters: _Counters) -> dict[int, np.ndarray]:
    started = perf_counter()
    records = np.frombuffer(raw, dtype=RECORD_DTYPE)
    pages = {start_page + offset // rows_per_page: records[offset:offset + rows_per_page]
             for offset in range(0, len(records), rows_per_page)}
    counters.decode_seconds += perf_counter() - started
    return pages


@dataclass(frozen=True)
class PrefixCache:
    """Immutable shared prefix pages; construct with :func:`prepare_prefix`.

    Private page tails are present but are not logically exposed. The arrays
    are read-only views over immutable bytes; stores keep no duplicate payload.
    """

    path: Path
    N: int
    initial_h: int
    page_bytes: int
    _source_signature: tuple[int, int, int, int]
    _pages: Mapping[int, np.ndarray]
    _setup_items: tuple[tuple[str, int | float], ...]

    @property
    def payload_bytes(self) -> int:
        return sum(page.nbytes for page in self._pages.values())

    @property
    def stats(self) -> dict:
        return dict(self._setup_items)


def prepare_prefix(path: os.PathLike | str, N: int, initial_h: int,
                   page_bytes: int = 4096) -> PrefixCache:
    """Read the initial prefix once, including at most one private page tail."""
    started = perf_counter()
    N, initial_h, page_bytes = _parameters(N, initial_h, page_bytes)
    path = Path(path).resolve()
    counters = _Counters()
    prefix_bytes = min(((initial_h * RECORD_BYTES + page_bytes - 1) // page_bytes)
                       * page_bytes, N * RECORD_BYTES)
    with path.open("rb", buffering=0) as handle:
        signature = _signature(handle, N * RECORD_BYTES)
        raw = _read_exact(handle, prefix_bytes, counters) if prefix_bytes else b""
        pages = _decode_pages(raw, 0, page_bytes // RECORD_BYTES, counters)
    counters.wall_seconds = perf_counter() - started
    return PrefixCache(path, N, initial_h, page_bytes, signature,
                       MappingProxyType(pages), tuple(counters.snapshot().items()))


class _FileStore:
    def __init__(self, path: os.PathLike | str, N: int, initial_h: int,
                 page_bytes: int = 4096, *, prefix: PrefixCache | None = None):
        started = perf_counter()
        self.N, self.initial_h, self.page_bytes = _parameters(N, initial_h, page_bytes)
        self.path = Path(path).resolve()
        self.rows_per_page = self.page_bytes // RECORD_BYTES
        self._h = self.initial_h
        self._setup = _Counters()
        self._continuation = _Counters()
        self._pages: dict[int, np.ndarray] = {}
        self._prefix_cached_pages = 0
        self._cached_payload_bytes = 0
        self._shared_payload_bytes = 0
        self._prefix_setup: dict | None = None
        self._peak_cache_payload_bytes = 0
        self._seen_extra: set[int] = set()
        self._seen_count = self.initial_h
        self._closed = True
        self._failed = False
        self._failure_type: str | None = None
        self._file = None
        pending_failure = False
        try:
            self._file = self.path.open("rb", buffering=0)
            signature = _signature(self._file, self.N * RECORD_BYTES)
            if prefix is not None:
                if not isinstance(prefix, PrefixCache):
                    raise TypeError("prefix must be a PrefixCache from prepare_prefix")
                if (prefix.path != self.path or prefix.N != self.N
                        or prefix.initial_h != self.initial_h
                        or prefix.page_bytes != self.page_bytes
                        or prefix._source_signature != signature):
                    raise ValueError("shared prefix does not match this source and configuration")
                self._pages.update(prefix._pages)
                self._prefix_cached_pages = len(prefix._pages)
                self._cached_payload_bytes = prefix.payload_bytes
                self._shared_payload_bytes = prefix.payload_bytes
                self._prefix_setup = prefix.stats
                # A forward source resumes immediately after the private page tail.
                self._file.seek(prefix.payload_bytes)
            else:
                self._ensure_prefix(self.initial_h, self._setup)
            self._peak_cache_payload_bytes = self._cache_payload_bytes()
            self._closed = False
        except BaseException as exc:
            pending_failure = True
            # Cleanup must not replace a setup KeyboardInterrupt/SystemExit
            # with a secondary close failure. All fields exist before try.
            self._fail_stop(exc)
            raise
        finally:
            self._finish_timing(self._setup, started, pending_failure)

    def _finish_timing(self, counters, started, pending_failure, copy_started=None):
        """A terminal timer failure is an operation failure too.

        A secondary diagnostic error must not replace a primary exception that
        has already disabled this instance. This helper is called only after
        public-request validation and from the operation's finally block.
        """
        try:
            ended = perf_counter()
            if copy_started is not None:
                counters.copy_seconds += ended - copy_started
            counters.wall_seconds += ended - started
        except BaseException as exc:
            if not pending_failure:
                self._fail_stop(exc)
                raise

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("file store is closed")

    def _cache_payload_bytes(self) -> int:
        # Page views are disjoint slices; raw bytes have exactly this total size.
        return self._cached_payload_bytes

    def _read_run(self, first_page: int, past_page: int, counters: _Counters) -> None:
        offset = first_page * self.page_bytes
        stop = min(past_page * self.page_bytes, self.N * RECORD_BYTES)
        self._position_file(offset)
        raw = _read_exact(self._file, stop - offset, counters)
        self._pages.update(_decode_pages(raw, first_page, self.rows_per_page, counters))
        self._cached_payload_bytes += len(raw)
        self._peak_cache_payload_bytes = max(self._peak_cache_payload_bytes,
                                              self._cached_payload_bytes)

    def _position_file(self, offset: int) -> None:
        if self._file.tell() != offset:
            self._file.seek(offset)

    def _ensure_prefix(self, h: int, counters: _Counters) -> None:
        past_page = (h + self.rows_per_page - 1) // self.rows_per_page
        if past_page > self._prefix_cached_pages:
            self._ensure_pages(range(self._prefix_cached_pages, past_page), counters)
            self._prefix_cached_pages = past_page

    def _ensure_pages(self, page_numbers: Iterable[int], counters: _Counters) -> None:
        missing = [int(page) for page in page_numbers if int(page) not in self._pages]
        if not missing:
            return
        first = previous = missing[0]
        for page in missing[1:]:
            if page != previous + 1:
                self._read_run(first, previous + 1, counters)
                first = page
            previous = page
        self._read_run(first, previous + 1, counters)

    def _positions(self, positions: object, upper: int) -> np.ndarray:
        if isinstance(positions, (list, tuple)) and any(
                isinstance(x, (bool, np.bool_)) for x in positions):
            raise TypeError("positions must contain integers, not booleans")
        positions = np.asarray(positions)
        if positions.ndim != 1:
            raise ValueError("positions must be a one-dimensional integer batch")
        if positions.size == 0:
            # [] is a useful batch despite NumPy inferring float64 for it.
            if positions.dtype.kind not in "iuf":
                raise TypeError("positions must contain integers")
            return np.empty(0, dtype=np.int64)
        if positions.dtype.kind not in "iu":
            raise TypeError("positions must contain integers")
        # Check in the original signedness before converting uint64 to int64.
        if np.any(positions < 0) or np.any(positions >= upper):
            raise IndexError(f"positions must be in [0, {upper})")
        return positions.astype(np.int64, copy=False)

    def _gather_cached(self, positions: np.ndarray) -> np.ndarray:
        started = perf_counter()
        result = np.empty(len(positions), dtype=RECORD_DTYPE)
        if len(positions):
            page_numbers = positions // self.rows_per_page
            order = np.argsort(page_numbers, kind="stable")
            sorted_pages = page_numbers[order]
            boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_pages)) + 1, len(order)]
            for first, past in zip(boundaries[:-1], boundaries[1:]):
                output_positions = order[first:past]
                page = self._pages[int(sorted_pages[first])]
                result[output_positions] = page[positions[output_positions] % self.rows_per_page]
        self._continuation.copy_seconds += perf_counter() - started
        self._continuation.copied_output_bytes += result.nbytes
        return result

    def read_range(self, start: int, stop: int) -> np.ndarray:
        """Copy the cached logical prefix slice [start, stop), without I/O.

        Both backends return an independent, writeable structured array. This
        uses one slice copy per underlying read extent, with no position array
        or sorting, and never admits private page tails or indexed-only rows.
        """
        self._require_open()
        start = _integer(start, "start", self._h)
        stop = _integer(stop, "stop", self._h)
        if start > stop:
            raise ValueError("start must not exceed stop")
        started = perf_counter()
        copied_started = perf_counter()
        pending_failure = False
        try:
            result = np.empty(stop - start, dtype=RECORD_DTYPE)
            position = start
            segments = 0
            while position < stop:
                page_number = position // self.rows_per_page
                page = self._pages[page_number]
                # _decode_pages always makes page views of the complete,
                # immutable records array for one contiguous FileIO.read run.
                # Jump directly to that base's end instead of visiting rows
                # or even every page belonging to the same source extent.
                records = page.base
                if not isinstance(records, np.ndarray) or records.dtype != RECORD_DTYPE:
                    raise RuntimeError("invalid cached page base")
                base_start = (page_number * self.rows_per_page
                              - (page.ctypes.data - records.ctypes.data) // RECORD_BYTES)
                past = min(stop, base_start + len(records))
                if past <= position:
                    raise RuntimeError("invalid cached extent bounds")
                result[position - start:past - start] = records[
                    position - base_start:past - base_start]
                position = past
                segments += 1
            self._continuation.range_copy_calls += 1
            self._continuation.range_copy_segments += segments
            self._continuation.range_output_bytes += result.nbytes
            self._continuation.copied_output_bytes += result.nbytes
            return result
        except BaseException as exc:
            pending_failure = True
            self._fail_stop(exc)
            raise
        finally:
            self._finish_timing(self._continuation, started, pending_failure, copied_started)

    def expose_to(self, new_h: int, *, return_positions: bool = True) -> np.ndarray | None:
        """Expose [old_h, new_h), returning all those physical positions.

        Equal frontiers are allowed. Moving backwards or beyond N is rejected.
        Indexed stores also provide this method for a matched initial/scan path.
        Set return_positions=False for a contiguous scan without an O(rows)
        address array; then the return value is None.
        """
        self._require_open()
        new_h = _integer(new_h, "new_h", self.N)
        if new_h < self._h:
            raise ValueError("exposure must be monotone")
        if not isinstance(return_positions, (bool, np.bool_)):
            raise TypeError("return_positions must be a boolean")
        started = perf_counter()
        old_h = self._h
        pending_failure = False
        try:
            self._ensure_prefix(new_h, self._continuation)
            exposed = np.arange(old_h, new_h, dtype=np.int64) if return_positions else None
            self._seen_extra = {position for position in self._seen_extra if position >= new_h}
            self._h = new_h
            self._seen_count = self._h + len(self._seen_extra)
            self._peak_cache_payload_bytes = max(self._peak_cache_payload_bytes,
                                                  self._cache_payload_bytes())
            return exposed
        except BaseException as exc:
            pending_failure = True
            self._fail_stop(exc)
            raise
        finally:
            self._finish_timing(self._continuation, started, pending_failure)

    @property
    def physical_rows(self) -> int:
        """Logical prefix frontier, excluding private page tails."""
        return self._h

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def stats(self) -> dict:
        """Detached counters; local read totals exclude shared preparation."""
        setup = self._setup.snapshot()
        continuation = self._continuation.snapshot()
        cached_bytes = self._cache_payload_bytes()
        shared_bytes = self._shared_payload_bytes
        return {
            "backend": type(self).__name__, "closed": self.closed,
            "failed": self._failed, "failure_type": self._failure_type,
            "N": self.N, "initial_h": self.initial_h, "page_bytes": self.page_bytes,
            "setup": setup, "continuation": continuation,
            "prefix_setup": (self._prefix_setup.copy() if self._prefix_setup is not None
                             else setup.copy()),
            "prefix_setup_ownership": ("shared" if self._prefix_setup is not None else "local"),
            "local_total_read_calls": setup["read_calls"] + continuation["read_calls"],
            "local_total_requested_bytes": (setup["read_requested_bytes"]
                                             + continuation["read_requested_bytes"]),
            "local_total_returned_bytes": (setup["read_returned_bytes"]
                                            + continuation["read_returned_bytes"]),
            "physical_count": self._h, "seen_count": self._seen_count,
            "initial_prefix_logical_bytes": self.initial_h * RECORD_BYTES,
            "logical_prefix_bytes": self._h * RECORD_BYTES,
            "policy_visible_payload_bytes": self._seen_count * RECORD_BYTES,
            "retained_cache_payload_bytes": cached_bytes,
            "shared_prefix_payload_bytes": shared_bytes,
            "owned_cache_payload_bytes": cached_bytes - shared_bytes,
            "private_prefetch_payload_bytes": max(0, cached_bytes - self._seen_count * RECORD_BYTES),
            "peak_cache_payload_bytes": self._peak_cache_payload_bytes,
            "cached_pages": len(self._pages), "seen_ledger_entries": len(self._seen_extra),
        }

    def close(self) -> None:
        """Release the handle and this instance's cache references, idempotently."""
        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._closed = True
            self._pages.clear()
            self._cached_payload_bytes = 0
            self._shared_payload_bytes = 0
            self._seen_extra.clear()

    def _fail_stop(self, exc: BaseException) -> None:
        self._failed = True
        self._failure_type = type(exc).__name__
        try:
            self.close()
        except BaseException:
            # Retain the original operation failure; close's finally block has
            # already disabled every later operation and released cache refs.
            pass

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                # Preserve the original body failure; close disables this
                # instance in its finally block even when the handle fails.
                pass
        return False


class ForwardFileStore(_FileStore):
    """Monotone source reads plus a retained cache of the entire exposed prefix."""

    def _position_file(self, offset: int) -> None:
        if self._file.tell() != offset:
            raise RuntimeError("forward backend attempted a non-forward read")

    def gather(self, physical_positions: object) -> np.ndarray:
        self._require_open()
        positions = self._positions(physical_positions, self._h)
        started = perf_counter()
        pending_failure = False
        try:
            return self._gather_cached(positions)
        except BaseException as exc:
            pending_failure = True
            self._fail_stop(exc)
            raise
        finally:
            self._finish_timing(self._continuation, started, pending_failure)


class IndexedFileStore(_FileStore):
    """Arbitrary physical addresses; sorted missing pages use contiguous runs."""

    def gather(self, physical_positions: object) -> np.ndarray:
        self._require_open()
        positions = self._positions(physical_positions, self.N)
        started = perf_counter()
        pending_failure = False
        try:
            self._ensure_pages(np.unique(positions // self.rows_per_page), self._continuation)
            result = self._gather_cached(positions)
            self._seen_extra.update(int(position) for position in np.unique(positions)
                                    if int(position) >= self._h)
            self._seen_count = self._h + len(self._seen_extra)
            self._peak_cache_payload_bytes = max(self._peak_cache_payload_bytes,
                                                  self._cache_payload_bytes())
            return result
        except BaseException as exc:
            pending_failure = True
            self._fail_stop(exc)
            raise
        finally:
            self._finish_timing(self._continuation, started, pending_failure)
