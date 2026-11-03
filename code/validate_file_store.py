"""Independent, small-file checks for the real read-only file backends.

Run from the repository root: python -B code/validate_file_store.py
Only this validation's output directory is created/written. No population
dataset is opened. Fixtures use struct.pack; the reference uses np.fromfile.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np

import anq.file_store as backend
from anq.file_store import ForwardFileStore, IndexedFileStore, prepare_prefix


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviews/innovation_20261017/file_store"
ORACLE_DTYPE = np.dtype([("id", "<u8"), ("p", "<f8"), ("r", "<f8"), ("q", "<f8")])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_error(kind, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except kind:
        return
    raise AssertionError(f"expected {kind.__name__}: {function.__name__}")


def assert_records(actual, reference):
    assert actual.dtype == ORACLE_DTYPE and actual.shape == reference.shape
    assert actual.tobytes() == reference.tobytes(), "record bytes differ from np.fromfile oracle"
    assert actual.flags.writeable, "returned records should be independently editable"


def make_fixture(path, N, seed=17519):
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    labels = [(2**63 + i) for i in range(N)]
    if N:
        labels[0] = 2**64 - 1
    if N > 1:
        labels[1] = 0
    raw = b"".join(struct.pack("<Qddd", labels[int(i)],
                              float(i) / 7, -0.0 if i == 0 else -float(i) / 11,
                              float("nan") if i == 1 else float(i) / 13)
                   for i in order)
    path.write_bytes(raw)
    # Windows honors the write bit as the read-only file attribute.
    path.chmod(0o444)
    return np.fromfile(path, dtype=ORACLE_DTYPE), hashlib.sha256(raw).hexdigest()


class AuditFile:
    """An independent observer around the actual unbuffered FileIO object."""

    def __init__(self, handle, log):
        self.handle = handle
        self.log = log

    def read(self, count):
        offset = self.handle.tell()
        data = self.handle.read(count)
        self.log["reads"].append(dict(offset=offset, requested=count, returned=len(data)))
        return data

    def seek(self, offset, whence=0):
        self.log["seeks"].append(dict(offset=offset, whence=whence))
        return self.handle.seek(offset, whence)

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.handle.close()
        return False


@contextmanager
def audit_reads():
    original_open = Path.open
    logs = []

    def observed_open(path, mode="r", buffering=-1, *args, **kwargs):
        assert mode == "rb" and buffering == 0, "backend must use read-only unbuffered FileIO"
        log = dict(path=str(path), mode=mode, reads=[], seeks=[])
        logs.append(log)
        return AuditFile(original_open(path, mode, buffering, *args, **kwargs), log)

    with patch.object(backend.Path, "open", observed_open):
        yield logs


def assert_io(counters, events, *, consumed_on_error=False):
    assert counters["read_calls"] == len(events)
    assert counters["read_requested_bytes"] == sum(x["requested"] for x in events)
    observed_bytes = sum(x["returned"] for x in events)
    if consumed_on_error:
        assert counters["observed_api_read_bytes"] == observed_bytes
    else:
        assert counters["read_returned_bytes"] == observed_bytes
        assert counters["read_consumed_on_error_bytes"] == 0
    assert counters["observed_api_read_bytes"] == (counters["read_returned_bytes"]
                                                   + counters["read_consumed_on_error_bytes"])
    assert counters["decode_copy_seconds"] == counters["decode_seconds"] + counters["copy_seconds"]
    for key in ["read_seconds", "decode_seconds", "copy_seconds", "wall_seconds"]:
        assert np.isfinite(counters[key]) and counters[key] >= 0


def exact_forward_case(path, ref):
    with audit_reads() as logs:
        with ForwardFileStore(path, 23, 3, page_bytes=128) as store:
            assert store._file.handle.writable() is False
            assert logs[0]["reads"] == [dict(offset=0, requested=128, returned=128)]
            assert_io(store.stats["setup"], logs[0]["reads"])
            assert store.stats["retained_cache_payload_bytes"] == 128
            assert store.stats["private_prefetch_payload_bytes"] == 32
            before = store.stats
            expect_error(IndexError, store.gather, [3])
            assert before == store.stats, "private page tail became visible"
            for h, expected_events, cached in [
                (4, [], 128),
                (5, [dict(offset=128, requested=128, returned=128)], 256),
                (14, [dict(offset=256, requested=256, returned=256)], 512),
                (23, [dict(offset=512, requested=224, returned=224)], 736),
            ]:
                old_h = store.physical_rows
                old_reads = len(logs[0]["reads"])
                assert np.array_equal(store.expose_to(h), np.arange(old_h, h))
                assert logs[0]["reads"][old_reads:] == expected_events
                positions = np.array([h - 1, 0, h - 1], dtype=">u8")
                actual = store.gather(positions)
                assert_records(actual, ref[positions.astype(np.int64)])
                actual["id"][0] = 7
                assert_records(store.gather(positions), ref[positions.astype(np.int64)])
                assert store.stats["retained_cache_payload_bytes"] == cached
                assert store.stats["seen_count"] == h
                assert store.stats["physical_count"] == h
                assert store.stats["private_prefetch_payload_bytes"] == cached - h * 32
            assert not logs[0]["seeks"], "direct forward continuation must never seek"
            assert_io(store.stats["continuation"], logs[0]["reads"][1:])
            assert store.stats["continuation"]["read_calls"] == 3
            assert store.stats["continuation"]["read_returned_bytes"] == 608
            assert store.stats["local_total_returned_bytes"] == ref.nbytes
            result = dict(stats=store.stats, observed_file_api=logs[0])
        assert store.closed and store.stats["retained_cache_payload_bytes"] == 0
        assert store.stats["peak_cache_payload_bytes"] == ref.nbytes
        expect_error(ValueError, store.gather, [])
        expect_error(ValueError, store.expose_to, 23)
        expect_error(ValueError, store.read_range, 0, 0)
        expect_error(ValueError, store.__enter__)
        store.close()
    return result


def exact_indexed_case(path, ref):
    with audit_reads() as logs:
        with IndexedFileStore(path, 23, 3, page_bytes=128) as store:
            requested = np.array([18, 8, 8, 0, 22, 17])
            assert_records(store.gather(requested), ref[requested])
            expected = [dict(offset=0, requested=128, returned=128),
                        dict(offset=256, requested=128, returned=128),
                        dict(offset=512, requested=224, returned=224)]
            assert logs[0]["reads"] == expected
            assert store.stats["seen_count"] == 7
            assert store.stats["physical_count"] == 3
            assert store.stats["retained_cache_payload_bytes"] == 480
            assert store.stats["private_prefetch_payload_bytes"] == 256
            assert_records(store.gather([9, 20, 18]), ref[[9, 20, 18]])
            assert store.stats["seen_count"] == 9
            assert logs[0]["reads"] == expected, "cached pages were read twice"
            assert_io(store.stats["setup"], logs[0]["reads"][:1])
            assert_io(store.stats["continuation"], logs[0]["reads"][1:])
            snapshot = dict(stats=store.stats, observed_file_api=copy.deepcopy(logs[0]))
            assert np.array_equal(store.expose_to(10), np.arange(3, 10))
            assert store.stats["seen_count"] == 14  # prefix 0..9 plus 17,18,20,22
            assert store.stats["seen_ledger_entries"] == 4
            assert_records(store.gather([3, 9]), ref[[3, 9]])
            assert np.array_equal(store.expose_to(23), np.arange(10, 23))
            assert store.stats["seen_count"] == 23
            assert store.stats["retained_cache_payload_bytes"] == ref.nbytes
    return snapshot


def shared_prefix_case(path, ref):
    with audit_reads() as logs:
        prefix = prepare_prefix(path, 23, 3, page_bytes=128)
        assert prefix.payload_bytes == 128
        assert_io(prefix.stats, logs[0]["reads"])
        assert prefix.stats["read_calls"] == 1
        expect_error(ValueError, prefix._pages[0].setflags, write=True)
        with ForwardFileStore(path, 23, 3, 128, prefix=prefix) as forward, \
                IndexedFileStore(path, 23, 3, 128, prefix=prefix) as indexed:
            assert np.shares_memory(forward._pages[0], indexed._pages[0])
            for store in [forward, indexed]:
                assert store.stats["setup"]["read_calls"] == 0
                assert store.stats["prefix_setup_ownership"] == "shared"
                assert store.stats["prefix_setup"]["read_returned_bytes"] == 128
                assert store.stats["shared_prefix_payload_bytes"] == 128
                assert store.stats["owned_cache_payload_bytes"] == 0
            expect_error(IndexError, forward.gather, [3])
            assert_records(indexed.gather([3, 3]), ref[[3, 3]])
            assert indexed.stats["seen_count"] == 4
            assert indexed.stats["continuation"]["read_calls"] == 0
            forward.expose_to(23)
            assert_records(forward.gather([22, 3, 0]), ref[[22, 3, 0]])
            assert len(logs[1]["reads"]) == 1
            assert logs[1]["reads"][0] == dict(offset=128, requested=608, returned=608)
            assert logs[1]["seeks"] == [dict(offset=128, whence=0)]
            assert not logs[2]["reads"]
            result = dict(prefix_setup=prefix.stats, forward=forward.stats, indexed=indexed.stats,
                          observed_file_api=logs)
        assert prefix.payload_bytes == 128, "closing a store must preserve the reusable shared prefix"
    expect_error(ValueError, ForwardFileStore, path, 23, 4, 128, prefix=prefix)
    expect_error(ValueError, IndexedFileStore, path, 23, 3, 256, prefix=prefix)
    expect_error(TypeError, ForwardFileStore, path, 23, 3, 128, prefix="invalid")
    return result


def boundary_and_input_cases(directory):
    count = 0
    rng = np.random.default_rng(4017)
    for N in [0, 1, 2, 3, 4, 5, 127, 128, 129, 1031]:
        path = directory / f"boundary_{N}.bin"
        reference, original_hash = make_fixture(path, N, seed=700 + N)
        for page_bytes in [32, 64, 96, 4096]:
            for h in sorted(set([0, min(N, 1), N // 2, N])):
                with ForwardFileStore(path, N, h, page_bytes) as forward, \
                        IndexedFileStore(path, N, h, page_bytes) as indexed:
                    assert_records(forward.gather([]), reference[:0])
                    assert_records(indexed.gather(np.array([], dtype=">u8")), reference[:0])
                    expected_setup = min(((h * 32 + page_bytes - 1) // page_bytes)
                                         * page_bytes, N * 32)
                    for store in [forward, indexed]:
                        assert store.stats["setup"]["read_returned_bytes"] == expected_setup
                        assert store.stats["setup"]["read_calls"] == int(bool(h))
                        assert store.expose_to(h).size == 0
                        assert_records(store.read_range(0, h), reference[:h])
                        assert_records(store.read_range(h, h), reference[:0])
                    seen = set(range(h))
                    frontier = h
                    for _ in range(8):
                        next_h = int(rng.integers(frontier, N + 1))
                        assert np.array_equal(forward.expose_to(next_h), np.arange(frontier, next_h))
                        frontier = next_h
                        positions = rng.integers(0, frontier, size=17) if frontier else []
                        assert_records(forward.gather(positions), reference[positions])
                        positions = rng.integers(0, N, size=19) if N else []
                        assert_records(indexed.gather(positions), reference[positions])
                        seen.update(int(x) for x in positions)
                        assert indexed.stats["seen_count"] == len(seen)
                    forward.expose_to(N)
                    assert_records(forward.gather(np.arange(N, dtype=">i8")), reference)
                    assert forward.stats["local_total_returned_bytes"] == N * 32
                    assert forward.stats["retained_cache_payload_bytes"] == N * 32
                    assert forward.stats["private_prefetch_payload_bytes"] == 0
                    assert_records(forward.read_range(0, N), reference)
                    indexed.gather(np.arange(N))
                    # Indexed-only cached rows must not extend its prefix API.
                    assert_records(indexed.read_range(0, h), reference[:h])
                    assert indexed.stats["local_total_returned_bytes"] == N * 32
                    assert indexed.stats["seen_count"] == N
                    count += 1
        assert sha256(path) == original_hash, "source fixture was modified"
    return count


def invalid_input_cases(path, directory):
    count = 0
    for cls in [ForwardFileStore, IndexedFileStore]:
        for kwargs, kind in [
            (dict(N=-1, initial_h=0), ValueError),
            (dict(N=23.0, initial_h=0), TypeError),
            (dict(N=True, initial_h=0), TypeError),
            (dict(N=23, initial_h=-1), ValueError),
            (dict(N=23, initial_h=24), ValueError),
            (dict(N=23, initial_h=1.0), TypeError),
            (dict(N=23, initial_h=0, page_bytes=0), ValueError),
            (dict(N=23, initial_h=0, page_bytes=31), ValueError),
            (dict(N=23, initial_h=0, page_bytes=33), ValueError),
            (dict(N=23, initial_h=0, page_bytes=True), TypeError),
            (dict(N=2**64, initial_h=0), ValueError),
        ]:
            expect_error(kind, cls, path, **kwargs)
            count += 1
        with cls(path, np.uint64(23), np.int64(3), np.int64(128)) as store:
            assert_records(store.gather(np.array([0, 1], dtype=">i8")),
                           np.fromfile(path, dtype=ORACLE_DTYPE)[[0, 1]])
            for positions, kind in [
                ([-1], IndexError), ([23], IndexError),
                (np.array([2**64 - 1], dtype=np.uint64), IndexError),
                ([1.0], TypeError), ([True], TypeError), ([0, True], TypeError),
                (["1"], TypeError), ([None], TypeError),
                (np.array([1], dtype=object), TypeError),
                ([[0, 1]], ValueError), (1, ValueError),
                (np.array([], dtype=bool), TypeError),
            ]:
                before = store.stats
                expect_error(kind, store.gather, positions)
                assert before == store.stats
                count += 1
            for new_h, kind in [(2, ValueError), (24, ValueError),
                                (-1, ValueError), (3.0, TypeError), (True, TypeError)]:
                before = store.stats
                expect_error(kind, store.expose_to, new_h)
                assert before == store.stats
                count += 1
        # Exceptions inside a context must close the handle without swallowing.
        sentinel = RuntimeError("context sentinel")
        try:
            with cls(path, 23, 0) as store:
                raise sentinel
        except RuntimeError as exc:
            assert exc is sentinel and store.closed
        missing = directory / "absent.bin"
        expect_error(FileNotFoundError, cls, missing, 1, 0)
        for byte_count in [0, 31, 32, 33, 735, 737, 768]:
            bad = directory / f"bad_{byte_count}.bin"
            bad.write_bytes(bytes(byte_count))
            expect_error(ValueError, cls, bad, 23, 0)
            count += 1
    return count


def truncated_source_case(directory):
    path = directory / "concurrent_truncation.bin"
    path.write_bytes(struct.pack("<Qddd", 99, 1.0, 2.0, 3.0) * 4)
    with audit_reads() as logs:
        with ForwardFileStore(path, 4, 1, 32) as store:
            # External mutation is explicitly unsupported; unexpected EOF still
            # raises instead of exposing incomplete records or hanging forever.
            with open(path, "r+b", buffering=0) as other:
                other.truncate(40)
            expect_error(OSError, store.expose_to, 4)
            assert store.physical_rows == 1
            assert logs[0]["reads"][1:] == [dict(offset=32, requested=96, returned=8),
                                            dict(offset=40, requested=88, returned=0)]
            assert_io(store.stats["continuation"], logs[0]["reads"][1:])
            assert_failed_closed(store, OSError)
    return dict(status="PASS", actual_read_calls=3, actual_returned_bytes=40)


def assert_failed_closed(store, failure_kind):
    assert store.closed and store._file.closed
    assert store.stats["failed"] is True
    assert store.stats["failure_type"] == failure_kind.__name__
    assert store.stats["retained_cache_payload_bytes"] == 0
    before = store.stats
    for operation, args in [(store.expose_to, (store.physical_rows,)),
                            (store.gather, ([],)), (store.read_range, (0, 0)),
                            (store.__enter__, ())]:
        expect_error(ValueError, operation, *args)
    assert store.stats == before, "failed store continued to operate"
    store.close()


def range_cases(path, reference):
    checked = 0
    invalid = 0
    evidence = []
    for shared in [False, True]:
        prefix = prepare_prefix(path, 23, 3, 128) if shared else None
        for cls in [ForwardFileStore, IndexedFileStore]:
            with audit_reads() as logs:
                with cls(path, 23, 3, 128, prefix=prefix) as store:
                    def check(start, stop, segments=None):
                        nonlocal checked
                        before = store.stats["continuation"]
                        reads_before = copy.deepcopy(logs[0]["reads"])
                        # Enforce absence of both row-address allocation and
                        # gather/sort machinery on the contiguous cached path.
                        with patch.object(backend.np, "arange", side_effect=AssertionError("range allocated addresses")), \
                                patch.object(backend.np, "argsort", side_effect=AssertionError("range sorted rows")), \
                                patch.object(store, "_gather_cached", side_effect=AssertionError("range used gather")):
                            actual = store.read_range(start, stop)
                        assert_records(actual, reference[start:stop])
                        assert logs[0]["reads"] == reads_before
                        after = store.stats["continuation"]
                        assert after["read_calls"] == before["read_calls"]
                        assert after["decode_seconds"] == before["decode_seconds"]
                        assert after["range_copy_calls"] == before["range_copy_calls"] + 1
                        assert after["range_output_bytes"] - before["range_output_bytes"] == (stop - start) * 32
                        assert after["copied_output_bytes"] - before["copied_output_bytes"] == (stop - start) * 32
                        if segments is not None:
                            assert after["range_copy_segments"] - before["range_copy_segments"] == segments
                        for page in store._pages.values():
                            assert not page.flags.writeable
                            assert not np.shares_memory(actual, page)
                        if stop > start:
                            actual["id"][0] = 4
                            assert_records(store.read_range(start, stop), reference[start:stop])
                        checked += 1

                    check(0, 3, 1)
                    check(1, 2, 1)
                    check(3, 3, 0)
                    if cls is IndexedFileStore:
                        store.gather([22])
                        before = store.stats
                        expect_error(ValueError, store.read_range, 22, 23)
                        assert before == store.stats
                        invalid += 1
                    for start, stop, kind in [(-1, 1, ValueError), (0, 4, ValueError),
                                               (2, 1, ValueError), (4, 4, ValueError),
                                               (True, 1, TypeError), (0, 1.0, TypeError),
                                               (0, np.uint64(2**64 - 1), ValueError)]:
                        before = store.stats
                        reads_before = copy.deepcopy(logs[0]["reads"])
                        expect_error(kind, store.read_range, start, stop)
                        assert store.stats == before and logs[0]["reads"] == reads_before
                        invalid += 1
                    before = store.stats
                    expect_error(TypeError, store.expose_to, 14, return_positions="no")
                    assert before == store.stats
                    invalid += 1
                    # No np.arange is needed even while exposing new pages.
                    with patch.object(backend.np, "arange", side_effect=AssertionError("scan allocated addresses")):
                        assert store.expose_to(14, return_positions=False) is None
                    check(0, 14, 2)
                    check(5, 14, 1)  # spans several pages of the same read base
                    check(3, 5, 2)   # shared/private initial-page tail + new extent
                    check(14, 14, 0)
                    assert store.expose_to(23, return_positions=False) is None
                    check(21, 23, 1)
                    check(0, 23)
                    check(23, 23, 0)
                    assert_io(store.stats["continuation"], logs[0]["reads"][(0 if shared else 1):])
                    evidence.append(dict(shared_prefix=shared, backend=cls.__name__,
                                         stats=store.stats, observed_file_api=copy.deepcopy(logs[0])))
                expect_error(ValueError, store.read_range, 0, 0)
    return dict(status="PASS", checked_ranges=checked, rejected_range_or_flag_inputs=invalid,
                output_contract="independent writeable copy; no cache alias; no read/row-sort/address array",
                evidence=evidence)


def failure_transition_cases(path):
    evidence = []
    for failure_kind in [OSError, KeyboardInterrupt]:
        for cls in [ForwardFileStore, IndexedFileStore]:
            for stage in ["decode_after_read", "positions_after_read", "read_consumed_then_raised"]:
                with audit_reads() as logs:
                    with cls(path, 23, 3, 128) as store:
                        if stage == "decode_after_read":
                            target = patch.object(backend, "_decode_pages", side_effect=failure_kind("injected after read"))
                        elif stage == "positions_after_read":
                            target = patch.object(backend.np, "arange", side_effect=failure_kind("injected after cache install"))
                        else:
                            original_read = store._file.read

                            def consume_then_raise(count):
                                original_read(count)
                                raise failure_kind("injected after real FileIO read advanced cursor")

                            target = patch.object(store._file, "read", consume_then_raise)
                        with target:
                            expect_error(failure_kind, store.expose_to, 23)
                        assert_failed_closed(store, failure_kind)
                        assert store.physical_rows == 3
                        assert logs[0]["reads"][1:] == [dict(offset=128, requested=608, returned=608)]
                        counters = store.stats["continuation"]
                        assert_io(counters, logs[0]["reads"][1:],
                                  consumed_on_error=(stage == "read_consumed_then_raised"))
                        if stage == "read_consumed_then_raised":
                            assert counters["read_returned_bytes"] == 0
                            assert counters["read_consumed_on_error_bytes"] == 608
                            assert counters["read_exception_calls"] == 1
                            assert counters["read_error_position_unknown"] == 0
                        else:
                            assert counters["read_returned_bytes"] == 608
                        evidence.append(dict(backend=cls.__name__, failure=failure_kind.__name__,
                                             stage=stage, stats=store.stats,
                                             observed_file_api=copy.deepcopy(logs[0])))
        # Indexed missing pages can form several runs. Fail after one run is
        # already installed and the next one has actually been read.
        for operation in ["expose", "gather"]:
            with audit_reads() as logs:
                with IndexedFileStore(path, 23, 3, 128) as store:
                    if operation == "expose":
                        store.gather([8])  # page 2 splits later missing runs
                    original_decode = backend._decode_pages
                    decode_calls = 0

                    def fail_second_decode(*args):
                        nonlocal decode_calls
                        decode_calls += 1
                        decoded = original_decode(*args)
                        if decode_calls == 2:
                            raise failure_kind("injected after earlier run committed")
                        return decoded

                    with patch.object(backend, "_decode_pages", fail_second_decode):
                        if operation == "expose":
                            expect_error(failure_kind, store.expose_to, 23)
                        else:
                            expect_error(failure_kind, store.gather, [8, 18, 22])
                    assert decode_calls == 2
                    assert_failed_closed(store, failure_kind)
                    assert store.physical_rows == 3
                    assert_io(store.stats["continuation"], logs[0]["reads"][1:])
                    expected = 608 if operation == "expose" else 352
                    assert store.stats["continuation"]["read_returned_bytes"] == expected
                    evidence.append(dict(backend="IndexedFileStore", failure=failure_kind.__name__,
                                         stage=f"partial_multirun_{operation}", stats=store.stats,
                                         observed_file_api=copy.deepcopy(logs[0])))
    return dict(status="PASS", injected_cases=len(evidence), evidence=evidence)


def main():
    if not __debug__:
        raise RuntimeError("validation requires assertions enabled; do not use python -O")
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    assert backend.RECORD_DTYPE.descr == ORACLE_DTYPE.descr
    assert backend.RECORD_BYTES == struct.calcsize("<Qddd") == 32
    with tempfile.TemporaryDirectory(prefix="fixture_", dir=OUT) as temporary:
        directory = Path(temporary)
        try:
            path = directory / "random_physical_order_23.bin"
            reference, original_hash = make_fixture(path, 23)
            result = dict(
                status="PASS", api_version=2, command="python -B code/validate_file_store.py",
                numpy_version=np.__version__, python_version=platform.python_version(),
                platform=platform.platform(),
                fixture=dict(N=23, seed=17519, physical_order="numpy Generator.permutation",
                             sha256=original_hash, bytes=path.stat().st_size,
                             oracle="independently packed struct <Qddd; np.fromfile reference"),
                exact_forward=exact_forward_case(path, reference),
                exact_indexed=exact_indexed_case(path, reference),
                shared_prefix=shared_prefix_case(path, reference),
                contiguous_range=range_cases(path, reference),
                fail_stop=failure_transition_cases(path),
                boundary_configurations=boundary_and_input_cases(directory),
                rejected_invalid_inputs=invalid_input_cases(path, directory),
                truncated_source=truncated_source_case(directory),
            )
            assert sha256(path) == original_hash
        finally:
            # Restore only generated fixture attributes so Windows tempfile can
            # remove this known child directory. No project source is chmod'ed.
            for fixture in directory.iterdir():
                fixture.chmod(0o666)
    result["executed_checks"] = [
        "independent real FileIO.read observer and exact offsets/requests/returns",
        "no hidden full-file preload; forward monotone reads; indexed sorted/coalesced missing pages",
        "partial initial pages remain private; exact EOF tail; copied output independence",
        "shared immutable prefix payload without duplicate read or cache copy",
        "N=0/1 and page boundaries, empty/duplicate batches, uint64 labels and big-endian indices",
        "invalid/negative/overflow indices, invalid frontiers, malformed lengths, closed/context states",
        "randomized records versus np.fromfile; source content hashes unchanged",
        "externally truncated source raises with accurate short/zero-read counters",
        "contiguous ranges use base slices, no argsort/gather/per-row address array, and no I/O",
        "range copies are writeable and do not alias immutable cache or shared prefix",
        "partial I/O/cache-install/position-allocation OSError and KeyboardInterrupt failures close the store",
        "post-failure operations are disabled; consumed-but-unreturned read bytes reported separately",
    ]
    result["not_checked"] = [
        "large original population datasets, which this validation never opens",
        "policy/driver integration and end-to-end scientific claims",
        "physical device bytes, cold OS page cache, or administrator cache flushing",
        "performance representativeness and out-of-memory behavior at production population size",
        "execution on a native big-endian machine or a 32-bit Python build",
        "global label uniqueness, uniform permutation, or semantic schema of arbitrary headerless inputs",
        "concurrent writers other than the explicit unexpected-EOF failure check",
    ]
    result["production_sha256"] = sha256(ROOT / "code/anq/file_store.py")
    result["validator_sha256"] = sha256(Path(__file__))
    result["contiguous_range_checks"] = result["contiguous_range"]["checked_ranges"]
    result["rejected_range_or_flag_inputs"] = result["contiguous_range"]["rejected_range_or_flag_inputs"]
    result["fail_stop_injected_cases"] = result["fail_stop"]["injected_cases"]
    result["validation_wall_seconds"] = perf_counter() - started
    (OUT / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in [
        "status", "boundary_configurations", "rejected_invalid_inputs", "validation_wall_seconds",
        "contiguous_range_checks", "rejected_range_or_flag_inputs", "fail_stop_injected_cases",
        "production_sha256", "validator_sha256", "not_checked"]}, indent=2))


if __name__ == "__main__":
    main()
