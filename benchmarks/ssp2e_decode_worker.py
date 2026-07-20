"""Fresh-process bytes-to-boundary worker for the frozen COMP-009 resource assay.

Imports, blob I/O, native-library loading, static-asset loading, and CPU affinity setup happen
before :func:`decode_blob` starts its timer.  The timed region contains only strict parsing,
per-stream model/CDF setup, zlib or native arithmetic decoding, and exact float32 boundary-array
construction.  No renderer is imported or called here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_execute as execute


WORKER_SCHEMA = "structsplat.comp009.ssp2e-decode-worker.v1"
MAGIC_TO_MODE = {
    b"SSPL1": "sspl1",
    b"SSP2E": "ssp2e",
    b"SSP2S": "ssp2s",
    b"SSP2L": "ssp2l",
    b"SSP2F": "ssp2f",
}
TIMING_SCOPE = (
    "strict_container_parse",
    "per_stream_model_and_cdf_setup",
    "zlib_or_native_arithmetic_decode",
    "native_internal_canonical_reencode_validation",
    "proof_adapter_exact_cdf_snapshot_and_compare",
    "exact_symbol_hash_construction",
    "exact_float32_boundary_array_construction",
)


class DecodeWorkerError(RuntimeError):
    """A worker input, runtime, or exactness invariant was invalid."""


@dataclass(frozen=True)
class DecodeRuntime:
    """All source state loaded before the measured decode region."""

    native: arithmetic.NativeArithmeticCoder
    tables: Mapping[int, assets.StaticCDFTable]
    native_library_bytes: int
    native_library_sha256: str


@dataclass(frozen=True)
class _NativeDecodeProof:
    payload: bytes
    cdf_identity: np.ndarray
    cdf_snapshot: np.ndarray
    decoded_snapshot: np.ndarray


class StrictNativeDecodeProofAdapter:
    """Bridge native canonical decode into the generic container's proof callback.

    The C++ decoder already re-encodes and byte-compares the canonical payload before returning.
    The generic Python container then asks its decoder adapter to re-encode the same symbols a
    second time.  This one-shot adapter returns the native-proven payload only for the immediately
    following callback and only after exact identity/content checks; it never silently skips or
    delegates a mismatch.
    """

    def __init__(self, native: arithmetic.NativeArithmeticCoder) -> None:
        if not isinstance(native, arithmetic.NativeArithmeticCoder):
            raise DecodeWorkerError("strict proof adapter requires NativeArithmeticCoder")
        self._native = native
        self._pending: _NativeDecodeProof | None = None

    def decode(self, payload: bytes, cdf_rows: np.ndarray, n: int) -> np.ndarray:
        if self._pending is not None:
            raise DecodeWorkerError("native proof adapter has an unconsumed decode")
        rows = np.asarray(cdf_rows)
        if rows.dtype != np.uint32 or not rows.flags.c_contiguous:
            raise DecodeWorkerError("native proof adapter requires contiguous uint32 CDF rows")
        physical = bytes(payload)
        cdf_snapshot = rows.copy(order="C")
        decoded = self._native.decode(physical, rows, n)
        values = np.asarray(decoded)
        if values.dtype != np.uint32 or not values.flags.c_contiguous:
            raise DecodeWorkerError("native decoder returned a noncanonical symbol array")
        self._pending = _NativeDecodeProof(
            payload=physical,
            cdf_identity=rows,
            cdf_snapshot=cdf_snapshot,
            decoded_snapshot=values.copy(order="C"),
        )
        return decoded

    def encode(self, symbols: np.ndarray, cdf_rows: np.ndarray) -> bytes:
        pending = self._pending
        if pending is None:
            raise DecodeWorkerError("native proof adapter encode has no preceding decode")
        self._pending = None
        rows = np.asarray(cdf_rows)
        values = np.asarray(symbols)
        if rows is not pending.cdf_identity:
            raise DecodeWorkerError("native proof adapter CDF identity changed")
        if (
            rows.shape != pending.cdf_snapshot.shape
            or rows.dtype != pending.cdf_snapshot.dtype
            or not rows.flags.c_contiguous
            or not np.array_equal(rows, pending.cdf_snapshot)
        ):
            raise DecodeWorkerError("native proof adapter CDF shape, dtype, or content changed")
        if (
            values.shape != pending.decoded_snapshot.shape
            or values.dtype != pending.decoded_snapshot.dtype
            or not values.flags.c_contiguous
            or not np.array_equal(values, pending.decoded_snapshot)
        ):
            raise DecodeWorkerError("native proof adapter decoded symbols changed")
        return pending.payload

    def assert_consumed(self) -> None:
        if self._pending is not None:
            raise DecodeWorkerError("native proof adapter has a dangling unconsumed decode")


def canonical_json(value: Any) -> bytes:
    """Serialize one finite canonical JSON object with a terminal newline."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DecodeWorkerError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def load_runtime(native_library: Path) -> DecodeRuntime:
    """Load and identify the persisted native arithmetic core and universal CDF tables."""
    path = Path(native_library).resolve()
    if not path.is_file():
        raise DecodeWorkerError(f"native arithmetic library is not a regular file: {path}")
    native_bytes = path.stat().st_size
    native_sha256 = _sha256_file(path)
    native = arithmetic.NativeArithmeticCoder(path)
    loaded = dict(assets.load_tables())
    if set(loaded) != set(assets.ALPHABETS):
        raise DecodeWorkerError("static CDF asset is missing a frozen alphabet")
    return DecodeRuntime(
        native=native,
        tables=MappingProxyType(loaded),
        native_library_bytes=native_bytes,
        native_library_sha256=native_sha256,
    )


def pin_lowest_available_cpu() -> int:
    """Pin this fresh worker process to the lowest CPU in its inherited affinity set."""
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        raise DecodeWorkerError("the frozen resource worker requires Linux CPU affinity")
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise DecodeWorkerError("the inherited CPU affinity set is empty")
    selected = available[0]
    os.sched_setaffinity(0, {selected})
    observed = sorted(os.sched_getaffinity(0))
    if observed != [selected]:
        raise DecodeWorkerError("failed to pin the worker to exactly one CPU")
    return selected


def _frequency_provider(runtime: DecodeRuntime):
    def provide(alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        return runtime.tables[alphabet].frequency_row(location, scale)

    return provide


def _peak_rss_bytes_linux() -> int:
    if not sys.platform.startswith("linux"):
        raise DecodeWorkerError("ru_maxrss byte conversion is frozen for Linux only")
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak_kib <= 0:
        raise DecodeWorkerError("Linux ru_maxrss did not return a positive value")
    return peak_kib * 1024


def _mode(blob: bytes) -> str:
    if not isinstance(blob, bytes):
        raise DecodeWorkerError("worker input must be immutable bytes")
    mode = MAGIC_TO_MODE.get(blob[:5])
    if mode is None:
        raise DecodeWorkerError("worker input has an unsupported or truncated magic")
    return mode


def decode_blob(
    blob: bytes,
    runtime: DecodeRuntime,
    *,
    expected_symbol_sha256: str,
    expected_boundary_state_sha256: str,
    pinned_cpu: int | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Decode one already-read blob and return a JSON-ready resource sample.

    ``runtime`` and ``blob`` must be prepared before this call so their I/O and initialization are
    outside the timed region.  A mismatch against either expected hash is reported in
    ``exactness``; callers must reject records where ``all_exact`` is false.
    """
    if not isinstance(runtime, DecodeRuntime):
        raise DecodeWorkerError("runtime must be prepared by load_runtime")
    mode = _mode(blob)
    expected_symbol = _expected_digest(expected_symbol_sha256, "expected_symbol_sha256")
    expected_boundary = _expected_digest(
        expected_boundary_state_sha256, "expected_boundary_state_sha256"
    )
    if not isinstance(synthetic, bool):
        raise DecodeWorkerError("synthetic must be exactly boolean")
    if pinned_cpu is not None and (
        isinstance(pinned_cpu, bool) or not isinstance(pinned_cpu, int) or pinned_cpu < 0
    ):
        raise DecodeWorkerError("pinned_cpu must be a nonnegative integer or null")
    if pinned_cpu is not None and sorted(os.sched_getaffinity(0)) != [pinned_cpu]:
        raise DecodeWorkerError("pinned_cpu disagrees with the process CPU affinity")

    provider = _frequency_provider(runtime)
    started_ns = time.perf_counter_ns()
    if mode == "sspl1":
        parts = container.extract_sspl1_parts(blob)
        header = parts.header
        symbols = parts.symbols
        symbol_sha256 = parts.symbol_sha256
    else:
        proof_adapter = StrictNativeDecodeProofAdapter(runtime.native)
        decoded = container.decode_container(
            blob,
            arithmetic_decode=proof_adapter.decode,
            arithmetic_encode=proof_adapter.encode,
            frequency_provider=None if mode == "ssp2f" else provider,
        )
        proof_adapter.assert_consumed()
        header = decoded.parsed.header
        symbols = decoded.symbols
        symbol_sha256 = decoded.symbol_sha256
    arrays = execute.boundary_arrays(header, symbols)
    elapsed_ns = time.perf_counter_ns() - started_ns
    if not synthetic and header.n != 8192:
        raise DecodeWorkerError(f"production resource decode requires N=8192, found {header.n}")

    # Hashing and reporting are intentionally outside the bytes-to-boundary timed region.
    boundary_sha256 = execute._array_state_digest(arrays)  # noqa: SLF001
    symbol_exact = symbol_sha256 == expected_symbol
    boundary_exact = boundary_sha256 == expected_boundary
    components = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes": int(value.nbytes),
            "sha256": _sha256_bytes(value.tobytes(order="C")),
        }
        for name, value in sorted(arrays.items())
    }
    affinity = (
        sorted(os.sched_getaffinity(0))
        if sys.platform.startswith("linux") and hasattr(os, "sched_getaffinity")
        else []
    )
    return {
        "schema": WORKER_SCHEMA,
        "mode": mode,
        "magic": blob[:5].decode("ascii"),
        "blob_bytes": len(blob),
        "blob_sha256": _sha256_bytes(blob),
        "n": header.n,
        "bit_tuple": list(header.bits),
        "execution_mode": "synthetic-proof" if synthetic else "production-resource",
        "elapsed_ns": elapsed_ns,
        "peak_rss_bytes": _peak_rss_bytes_linux(),
        "symbol_sha256": symbol_sha256,
        "boundary_state_sha256": boundary_sha256,
        "boundary_components": components,
        "exactness": {
            "expected_symbol_sha256": expected_symbol,
            "expected_boundary_state_sha256": expected_boundary,
            "symbol_hash_exact": symbol_exact,
            "boundary_hash_exact": boundary_exact,
            "all_exact": symbol_exact and boundary_exact,
        },
        "native": {
            "library_bytes": runtime.native_library_bytes,
            "library_sha256": runtime.native_library_sha256,
            "library_loaded_before_timing": True,
            "arithmetic_core_used_in_timed_region": mode != "sspl1",
            "canonical_validation": (
                "native_decode_internal_reencode_and_byte_compare"
                if mode != "sspl1"
                else "not_applicable_sspl1"
            ),
            "redundant_second_native_encode_eliminated": mode != "sspl1",
            "proof_adapter": (
                "one_shot_exact_payload_cdf_identity_content_and_symbol_content"
                if mode != "sspl1"
                else "not_applicable_sspl1"
            ),
            "proof_adapter_measured_overhead": (
                "full_cdf_copy_and_array_equal_inside_timed_region"
                if mode != "sspl1"
                else "not_applicable_sspl1"
            ),
        },
        "assets": {
            "cdf_sha256": assets.CDF_ASSET_SHA256,
            "cdf_bytes": assets.CDF_ASSET_SIZE,
            "loaded_before_timing": True,
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "scope": list(TIMING_SCOPE),
            "imports_before_timing": True,
            "blob_read_before_timing": True,
            "native_library_load_before_timing": True,
            "static_asset_load_before_timing": True,
            "boundary_hash_after_timing": True,
            "renderer_included": False,
            "peak_rss_scope": "full_fresh_process_including_pre_and_post_timing",
        },
        "process": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "platform": sys.platform,
            "pinned_cpu": pinned_cpu,
            "cpu_affinity": affinity,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss_linux_kib_times_1024",
        },
    }


def run_file(
    blob_path: Path,
    native_library: Path,
    *,
    expected_symbol_sha256: str,
    expected_boundary_state_sha256: str,
    pin_cpu: bool = True,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Prepare one fresh-process runtime, read the blob, then invoke the timed decode."""
    if not isinstance(pin_cpu, bool):
        raise DecodeWorkerError("pin_cpu must be exactly boolean")
    path = Path(blob_path).resolve()
    if not path.is_file():
        raise DecodeWorkerError(f"codec blob is not a regular file: {path}")
    blob = path.read_bytes()
    runtime = load_runtime(native_library)
    selected_cpu = pin_lowest_available_cpu() if pin_cpu else None
    return decode_blob(
        blob,
        runtime,
        expected_symbol_sha256=expected_symbol_sha256,
        expected_boundary_state_sha256=expected_boundary_state_sha256,
        pinned_cpu=selected_cpu,
        synthetic=synthetic,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blob", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--expected-symbol-sha256", required=True)
    parser.add_argument("--expected-boundary-state-sha256", required=True)
    parser.add_argument(
        "--no-pin-lowest-cpu",
        action="store_true",
        help="testing-only escape hatch; frozen resource measurements must not use it",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="testing-only N!=8192 mode; frozen resource measurements must not use it",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    record = run_file(
        args.blob,
        args.native_library,
        expected_symbol_sha256=args.expected_symbol_sha256,
        expected_boundary_state_sha256=args.expected_boundary_state_sha256,
        pin_cpu=not args.no_pin_lowest_cpu,
        synthetic=args.synthetic,
    )
    sys.stdout.buffer.write(canonical_json(record))
    sys.stdout.buffer.flush()
    return 0 if record["exactness"]["all_exact"] is True else 2


if __name__ == "__main__":  # pragma: no cover - exercised through fresh subprocess tests
    raise SystemExit(main())


__all__ = [
    "DecodeRuntime",
    "DecodeWorkerError",
    "MAGIC_TO_MODE",
    "TIMING_SCOPE",
    "StrictNativeDecodeProofAdapter",
    "WORKER_SCHEMA",
    "build_parser",
    "canonical_json",
    "decode_blob",
    "load_runtime",
    "main",
    "pin_lowest_available_cpu",
    "run_file",
]
