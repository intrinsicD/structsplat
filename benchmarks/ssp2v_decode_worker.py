"""Unified strict-cold-seal and resource worker for COMP-011.

Production entry points accept only artifact-root-relative blob/native paths.
They walk every path component through no-follow directory descriptors, read a
stable regular-file description, verify those exact bytes, and load executable
dependencies only from sealed verified copies.  Candidate cold seals are
untimed and immutable-verifiable.  Resource samples retain the frozen paired
primary/secondary schedules; their timer starts immediately before strict
container parsing and ends after exact float32 boundary construction.  This
module imports no renderer, target, CUDA JIT, or field fitter.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import resource
import stat
import struct
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as comp009_container
from benchmarks import ssp2e_decode_worker as comp009_worker
from benchmarks import ssp2e_execute as comp009_execute
from benchmarks import ssp2v_container as container


PROTOCOL = "COMP-011-complete-stream-rgb-vq-v2"
TASK_SHA256 = "e715e979b4453e45a8352fbc8fc06614adbeb47193142e0c6b8b02f5523676d0"
WORKER_SCHEMA = "structsplat.comp011.ssp2v-decode-worker.v3"
SUMMARY_SCHEMA = "structsplat.comp011.ssp2v-decode-summary.v3"
COLD_SEAL_SCHEMA = "structsplat.comp011.ssp2v-candidate-cold-seal.v1"
WORKER_MODULE_LABEL = "benchmarks/ssp2v_decode_worker.py"

MAGIC_TO_MODE = {
    b"SSPL1": "sspl1",
    b"SSP2Z": "ssp2z",
    b"SSP2V": "ssp2v",
    b"SSP2E": "ssp2e",
    b"SSP2S": "ssp2s",
    b"SSP2L": "ssp2l",
    b"SSP2F": "ssp2f",
}
BINARY_MAGICS = frozenset(MAGIC_TO_MODE) - {b"SSPL1"}
FROZEN_BIT_TUPLES = frozenset({(12, 6, 6, 8), (16, 8, 8, 8)})
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
TIMING_SCOPE = (
    "strict_complete_container_parse_crc_and_canonicality",
    "zlib_payload_decode",
    "static_or_empirical_model_and_cdf_construction",
    "native_arithmetic_decode_with_internal_canonical_reencode",
    "canonical_empirical_model_regeneration",
    "codebook_lookup_signed_accumulation_and_final_clip",
    "exact_symbol_hash_construction",
    "exact_float32_boundary_array_construction",
)

ScheduleKind = Literal["primary", "secondary"]
LaunchPhase = Literal["warmup", "measured"]


class DecodeWorkerError(RuntimeError):
    """A runtime, stream, launch, timing, or freshness invariant failed."""


_RUNTIME_VERIFICATION_TOKEN = object()


@dataclass(frozen=True)
class DecodeRuntime:
    """Every native/static dependency loaded and identified before timing."""

    native: arithmetic.NativeArithmeticCoder
    tables: Mapping[int, assets.StaticCDFTable]
    native_library_bytes: int
    native_library_sha256: str
    native_library_relative_path: str
    cdf_asset_bytes: int
    cdf_asset_sha256: str
    worker_module_bytes: int
    worker_module_sha256: str
    verification_token: object


@dataclass(frozen=True)
class OpenedArtifact:
    """Immutable bytes obtained through one stable no-follow file description."""

    relative_path: str
    payload: bytes
    device: int
    inode: int


@dataclass(frozen=True)
class LaunchSpec:
    """One exact position in either frozen twenty-child paired schedule."""

    schedule_kind: ScheduleKind
    launch_index: int
    phase: LaunchPhase
    repetition: int | None
    arm: str

    def validated(self) -> LaunchSpec:
        if self.schedule_kind not in ("primary", "secondary"):
            raise DecodeWorkerError("schedule_kind must be primary or secondary")
        if isinstance(self.launch_index, bool) or not isinstance(self.launch_index, int):
            raise DecodeWorkerError("launch_index must be an integer")
        if not 0 <= self.launch_index < 20:
            raise DecodeWorkerError("launch_index must lie in [0,19]")
        allowed_arms = (
            {"q", "candidate"} if self.schedule_kind == "primary" else {"sspl1", "ssp2l"}
        )
        if self.arm not in allowed_arms:
            raise DecodeWorkerError("launch arm is outside its frozen schedule")
        if self.phase == "warmup":
            if self.repetition is not None:
                raise DecodeWorkerError("a warmup launch cannot have a repetition")
        elif self.phase == "measured":
            if (
                isinstance(self.repetition, bool)
                or not isinstance(self.repetition, int)
                or not 0 <= self.repetition < 9
            ):
                raise DecodeWorkerError("a measured repetition must lie in [0,8]")
        else:
            raise DecodeWorkerError("launch phase must be warmup or measured")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validated()
        return {
            "schedule_kind": self.schedule_kind,
            "launch_index": self.launch_index,
            "phase": self.phase,
            "repetition": self.repetition,
            "arm": self.arm,
        }


def canonical_json(value: Any) -> bytes:
    """Serialize one finite canonical JSON value with a terminal newline."""

    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DecodeWorkerError(f"value is not finite canonical JSON: {exc}") from exc


def _digest(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DecodeWorkerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DecodeWorkerError(f"{name} must be a positive integer")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative_parts(value: str | os.PathLike[str], name: str) -> tuple[str, ...]:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise DecodeWorkerError(f"{name} must be a nonempty canonical POSIX relative path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise DecodeWorkerError(f"{name} must be artifact-root-relative without dot components")
    return path.parts


def _directory_flags() -> int:
    required = ("O_PATH", "O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise DecodeWorkerError("secure artifact access requires Linux no-follow directory opens")
    # Directory descriptors are traversal anchors only.  O_PATH preserves the
    # component-by-component no-symlink walk without requesting READ_DIR on
    # every ancestor (notably ``/``), so a Landlock worker can reach an exact
    # allowlisted artifact subtree without broadening its readable surface.
    # The eventual regular-file open still requests O_RDONLY and is therefore
    # mediated by Landlock's READ_FILE right.
    return os.O_PATH | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_artifact_root(artifact_root: Path) -> int:
    """Open an absolute directory by walking every component without following links."""

    raw = os.fspath(artifact_root)
    path = PurePosixPath(raw)
    if not path.is_absolute() or any(part == ".." for part in path.parts):
        raise DecodeWorkerError("artifact_root must be an absolute path without parent traversal")
    descriptor = os.open("/", _directory_flags())
    try:
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                raise DecodeWorkerError("artifact_root contains a noncanonical component")
            next_descriptor = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except (OSError, DecodeWorkerError) as exc:
        os.close(descriptor)
        if isinstance(exc, DecodeWorkerError):
            raise
        raise DecodeWorkerError(f"artifact_root is not a no-symlink directory: {exc}") from exc


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_fd(descriptor: int, name: str) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink <= 0:
        raise DecodeWorkerError(f"{name} is not a linked regular file")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    payload = b"".join(chunks)
    if _stable_stat_identity(before) != _stable_stat_identity(after):
        raise DecodeWorkerError(f"{name} changed while its opened bytes were read")
    if len(payload) != before.st_size:
        raise DecodeWorkerError(f"{name} byte count differs from its stable opened inode")
    return payload, before


def _read_artifact_file(
    root_descriptor: int,
    relative_path: str | os.PathLike[str],
    name: str,
) -> OpenedArtifact:
    parts = _relative_parts(relative_path, name)
    directory = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            next_directory = os.open(component, _directory_flags(), dir_fd=directory)
            os.close(directory)
            directory = next_directory
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(parts[-1], flags, dir_fd=directory)
        try:
            payload, opened_stat = _read_regular_fd(descriptor, name)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise DecodeWorkerError(f"unsafe or unavailable {name}: {exc}") from exc
    finally:
        os.close(directory)
    return OpenedArtifact(
        relative_path=PurePosixPath(*parts).as_posix(),
        payload=payload,
        device=opened_stat.st_dev,
        inode=opened_stat.st_ino,
    )


def _read_absolute_regular_file(path: Path, name: str) -> bytes:
    raw = os.fspath(path)
    absolute = PurePosixPath(raw)
    if not absolute.is_absolute():
        raise DecodeWorkerError(f"{name} source path is not absolute")
    root_descriptor = os.open("/", _directory_flags())
    try:
        opened = _read_artifact_file(
            root_descriptor,
            PurePosixPath(*absolute.parts[1:]).as_posix(),
            name,
        )
        return opened.payload
    finally:
        os.close(root_descriptor)


def _verify_identity(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_bytes: int,
    name: str,
) -> tuple[int, str]:
    expected_digest = _digest(expected_sha256, f"expected_{name}_sha256")
    expected_size = _positive_int(expected_bytes, f"expected_{name}_bytes")
    observed_digest = _sha256_bytes(payload)
    if len(payload) != expected_size or observed_digest != expected_digest:
        raise DecodeWorkerError(f"persisted {name} identity mismatch")
    return len(payload), observed_digest


def _sealed_memfd(payload: bytes, name: str) -> int:
    if not sys.platform.startswith("linux"):
        raise DecodeWorkerError("verified dependency loading requires Linux sealed memfd support")
    add_seals = getattr(fcntl, "F_ADD_SEALS", 1033)
    seal_seal = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
    seal_shrink = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
    seal_grow = getattr(fcntl, "F_SEAL_GROW", 0x0004)
    seal_write = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
    flags = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    if hasattr(os, "memfd_create"):
        descriptor = os.memfd_create(name, flags)
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        create = getattr(libc, "memfd_create", None)
        if create is None:
            raise DecodeWorkerError("the Linux C library does not expose memfd_create")
        create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        create.restype = ctypes.c_int
        descriptor = int(create(name.encode("ascii"), flags))
        if descriptor < 0:
            error = ctypes.get_errno()
            raise DecodeWorkerError(f"memfd_create failed: {os.strerror(error)}")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise DecodeWorkerError("sealed dependency copy made no forward progress")
            view = view[written:]
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = seal_seal | seal_shrink | seal_grow | seal_write
        fcntl.fcntl(descriptor, add_seals, seals)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _native_from_verified_bytes(payload: bytes) -> arithmetic.NativeArithmeticCoder:
    descriptor = _sealed_memfd(payload, "ssp2e-range-v1")
    try:
        library = ctypes.CDLL(f"/proc/self/fd/{descriptor}")
    except Exception as exc:
        raise DecodeWorkerError(f"verified native arithmetic load failed: {exc}") from exc
    finally:
        os.close(descriptor)
    coder = arithmetic.NativeArithmeticCoder.__new__(arithmetic.NativeArithmeticCoder)
    coder._library = library  # noqa: SLF001 - construct the frozen wrapper from verified bytes.
    uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
    uint32_pointer = ctypes.POINTER(ctypes.c_uint32)
    library.ssp2e_encode.argtypes = [
        uint32_pointer,
        uint32_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        uint8_pointer,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.ssp2e_encode.restype = ctypes.c_int
    library.ssp2e_decode.argtypes = [
        uint8_pointer,
        ctypes.c_size_t,
        uint32_pointer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        uint32_pointer,
    ]
    library.ssp2e_decode.restype = ctypes.c_int
    library.ssp2e_last_error.argtypes = []
    library.ssp2e_last_error.restype = ctypes.c_char_p
    return coder


def _tables_from_verified_bytes(payload: bytes) -> Mapping[int, assets.StaticCDFTable]:
    descriptor = _sealed_memfd(payload, "ssp2e-cdf-v1")
    try:
        return assets.load_tables(Path(f"/proc/self/fd/{descriptor}"))
    except Exception as exc:
        raise DecodeWorkerError(f"verified static CDF load failed: {exc}") from exc
    finally:
        os.close(descriptor)


def load_runtime(
    artifact_root: Path,
    native_library: str | os.PathLike[str],
    *,
    expected_native_library_sha256: str,
    expected_native_library_bytes: int,
    expected_worker_module_sha256: str,
    expected_worker_module_bytes: int,
) -> DecodeRuntime:
    """Securely acquire, verify, then load every executable decode dependency."""

    root_descriptor = _open_artifact_root(Path(artifact_root))
    try:
        opened_native = _read_artifact_file(
            root_descriptor, native_library, "native_library"
        )
    finally:
        os.close(root_descriptor)
    return _runtime_from_opened_native(
        opened_native,
        expected_native_library_sha256=expected_native_library_sha256,
        expected_native_library_bytes=expected_native_library_bytes,
        expected_worker_module_sha256=expected_worker_module_sha256,
        expected_worker_module_bytes=expected_worker_module_bytes,
    )


def _runtime_from_opened_native(
    opened_native: OpenedArtifact,
    *,
    expected_native_library_sha256: str,
    expected_native_library_bytes: int,
    expected_worker_module_sha256: str,
    expected_worker_module_bytes: int,
) -> DecodeRuntime:
    observed_bytes, observed_sha256 = _verify_identity(
        opened_native.payload,
        expected_sha256=expected_native_library_sha256,
        expected_bytes=expected_native_library_bytes,
        name="native_library",
    )
    module_payload = _read_absolute_regular_file(Path(__file__), "worker_module")
    module_bytes, module_sha256 = _verify_identity(
        module_payload,
        expected_sha256=expected_worker_module_sha256,
        expected_bytes=expected_worker_module_bytes,
        name="worker_module",
    )
    cdf_path = assets.ASSET_DIRECTORY / assets.CDF_ASSET_NAME
    cdf_payload = _read_absolute_regular_file(cdf_path, "cdf_asset")
    cdf_bytes, cdf_sha256 = _verify_identity(
        cdf_payload,
        expected_sha256=assets.CDF_ASSET_SHA256,
        expected_bytes=assets.CDF_ASSET_SIZE,
        name="cdf_asset",
    )
    native = _native_from_verified_bytes(opened_native.payload)
    loaded = dict(_tables_from_verified_bytes(cdf_payload))
    if set(loaded) != set(assets.ALPHABETS):
        raise DecodeWorkerError("static CDF asset is missing a frozen alphabet")
    return DecodeRuntime(
        native=native,
        tables=MappingProxyType(loaded),
        native_library_bytes=observed_bytes,
        native_library_sha256=observed_sha256,
        native_library_relative_path=opened_native.relative_path,
        cdf_asset_bytes=cdf_bytes,
        cdf_asset_sha256=cdf_sha256,
        worker_module_bytes=module_bytes,
        worker_module_sha256=module_sha256,
        verification_token=_RUNTIME_VERIFICATION_TOKEN,
    )


def pin_lowest_available_cpu() -> int:
    """Pin this fresh Linux process to the lowest inherited CPU."""

    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        raise DecodeWorkerError("the resource worker requires Linux CPU affinity")
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise DecodeWorkerError("the inherited CPU affinity set is empty")
    selected = available[0]
    os.sched_setaffinity(0, {selected})
    if sorted(os.sched_getaffinity(0)) != [selected]:
        raise DecodeWorkerError("failed to pin the worker to one CPU")
    return selected


def _thread_environment() -> dict[str, str]:
    observed = {name: os.environ.get(name, "") for name in THREAD_VARIABLES}
    if any(value != "1" for value in observed.values()):
        raise DecodeWorkerError("all frozen thread environment variables must equal one")
    return observed


def _peak_rss_bytes_linux() -> int:
    if not sys.platform.startswith("linux"):
        raise DecodeWorkerError("ru_maxrss conversion is frozen for Linux only")
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak_kib <= 0:
        raise DecodeWorkerError("Linux ru_maxrss did not return a positive value")
    return peak_kib * 1024


def _dispatch(blob: bytes) -> tuple[str, int]:
    """Dispatch solely from complete-stream magic and its frozen wire version."""

    if not isinstance(blob, bytes):
        raise DecodeWorkerError("worker input must be immutable bytes")
    magic = blob[:5]
    mode = MAGIC_TO_MODE.get(magic)
    if mode is None:
        raise DecodeWorkerError("worker input has unsupported or truncated magic")
    if magic == b"SSPL1":
        return mode, 1
    if len(blob) < 6 or blob[5] != 1:
        raise DecodeWorkerError("worker input has an unsupported or truncated binary version")
    return mode, 1


def _frequency_provider(runtime: DecodeRuntime):
    def provide(alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        try:
            return runtime.tables[alphabet].frequency_row(location, scale)
        except (KeyError, IndexError) as exc:
            raise DecodeWorkerError("static CDF lookup is outside the frozen table") from exc

    return provide


def _decode_ssp2v_native(
    blob: bytes, runtime: DecodeRuntime
) -> tuple[container.SSP2VHeader, dict[str, np.ndarray], str]:
    """Decode SSP2V without an extra Python arithmetic encode in the timed path."""

    parsed = container.parse_container(blob)
    if parsed.magic != b"SSP2V" or parsed.descriptor is None:
        raise DecodeWorkerError("SSP2V dispatch did not produce an SSP2V descriptor")
    descriptor = parsed.descriptor
    codebooks = container.parse_codebooks(parsed.payloads["codebook_raw"], descriptor)
    frequencies = container.parse_index_model(parsed.payloads["index_model_zlib"], descriptor)
    decoded_indices: list[np.ndarray] = []
    for stage, (alphabet, stage_frequencies) in enumerate(
        zip(descriptor.stage_alphabets, frequencies, strict=True)
    ):
        rows = container.cdf_rows(stage_frequencies)
        name = f"index_{stage}_arith"
        # Native decode performs its own canonical re-encode and byte comparison.
        decoded = runtime.native.decode(parsed.payloads[name], rows, container.ROW_COUNT)
        values = np.asarray(decoded)
        if (
            values.shape != (container.ROW_COUNT,)
            or values.dtype != np.uint32
            or not values.flags.c_contiguous
            or (values.size and int(values.max()) >= alphabet)
        ):
            raise DecodeWorkerError(f"native index_{stage} output is not canonical")
        decoded_indices.append(values)

    canonical_model = container.serialize_index_model(decoded_indices, descriptor)
    if canonical_model != parsed.payloads["index_model_zlib"]:
        raise DecodeWorkerError("SSP2V empirical model regeneration mismatch")
    if container.serialize_codebooks(codebooks, descriptor) != parsed.payloads["codebook_raw"]:
        raise DecodeWorkerError("SSP2V codebook regeneration mismatch")
    if descriptor.to_bytes() != parsed.payloads["descriptor_raw"]:
        raise DecodeWorkerError("SSP2V descriptor regeneration mismatch")

    _raw_rgb, clipped_rgb = container.reconstruct_rgb(descriptor, codebooks, decoded_indices)
    common = container._decode_common(parsed, include_colors=False)  # noqa: SLF001
    common.update(
        {
            "red": np.ascontiguousarray(clipped_rgb[:, 0], dtype=np.uint32),
            "green": np.ascontiguousarray(clipped_rgb[:, 1], dtype=np.uint32),
            "blue": np.ascontiguousarray(clipped_rgb[:, 2], dtype=np.uint32),
        }
    )
    symbols = {name: common[name] for name in container.SYMBOL_CHANNELS}
    symbol_sha256 = container._symbol_sha256(symbols)  # noqa: SLF001
    canonical_payloads = dict(parsed.payloads)
    canonical_payloads["index_model_zlib"] = canonical_model
    canonical_payloads["codebook_raw"] = container.serialize_codebooks(codebooks, descriptor)
    canonical_payloads["descriptor_raw"] = descriptor.to_bytes()
    reencoded = container._assemble(parsed.magic, parsed.header, canonical_payloads)  # noqa: SLF001
    if reencoded != blob:
        raise DecodeWorkerError("SSP2V complete canonical re-encoding mismatch")
    return parsed.header, symbols, symbol_sha256


def _strict_decode(
    mode: str, blob: bytes, runtime: DecodeRuntime
) -> tuple[Any, dict[str, np.ndarray], str]:
    try:
        if mode == "sspl1":
            parts = comp009_container.extract_sspl1_parts(blob)
            return parts.header, parts.symbols, parts.symbol_sha256
        if mode == "ssp2z":
            decoded_z = container.decode_ssp2z(blob)
            return decoded_z.parsed.header, decoded_z.symbols, decoded_z.symbol_sha256
        if mode == "ssp2v":
            return _decode_ssp2v_native(blob, runtime)
        proof = comp009_worker.StrictNativeDecodeProofAdapter(runtime.native)
        decoded = comp009_container.decode_container(
            blob,
            arithmetic_decode=proof.decode,
            arithmetic_encode=proof.encode,
            frequency_provider=None if mode == "ssp2f" else _frequency_provider(runtime),
        )
        proof.assert_consumed()
        return decoded.parsed.header, decoded.symbols, decoded.symbol_sha256
    except DecodeWorkerError:
        raise
    except Exception as exc:
        raise DecodeWorkerError(f"{mode} strict decode failed: {exc}") from exc


def _bytes_record(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": _sha256_bytes(payload)}


def component_records_sha256(records: Mapping[str, Mapping[str, Any]]) -> str:
    """Seal a canonical complete-stream component map."""

    if not isinstance(records, Mapping) or "total" not in records:
        raise DecodeWorkerError("stream component records must contain a total record")
    normalized: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(record, Mapping)
            or set(record) != {"bytes", "sha256"}
        ):
            raise DecodeWorkerError("stream component record shape is invalid")
        normalized[name] = {
            "bytes": _positive_int(record["bytes"], f"components.{name}.bytes"),
            "sha256": _digest(record["sha256"], f"components.{name}.sha256"),
        }
    return _sha256_bytes(canonical_json(normalized))


def stream_component_records(blob: bytes) -> dict[str, dict[str, Any]]:
    """Return the exact framing/payload partition used by candidate cold seals."""

    mode, _version = _dispatch(blob)
    if mode == "sspl1":
        parts = comp009_container.extract_sspl1_parts(blob)
        if len(blob) < 9:
            raise DecodeWorkerError("SSPL1 component parse found truncated framing")
        header_length = struct.unpack_from("<I", blob, 5)[0]
        header_begin = 9
        header_end = header_begin + header_length
        if header_end > len(blob):
            raise DecodeWorkerError("SSPL1 component parse found a truncated header")
        records = {
            "magic": _bytes_record(blob[:5]),
            "header_length": _bytes_record(blob[5:9]),
            "header_json": _bytes_record(blob[header_begin:header_end]),
        }
        offset = header_end
        for name in ("means", "scales", "rotation", "colors"):
            if offset + 4 > len(blob):
                raise DecodeWorkerError(f"SSPL1 component parse truncated {name} length")
            payload_length = struct.unpack_from("<I", blob, offset)[0]
            prefix = blob[offset : offset + 4]
            offset += 4
            end = offset + payload_length
            if end > len(blob):
                raise DecodeWorkerError(f"SSPL1 component parse truncated {name} payload")
            payload = blob[offset:end]
            if payload != parts.payloads[name]:
                raise DecodeWorkerError(f"SSPL1 component {name} differs from strict parse")
            records[f"{name}_length"] = _bytes_record(prefix)
            records[f"{name}_zlib"] = _bytes_record(payload)
            offset = end
        if offset != len(blob):
            raise DecodeWorkerError("SSPL1 component parse found an unclaimed suffix")
    elif mode in {"ssp2z", "ssp2v"}:
        parsed = container.parse_container(blob)
        directory_begin = container.OUTER.size + container.HEADER.size
        payloads = {
            "outer": blob[: container.OUTER.size],
            "header": blob[container.OUTER.size : directory_begin],
            "directory": blob[directory_begin : parsed.prefix_bytes],
            **parsed.payloads,
        }
        records = {name: _bytes_record(payload) for name, payload in payloads.items()}
    else:
        parsed = comp009_container.parse_container(blob)
        directory_begin = comp009_container.OUTER.size + comp009_container.HEADER.size
        payloads = {
            "outer": blob[: comp009_container.OUTER.size],
            "header": blob[comp009_container.OUTER.size : directory_begin],
            "directory": blob[directory_begin : parsed.prefix_bytes],
            **parsed.payloads,
        }
        records = {name: _bytes_record(payload) for name, payload in payloads.items()}
    records["total"] = _bytes_record(blob)
    if sum(record["bytes"] for name, record in records.items() if name != "total") != len(blob):
        raise DecodeWorkerError("stream component records do not partition the complete blob")
    component_records_sha256(records)
    return records


def _array_component_records(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, value in sorted(arrays.items()):
        array = np.asarray(value)
        if not array.flags.c_contiguous:
            raise DecodeWorkerError(f"decoded array {name} is not C-contiguous")
        records[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes": int(array.nbytes),
            "sha256": _sha256_bytes(array.tobytes(order="C")),
        }
    return records


def _array_component_sha256(records: Mapping[str, Mapping[str, Any]]) -> str:
    if not isinstance(records, Mapping) or not records:
        raise DecodeWorkerError("array component records must be a nonempty mapping")
    for name, record in records.items():
        if (
            not isinstance(name, str)
            or not isinstance(record, Mapping)
            or set(record) != {"dtype", "shape", "bytes", "sha256"}
            or not isinstance(record["dtype"], str)
            or not isinstance(record["shape"], list)
        ):
            raise DecodeWorkerError("array component record shape is invalid")
        _positive_int(record["bytes"], f"array_components.{name}.bytes")
        _digest(record["sha256"], f"array_components.{name}.sha256")
    return _sha256_bytes(canonical_json(dict(records)))


def _validate_array_component_layout(
    records: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
    name: str,
) -> None:
    if set(records) != set(expected):
        raise DecodeWorkerError(f"{name} component names differ from the frozen boundary")
    for component, (dtype, shape) in expected.items():
        record = records[component]
        if (
            record.get("dtype") != dtype
            or record.get("shape") != list(shape)
            or record.get("bytes") != int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        ):
            raise DecodeWorkerError(f"{name}.{component} layout differs from the frozen boundary")


def _source_bindings(runtime: DecodeRuntime) -> dict[str, Any]:
    return {
        "worker_module": {
            "source_label": WORKER_MODULE_LABEL,
            "bytes": runtime.worker_module_bytes,
            "sha256": runtime.worker_module_sha256,
            "identity_verified_before_operation": True,
        },
        "native_library": {
            "artifact_relative_path": runtime.native_library_relative_path,
            "bytes": runtime.native_library_bytes,
            "sha256": runtime.native_library_sha256,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
        "cdf_asset": {
            "source_label": f"benchmarks/assets/{assets.CDF_ASSET_NAME}",
            "bytes": runtime.cdf_asset_bytes,
            "sha256": runtime.cdf_asset_sha256,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
    }


def _validate_source_bindings(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "worker_module",
        "native_library",
        "cdf_asset",
    }:
        raise DecodeWorkerError("decode source bindings are absent or malformed")
    worker_module = value["worker_module"]
    native = value["native_library"]
    cdf = value["cdf_asset"]
    if (
        not isinstance(worker_module, Mapping)
        or set(worker_module)
        != {"source_label", "bytes", "sha256", "identity_verified_before_operation"}
        or worker_module.get("source_label") != WORKER_MODULE_LABEL
        or worker_module.get("identity_verified_before_operation") is not True
        or not isinstance(native, Mapping)
        or set(native)
        != {
            "artifact_relative_path",
            "bytes",
            "sha256",
            "identity_verified_from_same_opened_bytes_before_load",
            "loaded_from_sealed_verified_bytes",
        }
        or native.get("identity_verified_from_same_opened_bytes_before_load") is not True
        or native.get("loaded_from_sealed_verified_bytes") is not True
        or not isinstance(cdf, Mapping)
        or set(cdf)
        != {
            "source_label",
            "bytes",
            "sha256",
            "identity_verified_from_same_opened_bytes_before_load",
            "loaded_from_sealed_verified_bytes",
        }
        or cdf.get("source_label") != f"benchmarks/assets/{assets.CDF_ASSET_NAME}"
        or cdf.get("bytes") != assets.CDF_ASSET_SIZE
        or cdf.get("sha256") != assets.CDF_ASSET_SHA256
        or cdf.get("identity_verified_from_same_opened_bytes_before_load") is not True
        or cdf.get("loaded_from_sealed_verified_bytes") is not True
    ):
        raise DecodeWorkerError("decode source bindings are absent or malformed")
    _positive_int(worker_module.get("bytes"), "worker_module.bytes")
    _digest(worker_module.get("sha256"), "worker_module.sha256")
    _relative_parts(native.get("artifact_relative_path"), "native artifact path")
    _positive_int(native.get("bytes"), "native.bytes")
    _digest(native.get("sha256"), "native.sha256")


def _observed_affinity() -> list[int]:
    if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
        raise DecodeWorkerError("the frozen worker requires Linux CPU affinity observation")
    observed = sorted(os.sched_getaffinity(0))
    if not observed or any(isinstance(value, bool) or value < 0 for value in observed):
        raise DecodeWorkerError("the observed CPU affinity set is invalid")
    return observed


_COLD_SEAL_KEYS = frozenset(
    {
        "schema",
        "protocol",
        "task_sha256",
        "operation",
        "mode",
        "magic",
        "format_version",
        "n",
        "bit_tuple",
        "blob",
        "symbols",
        "boundary",
        "stream_components",
        "source_bindings",
        "process_environment",
        "canonical_decode_reencode_verified",
        "cold_decode_state_sha256",
        "cold_seal_sha256",
    }
)

_RESOURCE_SAMPLE_KEYS = frozenset(
    {
        "schema",
        "protocol",
        "task_sha256",
        "operation",
        "mode",
        "magic",
        "format_version",
        "blob_bytes",
        "blob_sha256",
        "blob_hash_verified_before_timing",
        "blob_acquisition",
        "n",
        "bit_tuple",
        "execution_mode",
        "launch",
        "freshness",
        "elapsed_ns",
        "peak_rss_bytes",
        "symbol_sha256",
        "boundary_state_sha256",
        "boundary_components",
        "exactness",
        "source_bindings",
        "module",
        "native",
        "assets",
        "timing",
        "process",
    }
)


def candidate_cold_seal(
    blob: bytes,
    runtime: DecodeRuntime,
    *,
    artifact_relative_path: str,
    expected_blob_sha256: str,
    expected_blob_bytes: int,
    expected_symbol_sha256: str,
    expected_boundary_state_sha256: str,
    expected_stream_components_sha256: str,
) -> dict[str, Any]:
    """Strictly cold-decode and seal one prebound candidate without resource timing."""

    if (
        not isinstance(runtime, DecodeRuntime)
        or runtime.verification_token is not _RUNTIME_VERIFICATION_TOKEN
    ):
        raise DecodeWorkerError("runtime must be prepared by load_runtime")
    _relative_parts(artifact_relative_path, "artifact_relative_path")
    expected_blob = _digest(expected_blob_sha256, "expected_blob_sha256")
    expected_bytes = _positive_int(expected_blob_bytes, "expected_blob_bytes")
    expected_symbol = _digest(expected_symbol_sha256, "expected_symbol_sha256")
    expected_boundary = _digest(
        expected_boundary_state_sha256, "expected_boundary_state_sha256"
    )
    expected_components = _digest(
        expected_stream_components_sha256, "expected_stream_components_sha256"
    )
    if len(blob) != expected_bytes or _sha256_bytes(blob) != expected_blob:
        raise DecodeWorkerError("candidate cold-seal blob identity mismatch")
    mode, version = _dispatch(blob)
    header, symbols, symbol_sha256 = _strict_decode(mode, blob, runtime)
    if header.n != container.ROW_COUNT:
        raise DecodeWorkerError(
            f"candidate cold seal requires N={container.ROW_COUNT}, found {header.n}"
        )
    if tuple(header.bits) not in FROZEN_BIT_TUPLES:
        raise DecodeWorkerError("candidate cold seal has a non-frozen COMP-011 bit tuple")
    boundary = comp009_execute.boundary_arrays(header, symbols)
    boundary_sha256 = comp009_execute._array_state_digest(boundary)  # noqa: SLF001
    stream_records = stream_component_records(blob)
    stream_sha256 = component_records_sha256(stream_records)
    if (
        symbol_sha256 != expected_symbol
        or boundary_sha256 != expected_boundary
        or stream_sha256 != expected_components
    ):
        raise DecodeWorkerError("candidate cold-seal decoded evidence mismatch")
    symbol_records = _array_component_records(symbols)
    boundary_records = _array_component_records(boundary)
    state = {
        "mode": mode,
        "blob_bytes": len(blob),
        "blob_sha256": expected_blob,
        "symbol_sha256": symbol_sha256,
        "symbol_components_sha256": _array_component_sha256(symbol_records),
        "boundary_state_sha256": boundary_sha256,
        "boundary_components_sha256": _array_component_sha256(boundary_records),
        "stream_components_sha256": stream_sha256,
    }
    record: dict[str, Any] = {
        "schema": COLD_SEAL_SCHEMA,
        "protocol": PROTOCOL,
        "task_sha256": TASK_SHA256,
        "operation": "candidate-cold-seal",
        "mode": mode,
        "magic": blob[:5].decode("ascii"),
        "format_version": version,
        "n": int(header.n),
        "bit_tuple": list(header.bits),
        "blob": {
            "artifact_relative_path": artifact_relative_path,
            "bytes": len(blob),
            "sha256": expected_blob,
            "opened_artifact_root_relative_no_symlink": True,
            "identity_verified_from_same_stable_opened_bytes": True,
        },
        "symbols": {
            "state_sha256": symbol_sha256,
            "components_sha256": state["symbol_components_sha256"],
            "components": symbol_records,
        },
        "boundary": {
            "state_sha256": boundary_sha256,
            "components_sha256": state["boundary_components_sha256"],
            "components": boundary_records,
        },
        "stream_components": {
            "state_sha256": stream_sha256,
            "records": stream_records,
        },
        "source_bindings": _source_bindings(runtime),
        "process_environment": {
            "thread_environment": _thread_environment(),
            "cpu_affinity": _observed_affinity(),
            "observed_exactly": True,
            "resource_schedule_sample": False,
        },
        "canonical_decode_reencode_verified": True,
        "cold_decode_state_sha256": _sha256_bytes(canonical_json(state)),
    }
    record["cold_seal_sha256"] = _sha256_bytes(canonical_json(record))
    verify_candidate_cold_seal(
        record,
        expected_blob_sha256=expected_blob,
        expected_blob_bytes=expected_bytes,
        expected_symbol_sha256=expected_symbol,
        expected_boundary_state_sha256=expected_boundary,
        expected_stream_components_sha256=expected_components,
    )
    return record


def verify_candidate_cold_seal(
    record: Mapping[str, Any],
    *,
    expected_blob_sha256: str | None = None,
    expected_blob_bytes: int | None = None,
    expected_symbol_sha256: str | None = None,
    expected_boundary_state_sha256: str | None = None,
    expected_stream_components_sha256: str | None = None,
) -> None:
    """Verify one immutable candidate cold-seal record and reject stale schemas."""

    if not isinstance(record, Mapping) or frozenset(record) != _COLD_SEAL_KEYS:
        raise DecodeWorkerError("candidate cold seal has an old or malformed schema")
    blob = record.get("blob")
    symbols = record.get("symbols")
    boundary = record.get("boundary")
    stream_components = record.get("stream_components")
    source_bindings = record.get("source_bindings")
    process_environment = record.get("process_environment")
    if (
        record.get("schema") != COLD_SEAL_SCHEMA
        or record.get("protocol") != PROTOCOL
        or record.get("task_sha256") != TASK_SHA256
        or record.get("operation") != "candidate-cold-seal"
        or record.get("canonical_decode_reencode_verified") is not True
        or not isinstance(blob, Mapping)
        or set(blob)
        != {
            "artifact_relative_path",
            "bytes",
            "sha256",
            "opened_artifact_root_relative_no_symlink",
            "identity_verified_from_same_stable_opened_bytes",
        }
        or blob.get("opened_artifact_root_relative_no_symlink") is not True
        or blob.get("identity_verified_from_same_stable_opened_bytes") is not True
        or not isinstance(symbols, Mapping)
        or set(symbols) != {"state_sha256", "components_sha256", "components"}
        or not isinstance(boundary, Mapping)
        or set(boundary) != {"state_sha256", "components_sha256", "components"}
        or not isinstance(stream_components, Mapping)
        or set(stream_components) != {"state_sha256", "records"}
        or not isinstance(source_bindings, Mapping)
        or set(source_bindings) != {"worker_module", "native_library", "cdf_asset"}
        or not isinstance(process_environment, Mapping)
        or set(process_environment)
        != {
            "thread_environment",
            "cpu_affinity",
            "observed_exactly",
            "resource_schedule_sample",
        }
        or process_environment.get("observed_exactly") is not True
        or process_environment.get("resource_schedule_sample") is not False
    ):
        raise DecodeWorkerError("candidate cold seal has an old or malformed schema")
    _validate_source_bindings(source_bindings)
    _relative_parts(blob["artifact_relative_path"], "artifact_relative_path")
    blob_bytes = _positive_int(blob["bytes"], "blob.bytes")
    blob_sha256 = _digest(blob["sha256"], "blob.sha256")
    expected_mode = MAGIC_TO_MODE.get(str(record.get("magic", "")).encode("ascii", errors="ignore"))
    bit_tuple = record.get("bit_tuple")
    if (
        record.get("mode") not in set(MAGIC_TO_MODE.values())
        or expected_mode != record.get("mode")
        or record.get("format_version") != 1
        or record.get("n") != container.ROW_COUNT
        or not isinstance(bit_tuple, list)
        or tuple(bit_tuple) not in FROZEN_BIT_TUPLES
    ):
        raise DecodeWorkerError("candidate cold seal format identity is invalid")
    symbol_sha256 = _digest(symbols["state_sha256"], "symbols.state_sha256")
    boundary_sha256 = _digest(boundary["state_sha256"], "boundary.state_sha256")
    if _array_component_sha256(symbols["components"]) != symbols["components_sha256"]:
        raise DecodeWorkerError("candidate cold seal symbol components were altered")
    if _array_component_sha256(boundary["components"]) != boundary["components_sha256"]:
        raise DecodeWorkerError("candidate cold seal boundary components were altered")
    stream_sha256 = component_records_sha256(stream_components["records"])
    if stream_sha256 != stream_components["state_sha256"]:
        raise DecodeWorkerError("candidate cold seal stream components were altered")
    total = stream_components["records"].get("total")
    if (
        not isinstance(total, Mapping)
        or total.get("bytes") != blob_bytes
        or total.get("sha256") != blob_sha256
    ):
        raise DecodeWorkerError("candidate cold seal total component differs from the blob")
    _validate_array_component_layout(
        symbols["components"],
        {
            name: (np.dtype(np.uint32).str, (container.ROW_COUNT,))
            for name in container.SYMBOL_CHANNELS
        },
        "symbols",
    )
    _validate_array_component_layout(
        boundary["components"],
        {
            "means": (np.dtype(np.float32).str, (container.ROW_COUNT, 2)),
            "log_scales": (np.dtype(np.float32).str, (container.ROW_COUNT, 2)),
            "rotations": (np.dtype(np.float32).str, (container.ROW_COUNT,)),
            "colors": (np.dtype(np.float32).str, (container.ROW_COUNT, 3)),
        },
        "boundary",
    )
    threads = process_environment["thread_environment"]
    affinity = process_environment["cpu_affinity"]
    if (
        not isinstance(threads, Mapping)
        or set(threads) != set(THREAD_VARIABLES)
        or any(threads[name] != "1" for name in THREAD_VARIABLES)
        or not isinstance(affinity, list)
        or not affinity
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in affinity)
        or affinity != sorted(set(affinity))
    ):
        raise DecodeWorkerError("candidate cold seal process environment is invalid")
    state = {
        "mode": record["mode"],
        "blob_bytes": blob_bytes,
        "blob_sha256": blob_sha256,
        "symbol_sha256": symbol_sha256,
        "symbol_components_sha256": symbols["components_sha256"],
        "boundary_state_sha256": boundary_sha256,
        "boundary_components_sha256": boundary["components_sha256"],
        "stream_components_sha256": stream_sha256,
    }
    if _sha256_bytes(canonical_json(state)) != record.get("cold_decode_state_sha256"):
        raise DecodeWorkerError("candidate cold decode state seal mismatch")
    unsigned = dict(record)
    observed_seal = _digest(unsigned.pop("cold_seal_sha256"), "cold_seal_sha256")
    if _sha256_bytes(canonical_json(unsigned)) != observed_seal:
        raise DecodeWorkerError("candidate cold seal integrity mismatch")
    expected_values = (
        (expected_blob_sha256, blob_sha256, "blob SHA-256"),
        (expected_symbol_sha256, symbol_sha256, "symbol SHA-256"),
        (expected_boundary_state_sha256, boundary_sha256, "boundary SHA-256"),
        (expected_stream_components_sha256, stream_sha256, "component SHA-256"),
    )
    for expected, observed, name in expected_values:
        if expected is not None and _digest(expected, f"expected_{name}") != observed:
            raise DecodeWorkerError(f"candidate cold seal {name} differs from authority")
    if expected_blob_bytes is not None and _positive_int(
        expected_blob_bytes, "expected_blob_bytes"
    ) != blob_bytes:
        raise DecodeWorkerError("candidate cold seal byte count differs from authority")


def decode_blob(
    blob: bytes,
    runtime: DecodeRuntime,
    *,
    expected_blob_sha256: str,
    expected_blob_bytes: int,
    expected_symbol_sha256: str,
    expected_boundary_state_sha256: str,
    measurement_session_sha256: str,
    launch: LaunchSpec,
    pinned_cpu: int | None,
    artifact_relative_path: str | None = None,
    affinity_before_pin: Sequence[int] | None = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Time one already-loaded, prebound complete stream through float32 boundary."""

    if (
        not isinstance(runtime, DecodeRuntime)
        or runtime.verification_token is not _RUNTIME_VERIFICATION_TOKEN
    ):
        raise DecodeWorkerError("runtime must be prepared by load_runtime")
    mode, version = _dispatch(blob)
    expected_blob = _digest(expected_blob_sha256, "expected_blob_sha256")
    expected_bytes = _positive_int(expected_blob_bytes, "expected_blob_bytes")
    expected_symbol = _digest(expected_symbol_sha256, "expected_symbol_sha256")
    expected_boundary = _digest(
        expected_boundary_state_sha256, "expected_boundary_state_sha256"
    )
    session = _digest(measurement_session_sha256, "measurement_session_sha256")
    launch = launch.validated()
    if not isinstance(synthetic, bool):
        raise DecodeWorkerError("synthetic must be exactly boolean")
    observed_blob = _sha256_bytes(blob)
    if observed_blob != expected_blob or len(blob) != expected_bytes:
        raise DecodeWorkerError("complete blob identity mismatch before timing")
    threads = _thread_environment()
    if pinned_cpu is not None and (
        isinstance(pinned_cpu, bool) or not isinstance(pinned_cpu, int) or pinned_cpu < 0
    ):
        raise DecodeWorkerError("pinned_cpu must be a nonnegative integer or null")
    affinity = _observed_affinity()
    before_affinity = (
        list(affinity)
        if affinity_before_pin is None
        else [int(value) for value in affinity_before_pin]
    )
    if (
        not before_affinity
        or any(isinstance(value, bool) or value < 0 for value in before_affinity)
        or before_affinity != sorted(set(before_affinity))
    ):
        raise DecodeWorkerError("affinity_before_pin must be a sorted nonempty CPU set")
    if pinned_cpu is not None and affinity != [pinned_cpu]:
        raise DecodeWorkerError("pinned_cpu disagrees with process affinity")
    if not synthetic and (
        pinned_cpu is None
        or pinned_cpu != before_affinity[0]
        or affinity != [pinned_cpu]
        or artifact_relative_path is None
    ):
        raise DecodeWorkerError(
            "production resource samples require secure artifact bytes and lowest-CPU pinning"
        )
    if artifact_relative_path is not None:
        _relative_parts(artifact_relative_path, "artifact_relative_path")
    worker_started_wall_ns = time.time_ns()

    started_ns = time.perf_counter_ns()
    header, symbols, symbol_sha256 = _strict_decode(mode, blob, runtime)
    boundary = comp009_execute.boundary_arrays(header, symbols)
    elapsed_ns = time.perf_counter_ns() - started_ns
    if elapsed_ns <= 0:
        raise DecodeWorkerError("measured decode duration must be positive")
    if not synthetic and header.n != container.ROW_COUNT:
        raise DecodeWorkerError(
            f"production resource decode requires N={container.ROW_COUNT}, found {header.n}"
        )
    if not synthetic and tuple(header.bits) not in FROZEN_BIT_TUPLES:
        raise DecodeWorkerError("production resource decode has a non-frozen COMP-011 bit tuple")

    # Boundary hashing, component reporting, RSS, and JSON construction are outside timing.
    boundary_sha256 = comp009_execute._array_state_digest(boundary)  # noqa: SLF001
    components = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes": int(value.nbytes),
            "sha256": _sha256_bytes(value.tobytes(order="C")),
        }
        for name, value in sorted(boundary.items())
    }
    symbol_exact = symbol_sha256 == expected_symbol
    boundary_exact = boundary_sha256 == expected_boundary
    arithmetic_used = mode in {"ssp2v", "ssp2e", "ssp2s", "ssp2l", "ssp2f"}
    return {
        "schema": WORKER_SCHEMA,
        "protocol": PROTOCOL,
        "task_sha256": TASK_SHA256,
        "operation": "resource-sample",
        "mode": mode,
        "magic": blob[:5].decode("ascii"),
        "format_version": version,
        "blob_bytes": len(blob),
        "blob_sha256": observed_blob,
        "blob_hash_verified_before_timing": True,
        "blob_acquisition": {
            "kind": (
                "artifact-root-relative-stable-open"
                if artifact_relative_path is not None
                else "in-memory-synthetic"
            ),
            "artifact_relative_path": artifact_relative_path,
            "absolute_and_parent_paths_rejected": artifact_relative_path is not None,
            "symlink_components_rejected_with_o_nofollow": artifact_relative_path is not None,
            "identity_verified_from_same_stable_opened_bytes": artifact_relative_path is not None,
        },
        "n": int(header.n),
        "bit_tuple": list(header.bits),
        "execution_mode": "synthetic-proof" if synthetic else "production-resource",
        "launch": launch.to_dict(),
        "freshness": {
            "fresh_comp011": True,
            "measurement_session_sha256": session,
            "worker_started_wall_ns": worker_started_wall_ns,
        },
        "elapsed_ns": elapsed_ns,
        "peak_rss_bytes": _peak_rss_bytes_linux(),
        "symbol_sha256": symbol_sha256,
        "boundary_state_sha256": boundary_sha256,
        "boundary_components": components,
        "exactness": {
            "expected_blob_bytes": expected_bytes,
            "expected_symbol_sha256": expected_symbol,
            "expected_boundary_state_sha256": expected_boundary,
            "blob_identity_exact": True,
            "symbol_hash_exact": symbol_exact,
            "boundary_hash_exact": boundary_exact,
            "all_exact": symbol_exact and boundary_exact,
        },
        "source_bindings": _source_bindings(runtime),
        "module": {
            "source_label": WORKER_MODULE_LABEL,
            "bytes": runtime.worker_module_bytes,
            "sha256": runtime.worker_module_sha256,
            "identity_verified_before_timing": True,
        },
        "native": {
            "artifact_relative_path": runtime.native_library_relative_path,
            "library_bytes": runtime.native_library_bytes,
            "library_sha256": runtime.native_library_sha256,
            "identity_verified_before_timing": True,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
            "loaded_before_timing": True,
            "arithmetic_core_used_in_timed_region": arithmetic_used,
            "canonical_validation": (
                "native_decode_internal_reencode_and_byte_compare"
                if arithmetic_used
                else "not_applicable"
            ),
        },
        "assets": {
            "cdf_bytes": runtime.cdf_asset_bytes,
            "cdf_sha256": runtime.cdf_asset_sha256,
            "source_label": f"benchmarks/assets/{assets.CDF_ASSET_NAME}",
            "identity_verified_and_loaded_before_timing": True,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
            "used_in_timed_region": mode in {"ssp2e", "ssp2s", "ssp2l"},
        },
        "timing": {
            "clock": "time.perf_counter_ns",
            "scope": list(TIMING_SCOPE),
            "imports_before_timing": True,
            "blob_read_and_hash_before_timing": True,
            "blob_identity_from_same_stable_opened_bytes_before_timing": (
                artifact_relative_path is not None
            ),
            "native_library_load_and_hash_before_timing": True,
            "static_tables_load_and_hash_before_timing": True,
            "boundary_hash_after_timing": True,
            "filesystem_read_inside_timing": False,
            "renderer_target_jit_inside_timing": False,
            "peak_rss_scope": "full_fresh_process_including_pre_and_post_timing",
        },
        "process": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "platform": sys.platform,
            "pinned_cpu": pinned_cpu,
            "cpu_affinity_before_pin": before_affinity,
            "cpu_affinity": affinity,
            "thread_environment": threads,
            "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss_linux_kib_times_1024",
        },
    }


def primary_schedule(image_index: int, tuple_index: int) -> tuple[LaunchSpec, ...]:
    """Return Q-then-candidate warmups and the frozen nine paired repetitions."""

    if image_index not in range(8) or tuple_index not in range(2):
        raise DecodeWorkerError("primary schedule indices are outside 8 images x 2 tuples")
    launches: list[LaunchSpec] = [
        LaunchSpec("primary", 0, "warmup", None, "q"),
        LaunchSpec("primary", 1, "warmup", None, "candidate"),
    ]
    for repetition in range(9):
        order = (
            ("candidate", "q")
            if (image_index + tuple_index + repetition) % 2 == 0
            else ("q", "candidate")
        )
        for arm in order:
            launches.append(
                LaunchSpec("primary", len(launches), "measured", repetition, arm)
            )
    return tuple(launch.validated() for launch in launches)


def secondary_schedule(image_index: int, tuple_index: int) -> tuple[LaunchSpec, ...]:
    """Return SSPL1-then-SSP2L warmups and the frozen secondary schedule."""

    if image_index not in range(8) or tuple_index not in range(2):
        raise DecodeWorkerError("secondary schedule indices are outside 8 images x 2 tuples")
    launches: list[LaunchSpec] = [
        LaunchSpec("secondary", 0, "warmup", None, "sspl1"),
        LaunchSpec("secondary", 1, "warmup", None, "ssp2l"),
    ]
    for repetition in range(9):
        order = (
            ("ssp2l", "sspl1")
            if (image_index + tuple_index + repetition) % 2 == 0
            else ("sspl1", "ssp2l")
        )
        for arm in order:
            launches.append(
                LaunchSpec("secondary", len(launches), "measured", repetition, arm)
            )
    return tuple(launch.validated() for launch in launches)


def validate_sample_for_launch(
    sample: Mapping[str, Any],
    launch: LaunchSpec,
    *,
    measurement_session_sha256: str,
    expected_blob_sha256: str | None = None,
    allow_synthetic: bool = False,
) -> None:
    """Reject old, cross-session, wrong-launch, inexact, or wrong-scope samples."""

    if not isinstance(sample, Mapping):
        raise DecodeWorkerError("decode sample must be a mapping")
    if frozenset(sample) != _RESOURCE_SAMPLE_KEYS:
        raise DecodeWorkerError("decode sample is old, inexact, cross-session, or wrong-scope")
    session = _digest(measurement_session_sha256, "measurement_session_sha256")
    if expected_blob_sha256 is not None:
        expected_blob_sha256 = _digest(expected_blob_sha256, "expected_blob_sha256")
    freshness = sample.get("freshness")
    timing = sample.get("timing")
    exactness = sample.get("exactness")
    native = sample.get("native")
    module = sample.get("module")
    assets_record = sample.get("assets")
    acquisition = sample.get("blob_acquisition")
    source_bindings = sample.get("source_bindings")
    process = sample.get("process")
    thread_environment = (
        process.get("thread_environment") if isinstance(process, Mapping) else None
    )
    _validate_source_bindings(source_bindings)
    if (
        sample.get("schema") != WORKER_SCHEMA
        or sample.get("protocol") != PROTOCOL
        or sample.get("task_sha256") != TASK_SHA256
        or sample.get("operation") != "resource-sample"
        or MAGIC_TO_MODE.get(
            str(sample.get("magic", "")).encode("ascii", errors="ignore")
        )
        != sample.get("mode")
        or sample.get("format_version") != 1
        or sample.get("n") != container.ROW_COUNT
        or not isinstance(sample.get("bit_tuple"), list)
        or tuple(sample["bit_tuple"]) not in FROZEN_BIT_TUPLES
        or sample.get("launch") != launch.validated().to_dict()
        or not isinstance(freshness, Mapping)
        or freshness.get("fresh_comp011") is not True
        or freshness.get("measurement_session_sha256") != session
        or not isinstance(timing, Mapping)
        or timing.get("scope") != list(TIMING_SCOPE)
        or timing.get("filesystem_read_inside_timing") is not False
        or timing.get("renderer_target_jit_inside_timing") is not False
        or not isinstance(exactness, Mapping)
        or exactness.get("all_exact") is not True
        or exactness.get("blob_identity_exact") is not True
        or not isinstance(acquisition, Mapping)
        or not isinstance(native, Mapping)
        or native.get("identity_verified_before_timing") is not True
        or native.get("identity_verified_from_same_opened_bytes_before_load") is not True
        or native.get("loaded_from_sealed_verified_bytes") is not True
        or not isinstance(module, Mapping)
        or module.get("identity_verified_before_timing") is not True
        or not isinstance(assets_record, Mapping)
        or assets_record.get("identity_verified_from_same_opened_bytes_before_load") is not True
        or assets_record.get("loaded_from_sealed_verified_bytes") is not True
        or sample.get("blob_hash_verified_before_timing") is not True
        or not isinstance(process, Mapping)
        or not isinstance(thread_environment, Mapping)
        or set(thread_environment) != set(THREAD_VARIABLES)
        or any(thread_environment[name] != "1" for name in THREAD_VARIABLES)
        or _positive_int(sample.get("elapsed_ns"), "elapsed_ns") <= 0
        or _positive_int(sample.get("peak_rss_bytes"), "peak_rss_bytes") <= 0
    ):
        raise DecodeWorkerError("decode sample is old, inexact, cross-session, or wrong-scope")
    expected_execution = "synthetic-proof" if allow_synthetic else "production-resource"
    if sample.get("execution_mode") != expected_execution:
        raise DecodeWorkerError("decode sample execution mode is not admissible")
    if (
        module.get("bytes") != source_bindings["worker_module"]["bytes"]
        or module.get("sha256") != source_bindings["worker_module"]["sha256"]
        or native.get("artifact_relative_path")
        != source_bindings["native_library"]["artifact_relative_path"]
        or native.get("library_bytes") != source_bindings["native_library"]["bytes"]
        or native.get("library_sha256") != source_bindings["native_library"]["sha256"]
        or assets_record.get("cdf_bytes") != source_bindings["cdf_asset"]["bytes"]
        or assets_record.get("cdf_sha256") != source_bindings["cdf_asset"]["sha256"]
    ):
        raise DecodeWorkerError("decode sample dependency identities disagree internally")
    before = process.get("cpu_affinity_before_pin")
    after = process.get("cpu_affinity")
    if (
        not isinstance(before, list)
        or not before
        or before != sorted(set(before))
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in before)
        or not isinstance(after, list)
        or not after
        or after != sorted(set(after))
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in after)
    ):
        raise DecodeWorkerError("decode sample CPU-affinity environment is invalid")
    expected_acquisition = (
        "in-memory-synthetic" if allow_synthetic else "artifact-root-relative-stable-open"
    )
    if acquisition.get("kind") != expected_acquisition:
        raise DecodeWorkerError("decode sample blob acquisition mode is not admissible")
    if not allow_synthetic:
        pinned_cpu = process.get("pinned_cpu")
        if (
            pinned_cpu is None
            or not isinstance(before, list)
            or not before
            or before != sorted(set(before))
            or pinned_cpu != before[0]
            or process.get("cpu_affinity") != [pinned_cpu]
            or acquisition.get("absolute_and_parent_paths_rejected") is not True
            or acquisition.get("symlink_components_rejected_with_o_nofollow") is not True
            or acquisition.get("identity_verified_from_same_stable_opened_bytes") is not True
            or timing.get("blob_identity_from_same_stable_opened_bytes_before_timing") is not True
        ):
            raise DecodeWorkerError("production decode sample was not exactly CPU-pinned")
    if expected_blob_sha256 is not None and sample.get("blob_sha256") != expected_blob_sha256:
        raise DecodeWorkerError("decode sample belongs to a different exact blob")


def summarize_schedule(
    samples: Sequence[Mapping[str, Any]],
    schedule: Sequence[LaunchSpec],
    *,
    image_index: int,
    tuple_index: int,
    measurement_session_sha256: str,
    expected_blob_sha256_by_arm: Mapping[str, str] | None = None,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Validate one complete fresh schedule and select exact median/RSS samples."""

    expected = tuple(schedule)
    if len(expected) != 20 or len(samples) != 20:
        raise DecodeWorkerError("a paired schedule requires exactly twenty fresh samples")
    if tuple(launch.launch_index for launch in expected) != tuple(range(20)):
        raise DecodeWorkerError("paired schedule launch indices are not canonical")
    kinds = {launch.schedule_kind for launch in expected}
    if len(kinds) != 1:
        raise DecodeWorkerError("paired schedule mixes primary and secondary launches")
    kind = next(iter(kinds))
    canonical_schedule = (
        primary_schedule(image_index, tuple_index)
        if kind == "primary"
        else secondary_schedule(image_index, tuple_index)
    )
    if expected != canonical_schedule:
        raise DecodeWorkerError("paired schedule differs from the frozen parity order")
    expected_arms = {launch.arm for launch in expected}
    if expected_blob_sha256_by_arm is not None:
        if set(expected_blob_sha256_by_arm) != expected_arms:
            raise DecodeWorkerError("expected blob mapping must bind both exact schedule arms")
        for arm, digest in expected_blob_sha256_by_arm.items():
            _digest(digest, f"expected_blob_sha256_by_arm[{arm}]")
    observed_pids: list[int] = []
    observed_source_bindings: set[bytes] = set()
    observed_thread_environments: set[bytes] = set()
    observed_inherited_affinities: set[tuple[int, ...]] = set()
    measured: dict[str, list[tuple[LaunchSpec, Mapping[str, Any]]]] = {}
    for sample, launch in zip(samples, expected, strict=True):
        expected_blob = (
            None
            if expected_blob_sha256_by_arm is None
            else expected_blob_sha256_by_arm.get(launch.arm)
        )
        validate_sample_for_launch(
            sample,
            launch,
            measurement_session_sha256=measurement_session_sha256,
            expected_blob_sha256=expected_blob,
            allow_synthetic=allow_synthetic,
        )
        pid = sample["process"].get("pid")
        observed_pids.append(_positive_int(pid, "process.pid"))
        observed_source_bindings.add(canonical_json(sample["source_bindings"]))
        observed_thread_environments.add(
            canonical_json(sample["process"]["thread_environment"])
        )
        observed_inherited_affinities.add(
            tuple(sample["process"]["cpu_affinity_before_pin"])
        )
        if launch.phase == "measured":
            measured.setdefault(launch.arm, []).append((launch, sample))
    if len(set(observed_pids)) != 20:
        raise DecodeWorkerError("paired samples were not produced by twenty fresh processes")
    if (
        len(observed_source_bindings) != 1
        or len(observed_thread_environments) != 1
        or len(observed_inherited_affinities) != 1
    ):
        raise DecodeWorkerError("paired samples do not share one exact source/environment binding")

    summaries: dict[str, Any] = {}
    for arm, rows in sorted(measured.items()):
        if len(rows) != 9 or {launch.repetition for launch, _sample in rows} != set(range(9)):
            raise DecodeWorkerError("each paired arm requires repetitions zero through eight")
        elapsed_order = sorted(
            rows, key=lambda item: (int(item[1]["elapsed_ns"]), int(item[0].repetition))
        )
        median_launch, median_sample = elapsed_order[4]
        rss_launch, rss_sample = min(
            rows,
            key=lambda item: (-int(item[1]["peak_rss_bytes"]), int(item[0].repetition)),
        )
        summaries[arm] = {
            "measured_count": 9,
            "median_ns": int(median_sample["elapsed_ns"]),
            "median_repetition": int(median_launch.repetition),
            "maximum_peak_rss_bytes": int(rss_sample["peak_rss_bytes"]),
            "maximum_peak_rss_repetition": int(rss_launch.repetition),
            "elapsed_order": [
                {
                    "elapsed_ns": int(sample["elapsed_ns"]),
                    "repetition": int(launch.repetition),
                }
                for launch, sample in elapsed_order
            ],
        }
    if len(summaries) != 2:
        raise DecodeWorkerError("paired schedule did not produce exactly two measured arms")
    source_binding_bytes = next(iter(observed_source_bindings))
    thread_environment_bytes = next(iter(observed_thread_environments))
    inherited_affinity = next(iter(observed_inherited_affinities))
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol": PROTOCOL,
        "task_sha256": TASK_SHA256,
        "schedule_kind": kind,
        "image_index": image_index,
        "tuple_index": tuple_index,
        "measurement_session_sha256": _digest(
            measurement_session_sha256, "measurement_session_sha256"
        ),
        "fresh_process_count": 20,
        "schedule_exact": True,
        "source_bindings_sha256": _sha256_bytes(source_binding_bytes),
        "thread_environment": json.loads(thread_environment_bytes),
        "cpu_affinity_before_pin": list(inherited_affinity),
        "environment_exact_across_schedule": True,
        "summaries": summaries,
    }


def run_file(
    artifact_root: Path,
    blob_path: str | os.PathLike[str],
    native_library: str | os.PathLike[str],
    *,
    expected_blob_sha256: str,
    expected_blob_bytes: int,
    expected_native_library_sha256: str,
    expected_native_library_bytes: int,
    expected_worker_module_sha256: str,
    expected_worker_module_bytes: int,
    expected_symbol_sha256: str,
    expected_boundary_state_sha256: str,
    measurement_session_sha256: str,
    launch: LaunchSpec,
    pin_cpu: bool = True,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Perform every file/native/affinity operation before invoking timed decode."""

    if not isinstance(pin_cpu, bool):
        raise DecodeWorkerError("pin_cpu must be exactly boolean")
    if not pin_cpu and not synthetic:
        raise DecodeWorkerError("production resource execution cannot disable CPU pinning")
    root_descriptor = _open_artifact_root(Path(artifact_root))
    try:
        opened_blob = _read_artifact_file(root_descriptor, blob_path, "blob")
        opened_native = _read_artifact_file(
            root_descriptor, native_library, "native_library"
        )
    finally:
        os.close(root_descriptor)
    runtime = _runtime_from_opened_native(
        opened_native,
        expected_native_library_sha256=expected_native_library_sha256,
        expected_native_library_bytes=expected_native_library_bytes,
        expected_worker_module_sha256=expected_worker_module_sha256,
        expected_worker_module_bytes=expected_worker_module_bytes,
    )
    before_affinity = _observed_affinity()
    selected_cpu = pin_lowest_available_cpu() if pin_cpu else None
    return decode_blob(
        opened_blob.payload,
        runtime,
        expected_blob_sha256=expected_blob_sha256,
        expected_blob_bytes=expected_blob_bytes,
        expected_symbol_sha256=expected_symbol_sha256,
        expected_boundary_state_sha256=expected_boundary_state_sha256,
        measurement_session_sha256=measurement_session_sha256,
        launch=launch,
        pinned_cpu=selected_cpu,
        artifact_relative_path=opened_blob.relative_path,
        affinity_before_pin=before_affinity,
        synthetic=synthetic,
    )


def candidate_cold_seal_file(
    artifact_root: Path,
    blob_path: str | os.PathLike[str],
    native_library: str | os.PathLike[str],
    *,
    expected_blob_sha256: str,
    expected_blob_bytes: int,
    expected_native_library_sha256: str,
    expected_native_library_bytes: int,
    expected_worker_module_sha256: str,
    expected_worker_module_bytes: int,
    expected_symbol_sha256: str,
    expected_boundary_state_sha256: str,
    expected_stream_components_sha256: str,
) -> dict[str, Any]:
    """Securely acquire and cold-seal one candidate with no timing semantics."""

    root_descriptor = _open_artifact_root(Path(artifact_root))
    try:
        opened_blob = _read_artifact_file(root_descriptor, blob_path, "blob")
        opened_native = _read_artifact_file(
            root_descriptor, native_library, "native_library"
        )
    finally:
        os.close(root_descriptor)
    runtime = _runtime_from_opened_native(
        opened_native,
        expected_native_library_sha256=expected_native_library_sha256,
        expected_native_library_bytes=expected_native_library_bytes,
        expected_worker_module_sha256=expected_worker_module_sha256,
        expected_worker_module_bytes=expected_worker_module_bytes,
    )
    return candidate_cold_seal(
        opened_blob.payload,
        runtime,
        artifact_relative_path=opened_blob.relative_path,
        expected_blob_sha256=expected_blob_sha256,
        expected_blob_bytes=expected_blob_bytes,
        expected_symbol_sha256=expected_symbol_sha256,
        expected_boundary_state_sha256=expected_boundary_state_sha256,
        expected_stream_components_sha256=expected_stream_components_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)

    def common(operation: argparse.ArgumentParser) -> None:
        operation.add_argument("--artifact-root", type=Path, required=True)
        operation.add_argument("--blob", required=True)
        operation.add_argument("--native-library", required=True)
        operation.add_argument("--expected-blob-sha256", required=True)
        operation.add_argument("--expected-blob-bytes", type=int, required=True)
        operation.add_argument("--expected-native-library-sha256", required=True)
        operation.add_argument("--expected-native-library-bytes", type=int, required=True)
        operation.add_argument("--expected-worker-module-sha256", required=True)
        operation.add_argument("--expected-worker-module-bytes", type=int, required=True)
        operation.add_argument("--expected-symbol-sha256", required=True)
        operation.add_argument("--expected-boundary-state-sha256", required=True)

    resource_parser = operations.add_parser(
        "resource", help="emit one production primary/secondary resource sample"
    )
    common(resource_parser)
    resource_parser.add_argument("--measurement-session-sha256", required=True)
    resource_parser.add_argument(
        "--schedule-kind", choices=("primary", "secondary"), required=True
    )
    resource_parser.add_argument("--launch-index", type=int, required=True)
    resource_parser.add_argument("--phase", choices=("warmup", "measured"), required=True)
    resource_parser.add_argument("--repetition", type=int)
    resource_parser.add_argument("--arm", required=True)

    cold_parser = operations.add_parser(
        "candidate-cold-seal", help="emit one strict untimed candidate cold seal"
    )
    common(cold_parser)
    cold_parser.add_argument("--expected-stream-components-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "expected_blob_sha256": args.expected_blob_sha256,
        "expected_blob_bytes": args.expected_blob_bytes,
        "expected_native_library_sha256": args.expected_native_library_sha256,
        "expected_native_library_bytes": args.expected_native_library_bytes,
        "expected_worker_module_sha256": args.expected_worker_module_sha256,
        "expected_worker_module_bytes": args.expected_worker_module_bytes,
        "expected_symbol_sha256": args.expected_symbol_sha256,
        "expected_boundary_state_sha256": args.expected_boundary_state_sha256,
    }
    if args.operation == "resource":
        launch = LaunchSpec(
            args.schedule_kind,
            args.launch_index,
            args.phase,
            args.repetition,
            args.arm,
        ).validated()
        record = run_file(
            args.artifact_root,
            args.blob,
            args.native_library,
            **common,
            measurement_session_sha256=args.measurement_session_sha256,
            launch=launch,
        )
        exit_code = 0 if record["exactness"]["all_exact"] is True else 2
    else:
        record = candidate_cold_seal_file(
            args.artifact_root,
            args.blob,
            args.native_library,
            **common,
            expected_stream_components_sha256=args.expected_stream_components_sha256,
        )
        exit_code = 0
    sys.stdout.buffer.write(canonical_json(record))
    sys.stdout.buffer.flush()
    return exit_code


if __name__ == "__main__":  # pragma: no cover - tested through a fresh subprocess
    raise SystemExit(main())


__all__ = [
    "COLD_SEAL_SCHEMA",
    "DecodeRuntime",
    "DecodeWorkerError",
    "LaunchSpec",
    "MAGIC_TO_MODE",
    "PROTOCOL",
    "SUMMARY_SCHEMA",
    "TASK_SHA256",
    "TIMING_SCOPE",
    "WORKER_SCHEMA",
    "build_parser",
    "candidate_cold_seal",
    "candidate_cold_seal_file",
    "canonical_json",
    "component_records_sha256",
    "decode_blob",
    "load_runtime",
    "main",
    "pin_lowest_available_cpu",
    "primary_schedule",
    "run_file",
    "secondary_schedule",
    "summarize_schedule",
    "stream_component_records",
    "validate_sample_for_launch",
    "verify_candidate_cold_seal",
]
