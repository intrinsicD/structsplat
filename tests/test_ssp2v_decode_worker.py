from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as comp009_container
from benchmarks import ssp2e_execute as comp009_execute
from benchmarks import ssp2v_container as container
from benchmarks import ssp2v_decode_worker as worker


ROOT = Path(__file__).resolve().parents[1]
SESSION = "a" * 64


def _pack(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    return np.asarray(values, dtype=dtype).tobytes(order="C")


def _planar(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    raw = np.asarray(values, dtype=dtype).tobytes(order="C")
    if dtype.itemsize == 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, dtype.itemsize).T.ravel().tobytes()


def _deltas(values: np.ndarray, bits: int) -> np.ndarray:
    preceding = np.vstack((np.zeros((1, values.shape[1]), dtype=np.uint32), values[:-1]))
    return (values - preceding) & ((1 << bits) - 1)


def _sspl1() -> bytes:
    n = container.ROW_COUNT
    bits = (12, 6, 6, 8)
    bm, bs, br, bc = bits
    row = np.arange(n, dtype=np.uint32)
    means = np.column_stack(((29 * row) % (1 << bm), (47 * row[::-1]) % (1 << bm)))
    scales = np.column_stack((row % (1 << bs), (3 * row + 2) % (1 << bs)))
    rotation = (5 * row + 1) % (1 << br)
    colors = np.column_stack(
        ((7 * row) % (1 << bc), (11 * row + 1) % (1 << bc), (13 * row + 2) % (1 << bc))
    )
    raw_streams = (
        _planar(_deltas(means, bm).T.ravel(), bm),
        _pack(scales.T.ravel(), bs),
        _pack(rotation, br),
        _pack(_deltas(colors, bc).T.ravel(), bc),
    )
    header = {
        "n": n,
        "H": 64,
        "W": 80,
        "bits": list(bits),
        "morton": True,
        "color_delta": True,
        "means_byte_planar": True,
        "rot_circular": True,
        "stream_codec": "zlib",
        "stream_raw_lengths": [len(raw) for raw in raw_streams],
        "color_lo": [-1.0, -0.5, 0.0],
        "color_hi": [1.0, 1.5, 2.0],
        "scale_lo": [-3.0, -2.0],
        "scale_hi": [2.0, 3.0],
        "means_lo": [0.0, 0.0],
        "means_hi": [79.0, 63.0],
        "has_opacity": False,
        "bits_opacity": 8,
        "renderer": "cuda",
        "aa_dilation": 0.0,
        "sigma_cutoff": 3.0,
        "support_fade": False,
        "render_chunk": 512,
    }
    header_raw = json.dumps(header).encode("utf-8")
    blob = b"SSPL1" + struct.pack("<I", len(header_raw)) + header_raw
    for raw in raw_streams:
        payload = zlib.compress(raw, level=9)
        blob += struct.pack("<I", len(payload)) + payload
    return blob


@pytest.fixture(autouse=True)
def one_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in worker.THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")


@pytest.fixture(scope="module")
def seven_streams(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("ssp2v-worker")
    native_path = arithmetic.build_native(directory / "libssp2e_range_v1.so")
    native_bytes = native_path.stat().st_size
    native_sha256 = hashlib.sha256(native_path.read_bytes()).hexdigest()
    worker_module_payload = Path(worker.__file__).read_bytes()
    worker_module_bytes = len(worker_module_payload)
    worker_module_sha256 = hashlib.sha256(worker_module_payload).hexdigest()
    source = _sspl1()
    tables = assets.load_tables()

    def frequencies(alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        return tables[alphabet].frequency_row(location, scale)

    head_channel = comp009_container.HeadChannel((0,) * 8, 32, (0,) * 8, 8)
    head = comp009_container.serialize_head((head_channel,) * 5)
    grid = bytes(range(256))
    global_models = {name: (32, 8) for name in comp009_container.MODELED_CHANNELS}
    descriptor = container.SSP2VDescriptor(0, 1, 2, 64, 192)
    row = np.arange(descriptor.codebook_entries, dtype=np.uint32)
    codebook = np.column_stack((row % 256, (3 * row + 7) % 256, (5 * row + 11) % 256))
    indices = ((37 * np.arange(container.ROW_COUNT, dtype=np.uint32) + 13) % len(row)).astype(
        np.uint32
    )
    blobs: dict[str, bytes] = {
        "sspl1": source,
        "ssp2z": container.encode_ssp2z(source),
        "ssp2v": container.encode_ssp2v(
            source,
            descriptor,
            (codebook.astype(np.uint8),),
            (indices,),
        ),
        "ssp2e": comp009_container.encode_contextual(
            source,
            grid,
            head,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=frequencies,
            magic=b"SSP2E",
        ),
        "ssp2s": comp009_container.encode_contextual(
            source,
            grid,
            head,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=frequencies,
            magic=b"SSP2S",
        ),
        "ssp2l": comp009_container.encode_ssp2l(
            source,
            global_models,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=frequencies,
        ),
        "ssp2f": comp009_container.encode_ssp2f(source, arithmetic_encode=arithmetic.encode),
    }
    paths: dict[str, Path] = {}
    expected: dict[str, dict[str, str]] = {}
    source_parts = comp009_container.extract_sspl1_parts(source)
    for mode, blob in blobs.items():
        path = directory / f"{mode}.bin"
        path.write_bytes(blob)
        paths[mode] = path
        if mode == "ssp2v":
            decoded = container.decode_ssp2v(blob)
            header, symbols, symbol_sha256 = (
                decoded.parsed.header,
                decoded.symbols,
                decoded.symbol_sha256,
            )
        else:
            header, symbols, symbol_sha256 = (
                source_parts.header,
                source_parts.symbols,
                source_parts.symbol_sha256,
            )
        boundary = comp009_execute.boundary_arrays(header, symbols)
        expected[mode] = {
            "blob": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "symbol": symbol_sha256,
            "boundary": comp009_execute._array_state_digest(boundary),  # noqa: SLF001
            "components": worker.component_records_sha256(
                worker.stream_component_records(blob)
            ),
        }
    runtime = worker.load_runtime(
        directory,
        native_path.name,
        expected_native_library_sha256=native_sha256,
        expected_native_library_bytes=native_bytes,
        expected_worker_module_sha256=worker_module_sha256,
        expected_worker_module_bytes=worker_module_bytes,
    )
    return {
        "directory": directory,
        "native_path": native_path,
        "native_bytes": native_bytes,
        "native_sha256": native_sha256,
        "worker_module_bytes": worker_module_bytes,
        "worker_module_sha256": worker_module_sha256,
        "runtime": runtime,
        "blobs": blobs,
        "paths": paths,
        "expected": expected,
    }


def _decode(
    frozen: dict[str, object], mode: str, launch: worker.LaunchSpec | None = None
) -> dict[str, object]:
    expected = frozen["expected"][mode]
    return worker.decode_blob(
        frozen["blobs"][mode],
        frozen["runtime"],
        expected_blob_sha256=expected["blob"],
        expected_blob_bytes=expected["bytes"],
        expected_symbol_sha256=expected["symbol"],
        expected_boundary_state_sha256=expected["boundary"],
        measurement_session_sha256=SESSION,
        launch=launch or worker.primary_schedule(0, 0)[0],
        pinned_cpu=None,
        synthetic=True,
    )


def _runtime_kwargs(frozen: dict[str, object]) -> dict[str, object]:
    return {
        "expected_native_library_sha256": frozen["native_sha256"],
        "expected_native_library_bytes": frozen["native_bytes"],
        "expected_worker_module_sha256": frozen["worker_module_sha256"],
        "expected_worker_module_bytes": frozen["worker_module_bytes"],
    }


def _cold(frozen: dict[str, object], mode: str) -> dict[str, object]:
    expected = frozen["expected"][mode]
    return worker.candidate_cold_seal_file(
        frozen["directory"],
        frozen["paths"][mode].name,
        frozen["native_path"].name,
        expected_blob_sha256=expected["blob"],
        expected_blob_bytes=expected["bytes"],
        expected_symbol_sha256=expected["symbol"],
        expected_boundary_state_sha256=expected["boundary"],
        expected_stream_components_sha256=expected["components"],
        **_runtime_kwargs(frozen),
    )


@pytest.mark.parametrize("mode", tuple(worker.MAGIC_TO_MODE.values()))
def test_unified_worker_strictly_dispatches_all_seven_formats(
    seven_streams: dict[str, object], mode: str
) -> None:
    record = _decode(seven_streams, mode)
    assert record["schema"] == worker.WORKER_SCHEMA
    assert record["protocol"] == worker.PROTOCOL
    assert record["task_sha256"] == worker.TASK_SHA256
    assert record["operation"] == "resource-sample"
    assert record["mode"] == mode
    assert record["magic"] == mode.upper().encode().replace(b"SSPL1", b"SSPL1").decode()[:5]
    assert record["format_version"] == 1
    assert record["n"] == container.ROW_COUNT
    assert record["bit_tuple"] == [12, 6, 6, 8]
    assert record["blob_hash_verified_before_timing"] is True
    assert record["exactness"]["all_exact"] is True
    assert record["elapsed_ns"] > 0
    assert record["peak_rss_bytes"] > 0
    assert record["timing"]["scope"] == list(worker.TIMING_SCOPE)
    assert record["timing"]["filesystem_read_inside_timing"] is False
    assert record["timing"]["renderer_target_jit_inside_timing"] is False
    assert record["native"]["arithmetic_core_used_in_timed_region"] is (
        mode not in {"sspl1", "ssp2z"}
    )
    assert record["assets"]["used_in_timed_region"] is (
        mode in {"ssp2e", "ssp2s", "ssp2l"}
    )
    assert json.loads(worker.canonical_json(record)) == record


@pytest.mark.parametrize("mode", tuple(worker.MAGIC_TO_MODE.values()))
def test_candidate_cold_seal_is_strict_distinct_and_self_verifying(
    seven_streams: dict[str, object], mode: str
) -> None:
    record = _cold(seven_streams, mode)
    expected = seven_streams["expected"][mode]
    assert record["schema"] == worker.COLD_SEAL_SCHEMA
    assert record["operation"] == "candidate-cold-seal"
    assert record["mode"] == mode
    assert record["blob"]["bytes"] == expected["bytes"]
    assert record["blob"]["sha256"] == expected["blob"]
    assert record["symbols"]["state_sha256"] == expected["symbol"]
    assert record["boundary"]["state_sha256"] == expected["boundary"]
    assert record["stream_components"]["state_sha256"] == expected["components"]
    assert record["canonical_decode_reencode_verified"] is True
    assert "elapsed_ns" not in record
    assert "launch" not in record
    assert "exactness" not in record
    worker.verify_candidate_cold_seal(
        record,
        expected_blob_sha256=expected["blob"],
        expected_blob_bytes=expected["bytes"],
        expected_symbol_sha256=expected["symbol"],
        expected_boundary_state_sha256=expected["boundary"],
        expected_stream_components_sha256=expected["components"],
    )
    assert json.loads(worker.canonical_json(record)) == record


def test_candidate_cold_seal_fails_closed_on_component_authority_and_old_schema(
    seven_streams: dict[str, object],
) -> None:
    expected = seven_streams["expected"]["ssp2v"]
    with pytest.raises(worker.DecodeWorkerError, match="decoded evidence mismatch"):
        worker.candidate_cold_seal_file(
            seven_streams["directory"],
            seven_streams["paths"]["ssp2v"].name,
            seven_streams["native_path"].name,
            expected_blob_sha256=expected["blob"],
            expected_blob_bytes=expected["bytes"],
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            expected_stream_components_sha256="0" * 64,
            **_runtime_kwargs(seven_streams),
        )

    record = _cold(seven_streams, "ssp2v")
    old = copy.deepcopy(record)
    old["schema"] = "structsplat.comp011.ssp2v-candidate-cold-seal.v0"
    with pytest.raises(worker.DecodeWorkerError, match="old or malformed"):
        worker.verify_candidate_cold_seal(old)
    altered = copy.deepcopy(record)
    altered["symbols"]["components"]["red"]["sha256"] = "0" * 64
    with pytest.raises(worker.DecodeWorkerError, match="symbol components"):
        worker.verify_candidate_cold_seal(altered)
    with pytest.raises(worker.DecodeWorkerError, match="old, inexact"):
        worker.validate_sample_for_launch(
            record,
            worker.primary_schedule(0, 0)[0],
            measurement_session_sha256=SESSION,
        )


def test_ssp2v_native_path_regenerates_model_and_matches_generic_decode(
    seven_streams: dict[str, object],
) -> None:
    record = _decode(seven_streams, "ssp2v")
    generic = container.decode_ssp2v(seven_streams["blobs"]["ssp2v"])
    assert record["symbol_sha256"] == generic.symbol_sha256
    assert record["native"]["canonical_validation"] == (
        "native_decode_internal_reencode_and_byte_compare"
    )
    assert record["boundary_state_sha256"] == seven_streams["expected"]["ssp2v"]["boundary"]


@pytest.mark.parametrize("mode", tuple(worker.MAGIC_TO_MODE.values()))
def test_every_format_rejects_complete_stream_corruption(
    seven_streams: dict[str, object], mode: str
) -> None:
    changed = bytearray(seven_streams["blobs"][mode])
    changed[-1] ^= 1
    corrupted = bytes(changed)
    expected = seven_streams["expected"][mode]
    with pytest.raises(worker.DecodeWorkerError, match="strict decode|canonical|rejected|invalid"):
        worker.decode_blob(
            corrupted,
            seven_streams["runtime"],
            expected_blob_sha256=hashlib.sha256(corrupted).hexdigest(),
            expected_blob_bytes=len(corrupted),
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            measurement_session_sha256=SESSION,
            launch=worker.primary_schedule(0, 0)[0],
            pinned_cpu=None,
            synthetic=True,
        )


def test_blob_hash_thread_and_version_fail_before_timer(
    seven_streams: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = seven_streams["blobs"]["ssp2v"]
    expected = seven_streams["expected"]["ssp2v"]

    def forbidden_clock() -> int:
        raise AssertionError("timer must not start")

    monkeypatch.setattr(worker.time, "perf_counter_ns", forbidden_clock)
    with pytest.raises(worker.DecodeWorkerError, match="blob identity"):
        worker.decode_blob(
            blob,
            seven_streams["runtime"],
            expected_blob_sha256="0" * 64,
            expected_blob_bytes=len(blob),
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            measurement_session_sha256=SESSION,
            launch=worker.primary_schedule(0, 0)[0],
            pinned_cpu=None,
            synthetic=True,
        )
    with pytest.raises(worker.DecodeWorkerError, match="blob identity"):
        worker.decode_blob(
            blob,
            seven_streams["runtime"],
            expected_blob_sha256=expected["blob"],
            expected_blob_bytes=expected["bytes"] + 1,
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            measurement_session_sha256=SESSION,
            launch=worker.primary_schedule(0, 0)[0],
            pinned_cpu=None,
            synthetic=True,
        )
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(worker.DecodeWorkerError, match="thread environment"):
        worker.decode_blob(
            blob,
            seven_streams["runtime"],
            expected_blob_sha256=expected["blob"],
            expected_blob_bytes=expected["bytes"],
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            measurement_session_sha256=SESSION,
            launch=worker.primary_schedule(0, 0)[0],
            pinned_cpu=None,
            synthetic=True,
        )

    version_changed = bytearray(blob)
    version_changed[5] = 2
    with pytest.raises(worker.DecodeWorkerError, match="version"):
        worker.decode_blob(
            bytes(version_changed),
            seven_streams["runtime"],
            expected_blob_sha256=hashlib.sha256(version_changed).hexdigest(),
            expected_blob_bytes=len(version_changed),
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            measurement_session_sha256=SESSION,
            launch=worker.primary_schedule(0, 0)[0],
            pinned_cpu=None,
            synthetic=True,
        )
    with pytest.raises(worker.DecodeWorkerError, match="unsupported"):
        worker.decode_blob(
            b"BAD!!payload",
            seven_streams["runtime"],
            expected_blob_sha256=hashlib.sha256(b"BAD!!payload").hexdigest(),
            expected_blob_bytes=len(b"BAD!!payload"),
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            measurement_session_sha256=SESSION,
            launch=worker.primary_schedule(0, 0)[0],
            pinned_cpu=None,
            synthetic=True,
        )


def test_runtime_rejects_wrong_persisted_native_identity(
    seven_streams: dict[str, object],
) -> None:
    with pytest.raises(worker.DecodeWorkerError, match="identity mismatch"):
        worker.load_runtime(
            seven_streams["directory"],
            seven_streams["native_path"].name,
            expected_native_library_sha256="0" * 64,
            expected_native_library_bytes=seven_streams["native_bytes"],
            expected_worker_module_sha256=seven_streams["worker_module_sha256"],
            expected_worker_module_bytes=seven_streams["worker_module_bytes"],
        )
    with pytest.raises(worker.DecodeWorkerError, match="identity mismatch"):
        worker.load_runtime(
            seven_streams["directory"],
            seven_streams["native_path"].name,
            expected_native_library_sha256=seven_streams["native_sha256"],
            expected_native_library_bytes=seven_streams["native_bytes"] + 1,
            expected_worker_module_sha256=seven_streams["worker_module_sha256"],
            expected_worker_module_bytes=seven_streams["worker_module_bytes"],
        )


def test_decode_rejects_fabricated_runtime_attestation(
    seven_streams: dict[str, object],
) -> None:
    expected = seven_streams["expected"]["ssp2v"]
    forged = replace(seven_streams["runtime"], verification_token=object())
    with pytest.raises(worker.DecodeWorkerError, match="prepared by load_runtime"):
        worker.decode_blob(
            seven_streams["blobs"]["ssp2v"],
            forged,
            expected_blob_sha256=expected["blob"],
            expected_blob_bytes=expected["bytes"],
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            measurement_session_sha256=SESSION,
            launch=worker.primary_schedule(0, 0)[0],
            pinned_cpu=None,
            synthetic=True,
        )


def test_artifact_relative_paths_reject_absolute_parent_and_symlink_components(
    seven_streams: dict[str, object], tmp_path: Path
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    blob_name = "candidate.bin"
    native_name = "runtime/libssp2e.so"
    (root / "runtime").mkdir()
    (root / blob_name).write_bytes(seven_streams["blobs"]["ssp2v"])
    shutil.copyfile(seven_streams["native_path"], root / native_name)
    expected = seven_streams["expected"]["ssp2v"]

    def seal(blob_path: str, native_path: str, artifact_root: Path = root) -> None:
        worker.candidate_cold_seal_file(
            artifact_root,
            blob_path,
            native_path,
            expected_blob_sha256=expected["blob"],
            expected_blob_bytes=expected["bytes"],
            expected_symbol_sha256=expected["symbol"],
            expected_boundary_state_sha256=expected["boundary"],
            expected_stream_components_sha256=expected["components"],
            **_runtime_kwargs(seven_streams),
        )

    with pytest.raises(worker.DecodeWorkerError, match="artifact-root-relative"):
        seal(str(root / blob_name), native_name)
    with pytest.raises(worker.DecodeWorkerError, match="artifact-root-relative"):
        seal("../candidate.bin", native_name)
    with pytest.raises(worker.DecodeWorkerError, match="artifact-root-relative"):
        seal(blob_name, str(root / native_name))
    with pytest.raises(worker.DecodeWorkerError, match="artifact-root-relative"):
        seal("./candidate.bin", native_name)

    os.symlink(blob_name, root / "candidate-link.bin")
    with pytest.raises(worker.DecodeWorkerError, match="unsafe or unavailable blob"):
        seal("candidate-link.bin", native_name)
    os.symlink("runtime", root / "runtime-link")
    with pytest.raises(worker.DecodeWorkerError, match="unsafe or unavailable native_library"):
        seal(blob_name, "runtime-link/libssp2e.so")
    root_link = tmp_path / "artifact-link"
    os.symlink(root, root_link)
    with pytest.raises(worker.DecodeWorkerError, match="no-symlink directory"):
        seal(blob_name, native_name, root_link)


def test_directory_walk_uses_path_only_anchors() -> None:
    flags = worker._directory_flags()  # noqa: SLF001 - security invariant under test.
    assert flags & os.O_PATH
    assert flags & os.O_DIRECTORY
    assert flags & os.O_NOFOLLOW
    assert not flags & os.O_RDWR


def test_opened_descriptor_rejects_path_swap_and_in_place_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_descriptor = worker._open_artifact_root(tmp_path)
    original = b"original-opened-bytes"
    path = tmp_path / "blob.bin"
    path.write_bytes(original)
    real_read = worker.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, count)
        if chunk and not swapped:
            swapped = True
            path.rename(tmp_path / "opened-original.bin")
            path.write_bytes(b"replacement-path-bytes")
        return chunk

    monkeypatch.setattr(worker.os, "read", swapping_read)
    try:
        with pytest.raises(worker.DecodeWorkerError, match="changed while"):
            worker._read_artifact_file(root_descriptor, "blob.bin", "blob")
    finally:
        os.close(root_descriptor)
    assert (tmp_path / "opened-original.bin").read_bytes() == original
    assert path.read_bytes() == b"replacement-path-bytes"

    monkeypatch.setattr(worker.os, "read", real_read)
    mutating_path = tmp_path / "mutating.bin"
    mutating_path.write_bytes(b"a" * ((1 << 20) + 16))
    root_descriptor = worker._open_artifact_root(tmp_path)
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            with mutating_path.open("r+b") as handle:
                handle.seek(1 << 20)
                handle.write(b"b" * 16)
                handle.flush()
                os.fsync(handle.fileno())
        return chunk

    monkeypatch.setattr(worker.os, "read", mutating_read)
    try:
        with pytest.raises(worker.DecodeWorkerError, match="changed while"):
            worker._read_artifact_file(root_descriptor, "mutating.bin", "blob")
    finally:
        os.close(root_descriptor)


def test_native_load_uses_verified_opened_bytes_and_module_binds_before_load(
    seven_streams: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = tmp_path / "native.so"
    shutil.copyfile(seven_streams["native_path"], native)
    real_loader = worker._native_from_verified_bytes
    swapped = False

    def swapping_loader(payload: bytes):
        nonlocal swapped
        assert hashlib.sha256(payload).hexdigest() == seven_streams["native_sha256"]
        native.rename(tmp_path / "native-opened.so")
        native.write_bytes(b"not an ELF library")
        swapped = True
        return real_loader(payload)

    monkeypatch.setattr(worker, "_native_from_verified_bytes", swapping_loader)
    runtime = worker.load_runtime(
        tmp_path,
        native.name,
        **_runtime_kwargs(seven_streams),
    )
    assert swapped is True
    assert runtime.native_library_sha256 == seven_streams["native_sha256"]
    assert runtime.native_library_bytes == seven_streams["native_bytes"]

    def forbidden_loader(payload: bytes):
        del payload
        raise AssertionError("native loading must follow module identity verification")

    monkeypatch.setattr(worker, "_native_from_verified_bytes", forbidden_loader)
    with pytest.raises(worker.DecodeWorkerError, match="worker_module identity mismatch"):
        worker.load_runtime(
            tmp_path,
            "native-opened.so",
            expected_native_library_sha256=seven_streams["native_sha256"],
            expected_native_library_bytes=seven_streams["native_bytes"],
            expected_worker_module_sha256="0" * 64,
            expected_worker_module_bytes=seven_streams["worker_module_bytes"],
        )


def test_blob_hash_is_before_and_boundary_component_hashes_are_after_timer(
    seven_streams: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_hash = worker._sha256_bytes
    real_clock = worker.time.perf_counter_ns

    def observed_hash(payload: bytes) -> str:
        events.append("hash")
        return real_hash(payload)

    def observed_clock() -> int:
        events.append("clock")
        return real_clock()

    monkeypatch.setattr(worker, "_sha256_bytes", observed_hash)
    monkeypatch.setattr(worker.time, "perf_counter_ns", observed_clock)
    _decode(seven_streams, "ssp2v")
    assert events[0] == "hash"
    assert events[1:3] == ["clock", "clock"]
    assert events[3:] == ["hash"] * 4


def test_decode_blob_performs_no_filesystem_read_inside_timed_api(
    seven_streams: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_open(*args, **kwargs):
        del args, kwargs
        raise AssertionError("decode_blob must not open a file")

    monkeypatch.setattr(Path, "open", forbidden_open)
    record = _decode(seven_streams, "ssp2e")
    assert record["exactness"]["all_exact"] is True


def test_hash_mismatch_is_reported_but_not_promoted_to_a_valid_sample(
    seven_streams: dict[str, object],
) -> None:
    expected = seven_streams["expected"]["sspl1"]
    record = worker.decode_blob(
        seven_streams["blobs"]["sspl1"],
        seven_streams["runtime"],
        expected_blob_sha256=expected["blob"],
        expected_blob_bytes=expected["bytes"],
        expected_symbol_sha256="0" * 64,
        expected_boundary_state_sha256=expected["boundary"],
        measurement_session_sha256=SESSION,
        launch=worker.primary_schedule(0, 0)[0],
        pinned_cpu=None,
        synthetic=True,
    )
    assert record["exactness"]["all_exact"] is False
    with pytest.raises(worker.DecodeWorkerError, match="old, inexact"):
        worker.validate_sample_for_launch(
            record,
            worker.primary_schedule(0, 0)[0],
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )


def test_resource_validator_binds_exact_thread_affinity_and_source_environment(
    seven_streams: dict[str, object],
) -> None:
    launch = worker.primary_schedule(0, 0)[0]
    base = _decode(seven_streams, "sspl1", launch)
    wrong_threads = copy.deepcopy(base)
    wrong_threads["process"]["thread_environment"]["EXTRA_THREADS"] = "1"
    with pytest.raises(worker.DecodeWorkerError, match="old, inexact"):
        worker.validate_sample_for_launch(
            wrong_threads,
            launch,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )
    wrong_affinity = copy.deepcopy(base)
    wrong_affinity["process"]["cpu_affinity_before_pin"] = []
    with pytest.raises(worker.DecodeWorkerError, match="CPU-affinity"):
        worker.validate_sample_for_launch(
            wrong_affinity,
            launch,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )

    schedule = worker.primary_schedule(0, 0)
    samples = _synthetic_schedule_samples(base, schedule)
    samples[0]["source_bindings"]["worker_module"]["sha256"] = "b" * 64
    samples[0]["module"]["sha256"] = "b" * 64
    with pytest.raises(worker.DecodeWorkerError, match="source/environment binding"):
        worker.summarize_schedule(
            samples,
            schedule,
            image_index=0,
            tuple_index=0,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )


def test_primary_and_secondary_schedule_helpers_are_exact() -> None:
    primary = worker.primary_schedule(0, 0)
    assert len(primary) == 20
    assert [launch.to_dict() for launch in primary[:6]] == [
        worker.LaunchSpec("primary", 0, "warmup", None, "q").to_dict(),
        worker.LaunchSpec("primary", 1, "warmup", None, "candidate").to_dict(),
        worker.LaunchSpec("primary", 2, "measured", 0, "candidate").to_dict(),
        worker.LaunchSpec("primary", 3, "measured", 0, "q").to_dict(),
        worker.LaunchSpec("primary", 4, "measured", 1, "q").to_dict(),
        worker.LaunchSpec("primary", 5, "measured", 1, "candidate").to_dict(),
    ]
    flipped = worker.primary_schedule(0, 1)
    assert [launch.arm for launch in flipped[2:6]] == ["q", "candidate", "candidate", "q"]

    secondary = worker.secondary_schedule(0, 0)
    assert [launch.arm for launch in secondary[:6]] == [
        "sspl1",
        "ssp2l",
        "ssp2l",
        "sspl1",
        "sspl1",
        "ssp2l",
    ]
    assert all(launch.launch_index == index for index, launch in enumerate(secondary))
    with pytest.raises(worker.DecodeWorkerError):
        worker.primary_schedule(8, 0)
    with pytest.raises(worker.DecodeWorkerError):
        worker.secondary_schedule(0, 2)


def _synthetic_schedule_samples(
    base: dict[str, object], schedule: tuple[worker.LaunchSpec, ...]
) -> list[dict[str, object]]:
    elapsed = (90, 10, 80, 20, 70, 30, 60, 40, 50)
    result = []
    for index, launch in enumerate(schedule):
        sample = copy.deepcopy(base)
        sample["launch"] = launch.to_dict()
        sample["freshness"]["measurement_session_sha256"] = SESSION
        sample["process"]["pid"] = 10_000 + index
        if launch.phase == "warmup":
            sample["elapsed_ns"] = 999_999
            sample["peak_rss_bytes"] = 999_999
        else:
            sample["elapsed_ns"] = elapsed[launch.repetition]
            sample["peak_rss_bytes"] = 5000 if launch.repetition in (2, 5) else 1000
        result.append(sample)
    return result


def test_schedule_summary_uses_zero_based_fifth_and_rss_tie_provenance(
    seven_streams: dict[str, object],
) -> None:
    schedule = worker.primary_schedule(3, 1)
    samples = _synthetic_schedule_samples(_decode(seven_streams, "sspl1"), schedule)
    blob_hash = seven_streams["expected"]["sspl1"]["blob"]
    summary = worker.summarize_schedule(
        samples,
        schedule,
        image_index=3,
        tuple_index=1,
        measurement_session_sha256=SESSION,
        expected_blob_sha256_by_arm={"q": blob_hash, "candidate": blob_hash},
        allow_synthetic=True,
    )
    assert summary["schema"] == worker.SUMMARY_SCHEMA
    assert summary["fresh_process_count"] == 20
    assert summary["environment_exact_across_schedule"] is True
    assert summary["thread_environment"] == {name: "1" for name in worker.THREAD_VARIABLES}
    for arm in ("q", "candidate"):
        assert summary["summaries"][arm]["median_ns"] == 50
        assert summary["summaries"][arm]["median_repetition"] == 8
        assert summary["summaries"][arm]["maximum_peak_rss_bytes"] == 5000
        assert summary["summaries"][arm]["maximum_peak_rss_repetition"] == 2

    tied = copy.deepcopy(samples)
    for sample, launch in zip(tied, schedule, strict=True):
        if launch.phase == "measured":
            sample["elapsed_ns"] = 50
    tied_summary = worker.summarize_schedule(
        tied,
        schedule,
        image_index=3,
        tuple_index=1,
        measurement_session_sha256=SESSION,
        expected_blob_sha256_by_arm={"q": blob_hash, "candidate": blob_hash},
        allow_synthetic=True,
    )
    assert all(
        arm_summary["median_repetition"] == 4
        for arm_summary in tied_summary["summaries"].values()
    )


def test_old_cross_session_wrong_launch_and_reused_process_samples_are_rejected(
    seven_streams: dict[str, object],
) -> None:
    schedule = worker.secondary_schedule(1, 0)
    samples = _synthetic_schedule_samples(_decode(seven_streams, "sspl1"), schedule)
    old = copy.deepcopy(samples)
    old[0]["schema"] = "structsplat.comp009.ssp2e-decode-worker.v1"
    with pytest.raises(worker.DecodeWorkerError, match="old, inexact"):
        worker.summarize_schedule(
            old,
            schedule,
            image_index=1,
            tuple_index=0,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )

    cross_session = copy.deepcopy(samples)
    cross_session[4]["freshness"]["measurement_session_sha256"] = "b" * 64
    with pytest.raises(worker.DecodeWorkerError, match="old, inexact"):
        worker.summarize_schedule(
            cross_session,
            schedule,
            image_index=1,
            tuple_index=0,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )

    wrong_launch = copy.deepcopy(samples)
    wrong_launch[3]["launch"] = schedule[2].to_dict()
    with pytest.raises(worker.DecodeWorkerError, match="old, inexact"):
        worker.summarize_schedule(
            wrong_launch,
            schedule,
            image_index=1,
            tuple_index=0,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )

    reused = copy.deepcopy(samples)
    reused[7]["process"]["pid"] = reused[6]["process"]["pid"]
    with pytest.raises(worker.DecodeWorkerError, match="twenty fresh"):
        worker.summarize_schedule(
            reused,
            schedule,
            image_index=1,
            tuple_index=0,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )

    reordered_schedule = list(schedule)
    reordered_schedule[2], reordered_schedule[3] = reordered_schedule[3], reordered_schedule[2]
    reordered_schedule = tuple(
        worker.LaunchSpec(
            launch.schedule_kind,
            index,
            launch.phase,
            launch.repetition,
            launch.arm,
        )
        for index, launch in enumerate(reordered_schedule)
    )
    with pytest.raises(worker.DecodeWorkerError, match="parity order"):
        worker.summarize_schedule(
            samples,
            reordered_schedule,
            image_index=1,
            tuple_index=0,
            measurement_session_sha256=SESSION,
            allow_synthetic=True,
        )


def test_cli_emits_one_canonical_fresh_record(seven_streams: dict[str, object]) -> None:
    mode = "ssp2v"
    expected = seven_streams["expected"][mode]
    launch = worker.primary_schedule(0, 0)[2]
    command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_decode_worker",
        "resource",
        "--artifact-root",
        str(seven_streams["directory"]),
        "--blob",
        seven_streams["paths"][mode].name,
        "--native-library",
        seven_streams["native_path"].name,
        "--expected-blob-sha256",
        expected["blob"],
        "--expected-blob-bytes",
        str(expected["bytes"]),
        "--expected-native-library-sha256",
        seven_streams["native_sha256"],
        "--expected-native-library-bytes",
        str(seven_streams["native_bytes"]),
        "--expected-worker-module-sha256",
        seven_streams["worker_module_sha256"],
        "--expected-worker-module-bytes",
        str(seven_streams["worker_module_bytes"]),
        "--expected-symbol-sha256",
        expected["symbol"],
        "--expected-boundary-state-sha256",
        expected["boundary"],
        "--measurement-session-sha256",
        SESSION,
        "--schedule-kind",
        launch.schedule_kind,
        "--launch-index",
        str(launch.launch_index),
        "--phase",
        launch.phase,
        "--repetition",
        str(launch.repetition),
        "--arm",
        launch.arm,
    ]
    environment = dict(os.environ)
    environment.update({name: "1" for name in worker.THREAD_VARIABLES})
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.stderr == b""
    record = json.loads(completed.stdout)
    assert completed.stdout == worker.canonical_json(record)
    assert record["mode"] == mode
    assert record["launch"] == launch.to_dict()
    assert record["process"]["pid"] != os.getpid()
    assert record["exactness"]["all_exact"] is True
    assert record["execution_mode"] == "production-resource"
    assert record["operation"] == "resource-sample"
    assert record["blob_acquisition"]["kind"] == "artifact-root-relative-stable-open"
    assert record["process"]["pinned_cpu"] == record["process"]["cpu_affinity_before_pin"][0]


def test_cli_candidate_cold_seal_is_distinct_and_rejects_old_absolute_interface(
    seven_streams: dict[str, object],
) -> None:
    mode = "ssp2v"
    expected = seven_streams["expected"][mode]
    command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_decode_worker",
        "candidate-cold-seal",
        "--artifact-root",
        str(seven_streams["directory"]),
        "--blob",
        seven_streams["paths"][mode].name,
        "--native-library",
        seven_streams["native_path"].name,
        "--expected-blob-sha256",
        expected["blob"],
        "--expected-blob-bytes",
        str(expected["bytes"]),
        "--expected-native-library-sha256",
        seven_streams["native_sha256"],
        "--expected-native-library-bytes",
        str(seven_streams["native_bytes"]),
        "--expected-worker-module-sha256",
        seven_streams["worker_module_sha256"],
        "--expected-worker-module-bytes",
        str(seven_streams["worker_module_bytes"]),
        "--expected-symbol-sha256",
        expected["symbol"],
        "--expected-boundary-state-sha256",
        expected["boundary"],
        "--expected-stream-components-sha256",
        expected["components"],
    ]
    environment = dict(os.environ)
    environment.update({name: "1" for name in worker.THREAD_VARIABLES})
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        timeout=120,
    )
    record = json.loads(completed.stdout)
    assert completed.stdout == worker.canonical_json(record)
    assert record["schema"] == worker.COLD_SEAL_SCHEMA
    assert record["operation"] == "candidate-cold-seal"
    worker.verify_candidate_cold_seal(record)

    absolute = list(command)
    absolute[absolute.index("--blob") + 1] = str(seven_streams["paths"][mode])
    rejected = subprocess.run(
        absolute,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        timeout=120,
    )
    assert rejected.returncode != 0
    assert b"artifact-root-relative" in rejected.stderr

    old = subprocess.run(
        [sys.executable, "-m", "benchmarks.ssp2v_decode_worker", "--blob", "ssp2v.bin"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        timeout=120,
    )
    assert old.returncode != 0
    assert b"operation" in old.stderr
