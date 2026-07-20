from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_decode_worker as worker
from benchmarks import ssp2e_execute as execute


ROOT = Path(__file__).resolve().parents[1]


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


def _synthetic_sspl1(n: int = 19) -> bytes:
    bits = (12, 6, 6, 8)
    bm, bs, br, bc = bits
    row = np.arange(n, dtype=np.uint32)
    means = np.column_stack(((29 * row) % (1 << bm), (47 * row[::-1]) % (1 << bm))).astype(
        np.uint32
    )
    scales = np.column_stack((row % (1 << bs), (3 * row + 2) % (1 << bs))).astype(np.uint32)
    rotation = ((5 * row + 1) % (1 << br)).astype(np.uint32)
    colors = np.column_stack(
        ((7 * row) % (1 << bc), (11 * row + 1) % (1 << bc), (13 * row + 2) % (1 << bc))
    ).astype(np.uint32)
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


@pytest.fixture(scope="module")
def five_streams(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("ssp2e-worker")
    native_path = arithmetic.build_native(directory / "libssp2e_range_v1.so")
    source = _synthetic_sspl1()
    tables = assets.load_tables()

    def frequencies(alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        return tables[alphabet].frequency_row(location, scale)

    head_channel = container.HeadChannel((0,) * 8, 32, (0,) * 8, 8)
    head = container.serialize_head((head_channel,) * 5)
    grid = bytes(range(256))
    globals_by_channel = {name: (32, 8) for name in container.MODELED_CHANNELS}
    blobs = {
        "sspl1": source,
        "ssp2e": container.encode_contextual(
            source,
            grid,
            head,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=frequencies,
            magic=b"SSP2E",
        ),
        "ssp2s": container.encode_contextual(
            source,
            grid,
            head,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=frequencies,
            magic=b"SSP2S",
        ),
        "ssp2l": container.encode_ssp2l(
            source,
            globals_by_channel,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=frequencies,
        ),
        "ssp2f": container.encode_ssp2f(source, arithmetic_encode=arithmetic.encode),
    }
    paths = {}
    for name, blob in blobs.items():
        path = directory / f"{name}.bin"
        path.write_bytes(blob)
        paths[name] = path

    parts = container.extract_sspl1_parts(source)
    boundary = execute.boundary_arrays(parts.header, parts.symbols)
    return {
        "native_path": native_path,
        "blobs": blobs,
        "paths": paths,
        "symbol_sha256": parts.symbol_sha256,
        "boundary_sha256": execute._array_state_digest(boundary),  # noqa: SLF001
    }


@pytest.mark.parametrize("mode", ["sspl1", "ssp2e", "ssp2s", "ssp2l", "ssp2f"])
def test_decode_blob_covers_all_five_strict_modes(
    five_streams: dict[str, object], mode: str
) -> None:
    runtime = worker.load_runtime(five_streams["native_path"])
    record = worker.decode_blob(
        five_streams["blobs"][mode],
        runtime,
        expected_symbol_sha256=five_streams["symbol_sha256"],
        expected_boundary_state_sha256=five_streams["boundary_sha256"],
        synthetic=True,
    )

    assert record["schema"] == worker.WORKER_SCHEMA
    assert record["mode"] == mode
    assert record["n"] == 19
    assert record["bit_tuple"] == [12, 6, 6, 8]
    assert isinstance(record["elapsed_ns"], int) and record["elapsed_ns"] > 0
    assert isinstance(record["peak_rss_bytes"], int) and record["peak_rss_bytes"] > 0
    assert record["symbol_sha256"] == five_streams["symbol_sha256"]
    assert record["boundary_state_sha256"] == five_streams["boundary_sha256"]
    assert set(record["exactness"].values()) >= {True}
    assert record["exactness"]["all_exact"] is True
    assert record["native"]["arithmetic_core_used_in_timed_region"] is (mode != "sspl1")
    assert record["native"]["redundant_second_native_encode_eliminated"] is (mode != "sspl1")
    assert record["timing"]["scope"] == list(worker.TIMING_SCOPE)
    assert record["timing"]["renderer_included"] is False
    assert json.loads(worker.canonical_json(record)) == record


@pytest.mark.parametrize("mode", ["sspl1", "ssp2e", "ssp2s", "ssp2l", "ssp2f"])
def test_cli_is_a_canonical_fresh_process_for_all_five_modes(
    five_streams: dict[str, object], mode: str
) -> None:
    command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2e_decode_worker",
        "--blob",
        str(five_streams["paths"][mode]),
        "--native-library",
        str(five_streams["native_path"]),
        "--expected-symbol-sha256",
        five_streams["symbol_sha256"],
        "--expected-boundary-state-sha256",
        five_streams["boundary_sha256"],
        "--synthetic",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        check=True,
        capture_output=True,
    )
    assert completed.stderr == b""
    record = json.loads(completed.stdout)
    assert completed.stdout == worker.canonical_json(record)
    assert record["mode"] == mode
    assert record["exactness"]["all_exact"] is True
    assert record["process"]["pid"] != os.getpid()
    assert record["process"]["cpu_affinity"] == [record["process"]["pinned_cpu"]]


def test_hash_mismatch_is_reported_as_nonexact(five_streams: dict[str, object]) -> None:
    runtime = worker.load_runtime(five_streams["native_path"])
    record = worker.decode_blob(
        five_streams["blobs"]["sspl1"],
        runtime,
        expected_symbol_sha256="0" * 64,
        expected_boundary_state_sha256=five_streams["boundary_sha256"],
        synthetic=True,
    )
    assert record["exactness"] == {
        "expected_symbol_sha256": "0" * 64,
        "expected_boundary_state_sha256": five_streams["boundary_sha256"],
        "symbol_hash_exact": False,
        "boundary_hash_exact": True,
        "all_exact": False,
    }


def test_production_worker_rejects_non_8192_stream(five_streams: dict[str, object]) -> None:
    runtime = worker.load_runtime(five_streams["native_path"])
    with pytest.raises(
        worker.DecodeWorkerError, match="production resource decode requires N=8192"
    ):
        worker.decode_blob(
            five_streams["blobs"]["sspl1"],
            runtime,
            expected_symbol_sha256=five_streams["symbol_sha256"],
            expected_boundary_state_sha256=five_streams["boundary_sha256"],
        )


@pytest.mark.parametrize("alphabet", [64, 256])
def test_vectorized_cdf_rows_match_the_old_per_row_reference_for_all_table_keys(
    alphabet: int,
) -> None:
    table = assets.load_tables()[alphabet]
    keys = np.tile(np.arange(64 * 16, dtype=np.uint16), 8)
    order = np.random.default_rng(20260716).permutation(keys.size)
    keys = keys[order]
    locations = (keys // 16).astype(np.uint8)
    scales = (keys % 16).astype(np.uint8)

    def frequencies(selected_alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        assert selected_alphabet == alphabet
        return table.frequency_row(location, scale)

    observed = container.cdf_rows_from_models(alphabet, locations, scales, frequencies)
    expected = np.asarray(
        [table.row(int(location), int(scale)) for location, scale in zip(locations, scales)],
        dtype=np.uint32,
    )
    assert observed.flags.c_contiguous
    assert observed.dtype == np.uint32
    np.testing.assert_array_equal(observed, expected)


def _native_proof_case(
    runtime: worker.DecodeRuntime,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    table = runtime.tables[64]
    symbols = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint32)
    cdfs = np.asarray(
        [table.row(index, index % 16) for index in range(symbols.size)],
        dtype=np.uint32,
    )
    return arithmetic.encode(symbols, cdfs), symbols, cdfs


def test_native_proof_adapter_is_one_shot_and_consumed(five_streams: dict[str, object]) -> None:
    runtime = worker.load_runtime(five_streams["native_path"])
    payload, symbols, cdfs = _native_proof_case(runtime)
    adapter = worker.StrictNativeDecodeProofAdapter(runtime.native)
    decoded = adapter.decode(payload, cdfs, symbols.size)
    np.testing.assert_array_equal(decoded, symbols)
    assert adapter.encode(decoded, cdfs) == payload
    adapter.assert_consumed()
    with pytest.raises(worker.DecodeWorkerError, match="no preceding decode"):
        adapter.encode(decoded, cdfs)


def test_native_proof_adapter_rejects_encode_before_decode_and_dangling_decode(
    five_streams: dict[str, object],
) -> None:
    runtime = worker.load_runtime(five_streams["native_path"])
    payload, symbols, cdfs = _native_proof_case(runtime)
    adapter = worker.StrictNativeDecodeProofAdapter(runtime.native)
    with pytest.raises(worker.DecodeWorkerError, match="no preceding decode"):
        adapter.encode(symbols, cdfs)
    adapter.decode(payload, cdfs, symbols.size)
    with pytest.raises(worker.DecodeWorkerError, match="dangling"):
        adapter.assert_consumed()
    with pytest.raises(worker.DecodeWorkerError, match="unconsumed decode"):
        adapter.decode(payload, cdfs, symbols.size)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("symbols", "decoded symbols changed"),
        ("cdf_identity", "CDF identity changed"),
        ("cdf_content", "CDF shape, dtype, or content changed"),
        ("cdf_order", "CDF identity changed"),
    ],
)
def test_native_proof_adapter_rejects_every_cached_proof_mutation(
    five_streams: dict[str, object], mutation: str, match: str
) -> None:
    runtime = worker.load_runtime(five_streams["native_path"])
    payload, _, cdfs = _native_proof_case(runtime)
    adapter = worker.StrictNativeDecodeProofAdapter(runtime.native)
    decoded = adapter.decode(payload, cdfs, cdfs.shape[0])
    selected_symbols = decoded
    selected_cdfs = cdfs
    if mutation == "symbols":
        selected_symbols = decoded.copy()
        selected_symbols[0] += 1
    elif mutation == "cdf_identity":
        selected_cdfs = cdfs.copy()
    elif mutation == "cdf_content":
        cdfs[0, 1] += 1
    elif mutation == "cdf_order":
        selected_cdfs = cdfs[::-1].copy()
    else:  # pragma: no cover - parameter list is frozen above
        raise AssertionError(mutation)
    with pytest.raises(worker.DecodeWorkerError, match=match):
        adapter.encode(selected_symbols, selected_cdfs)
    adapter.assert_consumed()
