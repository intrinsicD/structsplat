from __future__ import annotations

import hashlib
import json
import struct
import zlib

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_execute as execute
from benchmarks import ssp2e_model as model


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


def _synthetic_sspl1(
    n: int = 19,
    bits: tuple[int, int, int, int] = (12, 6, 6, 8),
) -> bytes:
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
def native_coder(tmp_path_factory: pytest.TempPathFactory):
    path = arithmetic.build_native(tmp_path_factory.mktemp("ssp2e-native") / "libssp2e_range_v1.so")
    return arithmetic.NativeArithmeticCoder(path)


def test_execute_cell_fits_selects_encodes_and_cold_decodes_end_to_end(native_coder) -> None:
    source = _synthetic_sspl1()
    result = execute.execute_cell(source, native_coder, synthetic=True)
    record = result.record

    assert record["schema"] == execute.EXECUTION_SCHEMA
    assert record["source"]["sha256"] == hashlib.sha256(source).hexdigest()
    assert record["source"]["n"] == 19
    assert set(record["integrity"].values()) == {True}
    assert set(result.blobs) == set(execute.STREAM_NAMES)
    assert set(result.boundary_arrays) == {"means", "log_scales", "rotations", "colors"}
    assert all(value.dtype == np.float32 for value in result.boundary_arrays.values())

    core = dict(record)
    observed_seal = core.pop("execution_sha256")
    serialized = (
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    assert observed_seal == hashlib.sha256(serialized).hexdigest()

    for name, magic in zip(
        execute.STREAM_NAMES, (b"SSP2E", b"SSP2S", b"SSP2L", b"SSP2F"), strict=True
    ):
        parsed = container.parse_container(result.blobs[name], reference_sspl1=source)
        stream = record["streams"][name]
        assert parsed.magic == magic
        assert stream["complete_bytes"] == len(result.blobs[name])
        assert stream["blob_sha256"] == hashlib.sha256(result.blobs[name]).hexdigest()
        assert stream["modeled_bytes"] + stream["shared_passthrough_bytes"] == len(
            result.blobs[name]
        )
        assert len(stream["certificates"]) == 5
        assert all(
            certificate["inequality_holds"] is True
            for certificate in stream["certificates"].values()
        )
        assert (
            record["boundary"]["decoded_state_sha256"][name] == record["boundary"]["state_sha256"]
        )

    for key, selection in (
        ("ssp2e", result.true_selection),
        ("ssp2s", result.shuffled_selection),
    ):
        diagnostics = record["models"][key]
        assert diagnostics["start_count"] == 8
        assert [item["start_id"] for item in diagnostics["starts"]] == list(range(8))
        assert all(item["fixed_point_verified"] for item in diagnostics["starts"])
        assert all(item["python_native_byte_parity"] for item in diagnostics["starts"])
        selected = selection.selected
        assert diagnostics["selection_reason"] == selection.selection_reason
        final_lengths = tuple(
            record["streams"][key]["arithmetic_payload_bytes"][name] for name in model.CHANNELS
        )
        assert final_lengths == selected.payload_lengths


def test_boundary_arrays_reject_partial_symbols_and_are_independent() -> None:
    source = _synthetic_sspl1(n=3)
    parts = container.extract_sspl1_parts(source)
    arrays = execute.boundary_arrays(parts.header, parts.symbols)
    assert arrays["means"].shape == (3, 2)
    assert arrays["colors"].shape == (3, 3)
    assert arrays["rotations"].shape == (3,)
    assert not any(
        np.shares_memory(value, symbol)
        for value in arrays.values()
        for symbol in parts.symbols.values()
    )

    incomplete = dict(parts.symbols)
    incomplete.pop("blue")
    with pytest.raises(execute.ExecutionError, match="all eight"):
        execute.boundary_arrays(parts.header, incomplete)


def test_execute_cell_fails_closed_on_start_level_native_byte_drift(monkeypatch) -> None:
    source = _synthetic_sspl1(n=3)

    def synthetic_fits(objective: model.ModelObjective) -> tuple[model.FitResult, ...]:
        pairs = model.best_global_pairs(objective)
        zero = (0,) * model.FEATURE_COUNT
        heads = tuple(model.ChannelHead(zero, location, zero, scale) for location, scale in pairs)
        fits = []
        for start_id in range(8):
            fitted = model.SSP2EModel(model.initial_grid(start_id), heads)
            objective_q24 = objective.objective_q24(fitted)
            trajectory = (
                model.SweepRecord(0, objective_q24, 0, 0, 0),
                model.SweepRecord(1, objective_q24, 0, 0, 0),
            )
            fits.append(model.FitResult(start_id, fitted, objective_q24, 1, trajectory))
        return tuple(fits)

    class DriftingNative:
        @staticmethod
        def encode(symbols: np.ndarray, cdf_rows: np.ndarray) -> bytes:
            return arithmetic.encode(symbols, cdf_rows) + b"\0"

        @staticmethod
        def decode(payload: bytes, cdf_rows: np.ndarray, n: int) -> np.ndarray:
            return arithmetic.decode(payload, cdf_rows, n)

    monkeypatch.setattr(model, "fit_eight_starts", synthetic_fits)
    monkeypatch.setattr(model, "verify_fixed_point", lambda objective, fitted: True)
    with pytest.raises(execute.ExecutionError, match="start 0/scale_x"):
        execute.execute_cell(source, DriftingNative(), synthetic=True)


def test_execute_cell_rejects_non_native_adapter_before_parsing() -> None:
    with pytest.raises(execute.ExecutionError, match="encode and decode"):
        execute.execute_cell(b"not-an-sspl1", object())


def test_production_mode_rejects_non_8192_synthetic_source_before_fitting() -> None:
    class PythonStandIn:
        encode = staticmethod(arithmetic.encode)
        decode = staticmethod(arithmetic.decode)

    with pytest.raises(execute.ExecutionError, match="production execution requires N=8192"):
        execute.execute_cell(_synthetic_sspl1(n=3), PythonStandIn())
