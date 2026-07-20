from __future__ import annotations

from dataclasses import replace
import inspect
import json
import struct
import zlib

import numpy as np
import pytest
import torch

from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_execute as execute
from benchmarks import ssp2f_rdo_operator as operator
from structsplat.gaussians import GaussianField


def _modular_deltas(values: np.ndarray, bits: int) -> np.ndarray:
    absolute = np.asarray(values, dtype=np.uint64)
    result = np.empty_like(absolute)
    result[0] = absolute[0]
    result[1:] = absolute[1:] - absolute[:-1]
    result &= np.uint64((1 << bits) - 1)
    return result.astype(np.uint32)


def _packed(values: np.ndarray, bits: int, *, byte_planar: bool = False) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    packed = np.ascontiguousarray(values, dtype=dtype)
    if not byte_planar or dtype.itemsize == 1:
        return packed.tobytes(order="C")
    interleaved = np.frombuffer(packed.tobytes(order="C"), dtype=np.uint8).reshape(
        packed.size,
        dtype.itemsize,
    )
    return np.ascontiguousarray(interleaved.T).tobytes(order="C")


def _synthetic_sspl1(
    *,
    rgb_shift: int = 0,
    aa_dilation: float = 0.125,
    sigma_cutoff: float = 2.75,
) -> bytes:
    n = 7
    header = container.ContainerHeader(
        n=n,
        height=9,
        width=11,
        bits=(12, 6, 6, 8),
        means_lo=(-3, -2),
        means_hi=(13, 11),
        scale_lo=(-1.0, -0.5),
        scale_hi=(1.0, 1.5),
        color_lo=(-0.5, -0.25, 0.0),
        color_hi=(1.5, 1.25, 2.0),
        aa_dilation=aa_dilation,
        sigma_cutoff=sigma_cutoff,
        render_chunk=64,
    ).validated()
    means = np.asarray(
        [
            [760, 620],
            [1200, 1110],
            [1720, 1410],
            [2250, 1940],
            [2790, 2380],
            [3300, 2890],
            [3900, 3450],
        ],
        dtype=np.uint32,
    )
    scales = np.asarray(
        [
            [9, 14],
            [18, 27],
            [30, 22],
            [42, 37],
            [51, 45],
            [24, 55],
            [36, 18],
        ],
        dtype=np.uint32,
    )
    rotations = np.asarray([0, 7, 18, 29, 41, 53, 61], dtype=np.uint32)
    colors = np.asarray(
        [
            [12, 230, 98],
            [245, 21, 180],
            [88, 140, 250],
            [200, 70, 35],
            [44, 192, 122],
            [155, 110, 76],
            [232, 33, 205],
        ],
        dtype=np.uint32,
    )
    colors[3, 1] = np.uint32((int(colors[3, 1]) + rgb_shift) & 255)

    mean_delta = _modular_deltas(means, header.bits[0])
    color_delta = _modular_deltas(colors, header.bits[3])
    raw_streams = (
        _packed(mean_delta.T.reshape(-1), header.bits[0], byte_planar=True),
        _packed(scales.T.reshape(-1), header.bits[1]),
        _packed(rotations, header.bits[2]),
        _packed(color_delta.T.reshape(-1), header.bits[3]),
    )
    semantic_header = header.to_sspl1_header()
    header_raw = json.dumps(semantic_header).encode("utf-8")
    blob = b"SSPL1" + struct.pack("<I", len(header_raw)) + header_raw
    for raw in raw_streams:
        payload = zlib.compress(raw, level=9)
        blob += struct.pack("<I", len(payload)) + payload
    return blob


@pytest.fixture(scope="module")
def source_blob() -> bytes:
    return _synthetic_sspl1()


@pytest.fixture(scope="module")
def bound(source_blob: bytes) -> operator.BoundSSP2FRDOOperator:
    parts = container.extract_sspl1_parts(source_blob)
    return operator.build_source_bound_operator(
        source_blob,
        cold_parts=parts,
        synthetic=True,
    )


def test_source_binding_reconstructs_exact_field_header_and_objective(
    source_blob: bytes,
    bound: operator.BoundSSP2FRDOOperator,
) -> None:
    parts = container.extract_sspl1_parts(source_blob)
    boundary = execute.boundary_arrays(parts.header, parts.symbols)
    record = bound.binding_record

    operator.verify_binding_record(record)
    assert record["schema"] == operator.OPERATOR_BINDING_SCHEMA
    assert record["task_sha256"] == operator.TASK_SHA256
    assert record["synthetic"] is True
    assert record["authority_scope"] == "synthetic-mechanics-only"
    assert record["target_input_surface"] == "absent"
    assert record["target_accessed"] is False
    assert record["source"]["blob_sha256"] == parts.blob_sha256
    assert record["source"]["symbol_sha256"] == parts.symbol_sha256
    assert record["source"]["cold_parts_cross_checked"] is True
    assert record["header"] == {
        "bytes": container.HEADER.size,
        "sha256": record["header"]["sha256"],
        "n": parts.header.n,
        "height": parts.header.height,
        "width": parts.header.width,
        "bits": list(parts.header.bits),
        "aa_dilation_hex": float(parts.header.aa_dilation).hex(),
        "sigma_cutoff_hex": float(parts.header.sigma_cutoff).hex(),
        "render_chunk": parts.header.render_chunk,
        "renderer_id": parts.header.renderer_id,
        "support_fade": False,
    }
    assert record["boundary"]["state_sha256"] == execute._array_state_digest(  # noqa: SLF001
        boundary
    )
    assert bound.operator.pixel_count == parts.header.height * parts.header.width
    assert bound.operator.column_count == parts.header.n

    returned_boundary = bound.boundary_arrays
    for name in boundary:
        np.testing.assert_array_equal(returned_boundary[name], boundary[name])
    source_rgb = np.column_stack(
        (parts.symbols["red"], parts.symbols["green"], parts.symbols["blue"])
    ).astype(np.uint8)
    np.testing.assert_array_equal(bound.source_rgb_symbols, source_rgb)

    field = GaussianField(
        torch.from_numpy(boundary["means"]).clone(),
        torch.from_numpy(boundary["log_scales"]).clone(),
        torch.from_numpy(boundary["rotations"]).clone(),
        torch.from_numpy(boundary["colors"]).clone(),
        opacities=None,
    )
    np.testing.assert_array_equal(
        bound.conics,
        field.conics(parts.header.aa_dilation).detach().numpy(),
    )
    np.testing.assert_array_equal(
        bound.radii,
        field.radii(parts.header.sigma_cutoff, parts.header.aa_dilation).numpy(),
    )
    assert record["construction"]["opacities"] == "absent"
    assert record["construction"]["support_fade"] is False
    assert record["construction"]["eps_hex"] == float(1e-8).hex()
    assert record["construction"]["thread_state"]["torch_num_threads"] == 1
    assert record["construction"]["thread_state"]["torch_num_interop_threads"] == 1
    assert record["construction"]["thread_state"]["deterministic_algorithms"] is True

    source_engine = bound.source_objective()
    assert source_engine.operator_sha256 == bound.operator.sha256
    assert source_engine.objective == 0.0
    assert record["objective"]["source_reference"] == record["operator"][
        "source_clamped_render"
    ]
    np.testing.assert_array_equal(source_engine.rgb_symbols, source_rgb)


def test_synthetic_gate_matches_cpu_forward_column_action_and_clamped_sse(
    bound: operator.BoundSSP2FRDOOperator,
) -> None:
    source = bound.source_rgb_symbols
    replacement = (int(source[4, 2]) + 61) & 255
    report = operator.validate_synthetic_cpu_parity(
        bound,
        row=4,
        channel=2,
        new_symbol=replacement,
    )

    assert report.schema == operator.SYNTHETIC_GATE_SCHEMA
    assert report.binding_sha256 == bound.binding_sha256
    assert report.forward_passed
    assert report.column_delta_passed
    assert report.objective_delta_passed
    assert report.passed
    assert report.forward_max_abs_error <= report.forward_atol + report.forward_rtol
    assert report.column_delta_max_abs_error <= (
        report.column_delta_atol + report.column_delta_rtol
    )

    source_engine = bound.source_objective()
    proposal = source_engine.propose(4, 2, replacement)
    assert proposal.objective.hex() == report.objective_operator_delta.hex()


def test_operator_is_target_independent_and_rgb_independent() -> None:
    first = operator.build_source_bound_operator(_synthetic_sspl1(), synthetic=True)
    rgb_changed = operator.build_source_bound_operator(
        _synthetic_sspl1(rgb_shift=17),
        synthetic=True,
    )
    first_record = first.binding_record
    changed_record = rgb_changed.binding_record

    assert "target" not in inspect.signature(operator.build_source_bound_operator).parameters
    assert first.operator.sha256 == rgb_changed.operator.sha256
    assert (
        first_record["source"]["common_symbol_sha256"]
        == changed_record["source"]["common_symbol_sha256"]
    )
    assert (
        first_record["boundary"]["geometry_state_sha256"]
        == changed_record["boundary"]["geometry_state_sha256"]
    )
    assert first_record["source"]["rgb_symbol_record"] != changed_record["source"][
        "rgb_symbol_record"
    ]
    assert first.binding_sha256 != rgb_changed.binding_sha256

    first_binding = first.binding_sha256
    all_black = np.zeros((first.header.height, first.header.width, 3), dtype=np.uint8)
    all_white = np.full_like(all_black, 255)
    black_objective = first.target_objective(all_black)
    white_objective = first.target_objective(all_white)
    assert black_objective.operator_sha256 == white_objective.operator_sha256
    assert black_objective.config_sha256 != white_objective.config_sha256
    assert first.binding_sha256 == first_binding


def test_header_dilation_and_cutoff_are_operator_inputs() -> None:
    first = operator.build_source_bound_operator(
        _synthetic_sspl1(aa_dilation=0.0, sigma_cutoff=2.5),
        synthetic=True,
    )
    second = operator.build_source_bound_operator(
        _synthetic_sspl1(aa_dilation=0.25, sigma_cutoff=3.0),
        synthetic=True,
    )

    assert first.binding_record["header"]["aa_dilation_hex"] == float(0.0).hex()
    assert first.binding_record["header"]["sigma_cutoff_hex"] == float(2.5).hex()
    assert second.binding_record["header"]["aa_dilation_hex"] == float(0.25).hex()
    assert second.binding_record["header"]["sigma_cutoff_hex"] == float(3.0).hex()
    assert first.operator.sha256 != second.operator.sha256
    assert not np.array_equal(first.conics, second.conics)
    assert not np.array_equal(first.radii, second.radii)


def test_public_arrays_and_records_are_detached(
    bound: operator.BoundSSP2FRDOOperator,
) -> None:
    binding_sha256 = bound.binding_sha256
    source = bound.source_rgb_symbols
    boundary = bound.boundary_arrays
    conics = bound.conics
    radii = bound.radii
    denominator = bound.denominator
    source_reference = bound.source_reference
    source[:] = 0
    boundary["means"][:] = 0.0
    conics[:] = 0.0
    radii[:] = 1
    denominator[:] = 0.0
    source_reference[:] = 0.0

    assert bound.binding_sha256 == binding_sha256
    assert np.any(bound.source_rgb_symbols != 0)
    assert np.any(bound.boundary_arrays["means"] != 0.0)
    assert np.any(bound.conics != 0.0)
    assert np.any(bound.radii != 1)
    assert np.any(bound.denominator != 0.0)
    assert np.any(bound.source_reference != 0.0)

    record = bound.binding_record
    record["operator"]["sha256"] = "0" * 64
    with pytest.raises(operator.SSP2FRDOOperatorError, match="SHA-256 mismatch"):
        operator.verify_binding_record(record)
    assert bound.binding_sha256 == binding_sha256


def test_fail_closed_on_malformed_mismatched_or_mislabeled_inputs(
    source_blob: bytes,
    bound: operator.BoundSSP2FRDOOperator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(operator.SSP2FRDOOperatorError, match="immutable bytes"):
        operator.build_source_bound_operator(bytearray(source_blob), synthetic=True)  # type: ignore[arg-type]
    with pytest.raises(operator.SSP2FRDOOperatorError, match="strict SSPL1 extraction"):
        operator.build_source_bound_operator(b"not-an-sspl1", synthetic=True)
    with pytest.raises(operator.SSP2FRDOOperatorError, match="cold-decoded"):
        operator.build_source_bound_operator(source_blob)
    exact_parts = container.extract_sspl1_parts(source_blob)
    with pytest.raises(operator.SSP2FRDOOperatorError, match="production operator requires"):
        operator.build_source_bound_operator(source_blob, cold_parts=exact_parts)

    other_parts = container.extract_sspl1_parts(_synthetic_sspl1(rgb_shift=9))
    with pytest.raises(operator.SSP2FRDOOperatorError, match="supplied cold parts differ"):
        operator.build_source_bound_operator(
            source_blob,
            cold_parts=other_parts,
            synthetic=True,
        )
    forged_parts = replace(exact_parts, symbol_sha256="0" * 64)
    with pytest.raises(operator.SSP2FRDOOperatorError, match="supplied cold parts differ"):
        operator.build_source_bound_operator(
            source_blob,
            cold_parts=forged_parts,
            synthetic=True,
        )

    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(operator.SSP2FRDOOperatorError, match="OMP_NUM_THREADS"):
        operator.build_source_bound_operator(source_blob, synthetic=True)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    wrong_target = np.zeros((bound.header.height, bound.header.width), dtype=np.uint8)
    with pytest.raises(operator.SSP2FRDOOperatorError, match="exact header shape"):
        bound.target_objective(wrong_target)
    with pytest.raises(operator.SSP2FRDOOperatorError, match="must change"):
        operator.validate_synthetic_cpu_parity(
            bound,
            row=0,
            channel=0,
            new_symbol=int(bound.source_rgb_symbols[0, 0]),
        )
    with pytest.raises(operator.SSP2FRDOOperatorError, match="synthetic bound"):
        operator.validate_synthetic_cpu_parity(  # type: ignore[arg-type]
            object(),
            row=0,
            channel=0,
            new_symbol=1,
        )


@pytest.mark.parametrize(
    ("aa_dilation", "sigma_cutoff", "message"),
    (
        (-0.25, 2.75, "aa_dilation"),
        (0.125, 0.0, "sigma_cutoff"),
    ),
)
def test_fail_closed_on_invalid_header_geometry_parameters(
    aa_dilation: float,
    sigma_cutoff: float,
    message: str,
) -> None:
    blob = _synthetic_sspl1(
        aa_dilation=aa_dilation,
        sigma_cutoff=sigma_cutoff,
    )
    with pytest.raises(operator.SSP2FRDOOperatorError, match=message):
        operator.build_source_bound_operator(blob, synthetic=True)
