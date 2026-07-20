from __future__ import annotations

import inspect
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest
import torch

from benchmarks import affine_carrier_assay as assay
from benchmarks import affine_carrier_core as core


ROOT = Path(__file__).resolve().parents[1]


def _broad_arrays(count: int = assay.PROXY_COUNT) -> dict[str, np.ndarray]:
    index = np.arange(count, dtype=np.float32)
    return {
        "means": np.stack((index % 9 + 1.0, index // 9), axis=1),
        "log_scales": np.full((count, 2), np.log(18.0), dtype=np.float32),
        "rotations": np.zeros(count, dtype=np.float32),
        "colors": np.stack(
            (
                0.35 + 0.001 * index,
                0.50 - 0.001 * index,
                0.60 + 0.0005 * index,
            ),
            axis=1,
        ).astype(np.float32),
    }


def _replace_crc(payload_without_crc: bytes) -> bytes:
    return payload_without_crc + struct.pack(
        "<I", zlib.crc32(payload_without_crc) & 0xFFFFFFFF
    )


def test_frozen_matrix_and_cell_field_roles() -> None:
    assert (assay.HEIGHT, assay.WIDTH) == (73, 89)
    assert assay.SEEDS == (307, 311, 313)
    assert set(assay.COUNTS) == {81, 83}
    assert assay.ARMS == ("nw81", "nw83", "ac81")
    assert len(assay.field_specs()) == 54
    assert len(assay.cell_specs()) == 48
    assert assay.EXPECTED_FORWARD_ROWS == 144
    assert assay.EXPECTED_PERMUTATION_ROWS == 576
    assert assay.EXPECTED_CONVERGENCE_TRAJECTORIES == 96
    assert assay.EXPECTED_CONVERGENCE_ROWS == 9_696
    for cell in assay.cell_specs():
        assert "__n0081__" in cell["baseline_field_id"]
        assert "__n0083__" in cell["challenger_field_id"]


def test_formula_bindings_are_the_exact_frozen_strings() -> None:
    assert assay.TARGET_FORMULAS == {
        "constant": "(0.25,0.50,0.75)",
        "affine": (
            "(0.45+0.18*q_x+0.12*q_y, 0.55-0.16*q_x-0.10*q_y, "
            "0.45+0.10*q_x+0.20*q_y)"
        ),
        "affine_sin": (
            "(0.45+0.16*q_x+0.10*q_y+0.07*sin(2*pi*u)*sin(pi*v), "
            "0.55-0.14*q_x-0.08*q_y+0.06*sin(2*pi*v), "
            "0.45+0.08*q_x+0.17*q_y+0.06*sin(2*pi*u)*sin(2*pi*v))"
        ),
        "affine_bump": (
            "g=exp(-((u-0.68)^2/(2*0.11^2)+(v-0.35)^2/(2*0.14^2))); "
            "(0.38+0.13*q_x+0.09*q_y+0.18*g, "
            "0.56-0.12*q_x-0.08*q_y-0.12*g, "
            "0.43+0.08*q_x+0.14*q_y+0.15*g)"
        ),
        "saddle": (
            "(0.48+0.10*q_x+0.07*q_y+0.10*q_x*q_y, "
            "0.52-0.08*q_x+0.06*q_y+0.09*(q_x^2-q_y^2), "
            "0.45+0.06*q_x+0.12*q_y-0.08*q_x*q_y)"
        ),
        "zero_linear": (
            "(0.45+0.12*cos(2*pi*u)*cos(2*pi*v), "
            "0.55+0.10*cos(2*pi*u)+0.06*cos(2*pi*v), "
            "0.45+0.10*sin(2*pi*u)*sin(2*pi*v))"
        ),
        "vertical_step": (
            "(0.1875,0.3125,0.75) for u<0.5, else (0.8125,0.6875,0.25)"
        ),
        "checker9x7": (
            "parity is (floor(9*u)+floor(7*v)) mod 2, using the two "
            "vertical_step colors in parity order zero/one"
        ),
    }


def test_checker_uses_nine_by_seven_normalized_frequency() -> None:
    points = np.array(
        [
            (0.0, 0.0),
            (assay.WIDTH / 9.0, 0.0),
            (0.0, assay.HEIGHT / 7.0),
            (assay.WIDTH - 1.0, assay.HEIGHT - 1.0),
        ]
    )
    values = assay.target_values("checker9x7", points)
    low = np.array((0.1875, 0.3125, 0.75))
    high = np.array((0.8125, 0.6875, 0.25))
    assert np.array_equal(values[0], low)
    assert np.array_equal(values[1], high)
    assert np.array_equal(values[2], high)
    assert np.array_equal(values[3], low)


def test_target_arrays_have_frozen_shapes_dtypes_and_ranges() -> None:
    for target in assay.TARGETS:
        image64 = assay.target_image(target, dtype=np.float64)
        image32 = assay.target_image(target, dtype=np.float32)
        assert image64.shape == (73, 89, 3)
        assert image64.dtype == np.float64
        assert image32.dtype == np.float32
        assert np.array_equal(image32, image64.astype(np.float32))
        assert np.isfinite(image64).all()


def test_afcr_roundtrip_and_exact_accounting() -> None:
    arrays = _broad_arrays()
    beta = torch.tensor(
        ((0.125, -0.0625, 0.03125), (0.0625, 0.125, -0.03125))
    )
    tail = core.encode_beta_f16(beta)
    payload = assay.encode_stream(
        "ac81", arrays, height=9, width=11, beta_payload=tail
    )
    decoded = assay.decode_stream(payload, "cpu")
    record = assay._stream_component_record(payload, decoded)  # noqa: SLF001
    assert decoded["arm"] == "ac81"
    assert decoded["count"] == 81
    assert (decoded["height"], decoded["width"]) == (9, 11)
    assert np.array_equal(decoded["beta"], beta.numpy())
    assert record["common_wrapper_overhead_bytes"] == 24
    assert record["components"]["tail_bytes"] == 12
    assert record["component_accounting_pass"]
    assert record["ordinary_decoded_reencode_pass"]
    assert record["deterministic_encode_pass"]


def test_nw_ordinary_state_samples_target_at_original_means() -> None:
    source = _broad_arrays()
    ordinary = assay._ordinary_stream_arrays(  # noqa: SLF001
        source, "affine", height=9, width=11
    )
    expected = assay.target_values(
        "affine", source["means"], height=9, width=11, dtype=np.float64
    ).astype(np.float32)
    assert np.array_equal(ordinary["means"], source["means"])
    assert np.array_equal(ordinary["colors"], expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (("version", 2), ("variant", 2), ("reserved", 1)),
)
def test_afcr_rejects_invalid_fixed_header(field: str, value: int) -> None:
    payload = assay.encode_stream("nw81", _broad_arrays(), height=9, width=11)
    header = list(assay.STREAM_HEADER.unpack_from(payload))
    header[{"version": 1, "variant": 2, "reserved": 3}[field]] = value
    malformed = _replace_crc(
        assay.STREAM_HEADER.pack(*header)
        + payload[assay.STREAM_HEADER.size : -assay.STREAM_CRC.size]
    )
    with pytest.raises(ValueError):
        assay.decode_stream(malformed, "cpu")


def test_afcr_rejects_tail_mismatch_crc_trailing_and_nonfinite() -> None:
    tail = core.encode_beta_f16(torch.zeros(core.BETA_SHAPE))
    payload = assay.encode_stream(
        "ac81", _broad_arrays(), height=9, width=11, beta_payload=tail
    )
    header = list(assay.STREAM_HEADER.unpack_from(payload))
    wrong_tail = header.copy()
    wrong_tail[5] = 10
    malformed_tail = _replace_crc(
        assay.STREAM_HEADER.pack(*wrong_tail)
        + payload[assay.STREAM_HEADER.size : -assay.STREAM_CRC.size]
    )
    corrupted_crc = bytearray(payload)
    corrupted_crc[-1] ^= 1
    nonfinite = bytearray(payload[:-assay.STREAM_CRC.size])
    offset = assay.STREAM_HEADER.size + int(header[4])
    nonfinite[offset : offset + 2] = b"\x00\x7c"
    for malformed in (
        malformed_tail,
        bytes(corrupted_crc),
        payload + b"\0",
        _replace_crc(bytes(nonfinite)),
    ):
        with pytest.raises(ValueError):
            assay.decode_stream(malformed, "cpu")


def test_ac83_is_not_a_legal_decoder_arm() -> None:
    arrays83 = _broad_arrays(assay.PRIMARY_COUNT)
    nw83 = assay.encode_stream("nw83", arrays83, height=9, width=11)
    header = assay.STREAM_HEADER.unpack_from(nw83)
    inner = nw83[assay.STREAM_HEADER.size : assay.STREAM_HEADER.size + header[4]]
    prefix = assay.STREAM_HEADER.pack(
        assay.STREAM_MAGIC, assay.STREAM_VERSION, 1, 0, len(inner), 12
    )
    with pytest.raises(ValueError):
        assay.decode_stream(_replace_crc(prefix + inner + bytes(12)), "cpu")


def test_decoder_api_has_no_source_or_target_argument_or_solve() -> None:
    assert tuple(inspect.signature(assay.decode_stream).parameters) == (
        "payload",
        "device",
    )
    source = inspect.getsource(assay._decode_stream_impl).lower()  # noqa: SLF001
    assert all(token not in source for token in ("target", "lstsq", "solve", "qr"))


def test_static_float32_float64_parity_is_diagnostic_not_a_gate() -> None:
    payload = assay.encode_stream("nw81", _broad_arrays(), height=9, width=11)
    decoded = assay.decode_stream(payload, "cpu")
    candidate, reference = assay._render_decoded(decoded)  # noqa: SLF001
    target = candidate.output.numpy().astype(np.float64)
    metrics, gates = assay._static_metrics_and_gates(  # noqa: SLF001
        candidate, reference, target
    )
    assert "candidate_reference_max_abs" in metrics
    assert "candidate_reference" not in gates


def test_nw_timing_path_does_not_touch_pixel_tail_design() -> None:
    payload = assay.encode_stream("nw81", _broad_arrays(), height=9, width=11)
    decoded = assay.decode_stream(payload, "cpu")
    state, radii, beta = assay._prepared_timing_state(decoded)  # noqa: SLF001
    assert beta is None
    unusable_design = torch.empty((0, 2), dtype=torch.float32)
    output = assay._timed_packed_output(  # noqa: SLF001
        state, radii, None, unusable_design, 9, 11
    )
    assert output.shape == (9, 11, 3)
    with pytest.raises(RuntimeError):
        assay._timed_packed_output(  # noqa: SLF001
            state,
            radii,
            torch.zeros(core.BETA_SHAPE),
            unusable_design,
            9,
            11,
        )


def test_plumbing_controls_pass_and_cover_hostile_cases() -> None:
    controls, arrays = assay.run_plumbing_controls()
    assert controls["valid"]
    assert controls["symmetric_constant_gauge"]["weight_reflection_pass"]
    assert controls["symmetric_constant_gauge"]["pixel_reflection_pass"]
    assert controls["symmetric_constant_gauge"]["mean_reflection_pass"]
    assert controls["broad_affine"]["decoded_beta_exact"]
    assert controls["broad_affine"]["stream_reencode_pass"]
    assert controls["broad_affine"]["stream_accounting_pass"]
    assert controls["alpha_zero"]["channel_sse_pass"]
    assert all(controls["wrapper_rejections"]["cases"].values())
    assert "symmetric_defect" in arrays


def test_source_binding_and_archive_closure_lists_all_executed_sources() -> None:
    manifest = assay._source_manifest(ROOT)  # noqa: SLF001
    assert set(manifest) == set(assay.SOURCE_PATHS)
    assert assay.TASK_PATH in manifest
    assert "benchmarks/affine_carrier_core.py" in manifest
    assert "benchmarks/affine_carrier_assay.py" in manifest
    assert "benchmarks/gauge_free_covariance_codec.py" in manifest
    assert "tests/test_affine_carrier_assay.py" in assay.ARCHIVE_PATHS


def test_run_source_orders_binding_before_target_generation_and_controls_before_fields() -> None:
    source = inspect.getsource(assay._run_impl)  # noqa: SLF001
    assert source.index('"config.json"') < source.index("_target_manifest")
    assert source.index('"source_manifest.json"') < source.index("_target_manifest")
    assert source.index("_write_source_archive") < source.index("_target_manifest")
    assert source.index("run_plumbing_controls") < source.index("_build_field")
    assert source.rindex('"completion.json"') > source.index("_artifact_manifest")


def test_run_refuses_any_existing_output_directory(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        assay.run(existing, ROOT)


def test_expected_key_sets_have_exact_unique_counts() -> None:
    keys = assay._expected_keys()  # noqa: SLF001
    expected = {
        "fields": 54,
        "cells": 48,
        "renderers": 144,
        "permutations": 576,
        "trajectories": 96,
        "checkpoints": 9_696,
        "timing": 288,
    }
    for name, count in expected.items():
        assert len(keys[name]) == count
        assert len(set(keys[name])) == count


def test_ledger_contract_requires_an_explicit_terminal_status() -> None:
    binding = "a" * 64
    expected = ["row0"]
    complete = [{"id": "row0", "binding_sha256": binding, "status": "ok"}]
    missing = [{"id": "row0", "binding_sha256": binding}]
    assert assay._ledger_contract(  # noqa: SLF001
        complete, key="id", expected=expected, binding_sha256=binding
    )
    assert not assay._ledger_contract(  # noqa: SLF001
        missing, key="id", expected=expected, binding_sha256=binding
    )


def test_json_safe_bytes_are_hashed_not_embedded() -> None:
    record = assay._json_safe(b"secret bytes")  # noqa: SLF001
    assert record["nbytes"] == 12
    assert len(record["sha256"]) == 64
    assert "secret" not in str(record)


def test_ledger_contract_fails_closed_on_duplicates_stale_binding_and_nonfinite() -> None:
    valid = {
        "id": "row0",
        "binding_sha256": "bound",
        "status": "ok",
        "metric": 1.0,
    }
    assert assay._ledger_contract(  # noqa: SLF001
        [valid], key="id", expected=["row0"], binding_sha256="bound"
    )
    duplicate = [valid, dict(valid)]
    assert not assay._ledger_contract(  # noqa: SLF001
        duplicate,
        key="id",
        expected=["row0", "row1"],
        binding_sha256="bound",
    )
    assert not assay._ledger_contract(  # noqa: SLF001
        [{**valid, "binding_sha256": "stale"}],
        key="id",
        expected=["row0"],
        binding_sha256="bound",
    )
    assert not assay._ledger_contract(  # noqa: SLF001
        [{**valid, "metric": float("nan")}],
        key="id",
        expected=["row0"],
        binding_sha256="bound",
    )


def test_artifact_manifest_detects_tampering_and_excludes_lifecycle(
    tmp_path: Path,
) -> None:
    (tmp_path / "raw.bin").write_bytes(b"raw")
    for name in (
        "artifact_manifest.json",
        "replay.json",
        "analysis.json",
        "completion.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    manifest = assay._artifact_manifest(tmp_path)  # noqa: SLF001
    assert set(manifest["files"]) == {"raw.bin"}
    assert assay._validate_artifact_manifest(tmp_path, manifest)  # noqa: SLF001
    (tmp_path / "raw.bin").write_bytes(b"tampered")
    assert not assay._validate_artifact_manifest(tmp_path, manifest)  # noqa: SLF001


def test_post_creation_exception_is_sealed_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "failed"

    def fail_after_creation(path: Path, _root: Path) -> dict[str, object]:
        path.mkdir()
        raise RuntimeError("injected failure")

    monkeypatch.setattr(assay, "_run_impl", fail_after_creation)
    result = assay.run(outdir, ROOT)
    assert result["decision"] == "invalid"
    completion = (outdir / "completion.json").read_text(encoding="utf-8")
    assert "injected failure" in (outdir / "failure.json").read_text(encoding="utf-8")
    assert '"complete": false' in completion
    assert {
        "failure.json",
        "artifact_manifest.json",
        "replay.json",
        "analysis.json",
        "completion.json",
    } <= {path.name for path in outdir.iterdir()}
