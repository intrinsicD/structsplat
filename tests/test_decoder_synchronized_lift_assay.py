from __future__ import annotations

import inspect
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest
import torch

from benchmarks import decoder_synchronized_lift_assay as assay


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_matrix_has_exact_stable_counts_and_disjoint_keys() -> None:
    fields = assay.field_specs()
    cells = assay.cell_specs()
    assert len(fields) == assay.EXPECTED_FIELDS == 60
    assert len(cells) == assay.EXPECTED_CELLS == 54
    assert len({row["field_id"] for row in fields}) == len(fields)
    assert len({row["cell_id"] for row in cells}) == len(cells)
    assert {row["count"] for row in fields} == {78, 80}
    assert {row["seed"] for row in fields} == {401, 409, 419}
    assert len(assay._expected_keys()["static"]) == 162  # noqa: SLF001
    assert len(assay._expected_keys()["permutations"]) == 486  # noqa: SLF001


def test_target_formulas_evaluate_on_noncanonical_extent() -> None:
    constant = assay.target_image("constant", height=7, width=9)
    assert constant.shape == (7, 9, 3)
    assert constant.dtype == np.float64
    assert np.array_equal(constant[0, 0], np.array((0.30, 0.52, 0.68)))

    points = np.array(((0.0, 0.0), (18.0, 16.0), (11.88, 5.76)))
    affine = assay.target_values("affine", points, height=17, width=19)
    assert np.allclose(affine[0], (0.42, 0.53, 0.15))
    assert np.allclose(affine[1], (0.50, 0.55, 0.69))
    occluded = assay.target_values("affine_occlusion", points, height=17, width=19)
    assert np.allclose(occluded[2] - affine[2], (0.16, -0.14, 0.12))
    assert "abs(sqrt(rho)-1)<=0.18" == assay.REGION_FORMULAS["affine_occlusion_edge"]


def test_dslr015_wrapper_prices_mode_without_a_tail_and_cold_decodes() -> None:
    arrays = assay._control_field_arrays()  # noqa: SLF001
    nw = assay.encode_stream("nw78", arrays)
    dsl = assay.encode_stream("dsl78", arrays)
    assert assay.STREAM_HEADER.size == 20
    assert assay.STREAM_CRC.size == 4
    assert len(nw) == len(dsl)
    nw_decoded = assay.decode_stream(nw)
    dsl_decoded = assay.decode_stream(dsl)
    assert nw_decoded["mode"] == 0
    assert dsl_decoded["mode"] == 1
    assert nw_decoded["inner"] == dsl_decoded["inner"]
    assert nw_decoded["lift_state"] is None
    assert dsl_decoded["lift_state"] is not None
    assert np.array_equal(nw_decoded["arrays"]["colors"], dsl_decoded["arrays"]["colors"])
    assert assay.encode_stream("nw78", nw_decoded["arrays"]) == nw
    assert assay.encode_stream("dsl78", dsl_decoded["arrays"]) == dsl


def test_normal_pair_contributors_and_unavailable_sentinel_are_explicit(
    tmp_path: Path,
) -> None:
    arrays = assay._control_field_arrays()  # noqa: SLF001
    nw_blob = assay.encode_stream("nw78", arrays)
    dsl_blob = assay.encode_stream("dsl78", arrays)
    nw = assay.decode_stream(nw_blob)
    dsl = assay.decode_stream(dsl_blob)
    nw_candidate, nw_reference, _, _ = assay._render_decoded(nw)  # noqa: SLF001
    dsl_candidate, dsl_reference, _, _ = assay._render_decoded(dsl)  # noqa: SLF001
    assert assay._contributor_hash(nw_candidate.contributors)  # noqa: SLF001
    assert assay._contributor_hash(nw_candidate.contributors) == assay._contributor_hash(  # noqa: SLF001
        dsl_candidate.contributors
    )
    assert assay._contributor_hash(nw_reference.contributors) == assay._contributor_hash(  # noqa: SLF001
        dsl_reference.contributors
    )
    nw_timing = assay._prepare_timing_state(nw)  # noqa: SLF001
    dsl_timing = assay._prepare_timing_state(dsl)  # noqa: SLF001
    assert nw_timing["gaussian_pixel_weight_evaluations"] > 0
    assert (
        nw_timing["gaussian_pixel_weight_evaluations"]
        == dsl_timing["gaussian_pixel_weight_evaluations"]
    )

    unavailable = {
        **dsl,
        "lift_state": None,
        "method_available": False,
        "method_error": {"error_type": "ValueError", "error": "fixture"},
    }
    cell = {
        "cell_id": "synthetic__unavailable__s0",
        "cohort": "synthetic",
        "target": "synthetic",
        "seed": 0,
    }
    row = assay._static_method_unavailable(  # noqa: SLF001
        cell=cell,
        arm="dsl78",
        payload=dsl_blob,
        decoded=unavailable,
        outdir=tmp_path,
        binding_sha256="bound",
    )
    assert row["status"] == "method_unavailable"
    assert row["decoded_geometry_sha256"]
    assert row["decoded_colors_sha256"]
    assert row["candidate_contributor_sha256"] is None
    assert row["reference_contributor_sha256"] is None


def test_cold_output_skips_ledger_only_contributor_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = assay.encode_stream("nw78", assay._control_field_arrays())  # noqa: SLF001
    decoded = assay.decode_stream(blob)
    calls = {"production_render": 0}

    def forbidden_accounting_enumeration(*args: object, **kwargs: object) -> None:
        raise AssertionError("cold timing must not enumerate solely for operation accounting")

    def fake_production_render(
        means: torch.Tensor,
        conics: torch.Tensor,
        colors: torch.Tensor,
        radii: torch.Tensor,
        height: int,
        width: int,
        **kwargs: object,
    ) -> torch.Tensor:
        del means, conics, colors, radii, kwargs
        calls["production_render"] += 1
        return torch.zeros((height, width, 3), dtype=torch.float32)

    monkeypatch.setattr(
        assay.render_core,
        "enumerate_contributors",
        forbidden_accounting_enumeration,
    )
    monkeypatch.setattr(assay, "production_render", fake_production_render)
    output = assay._candidate_output_only(decoded)  # noqa: SLF001
    assert output.shape == (assay.HEIGHT, assay.WIDTH, 3)
    assert calls == {"production_render": 1}


def test_wrapper_rejects_header_crc_length_trailing_and_bad_mode() -> None:
    blob = assay.encode_stream("dsl78", assay._control_field_arrays())  # noqa: SLF001
    bad = bytearray(blob)
    bad[0] ^= 1
    with pytest.raises(ValueError):
        assay.decode_stream(bytes(bad))
    bad = bytearray(blob)
    bad[9] = 3
    with pytest.raises(ValueError):
        assay.decode_stream(bytes(bad))
    bad = bytearray(blob)
    bad[-1] ^= 1
    with pytest.raises(ValueError):
        assay.decode_stream(bytes(bad))
    with pytest.raises(ValueError):
        assay.decode_stream(blob + b"trailing")
    bad = bytearray(blob[: -assay.STREAM_CRC.size])
    struct.pack_into("<I", bad, 12, len(blob))
    repaired = bytes(bad) + assay.STREAM_CRC.pack(zlib.crc32(bad) & 0xFFFFFFFF)
    with pytest.raises(ValueError):
        assay.decode_stream(repaired)


def test_decoder_api_is_blob_and_device_only() -> None:
    assert list(inspect.signature(assay.decode_stream).parameters) == ["blob", "device"]
    source = inspect.getsource(assay.decode_stream)
    for forbidden in ("target", "source", "beta", "confidence"):
        assert forbidden not in source


def test_target_free_plumbing_controls_pass() -> None:
    controls, arrays = assay.run_plumbing_controls()
    assert controls["valid"]
    assert all(controls["gates"].values())
    assert min(controls["step_alpha"]) <= 0.01
    assert controls["decoder_blob_device_only"]
    assert "valid_stream" in arrays


def test_binding_ignores_timestamp_but_binds_stable_environment() -> None:
    science = {"protocol": "x", "count": 78}
    environment = {
        "created_at": "first",
        "python": "3.x",
        "platform": "test",
        "numpy": "2",
        "torch": "2",
        "device": "cpu",
    }
    first = assay._binding_sha256(science, environment)  # noqa: SLF001
    second = assay._binding_sha256(  # noqa: SLF001
        science, {**environment, "created_at": "second"}
    )
    changed = assay._binding_sha256(  # noqa: SLF001
        science, {**environment, "device": "cuda"}
    )
    assert first == second
    assert first != changed


def test_preexisting_empty_directory_is_sealed_only_after_assay_acquires_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    untouched = tmp_path / "untouched"
    untouched.mkdir()

    def fail_before_acquisition(outdir: Path, root: Path) -> dict[str, object]:
        del outdir, root
        raise RuntimeError("before acquisition")

    monkeypatch.setattr(assay, "_run_impl", fail_before_acquisition)
    with pytest.raises(RuntimeError, match="before acquisition"):
        assay.run(untouched, ROOT)
    assert not any(untouched.iterdir())

    acquired = tmp_path / "acquired"
    acquired.mkdir()

    def fail_after_acquisition(outdir: Path, root: Path) -> dict[str, object]:
        del root
        assay._atomic_json(outdir / "config.json", {"binding_sha256": "fixture"})  # noqa: SLF001
        raise RuntimeError("after acquisition")

    monkeypatch.setattr(assay, "_run_impl", fail_after_acquisition)
    analysis = assay.run(acquired, ROOT)
    assert analysis["decision"] == "invalid"
    completion = json.loads((acquired / "completion.json").read_text(encoding="utf-8"))
    assert not completion["complete"]
    assert completion["decision"] == "invalid"


def test_ledger_contract_fails_closed() -> None:
    row = {
        "id": "row0",
        "binding_sha256": "bound",
        "status": "ok",
        "metric": 1.0,
    }
    assert assay._ledger_contract(  # noqa: SLF001
        [row], key="id", expected=["row0"], binding_sha256="bound"
    )
    assert not assay._ledger_contract(  # noqa: SLF001
        [row, dict(row)],
        key="id",
        expected=["row0", "row1"],
        binding_sha256="bound",
    )
    assert not assay._ledger_contract(  # noqa: SLF001
        [{**row, "binding_sha256": "stale"}],
        key="id",
        expected=["row0"],
        binding_sha256="bound",
    )
    assert not assay._ledger_contract(  # noqa: SLF001
        [{**row, "metric": float("nan")}],
        key="id",
        expected=["row0"],
        binding_sha256="bound",
    )


def test_artifact_manifest_excludes_lifecycle_and_detects_tampering(
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


def test_json_safe_hashes_bytes() -> None:
    record = assay._json_safe(b"do not embed")  # noqa: SLF001
    assert record["nbytes"] == 12
    assert len(record["sha256"]) == 64
    assert "embed" not in json.dumps(record)


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
    assert "injected failure" in (outdir / "failure.json").read_text(encoding="utf-8")
    completion = json.loads((outdir / "completion.json").read_text(encoding="utf-8"))
    assert not completion["complete"]
    assert completion["decision"] == "invalid"


def test_gradient_step_and_full_timing_scope_match_frozen_task() -> None:
    assert assay.GRADIENT_STEP == 2.0**-16
    assert assay.EXPECTED_TIMING_UNITS == assay.EXPECTED_CELLS == 54
    assert assay.TIMING_WARMUPS == 10
    assert assay.TIMING_REPETITIONS == 40
