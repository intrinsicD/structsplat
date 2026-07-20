from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks import marginal_cold_stream_rd as B
from benchmarks.perturb_recover_spectroscopy import _target_suite as fit020_target_suite
from structsplat.gaussians import GaussianField


def _field(seed: int = 7) -> GaussianField:
    rng = np.random.default_rng(seed)
    return GaussianField.from_numpy(
        rng.uniform(1.0, B.SIZE - 2.0, size=(B.START_COUNT, 2)).astype(np.float32),
        rng.uniform(1.0, 5.0, size=(B.START_COUNT, 2)).astype(np.float32),
        rng.uniform(-np.pi, np.pi, size=B.START_COUNT).astype(np.float32),
        rng.uniform(0.05, 0.95, size=(B.START_COUNT, 3)).astype(np.float32),
    )


def test_frozen_constants_precision_grid_and_protocol_hash():
    assert B.SIZE == 64
    assert B.START_COUNT == 64
    assert B.RECOVERY_STEPS == (0, 20)
    assert B.SEEDS == (0, 1)
    assert B.BASE_BITS == (12, 6, 6, 6)
    assert B.PRIMARY_CAP_INCREMENT == 16
    grid = B.precision_bit_grid()
    assert len(grid) == len(set(grid)) == 875
    assert grid[0] == (10, 4, 4, 4)
    assert grid[-1] == (16, 8, 8, 8)
    assert grid.count(B.BASE_BITS) == 1
    assert B.precision_grid_sha256() == (
        "c92e2fbb773e955b5c2b60a18592cbc9f012a6cc025a0f7d0578b90a49c42ab4"
    )
    manifest = B.protocol_manifest()
    assert manifest["target_manifest"]["manifest_sha256"] == B.TARGET_MANIFEST_SHA256
    assert manifest["protocol_sha256"] == (
        "137359fbe8447dad5e585d27f0b2b1fe58bc8ec89b0f31e92cd37c2fa543acf2"
    )
    assert manifest["inference"]["bootstrap_replicates"] == 10_000
    assert "range_policy" in manifest["qat"]
    assert "benchmarks/perturb_recover_spectroscopy.py" in B.SOURCE_PATHS


def test_target_suite_is_deterministic_complete_and_fit020_disjoint():
    suite = B.target_suite()
    assert len(suite) == 36
    assert {item["family"] for item in suite.values()} == set(B.TARGET_FAMILIES)
    assert sum(item["split"] == "development" for item in suite.values()) == 18
    assert sum(item["split"] == "confirmation" for item in suite.values()) == 18
    hashes = [item["pixel_sha256"] for item in suite.values()]
    assert len(set(hashes)) == 36
    for item in suite.values():
        image = item["image"]
        assert image.dtype == np.float32
        assert image.shape == (64, 64, 3)
        assert np.isfinite(image).all()
        assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
        assert item["pixel_sha256"] == B._tensor_sha256(image.copy())
    fit020_hashes = {
        B._tensor_sha256(item["image"]) for item in fit020_target_suite().values()
    }
    assert set(hashes).isdisjoint(fit020_hashes)
    assert B.target_manifest()["manifest_sha256"] == (
        "a1ea1ea5be41e36c3e4a8557d01ce721167545d680a6945522c935b832f60f0e"
    )


def test_action_bank_and_matched_actions_are_unique_no_opacity_and_nonmutating():
    parent = _field()
    target = torch.as_tensor(
        B.target_suite()["hard_disks_v0"]["image"], dtype=torch.float32
    )
    before = B._field_sha256(parent)
    rows, donor, metadata = B._action_bank(parent, target)
    assert len(rows) == B.ACTION_COUNT
    assert len(set(metadata["candidate_row_hashes"])) == B.ACTION_COUNT
    assert 0 <= donor < B.START_COUNT
    assert B._field_sha256(parent) == before
    assert all(row.n == 1 and row.opacities is None for row in rows)

    for row in rows:
        birth = B._birth_field(parent, row)
        replacement = B._replace_field(parent, row, donor)
        assert birth.n == 65 and birth.opacities is None
        assert replacement.n == 64 and replacement.opacities is None
        assert B._field_sha256(birth.subset(torch.tensor([64]))) == B._field_sha256(row)
        assert B._field_sha256(replacement.subset(torch.tensor([63]))) == B._field_sha256(row)
    assert B._field_sha256(parent) == before


def test_precision_configs_preserve_matched_ranges_and_transforms():
    field = _field()
    base = B._freeze_ranges(field, B._base_codec_config())
    changed = replace(
        base, bits_means=16, bits_scales=4, bits_rot=8, bits_colors=5
    )
    assert changed.color_lo == base.color_lo and changed.color_hi == base.color_hi
    assert changed.scale_lo == base.scale_lo and changed.scale_hi == base.scale_hi
    assert changed.morton_reorder and changed.color_delta
    assert changed.means_byte_planar and changed.rot_circular
    assert changed.stream_codec == "zlib" and changed.zlib_level == 9


def test_persisted_cold_stream_integrity_and_exact_components(tmp_path: Path):
    field = _field()
    target = torch.as_tensor(
        B.target_suite()["quadratic_patch_v0"]["image"], dtype=torch.float32
    )
    ccfg = B._freeze_ranges(field, B._base_codec_config())
    path = tmp_path / "stream.sspl"
    row = B._persist_and_score(field, target, ccfg, path, expected_n=64)
    blob = path.read_bytes()
    assert row["bytes"] == len(blob)
    assert row["stream_sha256"] == B._sha256_bytes(blob)
    assert set(row["components"]["streams"]) == {
        "means",
        "scales",
        "rotation",
        "colors",
    }
    assert row["components"]["header_bytes"] + sum(
        item["framed_bytes"] for item in row["components"]["streams"].values()
    ) == len(blob)
    assert not row["header"]["has_opacity"]
    assert np.isfinite([row["mse"], row["psnr"], row["ssim"], row["ms_ssim"]]).all()
    with pytest.raises(ValueError, match="trailing"):
        B._validate_blob(blob + b"x", expected_n=64, expected_bits=B.BASE_BITS)
    with pytest.raises(ValueError):
        B._validate_blob(blob[:-1], expected_n=64, expected_bits=B.BASE_BITS)


def _fake_row(
    key: str,
    branch: str,
    *,
    mse: float,
    byte_count: int,
    action: int | None = None,
    bits: tuple[int, int, int, int] = B.BASE_BITS,
) -> dict:
    n = 65 if branch == "birth" else 64
    psnr = -10.0 * np.log10(mse)
    streams = {
        name: {"framed_bytes": value}
        for name, value in (("means", 100), ("scales", 80), ("rotation", 50), ("colors", 120))
    }
    return {
        "row_key": key,
        "cell_key": "cell",
        "split": "development",
        "target": "hard_disks_v0",
        "family": "hard_disks",
        "variant": 0,
        "seed": 0,
        "recovery_steps": 20,
        "branch": branch,
        "action_index": action,
        "bits": list(bits),
        "n_gaussians": n,
        "bytes": byte_count,
        "bpp": 8.0 * byte_count / (64 * 64),
        "raw_bits": B._raw_bits(n, bits),
        "mse": mse,
        "psnr": float(psnr),
        "ssim": 0.9,
        "ms_ssim": 0.9,
        "stream_sha256": key * 4,
        "stream_path": f"streams/{key}.sspl",
        "components": {"header_bytes": 400, "streams": streams},
    }


def test_integer_cap_selection_excludes_overcap_and_never_divides():
    rows = [
        _fake_row("base", "precision", mse=0.04, byte_count=1000),
        _fake_row("prec", "precision", mse=0.03, byte_count=1015, bits=(12, 6, 7, 6)),
        _fake_row("rep", "replace", mse=0.029, byte_count=1008, action=0),
        _fake_row("birth0", "birth", mse=0.02, byte_count=1016, action=0),
        _fake_row("birth1", "birth", mse=0.001, byte_count=1017, action=1),
    ]
    selected = B._select_cell(rows, cap_increment=16)
    assert selected["byte_cap"] == 1016
    assert selected["best_birth"]["row_key"] == "birth0"
    assert selected["best_control"]["row_key"] == "rep"
    assert selected["exact_winner"]["row_key"] == "birth0"
    assert selected["birth_delta_bytes"] == 16
    assert selected["birth_advantage_mse"] == pytest.approx(0.009)


def test_family_bootstrap_uses_three_target_means_per_family_and_is_deterministic():
    rows = [
        {
            "target": f"{family}_v{variant}",
            "family": family,
            "birth_advantage_psnr": 0.2 + 0.01 * index,
        }
        for index, family in enumerate(B.TARGET_FAMILIES)
        for variant in B.DEVELOPMENT_VARIANTS
    ]
    first = B._family_stratified_bootstrap(rows)
    second = B._family_stratified_bootstrap(rows)
    assert first == second
    assert first["replicates"] == 10_000
    assert first["ci95"][0] > 0.0
    with pytest.raises(RuntimeError, match="three target means"):
        B._family_stratified_bootstrap(rows[:-1])


def test_confirmation_requires_exact_final_gate_schema(tmp_path: Path):
    path = tmp_path / "final.json"
    assert not B._confirmation_authorized(path)
    payload = {
        "schema": "comp006-final-v1",
        "split": "development",
        "decision": "authorize_confirmation",
        "protocol_sha256": B.protocol_manifest()["protocol_sha256"],
        "source_combined_sha256": B._source_provenance(
            Path(B.__file__).resolve().parents[1]
        )["combined_sha256"],
        "gates": {key: True for key in B.FINAL_GATE_KEYS},
        "checks": {"streams": True, "analysis": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert B._confirmation_authorized(path)
    payload["gates"].pop("replay_integrity")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not B._confirmation_authorized(path)


def test_replay_normalization_ignores_only_timing():
    left = {"row_key": "a", "bytes": 10, "timing": {"encode_seconds": 1.0}}
    right = {"row_key": "a", "bytes": 10, "timing": {"encode_seconds": 9.0}}
    assert B._without_timing(left) == B._without_timing(right)
    right["bytes"] = 11
    assert B._without_timing(left) != B._without_timing(right)


def test_stream_validation_recomputes_key_and_cross_binds_cell(tmp_path: Path):
    field = _field()
    target_item = B.target_suite()["hard_disks_v0"]
    target = torch.as_tensor(target_item["image"], dtype=torch.float32)
    cell = {
        "protocol_sha256": "protocol",
        "source_combined_sha256": "source",
        "cell_key": "cell",
        "split": "development",
        "target": "hard_disks_v0",
        "family": "hard_disks",
        "variant": 0,
        "target_pixel_sha256": target_item["pixel_sha256"],
        "seed": 0,
        "parent_field_sha256": B._field_sha256(field),
        "parent": {"stream_sha256": "parent-stream"},
        "action_bank": {"candidate_action_sha256": "actions"},
    }
    ccfg = B._freeze_ranges(field, B._base_codec_config())
    row = B._stream_row(
        outdir=tmp_path,
        cell=cell,
        target=target,
        recovered_field=field,
        ccfg=ccfg,
        recovery_steps=0,
        recovery_seconds=0.0,
        action_seconds=0.0,
        branch="precision",
        action_index=None,
        action_start_sha256=cell["parent_field_sha256"],
    )
    bindings = {
        "protocol_sha256": "protocol",
        "source_combined_sha256": "source",
    }
    B._validate_existing_stream_row(tmp_path, row, bindings, cell=cell)

    wrong_target = dict(row)
    wrong_target["target"] = "hard_disks_v2"
    with pytest.raises(RuntimeError, match="row/cell binding drift"):
        B._validate_existing_stream_row(tmp_path, wrong_target, bindings, cell=cell)

    wrong_key = dict(row)
    wrong_key["row_key"] = "not-the-derived-key"
    with pytest.raises(RuntimeError, match="row key drift"):
        B._validate_existing_stream_row(tmp_path, wrong_key, bindings, cell=cell)


def test_resume_rejects_environment_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    protocol = {"protocol_sha256": "protocol", "target_manifest": {"targets": []}}
    source = {"combined_sha256": "source", "files": {}}
    environment = {"python": "same", "packages": {"torch": "test"}}
    monkeypatch.setattr(B, "protocol_manifest", lambda: protocol)
    monkeypatch.setattr(B, "_source_provenance", lambda _root: source)
    monkeypatch.setattr(B, "_write_source_snapshot", lambda *_args: None)
    monkeypatch.setattr(B, "_environment_provenance", lambda: environment)
    B._prepare_run_snapshot(tmp_path, "development", tmp_path)
    B._prepare_run_snapshot(tmp_path, "development", tmp_path)

    monkeypatch.setattr(
        B,
        "_environment_provenance",
        lambda: {"python": "changed", "packages": {"torch": "test"}},
    )
    with pytest.raises(RuntimeError, match="environment changed"):
        B._prepare_run_snapshot(tmp_path, "development", tmp_path)


def test_verify_replay_binds_environment_and_writes_final_overview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    primary = tmp_path / "primary"
    replay = tmp_path / "replay"
    primary.mkdir()
    replay.mkdir()
    config = {
        "split": "development",
        "protocol": {"protocol_sha256": "protocol"},
        "source": {"combined_sha256": "source"},
        "environment": {"python": "same", "packages": {}},
    }
    for path in (primary, replay):
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    scientific_gates = {
        "feasible_cells_at_least_90pct": True,
        "mean_advantage_at_least_0_15_db": False,
        "bootstrap_lower_bound_above_zero": False,
        "median_advantage_at_least_0_10_db": False,
        "at_least_four_positive_families": False,
        "integrity": True,
    }
    primary_metrics = {
        "mean_birth_advantage_psnr": -1.0,
        "median_birth_advantage_psnr": -0.9,
        "bootstrap": {"ci95": [-1.2, -0.8]},
        "positive_family_means": 0,
        "feasible_cells": 36,
        "total_cells": 36,
    }
    analysis = {"gates": scientific_gates, "primary": primary_metrics}
    monkeypatch.setattr(B, "analyze", lambda *_args, **_kwargs: analysis)
    monkeypatch.setattr(B, "_journal_hash", lambda *_args, **_kwargs: "same")

    final = B.verify_replay(primary, replay)
    assert final["decision"] == "stop"
    assert final["checks"]["environment_sha256"] is True
    markdown = (primary / "summary.md").read_text(encoding="utf-8")
    assert "Final decision: **stop**" in markdown
    assert "Exact replay integrity: `pass`" in markdown
    assert "stop_pending_replay" not in markdown
    assert "Final decision: **stop**" in (primary / "index.html").read_text(
        encoding="utf-8"
    )

    changed = {**config, "environment": {"python": "changed", "packages": {}}}
    (replay / "config.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="replay mismatch"):
        B.verify_replay(primary, replay)
