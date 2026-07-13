from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from benchmarks import fair_density_control_compare as F
from structsplat.config import FitConfig, StructureTensorConfig
from structsplat.gaussians import GaussianField


def test_artifact_source_id_binds_canonical_path_and_content(tmp_path: Path):
    left_path = tmp_path / "left" / "foo.png"
    right_path = tmp_path / "right" / "foo.png"
    left_path.parent.mkdir()
    right_path.parent.mkdir()
    left_path.write_bytes(b"same bytes")
    right_path.write_bytes(b"same bytes")
    alias = tmp_path / "foo-alias.png"
    alias.symlink_to(left_path)
    content_hash = "a" * 64
    left = F._artifact_source_id(left_path, content_hash)
    right = F._artifact_source_id(right_path, content_hash)

    assert left.startswith("foo_")
    assert right.startswith("foo_")
    assert left != right
    assert F._artifact_source_id(alias, content_hash) == left


def test_growth_fit_cfg_reaches_final_cap_with_shared_schedule():
    base = FitConfig(iters=100)
    cfg = F._growth_fit_cfg(base, final_budget=1000, start_budget=500, split_mode="residual_add", growth_waves=4)
    assert cfg.max_gaussians == 1000
    assert cfg.split_mode == "residual_add"
    assert cfg.split_every == 20
    assert cfg.split_count == 125


def test_representation_layout_counts_stored_extra_attributes() -> None:
    field = GaussianField.from_numpy(
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        opacities=np.zeros(2, dtype=np.float32),
        color_grads=np.zeros((2, 2, 3), dtype=np.float32),
    )

    layout = F._representation_layout(field)

    assert layout["representation_payload_scalars_per_gaussian"] == 15
    assert layout["representation_extra_scalars_per_gaussian"] == 7
    assert layout["representation_actual_bytes_per_gaussian"] == 60


def test_fixed_storage_growth_finishes_before_convergence_gate() -> None:
    base = FitConfig(
        iters=10_000,
        early_stop_patience=6,
        early_stop_min_iters=6_500,
        loss_target_downsample=2,
        loss_target_full_frac=0.1,
    )
    growth = F._growth_fit_cfg(
        base,
        final_budget=5_376,
        start_budget=2_688,
        split_mode="residual_tensor_add",
        growth_waves=5,
    )
    storage = F._storage_growth_fit_cfg(growth, 5, enabled=True)

    assert storage.split_every == 1_083
    assert storage.split_schedule_stops_at_max is True
    assert storage.split_every - 1 >= 1_000
    assert storage.split_every * 5 - 1 < storage.early_stop_min_iters


def test_best_default_is_first_default_and_pinned(tmp_path: Path):
    assert F.DEFAULT_METHODS[0] == F.BEST_DEFAULT_METHOD
    assert F.METHOD_LABELS[F.BEST_DEFAULT_METHOD] == "SS best default"

    img = np.full((24, 24, 3), 0.5, dtype=np.float32)
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)
    base = FitConfig(iters=20, pixel_loss="l2", ssim_weight=0.9)

    field, cfg, _seconds, actual_start, meta = F._build_method(
        method=F.BEST_DEFAULT_METHOD,
        img=img,
        image_path=image_path,
        final_budget=200,
        start_budget=100,
        seed=3,
        base_fit=base,
        scfg=StructureTensorConfig(),
        growth_waves=4,
        device="cpu",
        feature_cap=99.0,
        feature_cap_reference_side=24.0,
    )

    assert field.n == 100
    assert actual_start == 100
    assert cfg.pixel_loss == "l1"
    assert cfg.ssim_weight == 0.3
    assert cfg.split_mode == "residual_tensor_add"
    assert cfg.split_count == 20
    assert cfg.split_every == 3
    assert meta["configured_growth_waves"] == F.BEST_DEFAULT_GROWTH_WAVES
    assert meta["variant_base"] == F.BEST_DEFAULT_BASE_METHOD
    assert meta["variant_overrides"]["pinned_best_default"] is True
    assert meta["scale_cap_input"] == F.BEST_DEFAULT_FEATURE_CAP
    assert meta["scale_cap_reference_side"] == F.DEFAULT_FEATURE_CAP_REFERENCE_SIDE
    assert abs(meta["feature_cap_px"] - 1.8) < 1e-6


def test_best_diff_reduction_variants_apply_fit_overrides(tmp_path: Path):
    img = np.full((24, 24, 3), 0.5, dtype=np.float32)
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)
    base = FitConfig(
        iters=50,
        pixel_loss="l2",
        ssim_weight=0.9,
        loss_weighting="tensor",
        geometry_loss_weight=0.9,
    )

    expected = {
        "structsplat_best_ssim010": ("ssim_weight", 0.1),
        "structsplat_best_l1_only": ("ssim_weight", 0.0),
        "structsplat_best_charbonnier": ("pixel_loss", "charbonnier"),
        "structsplat_best_tensor_loss": ("loss_weighting", "tensor"),
        F.BEST_COSINE_METHOD: ("lr_schedule", "cosine"),
        F.BEST_COSINE_TAIL_METHOD: ("lr_schedule", "cosine"),
        F.BEST_CHECKPOINT_METHOD: ("checkpoint_policy", "best_psnr_final_count"),
        F.BEST_CHECKPOINT_LOWPASS_METHOD: ("loss_target_downsample", 2),
        "structsplat_best_gcr015": ("geometry_loss_weight", 0.015),
        "structsplat_best_gcr015_e2": ("geometry_loss_weight", 0.015),
        "structsplat_best_gcr015_e4": ("geometry_loss_weight", 0.015),
        "structsplat_best_gcr030": ("geometry_loss_weight", 0.030),
        "structsplat_best_gcr060": ("geometry_loss_weight", 0.060),
        F.BEST_CAF_METHOD: ("covariance_filter_mode", "generation_density"),
        "structsplat_best_color_final": ("color_solve_schedule", "final"),
    }
    for method, (attr, value) in expected.items():
        _field, cfg, _seconds, _actual_start, meta = F._build_method(
            method=method,
            img=img,
            image_path=image_path,
            final_budget=200,
            start_budget=100,
            seed=3,
            base_fit=base,
            scfg=StructureTensorConfig(),
            growth_waves=4,
            device="cpu",
        )
        assert getattr(cfg, attr) == value
        if method != "structsplat_best_charbonnier":
            assert cfg.pixel_loss == "l1"
        if method not in {
            "structsplat_best_ssim010",
            "structsplat_best_l1_only",
            "structsplat_best_charbonnier",
        }:
            assert cfg.ssim_weight == 0.3
        if method != "structsplat_best_tensor_loss":
            assert cfg.loss_weighting == "none"
        assert meta["configured_growth_waves"] == F.BEST_DEFAULT_GROWTH_WAVES
        assert meta["scale_cap_input"] == F.BEST_DEFAULT_FEATURE_CAP

    _field, cfg, _seconds, _actual_start, meta = F._build_method(
        method=F.BEST_COSINE_TAIL_METHOD,
        img=img,
        image_path=image_path,
        final_budget=200,
        start_budget=100,
        seed=3,
        base_fit=base,
        scfg=StructureTensorConfig(),
        growth_waves=4,
        device="cpu",
    )
    expected_start = F.BEST_DEFAULT_GROWTH_WAVES / (F.BEST_DEFAULT_GROWTH_WAVES + 1)
    assert cfg.lr_cosine_start_frac == expected_start
    assert meta["variant_overrides"]["lr_cosine_start_frac"] == expected_start

    for method, every in (
        ("structsplat_best_gcr015_e2", 2),
        ("structsplat_best_gcr015_e4", 4),
    ):
        _field, cfg, _seconds, _actual_start, meta = F._build_method(
            method=method,
            img=img,
            image_path=image_path,
            final_budget=200,
            start_budget=100,
            seed=3,
            base_fit=base,
            scfg=StructureTensorConfig(),
            growth_waves=4,
            device="cpu",
        )
        assert cfg.geometry_loss_every == every
        assert meta["variant_overrides"]["geometry_loss_every"] == every

    for method, alpha in F.BEST_CAF_METHODS.items():
        _field, cfg, _seconds, _actual_start, meta = F._build_method(
            method=method,
            img=img,
            image_path=image_path,
            final_budget=200,
            start_budget=100,
            seed=3,
            base_fit=base,
            scfg=StructureTensorConfig(),
            growth_waves=4,
            device="cpu",
        )
        assert cfg.covariance_filter_mode == "generation_density"
        assert cfg.covariance_filter_alpha == alpha
        assert meta["variant_overrides"]["covariance_filter_alpha"] == alpha

    _field, cfg, _seconds, _actual_start, meta = F._build_method(
        method="structsplat_best_adaptive_1p5x",
        img=img,
        image_path=image_path,
        final_budget=200,
        start_budget=100,
        seed=3,
        base_fit=base,
        scfg=StructureTensorConfig(),
        growth_waves=4,
        device="cpu",
    )
    assert cfg.adaptive_count is True
    assert cfg.max_gaussians == 304
    assert meta["variant_overrides"]["adaptive_cap_multiplier"] == F.ADAPTIVE_EXTRA_CAP_MULT

    storage_cfg, storage_meta = F._apply_method_fit_overrides(
        "structsplat_best_adaptive_1p5x",
        F._growth_fit_cfg(base, 5_376, 2_688, "residual_tensor_add", 5),
        5_376,
        storage_gaussian_cap=5_376,
    )
    assert storage_cfg.max_gaussians == 5_376
    assert storage_cfg.adaptive_continue_after_stop is True
    assert storage_meta["adaptive_cap_multiplier"] == 1.0


def test_lowpass_candidate_differs_from_checkpoint_control_in_exactly_two_fields(tmp_path):
    img = np.full((24, 24, 3), 0.5, dtype=np.float32)
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)
    base = FitConfig(iters=50, pixel_loss="l2", ssim_weight=0.9)

    built = {}
    metadata = {}
    for method in (F.BEST_CHECKPOINT_METHOD, F.BEST_CHECKPOINT_LOWPASS_METHOD):
        _field, cfg, _seconds, _actual_start, meta = F._build_method(
            method=method,
            img=img,
            image_path=image_path,
            final_budget=200,
            start_budget=100,
            seed=3,
            base_fit=base,
            scfg=StructureTensorConfig(),
            growth_waves=4,
            device="cpu",
        )
        built[method] = cfg
        metadata[method] = meta

    control = built[F.BEST_CHECKPOINT_METHOD]
    candidate = built[F.BEST_CHECKPOINT_LOWPASS_METHOD]
    differences = {
        name: (getattr(control, name), getattr(candidate, name))
        for name in control.__dict__
        if getattr(control, name) != getattr(candidate, name)
    }
    assert differences == {
        "loss_target_downsample": (1, 2),
        "loss_target_full_frac": (0.0, 0.1),
    }
    assert candidate.checkpoint_policy == "best_psnr_final_count"
    assert metadata[F.BEST_CHECKPOINT_LOWPASS_METHOD]["variant_overrides"][
        "loss_target_downsample"
    ] == 2
    assert metadata[F.BEST_CHECKPOINT_LOWPASS_METHOD]["variant_overrides"][
        "loss_target_full_frac"
    ] == pytest.approx(0.1)


def test_fair_method_registry_metadata_is_complete():
    registered = set(F.DEFAULT_METHODS)
    assert set(F.METHOD_LABELS) == registered
    assert set(F.METHOD_NOTES) == registered
    assert set(F.METHOD_TRACKS) == registered
    assert F.BEST_CHECKPOINT_LOWPASS_METHOD in F.STRUCTSPLAT_INIT
    assert F.BEST_CHECKPOINT_LOWPASS_METHOD in F.STRUCTSPLAT_SPLIT_MODE


def test_method_tracks_keep_growth_methods_at_same_start_budget(tmp_path):
    args = SimpleNamespace(
        iters=10,
        target_psnr=35.0,
        target_psnrs=[22.0],
        render_chunk=64,
        renderer="normalized",
        support_fade=False,
        pixel_loss="l1",
        ssim_weight=0.3,
    )
    base = F._base_fit(args)
    start = F._start_budget(2000, 0.5)
    assert start == 1000
    growth_methods = [
        "gaussianimage_plus_residual",
        "image_gs_residual",
        "structsplat_onedge_residual",
        "structsplat_onedge_residual_relocate",
        "structsplat_onedge_residual_featurecap",
        "structsplat_onedge_residual_feature_rel",
        "structsplat_onedge_tensor",
        "structsplat_onedge_tensor_featurecap",
        "structsplat_onedge_tensor_feature_rel",
        "structsplat_quadtree_wse_tensor",
        "structsplat_quadtree_wse_tensor_feature_rel",
    ]
    for method in growth_methods:
        split_mode = F.STRUCTSPLAT_SPLIT_MODE.get(method, "residual_add")
        cfg = F._growth_fit_cfg(base, 2000, start, split_mode, growth_waves=4)
        assert cfg.max_gaussians == 2000
        assert cfg.split_every == 2
        assert cfg.split_count == 250


def test_checkpoint_audit_is_same_trajectory_and_same_count():
    row = {
        "status": "ok",
        "checkpoint_policy": "best_psnr_final_count",
        "image": "x",
        "source_path": "x.png",
        "final_budget": 640,
        "seed": 0,
        "method": F.BEST_CHECKPOINT_METHOD,
        "trajectory_terminal_iter": 5000,
        "selected_iter": 4251,
        "trajectory_terminal_n_gaussians": 640,
        "selected_n_gaussians": 640,
        "selected_from_checkpoint": True,
        "trajectory_terminal_psnr": 24.0,
        "selected_fit_psnr": 25.0,
        "trajectory_terminal_ssim": 0.90,
        "selected_fit_ssim": 0.91,
        "trajectory_terminal_ms_ssim": 0.92,
        "selected_fit_ms_ssim": 0.94,
        "trajectory_terminal_lpips": 0.2,
        "selected_fit_lpips": 0.15,
    }
    audit = F._checkpoint_selection_audit([row])
    assert len(audit) == 1
    assert audit[0]["gain_psnr"] == 1.0
    assert audit[0]["gain_ms_ssim"] == pytest.approx(0.02)
    assert audit[0]["gain_lpips"] == pytest.approx(0.05)
    summary = F._checkpoint_selection_summary(audit)
    assert len(summary) == 2
    assert summary[0]["scope"] == "all"
    assert summary[0]["runs"] == 1
    assert summary[0]["images"] == 1
    assert summary[0]["earlier_selected"] == 1
    assert summary[0]["gain_psnr"] == 1.0
    assert summary[0]["ci_low_psnr"] == 1.0
    assert summary[1]["scope"] == "final_budget"
    assert summary[1]["final_budget"] == 640

    row["selected_n_gaussians"] = 639
    assert F._checkpoint_selection_audit([row]) == []


def test_lowpass_audit_pairs_candidate_with_checkpoint_control():
    rows = []
    control_cfg = FitConfig(
        iters=500,
        checkpoint_policy="best_psnr_final_count",
        color_solve_schedule="none",
    )
    candidate_cfg = FitConfig(**{
        **control_cfg.__dict__,
        "loss_target_downsample": 2,
        "loss_target_full_frac": 0.1,
    })
    for image in ("a", "b"):
        for seed in (0, 1):
            common = {
                "status": "ok",
                "image": image,
                "source_path": f"/{image}.png",
                "target_pixel_sha256": f"pixels-{image}",
                "max_side": 160,
                "height": 120,
                "width": 160,
                "final_budget": 640,
                "start_budget": 320,
                "start_gaussians": 320,
                "start_fraction": 0.5,
                "growth_waves": 5,
                "seed": seed,
                "iters": 500,
                "renderer": "cuda",
                "support_fade": False,
                "run_protocol_sha256": "same-protocol",
                "selected_iter": 450,
                "checkpoint_policy": "best_psnr_final_count",
                "n_gaussians": 640,
                "trajectory_terminal_n_gaussians": 640,
                "selected_n_gaussians": 640,
            }
            rows.append({
                **common,
                "method": F.BEST_CHECKPOINT_METHOD,
                "fit_config": dict(control_cfg.__dict__),
                "psnr": 30.0,
                "ms_ssim": 0.90,
                "auc_psnr": 24.0,
                "fit_seconds": 2.0,
                "total_seconds": 2.1,
                "lpips": 0.20,
                "trajectory_terminal_psnr": 29.0,
                "trajectory_terminal_ssim": 0.88,
                "trajectory_terminal_ms_ssim": 0.89,
                "trajectory_terminal_lpips": 0.21,
            })
            rows.append({
                **common,
                "method": F.BEST_CHECKPOINT_LOWPASS_METHOD,
                "fit_config": dict(candidate_cfg.__dict__),
                "psnr": 31.0,
                "ms_ssim": 0.91,
                "auc_psnr": 24.5,
                "fit_seconds": 2.1,
                "total_seconds": 2.2,
                "lpips": 0.18,
                "trajectory_terminal_psnr": 29.5,
                "trajectory_terminal_ssim": 0.89,
                "trajectory_terminal_ms_ssim": 0.90,
                "trajectory_terminal_lpips": 0.19,
            })

    pairs, summary = F._paired_method_audit(
        rows,
        baseline_method=F.BEST_CHECKPOINT_METHOD,
        candidate_method=F.BEST_CHECKPOINT_LOWPASS_METHOD,
    )

    assert len(pairs) == 4
    assert all(row["gain_psnr"] == 1.0 for row in pairs)
    assert all(row["gain_lpips"] == pytest.approx(0.02) for row in pairs)
    assert summary is not None
    assert summary["pairs"] == 4
    assert summary["paired_images"] == 2
    assert summary["gain_psnr"] == 1.0
    assert summary["gain_ms_ssim"] == pytest.approx(0.01)
    assert summary["gain_fit_seconds"] == pytest.approx(-0.1)
    assert summary["gain_lpips"] == pytest.approx(0.02)
    assert summary["gain_terminal_psnr"] == pytest.approx(0.5)
    assert summary["gain_terminal_ms_ssim"] == pytest.approx(0.01)
    assert summary["gain_terminal_lpips"] == pytest.approx(0.02)

    bad_rows = [dict(row) for row in rows]
    bad_candidate = next(
        row for row in bad_rows if row["method"] == F.BEST_CHECKPOINT_LOWPASS_METHOD
    )
    bad_candidate["fit_config"] = {
        **bad_candidate["fit_config"],
        "ssim_weight": 0.2,
    }
    with pytest.raises(ValueError, match="fit configs differ outside"):
        F._paired_method_audit(
            bad_rows,
            baseline_method=F.BEST_CHECKPOINT_METHOD,
            candidate_method=F.BEST_CHECKPOINT_LOWPASS_METHOD,
        )


def test_base_fit_and_cell_key_include_support_fade_axis():
    args = SimpleNamespace(
        iters=10,
        target_psnr=35.0,
        target_psnrs=[22.0],
        render_chunk=64,
        renderer="normalized",
        support_fade=True,
        pixel_loss="l1",
        ssim_weight=0.3,
    )
    cfg = F._base_fit(args)
    assert cfg.support_fade is True
    assert cfg.color_solve_every is None
    assert cfg.color_solve_schedule == "none"

    row = {
        "image": "x",
        "source_path": "x.png",
        "max_side": 64,
        "final_budget": 100,
        "start_budget": 50,
        "start_fraction": 0.5,
        "growth_waves": 4,
        "seed": 0,
        "method": "structsplat_onedge_residual",
        "iters": 10,
        "renderer": "normalized",
        "pixel_loss": "l1",
        "ssim_weight": 0.3,
    }
    fade_off = F._cell_key(row)
    fade_on = F._cell_key({**row, "support_fade": True})
    assert fade_off != fade_on
    assert fade_off[11] is False
    assert fade_on[11] is True
    assert F._cell_key(row) != F._cell_key({
        **row,
        "loss_target_downsample": 2,
        "loss_target_full_frac": 0.1,
    })
    assert F._cell_key({
        **row,
        "loss_target_downsample": 2,
        "loss_target_full_frac": 0.1,
    }) != F._cell_key({
        **row,
        "loss_target_downsample": 2,
        "loss_target_full_frac": 0.2,
    })


def test_promotion_pair_key_requires_identical_input_provenance_and_protocol_axes():
    base = {
        "source_path": "x.png",
        "source_sha256": "source-a",
        "target_pixel_sha256": "pixels-a",
        "max_side": 64,
        "final_budget": 100,
        "start_budget": 50,
        "start_fraction": 0.5,
        "growth_waves": 5,
        "seed": 0,
        "iters": 10,
        "renderer": "normalized",
        "support_fade": False,
    }
    key = F._promotion_pair_key(base)
    for name, value in (
        ("source_sha256", "source-b"),
        ("target_pixel_sha256", "pixels-b"),
        ("start_fraction", 0.25),
        ("growth_waves", 4),
        ("support_fade", True),
    ):
        assert F._promotion_pair_key({**base, name: value}) != key


def test_run_protocol_fingerprint_covers_scientific_axes_and_environment():
    protocol = {
        "flat_frac": 0.02,
        "corner_frac": 0.15,
        "render_chunk": 512,
        "target_psnrs": [22.0, 24.0],
        "lpips": False,
        "relocate_fraction": 0.25,
        "relocate_downsample": 4,
    }
    kwargs = {
        "device": "cuda",
        "versions": {"torch": "1"},
        "repo_state": {"commit": "a", "tracked_diff_sha256": "diff-a"},
        "source_fingerprint": {"fit_source_sha256": "fit-a"},
    }
    fingerprint = F._run_protocol_sha256(protocol, **kwargs)
    for name, value in (
        ("flat_frac", 0.03),
        ("corner_frac", 0.20),
        ("render_chunk", 256),
        ("target_psnrs", [22.0, 26.0]),
        ("lpips", True),
        ("relocate_fraction", 0.5),
        ("relocate_downsample", 2),
    ):
        assert F._run_protocol_sha256({**protocol, name: value}, **kwargs) != fingerprint
    assert F._run_protocol_sha256(protocol, **{**kwargs, "device": "cpu"}) != fingerprint
    assert F._run_protocol_sha256(
        protocol,
        **{**kwargs, "versions": {"torch": "2"}},
    ) != fingerprint


def test_write_outputs_compacts_resume_journal_to_selected_rows(tmp_path: Path, monkeypatch):
    stale = {"status": "ok", "method": "stale", "source_sha256": "old"}
    current = {"status": "ok", "method": "current", "source_sha256": "new"}
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(stale) + "\n" + json.dumps(current) + "\n",
        encoding="utf-8",
    )
    optional_outputs = (
        "checkpoint_selection.csv",
        "checkpoint_selection_summary.csv",
        "lowpass_vs_checkpoint.csv",
        "lowpass_vs_checkpoint_summary.csv",
    )
    for name in optional_outputs:
        (tmp_path / name).write_text("stale\n", encoding="utf-8")
    for name in (
        "write_json",
        "write_csv",
        "_write_storage_tables",
        "_write_default_dominance",
        "_write_convergence_tables",
        "_write_summary",
        "_write_plots",
        "_write_grids",
        "_write_index",
    ):
        monkeypatch.setattr(F, name, lambda *args, **kwargs: None)

    F._write_outputs([current], tmp_path, [F.BEST_DEFAULT_METHOD])

    compacted = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert compacted == [current]
    assert all(not (tmp_path / name).exists() for name in optional_outputs)


def test_cached_row_rejects_same_size_reconstruction_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    target = tmp_path / "target.png"
    recon = tmp_path / "recon.png"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    recon.write_bytes(b"before")
    row = {
        "source_path": str(source),
        "source_file_bytes": source.stat().st_size,
        "source_sha256": F._file_sha256(source),
        "target_path": str(target),
        "target_file_bytes": target.stat().st_size,
        "target_file_sha256": F._file_sha256(target),
        "reconstruction_path": str(recon),
        "reconstruction_file_bytes": recon.stat().st_size,
        "reconstruction_sha256": F._file_sha256(recon),
    }
    assert F._cached_row_artifacts_valid(row)

    recon.write_bytes(b"after!")
    assert recon.stat().st_size == row["reconstruction_file_bytes"]
    assert not F._cached_row_artifacts_valid(row)


def test_output_aligned_history_uses_scored_reconstruction_endpoint() -> None:
    history, previous = F._output_aligned_history(
        {
            "iter": [0, 9],
            "psnr": [20.0, 25.0],
            "elapsed": [0.1, 1.0],
            "n_gaussians": [50, 100],
        },
        terminal_iter=9,
        terminal_psnr=30.0,
        terminal_elapsed=1.2,
        terminal_gaussians=100,
    )

    assert previous == 25.0
    assert history["iter"] == [0, 9]
    assert history["psnr"] == [20.0, 30.0]
    assert history["elapsed"][-1] == 1.2


def test_repo_growth_methods_honor_non_half_start_budget(tmp_path: Path):
    img = np.full((24, 24, 3), 0.5, dtype=np.float32)
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)
    base = FitConfig(iters=20)
    start_budget = 60

    for method in ["gaussianimage_plus_residual", "image_gs_residual"]:
        field, cfg, _seconds, actual_start, meta = F._build_method(
            method=method,
            img=img,
            image_path=image_path,
            final_budget=200,
            start_budget=start_budget,
            seed=3,
            base_fit=base,
            scfg=StructureTensorConfig(),
            growth_waves=4,
            device="cpu",
        )
        assert field.n == start_budget
        assert actual_start == start_budget
        assert meta["init_config"]["num_gaussians"] == start_budget
        assert cfg.split_count == 35


def test_storage_lane_pads_instant_gi_underfill_to_exact_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)
    native = F.build_field(
        img,
        F.InitConfig(strategy="random", num_gaussians=12, seed=4),
        StructureTensorConfig(),
        device="cpu",
    )

    def fake_analogue(*args, **kwargs):
        return native, FitConfig(iters=5), 0.1, native.n

    monkeypatch.setattr(F, "build_comparison_analogue", fake_analogue)
    field, _cfg, _seconds, actual_start, meta = F._build_method(
        method="instant_gi_quadtree_fixed",
        img=img,
        image_path=image_path,
        final_budget=20,
        start_budget=10,
        seed=3,
        base_fit=FitConfig(iters=5),
        scfg=StructureTensorConfig(),
        growth_waves=4,
        device="cpu",
        storage_gaussian_cap=20,
    )

    assert field.n == 20
    assert actual_start == 20
    assert meta["instant_gi_native_gaussians"] == 12
    assert meta["storage_fill_gaussians"] == 8
    assert meta["storage_fill_rule"] == "seeded_random_target_samples"


def test_relocation_method_uses_split_scheduled_coarse_residual(tmp_path: Path):
    img = np.full((24, 24, 3), 0.5, dtype=np.float32)
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)

    field, cfg, _seconds, actual_start, meta = F._build_method(
        method="structsplat_onedge_residual_relocate",
        img=img,
        image_path=image_path,
        final_budget=200,
        start_budget=100,
        seed=3,
        base_fit=FitConfig(iters=20),
        scfg=StructureTensorConfig(),
        growth_waves=4,
        device="cpu",
        relocate_fraction=0.25,
        relocate_downsample=4,
    )

    assert field.n == 100
    assert actual_start == 100
    assert cfg.split_mode == "residual_add"
    assert cfg.split_every == 4
    assert cfg.split_count == 25
    assert cfg.relocate_at_split is True
    assert cfg.relocate_every is None
    assert cfg.relocate_count == 7
    assert cfg.relocate_residual_downsample == 4
    assert meta["growth_rule"] == "residual_add+relocate"
    assert meta["relocate_rule"] == "at_split"
    assert meta["relocate_count_per_event"] == 7


def test_featurecap_method_sets_init_cap_and_growth(tmp_path: Path):
    img = np.full((24, 24, 3), 0.5, dtype=np.float32)
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)

    field, cfg, _seconds, actual_start, meta = F._build_method(
        method="structsplat_quadtree_wse_tensor_featurecap",
        img=img,
        image_path=image_path,
        final_budget=200,
        start_budget=100,
        seed=3,
        base_fit=FitConfig(iters=20),
        scfg=StructureTensorConfig(),
        growth_waves=4,
        device="cpu",
        feature_cap=9.0,
        feature_cap_reference_side=24.0,
    )

    assert field.n == 100
    assert actual_start == 100
    assert field.scale_max is not None
    assert float(field.scale_max.max()) <= 9.0
    assert cfg.split_mode == "residual_tensor_add"
    assert cfg.split_count == 25
    assert meta["init_config"]["scale_cap_mode"] == "feature"
    assert meta["init_config"]["scale_cap_max"] == 9.0
    assert meta["scale_cap_rule"] == "feature"
    assert meta["scale_cap_input"] == 9.0
    assert meta["scale_cap_reference_side"] == 24.0
    assert meta["scale_cap_max"] == 9.0
    assert meta["feature_cap_px"] == 9.0


def test_feature_relative_method_sets_init_cap_and_growth(tmp_path: Path):
    img = np.zeros((24, 24, 3), dtype=np.float32)
    img[:, 12:] = 1.0
    image_path = tmp_path / "target.png"
    Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(image_path)

    field, cfg, _seconds, actual_start, meta = F._build_method(
        method="structsplat_quadtree_wse_tensor_feature_rel",
        img=img,
        image_path=image_path,
        final_budget=200,
        start_budget=100,
        seed=3,
        base_fit=FitConfig(iters=20),
        scfg=StructureTensorConfig(),
        growth_waves=4,
        device="cpu",
    )

    assert field.n == 100
    assert actual_start == 100
    assert field.scale_max is not None
    assert cfg.split_mode == "residual_tensor_add"
    assert cfg.split_count == 25
    assert meta["init_config"]["scale_cap_mode"] == "feature_rel"
    assert meta["init_config"]["scale_cap_max"] is None
    assert meta["scale_cap_rule"] == "feature_rel"
    assert meta["scale_cap_input"] is None
    assert meta["scale_cap_reference_side"] is None
    assert meta["scale_cap_max"] is None
    assert meta["feature_cap_px"] is None


def test_write_index_links_summary_metrics_and_images(tmp_path: Path):
    (tmp_path / "plots").mkdir()
    (tmp_path / "grids" / "by_image").mkdir(parents=True)
    (tmp_path / "grids" / "by_budget").mkdir(parents=True)
    (tmp_path / "convergence_curves.csv").write_text("iter\n", encoding="utf-8")
    (tmp_path / "target_hit_rates.csv").write_text("target\n", encoding="utf-8")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "plots" / "mean_psnr_by_budget.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "grids" / "by_image" / "example.png")
    F._write_index(tmp_path, ["gaussianimage_fixed_full", "gaussianimage_plus_residual"])
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "summary.md" in text
    assert "metrics.csv" in text
    assert "convergence_curves.csv" in text
    assert "target_hit_rates.csv" in text
    assert "lowpass_vs_checkpoint.csv" not in text
    assert "absolute difference row" in text
    assert "mean_psnr_by_budget.png" in text
    assert "example.png" in text


def test_write_abs_diff_image_amplifies_target_error(tmp_path: Path):
    target = tmp_path / "target.png"
    recon = tmp_path / "recon.png"
    diff = tmp_path / "diff.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(target)
    Image.fromarray(np.full((2, 2, 3), 10, dtype=np.uint8), mode="RGB").save(recon)

    out = F._write_abs_diff_image(target, recon, diff, gain=3.0)

    assert out == diff
    arr = np.asarray(Image.open(diff).convert("RGB"))
    assert arr.shape == (2, 2, 3)
    assert np.all(arr == 30)


def test_write_convergence_tables_from_histories(tmp_path: Path):
    rows = [
        {
            "status": "ok",
            "final_budget": 100,
            "method": "gaussianimage_fixed_full",
            "method_label": F.METHOD_LABELS["gaussianimage_fixed_full"],
            "history": {"iter": [0, 10, 20], "psnr": [18.0, 22.0, 25.0], "elapsed": [0.1, 0.2, 0.3]},
            "iters_to_targets": {"22.0": 10.0, "24.0": 20.0},
        },
        {
            "status": "ok",
            "final_budget": 100,
            "method": "gaussianimage_plus_residual",
            "method_label": F.METHOD_LABELS["gaussianimage_plus_residual"],
            "history": {"iter": [0, 10, 20], "psnr": [17.0, 21.0, 23.0], "elapsed": [0.1, 0.2, 0.3]},
            "iters_to_targets": {"22.0": 20.0, "24.0": None},
        },
    ]

    F._write_convergence_tables(
        rows,
        tmp_path,
        ["gaussianimage_fixed_full", "gaussianimage_plus_residual"],
    )

    curves = (tmp_path / "convergence_curves.csv").read_text(encoding="utf-8")
    targets = (tmp_path / "target_hit_rates.csv").read_text(encoding="utf-8")
    assert "mean_psnr" in curves
    assert "gaussianimage_fixed_full" in curves
    assert "target_psnr" in targets
    assert "gaussianimage_plus_residual" in targets
    assert ",24.0,1,0," in targets


def test_convergence_table_holds_early_terminal_values_to_nominal_horizon(
    tmp_path: Path,
):
    method = "gaussianimage_fixed_full"
    rows = [
        {
            "status": "ok",
            "final_budget": 100,
            "method": method,
            "iters": 300,
            "history": {"iter": [0, 100], "psnr": [20.0, 30.0]},
            "iters_to_targets": {},
        },
        {
            "status": "ok",
            "final_budget": 100,
            "method": method,
            "iters": 300,
            "history": {"iter": [0, 100, 200], "psnr": [22.0, 32.0, 34.0]},
            "iters_to_targets": {},
        },
    ]

    F._write_convergence_tables(rows, tmp_path, [method])
    with (tmp_path / "convergence_curves.csv").open(newline="", encoding="utf-8") as stream:
        curves = list(csv.DictReader(stream))

    terminal = next(row for row in curves if int(row["iter"]) == 299)
    assert float(terminal["mean_psnr"]) == 32.0
    assert int(terminal["runs"]) == 2
    assert int(terminal["active_runs"]) == 0
    assert int(terminal["carried_runs"]) == 2


def test_storage_underfill_is_excluded_from_equal_capacity_ranking() -> None:
    rows = [
        {
            "status": "ok",
            "method": "instant_gi_quadtree_fixed",
            "final_budget": 5_376,
            "n_gaussians": 4_000,
            "storage_budget_bytes": 172_032,
            "storage_exact_budget": False,
        }
    ]

    mismatched = F._over_budget_methods(rows, ["instant_gi_quadtree_fixed"])
    assert mismatched == {"instant_gi_quadtree_fixed"}


def test_storage_tables_and_index_expose_unambiguous_image_bytes(tmp_path: Path) -> None:
    method = "gaussianimage_fixed_full"
    row = {
        "status": "ok",
        "image": "source_deadbeef",
        "image_stem": "source",
        "source_path": "/data/source.jpg",
        "source_width": 640,
        "source_height": 480,
        "width": 160,
        "height": 120,
        "source_file_bytes": 224_297,
        "benchmark_rgb8_payload_bytes": 57_600,
        "benchmark_array_bytes": 230_400,
        "target_file_bytes": 43_026,
        "storage_budget_bytes": 172_032,
        "storage_gaussian_cap": 5_376,
        "storage_bytes_per_gaussian": 32,
        "storage_accounting": "float32_constant_rgb_rs_payload_v1",
        "method": method,
        "method_label": F.METHOD_LABELS[method],
        "seed": 0,
        "iterations_run": 4_900,
        "convergence_stop_reason": "psnr_plateau",
        "stopped_early": True,
        "start_gaussians": 5_376,
        "n_gaussians": 5_376,
        "storage_budget_status": "exact",
        "storage_exact_budget": True,
        "representation_payload_bytes": 172_032,
        "actual_codec_bytes": 54_321,
        "reconstruction_file_bytes": 51_234,
        "psnr": 31.25,
        "ssim": 0.91,
        "ms_ssim": 0.95,
        "lpips": 0.12,
        "auc_psnr": 29.5,
        "fit_seconds": 10.0,
        "total_seconds": 10.2,
        "codec_psnr": 30.8,
        "codec_ms_ssim": 0.94,
        "actual_codec_bpp": 2.0,
    }
    (tmp_path / "plots").mkdir()
    (tmp_path / "grids" / "by_image").mkdir(parents=True)
    (tmp_path / "grids" / "by_budget").mkdir(parents=True)
    failed_method = "gaussianimage_plus_residual"
    failed = {
        **{
            key: row[key]
            for key in (
                "image",
                "image_stem",
                "source_path",
                "source_width",
                "source_height",
                "width",
                "height",
                "source_file_bytes",
                "target_file_bytes",
                "benchmark_rgb8_payload_bytes",
                "benchmark_array_bytes",
                "storage_budget_bytes",
                "storage_gaussian_cap",
                "storage_bytes_per_gaussian",
                "storage_accounting",
            )
        },
        "status": "error",
        "method": failed_method,
        "seed": 0,
        "error": "RuntimeError: boom",
    }

    methods = [method, failed_method]
    F._write_storage_tables([row, failed], tmp_path, methods)
    F._write_index(tmp_path, methods, rows=[row, failed])

    image_csv = (tmp_path / "image_storage.csv").read_text(encoding="utf-8")
    metrics_csv = (tmp_path / "image_metrics.csv").read_text(encoding="utf-8")
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "source_file_bytes" in image_csv
    assert "224297" in image_csv
    assert "source_path" in metrics_csv
    assert "representation_payload_bytes" in metrics_csv
    assert "actual_codec_bytes" in metrics_csv
    assert "172,032 bytes = 168 KiB" in html
    assert "224,297" in html
    assert "230,400" in html
    assert "54,321" in html
    assert "Failed cells" in html
    assert "RuntimeError: boom" in html


def test_zero_success_output_writers_remove_stale_derived_evidence(tmp_path: Path) -> None:
    for name in ("image_metrics.csv", "convergence_curves.csv", "target_hit_rates.csv"):
        (tmp_path / name).write_text("stale\n", encoding="utf-8")
    for directory in (tmp_path / "plots", tmp_path / "grids", tmp_path / "diffs"):
        directory.mkdir(parents=True)
        (directory / "stale.png").write_bytes(b"stale")
    error = {
        "status": "error",
        "image": "source_deadbeef",
        "source_path": "/data/source.jpg",
        "method": "gaussianimage_fixed_full",
    }

    F._write_storage_tables([error], tmp_path, ["gaussianimage_fixed_full"])
    F._write_convergence_tables([error], tmp_path, ["gaussianimage_fixed_full"])
    F._write_plots([error], tmp_path, ["gaussianimage_fixed_full"])
    F._write_grids([error], tmp_path, ["gaussianimage_fixed_full"])

    assert not (tmp_path / "image_metrics.csv").exists()
    assert not (tmp_path / "convergence_curves.csv").exists()
    assert not (tmp_path / "target_hit_rates.csv").exists()
    assert not (tmp_path / "plots").exists()
    assert not (tmp_path / "grids").exists()
    assert not (tmp_path / "diffs").exists()


def test_summary_headline_targets_are_bench004_set(tmp_path: Path):
    rows = [
        {
            "status": "ok",
            "image": "toy",
            "final_budget": 100,
            "start_gaussians": 50,
            "n_gaussians": 100,
            "method": "gaussianimage_fixed_full",
            "method_label": F.METHOD_LABELS["gaussianimage_fixed_full"],
            "psnr": 30.0,
            "ms_ssim": 0.9,
            "auc_psnr": 28.0,
            "lpips": None,
            "init_seconds": 0.1,
            "fit_seconds": 0.2,
            "total_seconds": 0.3,
            "history": {"iter": [0, 10], "psnr": [24.0, 30.0]},
            "iters_to_targets": {"24.0": 1, "28.0": 5, "30.0": 10, "32.0": None, "35.0": None},
        }
    ]

    F._write_summary(rows, tmp_path, ["gaussianimage_fixed_full"])

    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Hit 28" in text
    assert "Iter 30" in text
    assert "Hit 32" in text
    assert "Hit 24" not in text
    assert "Hit 35" not in text


def test_summary_reports_default_promotion_check(tmp_path: Path):
    base = {
        "status": "ok",
        "image": "toy",
        "source_path": "toy.png",
        "max_side": 16,
        "final_budget": 100,
        "start_budget": 50,
        "seed": 0,
        "iters": 10,
        "renderer": "normalized",
        "start_gaussians": 50,
        "n_gaussians": 100,
        "lpips": None,
        "init_seconds": 0.1,
        "history": {"iter": [0, 10], "psnr": [20.0, 30.0]},
        "iters_to_targets": {},
    }
    rows = [
        {
            **base,
            "method": F.BEST_DEFAULT_METHOD,
            "method_label": F.METHOD_LABELS[F.BEST_DEFAULT_METHOD],
            "psnr": 30.0,
            "ms_ssim": 0.900,
            "auc_psnr": 24.0,
            "fit_seconds": 1.0,
            "total_seconds": 1.1,
        },
        {
            **base,
            "method": "structsplat_best_tensor_loss",
            "method_label": F.METHOD_LABELS["structsplat_best_tensor_loss"],
            "psnr": 30.2,
            "ms_ssim": 0.901,
            "auc_psnr": 24.1,
            "fit_seconds": 0.9,
            "total_seconds": 1.0,
        },
        {
            **base,
            "method": "structsplat_best_color_final",
            "method_label": F.METHOD_LABELS["structsplat_best_color_final"],
            "psnr": 30.3,
            "ms_ssim": 0.902,
            "auc_psnr": 24.2,
            "fit_seconds": 1.2,
            "total_seconds": 1.3,
        },
    ]

    F._write_summary(
        rows,
        tmp_path,
        [F.BEST_DEFAULT_METHOD, "structsplat_best_tensor_loss", "structsplat_best_color_final"],
    )

    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Default Promotion Check" in text
    assert "SS best + tensor loss" in text
    assert (
        "| SS best + tensor loss | 1 | +0.2000 | +0.00100 | +0.1000 | "
        "-0.1000 | -0.1000 | 1/1 | 1/1 | 1/1 | 1/1 | yes |"
    ) in text
    assert (
        "| SS best + final color solve | 1 | +0.3000 | +0.00200 | +0.2000 | "
        "+0.2000 | +0.2000 | 1/1 | 1/1 | 1/1 | 0/1 | no |"
    ) in text


def test_default_dominance_audit_clusters_images_and_marks_tradeoffs(tmp_path: Path):
    rows = []
    methods = [
        F.BEST_DEFAULT_METHOD,
        "structsplat_best_tensor_loss",
        "structsplat_best_color_final",
        "gaussianimage_fixed_full",
        "structsplat_best_adaptive_1p5x",
    ]
    for image in ("a", "b"):
        for seed in (0, 1):
            common = {
                "status": "ok",
                "image": image,
                "source_path": f"{image}.png",
                "max_side": 16,
                "final_budget": 100,
                "start_budget": 50,
                "seed": seed,
                "iters": 10,
                "renderer": "normalized",
                "init_seconds": 0.1,
            }
            rows.append({
                **common,
                "method": F.BEST_DEFAULT_METHOD,
                "n_gaussians": 100,
                "psnr": 30.0,
                "ms_ssim": 0.900,
                "auc_psnr": 24.0,
                "fit_seconds": 1.0,
                "total_seconds": 1.1,
                "lpips": 0.20,
            })
            rows.append({
                **common,
                "method": "structsplat_best_tensor_loss",
                "n_gaussians": 100,
                "psnr": 30.2,
                "ms_ssim": 0.901,
                "auc_psnr": 24.1,
                "fit_seconds": 0.9,
                "total_seconds": 1.0,
                "lpips": 0.18,
            })
            rows.append({
                **common,
                "method": "structsplat_best_color_final",
                "n_gaussians": 100,
                "psnr": 30.3,
                "ms_ssim": 0.902,
                "auc_psnr": 23.9,
                "fit_seconds": 1.2,
                "total_seconds": 1.3,
                "lpips": 0.19,
            })
            rows.append({
                **common,
                "method": "gaussianimage_fixed_full",
                "n_gaussians": 100,
                "psnr": 29.8,
                "ms_ssim": 0.899,
                "auc_psnr": 23.9,
                "fit_seconds": 1.1,
                "total_seconds": 1.2,
                "lpips": 0.22,
            })
            rows.append({
                **common,
                "method": "structsplat_best_adaptive_1p5x",
                "n_gaussians": 150,
                "psnr": 31.0,
                "ms_ssim": 0.910,
                "auc_psnr": 25.0,
                "fit_seconds": 0.8,
                "total_seconds": 0.9,
                "lpips": 0.15,
            })

    audit = {row["method"]: row for row in F._default_dominance_audit(rows, methods)}
    tensor = audit["structsplat_best_tensor_loss"]
    assert tensor["pairs"] == 4
    assert tensor["paired_images"] == 2
    assert tensor["sample_relation"] == "candidate_dominates"
    assert tensor["supported_relation_95ci"] == "candidate_dominates"
    assert tensor["gain_fit_seconds"] > 0.0
    assert tensor["gain_lpips"] > 0.0

    color = audit["structsplat_best_color_final"]
    assert color["sample_relation"] == "tradeoff"
    assert color["supported_relation_95ci"] == "tradeoff"

    fixed = audit["gaussianimage_fixed_full"]
    assert fixed["sample_relation"] == "default_dominates"
    assert fixed["supported_relation_95ci"] == "default_dominates"

    adaptive = audit["structsplat_best_adaptive_1p5x"]
    assert adaptive["budget_matched"] is False
    assert adaptive["sample_relation"] == "over_budget"
    assert adaptive["supported_relation_95ci"] == "not_comparable"

    F._write_default_dominance(rows, tmp_path, methods)
    csv_text = (tmp_path / "default_dominance.csv").read_text(encoding="utf-8")
    assert "supported_relation_95ci" in csv_text
    assert "candidate_dominates" in csv_text
    assert "over_budget" in csv_text


def test_default_dominance_relation_uses_familywise_not_marginal_bounds():
    rows = []
    methods = [F.BEST_DEFAULT_METHOD, "structsplat_best_tensor_loss"]
    gains = [0.01] * 7 + [-0.012]
    for image_index, gain in enumerate(gains):
        common = {
            "status": "ok",
            "image": f"image{image_index}",
            "source_path": f"image{image_index}.png",
            "max_side": 16,
            "final_budget": 100,
            "start_budget": 50,
            "seed": 0,
            "iters": 10,
            "renderer": "normalized",
            "n_gaussians": 100,
        }
        rows.append({
            **common,
            "method": F.BEST_DEFAULT_METHOD,
            "psnr": 30.0,
            "ms_ssim": 0.9,
            "auc_psnr": 24.0,
            "fit_seconds": 1.0,
            "total_seconds": 1.1,
        })
        rows.append({
            **common,
            "method": "structsplat_best_tensor_loss",
            "psnr": 30.0 + gain,
            "ms_ssim": 0.9 + gain,
            "auc_psnr": 24.0 + gain,
            "fit_seconds": 1.0 - gain,
            "total_seconds": 1.1 - gain,
        })

    result = F._default_dominance_audit(rows, methods)[0]

    assert result["ci_low_psnr"] > 0.0
    assert result["sample_relation"] == "candidate_dominates"
    assert result["supported_relation_95ci"] == "inconclusive"
