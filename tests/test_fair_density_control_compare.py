from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from benchmarks import fair_density_control_compare as F
from structsplat.config import FitConfig, StructureTensorConfig


def test_growth_fit_cfg_reaches_final_cap_with_shared_schedule():
    base = FitConfig(iters=100)
    cfg = F._growth_fit_cfg(base, final_budget=1000, start_budget=500, split_mode="residual_add", growth_waves=4)
    assert cfg.max_gaussians == 1000
    assert cfg.split_mode == "residual_add"
    assert cfg.split_every == 20
    assert cfg.split_count == 125


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
        "structsplat_onedge_tensor",
        "structsplat_onedge_tensor_featurecap",
        "structsplat_quadtree_wse_tensor",
    ]
    for method in growth_methods:
        split_mode = F.STRUCTSPLAT_SPLIT_MODE.get(method, "residual_add")
        cfg = F._growth_fit_cfg(base, 2000, start, split_mode, growth_waves=4)
        assert cfg.max_gaussians == 2000
        assert cfg.split_every == 2
        assert cfg.split_count == 250


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


def test_write_index_links_summary_metrics_and_images(tmp_path: Path):
    (tmp_path / "plots").mkdir()
    (tmp_path / "grids" / "by_image").mkdir(parents=True)
    (tmp_path / "grids" / "by_budget").mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(tmp_path / "plots" / "mean_psnr_by_budget.png")
    Image.new("RGB", (8, 8), "white").save(tmp_path / "grids" / "by_image" / "example.png")
    F._write_index(tmp_path, ["gaussianimage_fixed_full", "gaussianimage_plus_residual"])
    text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "summary.md" in text
    assert "metrics.csv" in text
    assert "convergence_curves.csv" in text
    assert "target_hit_rates.csv" in text
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
