from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from benchmarks import fair_density_control_compare as F
from structsplat.config import FitConfig


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
        "structsplat_onedge_tensor",
        "structsplat_quadtree_wse_tensor",
    ]
    for method in growth_methods:
        split_mode = "residual_add" if method.endswith("_residual") or method in {
            "gaussianimage_plus_residual",
            "image_gs_residual",
        } else "residual_tensor_add"
        cfg = F._growth_fit_cfg(base, 2000, start, split_mode, growth_waves=4)
        assert cfg.max_gaussians == 2000
        assert cfg.split_every == 2
        assert cfg.split_count == 250


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
    assert "mean_psnr_by_budget.png" in text
    assert "example.png" in text
