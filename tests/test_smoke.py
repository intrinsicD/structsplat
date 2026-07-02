# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.config import InitConfig, FitConfig, PyramidConfig
from structsplat import init as _init
from structsplat.init import STRATEGIES
from structsplat.fit import fit
from structsplat.pyramid import fit_pyramid


def _toy(H=32, W=32):
    img = np.zeros((H, W, 3), np.float32)
    img[:, W // 2:, :] = 1.0
    img[8:12, 8:12] = [1.0, 0.5, 0.0]
    return img


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_build_field_all_strategies(strategy):
    img = _toy()
    icfg = InitConfig(strategy=strategy, num_gaussians=64, seed=0)
    field = _init.build_field(img, icfg)
    assert 32 <= field.n <= 90
    for p in (field.means, field.log_scales, field.rotations, field.colors):
        assert torch.isfinite(p).all()


def test_build_field_stage_variants():
    img = _toy()
    icfg = InitConfig(
        strategy="aniso_flanking",
        num_gaussians=32,
        density_mode="hybrid",
        sampling_mode="density_random",
        color_mode="two_sided",
        scale_mode="uniform",
        opacity_mode="constant",
        seed=1,
    )
    field = _init.build_field(img, icfg)
    assert field.n == 32
    assert field.opacities is not None
    assert torch.isfinite(field.colors).all()


def test_fit_runs_and_improves():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="aniso_flanking", num_gaussians=64)
    field = _init.build_field(img, icfg)
    out = fit(field, target, FitConfig(iters=15, log_every=5))
    assert np.isfinite(out["psnr"]) and out["psnr"] > 0


def test_fit_dynamic_controls_prune_split_and_track_targets():
    img = _toy(24, 24)
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="aniso_flanking", num_gaussians=24)
    field = _init.build_field(img, icfg)
    out = fit(
        field,
        target,
        FitConfig(
            iters=4,
            log_every=1,
            pixel_loss="l2",
            target_psnrs=[0.0],
            lr_decay_every=2,
            prune_every=2,
            prune_min_activity=1e9,
            prune_keep_min=8,
            split_every=2,
            split_count=2,
            split_scale=0.8,
            max_gaussians=30,
        ),
        verbose=False,
    )
    assert out["lpips"] is None
    assert out["iters_to_targets"]["0.0"] is not None
    assert 8 <= out["n_gaussians"] <= 30
    assert len(out["history"]["n_gaussians"]) == 4


def test_fit_residual_add_charbonnier_cosine():
    img = _toy(24, 24)
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=20))
    out = fit(
        field,
        target,
        FitConfig(
            iters=4,
            log_every=1,
            pixel_loss="charbonnier",
            loss_warmup_iters=1,
            loss_warmup_pixel_loss="l2",
            lr_schedule="cosine",
            split_every=2,
            split_count=3,
            split_mode="residual_add",
            max_gaussians=28,
        ),
        verbose=False,
    )
    assert np.isfinite(out["psnr"])
    assert 20 < out["n_gaussians"] <= 28


def test_pyramid_runs():
    img = _toy()
    target = torch.as_tensor(img)
    icfg = InitConfig(strategy="aniso_flanking", num_gaussians=48)
    out = fit_pyramid(img, target, icfg, FitConfig(iters=10, log_every=5),
                      PyramidConfig(levels=2, level_fractions=[0.5, 0.5], iters_per_level=8),
                      verbose=False)
    assert np.isfinite(out["psnr"])
    assert len(out["level_counts"]) == 2
    assert len(out["prefix_metrics"]) == 2
    assert out["prefix_metrics"][-1]["n_gaussians"] == out["n_gaussians"]
