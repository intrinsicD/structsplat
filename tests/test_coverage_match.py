"""FIT-022 coverage-matching regularizer (feature-targeted fit-time Gaussian blue noise)."""
import numpy as np
import pytest
import torch

from structsplat.config import FitConfig
from structsplat.fit import (
    fit,
    _coverage_match_factor,
    _coverage_match_loss,
    _prepare_coverage_profile,
    _raw_weight_map_field,
    _MaskConstraint,
)
from structsplat.gaussians import GaussianField


def _test_image(H=40, W=40):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([
        0.2 + 0.6 * (xx / W),
        0.3 + 0.4 * (yy / H),
        0.5 + 0.3 * np.sin(xx / 5.0) * np.cos(yy / 5.0),
    ], axis=2).clip(0.0, 1.0).astype(np.float32)
    mask = (((yy - H / 2) ** 2 + (xx - W / 2) ** 2) <= (0.42 * H) ** 2).astype(np.float32)
    return img, mask


def _clustered_field(n=24, H=40, W=40, seed=0):
    """All Gaussians clumped in one quadrant — maximal coverage mismatch to relax."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(6, 14, size=(n, 2)).astype(np.float32)
    return GaussianField.from_numpy(
        pos, np.full((n, 2), 2.5, np.float32), np.zeros(n, np.float32),
        np.full((n, 3), 0.5, np.float32), opacities=np.full(n, 2.0, np.float32),
    )


def test_config_validation():
    with pytest.raises(ValueError, match="coverage_match_target"):
        FitConfig(coverage_match_target="blue_noise")
    with pytest.raises(ValueError, match="coverage_match_weight"):
        FitConfig(coverage_match_weight=-1.0)
    with pytest.raises(ValueError, match="coverage_match_decay_frac"):
        FitConfig(coverage_match_decay_frac=1.5)
    with pytest.raises(ValueError, match="coverage_match_error_alpha"):
        FitConfig(coverage_match_error_alpha=2.0)


def test_tensor_boundary_requires_mask():
    img, _ = _test_image()
    cfg = FitConfig(coverage_match_weight=0.1, coverage_match_target="tensor_boundary")
    with pytest.raises(ValueError, match="requires a mask"):
        _prepare_coverage_profile(torch.as_tensor(img), cfg, None)


def test_gradients_reach_geometry_but_not_opacity():
    H = W = 40
    img, _ = _test_image(H, W)
    target = torch.as_tensor(img)
    cfg = FitConfig(coverage_match_weight=0.1, render_chunk=64)
    field = _clustered_field().trainable()
    profile, inside = _prepare_coverage_profile(target, cfg, None)
    loss = _coverage_match_loss(field, cfg, H, W, profile, inside, None, target)
    loss.backward()
    assert field.means.grad is not None
    assert float(field.means.grad.abs().max()) > 0.0
    assert float(field.log_scales.grad.abs().max()) > 0.0
    # opacities are detached inside S by design: transport geometry, never dim rows
    assert field.opacities.grad is None or float(field.opacities.grad.abs().max()) == 0.0


def test_pure_energy_descent_reduces_coverage_mismatch():
    H = W = 40
    img, _ = _test_image(H, W)
    target = torch.as_tensor(img)
    cfg = FitConfig(coverage_match_weight=1.0, render_chunk=64)
    field = _clustered_field().trainable()
    profile, inside = _prepare_coverage_profile(target, cfg, None)
    opt = torch.optim.Adam([field.means], lr=0.5)
    start = float(
        _coverage_match_loss(field, cfg, H, W, profile, inside, None, target).detach()
    )
    for _ in range(40):
        opt.zero_grad()
        loss = _coverage_match_loss(field, cfg, H, W, profile, inside, None, target)
        loss.backward()
        opt.step()
    end = float(
        _coverage_match_loss(field, cfg, H, W, profile, inside, None, target).detach()
    )
    assert end < 0.7 * start, f"coverage mismatch did not relax: {start} -> {end}"


def test_target_is_mass_neutral():
    H = W = 40
    img, _ = _test_image(H, W)
    target = torch.as_tensor(img)
    cfg = FitConfig(coverage_match_weight=0.1, render_chunk=64)
    field = _clustered_field()
    profile, inside = _prepare_coverage_profile(target, cfg, None)
    S = _raw_weight_map_field(field, cfg, H, W, detach_opacities=True)
    c = profile / profile.sum().clamp_min(1e-12) * (S.detach() * inside).sum()
    assert float((c.sum() - (S * inside).sum()).abs()) < 1e-3 * float(S.sum())


def test_decay_factor_schedule():
    cfg = FitConfig(iters=100, coverage_match_weight=0.1, coverage_match_decay_frac=0.5)
    assert _coverage_match_factor(cfg, 0) == 1.0
    mid = _coverage_match_factor(cfg, 25)
    assert 0.0 < mid < 1.0
    assert _coverage_match_factor(cfg, 50) == 0.0
    assert _coverage_match_factor(cfg, 99) == 0.0
    flat = FitConfig(iters=100, coverage_match_weight=0.1)
    assert _coverage_match_factor(flat, 99) == 1.0


def test_fit_with_coverage_match_runs_maskless():
    H = W = 40
    img, _ = _test_image(H, W)
    target = torch.as_tensor(img)
    torch.manual_seed(0)
    field = _clustered_field()
    cfg = FitConfig(iters=40, log_every=10, ssim_weight=0.0, render_chunk=64,
                    coverage_match_weight=0.05, coverage_match_decay_frac=0.5)
    out = fit(field, target, cfg, verbose=False)
    assert np.isfinite(out["psnr"])
    logged = out["history"]["coverage_match_loss"]
    assert any(v is not None for v in logged), "coverage term must be logged while active"
    assert logged[-1] is None, "decayed-to-zero term must log None"


def test_fit_masked_pooled_and_coverage_compose():
    H = W = 48
    img, mask = _test_image(H, W)
    img = img * mask[..., None]
    target = torch.as_tensor(img)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    pos = rng.uniform(14, 34, size=(40, 2)).astype(np.float32)
    field = GaussianField.from_numpy(
        pos, np.full((40, 2), 2.0, np.float32), np.zeros(40, np.float32),
        img[pos[:, 1].astype(int), pos[:, 0].astype(int)],
    )
    cfg = FitConfig(
        iters=45, log_every=15, ssim_weight=0.0, render_chunk=64,
        mask_contain=True, support_fade=True, loss_weighting="mask",
        triage_every=15, triage_spawn_count=8, triage_split_count=2,
        triage_merge_count=2, triage_park_count=2, pool_capacity=96,
        coverage_match_weight=0.05, coverage_match_target="tensor_boundary",
    )
    out = fit(field, target, cfg, verbose=False, mask=mask)
    assert np.isfinite(out["psnr"])
    assert out["pool"] is not None and out["pool"]["live_n"] == out["field"].n
    outside = torch.as_tensor(1.0 - mask)[..., None]
    assert float((out["render"] * outside).abs().max()) == 0.0


def test_boundary_band_raises_profile_inside_band():
    H = W = 40
    img, mask = _test_image(H, W)
    target = torch.as_tensor(img)
    cfg = FitConfig(coverage_match_weight=0.1, coverage_match_target="tensor_boundary",
                    coverage_match_boundary_boost=2.0)
    mc = _MaskConstraint.from_mask(mask, target.device, target.dtype, 3.0, 1.5)
    profile_b, _ = _prepare_coverage_profile(target, cfg, mc)
    plain = FitConfig(coverage_match_weight=0.1)
    profile_p, _ = _prepare_coverage_profile(target, plain, mc)
    band = (mc.sdf_flat > 0.0) & (mc.sdf_flat <= 3.0)
    ratio = profile_b[band] / profile_p[band].clamp_min(1e-12)
    assert float(ratio.min()) > 2.9  # 1 + boost = 3x inside the band
