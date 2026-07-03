# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.config import FitConfig, InitConfig
from structsplat import init as _init
from structsplat.gaussians import GaussianField
from structsplat.fit import (
    _add_from_residual,
    _carry_adam_state,
    _lr_factor,
    _make_optimizer,
    _maybe_prune,
    _split_from_residual,
    fit,
)
from structsplat import metrics as M


def test_lr_factor_is_global_step_decay():
    cfg = FitConfig(lr_decay_every=100, lr_decay_gamma=0.5)
    assert _lr_factor(cfg, 0) == 1.0
    assert _lr_factor(cfg, 99) == 1.0
    assert _lr_factor(cfg, 100) == 0.5
    assert _lr_factor(cfg, 250) == 0.25
    assert _lr_factor(FitConfig(lr_decay_every=None), 1000) == 1.0


def test_carry_adam_state_preserves_moments_across_prune_and_split():
    img = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))
    field.trainable()
    cfg = FitConfig()
    opt = _make_optimizer(field, cfg)
    target = torch.as_tensor(img)
    from structsplat.render import render
    im = render(field.means, field.conics(), field.colors, field.radii(3.0), 16, 16)
    (im - target).abs().mean().backward()
    opt.step()
    old_exp_avg = opt.state[field.means]["exp_avg"].clone()

    keep = torch.zeros(8, dtype=torch.bool)
    keep[:5] = True
    pruned = field.subset(keep)
    n_new = 3
    grown = pruned.append(pruned.subset(slice(0, n_new)))
    grown.trainable()
    opt2 = _carry_adam_state(opt, grown, cfg, keep, n_new)
    st = opt2.state[grown.means]
    assert torch.allclose(st["exp_avg"][:5], old_exp_avg[:5])   # survivors keep moments
    assert torch.all(st["exp_avg"][5:] == 0)                    # new rows start clean
    assert st["exp_avg"].shape[0] == 8


def test_fit_with_decay_prune_split_still_improves():
    img = np.zeros((24, 24, 3), np.float32)
    img[:, 12:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="iso_blue_noise", num_gaussians=32, seed=0))
    out = fit(field, target, FitConfig(
        iters=30, log_every=5, lr_decay_every=10, lr_decay_gamma=0.5,
        prune_every=8, prune_min_activity=1e-6, prune_keep_min=8,
        split_every=8, split_count=4, max_gaussians=64,
    ), verbose=False)
    assert np.isfinite(out["psnr"])
    assert out["history"]["psnr"][-1] >= out["history"]["psnr"][0] - 0.5


def test_no_restructure_on_final_iteration():
    # iters divisible by split_every: the last split window must be skipped, otherwise the
    # returned field contains Gaussians no optimizer step ever touched
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))
    out = fit(field, target, FitConfig(iters=10, log_every=5, split_every=5, split_count=4,
                                       split_mode="residual_add"), verbose=False)
    assert out["n_gaussians"] == 12  # one split at it=4, none at it=9


def test_int_target_psnrs_are_recorded():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=16, seed=0))
    out = fit(field, target, FitConfig(iters=3, log_every=1, target_psnr=1,
                                       target_psnrs=[1, 2.0]), verbose=False)
    assert out["iters_to_target"] is not None       # str(int) vs str(float) key mismatch
    assert set(out["iters_to_targets"]) == {"1.0", "2.0"}


def test_target_psnr_tracking_matches_logged_psnr_every_iteration():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=16, seed=1))
    out = fit(
        field,
        target,
        FitConfig(iters=6, log_every=1, target_psnrs=[1.0, 2.0, 99.0]),
        verbose=False,
    )
    hist = out["history"]
    for target_psnr in [1.0, 2.0, 99.0]:
        expected = next(
            (it for it, psnr in zip(hist["iter"], hist["psnr"]) if psnr >= target_psnr),
            None,
        )
        assert out["iters_to_targets"][str(target_psnr)] == expected


def test_ssim_loss_is_skipped_when_weight_is_zero(monkeypatch):
    original_ssim = M.ssim
    calls = {"n": 0}

    def counting_ssim(*args, **kwargs):
        calls["n"] += 1
        return original_ssim(*args, **kwargs)

    monkeypatch.setattr(M, "ssim", counting_ssim)
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)

    field0 = _init.build_field(img, InitConfig(strategy="random", num_gaussians=12, seed=2))
    fit(field0, target, FitConfig(iters=4, log_every=1, ssim_weight=0.0), verbose=False)
    zero_weight_calls = calls["n"]

    calls["n"] = 0
    field1 = _init.build_field(img, InitConfig(strategy="random", num_gaussians=12, seed=2))
    fit(field1, target, FitConfig(iters=4, log_every=1, ssim_weight=0.3), verbose=False)
    weighted_calls = calls["n"]

    assert weighted_calls - zero_weight_calls == 4


def test_early_stop_is_opt_in_and_reports_iterations():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=16, seed=0))
    out = fit(
        field,
        target,
        FitConfig(
            iters=20,
            log_every=1,
            early_stop_patience=1,
            early_stop_min_delta=1e6,
        ),
        verbose=False,
    )
    assert out["stopped_early"]
    assert out["iterations_run"] < 20


def test_fit_respects_field_scale_caps():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(
        img,
        InitConfig(
            strategy="aniso_flanking",
            num_gaussians=16,
            scale_cap_mode="hard",
            scale_cap_max=1.25,
            seed=0,
        ),
    )
    out = fit(
        field,
        target,
        FitConfig(iters=6, log_every=1, lr_scales=1.0),
        verbose=False,
    )
    assert float(torch.exp(out["field"].log_scales).max().detach()) <= 1.25 + 1e-6


def test_split_uses_residual_colors_under_additive_renderer():
    # additive splits must carry the residual (target - render), not the full target color,
    # or the child double-counts brightness (FIT-002 #1).
    H, W = 12, 12
    target = torch.zeros(H, W, 3)
    target[:, :, 0] = 0.8
    render_img = torch.full((H, W, 3), 0.5)  # already contributes 0.5 red everywhere
    field = _init.build_field(
        target.numpy(), InitConfig(strategy="random", num_gaussians=6, seed=0))
    base = dict(split_count=3, split_mode="duplicate", max_gaussians=64)
    add_field, k = _split_from_residual(
        field, target, render_img, FitConfig(renderer="additive", **base))
    norm_field, _ = _split_from_residual(
        field, target, render_img, FitConfig(renderer="normalized", **base))
    assert k == 3
    add_children = add_field.colors[-k:]
    norm_children = norm_field.colors[-k:]
    # normalized children ~= full target red (0.8); additive children ~= residual (0.8-0.5=0.3)
    assert torch.allclose(norm_children[:, 0], torch.full((k,), 0.8), atol=1e-5)
    assert torch.allclose(add_children[:, 0], torch.full((k,), 0.3), atol=1e-5)


def test_zero_opacity_gaussian_is_pruned():
    # a fully-transparent Gaussian contributes nothing but survived opacity-free pruning (#2).
    means = np.array([[8.0, 8.0], [4.0, 4.0], [12.0, 12.0], [2.0, 14.0]], dtype=np.float32)
    scales = np.full((4, 2), 2.0, dtype=np.float32)
    # logits: three saturated-opaque, one saturated-transparent (index 1)
    opac_logits = np.array([10.0, -20.0, 10.0, 10.0], dtype=np.float32)
    field = GaussianField.from_numpy(means, scales, np.zeros(4), np.ones((4, 3)),
                                     opacities=opac_logits)
    cfg = FitConfig(prune_min_activity=1e-3, prune_keep_min=1)
    pruned, keep = _maybe_prune(field, cfg, 16, 16)
    assert keep is not None
    assert not bool(keep[1])          # transparent Gaussian removed
    assert bool(keep[0]) and bool(keep[2]) and bool(keep[3])


def test_history_n_gaussians_pairs_with_prestep_psnr():
    # on a split iteration the logged n_gaussians must be the pre-split count that produced the
    # logged (pre-step) PSNR, not the post-split count (#3).
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))
    out = fit(field, target, FitConfig(iters=10, log_every=1, split_every=5, split_count=4,
                                       split_mode="residual_add", max_gaussians=64),
              verbose=False)
    ng = out["history"]["n_gaussians"]
    assert ng[4] == 8    # split fires at it=4; the row for it=4 is pre-split
    assert ng[5] == 12   # subsequent rows see the grown field


def test_densify_anisotropy_threaded_from_config():
    # a larger densify_max_axis_ratio must make residual_tensor_add children more anisotropic (#4).
    H, W = 24, 24
    target = torch.zeros(H, W, 3)
    target[:, 12:] = 1.0                    # a strong vertical edge -> high coherence
    render_img = target.clone()
    render_img[:, 11:13] = 0.0              # residual concentrated on the edge columns
    field = _init.build_field(
        target.numpy(), InitConfig(strategy="random", num_gaussians=6, seed=0))
    common = dict(split_count=4, split_mode="residual_tensor_add", max_gaussians=64)

    def child_ratio(max_ratio):
        f, k = _add_from_residual(field, target, render_img,
                                  FitConfig(densify_max_axis_ratio=max_ratio, **common),
                                  tensor_aligned=True)
        scales = torch.exp(f.log_scales[-k:])
        return (scales.max(dim=1).values / scales.min(dim=1).values).max().item()

    assert child_ratio(8.0) > child_ratio(2.0) + 0.5
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(
        img,
        InitConfig(
            strategy="aniso_flanking",
            num_gaussians=12,
            scale_cap_mode="hard",
            scale_cap_max=1.25,
            seed=0,
        ),
    )
    out = fit(
        field,
        target,
        FitConfig(
            iters=7,
            log_every=1,
            split_every=3,
            split_count=4,
            split_mode="residual_tensor_add",
            max_gaussians=24,
            lr_scales=1.0,
        ),
        verbose=False,
    )
    assert out["n_gaussians"] > 12
    assert out["field"].scale_max is not None
    assert float(torch.exp(out["field"].log_scales).max().detach()) <= 1.25 + 1e-6
