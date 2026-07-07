# ruff: noqa: E402
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from structsplat.config import FitConfig, InitConfig
from structsplat import init as _init
from structsplat.gaussians import GaussianField
from structsplat.fit import (
    _add_from_residual,
    _carry_adam_state,
    _freq_violation_scores,
    _lr_factor,
    _make_optimizer,
    _maybe_prune,
    _moment_preserving_duplicate_indices,
    _residual_candidate_pixels,
    _relocate_from_residual,
    _solve_colors_normalized,
    _split_from_residual,
    differentiable_rate_bpp,
    estimated_raw_bpp,
    fit,
)
from structsplat import codec
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


def test_carry_adan_state_preserves_three_moments_across_split():
    img = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))
    field.trainable()
    cfg = FitConfig(optimizer="adan")
    opt = _make_optimizer(field, cfg)
    target = torch.as_tensor(img)
    from structsplat.render import render
    im = render(field.means, field.conics(), field.colors, field.radii(3.0), 16, 16)
    (im - target).abs().mean().backward()
    opt.step()
    old = opt.state[field.means]

    grown = field.subset(slice(0, 8)).append(field.subset(slice(0, 2)))
    grown.trainable()
    opt2 = _carry_adam_state(opt, grown, cfg, keep=None, n_new=2)
    st = opt2.state[grown.means]

    for key in ("exp_avg", "exp_avg_diff", "exp_avg_sq", "prev_grad"):
        assert key in st
        assert st[key].shape[0] == 10
        assert torch.allclose(st[key][:8], old[key])
        assert torch.all(st[key][8:] == 0)


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


def _color_solve_fixture():
    H, W = 12, 12
    means = torch.tensor([[3.0, 3.0], [8.0, 8.0]])
    log_scales = torch.log(torch.full((2, 2), 0.45))
    rotations = torch.zeros(2)
    true_colors = torch.tensor([[0.2, 0.7, 0.1], [0.9, 0.1, 0.4]])
    true_field = GaussianField(
        means.clone(), log_scales.clone(), rotations.clone(), true_colors.clone()
    )
    cfg = FitConfig(
        renderer="normalized",
        render_chunk=1,
        sigma_cutoff=2.0,
        color_solve_lambda=0.0,
        color_solve_maxiter=8,
        ssim_weight=0.0,
    )
    from structsplat.render import render_field
    target = render_field(
        true_field.means,
        true_field.conics(cfg.aa_dilation),
        true_field.colors,
        true_field.radii(cfg.sigma_cutoff, cfg.aa_dilation),
        H,
        W,
        cfg.render_chunk,
        cfg.renderer,
        true_field.opacity_values(),
        support_fade=cfg.support_fade,
        sigma_cutoff=cfg.sigma_cutoff,
    )
    field = GaussianField(
        means.clone(),
        log_scales.clone(),
        rotations.clone(),
        torch.zeros_like(true_colors),
    )
    return field, true_colors, target, cfg, H, W


def _field_to(field: GaussianField, device: str) -> GaussianField:
    return GaussianField(
        field.means.to(device),
        field.log_scales.to(device),
        field.rotations.to(device),
        field.colors.to(device),
        None if field.opacities is None else field.opacities.to(device),
        None if field.scale_max is None else field.scale_max.to(device),
        None if field.color_grads is None else field.color_grads.to(device),
    )


def test_color_solve_recovers_fixed_geometry_colors():
    field, true_colors, target, cfg, H, W = _color_solve_fixture()

    stats = _solve_colors_normalized(field, target, cfg, H, W)

    assert stats["iterations"] <= cfg.color_solve_maxiter
    assert stats["relative_residual"] < 1e-4
    assert torch.allclose(field.colors, true_colors, atol=1e-4)


def test_fit_time_qat_ste_returns_codec_config_and_rate_history():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(
        img,
        InitConfig(strategy="random", num_gaussians=12, opacity_mode="constant", seed=0),
    )

    cfg = FitConfig(
        iters=3,
        log_every=1,
        ssim_weight=0.0,
        qat_mode="ste",
        lambda_rate=0.01,
        qat_bits_means=10,
        qat_bits_scales=5,
        qat_bits_rot=5,
        qat_bits_colors=5,
    )
    out = fit(field, target, cfg, verbose=False)

    assert out["qat_mode"] == "ste"
    assert out["lambda_rate"] == pytest.approx(0.01)
    assert out["qat_rate_bpp"] is not None and out["qat_rate_bpp"] > 0
    assert out["qat_codec_config"] is not None
    assert len(out["history"]["qat_rate_bpp"]) == len(out["history"]["iter"])
    assert all(v is not None and v > 0 for v in out["history"]["qat_rate_bpp"])
    rd = codec.rd_point(out["field"], target, cfg, out["qat_codec_config"])
    assert np.isfinite(rd["psnr"])
    assert rd["bits"] == [10, 5, 5, 5]


def test_fit_time_qat_noise_runs_and_affine_fails_closed():
    img = np.zeros((12, 12, 3), np.float32)
    img[:, 6:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))
    out = fit(
        field,
        target,
        FitConfig(iters=2, log_every=1, ssim_weight=0.0, qat_mode="noise"),
        verbose=False,
    )
    assert np.isfinite(out["psnr"])
    assert out["qat_codec_config"] is not None

    affine_field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))
    with pytest.raises(ValueError, match="color_basis='constant'"):
        fit(
            affine_field,
            target,
            FitConfig(iters=1, color_basis="affine", qat_mode="ste"),
            verbose=False,
        )


def test_differentiable_rate_proxy_has_gradients():
    img = np.zeros((12, 12, 3), np.float32)
    field = _init.build_field(
        img,
        InitConfig(strategy="random", num_gaussians=8, opacity_mode="constant", seed=0),
    ).trainable()
    rate = differentiable_rate_bpp(field, 12, 12, FitConfig(lambda_rate=0.1))
    rate.backward()

    assert float(rate.detach()) > 0.0
    assert field.means.grad is not None
    assert field.colors.grad is not None
    assert field.opacities is not None and field.opacities.grad is not None
    assert torch.isfinite(field.means.grad).all()
    assert torch.isfinite(field.colors.grad).all()


def test_fit_color_solve_runs_in_loop_and_fails_closed_for_other_renderers():
    field, _true_colors, target, cfg, _H, _W = _color_solve_fixture()
    out = fit(
        field,
        target,
        FitConfig(
            **{
                **cfg.__dict__,
                "iters": 1,
                "log_every": 1,
                "color_solve_every": 1,
                "lr_means": 0.0,
                "lr_scales": 0.0,
                "lr_rot": 0.0,
                "lr_color": 0.0,
            }
        ),
        verbose=False,
    )

    events = out["history"]["color_solve_events"]
    assert len(events) == 1
    assert np.isfinite(out["psnr"])
    assert out["psnr"] > 70.0

    if torch.cuda.is_available():
        cuda_field, _true_colors, cuda_target, cuda_cfg, _H, _W = _color_solve_fixture()
        cuda_out = fit(
            _field_to(cuda_field, "cuda"),
            cuda_target.to("cuda"),
            FitConfig(
                **{
                    **cuda_cfg.__dict__,
                    "renderer": "cuda",
                    "iters": 1,
                    "log_every": 1,
                    "color_solve_every": 1,
                    "lr_means": 0.0,
                    "lr_scales": 0.0,
                    "lr_rot": 0.0,
                    "lr_color": 0.0,
                }
            ),
            verbose=False,
        )
        assert len(cuda_out["history"]["color_solve_events"]) == 1
        assert np.isfinite(cuda_out["psnr"])

    bad_field, _true_colors, bad_target, _cfg, _H, _W = _color_solve_fixture()
    with pytest.raises(ValueError, match="normalized renderers"):
        fit(
            bad_field,
            bad_target,
            FitConfig(iters=1, color_solve_every=1, renderer="additive"),
            verbose=False,
        )
    affine_field, _true_colors, affine_target, _cfg, _H, _W = _color_solve_fixture()
    with pytest.raises(ValueError, match="color_basis='constant'"):
        fit(
            affine_field,
            affine_target,
            FitConfig(iters=1, color_basis="affine", color_solve_every=1),
            verbose=False,
        )


def _adaptive_fixture(n=8):
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=n, seed=3))
    return img, target, field


def test_adaptive_count_stops_when_target_psnr_reached_without_growth():
    _img, target, field = _adaptive_fixture(8)
    out = fit(
        field,
        target,
        FitConfig(
            iters=5,
            log_every=1,
            target_psnr=0.0,
            adaptive_count=True,
            max_gaussians=16,
            adaptive_growth_every=1,
            adaptive_growth_count=4,
            ssim_weight=0.0,
        ),
        verbose=False,
    )

    assert out["adaptive_stop_reason"] == "target_psnr_reached"
    assert out["n_gaussians"] == 8
    assert out["history"]["adaptive_events"][0]["action"] == "stop"
    assert out["history"]["adaptive_selected_n"] == 8


def test_adaptive_count_stops_at_max_gaussians():
    _img, target, field = _adaptive_fixture(8)
    out = fit(
        field,
        target,
        FitConfig(
            iters=5,
            log_every=1,
            target_psnr=99.0,
            adaptive_count=True,
            max_gaussians=8,
            adaptive_growth_every=1,
            adaptive_growth_count=4,
            ssim_weight=0.0,
        ),
        verbose=False,
    )

    assert out["adaptive_stop_reason"] == "max_gaussians_reached"
    assert out["n_gaussians"] == 8


def test_adaptive_count_stops_at_target_bpp_cap():
    _img, target, field = _adaptive_fixture(8)
    cap = estimated_raw_bpp(field, *target.shape[:2])
    out = fit(
        field,
        target,
        FitConfig(
            iters=5,
            log_every=1,
            target_psnr=99.0,
            target_bpp=cap,
            adaptive_count=True,
            adaptive_growth_every=1,
            adaptive_growth_count=4,
            ssim_weight=0.0,
        ),
        verbose=False,
    )

    assert out["adaptive_stop_reason"] == "target_bpp_reached"
    assert out["n_gaussians"] == 8
    assert out["estimated_bpp"] == pytest.approx(cap)


def test_adaptive_count_grows_then_stops_on_stall():
    _img, target, field = _adaptive_fixture(8)
    out = fit(
        field,
        target,
        FitConfig(
            iters=6,
            log_every=1,
            target_psnr=99.0,
            adaptive_count=True,
            max_gaussians=12,
            adaptive_growth_every=1,
            adaptive_growth_count=2,
            adaptive_split_mode="residual_add",
            adaptive_min_delta_psnr=1e6,
            adaptive_patience=1,
            ssim_weight=0.0,
        ),
        verbose=False,
    )

    events = out["history"]["adaptive_events"]
    assert events[0]["action"] == "grow"
    assert events[0]["added"] == 2
    assert out["adaptive_stop_reason"] == "stalled"
    assert 10 <= out["n_gaussians"] <= 12


def test_fit_affine_color_basis_adds_trainable_coefficients():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, :, 0] = np.linspace(0.0, 1.0, 16)[None, :]
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="grid", num_gaussians=8, seed=0))
    out = fit(
        field,
        target,
        FitConfig(iters=3, log_every=1, color_basis="affine", ssim_weight=0.0),
        verbose=False,
    )
    assert out["field"].color_grads is not None
    assert out["field"].color_grads.requires_grad
    assert torch.isfinite(out["field"].color_grads).all()
    assert np.isfinite(out["psnr"])


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
    assert out["stopped_at"] == out["history"]["iter"][-1]


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


def test_residual_add_min_spacing_suppresses_same_error_cluster():
    H, W = 32, 32
    target = torch.zeros(H, W, 3)
    render_img = torch.zeros(H, W, 3)
    target[8:11, 8:11] = 1.0
    target[8, 24] = 0.95
    target[24, 8] = 0.9
    target[24, 24] = 0.85
    field = _init.build_field(
        target.numpy(), InitConfig(strategy="random", num_gaussians=8, seed=0))
    cfg = FitConfig(
        split_count=4,
        split_mode="residual_add",
        split_min_spacing=0.6,
        split_oversample=32.0,
        max_gaussians=64,
    )

    torch.manual_seed(0)
    grown, k = _add_from_residual(field, target, render_img, cfg, tensor_aligned=False)
    assert k == 4
    children = grown.means[-k:].detach()
    pixels = children.round().to(torch.long)
    in_cluster = (
        (pixels[:, 0] >= 8) & (pixels[:, 0] <= 10)
        & (pixels[:, 1] >= 8) & (pixels[:, 1] <= 10)
    )
    base_scale = math.sqrt((H * W) / (field.n + k)) * cfg.split_scale

    assert int(in_cluster.sum()) == 1
    assert float(torch.pdist(children).min()) >= cfg.split_min_spacing * base_scale - 1.0


def test_residual_add_color_init_is_configurable_and_additive_safe():
    H, W = 12, 12
    target = torch.zeros(H, W, 3)
    target[6, 6, 0] = 0.8
    render_img = torch.zeros(H, W, 3)
    render_img[6, 6, 0] = 0.5
    field = _init.build_field(
        target.numpy(), InitConfig(strategy="random", num_gaussians=4, seed=0))
    base = dict(split_count=1, split_mode="residual_add", max_gaussians=16)

    target_field, k = _add_from_residual(
        field, target, render_img, FitConfig(renderer="normalized", **base),
        tensor_aligned=False,
    )
    residual_field, _ = _add_from_residual(
        field, target, render_img,
        FitConfig(renderer="normalized", split_color_init="residual", **base),
        tensor_aligned=False,
    )
    additive_field, _ = _add_from_residual(
        field, target, render_img,
        FitConfig(renderer="additive", split_color_init="target", **base),
        tensor_aligned=False,
    )

    assert k == 1
    assert torch.allclose(target_field.colors[-1, 0], torch.tensor(0.8), atol=1e-5)
    assert torch.allclose(residual_field.colors[-1, 0], torch.tensor(1.1), atol=1e-5)
    assert torch.allclose(additive_field.colors[-1, 0], torch.tensor(0.3), atol=1e-5)


def test_freq_violation_scores_oversized_edge_before_smooth_region():
    H, W = 32, 32
    target = torch.zeros(H, W, 3)
    target[:, 16:] = 1.0
    render_img = torch.clamp(target + 0.1, 0.0, 1.0)
    field = GaussianField(
        means=torch.tensor([[16.0, 16.0], [8.0, 8.0]]),
        log_scales=torch.log(torch.full((2, 2), 8.0)),
        rotations=torch.zeros(2),
        colors=torch.zeros(2, 3),
    )

    score, split_axis, components = _freq_violation_scores(
        field,
        target,
        render_img,
        FitConfig(split_mode="freq_violation", split_count=1),
    )

    assert score[0] > score[1]
    assert components["freq"][0] > components["freq"][1]
    assert int(split_axis[0]) == 0  # x-axis crosses the vertical step edge


def test_freq_violation_split_logs_stats_and_respects_max_gaussians():
    img = np.zeros((32, 32, 3), np.float32)
    img[:, 16:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(
        img,
        InitConfig(strategy="random", num_gaussians=12, seed=0),
    )

    out = fit(
        field,
        target,
        FitConfig(
            iters=6,
            log_every=1,
            split_every=3,
            split_count=4,
            split_mode="freq_violation",
            max_gaussians=16,
        ),
        verbose=False,
    )

    assert out["n_gaussians"] <= 16
    events = [e for e in out["history"]["split_events"] if e["mode"] == "freq_violation"]
    assert len(events) == 1
    event = events[0]
    assert event["added"] == 4
    assert "freq_violation_score_mean" in event
    assert "freq_violation_score_max" in event
    assert event["freq_violation_axis0_count"] + event["freq_violation_axis1_count"] == 4


def test_fp_duplicate_has_small_immediate_psnr_dip():
    from structsplat.render import render_field

    img = np.zeros((32, 32, 3), np.float32)
    img[:, :16] = [0.2, 0.4, 0.8]
    img[:, 16:] = [0.9, 0.6, 0.1]
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=48,
                                              seed=0))
    cfg = FitConfig(renderer="normalized", split_count=8, split_mode="fp_duplicate",
                    max_gaussians=80, render_chunk=32)
    before = render_field(field.means, field.conics(), field.colors, field.radii(3.0),
                          32, 32, chunk=32, mode="normalized", opacities=field.opacity_values())
    before_psnr = M.psnr(before, target)

    grown, k = _split_from_residual(field, target, before, cfg)
    after = render_field(grown.means, grown.conics(), grown.colors, grown.radii(3.0),
                         32, 32, chunk=32, mode="normalized", opacities=grown.opacity_values())
    after_psnr = M.psnr(after, target)

    assert k == 8
    assert after_psnr >= before_psnr - 0.05
    assert grown.opacities is not None


def _field_covariance(field: GaussianField) -> torch.Tensor:
    scales = field.scales()
    c = torch.cos(field.rotations)
    s = torch.sin(field.rotations)
    r = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1),
    ], dim=-2)
    d = torch.zeros(field.n, 2, 2)
    d[:, 0, 0] = scales[:, 0] ** 2
    d[:, 1, 1] = scales[:, 1] ** 2
    return r @ d @ r.transpose(-1, -2)


def test_moment_preserving_split_preserves_mass_mean_and_covariance():
    mean = torch.tensor([[20.0, 20.0]])
    log_scales = torch.log(torch.tensor([[4.0, 2.0]]))
    rotations = torch.tensor([0.37])
    field = GaussianField(mean, log_scales, rotations, torch.ones(1, 3))
    parent_cov = _field_covariance(field)[0]

    grown, k = _moment_preserving_duplicate_indices(
        field,
        torch.tensor([0]),
        FitConfig(split_mode="moment_preserving", split_scale=0.7),
        H=64,
        W=64,
    )

    assert k == 1
    opac = grown.opacity_values()
    weights = opac / opac.sum()
    mix_mean = (weights[:, None] * grown.means).sum(dim=0)
    covs = _field_covariance(grown)
    centered = grown.means - mix_mean
    mix_cov = torch.sum(
        weights[:, None, None] * (covs + centered[:, :, None] * centered[:, None, :]),
        dim=0,
    )

    assert torch.allclose(opac.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(mix_mean, mean[0], atol=1e-5)
    assert torch.allclose(mix_cov, parent_cov, atol=1e-4)


def test_moment_preserving_split_has_no_worse_dip_than_fp_duplicate():
    from structsplat.render import render_field

    img = np.zeros((32, 32, 3), np.float32)
    img[:, :16] = [0.2, 0.4, 0.8]
    img[:, 16:] = [0.9, 0.6, 0.1]
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="aniso_flanking", num_gaussians=48,
                                              seed=0))
    base = dict(renderer="normalized", split_count=8, max_gaussians=80, render_chunk=32)
    before = render_field(field.means, field.conics(), field.colors, field.radii(3.0),
                          32, 32, chunk=32, mode="normalized", opacities=field.opacity_values())
    before_psnr = M.psnr(before, target)

    fp, _ = _split_from_residual(
        field, target, before, FitConfig(split_mode="fp_duplicate", **base)
    )
    mp, _ = _split_from_residual(
        field, target, before, FitConfig(split_mode="moment_preserving", **base)
    )
    fp_after = render_field(fp.means, fp.conics(), fp.colors, fp.radii(3.0),
                            32, 32, chunk=32, mode="normalized", opacities=fp.opacity_values())
    mp_after = render_field(mp.means, mp.conics(), mp.colors, mp.radii(3.0),
                            32, 32, chunk=32, mode="normalized", opacities=mp.opacity_values())
    fp_drop = min(0.0, M.psnr(fp_after, target) - before_psnr)
    mp_drop = min(0.0, M.psnr(mp_after, target) - before_psnr)

    assert mp_drop >= fp_drop - 1e-4


def test_moment_preserving_split_runs_under_additive_renderer():
    img = np.zeros((20, 20, 3), np.float32)
    img[:, 10:] = 0.5
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=8, seed=0))

    out = fit(
        field,
        target,
        FitConfig(
            renderer="additive",
            iters=5,
            log_every=1,
            split_every=2,
            split_count=2,
            split_mode="moment_preserving",
            max_gaussians=12,
        ),
        verbose=False,
    )

    assert np.isfinite(out["psnr"])
    assert out["n_gaussians"] == 12


def test_ranked_wave_grows_exactly_and_logs_components():
    img = np.zeros((20, 20, 3), np.float32)
    img[:, 10:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=12, seed=0))
    out = fit(
        field,
        target,
        FitConfig(
            iters=5,
            log_every=1,
            split_every=2,
            split_count=3,
            split_mode="ranked_wave",
            max_gaussians=18,
        ),
        verbose=False,
    )

    assert out["n_gaussians"] == 18
    events = out["history"]["split_events"]
    assert [e["added"] for e in events] == [3, 3]
    assert all(e["mode"] == "ranked_wave" for e in events)
    for key in ("score_mean", "residual_support_mean", "activity_mean", "footprint_mean"):
        assert all(key in e and np.isfinite(e[key]) for e in events)


def test_absgrad_wave_grows_exactly_and_logs_scores():
    img = np.zeros((20, 20, 3), np.float32)
    img[:, 10:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=12, seed=0))
    out = fit(
        field,
        target,
        FitConfig(
            iters=5,
            log_every=1,
            split_every=2,
            split_count=3,
            split_mode="absgrad_wave",
            max_gaussians=18,
        ),
        verbose=False,
    )

    assert out["n_gaussians"] == 18
    events = out["history"]["split_events"]
    assert [e["added"] for e in events] == [3, 3]
    assert all(e["mode"] == "absgrad_wave" for e in events)
    assert all(e["absgrad_score_mean"] >= 0 and e["absgrad_score_max"] >= 0 for e in events)


def test_adan_fit_runs_and_reports_finite_metrics():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=12, seed=0))
    out = fit(field, target, FitConfig(iters=4, log_every=1, optimizer="adan"), verbose=False)
    assert np.isfinite(out["psnr"])
    assert np.isfinite(out["loss"] if "loss" in out else out["history"]["loss"][-1])


def test_relocation_keeps_count_and_moves_low_activity_rows():
    H, W = 20, 20
    target = torch.zeros(H, W, 3)
    target[15, 15] = 1.0
    render_img = torch.zeros(H, W, 3)
    means = np.array([[2.0, 2.0], [6.0, 2.0], [10.0, 2.0], [14.0, 2.0]], dtype=np.float32)
    scales = np.full((4, 2), 1.0, dtype=np.float32)
    colors = np.full((4, 3), 0.5, dtype=np.float32)
    opacities = np.array([8.0, 8.0, -12.0, -12.0], dtype=np.float32)
    field = GaussianField.from_numpy(means, scales, np.zeros(4), colors, opacities=opacities)

    moved, k, reset_idx, stats = _relocate_from_residual(
        field,
        target,
        render_img,
        FitConfig(relocate_count=2, split_min_spacing=0.0),
    )

    assert k == 2
    assert moved.n == field.n
    assert reset_idx is not None and set(reset_idx.tolist()) == {2, 3}
    assert torch.allclose(moved.means[reset_idx],
                          torch.tensor([[15.0, 15.0], [0.0, 0.0]]),
                          atol=1e-5)
    assert moved.opacities is not None
    assert torch.allclose(torch.sigmoid(moved.opacities[reset_idx]),
                          torch.full((2,), 0.05), atol=1e-5)
    assert stats["residual_mean"] > 0


def test_coarse_residual_candidates_snap_to_best_full_res_pixel():
    scores = torch.zeros(8, 8)
    scores[2, 3] = 4.0
    scores[6, 7] = 9.0

    idx = _residual_candidate_pixels(scores, k=2, min_spacing=0.0, oversample=1.0, downsample=4)

    assert idx.tolist() == [6 * 8 + 7, 2 * 8 + 3]


def test_fit_relocation_keeps_n_constant():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=10, seed=0))
    out = fit(
        field,
        target,
        FitConfig(iters=5, log_every=1, relocate_every=2, relocate_count=3),
        verbose=False,
    )

    assert out["n_gaussians"] == 10
    assert out["history"]["relocate_events"]
    assert all(n == 10 for n in out["history"]["n_gaussians"])


def test_fit_relocation_can_follow_split_schedule():
    img = np.zeros((16, 16, 3), np.float32)
    img[:, 8:] = 1.0
    target = torch.as_tensor(img)
    field = _init.build_field(img, InitConfig(strategy="random", num_gaussians=10, seed=0))
    out = fit(
        field,
        target,
        FitConfig(
            iters=6,
            log_every=1,
            split_every=2,
            split_count=2,
            split_mode="residual_add",
            max_gaussians=14,
            relocate_at_split=True,
            relocate_count=1,
            relocate_residual_downsample=2,
        ),
        verbose=False,
    )

    events = out["history"]["relocate_events"]
    assert [e["iter"] for e in events] == [1, 3]
    assert all(e["trigger"] == "split" for e in events)
    assert all(e["residual_downsample"] == 2.0 for e in events)
    assert out["n_gaussians"] == 14


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
