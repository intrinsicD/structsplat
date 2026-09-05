"""Tiny FIT-051 CPU oracles and marked CUDA diagnostics; no research outcomes."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest
import torch

import structsplat.actual_color_ray as ray
from structsplat.config import FitConfig
from structsplat.fit import _MaskConstraint, _render, _solve_colors_normalized
from structsplat.gaussians import GaussianField
from structsplat.safe_schedule import SafeScheduleConfig, evaluate_quality


DIRECTIONS = ("cg", "streaming_gradient", "streaming_jacobi", "native_gradient")


def _fixture(dtype=torch.float64):
    field = GaussianField(
        torch.tensor([[2., 3.], [6., 5.]], dtype=dtype),
        torch.full((2, 2), math.log(1.2), dtype=dtype),
        torch.tensor([.2, -.4], dtype=dtype), torch.full((2, 3), .03, dtype=dtype),
        torch.tensor([.3, -.2], dtype=dtype), torch.full((2, 2), 4., dtype=dtype),
        background_mask=torch.tensor([False, False]),
        filter_variance=torch.full((2,), .08, dtype=dtype),
    )
    cfg = FitConfig(renderer="normalized", pixel_loss="l2", ssim_weight=0.,
                    support_fade=True, aa_dilation=.04, color_solve_lambda=1e-4,
                    render_chunk=1)
    truth = field.detached()
    truth.colors.copy_(torch.tensor([[.2, .7, .3], [.8, .2, .6]], dtype=dtype))
    target = _render(truth, cfg, 9, 9, support_fade_alpha=1.)
    mask = torch.ones(9, 9, dtype=torch.bool)
    constraint = _MaskConstraint.from_mask(
        mask.numpy(), target.device, dtype, cfg.sigma_cutoff, cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode, undercoverage_band=cfg.mask_undercoverage_band)
    schedule = SafeScheduleConfig(capacity=2, coverage_target_gaussians=2,
                                  detail_target_gaussians=2)
    return field, target, cfg, mask, constraint, schedule


def _dense_basis(field, cfg):
    """Independent dense covariance/opacity/fade oracle, restricted to the 9x9 fixture."""
    yy, xx = np.mgrid[:9, :9]
    columns = []
    for row in range(field.n):
        theta = float(field.rotations[row])
        rotation = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
        variance = np.exp(2 * field.log_scales[row].numpy()) + cfg.aa_dilation
        variance += float(field.filter_variance[row])
        covariance = rotation @ np.diag(variance) @ rotation.T
        delta = np.stack([xx - float(field.means[row, 0]),
                          yy - float(field.means[row, 1])], axis=-1)
        distance = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta)
        weight = np.maximum(np.exp(-.5 * distance) - math.exp(-.5 * cfg.sigma_cutoff**2), 0.)
        weight *= 1 / (1 + math.exp(-float(field.opacities[row])))
        columns.append(weight.reshape(-1))
    raw = np.stack(columns, axis=1)
    return torch.as_tensor(raw / (raw.sum(axis=1, keepdims=True) + cfg.normalization_eps),
                           dtype=field.colors.dtype)


def _assert_owned(result, source, *, equal_colors=False):
    for name in ("means", "log_scales", "rotations", "colors", "opacities", "scale_max",
                 "color_grads", "background_mask", "filter_variance"):
        value, original = getattr(result, name), getattr(source, name)
        if original is None:
            assert value is None
            continue
        assert value.data_ptr() != original.data_ptr()
        assert not value.requires_grad
        if name != "colors" or equal_colors:
            torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize("direction", DIRECTIONS[1:])
def test_gradient_and_actual_line_minimizer_match_dense_float64(direction):
    field, target, cfg, mask, constraint, schedule = _fixture()
    dense = _dense_basis(field, cfg)
    residual = target.reshape(-1, 3) - dense @ field.colors
    expected_gradient = dense.T @ residual
    _, _, metadata, tensors = ray.refine_actual_color_ray(
        field, target, cfg, mask, constraint, schedule, direction=direction)
    torch.testing.assert_close(tensors["gradient"], expected_gradient, rtol=3e-13, atol=3e-13)
    expected_vector = expected_gradient
    if direction == "streaming_jacobi":
        diagonal = dense.square().sum(dim=0)
        torch.testing.assert_close(tensors["diagonal"], diagonal, rtol=3e-13, atol=3e-13)
        expected_vector = expected_gradient / (diagonal[:, None] + cfg.color_solve_lambda)
    torch.testing.assert_close(tensors["direction"], expected_vector, rtol=3e-13, atol=3e-13)
    q = dense @ expected_vector
    torch.testing.assert_close(tensors["direction_render"].reshape(-1, 3), q, rtol=3e-13, atol=3e-13)
    expected_alpha = float((residual * q).sum() / (
        q.square().sum() + cfg.color_solve_lambda * expected_vector.square().sum()))
    assert metadata["alpha_star"] == pytest.approx(expected_alpha, rel=3e-13)
    def objective(alpha):
        return float((residual - alpha * q).square().sum()
                     + cfg.color_solve_lambda * (alpha * expected_vector).square().sum())
    epsilon = 1e-4
    derivative = (objective(expected_alpha + epsilon) - objective(expected_alpha - epsilon)) / (2 * epsilon)
    assert abs(derivative) < 1e-9
    assert objective(expected_alpha) <= objective(expected_alpha * .99)
    assert objective(expected_alpha) <= objective(expected_alpha * 1.01)


def test_native_vjp_matches_finite_difference_and_touches_no_caller_gradients():
    field, target, cfg, *_ = _fixture()
    for value in (field.means, field.log_scales, field.rotations, field.colors, field.opacities):
        value.requires_grad_(True)
        value.grad = torch.full_like(value, 7.)
    residual = (target - _render(field, cfg, 9, 9, support_fade_alpha=1.)).detach()
    gradient, native = ray._native_color_gradient(field, residual, cfg)
    probe = torch.arange(6, dtype=torch.float64).reshape(2, 3) / 5 - .5
    plus, minus = field.detached(), field.detached()
    plus.colors.add_(1e-6 * probe)
    minus.colors.sub_(1e-6 * probe)
    positive = (_render(plus, cfg, 9, 9, support_fade_alpha=1.) - target).square().sum()
    negative = (_render(minus, cfg, 9, 9, support_fade_alpha=1.) - target).square().sum()
    assert float((positive - negative) / 2e-6) == pytest.approx(
        float(-2 * (gradient * probe).sum()), rel=2e-8, abs=1e-8)
    assert not gradient.requires_grad and not native.requires_grad
    for value in (field.means, field.log_scales, field.rotations, field.colors, field.opacities):
        assert value.requires_grad
        torch.testing.assert_close(value.grad, torch.full_like(value, 7.), rtol=0, atol=0)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_transaction_replays_owned_state_and_exact_work(direction, dtype):
    args = _fixture(dtype)
    snapshot = args[0].detached()
    selected, quality, metadata, tensors = ray.refine_actual_color_ray(*args, direction=direction)
    assert metadata["accepted"] and metadata["foreground_mse_improved"]
    assert metadata["selected_fraction"] > 0 and metadata["coefficients_changed"]
    _assert_owned(selected, args[0])
    _assert_owned(args[0], snapshot, equal_colors=True)
    actual, raw = evaluate_quality(selected, args[1], args[3], args[2], args[4], args[5].coverage_tau)
    assert actual == quality
    torch.testing.assert_close(raw, tensors["replay_render"], rtol=0, atol=0)
    trial_count = len(metadata["trials"])
    counts = metadata["counts"]
    assert counts["quality_evaluations"] == counts["raw_coverage_passes"] == trial_count + 2
    assert counts["actual_direction_render_calls"] == 1
    assert counts["native_gradient_forward_calls"] == counts["native_color_vjp_calls"] == int(direction == "native_gradient")
    assert counts["gaussian_renders"] == trial_count + 3 + int(direction == "native_gradient")
    assert counts["basis_diagonal_passes"] == int(direction == "streaming_jacobi")
    if direction == "cg":
        stats = metadata["legacy_cg"]
        assert counts["basis_apply_calls"] == stats["basis_apply_calls"]
        assert counts["basis_transpose_calls"] == stats["basis_transpose_calls"]
        assert counts["basis_denominator_passes"] == stats["denominator_calls"]
        assert counts["legacy_cg_iterations"] == stats["iterations"]
    else:
        assert counts["basis_apply_calls"] == counts["legacy_cg_iterations"] == 0
        assert counts["basis_transpose_calls"] == counts["basis_denominator_passes"] == int(direction != "native_gradient")
    assert len(tensors["trial_renders"]) == len(tensors["trial_denominators"]) == trial_count
    for trial, trial_raw, trial_den in zip(metadata["trials"], tensors["trial_renders"], tensors["trial_denominators"], strict=True):
        replayed = ray._quality_from_render(trial_raw, args[1], trial_den, args[3], args[4],
                                            args[5].coverage_tau, selected.n)
        assert replayed.to_dict() == trial["actual_metrics"]
        assert trial["raw_sse"] == float((trial_raw - args[1]).square().sum())
    assert metadata["replay_max_abs_error"] <= 2e-5
    assert metadata["elapsed_seconds"] >= sum(metadata["phase_seconds"].values())
    assert metadata["image_interpolation"] is False
    json.dumps(metadata, allow_nan=False)


def test_cg_first_actual_trial_is_exact_independently_solved_endpoint(monkeypatch):
    args = _fixture(torch.float32)
    endpoint = args[0].detached()
    _solve_colors_normalized(endpoint, args[1], replace(args[2], color_solve_maxiter=32), 9, 9,
                             support_fade_alpha=1.)
    original, actual_colors = ray._quality_render_inputs, []
    def capture(field, *a, **k):
        actual_colors.append(field.colors.clone())
        return original(field, *a, **k)
    monkeypatch.setattr(ray, "_quality_render_inputs", capture)
    _, _, metadata, tensors = ray.refine_actual_color_ray(*args, direction="cg")
    assert metadata["alpha_star"] == metadata["trials"][0]["alpha"] == 1.
    assert metadata["cg_endpoint_exact_alpha1"]
    torch.testing.assert_close(actual_colors[1], endpoint.colors, rtol=0, atol=0)
    torch.testing.assert_close(tensors["cg_endpoint_colors"], endpoint.colors, rtol=0, atol=0)


def test_negative_signed_direction_is_not_clamped():
    args = list(_fixture())
    args[0].colors.fill_(.9)
    args[1].mul_(.1)
    _, _, metadata, tensors = ray.refine_actual_color_ray(*args)
    assert metadata["accepted"]
    assert bool((tensors["direction"] < 0).all())
    assert bool((tensors["direction_render"] < 0).any())
    assert metadata["alpha_star"] > 0


def test_zero_support_column_has_zero_streaming_jacobi_direction():
    args = list(_fixture())
    args[0].means[1] = 1000.
    args[2] = replace(args[2], color_solve_lambda=0.)
    selected, _, metadata, tensors = ray.refine_actual_color_ray(*args, direction="streaming_jacobi")
    assert tensors["diagonal"][1] == 0.
    torch.testing.assert_close(tensors["direction"][1], torch.zeros_like(args[0].colors[1]), rtol=0, atol=0)
    torch.testing.assert_close(selected.colors[1], args[0].colors[1], rtol=0, atol=0)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("stage", ["parent", "trial", "replay"])
def test_nonfinite_quality_reduction_fails_closed_without_changing_metric_vector(monkeypatch, stage):
    args = _fixture()
    original, calls = ray._quality_from_render, []
    def overflow(*a, **k):
        quality = original(*a, **k)
        calls.append(1)
        bad = ((stage == "parent" and len(calls) == 1)
               or (stage == "trial" and len(calls) > 1)
               or (stage == "replay" and len(calls) == 3))
        return replace(quality, cvar99_mse=float("nan")) if bad else quality
    monkeypatch.setattr(ray, "_quality_from_render", overflow)
    selected, _, metadata, _ = ray.refine_actual_color_ray(*args)
    assert not metadata["accepted"]
    assert metadata["rollback_reason"] == {
        "parent": "nonfinite_parent_render", "trial": "all_trials_rejected",
        "replay": "selected_replay_failed"}[stage]
    _assert_owned(selected, args[0], equal_colors=True)
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("metric", ["foreground_mse", "boundary_mse", "cvar99_mse", "p99_mse",
    "interior_hole_fraction", "boundary_hole_fraction", "outside_max_abs", "outside_coverage_max", "finite"])
def test_each_complete_quality_guard_rejects_all_actual_trials(monkeypatch, metric):
    args = list(_fixture())
    args[5] = replace(args[5], tolerances=replace(args[5].tolerances, guard_p99=True))
    original, calls = ray._quality_from_render, []
    def reject(*a, **k):
        quality = original(*a, **k)
        calls.append(1)
        return quality if len(calls) == 1 else replace(quality, **{metric: False if metric == "finite" else 100.})
    monkeypatch.setattr(ray, "_quality_from_render", reject)
    selected, _, metadata, tensors = ray.refine_actual_color_ray(*args)
    assert metadata["rollback_reason"] == "all_trials_rejected"
    assert [trial["fraction"] for trial in metadata["trials"]] == [1., .5, .25, .125, .0625, .03125]
    assert all(not trial["accepted"] and trial["reasons"] for trial in metadata["trials"])
    assert metadata["counts"]["quality_evaluations"] == len(calls) == 7
    assert len(tensors["trial_renders"]) == len(tensors["trial_denominators"]) == 6
    assert tensors["replay_render"] is None
    _assert_owned(selected, args[0], equal_colors=True)


@pytest.mark.parametrize("failure", ["pixels", "gate"])
def test_selected_replay_failure_restores_exact_parent_without_retry(monkeypatch, failure):
    args = _fixture()
    if failure == "pixels":
        original = ray._quality_render_inputs
        calls = []
        def invalidate(*a, **k):
            raw, den = original(*a, **k)
            calls.append(1)
            return (raw + 1e-3 if len(calls) == 3 else raw), den
        monkeypatch.setattr(ray, "_quality_render_inputs", invalidate)
    else:
        original = ray._quality_from_render
        calls = []
        def invalidate(*a, **k):
            quality = original(*a, **k)
            calls.append(1)
            return replace(quality, foreground_mse=100.) if len(calls) == 3 else quality
        monkeypatch.setattr(ray, "_quality_from_render", invalidate)
    selected, _, metadata, tensors = ray.refine_actual_color_ray(*args)
    assert metadata["rollback_reason"] == "selected_replay_failed"
    assert not metadata["accepted"] and len(metadata["trials"]) == 1 and len(calls) == 3
    assert metadata["selected_fraction"] == 0. and tensors["replay_render"] is not None
    _assert_owned(selected, args[0], equal_colors=True)


@pytest.mark.parametrize("failure", ["zero", "nan", "bad_q", "negative_alpha", "parent"])
def test_early_invalid_proposals_record_exact_rollbacks(monkeypatch, failure):
    args = list(_fixture())
    if failure == "zero":
        args[0].colors.zero_()
        args[1].zero_()
        reason = "invalid_or_zero_direction"
    elif failure == "nan":
        monkeypatch.setattr(ray, "_native_color_gradient", lambda *a: (
            torch.full_like(args[0].colors, float("nan")), torch.zeros_like(args[1])))
        reason = "invalid_or_zero_direction"
    elif failure in {"bad_q", "negative_alpha"}:
        original = ray._render
        def changed(field, *a, **k):
            if field.colors.requires_grad:
                return original(field, *a, **k)
            return (torch.full_like(args[1], float("nan")) if failure == "bad_q"
                    else -original(field, *a, **k))
        monkeypatch.setattr(ray, "_render", changed)
        reason = "nonfinite_direction_render" if failure == "bad_q" else "invalid_line_minimizer"
    else:
        original = ray._quality_from_render
        monkeypatch.setattr(ray, "_quality_from_render", lambda *a, **k: replace(original(*a, **k), finite=False))
        reason = "nonfinite_parent_render"
    selected, _, metadata, tensors = ray.refine_actual_color_ray(*args)
    assert metadata["rollback_reason"] == reason and not metadata["trials"]
    assert tensors["replay_render"] is None
    _assert_owned(selected, args[0], equal_colors=True)
    json.dumps(metadata, allow_nan=False)


def test_actual_trials_do_not_depend_on_direction_image_interpolation(monkeypatch):
    args = _fixture()
    original, seen_colors = ray._quality_render_inputs, []
    def capture(field, *a, **k):
        seen_colors.append(field.colors.clone())
        return original(field, *a, **k)
    monkeypatch.setattr(ray, "_quality_render_inputs", capture)
    monkeypatch.setattr(ray, "safe_commit_decision", lambda *a: (False, ["fixture_rejection"]))
    _, _, metadata, tensors = ray.refine_actual_color_ray(*args, direction="streaming_gradient")
    for index, trial in enumerate(metadata["trials"]):
        candidate = args[0].detached()
        candidate.colors.add_(trial["alpha"] * tensors["direction"])
        torch.testing.assert_close(seen_colors[index + 1], candidate.colors, rtol=0, atol=0)
        actual = _render(candidate, args[2], 9, 9, support_fade_alpha=1.)
        torch.testing.assert_close(tensors["trial_renders"][index], actual, rtol=0, atol=0)
    assert len(seen_colors) == 7


def test_reference_quality_backends_are_forced_without_mutating_config(monkeypatch):
    args = list(_fixture())
    args[2] = replace(args[2], quality_coverage_backend="renderer", quality_tail_backend="shared")
    original, seen = ray._quality_render_inputs, []
    def check(field, cfg, *a, **k):
        seen.append((cfg.quality_coverage_backend, cfg.quality_tail_backend))
        return original(field, cfg, *a, **k)
    monkeypatch.setattr(ray, "_quality_render_inputs", check)
    ray.refine_actual_color_ray(*args)
    assert seen and set(seen) == {("reference", "reference")}
    assert args[2].quality_coverage_backend == "renderer" and args[2].quality_tail_backend == "shared"


@pytest.mark.parametrize("bad", ["mask", "target", "field", "loss", "renderer", "trials", "direction", "ridge"])
def test_unsupported_inputs_fail_closed(bad):
    args = list(_fixture())
    kwargs = {}
    if bad == "mask":
        args[3][0, 0] = False
    elif bad == "target":
        args[1][0, 0, 0] = float("nan")
    elif bad == "field":
        args[0].colors[0, 0] = float("inf")
    elif bad == "loss":
        args[2] = replace(args[2], pixel_loss="l1")
    elif bad == "renderer":
        args[2] = replace(args[2], renderer="additive")
    elif bad == "ridge":
        # Construction already rejects negative ridge; exercise the API's independent guard
        # on a subsequently modified (mutable) configuration.
        args[2].color_solve_lambda = -1.
    else:
        kwargs["max_trials" if bad == "trials" else "direction"] = 7 if bad == "trials" else "wrong"
    with pytest.raises(ValueError):
        ray.refine_actual_color_ray(*args, **kwargs)


@pytest.mark.cuda
def test_cuda_native_vjp_signed_linearity_ownership_and_actual_replay():
    """Same-device diagnostics, not reference/CUDA equivalence or formal-image evidence.

    Basis columns are actual CUDA unit-color renders. Finite differences use float32 image
    evaluations with float64 scalar reductions and a 1e-2 step (fixed-color rendering is
    linear, so the squared-error objective is quadratic). The looser derivative tolerance
    accounts for subtraction of float32 images; replay keeps the unchanged 2e-5 check.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    field, _, cfg, mask, _, schedule = _fixture(torch.float32)
    for name in ("means", "log_scales", "rotations", "colors", "opacities", "scale_max",
                 "background_mask", "filter_variance"):
        setattr(field, name, getattr(field, name).to("cuda"))
    cfg = replace(cfg, renderer="cuda")
    constraint = _MaskConstraint.from_mask(
        mask.numpy(), field.colors.device, torch.float32, cfg.sigma_cutoff, cfg.mask_margin,
        cap_mode=cfg.mask_cap_mode, undercoverage_band=cfg.mask_undercoverage_band)
    mask = mask.to("cuda")
    field.colors.fill_(.9)
    truth = field.detached()
    truth.colors.fill_(.15)
    target = _render(truth, cfg, 9, 9, support_fade_alpha=1.)
    parent_image = _render(field, cfg, 9, 9, support_fade_alpha=1.)
    residual = (target - parent_image).detach()
    snapshot = field.detached()
    parameters = (field.means, field.log_scales, field.rotations, field.colors, field.opacities)
    for value in parameters:
        value.requires_grad_(True)
        value.grad = torch.full_like(value, 7.)
    gradient, native_image = ray._native_color_gradient(field, residual, cfg)

    columns = []
    for index in range(field.n):
        unit = field.detached()
        unit.colors.zero_()
        unit.colors[index].fill_(1.)
        columns.append(_render(unit, cfg, 9, 9, support_fade_alpha=1.)[..., 0].reshape(-1))
    dense = torch.stack(columns, dim=1)
    expected = dense.T @ residual.reshape(-1, 3)
    torch.testing.assert_close(gradient, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(native_image.reshape(-1, 3), dense @ field.colors.detach(),
                               rtol=2e-5, atol=2e-5)

    probe = torch.tensor([[-.6, .2, -.3], [-.2, -.4, .1]], device="cuda")
    signed = field.detached()
    signed.colors.copy_(probe)
    q = _render(signed, cfg, 9, 9, support_fade_alpha=1.)
    assert bool((q < 0).any())
    torch.testing.assert_close(q.reshape(-1, 3), dense @ probe, rtol=2e-5, atol=2e-5)
    plus, minus = field.detached(), field.detached()
    plus.colors.add_(1e-2 * probe)
    minus.colors.sub_(1e-2 * probe)
    plus_image = _render(plus, cfg, 9, 9, support_fade_alpha=1.)
    minus_image = _render(minus, cfg, 9, 9, support_fade_alpha=1.)
    torch.testing.assert_close(plus_image, parent_image + 1e-2 * q, rtol=2e-5, atol=2e-5)
    plus_loss = (plus_image.double() - target.double()).square().sum()
    minus_loss = (minus_image.double() - target.double()).square().sum()
    difference = float((plus_loss - minus_loss) / 2e-2)
    assert difference == pytest.approx(float(-2 * (gradient * probe).sum()), rel=2e-3, abs=2e-4)

    selected, quality, metadata, tensors = ray.refine_actual_color_ray(
        field, target, cfg, mask, constraint, schedule, direction="native_gradient")
    assert metadata["accepted"] and metadata["foreground_mse_improved"]
    assert bool((tensors["direction"] < 0).all())
    assert bool((tensors["direction_render"] < 0).any())
    assert metadata["replay_max_abs_error"] <= 2e-5
    actual, replay = evaluate_quality(selected, target, mask, cfg, constraint, schedule.coverage_tau)
    assert actual == quality
    torch.testing.assert_close(replay, tensors["replay_render"], rtol=0, atol=2e-5)
    assert metadata["counts"]["native_color_vjp_calls"] == 1
    assert metadata["counts"]["quality_evaluations"] == len(metadata["trials"]) + 2
    _assert_owned(selected, field)
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        value = getattr(field, name)
        assert value.requires_grad
        torch.testing.assert_close(value.detach(), getattr(snapshot, name), rtol=0, atol=0)
        torch.testing.assert_close(value.grad, torch.full_like(value, 7.), rtol=0, atol=0)
    assert not gradient.requires_grad and not native_image.requires_grad
    json.dumps(metadata, allow_nan=False)
