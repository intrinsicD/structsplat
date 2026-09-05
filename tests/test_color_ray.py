"""Deterministic tiny CPU oracles for FIT-050; no formal-image outcomes."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest
import torch

import structsplat.color_ray as ray
from structsplat.config import FitConfig
from structsplat.fit import _MaskConstraint, _normalized_color_denominator, _render
from structsplat.gaussians import GaussianField
from structsplat.safe_schedule import SafeScheduleConfig, evaluate_quality


def _fixture(dtype=torch.float64, *, identical=False):
    height = width = 9
    means = torch.tensor([[2.0, 3.0], [6.0, 5.0]], dtype=dtype)
    if identical:
        means[1].copy_(means[0])
    field = GaussianField(
        means, torch.full((2, 2), math.log(1.2), dtype=dtype),
        torch.tensor([0.2, -0.4], dtype=dtype), torch.full((2, 3), 0.03, dtype=dtype),
        torch.tensor([0.3, -0.2], dtype=dtype),
        torch.full((2, 2), 4.0, dtype=dtype),
        background_mask=torch.tensor([False, False]),
        filter_variance=torch.full((2,), 0.08, dtype=dtype),
    )
    if identical:
        field.rotations[1].copy_(field.rotations[0])
        field.opacities[1].copy_(field.opacities[0])
    cfg = FitConfig(renderer="normalized", pixel_loss="l2", ssim_weight=0.0,
                    support_fade=True, aa_dilation=0.04, color_solve_lambda=1e-4,
                    render_chunk=1)
    truth = field.detached()
    truth.colors.copy_(torch.tensor([[0.2, 0.7, 0.3], [0.8, 0.2, 0.6]], dtype=dtype))
    target = _render(truth, cfg, height, width, support_fade_alpha=1.0)
    mask = torch.ones(height, width, dtype=torch.bool)
    constraint = _MaskConstraint.from_mask(mask.numpy(), target.device, dtype,
        cfg.sigma_cutoff, cfg.mask_margin, cap_mode=cfg.mask_cap_mode,
        undercoverage_band=cfg.mask_undercoverage_band)
    schedule = SafeScheduleConfig(capacity=2, coverage_target_gaussians=2,
                                  detail_target_gaussians=2)
    return field, target, cfg, mask, constraint, schedule


def _dense_basis(field, cfg, height, width):
    """Independent dense RS covariance/opacity/fade calculation, only for tiny tests."""
    yy, xx = np.mgrid[:height, :width]
    columns = []
    for row in range(field.n):
        theta = float(field.rotations[row])
        rotation = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
        variance = np.exp(2 * field.log_scales[row].numpy()) + cfg.aa_dilation
        if field.filter_variance is not None:
            variance += float(field.filter_variance[row])
        covariance = rotation @ np.diag(variance) @ rotation.T
        delta = np.stack([xx - float(field.means[row, 0]), yy - float(field.means[row, 1])], axis=-1)
        distance = np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta)
        weight = np.maximum(np.exp(-0.5 * distance) - math.exp(-0.5 * cfg.sigma_cutoff**2), 0)
        if field.opacities is not None:
            weight *= 1 / (1 + math.exp(-float(field.opacities[row])))
        columns.append(weight.reshape(-1))
    raw = np.stack(columns, axis=1)
    return torch.as_tensor(raw / (raw.sum(axis=1, keepdims=True) + cfg.normalization_eps),
                           dtype=field.colors.dtype)


def _assert_owned_equal_noncolors(result, source):
    for name in ("means", "log_scales", "rotations", "opacities", "scale_max",
                 "background_mask", "filter_variance"):
        value, original = getattr(result, name), getattr(source, name)
        if original is None:
            assert value is None
        else:
            torch.testing.assert_close(value, original, rtol=0, atol=0)
            assert value.data_ptr() != original.data_ptr()


def test_exact_diagonal_matches_dense_float64_oracle():
    field, target, cfg, *_ = _fixture()
    height, width = target.shape[:2]
    dense = _dense_basis(field, cfg, height, width)
    denominator = _normalized_color_denominator(field, cfg, height, width, support_fade_alpha=1.0)
    diagonal = ray._normalized_color_basis_diagonal(field, cfg, height, width, denominator)
    torch.testing.assert_close(diagonal, dense.square().sum(dim=0), rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("direction", ["gradient", "jacobi"])
def test_line_minimizer_matches_dense_and_finite_difference(direction):
    field, target, cfg, mask, constraint, schedule = _fixture()
    dense = _dense_basis(field, cfg, *target.shape[:2])
    residual = target.reshape(-1, 3) - dense @ field.colors
    gradient = dense.T @ residual
    probe = torch.arange(6, dtype=field.colors.dtype).reshape(2, 3) / 5 - 0.5
    epsilon_gradient = 1e-6
    plus, minus = field.detached(), field.detached()
    plus.colors.add_(epsilon_gradient * probe)
    minus.colors.sub_(epsilon_gradient * probe)
    plus_loss = (_render(plus, cfg, *target.shape[:2], support_fade_alpha=1.0) - target).square().sum()
    minus_loss = (_render(minus, cfg, *target.shape[:2], support_fade_alpha=1.0) - target).square().sum()
    derivative = float((plus_loss - minus_loss) / (2 * epsilon_gradient))
    assert derivative == pytest.approx(float(-2 * (gradient * probe).sum()), rel=2e-8, abs=1e-8)
    vector = gradient if direction == "gradient" else gradient / (
        dense.square().sum(dim=0)[:, None] + cfg.color_solve_lambda)
    image_direction = dense @ vector
    expected = float((residual * image_direction).sum() / (
        image_direction.square().sum() + cfg.color_solve_lambda * vector.square().sum()))
    _, _, metadata = ray.refine_color_ray(field, target, cfg, mask, constraint, schedule,
                                          direction=direction)
    assert metadata["alpha_star"] == pytest.approx(expected, rel=2e-13)
    def objective(alpha):
        return float((residual - alpha * image_direction).square().sum()
                     + cfg.color_solve_lambda * (alpha * vector).square().sum())
    epsilon = 1e-4
    assert abs((objective(expected + epsilon) - objective(expected - epsilon)) / (2 * epsilon)) < 1e-9
    assert objective(expected) <= objective(expected * 0.99)
    assert objective(expected) <= objective(expected * 1.01)


@pytest.mark.parametrize("direction", ["gradient", "jacobi", "cg"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transaction_replays_owns_state_and_accounts_work(direction, dtype):
    field, target, cfg, mask, constraint, schedule = _fixture(dtype)
    snapshot = field.detached()
    selected, quality, metadata = ray.refine_color_ray(field, target, cfg, mask, constraint,
                                                       schedule, direction=direction)
    assert metadata["accepted"]
    assert metadata["selected_fraction"] > 0
    assert metadata["coefficients_changed"]
    assert metadata["foreground_mse_improved"]
    _assert_owned_equal_noncolors(selected, field)
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        torch.testing.assert_close(getattr(field, name), getattr(snapshot, name), rtol=0, atol=0)
    assert selected.colors.data_ptr() != field.colors.data_ptr()
    actual, _ = evaluate_quality(selected, target, mask, cfg, constraint, schedule.coverage_tau)
    assert actual == quality
    counts = metadata["counts"]
    assert counts["gaussian_renders"] == 2
    assert counts["quality_evaluations"] == 2
    assert counts["raw_coverage_passes"] == 3
    assert counts["interpolated_quality_evaluations"] == len(metadata["trials"])
    if direction == "cg":
        cg = metadata["legacy_cg"]
        assert counts["basis_apply_calls"] == 2 + cg["basis_apply_calls"]
        assert counts["basis_transpose_calls"] == cg["basis_transpose_calls"]
        assert counts["basis_denominator_passes"] == 2
    else:
        assert counts["basis_apply_calls"] == 2
        assert counts["basis_transpose_calls"] == 1
        assert counts["basis_denominator_passes"] == 1
    assert counts["basis_diagonal_passes"] == int(direction == "jacobi")
    assert metadata["replay_max_abs_error"] <= 2e-5
    assert metadata["elapsed_seconds"] >= sum(metadata["phase_seconds"].values())
    json.dumps(metadata, allow_nan=False)


def test_gradient_and_jacobi_agree_for_equal_diagonal_columns():
    args = _fixture(identical=True)
    gradient, _, _ = ray.refine_color_ray(*args, direction="gradient")
    jacobi, _, _ = ray.refine_color_ray(*args, direction="jacobi")
    torch.testing.assert_close(gradient.colors, jacobi.colors, rtol=2e-13, atol=2e-13)


def test_zero_support_column_has_zero_diagonal_and_does_not_change():
    field, target, cfg, mask, constraint, schedule = _fixture()
    field.means[1] = 1000
    cfg = replace(cfg, color_solve_lambda=0.0)
    den = _normalized_color_denominator(field, cfg, *target.shape[:2], support_fade_alpha=1.0)
    assert ray._normalized_color_basis_diagonal(field, cfg, *target.shape[:2], den)[1] == 0
    selected, _, metadata = ray.refine_color_ray(field, target, cfg, mask, constraint, schedule)
    torch.testing.assert_close(selected.colors[1], field.colors[1], rtol=0, atol=0)
    json.dumps(metadata, allow_nan=False)


def test_zero_direction_returns_owned_exact_rollback():
    field, target, cfg, mask, constraint, schedule = _fixture()
    field.colors.zero_()
    target.zero_()
    selected, _, metadata = ray.refine_color_ray(field, target, cfg, mask, constraint, schedule)
    assert metadata["rollback_reason"] == "invalid_or_zero_direction"
    assert not metadata["trials"]
    torch.testing.assert_close(selected.colors, field.colors, rtol=0, atol=0)
    _assert_owned_equal_noncolors(selected, field)


def test_nonfinite_direction_is_a_recorded_rollback(monkeypatch):
    args = _fixture()
    monkeypatch.setattr(ray, "_normalized_color_basis_transpose",
                        lambda *a, **k: torch.full_like(args[0].colors, float("nan")))
    selected, _, metadata = ray.refine_color_ray(*args)
    assert metadata["rollback_reason"] == "invalid_or_zero_direction"
    torch.testing.assert_close(selected.colors, args[0].colors, rtol=0, atol=0)
    json.dumps(metadata, allow_nan=False)


def test_all_six_rejections_are_retained(monkeypatch):
    args = _fixture()
    monkeypatch.setattr(ray, "safe_commit_decision", lambda *a: (False, ["test_guard"]))
    selected, _, metadata = ray.refine_color_ray(*args)
    assert metadata["rollback_reason"] == "all_trials_rejected"
    assert [record["fraction"] for record in metadata["trials"]] == [1, .5, .25, .125, .0625, .03125]
    assert metadata["counts"]["gaussian_renders"] == 1
    torch.testing.assert_close(selected.colors, args[0].colors, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["pixels", "gate"])
def test_replay_failure_restores_parent_without_retry(monkeypatch, failure):
    args = _fixture()
    original = ray.evaluate_quality
    calls = []
    def changed_replay(*a, **k):
        metrics, image = original(*a, **k)
        calls.append(1)
        if len(calls) == 2:
            if failure == "pixels":
                image = image + 1e-3
            else:
                metrics = replace(metrics, foreground_mse=100.0)
        return metrics, image
    monkeypatch.setattr(ray, "evaluate_quality", changed_replay)
    selected, _, metadata = ray.refine_color_ray(*args)
    assert metadata["rollback_reason"] == "selected_replay_failed"
    assert len(metadata["trials"]) == 1
    assert len(calls) == 2
    assert not metadata["accepted"]
    torch.testing.assert_close(selected.colors, args[0].colors, rtol=0, atol=0)


def test_basis_parent_parity_failure_happens_before_direction(monkeypatch):
    args = _fixture()
    original = ray._normalized_color_basis_apply
    monkeypatch.setattr(ray, "_normalized_color_basis_apply", lambda *a, **k: original(*a, **k) + 1e-3)
    _, _, metadata = ray.refine_color_ray(*args)
    assert metadata["rollback_reason"] == "basis_parent_parity_failed"
    assert metadata["counts"]["basis_transpose_calls"] == 0
    assert not metadata["trials"]


@pytest.mark.parametrize("bad", ["mask", "target", "field", "loss", "renderer", "trials"])
def test_unsupported_inputs_fail_closed(bad):
    field, target, cfg, mask, constraint, schedule = _fixture()
    kwargs = {}
    if bad == "mask":
        mask[0, 0] = False
    elif bad == "target":
        target[0, 0, 0] = float("nan")
    elif bad == "field":
        field.colors[0, 0] = float("inf")
    elif bad == "loss":
        cfg = replace(cfg, pixel_loss="l1")
    elif bad == "renderer":
        cfg = replace(cfg, renderer="additive")
    else:
        kwargs["max_trials"] = 7
    with pytest.raises(ValueError):
        ray.refine_color_ray(field, target, cfg, mask, constraint, schedule, **kwargs)


def test_reference_coverage_is_forced_even_if_caller_requests_renderer(monkeypatch):
    field, target, cfg, mask, constraint, schedule = _fixture()
    seen = []
    original = ray.evaluate_quality
    def observe(*args, **kwargs):
        seen.append((args[3].quality_coverage_backend, args[3].quality_tail_backend))
        return original(*args, **kwargs)
    monkeypatch.setattr(ray, "evaluate_quality", observe)
    ray.refine_color_ray(field, target, replace(cfg, quality_coverage_backend="renderer", quality_tail_backend="shared"),
                         mask, constraint, schedule)
    assert seen == [("reference", "reference"), ("reference", "reference")]
