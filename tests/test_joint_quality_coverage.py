"""PORT-007: same-render raw coverage, guarded fallback and no-grad ownership."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from structsplat.config import FitConfig
from structsplat.fit import _MaskConstraint, _raw_weight_map_field, _render
from structsplat.gaussians import GaussianField
from structsplat.pipeline import PipelineConfig, build_fit_config
from structsplat.safe_schedule import (
    _coverage_requires_reference, _quality_from_render, _quality_render_inputs,
    evaluate_quality, safe_commit_decision, CommitTolerances,
)


def fixture(device="cpu", n=5):
    rng = np.random.default_rng(812)
    field = GaussianField.from_numpy(
        rng.uniform(5, 25, (n, 2)), rng.uniform(0.8, 3, (n, 2)),
        rng.uniform(-2, 2, n), rng.uniform(-0.2, 1.1, (n, 3)),
        opacities=rng.uniform(-1, 2, n), filter_variance=rng.uniform(0, 0.2, n),
        device=device,
    )
    target = torch.as_tensor(rng.uniform(0, 1, (32, 32, 3)), dtype=torch.float32,
                             device=device)
    mask = torch.ones((32, 32), dtype=torch.bool, device=device)
    cfg = FitConfig(renderer="cuda" if device == "cuda" else "normalized",
                    support_fade=True, aa_dilation=0.1, render_chunk=64)
    constraint = _MaskConstraint.from_mask(
        mask.cpu().numpy(), device, target.dtype, cfg.sigma_cutoff, cfg.mask_margin,
        undercoverage_band=4,
    )
    return field, target, mask, cfg, constraint


@pytest.mark.parametrize("value,expected", [
    (0.0, False), (1.0, False), (0.05, True), (0.050001, True),
    (0.049999, True), (-1e-8, True), (float("inf"), True), (float("nan"), True),
])
def test_guard_threshold_nonfinite_and_negative(value, expected):
    den = torch.full((2, 2), value)
    mask = torch.ones((2, 2), dtype=torch.bool)
    assert _coverage_requires_reference(den, mask, 0.05) is expected


def test_guard_nonzero_outside_even_below_containment_slack():
    mask = torch.tensor([[True, False]])
    assert _coverage_requires_reference(torch.tensor([[1., 1e-12]]), mask, 0.05)
    assert not _coverage_requires_reference(torch.tensor([[1., 0.]]), mask, 0.05)


def test_configuration_defaults_and_fail_closed():
    assert FitConfig().quality_coverage_backend == "reference"
    assert PipelineConfig().quality_coverage_backend == "reference"
    cfg = PipelineConfig(quality_coverage_backend="renderer")
    cfg.validate()
    assert build_fit_config(cfg, "cpu").quality_coverage_backend == "renderer"
    with pytest.raises(ValueError, match="quality_coverage_backend"):
        FitConfig(quality_coverage_backend="unknown")
    with pytest.raises(ValueError, match="quality_coverage_backend"):
        PipelineConfig(quality_coverage_backend="unknown").validate()


def test_cpu_fallback_and_extracted_metrics_are_exact():
    field, target, mask, cfg, constraint = fixture()
    before = field.detached()
    render = _render(field, cfg, 32, 32, support_fade_alpha=1.)
    den = _raw_weight_map_field(field, cfg, 32, 32, support_fade_alpha=1.).reshape(32, 32)
    expected = _quality_from_render(render, target, den, mask, constraint, .05, field.n)
    actual, image = evaluate_quality(field, target, mask, cfg, constraint, .05)
    fallback, other = evaluate_quality(
        field, target, mask, replace(cfg, quality_coverage_backend="renderer"), constraint, .05)
    assert actual == fallback == expected
    torch.testing.assert_close(image, render, rtol=0, atol=0)
    torch.testing.assert_close(other, render, rtol=0, atol=0)
    for name in ("means", "log_scales", "rotations", "colors", "opacities"):
        torch.testing.assert_close(getattr(field, name), getattr(before, name), rtol=0, atol=0)
    assert safe_commit_decision(actual, fallback, CommitTolerances()) == (False, ["no_material_gain"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("n", [0, 1, 5])
def test_joint_cuda_matches_reference_no_grad(n):
    from structsplat.cuda_render import render_cuda_exact_with_coverage

    field, target, mask, cfg, constraint = fixture("cuda", n)
    field.trainable()
    render, den = render_cuda_exact_with_coverage(
        field.means, field.conics(cfg.aa_dilation), field.colors,
        field.radii(cfg.sigma_cutoff, cfg.aa_dilation), 32, 32,
        opacities=field.opacity_values(), support_fade=True, eps=cfg.normalization_eps,
    )
    assert not render.requires_grad and not den.requires_grad
    oracle = _render(field, replace(cfg, renderer="normalized"), 32, 32,
                     support_fade_alpha=1.)
    oracle_den = _raw_weight_map_field(field, cfg, 32, 32, support_fade_alpha=1.).reshape(32, 32)
    torch.testing.assert_close(render, oracle, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(den, oracle_den, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(den < .05, oracle_den < .05, rtol=0, atol=0)
    legacy, _ = evaluate_quality(field, target, mask, cfg, constraint, .05)
    reuse, _ = evaluate_quality(field, target, mask,
        replace(cfg, quality_coverage_backend="renderer"), constraint, .05)
    assert legacy.interior_hole_fraction == reuse.interior_hole_fraction
    assert legacy.boundary_hole_fraction == reuse.boundary_hole_fraction


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_joint_guard_recomputes_reference_den_and_active_prefix(monkeypatch):
    import structsplat.cuda_render as cr
    import structsplat.safe_schedule as ss

    field, target, mask, cfg, constraint = fixture("cuda")
    cfg = replace(cfg, quality_coverage_backend="renderer")
    fake_render = torch.zeros_like(target)
    fake_den = torch.full_like(target[..., 0], .05)
    monkeypatch.setattr(cr, "render_cuda_exact_with_coverage", lambda *a, **k: (fake_render, fake_den))
    calls = []
    real = ss._raw_weight_map_field
    def counted(active, *args, **kwargs):
        calls.append(active.n)
        return real(active, *args, **kwargs)
    monkeypatch.setattr(ss, "_raw_weight_map_field", counted)
    actual, image = evaluate_quality(field, target, mask, cfg, constraint, .05, active_n=2)
    assert calls == [2] and actual.n_gaussians == 2
    assert image is fake_render
    _, den = _quality_render_inputs(field, cfg, 32, 32, mask, .05)
    assert not torch.equal(den, fake_den)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_affine_field_falls_back_without_joint_call(monkeypatch):
    import structsplat.cuda_render as cr
    field, target, mask, cfg, constraint = fixture("cuda")
    field = field.with_affine_colors()
    def forbidden(*args, **kwargs):
        raise AssertionError("affine field must not call joint CUDA")
    monkeypatch.setattr(cr, "render_cuda_exact_with_coverage", forbidden)
    ref, _ = evaluate_quality(field, target, mask, cfg, constraint, .05)
    got, _ = evaluate_quality(field, target, mask,
        replace(cfg, quality_coverage_backend="renderer"), constraint, .05)
    assert got == ref
