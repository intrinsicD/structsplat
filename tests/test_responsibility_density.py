# ruff: noqa: E402
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from structsplat.config import FitConfig
from structsplat.fit import (
    _render,
    _responsibility_error_density_scores,
    _split_from_residual_with_stats,
)
from structsplat.gaussians import GaussianField


def _logits(values):
    values = np.asarray(values, dtype=np.float32)
    return np.log(values / (1.0 - values))


def _dense_oracle(field, target, render_img, cfg, support_fade_alpha=None):
    H, W = target.shape[:2]
    N = field.n
    weights = torch.zeros(N, H, W, dtype=field.means.dtype)
    conics = field.conics(cfg.aa_dilation).detach().cpu()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation).detach().cpu()
    means = field.means.detach().cpu()
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach().cpu()
    fade = (1.0 if cfg.support_fade else 0.0) \
        if support_fade_alpha is None else min(1.0, max(0.0, support_fade_alpha))
    tail = fade * math.exp(-0.5 * cfg.sigma_cutoff**2)
    for i in range(N):
        ix = int(torch.round(means[i, 0]))
        iy = int(torch.round(means[i, 1]))
        x0 = max(0, ix - int(radii[i, 0]))
        x1 = min(W - 1, ix + int(radii[i, 0]))
        y0 = max(0, iy - int(radii[i, 1]))
        y1 = min(H - 1, iy + int(radii[i, 1]))
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = [float(value) for value in conics[i]]
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                dx = x - float(means[i, 0])
                dy = y - float(means[i, 1])
                q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
                weight = max(math.exp(-0.5 * q) - tail, 0.0)
                if opacities is not None:
                    weight *= float(opacities[i])
                weights[i, y, x] = weight
    denominator = weights.sum(dim=0)
    gamma = weights / (denominator[None] + 1e-8)
    pixel_error = (render_img.detach().cpu() - target.detach().cpu()).square().mean(dim=2)
    mass = gamma.sum(dim=(1, 2))
    error = (gamma * pixel_error[None]).sum(dim=(1, 2))
    score = error / mass.clamp_min(1e-8).pow(cfg.responsibility_mass_alpha)
    return score, mass, error, denominator


def _overlap_field(background_mask=None):
    return GaussianField.from_numpy(
        means=np.array([[2.0, 2.0], [3.0, 2.0], [5.0, 4.0]], dtype=np.float32),
        scales=np.array([[1.4, 1.0], [1.1, 1.6], [0.9, 1.2]], dtype=np.float32),
        angles=np.array([0.2, 0.7, 0.0], dtype=np.float32),
        colors=np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.4], [0.2, 0.2, 0.9]],
                        dtype=np.float32),
        opacities=_logits([0.8, 0.25, 0.6]),
        background_mask=background_mask,
    )


def test_responsibility_score_matches_direct_dense_oracle_with_overlap_opacity_and_fade():
    field = _overlap_field()
    cfg = FitConfig(
        renderer="normalized",
        support_fade=True,
        sigma_cutoff=2.5,
        responsibility_mass_alpha=0.7,
        render_chunk=1,
    )
    H = W = 7
    render_img = _render(field, cfg, H, W, support_fade_alpha=0.4)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    target = torch.stack([
        xx / (W - 1), yy / (H - 1), (xx + yy) / (H + W - 2)
    ], dim=2).float()

    score, components = _responsibility_error_density_scores(
        field, target, render_img, cfg, support_fade_alpha=0.4
    )
    oracle_score, oracle_mass, oracle_error, oracle_den = _dense_oracle(
        field, target, render_img, cfg, support_fade_alpha=0.4
    )

    assert torch.allclose(components["denominator"], oracle_den, atol=1e-6, rtol=1e-6)
    assert torch.allclose(components["mass"], oracle_mass, atol=1e-6, rtol=1e-6)
    assert torch.allclose(components["error"], oracle_error, atol=1e-6, rtol=1e-6)
    assert torch.allclose(score, oracle_score, atol=1e-6, rtol=1e-6)


def test_responsibility_alpha_only_changes_mass_exponent():
    field = _overlap_field()
    target = torch.rand(7, 7, 3, generator=torch.Generator().manual_seed(3))
    alpha1 = FitConfig(renderer="normalized", responsibility_mass_alpha=1.0)
    render_img = _render(field, alpha1, 7, 7)
    score1, components1 = _responsibility_error_density_scores(
        field, target, render_img, alpha1
    )
    alpha07 = FitConfig(renderer="normalized", responsibility_mass_alpha=0.7)
    score07, components07 = _responsibility_error_density_scores(
        field, target, render_img, alpha07
    )

    assert torch.allclose(components07["mass"], components1["mass"])
    assert torch.allclose(components07["error"], components1["error"])
    expected = score1 * components1["mass"].clamp_min(1e-8).pow(0.3)
    assert torch.allclose(score07, expected, atol=1e-6, rtol=1e-6)


def test_identical_overlap_assigns_mass_and_error_in_opacity_ratio():
    field = GaussianField.from_numpy(
        means=np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32),
        scales=np.ones((2, 2), dtype=np.float32) * 1.5,
        angles=np.zeros(2, dtype=np.float32),
        colors=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        opacities=_logits([0.8, 0.2]),
    )
    cfg = FitConfig(renderer="normalized", responsibility_mass_alpha=1.0)
    render_img = _render(field, cfg, 7, 7)
    target = torch.ones(7, 7, 3) * 0.3

    score, components = _responsibility_error_density_scores(field, target, render_img, cfg)

    assert components["mass"][0] / components["mass"][1] == pytest.approx(4.0)
    assert components["error"][0] / components["error"][1] == pytest.approx(4.0)
    assert score[0] == pytest.approx(float(score[1]), rel=1e-6, abs=1e-6)


def test_responsibility_zero_coverage_is_finite_and_zero():
    field = GaussianField.from_numpy(
        np.array([[50.0, 50.0]], dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
    )
    cfg = FitConfig(renderer="normalized", responsibility_mass_alpha=0.7)
    target = torch.ones(6, 6, 3)
    render_img = _render(field, cfg, 6, 6)

    score, components = _responsibility_error_density_scores(field, target, render_img, cfg)

    assert torch.isfinite(score).all()
    assert score.tolist() == [0.0]
    assert components["mass"].tolist() == [0.0]
    assert components["error"].tolist() == [0.0]


def test_responsibility_split_masks_background_parent_and_grows_exactly():
    field = GaussianField.from_numpy(
        means=np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32),
        scales=np.ones((2, 2), dtype=np.float32) * 1.5,
        angles=np.zeros(2, dtype=np.float32),
        colors=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        opacities=_logits([0.9, 0.1]),
        background_mask=np.array([True, False]),
    )
    cfg = FitConfig(
        renderer="normalized",
        refine_site="responsibility",
        refine_primitive="moment_preserving",
        split_count=1,
        max_gaussians=3,
        responsibility_mass_alpha=0.7,
    )
    render_img = _render(field, cfg, 7, 7)
    target = torch.ones(7, 7, 3)
    scores, _ = _responsibility_error_density_scores(field, target, render_img, cfg)
    assert scores[0] > scores[1]

    grown, added, stats = _split_from_residual_with_stats(field, target, render_img, cfg)

    assert added == 1
    assert grown.n == 3
    assert getattr(grown, "_new_parent_idx").tolist() == [1]
    assert stats["site_scores_finite"] is True
    assert stats["responsibility_mass_alpha"] == pytest.approx(0.7)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.01, float("inf"), float("nan")])
def test_responsibility_alpha_validation(alpha):
    with pytest.raises(ValueError, match="responsibility_mass_alpha"):
        FitConfig(responsibility_mass_alpha=alpha)


@pytest.mark.parametrize("renderer", ["additive", "cuda_additive", "gsplat"])
def test_responsibility_rejects_non_normalized_renderer_semantics(renderer):
    with pytest.raises(ValueError, match="requires normalized renderer semantics"):
        FitConfig(refine_site="responsibility", renderer=renderer)


def test_responsibility_alpha_does_not_change_legacy_residual_split():
    field = _overlap_field()
    target = torch.rand(7, 7, 3, generator=torch.Generator().manual_seed(7))
    common = dict(
        renderer="normalized",
        refine_site="residual",
        refine_primitive="moment_preserving",
        split_count=2,
        max_gaussians=5,
    )
    cfg07 = FitConfig(**common, responsibility_mass_alpha=0.7)
    cfg1 = FitConfig(**common, responsibility_mass_alpha=1.0)
    render_img = _render(field, cfg07, 7, 7)

    grown07, added07, _ = _split_from_residual_with_stats(
        field.detached(), target, render_img, cfg07
    )
    grown1, added1, _ = _split_from_residual_with_stats(
        field.detached(), target, render_img, cfg1
    )

    assert added07 == added1 == 2
    assert torch.equal(grown07._new_parent_idx, grown1._new_parent_idx)
    for left, right in (
        (grown07.means, grown1.means),
        (grown07.log_scales, grown1.log_scales),
        (grown07.rotations, grown1.rotations),
        (grown07.colors, grown1.colors),
        (grown07.opacities, grown1.opacities),
    ):
        assert left is not None and right is not None
        assert torch.equal(left, right)
