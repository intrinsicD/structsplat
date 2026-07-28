import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.gauge_lifted_dipole import (
    DipoleSelection,
    apply_dipole_split,
    select_residual_dipoles,
    translation_jacobian,
)
from structsplat.config import FitConfig
from structsplat.fit import _raw_weight_map_field, _render
from structsplat.gaussians import GaussianField


def _logits(values):
    values = np.asarray(values, dtype=np.float32)
    return np.log(values / (1.0 - values))


def _field():
    return GaussianField.from_numpy(
        means=np.asarray([[6.2, 7.1], [14.0, 9.0], [9.0, 15.0]], dtype=np.float32),
        scales=np.asarray([[2.3, 1.6], [2.0, 2.4], [1.8, 1.4]], dtype=np.float32),
        angles=np.asarray([0.3, -0.4, 0.8], dtype=np.float32),
        colors=np.asarray(
            [[0.65, 0.25, 0.20], [0.15, 0.70, 0.35], [0.20, 0.30, 0.80]],
            dtype=np.float32,
        ),
        opacities=_logits([0.8, 0.55, 0.7]),
        scale_max=np.full((3, 2), 4.0, dtype=np.float32),
    )


def _selection(parent, displacement, contrast):
    return DipoleSelection(
        parents=torch.as_tensor([parent], dtype=torch.long),
        displacement=torch.as_tensor([displacement], dtype=torch.float32),
        contrast=torch.as_tensor([contrast], dtype=torch.float32),
        score=torch.ones(1),
        unclipped_score=torch.ones(1),
        color_clip=torch.ones(1),
        support_scale=torch.ones(1),
        candidate_count=1,
        rejected_background=0,
        rejected_mask=0,
        rejected_degenerate=0,
    )


@pytest.mark.parametrize("support_fade", [False, True])
def test_zero_lift_is_exact_half_opacity_gauge_split(support_fade):
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        support_fade=support_fade,
        sigma_cutoff=3.5,
        render_chunk=1,
    )
    selection = _selection(1, [0.7, -0.2], [0.2, -0.1, 0.3])
    split = apply_dipole_split(field, selection, lift_scale=0.0)
    before = _render(field, cfg, 22, 23, support_fade_alpha=1.0)
    after = _render(split, cfg, 22, 23, support_fade_alpha=1.0)
    before_den = _raw_weight_map_field(
        field, cfg, 22, 23, support_fade_alpha=1.0
    )
    after_den = _raw_weight_map_field(
        split, cfg, 22, 23, support_fade_alpha=1.0
    )

    assert split.n == field.n + 1
    assert float((after - before).abs().max()) <= 2e-6
    assert float((after_den - before_den).abs().max()) <= 2e-6


def test_translation_jacobian_matches_centered_dipole_finite_difference():
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        support_fade=False,
        sigma_cutoff=4.0,
        render_chunk=1,
    )
    direction = torch.tensor([0.6, -0.8])
    contrast = torch.tensor([0.17, -0.08, 0.11])
    epsilon = 2e-3
    selection = _selection(
        0,
        (epsilon * direction).tolist(),
        contrast.tolist(),
    )
    split = apply_dipole_split(field, selection)
    base = _render(field, cfg, 22, 23, support_fade_alpha=1.0)
    realized = (
        _render(split, cfg, 22, 23, support_fade_alpha=1.0) - base
    ) / epsilon
    jacobian = translation_jacobian(field, 0, cfg, 22, 23)
    expected_scalar = torch.einsum("hwk,k->hw", jacobian, direction)
    expected = expected_scalar[..., None] * contrast
    denominator = _raw_weight_map_field(
        field, cfg, 22, 23, support_fade_alpha=1.0
    ).reshape(22, 23)
    covered = denominator > 1e-3
    error = realized[covered] - expected[covered]

    assert float(error.square().sum().sqrt() / expected[covered].square().sum().sqrt()) < 1e-3
    assert float(error.abs().max()) < 7e-4


def test_residual_solve_is_finite_deterministic_and_reduces_matched_mode():
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        support_fade=False,
        sigma_cutoff=3.5,
        render_chunk=1,
    )
    height, width = 22, 23
    base = _render(field, cfg, height, width, support_fade_alpha=1.0)
    jacobian = translation_jacobian(field, 0, cfg, height, width)
    direction = torch.tensor([0.8, 0.6])
    color = torch.tensor([0.12, -0.05, 0.08])
    target = base + 0.4 * torch.einsum(
        "hwk,k,c->hwc", jacobian, direction, color
    )
    mask = torch.ones(height, width, dtype=torch.bool)

    first = select_residual_dipoles(
        field,
        target,
        base,
        cfg,
        mask,
        1,
        minimum_spacing=0.0,
    )
    second = select_residual_dipoles(
        field,
        target,
        base,
        cfg,
        mask,
        1,
        minimum_spacing=0.0,
    )
    proposal = apply_dipole_split(field, first)
    reconstructed = _render(
        proposal, cfg, height, width, support_fade_alpha=1.0
    )

    assert first.parents.tolist() == second.parents.tolist() == [0]
    assert torch.equal(first.displacement, second.displacement)
    assert torch.equal(first.contrast, second.contrast)
    assert bool(torch.isfinite(first.score).all())
    assert float((reconstructed - target).square().mean()) < float(
        (base - target).square().mean()
    )
    conic = field.conics()[0]
    delta = first.displacement[0]
    radius = math.sqrt(
        float(
            conic[0] * delta[0].square()
            + 2.0 * conic[1] * delta[0] * delta[1]
            + conic[2] * delta[1].square()
        )
    )
    assert radius == pytest.approx(
        0.35 * math.sqrt(float(first.support_scale[0])),
        abs=1e-5,
    )
