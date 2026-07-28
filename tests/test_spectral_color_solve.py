import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.spectral_color_solve import (
    gaussian_highpass_zero,
    solve_new_row_colors_spectral,
    spectral_normal_observation,
    spectral_objective,
)
from structsplat.config import FitConfig
from structsplat.fit import (
    _normalized_color_basis_apply,
    _normalized_color_denominator,
)
from structsplat.gaussians import GaussianField


def _logits(values):
    values = np.asarray(values, dtype=np.float32)
    return np.log(values / (1.0 - values))


def _field():
    return GaussianField.from_numpy(
        means=np.asarray(
            [[3.0, 3.5], [8.0, 7.0], [4.5, 8.0], [9.5, 3.0]],
            dtype=np.float32,
        ),
        scales=np.asarray(
            [[2.2, 1.8], [2.0, 2.4], [1.5, 2.1], [1.8, 1.4]],
            dtype=np.float32,
        ),
        angles=np.asarray([0.2, -0.3, 0.7, -0.9], dtype=np.float32),
        colors=np.asarray(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.6, 0.3],
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
            ],
            dtype=np.float32,
        ),
        opacities=_logits([0.8, 0.7, 0.3, 0.4]),
    )


def test_highpass_observation_is_symmetric_and_matches_objective():
    torch.manual_seed(4)
    x = torch.randn(9, 10, 3)
    y = torch.randn(9, 10, 3)
    detail = torch.zeros(9, 10, dtype=torch.bool)
    detail[1:8, 2:9] = True
    raw = torch.zeros(9, 10, dtype=torch.bool)
    raw[1:8, 1:9] = True
    sigma = 1.1
    raw_weight = 0.2

    hx = gaussian_highpass_zero(x, sigma)
    hy = gaussian_highpass_zero(y, sigma)
    assert torch.allclose((x * hy).sum(), (hx * y).sum(), atol=2e-5)

    normal_x = spectral_normal_observation(
        x,
        detail,
        raw,
        sigma=sigma,
        raw_weight=raw_weight,
    )
    expected = spectral_objective(
        x,
        detail,
        raw,
        sigma=sigma,
        raw_weight=raw_weight,
    )
    assert torch.allclose((x * normal_x).sum(), expected, atol=2e-5)


def test_spectral_partial_solve_matches_materialized_weighted_system():
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        support_fade=True,
        sigma_cutoff=3.5,
        render_chunk=1,
    )
    height, width = 12, 13
    rows = torch.tensor([2, 3])
    truth = field.colors.clone()
    truth[rows] = torch.tensor(
        [[0.85, -0.10, 0.55], [-0.20, 0.75, 0.40]]
    )
    denominator = _normalized_color_denominator(
        field,
        cfg,
        height,
        width,
        support_fade_alpha=1.0,
    )
    target = _normalized_color_basis_apply(
        field,
        truth,
        cfg,
        height,
        width,
        denominator,
        support_fade_alpha=1.0,
    )
    detail = torch.zeros(height, width, dtype=torch.bool)
    detail[2:10, 2:11] = True
    raw = torch.ones(height, width, dtype=torch.bool)
    sigma = 1.0
    raw_weight = 0.15
    ridge = 1e-4

    fixed_colors = field.colors.clone()
    fixed_colors[rows] = 0.0
    fixed = _normalized_color_basis_apply(
        field,
        fixed_colors,
        cfg,
        height,
        width,
        denominator,
        support_fade_alpha=1.0,
    )
    columns = []
    for row in rows:
        basis = torch.zeros(field.n, 1)
        basis[row, 0] = 1.0
        columns.append(
            _normalized_color_basis_apply(
                field,
                basis,
                cfg,
                height,
                width,
                denominator,
                support_fade_alpha=1.0,
            )
        )
    normal = torch.empty(2, 2)
    for i, left in enumerate(columns):
        for j, right in enumerate(columns):
            observed = spectral_normal_observation(
                right,
                detail,
                raw,
                sigma=sigma,
                raw_weight=raw_weight,
            )
            normal[i, j] = (left * observed).sum()
    normal = normal + ridge * torch.eye(2)
    right_image = target - fixed
    observed_right = spectral_normal_observation(
        right_image,
        detail,
        raw,
        sigma=sigma,
        raw_weight=raw_weight,
    )
    right = torch.stack(
        [(column * observed_right).sum(dim=(0, 1)) for column in columns]
    )
    expected = torch.linalg.solve(
        normal,
        right + ridge * field.colors[rows],
    )

    result = solve_new_row_colors_spectral(
        field,
        target,
        cfg,
        rows,
        detail,
        raw,
        sigma=sigma,
        raw_weight=raw_weight,
        ridge=ridge,
        max_iterations=16,
        tolerance=1e-9,
    )

    assert torch.equal(result.field.colors[:2], field.colors[:2])
    assert torch.allclose(result.field.colors[rows], expected, atol=2e-5, rtol=2e-5)
    assert result.relative_residual < 1e-5
    assert result.final_objective < result.initial_objective
