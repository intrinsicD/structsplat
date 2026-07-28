import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.residual_birth_color_solve import solve_new_row_colors
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


def test_partial_color_solve_matches_materialized_regularized_system():
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
            ).reshape(-1)
        )
    design = torch.stack(columns, dim=1)
    right = target - fixed
    normal = design.T @ design + ridge * torch.eye(rows.numel())
    expected = torch.linalg.solve(
        normal,
        design.T @ right.reshape(-1, 3) + ridge * field.colors[rows],
    )

    result = solve_new_row_colors(
        field,
        target,
        cfg,
        rows,
        ridge=ridge,
        max_iterations=16,
        tolerance=1e-9,
    )

    assert torch.equal(result.field.colors[:2], field.colors[:2])
    assert torch.allclose(result.field.colors[rows], expected, atol=2e-5, rtol=2e-5)
    assert result.relative_residual < 1e-5
    assert result.iterations <= 4


def test_partial_color_solve_reduces_exact_render_objective_and_validates_rows():
    field = _field()
    cfg = FitConfig(renderer="normalized", support_fade=True, render_chunk=1)
    rows = torch.tensor([2, 3])
    denominator = _normalized_color_denominator(
        field, cfg, 12, 13, support_fade_alpha=1.0
    )
    truth = field.colors.clone()
    truth[rows] = torch.tensor([[0.9, 0.0, 0.6], [0.0, 0.8, 0.5]])
    target = _normalized_color_basis_apply(
        field,
        truth,
        cfg,
        12,
        13,
        denominator,
        support_fade_alpha=1.0,
    )
    before = _normalized_color_basis_apply(
        field,
        field.colors,
        cfg,
        12,
        13,
        denominator,
        support_fade_alpha=1.0,
    )
    result = solve_new_row_colors(field, target, cfg, rows, ridge=0.0)
    after = _normalized_color_basis_apply(
        result.field,
        result.field.colors,
        cfg,
        12,
        13,
        denominator,
        support_fade_alpha=1.0,
    )

    assert float((after - target).square().mean()) < float(
        (before - target).square().mean()
    ) * 1e-5
    with pytest.raises(ValueError, match="nonempty"):
        solve_new_row_colors(field, target, cfg, torch.zeros(0, dtype=torch.long))
    with pytest.raises(IndexError, match="outside"):
        solve_new_row_colors(field, target, cfg, torch.tensor([field.n]))
