import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.affine_partial_color_solve import (
    affine_color_basis_apply,
    affine_color_basis_transpose,
    solve_new_row_affine_colors,
)
from scripts.experiments.fit035_janelle_affine_screen import (
    _aa_is_nondegrading,
)
from structsplat.config import FitConfig
from structsplat.fit import _normalized_color_denominator, _render
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


def test_affine_basis_matches_native_renderer_and_its_transpose():
    torch.manual_seed(5)
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        support_fade=True,
        sigma_cutoff=3.5,
        render_chunk=1,
    )
    height, width = 12, 13
    denominator = _normalized_color_denominator(
        field,
        cfg,
        height,
        width,
        support_fade_alpha=1.0,
    )
    coefficients = torch.randn(field.n, 3, 3) * 0.2
    coefficients[:, 0] = field.colors
    applied = affine_color_basis_apply(
        field,
        coefficients,
        cfg,
        height,
        width,
        denominator,
    )
    native = field.detached().with_affine_colors()
    native.colors = coefficients[:, 0].clone()
    native.color_grads = coefficients[:, 1:].clone()
    rendered = _render(native, cfg, height, width, support_fade_alpha=1.0)
    assert torch.allclose(applied, rendered, atol=2e-6, rtol=2e-6)

    probe_coefficients = torch.randn(field.n, 3, 2)
    probe_image = torch.randn(height, width, 2)
    forward = affine_color_basis_apply(
        field,
        probe_coefficients,
        cfg,
        height,
        width,
        denominator,
    )
    transposed = affine_color_basis_transpose(
        field,
        probe_image,
        cfg,
        height,
        width,
        denominator,
    )
    assert torch.allclose(
        (forward * probe_image).sum(),
        (probe_coefficients * transposed).sum(),
        atol=2e-5,
        rtol=2e-5,
    )


def test_affine_partial_solve_matches_materialized_regularized_system():
    field = _field()
    cfg = FitConfig(
        renderer="normalized",
        support_fade=True,
        sigma_cutoff=3.5,
        render_chunk=1,
    )
    height, width = 12, 13
    rows = torch.tensor([2, 3])
    truth = field.detached().with_affine_colors()
    truth.colors[rows] = torch.tensor(
        [[0.85, -0.10, 0.55], [-0.20, 0.75, 0.40]]
    )
    assert truth.color_grads is not None
    truth.color_grads[rows] = torch.tensor(
        [
            [[0.1, -0.2, 0.05], [0.2, 0.1, -0.1]],
            [[-0.1, 0.15, 0.2], [0.05, -0.1, 0.1]],
        ]
    )
    target = _render(truth, cfg, height, width, support_fade_alpha=1.0)
    denominator = _normalized_color_denominator(
        field,
        cfg,
        height,
        width,
        support_fade_alpha=1.0,
    )
    columns = []
    for row in rows:
        for basis in range(3):
            coefficient = torch.zeros(field.n, 3, 1)
            coefficient[row, basis, 0] = 1.0
            columns.append(
                affine_color_basis_apply(
                    field,
                    coefficient,
                    cfg,
                    height,
                    width,
                    denominator,
                ).reshape(-1)
            )
    design = torch.stack(columns, dim=1)
    fixed_coefficients = torch.zeros(field.n, 3, 3)
    fixed_coefficients[:, 0] = field.colors
    fixed_coefficients[rows] = 0.0
    fixed = affine_color_basis_apply(
        field,
        fixed_coefficients,
        cfg,
        height,
        width,
        denominator,
    )
    initial = torch.zeros(6, 3)
    initial[0, :] = field.colors[rows[0]]
    initial[3, :] = field.colors[rows[1]]
    diagonal = torch.tensor(
        [1e-4, 2e-3, 2e-3, 1e-4, 2e-3, 2e-3]
    )
    normal = design.T @ design + torch.diag(diagonal)
    expected = torch.linalg.solve(
        normal,
        design.T @ (target - fixed).reshape(-1, 3)
        + diagonal[:, None] * initial,
    )

    result = solve_new_row_affine_colors(
        field,
        target,
        cfg,
        rows,
        color_ridge=1e-4,
        gradient_ridge=2e-3,
        max_iterations=24,
        tolerance=1e-9,
    )
    actual = torch.cat(
        [
            result.field.colors[rows, None, :],
            result.field.color_grads[rows],
        ],
        dim=1,
    ).reshape(6, 3)

    assert torch.equal(result.field.colors[:2], field.colors[:2])
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert result.relative_residual < 1e-5
    assert result.final_objective < result.initial_objective


def test_affine_renderer_aa_allows_only_missing_material_gain():
    assert _aa_is_nondegrading(True, [])
    assert _aa_is_nondegrading(False, ["no_material_gain"])
    assert not _aa_is_nondegrading(
        False,
        ["foreground_mse_regressed", "no_material_gain"],
    )
