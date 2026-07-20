import pytest
import torch

from benchmarks.local_linear_reproducing_core import (
    GaussianGeometry,
    build_local_linear_operator,
    geometry_diagnostics,
    renderer_support_weights,
)
from structsplat.render import render


def _corner_geometry(dtype: torch.dtype = torch.float64) -> GaussianGeometry:
    means = torch.tensor(((0, 0), (8, 0), (0, 8), (8, 8)), dtype=dtype)
    conics = torch.tensor((1.0 / 64.0, 0.0, 1.0 / 64.0), dtype=dtype).repeat(4, 1)
    radii = torch.full((4, 2), 8, dtype=torch.int64)
    return GaussianGeometry(means, conics, radii)


def _affine_values(xy: torch.Tensor) -> torch.Tensor:
    u = xy[:, 0] / 8.0
    v = xy[:, 1] / 8.0
    return torch.stack(
        (
            0.10 + 0.35 * u + 0.25 * v,
            0.80 - 0.30 * u - 0.20 * v,
            0.20 + 0.20 * u + 0.40 * v,
        ),
        dim=1,
    )


def test_dense_support_and_constant_fit_match_the_production_renderer() -> None:
    geometry = _corner_geometry()
    colors = torch.tensor(
        ((0.1, 0.3, 0.9), (0.7, 0.1, 0.2), (0.2, 0.8, 0.4), (0.9, 0.6, 0.1)),
        dtype=torch.float64,
    )
    operator = build_local_linear_operator(geometry, 9, 9)
    expected = render(
        geometry.means,
        geometry.conics,
        colors,
        geometry.radii,
        9,
        9,
        support_fade=False,
        sigma_cutoff=3.0,
    )
    actual = operator.normalized_constant_render(colors)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-16)
    assert bool(operator.support_weights.support.all())
    assert torch.equal(operator.support_weights.active_counts, torch.full((81,), 4))


def test_float64_qr_oracle_exactly_reproduces_affine_rgb() -> None:
    geometry = _corner_geometry()
    operator = build_local_linear_operator(geometry, 9, 9)
    actual = operator.render(_affine_values(geometry.means))
    yy, xx = torch.meshgrid(
        torch.arange(9, dtype=torch.float64),
        torch.arange(9, dtype=torch.float64),
        indexing="ij",
    )
    expected = _affine_values(torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)).reshape(
        9, 9, 3
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-14)


def test_plumbing_rank_and_contributor_controls_have_expected_signs() -> None:
    positive = geometry_diagnostics(_corner_geometry(), 9, 9)
    assert bool(positive.early_pass.all())
    assert int(positive.ranks.min()) == 3

    collinear_means = torch.tensor(((1, 4), (4, 4), (7, 4)), dtype=torch.float64)
    collinear = GaussianGeometry(
        collinear_means,
        torch.tensor((1.0 / 64.0, 0.0, 1.0 / 64.0), dtype=torch.float64).repeat(3, 1),
        torch.full((3, 2), 8, dtype=torch.int64),
    )
    collinear_diagnostics = geometry_diagnostics(collinear, 9, 9)
    assert bool((~collinear_diagnostics.rank_pass).all())
    assert int(collinear_diagnostics.ranks.max()) == 2

    one = GaussianGeometry(
        torch.tensor(((4, 4),), dtype=torch.float64),
        torch.tensor(((1.0 / 64.0, 0.0, 1.0 / 64.0),), dtype=torch.float64),
        torch.tensor(((8, 8),), dtype=torch.int64),
    )
    one_diagnostics = geometry_diagnostics(one, 9, 9)
    assert bool((~one_diagnostics.contributor_pass).all())
    assert bool((~one_diagnostics.rank_pass).all())
    assert one_diagnostics.design_singular_values.shape == (81, 3)


def test_active_count_uses_aabb_membership_even_when_weight_underflows() -> None:
    geometry = GaussianGeometry(
        torch.tensor(((2.0, 2.0),), dtype=torch.float64),
        torch.tensor(((1e6, 0.0, 1e6),), dtype=torch.float64),
        torch.tensor(((2, 2),), dtype=torch.int64),
    )
    support = renderer_support_weights(geometry, 5, 5)
    assert int(support.active_counts.sum()) == 25
    assert int((support.weights == 0.0).sum()) >= 24


def test_qr_oracle_rejects_float32_and_stage0_rejects_opacity() -> None:
    geometry = _corner_geometry(torch.float32)
    with pytest.raises(ValueError, match="requires float64"):
        build_local_linear_operator(geometry, 9, 9)
    with_opacity = GaussianGeometry(
        geometry.means,
        geometry.conics,
        geometry.radii,
        torch.ones(4, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="opacity=None"):
        renderer_support_weights(with_opacity, 9, 9)


def test_geometry_validation_rejects_non_integer_radii() -> None:
    geometry = _corner_geometry()
    with pytest.raises(ValueError, match="integer dtype"):
        GaussianGeometry(
            geometry.means, geometry.conics, geometry.radii.to(torch.float64)
        ).validated()
