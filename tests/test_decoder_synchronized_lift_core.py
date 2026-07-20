from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks import affine_carrier_core as affine_core
from benchmarks import decoder_synchronized_lift_core as core
from structsplat.gaussians import GaussianField


def _irregular_means(count: int = 17) -> torch.Tensor:
    rng = np.random.default_rng(20260716031)
    points = np.column_stack((rng.uniform(1.0, 30.0, count), rng.uniform(1.0, 22.0, count)))
    return torch.from_numpy(points).to(torch.float64)


def _affine_colors(means: torch.Tensor, height: int = 25, width: int = 33) -> torch.Tensor:
    beta = torch.tensor(
        [[0.45, 0.55, 0.40], [0.12, -0.08, 0.10], [-0.07, 0.11, 0.16]],
        dtype=torch.float64,
    )
    return core.affine_design(means, height, width) @ beta


def test_affine_design_and_diagnostics() -> None:
    means = _irregular_means()
    design = core.affine_design(means, 25, 33)
    assert design.shape == (17, 3)
    assert torch.equal(design[:, 0], torch.ones(17, dtype=torch.float64))
    diagnostics = core.diagnose_design(design)
    assert diagnostics.accepted
    assert diagnostics.rank == 3
    assert diagnostics.condition_number < core.MAXIMUM_DESIGN_CONDITION


def test_robust_plane_exactly_recovers_affine_samples() -> None:
    means = _irregular_means()
    colors = _affine_colors(means)
    result = core.robust_plane_irls(means, colors, 25, 33)
    expected = torch.tensor(
        [[0.45, 0.55, 0.40], [0.12, -0.08, 0.10], [-0.07, 0.11, 0.16]],
        dtype=torch.float64,
    )
    assert result.iterations == core.IRLS_ITERATIONS
    assert result.weighted_condition_numbers.shape == (core.IRLS_ITERATIONS, 3)
    assert result.error_median_row_indices.shape == (core.IRLS_ITERATIONS, 3, 2)
    assert result.absolute_deviation_median_row_indices.shape == (
        core.IRLS_ITERATIONS,
        3,
        2,
    )
    assert torch.allclose(result.beta, expected, atol=2e-14, rtol=2e-14)
    assert torch.max(torch.abs(result.residual)).item() <= 2e-14


def test_robust_plane_order_statistic_trace_matches_fit() -> None:
    means = _irregular_means(18)
    generator = torch.Generator().manual_seed(20260716032)
    colors = _affine_colors(means) + 0.02 * torch.randn(
        (18, 3), dtype=torch.float64, generator=generator
    )
    result = core.robust_plane_irls(means, colors, 25, 33)
    trace = core.robust_plane_order_statistic_trace(means, colors, 25, 33)
    assert torch.equal(trace.error_median_row_indices, result.error_median_row_indices)
    assert torch.equal(
        trace.absolute_deviation_median_row_indices,
        result.absolute_deviation_median_row_indices,
    )
    for indices in (
        trace.error_median_row_indices,
        trace.absolute_deviation_median_row_indices,
    ):
        assert indices.dtype == torch.int64
        assert torch.all((indices >= 0) & (indices < means.shape[0]))
        assert torch.all(indices[..., 0] != indices[..., 1])
    initial_beta, _ = core.ordinary_plane_qr(means, colors, 25, 33)
    initial_error = colors - core.affine_design(means, 25, 33) @ initial_beta
    for channel in range(3):
        error = initial_error[:, channel]
        error_ids = trace.error_median_row_indices[0, channel]
        traced_center = torch.mean(error[error_ids])
        expected_center = torch.quantile(error, 0.5, interpolation="linear")
        assert torch.equal(traced_center, expected_center)
        deviation = torch.abs(error - expected_center)
        deviation_ids = trace.absolute_deviation_median_row_indices[0, channel]
        traced_deviation = torch.mean(deviation[deviation_ids])
        expected_deviation = torch.quantile(deviation, 0.5, interpolation="linear")
        assert torch.equal(traced_deviation, expected_deviation)


@pytest.mark.parametrize(
    ("raw", "expected_ids", "expected_gradient"),
    (
        ([0.0, 1.0, 1.0, 1.0, 2.0], [2, 2], [0.0, 0.0, 1.0, 0.0, 0.0]),
        ([0.0, 1.0, 1.0, 2.0], [1, 2], [0.0, 0.5, 0.5, 0.0]),
    ),
)
def test_linear_median_trace_is_the_actual_tied_autograd_support(
    raw: list[float],
    expected_ids: list[int],
    expected_gradient: list[float],
) -> None:
    values = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    median, row_ids = core._linear_median_with_support(values)  # noqa: SLF001
    median.backward()
    assert row_ids.tolist() == expected_ids
    assert values.grad is not None
    assert values.grad.tolist() == expected_gradient


def test_dense_lift_reproduces_constant_and_affine_for_nonstochastic_weights() -> None:
    height, width = 25, 33
    means = _irregular_means()
    rng = np.random.default_rng(9)
    weights = rng.random((height * width, means.shape[0]))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights64 = torch.from_numpy(weights).to(torch.float64)

    constant = torch.tensor([0.30, 0.52, 0.68], dtype=torch.float64).repeat(means.shape[0], 1)
    output, state = core.dense_weight_render(
        weights64, constant, height, width, lift=True, means=means
    )
    assert state is not None
    assert torch.max(torch.abs(output - constant[0])).item() <= 2e-14

    means = _irregular_means(77)
    weights = rng.random((height * width, means.shape[0]))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights64 = torch.from_numpy(weights).to(torch.float64)
    colors = _affine_colors(means, height, width)
    output, state = core.dense_weight_render(
        weights64, colors, height, width, lift=True, means=means
    )
    expected = (
        core.pixel_design(height, width, dtype=torch.float64) @ state.unconstrained_beta
    ).reshape(height, width, 3)
    assert torch.all(state.confidence.alpha > 0.999999)
    assert torch.max(torch.abs(output - expected)).item() <= 3e-14


def test_ols_effective_operator_reproduces_affine_without_adding_rank() -> None:
    height, width = 15, 19
    means = _irregular_means(17).clone()
    means[:, 0] *= (width - 2) / 30.0
    means[:, 1] *= (height - 2) / 22.0
    rng = np.random.default_rng(101)
    weights = rng.random((height * width, means.shape[0]))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights64 = torch.from_numpy(weights).to(torch.float64)
    diagnostics = core.dense_effective_operator_diagnostics(weights64, means, height, width)
    assert diagnostics.reproduction_max_abs <= 5e-15
    assert diagnostics.operator_rank == diagnostics.base_rank == means.shape[0]
    assert diagnostics.minimum_coefficient < 0.0
    assert diagnostics.maximum_row_l1 > 1.0

    colors = _affine_colors(means, height, width)
    beta, plane_diagnostics = core.ordinary_plane_qr(means, colors, height, width)
    assert plane_diagnostics.accepted
    output = core.dense_render_with_beta(weights64, means, colors, beta, height, width)
    expected = (core.pixel_design(height, width, dtype=torch.float64) @ beta).reshape(
        height, width, 3
    )
    assert torch.max(torch.abs(output - expected)).item() <= 5e-15


def test_smooth_max_confidence_rejects_step_and_keeps_smooth_field() -> None:
    means = _irregular_means(31)
    design = core.affine_design(means, 25, 33)
    smooth = design @ torch.tensor(
        [[0.4, 0.5, 0.6], [0.1, -0.08, 0.06], [0.07, 0.09, -0.04]],
        dtype=torch.float64,
    )
    smooth_residual = (
        smooth
        - core.affine_design(means, 25, 33)
        @ torch.linalg.lstsq(core.affine_design(means, 25, 33), smooth).solution
    )
    smooth_result = core.discontinuity_confidence(means, smooth, smooth_residual, 25, 33)
    assert torch.all(smooth_result.alpha > 0.999)

    step_bit = (means[:, 0] + 0.65 * means[:, 1] >= 16.0).to(torch.float64)
    low = torch.tensor([0.2, 0.34, 0.76], dtype=torch.float64)
    high = torch.tensor([0.8, 0.70, 0.24], dtype=torch.float64)
    step = low + step_bit[:, None] * (high - low)
    step_plane = core.robust_plane_irls(means, step, 25, 33)
    step_result = core.discontinuity_confidence(means, step, step_plane.residual, 25, 33)
    assert torch.all(step_result.alpha <= 0.01)
    assert torch.all(step_result.score >= core.CONFIDENCE_HIGH)

    rng = np.random.default_rng(83)
    weights = rng.random((25 * 33, means.shape[0]))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights64 = torch.from_numpy(weights).to(torch.float64)
    nw_output, _ = core.dense_weight_render(weights64, step, 25, 33, lift=False)
    dsl_output, state = core.dense_weight_render(weights64, step, 25, 33, lift=True, means=means)
    assert state is not None
    assert torch.equal(state.confidence.alpha, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(state.beta, torch.zeros((3, 3), dtype=torch.float64))
    assert torch.equal(dsl_output, nw_output)


def test_neighbor_distance_ties_use_decoded_xy_lexicographic_order() -> None:
    means = torch.tensor(
        (
            (10.0, 10.0),
            (11.0, 10.0),
            (10.0, 11.0),
            (9.0, 10.0),
            (10.0, 9.0),
            (14.0, 14.0),
        ),
        dtype=torch.float64,
    )
    colors = torch.linspace(0.1, 0.9, 18, dtype=torch.float64).reshape(6, 3)
    result = core.discontinuity_confidence(means, colors, colors - colors.mean(dim=0), 21, 21)
    expected = torch.tensor((3, 4, 2, 1), dtype=torch.int64)
    assert torch.equal(result.neighbor_indices[0], expected)


def test_permutation_invariance() -> None:
    height, width = 25, 33
    means = _irregular_means(23)
    colors = _affine_colors(means, height, width)
    colors = colors + 0.015 * torch.sin(means[:, :1] * torch.tensor([0.3, 0.5, 0.7]))
    rng = np.random.default_rng(17)
    weights = rng.random((height * width, means.shape[0]))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights = torch.from_numpy(weights).to(torch.float64)
    output, state = core.dense_weight_render(weights, colors, height, width, lift=True, means=means)
    permutation = torch.as_tensor(rng.permutation(means.shape[0]), dtype=torch.int64)
    permuted_output, permuted_state = core.dense_weight_render(
        weights[:, permutation],
        colors[permutation],
        height,
        width,
        lift=True,
        means=means[permutation],
    )
    assert state is not None and permuted_state is not None
    assert torch.max(torch.abs(output - permuted_output)).item() <= 2e-14
    assert (
        torch.max(torch.abs(state.confidence.alpha - permuted_state.confidence.alpha)).item()
        <= 2e-14
    )


def test_directional_color_gradient_matches_finite_difference() -> None:
    height, width = 13, 17
    means = _irregular_means(13).clone()
    means[:, 0] *= (width - 2) / 30.0
    means[:, 1] *= (height - 2) / 22.0
    generator = torch.Generator().manual_seed(20260716033)
    colors = 0.25 + 0.5 * torch.rand((means.shape[0], 3), generator=generator, dtype=torch.float64)
    colors = colors.clone().requires_grad_(True)
    rng = np.random.default_rng(23)
    weights = rng.random((height * width, means.shape[0]))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights = torch.from_numpy(weights).to(torch.float64)
    target = torch.full((height, width, 3), 0.5, dtype=torch.float64)
    output, _ = core.dense_weight_render(weights, colors, height, width, lift=True, means=means)
    loss = torch.mean(torch.square(output - target))
    (gradient,) = torch.autograd.grad(loss, colors)
    direction = torch.from_numpy(rng.normal(size=colors.shape)).to(torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    automatic = torch.sum(gradient * direction)
    step = 2.0**-16
    values = []
    for sign in (-1.0, 1.0):
        shifted = colors.detach() + sign * step * direction
        shifted_output, _ = core.dense_weight_render(
            weights, shifted, height, width, lift=True, means=means
        )
        values.append(torch.mean(torch.square(shifted_output - target)))
    finite = (values[1] - values[0]) / (2.0 * step)
    assert torch.allclose(automatic, finite, atol=1e-8, rtol=2e-4)


def test_mid_transition_confidence_gradient_matches_finite_difference() -> None:
    height, width = 25, 33
    means = _irregular_means(31)
    design = core.affine_design(means, height, width)
    beta = torch.tensor(
        [[0.45, 0.55, 0.40], [0.12, -0.08, 0.10], [-0.07, 0.11, 0.16]],
        dtype=torch.float64,
    )
    colors = design @ beta + 0.08 * torch.sin(
        means[:, :1] * torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    )
    colors = colors.requires_grad_(True)
    rng = np.random.default_rng(43)
    weights = rng.random((height * width, means.shape[0]))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    weights = torch.from_numpy(weights).to(torch.float64)
    target = torch.full((height, width, 3), 0.5, dtype=torch.float64)
    output, state = core.dense_weight_render(weights, colors, height, width, lift=True, means=means)
    assert state is not None
    assert bool(((state.confidence.alpha > 0.1) & (state.confidence.alpha < 0.9)).any())
    loss = torch.mean(torch.square(output - target))
    (gradient,) = torch.autograd.grad(loss, colors)
    direction = torch.from_numpy(rng.normal(size=colors.shape)).to(torch.float64)
    direction /= torch.linalg.vector_norm(direction)
    automatic = torch.sum(gradient * direction)
    step = 2.0**-16
    values = []
    for sign in (-1.0, 1.0):
        shifted_output, _ = core.dense_weight_render(
            weights,
            colors.detach() + sign * step * direction,
            height,
            width,
            lift=True,
            means=means,
        )
        values.append(torch.mean(torch.square(shifted_output - target)))
    finite = (values[1] - values[0]) / (2.0 * step)
    assert torch.allclose(automatic, finite, atol=1e-8, rtol=2e-4)


def test_reference_and_candidate_render_agree() -> None:
    height, width = 19, 23
    means64 = _irregular_means(13)
    means64[:, 0] *= (width - 2) / 30.0
    means64[:, 1] *= (height - 2) / 22.0
    colors64 = _affine_colors(means64, height, width)
    means = means64.to(torch.float32)
    colors = colors64.to(torch.float32)
    log_scales = torch.full((13, 2), 1.35, dtype=torch.float32)
    rotations = torch.linspace(-0.3, 0.4, 13, dtype=torch.float32)
    field = GaussianField(means, log_scales, rotations, colors)
    radii = field.radii(affine_core.SIGMA_CUTOFF, dilation=0.0)
    reference = core.reference_render_lift(
        means.to(torch.float64),
        log_scales.to(torch.float64),
        rotations.to(torch.float64),
        colors.to(torch.float64),
        radii,
        height,
        width,
    )
    candidate = core.candidate_render_lift(
        means, log_scales, rotations, colors, radii, height, width
    )
    assert (
        torch.max(torch.abs(candidate.output.to(torch.float64) - reference.output)).item() <= 2e-5
    )
    assert torch.equal(candidate.base.contributors.gid, reference.base.contributors.gid)


def test_invalid_inputs_are_rejected() -> None:
    means = _irregular_means(9)
    colors = _affine_colors(means)
    collinear = means.clone()
    collinear[:, 1] = 2.0 * collinear[:, 0]
    with pytest.raises(ValueError, match="rank/condition"):
        core.robust_plane_irls(collinear, colors, 25, 33)
    bad = colors.clone()
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        core.derive_lift(means, bad, 25, 33)

    # Duplicate decoded coordinates have no invariant row identity and are rejected.
    tied = torch.full((6, 2), 10.0, dtype=torch.float64)
    tied_colors = torch.linspace(0.2, 0.8, 18, dtype=torch.float64).reshape(6, 3)
    with pytest.raises(ValueError, match="unique"):
        core.discontinuity_confidence(tied, tied_colors, tied_colors - tied_colors.mean(0), 25, 33)
