import math

import numpy as np
import pytest
import torch

from benchmarks import affine_carrier_core as core
from structsplat.render import render


def _asymmetric_field(
    dtype: torch.dtype,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = 7, 8
    means = torch.tensor(
        ((-0.4, 0.7), (6.6, 0.2), (1.3, 5.8), (7.2, 6.4), (3.4, 2.1)),
        dtype=dtype,
    )
    conics = torch.tensor(
        (
            (0.08, 0.01, 0.13),
            (0.11, -0.02, 0.07),
            (0.06, 0.015, 0.10),
            (0.09, 0.025, 0.12),
            (0.13, -0.03, 0.08),
        ),
        dtype=dtype,
    )
    colors = torch.tensor(
        (
            (0.1, 0.2, 0.8),
            (0.7, 0.3, 0.2),
            (0.4, 0.9, 0.5),
            (0.8, 0.6, 0.1),
            (0.3, 0.1, 0.9),
        ),
        dtype=dtype,
    )
    radii = torch.tensor(((4, 4), (4, 3), (3, 4), (4, 4), (3, 3)), dtype=torch.int64)
    return height, width, means, conics, colors, radii


def _wide_corners(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = torch.tensor(((0, 0), (8, 0), (0, 8), (8, 8), (4, 4)), dtype=dtype)
    conics = torch.tensor((1.0 / 64.0, 0.0, 1.0 / 64.0), dtype=dtype).repeat(5, 1)
    radii = torch.full((5, 2), 8, dtype=torch.int64)
    return means, conics, radii


def test_gauge_fixed_design_has_two_centered_columns() -> None:
    points = torch.tensor(((0.0, 0.0), (8.0, 0.0), (0.0, 6.0), (4.0, 3.0)))
    design = core.carrier_design(points, 7, 9)
    expected = torch.tensor(((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)))
    torch.testing.assert_close(design, expected, rtol=0.0, atol=0.0)
    pixels = core.pixel_design(7, 9, dtype=torch.float64)
    assert pixels.shape == (63, 2)
    torch.testing.assert_close(pixels[0], torch.tensor((-1.0, -1.0), dtype=torch.float64))
    torch.testing.assert_close(pixels[-1], torch.tensor((1.0, 1.0), dtype=torch.float64))


def test_binary16_beta_payload_is_exactly_twelve_little_endian_bytes() -> None:
    beta = torch.tensor(
        ((0.125, -0.3333, 1.75), (-2.5, 0.0007, 17.25)), dtype=torch.float64
    )
    payload = core.encode_beta_f16(beta)
    assert len(payload) == core.BETA_PAYLOAD_BYTES == 12
    assert payload == np.asarray(beta.numpy(), dtype="<f2").tobytes(order="C")
    decoded = core.decode_beta_f16(payload)
    assert decoded.shape == (2, 3)
    assert decoded.dtype == torch.float32
    expected = torch.from_numpy(np.asarray(beta.numpy(), dtype="<f2").astype(np.float32))
    torch.testing.assert_close(decoded, expected, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="exactly 12 bytes"):
        core.decode_beta_f16(payload[:-1])
    with pytest.raises(ValueError, match="binary16 range"):
        core.encode_beta_f16(torch.full((2, 3), 70000.0))


def test_quantized_linear_tail_is_reproduced_without_local_rank_or_signed_weights() -> None:
    height = width = 9
    beta_source = torch.tensor(
        ((0.17321, -0.28519, 0.39117), (-0.31981, 0.24713, 0.11891)),
        dtype=torch.float64,
    )
    _, beta32 = core.quantize_beta_f16(beta_source)
    means32, conics32, radii = _wide_corners(torch.float32)
    colors32 = core.carrier_design(means32, height, width) @ beta32
    candidate = core.candidate_render(
        means32, conics32, colors32, beta32, radii, height, width
    )
    truth32 = (core.pixel_design(height, width, dtype=torch.float32) @ beta32).reshape(
        height, width, 3
    )
    torch.testing.assert_close(candidate.residual_colors, torch.zeros_like(colors32), rtol=0, atol=0)
    torch.testing.assert_close(candidate.output, truth32, rtol=0, atol=0)
    assert bool(candidate.nonnegative_pass.all())

    beta64 = beta32.to(torch.float64)
    means64, conics64, _ = _wide_corners(torch.float64)
    colors64 = core.carrier_design(means64, height, width) @ beta64
    reference = core.reference_render(
        means64, conics64, colors64, beta64, radii, height, width
    )
    truth64 = (core.pixel_design(height, width, dtype=torch.float64) @ beta64).reshape(
        height, width, 3
    )
    torch.testing.assert_close(reference.output, truth64, rtol=0.0, atol=0.0)


def test_positive_residual_weights_report_partition_a1_minimum_and_first_moment() -> None:
    height, width, means, conics, colors, radii = _asymmetric_field(torch.float64)
    beta = torch.tensor(((0.08, -0.04, 0.03), (-0.02, 0.06, -0.07)), dtype=torch.float64)
    result = core.reference_render(means, conics, colors, beta, radii, height, width)
    assert bool(result.nonnegative_pass.all())
    assert float(result.minimum_effective_weight.min()) >= 0.0
    torch.testing.assert_close(result.a1, result.amplification_l1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.amplification_l1, result.partition, rtol=0.0, atol=0.0)
    assert float(result.partition.min()) > 0.0
    assert float(result.partition.max()) <= 1.0

    pixel = 3 * width + 4
    sample_q = core.carrier_design(means, height, width)
    pixel_q = core.pixel_design(height, width, dtype=torch.float64)[pixel]
    manual_moment = torch.einsum(
        "n,nd->d", result.effective_weights[pixel], sample_q - pixel_q
    )
    torch.testing.assert_close(result.first_moment[pixel], manual_moment, rtol=0.0, atol=1e-15)


def test_zero_tail_matches_the_production_positive_normalized_renderer() -> None:
    height, width, means, conics, colors, radii = _asymmetric_field(torch.float32)
    candidate = core.candidate_render(
        means, conics, colors, torch.zeros((2, 3)), radii, height, width
    )
    production = render(means, conics, colors, radii, height, width)
    # The candidate fuses mass and RGB into one four-channel scatter; CPU vector scatter may differ
    # by one float32 summation ulp from production's two separate scatters.
    torch.testing.assert_close(candidate.output, production, rtol=0.0, atol=2e-7)
    torch.testing.assert_close(
        candidate.output, candidate.normalized_residual.reshape(height, width, 3), rtol=0, atol=0
    )
    assert candidate.accumulators.shape == (height * width, 4)


def test_candidate_all_four_channels_match_an_asymmetric_manual_oracle() -> None:
    height, width, means, conics, colors, radii = _asymmetric_field(torch.float32)
    beta = torch.tensor(((0.07, -0.11, 0.05), (-0.03, 0.09, 0.02)), dtype=torch.float32)
    sample_q = core.carrier_design(means, height, width)
    residual = colors - sample_q @ beta
    result = core.candidate_render_residual(
        means, conics, residual, beta, radii, height, width
    )
    expected = torch.zeros((height * width, 4), dtype=torch.float32)
    for gid in range(means.shape[0]):
        center_x = int(torch.round(means[gid, 0]).item())
        center_y = int(torch.round(means[gid, 1]).item())
        x0 = max(0, center_x - int(radii[gid, 0]))
        x1 = min(width - 1, center_x + int(radii[gid, 0]))
        y0 = max(0, center_y - int(radii[gid, 1]))
        y1 = min(height - 1, center_y + int(radii[gid, 1]))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                dx = torch.tensor(float(x)) - means[gid, 0]
                dy = torch.tensor(float(y)) - means[gid, 1]
                quadratic = (
                    conics[gid, 0] * dx.square()
                    + 2.0 * conics[gid, 1] * dx * dy
                    + conics[gid, 2] * dy.square()
                )
                weight = torch.exp(-0.5 * quadratic)
                contribution = torch.cat((weight.reshape(1), weight * residual[gid]))
                expected[y * width + x] += contribution
    assert bool((expected.abs().sum(dim=0) > 0.0).all())
    torch.testing.assert_close(result.accumulators, expected, rtol=0.0, atol=0.0)


def test_independent_residual_decoder_avoids_reconstruct_then_subtract_roundoff() -> None:
    height, width, means, conics, _, radii = _asymmetric_field(torch.float32)
    beta = torch.tensor(((1024.0, -768.0, 896.0), (-640.0, 512.0, -384.0)))
    residual = torch.tensor(
        (
            (1.0e-5, -2.0e-5, 3.0e-5),
            (-4.0e-5, 5.0e-5, -6.0e-5),
            (7.0e-5, -8.0e-5, 9.0e-5),
            (-1.0e-4, 1.1e-4, -1.2e-4),
            (1.3e-4, -1.4e-4, 1.5e-4),
        ),
        dtype=torch.float32,
    )
    decoded = core.candidate_render_residual(
        means, conics, residual, beta, radii, height, width
    )
    reconstructed_colors = core.carrier_design(means, height, width) @ beta + residual
    roundtrip = core.candidate_render(
        means, conics, reconstructed_colors, beta, radii, height, width
    )
    torch.testing.assert_close(decoded.residual_colors, residual, rtol=0.0, atol=0.0)
    assert not torch.equal(roundtrip.residual_colors, residual)
    assert not torch.equal(decoded.output, roundtrip.output)

    means64 = means.to(torch.float64)
    conics64 = conics.to(torch.float64)
    residual64 = residual.to(torch.float64)
    beta64 = beta.to(torch.float64)
    reference = core.reference_render_residual(
        means64, conics64, residual64, beta64, radii, height, width
    )
    torch.testing.assert_close(reference.residual_colors, residual64, rtol=0.0, atol=0.0)


def test_reproduction_defect_qr_recovers_tail_and_cannot_increase_prequant_sse() -> None:
    height, width, means, conics, colors, radii = _asymmetric_field(torch.float64)
    beta = torch.tensor(((0.04, -0.03, 0.06), (-0.05, 0.02, 0.01)), dtype=torch.float64)
    target = core.reference_render(means, conics, colors, beta, radii, height, width).output
    fit = core.fit_reproduction_defect_qr(
        means, conics, colors, radii, target, height, width
    )
    assert fit.diagnostics.accepted
    assert fit.diagnostics.rank == 2
    torch.testing.assert_close(fit.beta, beta, rtol=0.0, atol=2e-14)
    baseline_sse = (fit.baseline - target.reshape(-1, 3)).square().sum()
    fitted = fit.baseline + fit.reproduction_design @ fit.beta
    fitted_sse = (fitted - target.reshape(-1, 3)).square().sum()
    assert float(fitted_sse) <= float(baseline_sse) + 1e-25

    baseline = core.reference_render(
        means, conics, colors, torch.zeros((2, 3), dtype=torch.float64), radii, height, width
    )
    augmented = core.diagnose_augmented_basis(
        baseline.effective_weights, fit.reproduction_design
    )
    assert augmented.full_tail_gain
    assert augmented.rank_gain == 2
    assert augmented.augmented_rank_threshold > 0.0


def test_rank_deficient_reproduction_design_is_diagnostic_total_and_rejected() -> None:
    x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    collinear = torch.stack((x, 2.0 * x), dim=1)
    diagnostics = core.diagnose_reproduction_design(collinear)
    assert diagnostics.rank == 1
    assert not diagnostics.accepted
    assert math.isfinite(diagnostics.condition_number)
    with pytest.raises(ValueError, match="rank deficient"):
        core.solve_reproduction_design_qr(
            collinear, torch.ones((17, 3), dtype=torch.float64)
        )


def test_augmented_rank_gain_uses_one_common_threshold() -> None:
    # The weak third W column lies between W's old separately scaled tolerance (4 eps) and the
    # declared augmented tolerance (5 eps).  Separate thresholds report a spurious gain of one;
    # one common augmented threshold correctly reports the two independent D directions.
    weak = 1.0e-15
    weights = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, weak), (0.0, 0.0, 0.0)),
        dtype=torch.float64,
    )
    design = torch.tensor(
        ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=torch.float64
    )
    diagnostics = core.diagnose_augmented_basis(weights, design)
    old_base_threshold = 4 * torch.finfo(torch.float64).eps
    old_base_rank = int((torch.linalg.svdvals(weights) > old_base_threshold).sum())
    assert old_base_rank == 3
    assert diagnostics.base_rank == 2
    assert diagnostics.augmented_rank == 4
    assert diagnostics.rank_gain == 2
    assert diagnostics.base_rank_threshold == diagnostics.common_rank_threshold
    assert diagnostics.augmented_rank_threshold == diagnostics.common_rank_threshold


def test_frozen_row_permutation_preserves_reference_and_candidate_outputs() -> None:
    height, width, means64, conics64, colors64, radii = _asymmetric_field(torch.float64)
    beta64 = torch.tensor(((0.08, -0.04, 0.03), (-0.02, 0.06, -0.07)), dtype=torch.float64)
    permutation = torch.tensor((4, 3, 2, 1, 0), dtype=torch.int64)
    original64 = core.reference_render(
        means64, conics64, colors64, beta64, radii, height, width
    )
    permuted64 = core.reference_render(
        means64[permutation],
        conics64[permutation],
        colors64[permutation],
        beta64,
        radii[permutation],
        height,
        width,
    )
    torch.testing.assert_close(original64.output, permuted64.output, rtol=0.0, atol=1e-15)

    original32 = core.candidate_render(
        means64.float(), conics64.float(), colors64.float(), beta64.float(), radii, height, width
    )
    permuted32 = core.candidate_render(
        means64[permutation].float(),
        conics64[permutation].float(),
        colors64[permutation].float(),
        beta64.float(),
        radii[permutation],
        height,
        width,
    )
    torch.testing.assert_close(original32.output, permuted32.output, rtol=0.0, atol=3e-7)


def test_frozen_reference_autograd_matches_centered_difference() -> None:
    height = width = 9
    means, _, radii = _wide_corners(torch.float64)
    logs = torch.log(
        torch.tensor(
            ((7.0, 5.0), (6.0, 8.0), (8.0, 6.0), (5.0, 7.0), (7.0, 7.0)),
            dtype=torch.float64,
        )
    )
    rotations = torch.tensor((0.17, -0.31, 0.49, -0.73, 0.22), dtype=torch.float64)
    colors = torch.tensor(
        ((0.1, 0.2, 0.8), (0.7, 0.3, 0.2), (0.4, 0.9, 0.5),
         (0.8, 0.6, 0.1), (0.3, 0.1, 0.9)),
        dtype=torch.float64,
    )
    beta = torch.tensor(((0.08, -0.04, 0.03), (-0.02, 0.06, -0.07)), dtype=torch.float64)
    contributors = core.enumerate_contributors(means, radii, height, width)
    parameters = [
        value.clone().requires_grad_(True)
        for value in (means, logs, rotations, colors, beta)
    ]
    cotangent = torch.linspace(
        -1.0, 1.0, height * width * 3, dtype=torch.float64
    ).reshape(height, width, 3)
    cotangent /= torch.linalg.vector_norm(cotangent)
    output = core.reference_render_frozen(*parameters, radii, contributors)
    loss = torch.sum(output * cotangent)
    gradients = torch.autograd.grad(loss, parameters)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(float(torch.linalg.vector_norm(gradient)) > 1e-7 for gradient in gradients)

    direction = torch.arange(6, dtype=torch.float64).reshape(2, 3) - 2.5
    direction /= torch.linalg.vector_norm(direction)
    step = 2.0**-18
    plus = [value.detach().clone() for value in parameters]
    minus = [value.detach().clone() for value in parameters]
    plus[4] += step * direction
    minus[4] -= step * direction
    plus_loss = torch.sum(
        core.reference_render_frozen(*plus, radii, contributors) * cotangent
    )
    minus_loss = torch.sum(
        core.reference_render_frozen(*minus, radii, contributors) * cotangent
    )
    finite_difference = (plus_loss - minus_loss) / (2.0 * step)
    torch.testing.assert_close(
        torch.sum(gradients[4] * direction), finite_difference, rtol=2e-7, atol=1e-9
    )


def test_finite_grid_ray_projection_replays_payload_and_passes_exact_constraints() -> None:
    height, width, means, conics, colors, radii = _asymmetric_field(torch.float64)
    beta = torch.tensor(((0.001, -0.0015, 0.0007), (-0.0008, 0.0012, 0.001)), dtype=torch.float64)
    zero = torch.zeros((2, 3), dtype=torch.float64)
    baseline_result = core.reference_render(means, conics, colors, zero, radii, height, width)
    design = core.reproduction_defect_design(
        means, baseline_result.effective_weights, height, width
    )
    target = baseline_result.output.reshape(-1, 3) + design @ beta
    selected = core.project_beta_ray_on_finite_grid(
        beta, baseline_result.output.reshape(-1, 3), design, target
    )
    torch.testing.assert_close(selected.grid_index, torch.zeros(3, dtype=torch.int64))
    torch.testing.assert_close(selected.alpha, torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(selected.alpha_cap, torch.ones(3, dtype=torch.float64))
    assert len(selected.payload) == 12
    torch.testing.assert_close(
        core.decode_beta_f16(selected.payload), selected.decoded_beta, rtol=0, atol=0
    )
    torch.testing.assert_close(
        selected.output,
        baseline_result.output.reshape(-1, 3)
        + design @ selected.decoded_beta.to(torch.float64),
        rtol=0,
        atol=0,
    )
    assert selected.feasible
    assert bool((selected.excursion <= core.RAY_GATE_MARGIN).all())
    assert bool((selected.sse <= selected.sse_limit).all())
    assert set(selected.active_constraints) == {
        "unit_interval_rgb0",
        "unit_interval_rgb1",
        "unit_interval_rgb2",
    }


def test_ray_projection_asserts_the_declared_zero_beta_feasibility() -> None:
    design = torch.zeros((5, 2), dtype=torch.float64)
    target = torch.zeros((5, 3), dtype=torch.float64)
    baseline = torch.full((5, 3), 0.1, dtype=torch.float64)
    with pytest.raises(ValueError, match="zero beta is not feasible"):
        core.project_beta_ray_on_finite_grid(
            torch.ones((2, 3), dtype=torch.float64), baseline, design, target
        )


def test_ray_projection_checks_zero_direction_rows_at_the_design_margin() -> None:
    design = torch.zeros((5, 2), dtype=torch.float64)
    target = torch.zeros((5, 3), dtype=torch.float64)
    # This lies inside the terminal +/-2/255 gate but outside the stricter analytic
    # +/-1.75/255 intersection.  A zero direction cannot repair it for any alpha.
    baseline = torch.full((5, 3), 1.9 / 255.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="analytic range-ray intersection"):
        core.project_beta_ray_on_finite_grid(
            torch.ones((2, 3), dtype=torch.float64), baseline, design, target
        )


def test_ray_projection_rejects_overflowing_candidates_then_uses_zero() -> None:
    design = torch.tensor(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
        dtype=torch.float64,
    )
    beta = torch.full((2, 3), 1.0e9, dtype=torch.float64)
    baseline = torch.zeros((4, 3), dtype=torch.float64)
    target = design @ beta
    selected = core.project_beta_ray_on_finite_grid(beta, baseline, design, target)
    torch.testing.assert_close(
        selected.grid_index,
        torch.full((3,), core.RAY_GRID_DENOMINATOR, dtype=torch.int64),
    )
    torch.testing.assert_close(selected.decoded_beta, torch.zeros((2, 3)))
    assert selected.feasible
    assert set(selected.active_constraints) == {
        "binary16_range_rgb0",
        "binary16_range_rgb1",
        "binary16_range_rgb2",
    }


def test_shape_and_finiteness_guards_fail_closed() -> None:
    height, width, means, conics, colors, radii = _asymmetric_field(torch.float32)
    with pytest.raises(ValueError, match="shape"):
        core.candidate_render(
            means, conics, colors, torch.zeros((3, 3)), radii, height, width
        )
    with pytest.raises(ValueError, match="finite"):
        bad_colors = colors.clone()
        bad_colors[0, 0] = float("nan")
        core.candidate_render(
            means, conics, bad_colors, torch.zeros((2, 3)), radii, height, width
        )
    with pytest.raises(ValueError, match="height and width"):
        core.pixel_design(1, width, dtype=torch.float32)
    with pytest.raises(ValueError, match="matching shape"):
        core.squared_error_loss(torch.zeros((2, 2, 3)), torch.zeros((2, 3, 3)))
