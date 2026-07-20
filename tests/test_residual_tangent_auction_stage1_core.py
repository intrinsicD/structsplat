# ruff: noqa: E402
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.residual_tangent_auction_stage1_core import (
    BirthGeometry,
    DimensionlessScaleSpec,
    affine_six_columns,
    build_two_birth_packet,
    carrier_six_columns,
    centered_fd_jvp_validation,
    checkerboard_fold_masks,
    evaluate_six_coefficient_action,
    freeze_pixel_support,
    make_no_opacity_parameterization,
    normalized_responsibilities,
    packet_linear_realization,
    realize_affine_six,
    realize_carrier_six,
    realize_two_birth_six,
    render_normalized,
    render_normalized_frozen_support,
    solve_two_birth_colors,
    support_membership_change_count,
    support_membership_change_summary,
    pixel_membership_sha256,
    uniform_linf_trust_scale,
)
from structsplat.gaussians import GaussianField


def _toy_field(*, opacity: bool = False) -> GaussianField:
    probabilities = np.asarray([0.8, 0.7, 0.75], dtype=np.float64)
    logits = np.log(probabilities / (1.0 - probabilities)) if opacity else None
    return GaussianField.from_numpy(
        # Keep fixture centers away from half-pixel AABB rounding discontinuities. The
        # scientific preflight is expected to fail, not smooth over, a parent on that boundary.
        means=np.asarray([[4.2, 4.4], [11.1, 6.8], [7.6, 12.2]], dtype=np.float64),
        scales=np.asarray([[3.1, 2.0], [2.4, 3.0], [2.8, 2.2]], dtype=np.float64),
        angles=np.asarray([0.21, -0.47, 0.72], dtype=np.float64),
        colors=np.asarray(
            [[0.22, 0.51, 0.74], [0.78, 0.31, 0.42], [0.29, 0.72, 0.36]],
            dtype=np.float64,
        ),
        opacities=logits,
        dtype=torch.float64,
    )


def _nonzero_base_step(parameterization):
    step = torch.zeros_like(parameterization.physical_vector)
    # Geometry and appearance both move; this is not a packet-only parity fixture.
    step[0] = 0.23
    step[3] = -0.19
    step[8] = 0.17
    step[13] = -0.16
    step[-4] = 0.14
    return step


def test_no_opacity_chart_uses_rotated_parent_axes_and_unit_fair_coordinates():
    field = _toy_field()
    spec = DimensionlessScaleSpec()
    parameterization = make_no_opacity_parameterization(
        field, 18, 20, sigma_cutoff=3.0, scale_spec=spec
    )
    assert parameterization.parameter_count == field.n * 8
    assert torch.equal(parameterization.frozen_radii, field.radii(3.0))

    step = torch.zeros_like(parameterization.physical_vector)
    # One local-major-axis unit maps through R(theta) diag(sx,sy), not global x/y scaling.
    step[0] = 1.0
    # One log-scale, radian, and display-linear RGB unit are physical units exactly.
    n = field.n
    step[2 * n] = 1.0
    step[4 * n] = 1.0
    step[5 * n] = 1.0
    updated = parameterization.field_at(step)
    expected_delta = field.scales()[0, 0] * torch.stack(
        (torch.cos(field.rotations[0]), torch.sin(field.rotations[0]))
    )
    assert torch.allclose(updated.means[0] - field.means[0], expected_delta)
    assert updated.log_scales[0, 0] - field.log_scales[0, 0] == pytest.approx(1.0)
    assert updated.rotations[0] - field.rotations[0] == pytest.approx(1.0)
    assert updated.colors[0, 0] - field.colors[0, 0] == pytest.approx(1.0)
    assert updated.opacities is None
    assert updated.color_grads is None

    with pytest.raises(ValueError, match="forbids opacity"):
        make_no_opacity_parameterization(_toy_field(opacity=True), 18, 20)


def test_centered_finite_difference_matches_jvp_on_frozen_support():
    parameterization = make_no_opacity_parameterization(_toy_field(), 18, 20)
    check = centered_fd_jvp_validation(parameterization, seed=91, epsilon=1e-4)
    assert check["relative_l2"] <= 1e-7
    assert check["max_abs"] <= 2e-8


def test_frozen_linearization_membership_differs_from_native_after_half_pixel_crossing():
    field = GaussianField.from_numpy(
        means=np.asarray([[4.49, 5.0], [8.0, 5.0]], dtype=np.float64),
        scales=np.asarray([[1.0, 0.8], [1.2, 1.0]], dtype=np.float64),
        angles=np.asarray([0.0, 0.0], dtype=np.float64),
        colors=np.asarray([[0.9, 0.1, 0.2], [0.1, 0.8, 0.4]], dtype=np.float64),
        dtype=torch.float64,
    )
    parameterization = make_no_opacity_parameterization(field, 12, 12)
    step = torch.zeros_like(parameterization.physical_vector)
    step[0] = 0.03  # x=4.52 crosses torch.round's half-pixel AABB boundary.
    updated = parameterization.field_at(step)
    summary = support_membership_change_summary(
        parameterization.frozen_support, updated
    )
    assert summary.xor_count == support_membership_change_count(
        parameterization.frozen_support, updated
    )
    assert summary.added_count > 0
    assert summary.removed_count > 0
    assert summary.xor_count == summary.added_count + summary.removed_count
    assert len(summary.parent_membership_sha256) == 64
    assert len(summary.native_membership_sha256) == 64
    assert summary.parent_membership_sha256 != summary.native_membership_sha256
    frozen = parameterization.render_step(step)
    independently_frozen = render_normalized_frozen_support(
        updated, parameterization.frozen_support
    )
    native = render_normalized(updated, 12, 12)
    assert torch.equal(frozen, independently_frozen)
    assert float((native - frozen).abs().max()) > 1e-4


def test_checkerboard_folds_are_disjoint_complete_and_swap_at_eight_pixel_tiles():
    discovery0, heldout0 = checkerboard_fold_masks(64, 64, 0)
    discovery1, heldout1 = checkerboard_fold_masks(64, 64, 1)
    assert discovery0.shape == (64 * 64 * 3,)
    assert not bool((discovery0 & heldout0).any())
    assert bool((discovery0 | heldout0).all())
    assert int(discovery0.sum()) == int(heldout0.sum()) == 64 * 64 * 3 // 2
    assert torch.equal(discovery0, heldout1)
    assert torch.equal(heldout0, discovery1)


def test_six_coefficient_fit_never_uses_heldout_values_and_oracle_is_separate():
    generator = torch.Generator().manual_seed(17)
    design = torch.randn(180, 6, generator=generator, dtype=torch.float64)
    truth = torch.tensor([0.3, -0.2, 0.12, 0.07, -0.18, 0.22], dtype=torch.float64)
    discovery = torch.zeros(180, dtype=torch.bool)
    discovery[::2] = True
    heldout = ~discovery
    residual = design @ truth
    first = evaluate_six_coefficient_action(design, residual, discovery, heldout)

    perturbed = residual.clone()
    perturbed[heldout] += 2.5 * torch.randn(
        int(heldout.sum()), generator=generator, dtype=torch.float64
    )
    second = evaluate_six_coefficient_action(design, perturbed, discovery, heldout)

    assert torch.allclose(
        first.crossfit.coefficients,
        second.crossfit.coefficients,
        atol=0.0,
        rtol=0.0,
    )
    assert first.crossfit.discovery_post_mse == second.crossfit.discovery_post_mse
    assert first.crossfit.heldout_post_mse != second.crossfit.heldout_post_mse
    assert not torch.allclose(
        first.all_pixel_oracle.coefficients,
        second.all_pixel_oracle.coefficients,
    )
    assert first.crossfit.heldout_post_mse <= 1e-25


def test_uniform_linf_trust_uses_one_scale_not_coordinate_clipping():
    coefficients = torch.tensor([2.0, -1.0, 0.2, -0.4, 0.0, 1.5])
    limited, factor = uniform_linf_trust_scale(coefficients, 0.75)
    assert factor == pytest.approx(0.375)
    assert float(limited.abs().max()) == pytest.approx(0.75)
    nonzero = coefficients != 0
    assert torch.allclose(limited[nonzero] / coefficients[nonzero], torch.full((5,), factor))
    assert not torch.equal(limited, coefficients.clamp(-0.75, 0.75))


def test_analytic_responsibilities_and_joint_base_affine_realize_exactly():
    field = _toy_field()
    parameterization = make_no_opacity_parameterization(field, 18, 20)
    updated = parameterization.field_at(_nonzero_base_step(parameterization))
    support = parameterization.frozen_support
    responsibility, _ = normalized_responsibilities(updated, 18, 20, support=support)
    analytic_base = torch.einsum("nhw,nc->hwc", responsibility, updated.colors)
    frozen_base = render_normalized_frozen_support(updated, support)
    assert torch.allclose(analytic_base, frozen_base, atol=2e-15, rtol=2e-14)

    coefficients = torch.tensor(
        [0.042, -0.031, 0.025, -0.018, 0.036, 0.021], dtype=torch.float64
    )
    design = affine_six_columns(updated, 1, 18, 20, support=support)
    predicted = packet_linear_realization(frozen_base, design, coefficients)
    gradients = torch.zeros(updated.n, 2, 3, dtype=torch.float64)
    gradients[1] = coefficients.reshape(2, 3)
    frozen_direct = render_normalized_frozen_support(
        updated, support, color_grads=gradients
    )
    native_direct = realize_affine_six(updated, 1, coefficients, 18, 20)
    assert torch.allclose(predicted, frozen_direct, atol=3e-15, rtol=3e-14)
    assert bool(torch.isfinite(native_direct).all())


def test_joint_base_carrier_matches_independent_attached_carrier_formula():
    parameterization = make_no_opacity_parameterization(_toy_field(), 18, 20)
    updated = parameterization.field_at(_nonzero_base_step(parameterization))
    support = parameterization.frozen_support
    frequency = (0.31, -0.18)
    phase = 0.27
    coefficients = torch.tensor(
        [0.05, -0.03, 0.02, -0.04, 0.01, 0.035], dtype=torch.float64
    )
    design = carrier_six_columns(
        updated, 2, frequency, 18, 20, phase_radians=phase, support=support
    )
    base = render_normalized_frozen_support(updated, support)
    predicted = packet_linear_realization(base, design, coefficients)
    direct = realize_carrier_six(
        updated,
        2,
        frequency,
        coefficients,
        18,
        20,
        phase_radians=phase,
    )
    assert bool(torch.isfinite(direct).all())

    # Independent image-space construction checks the declared sin-RGB/cos-RGB ordering.
    responsibility, _ = normalized_responsibilities(updated, 18, 20, support=support)
    yy, xx = torch.meshgrid(
        torch.arange(18, dtype=torch.float64),
        torch.arange(20, dtype=torch.float64),
        indexing="ij",
    )
    dx = xx - updated.means[2, 0]
    dy = yy - updated.means[2, 1]
    cosine, sine = torch.cos(updated.rotations[2]), torch.sin(updated.rotations[2])
    scales = updated.effective_scales()[2]
    local_x = (cosine * dx + sine * dy) / scales[0]
    local_y = (-sine * dx + cosine * dy) / scales[1]
    wave_phase = 2.0 * math.pi * (frequency[0] * local_x + frequency[1] * local_y) + phase
    delta = responsibility[2][..., None] * (
        torch.sin(wave_phase)[..., None] * coefficients[:3]
        + torch.cos(wave_phase)[..., None] * coefficients[3:]
    )
    assert torch.allclose(predicted, base + delta, atol=1e-15, rtol=1e-15)


def test_exact_joint_two_birth_common_denominator_solve_and_direct_render():
    parameterization = make_no_opacity_parameterization(_toy_field(), 18, 20)
    updated = parameterization.field_at(_nonzero_base_step(parameterization))
    geometries = (
        BirthGeometry(mean_xy=(3.1, 13.8), scale_xy=(1.5, 2.1), angle_radians=-0.24),
        BirthGeometry(mean_xy=(15.2, 3.4), scale_xy=(2.0, 1.4), angle_radians=0.39),
    )
    packet = build_two_birth_packet(
        updated,
        geometries,
        18,
        20,
        frozen_parent_support=parameterization.frozen_support,
    )
    assert packet.frozen_parent_support.membership.shape[0] == updated.n
    assert packet.birth_support.membership.shape[0] == 2
    assert pixel_membership_sha256(packet.birth_support.membership) != (
        pixel_membership_sha256(packet.frozen_parent_support.membership)
    )
    colors = torch.tensor([0.83, 0.16, 0.64, 0.12, 0.76, 0.28], dtype=torch.float64)
    predicted = packet_linear_realization(
        packet.base_render,
        packet.design,
        colors,
        offset=packet.offset,
    )
    direct = realize_two_birth_six(updated, packet, colors)
    assert float(packet.offset.abs().max()) > 1e-5
    assert bool(torch.isfinite(direct).all())

    # The finite common-denominator solve is exact on the frozen linearization support.
    solved = solve_two_birth_colors(packet, predicted)
    assert torch.allclose(solved, colors, atol=2e-13, rtol=2e-13)


def test_two_birth_native_parity_is_exact_without_a_parent_geometry_update():
    field = _toy_field()
    support = freeze_pixel_support(field, 18, 20)
    geometries = (
        BirthGeometry(mean_xy=(3.1, 13.8), scale_xy=(1.5, 2.1), angle_radians=-0.24),
        BirthGeometry(mean_xy=(15.2, 3.4), scale_xy=(2.0, 1.4), angle_radians=0.39),
    )
    packet = build_two_birth_packet(
        field, geometries, 18, 20, frozen_parent_support=support
    )
    colors = torch.tensor([0.83, 0.16, 0.64, 0.12, 0.76, 0.28], dtype=torch.float64)
    predicted = packet_linear_realization(
        packet.base_render, packet.design, colors, offset=packet.offset
    )
    native = realize_two_birth_six(field, packet, colors)
    assert torch.allclose(predicted, native, atol=3e-15, rtol=3e-14)
