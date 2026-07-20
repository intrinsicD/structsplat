import math

import numpy as np
import torch

from benchmarks import local_linear_reproducing_downstream_core as core
from structsplat.gaussians import GaussianField


def _corners(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = torch.tensor(((0, 0), (8, 0), (0, 8), (8, 8)), dtype=dtype)
    conics = torch.tensor((1.0 / 64.0, 0.0, 1.0 / 64.0), dtype=dtype).repeat(4, 1)
    radii = torch.full((4, 2), 8, dtype=torch.int64)
    return means, conics, radii


def _affine(means: torch.Tensor) -> torch.Tensor:
    u = means[:, 0] / 8.0
    v = means[:, 1] / 8.0
    return torch.stack(
        (0.10 + 0.35 * u + 0.25 * v, 0.80 - 0.30 * u - 0.20 * v,
         0.20 + 0.20 * u + 0.40 * v),
        dim=1,
    )


def _truth(dtype: torch.dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(9, dtype=dtype), torch.arange(9, dtype=dtype), indexing="ij"
    )
    return _affine(torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)).reshape(9, 9, 3)


def test_math64_qr_reproduces_affine_and_effective_moments() -> None:
    means, conics, radii = _corners(torch.float64)
    result = core.reference_qr(means, conics, _affine(means), radii, 9, 9)
    torch.testing.assert_close(result.output, _truth(torch.float64), rtol=0.0, atol=1e-14)
    assert bool((result.ranks == 3).all())
    assert float((result.partition - 1.0).abs().max()) <= 2e-15
    assert float(result.first_moment.abs().max()) <= 1e-15
    assert float(result.amplification_l1.max()) < 1.5


def test_float32_candidate_has_exact_15_channel_layout_and_reproduces_affine() -> None:
    means, conics, radii = _corners(torch.float32)
    result = core.candidate_cholesky(means, conics, _affine(means), radii, 9, 9)
    assert result.accumulators.shape == (81, 15)
    torch.testing.assert_close(result.mass, result.accumulators[:, 0], rtol=0.0, atol=0.0)
    assert bool(result.mass_pass.all())
    assert bool(result.contributor_pass.all())
    assert bool(result.solve_pass.all())
    torch.testing.assert_close(result.output, _truth(torch.float32), rtol=0.0, atol=2e-6)
    assert float((result.partition - 1.0).abs().max()) <= 2e-6
    assert float(result.first_moment.abs().max()) <= 2e-6


def test_float32_candidate_all_15_channels_match_asymmetric_manual_oracle() -> None:
    height, width = 7, 8
    means = torch.tensor(
        ((-0.4, 0.7), (6.6, 0.2), (1.3, 5.8), (7.2, 6.4), (3.4, 2.1)),
        dtype=torch.float32,
    )
    conics = torch.tensor(
        (
            (0.08, 0.01, 0.13),
            (0.11, -0.02, 0.07),
            (0.06, 0.015, 0.10),
            (0.09, 0.025, 0.12),
            (0.13, -0.03, 0.08),
        ),
        dtype=torch.float32,
    )
    colors = torch.tensor(
        ((0.1, 0.2, 0.8), (0.7, 0.3, 0.2), (0.4, 0.9, 0.5),
         (0.8, 0.6, 0.1), (0.3, 0.1, 0.9)),
        dtype=torch.float32,
    )
    radii = torch.tensor(((4, 4), (4, 3), (3, 4), (4, 4), (3, 3)), dtype=torch.int64)
    result = core.candidate_cholesky(means, conics, colors, radii, height, width)

    expected = torch.zeros((height * width, 15), dtype=torch.float32)
    coordinate_scale = float(max(height - 1, width - 1))
    for gid in range(means.shape[0]):
        center_x = int(torch.round(means[gid, 0]).item())
        center_y = int(torch.round(means[gid, 1]).item())
        x0 = max(0, center_x - int(radii[gid, 0].item()))
        x1 = min(width - 1, center_x + int(radii[gid, 0].item()))
        y0 = max(0, center_y - int(radii[gid, 1].item()))
        y1 = min(height - 1, center_y + int(radii[gid, 1].item()))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                pixel_x = torch.tensor(x, dtype=torch.float32)
                pixel_y = torch.tensor(y, dtype=torch.float32)
                pixel_minus_mean_x = pixel_x - means[gid, 0]
                pixel_minus_mean_y = pixel_y - means[gid, 1]
                q = (
                    conics[gid, 0] * pixel_minus_mean_x.square()
                    + 2.0
                    * conics[gid, 1]
                    * pixel_minus_mean_x
                    * pixel_minus_mean_y
                    + conics[gid, 2] * pixel_minus_mean_y.square()
                )
                weight = torch.exp(-0.5 * q)
                offset = torch.stack(
                    (means[gid, 0] - pixel_x, means[gid, 1] - pixel_y)
                ) / coordinate_scale
                manual = torch.cat(
                    (
                        weight.reshape(1),
                        weight * offset,
                        weight
                        * torch.stack(
                            (offset[0].square(), offset[0] * offset[1], offset[1].square())
                        ),
                        weight * colors[gid],
                        weight * offset[0] * colors[gid],
                        weight * offset[1] * colors[gid],
                    )
                )
                expected[y * width + x] += manual

    assert bool((expected.abs().sum(dim=0) > 0.0).all())
    torch.testing.assert_close(result.accumulators, expected, rtol=0.0, atol=0.0)


def test_candidate_phi_matches_independent_full_moment_solve_at_boundaries() -> None:
    height = width = 9
    means = torch.tensor(
        ((-0.2, 0.4), (8.1, -0.1), (0.3, 8.2), (7.7, 7.8), (4.2, 3.1)),
        dtype=torch.float32,
    )
    conics = torch.tensor(
        (
            (0.012, 0.001, 0.018),
            (0.016, -0.002, 0.011),
            (0.010, 0.0015, 0.014),
            (0.013, 0.002, 0.017),
            (0.020, -0.003, 0.012),
        ),
        dtype=torch.float32,
    )
    colors = torch.tensor(
        ((0.1, 0.2, 0.8), (0.7, 0.3, 0.2), (0.4, 0.9, 0.5),
         (0.8, 0.6, 0.1), (0.3, 0.1, 0.9)),
        dtype=torch.float32,
    )
    radii = torch.full((5, 2), 9, dtype=torch.int64)
    result = core.candidate_cholesky(means, conics, colors, radii, height, width)
    assert bool(result.solve_pass.all())

    means64 = means.to(torch.float64)
    conics64 = conics.to(torch.float64)
    colors64 = colors.to(torch.float64)
    origin = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64)
    for x, y in ((0, 0), (8, 0), (0, 8), (8, 8), (3, 5)):
        pixel = torch.tensor((x, y), dtype=torch.float64)
        offset = (means64 - pixel) / 8.0
        pixel_minus_mean = -offset * 8.0
        q = (
            conics64[:, 0] * pixel_minus_mean[:, 0].square()
            + 2.0
            * conics64[:, 1]
            * pixel_minus_mean[:, 0]
            * pixel_minus_mean[:, 1]
            + conics64[:, 2] * pixel_minus_mean[:, 1].square()
        )
        probability = torch.exp(-0.5 * q)
        probability /= probability.sum()
        design = torch.cat((torch.ones((5, 1), dtype=torch.float64), offset), dim=1)
        moment = torch.einsum("n,ni,nj->ij", probability, design, design)
        h = torch.linalg.solve(moment, origin)
        phi = probability * (design @ h)
        index = y * width + x
        torch.testing.assert_close(
            result.output[y, x].to(torch.float64), phi @ colors64, rtol=0.0, atol=5e-7
        )
        torch.testing.assert_close(
            result.partition[index].to(torch.float64), phi.sum(), rtol=0.0, atol=5e-7
        )
        torch.testing.assert_close(
            result.first_moment[index].to(torch.float64),
            torch.einsum("n,nd->d", phi, offset),
            rtol=0.0,
            atol=5e-7,
        )
        torch.testing.assert_close(
            result.amplification_l1[index].to(torch.float64),
            phi.abs().sum(),
            rtol=0.0,
            atol=5e-7,
        )


def test_reference_phi_matches_an_independent_three_by_three_moment_inverse() -> None:
    means, conics, radii = _corners(torch.float64)
    result = core.reference_qr(means, conics, _affine(means), radii, 9, 9)
    pixel = 4 * 9 + 4
    phi = result.effective_weights[pixel]
    d = (means - torch.tensor((4.0, 4.0), dtype=torch.float64)) / 8.0
    z = torch.cat((torch.ones((4, 1), dtype=torch.float64), d), dim=1)
    q = torch.einsum("ni,ni->n", d, d) / (1.0 / 64.0 * 8.0**2)
    weight = torch.exp(-0.5 * q)
    probability = weight / weight.sum()
    moment = torch.einsum("n,ni,nj->ij", probability, z, z)
    h = torch.linalg.solve(moment, torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64))
    independent = probability * (z @ h)
    torch.testing.assert_close(phi, independent, rtol=1e-13, atol=1e-14)


def test_low_rank_controls_are_diagnostic_total_and_never_solved() -> None:
    means = torch.tensor(((1.0, 4.0), (4.0, 4.0), (7.0, 4.0)), dtype=torch.float64)
    conics = torch.tensor((1.0 / 64.0, 0.0, 1.0 / 64.0), dtype=torch.float64).repeat(3, 1)
    radii = torch.full((3, 2), 8, dtype=torch.int64)
    result = core.reference_qr(means, conics, _affine(means), radii, 9, 9)
    assert bool((result.ranks == 2).all())
    assert bool(torch.isnan(result.output).all())
    assert bool(torch.isnan(result.partition).all())

    one = core.candidate_cholesky(
        torch.tensor(((4.0, 4.0),), dtype=torch.float32),
        torch.tensor(((1.0 / 64.0, 0.0, 1.0 / 64.0),), dtype=torch.float32),
        torch.tensor(((0.4, 0.5, 0.6),), dtype=torch.float32),
        torch.tensor(((8, 8),), dtype=torch.int64),
        9,
        9,
    )
    assert bool((~one.solve_pass).all())
    assert bool(torch.isnan(one.output).all())
    assert bool(torch.isnan(one.partition).all())
    assert bool(torch.isnan(one.amplification_l1).all())


def test_row_permutation_changes_enumeration_order_but_preserves_membership_and_output() -> None:
    means, conics, radii = _corners(torch.float64)
    colors = torch.tensor(
        ((0.1, 0.3, 0.7), (0.8, 0.2, 0.4), (0.2, 0.9, 0.5), (0.7, 0.6, 0.1)),
        dtype=torch.float64,
    )
    permutation = torch.tensor((3, 2, 1, 0), dtype=torch.int64)
    original_contributors = core.enumerate_contributors(means, radii, 9, 9)
    permuted_contributors = core.enumerate_contributors(means[permutation], radii[permutation], 9, 9)
    original_members = set(
        zip(
            original_contributors.gid.tolist(),
            original_contributors.px.tolist(),
            original_contributors.py.tolist(),
            strict=True,
        )
    )
    mapped_gid = permutation[permuted_contributors.gid]
    permuted_members = set(
        zip(
            mapped_gid.tolist(),
            permuted_contributors.px.tolist(),
            permuted_contributors.py.tolist(),
            strict=True,
        )
    )
    assert original_members == permuted_members
    assert original_contributors.gid.tolist() != mapped_gid.tolist()
    original = core.reference_qr(means, conics, colors, radii, 9, 9)
    reversed_result = core.reference_qr(
        means[permutation], conics[permutation], colors[permutation], radii[permutation], 9, 9
    )
    torch.testing.assert_close(original.output, reversed_result.output, rtol=0.0, atol=1e-12)


def test_conics_recomputation_matches_gaussian_field() -> None:
    log_scales = torch.tensor(((0.2, -0.4), (1.1, 0.3)), dtype=torch.float64)
    rotations = torch.tensor((0.3, -1.2), dtype=torch.float64)
    field = GaussianField(
        torch.zeros((2, 2), dtype=torch.float64),
        log_scales,
        rotations,
        torch.zeros((2, 3), dtype=torch.float64),
    )
    torch.testing.assert_close(
        core.conics_from_parameters(log_scales, rotations), field.conics(), rtol=0.0, atol=0.0
    )


def test_frozen_reference_gradient_matches_centered_difference() -> None:
    means, _, radii = _corners(torch.float64)
    logs = torch.full((4, 2), float(np.log(8.0)), dtype=torch.float64)
    rotations = torch.zeros(4, dtype=torch.float64)
    colors = torch.tensor(
        ((0.1, 0.2, 0.8), (0.7, 0.3, 0.2), (0.4, 0.9, 0.5), (0.8, 0.6, 0.1)),
        dtype=torch.float64,
    )
    contributors = core.enumerate_contributors(means, radii, 9, 9)
    means_ad = means.clone().requires_grad_(True)
    cotangent = torch.linspace(-1.0, 1.0, 9 * 9 * 3, dtype=torch.float64).reshape(9, 9, 3)
    cotangent /= torch.linalg.vector_norm(cotangent)
    loss = torch.sum(
        core.reference_render_frozen(
            means_ad, logs, rotations, colors, radii, contributors
        )
        * cotangent
    )
    gradient = torch.autograd.grad(loss, means_ad)[0] * 8.0
    direction = torch.arange(8, dtype=torch.float64).reshape(4, 2) - 3.5
    direction /= torch.linalg.vector_norm(direction)
    h = 2.0**-12
    plus = means + h * 8.0 * direction
    minus = means - h * 8.0 * direction
    plus_loss = torch.sum(
        core.reference_render_frozen(plus, logs, rotations, colors, radii, contributors) * cotangent
    )
    minus_loss = torch.sum(
        core.reference_render_frozen(minus, logs, rotations, colors, radii, contributors)
        * cotangent
    )
    finite_difference = (plus_loss - minus_loss) / (2.0 * h)
    torch.testing.assert_close(
        torch.sum(gradient * direction), finite_difference, rtol=1e-4, atol=1e-8
    )


def test_frozen_candidate_autograd_is_finite_for_all_blocks_and_matches_fd() -> None:
    means = torch.tensor(
        ((0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (8.0, 8.0), (4.0, 3.0)),
        dtype=torch.float32,
    )
    log_scales = torch.log(
        torch.tensor(((7.0, 5.0), (6.0, 8.0), (8.0, 6.0), (5.0, 7.0), (7.0, 7.0)))
    )
    rotations = torch.tensor((0.17, -0.31, 0.49, -0.73, 0.22), dtype=torch.float32)
    colors = torch.tensor(
        ((0.1, 0.2, 0.8), (0.7, 0.3, 0.2), (0.4, 0.9, 0.5),
         (0.8, 0.6, 0.1), (0.3, 0.1, 0.9)),
        dtype=torch.float32,
    )
    radii = torch.full((5, 2), 8, dtype=torch.int64)
    contributors = core.enumerate_contributors(means, radii, 9, 9)
    frozen_radii = radii.clone()
    frozen_contributors = tuple(
        value.clone() for value in (contributors.gid, contributors.px, contributors.py)
    )
    parameters = [
        value.clone().requires_grad_(True)
        for value in (means, log_scales, rotations, colors)
    ]
    cotangent = torch.linspace(-1.0, 1.0, 9 * 9 * 3, dtype=torch.float32).reshape(9, 9, 3)
    cotangent /= torch.linalg.vector_norm(cotangent)
    loss = torch.sum(
        core.candidate_render_frozen(*parameters, radii, contributors) * cotangent
    )
    gradients = torch.autograd.grad(loss, parameters)
    scales = (8.0, 1.0, math.pi, 1.0)
    step = 2.0**-12
    for block_index, (parameter, gradient, scale) in enumerate(
        zip(parameters, gradients, scales, strict=True)
    ):
        assert bool(torch.isfinite(gradient).all())
        assert float(torch.linalg.vector_norm(gradient)) > 1e-4
        direction = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        direction -= (parameter.numel() - 1) / 2.0
        direction /= torch.linalg.vector_norm(direction)
        plus = [value.detach().clone() for value in parameters]
        minus = [value.detach().clone() for value in parameters]
        plus[block_index] += step * scale * direction
        minus[block_index] -= step * scale * direction
        plus_loss = torch.sum(
            core.candidate_render_frozen(*plus, radii, contributors) * cotangent
        )
        minus_loss = torch.sum(
            core.candidate_render_frozen(*minus, radii, contributors) * cotangent
        )
        finite_difference = (plus_loss - minus_loss) / (2.0 * step)
        directional_ad = torch.sum(gradient * scale * direction)
        torch.testing.assert_close(
            directional_ad, finite_difference, rtol=5e-3, atol=5e-4
        )

    torch.testing.assert_close(radii, frozen_radii, rtol=0.0, atol=0.0)
    for actual, frozen in zip(
        (contributors.gid, contributors.px, contributors.py),
        frozen_contributors,
        strict=True,
    ):
        torch.testing.assert_close(actual, frozen, rtol=0.0, atol=0.0)
