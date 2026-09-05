import math

import pytest
import torch

from structsplat.gaussians import GaussianField
from structsplat.pixel_gradient import additive_pixel_jacobians, pixel_gradient_packet
from structsplat.render import render_field


def _render(field, height=13, width=15):
    return render_field(field.means, field.conics(), field.colors, field.radii(3),
                        height, width, mode="additive", scales=field.scales(),
                        rotations=field.rotations, support_fade=True, sigma_cutoff=3)


def _field():
    return GaussianField(
        torch.tensor([[4.2, 5.1], [8.3, 6.7]], dtype=torch.float64),
        torch.tensor([[1.5, 0.8], [1.2, 1.8]], dtype=torch.float64).log(),
        torch.tensor([0.27, -0.42], dtype=torch.float64),
        torch.tensor([[0.4, -0.1, 0.6], [0.2, 0.7, -0.3]], dtype=torch.float64),
    )


@pytest.mark.parametrize("max_pairs", [7, 65536])
def test_all_analytic_contribution_sums_match_autograd(max_pairs):
    field = _field().trainable()
    adjoint = torch.sin(torch.arange(13 * 15 * 3, dtype=torch.float64)).reshape(13, 15, 3)
    (_render(field) * adjoint).sum().backward()
    expected = torch.cat((field.means.grad, field.log_scales.grad,
                          field.rotations.grad[:, None], field.colors.grad), dim=1)
    packet = pixel_gradient_packet(field, adjoint, max_pairs=max_pairs)
    torch.testing.assert_close(packet.signed, expected, atol=1e-12, rtol=1e-12)
    assert bool((packet.absolute + 1e-14 >= packet.signed.abs()).all())
    assert bool((torch.linalg.eigvalsh(packet.gram) >= -1e-12).all())
    assert not packet.signed.requires_grad


def test_streamed_jacobian_predicts_finite_perturbation():
    field = _field()
    delta = torch.sin(torch.arange(16, dtype=torch.float64)).reshape(2, 8)
    predicted = torch.zeros(13 * 15, 3, dtype=torch.float64)
    for row, pixel, _w, jacobian, _h in additive_pixel_jacobians(field, 13, 15, max_pairs=9):
        assert row.numel() <= 9
        predicted.index_add_(0, pixel, (jacobian * delta[row, None, :]).sum(2))
    eps = 1e-6
    plus, minus = field.detached(), field.detached()
    for candidate, sign in ((plus, 1), (minus, -1)):
        candidate.means += sign * eps * delta[:, :2]
        candidate.log_scales += sign * eps * delta[:, 2:4]
        candidate.rotations += sign * eps * delta[:, 4]
        candidate.colors += sign * eps * delta[:, 5:]
    observed = (_render(plus) - _render(minus)) / (2 * eps)
    torch.testing.assert_close(predicted.reshape(13, 15, 3), observed, atol=1e-9, rtol=1e-7)


def test_absolute_before_reduction_detects_symmetric_width_error():
    field = GaussianField(torch.tensor([[7., 6.]], dtype=torch.float64),
                          torch.tensor([[1.8, 1.3]], dtype=torch.float64).log(),
                          torch.zeros(1, dtype=torch.float64),
                          torch.full((1, 3), 0.6, dtype=torch.float64))
    target_field = field.detached()
    target_field.log_scales[:, 0] += math.log(1.2)
    packet = pixel_gradient_packet(field, _render(field) - _render(target_field))
    assert packet.signed[0, 0].abs() < 1e-12
    assert packet.absolute[0, 0] > 0.01
    assert packet.signed[0, 2].abs() > 0.01


def test_split_curvature_predicts_small_symmetric_finite_split():
    field = _field().subset(torch.tensor([0]))
    target = torch.sin(torch.arange(13 * 15 * 3, dtype=torch.float64)).reshape(13, 15, 3) * 0.2
    base = _render(field)
    packet = pixel_gradient_packet(field, base - target)
    eigenvalues, eigenvectors = torch.linalg.eigh(packet.split_matrix[0])
    direction, eps = eigenvectors[:, 0], 1e-4
    split = GaussianField(torch.cat((field.means + eps * direction, field.means - eps * direction)),
                          field.log_scales.repeat(2, 1), field.rotations.repeat(2),
                          field.colors.repeat(2, 1) * 0.5)
    observed = 0.5 * ((_render(split) - target).square().sum() - (base - target).square().sum())
    predicted = 0.5 * eps**2 * eigenvalues[0]
    torch.testing.assert_close(observed, predicted, atol=3e-14, rtol=2e-5)


def test_unsupported_semantics_and_invalid_adjoint_are_rejected():
    field = _field()
    field.opacities = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="without opacity"):
        pixel_gradient_packet(field, torch.zeros(13, 15, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="HWC"):
        pixel_gradient_packet(_field(), torch.zeros(13, 15, dtype=torch.float64))
