"""Deterministic sanity check for the additive HIER gradient derivation."""

from __future__ import annotations

import json
import math

import torch

from structsplat.gaussians import GaussianField
from structsplat.render import render_field


HEIGHT = 7
WIDTH = 7
SIGMA_CUTOFF = 3.0


def _field(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
) -> GaussianField:
    return GaussianField(means, log_scales, rotations, colors).trainable()


def _render(field: GaussianField) -> torch.Tensor:
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(SIGMA_CUTOFF),
        HEIGHT,
        WIDTH,
        mode="additive",
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=True,
        sigma_cutoff=SIGMA_CUTOFF,
    )


def main() -> None:
    torch.set_default_dtype(torch.float64)
    ys, xs = torch.meshgrid(
        torch.arange(HEIGHT),
        torch.arange(WIDTH),
        indexing="ij",
    )
    pixels = torch.stack([xs, ys], dim=-1)
    means = torch.tensor([[3.1, 2.8]])
    log_scales = torch.tensor([[math.log(1.7), math.log(1.1)]])
    rotations = torch.tensor([0.37])
    colors = torch.tensor([[0.55, -0.2, 0.8]])
    field = _field(means, log_scales, rotations, colors)
    target = (
        torch.sin(pixels[..., :1] * 0.7 + pixels[..., 1:] * 0.2)
        * torch.tensor([0.2, -0.3, 0.4])
    )

    rendered = _render(field)
    loss = 0.5 * (rendered - target).square().sum()
    loss.backward()

    delta = pixels - field.means.detach()[0]
    cosine = torch.cos(field.rotations.detach()[0])
    sine = torch.sin(field.rotations.detach()[0])
    xi_x = cosine * delta[..., 0] + sine * delta[..., 1]
    xi_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    sx, sy = field.scales().detach()[0]
    raw = torch.exp(
        -0.5 * (xi_x.square() / sx.square() + xi_y.square() / sy.square())
    )
    cutoff = math.exp(-0.5 * SIGMA_CUTOFF**2)
    faded = torch.clamp(raw - cutoff, min=0.0)
    active = (faded > 0.0).to(raw.dtype)
    adjoint = (rendered - target).detach()
    pressure = (adjoint * field.colors.detach()[0]).sum(dim=-1)
    common = active * raw * pressure

    inv_sx2 = 1.0 / sx.square()
    inv_sy2 = 1.0 / sy.square()
    conic = torch.stack(
        [
            torch.stack(
                [
                    cosine.square() * inv_sx2 + sine.square() * inv_sy2,
                    cosine * sine * (inv_sx2 - inv_sy2),
                ]
            ),
            torch.stack(
                [
                    cosine * sine * (inv_sx2 - inv_sy2),
                    sine.square() * inv_sx2 + cosine.square() * inv_sy2,
                ]
            ),
        ]
    )
    conic_delta = torch.einsum("ab,...b->...a", conic, delta)
    analytic = {
        "color": (faded[..., None] * adjoint).sum(dim=(0, 1)),
        "mean": (common[..., None] * conic_delta).sum(dim=(0, 1)),
        "log_scale": torch.stack(
            [
                (common * xi_x.square() * inv_sx2).sum(),
                (common * xi_y.square() * inv_sy2).sum(),
            ]
        ),
        "rotation": (
            common * xi_x * xi_y * (inv_sy2 - inv_sx2)
        ).sum(),
    }
    autograd = {
        "color": field.colors.grad[0],
        "mean": field.means.grad[0],
        "log_scale": field.log_scales.grad[0],
        "rotation": field.rotations.grad[0],
    }
    gradient_max_abs_error = {
        name: float((analytic[name] - autograd[name]).abs().max())
        for name in analytic
    }

    gaussian_hessian = raw[..., None, None] * (
        conic_delta[..., :, None] * conic_delta[..., None, :] - conic
    )
    split_matrix = (
        (active * pressure)[..., None, None] * gaussian_hessian
    ).sum(dim=(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(split_matrix)
    direction = eigenvectors[:, 0]
    base_loss = loss.detach()
    split_checks = []
    for epsilon in (1e-2, 3e-3, 1e-3):
        split_field = _field(
            torch.stack(
                [
                    field.means.detach()[0] + epsilon * direction,
                    field.means.detach()[0] - epsilon * direction,
                ]
            ),
            field.log_scales.detach().repeat(2, 1),
            field.rotations.detach().repeat(2),
            0.5 * field.colors.detach().repeat(2, 1),
        )
        split_loss = 0.5 * (_render(split_field) - target).square().sum()
        observed = float((split_loss - base_loss).detach())
        predicted = float(
            0.5 * epsilon**2 * (direction @ split_matrix @ direction)
        )
        split_checks.append(
            {
                "epsilon": epsilon,
                "observed_loss_change": observed,
                "predicted_loss_change": predicted,
                "observed_over_predicted": observed / predicted,
            }
        )

    assert max(gradient_max_abs_error.values()) < 1e-12
    assert abs(split_checks[-1]["observed_over_predicted"] - 1.0) < 1e-5
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "dtype": "float64",
                "gradient_max_abs_error": gradient_max_abs_error,
                "split_min_eigenvalue": float(eigenvalues[0]),
                "split_checks": split_checks,
                "verdict": "pass",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
