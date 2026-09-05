"""HIER-033's finite, count-funded operator atlas; not a production topology policy."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from benchmarks.hier_additive_controls import additive_render, pack, unpack
from structsplat.gaussians import GaussianField
from structsplat.pixel_gradient import pixel_gradient_packet

FAMILIES = ("translation", "width", "rotation", "color", "two_lobes", "support_gap")
GROUPS = {"move": (0, 1), "scale": (2, 3), "rotate": (4,), "color": (5, 6, 7)}
MAGNITUDES = (0.5, 1.0)
TRUST = (2.0, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
DONORS = (1, 2)
DAMPING = 0.01
SHAPE = (64, 64)


@dataclass(frozen=True)
class FiniteAction:
    name: str
    family: str
    field: GaussianField
    predicted_gain: float
    donor: int | None
    magnitude: float


def fixture(family, seed):
    if family not in FAMILIES:
        raise ValueError("unknown oracle fixture")
    phase = (seed * 0.25) % math.pi
    donor_amplitude = (0.01, 0.05, 0.12)[seed % 3]
    field = GaussianField(
        torch.tensor([[36., 36.], [10., 10.], [10., 54.]]),
        torch.tensor([[5., 3.2], [2., 2.], [2., 2.]]).log(),
        torch.tensor([phase, 0., 0.]),
        torch.tensor([[0.5, 0.45, 0.35], [donor_amplitude, donor_amplitude * 0.6, donor_amplitude * 0.8],
                      [0.20, 0.12, 0.16]]),
    )
    truth = field.detached()
    axis = torch.tensor([math.cos(phase), math.sin(phase)])
    if family == "translation":
        truth.means[0] += 2 * axis
    elif family == "width":
        truth.log_scales[0, 0] += math.log(1.4)
    elif family == "rotation":
        truth.rotations[0] += 0.45
    elif family == "color":
        truth.colors[0] *= torch.tensor([1.25, 0.75, 1.1])
    elif family == "two_lobes":
        values = pack(field)
        children = values[0:1].repeat(2, 1)
        children[:, :2] += torch.stack((4 * axis, -4 * axis))
        children[:, 5:] *= 0.5
        truth = unpack(torch.cat((children, values[1:])))
    else:
        gap = torch.tensor([[56., 10., math.log(1.6), math.log(1.6), 0., 0.6, 0.5, 0.4]])
        truth = unpack(torch.cat((pack(field), gap)))
    target = additive_render(truth, *SHAPE).detach()
    if not bool(torch.isfinite(target).all()) or not bool(((target >= 0) & (target <= 1)).all()):
        raise ValueError("oracle fixture escaped image bounds")
    return field, target


def action_names():
    names = ["noop"]
    names += [f"{group}_m{int(magnitude * 10)}" for group in GROUPS for magnitude in MAGNITUDES]
    names += [f"split_d{donor}_m{int(magnitude * 10)}" for donor in DONORS for magnitude in MAGNITUDES]
    names += [f"birth_d{donor}" for donor in DONORS]
    return names


@torch.no_grad()
def finite_actions(field, target):
    """All candidates keep three rows, with every split/birth explicitly funding its donor."""
    if field.n != 3 or target.shape != (*SHAPE, 3):
        raise ValueError("oracle requires its three-row 64x64 fixture")
    base = additive_render(field, *SHAPE)
    residual = base - target
    packet = pixel_gradient_packet(field, residual / target.numel())
    g, h = packet.signed, packet.gram / target.numel()
    values = pack(field).detach()
    actions = [FiniteAction("noop", "noop", field.detached(), 0.0, None, 0.0)]
    trust = values.new_tensor(TRUST)
    for group, indices in GROUPS.items():
        index = torch.tensor(indices, device=values.device)
        scale = trust[index]
        gradient = g[0, index] * scale
        gram = h[0][index[:, None], index[None, :]] * scale[:, None] * scale[None, :]
        ridge = DAMPING * gram.diagonal().max().clamp_min(1e-12)
        direction = -torch.linalg.solve(gram + torch.eye(len(indices), device=values.device) * ridge, gradient)
        direction /= direction.abs().max().clamp_min(1.0)
        for magnitude in MAGNITUDES:
            delta = values.new_zeros(8)
            delta[index] = direction * scale * magnitude
            candidate = values.clone()
            candidate[0] += delta
            predicted = -(g[0] @ delta + 0.5 * delta @ h[0] @ delta)
            actions.append(FiniteAction(f"{group}_m{int(magnitude * 10)}", group,
                                         unpack(candidate), float(predicted), None, magnitude))
    eigenvalues, eigenvectors = torch.linalg.eigh(packet.split_matrix[0])
    axis = eigenvectors[:, 0]
    for donor in DONORS:
        color = values[donor, 5:]
        donor_loss = -g[donor, 5:] @ color + 0.5 * color @ h[donor, 5:, 5:] @ color
        for magnitude in MAGNITUDES:
            distance = float(field.scales()[0].min()) * magnitude
            children = values[0:1].repeat(2, 1)
            children[:, :2] += torch.stack((distance * axis, -distance * axis))
            children[:, 5:] *= 0.5
            survivor = next(row for row in DONORS if row != donor)
            candidate = unpack(torch.cat((children, values[survivor:survivor + 1])))
            prediction = -(0.5 * distance ** 2 * eigenvalues[0] + donor_loss)
            actions.append(FiniteAction(f"split_d{donor}_m{int(magnitude * 10)}", "split",
                                         candidate, float(prediction), donor, magnitude))
    pixel = int(residual.square().mean(2).flatten().argmax())
    y, x = divmod(pixel, SHAPE[1])
    for donor in DONORS:
        color = values[donor, 5:]
        donor_loss = -g[donor, 5:] @ color + 0.5 * color @ h[donor, 5:, 5:] @ color
        candidate = values.clone()
        candidate[donor, :2] = values.new_tensor((x, y))
        candidate[donor, 2:4] = math.log(1.6)
        candidate[donor, 4] = 0
        candidate[donor, 5:] = -residual[y, x] / (1 - math.exp(-4.5))
        # A deliberately local proxy: it does not pretend to price the new atom's full support.
        pixel_gain = 0.5 * residual[y, x].square().sum() / target.numel()
        actions.append(FiniteAction(f"birth_d{donor}", "birth", unpack(candidate),
                                     float(pixel_gain - donor_loss), donor, 1.0))
    if [a.name for a in actions] != action_names():
        raise RuntimeError("oracle action identity drift")
    if any(a.field.n != field.n or not bool(torch.isfinite(pack(a.field)).all()) for a in actions):
        raise RuntimeError("oracle action changed count or produced nonfinite parameters")
    return actions, packet, base
