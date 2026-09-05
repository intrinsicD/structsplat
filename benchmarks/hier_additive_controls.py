"""Small fixed-count additive optimization controls for HIER-033/035.

These are research controls, not maintained fit defaults. Constant RGB, three-sigma C0 fade,
no mask/opacity/filter/affine semantics. Exact local GN blocks omit cross-Gaussian coupling;
a finite global line search is authoritative. All work is exposed in the returned trace.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from structsplat.gaussians import GaussianField
from structsplat.pixel_gradient import pixel_gradient_packet
from structsplat.render import render_field


@dataclass(frozen=True)
class ControlConfig:
    arm: str = "adam"
    steps: int = 160
    adam_multiplier: float = 1.0
    damping: float = 0.01
    max_backtracks: int = 6
    trust: tuple = (1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    scale_min: float = 0.35
    scale_max: float = 16.0
    color_limit: float = 2.0
    max_pairs: int = 65536

    def __post_init__(self):
        if self.arm not in ("adam", "diagonal", "block"):
            raise ValueError("unknown control arm")
        for key in ("steps", "max_backtracks", "max_pairs"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for value in (self.adam_multiplier, self.damping, self.scale_min, self.scale_max,
                      self.color_limit, *self.trust):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("control scales must be finite and positive")
        if len(self.trust) != 8 or self.scale_min > self.scale_max:
            raise ValueError("invalid trust or scale interval")


def pack(field):
    return torch.cat((field.means, field.log_scales, field.rotations[:, None], field.colors), 1)


def unpack(values):
    return GaussianField(values[:, :2], values[:, 2:4], values[:, 4], values[:, 5:])


def additive_render(field, height, width, *, renderer="additive"):
    return render_field(field.means, field.conics(), field.colors, field.radii(3),
                        height, width, chunk=256, mode=renderer, scales=field.scales(),
                        rotations=field.rotations, support_fade=True, sigma_cutoff=3)


def bounded(values, height, width, cfg):
    # The same parameter-domain projection is applied to all optimizer arms.
    result = values.clone()
    result[:, 0].clamp_(0, width - 1)
    result[:, 1].clamp_(0, height - 1)
    result[:, 2:4].clamp_(math.log(cfg.scale_min), math.log(cfg.scale_max))
    result[:, 5:].clamp_(-cfg.color_limit, cfg.color_limit)
    return result


@torch.no_grad()
def curvature_direction(gradient, gram, cfg):
    """Return an 8-parameter step, normalized by each row's maximum trust ratio."""
    scale = gradient.new_tensor(cfg.trust)
    g = gradient * scale
    h = gram * scale[None, :, None] * scale[None, None, :]
    diagonal = h.diagonal(dim1=1, dim2=2)
    ridge = cfg.damping * diagonal.amax(1, keepdim=True).clamp_min(1e-12)
    if cfg.arm == "diagonal":
        direction = -g / (diagonal + ridge)
    elif cfg.arm == "block":
        matrix = h + torch.diag_embed(ridge.expand_as(diagonal))
        direction = -torch.linalg.solve(matrix, g[..., None]).squeeze(-1)
    else:
        raise ValueError("curvature direction requires diagonal or block")
    direction /= direction.abs().amax(1, keepdim=True).clamp_min(1.0)
    return direction * scale


def fit_control(initial, target, cfg=ControlConfig(), *, renderer="additive", callback=None):
    """Fit exactly cfg.steps, returning the terminal field and unselected temporal trace.

    Raw objective is 0.5*mean((render-target)^2). No best-checkpoint selection, densification,
    held-out decisions, or free line-search work. Adam uses parameter-group learning rates.
    Caller owns warmup, external contention disclosure and perceptual metric calculation.
    """
    if target.ndim != 3 or target.shape[-1] != 3 or initial.n < 1:
        raise ValueError("nonempty Gaussian field and HWC target required")
    if target.dtype != initial.means.dtype or target.device != initial.means.device:
        raise ValueError("field and target must share dtype/device")
    if not bool(torch.isfinite(target).all()) or not bool(((target >= 0) & (target <= 1)).all()):
        raise ValueError("target must be finite in [0,1]")
    if any(value is not None for value in (initial.opacities, initial.color_grads,
            initial.filter_variance, initial.scale_max, initial.background_mask)):
        raise ValueError("unsupported nonconstant or constrained field semantics")
    if not bool(torch.isfinite(pack(initial)).all()):
        raise ValueError("initial parameters must be finite")
    height, width = target.shape[:2]
    field = initial.detached().trainable()
    if not torch.equal(pack(field), bounded(pack(field), height, width, cfg)):
        raise ValueError("initial field outside frozen parameter bounds")
    params = (field.means, field.log_scales, field.rotations, field.colors)
    optimizer = None
    if cfg.arm == "adam":
        optimizer = torch.optim.Adam(
            [{"params": [p], "lr": lr * cfg.adam_multiplier}
             for p, lr in zip(params, (0.1, 0.03, 0.03, 0.03))],
            betas=(0.9, 0.999), eps=1e-8, foreach=False)
    forward_evals = gradient_evals = 0

    def synchronize():
        if target.is_cuda:
            torch.cuda.synchronize(target.device)

    def evaluate(candidate):
        nonlocal forward_evals
        with torch.set_grad_enabled(cfg.arm == "adam"):
            raw = additive_render(candidate, height, width, renderer=renderer)
            loss = 0.5 * (raw - target).square().mean()
        forward_evals += 1
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite control loss")
        return raw, loss

    def install(values):
        with torch.no_grad():
            for param, value in zip(params, (values[:, :2], values[:, 2:4], values[:, 4], values[:, 5:])):
                param.copy_(value)

    synchronize()
    started = time.perf_counter()
    raw, loss = evaluate(field)
    history = []

    def record(step, accepted, trials):
        synchronize()
        objective = float(loss.detach())
        row = {"iteration": step, "objective": objective,
               "psnr": -10 * math.log10(max(2 * objective, 1e-12)),
               "elapsed_seconds": time.perf_counter() - started, "accepted": accepted,
               "line_search_trials": trials, "forward_evaluations": forward_evals,
               "gradient_evaluations": gradient_evals}
        history.append(row)
        if callback is not None:
            callback(row)

    record(0, True, 0)
    for step in range(1, cfg.steps + 1):
        trials, accepted = 0, True
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_evals += 1
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError("nonfinite or missing Adam gradient")
            optimizer.step()
            with torch.no_grad():
                install(bounded(pack(field), height, width, cfg))
            raw, loss = evaluate(field)
        else:
            packet = pixel_gradient_packet(field, (raw - target).detach() / target.numel(),
                                            max_pairs=cfg.max_pairs)
            gradient_evals += 1
            delta = curvature_direction(packet.signed, packet.gram / target.numel(), cfg)
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError("nonfinite curvature update")
            original = pack(field).detach().clone()
            accepted = False
            for attempt in range(cfg.max_backtracks):
                trials += 1
                values = bounded(original + (0.5 ** attempt) * delta, height, width, cfg)
                candidate_raw, candidate_loss = evaluate(unpack(values))
                if float(candidate_loss) <= float(loss):
                    install(values)
                    raw, loss, accepted = candidate_raw, candidate_loss, True
                    break
            # Rejection leaves the exact previous state and its objective in place.
        if not bool(torch.isfinite(pack(field)).all()):
            raise FloatingPointError("nonfinite terminal parameters")
        record(step, accepted, trials)
    synchronize()
    return field.detached(), raw.detach(), history, time.perf_counter() - started

