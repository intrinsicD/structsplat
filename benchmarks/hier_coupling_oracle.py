"""HIER-036 bounded dense GN oracle; deliberately not a production optimizer."""
from __future__ import annotations

import time
import math
import torch

from benchmarks.hier_additive_controls import (
    ControlConfig, additive_render, bounded, pack, unpack,
)
from structsplat.pixel_gradient import additive_pixel_jacobians

MODES = ("block_row", "block_shared", "full_row", "full_shared")
MAX_JACOBIAN_BYTES = 64 * 1024 * 1024
MAX_PARAMETERS = 256


@torch.no_grad()
def dense_system(field, residual, *, max_jacobian_bytes=MAX_JACOBIAN_BYTES,
                 max_parameters=MAX_PARAMETERS, max_pairs=65536):
    """Materialize J, then return g and full GN Gram for 0.5*mean(residual**2).

    The byte guard covers the retained J array only, not temporary/Gram/solve allocations.
    Pair ownership is the analytic generator's unique (Gaussian,pixel) contract.
    """
    if residual.ndim != 3 or residual.shape[-1] != 3:
        raise ValueError("HWC RGB residual required")
    if residual.dtype != field.means.dtype or residual.device != field.means.device:
        raise ValueError("field and residual must share dtype/device")
    if not bool(torch.isfinite(residual).all()) or field.n < 1:
        raise ValueError("finite residual and nonempty field required")
    for value in (max_jacobian_bytes, max_parameters):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("positive integer dense limits required")
    height, width = residual.shape[:2]
    p = field.n * 8
    required = residual.numel() * p * residual.element_size()
    if p > max_parameters or required > max_jacobian_bytes:
        raise MemoryError("dense oracle retained-array/parameter ceiling exceeded")
    matrix = residual.new_zeros((residual.numel(), p))
    channel = torch.arange(3, device=residual.device)[None, :, None]
    parameter = torch.arange(8, device=residual.device)[None, None, :]
    for gid, pixel, _weight, jacobian, _hessian in additive_pixel_jacobians(
            field, height, width, max_pairs=max_pairs):
        matrix[pixel[:, None, None] * 3 + channel,
               gid[:, None, None] * 8 + parameter] = jacobian
    gradient = matrix.T @ residual.reshape(-1) / residual.numel()
    gram = matrix.T @ matrix / residual.numel()
    if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(gram).all()):
        raise FloatingPointError("nonfinite dense GN system")
    return gradient.reshape(field.n, 8), gram, required


@torch.no_grad()
def dense_direction(gradient, gram, cfg, mode):
    """Only cross-row entries and cap scope vary across the four factorial arms."""
    if mode not in MODES or cfg.arm != "block":
        raise ValueError("unknown dense mode or incompatible control configuration")
    n, parameters = gradient.shape
    if parameters != 8 or gram.shape != (n * 8, n * 8):
        raise ValueError("invalid dense system shapes")
    scale = gradient.new_tensor(cfg.trust).repeat(n)
    scaled_gradient = gradient.reshape(-1) * scale
    scaled_gram = gram * scale[:, None] * scale[None, :]
    diagonal = scaled_gram.diagonal().reshape(n, 8)
    ridge = cfg.damping * diagonal.amax(1).clamp_min(1e-12)
    row = torch.arange(n * 8, device=gradient.device) // 8
    same_row = row[:, None] == row[None, :]
    cross_fraction = float(scaled_gram.masked_fill(same_row, 0).norm()
                           / scaled_gram.norm().clamp_min(1e-30))
    matrix = scaled_gram if mode.startswith("full") else scaled_gram * same_row
    matrix = matrix + torch.diag(ridge.repeat_interleave(8))
    direction = -torch.linalg.solve(matrix, scaled_gradient).reshape(n, 8)
    cap = direction.abs().amax(1, keepdim=True) if mode.endswith("_row") else direction.abs().amax()
    direction = direction / cap.clamp_min(1)
    delta = direction * scale.reshape(n, 8)
    if not bool(torch.isfinite(delta).all()):
        raise FloatingPointError("nonfinite dense direction")
    return delta, {"directional_derivative": float((gradient * delta).sum()),
                   "cross_gram_fraction": cross_fraction}


def fit_coupling(initial, target, cfg=ControlConfig(arm="block"), *,
                 mode="full_shared", renderer="additive", callback=None):
    """Exactly cfg.steps terminal updates; every dense construction/solve/render is charged."""
    if mode not in MODES or cfg.arm != "block":
        raise ValueError("dense curvature configuration required")
    if (target.ndim != 3 or target.shape[-1] != 3 or initial.n < 1
            or target.dtype != initial.means.dtype or target.device != initial.means.device):
        raise ValueError("nonempty field and matching HWC RGB target required")
    if not bool(torch.isfinite(target).all()) or not bool(((target >= 0) & (target <= 1)).all()):
        raise ValueError("target must be finite in [0,1]")
    if any(value is not None for value in (initial.opacities, initial.color_grads,
            initial.filter_variance, initial.scale_max, initial.background_mask)):
        raise ValueError("unsupported field semantics")
    if not bool(torch.isfinite(pack(initial)).all()):
        raise ValueError("nonfinite initial parameters")
    height, width = target.shape[:2]
    values = pack(initial).detach().clone()
    if not torch.equal(values, bounded(values, height, width, cfg)):
        raise ValueError("initial field outside frozen bounds")
    if initial.n * 8 > MAX_PARAMETERS or target.numel() * initial.n * 8 * target.element_size() > MAX_JACOBIAN_BYTES:
        raise MemoryError("dense oracle ceiling exceeded before fitting")
    forwards = 0

    def sync():
        if target.is_cuda:
            torch.cuda.synchronize(target.device)

    @torch.no_grad()
    def evaluate(candidate):
        nonlocal forwards
        raw = additive_render(unpack(candidate), height, width, renderer=renderer)
        loss = 0.5 * (raw - target).square().mean()
        forwards += 1
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite trial loss")
        return raw, loss

    sync()
    started = time.perf_counter()
    raw, loss = evaluate(values)
    history = []

    def record(step, accepted, trials, telemetry):
        sync()
        objective = float(loss)
        row = {"iteration": step, "objective": objective,
               "psnr": -10 * math.log10(max(2 * objective, 1e-12)),
               "elapsed_seconds": time.perf_counter() - started, "accepted": accepted,
               "line_search_trials": trials, "forward_evaluations": forwards,
               "gradient_evaluations": step, "jacobian_constructions": step,
               "linear_solves": step, **telemetry}
        history.append(row)
        if callback:
            callback(row)

    record(0, True, 0, {"directional_derivative": 0., "cross_gram_fraction": 0.})
    for step in range(1, cfg.steps + 1):
        gradient, gram, _bytes = dense_system(unpack(values), raw - target, max_pairs=cfg.max_pairs)
        delta, telemetry = dense_direction(gradient, gram, cfg, mode)
        accepted = False
        for attempt in range(cfg.max_backtracks):
            trial = bounded(values + 0.5 ** attempt * delta, height, width, cfg)
            trial_raw, trial_loss = evaluate(trial)
            if float(trial_loss) <= float(loss):
                values, raw, loss, accepted = trial, trial_raw, trial_loss, True
                break
        record(step, accepted, attempt + 1, telemetry)
    sync()
    return unpack(values).detached(), raw.detach(), history, time.perf_counter() - started
