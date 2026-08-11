"""Mass-free normalized-to-additive optimization continuation (HIER-023).

The maintained renderer dispatch remains unchanged.  The frozen path calls the ordinary
normalized renderer during its hold, composes existing additive accumulations only during the
short transition, and calls the ordinary additive renderer throughout its endpoint tail.  The
returned object is a plain additive ``GaussianField`` with no gauge or optimizer payload.

Torch imports stay local so importing :mod:`structsplat` remains NumPy-only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import TYPE_CHECKING, Literal

import numpy as np

from .additive_continuation import (
    _finite,
    _image,
    _integer,
    _readonly,
    _tensor_quantiles,
    _validate_field,
)


if TYPE_CHECKING:
    from .gaussians import GaussianField


UnitGaugePhase = Literal["hold", "anneal", "endpoint"]
PixelLoss = Literal["l1", "l2"]

_HOLD_FRACTION = 0.35
_ANNEAL_FRACTION = 0.15
_NORMALIZED_RENDERERS = frozenset(
    (
        "normalized",
        "cuda",
        "cuda_normalized",
        "cuda_block_reduce",
        "cuda_tiled",
        "cuda_tiled_normalized",
    )
)
_ADDITIVE_RENDERERS = frozenset(("additive", "cuda_additive", "cuda_tiled_additive"))


@dataclass(frozen=True)
class UnitGaugeContinuationConfig:
    """Frozen HIER-023 renderer and optimizer controls."""

    steps: int = 500
    checkpoint_every: int = 25
    lr_means: float = 5e-2
    lr_scales: float = 3e-2
    lr_rotations: float = 1e-2
    lr_coefficients: float = 3e-2
    pixel_loss: PixelLoss = "l1"
    ssim_weight: float = 0.3
    normalization_eps: float = 1e-8
    coefficient_abs_limit: float = 16.0
    min_scale_px: float = 0.35
    sigma_cutoff: float = 3.0
    aa_dilation: float = 0.0
    render_chunk: int = 256
    normalized_renderer: str = "normalized"
    additive_renderer: str = "additive"
    support_fade: bool = False
    render_checkpoint: bool = False
    reset_optimizer_at_endpoint: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", _integer(self.steps, "steps", minimum=4))
        object.__setattr__(
            self,
            "checkpoint_every",
            _integer(self.checkpoint_every, "checkpoint_every", minimum=1),
        )
        object.__setattr__(
            self,
            "render_chunk",
            _integer(self.render_chunk, "render_chunk", minimum=1),
        )
        for name in (
            "lr_means",
            "lr_scales",
            "lr_rotations",
            "lr_coefficients",
            "normalization_eps",
            "coefficient_abs_limit",
            "min_scale_px",
            "sigma_cutoff",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, minimum=0.0, strict_minimum=True),
            )
        object.__setattr__(
            self,
            "ssim_weight",
            _finite(self.ssim_weight, "ssim_weight", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "aa_dilation",
            _finite(self.aa_dilation, "aa_dilation", minimum=0.0),
        )
        if self.pixel_loss not in ("l1", "l2"):
            raise ValueError("pixel_loss must be 'l1' or 'l2'")
        if self.normalized_renderer not in _NORMALIZED_RENDERERS:
            expected = ", ".join(sorted(_NORMALIZED_RENDERERS))
            raise ValueError(f"unknown normalized renderer; expected one of {expected}")
        if self.additive_renderer not in _ADDITIVE_RENDERERS:
            expected = ", ".join(sorted(_ADDITIVE_RENDERERS))
            raise ValueError(f"unknown additive renderer; expected one of {expected}")
        for name in ("support_fade", "render_checkpoint", "reset_optimizer_at_endpoint"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        unit_gauge_phase_lengths(self.steps)


@dataclass(frozen=True)
class UnitGaugeSchedulePoint:
    step: int
    phase: UnitGaugePhase
    lambda_value: float
    endpoint_eligible: bool


def unit_gauge_phase_lengths(steps: int) -> tuple[int, int, int]:
    """Return integer 35/15/50 hold, anneal, and exact-additive lengths."""

    total = _integer(steps, "steps", minimum=4)
    hold = int(round(_HOLD_FRACTION * total))
    anneal = int(round(_ANNEAL_FRACTION * total))
    endpoint = total - hold - anneal
    if min(hold, anneal, endpoint) < 1:
        raise ValueError(
            "steps must allocate at least one operation to hold, anneal, and endpoint"
        )
    return hold, anneal, endpoint


def unit_gauge_schedule(step: int, steps: int) -> UnitGaugeSchedulePoint:
    """Return the frozen schedule; the final anneal value remains strictly positive."""

    current = _integer(step, "step", minimum=1)
    hold, anneal, _ = unit_gauge_phase_lengths(steps)
    if current > steps:
        raise ValueError(f"step must be <= steps ({steps}), got {current}")
    if current <= hold:
        return UnitGaugeSchedulePoint(current, "hold", 1.0, False)
    if current <= hold + anneal:
        progress = (current - hold) / float(anneal + 1)
        value = 0.5 * (1.0 + math.cos(math.pi * progress))
        return UnitGaugeSchedulePoint(current, "anneal", value, False)
    return UnitGaugeSchedulePoint(current, "endpoint", 0.0, True)


@dataclass(frozen=True)
class UnitGaugeRender:
    """Differentiable output plus optional transition components and call accounting."""

    image: object
    numerator: object | None
    denominator: object | None
    normalized_calls: int
    additive_numerator_calls: int
    additive_denominator_calls: int

    @property
    def renderer_calls(self) -> int:
        return (
            self.normalized_calls
            + self.additive_numerator_calls
            + self.additive_denominator_calls
        )


@dataclass(frozen=True)
class UnitGaugeCheckpoint:
    step: int
    phase: str
    lambda_value: float
    endpoint_eligible: bool
    selected: bool
    finite: bool
    reconstruction_loss: float
    raw_l1: float
    raw_mse: float
    raw_psnr_db: float
    ssim: float
    coverage_loss: float
    denominator_min: float
    denominator_q01: float
    denominator_q05: float
    denominator_q50: float
    denominator_q95: float
    denominator_q99: float
    denominator_max: float
    coefficient_abs_max: float
    optimizer_reset_count: int
    optimizer_reset_step: int | None
    normalized_calls: int
    additive_numerator_calls: int
    additive_denominator_calls: int
    renderer_calls: int
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class UnitGaugeContinuationResult:
    """A strict additive field and scalar-only path telemetry."""

    field: "GaussianField"
    reconstruction_raw: np.ndarray
    checkpoints: tuple[UnitGaugeCheckpoint, ...]
    selected_step: int | None
    attempted_steps: int
    completed_steps: int
    status: str
    optimizer_reset_count: int
    optimizer_reset_step: int | None
    normalized_calls: int
    additive_numerator_calls: int
    additive_denominator_calls: int
    renderer_calls: int
    endpoint_parity_max_abs: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        expected = 0 if self.selected_step is None else 1
        if sum(checkpoint.selected for checkpoint in self.checkpoints) != expected:
            raise ValueError("checkpoint selection and selected_step disagree")
        if self.field.opacities is not None:
            raise ValueError("unit-gauge endpoints cannot retain opacity or mass")
        if self.field.color_grads is not None:
            raise ValueError("HIER-023 endpoints require constant RGB coefficients")
        total = (
            self.normalized_calls
            + self.additive_numerator_calls
            + self.additive_denominator_calls
        )
        if self.renderer_calls != total:
            raise ValueError("renderer call total disagrees with per-equation counts")

    @property
    def completed(self) -> bool:
        return self.status == "completed" and self.completed_steps == self.attempted_steps

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


def _render(
    field: "GaussianField",
    colors,
    height: int,
    width: int,
    config: UnitGaugeContinuationConfig,
    renderer: str,
):
    from .render import render_field

    return render_field(
        field.means,
        field.conics(config.aa_dilation),
        colors,
        field.radii(config.sigma_cutoff, config.aa_dilation),
        height,
        width,
        chunk=config.render_chunk,
        mode=renderer,
        opacities=None,
        scales=field.effective_scales(),
        rotations=field.rotations,
        support_fade=config.support_fade,
        sigma_cutoff=config.sigma_cutoff,
        checkpoint_chunks=config.render_checkpoint,
        normalization_eps=config.normalization_eps,
    )


def render_unit_gauge(
    field: "GaussianField",
    height: int,
    width: int,
    lambda_value: float,
    *,
    config: UnitGaugeContinuationConfig | None = None,
) -> UnitGaugeRender:
    """Render the exact endpoint dispatch or the mass-free intermediate quotient.

    ``lambda=1`` calls the configured ordinary normalized renderer directly. ``lambda=0``
    returns the ordinary additive output as the very same object as ``numerator``.  Only strict
    intermediate values perform two additive accumulations.
    """

    import torch

    candidate = _validate_field(field)
    cfg = config or UnitGaugeContinuationConfig()
    h = _integer(height, "height", minimum=1)
    w = _integer(width, "width", minimum=1)
    lam = _finite(lambda_value, "lambda_value", minimum=0.0, maximum=1.0)
    if candidate.means.device.type == "cpu" and (
        cfg.normalized_renderer != "normalized" or cfg.additive_renderer != "additive"
    ):
        raise ValueError("CUDA renderers require a CUDA field")
    if lam == 1.0:
        image = _render(candidate, candidate.colors, h, w, cfg, cfg.normalized_renderer)
        return UnitGaugeRender(image, None, None, 1, 0, 0)
    numerator = _render(candidate, candidate.colors, h, w, cfg, cfg.additive_renderer)
    if lam == 0.0:
        return UnitGaugeRender(numerator, numerator, None, 0, 1, 0)
    denominator_rgb = _render(
        candidate,
        torch.ones_like(candidate.colors),
        h,
        w,
        cfg,
        cfg.additive_renderer,
    )
    denominator = denominator_rgb[..., :1]
    image = numerator / (lam * (denominator + cfg.normalization_eps) + (1.0 - lam))
    return UnitGaugeRender(image, numerator, denominator, 0, 1, 1)


def _unit_denominator(
    field: "GaussianField", height: int, width: int, config: UnitGaugeContinuationConfig
):
    import torch

    return _render(
        field,
        torch.ones_like(field.colors),
        height,
        width,
        config,
        config.additive_renderer,
    )[..., :1]


def _project_parameters(
    field: "GaussianField", height: int, width: int, config: UnitGaugeContinuationConfig
) -> None:
    import torch

    with torch.no_grad():
        field.means[:, 0].clamp_(0.0, float(width - 1))
        field.means[:, 1].clamp_(0.0, float(height - 1))
        field.rotations.remainder_(math.pi)
        lower = math.log(config.min_scale_px)
        upper = torch.full_like(field.log_scales, math.log(float(max(height, width))))
        if field.scale_max is not None:
            upper = torch.minimum(
                upper,
                torch.log(field.scale_max.to(upper).clamp_min(config.min_scale_px)),
            )
        field.log_scales.copy_(torch.maximum(field.log_scales, torch.full_like(upper, lower)))
        field.log_scales.copy_(torch.minimum(field.log_scales, upper))
        field.colors.clamp_(-config.coefficient_abs_limit, config.coefficient_abs_limit)


def _parameter_state_is_finite(field: "GaussianField") -> bool:
    import torch

    return all(
        bool(torch.isfinite(value).all())
        for value in (field.means, field.log_scales, field.rotations, field.colors)
    )


def _clone_state(field: "GaussianField") -> tuple[object, ...]:
    return (
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        field.colors.detach().clone(),
    )


def _restore_state(field: "GaussianField", state: tuple[object, ...]) -> None:
    import torch

    with torch.no_grad():
        for parameter, value in zip(
            (field.means, field.log_scales, field.rotations, field.colors),
            state,
            strict=True,
        ):
            parameter.copy_(value)


def _endpoint_field(field: "GaussianField") -> "GaussianField":
    from .gaussians import GaussianField

    return GaussianField(
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        field.colors.detach().clone(),
        opacities=None,
        scale_max=None,
        color_grads=None,
        background_mask=None,
        filter_variance=(
            None if field.filter_variance is None else field.filter_variance.detach().clone()
        ),
    )


def _optimizer(field: "GaussianField", config: UnitGaugeContinuationConfig):
    import torch

    return torch.optim.Adam(
        [
            {"params": [field.means], "lr": config.lr_means},
            {"params": [field.log_scales], "lr": config.lr_scales},
            {"params": [field.rotations], "lr": config.lr_rotations},
            {"params": [field.colors], "lr": config.lr_coefficients},
        ]
    )


def _account(calls: dict[str, int], rendered: UnitGaugeRender) -> None:
    calls["normalized"] += rendered.normalized_calls
    calls["additive_numerator"] += rendered.additive_numerator_calls
    calls["additive_denominator"] += rendered.additive_denominator_calls


def _checkpoint(
    *,
    step: int,
    phase: str,
    point_lambda: float,
    endpoint_eligible: bool,
    rendered: UnitGaugeRender,
    denominator,
    target,
    field: "GaussianField",
    reconstruction_loss: float,
    ssim_value: float,
    optimizer_reset_count: int,
    optimizer_reset_step: int | None,
    calls: dict[str, int],
    started: float,
) -> UnitGaugeCheckpoint:
    import torch

    with torch.no_grad():
        residual = rendered.image - target
        raw_l1 = float(residual.abs().mean().cpu())
        raw_mse = float(residual.square().mean().cpu())
        den_stats = _tensor_quantiles(denominator)
        finite = bool(
            torch.isfinite(rendered.image).all()
            and torch.isfinite(denominator).all()
            and _parameter_state_is_finite(field)
        )
    total = sum(calls.values())
    return UnitGaugeCheckpoint(
        step=step,
        phase=phase,
        lambda_value=float(point_lambda),
        endpoint_eligible=endpoint_eligible,
        selected=False,
        finite=finite,
        reconstruction_loss=float(reconstruction_loss),
        raw_l1=raw_l1,
        raw_mse=raw_mse,
        raw_psnr_db=-10.0 * math.log10(max(raw_mse, 1e-12)),
        ssim=float(ssim_value),
        coverage_loss=float((denominator - 1.0).square().mean().cpu()),
        denominator_min=den_stats[0],
        denominator_q01=den_stats[1],
        denominator_q05=den_stats[2],
        denominator_q50=den_stats[3],
        denominator_q95=den_stats[4],
        denominator_q99=den_stats[5],
        denominator_max=den_stats[6],
        coefficient_abs_max=float(field.colors.abs().max().cpu()),
        optimizer_reset_count=optimizer_reset_count,
        optimizer_reset_step=optimizer_reset_step,
        normalized_calls=calls["normalized"],
        additive_numerator_calls=calls["additive_numerator"],
        additive_denominator_calls=calls["additive_denominator"],
        renderer_calls=total,
        elapsed_seconds=time.perf_counter() - started,
    )


def fit_unit_gauge_continuation(
    field: "GaussianField",
    target: np.ndarray,
    *,
    config: UnitGaugeContinuationConfig | None = None,
    verbose: bool = False,
) -> UnitGaugeContinuationResult:
    """Fit the frozen path and return only the best exact-additive-tail checkpoint."""

    import torch

    from . import metrics as M

    started = time.perf_counter()
    source_field = _validate_field(field)
    source = _image(target, "target")
    height, width = source.shape[:2]
    cfg = config or UnitGaugeContinuationConfig()
    if source_field.means.device.type == "cpu" and (
        cfg.normalized_renderer != "normalized" or cfg.additive_renderer != "additive"
    ):
        raise ValueError("CUDA renderers require a CUDA field")

    candidate = source_field.detached()
    candidate.trainable()
    target_tensor = torch.as_tensor(
        source, device=candidate.means.device, dtype=candidate.means.dtype
    )
    optimizer = _optimizer(candidate, cfg)
    target_stats = M.SSIMTargetStats(target_tensor) if cfg.ssim_weight > 0.0 else None
    calls = {"normalized": 0, "additive_numerator": 0, "additive_denominator": 0}
    checkpoints: list[UnitGaugeCheckpoint] = []
    reset_count = 0
    reset_step: int | None = None

    with torch.no_grad():
        initial = render_unit_gauge(candidate, height, width, 1.0, config=cfg)
        _account(calls, initial)
        denominator = _unit_denominator(candidate, height, width, cfg)
        calls["additive_denominator"] += 1
        initial_residual = initial.image - target_tensor
        initial_pixel = (
            initial_residual.abs().mean()
            if cfg.pixel_loss == "l1"
            else initial_residual.square().mean()
        )
        initial_ssim = float(
            M.ssim(initial.image, target_tensor, target_stats=target_stats)
        )
        initial_loss = float(
            (
                (1.0 - cfg.ssim_weight) * initial_pixel
                + cfg.ssim_weight * (1.0 - initial_ssim)
            ).cpu()
        )
    checkpoints.append(
        _checkpoint(
            step=0,
            phase="initial",
            point_lambda=1.0,
            endpoint_eligible=False,
            rendered=initial,
            denominator=denominator,
            target=target_tensor,
            field=candidate,
            reconstruction_loss=initial_loss,
            ssim_value=initial_ssim,
            optimizer_reset_count=reset_count,
            optimizer_reset_step=reset_step,
            calls=calls,
            started=started,
        )
    )

    hold_steps, anneal_steps, _ = unit_gauge_phase_lengths(cfg.steps)
    endpoint_start = hold_steps + anneal_steps + 1
    best_state: tuple[object, ...] | None = None
    best_image = None
    best_index: int | None = None
    best_key: tuple[float, int] | None = None
    completed_steps = 0
    status = "completed"
    last_finite_state = _clone_state(candidate)

    for step in range(1, cfg.steps + 1):
        point = unit_gauge_schedule(step, cfg.steps)
        if (
            point.phase == "endpoint"
            and cfg.reset_optimizer_at_endpoint
            and reset_count == 0
        ):
            optimizer = _optimizer(candidate, cfg)
            reset_count = 1
            reset_step = step
        optimizer.zero_grad(set_to_none=True)
        rendered = render_unit_gauge(
            candidate, height, width, point.lambda_value, config=cfg
        )
        _account(calls, rendered)
        residual = rendered.image - target_tensor
        pixel = residual.abs().mean() if cfg.pixel_loss == "l1" else residual.square().mean()
        if cfg.ssim_weight > 0.0:
            ssim_value_t = M.ssim(
                rendered.image, target_tensor, target_stats=target_stats
            )
        else:
            ssim_value_t = pixel.new_tensor(1.0)
        loss_t = (
            (1.0 - cfg.ssim_weight) * pixel
            + cfg.ssim_weight * (1.0 - ssim_value_t)
        )
        if not bool(torch.isfinite(loss_t)):
            status = "nonfinite_objective"
            break
        loss_t.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in (
                candidate.means,
                candidate.log_scales,
                candidate.rotations,
                candidate.colors,
            )
        ):
            status = "nonfinite_gradient"
            break
        optimizer.step()
        _project_parameters(candidate, height, width, cfg)
        if not _parameter_state_is_finite(candidate):
            _restore_state(candidate, last_finite_state)
            status = "nonfinite_parameters"
            break
        completed_steps = step
        last_finite_state = _clone_state(candidate)

        if not (
            step == hold_steps
            or step == endpoint_start
            or step == cfg.steps
            or step % cfg.checkpoint_every == 0
        ):
            continue
        with torch.no_grad():
            evaluated = render_unit_gauge(
                candidate, height, width, point.lambda_value, config=cfg
            )
            _account(calls, evaluated)
            if evaluated.denominator is None:
                eval_denominator = _unit_denominator(candidate, height, width, cfg)
                calls["additive_denominator"] += 1
            else:
                eval_denominator = evaluated.denominator
            eval_residual = evaluated.image - target_tensor
            eval_pixel = (
                eval_residual.abs().mean()
                if cfg.pixel_loss == "l1"
                else eval_residual.square().mean()
            )
            eval_ssim = float(
                M.ssim(evaluated.image, target_tensor, target_stats=target_stats)
            )
            eval_loss = float(
                (
                    (1.0 - cfg.ssim_weight) * eval_pixel
                    + cfg.ssim_weight * (1.0 - eval_ssim)
                ).cpu()
            )
        checkpoint = _checkpoint(
            step=step,
            phase=point.phase,
            point_lambda=point.lambda_value,
            endpoint_eligible=point.endpoint_eligible,
            rendered=evaluated,
            denominator=eval_denominator,
            target=target_tensor,
            field=candidate,
            reconstruction_loss=eval_loss,
            ssim_value=eval_ssim,
            optimizer_reset_count=reset_count,
            optimizer_reset_step=reset_step,
            calls=calls,
            started=started,
        )
        checkpoints.append(checkpoint)
        if verbose:
            print(
                f"  unit-gauge {step:4d}/{cfg.steps} phase={point.phase:8s} "
                f"lambda={point.lambda_value:.6f} psnr={checkpoint.raw_psnr_db:6.2f} "
                f"reset={reset_count}"
            )
        if checkpoint.endpoint_eligible and checkpoint.finite:
            key = (checkpoint.raw_mse, step)
            if best_key is None or key < best_key:
                best_key = key
                best_index = len(checkpoints) - 1
                best_state = _clone_state(candidate)
                best_image = evaluated.image.detach().clone()

    if best_state is None:
        terminal = _endpoint_field(candidate)
        selected_step = None
        if status == "completed":
            status = "no_exact_endpoint_checkpoint"
    else:
        _restore_state(candidate, best_state)
        terminal = _endpoint_field(candidate)
        selected_step = checkpoints[best_index].step
        checkpoints[best_index] = replace(checkpoints[best_index], selected=True)

    with torch.no_grad():
        terminal_render = _render(
            terminal,
            terminal.colors,
            height,
            width,
            cfg,
            cfg.additive_renderer,
        )
        calls["additive_numerator"] += 1
    parity = (
        0.0
        if best_image is None
        else float(torch.max(torch.abs(terminal_render - best_image)).cpu())
    )
    reconstruction = terminal_render.detach().cpu().numpy().astype(np.float32, copy=False)
    return UnitGaugeContinuationResult(
        field=terminal,
        reconstruction_raw=reconstruction,
        checkpoints=tuple(checkpoints),
        selected_step=selected_step,
        attempted_steps=cfg.steps,
        completed_steps=completed_steps,
        status=status,
        optimizer_reset_count=reset_count,
        optimizer_reset_step=reset_step,
        normalized_calls=calls["normalized"],
        additive_numerator_calls=calls["additive_numerator"],
        additive_denominator_calls=calls["additive_denominator"],
        renderer_calls=sum(calls.values()),
        endpoint_parity_max_abs=parity,
        elapsed_seconds=time.perf_counter() - started,
    )
