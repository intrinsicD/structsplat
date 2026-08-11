"""Training-only normalized-to-additive continuation for strict Gaussian sums (HIER-022).

The maintained renderer dispatch is intentionally unchanged.  This default-off research module
composes two existing additive renders during optimization and returns a plain additive
``GaussianField``.  Torch imports stay local so NumPy-only package imports remain torch-free.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import TYPE_CHECKING, Literal

import numpy as np


if TYPE_CHECKING:
    from .gaussians import GaussianField


ContinuationPhase = Literal["hold", "anneal", "endpoint"]
PixelLoss = Literal["l1", "l2"]

_HOLD_FRACTION = 0.35
_ANNEAL_FRACTION = 0.50
_ADDITIVE_RENDERERS = frozenset(("additive", "cuda_additive", "cuda_tiled_additive"))


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite(value: object, name: str, *, minimum: float | None = None,
            maximum: float | None = None, strict_minimum: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            relation = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {relation} {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {result}")
    return result


def _image(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f"{name} must have non-empty HWC RGB shape")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class AdditiveContinuationConfig:
    """Frozen HIER-022 optimizer and renderer controls.

    The 35/50/15 continuation schedule is deliberately not configurable.  A caller can disable
    only the coverage objective (the registered causal control), not silently change the path.
    """

    steps: int = 500
    checkpoint_every: int = 25
    lr_means: float = 5e-2
    lr_scales: float = 3e-2
    lr_rotations: float = 1e-2
    lr_coefficients: float = 3e-2
    lr_masses: float = 1e-2
    pixel_loss: PixelLoss = "l1"
    ssim_weight: float = 0.3
    coverage_weight: float = 0.05
    normalization_eps: float = 1e-8
    coefficient_abs_limit: float = 16.0
    log_mass_abs_limit: float = 8.0
    min_scale_px: float = 0.35
    sigma_cutoff: float = 3.0
    aa_dilation: float = 0.0
    render_chunk: int = 256
    renderer: str = "additive"
    support_fade: bool = False
    render_checkpoint: bool = False

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
            "lr_masses",
            "normalization_eps",
            "coefficient_abs_limit",
            "log_mass_abs_limit",
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
            "coverage_weight",
            _finite(self.coverage_weight, "coverage_weight", minimum=0.0),
        )
        object.__setattr__(
            self,
            "aa_dilation",
            _finite(self.aa_dilation, "aa_dilation", minimum=0.0),
        )
        object.__setattr__(
            self,
            "ssim_weight",
            _finite(self.ssim_weight, "ssim_weight", minimum=0.0, maximum=1.0),
        )
        if self.pixel_loss not in ("l1", "l2"):
            raise ValueError("pixel_loss must be 'l1' or 'l2'")
        if self.renderer not in _ADDITIVE_RENDERERS:
            expected = ", ".join(sorted(_ADDITIVE_RENDERERS))
            raise ValueError(f"renderer must use additive semantics; expected one of {expected}")
        if not isinstance(self.support_fade, bool):
            raise TypeError("support_fade must be bool")
        if not isinstance(self.render_checkpoint, bool):
            raise TypeError("render_checkpoint must be bool")
        continuation_phase_lengths(self.steps)


@dataclass(frozen=True)
class ContinuationSchedulePoint:
    step: int
    phase: ContinuationPhase
    lambda_value: float
    endpoint_eligible: bool


def continuation_phase_lengths(steps: int) -> tuple[int, int, int]:
    """Return integer hold, anneal, and exact-additive tail lengths."""

    total = _integer(steps, "steps", minimum=4)
    hold = int(round(_HOLD_FRACTION * total))
    anneal = int(round(_ANNEAL_FRACTION * total))
    endpoint = total - hold - anneal
    if min(hold, anneal, endpoint) < 1:
        raise ValueError(
            "steps must allocate at least one optimization step to hold, anneal, and endpoint"
        )
    return hold, anneal, endpoint


def continuation_schedule(step: int, steps: int) -> ContinuationSchedulePoint:
    """Return the frozen 35%-hold, 50%-cosine, 15%-exact-additive schedule point.

    ``step`` is one-based.  The last cosine step reaches zero, but only the following dedicated
    endpoint tail is checkpoint-eligible.  This keeps the registered final 15% unambiguous.
    """

    current = _integer(step, "step", minimum=1)
    hold, anneal, _ = continuation_phase_lengths(steps)
    if current > steps:
        raise ValueError(f"step must be <= steps ({steps}), got {current}")
    if current <= hold:
        return ContinuationSchedulePoint(current, "hold", 1.0, False)
    if current <= hold + anneal:
        progress = (current - hold) / float(anneal)
        value = 0.5 * (1.0 + math.cos(math.pi * progress))
        return ContinuationSchedulePoint(current, "anneal", value, False)
    return ContinuationSchedulePoint(current, "endpoint", 0.0, True)


@dataclass(frozen=True)
class AdditiveContinuationRender:
    """Differentiable render components; mass itself is intentionally not retained."""

    image: object
    numerator: object
    denominator: object


@dataclass(frozen=True)
class AdditiveContinuationCheckpoint:
    step: int
    phase: str
    lambda_value: float
    endpoint_eligible: bool
    selected: bool
    finite: bool
    objective: float | None
    reconstruction_loss: float
    coverage_loss: float
    raw_l1: float
    raw_mse: float
    raw_psnr_db: float
    ssim: float
    denominator_min: float
    denominator_q01: float
    denominator_q05: float
    denominator_q50: float
    denominator_q95: float
    denominator_q99: float
    denominator_max: float
    coefficient_abs_max: float
    mass_min: float
    mass_max: float
    elapsed_seconds: float
    renderer_calls: int

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AdditiveContinuationResult:
    """A strict additive endpoint plus scalar-only training telemetry."""

    field: "GaussianField"
    reconstruction_raw: np.ndarray
    checkpoints: tuple[AdditiveContinuationCheckpoint, ...]
    selected_step: int | None
    attempted_steps: int
    completed_steps: int
    status: str
    initial_mass: float
    selected_coverage_loss: float | None
    renderer_calls: int
    endpoint_parity_max_abs: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        selected_count = sum(checkpoint.selected for checkpoint in self.checkpoints)
        expected = 0 if self.selected_step is None else 1
        if selected_count != expected:
            raise ValueError("checkpoint selection and selected_step disagree")
        if self.field.opacities is not None:
            raise ValueError("an additive-continuation endpoint cannot retain opacity/mass")
        if self.field.color_grads is not None:
            raise ValueError("HIER-022 endpoints require constant RGB coefficients")

    @property
    def completed(self) -> bool:
        return self.status == "completed" and self.completed_steps == self.attempted_steps

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


def _validate_field(field: object) -> "GaussianField":
    import torch

    from .gaussians import GaussianField

    if not isinstance(field, GaussianField):
        raise TypeError("field must be GaussianField")
    if field.n < 1:
        raise ValueError("field must contain at least one Gaussian")
    if field.opacities is not None:
        raise ValueError("continuation requires opacity-free input geometry")
    if field.color_grads is not None:
        raise ValueError("continuation currently requires constant RGB coefficients")
    expected = {
        "means": (field.n, 2),
        "log_scales": (field.n, 2),
        "rotations": (field.n,),
        "colors": (field.n, 3),
    }
    for name, shape in expected.items():
        value = getattr(field, name)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"field.{name} must have shape {shape}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"field.{name} must be finite")
    return field


def _validate_log_masses(field: "GaussianField", log_masses: object,
                         config: AdditiveContinuationConfig):
    import torch

    if not torch.is_tensor(log_masses):
        raise TypeError("log_masses must be a torch Tensor")
    if tuple(log_masses.shape) != (field.n,):
        raise ValueError(f"log_masses must have shape ({field.n},)")
    if log_masses.device != field.means.device or log_masses.dtype != field.means.dtype:
        raise ValueError("log_masses must match the field device and dtype")
    if not bool(torch.isfinite(log_masses).all()):
        raise ValueError("log_masses must be finite")
    limit = config.log_mass_abs_limit
    if bool((torch.abs(log_masses.detach()) > limit + 1e-7).any()):
        raise ValueError(f"log_masses must stay within [-{limit}, {limit}]")
    return log_masses


def _render_additive(field: "GaussianField", colors, height: int, width: int,
                     config: AdditiveContinuationConfig, *, opacities=None):
    from .render import render_field

    return render_field(
        field.means,
        field.conics(config.aa_dilation),
        colors,
        field.radii(config.sigma_cutoff, config.aa_dilation),
        height,
        width,
        chunk=config.render_chunk,
        mode=config.renderer,
        opacities=opacities,
        scales=field.effective_scales(),
        rotations=field.rotations,
        support_fade=config.support_fade,
        sigma_cutoff=config.sigma_cutoff,
        checkpoint_chunks=config.render_checkpoint,
    )


def render_additive_continuation(
    field: "GaussianField",
    log_masses,
    height: int,
    width: int,
    lambda_value: float,
    *,
    config: AdditiveContinuationConfig | None = None,
) -> AdditiveContinuationRender:
    """Render ``A / (lambda * (D + eps) + 1 - lambda)`` using additive kernels.

    At ``lambda=0`` the returned image is the numerator object itself, not an algebraic multiply
    by one.  Consequently the endpoint is exactly the maintained direct-additive equation and
    has no computational dependency on training mass.
    """

    import torch

    candidate = _validate_field(field)
    cfg = config or AdditiveContinuationConfig()
    h = _integer(height, "height", minimum=1)
    w = _integer(width, "width", minimum=1)
    lam = _finite(lambda_value, "lambda_value", minimum=0.0, maximum=1.0)
    log_mass = _validate_log_masses(candidate, log_masses, cfg)
    masses = torch.exp(log_mass)
    numerator = _render_additive(candidate, candidate.colors, h, w, cfg)
    denominator_rgb = _render_additive(
        candidate,
        torch.ones_like(candidate.colors),
        h,
        w,
        cfg,
        opacities=masses,
    )
    denominator = denominator_rgb[..., :1]
    image = numerator if lam == 0.0 else numerator / (
        lam * (denominator + cfg.normalization_eps) + (1.0 - lam)
    )
    return AdditiveContinuationRender(image, numerator, denominator)


def _project_parameters(field: "GaussianField", log_masses, height: int, width: int,
                        config: AdditiveContinuationConfig) -> None:
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
        log_masses.clamp_(-config.log_mass_abs_limit, config.log_mass_abs_limit)


def _parameter_state_is_finite(field: "GaussianField", log_masses) -> bool:
    import torch

    values = (field.means, field.log_scales, field.rotations, field.colors, log_masses)
    return all(bool(torch.isfinite(value).all()) for value in values)


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


def _tensor_quantiles(values) -> tuple[float, ...]:
    import torch

    quantiles = torch.as_tensor(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
        device=values.device,
        dtype=values.dtype,
    )
    return tuple(float(value) for value in torch.quantile(values.reshape(-1), quantiles).cpu())


def _checkpoint(
    *,
    step: int,
    phase: str,
    lambda_value: float,
    endpoint_eligible: bool,
    rendered: AdditiveContinuationRender,
    target,
    field: "GaussianField",
    log_masses,
    objective: float | None,
    reconstruction_loss: float,
    ssim_value: float,
    started: float,
    renderer_calls: int,
) -> AdditiveContinuationCheckpoint:
    import torch

    with torch.no_grad():
        residual = rendered.image - target
        raw_l1 = float(residual.abs().mean().cpu())
        raw_mse = float(residual.square().mean().cpu())
        psnr = -10.0 * math.log10(max(raw_mse, 1e-12))
        coverage = float((rendered.denominator - 1.0).square().mean().cpu())
        den_stats = _tensor_quantiles(rendered.denominator)
        masses = torch.exp(log_masses)
        coefficient_abs_max = float(field.colors.abs().max().cpu())
        mass_min = float(masses.min().cpu())
        mass_max = float(masses.max().cpu())
        finite = bool(
            torch.isfinite(rendered.image).all()
            and torch.isfinite(rendered.denominator).all()
            and _parameter_state_is_finite(field, log_masses)
        )
    return AdditiveContinuationCheckpoint(
        step=step,
        phase=phase,
        lambda_value=float(lambda_value),
        endpoint_eligible=endpoint_eligible,
        selected=False,
        finite=finite,
        objective=objective,
        reconstruction_loss=float(reconstruction_loss),
        coverage_loss=coverage,
        raw_l1=raw_l1,
        raw_mse=raw_mse,
        raw_psnr_db=psnr,
        ssim=float(ssim_value),
        denominator_min=den_stats[0],
        denominator_q01=den_stats[1],
        denominator_q05=den_stats[2],
        denominator_q50=den_stats[3],
        denominator_q95=den_stats[4],
        denominator_q99=den_stats[5],
        denominator_max=den_stats[6],
        coefficient_abs_max=coefficient_abs_max,
        mass_min=mass_min,
        mass_max=mass_max,
        elapsed_seconds=time.perf_counter() - started,
        renderer_calls=renderer_calls,
    )


def fit_additive_continuation(
    field: "GaussianField",
    target: np.ndarray,
    *,
    config: AdditiveContinuationConfig | None = None,
    verbose: bool = False,
) -> AdditiveContinuationResult:
    """Fit through normalization and return only a strict additive Gaussian field.

    The input field is never mutated.  Only exact-additive checkpoints from the registered final
    tail compete by raw MSE.  If optimization fails before that tail, the result is explicitly
    unsuccessful and returns an exact-additive initialization rather than pretending it passed.
    """

    import torch

    from . import metrics as M

    started = time.perf_counter()
    source_field = _validate_field(field)
    source = _image(target, "target")
    height, width = source.shape[:2]
    cfg = config or AdditiveContinuationConfig()
    if source_field.means.device.type == "cpu" and cfg.renderer != "additive":
        raise ValueError(f"renderer={cfg.renderer!r} requires a CUDA field")

    candidate = source_field.detached()
    target_tensor = torch.as_tensor(
        source,
        device=candidate.means.device,
        dtype=candidate.means.dtype,
    )
    ones = torch.ones_like(candidate.colors)
    with torch.no_grad():
        unit_denominator = _render_additive(
            candidate, ones, height, width, cfg, opacities=torch.ones(candidate.n,
                                                                      device=ones.device,
                                                                      dtype=ones.dtype)
        )[..., :1]
        mean_coverage = float(unit_denominator.mean().cpu())
        if not math.isfinite(mean_coverage) or mean_coverage <= 0.0:
            raise ValueError("input geometry has no finite visible support")
        initial_log_mass = max(
            -cfg.log_mass_abs_limit,
            min(cfg.log_mass_abs_limit, math.log(1.0 / mean_coverage)),
        )
        initial_mass = math.exp(initial_log_mass)
        candidate.colors.mul_(initial_mass)
        candidate.colors.clamp_(-cfg.coefficient_abs_limit, cfg.coefficient_abs_limit)
    log_masses = torch.full(
        (candidate.n,),
        initial_log_mass,
        device=candidate.means.device,
        dtype=candidate.means.dtype,
        requires_grad=True,
    )
    candidate.trainable()
    optimizer = torch.optim.Adam(
        [
            {"params": [candidate.means], "lr": cfg.lr_means},
            {"params": [candidate.log_scales], "lr": cfg.lr_scales},
            {"params": [candidate.rotations], "lr": cfg.lr_rotations},
            {"params": [candidate.colors], "lr": cfg.lr_coefficients},
            {"params": [log_masses], "lr": cfg.lr_masses},
        ]
    )
    ssim_stats = M.SSIMTargetStats(target_tensor) if cfg.ssim_weight > 0.0 else None
    renderer_calls = 1
    checkpoints: list[AdditiveContinuationCheckpoint] = []

    with torch.no_grad():
        initial_render = render_additive_continuation(
            candidate, log_masses, height, width, 1.0, config=cfg
        )
        renderer_calls += 2
        initial_ssim = float(
            M.ssim(initial_render.image, target_tensor, target_stats=ssim_stats)
        )
        initial_pixel = (
            initial_render.image - target_tensor
        ).abs().mean() if cfg.pixel_loss == "l1" else (
            initial_render.image - target_tensor
        ).square().mean()
        initial_reconstruction_loss = float(
            ((1.0 - cfg.ssim_weight) * initial_pixel
             + cfg.ssim_weight * (1.0 - initial_ssim)).cpu()
        )
    checkpoints.append(
        _checkpoint(
            step=0,
            phase="initial",
            lambda_value=1.0,
            endpoint_eligible=False,
            rendered=initial_render,
            target=target_tensor,
            field=candidate,
            log_masses=log_masses,
            objective=None,
            reconstruction_loss=initial_reconstruction_loss,
            ssim_value=initial_ssim,
            started=started,
            renderer_calls=renderer_calls,
        )
    )

    hold_steps, anneal_steps, _ = continuation_phase_lengths(cfg.steps)
    endpoint_start = hold_steps + anneal_steps + 1
    best_state: tuple[object, ...] | None = None
    best_image = None
    best_index: int | None = None
    best_key: tuple[float, int] | None = None
    attempted_steps = cfg.steps
    completed_steps = 0
    status = "completed"
    last_finite_state = (
        candidate.means.detach().clone(),
        candidate.log_scales.detach().clone(),
        candidate.rotations.detach().clone(),
        candidate.colors.detach().clone(),
        log_masses.detach().clone(),
    )

    for step in range(1, cfg.steps + 1):
        point = continuation_schedule(step, cfg.steps)
        optimizer.zero_grad(set_to_none=True)
        rendered = render_additive_continuation(
            candidate,
            log_masses,
            height,
            width,
            point.lambda_value,
            config=cfg,
        )
        renderer_calls += 2
        residual = rendered.image - target_tensor
        pixel = residual.abs().mean() if cfg.pixel_loss == "l1" else residual.square().mean()
        if cfg.ssim_weight > 0.0:
            ssim_value_t = M.ssim(
                rendered.image,
                target_tensor,
                target_stats=ssim_stats,
            )
        else:
            ssim_value_t = pixel.new_tensor(1.0)
        reconstruction_loss_t = (
            (1.0 - cfg.ssim_weight) * pixel
            + cfg.ssim_weight * (1.0 - ssim_value_t)
        )
        coverage_loss_t = (rendered.denominator - 1.0).square().mean()
        objective_t = reconstruction_loss_t + cfg.coverage_weight * coverage_loss_t
        if not bool(torch.isfinite(objective_t)):
            status = "nonfinite_objective"
            break
        objective_t.backward()
        parameters = (
            candidate.means,
            candidate.log_scales,
            candidate.rotations,
            candidate.colors,
            log_masses,
        )
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            status = "nonfinite_gradient"
            break
        optimizer.step()
        _project_parameters(candidate, log_masses, height, width, cfg)
        if not _parameter_state_is_finite(candidate, log_masses):
            with torch.no_grad():
                for parameter, safe_value in zip(
                    (
                        candidate.means,
                        candidate.log_scales,
                        candidate.rotations,
                        candidate.colors,
                        log_masses,
                    ),
                    last_finite_state,
                    strict=True,
                ):
                    parameter.copy_(safe_value)
            status = "nonfinite_parameters"
            break
        completed_steps = step
        last_finite_state = (
            candidate.means.detach().clone(),
            candidate.log_scales.detach().clone(),
            candidate.rotations.detach().clone(),
            candidate.colors.detach().clone(),
            log_masses.detach().clone(),
        )

        should_checkpoint = (
            step == endpoint_start
            or step == cfg.steps
            or step % cfg.checkpoint_every == 0
        )
        if not should_checkpoint:
            continue
        with torch.no_grad():
            evaluated = render_additive_continuation(
                candidate,
                log_masses,
                height,
                width,
                point.lambda_value,
                config=cfg,
            )
            renderer_calls += 2
            eval_residual = evaluated.image - target_tensor
            eval_pixel = (
                eval_residual.abs().mean()
                if cfg.pixel_loss == "l1"
                else eval_residual.square().mean()
            )
            eval_ssim = float(
                M.ssim(evaluated.image, target_tensor, target_stats=ssim_stats)
            )
            eval_reconstruction = float(
                ((1.0 - cfg.ssim_weight) * eval_pixel
                 + cfg.ssim_weight * (1.0 - eval_ssim)).cpu()
            )
            eval_coverage = (evaluated.denominator - 1.0).square().mean()
            eval_objective = float(
                (eval_reconstruction + cfg.coverage_weight * eval_coverage).cpu()
            )
        checkpoint = _checkpoint(
            step=step,
            phase=point.phase,
            lambda_value=point.lambda_value,
            endpoint_eligible=point.endpoint_eligible,
            rendered=evaluated,
            target=target_tensor,
            field=candidate,
            log_masses=log_masses,
            objective=eval_objective,
            reconstruction_loss=eval_reconstruction,
            ssim_value=eval_ssim,
            started=started,
            renderer_calls=renderer_calls,
        )
        checkpoints.append(checkpoint)
        if verbose:
            print(
                f"  continuation {step:4d}/{cfg.steps} phase={point.phase:8s} "
                f"lambda={point.lambda_value:.5f} psnr={checkpoint.raw_psnr_db:6.2f} "
                f"coverage={checkpoint.coverage_loss:.4g} "
                f"|p|max={checkpoint.coefficient_abs_max:.3f}"
            )
        if checkpoint.endpoint_eligible and checkpoint.finite:
            key = (checkpoint.raw_mse, step)
            if best_key is None or key < best_key:
                best_key = key
                best_index = len(checkpoints) - 1
                best_state = (
                    candidate.means.detach().clone(),
                    candidate.log_scales.detach().clone(),
                    candidate.rotations.detach().clone(),
                    candidate.colors.detach().clone(),
                    None if candidate.filter_variance is None
                    else candidate.filter_variance.detach().clone(),
                )
                best_image = evaluated.image.detach().clone()

    if best_state is None:
        terminal = _endpoint_field(candidate)
        selected_step = None
        selected_coverage = None
        if status == "completed":
            status = "no_exact_endpoint_checkpoint"
    else:
        from .gaussians import GaussianField

        means, log_scales, rotations, colors, filter_variance = best_state
        terminal = GaussianField(
            means,
            log_scales,
            rotations,
            colors,
            opacities=None,
            scale_max=None,
            color_grads=None,
            background_mask=None,
            filter_variance=filter_variance,
        )
        selected_step = checkpoints[best_index].step
        selected_coverage = checkpoints[best_index].coverage_loss
        checkpoints[best_index] = replace(checkpoints[best_index], selected=True)

    with torch.no_grad():
        terminal_render = _render_additive(terminal, terminal.colors, height, width, cfg)
        renderer_calls += 1
    if best_image is None:
        parity = 0.0
    else:
        parity = float(torch.max(torch.abs(terminal_render - best_image)).cpu())
    reconstruction = terminal_render.detach().cpu().numpy().astype(np.float32, copy=False)
    return AdditiveContinuationResult(
        field=terminal,
        reconstruction_raw=reconstruction,
        checkpoints=tuple(checkpoints),
        selected_step=selected_step,
        attempted_steps=attempted_steps,
        completed_steps=completed_steps,
        status=status,
        initial_mass=initial_mass,
        selected_coverage_loss=selected_coverage,
        renderer_calls=renderer_calls,
        endpoint_parity_max_abs=parity,
        elapsed_seconds=time.perf_counter() - started,
    )
