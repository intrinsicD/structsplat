"""Count-neutral residual anchoring and RGB projection for HIER-010.

This module is a deterministic, default-off research reference.  It keeps the direct-additive
Observation Field V2 geometry fixed and never materializes the dense pixel-by-Gaussian matrix.
Torch is imported lazily so the package's NumPy-only analysis boundary remains intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
import math
import time
from typing import Literal

import numpy as np

from .observation_field import ObservationField2D
from .progressive_residual_quadtree import progressive_artifact_metrics


ProjectionCenter = Literal["input", "zero"]
ProjectionStart = Literal["input", "zero"]
FrozenBaseMode = Literal["subtract", "explicit"]
ProjectionSelectionMode = Literal["transaction", "bounded_intermediate"]


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite(value: object, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    invalid = result <= minimum if strict else result < minimum
    if not math.isfinite(result) or invalid:
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}, got {result}")
    return result


def _image(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f"{name} must have non-empty HWC RGB shape")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _mask(value: object, shape: tuple[int, int], name: str = "mask") -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != shape:
        raise ValueError(f"{name} must be a bool NumPy array with shape {shape}")
    result = np.array(value, dtype=bool, order="C", copy=True)
    if not result.any():
        raise ValueError(f"{name} must contain at least one active pixel")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, order="C", copy=True)
    result.flags.writeable = False
    return result


def _centered_box_mean(values: np.ndarray, mask: np.ndarray, side: int) -> np.ndarray:
    """Return a zero-padded, mask-normalized centered box mean."""

    radius = side // 2

    def box_sum(array: np.ndarray) -> np.ndarray:
        padded = np.pad(array, ((radius, radius), (radius, radius)), mode="constant")
        integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (
            integral[side:, side:]
            - integral[:-side, side:]
            - integral[side:, :-side]
            + integral[:-side, :-side]
        )

    numerator = box_sum(np.where(mask, values, 0.0).astype(np.float64))
    denominator = box_sum(mask.astype(np.float64))
    return numerator / np.maximum(denominator, 1.0)


@dataclass(frozen=True)
class ResidualAnchorSelection:
    """Exact residual-ranked source leaves reserved for a second contraction pass."""

    protected_mask: np.ndarray
    score: np.ndarray
    pixel_mse: np.ndarray
    patch_mse: np.ndarray
    requested_count: int
    selected_count: int
    nms_selected_count: int
    pixel_reference_q99: float
    patch_reference_q99: float
    patch_side: int
    nms_radius_px: int

    def __post_init__(self) -> None:
        shape = self.protected_mask.shape
        if self.protected_mask.dtype != np.bool_ or self.protected_mask.ndim != 2:
            raise ValueError("protected_mask must be a two-dimensional bool array")
        for name in ("score", "pixel_mse", "patch_mse"):
            value = getattr(self, name)
            if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite floating array matching the mask")
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "protected_mask", _readonly(self.protected_mask))
        if self.selected_count != int(self.protected_mask.sum()):
            raise ValueError("selected_count disagrees with protected_mask")
        if self.selected_count != self.requested_count:
            raise ValueError("residual anchor selection did not reach the requested count")


def select_residual_anchor_leaves(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    count: int,
    *,
    patch_side: int = 7,
    nms_radius_px: int = 1,
) -> ResidualAnchorSelection:
    """Select exact source leaves from isolated and patch-scale residual evidence.

    Pixel and mask-aware patch MSE are independently normalized by their active-pixel q99,
    combined by a pointwise maximum, and ranked with stable row-major ties.  Chebyshev NMS spreads
    the first tranche; a stable no-NMS fill guarantees the exact requested count.
    """

    candidate = _image(reconstruction, "reconstruction")
    source = _image(target, "target")
    if candidate.shape != source.shape:
        raise ValueError("reconstruction and target must have matching shapes")
    active = _mask(mask, source.shape[:2])
    requested = _integer(count, "count")
    if requested > int(active.sum()):
        raise ValueError("count cannot exceed the active mask population")
    side = _integer(patch_side, "patch_side", minimum=1)
    if side % 2 == 0:
        raise ValueError("patch_side must be odd")
    radius = _integer(nms_radius_px, "nms_radius_px")

    residual = candidate.astype(np.float64) - source.astype(np.float64)
    pixel_mse = np.mean(residual * residual, axis=2)
    patch_mse = _centered_box_mean(pixel_mse, active, side)
    epsilon = np.finfo(np.float64).eps
    pixel_reference = max(float(np.quantile(pixel_mse[active], 0.99)), epsilon)
    patch_reference = max(float(np.quantile(patch_mse[active], 0.99)), epsilon)
    score = np.maximum(pixel_mse / pixel_reference, patch_mse / patch_reference)
    score[~active] = 0.0

    order = np.argsort(-score.reshape(-1), kind="stable")
    height, width = active.shape
    protected = np.zeros(active.shape, dtype=bool)
    blocked = np.zeros(active.shape, dtype=bool)
    selected = 0
    for flat_index in order:
        if selected >= requested:
            break
        y, x = divmod(int(flat_index), width)
        if not active[y, x] or blocked[y, x]:
            continue
        protected[y, x] = True
        selected += 1
        blocked[
            max(0, y - radius) : min(height, y + radius + 1),
            max(0, x - radius) : min(width, x + radius + 1),
        ] = True
    nms_selected = selected
    if selected < requested:
        for flat_index in order:
            if selected >= requested:
                break
            y, x = divmod(int(flat_index), width)
            if active[y, x] and not protected[y, x]:
                protected[y, x] = True
                selected += 1
    if selected != requested:
        raise RuntimeError("residual anchor fill did not reach the requested count")

    return ResidualAnchorSelection(
        protected_mask=protected,
        score=score.astype(np.float32),
        pixel_mse=pixel_mse.astype(np.float32),
        patch_mse=patch_mse.astype(np.float32),
        requested_count=requested,
        selected_count=selected,
        nms_selected_count=nms_selected,
        pixel_reference_q99=pixel_reference,
        patch_reference_q99=patch_reference,
        patch_side=side,
        nms_radius_px=radius,
    )


@dataclass(frozen=True)
class CoefficientProjectionConfig:
    """Fail-closed PCG settings for fixed-geometry direct-additive RGB projection."""

    tolerance: float = 1e-6
    max_iterations: int = 48
    ridge: float = 1e-8
    coefficient_abs_limit: float = 16.0
    pixel_rmse_threshold: float = 0.02
    patch7_rmse_threshold: float = 0.01
    sse_relative_tolerance: float = 1e-8
    violation_absolute_tolerance: float = 1e-9
    regularization_center: ProjectionCenter = "input"
    solver_start: ProjectionStart = "input"
    frozen_base_mode: FrozenBaseMode = "subtract"
    allow_unsafe_stage_zero_reconditioning: bool = False
    selection_mode: ProjectionSelectionMode = "transaction"

    def __post_init__(self) -> None:
        for name in (
            "tolerance",
            "coefficient_abs_limit",
            "pixel_rmse_threshold",
            "patch7_rmse_threshold",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name, strict=True))
        for name in ("ridge", "sse_relative_tolerance", "violation_absolute_tolerance"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_iterations",
            _integer(self.max_iterations, "max_iterations", minimum=1),
        )
        if self.regularization_center not in ("input", "zero"):
            raise ValueError("regularization_center must be 'input' or 'zero'")
        if self.solver_start not in ("input", "zero"):
            raise ValueError("solver_start must be 'input' or 'zero'")
        if self.frozen_base_mode not in ("subtract", "explicit"):
            raise ValueError("frozen_base_mode must be 'subtract' or 'explicit'")
        if not isinstance(self.allow_unsafe_stage_zero_reconditioning, bool):
            raise TypeError("allow_unsafe_stage_zero_reconditioning must be bool")
        if self.selection_mode not in ("transaction", "bounded_intermediate"):
            raise ValueError("selection_mode must be 'transaction' or 'bounded_intermediate'")


@dataclass(frozen=True)
class CoefficientProjectionCheckpoint:
    iteration: int
    selected: bool
    selectable: bool
    bounded: bool
    transaction_safe: bool
    raw_sse: float
    raw_pixel_rmse_max: float
    raw_patch7_rmse_max: float
    display_pixel_rmse_max: float
    display_patch7_rmse_max: float
    display_normalized_violation: float
    display_gate_pass: bool
    relative_normal_residual_max: float
    coefficient_abs_max: float
    elapsed_seconds: float
    forward_applications: int
    transpose_applications: int

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CoefficientProjectionResult:
    field: ObservationField2D
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    checkpoints: tuple[CoefficientProjectionCheckpoint, ...]
    selected_iteration: int
    initial_sse: float
    final_sse: float
    trainable_rows: int
    frozen_rows: int
    protected_rows: int
    forward_applications: int
    transpose_applications: int
    relative_normal_residual_max: float
    adjoint_relative_error: float
    initial_operator_parity_max_abs: float
    normal_diagonal_min: float
    normal_diagonal_max: float
    maintained_render_parity_max_abs: float
    elapsed_seconds: float
    selection_mode: ProjectionSelectionMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        object.__setattr__(self, "reconstruction", _readonly(self.reconstruction))
        if sum(checkpoint.selected for checkpoint in self.checkpoints) != 1:
            raise ValueError("exactly one projection checkpoint must be selected")

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


def project_contracted_coefficients(
    field: ObservationField2D,
    target: np.ndarray,
    mask: np.ndarray,
    touched_row_mask: np.ndarray,
    protected_row_mask: np.ndarray | None = None,
    *,
    config: CoefficientProjectionConfig | None = None,
    device: str = "cpu",
    renderer: str = "additive",
    render_chunk: int = 256,
) -> CoefficientProjectionResult:
    """Project topology-touched RGB rows while freezing geometry and exact leaves.

    The operator is evaluated from the maintained finite-support kernel in sparse tiles.  It does
    not allocate the dense ``pixels x rows`` basis matrix.  In the default ``transaction`` mode,
    every PCG iterate is checked against stage-zero raw SSE and displayed artifact violation and
    an unsafe solve returns step zero.  ``bounded_intermediate`` may return a finite bounded
    lower-SSE iterate for use *inside* a wider fail-closed optimization transaction; callers must
    not expose that state without their own final quality guard.
    """

    started = time.perf_counter()
    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be ObservationField2D")
    if field.semantics.renderer_equation != "additive_rgb_peak_one_v1":
        raise ValueError("projection requires the direct additive peak-one equation")
    if field.semantics.support.mode != "axis_aligned_bbox":
        raise ValueError("projection currently requires axis-aligned AABB support")
    if field.background_rgb is not None:
        raise ValueError("projection currently requires a zero-DC field")
    if renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
        raise ValueError("renderer must be additive, cuda_additive, or cuda_tiled_additive")
    chunk = _integer(render_chunk, "render_chunk", minimum=1)
    source = _image(target, "target")
    if source.shape[:2] != field.crop_shape:
        raise ValueError("target shape must match the field crop")
    active = _mask(mask, source.shape[:2])
    touched = _mask_vector(touched_row_mask, field.n, "touched_row_mask")
    protected = (
        np.zeros(field.n, dtype=bool)
        if protected_row_mask is None
        else _mask_vector(protected_row_mask, field.n, "protected_row_mask")
    )
    cfg = config or CoefficientProjectionConfig()
    trainable_numpy = touched & ~protected

    import torch

    from .pixel_contraction import observation_to_gaussian_field, render_observation_field
    from .render import (
        _element_budget,
        _flat_tile_slices,
        _support_weight,
        _tile_bounds,
        _tile_coords,
        render_field,
    )

    gaussian = observation_to_gaussian_field(field, device=device)
    means = gaussian.means.detach()
    dilation = field.semantics.filtering.aa_dilation_px2
    conics = gaussian.conics(dilation).detach()
    radii = gaussian.radii(field.semantics.support.sigma_cutoff, dilation)
    height, width = source.shape[:2]
    torch_device = means.device
    dtype = means.dtype
    mask_tensor = torch.as_tensor(active, device=torch_device, dtype=torch.bool)
    mask_flat = mask_tensor.reshape(-1)
    target_tensor = torch.as_tensor(source, device=torch_device, dtype=dtype)
    trainable_ids = torch.as_tensor(
        np.flatnonzero(trainable_numpy), device=torch_device, dtype=torch.long
    )
    local_of_global = torch.full((field.n,), -1, device=torch_device, dtype=torch.long)
    if trainable_ids.numel():
        local_of_global[trainable_ids] = torch.arange(trainable_ids.numel(), device=torch_device)
    x0 = gaussian.colors.detach()[trainable_ids].clone()

    x0_bounds = _tile_bounds(means, radii, height, width)
    x0_tile, y0_tile, tile_width, tile_elements = x0_bounds
    budget = _element_budget(chunk)
    calls = {"forward": 0, "transpose": 0}
    sigma_cutoff = field.semantics.support.sigma_cutoff
    fade_alpha = field.semantics.support.fade_alpha

    def maintained_apply(colors):
        return render_field(
            means,
            conics,
            colors,
            radii,
            height,
            width,
            chunk=chunk,
            mode=renderer,
            opacities=None,
            scales=gaussian.effective_scales(dilation).detach(),
            rotations=gaussian.rotations.detach(),
            support_fade=fade_alpha > 0.0,
            sigma_cutoff=sigma_cutoff,
            support_fade_alpha=fade_alpha,
        )

    def tile_values(start: int, end: int):
        gid, px, py = _tile_coords(
            x0_tile,
            y0_tile,
            tile_width,
            tile_elements,
            start,
            end,
            torch_device,
        )
        local = local_of_global[gid]
        keep = (local >= 0) & mask_flat[py * width + px]
        if not bool(torch.any(keep)):
            return local[:0], px[:0], py[:0], means.new_zeros((0,))
        gid = gid[keep]
        local = local[keep]
        px = px[keep]
        py = py[keep]
        dx = px.to(dtype) - means[gid, 0]
        dy = py.to(dtype) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        quadratic = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        weights = _support_weight(
            quadratic,
            sigma_cutoff,
            fade_alpha > 0.0,
            fade_alpha,
        )
        return local, px, py, weights

    def basis_apply(values):
        calls["forward"] += 1
        out = torch.zeros(height * width, 3, device=torch_device, dtype=dtype)
        for start, end in _flat_tile_slices(tile_elements, budget):
            local, px, py, weights = tile_values(start, end)
            if local.numel():
                out.index_add_(
                    0,
                    py * width + px,
                    weights[:, None] * values[local],
                )
        return out.view(height, width, 3)

    def basis_transpose(image):
        calls["transpose"] += 1
        out = torch.zeros(trainable_ids.numel(), 3, device=torch_device, dtype=dtype)
        flat_image = image.reshape(-1, 3)
        for start, end in _flat_tile_slices(tile_elements, budget):
            local, px, py, weights = tile_values(start, end)
            if local.numel():
                flat = py * width + px
                out.index_add_(0, local, weights[:, None] * flat_image[flat])
        return out

    def diagonal_values():
        diagonal = torch.zeros(trainable_ids.numel(), device=torch_device, dtype=dtype)
        for start, end in _flat_tile_slices(tile_elements, budget):
            local, _px, _py, weights = tile_values(start, end)
            if local.numel():
                diagonal.index_add_(0, local, weights.square())
        return diagonal

    with torch.no_grad():
        maintained_raw = maintained_apply(gaussian.colors.detach())
        initial_variable = (
            basis_apply(x0) if trainable_ids.numel() else torch.zeros_like(maintained_raw)
        )
        if cfg.frozen_base_mode == "explicit":
            if trainable_ids.numel() == field.n:
                frozen_base = torch.zeros_like(maintained_raw)
            else:
                frozen_colors = gaussian.colors.detach().clone()
                frozen_colors[trainable_ids] = 0.0
                frozen_base = maintained_apply(frozen_colors)
        else:
            frozen_base = maintained_raw - initial_variable
        initial_operator_reconstruction = frozen_base + initial_variable
        initial_operator_parity_max_abs = float(
            torch.max(torch.abs(initial_operator_reconstruction - maintained_raw)).cpu()
        )
        objective_target = torch.where(
            mask_tensor[:, :, None], target_tensor - frozen_base, torch.zeros_like(target_tensor)
        )

        # A deterministic bilinear identity check makes a forward/transpose mismatch visible in
        # every real result without introducing a random seed.
        if trainable_ids.numel():
            probe_x = torch.sin(
                torch.arange(trainable_ids.numel() * 3, device=torch_device, dtype=dtype)
            ).view(-1, 3)
            probe_y = torch.cos(
                torch.arange(height * width * 3, device=torch_device, dtype=dtype)
            ).view(height, width, 3)
            probe_y = torch.where(mask_tensor[:, :, None], probe_y, torch.zeros_like(probe_y))
            left = torch.sum(basis_apply(probe_x) * probe_y)
            right = torch.sum(probe_x * basis_transpose(probe_y))
            adjoint_relative_error = float(
                (left - right)
                .abs()
                .div(torch.maximum(left.abs(), right.abs()).clamp_min(1.0))
                .cpu()
            )
        else:
            adjoint_relative_error = 0.0

        checkpoints: list[CoefficientProjectionCheckpoint] = []
        best_x = x0.clone()
        best_reconstruction = maintained_raw.clone()
        best_index = 0
        initial_raw_metrics = progressive_artifact_metrics(
            maintained_raw.cpu().numpy(),
            source,
            active,
            pixel_threshold=cfg.pixel_rmse_threshold,
            patch7_threshold=cfg.patch7_rmse_threshold,
            displayed=False,
        )
        initial_display_metrics = progressive_artifact_metrics(
            maintained_raw.cpu().numpy(),
            source,
            active,
            pixel_threshold=cfg.pixel_rmse_threshold,
            patch7_threshold=cfg.patch7_rmse_threshold,
            displayed=True,
        )
        initial_sse = float(initial_raw_metrics["sse"])
        sse_tolerance = cfg.sse_relative_tolerance * max(initial_sse, 1.0)

        def append_checkpoint(iteration: int, values, reconstruction, relative: float) -> None:
            nonlocal best_index, best_x, best_reconstruction
            reconstruction_numpy = reconstruction.cpu().numpy()
            raw = progressive_artifact_metrics(
                reconstruction_numpy,
                source,
                active,
                pixel_threshold=cfg.pixel_rmse_threshold,
                patch7_threshold=cfg.patch7_rmse_threshold,
                displayed=False,
            )
            display = progressive_artifact_metrics(
                reconstruction_numpy,
                source,
                active,
                pixel_threshold=cfg.pixel_rmse_threshold,
                patch7_threshold=cfg.patch7_rmse_threshold,
                displayed=True,
            )
            coefficient_abs_max = float(
                torch.max(torch.abs(values)).cpu() if values.numel() else 0.0
            )
            bounded = bool(
                math.isfinite(float(raw["sse"]))
                and coefficient_abs_max <= cfg.coefficient_abs_limit
            )
            transaction_safe = bool(
                bounded
                and float(raw["sse"]) <= initial_sse + sse_tolerance
                and float(display["normalized_violation"])
                <= float(initial_display_metrics["normalized_violation"])
                + cfg.violation_absolute_tolerance
            )
            selectable = bool(transaction_safe if cfg.selection_mode == "transaction" else bounded)
            checkpoints.append(
                CoefficientProjectionCheckpoint(
                    iteration=iteration,
                    selected=False,
                    selectable=selectable,
                    bounded=bounded,
                    transaction_safe=transaction_safe,
                    raw_sse=float(raw["sse"]),
                    raw_pixel_rmse_max=float(raw["pixel_rmse_max"]),
                    raw_patch7_rmse_max=float(raw["patch7_rmse_max"]),
                    display_pixel_rmse_max=float(display["pixel_rmse_max"]),
                    display_patch7_rmse_max=float(display["patch7_rmse_max"]),
                    display_normalized_violation=float(display["normalized_violation"]),
                    display_gate_pass=bool(display["gate_pass"]),
                    relative_normal_residual_max=relative,
                    coefficient_abs_max=coefficient_abs_max,
                    elapsed_seconds=time.perf_counter() - started,
                    forward_applications=calls["forward"],
                    transpose_applications=calls["transpose"],
                )
            )
            if not selectable:
                return
            best = checkpoints[best_index]
            key = (
                float(raw["sse"]),
                float(display["normalized_violation"]),
                iteration,
            )
            best_key = (best.raw_sse, best.display_normalized_violation, best.iteration)
            if key < best_key:
                best_index = len(checkpoints) - 1
                best_x = values.clone()
                best_reconstruction = reconstruction.clone()

        normal_diagonal_min = 0.0
        normal_diagonal_max = 0.0
        if trainable_ids.numel():
            ridge = cfg.ridge
            raw_diagonal = diagonal_values()
            normal_diagonal_min = float(torch.min(raw_diagonal).cpu())
            normal_diagonal_max = float(torch.max(raw_diagonal).cpu())
            diagonal = (raw_diagonal + ridge).clamp_min(torch.finfo(dtype).tiny)
            right_hand_side = basis_transpose(objective_target)
            if ridge > 0.0:
                regularization_reference = (
                    x0 if cfg.regularization_center == "input" else torch.zeros_like(x0)
                )
                right_hand_side = right_hand_side + ridge * regularization_reference

            def normal(values):
                result = basis_transpose(basis_apply(values))
                return result + ridge * values if ridge > 0.0 else result

            x = x0.clone() if cfg.solver_start == "input" else torch.zeros_like(x0)
            residual = right_hand_side - normal(x)
            z = residual / diagonal[:, None]
            direction = z.clone()
            residual_dot = torch.sum(residual * z, dim=0)
            right_norm = torch.sqrt(torch.sum(right_hand_side.square(), dim=0)).clamp_min(1e-12)
            relative = torch.sqrt(torch.sum(residual.square(), dim=0)) / right_norm
            append_checkpoint(0, x0, maintained_raw, float(torch.max(relative).cpu()))
            # Stage zero is the unconditional fail-closed return even when the incoming field is
            # already outside the optional coefficient bound.  In that case retain its explicit
            # non-selectable record, run no iterations, and select the unchanged field below.
            if checkpoints[0].selectable or cfg.allow_unsafe_stage_zero_reconditioning:
                epsilon = torch.finfo(dtype).eps
                for iteration in range(1, cfg.max_iterations + 1):
                    if bool(torch.all(relative <= cfg.tolerance)):
                        break
                    product = normal(direction)
                    denominator = torch.sum(direction * product, dim=0)
                    valid = (denominator > epsilon) & (relative > cfg.tolerance)
                    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
                    alpha = torch.where(
                        valid, residual_dot / safe_denominator, torch.zeros_like(residual_dot)
                    )
                    x = x + direction * alpha[None, :]
                    residual = residual - product * alpha[None, :]
                    updated_dot_input = residual / diagonal[:, None]
                    updated_dot = torch.sum(residual * updated_dot_input, dim=0)
                    relative = torch.sqrt(torch.sum(residual.square(), dim=0)) / right_norm
                    candidate = frozen_base + basis_apply(x)
                    append_checkpoint(iteration, x, candidate, float(torch.max(relative).cpu()))
                    safe_residual_dot = torch.where(
                        residual_dot.abs() > epsilon,
                        residual_dot,
                        torch.ones_like(residual_dot),
                    )
                    beta = torch.where(
                        valid, updated_dot / safe_residual_dot, torch.zeros_like(updated_dot)
                    )
                    direction = updated_dot_input + direction * beta[None, :]
                    residual_dot = updated_dot
        else:
            append_checkpoint(0, x0, maintained_raw, 0.0)

        checkpoints[best_index] = replace(checkpoints[best_index], selected=True)
        selected = checkpoints[best_index]
        coefficients = gaussian.colors.detach().clone()
        if trainable_ids.numel():
            coefficients[trainable_ids] = best_x
        projected_field = replace(
            field,
            rgb_coeff=coefficients.cpu().numpy().astype(field.rgb_coeff.dtype, copy=False),
        )
        reconstruction_raw = best_reconstruction.cpu().numpy().astype(np.float32, copy=False)
        reconstruction = np.where(active[:, :, None], reconstruction_raw, 0.0).astype(
            np.float32, copy=False
        )

    maintained = render_observation_field(
        projected_field,
        device=device,
        renderer=renderer,
        render_chunk=chunk,
    )
    # The matrix-free operator is intentionally restricted to ``active`` pixels, so its
    # candidate buffer is undefined (zero apart from an explicit frozen base) elsewhere.
    # Certify parity on that common domain, then expose the actual maintained full-crop replay
    # as ``reconstruction_raw`` and retain the black-matted view as ``reconstruction``.
    parity = float(np.max(np.abs(maintained[active] - reconstruction_raw[active])))
    reconstruction_raw = maintained.astype(np.float32, copy=False)
    reconstruction = np.where(active[:, :, None], reconstruction_raw, 0.0).astype(
        np.float32, copy=False
    )
    return CoefficientProjectionResult(
        field=projected_field,
        reconstruction_raw=reconstruction_raw,
        reconstruction=reconstruction,
        checkpoints=tuple(checkpoints),
        selected_iteration=selected.iteration,
        initial_sse=initial_sse,
        final_sse=selected.raw_sse,
        trainable_rows=int(trainable_numpy.sum()),
        frozen_rows=int((~trainable_numpy).sum()),
        protected_rows=int(protected.sum()),
        forward_applications=calls["forward"],
        transpose_applications=calls["transpose"],
        relative_normal_residual_max=selected.relative_normal_residual_max,
        adjoint_relative_error=adjoint_relative_error,
        initial_operator_parity_max_abs=initial_operator_parity_max_abs,
        normal_diagonal_min=normal_diagonal_min,
        normal_diagonal_max=normal_diagonal_max,
        maintained_render_parity_max_abs=parity,
        elapsed_seconds=time.perf_counter() - started,
        selection_mode=cfg.selection_mode,
    )


@dataclass(frozen=True)
class GeometryRelaxationConfig:
    """Geometry-only Adam block used inside HIER-015's wider transaction."""

    steps: int = 400
    checkpoint_every: int = 25
    lr_means: float = 0.01
    lr_scales: float = 0.006
    lr_rotations: float = 0.002
    max_mean_shift_px: float = 4.0
    max_log_scale_shift: float = 0.7
    max_rotation_shift_rad: float = 0.7

    def __post_init__(self) -> None:
        for name in ("steps", "checkpoint_every"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, minimum=1),
            )
        for name in (
            "lr_means",
            "lr_scales",
            "lr_rotations",
            "max_mean_shift_px",
            "max_log_scale_shift",
            "max_rotation_shift_rad",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, strict=True),
            )


@dataclass(frozen=True)
class GeometryRelaxationCheckpoint:
    step: int
    selected: bool
    finite: bool
    raw_sse: float
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class GeometryRelaxationResult:
    """Best raw-SSE geometry state from one RGB-frozen optimization block."""

    field: ObservationField2D
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    checkpoints: tuple[GeometryRelaxationCheckpoint, ...]
    selected_step: int
    initial_sse: float
    final_sse: float
    mean_shift_max_px: float
    log_scale_shift_max: float
    rotation_shift_max_rad: float
    maintained_render_parity_max_abs: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        object.__setattr__(self, "reconstruction", _readonly(self.reconstruction))
        if sum(checkpoint.selected for checkpoint in self.checkpoints) != 1:
            raise ValueError("exactly one geometry checkpoint must be selected")

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


def _field_from_geometry(
    base: ObservationField2D,
    means,
    log_scales,
    rotations,
) -> ObservationField2D:
    """Copy torch geometry into an immutable field without changing any other array."""

    return replace(
        base,
        means_xy=means.detach().cpu().numpy().astype(base.means_xy.dtype, copy=False),
        log_scales_xy=(
            log_scales.detach().cpu().numpy().astype(base.log_scales_xy.dtype, copy=False)
        ),
        rotations_rad=(
            rotations.detach().cpu().numpy().astype(base.rotations_rad.dtype, copy=False)
        ),
    )


def relax_contracted_geometry(
    field: ObservationField2D,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    config: GeometryRelaxationConfig | None = None,
    anchor_field: ObservationField2D | None = None,
    device: str = "cpu",
    renderer: str = "additive",
    render_chunk: int = 256,
) -> GeometryRelaxationResult:
    """Optimize geometry with RGB frozen, returning the lowest finite raw-SSE checkpoint.

    This is intentionally an *intermediate* primitive.  It guarantees finite trust-bounded
    geometry and exact preservation of RGB/non-geometry arrays, but it does not apply the final
    displayed-artifact transaction.  :func:`alternate_projected_geometry` owns that outer guard.
    """

    started = time.perf_counter()
    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be ObservationField2D")
    if field.semantics.renderer_equation != "additive_rgb_peak_one_v1":
        raise ValueError("geometry relaxation requires direct additive peak-one semantics")
    if field.background_rgb is not None:
        raise ValueError("geometry relaxation currently requires a zero-DC field")
    if renderer not in ("additive", "cuda_additive", "cuda_tiled_additive"):
        raise ValueError("renderer must be additive, cuda_additive, or cuda_tiled_additive")
    source = _image(target, "target")
    if source.shape[:2] != field.crop_shape:
        raise ValueError("target shape must match the field crop")
    active = _mask(mask, source.shape[:2])
    cfg = config or GeometryRelaxationConfig()
    anchor = field if anchor_field is None else anchor_field
    if not isinstance(anchor, ObservationField2D) or anchor.n != field.n:
        raise ValueError("anchor_field must be an ObservationField2D with matching row count")
    if anchor.crop_shape != field.crop_shape:
        raise ValueError("anchor_field crop must match field crop")
    chunk = _integer(render_chunk, "render_chunk", minimum=1)

    import torch

    from .pixel_contraction import observation_to_gaussian_field, render_observation_field
    from .render import render_field

    gaussian = observation_to_gaussian_field(field, device=device).trainable()
    gaussian.colors.requires_grad_(False)
    if gaussian.opacities is not None:
        gaussian.opacities.requires_grad_(False)
    dtype = gaussian.means.dtype
    target_tensor = torch.as_tensor(source, device=gaussian.means.device, dtype=dtype)
    mask_tensor = torch.as_tensor(active, device=gaussian.means.device, dtype=torch.bool)
    anchor_means = torch.as_tensor(
        np.array(anchor.means_xy, copy=True), device=gaussian.means.device, dtype=dtype
    )
    anchor_log_scales = torch.as_tensor(
        np.array(anchor.log_scales_xy, copy=True),
        device=gaussian.means.device,
        dtype=dtype,
    )
    anchor_rotations = torch.as_tensor(
        np.array(anchor.rotations_rad, copy=True),
        device=gaussian.means.device,
        dtype=dtype,
    )
    height, width = source.shape[:2]
    dilation = field.semantics.filtering.aa_dilation_px2
    sigma_cutoff = field.semantics.support.sigma_cutoff
    fade_alpha = field.semantics.support.fade_alpha

    def render_current():
        return render_field(
            gaussian.means,
            gaussian.conics(dilation),
            gaussian.colors,
            gaussian.radii(sigma_cutoff, dilation),
            height,
            width,
            chunk=chunk,
            mode=renderer,
            opacities=None,
            scales=gaussian.effective_scales(dilation),
            rotations=gaussian.rotations,
            support_fade=fade_alpha > 0.0,
            sigma_cutoff=sigma_cutoff,
            support_fade_alpha=fade_alpha,
        )

    optimizer = torch.optim.Adam(
        [
            {"params": [gaussian.means], "lr": cfg.lr_means},
            {"params": [gaussian.log_scales], "lr": cfg.lr_scales},
            {"params": [gaussian.rotations], "lr": cfg.lr_rotations},
        ]
    )
    checkpoints: list[GeometryRelaxationCheckpoint] = []
    best_state = (
        gaussian.means.detach().clone(),
        gaussian.log_scales.detach().clone(),
        gaussian.rotations.detach().clone(),
    )
    with torch.no_grad():
        initial_render = render_current()
        initial_residual = initial_render[mask_tensor] - target_tensor[mask_tensor]
        initial_sse = float(torch.sum(initial_residual.square()).cpu())
    best_sse = initial_sse
    best_render = initial_render.detach().clone()
    best_index = 0
    checkpoints.append(
        GeometryRelaxationCheckpoint(
            step=0,
            selected=False,
            finite=math.isfinite(initial_sse),
            raw_sse=initial_sse,
            elapsed_seconds=time.perf_counter() - started,
        )
    )
    tolerance = 1e-10 * max(initial_sse, 1.0)
    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered = render_current()
        residual = rendered[mask_tensor] - target_tensor[mask_tensor]
        loss = torch.mean(residual.square())
        if not bool(torch.isfinite(loss)):
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mean_delta = gaussian.means - anchor_means
            mean_norm = torch.linalg.vector_norm(mean_delta, dim=1, keepdim=True)
            mean_factor = torch.clamp(cfg.max_mean_shift_px / mean_norm.clamp_min(1e-12), max=1.0)
            gaussian.means.copy_(anchor_means + mean_delta * mean_factor)
            gaussian.means[:, 0].clamp_(min=0.0, max=float(width - 1))
            gaussian.means[:, 1].clamp_(min=0.0, max=float(height - 1))
            gaussian.log_scales.clamp_(
                anchor_log_scales - cfg.max_log_scale_shift,
                anchor_log_scales + cfg.max_log_scale_shift,
            )
            gaussian.log_scales.clamp_(min=math.log(1e-3), max=math.log(float(max(height, width))))
            gaussian.rotations.clamp_(
                anchor_rotations - cfg.max_rotation_shift_rad,
                anchor_rotations + cfg.max_rotation_shift_rad,
            )
            finite_parameters = bool(
                torch.isfinite(gaussian.means).all()
                and torch.isfinite(gaussian.log_scales).all()
                and torch.isfinite(gaussian.rotations).all()
            )
            if not finite_parameters:
                break
            if step % cfg.checkpoint_every != 0 and step != cfg.steps:
                continue
            candidate_render = render_current()
            candidate_residual = candidate_render[mask_tensor] - target_tensor[mask_tensor]
            candidate_sse = float(torch.sum(candidate_residual.square()).cpu())
            finite = math.isfinite(candidate_sse)
            checkpoints.append(
                GeometryRelaxationCheckpoint(
                    step=step,
                    selected=False,
                    finite=finite,
                    raw_sse=candidate_sse,
                    elapsed_seconds=time.perf_counter() - started,
                )
            )
            if finite and candidate_sse < best_sse - tolerance:
                best_sse = candidate_sse
                best_index = len(checkpoints) - 1
                best_state = (
                    gaussian.means.detach().clone(),
                    gaussian.log_scales.detach().clone(),
                    gaussian.rotations.detach().clone(),
                )
                best_render = candidate_render.detach().clone()

    checkpoints[best_index] = replace(checkpoints[best_index], selected=True)
    best_means, best_log_scales, best_rotations = best_state
    relaxed_field = _field_from_geometry(field, best_means, best_log_scales, best_rotations)
    reconstruction_raw = best_render.cpu().numpy().astype(np.float32, copy=False)
    reconstruction = np.where(active[:, :, None], reconstruction_raw, 0.0).astype(
        np.float32, copy=False
    )
    maintained = render_observation_field(
        relaxed_field,
        device=device,
        renderer=renderer,
        render_chunk=chunk,
    )
    parity = float(np.max(np.abs(maintained - reconstruction)))
    mean_shift = torch.linalg.vector_norm(best_means - anchor_means, dim=1)
    rotation_shift = torch.abs(best_rotations - anchor_rotations)
    return GeometryRelaxationResult(
        field=relaxed_field,
        reconstruction_raw=reconstruction_raw,
        reconstruction=reconstruction,
        checkpoints=tuple(checkpoints),
        selected_step=checkpoints[best_index].step,
        initial_sse=initial_sse,
        final_sse=best_sse,
        mean_shift_max_px=float(torch.max(mean_shift).cpu()) if field.n else 0.0,
        log_scale_shift_max=(
            float(torch.max(torch.abs(best_log_scales - anchor_log_scales)).cpu())
            if field.n
            else 0.0
        ),
        rotation_shift_max_rad=(float(torch.max(rotation_shift).cpu()) if field.n else 0.0),
        maintained_render_parity_max_abs=parity,
        elapsed_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class AlternatingGeometryConfig:
    """Fail-closed bounded appearance/geometry alternation."""

    rounds: int = 2
    geometry: GeometryRelaxationConfig = dataclass_field(
        default_factory=lambda: GeometryRelaxationConfig(steps=200)
    )
    projection: CoefficientProjectionConfig = dataclass_field(
        default_factory=lambda: CoefficientProjectionConfig(
            tolerance=1e-6,
            max_iterations=96,
            ridge=1e-8,
            coefficient_abs_limit=16.0,
            regularization_center="zero",
            solver_start="zero",
            frozen_base_mode="explicit",
            allow_unsafe_stage_zero_reconditioning=True,
            selection_mode="bounded_intermediate",
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rounds", _integer(self.rounds, "rounds", minimum=1))
        if not isinstance(self.geometry, GeometryRelaxationConfig):
            raise TypeError("geometry must be GeometryRelaxationConfig")
        if not isinstance(self.projection, CoefficientProjectionConfig):
            raise TypeError("projection must be CoefficientProjectionConfig")
        if self.projection.selection_mode != "bounded_intermediate":
            raise ValueError("alternating projection must use bounded_intermediate selection")


@dataclass(frozen=True)
class AlternatingGeometryCheckpoint:
    stage: str
    round_index: int
    selected: bool
    transaction_safe: bool
    raw_sse: float
    display_pixel_rmse_max: float
    display_patch7_rmse_max: float
    display_normalized_violation: float
    coefficient_abs_max: float
    geometry_step: int
    projection_iteration: int
    elapsed_seconds: float

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class AlternatingGeometryResult:
    field: ObservationField2D
    reconstruction_raw: np.ndarray
    reconstruction: np.ndarray
    checkpoints: tuple[AlternatingGeometryCheckpoint, ...]
    geometry_results: tuple[GeometryRelaxationResult, ...]
    projection_results: tuple[CoefficientProjectionResult, ...]
    selected_stage: str
    selected_round: int
    initial_sse: float
    final_sse: float
    total_geometry_steps: int
    forward_applications: int
    transpose_applications: int
    maintained_render_parity_max_abs: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        object.__setattr__(self, "reconstruction", _readonly(self.reconstruction))
        if sum(checkpoint.selected for checkpoint in self.checkpoints) != 1:
            raise ValueError("exactly one alternating checkpoint must be selected")

    def checkpoint_records(self) -> list[dict[str, object]]:
        return [checkpoint.to_record() for checkpoint in self.checkpoints]


def alternate_projected_geometry(
    field: ObservationField2D,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    config: AlternatingGeometryConfig | None = None,
    device: str = "cpu",
    renderer: str = "additive",
    render_chunk: int = 256,
) -> AlternatingGeometryResult:
    """Alternate bounded RGB solves and RGB-frozen geometry blocks, then fail closed.

    Intermediate states may regress the displayed local metric so long as coefficients are finite
    and bounded.  The returned field is selected against the *original* input and may only improve
    raw SSE without increasing its displayed pixel/7x7 normalized violation.  If no candidate is
    safe, the exact input field is returned.
    """

    started = time.perf_counter()
    if not isinstance(field, ObservationField2D):
        raise TypeError("field must be ObservationField2D")
    source = _image(target, "target")
    if source.shape[:2] != field.crop_shape:
        raise ValueError("target shape must match the field crop")
    active = _mask(mask, source.shape[:2])
    cfg = config or AlternatingGeometryConfig()
    chunk = _integer(render_chunk, "render_chunk", minimum=1)
    all_rows = np.ones(field.n, dtype=bool)

    from .pixel_contraction import render_observation_field

    baseline_raw = render_observation_field(
        field, device=device, renderer=renderer, render_chunk=chunk
    )
    baseline_metrics = progressive_artifact_metrics(
        baseline_raw,
        source,
        active,
        pixel_threshold=cfg.projection.pixel_rmse_threshold,
        patch7_threshold=cfg.projection.patch7_rmse_threshold,
        displayed=False,
    )
    baseline_display = progressive_artifact_metrics(
        baseline_raw,
        source,
        active,
        pixel_threshold=cfg.projection.pixel_rmse_threshold,
        patch7_threshold=cfg.projection.patch7_rmse_threshold,
        displayed=True,
    )
    initial_sse = float(baseline_metrics["sse"])
    sse_tolerance = cfg.projection.sse_relative_tolerance * max(initial_sse, 1.0)
    checkpoints: list[AlternatingGeometryCheckpoint] = []
    candidates: list[tuple[ObservationField2D, np.ndarray]] = []
    geometry_results: list[GeometryRelaxationResult] = []
    projection_results: list[CoefficientProjectionResult] = []
    best_index = 0
    best_key = (initial_sse, float(baseline_display["normalized_violation"]), 0)

    def append_candidate(
        *,
        stage: str,
        round_index: int,
        candidate_field: ObservationField2D,
        reconstruction: np.ndarray,
        geometry_step: int,
        projection_iteration: int,
    ) -> None:
        nonlocal best_index, best_key
        raw = progressive_artifact_metrics(
            reconstruction,
            source,
            active,
            pixel_threshold=cfg.projection.pixel_rmse_threshold,
            patch7_threshold=cfg.projection.patch7_rmse_threshold,
            displayed=False,
        )
        display = progressive_artifact_metrics(
            reconstruction,
            source,
            active,
            pixel_threshold=cfg.projection.pixel_rmse_threshold,
            patch7_threshold=cfg.projection.patch7_rmse_threshold,
            displayed=True,
        )
        coefficient_abs_max = float(np.max(np.abs(candidate_field.rgb_coeff)))
        transaction_safe = bool(
            math.isfinite(float(raw["sse"]))
            and coefficient_abs_max <= cfg.projection.coefficient_abs_limit
            and float(raw["sse"]) <= initial_sse + sse_tolerance
            and float(display["normalized_violation"])
            <= float(baseline_display["normalized_violation"])
            + cfg.projection.violation_absolute_tolerance
        )
        checkpoints.append(
            AlternatingGeometryCheckpoint(
                stage=stage,
                round_index=round_index,
                selected=False,
                transaction_safe=transaction_safe,
                raw_sse=float(raw["sse"]),
                display_pixel_rmse_max=float(display["pixel_rmse_max"]),
                display_patch7_rmse_max=float(display["patch7_rmse_max"]),
                display_normalized_violation=float(display["normalized_violation"]),
                coefficient_abs_max=coefficient_abs_max,
                geometry_step=geometry_step,
                projection_iteration=projection_iteration,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        candidates.append((candidate_field, np.array(reconstruction, copy=True)))
        if not transaction_safe:
            return
        key = (
            float(raw["sse"]),
            float(display["normalized_violation"]),
            len(checkpoints) - 1,
        )
        if key < best_key:
            best_key = key
            best_index = len(checkpoints) - 1

    append_candidate(
        stage="input",
        round_index=0,
        candidate_field=field,
        reconstruction=baseline_raw,
        geometry_step=0,
        projection_iteration=0,
    )
    current = field
    projection = project_contracted_coefficients(
        current,
        source,
        active,
        all_rows,
        config=cfg.projection,
        device=device,
        renderer=renderer,
        render_chunk=chunk,
    )
    projection_results.append(projection)
    current = projection.field
    append_candidate(
        stage="initial_projection",
        round_index=0,
        candidate_field=current,
        reconstruction=projection.reconstruction_raw,
        geometry_step=0,
        projection_iteration=projection.selected_iteration,
    )

    selected_projection_checkpoint = next(
        checkpoint for checkpoint in projection.checkpoints if checkpoint.selected
    )
    runnable_rounds = cfg.rounds if selected_projection_checkpoint.bounded else 0
    for round_index in range(1, runnable_rounds + 1):
        geometry = relax_contracted_geometry(
            current,
            source,
            active,
            config=cfg.geometry,
            anchor_field=field,
            device=device,
            renderer=renderer,
            render_chunk=chunk,
        )
        geometry_results.append(geometry)
        current = geometry.field
        append_candidate(
            stage="geometry",
            round_index=round_index,
            candidate_field=current,
            reconstruction=geometry.reconstruction_raw,
            geometry_step=geometry.selected_step,
            projection_iteration=0,
        )
        projection = project_contracted_coefficients(
            current,
            source,
            active,
            all_rows,
            config=cfg.projection,
            device=device,
            renderer=renderer,
            render_chunk=chunk,
        )
        projection_results.append(projection)
        current = projection.field
        append_candidate(
            stage="projection",
            round_index=round_index,
            candidate_field=current,
            reconstruction=projection.reconstruction_raw,
            geometry_step=geometry.selected_step,
            projection_iteration=projection.selected_iteration,
        )

    checkpoints[best_index] = replace(checkpoints[best_index], selected=True)
    selected_field, selected_raw = candidates[best_index]
    selected = checkpoints[best_index]
    selected_masked = np.where(active[:, :, None], selected_raw, 0.0).astype(np.float32, copy=False)
    maintained = render_observation_field(
        selected_field,
        device=device,
        renderer=renderer,
        render_chunk=chunk,
    )
    parity = float(np.max(np.abs(maintained - selected_raw)))
    return AlternatingGeometryResult(
        field=selected_field,
        reconstruction_raw=selected_raw,
        reconstruction=selected_masked,
        checkpoints=tuple(checkpoints),
        geometry_results=tuple(geometry_results),
        projection_results=tuple(projection_results),
        selected_stage=selected.stage,
        selected_round=selected.round_index,
        initial_sse=initial_sse,
        final_sse=selected.raw_sse,
        total_geometry_steps=sum(cfg.geometry.steps for _result in geometry_results),
        forward_applications=sum(result.forward_applications for result in projection_results),
        transpose_applications=sum(result.transpose_applications for result in projection_results),
        maintained_render_parity_max_abs=parity,
        elapsed_seconds=time.perf_counter() - started,
    )


def _mask_vector(value: object, size: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != (size,):
        raise ValueError(f"{name} must be a bool NumPy array with shape ({size},)")
    return np.array(value, dtype=bool, order="C", copy=True)


__all__ = [
    "AlternatingGeometryCheckpoint",
    "AlternatingGeometryConfig",
    "AlternatingGeometryResult",
    "CoefficientProjectionCheckpoint",
    "CoefficientProjectionConfig",
    "CoefficientProjectionResult",
    "GeometryRelaxationCheckpoint",
    "GeometryRelaxationConfig",
    "GeometryRelaxationResult",
    "ResidualAnchorSelection",
    "alternate_projected_geometry",
    "project_contracted_coefficients",
    "relax_contracted_geometry",
    "select_residual_anchor_leaves",
]
