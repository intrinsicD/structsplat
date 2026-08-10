"""Count-neutral residual anchoring and RGB projection for HIER-010.

This module is a deterministic, default-off research reference.  It keeps the direct-additive
Observation Field V2 geometry fixed and never materializes the dense pixel-by-Gaussian matrix.
Torch is imported lazily so the package's NumPy-only analysis boundary remains intact.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from .observation_field import ObservationField2D
from .progressive_residual_quadtree import progressive_artifact_metrics


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite(value: object, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
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

    def __post_init__(self) -> None:
        for name in (
            "tolerance",
            "coefficient_abs_limit",
            "pixel_rmse_threshold",
            "patch7_rmse_threshold",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, strict=True)
            )
        for name in ("ridge", "sse_relative_tolerance", "violation_absolute_tolerance"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_iterations",
            _integer(self.max_iterations, "max_iterations", minimum=1),
        )


@dataclass(frozen=True)
class CoefficientProjectionCheckpoint:
    iteration: int
    selected: bool
    selectable: bool
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
    maintained_render_parity_max_abs: float
    elapsed_seconds: float

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
    not allocate the dense ``pixels x rows`` basis matrix.  Every PCG iterate is checked against
    stage-zero raw SSE and displayed artifact violation; an unsafe solve returns step zero.
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
    local_of_global = torch.full(
        (field.n,), -1, device=torch_device, dtype=torch.long
    )
    if trainable_ids.numel():
        local_of_global[trainable_ids] = torch.arange(
            trainable_ids.numel(), device=torch_device
        )
    x0 = gaussian.colors.detach()[trainable_ids].clone()

    x0_bounds = _tile_bounds(means, radii, height, width)
    x0_tile, y0_tile, tile_width, tile_elements = x0_bounds
    budget = _element_budget(chunk)
    calls = {"forward": 0, "transpose": 0}
    sigma_cutoff = field.semantics.support.sigma_cutoff
    fade_alpha = field.semantics.support.fade_alpha

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
        maintained_raw = render_field(
            means,
            conics,
            gaussian.colors.detach(),
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
        initial_variable = basis_apply(x0) if trainable_ids.numel() else torch.zeros_like(
            maintained_raw
        )
        frozen_base = maintained_raw - initial_variable
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
                (left - right).abs()
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
            selectable = bool(
                math.isfinite(float(raw["sse"]))
                and coefficient_abs_max <= cfg.coefficient_abs_limit
                and float(raw["sse"]) <= initial_sse + sse_tolerance
                and float(display["normalized_violation"])
                <= float(initial_display_metrics["normalized_violation"])
                + cfg.violation_absolute_tolerance
            )
            checkpoints.append(
                CoefficientProjectionCheckpoint(
                    iteration=iteration,
                    selected=False,
                    selectable=selectable,
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

        if trainable_ids.numel():
            ridge = cfg.ridge
            diagonal = (diagonal_values() + ridge).clamp_min(torch.finfo(dtype).tiny)
            right_hand_side = basis_transpose(objective_target)
            if ridge > 0.0:
                right_hand_side = right_hand_side + ridge * x0

            def normal(values):
                result = basis_transpose(basis_apply(values))
                return result + ridge * values if ridge > 0.0 else result

            x = x0.clone()
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
            if checkpoints[0].selectable:
                epsilon = torch.finfo(dtype).eps
                for iteration in range(1, cfg.max_iterations + 1):
                    if bool(torch.all(relative <= cfg.tolerance)):
                        break
                    product = normal(direction)
                    denominator = torch.sum(direction * product, dim=0)
                    valid = (denominator > epsilon) & (relative > cfg.tolerance)
                    safe_denominator = torch.where(
                        valid, denominator, torch.ones_like(denominator)
                    )
                    alpha = torch.where(
                        valid, residual_dot / safe_denominator, torch.zeros_like(residual_dot)
                    )
                    x = x + direction * alpha[None, :]
                    residual = residual - product * alpha[None, :]
                    updated_dot_input = residual / diagonal[:, None]
                    updated_dot = torch.sum(residual * updated_dot_input, dim=0)
                    relative = torch.sqrt(torch.sum(residual.square(), dim=0)) / right_norm
                    candidate = frozen_base + basis_apply(x)
                    append_checkpoint(
                        iteration, x, candidate, float(torch.max(relative).cpu())
                    )
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
    parity = float(np.max(np.abs(maintained - reconstruction)))
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
        maintained_render_parity_max_abs=parity,
        elapsed_seconds=time.perf_counter() - started,
    )


def _mask_vector(value: object, size: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.bool_ or value.shape != (size,):
        raise ValueError(f"{name} must be a bool NumPy array with shape ({size},)")
    return np.array(value, dtype=bool, order="C", copy=True)


__all__ = [
    "CoefficientProjectionCheckpoint",
    "CoefficientProjectionConfig",
    "CoefficientProjectionResult",
    "ResidualAnchorSelection",
    "project_contracted_coefficients",
    "select_residual_anchor_leaves",
]
