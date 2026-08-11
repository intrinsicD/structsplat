"""Fold a low-pass/residual hierarchy into one additive Gaussian field (HIER-025).

The method is deliberately default-off.  It optimizes two counted Gaussian levels against a
low-pass image and its signed residual, concatenates them, performs a short joint additive polish,
and returns one ordinary opacity-free ``GaussianField``.  Torch remains a lazy dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import time
from typing import TYPE_CHECKING, Mapping

import numpy as np


if TYPE_CHECKING:
    from .gaussians import GaussianField


_ADDITIVE_RENDERERS = frozenset(("additive", "cuda_additive", "cuda_tiled_additive"))


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {result}")
    return result


def _image(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError(f"{name} must have non-empty HWC RGB shape")
    if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FoldedMultiscaleAdditiveConfig:
    """Explicit HIER-025 counts, stages, optimizer, and additive-renderer controls."""

    total_gaussians: int = 640
    coarse_gaussians: int = 16
    coarse_steps: int = 100
    detail_steps: int = 300
    joint_steps: int = 100
    checkpoint_every: int = 25
    lr_means: float = 5e-2
    lr_scales: float = 3e-2
    lr_rotations: float = 1e-2
    lr_coefficients: float = 3e-2
    detail_feature_cap_px: float = 12.0
    coefficient_abs_limit: float = 16.0
    sigma_cutoff: float = 3.0
    render_chunk: int = 256
    renderer: str = "additive"

    def __post_init__(self) -> None:
        for name in (
            "total_gaussians",
            "coarse_gaussians",
            "coarse_steps",
            "detail_steps",
            "joint_steps",
            "checkpoint_every",
            "render_chunk",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.coarse_gaussians >= self.total_gaussians:
            raise ValueError("coarse_gaussians must be smaller than total_gaussians")
        for name in (
            "lr_means",
            "lr_scales",
            "lr_rotations",
            "lr_coefficients",
            "detail_feature_cap_px",
            "coefficient_abs_limit",
            "sigma_cutoff",
        ):
            object.__setattr__(
                self, name, _finite_positive(getattr(self, name), name)
            )
        if self.renderer not in _ADDITIVE_RENDERERS:
            expected = ", ".join(sorted(_ADDITIVE_RENDERERS))
            raise ValueError(f"renderer must be one of {expected}")

    @property
    def detail_gaussians(self) -> int:
        return self.total_gaussians - self.coarse_gaussians

    @property
    def total_steps(self) -> int:
        return self.coarse_steps + self.detail_steps + self.joint_steps


@dataclass(frozen=True)
class FoldedTrajectoryPoint:
    step: int
    stage: str
    raw_mse: float
    psnr_db: float
    observer_renderer_calls: int = 1

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class FoldedMultiscaleAdditiveResult:
    """One pure-additive endpoint plus scalar/history-only training telemetry."""

    field: "GaussianField"
    reconstruction_raw: np.ndarray
    trajectory: tuple[FoldedTrajectoryPoint, ...]
    stage_histories: Mapping[str, object]
    selected_steps: Mapping[str, int]
    attempted_steps: int
    completed_steps: int
    coarse_count: int
    detail_count: int
    fit_renderer_calls: int
    observer_renderer_calls: int
    diagnostic_renderer_calls: int
    fold_parity_max_abs: float
    endpoint_parity_max_abs: float
    coarse_geometry_exact: bool
    training_mask_removed: bool
    coefficient_abs_max: float
    lowpass_sha256: str
    residual_sha256: str
    coarse_field_digest: str
    detail_field_digest: str
    folded_initial_digest: str
    endpoint_field_digest: str
    coarse_geometry_digest: str
    status: str
    fit_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
        object.__setattr__(self, "stage_histories", dict(self.stage_histories))
        object.__setattr__(self, "selected_steps", dict(self.selected_steps))
        if self.field.opacities is not None or self.field.color_grads is not None:
            raise ValueError("folded endpoint must be opacity-free constant-color additive")
        if self.field.background_mask is not None:
            raise ValueError("folded endpoint retained its training-only level mask")
        if self.field.n != self.coarse_count + self.detail_count:
            raise ValueError("folded endpoint count does not match its level accounting")
        if self.completed_steps > self.attempted_steps:
            raise ValueError("completed_steps cannot exceed attempted_steps")

    @property
    def completed(self) -> bool:
        return self.status == "completed" and self.completed_steps == self.attempted_steps

    def trajectory_records(self) -> list[dict[str, object]]:
        return [point.to_record() for point in self.trajectory]


def area_bilinear_lowpass(target: np.ndarray, factor: int = 2) -> np.ndarray:
    """Return area-downsampled/bilinear-upsampled HWC data at the original shape."""

    source = _image(target, "target")
    downsample = _integer(factor, "factor")
    if downsample == 1:
        return _readonly(source)
    import torch
    import torch.nn.functional as functional

    height, width = source.shape[:2]
    low_size = (max(1, height // downsample), max(1, width // downsample))
    with torch.no_grad():
        bchw = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0)
        low = functional.interpolate(bchw, size=low_size, mode="area")
        restored = functional.interpolate(
            low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    return _readonly(restored.squeeze(0).permute(1, 2, 0).numpy())


def _render_field(
    field: "GaussianField",
    height: int,
    width: int,
    config: FoldedMultiscaleAdditiveConfig,
):
    from .render import render_field

    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(config.sigma_cutoff),
        height,
        width,
        chunk=config.render_chunk,
        mode=config.renderer,
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=False,
        sigma_cutoff=config.sigma_cutoff,
    )


def _fit_config(
    config: FoldedMultiscaleAdditiveConfig,
    *,
    steps: int,
    count: int,
    pixel_loss: str,
    ssim_weight: float,
):
    from .config import FitConfig

    return FitConfig(
        iters=steps,
        lr_means=config.lr_means,
        lr_scales=config.lr_scales,
        lr_rot=config.lr_rotations,
        lr_color=config.lr_coefficients,
        optimizer="adam",
        pixel_loss=pixel_loss,
        ssim_weight=ssim_weight,
        log_every=config.checkpoint_every,
        checkpoint_policy="best_psnr_final_count",
        sigma_cutoff=config.sigma_cutoff,
        support_fade=False,
        aa_dilation=0.0,
        render_chunk=config.render_chunk,
        renderer=config.renderer,
        color_basis="constant",
        compute_lpips=False,
        max_gaussians=count,
    )


def _fit_renderer_calls(result: Mapping[str, object]) -> int:
    checkpoint_history = result["checkpoint_history"]
    if not isinstance(checkpoint_history, Mapping):
        raise TypeError("fit checkpoint_history must be a mapping")
    iterations = checkpoint_history.get("iter")
    if not isinstance(iterations, list):
        raise TypeError("fit checkpoint_history.iter must be a list")
    return (
        int(result["iterations_run"])
        + len(iterations)
        + int(bool(result["selected_from_checkpoint"]))
    )


def _raw_error(prediction, target) -> tuple[float, float]:
    import torch

    mse_t = torch.mean((prediction - target) ** 2)
    mse = float(mse_t.detach().cpu())
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return mse, psnr


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _field_digest(field: "GaussianField") -> str:
    digest = sha256()
    for name in ("means", "log_scales", "rotations", "colors"):
        value = np.ascontiguousarray(getattr(field, name).detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _geometry_digest(geometry) -> str:
    digest = sha256()
    for name, value in zip(("means", "log_scales", "rotations"), geometry):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _geometry_tuple(field: "GaussianField", count: int):
    return tuple(
        getattr(field, name)[:count].detach().clone()
        for name in ("means", "log_scales", "rotations")
    )


def _geometry_equal(before, field: "GaussianField", count: int) -> bool:
    import torch

    return all(
        torch.equal(reference, getattr(field, name)[:count])
        for reference, name in zip(before, ("means", "log_scales", "rotations"))
    )


def _stage_history(result: Mapping[str, object]) -> dict[str, object]:
    return {
        "history": result["history"],
        "checkpoint_history": result["checkpoint_history"],
        "selected_iter": int(result["selected_iter"]),
        "selected_psnr": float(result["selected_psnr"]),
        "selected_from_checkpoint": bool(result["selected_from_checkpoint"]),
        "iterations_run": int(result["iterations_run"]),
        "fit_seconds": float(result["fit_seconds"]),
    }


def fit_folded_multiscale_additive(
    target: np.ndarray,
    *,
    seed: int = 0,
    config: FoldedMultiscaleAdditiveConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> FoldedMultiscaleAdditiveResult:
    """Fit HIER-025's counted levels and return a single direct-additive field."""

    import torch

    from .config import InitConfig, StructureTensorConfig
    from .fit import fit
    from .init import build_field

    source = _image(target, "target")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    resolved_seed = int(seed)
    cfg = config or FoldedMultiscaleAdditiveConfig()
    height, width = source.shape[:2]
    target_tensor = torch.as_tensor(source, device=device, dtype=torch.float32)
    lowpass = area_bilinear_lowpass(source, 2)
    lowpass_tensor = torch.as_tensor(
        np.array(lowpass, dtype=np.float32, order="C", copy=True),
        device=device,
        dtype=torch.float32,
    )
    tensor_config = StructureTensorConfig()
    trajectory: list[FoldedTrajectoryPoint] = []
    observer_calls = 0
    fit_started = time.perf_counter()

    def observe(
        stage: str,
        offset: int,
        full_target,
        frozen_prefix=None,
    ):
        def callback(field, step: int, _loss: float) -> None:
            nonlocal observer_calls
            rendered = _render_field(field, height, width, cfg)
            if frozen_prefix is not None:
                rendered = frozen_prefix + rendered
            mse, psnr = _raw_error(rendered, full_target)
            trajectory.append(
                FoldedTrajectoryPoint(offset + int(step), stage, mse, psnr)
            )
            observer_calls += 1

        return callback

    coarse_initial = build_field(
        lowpass,
        InitConfig(
            strategy="grid",
            num_gaussians=cfg.coarse_gaussians,
            seed=resolved_seed,
            scale_cap_mode="none",
            color_mode="bilinear",
        ),
        tensor_config,
        device=device,
    )
    coarse_result = fit(
        coarse_initial,
        lowpass_tensor,
        _fit_config(
            cfg,
            steps=cfg.coarse_steps,
            count=cfg.coarse_gaussians,
            pixel_loss="l2",
            ssim_weight=0.0,
        ),
        verbose=verbose,
        iteration_observer=observe("coarse", 0, target_tensor),
        observer_every=cfg.checkpoint_every,
    )
    coarse_field = coarse_result["field"].detached()
    coarse_render = coarse_result["render"].detach()
    residual_tensor = target_tensor - coarse_render
    residual = np.array(
        residual_tensor.detach().cpu().numpy(), dtype=np.float32, order="C", copy=True
    )

    detail_initial = build_field(
        residual,
        InitConfig(
            strategy="aniso_onedge",
            num_gaussians=cfg.detail_gaussians,
            seed=resolved_seed,
            sampling_mode="wse",
            flank_offset_frac=0.0,
            scale_cap_mode="feature",
            scale_cap_max=cfg.detail_feature_cap_px,
            color_mode="bilinear",
        ),
        tensor_config,
        device=device,
    )
    detail_result = fit(
        detail_initial,
        residual_tensor,
        _fit_config(
            cfg,
            steps=cfg.detail_steps,
            count=cfg.detail_gaussians,
            pixel_loss="l2",
            ssim_weight=0.0,
        ),
        verbose=verbose,
        iteration_observer=observe(
            "detail",
            cfg.coarse_steps,
            target_tensor,
            frozen_prefix=coarse_render,
        ),
        observer_every=cfg.checkpoint_every,
    )
    detail_field = detail_result["field"].detached()
    detail_render = detail_result["render"].detach()

    coarse_field.background_mask = torch.ones(
        coarse_field.n, device=coarse_field.means.device, dtype=torch.bool
    )
    folded = coarse_field.append(detail_field)
    if folded.n != cfg.total_gaussians or folded.background_count != cfg.coarse_gaussians:
        raise RuntimeError("folded level allocation does not match the configured exact count")
    coarse_geometry = _geometry_tuple(folded, cfg.coarse_gaussians)
    folded_initial_digest = _field_digest(folded)
    folded_initial_render = _render_field(folded, height, width, cfg)
    fold_parity = float(
        (folded_initial_render - (coarse_render + detail_render)).abs().max().detach().cpu()
    )

    joint_result = fit(
        folded,
        target_tensor,
        _fit_config(
            cfg,
            steps=cfg.joint_steps,
            count=cfg.total_gaussians,
            pixel_loss="l1",
            ssim_weight=0.3,
        ),
        verbose=verbose,
        iteration_observer=observe(
            "joint", cfg.coarse_steps + cfg.detail_steps, target_tensor
        ),
        observer_every=cfg.checkpoint_every,
    )
    joint_field = joint_result["field"].detached()
    coarse_geometry_exact = _geometry_equal(
        coarse_geometry, joint_field, cfg.coarse_gaussians
    )
    joint_expected = joint_result["render"].detach()
    joint_field.background_mask = None
    stripped_render = _render_field(joint_field, height, width, cfg)
    endpoint_parity = float(
        (stripped_render - joint_expected).abs().max().detach().cpu()
    )
    reconstruction = np.array(
        stripped_render.detach().cpu().numpy(), dtype=np.float32, order="C", copy=True
    )
    coefficient_abs_max = float(joint_field.colors.detach().abs().max().cpu())
    stage_results = (coarse_result, detail_result, joint_result)
    completed_steps = sum(int(result["iterations_run"]) for result in stage_results)
    finite = bool(np.isfinite(reconstruction).all()) and bool(
        torch.isfinite(joint_field.means).all()
        and torch.isfinite(joint_field.log_scales).all()
        and torch.isfinite(joint_field.rotations).all()
        and torch.isfinite(joint_field.colors).all()
    )
    if not finite:
        status = "nonfinite_endpoint"
    elif coefficient_abs_max > cfg.coefficient_abs_limit:
        status = "coefficient_limit"
    elif not coarse_geometry_exact:
        status = "coarse_geometry_changed"
    elif fold_parity > 2e-5 or endpoint_parity > 2e-5:
        status = "parity_failure"
    elif completed_steps != cfg.total_steps:
        status = "incomplete"
    else:
        status = "completed"
    fit_seconds = time.perf_counter() - fit_started
    fit_renderer_calls = sum(_fit_renderer_calls(result) for result in stage_results)
    return FoldedMultiscaleAdditiveResult(
        field=joint_field,
        reconstruction_raw=reconstruction,
        trajectory=tuple(sorted(trajectory, key=lambda point: point.step)),
        stage_histories={
            "coarse": _stage_history(coarse_result),
            "detail": _stage_history(detail_result),
            "joint": _stage_history(joint_result),
        },
        selected_steps={
            "coarse": int(coarse_result["selected_iter"]),
            "detail": cfg.coarse_steps + int(detail_result["selected_iter"]),
            "joint": (
                cfg.coarse_steps
                + cfg.detail_steps
                + int(joint_result["selected_iter"])
            ),
        },
        attempted_steps=cfg.total_steps,
        completed_steps=completed_steps,
        coarse_count=cfg.coarse_gaussians,
        detail_count=cfg.detail_gaussians,
        fit_renderer_calls=fit_renderer_calls + observer_calls,
        observer_renderer_calls=observer_calls,
        diagnostic_renderer_calls=2,
        fold_parity_max_abs=fold_parity,
        endpoint_parity_max_abs=endpoint_parity,
        coarse_geometry_exact=coarse_geometry_exact,
        training_mask_removed=joint_field.background_mask is None,
        coefficient_abs_max=coefficient_abs_max,
        lowpass_sha256=_array_digest(lowpass),
        residual_sha256=_array_digest(residual),
        coarse_field_digest=_field_digest(coarse_field),
        detail_field_digest=_field_digest(detail_field),
        folded_initial_digest=folded_initial_digest,
        endpoint_field_digest=_field_digest(joint_field),
        coarse_geometry_digest=_geometry_digest(coarse_geometry),
        status=status,
        fit_seconds=fit_seconds,
    )
