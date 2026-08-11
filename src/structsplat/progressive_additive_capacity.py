"""Progressive pure-additive capacity fitter for HIER-026.

The method is deliberately default-off.  It fits one full-target additive base, inserts a counted
set of signed residual Gaussians, jointly optimizes the resulting field against the original
target, and returns cold-replayable four-array Gaussian endpoints.  Torch remains a lazy
dependency.
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
_PARITY_LIMIT = 2e-5


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
class ProgressiveAdditiveCapacityConfig:
    """Explicit HIER-026 counts, stages, optimizer, and additive-renderer controls."""

    base_gaussians: int = 640
    residual_gaussians: int = 256
    base_steps: int = 500
    joint_steps: int = 200
    checkpoint_every: int = 25
    lr_means: float = 5e-2
    lr_scales: float = 3e-2
    lr_rotations: float = 1e-2
    lr_coefficients: float = 3e-2
    feature_cap_px: float = 12.0
    coefficient_abs_limit: float = 16.0
    sigma_cutoff: float = 3.0
    render_chunk: int = 256
    renderer: str = "additive"

    def __post_init__(self) -> None:
        for name in (
            "base_gaussians",
            "residual_gaussians",
            "base_steps",
            "joint_steps",
            "checkpoint_every",
            "render_chunk",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        for name in (
            "lr_means",
            "lr_scales",
            "lr_rotations",
            "lr_coefficients",
            "feature_cap_px",
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
    def total_gaussians(self) -> int:
        return self.base_gaussians + self.residual_gaussians

    @property
    def total_steps(self) -> int:
        return self.base_steps + self.joint_steps

    @property
    def gaussian_row_updates(self) -> int:
        return (
            self.base_gaussians * self.base_steps
            + self.total_gaussians * self.joint_steps
        )


@dataclass(frozen=True)
class ProgressiveTrajectoryPoint:
    step: int
    stage: str
    raw_mse: float
    psnr_db: float
    observer_renderer_calls: int = 1

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ProgressiveAdditiveCapacityResult:
    """Pure endpoints plus explicit encoder-side audit snapshots and training telemetry."""

    initial_field: "GaussianField"
    base_training_field: "GaussianField"
    birth_field: "GaussianField"
    appended_initial_field: "GaussianField"
    base_field: "GaussianField"
    field: "GaussianField"
    base_reconstruction_raw: np.ndarray
    reconstruction_raw: np.ndarray
    trajectory: tuple[ProgressiveTrajectoryPoint, ...]
    stage_histories: Mapping[str, object]
    selected_steps: Mapping[str, int]
    attempted_steps: int
    completed_steps: int
    base_count: int
    residual_count: int
    total_count: int
    gaussian_row_updates: int
    base_fit_renderer_calls: int
    joint_fit_renderer_calls: int
    observer_renderer_calls: int
    diagnostic_renderer_calls: int
    base_endpoint_parity_max_abs: float
    append_parity_max_abs: float
    endpoint_parity_max_abs: float
    base_endpoint_unchanged: bool
    joint_training_mask_absent: bool
    training_payload_removed: bool
    coefficient_abs_max: float
    negative_birth_coefficients: int
    initial_field_digest: str
    base_training_field_digest: str
    base_endpoint_field_digest: str
    residual_sha256: str
    birth_field_digest: str
    appended_initial_digest: str
    endpoint_field_digest: str
    status: str
    fit_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_reconstruction_raw", _readonly(self.base_reconstruction_raw)
        )
        object.__setattr__(self, "reconstruction_raw", _readonly(self.reconstruction_raw))
        object.__setattr__(self, "trajectory", tuple(self.trajectory))
        object.__setattr__(self, "stage_histories", dict(self.stage_histories))
        object.__setattr__(self, "selected_steps", dict(self.selected_steps))
        if not _pure_payload(self.base_field) or not _pure_payload(self.field):
            raise ValueError("returned endpoints must contain exactly four Gaussian arrays")
        if self.base_field.n != self.base_count:
            raise ValueError("base endpoint count does not match its accounting")
        if self.field.n != self.total_count:
            raise ValueError("progressive endpoint count does not match its accounting")
        if self.initial_field.n != self.base_count:
            raise ValueError("initial audit field count does not match its accounting")
        if self.base_training_field.n != self.base_count:
            raise ValueError("base training audit field count does not match its accounting")
        if self.birth_field.n != self.residual_count:
            raise ValueError("birth audit field count does not match its accounting")
        if self.appended_initial_field.n != self.total_count:
            raise ValueError("appended audit field count does not match its accounting")
        if self.total_count != self.base_count + self.residual_count:
            raise ValueError("progressive count allocation is inconsistent")
        if self.completed_steps > self.attempted_steps:
            raise ValueError("completed_steps cannot exceed attempted_steps")

    @property
    def completed(self) -> bool:
        return self.status == "completed" and self.completed_steps == self.attempted_steps

    @property
    def fit_renderer_calls(self) -> int:
        return self.base_fit_renderer_calls + self.joint_fit_renderer_calls

    def trajectory_records(self) -> list[dict[str, object]]:
        return [point.to_record() for point in self.trajectory]


def _pure_payload(field: "GaussianField") -> bool:
    return all(
        getattr(field, name) is None
        for name in (
            "opacities",
            "scale_max",
            "color_grads",
            "background_mask",
            "filter_variance",
        )
    )


def _pure_endpoint(field: "GaussianField") -> "GaussianField":
    from .gaussians import GaussianField

    if field.opacities is not None or field.color_grads is not None:
        raise ValueError("HIER-026 supports only opacity-free constant-color fields")
    return GaussianField(
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        field.colors.detach().clone(),
    )


def _render_field(
    field: "GaussianField",
    height: int,
    width: int,
    config: ProgressiveAdditiveCapacityConfig,
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
    config: ProgressiveAdditiveCapacityConfig, *, steps: int, count: int
):
    from .config import FitConfig

    return FitConfig(
        iters=steps,
        lr_means=config.lr_means,
        lr_scales=config.lr_scales,
        lr_rot=config.lr_rotations,
        lr_color=config.lr_coefficients,
        optimizer="adam",
        pixel_loss="l1",
        ssim_weight=0.3,
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

    mse_tensor = torch.mean((prediction - target) ** 2)
    mse = float(mse_tensor.detach().cpu())
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


def fit_progressive_additive_capacity(
    target: np.ndarray,
    *,
    seed: int = 0,
    config: ProgressiveAdditiveCapacityConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> ProgressiveAdditiveCapacityResult:
    """Fit HIER-026's shared base and counted residual continuation."""

    import torch

    from .config import InitConfig, StructureTensorConfig
    from .fit import fit
    from .init import build_field

    source = _image(target, "target")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    resolved_seed = int(seed)
    cfg = config or ProgressiveAdditiveCapacityConfig()
    height, width = source.shape[:2]
    target_tensor = torch.as_tensor(source, device=device, dtype=torch.float32)
    tensor_config = StructureTensorConfig()
    trajectory: list[ProgressiveTrajectoryPoint] = []
    observer_calls = 0
    fit_started = time.perf_counter()

    def observe(stage: str, offset: int):
        def callback(field, step: int, _loss: float) -> None:
            nonlocal observer_calls
            rendered = _render_field(field, height, width, cfg)
            mse, psnr = _raw_error(rendered, target_tensor)
            trajectory.append(
                ProgressiveTrajectoryPoint(offset + int(step), stage, mse, psnr)
            )
            observer_calls += 1

        return callback

    base_initial = build_field(
        source,
        InitConfig(
            strategy="aniso_onedge",
            num_gaussians=cfg.base_gaussians,
            seed=resolved_seed,
            sampling_mode="wse",
            flank_offset_frac=0.0,
            scale_cap_mode="feature",
            scale_cap_max=cfg.feature_cap_px,
            color_mode="bilinear",
        ),
        tensor_config,
        device=device,
    )
    initial_field = base_initial.detached()
    initial_field_digest = _field_digest(base_initial)
    base_result = fit(
        base_initial,
        target_tensor,
        _fit_config(cfg, steps=cfg.base_steps, count=cfg.base_gaussians),
        verbose=verbose,
        iteration_observer=observe("base", 0),
        observer_every=cfg.checkpoint_every,
    )
    base_training_field = base_result["field"].detached()
    base_training_digest = _field_digest(base_training_field)
    base_expected = base_result["render"].detach()
    base_endpoint = _pure_endpoint(base_training_field)
    base_endpoint_render = _render_field(base_endpoint, height, width, cfg)
    base_endpoint_parity = float(
        (base_endpoint_render - base_expected).abs().max().detach().cpu()
    )
    base_reconstruction = np.array(
        base_endpoint_render.detach().cpu().numpy(),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    base_endpoint_digest = _field_digest(base_endpoint)

    residual_tensor = target_tensor - base_expected
    residual = np.array(
        residual_tensor.detach().cpu().numpy(), dtype=np.float32, order="C", copy=True
    )
    birth_field = build_field(
        residual,
        InitConfig(
            strategy="aniso_onedge",
            num_gaussians=cfg.residual_gaussians,
            seed=resolved_seed,
            sampling_mode="wse",
            flank_offset_frac=0.0,
            scale_cap_mode="feature",
            scale_cap_max=cfg.feature_cap_px,
            color_mode="bilinear",
        ),
        tensor_config,
        device=device,
    )
    negative_birth_coefficients = int((birth_field.colors < 0.0).sum().detach().cpu())
    birth_render = _render_field(birth_field, height, width, cfg)
    appended = base_training_field.append(birth_field)
    appended_initial_field = appended.detached()
    appended_initial_digest = _field_digest(appended)
    appended_render = _render_field(appended, height, width, cfg)
    append_parity = float(
        (appended_render - (base_expected + birth_render)).abs().max().detach().cpu()
    )
    joint_training_mask_absent = appended.background_mask is None
    joint_result = fit(
        appended,
        target_tensor,
        _fit_config(cfg, steps=cfg.joint_steps, count=cfg.total_gaussians),
        verbose=verbose,
        iteration_observer=observe("joint", cfg.base_steps),
        observer_every=cfg.checkpoint_every,
    )
    joint_training_field = joint_result["field"].detached()
    joint_expected = joint_result["render"].detach()
    endpoint = _pure_endpoint(joint_training_field)
    endpoint_render = _render_field(endpoint, height, width, cfg)
    endpoint_parity = float(
        (endpoint_render - joint_expected).abs().max().detach().cpu()
    )
    reconstruction = np.array(
        endpoint_render.detach().cpu().numpy(),
        dtype=np.float32,
        order="C",
        copy=True,
    )
    completed_steps = int(base_result["iterations_run"]) + int(
        joint_result["iterations_run"]
    )
    coefficient_abs_max = float(endpoint.colors.detach().abs().max().cpu())
    base_endpoint_unchanged = _field_digest(base_endpoint) == base_endpoint_digest
    training_payload_removed = _pure_payload(base_endpoint) and _pure_payload(endpoint)
    finite = bool(np.isfinite(base_reconstruction).all()) and bool(
        np.isfinite(reconstruction).all()
    ) and all(
        bool(torch.isfinite(getattr(field, name)).all())
        for field in (base_endpoint, endpoint)
        for name in ("means", "log_scales", "rotations", "colors")
    )
    if not finite:
        status = "nonfinite_endpoint"
    elif coefficient_abs_max > cfg.coefficient_abs_limit:
        status = "coefficient_limit"
    elif base_endpoint.n != cfg.base_gaussians or endpoint.n != cfg.total_gaussians:
        status = "count_mismatch"
    elif not base_endpoint_unchanged:
        status = "shared_base_mutated"
    elif not joint_training_mask_absent:
        status = "training_mask_present"
    elif not training_payload_removed:
        status = "payload_present"
    elif max(base_endpoint_parity, append_parity, endpoint_parity) > _PARITY_LIMIT:
        status = "parity_failure"
    elif completed_steps != cfg.total_steps:
        status = "incomplete"
    else:
        status = "completed"
    fit_seconds = time.perf_counter() - fit_started
    return ProgressiveAdditiveCapacityResult(
        initial_field=initial_field,
        base_training_field=base_training_field,
        birth_field=birth_field.detached(),
        appended_initial_field=appended_initial_field,
        base_field=base_endpoint,
        field=endpoint,
        base_reconstruction_raw=base_reconstruction,
        reconstruction_raw=reconstruction,
        trajectory=tuple(sorted(trajectory, key=lambda point: point.step)),
        stage_histories={
            "base": _stage_history(base_result),
            "joint": _stage_history(joint_result),
        },
        selected_steps={
            "base": int(base_result["selected_iter"]),
            "joint": cfg.base_steps + int(joint_result["selected_iter"]),
        },
        attempted_steps=cfg.total_steps,
        completed_steps=completed_steps,
        base_count=cfg.base_gaussians,
        residual_count=cfg.residual_gaussians,
        total_count=cfg.total_gaussians,
        gaussian_row_updates=cfg.gaussian_row_updates,
        base_fit_renderer_calls=_fit_renderer_calls(base_result),
        joint_fit_renderer_calls=_fit_renderer_calls(joint_result),
        observer_renderer_calls=observer_calls,
        diagnostic_renderer_calls=4,
        base_endpoint_parity_max_abs=base_endpoint_parity,
        append_parity_max_abs=append_parity,
        endpoint_parity_max_abs=endpoint_parity,
        base_endpoint_unchanged=base_endpoint_unchanged,
        joint_training_mask_absent=joint_training_mask_absent,
        training_payload_removed=training_payload_removed,
        coefficient_abs_max=coefficient_abs_max,
        negative_birth_coefficients=negative_birth_coefficients,
        initial_field_digest=initial_field_digest,
        base_training_field_digest=base_training_digest,
        base_endpoint_field_digest=base_endpoint_digest,
        residual_sha256=_array_digest(residual),
        birth_field_digest=_field_digest(birth_field),
        appended_initial_digest=appended_initial_digest,
        endpoint_field_digest=_field_digest(endpoint),
        status=status,
        fit_seconds=fit_seconds,
    )
