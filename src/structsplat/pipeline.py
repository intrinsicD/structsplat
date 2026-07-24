"""Frozen operational pipeline shared by conversion and benchmark workflows.

The profile is the bounded Janelle development winner from 2026-07-24: fixed
12,024-row storage, an 11,512-row ordinary active ceiling, global transactional
refinement, Pareto-safe checkpoints every 50 steps, and no specialized detail
tail or event color solve. This is an operational profile, not a repository-wide
quality or SOTA claim.

Masked and unmasked execution intentionally share the same 5,000-row start,
phase budgets, active ceiling, optimizer, proposal auction, checkpoint policy,
and polish. Masked execution replaces 500 general initial rows with explicit
boundary rows and enables boundary losses/proposals. Unmasked execution keeps
all 5,000 general rows and substitutes count-matched coverage/detail proposals
in the closure phase; no boundary metric, loss, cap, or proposal is active.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import time
from typing import Any, Callable

import numpy as np

from .config import FitConfig, InitConfig, StructureTensorConfig
from .fit import _MaskConstraint, _boundary_tangent_add, _render
from .init import build_field, build_masked_field
from .safe_schedule import SafeScheduleConfig, run_safe_schedule


CURRENT_PROFILE_NAME = "safe_schedule_2026_07_24"
CURRENT_PROFILE_EVIDENCE_SCOPE = (
    "single-image Janelle development winner; operational default for these "
    "workflow scripts, not a repository-wide superiority claim"
)
INITIAL_GAUSSIANS = 5_000
MASKED_GENERAL_GAUSSIANS = 4_500
MASKED_BOUNDARY_GAUSSIANS = 500
PHYSICAL_CAPACITY = 12_024
ACTIVE_LIMIT = 11_512

PipelineObserver = Callable[[Any, dict[str, Any], Any, FitConfig], None]
ScheduleTransform = Callable[[SafeScheduleConfig], SafeScheduleConfig]


def build_current_schedule(*, boundary_enabled: bool) -> SafeScheduleConfig:
    """Return the fully resolved current workflow schedule."""

    defaults = SafeScheduleConfig()
    closure_name = "boundary_closure" if boundary_enabled else "general_closure"
    return SafeScheduleConfig(
        capacity=PHYSICAL_CAPACITY,
        storage_policy="fixed_capacity",
        boundary_enabled=bool(boundary_enabled),
        base_active_limit=ACTIVE_LIMIT,
        detail_tail_max_rows=0,
        detail_tail_batch_rows=defaults.detail_tail_batch_rows,
        detail_tail_min_gain_per_row=0.0,
        coverage_target_gaussians=8_000,
        detail_target_gaussians=10_000,
        coverage_tau=defaults.coverage_tau,
        boundary_band=defaults.boundary_band,
        interior_hole_target=defaults.interior_hole_target,
        boundary_hole_target=defaults.boundary_hole_target,
        stale_patience=defaults.stale_patience,
        event_min_count=defaults.event_min_count,
        recovery_steps=defaults.recovery_steps,
        event_spacing_px=defaults.event_spacing_px,
        event_oversample=defaults.event_oversample,
        coverage_birth_count=defaults.coverage_birth_count,
        detail_birth_count=defaults.detail_birth_count,
        detail_split_count=defaults.detail_split_count,
        boundary_birth_count=defaults.boundary_birth_count,
        redistribution_count=defaults.redistribution_count,
        redistribution_min_responsibility=defaults.redistribution_min_responsibility,
        split_shrink=defaults.split_shrink,
        hole_opacity=defaults.hole_opacity,
        covered_birth_opacity=defaults.covered_birth_opacity,
        merge_envelope_inflation=defaults.merge_envelope_inflation,
        refinement_policy="global",
        local_start_phase=defaults.local_start_phase,
        local_seed_count=defaults.local_seed_count,
        local_neighbor_count=defaults.local_neighbor_count,
        topology_neighbor_count=defaults.topology_neighbor_count,
        boundary_recycle_at_capacity=False,
        pareto_safe_checkpoints=True,
        pareto_checkpoint_every=50,
        event_color_solve=False,
        boundary_residual_mse_threshold=defaults.boundary_residual_mse_threshold,
        boundary_residual_component_target=defaults.boundary_residual_component_target,
        boundary_residual_min_pixels=defaults.boundary_residual_min_pixels,
        tolerances=defaults.tolerances,
        bootstrap=replace(defaults.bootstrap, target_gaussians=INITIAL_GAUSSIANS),
        coverage=replace(defaults.coverage, target_gaussians=8_000),
        detail=replace(defaults.detail, target_gaussians=10_000),
        boundary=replace(
            defaults.boundary,
            name=closure_name,
            target_gaussians=ACTIVE_LIMIT,
        ),
        redistribution=replace(
            defaults.redistribution, target_gaussians=ACTIVE_LIMIT
        ),
        polish=replace(defaults.polish, target_gaussians=ACTIVE_LIMIT),
    )


def build_current_fit_config(*, masked: bool, mask_margin: float = 0.75) -> FitConfig:
    """Return the shared optimizer/renderer configuration."""

    return FitConfig(
        iters=1,
        renderer="cuda_tiled",
        render_chunk=512,
        pixel_loss="l2",
        ssim_weight=0.0,
        loss_weighting="mask",
        mask_contain=bool(masked),
        mask_margin=float(mask_margin),
        mask_cap_mode="anisotropic",
        mask_cap_refresh_every=100,
        mask_undercoverage_band=4.0,
        mask_undercoverage_tau=0.05,
        mask_undercoverage_every=8,
        support_fade=True,
        checkpoint_policy="terminal",
        split_scale=0.35,
        split_oversample=8.0,
        split_min_spacing=1.0,
        densify_max_axis_ratio=6.0,
        densify_coherence_power=1.0,
        color_solve_lambda=1e-4,
        color_solve_maxiter=32,
        compute_lpips=False,
        log_every=1,
    )


def build_initialization_config(
    *, seed: int, strategy: str = "quadtree_wse"
) -> InitConfig:
    """Return the initialization shared by masked and unmasked paths."""

    return InitConfig(
        strategy=str(strategy),
        num_gaussians=INITIAL_GAUSSIANS,
        seed=int(seed),
        sampling_mode="wse",
        wse_progressive_order=True,
        init_scale_mult=0.35,
        scale_cap_mode="none",
        background_fraction=0.0,
        background_grid=0,
    )


def _initialize(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    device: str,
    seed: int,
    strategy: str,
    boundary_enabled: bool,
    mask_margin: float,
):
    import torch

    init_cfg = build_initialization_config(seed=seed, strategy=strategy)
    tensor_cfg = StructureTensorConfig()
    if mask is None:
        field = build_field(
            image, init_cfg, tensor_cfg, device=device
        )
        return field, init_cfg, tensor_cfg, 0

    mask_bool = np.asarray(mask, dtype=bool)
    target_np = image * mask_bool[..., None].astype(np.float32)
    pool = build_masked_field(
        target_np,
        mask_bool,
        init_cfg,
        tensor_cfg,
        device=device,
        sigma_cutoff=3.0,
        mask_margin=float(mask_margin),
        contain=False,
    )
    if not boundary_enabled:
        return pool, init_cfg, tensor_cfg, 0

    field = pool.subset(slice(0, MASKED_GENERAL_GAUSSIANS))
    cfg = replace(
        build_current_fit_config(masked=True, mask_margin=mask_margin),
        mask_boundary_add_every=1,
        mask_boundary_add_count=MASKED_BOUNDARY_GAUSSIANS,
        mask_boundary_add_band=4.0,
        mask_boundary_add_spacing=3.0,
        max_gaussians=INITIAL_GAUSSIANS,
        split_color_init="target",
    )
    constraint = _MaskConstraint.from_mask(
        mask_bool,
        field.means.device,
        field.means.dtype,
        cfg.sigma_cutoff,
        float(mask_margin),
        aa_dilation=cfg.aa_dilation,
        cap_mode="anisotropic",
    )
    constraint.apply(field, cfg, refresh=True)
    target = torch.as_tensor(
        target_np, device=device, dtype=torch.float32
    )
    with torch.no_grad():
        rendered = _render(
            field,
            cfg,
            target.shape[0],
            target.shape[1],
            support_fade_alpha=1.0,
        )
        field, boundary_added = _boundary_tangent_add(
            field, target, rendered, cfg, constraint
        )
        constraint.apply(field, cfg, refresh=True)
    if field.n != INITIAL_GAUSSIANS or boundary_added != MASKED_BOUNDARY_GAUSSIANS:
        raise RuntimeError(
            "boundary initialization must produce exactly "
            f"{INITIAL_GAUSSIANS} rows ({MASKED_GENERAL_GAUSSIANS} general + "
            f"{MASKED_BOUNDARY_GAUSSIANS} boundary), got {field.n}/{boundary_added}"
        )
    return field, init_cfg, tensor_cfg, boundary_added


def render_field(field, target, cfg: FitConfig):
    """Render a field with the current profile's final display semantics."""

    return _render(
        field,
        cfg,
        target.shape[0],
        target.shape[1],
        support_fade_alpha=1.0,
    )


def run_current_pipeline(
    image: np.ndarray,
    *,
    mask: np.ndarray | None,
    device: str,
    seed: int,
    strategy: str = "quadtree_wse",
    mask_margin: float = 0.75,
    schedule_transform: ScheduleTransform | None = None,
    observer: PipelineObserver | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute the current profile and return its field, pixels, and audit records."""

    import torch

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB image (H,W,3), got {image.shape}")
    mask_bool = None if mask is None else np.asarray(mask, dtype=bool)
    if mask_bool is not None and mask_bool.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask_bool.shape} does not match image {image.shape[:2]}"
        )
    if mask_bool is not None and not mask_bool.any():
        raise ValueError("mask contains no foreground pixels")
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            f"{CURRENT_PROFILE_NAME} requires a CUDA device; got {device!r}"
        )
    torch.cuda.set_device(torch_device)

    boundary_enabled = mask_bool is not None
    schedule = build_current_schedule(boundary_enabled=boundary_enabled)
    if schedule_transform is not None:
        schedule = schedule_transform(schedule)
    if mask_bool is None and schedule.boundary_enabled:
        raise ValueError("an unmasked run cannot enable boundary specialization")

    started = time.perf_counter()
    init_started = time.perf_counter()
    field, init_cfg, tensor_cfg, boundary_added = _initialize(
        image,
        mask_bool,
        device=str(torch_device),
        seed=seed,
        strategy=strategy,
        boundary_enabled=schedule.boundary_enabled,
        mask_margin=mask_margin,
    )
    init_seconds = time.perf_counter() - init_started
    schedule.validate(field.n)
    target_np = (
        image
        if mask_bool is None
        else image * mask_bool[..., None].astype(np.float32)
    )
    target = torch.as_tensor(
        target_np, device=torch_device, dtype=torch.float32
    ).contiguous()
    base_cfg = build_current_fit_config(
        masked=mask_bool is not None, mask_margin=mask_margin
    )

    def observe(selected_field, record: dict[str, Any]) -> None:
        if observer is not None:
            observer(selected_field, record, target, base_cfg)

    schedule_started = time.perf_counter()
    result = run_safe_schedule(
        field,
        target,
        mask_bool,
        base_cfg,
        schedule,
        observer=observe,
        verbose=verbose,
    )
    schedule_seconds = time.perf_counter() - schedule_started
    final_field = result["field"]
    render_started = time.perf_counter()
    final_render = render_field(final_field, target, base_cfg)
    render_seconds = time.perf_counter() - render_started
    return {
        "profile": profile_manifest(masked=mask_bool is not None),
        "field": final_field,
        "target": target,
        "render": final_render,
        "mask": mask_bool,
        "initialization": {
            "config": asdict(init_cfg),
            "structure_tensor": asdict(tensor_cfg),
            "general_rows": (
                MASKED_GENERAL_GAUSSIANS
                if schedule.boundary_enabled
                else INITIAL_GAUSSIANS
            ),
            "boundary_rows": int(boundary_added),
        },
        "fit_config": asdict(base_cfg),
        "schedule": asdict(schedule),
        "schedule_result": result,
        "timing": {
            "initialization_seconds": init_seconds,
            "schedule_seconds": schedule_seconds,
            "final_render_seconds": render_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }


def profile_manifest(*, masked: bool) -> dict[str, Any]:
    """Return the reader-facing frozen profile contract."""

    schedule = build_current_schedule(boundary_enabled=masked)
    return {
        "name": CURRENT_PROFILE_NAME,
        "evidence_scope": CURRENT_PROFILE_EVIDENCE_SCOPE,
        "masked": bool(masked),
        "initial_gaussians": INITIAL_GAUSSIANS,
        "masked_general_gaussians": (
            MASKED_GENERAL_GAUSSIANS if masked else None
        ),
        "masked_boundary_gaussians": (
            MASKED_BOUNDARY_GAUSSIANS if masked else 0
        ),
        "physical_capacity": PHYSICAL_CAPACITY,
        "active_limit": ACTIVE_LIMIT,
        "requested_optimizer_steps": sum(
            phase.max_steps for phase in schedule.phases
        ),
        "boundary_contract": (
            "authoritative mask containment + boundary initialization/loss/proposals"
            if masked
            else "disabled; all initial rows and closure proposals are general"
        ),
    }
