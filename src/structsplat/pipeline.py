"""The single entrypoint for converting an image into a Gaussian field (ADR-0025).

``run_pipeline`` is the one place that answers "what is StructSplat's current best pipeline?".
Everything else in the package is a mechanism it composes: ``config`` holds the *conservative*
library defaults (ADR-0009/0013), while this module holds the *measured best-known recipe*, which
is deliberately not the same thing. A knob is set here when evidence says it wins even though the
library default stays off pending broader confirmation.

Two arms, one schedule
----------------------
``mask`` selects the arm and nothing else does:

* **masked** (``mask`` given) — the dome/alpha-matted case. Mask containment, boundary-tangent
  initialization, and the boundary-closure phase are active.
* **full frame** (``mask=None``) — the ordinary-image case. The containment machinery degenerates
  rather than being replaced (see :func:`safe_schedule.run_safe_schedule`), and boundary closure is
  skipped. Every other phase, the Pareto commit gate, and the metric vector are identical.

Updating the recipe when a new approach wins
--------------------------------------------
Edit :data:`RECIPE` and the defaults on :class:`PipelineConfig` in the same commit, bump
``RECIPE["version"]``, and cite the claim that authorizes the change in ``evidence``. That keeps
"the best pipeline" a single reviewable diff instead of an inference across README prose, task
status lines, and benchmark constants. A change here is a results-bearing change: it needs the
``structsplat-results-audit`` pass and an ``ara/logic/claims.md`` row like any other promotion.

Evidence status
---------------
The schedule's quality evidence (C50/C51/C52) is **one masked image, one seed** on an RTX 4090.
This module ships it as the recommended path because it is the best measured pipeline in the
repository, not because it has multi-image confirmation. The full-frame arm is a mechanism
extension of that schedule and has **no** benchmark screen of its own yet; ``BENCH-017`` is the
task that would give it one. Neither arm's defaults are promoted to ``FitConfig``/``InitConfig``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as _field, replace
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import FitConfig, InitConfig, StructureTensorConfig

# ``safe_schedule`` pulls in torch, so it is imported inside the functions that need it: importing
# ``PipelineConfig`` (which the CLI parser does for its defaults) must stay torch-free.
if TYPE_CHECKING:
    from .safe_schedule import PhaseBudget, SafeScheduleConfig

RECIPE: dict[str, Any] = {
    "name": "safe-commit-schedule",
    "version": "2026-07-25.1",
    "summary": (
        "Phase-ordered transactional fit (bootstrap / coverage / detail / boundary / "
        "redistribution / polish) with a Pareto-safe commit gate on a full-frame metric vector."
    ),
    "choices": {
        "init_strategy": "quadtree_wse (ADR-0013, ABL-006 / C05)",
        "wse_progressive_order": "on (INIT-009 / C25: 32/32 uniform prefix wins, terminal set unchanged)",
        "pareto_safe_checkpoints": "on every 50 steps (FIT-023 / C50: +0.502/+0.519 dB fg/boundary)",
        "event_color_solve": "off (FIT-023 / C50: worse on every protected metric, 7.8% slower)",
        "detail_tail_max_rows": "0 (FIT-025 / C52: generic activation beats the specialized tail)",
        "storage_policy": "dynamic (FIT-024 / C51: fixed capacity is quality-neutral, not faster)",
        "refinement_policy": "global (README: the reproducible baseline; local is experimental)",
        "renderer": "cuda where available (ADR-0011 / C03; C53 keeps the tiled path opt-in)",
    },
    "evidence": ["C25", "C50", "C51", "C52"],
    "evidence_scope": (
        "single masked image, single seed, one GPU; the full-frame arm is unscreened (BENCH-017)"
    ),
}

# Phase capacities as fractions of the row budget, from the Janelle schedule this generalizes
# (5,000 / 8,000 / 10,000 / 11,000 rows). Overriding `capacity` alone rescales the whole schedule.
_INITIAL_FRACTION = 5.0 / 11.0
_COVERAGE_FRACTION = 8.0 / 11.0
_DETAIL_FRACTION = 10.0 / 11.0
# Share of the initial rows placed on the mask boundary as tangent-aligned seeds (500 of 5,000).
_BOUNDARY_INIT_FRACTION = 0.10


@dataclass
class PipelineConfig:
    """User-facing knobs. Defaults are the recipe; everything here is an override, not a policy."""

    capacity: int = 11_000
    initial_gaussians: int | None = None      # default: capacity * 5/11
    boundary_gaussians: int | None = None     # masked arm only; default: 10% of the initial rows
    coverage_target: int | None = None        # default: capacity * 8/11
    detail_target: int | None = None          # default: capacity * 10/11
    seed: int = 0
    device: str | None = None                 # default: cuda when available, else cpu
    renderer: str | None = None               # default: cuda when on a CUDA device, else normalized
    step_scale: float = 1.0                   # scales every phase's step ceiling
    block_steps: int | None = None            # commit-gate granularity; default: schedule default
    storage_policy: str = "dynamic"
    pareto_safe_checkpoints: bool = True
    pareto_checkpoint_every: int = 50
    event_color_solve: bool = False
    mask_margin: float = 1.5
    boundary_band: float = 4.0
    coverage_tau: float = 0.05
    schedule_overrides: dict[str, Any] = _field(default_factory=dict)

    def resolved_initial(self) -> int:
        if self.initial_gaussians is not None:
            return int(self.initial_gaussians)
        return max(1, int(round(self.capacity * _INITIAL_FRACTION)))

    def resolved_boundary(self) -> int:
        if self.boundary_gaussians is not None:
            return int(self.boundary_gaussians)
        return int(round(self.resolved_initial() * _BOUNDARY_INIT_FRACTION))

    def resolved_coverage_target(self) -> int:
        if self.coverage_target is not None:
            return int(self.coverage_target)
        return max(
            self.resolved_initial(),
            int(round(self.capacity * _COVERAGE_FRACTION)),
        )

    def resolved_detail_target(self) -> int:
        if self.detail_target is not None:
            return int(self.detail_target)
        return max(
            self.resolved_coverage_target(),
            int(round(self.capacity * _DETAIL_FRACTION)),
        )

    def validate(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        initial = self.resolved_initial()
        if initial > self.capacity:
            raise ValueError(
                f"initial_gaussians {initial} exceeds capacity {self.capacity}"
            )
        boundary = self.resolved_boundary()
        if boundary < 0:
            raise ValueError(f"boundary_gaussians must be nonnegative, got {boundary}")
        if boundary >= initial:
            raise ValueError(
                f"boundary_gaussians {boundary} must leave room inside the initial "
                f"{initial} rows"
            )
        if not math.isfinite(self.step_scale) or self.step_scale <= 0.0:
            raise ValueError(f"step_scale must be finite and positive, got {self.step_scale}")
        if self.block_steps is not None and self.block_steps <= 0:
            raise ValueError(f"block_steps must be positive, got {self.block_steps}")
        if not (
            initial
            <= self.resolved_coverage_target()
            <= self.resolved_detail_target()
            <= self.capacity
        ):
            raise ValueError(
                "expected initial <= coverage_target <= detail_target <= capacity, got "
                f"{initial} / {self.resolved_coverage_target()} / "
                f"{self.resolved_detail_target()} / {self.capacity}"
            )


def _resolve_device(requested: str | None) -> str:
    if requested is not None:
        return str(requested)
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_renderer(requested: str | None, device: str) -> str:
    """Default to the shipped GPU renderer (ADR-0011); C53 keeps `cuda_tiled` an explicit opt-in."""
    if requested is not None:
        return str(requested)
    return "cuda" if str(device).startswith("cuda") else "normalized"


def _scaled_phase(phase: "PhaseBudget", cfg: PipelineConfig, target: int | None) -> "PhaseBudget":
    max_steps = max(1, int(round(phase.max_steps * cfg.step_scale)))
    block = phase.block_steps if cfg.block_steps is None else int(cfg.block_steps)
    return replace(
        phase,
        max_steps=max_steps,
        block_steps=max(1, min(int(block), max_steps)),
        target_gaussians=target,
    )


def build_schedule(cfg: PipelineConfig) -> "SafeScheduleConfig":
    """Translate the user-facing config into the schedule policy the recipe pins."""
    from .safe_schedule import SafeScheduleConfig

    cfg.validate()
    defaults = SafeScheduleConfig()
    coverage_target = cfg.resolved_coverage_target()
    detail_target = cfg.resolved_detail_target()
    schedule = SafeScheduleConfig(
        capacity=int(cfg.capacity),
        storage_policy=str(cfg.storage_policy),
        coverage_target_gaussians=coverage_target,
        detail_target_gaussians=detail_target,
        coverage_tau=float(cfg.coverage_tau),
        boundary_band=float(cfg.boundary_band),
        pareto_safe_checkpoints=bool(cfg.pareto_safe_checkpoints),
        pareto_checkpoint_every=int(cfg.pareto_checkpoint_every),
        event_color_solve=bool(cfg.event_color_solve),
        bootstrap=_scaled_phase(defaults.bootstrap, cfg, cfg.resolved_initial()),
        coverage=_scaled_phase(defaults.coverage, cfg, coverage_target),
        detail=_scaled_phase(defaults.detail, cfg, detail_target),
        boundary=_scaled_phase(defaults.boundary, cfg, int(cfg.capacity)),
        redistribution=_scaled_phase(defaults.redistribution, cfg, int(cfg.capacity)),
        polish=_scaled_phase(defaults.polish, cfg, int(cfg.capacity)),
    )
    if cfg.schedule_overrides:
        unknown = set(cfg.schedule_overrides) - set(asdict(schedule))
        if unknown:
            raise ValueError(
                f"unknown schedule override(s): {', '.join(sorted(unknown))}"
            )
        schedule = replace(schedule, **cfg.schedule_overrides)
    return schedule


def build_fit_config(cfg: PipelineConfig, device: str) -> FitConfig:
    """The base fit config; ``run_safe_schedule`` owns the loss/mask fields it must control."""

    return FitConfig(
        iters=1,
        renderer=_resolve_renderer(cfg.renderer, device),
        render_chunk=512,
        pixel_loss="l2",
        ssim_weight=0.0,
        mask_margin=float(cfg.mask_margin),
        mask_cap_refresh_every=100,
        support_fade=True,
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


def build_init_config(cfg: PipelineConfig, count: int) -> InitConfig:
    """The measured-best initialization (ADR-0013 strategy + INIT-009 ordering)."""

    return InitConfig(
        strategy="quadtree_wse",
        num_gaussians=int(count),
        seed=int(cfg.seed),
        sampling_mode="wse",
        wse_progressive_order=True,
        init_scale_mult=0.35,
        scale_cap_mode="none",
        background_fraction=0.0,
        background_grid=0,
    )


def _initial_masked_field(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: PipelineConfig,
    fit_cfg: FitConfig,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Interior quadtree-WSE rows plus tangent-aligned boundary rows (CORE-011)."""
    import torch

    from .fit import _MaskConstraint, _boundary_tangent_add, _render
    from .init import build_masked_field

    total = cfg.resolved_initial()
    boundary_count = cfg.resolved_boundary()
    interior_count = total - boundary_count
    init_cfg = build_init_config(cfg, total)
    pool = build_masked_field(
        image,
        mask,
        init_cfg,
        StructureTensorConfig(),
        device=device,
        sigma_cutoff=fit_cfg.sigma_cutoff,
        mask_margin=float(cfg.mask_margin),
        contain=False,
    )
    field = pool.subset(slice(0, min(interior_count, pool.n)))
    add_cfg = replace(
        fit_cfg,
        loss_weighting="mask",
        mask_contain=True,
        mask_cap_mode="anisotropic",
        mask_boundary_add_every=1,
        mask_boundary_add_count=boundary_count,
        mask_boundary_add_band=float(cfg.boundary_band),
        mask_boundary_add_spacing=3.0,
        max_gaussians=total,
        split_color_init="target",
    )
    constraint = _MaskConstraint.from_mask(
        mask,
        device,
        torch.float32,
        add_cfg.sigma_cutoff,
        float(cfg.mask_margin),
        aa_dilation=add_cfg.aa_dilation,
        cap_mode="anisotropic",
    )
    constraint.apply(field, add_cfg, refresh=True)
    added = 0
    if boundary_count > 0:
        target = torch.as_tensor(image, device=device, dtype=torch.float32)
        H, W = image.shape[:2]
        with torch.no_grad():
            rendered = _render(field, add_cfg, H, W)
            field, added = _boundary_tangent_add(
                field, target, rendered, add_cfg, constraint
            )
            constraint.apply(field, add_cfg, refresh=True)
    record = {
        "strategy": init_cfg.strategy,
        "requested": total,
        "interior_rows": int(field.n) - int(added),
        "boundary_rows": int(added),
        "boundary_requested": boundary_count,
        "n": int(field.n),
    }
    return field, record


def _initial_full_frame_field(
    image: np.ndarray,
    cfg: PipelineConfig,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """The same initialization without the mask-specific boundary allocation."""
    from .init import build_field

    total = cfg.resolved_initial()
    init_cfg = build_init_config(cfg, total)
    field = build_field(image, init_cfg, StructureTensorConfig(), device=device)
    record = {
        "strategy": init_cfg.strategy,
        "requested": total,
        "interior_rows": int(field.n),
        "boundary_rows": 0,
        "boundary_requested": 0,
        "n": int(field.n),
    }
    return field, record


def run_pipeline(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: PipelineConfig | None = None,
    *,
    observer=None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Convert one image into a Gaussian field with the current best pipeline.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` float32 in ``[0, 1]`` (core invariant 2).
    mask:
        ``(H, W)`` in ``[0, 1]`` or bool selects the masked arm; ``None`` runs the full-frame arm.
    cfg:
        :class:`PipelineConfig` overrides. ``None`` uses the recipe as shipped.

    Returns the :func:`safe_schedule.run_safe_schedule` payload plus ``recipe``, ``arm``, ``init``,
    and ``device`` provenance, so a run is self-describing from its own output.
    """
    import torch

    from .safe_schedule import run_safe_schedule

    cfg = PipelineConfig() if cfg is None else cfg
    cfg.validate()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    device = _resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))

    mask_bool: np.ndarray | None = None
    if mask is not None:
        from . import mask as _mask

        mask_bool = _mask.as_bool_mask(np.asarray(mask))
        if mask_bool.shape != image.shape[:2]:
            raise ValueError(
                f"mask shape {mask_bool.shape} does not match image {image.shape[:2]}"
            )
        if not mask_bool.any():
            raise ValueError("mask selects no pixels; pass mask=None for a full-frame fit")

    fit_cfg = build_fit_config(cfg, device)
    schedule = build_schedule(cfg)
    if mask_bool is None:
        field, init_record = _initial_full_frame_field(image, cfg, device)
    else:
        field, init_record = _initial_masked_field(image, mask_bool, cfg, fit_cfg, device)

    target = torch.as_tensor(image, device=device, dtype=torch.float32)
    result = run_safe_schedule(
        field,
        target,
        mask_bool,
        fit_cfg,
        schedule,
        observer=observer,
        verbose=verbose,
    )
    result["recipe"] = dict(RECIPE)
    result["init"] = init_record
    result["device"] = device
    result["pipeline_config"] = asdict(cfg)
    return result
