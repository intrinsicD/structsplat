"""Pooled GaussianField lifecycle: byte-budgeted capacity, parking, free rows (FIT-021).

Pooled fitting (ADR-0020) allocates parameter tensors once at a fixed capacity and never
resizes them during the fit. Rows are alive or *parked*: a parked row's mean sits at a far
off-image sentinel, so CORE-003 tile clipping gives it an empty support rectangle — exactly
zero render contribution, zero gradient, zero tile cost, with no renderer change. Parked rows
form the free list that triage events (``triage.py``) spend on spawns/splits and refill by
parks/merges; one ``subset(live)`` compaction at the save boundary turns donors that never
found a destination into actual deletions.

Capacity is derived from a target encoded-file byte budget through the SSPL1 raw fixed-length
stream layout (ADR-0007) plus the compressed alpha payload for masked inputs and a header
allowance. zlib stream coding only shrinks the container further and is never counted on; the
final blob is asserted against the budget where it is written.

Requires torch (fit-time module; init-time math stays NumPy per invariant 1).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .codec import CodecConfig, _encode_stream, _pack_dtype, alpha_stream_raw
from .config import FitConfig
from .gaussians import GaussianField

# Far enough off-image that mean + support radius stays negative for any practical image, so
# _tile_bounds clips the support rectangle to zero area (CORE-003) and the row contributes
# nothing to render, gradient, activity, or coverage sums.
POOL_PARK_COORD = -1.0e4
POOL_PARK_LOG_SCALE = math.log(0.35)  # the fitter's log-scale clamp floor
POOL_PARK_OPACITY_LOGIT = -12.0
_STREAM_FRAME_BYTES = 4  # struct.pack("<I", len(payload)) per framed stream


@dataclass
class PoolState:
    """Fit-local pooled-lifecycle bookkeeping (deliberately not part of GaussianField)."""

    live: torch.Tensor        # (capacity,) bool
    capacity: int
    initial_live: int
    budget: dict | None       # byte breakdown when capacity came from target_file_bytes

    @property
    def live_count(self) -> int:
        return int(self.live.sum().item())

    def free_rows(self) -> torch.Tensor:
        """Free (parked) row indices in ascending order — the deterministic spend order."""
        return (~self.live).nonzero(as_tuple=False).reshape(-1)


def _bits_itemsize(bits: int) -> int:
    return int(_pack_dtype(bits).itemsize)


def gaussian_row_bytes(has_opacity: bool, ccfg: CodecConfig | None = None) -> int:
    """Raw packed SSPL1 payload bytes per Gaussian row (fixed length, before stream coding)."""
    ccfg = ccfg or CodecConfig()
    total = 2 * _bits_itemsize(ccfg.bits_means)      # x, y
    total += 2 * _bits_itemsize(ccfg.bits_scales)    # log sx, log sy
    total += _bits_itemsize(ccfg.bits_rot)
    total += 3 * _bits_itemsize(ccfg.bits_colors)
    if has_opacity:
        total += _bits_itemsize(ccfg.bits_opacity)
    return total


def alpha_payload_bytes(mask_arr, ccfg: CodecConfig | None = None) -> int:
    """Encoded alpha-stream size exactly as ``codec.encode`` will frame it (payload only)."""
    ccfg = ccfg or CodecConfig()
    return len(_encode_stream(alpha_stream_raw(mask_arr), ccfg))


def derive_capacity(cfg: FitConfig, has_opacity: bool, mask_arr=None,
                    ccfg: CodecConfig | None = None) -> tuple[int, dict]:
    """Largest live-row count whose raw-layout SSPL1 container fits ``target_file_bytes``.

    The derivation is conservative in exactly one place: ``pool_header_allowance`` must cover
    the JSON header, magic + length prefixes, and any zlib worst-case expansion of the raw
    streams. Everything else (row payload, alpha payload, stream framing) is exact.
    """
    if cfg.target_file_bytes is None:
        raise ValueError("derive_capacity requires cfg.target_file_bytes")
    ccfg = ccfg or CodecConfig()
    budget = int(cfg.target_file_bytes)
    row_bytes = gaussian_row_bytes(has_opacity, ccfg)
    n_streams = 5 if has_opacity else 4
    alpha_bytes = 0
    if mask_arr is not None:
        alpha_bytes = alpha_payload_bytes(mask_arr, ccfg)
        n_streams += 1
    overhead = (int(cfg.pool_header_allowance) + alpha_bytes
                + _STREAM_FRAME_BYTES * n_streams)
    capacity = (budget - overhead) // row_bytes
    breakdown = {
        "target_file_bytes": budget,
        "header_allowance": int(cfg.pool_header_allowance),
        "alpha_payload_bytes": alpha_bytes,
        "stream_frame_bytes": _STREAM_FRAME_BYTES * n_streams,
        "gaussian_row_bytes": row_bytes,
        "capacity": int(capacity),
    }
    if capacity <= 0:
        raise ValueError(
            f"target_file_bytes={budget} leaves no room for Gaussian rows after "
            f"{overhead} bytes of alpha/header/framing overhead")
    return int(capacity), breakdown


def prepare_pooled_field(field: GaussianField, cfg: FitConfig, H: int, W: int,
                         mask_arr=None, ccfg: CodecConfig | None = None
                         ) -> tuple[GaussianField, PoolState]:
    """Expand a freshly initialized field to pool capacity with parked spare rows.

    Must run before ``trainable()`` / optimizer creation: it may create the opacity tensor and
    it resizes the field once — the last resize of the whole fit.
    """
    device = field.means.device
    dtype = field.means.dtype
    if field.opacities is None:
        # Pooled activation/teleport needs per-row opacity for low-alpha function-preserving
        # births; sigmoid(10) ~ 0.99995 preserves the opacity-free appearance (FIT-002).
        field.opacities = torch.full((field.n,), 10.0, device=device, dtype=dtype)
    if field.color_grads is not None:
        raise ValueError("pooled fitting supports color_basis='constant' only (FIT-021)")
    budget = None
    if cfg.pool_capacity is not None:
        capacity = int(cfg.pool_capacity)
    else:
        alpha = None if mask_arr is None else np.asarray(mask_arr, dtype=np.float32)
        capacity, budget = derive_capacity(cfg, True, alpha, ccfg)
    n0 = field.n
    if capacity < n0:
        raise ValueError(
            f"pool capacity {capacity} is smaller than the initial field ({n0} rows); "
            "raise target_file_bytes/pool_capacity or lower the init num_gaussians")
    extra = capacity - n0
    if extra > 0:
        parked = GaussianField(
            torch.full((extra, 2), POOL_PARK_COORD, device=device, dtype=dtype),
            torch.full((extra, 2), POOL_PARK_LOG_SCALE, device=device, dtype=dtype),
            torch.zeros(extra, device=device, dtype=dtype),
            torch.zeros(extra, 3, device=device, dtype=dtype),
            torch.full((extra,), POOL_PARK_OPACITY_LOGIT, device=device, dtype=dtype),
        )
        field = field.append(parked)
    live = torch.zeros(field.n, device=device, dtype=torch.bool)
    live[:n0] = True
    return field, PoolState(live=live, capacity=int(capacity), initial_live=int(n0),
                            budget=budget)


@torch.no_grad()
def park_rows(field: GaussianField, rows: torch.Tensor) -> None:
    """Park rows in place at the off-image sentinel (exact-zero contribution and gradient)."""
    if rows.numel() == 0:
        return
    field.means[rows] = POOL_PARK_COORD
    field.log_scales[rows] = POOL_PARK_LOG_SCALE
    field.rotations[rows] = 0.0
    field.colors[rows] = 0.0
    if field.opacities is not None:
        field.opacities[rows] = POOL_PARK_OPACITY_LOGIT
    if field.color_grads is not None:
        field.color_grads[rows] = 0.0
    if field.filter_variance is not None:
        field.filter_variance[rows] = 0.0
