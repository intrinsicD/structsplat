"""Error-triage topology events over a pooled GaussianField (FIT-021, ADR-0020).

One event runs a shared responsibility/attribution pass over the field, then applies
park -> merge -> split -> spawn with per-op budgets, entirely in place on fixed-capacity
tensors:

- **park**: function-preserving donor marking, gated on the row's *maximum* normalized
  responsibility (summed activity cannot see that a low-mass row is the sole owner of some
  pixels — under the normalized compositor such a row fully determines those pixels);
- **merge**: prototype-parity envelope merge of mutual-nearest redundant pairs
  (``deprecated_scripts/fit_janelle_complete_refinement.py``): exact midpoint mean, smallest
  generalized-eigen covariance envelope of both translated inputs with explicit inflation,
  amplitude-weighted color, union opacity; under ``mask_contain`` pairs whose certified caps
  cannot preserve the envelope are batch-restored. The absorbed partner is parked;
- **split**: major-axis split of the highest attributed-error live owners into free rows;
- **spawn**: activation of free rows at masked residual peaks (max-pool NMS), with
  hole-vs-covered opacity init from the shared per-pixel weight sum and tensor-aligned
  anisotropy from local target gradients.

Parks and merges refill the free list; splits and spawns consume it, so ``live_n <= capacity``
is an invariant, not a schedule. Parked rows have empty support tiles, so the attribution pass
assigns them exactly zero mass/error at zero cost — no special-casing in the hot loops.

Requires torch.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .config import FitConfig
from .gaussians import GaussianField
from .pool import PoolState, park_rows
from .render import (
    _element_budget,
    _flat_tile_slices,
    _support_weight,
    _tile_bounds,
    _tile_coords,
)
from .fit import (
    _EPS,
    _MIN_DENSIFY_SCALE,
    _image_gradients,
    _logit,
    _major_axis,
)


@dataclass
class TriageStats:
    parked: int = 0
    merged: int = 0
    merge_candidates: int = 0
    merge_containment_rejected: int = 0
    split: int = 0
    spawned: int = 0
    spawned_holes: int = 0
    touched: torch.Tensor | None = None


_SNAPSHOT_ATTRS = ("means", "log_scales", "rotations", "colors", "opacities",
                   "color_grads", "filter_variance")


@torch.no_grad()
def attribution_pass(field: GaussianField, target: torch.Tensor, render_img: torch.Tensor,
                     cfg: FitConfig, support_fade_alpha: float | None = None
                     ) -> dict[str, torch.Tensor]:
    """Shared per-pixel weight sum and per-row responsibility statistics.

    Mirrors the reference renderer's clipped support tiles, tail fade, opacity, and AA
    dilation exactly (same discipline as FIT-018's responsibility scores). Returns:
    ``denominator`` (H, W) raw opacity-weighted weight sum; per-row ``mass`` (summed
    responsibility), ``error`` (responsibility-weighted squared pixel error),
    ``max_responsibility``, and ``activity`` (opacity-weighted raw weight sum).
    """
    H, W = target.shape[:2]
    dev, dt = field.means.device, field.means.dtype
    means = field.means.detach()
    conics = field.conics(cfg.aa_dilation).detach()
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    if opacities is not None:
        opacities = opacities.detach().to(device=dev, dtype=dt)
    x0, y0, Tx, n = _tile_bounds(means, radii, H, W)
    budget = _element_budget(cfg.render_chunk)

    denominator = torch.zeros(H * W, device=dev, dtype=dt)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha)
        if opacities is not None:
            w = w * opacities[gid]
        denominator.index_add_(0, py * W + px, w)

    pixel_error = (render_img.detach() - target.detach()).square().mean(dim=2)
    pixel_error = pixel_error.reshape(-1).to(device=dev, dtype=dt)
    mass = torch.zeros(field.n, device=dev, dtype=dt)
    error = torch.zeros(field.n, device=dev, dtype=dt)
    max_resp = torch.zeros(field.n, device=dev, dtype=dt)
    activity = torch.zeros(field.n, device=dev, dtype=dt)
    for s, e in _flat_tile_slices(n, budget):
        gid, px, py = _tile_coords(x0, y0, Tx, n, s, e, dev)
        dx = px.to(dt) - means[gid, 0]
        dy = py.to(dt) - means[gid, 1]
        a, b, c = conics[gid, 0], conics[gid, 1], conics[gid, 2]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        w = _support_weight(q, cfg.sigma_cutoff, cfg.support_fade, support_fade_alpha)
        if opacities is not None:
            w = w * opacities[gid]
        flat = py * W + px
        resp = w / (denominator[flat] + _EPS)
        mass.index_add_(0, gid, resp)
        error.index_add_(0, gid, resp * pixel_error[flat])
        activity.index_add_(0, gid, w)
        max_resp.scatter_reduce_(0, gid, resp, reduce="amax", include_self=True)
    return {
        "denominator": denominator.view(H, W),
        "mass": mass,
        "error": error,
        "max_responsibility": max_resp,
        "activity": activity,
    }


def _detail_live_mask(field: GaussianField, live: torch.Tensor) -> torch.Tensor:
    m = live.clone()
    if field.background_mask is not None:
        m &= ~field.background_mask.to(device=live.device)
    return m


def _row_covariances(field: GaussianField, idx: torch.Tensor) -> torch.Tensor:
    """(k, 2, 2) covariance of the base RS scales (prototype parity: no filter variance)."""
    scales = field.scales().detach()[idx]
    theta = field.rotations.detach()[idx]
    c, s = torch.cos(theta), torch.sin(theta)
    sx2, sy2 = scales[:, 0].square(), scales[:, 1].square()
    xx = c * c * sx2 + s * s * sy2
    xy = c * s * (sx2 - sy2)
    yy = s * s * sx2 + c * c * sy2
    return torch.stack(
        [torch.stack([xx, xy], dim=-1), torch.stack([xy, yy], dim=-1)], dim=-2
    )


def _mutual_nearest_pairs(means: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Disjoint mutual-nearest-neighbour pairs (local indices) with their distances.

    Deliberately a chunked exact O(M^2) scan: it runs only at the event cadence over live rows,
    stays dependency-free (the prototype's cKDTree would add scipy), and chunking bounds the
    transient distance matrix. A grid-hash neighborhood search is the FIT-021 phase-2 upgrade
    if event cost ever shows up at production capacities.
    """
    M = means.shape[0]
    dev = means.device
    nn = torch.empty(M, dtype=torch.long, device=dev)
    nnd = torch.empty(M, dtype=means.dtype, device=dev)
    chunk = 2048
    for s0 in range(0, M, chunk):
        e0 = min(s0 + chunk, M)
        d = torch.cdist(means[s0:e0], means)
        d[torch.arange(e0 - s0, device=dev), torch.arange(s0, e0, device=dev)] = float("inf")
        vals, idx = d.min(dim=1)
        nn[s0:e0] = idx
        nnd[s0:e0] = vals
    ar = torch.arange(M, device=dev)
    mutual = nn[nn] == ar
    first = ar[mutual & (ar < nn)]
    return first, nn[first], nnd[first]


def _snapshot_rows(field: GaussianField, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    snap = {}
    for name in _SNAPSHOT_ATTRS:
        value = getattr(field, name)
        if value is not None:
            snap[name] = value.detach()[rows].clone()
    return snap


def _restore_rows(field: GaussianField, rows: torch.Tensor,
                  snap: dict[str, torch.Tensor], select: torch.Tensor) -> None:
    take = rows[select]
    for name, value in snap.items():
        getattr(field, name)[take] = value[select]


@torch.no_grad()
def _park_donors(field: GaussianField, pool: PoolState, stats: dict[str, torch.Tensor],
                 cfg: FitConfig) -> torch.Tensor:
    empty = torch.zeros(0, dtype=torch.long, device=field.means.device)
    if cfg.triage_park_count <= 0:
        return empty
    detail = _detail_live_mask(field, pool.live)
    room = int(detail.sum()) - max(int(cfg.prune_keep_min), 1)
    if room <= 0:
        return empty
    eligible = detail & (
        stats["max_responsibility"] < float(cfg.triage_park_max_responsibility)
    )
    idx = eligible.nonzero(as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return empty
    k = min(int(cfg.triage_park_count), int(idx.numel()), room)
    take = idx[torch.topk(-stats["activity"][idx], k=k).indices]
    park_rows(field, take)
    pool.live[take] = False
    return take


@torch.no_grad()
def _merge_pairs(field: GaussianField, pool: PoolState, cfg: FitConfig,
                 mask_constraint) -> tuple[torch.Tensor, dict[str, int]]:
    dev = field.means.device
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    out = {"merged": 0, "candidates": 0, "containment_rejected": 0}
    if cfg.triage_merge_count <= 0:
        return empty, out
    detail = _detail_live_mask(field, pool.live)
    live_idx = detail.nonzero(as_tuple=False).reshape(-1)
    if live_idx.numel() < 4:
        return empty, out
    means = field.means.detach()
    first_l, second_l, distance = _mutual_nearest_pairs(means[live_idx])
    if first_l.numel() == 0:
        return empty, out
    first = live_idx[first_l]
    second = live_idx[second_l]

    colors = field.colors.detach()
    color_delta = torch.linalg.norm(colors[first] - colors[second], dim=1)
    sorted_scales = torch.sort(field.scales().detach(), dim=1).values.clamp_min(1e-6)
    scale_delta = torch.linalg.norm(
        torch.log(sorted_scales[first]) - torch.log(sorted_scales[second]), dim=1
    )
    angle_delta = field.rotations.detach()[first] - field.rotations.detach()[second]
    axial_delta = 0.5 * torch.abs(
        torch.atan2(torch.sin(2 * angle_delta), torch.cos(2 * angle_delta))
    )
    minor = 0.5 * (sorted_scales[first, 0] + sorted_scales[second, 0])
    allowed = torch.clamp(
        float(cfg.triage_merge_distance_factor) * minor,
        min=float(cfg.triage_merge_distance_min),
        max=float(cfg.triage_merge_distance_max),
    )
    valid = (
        (distance <= allowed)
        & (color_delta <= float(cfg.triage_merge_color_max))
        & (scale_delta <= float(cfg.triage_merge_log_scale_max))
        & (axial_delta <= float(cfg.triage_merge_angle_max))
    )
    midpoint = 0.5 * (means[first] + means[second])
    if mask_constraint is not None and cfg.mask_contain:
        W = mask_constraint.W
        mx = torch.round(midpoint[:, 0]).long().clamp(0, W - 1)
        my = torch.round(midpoint[:, 1]).long().clamp(0, mask_constraint.H - 1)
        valid &= mask_constraint.eroded_flat[my * W + mx]
    out["candidates"] = int(valid.sum())
    k = min(int(cfg.triage_merge_count), out["candidates"])
    if k <= 0:
        return empty, out
    score = (
        distance / allowed.clamp_min(1e-6)
        + 2.0 * color_delta
        + 0.25 * scale_delta
        + 0.15 * axial_delta
    )
    vidx = valid.nonzero(as_tuple=False).reshape(-1)
    sel = vidx[torch.argsort(score[vidx])[:k]]
    keep = first[sel]
    part = second[sel]

    # All merged values are computed from detached reads before any write; keep/part rows are
    # disjoint by mutual-NN construction, so ordering within the batch cannot alias.
    amp = torch.sigmoid(field.opacities.detach())
    wa, wb = amp[keep], amp[part]
    total = (wa + wb).clamp_min(1e-6)
    merged_mean = 0.5 * (means[keep] + means[part])
    cov_a = _row_covariances(field, keep)
    cov_b = _row_covariances(field, part)
    da = means[keep] - merged_mean
    db = means[part] - merged_mean
    spatial_a = cov_a + da[:, :, None] * da[:, None, :]
    spatial_b = cov_b + db[:, :, None] * db[:, None, :]
    moment = 0.5 * (spatial_a + spatial_b)
    # Smallest scaled pair moment whose ellipse contains both translated inputs, then the
    # explicit inflation — the prototype's vectorized generalized-eigen envelope.
    mvals, mvecs = torch.linalg.eigh(moment)
    inv_sqrt = (
        mvecs @ torch.diag_embed(mvals.clamp_min(1e-8).rsqrt()) @ mvecs.transpose(1, 2)
    )
    envelope = torch.maximum(
        torch.linalg.eigvalsh(inv_sqrt @ spatial_a @ inv_sqrt)[:, -1],
        torch.linalg.eigvalsh(inv_sqrt @ spatial_b @ inv_sqrt)[:, -1],
    ).clamp_min(1.0)
    inflation = float(cfg.triage_merge_envelope_inflation) ** 2
    merged_cov = moment * envelope[:, None, None] * inflation
    evals, evecs = torch.linalg.eigh(merged_cov)
    evals = evals.clamp_min(_MIN_DENSIFY_SCALE ** 2)
    major = evecs[:, :, 1]
    merged_log_scales = torch.log(
        torch.stack([torch.sqrt(evals[:, 1]), torch.sqrt(evals[:, 0])], dim=1)
    )
    merged_rot = torch.atan2(major[:, 1], major[:, 0])
    merged_color = (wa[:, None] * colors[keep] + wb[:, None] * colors[part]) / total[:, None]
    combined = (1.0 - (1.0 - wa) * (1.0 - wb)).clamp(1e-4, 1.0 - 1e-4)

    snap_rows = torch.cat([keep, part])
    snap = _snapshot_rows(field, snap_rows)

    field.means[keep] = merged_mean
    field.log_scales[keep] = merged_log_scales
    field.rotations[keep] = merged_rot
    field.colors[keep] = merged_color
    field.opacities[keep] = torch.log(combined / (1.0 - combined))
    if field.color_grads is not None:
        grads = field.color_grads.detach()
        field.color_grads[keep] = (
            wa[:, None, None] * grads[keep] + wb[:, None, None] * grads[part]
        ) / total[:, None, None]
    if field.filter_variance is not None:
        fv = field.filter_variance.detach()
        field.filter_variance[keep] = (wa * fv[keep] + wb * fv[part]) / total
    park_rows(field, part)
    pool.live[part] = False

    if mask_constraint is not None and cfg.mask_contain:
        # Containment is non-negotiable (prototype rule): recertify caps, then reject as a
        # batch every pair whose certified cap would undo the envelope, restoring both rows.
        mask_constraint.apply(field, cfg, exempt=~pool.live)
        cert = _row_covariances(field, keep)
        cvals, cvecs = torch.linalg.eigh(cert)
        cinv = (
            cvecs @ torch.diag_embed(cvals.clamp_min(1e-8).rsqrt()) @ cvecs.transpose(1, 2)
        )
        ratio = torch.maximum(
            torch.linalg.eigvalsh(cinv @ spatial_a @ cinv)[:, -1],
            torch.linalg.eigvalsh(cinv @ spatial_b @ cinv)[:, -1],
        )
        bad = ratio > 1.0 + 1e-4
        if bool(bad.any()):
            select = torch.cat([bad, bad])
            _restore_rows(field, snap_rows, snap, select)
            pool.live[part[bad]] = True
            out["containment_rejected"] = int(bad.sum())
            keep = keep[~bad]
            part = part[~bad]
            # Caps were certified against the now-reverted geometry; refresh once more so the
            # restored rows leave the event with caps matching their actual parameters.
            mask_constraint.apply(field, cfg, exempt=~pool.live)
    out["merged"] = int(keep.numel())
    return torch.cat([keep, part]), out


@torch.no_grad()
def _split_owners(field: GaussianField, pool: PoolState, stats: dict[str, torch.Tensor],
                  cfg: FitConfig, free: torch.Tensor, k_budget: int
                  ) -> tuple[torch.Tensor, int]:
    empty = torch.zeros(0, dtype=torch.long, device=field.means.device)
    if k_budget <= 0 or free.numel() == 0:
        return empty, 0
    cand = _detail_live_mask(field, pool.live) & (stats["mass"] > 0)
    idx = cand.nonzero(as_tuple=False).reshape(-1)
    if idx.numel() == 0:
        return empty, 0
    k = min(int(k_budget), int(idx.numel()), int(free.numel()))
    parents = idx[torch.topk(stats["error"][idx], k=k).indices]
    children = free[:k]

    scales_p = field.scales().detach()[parents]
    direction, s_major = _major_axis(scales_p, field.rotations.detach()[parents])
    offset = 0.5 * s_major[:, None] * direction
    parent_means = field.means.detach()[parents]
    shrink = float(cfg.triage_split_shrink)
    use_x = scales_p[:, 0] >= scales_p[:, 1]
    new_scales = scales_p.clone()
    new_scales[use_x, 0] *= shrink
    new_scales[~use_x, 1] *= shrink
    new_log_scales = torch.log(new_scales.clamp_min(_MIN_DENSIFY_SCALE))

    field.means[children] = parent_means + offset
    field.means[parents] = parent_means - offset
    field.log_scales[parents] = new_log_scales
    field.log_scales[children] = new_log_scales
    field.rotations[children] = field.rotations.detach()[parents]
    field.colors[children] = field.colors.detach()[parents]
    field.opacities[children] = field.opacities.detach()[parents]
    if field.color_grads is not None:
        field.color_grads[children] = field.color_grads.detach()[parents]
    if field.filter_variance is not None:
        field.filter_variance[children] = field.filter_variance.detach()[parents]
    if field.scale_max is not None:
        field.scale_max[children] = field.scale_max.detach()[parents]
    pool.live[children] = True
    return torch.cat([parents, children]), k


@torch.no_grad()
def _activate_sites(field: GaussianField, pool: PoolState, stats: dict[str, torch.Tensor],
                    cfg: FitConfig, target: torch.Tensor, render_img: torch.Tensor,
                    mask_constraint, free: torch.Tensor, k_budget: int
                    ) -> tuple[torch.Tensor, int, int]:
    empty = torch.zeros(0, dtype=torch.long, device=field.means.device)
    if k_budget <= 0 or free.numel() == 0:
        return empty, 0, 0
    H, W = target.shape[:2]
    dt = target.dtype
    residual = (render_img.detach() - target.detach()).abs().mean(dim=2)
    score = residual.reshape(-1)
    neg = torch.full_like(score, -float("inf"))
    if mask_constraint is not None:
        allowed = (
            mask_constraint.eroded_flat if cfg.mask_contain
            else mask_constraint.inside.reshape(-1)
        )
        score = torch.where(allowed, score, neg)
    r = max(int(cfg.triage_site_nms_radius), 0)
    if r > 0:
        pooled = F.max_pool2d(
            score.view(1, 1, H, W), kernel_size=2 * r + 1, stride=1, padding=r
        )[0, 0].reshape(-1)
        peaks = torch.where(score >= pooled, score, neg)
    else:
        peaks = score
    finite = int(torch.isfinite(peaks).sum())
    k = min(int(k_budget), int(free.numel()), finite)
    if k <= 0:
        return empty, 0, 0
    sites = torch.topk(peaks, k=k).indices
    site_y = torch.div(sites, W, rounding_mode="floor")
    site_x = sites - site_y * W
    hole = stats["denominator"].reshape(-1)[sites] < float(cfg.triage_hole_coverage_tau)

    gx, gy = _image_gradients(target)
    grad = torch.sqrt(gx[site_y, site_x].square() + gy[site_y, site_x].square())
    ref = torch.quantile(torch.sqrt(gx.square() + gy.square()).reshape(-1), 0.95)
    coherence = torch.clamp(grad / ref.clamp_min(1e-6), 0.0, 1.0)
    ratio = 1.0 + (cfg.densify_max_axis_ratio - 1.0) * coherence ** cfg.densify_coherence_power
    base = max(
        math.sqrt((H * W) / max(pool.live_count, 1)) * cfg.split_scale, _MIN_DENSIFY_SCALE
    )
    along = base * torch.sqrt(ratio)
    across = base / torch.sqrt(ratio)

    rows = free[:k]
    field.means[rows] = torch.stack([site_x.to(dt), site_y.to(dt)], dim=1)
    field.log_scales[rows] = torch.log(
        torch.stack([along, across], dim=1).clamp_min(_MIN_DENSIFY_SCALE)
    )
    field.rotations[rows] = torch.atan2(gy[site_y, site_x], gx[site_y, site_x]) + math.pi * 0.5
    field.colors[rows] = target[site_y, site_x]
    opacity = torch.where(
        hole,
        torch.full((k,), float(cfg.triage_hole_opacity), device=hole.device, dtype=dt),
        torch.full((k,), float(cfg.relocate_init_opacity), device=hole.device, dtype=dt),
    )
    field.opacities[rows] = _logit(opacity)
    if field.color_grads is not None:
        field.color_grads[rows] = 0.0
    if field.filter_variance is not None and cfg.covariance_filter_mode == "generation_density":
        from .fit import _generation_density_filter_variance
        field.filter_variance[rows] = _generation_density_filter_variance(
            cfg, H, W, pool.live_count + k
        )
    if field.scale_max is not None:
        # A row activated far from where it was parked must not inherit a stale positional cap
        # (prototype rule). Under mask_contain the post-event apply() refresh re-derives real
        # caps from the signed distance immediately.
        field.scale_max[rows] = float("inf")
    pool.live[rows] = True
    return rows, k, int(hole.sum())


@torch.no_grad()
def triage_event(field: GaussianField, pool: PoolState, target: torch.Tensor,
                 render_img: torch.Tensor, cfg: FitConfig, mask_constraint=None,
                 support_fade_alpha: float | None = None) -> TriageStats:
    """Run one park -> merge -> split -> spawn event in place. Returns per-op counts and the
    union of touched rows (whose optimizer moments the caller must zero)."""
    stats = attribution_pass(field, target, render_img, cfg, support_fade_alpha)
    parked = _park_donors(field, pool, stats, cfg)
    merge_touched, merge_stats = _merge_pairs(field, pool, cfg, mask_constraint)
    split_touched, n_split = _split_owners(
        field, pool, stats, cfg, pool.free_rows(), cfg.triage_split_count
    )
    spawn_rows, n_spawn, n_holes = _activate_sites(
        field, pool, stats, cfg, target, render_img, mask_constraint,
        pool.free_rows(), cfg.triage_spawn_count,
    )
    touched = torch.unique(torch.cat([parked, merge_touched, split_touched, spawn_rows]))
    return TriageStats(
        parked=int(parked.numel()),
        merged=merge_stats["merged"],
        merge_candidates=merge_stats["candidates"],
        merge_containment_rejected=merge_stats["containment_rejected"],
        split=n_split,
        spawned=n_spawn,
        spawned_holes=n_holes,
        touched=touched,
    )
