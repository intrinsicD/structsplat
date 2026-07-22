"""Initialization strategies — the variables the core ablation (ABL-001) compares.

STRATEGIES:
  random            uniform positions, isotropic                         (3DGS-style baseline)
  grid              uniform grid, isotropic                              (GaussianVision baseline)
  iso_blue_noise    density-adaptive isotropic blue noise                (feature-aware, no anisotropy)
  aniso_onedge      anisotropic blue noise, centers ON features          (tensor-oriented)
  aniso_flanking    anisotropic blue noise, edge centers pushed to flanks (control arm)
  feedforward       FF-001 predictor API: saved-field warm start or tensor-prior fallback
  quadtree_aggregate density-adaptive quadtree cells with aggregate color/features
  quadtree_hybrid   aggregate smooth cells, WSE/flanking samples for detailed cells
  quadtree_wse      quadtree budget cells with local WSE samples (shipped default)
  local_slic_sobel_control  local SLIC/Sobel complexity allocation (publication control)

The feature-aware strategies all share the structure tensor (orientation + density).
`build_field` also accepts a precomputed density/tensor so the pyramid can drive placement from
the residual (HIER-001). Colors are always sampled from the *target* image (see ADR-0003).
Requires torch (only to assemble the GaussianField); the heavy math stays NumPy.
"""
from __future__ import annotations
import heapq
from dataclasses import replace
import numpy as np

from . import structure_tensor as st
from . import density as de
from . import sampling as sa
from . import mask as _mask
from .config import InitConfig, StructureTensorConfig
from .gaussians import GaussianField

STRATEGIES = (
    "random", "grid", "iso_blue_noise", "aniso_onedge", "aniso_flanking",
    "feedforward", "quadtree_aggregate", "quadtree_hybrid", "quadtree_wse",
    "local_slic_sobel_control",
)


def _bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x = np.clip(pts[:, 0], 0, W - 1.001)
    y = np.clip(pts[:, 1], 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    c = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
         + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return c.astype(np.float32)


def _local_mean_colors(img: np.ndarray, pts: np.ndarray, radius: float) -> np.ndarray:
    H, W = img.shape[:2]
    r = max(0, int(round(radius)))
    if r == 0:
        return _bilinear(img, pts)
    integ = np.pad(img.astype(np.float64), ((1, 0), (1, 0), (0, 0)), mode="constant")
    integ = integ.cumsum(axis=0).cumsum(axis=1)
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    x0 = np.clip(xi - r, 0, W - 1)
    x1 = np.clip(xi + r + 1, 0, W)
    y0 = np.clip(yi - r, 0, H - 1)
    y1 = np.clip(yi + r + 1, 0, H)
    sums = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
    area = ((y1 - y0) * (x1 - x0))[:, None]
    return (sums / np.maximum(area, 1)).astype(np.float32)


def _nearest(map2d: np.ndarray, pts: np.ndarray) -> np.ndarray:
    H, W = map2d.shape[:2]
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    return map2d[yi, xi]


def _radius_map(density: np.ndarray, n: int, r_min=0.5, r_max=20.0) -> np.ndarray:
    lam = np.maximum(n * density, 1e-9)          # expected points per pixel
    return np.clip(np.sqrt(1.0 / (np.pi * lam)), r_min, r_max)


# The samplers work with the exclusion-disk radius r = sqrt(area/pi); the *scale* init works
# with the per-point cell side sqrt(area) — the definition random/grid use. Returning r as
# "spacing" would hand the feature-aware strategies a systematically sqrt(pi)~1.77x smaller
# initial scale at identical local density: a confound in the strategy ablation, not a choice.
_SPACING_PER_RADIUS = float(np.sqrt(np.pi))
_NN_SPACING_MAX_MATRIX_ELEMS = 16_000_000  # 128 MB at float64
_FEATURE_RUN_LENGTH_MAX_GRID_ELEMS = 2_000_000


def _sat(a: np.ndarray) -> np.ndarray:
    pad = [(1, 0), (1, 0)] + [(0, 0)] * max(0, a.ndim - 2)
    return np.pad(a.astype(np.float64), pad, mode="constant").cumsum(axis=0).cumsum(axis=1)


def _rect_sum(sat: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]


def _split_cell(cell: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = cell
    if x1 - x0 <= 1 and y1 - y0 <= 1:
        return []
    xs = [(x0, x1)]
    ys = [(y0, y1)]
    if x1 - x0 > 1:
        xm = (x0 + x1) // 2
        xs = [(x0, xm), (xm, x1)]
    if y1 - y0 > 1:
        ym = (y0 + y1) // 2
        ys = [(y0, ym), (ym, y1)]
    return [(xa, ya, xb, yb) for xa, xb in xs for ya, yb in ys if xb > xa and yb > ya]


def _quadtree_leaves(density: np.ndarray, n: int) -> list[tuple[int, int, int, int]]:
    H, W = density.shape
    mass_sat = _sat(density)

    def priority(cell: tuple[int, int, int, int]) -> tuple[float, int]:
        x0, y0, x1, y1 = cell
        area = (x1 - x0) * (y1 - y0)
        return float(_rect_sum(mass_sat, x0, y0, x1, y1)), area

    leaves: list[tuple[int, int, int, int]] = [(0, 0, W, H)]
    active = [True]
    heap: list[tuple[float, int, int, list[tuple[int, int, int, int]]]] = []

    def push_entry(idx: int, cell: tuple[int, int, int, int]) -> None:
        children = _split_cell(cell)
        if not children:
            return
        mass, area = priority(cell)
        heapq.heappush(heap, (-mass, -area, idx, children))

    push_entry(0, leaves[0])
    leaf_count = 1
    while leaf_count < n and heap:
        remaining = n - leaf_count
        deferred = []
        chosen: tuple[int, list[tuple[int, int, int, int]]] | None = None
        while heap:
            entry = heapq.heappop(heap)
            _, _, idx, children = entry
            if not active[idx]:
                continue
            if len(children) - 1 <= remaining:
                chosen = idx, children
                break
            deferred.append(entry)
        for entry in deferred:
            heapq.heappush(heap, entry)
        if chosen is None:
            break
        idx, children = chosen
        active[idx] = False
        leaf_count += len(children) - 1
        for child in children:
            child_idx = len(leaves)
            leaves.append(child)
            active.append(True)
            push_entry(child_idx, child)

    leaves = [cell for cell, is_active in zip(leaves, active) if is_active]
    if len(leaves) < n:
        child_pool = []
        for cell in leaves:
            child_pool.extend(_split_cell(cell))
        child_pool.sort(key=priority, reverse=True)
        leaves.extend(child_pool[:n - len(leaves)])

    if len(leaves) > n:  # defensive fallback for degenerate tiny images
        leaves = sorted(leaves, key=priority, reverse=True)[:n]
    return leaves


def _quadtree_cell_aggregates(
    img: np.ndarray,
    density: np.ndarray,
    tensor: st.StructureTensor,
    icfg: InitConfig,
    leaves: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W = density.shape
    mass_sat = _sat(density)
    yy, xx = np.mgrid[0:H, 0:W]
    xmass_sat = _sat(density * xx)
    ymass_sat = _sat(density * yy)
    color_sat = _sat(img[..., :3])
    energy = np.maximum(tensor.energy, 0.0)
    coherence = np.clip(tensor.coherence, 0.0, 1.0)
    orient_weight = energy * coherence
    along = tensor.along_edge_angle
    energy_sat = _sat(energy)
    coh_sat = _sat(orient_weight)
    cos_sat = _sat(orient_weight * np.cos(2.0 * along))
    sin_sat = _sat(orient_weight * np.sin(2.0 * along))

    pts = np.empty((len(leaves), 2), dtype=np.float64)
    spacing = np.empty(len(leaves), dtype=np.float64)
    angles = np.empty(len(leaves), dtype=np.float64)
    ratios = np.empty(len(leaves), dtype=np.float64)
    colors = np.empty((len(leaves), 3), dtype=np.float32)
    detail = np.empty(len(leaves), dtype=np.float64)
    for i, (x0, y0, x1, y1) in enumerate(leaves):
        area = max((x1 - x0) * (y1 - y0), 1)
        mass = float(_rect_sum(mass_sat, x0, y0, x1, y1))
        detail[i] = mass / area
        if mass > 1e-12:
            x = float(_rect_sum(xmass_sat, x0, y0, x1, y1) / mass)
            y = float(_rect_sum(ymass_sat, x0, y0, x1, y1) / mass)
        else:
            x = 0.5 * (x0 + x1 - 1)
            y = 0.5 * (y0 + y1 - 1)
        pts[i] = [np.clip(x, 0.0, W - 1.0), np.clip(y, 0.0, H - 1.0)]
        spacing[i] = np.sqrt(area)
        colors[i] = (_rect_sum(color_sat, x0, y0, x1, y1) / area).astype(np.float32)

        e = float(_rect_sum(energy_sat, x0, y0, x1, y1))
        c = float(_rect_sum(coh_sat, x0, y0, x1, y1))
        coh = np.clip(c / (e + 1e-12), 0.0, 1.0)
        ratios[i] = 1.0 + (icfg.max_axis_ratio - 1.0) * (coh ** icfg.coherence_power)
        if c > 1e-12:
            angles[i] = 0.5 * np.arctan2(
                _rect_sum(sin_sat, x0, y0, x1, y1),
                _rect_sum(cos_sat, x0, y0, x1, y1),
            )
        else:
            angles[i] = 0.0
    return pts, spacing, angles, ratios, colors, detail


def _quadtree_aggregate_init(img: np.ndarray, density: np.ndarray, tensor: st.StructureTensor,
                             icfg: InitConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray, np.ndarray]:
    """Density-prioritized quadtree initialization with cell aggregate attributes."""
    leaves = _quadtree_leaves(density, icfg.num_gaussians)
    pts, spacing, angles, ratios, colors, _ = _quadtree_cell_aggregates(
        img, density, tensor, icfg, leaves
    )
    return pts, spacing, spacing.copy(), angles, ratios, colors


def _tensor_attrs_at_points(tensor: st.StructureTensor, pts: np.ndarray,
                            icfg: InitConfig, anisotropic: bool) -> tuple[np.ndarray, np.ndarray]:
    angles = _nearest(tensor.along_edge_angle, pts)
    if not anisotropic:
        return angles, np.ones(len(pts))
    coh = np.clip(_nearest(tensor.coherence, pts), 0.0, 1.0) ** icfg.coherence_power
    return angles, 1.0 + (icfg.max_axis_ratio - 1.0) * coh


def _flank_edge_points(pts: np.ndarray, spacing: np.ndarray, ratios: np.ndarray,
                       tensor: st.StructureTensor, scfg: StructureTensorConfig | None,
                       icfg: InitConfig, two_sided: bool = False
                       ) -> tuple[np.ndarray, np.ndarray | None]:
    """Push edge Gaussians off the ridge onto one flank; optionally sample a two-sided color.

    The single flanking implementation used by both `aniso_flanking` and the quadtree strategies
    (INIT-005). Returns `(flanked_pts, color_pts)`; `color_pts` is None unless `two_sided`.
    """
    label = _nearest(tensor.label, pts)
    across = _nearest(tensor.across_edge_angle, pts)
    normal = np.stack([np.cos(across), np.sin(across)], 1)
    s_across = spacing / np.sqrt(np.maximum(ratios, 1.0))
    edge_w = 2.0 * (scfg or StructureTensorConfig()).grad_sigma
    sign = np.where((np.arange(len(pts)) % 2) == 0, 1.0, -1.0)
    # Floor the offset distance at the blur width AFTER scaling by the fraction, so the flank
    # actually clears the blurred transition zone. Flooring before the fraction (the old
    # max(s_across, edge_w)*frac) only guaranteed edge_w*frac ~ 1px at defaults, not the blur
    # width the comment claimed, so flanking degenerated toward on-edge placement (INIT-005).
    offset_dist = np.maximum(s_across * icfg.flank_offset_frac, edge_w)
    offset = (sign * offset_dist)[:, None] * normal
    is_edge = (label == 1)[:, None]
    out = pts + offset * is_edge
    H, W = tensor.energy.shape
    out[:, 0] = np.clip(out[:, 0], 0, W - 1)
    out[:, 1] = np.clip(out[:, 1], 0, H - 1)
    if not two_sided:
        return out, None
    # Sample the flat color on the DOWN-energy side of the edge as seen from the flanked center.
    # The parity sign only says which flank the center was pushed toward; for off-ridge starts it
    # can point back across the edge, so pick the side by comparing energy. Cap the probe by the
    # flank distance (the local across-edge feature half-width proxy) so it does not overshoot a
    # thin structure and sample the far side (INIT-005).
    eps = np.minimum(np.maximum(s_across * icfg.color_radius, edge_w), offset_dist)
    e_pos = _nearest(tensor.energy, out + eps[:, None] * normal)
    e_neg = _nearest(tensor.energy, out - eps[:, None] * normal)
    away = np.where(e_pos <= e_neg, 1.0, -1.0)
    color_pts = out + (away * eps)[:, None] * normal * is_edge
    color_pts[:, 0] = np.clip(color_pts[:, 0], 0, W - 1)
    color_pts[:, 1] = np.clip(color_pts[:, 1], 0, H - 1)
    return out, color_pts


def _feature_run_lengths(tensor: st.StructureTensor, pts: np.ndarray, angles: np.ndarray,
                         scfg: StructureTensorConfig | None,
                         icfg: InitConfig) -> np.ndarray:
    H, W = tensor.energy.shape
    scfg = scfg or StructureTensorConfig()
    energy = np.maximum(tensor.energy, 0.0)
    ref = getattr(tensor, "energy_ref", None)
    ref = st.energy_reference(energy) if ref is None else float(ref)
    floor = st.flat_threshold(energy, scfg.flat_frac, ref)
    local_energy = _nearest(energy, pts)
    max_steps = int(np.ceil(max(H, W)))
    if icfg.scale_cap_max is not None:
        max_steps = max(1, min(max_steps, int(np.ceil(icfg.scale_cap_max
                                                      * icfg.scale_feature_sigma))))
    lengths = np.full(len(pts), np.inf, dtype=np.float64)
    active_idx = np.flatnonzero(~(local_energy < floor))
    if len(active_idx) == 0:
        return lengths

    active_pts = np.asarray(pts[active_idx], dtype=np.float64)
    active_angles = np.asarray(angles[active_idx], dtype=np.float64)
    thresholds = np.fmax(floor, local_energy[active_idx] * icfg.scale_feature_energy_frac)
    directions = np.stack([np.cos(active_angles), np.sin(active_angles)], axis=1)
    steps = np.arange(1, max_steps + 1, dtype=np.float64)
    chunk = max(1, min(len(active_idx), _FEATURE_RUN_LENGTH_MAX_GRID_ELEMS // max_steps))

    def walk_counts(sign: float) -> np.ndarray:
        counts = np.empty(len(active_idx), dtype=np.int64)
        for start in range(0, len(active_idx), chunk):
            end = min(start + chunk, len(active_idx))
            p = active_pts[start:end]
            direction = directions[start:end]
            xi = np.rint(p[:, 0, None] + sign * direction[:, 0, None] * steps).astype(np.int64)
            yi = np.rint(p[:, 1, None] + sign * direction[:, 1, None] * steps).astype(np.int64)
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            xq = np.clip(xi, 0, W - 1)
            yq = np.clip(yi, 0, H - 1)
            valid &= tensor.label[yq, xq] != 2
            valid &= ~(energy[yq, xq] < thresholds[start:end, None])
            invalid = ~valid
            has_invalid = invalid.any(axis=1)
            first_invalid = invalid.argmax(axis=1)
            counts[start:end] = np.where(has_invalid, first_invalid, max_steps)
        return counts

    lengths[active_idx] = walk_counts(1.0) + walk_counts(-1.0) + 1.0
    return lengths


def _scale_caps(tensor: st.StructureTensor | None, pts: np.ndarray, angles: np.ndarray,
                scales: np.ndarray, feature_scale: np.ndarray | None, icfg: InitConfig,
                scfg: StructureTensorConfig | None) -> np.ndarray | None:
    mode = icfg.scale_cap_mode
    if mode == "none":
        return None
    if mode not in ("hard", "feature", "feature_rel"):
        raise ValueError("unknown scale_cap_mode "
                         f"{mode!r}; expected none, hard, feature, or feature_rel")
    if icfg.scale_cap_max is None and mode == "hard":
        raise ValueError("scale_cap_mode='hard' requires scale_cap_max")

    caps = np.full_like(scales, np.inf, dtype=np.float64)
    if icfg.scale_cap_max is not None:
        caps[:] = max(float(icfg.scale_cap_max), 1e-3)
    if mode == "feature":
        if tensor is None:
            raise ValueError("scale_cap_mode='feature' requires a structure tensor")
        run = _feature_run_lengths(tensor, pts, angles, scfg, icfg)
        adaptive = run / max(float(icfg.scale_feature_sigma), 1e-3)
        adaptive = np.maximum(adaptive, float(icfg.scale_feature_min))
        adaptive = np.maximum(adaptive, scales[:, 1])
        finite = np.isfinite(adaptive)
        caps[finite, 0] = np.minimum(caps[finite, 0], adaptive[finite])
    elif mode == "feature_rel":
        has_feature_scale = feature_scale is not None
        if feature_scale is None:
            # Fallback for direct helper use: geometric-mean scale is resolution relative, but
            # all shipped init strategies pass an explicit WSE radius or quadtree leaf side.
            feature_scale = np.sqrt(np.maximum(scales[:, 0] * scales[:, 1], 1e-12))
        base = np.maximum(np.asarray(feature_scale, dtype=np.float64), icfg.scale_feature_rel_min)
        if base.shape[0] != scales.shape[0]:
            raise ValueError(
                f"feature_scale length {base.shape[0]} does not match scales length {scales.shape[0]}"
            )
        rel = np.full_like(scales, np.inf, dtype=np.float64)
        finite_base = base[np.isfinite(base)]
        if (
            has_feature_scale
            and finite_base.size > 1
            and float(finite_base.max() - finite_base.min()) > 1e-6
        ):
            # `feature_rel` should track local content scale, not the fraction of pixels a
            # fixed-sigma tensor labels as non-flat after resizing. Sparse/high-radius samples
            # are the flat coverage rows and stay loose by default.
            threshold = float(np.quantile(finite_base, icfg.scale_feature_rel_quantile))
            detail = base < threshold
            if not detail.any() and icfg.scale_feature_rel_quantile > 0.0:
                detail = base <= threshold
        elif tensor is not None:
            detail = _nearest(tensor.label, pts) != 0
        else:
            detail = np.ones(scales.shape[0], dtype=bool)
        rel[detail, 0] = icfg.scale_feature_rel_along * base[detail]
        rel[detail, 1] = icfg.scale_feature_rel_across * base[detail]
        flat = ~detail
        if icfg.scale_feature_rel_flat_mult is not None and flat.any():
            rel[flat] = icfg.scale_feature_rel_flat_mult * base[flat, None]
        finite = np.isfinite(rel)
        rel[finite] = np.maximum(rel[finite], icfg.scale_feature_rel_min)
        caps = np.minimum(caps, rel)
    if np.isinf(caps).all():
        return None
    return caps.astype(np.float32)


def _normalize_density(density: np.ndarray) -> np.ndarray:
    d = np.maximum(density.astype(np.float64), 0.0)
    s = d.sum()
    if s <= 0.0:
        return np.full(d.shape, 1.0 / d.size, dtype=np.float64)
    return d / s


def _quadtree_wse_init(img: np.ndarray, density: np.ndarray, tensor: st.StructureTensor,
                       icfg: InitConfig, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use quadtree cells only for budget allocation; place local WSE samples inside cells."""
    n = icfg.num_gaussians
    leaf_count = max(1, min(n, n // 4 if n >= 4 else n))
    leaves = _quadtree_leaves(density, leaf_count)
    mass_sat = _sat(density)
    masses = np.array([float(_rect_sum(mass_sat, *cell)) for cell in leaves], dtype=np.float64)
    if masses.sum() <= 0.0:
        masses = np.ones(len(leaves), dtype=np.float64)
    counts = np.ones(len(leaves), dtype=np.int64)
    remaining = n - len(leaves)
    if remaining > 0:
        raw = masses / masses.sum() * remaining
        extra = np.floor(raw).astype(np.int64)
        counts += extra
        remainder = n - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - extra))
            counts[order[:remainder]] += 1

    pts_parts, spacing_parts, feature_scale_parts = [], [], []
    for (x0, y0, x1, y1), k in zip(leaves, counts):
        if k <= 0:
            continue
        local = _normalize_density(density[y0:y1, x0:x1])
        n_cand = max(k, int(np.ceil(icfg.candidate_oversample * k)))
        cand_local = de.sample_candidates(local, n_cand, rng)
        cand = cand_local + np.array([x0, y0], dtype=np.float64)
        local_r = _radius_map(local, k)
        local_feature_r = _radius_map(local, k, r_max=np.inf)
        r_i = _nearest(local_r, cand_local)
        feature_r_i = _nearest(local_feature_r, cand_local)
        angle = _nearest(tensor.across_edge_angle, cand)
        coh = np.clip(_nearest(tensor.coherence, cand), 0.0, 1.0) ** icfg.coherence_power
        metric = sa.anisotropy_metric(angle, 1.0 + (icfg.max_axis_ratio - 1.0) * coh)
        keep = sa.eliminate(cand, k, r_i, metric=metric)
        pts_parts.append(cand[keep])
        spacing_parts.append(r_i[keep] * _SPACING_PER_RADIUS)
        feature_scale_parts.append(feature_r_i[keep])

    pts = np.concatenate(pts_parts, axis=0) if pts_parts else np.empty((0, 2), dtype=np.float64)
    spacing = np.concatenate(spacing_parts, axis=0) if spacing_parts else np.empty(0, dtype=np.float64)
    feature_scale = (
        np.concatenate(feature_scale_parts, axis=0)
        if feature_scale_parts else np.empty(0, dtype=np.float64)
    )
    if len(pts) > n:
        pts = pts[:n]
        spacing = spacing[:n]
        feature_scale = feature_scale[:n]
    angles, ratios = _tensor_attrs_at_points(tensor, pts, icfg, anisotropic=True)
    return pts, spacing, feature_scale, angles, ratios


def _quadtree_hybrid_init(
    img: np.ndarray,
    density: np.ndarray,
    tensor: st.StructureTensor,
    scfg: StructureTensorConfig | None,
    icfg: InitConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Aggregate low-detail cells and use WSE/flanking samples for the remaining budget."""
    n = icfg.num_gaussians
    n_agg = min(n, max(1, int(round(0.25 * n))))
    leaves = _quadtree_leaves(density, n)
    q_pts, q_spacing, q_angles, q_ratios, q_colors, detail = _quadtree_cell_aggregates(
        img, density, tensor, icfg, leaves
    )
    q_feature_scale = q_spacing.copy()
    agg_idx = np.argsort(detail)[:n_agg]

    smooth_mask = np.zeros(density.shape, dtype=bool)
    for i in agg_idx:
        x0, y0, x1, y1 = leaves[int(i)]
        smooth_mask[y0:y1, x0:x1] = True
    high_density = density.copy()
    high_density[smooth_mask] *= 0.02
    high_density = _normalize_density(high_density)

    n_wse = n - n_agg
    if n_wse > 0:
        wse_cfg = replace(icfg, num_gaussians=n_wse)
        wse_pts, wse_spacing, wse_feature_scale = _blue_noise_positions(
            img, high_density, tensor, wse_cfg, anisotropic=True, rng=rng
        )
        wse_angles, wse_ratios = _tensor_attrs_at_points(tensor, wse_pts, icfg, anisotropic=True)
        wse_pts, _ = _flank_edge_points(wse_pts, wse_spacing, wse_ratios, tensor, scfg, icfg)
    else:
        wse_pts = np.empty((0, 2), dtype=np.float64)
        wse_spacing = np.empty(0, dtype=np.float64)
        wse_feature_scale = np.empty(0, dtype=np.float64)
        wse_angles = np.empty(0, dtype=np.float64)
        wse_ratios = np.empty(0, dtype=np.float64)

    pts = np.concatenate([q_pts[agg_idx], wse_pts], axis=0)
    spacing = np.concatenate([q_spacing[agg_idx], wse_spacing], axis=0)
    feature_scale = np.concatenate([q_feature_scale[agg_idx], wse_feature_scale], axis=0)
    angles = np.concatenate([q_angles[agg_idx], wse_angles], axis=0)
    ratios = np.concatenate([q_ratios[agg_idx], wse_ratios], axis=0)
    colors = None
    if icfg.color_mode == "aggregate":
        colors = np.concatenate([q_colors[agg_idx], _bilinear(img, wse_pts)], axis=0)
    return pts, spacing, feature_scale, angles, ratios, colors


def _opacity_logits(n: int, mode: str, init_opacity: float) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "constant":
        p = np.clip(init_opacity, 1e-4, 1.0 - 1e-4)
        return np.full(n, np.log(p / (1.0 - p)), dtype=np.float32)
    raise ValueError(f"unknown opacity_mode {mode!r}; expected none or constant")


def background_count(icfg: InitConfig) -> int:
    if icfg.background_fraction <= 0.0 or icfg.background_grid <= 0 or icfg.num_gaussians <= 1:
        return 0
    requested = max(1, int(round(icfg.num_gaussians * icfg.background_fraction)))
    grid_cap = int(icfg.background_grid) * int(icfg.background_grid)
    return min(icfg.num_gaussians - 1, grid_cap, requested)


def _background_layer(img: np.ndarray, icfg: InitConfig, rng: np.random.Generator,
                      device: str) -> GaussianField | None:
    n = background_count(icfg)
    if n <= 0:
        return None
    H, W = img.shape[:2]
    grid = int(icfg.background_grid)
    cell_w = W / max(grid, 1)
    cell_h = H / max(grid, 1)
    gx, gy = np.meshgrid(np.arange(grid), np.arange(grid))
    xs = (gx + rng.random((grid, grid))) * cell_w - 0.5
    ys = (gy + rng.random((grid, grid))) * cell_h - 0.5
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    if len(pts) > n:
        pts = pts[np.round(np.linspace(0, len(pts) - 1, n)).astype(int)]
    pts[:, 0] = np.clip(pts[:, 0], -0.5, W - 0.5)
    pts[:, 1] = np.clip(pts[:, 1], -0.5, H - 0.5)
    sigma = max(float(np.sqrt(cell_w * cell_h)), 1e-3)
    scales = np.full((len(pts), 2), sigma, dtype=np.float32)
    colors = _local_mean_colors(img, pts, max(cell_w, cell_h) * 0.5)
    return GaussianField.from_numpy(
        pts,
        scales,
        np.zeros(len(pts), dtype=np.float32),
        colors,
        _opacity_logits(len(pts), icfg.opacity_mode, icfg.init_opacity),
        device=device,
        background_mask=np.ones(len(pts), dtype=bool),
    )


def _jittered_grid_positions(H: int, W: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    gw = int(round(np.sqrt(n * W / H)))
    gh = int(np.ceil(n / max(gw, 1)))
    cell_w, cell_h = W / max(gw, 1), H / max(gh, 1)
    gx, gy = np.meshgrid(np.arange(gw), np.arange(gh))
    # independent jitter per CELL (stratified sampling); a shared jitter per row/column would
    # collapse this to a randomized lattice. -0.5 keeps pixel centers at integer coords.
    u = rng.random((gh, gw))
    v = rng.random((gh, gw))
    xs = (gx + u) * cell_w - 0.5
    ys = (gy + v) * cell_h - 0.5
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    if len(pts) > n:  # drop evenly across the grid, not the bottom rows
        pts = pts[np.round(np.linspace(0, len(pts) - 1, n)).astype(int)]
    spacing = np.full(len(pts), np.sqrt(cell_w * cell_h))
    return pts, spacing


def _nn_spacing(pts: np.ndarray, r_min: float = 0.5, r_max: float = 40.0,
                chunk: int = 2048) -> np.ndarray:
    """Per-point distance to the nearest other point (chunked GEMM distance matrix)."""
    n = len(pts)
    if n < 2:
        return np.full(n, r_max)
    if chunk <= 0:
        raise ValueError(f"chunk must be > 0, got {chunk}")
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.einsum("ij,ij->i", pts, pts)
    # Bound the only O(chunk*N) allocation. The old broadcast path allocated
    # (chunk, N, 2); this keeps the distance matrix itself under the fixed budget.
    eff_chunk = min(chunk, max(1, _NN_SPACING_MAX_MATRIX_ELEMS // n))
    out = np.empty(n)
    for s in range(0, n, eff_chunk):
        e = min(s + eff_chunk, n)
        block = pts[s:e]
        d2 = block @ pts.T
        d2 *= -2.0
        d2 += norms[s:e, None]
        d2 += norms[None, :]
        np.maximum(d2, 0.0, out=d2)
        rows = np.arange(e - s)
        d2[rows, s + rows] = np.inf
        nearest2 = d2.min(axis=1)
        out[s:e] = np.sqrt(nearest2)
    return np.clip(out, r_min, r_max)


SAMPLING_MODES = ("wse", "density_random", "floyd_steinberg", "jittered_grid",
                  "dart_throwing", "halton", "farthest_point", "cvt")


def _blue_noise_positions(img, density, tensor, icfg, anisotropic, rng):
    n = icfg.num_gaussians
    rmap = _radius_map(density, n)
    feature_rmap = _radius_map(density, n, r_max=np.inf)
    H, W = density.shape
    mode = icfg.sampling_mode
    if mode == "density_random":
        pts = de.sample_candidates(density, n, rng)
        spacing = _nearest(rmap, pts) * _SPACING_PER_RADIUS
        feature_scale = _nearest(feature_rmap, pts)
        return pts, spacing, feature_scale
    if mode == "floyd_steinberg":
        pts = sa.floyd_steinberg(density, n)
        spacing = _nearest(rmap, pts) * _SPACING_PER_RADIUS
        feature_scale = _nearest(feature_rmap, pts)
        return pts, spacing, feature_scale
    if mode == "jittered_grid":
        pts, spacing = _jittered_grid_positions(H, W, n, rng)
        return pts, spacing, spacing.copy()
    if mode == "halton":
        pts = de.warp_unit_points(sa.halton_unit(n, rng), density)
        spacing = _nearest(rmap, pts) * _SPACING_PER_RADIUS
        feature_scale = _nearest(feature_rmap, pts)
        return pts, spacing, feature_scale
    if mode not in ("wse", "dart_throwing", "farthest_point", "cvt"):
        raise ValueError(
            f"unknown sampling_mode {mode!r}; expected one of {SAMPLING_MODES}"
        )
    cand = de.sample_candidates(density, int(icfg.candidate_oversample * n), rng)
    if mode == "cvt":
        pts = sa.cvt(cand, n, rng=rng)
        pts[:, 0] = np.clip(pts[:, 0], 0.0, W - 1.0)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, H - 1.0)
        spacing = _nearest(rmap, pts) * _SPACING_PER_RADIUS
        feature_scale = _nearest(feature_rmap, pts)
        return pts, spacing, feature_scale
    r_i = _nearest(rmap, cand)
    feature_r_i = _nearest(feature_rmap, cand)
    metric = None
    if anisotropic:
        angle = _nearest(tensor.across_edge_angle, cand)
        coh = np.clip(_nearest(tensor.coherence, cand), 0.0, 1.0) ** icfg.coherence_power
        ratio = 1.0 + (icfg.max_axis_ratio - 1.0) * coh
        metric = sa.anisotropy_metric(angle, ratio)
    if mode == "dart_throwing":
        keep = sa.dart_throwing(cand, n, r_i, metric=metric, rng=rng)
    elif mode == "farthest_point":
        keep = sa.farthest_point(cand, n, r_i=r_i, metric=metric, rng=rng)
    else:
        keep = sa.eliminate(cand, n, r_i, metric=metric)
    return cand[keep], r_i[keep] * _SPACING_PER_RADIUS, feature_r_i[keep]


def build_field(img: np.ndarray, icfg: InitConfig,
                scfg: StructureTensorConfig | None = None,
                density: np.ndarray | None = None,
                tensor: st.StructureTensor | None = None,
                device: str = "cpu") -> GaussianField:
    H, W = img.shape[:2]
    rng = np.random.default_rng(icfg.seed)
    bg_count = background_count(icfg)
    if bg_count > 0:
        detail_cfg = replace(
            icfg,
            num_gaussians=icfg.num_gaussians - bg_count,
            background_fraction=0.0,
            background_grid=0,
        )
        bg = _background_layer(img, icfg, rng, device)
        detail = build_field(
            img, detail_cfg, scfg, density=density, tensor=tensor, device=device
        )
        return bg.append(detail) if bg is not None else detail

    n = icfg.num_gaussians
    strat = icfg.strategy
    diag = float(np.hypot(H, W))
    color_pts = None
    colors = None

    if strat == "feedforward":
        from .predictor import predict_field
        return predict_field(img, icfg, scfg, density=density, tensor=tensor, device=device)
    if strat == "random":
        pts = rng.random((n, 2)) * np.array([W, H]) - 0.5   # pixel centers at integer coords
        spacing = np.full(n, np.sqrt(H * W / n))            # mean per-point area -> spacing
        feature_scale = spacing.copy()
        angles = np.zeros(n)
        ratios = np.ones(n)
    elif strat == "grid":
        gw = int(round(np.sqrt(n * W / H)))
        gh = int(np.ceil(n / max(gw, 1)))
        xs = (np.arange(gw) + 0.5) * W / gw - 0.5
        ys = (np.arange(gh) + 0.5) * H / gh - 0.5
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        if len(pts) > n:  # drop evenly across the grid, not the bottom rows
            pts = pts[np.round(np.linspace(0, len(pts) - 1, n)).astype(int)]
        spacing = np.full(len(pts), np.sqrt((W / gw) * (H / gh)))
        feature_scale = spacing.copy()
        angles = np.zeros(len(pts))
        ratios = np.ones(len(pts))
    elif strat == "local_slic_sobel_control":
        from .structural_controls import slic_sobel_placement

        placement = slic_sobel_placement(img, n, seed=icfg.seed)
        pts = placement.points_xy
        spacing = placement.spacing
        feature_scale = spacing.copy()
        angles = np.zeros(n)
        ratios = np.ones(n)
    else:
        if tensor is None:
            tensor = st.compute(img, scfg)
        if density is None:
            density = de.density_from_tensor_and_image(img, tensor, icfg, scfg)
        if strat == "quadtree_aggregate":
            pts, spacing, feature_scale, angles, ratios, colors = _quadtree_aggregate_init(
                img, density, tensor, icfg
            )
            if icfg.color_mode == "bilinear":
                colors = None
            elif icfg.color_mode == "local_mean":
                colors = None
            elif icfg.color_mode != "aggregate":
                raise ValueError(
                    "quadtree_aggregate supports color_mode aggregate, bilinear, or local_mean"
                )
        elif strat == "quadtree_wse":
            pts, spacing, feature_scale, angles, ratios = _quadtree_wse_init(
                img, density, tensor, icfg, rng
            )
            if icfg.color_mode == "aggregate":
                raise ValueError(
                    "quadtree_wse uses sampled colors; use color_mode bilinear or local_mean"
                )
            pts, _ = _flank_edge_points(pts, spacing, ratios, tensor, scfg, icfg)
        elif strat == "quadtree_hybrid":
            pts, spacing, feature_scale, angles, ratios, colors = _quadtree_hybrid_init(
                img, density, tensor, scfg, icfg, rng
            )
            if icfg.color_mode not in ("aggregate", "bilinear", "local_mean"):
                raise ValueError(
                    "quadtree_hybrid supports color_mode aggregate, bilinear, or local_mean"
                )
        else:
            anisotropic = strat in ("aniso_onedge", "aniso_flanking")
            pts, spacing, feature_scale = _blue_noise_positions(
                img, density, tensor, icfg, anisotropic, rng
            )
            angles = _nearest(tensor.along_edge_angle, pts)      # elongate along the edge
            coh = np.clip(_nearest(tensor.coherence, pts), 0.0, 1.0) ** icfg.coherence_power
            ratios = 1.0 + (icfg.max_axis_ratio - 1.0) * coh if anisotropic else np.ones(len(pts))
        if strat == "aniso_flanking":
            # one flanking implementation, shared with the quadtree strategies (INIT-005)
            pts, color_pts = _flank_edge_points(
                pts, spacing, ratios, tensor, scfg, icfg,
                two_sided=icfg.color_mode == "two_sided")

    progressive_perm = None
    pure_wse = strat == "quadtree_wse" or (
        strat in ("iso_blue_noise", "aniso_onedge", "aniso_flanking")
        and icfg.sampling_mode == "wse"
    )
    if icfg.wse_progressive_order and pure_wse and len(pts) >= 3:
        # Compute after parity-based flanking so ordering cannot change the represented geometry,
        # but before orientation ablations overwrite the sampling metric. Apply the permutation
        # only after every row-aligned Gaussian attribute has been constructed.
        progressive_metric = None
        if strat != "iso_blue_noise":
            progressive_metric = sa.anisotropy_metric(
                np.asarray(angles) - np.pi * 0.5,
                np.asarray(ratios),
            )
        progressive_perm = sa.progressive_order(
            np.asarray(pts),
            np.asarray(spacing) / _SPACING_PER_RADIUS,
            metric=progressive_metric,
        )

    n_out = len(pts)
    feature_scale = np.asarray(feature_scale, dtype=np.float64)[:n_out]
    m = icfg.init_scale_mult
    ratios = np.maximum(ratios, 1.0)
    # orientation stage: 'tensor' keeps the strategy's angles (structure-tensor tangent for the
    # feature-aware strategies, zero for random/grid); the alternatives ablate how much the
    # tensor *orientation* specifically contributes, independent of the anisotropy magnitude.
    if icfg.orientation_mode == "random":
        angles = rng.uniform(0.0, np.pi, n_out)
    elif icfg.orientation_mode == "zero":
        angles = np.zeros(n_out)
    elif icfg.orientation_mode != "tensor":
        raise ValueError(
            f"unknown orientation_mode {icfg.orientation_mode!r}; expected tensor, random, or zero"
        )
    if icfg.scale_mode == "uniform":
        spacing = np.full(n_out, diag / np.sqrt(max(n_out, 1)))
    elif icfg.scale_mode == "knn":
        spacing = _nn_spacing(np.asarray(pts, dtype=np.float64))
    elif icfg.scale_mode != "spacing":
        raise ValueError(f"unknown scale_mode {icfg.scale_mode!r}; expected spacing, uniform, or knn")
    s_along = spacing * np.sqrt(ratios) * m
    s_across = spacing / np.sqrt(ratios) * m
    scales = np.stack([s_along, s_across], 1)                # sx along tangent, sy across
    if icfg.scale_cap_mode in ("feature", "feature_rel") and tensor is None:
        tensor = st.compute(img, scfg)
    scale_max = _scale_caps(tensor, pts, angles[:n_out], scales, feature_scale, icfg, scfg)
    if scale_max is not None:
        scales = np.minimum(scales, scale_max)
    if colors is None:
        sample_pts = pts if color_pts is None else color_pts
        if icfg.color_mode in ("bilinear", "two_sided"):
            colors = _bilinear(img, sample_pts)
        elif icfg.color_mode == "local_mean":
            colors = _local_mean_colors(img, sample_pts, icfg.color_radius)
        else:
            raise ValueError(
                f"unknown color_mode {icfg.color_mode!r}; expected bilinear, local_mean, "
                "two_sided, or aggregate"
            )
    opacities = _opacity_logits(n_out, icfg.opacity_mode, icfg.init_opacity)
    if progressive_perm is not None:
        pts = np.asarray(pts)[progressive_perm]
        scales = np.asarray(scales)[progressive_perm]
        angles = np.asarray(angles)[progressive_perm]
        colors = np.asarray(colors)[progressive_perm]
        if opacities is not None:
            opacities = np.asarray(opacities)[progressive_perm]
        if scale_max is not None:
            scale_max = np.asarray(scale_max)[progressive_perm]
    return GaussianField.from_numpy(pts, scales, angles[:n_out], colors, opacities,
                                    scale_max, device=device)


def build_masked_field(img: np.ndarray, mask: np.ndarray, icfg: InitConfig,
                       scfg: StructureTensorConfig | None = None,
                       device: str = "cpu", *, sigma_cutoff: float = 3.0,
                       mask_margin: float = 1.5, dilate_colors: bool = True,
                       contain: bool = True, cap_mode: str = "isotropic") -> GaussianField:
    """Initialize a mask-contained field for an alpha-masked image (CORE-010).

    Any strategy works. Structure/orientation come from the (matted) image; seeds are restricted to
    the eroded mask via a masked density; colors are sampled from a boundary-dilated copy so the
    matte does not contaminate boundary colors (the matte *edge* is intentionally kept in the
    structure tensor as a tangent-aligned boundary attractor). With ``contain=True`` (default) means
    are then projected inside and effective scales capped from the signed distance, so the returned
    field's sigma_cutoff support is already contained and ``scale_cap_mode`` must be ``none`` (mask
    containment owns ``scale_max``; see :class:`fit._MaskConstraint`). With ``contain=False`` only
    the masked density and dilated colors apply, leaving scales free — for the soft-penalty /
    masked-loss-only arms.
    """
    if contain and icfg.scale_cap_mode != "none":
        raise ValueError(
            "build_masked_field(contain=True) manages scale_max; set "
            "InitConfig.scale_cap_mode='none'")
    inside = _mask.as_bool_mask(np.asarray(mask))
    H, W = img.shape[:2]
    if inside.shape != (H, W):
        raise ValueError(f"mask shape {inside.shape} does not match image {(H, W)}")
    scfg = scfg or StructureTensorConfig()
    tensor = st.compute(img, scfg)
    density = de.density_from_tensor_and_image(img, tensor, icfg, scfg)
    erode_radius = float(mask_margin) + float(sigma_cutoff) * _mask.MIN_SCALE
    eroded = _mask.erode(inside, erode_radius)
    region = eroded if eroded.any() else inside  # mask thinner than the erosion radius
    masked_density = density.astype(np.float64) * region
    total = masked_density.sum()
    if total > 0.0:
        masked_density = masked_density / total
    else:  # degenerate: place uniformly inside the raw mask
        masked_density = region.astype(np.float64)
        masked_density = masked_density / max(masked_density.sum(), 1.0)
    color_img = _mask.color_dilate(img, inside) if dilate_colors else img
    field = build_field(color_img, icfg, scfg, density=masked_density, tensor=tensor,
                        device=device)
    if contain:
        from .fit import _MaskConstraint  # local import: init is torch-bridge, avoids load cycle

        constraint = _MaskConstraint.from_mask(
            inside, field.means.device, field.means.dtype, sigma_cutoff, mask_margin,
            cap_mode=cap_mode,
        )
        constraint.apply(field, aa_dilation=0.0)
    return field
