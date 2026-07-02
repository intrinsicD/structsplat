"""Initialization strategies — the variables the core ablation (ABL-001) compares.

STRATEGIES:
  random            uniform positions, isotropic                         (3DGS-style baseline)
  grid              uniform grid, isotropic                              (GaussianVision baseline)
  iso_blue_noise    density-adaptive isotropic blue noise                (feature-aware, no anisotropy)
  aniso_onedge      anisotropic blue noise, centers ON features          (tensor-oriented)
  aniso_flanking    anisotropic blue noise, edge centers pushed to flanks (the proposed default)

The last three all share the structure tensor (orientation + density) and WSE (INIT-003).
`build_field` also accepts a precomputed density/tensor so the pyramid can drive placement from
the residual (HIER-001). Colors are always sampled from the *target* image (see ADR-0003).
Requires torch (only to assemble the GaussianField); the heavy math stays NumPy.
"""
from __future__ import annotations
import numpy as np

from . import structure_tensor as st
from . import density as de
from . import sampling as sa
from .config import InitConfig, StructureTensorConfig
from .gaussians import GaussianField

STRATEGIES = ("random", "grid", "iso_blue_noise", "aniso_onedge", "aniso_flanking")


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


def _opacity_logits(n: int, mode: str, init_opacity: float) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "constant":
        p = np.clip(init_opacity, 1e-4, 1.0 - 1e-4)
        return np.full(n, np.log(p / (1.0 - p)), dtype=np.float32)
    raise ValueError(f"unknown opacity_mode {mode!r}; expected none or constant")


def _jittered_grid_positions(H: int, W: int, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    gw = int(round(np.sqrt(n * W / H)))
    gh = int(np.ceil(n / max(gw, 1)))
    cell_w, cell_h = W / max(gw, 1), H / max(gh, 1)
    xs = (np.arange(gw) + rng.random(gw)) * cell_w
    ys = (np.arange(gh) + rng.random(gh)) * cell_h
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)[:n]
    spacing = np.full(len(pts), np.sqrt(cell_w * cell_h))
    return pts, spacing


def _blue_noise_positions(img, density, tensor, icfg, anisotropic, rng):
    n = icfg.num_gaussians
    rmap = _radius_map(density, n)
    H, W = density.shape
    if icfg.sampling_mode == "density_random":
        pts = de.sample_candidates(density, n, rng)
        return pts, _nearest(rmap, pts)
    if icfg.sampling_mode == "jittered_grid":
        return _jittered_grid_positions(H, W, n, rng)
    if icfg.sampling_mode != "wse":
        raise ValueError(
            f"unknown sampling_mode {icfg.sampling_mode!r}; expected wse, density_random, or jittered_grid"
        )
    cand = de.sample_candidates(density, int(icfg.candidate_oversample * n), rng)
    r_i = _nearest(rmap, cand)
    metric = None
    if anisotropic:
        angle = _nearest(tensor.across_edge_angle, cand)
        coh = np.clip(_nearest(tensor.coherence, cand), 0.0, 1.0) ** icfg.coherence_power
        ratio = 1.0 + (icfg.max_axis_ratio - 1.0) * coh
        metric = sa.anisotropy_metric(angle, ratio)
    keep = sa.eliminate(cand, n, r_i, metric=metric)
    return cand[keep], r_i[keep]


def build_field(img: np.ndarray, icfg: InitConfig,
                scfg: StructureTensorConfig | None = None,
                density: np.ndarray | None = None,
                tensor: st.StructureTensor | None = None,
                device: str = "cpu") -> GaussianField:
    H, W = img.shape[:2]
    rng = np.random.default_rng(icfg.seed)
    n = icfg.num_gaussians
    strat = icfg.strategy
    diag = float(np.hypot(H, W))
    color_pts = None

    if strat == "random":
        pts = rng.random((n, 2)) * np.array([W, H]) - 0.5   # pixel centers at integer coords
        spacing = np.full(n, np.sqrt(H * W / n))            # mean per-point area -> spacing
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
        angles = np.zeros(len(pts))
        ratios = np.ones(len(pts))
    else:
        if tensor is None:
            tensor = st.compute(img, scfg)
        if density is None:
            density = de.density_from_tensor_and_image(img, tensor, icfg, scfg)
        anisotropic = strat in ("aniso_onedge", "aniso_flanking")
        pts, spacing = _blue_noise_positions(img, density, tensor, icfg, anisotropic, rng)
        angles = _nearest(tensor.along_edge_angle, pts)      # elongate along the edge
        coh = np.clip(_nearest(tensor.coherence, pts), 0.0, 1.0) ** icfg.coherence_power
        ratios = 1.0 + (icfg.max_axis_ratio - 1.0) * coh if anisotropic else np.ones(len(pts))
        if strat == "aniso_flanking":
            label = _nearest(tensor.label, pts)
            across = _nearest(tensor.across_edge_angle, pts)
            normal = np.stack([np.cos(across), np.sin(across)], 1)
            s_across = spacing / np.sqrt(np.maximum(ratios, 1.0))
            # Floor the flank distance at the edge blur width: at realistic budgets the
            # across-edge spacing is sub-pixel, so a spacing-only offset never clears the
            # blurred transition zone and flanking degenerates to on-edge placement.
            edge_w = 2.0 * (scfg or StructureTensorConfig()).grad_sigma
            sign = np.where((np.arange(len(pts)) % 2) == 0, 1.0, -1.0)
            offset = (sign * np.maximum(s_across, edge_w) * icfg.flank_offset_frac)[:, None] * normal
            is_edge = (label == 1)[:, None]
            pts = pts + offset * is_edge                     # only edges get flanked
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            if icfg.color_mode == "two_sided":
                color_pts = pts + (sign * s_across * icfg.color_radius)[:, None] * normal * is_edge
                color_pts[:, 0] = np.clip(color_pts[:, 0], 0, W - 1)
                color_pts[:, 1] = np.clip(color_pts[:, 1], 0, H - 1)

    n_out = len(pts)
    m = icfg.init_scale_mult
    ratios = np.maximum(ratios, 1.0)
    if icfg.scale_mode == "uniform":
        spacing = np.full(n_out, diag / np.sqrt(max(n_out, 1)))
    elif icfg.scale_mode != "spacing":
        raise ValueError(f"unknown scale_mode {icfg.scale_mode!r}; expected spacing or uniform")
    s_along = spacing * np.sqrt(ratios) * m
    s_across = spacing / np.sqrt(ratios) * m
    scales = np.stack([s_along, s_across], 1)                # sx along tangent, sy across
    sample_pts = pts if color_pts is None else color_pts
    if icfg.color_mode in ("bilinear", "two_sided"):
        colors = _bilinear(img, sample_pts)
    elif icfg.color_mode == "local_mean":
        colors = _local_mean_colors(img, sample_pts, icfg.color_radius)
    else:
        raise ValueError(
            f"unknown color_mode {icfg.color_mode!r}; expected bilinear, local_mean, or two_sided"
        )
    opacities = _opacity_logits(n_out, icfg.opacity_mode, icfg.init_opacity)
    return GaussianField.from_numpy(pts, scales, angles[:n_out], colors, opacities, device=device)
