"""Local structural-placement controls used by publication benchmarks.

The SLIC/Sobel control follows the public algorithmic description in
Structure-Guided Allocation (arXiv:2512.24018): SLIC regions are ranked by the variance of Sobel gradient
magnitude, split into three complexity groups, and receive Gaussian budgets in
a 6:2:1 ratio.  The paper does not publish its SLIC hyperparameters or the
constants needed for its dynamic ratio schedule, so this module freezes and
exposes those fidelity assumptions instead of presenting them as upstream code.

All placement math is NumPy.  ``scikit-image`` is imported lazily because this
control is benchmark-only and should not make importing StructSplat heavier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SlicSobelConfig:
    """Frozen assumptions for the local Image-GS-style placement control."""

    target_superpixel_area: int = 1024
    compactness: float = 10.0
    sigma: float = 1.0
    convert2lab: bool = True
    enforce_connectivity: bool = True
    category_ratios: tuple[int, int, int] = (6, 2, 1)
    grayscale: str = "rec709"
    sobel_padding: str = "reflect"
    allocation_schedule: str = "fixed_sparse_6_2_1"
    implementation: str = "local_skimage_slic_numpy_sobel_v1"

    def __post_init__(self) -> None:
        if self.target_superpixel_area <= 0:
            raise ValueError("target_superpixel_area must be > 0")
        if self.compactness <= 0.0:
            raise ValueError("compactness must be > 0")
        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        if len(self.category_ratios) != 3 or any(v <= 0 for v in self.category_ratios):
            raise ValueError("category_ratios must contain three positive integers")

    def provenance(self) -> dict[str, object]:
        return {
            **asdict(self),
            "algorithm_source": "https://arxiv.org/abs/2512.24018",
            "fidelity": "local_control_not_upstream_code",
            "unresolved_upstream_details": [
                "SLIC hyperparameters",
                "dynamic allocation constants k and alpha",
                "exact category boundary convention",
            ],
        }


@dataclass(frozen=True)
class SlicSobelPlacement:
    """Exact-N placement plus audit maps and allocation diagnostics."""

    points_xy: np.ndarray
    spacing: np.ndarray
    labels: np.ndarray
    sobel_magnitude: np.ndarray
    region_complexity: np.ndarray
    region_counts: np.ndarray
    region_categories: np.ndarray
    config: SlicSobelConfig


def _convolve3_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded = np.pad(image.astype(np.float64), 1, mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    for iy in range(3):
        for ix in range(3):
            out += float(kernel[iy, ix]) * padded[iy:iy + image.shape[0], ix:ix + image.shape[1]]
    return out


def sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Return native-resolution Sobel magnitude from an HWC RGB float image."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {img.shape}")
    rgb = np.asarray(img, dtype=np.float64)
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    ky = kx.T
    gx = _convolve3_reflect(gray, kx)
    gy = _convolve3_reflect(gray, ky)
    return np.hypot(gx, gy).astype(np.float32)


def _largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0 or weights.ndim != 1 or weights.size == 0 or np.any(weights < 0.0):
        raise ValueError("invalid largest-remainder inputs")
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)
    raw = total * weights / weights.sum()
    out = np.floor(raw).astype(np.int64)
    remainder = total - int(out.sum())
    if remainder:
        # Stable index tie-break makes allocation byte-for-byte reproducible.
        order = np.argsort(-(raw - out), kind="stable")
        out[order[:remainder]] += 1
    return out


def _ranked_region_allocation(
    complexity: np.ndarray,
    num_points: int,
    ratios: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate groups 6:2:1, then uniformly within each complexity group."""
    if num_points < 0:
        raise ValueError("num_points must be >= 0")
    n_regions = len(complexity)
    counts = np.zeros(n_regions, dtype=np.int64)
    categories = np.full(n_regions, -1, dtype=np.int8)
    if n_regions == 0 or num_points == 0:
        return counts, categories

    ranked = np.argsort(-np.asarray(complexity), kind="stable")
    groups = np.array_split(ranked, 3)
    active = np.array([len(g) > 0 for g in groups])
    group_weights = np.asarray(ratios, dtype=np.float64) * active
    group_budgets = _largest_remainder(num_points, group_weights)
    for category, (group, budget) in enumerate(zip(groups, group_budgets)):
        if not len(group):
            continue
        categories[group] = category
        base, remainder = divmod(int(budget), len(group))
        counts[group] = base
        # ``group`` remains in descending complexity order, matching the reported
        # rule that leftover points go sequentially to the most complex regions.
        counts[group[:remainder]] += 1
    if int(counts.sum()) != num_points:
        raise RuntimeError("SLIC/Sobel allocation violated the exact-N contract")
    return counts, categories


def slic_sobel_placement(
    img: np.ndarray,
    num_points: int,
    seed: int = 0,
    cfg: SlicSobelConfig | None = None,
) -> SlicSobelPlacement:
    """Build a deterministic exact-N local SLIC/Sobel placement control."""
    cfg = cfg or SlicSobelConfig()
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape {img.shape}")
    try:
        from skimage.segmentation import slic
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installations
        raise ImportError(
            "local_slic_sobel_control requires the 'benchmark' extra: "
            "pip install -e '.[benchmark]'"
        ) from exc

    h, w = img.shape[:2]
    n_segments = max(1, int(round((h * w) / cfg.target_superpixel_area)))
    labels = slic(
        np.clip(img, 0.0, 1.0),
        n_segments=n_segments,
        compactness=cfg.compactness,
        sigma=cfg.sigma,
        start_label=0,
        convert2lab=cfg.convert2lab,
        enforce_connectivity=cfg.enforce_connectivity,
        channel_axis=-1,
    ).astype(np.int32, copy=False)
    # SLIC labels are contiguous with start_label=0, but relabel defensively so the
    # arrays below remain compact across scikit-image releases.
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(h, w).astype(np.int32, copy=False)
    n_regions = int(labels.max()) + 1

    magnitude = sobel_magnitude(img)
    flat_labels = labels.ravel()
    area = np.bincount(flat_labels, minlength=n_regions).astype(np.int64)
    sum_g = np.bincount(flat_labels, weights=magnitude.ravel(), minlength=n_regions)
    sum_g2 = np.bincount(flat_labels, weights=np.square(magnitude).ravel(), minlength=n_regions)
    mean_g = sum_g / np.maximum(area, 1)
    complexity = np.maximum(sum_g2 / np.maximum(area, 1) - np.square(mean_g), 0.0)
    counts, categories = _ranked_region_allocation(
        complexity, num_points, cfg.category_ratios
    )

    rng = np.random.default_rng(seed)
    ys, xs = np.indices((h, w))
    points: list[np.ndarray] = []
    spacing: list[np.ndarray] = []
    for region, count in enumerate(counts):
        if count <= 0:
            continue
        pixels = np.flatnonzero(flat_labels == region)
        replace = int(count) > len(pixels)
        chosen = rng.choice(pixels, size=int(count), replace=replace)
        px = xs.ravel()[chosen].astype(np.float64)
        py = ys.ravel()[chosen].astype(np.float64)
        if replace:
            # Avoid exact duplicate centers if a tiny region is oversubscribed.
            px += rng.uniform(-0.45, 0.45, size=int(count))
            py += rng.uniform(-0.45, 0.45, size=int(count))
        px = np.clip(px, -0.5, w - 0.5)
        py = np.clip(py, -0.5, h - 0.5)
        points.append(np.stack([px, py], axis=1))
        spacing.append(np.full(int(count), np.sqrt(area[region] / float(count))))

    points_xy = np.concatenate(points, axis=0).astype(np.float32, copy=False)
    local_spacing = np.concatenate(spacing, axis=0).astype(np.float32, copy=False)
    if len(points_xy) != num_points or len(local_spacing) != num_points:
        raise RuntimeError("SLIC/Sobel sampling violated the exact-N contract")
    return SlicSobelPlacement(
        points_xy=points_xy,
        spacing=local_spacing,
        labels=labels,
        sobel_magnitude=magnitude,
        region_complexity=complexity.astype(np.float32),
        region_counts=counts,
        region_categories=categories,
        config=cfg,
    )


__all__ = [
    "SlicSobelConfig",
    "SlicSobelPlacement",
    "slic_sobel_placement",
    "sobel_magnitude",
]
