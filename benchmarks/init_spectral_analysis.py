"""INIT-003 spectral and anisotropy diagnostics for initializer point sets.

This benchmark does not fit or render. It builds initial Gaussian fields, then measures the
placement-only signals INIT-003 needs: low-frequency spectral trough, pair-correlation exclusion,
nearest-neighbor spacing, edge-local directional spacing, and realized scale-axis ratios under
coherence -> axis-ratio sweeps.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from benchmarks.common import load_image, run_config, write_config, write_csv, write_json
from structsplat import init as init_mod
from structsplat import structure_tensor as st
from structsplat.config import InitConfig, StructureTensorConfig

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ANISOTROPIC_STRATEGIES = {
    "aniso_onedge",
    "aniso_flanking",
    "quadtree_aggregate",
    "quadtree_hybrid",
    "quadtree_wse",
}


def _iter_images(paths: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(path)
    return files


def _sample_map(map2d: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = map2d.shape[:2]
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, h - 1)
    return map2d[y, x]


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.mean())


def radial_power_profile(
    points: np.ndarray,
    height: int,
    width: int,
    *,
    bins: int = 40,
) -> dict[str, Any]:
    """Rasterize points and return a radially averaged FFT power profile."""
    points = np.asarray(points, dtype=np.float64)
    occupancy = np.zeros((height, width), dtype=np.float64)
    if len(points):
        x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
        y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
        np.add.at(occupancy, (y, x), 1.0)

    centered = occupancy - occupancy.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2 / max(height * width, 1)
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    edges = np.linspace(0.0, float(rr.max()) + 1e-12, bins + 1)
    which = np.clip(np.searchsorted(edges, rr.ravel(), side="right") - 1, 0, bins - 1)
    sums = np.bincount(which, weights=power.ravel(), minlength=bins).astype(np.float64)
    counts = np.bincount(which, minlength=bins).astype(np.float64)
    profile = np.divide(sums, counts, out=np.full(bins, np.nan), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {"frequency": centers.tolist(), "power": profile.tolist()}


def spectral_trough_metrics(profile: dict[str, Any]) -> dict[str, float | None]:
    """Return compact blue-noise spectral metrics; lower low/mid is a deeper trough."""
    power = np.asarray(profile["power"], dtype=np.float64)
    n = len(power)
    low_stop = max(2, n // 8)
    mid_start = max(low_stop, n // 5)
    mid_stop = max(mid_start + 1, n // 2)
    high_start = max(mid_stop, (2 * n) // 3)

    low = _finite_mean(power[1:low_stop])
    mid = _finite_mean(power[mid_start:mid_stop])
    high = _finite_mean(power[high_start:])
    ratio = None if low is None or mid is None or mid <= 0 else low / mid
    return {
        "spectral_low": low,
        "spectral_mid": mid,
        "spectral_high": high,
        "spectral_low_mid_ratio": ratio,
        "spectral_trough_depth": None if ratio is None else 1.0 - ratio,
    }


def pair_correlation_metrics(
    points: np.ndarray,
    height: int,
    width: int,
    *,
    bins: int = 32,
    r_max: float | None = None,
    max_points: int = 2500,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Approximate pair-correlation curve summary with uniform-edge correction omitted."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return {
            "pair_samples": int(len(points)),
            "pair_r_max": r_max,
            "pair_near_g": None,
            "pair_min_g": None,
            "pair_peak_g": None,
        }

    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        points = points[np.sort(rng.choice(len(points), size=max_points, replace=False))]
    n = len(points)
    if r_max is None:
        nominal_spacing = math.sqrt(max(height * width, 1) / n)
        r_max = min(0.25 * min(height, width), max(2.0, 4.0 * nominal_spacing))
    r_max = max(float(r_max), 1e-6)

    delta = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))
    dist = dist[np.triu_indices(n, k=1)]
    edges = np.linspace(0.0, r_max, bins + 1)
    hist, _ = np.histogram(dist, bins=edges)
    annulus = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    expected = 0.5 * n * (n - 1) * annulus / max(height * width, 1)
    g = np.divide(hist, expected, out=np.full_like(expected, np.nan), where=expected > 0)
    early = g[:max(1, bins // 6)]

    return {
        "pair_samples": int(n),
        "pair_r_max": float(r_max),
        "pair_near_g": _finite_mean(early),
        "pair_min_g": _finite_or_none(float(np.nanmin(g)) if np.isfinite(g).any() else None),
        "pair_peak_g": _finite_or_none(float(np.nanmax(g)) if np.isfinite(g).any() else None),
    }


def _nearest_vectors(points: np.ndarray, anchor_indices: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if anchor_indices is None:
        anchor_indices = np.arange(len(points), dtype=np.int64)
    else:
        anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    if len(points) < 2 or len(anchor_indices) == 0:
        return np.empty((0, 2), dtype=np.float64)

    max_elems = 8_000_000
    chunk = max(1, min(len(anchor_indices), max_elems // max(len(points), 1)))
    nearest = np.empty(len(anchor_indices), dtype=np.int64)
    norms = np.einsum("ij,ij->i", points, points)
    for start in range(0, len(anchor_indices), chunk):
        end = min(start + chunk, len(anchor_indices))
        idx = anchor_indices[start:end]
        block = points[idx]
        d2 = block @ points.T
        d2 *= -2.0
        d2 += norms[idx, None]
        d2 += norms[None, :]
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(end - start), idx] = np.inf
        nearest[start:end] = d2.argmin(axis=1)
    return points[nearest] - points[anchor_indices]


def nearest_spacing_metrics(
    points: np.ndarray,
    *,
    max_points: int = 2500,
    seed: int = 0,
) -> dict[str, float | int | None]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return {
            "nn_samples": int(len(points)),
            "nn_mean": None,
            "nn_p05": None,
            "nn_cv": None,
        }
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        points = points[np.sort(rng.choice(len(points), size=max_points, replace=False))]
    vectors = _nearest_vectors(points)
    distances = np.linalg.norm(vectors, axis=1)
    mean = float(distances.mean())
    return {
        "nn_samples": int(len(points)),
        "nn_mean": mean,
        "nn_p05": float(np.percentile(distances, 5.0)),
        "nn_cv": float(distances.std() / max(mean, 1e-12)),
    }


def edge_anisotropy_signature(
    points: np.ndarray,
    tensor: st.StructureTensor,
    *,
    coherence_min: float = 0.35,
    max_points: int = 3000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Measure nearest-neighbor displacement components near coherent edge pixels."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return {
            "edge_samples": 0,
            "edge_nn_tangent": None,
            "edge_nn_normal": None,
            "edge_nn_tangent_normal_ratio": None,
        }

    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        points = points[np.sort(rng.choice(len(points), size=max_points, replace=False))]

    coherence = np.clip(_sample_map(tensor.coherence, points), 0.0, 1.0)
    labels = _sample_map(tensor.label, points)
    anchor_indices = np.nonzero((labels == 1) & (coherence >= coherence_min))[0]
    if len(anchor_indices) < 2:
        anchor_indices = np.nonzero(coherence >= coherence_min)[0]
    if len(anchor_indices) == 0:
        return {
            "edge_samples": 0,
            "edge_nn_tangent": None,
            "edge_nn_normal": None,
            "edge_nn_tangent_normal_ratio": None,
        }

    vectors = _nearest_vectors(points, anchor_indices)
    tangent_angle = _sample_map(tensor.along_edge_angle, points[anchor_indices])
    normal_angle = _sample_map(tensor.across_edge_angle, points[anchor_indices])
    tangent = np.stack([np.cos(tangent_angle), np.sin(tangent_angle)], axis=1)
    normal = np.stack([np.cos(normal_angle), np.sin(normal_angle)], axis=1)
    tangent_abs = np.abs(np.einsum("ij,ij->i", vectors, tangent))
    normal_abs = np.abs(np.einsum("ij,ij->i", vectors, normal))
    tangent_mean = float(tangent_abs.mean())
    normal_mean = float(normal_abs.mean())
    ratio = None if normal_mean <= 1e-12 else tangent_mean / normal_mean
    return {
        "edge_samples": int(len(anchor_indices)),
        "edge_nn_tangent": tangent_mean,
        "edge_nn_normal": normal_mean,
        "edge_nn_tangent_normal_ratio": ratio,
    }


def scale_axis_metrics(
    points: np.ndarray,
    scales: np.ndarray,
    tensor: st.StructureTensor,
    *,
    max_axis_ratio: float,
    coherence_power: float,
    anisotropic: bool,
) -> dict[str, float | int | None]:
    ratio = np.maximum(scales[:, 0] / np.maximum(scales[:, 1], 1e-12), 1.0)
    coherence = np.clip(_sample_map(tensor.coherence, points), 0.0, 1.0)
    labels = _sample_map(tensor.label, points)
    edge = (labels == 1) & (coherence > 0.1)
    expected = (
        1.0 + (float(max_axis_ratio) - 1.0) * (coherence ** float(coherence_power))
        if anisotropic
        else np.ones_like(coherence)
    )
    return {
        "coherence_mean": float(coherence.mean()) if len(coherence) else None,
        "coherence_edge_mean": _finite_mean(coherence[edge]),
        "axis_ratio_mean": float(ratio.mean()) if len(ratio) else None,
        "axis_ratio_p95": float(np.percentile(ratio, 95.0)) if len(ratio) else None,
        "axis_ratio_edge_mean": _finite_mean(ratio[edge]),
        "axis_ratio_rmse_vs_mapping": float(np.sqrt(np.mean((ratio - expected) ** 2)))
        if len(ratio)
        else None,
    }


def analyze_field(
    points: np.ndarray,
    scales: np.ndarray,
    tensor: st.StructureTensor,
    *,
    max_axis_ratio: float,
    coherence_power: float,
    anisotropic: bool,
    bins: int = 40,
    seed: int = 0,
) -> dict[str, Any]:
    height, width = tensor.energy.shape
    profile = radial_power_profile(points, height, width, bins=bins)
    metrics: dict[str, Any] = {}
    metrics.update(spectral_trough_metrics(profile))
    metrics.update(
        pair_correlation_metrics(points, height, width, bins=max(12, bins // 2), seed=seed)
    )
    metrics.update(nearest_spacing_metrics(points, seed=seed))
    metrics.update(edge_anisotropy_signature(points, tensor, seed=seed))
    metrics.update(
        scale_axis_metrics(
            points,
            scales,
            tensor,
            max_axis_ratio=max_axis_ratio,
            coherence_power=coherence_power,
            anisotropic=anisotropic,
        )
    )
    return {k: _finite_or_none(v) if isinstance(v, float) else v for k, v in metrics.items()}


def _strategy_sweep_values(
    strategy: str,
    max_axis_ratios: list[float],
    coherence_powers: list[float],
) -> tuple[list[float], list[float]]:
    if strategy in ANISOTROPIC_STRATEGIES:
        return max_axis_ratios, coherence_powers
    return [max_axis_ratios[0]], [coherence_powers[0]]


def run_analysis(
    images: Iterable[str | Path],
    *,
    outdir: str | Path = "results/init_spectral_analysis",
    strategies: Iterable[str] = ("iso_blue_noise", "aniso_onedge", "aniso_flanking"),
    num_gaussians: int = 2048,
    max_axis_ratios: Iterable[float] = (2.0, 4.0, 6.0, 8.0),
    coherence_powers: Iterable[float] = (0.5, 1.0, 2.0),
    seed: int = 0,
    candidate_oversample: float = 4.0,
    sampling_mode: str = "wse",
    density_mode: str = "structure",
    max_side: int | None = 320,
    bins: int = 40,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    image_files = _iter_images(images)
    if not image_files:
        raise SystemExit("no images found")
    strategies = list(strategies)
    max_axis_ratios = [float(v) for v in max_axis_ratios]
    coherence_powers = [float(v) for v in coherence_powers]
    if not max_axis_ratios or not coherence_powers:
        raise ValueError("axis-ratio and coherence-power sweeps must be non-empty")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = run_config(
        {
            "images": [str(p) for p in image_files],
            "strategies": strategies,
            "num_gaussians": num_gaussians,
            "max_axis_ratios": max_axis_ratios,
            "coherence_powers": coherence_powers,
            "seed": seed,
            "candidate_oversample": candidate_oversample,
            "sampling_mode": sampling_mode,
            "density_mode": density_mode,
            "max_side": max_side,
            "bins": bins,
        },
        device=device,
    )
    write_config(str(outdir), config, name="init_spectral_config.json")

    scfg = StructureTensorConfig()
    rows: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    for image_path in image_files:
        img = load_image(image_path, max_side=max_side)
        tensor = st.compute(img, scfg)
        height, width = img.shape[:2]
        image_name = image_path.stem
        for strategy in strategies:
            ratios, powers = _strategy_sweep_values(strategy, max_axis_ratios, coherence_powers)
            for ratio_cap in ratios:
                for coherence_power in powers:
                    icfg = InitConfig(
                        strategy=strategy,
                        num_gaussians=int(num_gaussians),
                        candidate_oversample=float(candidate_oversample),
                        density_mode=density_mode,
                        sampling_mode=sampling_mode,
                        max_axis_ratio=float(ratio_cap),
                        coherence_power=float(coherence_power),
                        seed=int(seed),
                    )
                    field = init_mod.build_field(img, icfg, scfg=scfg, tensor=tensor, device=device)
                    points = field.means.detach().cpu().numpy()
                    scales = field.scales().detach().cpu().numpy()
                    anisotropic = strategy in ANISOTROPIC_STRATEGIES
                    metrics = analyze_field(
                        points,
                        scales,
                        tensor,
                        max_axis_ratio=ratio_cap,
                        coherence_power=coherence_power,
                        anisotropic=anisotropic,
                        bins=bins,
                        seed=seed,
                    )
                    key = f"{image_name}/{strategy}/r{ratio_cap:g}/c{coherence_power:g}"
                    profiles[key] = radial_power_profile(points, height, width, bins=bins)
                    row = {
                        "image": image_name,
                        "source_path": str(image_path),
                        "height": height,
                        "width": width,
                        "strategy": strategy,
                        "num_gaussians": int(num_gaussians),
                        "n_out": int(len(points)),
                        "seed": int(seed),
                        "sampling_mode": sampling_mode,
                        "density_mode": density_mode,
                        "max_axis_ratio": float(ratio_cap),
                        "coherence_power": float(coherence_power),
                        **metrics,
                    }
                    rows.append(row)
                    print(row)

    write_json(outdir / "spectral_profiles.json", profiles)
    write_json(outdir / "init_spectral_metrics.json", {"config": config, "rows": rows})
    write_csv(outdir / "init_spectral_metrics.csv", rows)
    write_summary(rows, outdir / "summary.md")
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(fv):
        return "-"
    return f"{fv:.{digits}f}"


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["strategy"]),
            float(row["max_axis_ratio"]),
            float(row["coherence_power"]),
        )
        groups.setdefault(key, []).append(row)
    out = []
    fields = [
        "spectral_low_mid_ratio",
        "pair_near_g",
        "nn_p05",
        "edge_nn_tangent_normal_ratio",
        "axis_ratio_edge_mean",
        "axis_ratio_rmse_vs_mapping",
    ]
    for (strategy, ratio, power), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "strategy": strategy,
            "max_axis_ratio": ratio,
            "coherence_power": power,
            "runs": len(items),
        }
        for field in fields:
            vals = [float(item[field]) for item in items if item.get(field) is not None]
            row[field] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


def write_summary(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    lines = ["# INIT-003 Spectral / Anisotropy Analysis", ""]
    lines.append(
        "Lower spectral_low_mid_ratio and pair_near_g indicate a stronger low-frequency "
        "blue-noise trough/exclusion. edge_nn_tangent_normal_ratio reports directional "
        "nearest-neighbor spacing near coherent edges; values away from 1 indicate an "
        "anisotropic edge signature."
    )
    lines.append("")
    lines.extend(
        [
            "| strategy | max_axis_ratio | coherence_power | runs | spectral low/mid | "
            "pair near g | nn p05 | edge tangent/normal | edge scale ratio | mapping RMSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _aggregate(rows):
        lines.append(
            f"| {row['strategy']} | {_fmt(row['max_axis_ratio'], 1)} | "
            f"{_fmt(row['coherence_power'], 2)} | {row['runs']} | "
            f"{_fmt(row['spectral_low_mid_ratio'])} | {_fmt(row['pair_near_g'])} | "
            f"{_fmt(row['nn_p05'])} | {_fmt(row['edge_nn_tangent_normal_ratio'])} | "
            f"{_fmt(row['axis_ratio_edge_mean'])} | "
            f"{_fmt(row['axis_ratio_rmse_vs_mapping'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze INIT-003 initializer spectra and edge anisotropy without fitting."
    )
    parser.add_argument("images", nargs="+")
    parser.add_argument("--outdir", default="results/init_spectral_analysis")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["iso_blue_noise", "aniso_onedge", "aniso_flanking"],
        choices=init_mod.STRATEGIES,
    )
    parser.add_argument("--num-gaussians", type=int, default=2048)
    parser.add_argument("--max-axis-ratios", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--coherence-powers", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-oversample", type=float, default=4.0)
    parser.add_argument("--sampling-mode", default="wse", choices=init_mod.SAMPLING_MODES)
    parser.add_argument("--density-mode", default="structure")
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_analysis(
        args.images,
        outdir=args.outdir,
        strategies=args.strategies,
        num_gaussians=args.num_gaussians,
        max_axis_ratios=args.max_axis_ratios,
        coherence_powers=args.coherence_powers,
        seed=args.seed,
        candidate_oversample=args.candidate_oversample,
        sampling_mode=args.sampling_mode,
        density_mode=args.density_mode,
        max_side=args.max_side,
        bins=args.bins,
        device=args.device,
    )


if __name__ == "__main__":
    main()
