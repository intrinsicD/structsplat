#!/usr/bin/env python3
"""Exact-7k masked boundary/thin-detail allocation diagnostic (HIER-031)."""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import asdict, replace
from html import escape
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier022_additive_continuation as h22  # noqa: E402
from scripts.experiments import hier029_janelle_mask_diagnostic as h29  # noqa: E402
from scripts.experiments import hier030_janelle_7k_contained_diagnostic as h30  # noqa: E402
from structsplat import mask as mask_geometry  # noqa: E402
from structsplat.cli import save_error_heatmap, save_image  # noqa: E402
from structsplat.detail_pursuit import _laplacian, gaussian_blur  # noqa: E402
from structsplat.endpoint_appearance_projection import project_additive_endpoint  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402


REPORT_SCHEMA = "structsplat.hier031_exact7k_masked_boundary_detail.diagnostic.v1"
SOURCE_SHA256 = h30.SOURCE_SHA256
MASK_SHA256 = h30.MASK_SHA256
NATIVE_SHAPE = h30.NATIVE_SHAPE
EVALUATION_SHAPE = h30.EVALUATION_SHAPE
CAPACITY = 7_000
MASK_MARGIN = 0.75
SIGMA_CUTOFF = 3.0
ORDINARY_MIN_SCALE = 0.35
MICRO_SCALE = 0.08
MICRO_CERTIFICATE_RADIUS = MASK_MARGIN + SIGMA_CUTOFF * MICRO_SCALE
DETAIL_ROWS = 768
DETAIL_BATCH = 128
DETAIL_DEEP_OFFSET = 6.0
DETAIL_BLUR_SIGMA = 1.5
DETAIL_NMS_RADIUS = 2
CONTAINMENT_LIMIT = 1e-7
FOUR_ARRAY_KEYS = frozenset(("means", "log_scales", "rotations", "colors"))
ARMS = (
    "hier030_cold_additive_n7000",
    "micro_hole_reallocation_n7000",
    "detail_reallocation_n7000",
    "combined_micro_detail_n7000",
    "merge_funded_micro_n7000",
    "merge_funded_micro_exempt_n7000",
    "merge_micro_geometry_recovery_n7000",
    "geometry_recovery_terminal_closure_n7000",
    "coverage_constrained_geometry_recovery_n7000",
    "deep_only_geometry_recovery_n7000",
    "deep_only_terminal_closure_n7000",
    "pipeline_fixed_n7000",
    "pipeline_boundary_recycle_n7000",
)
ADDITIVE_ARMS = frozenset(ARMS[:11])
BASE_BUNDLE = ROOT / "results/hier030_janelle_c0001_s1200_7k_contained_s0_diagnostic_2026-08-11"
BASE_FIELD_REL = Path(
    "artifacts/C0001__masked_contained__s0__cold_additive_projected_n7000/field.gaussian.npz"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--base-bundle", type=Path, default=BASE_BUNDLE)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
        help="Frozen arm subset; the final evidence command runs all arms.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "max_side": 1200,
        "seed": 0,
        "device": "cuda",
        "render_chunk": 256,
        "lpips": True,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-031 diagnostic requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    for name in ("image", "mask"):
        if not getattr(args, name).is_file():
            raise SystemExit(f"{name} does not exist: {getattr(args, name)}")
    if h22.report_utils._sha256(args.image) != SOURCE_SHA256:
        raise SystemExit("Janelle source SHA-256 differs from the frozen HIER-031 binding")
    if h22.report_utils._sha256(args.mask) != MASK_SHA256:
        raise SystemExit("Janelle mask SHA-256 differs from the frozen HIER-031 binding")
    base_field = args.base_bundle / BASE_FIELD_REL
    if not base_field.is_file() or not (args.base_bundle / "COMPLETED").is_file():
        raise SystemExit(f"immutable HIER-030 base bundle is unavailable: {args.base_bundle}")


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _pure_field(field: GaussianField) -> GaussianField:
    return GaussianField(
        field.means.detach().clone(),
        field.log_scales.detach().clone(),
        field.rotations.detach().clone(),
        field.colors.detach().clone(),
    )


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "scripts/check_report_bundle.py",
        ROOT / "src/structsplat/pipeline.py",
        ROOT / "src/structsplat/safe_schedule.py",
        ROOT / "src/structsplat/detail_pursuit.py",
        ROOT / "src/structsplat/endpoint_appearance_projection.py",
        ROOT / "src/structsplat/fit.py",
        ROOT / "src/structsplat/mask.py",
        ROOT / "tasks/HIER-031-exact7k-masked-boundary-detail-allocation.md",
    )
    records: list[dict[str, object]] = []
    for source in sources:
        destination = output_root / "source_snapshot" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(source.relative_to(ROOT)),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": h22.report_utils._sha256(destination),
            }
        )
    return records


def _component_summary(binary: np.ndarray) -> dict[str, int]:
    remaining = np.asarray(binary, dtype=bool).copy()
    height, width = remaining.shape
    sizes: list[int] = []
    for y0, x0 in np.argwhere(remaining):
        if not remaining[y0, x0]:
            continue
        remaining[y0, x0] = False
        queue: deque[tuple[int, int]] = deque([(int(y0), int(x0))])
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and remaining[yy, xx]:
                        remaining[yy, xx] = False
                        queue.append((yy, xx))
        sizes.append(size)
    return {
        "components": len(sizes),
        "largest_component": max(sizes, default=0),
        "singleton_components": sum(size == 1 for size in sizes),
    }


def _ridge_mask(sdf: np.ndarray, inside: np.ndarray) -> np.ndarray:
    padded = np.pad(sdf, 1, mode="constant", constant_values=-1e12)
    maximum = np.full_like(sdf, -1e12)
    for dy in range(3):
        for dx in range(3):
            if dx == 1 and dy == 1:
                continue
            maximum = np.maximum(
                maximum,
                padded[dy : dy + sdf.shape[0], dx : dx + sdf.shape[1]],
            )
    return inside & (sdf >= maximum - 1e-9)


def _feasibility_audit(inside: np.ndarray) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    sdf = mask_geometry.signed_distance(inside)
    ordinary_erode_radius = MASK_MARGIN + SIGMA_CUTOFF * ORDINARY_MIN_SCALE
    eroded = inside & (sdf >= ordinary_erode_radius)
    nearest_eroded = np.sqrt(mask_geometry.squared_edt(eroded))
    isotropic_unreachable = inside & (
        nearest_eroded > SIGMA_CUTOFF * ORDINARY_MIN_SCALE + 1e-9
    )
    ridge = _ridge_mask(sdf, inside)

    # A connected component without any admissible centre cannot be covered by a support wholly
    # contained in that component, regardless of row count or tangent elongation.
    component_labels = np.zeros(inside.shape, dtype=np.int32)
    component_records: list[dict[str, object]] = []
    remaining = inside.copy()
    label = 0
    for y0, x0 in np.argwhere(remaining):
        if not remaining[y0, x0]:
            continue
        label += 1
        remaining[y0, x0] = False
        queue: deque[tuple[int, int]] = deque([(int(y0), int(x0))])
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            component_labels[y, x] = label
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < inside.shape[0]
                        and 0 <= xx < inside.shape[1]
                        and remaining[yy, xx]
                    ):
                        remaining[yy, xx] = False
                        queue.append((yy, xx))
        values = np.asarray([sdf[y, x] for y, x in pixels])
        component_records.append(
            {
                "label": label,
                "pixels": len(pixels),
                "max_sdf": float(values.max()),
                "ordinary_admissible_centres": int(
                    sum(bool(eroded[y, x]) for y, x in pixels)
                ),
            }
        )
    centreless = [
        record for record in component_records if record["ordinary_admissible_centres"] == 0
    ]
    sdf_bins = {}
    active_sdf = sdf[inside]
    for threshold in (1.0, math.sqrt(2.0) + 1e-6, 1.8, 2.0, 3.0, 4.0, 6.0, 8.0):
        sdf_bins[f"le_{threshold:.4f}"] = int(np.count_nonzero(active_sdf <= threshold))
    record: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "active_pixels": int(inside.sum()),
        "active_fraction": float(inside.mean()),
        "ordinary_min_scale_px": ORDINARY_MIN_SCALE,
        "ordinary_erode_radius_px": ordinary_erode_radius,
        "ordinary_support_radius_px": SIGMA_CUTOFF * ORDINARY_MIN_SCALE,
        "ordinary_admissible_centres": int(eroded.sum()),
        "isotropic_unreachable_pixels": int(isotropic_unreachable.sum()),
        "isotropic_unreachable_fraction": float(isotropic_unreachable.sum() / inside.sum()),
        "components": len(component_records),
        "centreless_components": len(centreless),
        "centreless_component_pixels": int(sum(int(row["pixels"]) for row in centreless)),
        "centreless_component_records": centreless,
        "micro_scale_px": MICRO_SCALE,
        "micro_certificate_radius_px": MICRO_CERTIFICATE_RADIUS,
        "micro_all_mask_centres_certified": bool(MICRO_CERTIFICATE_RADIUS <= 1.0),
        "ridge_pixels": int(ridge.sum()),
        "thin_ridge_pixels_sdf_le_2": int((ridge & (sdf <= 2.0)).sum()),
        "thin_ridge_pixels_sdf_le_3": int((ridge & (sdf <= 3.0)).sum()),
        "sdf_bins": sdf_bins,
        "interpretation": (
            "isotropic_unreachable is an upper bound because certified tangent support may reach "
            "some sites; centreless component pixels are a hard current-scale lower bound"
        ),
    }
    return record, {
        "sdf": sdf,
        "eroded": eroded,
        "isotropic_unreachable": isotropic_unreachable,
        "ridge": ridge,
        "component_labels": component_labels,
    }


def _render(field: GaussianField, renderer: str, shape: tuple[int, int], args, torch):
    with torch.no_grad():
        return h30._render(
            field,
            shape[0],
            shape[1],
            renderer,
            args.render_chunk,
            support_fade=True,
        )


def _coverage(field: GaussianField, shape: tuple[int, int], args, torch) -> np.ndarray:
    return h30._unit_coverage(
        field,
        shape[0],
        shape[1],
        args,
        torch,
        support_fade=True,
    )


def _projection_record(result) -> dict[str, object]:
    projection = result.projection
    return {
        "selected_iteration": projection.selected_iteration,
        "initial_sse": projection.initial_sse,
        "final_sse": projection.final_sse,
        "forward_applications": projection.forward_applications,
        "transpose_applications": projection.transpose_applications,
        "relative_normal_residual_max": projection.relative_normal_residual_max,
        "adjoint_relative_error": projection.adjoint_relative_error,
        "maintained_render_parity_max_abs": projection.maintained_render_parity_max_abs,
        "elapsed_seconds": projection.elapsed_seconds,
        "checkpoints": projection.checkpoint_records(),
    }


def _donor_order(
    field: GaussianField,
    coverage: np.ndarray,
    sdf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = field.means.detach().cpu().numpy()
    scales = np.exp(field.log_scales.detach().cpu().numpy())
    colors = field.colors.detach().cpu().numpy()
    ix = np.rint(means[:, 0]).astype(np.int64).clip(0, coverage.shape[1] - 1)
    iy = np.rint(means[:, 1]).astype(np.int64).clip(0, coverage.shape[0] - 1)
    local_coverage = np.maximum(coverage[iy, ix], 1e-6)
    # Column-energy / local-overlap is a deterministic redundancy proxy: low coefficient energy,
    # small footprint, and high local overlap are cheapest to exchange. Micro rows are protected.
    cost = (
        np.mean(np.square(colors.astype(np.float64)), axis=1)
        * np.prod(scales.astype(np.float64), axis=1)
        / local_coverage.astype(np.float64)
    )
    ordinary = np.min(scales, axis=1) > 0.20
    deep = sdf[iy, ix] > MASK_MARGIN + DETAIL_DEEP_OFFSET
    cohorts = (
        np.flatnonzero(ordinary & deep & (local_coverage >= 1.0)),
        np.flatnonzero(ordinary & deep & (local_coverage < 1.0)),
        np.flatnonzero(ordinary & ~deep),
    )
    ordered: list[np.ndarray] = []
    used = np.zeros(field.n, dtype=bool)
    for cohort in cohorts:
        cohort = cohort[~used[cohort]]
        cohort = cohort[np.argsort(cost[cohort], kind="stable")]
        used[cohort] = True
        ordered.append(cohort)
    rest = np.flatnonzero(~used)
    rest = rest[np.argsort(cost[rest], kind="stable")]
    ordered.append(rest)
    return np.concatenate(ordered), cost


def _swap_sites_additive(
    field: GaussianField,
    target: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    sites: np.ndarray,
    scale: float,
    args: argparse.Namespace,
    torch,
) -> tuple[GaussianField, dict[str, object]]:
    sites = np.asarray(sites, dtype=np.int64).reshape(-1)
    if not sites.size:
        raise ValueError("site exchange requires at least one site")
    if len(np.unique(sites)) != len(sites):
        raise ValueError("site exchange contains duplicate sites")
    if field.n != CAPACITY:
        raise ValueError(f"site exchange requires exactly {CAPACITY} rows")
    rendered = _render(field, "cuda_additive", inside.shape, args, torch)
    reconstruction = rendered.detach().cpu().numpy()
    coverage = _coverage(field, inside.shape, args, torch)
    order, donor_cost = _donor_order(field, coverage, sdf)
    donor = order[: len(sites)]
    keep = np.ones(field.n, dtype=bool)
    keep[donor] = False
    inherited = field.subset(
        torch.as_tensor(np.flatnonzero(keep), device=field.means.device, dtype=torch.long)
    )
    y = sites // inside.shape[1]
    x = sites - y * inside.shape[1]
    if not np.all(inside[y, x]):
        raise ValueError("site exchange selected pixels outside the mask")
    if scale == MICRO_SCALE and not np.all(sdf[y, x] >= MICRO_CERTIFICATE_RADIUS):
        raise ValueError("micro sites fail the per-site support certificate")
    peak = 1.0 - math.exp(-0.5 * SIGMA_CUTOFF**2)
    initial_colors = (target - reconstruction)[y, x] / peak
    components = GaussianField(
        means=torch.as_tensor(
            np.stack([x, y], axis=1),
            device=field.means.device,
            dtype=field.means.dtype,
        ),
        log_scales=torch.full(
            (len(sites), 2),
            math.log(float(scale)),
            device=field.means.device,
            dtype=field.means.dtype,
        ),
        rotations=torch.zeros(
            len(sites), device=field.means.device, dtype=field.means.dtype
        ),
        colors=torch.as_tensor(
            initial_colors,
            device=field.means.device,
            dtype=field.means.dtype,
        ),
    )
    proposal = inherited.append(components)
    if proposal.n != CAPACITY:
        raise RuntimeError("count-neutral exchange changed the row count")
    projection = project_additive_endpoint(
        proposal,
        target,
        config=h30._projection_config(args, contained=True),
        device=args.device,
        mask=inside,
    )
    output = _pure_field(projection.field)
    return output, {
        "sites": sites.tolist(),
        "site_count": int(len(sites)),
        "scale_px": float(scale),
        "donor_rows": donor.tolist(),
        "donor_cost_min": float(donor_cost[donor].min()),
        "donor_cost_mean": float(donor_cost[donor].mean()),
        "donor_cost_max": float(donor_cost[donor].max()),
        "initial_color_abs_max": float(np.abs(initial_colors).max()),
        "projection": _projection_record(projection),
    }


def _micro_reallocation(
    base: GaussianField,
    target: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
    *,
    max_waves: int = 4,
) -> tuple[GaussianField, list[dict[str, object]]]:
    field = _pure_field(base)
    history: list[dict[str, object]] = []
    for wave in range(max_waves):
        coverage = _coverage(field, inside.shape, args, torch)
        sites = np.flatnonzero((inside & (coverage <= 0.0)).reshape(-1))
        record: dict[str, object] = {
            "wave": wave,
            "holes_before": int(len(sites)),
        }
        if not len(sites):
            record["status"] = "zero_holes"
            history.append(record)
            break
        field, exchange = _swap_sites_additive(
            field, target, inside, sdf, sites, MICRO_SCALE, args, torch
        )
        after = _coverage(field, inside.shape, args, torch)
        record.update(exchange)
        record["holes_after"] = int((inside & (after <= 0.0)).sum())
        record["status"] = "accepted"
        history.append(record)
    return field, history


def _merge_swap_sites_additive(
    field: GaussianField,
    target: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    sites: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> tuple[GaussianField, dict[str, object]]:
    """Fund micro sites by envelope-merging disjoint ordinary mutual-nearest pairs."""

    from structsplat.fit import _MaskConstraint
    from structsplat.safe_schedule import _production_mutual_nearest_pairs
    from structsplat.triage import _row_covariances

    sites = np.asarray(sites, dtype=np.int64).reshape(-1)
    if not len(sites):
        raise ValueError("merge-funded exchange requires sites")
    first, second, distance = _production_mutual_nearest_pairs(field.means.detach())
    if first.numel() < len(sites):
        raise RuntimeError("too few mutual-nearest pairs to fund the requested sites")
    scales = field.scales().detach()
    colors = field.colors.detach()
    ordinary = (scales[first].amin(dim=1) > 0.20) & (scales[second].amin(dim=1) > 0.20)
    means = field.means.detach()
    ia = means[first].round().long()
    ib = means[second].round().long()
    sdf_tensor = torch.as_tensor(sdf, device=means.device, dtype=means.dtype)
    depth = (sdf_tensor[ia[:, 1], ia[:, 0]] > 2.0) & (
        sdf_tensor[ib[:, 1], ib[:, 0]] > 2.0
    )
    color_delta = torch.linalg.norm(colors[first] - colors[second], dim=1)
    sorted_scales = torch.sort(scales, dim=1).values.clamp_min(1e-6)
    scale_delta = torch.linalg.norm(
        torch.log(sorted_scales[first]) - torch.log(sorted_scales[second]), dim=1
    )
    angle_delta = field.rotations.detach()[first] - field.rotations.detach()[second]
    axial_delta = 0.5 * torch.abs(
        torch.atan2(torch.sin(2.0 * angle_delta), torch.cos(2.0 * angle_delta))
    )
    pair_scale = 0.5 * (
        torch.sqrt(scales[first].prod(dim=1))
        + torch.sqrt(scales[second].prod(dim=1))
    ).clamp_min(0.35)
    score = (
        distance / pair_scale
        + 0.25 * color_delta
        + 0.10 * scale_delta
        + 0.10 * axial_delta
    )
    valid = ordinary & depth
    candidates = valid.nonzero(as_tuple=False).reshape(-1)
    if candidates.numel() < len(sites):
        raise RuntimeError(
            f"only {int(candidates.numel())} SDF>2 ordinary pairs for {len(sites)} sites"
        )
    selected = candidates[torch.argsort(score[candidates])[: len(sites)]]
    keep_rows = first[selected]
    absorbed_rows = second[selected]
    midpoint = 0.5 * (means[keep_rows] + means[absorbed_rows])
    cov_a = _row_covariances(field, keep_rows)
    cov_b = _row_covariances(field, absorbed_rows)
    da = means[keep_rows] - midpoint
    db = means[absorbed_rows] - midpoint
    spatial_a = cov_a + da[:, :, None] * da[:, None, :]
    spatial_b = cov_b + db[:, :, None] * db[:, None, :]
    moment = 0.5 * (spatial_a + spatial_b)
    values, vectors = torch.linalg.eigh(moment)
    inverse_sqrt = (
        vectors
        @ torch.diag_embed(values.clamp_min(1e-8).rsqrt())
        @ vectors.transpose(1, 2)
    )
    envelope = torch.maximum(
        torch.linalg.eigvalsh(inverse_sqrt @ spatial_a @ inverse_sqrt)[:, -1],
        torch.linalg.eigvalsh(inverse_sqrt @ spatial_b @ inverse_sqrt)[:, -1],
    ).clamp_min(1.0)
    covariance = moment * envelope[:, None, None] * 1.05**2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(ORDINARY_MIN_SCALE**2)
    major = eigenvectors[:, :, 1]
    merged_scales = torch.stack(
        [torch.sqrt(eigenvalues[:, 1]), torch.sqrt(eigenvalues[:, 0])], dim=1
    )
    area_a = scales[keep_rows].prod(dim=1)
    area_b = scales[absorbed_rows].prod(dim=1)
    merged_area = merged_scales.prod(dim=1).clamp_min(1e-8)
    merged_colors = (
        area_a[:, None] * colors[keep_rows]
        + area_b[:, None] * colors[absorbed_rows]
    ) / merged_area[:, None]
    trial = field.detached()
    trial.means[keep_rows] = midpoint
    trial.log_scales[keep_rows] = torch.log(merged_scales)
    trial.rotations[keep_rows] = torch.atan2(major[:, 1], major[:, 0])
    trial.colors[keep_rows] = merged_colors
    retain = torch.ones(field.n, device=means.device, dtype=torch.bool)
    retain[absorbed_rows] = False
    funded = trial.subset(retain)

    fit_cfg = h30._fit_config(args, "cuda_additive", CAPACITY, contained=True)
    constraint = _MaskConstraint.from_mask(
        inside,
        means.device,
        means.dtype,
        SIGMA_CUTOFF,
        MASK_MARGIN,
        aa_dilation=0.0,
        min_scale=ORDINARY_MIN_SCALE,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    existing_micro = funded.scales().detach().amax(dim=1) <= 0.081
    constraint.apply(funded, fit_cfg, refresh=True, exempt=existing_micro)
    funded = _pure_field(funded)
    y = sites // inside.shape[1]
    x = sites - y * inside.shape[1]
    if not np.all(sdf[y, x] >= MICRO_CERTIFICATE_RADIUS):
        raise RuntimeError("merge-funded micro sites fail their support certificate")
    funded_render = _render(funded, "cuda_additive", inside.shape, args, torch)
    residual = target - funded_render.detach().cpu().numpy()
    peak = 1.0 - math.exp(-0.5 * SIGMA_CUTOFF**2)
    micro = GaussianField(
        means=torch.as_tensor(
            np.stack([x, y], axis=1), device=means.device, dtype=means.dtype
        ),
        log_scales=torch.full(
            (len(sites), 2), math.log(MICRO_SCALE), device=means.device, dtype=means.dtype
        ),
        rotations=torch.zeros(len(sites), device=means.device, dtype=means.dtype),
        colors=torch.as_tensor(residual[y, x] / peak, device=means.device, dtype=means.dtype),
    )
    proposal = funded.append(micro)
    if proposal.n != CAPACITY:
        raise RuntimeError(f"merge-funded proposal has {proposal.n} rows")
    projection = project_additive_endpoint(
        proposal,
        target,
        config=h30._projection_config(args, contained=True),
        device=args.device,
        mask=inside,
    )
    return _pure_field(projection.field), {
        "site_count": int(len(sites)),
        "sites": sites.tolist(),
        "candidate_pairs": int(first.numel()),
        "eligible_pairs": int(candidates.numel()),
        "pair_score_min": float(score[selected].min()),
        "pair_score_mean": float(score[selected].mean()),
        "pair_score_max": float(score[selected].max()),
        "merge_rule": "mutual nearest + 1.05x covariance envelope + anisotropic recertificate",
        "projection": _projection_record(projection),
    }


def _merge_funded_micro_reallocation(
    base: GaussianField,
    target: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
    *,
    max_waves: int = 4,
) -> tuple[GaussianField, list[dict[str, object]]]:
    field = _pure_field(base)
    history: list[dict[str, object]] = []
    for wave in range(max_waves):
        coverage = _coverage(field, inside.shape, args, torch)
        sites = np.flatnonzero((inside & (coverage <= 0.0)).reshape(-1))
        record: dict[str, object] = {"wave": wave, "holes_before": int(len(sites))}
        if not len(sites):
            record["status"] = "zero_holes"
            history.append(record)
            break
        try:
            field, exchange = _merge_swap_sites_additive(
                field, target, inside, sdf, sites, args, torch
            )
        except RuntimeError as exc:
            if "ordinary pairs" not in str(exc) and "mutual-nearest pairs" not in str(exc):
                raise
            record["status"] = "insufficient_pairs"
            record["error"] = str(exc)
            history.append(record)
            break
        after = _coverage(field, inside.shape, args, torch)
        record.update(exchange)
        record["holes_after"] = int((inside & (after <= 0.0)).sum())
        record["status"] = "accepted"
        history.append(record)
    return field, history


def _detail_sites(
    field: GaussianField,
    target: np.ndarray,
    sdf: np.ndarray,
    forbidden: np.ndarray,
    count: int,
    args: argparse.Namespace,
    torch,
) -> tuple[np.ndarray, dict[str, object]]:
    rendered = _render(field, "cuda_additive", sdf.shape, args, torch)
    target_tensor = torch.as_tensor(target, device=rendered.device, dtype=rendered.dtype)
    residual = rendered - target_tensor
    highpass = residual - gaussian_blur(residual, DETAIL_BLUR_SIGMA)
    raw = highpass.square().mean(dim=2)
    eligible_np = (sdf > MASK_MARGIN + DETAIL_DEEP_OFFSET) & ~forbidden
    eligible = torch.as_tensor(eligible_np, device=raw.device, dtype=torch.bool)
    negative = torch.full_like(raw, -float("inf"))
    score = torch.where(eligible, raw, negative)
    pooled = torch.nn.functional.max_pool2d(
        score[None, None],
        2 * DETAIL_NMS_RADIUS + 1,
        stride=1,
        padding=DETAIL_NMS_RADIUS,
    )[0, 0]
    peaks = (score >= pooled) & eligible
    peak_scores = torch.where(peaks, score, negative).reshape(-1)
    finite = int(torch.isfinite(peak_scores).sum())
    if finite < count:
        peak_scores = score.reshape(-1)
        finite = int(torch.isfinite(peak_scores).sum())
    selected = min(int(count), finite)
    ranked = torch.topk(peak_scores, selected, sorted=True)
    sites = ranked.indices.detach().cpu().numpy().astype(np.int64, copy=False)
    return sites, {
        "requested": int(count),
        "selected": int(selected),
        "eligible_pixels": int(eligible_np.sum()),
        "finite_candidates": finite,
        "score_min": float(ranked.values.min()),
        "score_mean": float(ranked.values.mean()),
        "score_max": float(ranked.values.max()),
        "blur_sigma": DETAIL_BLUR_SIGMA,
        "nms_radius": DETAIL_NMS_RADIUS,
        "deep_offset": DETAIL_DEEP_OFFSET,
    }


def _detail_reallocation(
    base: GaussianField,
    target: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> tuple[GaussianField, list[dict[str, object]]]:
    field = _pure_field(base)
    forbidden = np.zeros(inside.shape, dtype=bool)
    history: list[dict[str, object]] = []
    for wave in range(DETAIL_ROWS // DETAIL_BATCH):
        sites, selection = _detail_sites(
            field, target, sdf, forbidden, DETAIL_BATCH, args, torch
        )
        if len(sites) != DETAIL_BATCH:
            raise RuntimeError(
                f"detail wave {wave} selected {len(sites)} of {DETAIL_BATCH} sites"
            )
        forbidden.reshape(-1)[sites] = True
        field, exchange = _swap_sites_additive(
            field, target, inside, sdf, sites, ORDINARY_MIN_SCALE, args, torch
        )
        history.append({"wave": wave, "selection": selection, **exchange})
    return field, history


def _pipeline_arm(
    source: np.ndarray,
    inside: np.ndarray,
    args: argparse.Namespace,
    torch,
    *,
    boundary_recycle: bool,
) -> tuple[GaussianField, list[dict[str, object]], dict[str, object]]:
    from structsplat.pipeline import PipelineConfig, run_pipeline

    cfg = PipelineConfig(
        capacity=CAPACITY,
        initial_gaussians=CAPACITY,
        boundary_gaussians=700,
        coverage_target=CAPACITY,
        detail_target=CAPACITY,
        seed=args.seed,
        device=args.device,
        renderer="cuda",
    )

    def transform(schedule):
        return replace(
            schedule,
            boundary_recycle_at_capacity=bool(boundary_recycle),
        )

    result = run_pipeline(
        source,
        inside,
        cfg,
        schedule_transform=transform,
        verbose=not args.quiet,
    )
    field = _pure_field(result["field"])
    if field.n != CAPACITY:
        raise RuntimeError(f"fixed-N pipeline returned {field.n} rows, expected {CAPACITY}")
    metadata = {
        "pipeline_config": asdict(cfg),
        "schedule": result["schedule"],
        "storage": result["storage"],
        "attempted_steps": result["attempted_steps"],
        "accepted_steps": result["accepted_steps"],
        "timing": result["timing"],
        "boundary_recycle_at_capacity": bool(boundary_recycle),
    }
    return field, list(result["history"]), metadata


def _geometry_recovery(
    base: GaussianField,
    target: np.ndarray,
    inside: np.ndarray,
    args: argparse.Namespace,
    torch,
    *,
    coverage_constrained: bool = False,
    deep_only: bool = False,
    sdf: np.ndarray | None = None,
) -> tuple[GaussianField, dict[str, object], dict[str, object]]:
    from structsplat.fit import _MaskConstraint, fit

    field = _pure_field(base)
    micro = field.scales().detach().amax(dim=1) <= 0.081
    if int(micro.sum()) == 0:
        raise RuntimeError("geometry recovery base contains no certified micro cohort")
    trainable = ~micro
    if deep_only:
        if sdf is None:
            raise ValueError("deep-only geometry recovery requires the mask SDF")
        means = field.means.detach()
        x = means[:, 0].round().long().clamp(0, inside.shape[1] - 1)
        y = means[:, 1].round().long().clamp(0, inside.shape[0] - 1)
        sdf_tensor = torch.as_tensor(sdf, device=means.device, dtype=means.dtype)
        trainable &= sdf_tensor[y, x] > MASK_MARGIN + DETAIL_DEEP_OFFSET
    cfg = h30._fit_config(args, "cuda_additive", CAPACITY, contained=True)
    if coverage_constrained:
        cfg = replace(
            cfg,
            mask_undercoverage_weight=0.05,
            mask_undercoverage_band=4.0,
            mask_undercoverage_tau=0.05,
            mask_undercoverage_every=8,
        )
    constraint = _MaskConstraint.from_mask(
        inside,
        field.means.device,
        field.means.dtype,
        SIGMA_CUTOFF,
        MASK_MARGIN,
        aa_dilation=0.0,
        min_scale=ORDINARY_MIN_SCALE,
        cap_mode="anisotropic",
        undercoverage_band=4.0,
    )
    target_tensor = torch.as_tensor(target, device=args.device, dtype=torch.float32).contiguous()
    result = fit(
        field,
        target_tensor,
        cfg,
        mask=inside,
        trainable_row_mask=trainable,
        constraint_exempt_row_mask=micro,
        mask_constraint_override=constraint,
        verbose=not args.quiet,
    )
    fitted = _pure_field(result["field"])
    projection = project_additive_endpoint(
        fitted,
        target,
        config=h30._projection_config(args, contained=True),
        device=args.device,
        mask=inside,
    )
    metadata = {
        "micro_rows": int(micro.sum()),
        "ordinary_trainable_rows": int(trainable.sum()),
        "fit_config": asdict(cfg),
        "fit_seconds": result["fit_seconds"],
        "selected_step": result["selected_iter"],
        "selected_from_checkpoint": result["selected_from_checkpoint"],
        "projection": _projection_record(projection),
        "coverage_constrained": bool(coverage_constrained),
        "deep_only": bool(deep_only),
    }
    return _pure_field(projection.field), result["history"], metadata


def _coverage_metrics(
    coverage: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
    ridge: np.ndarray,
    isotropic_unreachable: np.ndarray,
) -> dict[str, object]:
    raw_holes = inside & (coverage <= 0.0)
    threshold_holes = inside & (coverage < 0.05)
    boundary = inside & (sdf <= 4.0)
    interior = inside & (sdf > 4.0)
    thin_ridge = ridge & (sdf <= 3.0)
    raw_components = _component_summary(raw_holes)
    threshold_components = _component_summary(threshold_holes)

    def fraction(selection: np.ndarray, domain: np.ndarray) -> float:
        return float(selection[domain].mean()) if bool(domain.any()) else 0.0

    return {
        "raw_hole_pixels": int(raw_holes.sum()),
        "raw_hole_fraction": fraction(raw_holes, inside),
        "raw_hole_reachable_isotropic_pixels": int(
            (raw_holes & ~isotropic_unreachable).sum()
        ),
        "raw_hole_isotropic_unreachable_pixels": int(
            (raw_holes & isotropic_unreachable).sum()
        ),
        "raw_hole_components": raw_components["components"],
        "raw_hole_largest_component": raw_components["largest_component"],
        "coverage_lt_005_pixels": int(threshold_holes.sum()),
        "coverage_lt_005_fraction": fraction(threshold_holes, inside),
        "coverage_lt_005_components": threshold_components["components"],
        "coverage_lt_005_largest_component": threshold_components["largest_component"],
        "boundary_raw_hole_fraction": fraction(raw_holes, boundary),
        "boundary_coverage_lt_005_fraction": fraction(threshold_holes, boundary),
        "interior_raw_hole_fraction": fraction(raw_holes, interior),
        "interior_coverage_lt_005_fraction": fraction(threshold_holes, interior),
        "ridge_raw_hole_fraction": fraction(raw_holes, ridge),
        "ridge_coverage_lt_005_fraction": fraction(threshold_holes, ridge),
        "thin_ridge_raw_hole_fraction": fraction(raw_holes, thin_ridge),
        "thin_ridge_coverage_lt_005_fraction": fraction(threshold_holes, thin_ridge),
        "coverage_inside_min": float(coverage[inside].min()),
        "coverage_inside_mean": float(coverage[inside].mean()),
        "coverage_inside_q01": float(np.quantile(coverage[inside], 0.01)),
        "coverage_inside_q05": float(np.quantile(coverage[inside], 0.05)),
        "coverage_inside_max": float(coverage[inside].max(initial=0.0)),
    }


def _regional_quality(
    reconstruction: np.ndarray,
    target: np.ndarray,
    inside: np.ndarray,
    sdf: np.ndarray,
) -> dict[str, float]:
    squared = np.square(
        reconstruction.astype(np.float64) - target.astype(np.float64)
    ).mean(axis=2)

    def record(prefix: str, domain: np.ndarray) -> dict[str, float]:
        mse = float(squared[domain].mean()) if bool(domain.any()) else 0.0
        return {
            f"{prefix}_mse": mse,
            f"{prefix}_psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        }

    return {
        **record("foreground", inside),
        **record("boundary_le4", inside & (sdf <= 4.0)),
        **record("interior_gt4", inside & (sdf > 4.0)),
        **record("deep_gt675", inside & (sdf > MASK_MARGIN + DETAIL_DEEP_OFFSET)),
    }


def _detail_metrics(
    reconstruction: np.ndarray,
    target: np.ndarray,
    sdf: np.ndarray,
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    rendered = torch.as_tensor(reconstruction, device=args.device, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, device=args.device, dtype=torch.float32)
    residual = rendered - target_tensor
    deep = torch.as_tensor(
        sdf > MASK_MARGIN + DETAIL_DEEP_OFFSET,
        device=args.device,
        dtype=torch.bool,
    )
    highpass = residual - gaussian_blur(residual, DETAIL_BLUR_SIGMA)
    laplacian = _laplacian(residual)
    sobel_x = torch.zeros_like(residual)
    sobel_y = torch.zeros_like(residual)
    sobel_x[:, 1:-1] = 0.5 * (residual[:, 2:] - residual[:, :-2])
    sobel_y[1:-1] = 0.5 * (residual[2:] - residual[:-2])
    return {
        "detail_deep_pixels": int(deep.sum()),
        "detail_highpass_mse": float(highpass[deep].square().mean()),
        "detail_laplacian_mse": float(laplacian[deep].square().mean()),
        "detail_sobel_mse": float(
            (sobel_x[deep].square() + sobel_y[deep].square()).mean()
        ),
    }


def _scaling_readiness(
    field: GaussianField,
    reconstruction: np.ndarray,
    target: np.ndarray,
    inside: np.ndarray,
) -> dict[str, object]:
    x0, y0, x1, y1 = h29._foreground_bounds(inside, padding=0)
    gray = target.astype(np.float64).mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1] = 0.5 * (gray[2:] - gray[:-2])
    complexity = np.hypot(gx, gy) + 0.02
    pixel_mse = np.square(
        reconstruction.astype(np.float64) - target.astype(np.float64)
    ).mean(axis=2)
    means = field.means.detach().cpu().numpy()
    mx = np.rint(means[:, 0]).astype(np.int64)
    my = np.rint(means[:, 1]).astype(np.int64)
    tiles: list[dict[str, object]] = []
    for row in range(8):
        ya = y0 + (y1 - y0) * row // 8
        yb = y0 + (y1 - y0) * (row + 1) // 8
        for col in range(8):
            xa = x0 + (x1 - x0) * col // 8
            xb = x0 + (x1 - x0) * (col + 1) // 8
            domain = np.zeros(inside.shape, dtype=bool)
            domain[ya:yb, xa:xb] = inside[ya:yb, xa:xb]
            pixels = int(domain.sum())
            if not pixels:
                continue
            row_count = int(
                np.count_nonzero((mx >= xa) & (mx < xb) & (my >= ya) & (my < yb))
            )
            sse = float(pixel_mse[domain].sum())
            complexity_sum = float(complexity[domain].sum())
            tiles.append(
                {
                    "row": row,
                    "column": col,
                    "pixels": pixels,
                    "gaussians": row_count,
                    "sse": sse,
                    "mse": sse / pixels,
                    "complexity": complexity_sum,
                    "complexity_normalized_error": sse / complexity_sum,
                    "gaussians_per_kpixel": 1000.0 * row_count / pixels,
                    # A common compact row placed on the largest-residual pixel has first-order
                    # value proportional to this peak. This is a comparable allocation proxy,
                    # not a fitted counterfactual.
                    "next_row_gain_proxy": float(pixel_mse[domain].max()),
                }
            )

    def cv(key: str) -> float:
        values = np.asarray([float(tile[key]) for tile in tiles], dtype=np.float64)
        mean = float(values.mean())
        return float(values.std() / mean) if mean > 0.0 else 0.0

    total_sse = sum(float(tile["sse"]) for tile in tiles)
    return {
        "tiles": tiles,
        "tile_count": len(tiles),
        "tile_sse_cv": cv("sse"),
        "tile_mse_cv": cv("mse"),
        "tile_complexity_normalized_error_cv": cv("complexity_normalized_error"),
        "tile_gaussian_density_cv": cv("gaussians_per_kpixel"),
        "tile_next_row_gain_proxy_cv": cv("next_row_gain_proxy"),
        "tile_max_sse_fraction": (
            max((float(tile["sse"]) for tile in tiles), default=0.0) / max(total_sse, 1e-12)
        ),
        "interpretation": (
            "lower next-row-gain CV means less spatial starvation; raw-error equality is not "
            "the objective because local image complexity differs"
        ),
    }


def _save_hair_crop(
    artifact_dir: Path,
    source: np.ndarray,
    reconstruction: np.ndarray,
    inside: np.ndarray,
    error_scale: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = h29._foreground_bounds(inside, padding=8)
    head_bottom = min(y1, y0 + max(96, int(round(0.38 * (y1 - y0)))))
    bounds = (x0, y0, x1, head_bottom)
    target = source * inside[..., None].astype(np.float32)
    error = np.repeat(
        np.clip(
            np.mean(
                np.abs(reconstruction.astype(np.float64) - target.astype(np.float64)),
                axis=2,
            )
            * error_scale,
            0.0,
            1.0,
        )[..., None],
        3,
        axis=2,
    )
    h22.viz_utils._save_crop(artifact_dir / "hair_source_crop.png", target, bounds)
    h22.viz_utils._save_crop(
        artifact_dir / "hair_reconstruction_crop.png", reconstruction, bounds
    )
    h22.viz_utils._save_crop(artifact_dir / "hair_error_crop.png", error, bounds)
    return bounds


def _save_objective_crops(
    artifact_dir: Path,
    source: np.ndarray,
    reconstruction: np.ndarray,
    inside: np.ndarray,
    bounds: tuple[int, int, int, int],
    error_scale: float,
) -> None:
    target = source * inside[..., None].astype(np.float32)
    error = np.repeat(
        np.clip(
            np.mean(
                np.abs(reconstruction.astype(np.float64) - target.astype(np.float64)),
                axis=2,
            )
            * error_scale,
            0.0,
            1.0,
        )[..., None],
        3,
        axis=2,
    )
    h22.viz_utils._save_crop(artifact_dir / "source_crop.png", target, bounds)
    h22.viz_utils._save_crop(
        artifact_dir / "reconstruction_crop.png", reconstruction, bounds
    )
    h22.viz_utils._save_crop(artifact_dir / "error_crop.png", error, bounds)


def _write_arm(
    *,
    output_root: Path,
    arm: str,
    method: dict[str, object],
    source: np.ndarray,
    inside: np.ndarray,
    geometry: dict[str, np.ndarray],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    artifact_dir = output_root / "artifacts" / arm
    artifact_dir.mkdir(parents=True, exist_ok=False)
    field = _pure_field(method["field"])
    if field.n != CAPACITY:
        raise RuntimeError(f"{arm} has {field.n} rows, expected {CAPACITY}")
    field_path = artifact_dir / "field.gaussian.npz"
    field.save(str(field_path))
    with np.load(field_path) as payload:
        field_keys = sorted(payload.files)
    if set(field_keys) != FOUR_ARRAY_KEYS:
        raise RuntimeError(f"{arm} persisted non-endpoint arrays: {field_keys}")
    decoded = GaussianField.load(str(field_path), device=args.device)
    renderer = str(method["renderer"])
    expected = _render(field, renderer, inside.shape, args, torch)
    cold = _render(decoded, renderer, inside.shape, args, torch)
    repeated = _render(decoded, renderer, inside.shape, args, torch)
    expected_np = expected.detach().cpu().numpy().astype(np.float32, copy=False)
    cold_np = cold.detach().cpu().numpy().astype(np.float32, copy=False)
    repeated_np = repeated.detach().cpu().numpy().astype(np.float32, copy=False)
    maintained_parity = float(np.max(np.abs(expected_np.astype(np.float64) - cold_np)))
    repeated_parity = float(np.max(np.abs(repeated_np.astype(np.float64) - cold_np)))
    target = source * inside[..., None].astype(np.float32)
    coverage = _coverage(decoded, inside.shape, args, torch)
    outside_reconstruction = np.abs(cold_np[~inside])
    outside_coverage = np.abs(coverage[~inside])
    means = decoded.means.detach().cpu().numpy()
    ix = np.rint(means[:, 0]).astype(np.int64)
    iy = np.rint(means[:, 1]).astype(np.int64)
    valid = (ix >= 0) & (ix < inside.shape[1]) & (iy >= 0) & (iy < inside.shape[0])
    centres_inside = np.zeros(decoded.n, dtype=bool)
    centres_inside[valid] = inside[iy[valid], ix[valid]]
    containment_pass = bool(
        centres_inside.all()
        and float(outside_reconstruction.max(initial=0.0)) <= CONTAINMENT_LIMIT
        and float(outside_coverage.max(initial=0.0)) <= CONTAINMENT_LIMIT
    )
    if not containment_pass:
        raise RuntimeError(f"{arm} failed exact containment")

    full_metrics, foreground_metrics = h29._metric_domains(cold_np, source, inside, args)
    regional = _regional_quality(cold_np, target, inside, geometry["sdf"])
    detail = _detail_metrics(cold_np, target, geometry["sdf"], args, torch)
    holes = _coverage_metrics(
        coverage,
        inside,
        geometry["sdf"],
        geometry["ridge"],
        geometry["isotropic_unreachable"],
    )
    scaling = _scaling_readiness(decoded, cold_np, target, inside)
    bounds = h29._save_visuals(
        artifact_dir,
        source,
        cold_np,
        inside,
        "masked_foreground",
        args.error_scale,
    )
    _save_objective_crops(
        artifact_dir,
        source,
        cold_np,
        inside,
        bounds,
        args.error_scale,
    )
    hair_bounds = _save_hair_crop(
        artifact_dir, source, cold_np, inside, args.error_scale
    )
    coverage_scale = max(float(coverage.max(initial=0.0)), 1e-8)
    save_image(
        str(artifact_dir / "unit_coverage.png"),
        np.clip(coverage / coverage_scale, 0.0, 1.0),
    )
    hole_view = np.zeros((*inside.shape, 3), dtype=np.float32)
    hole_view[inside] = np.asarray([0.08, 0.08, 0.08], dtype=np.float32)
    hole_view[inside & (coverage < 0.05)] = np.asarray([1.0, 0.55, 0.0], dtype=np.float32)
    hole_view[inside & (coverage <= 0.0)] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    save_image(str(artifact_dir / "holes.png"), hole_view)
    save_image(
        str(artifact_dir / "placement.png"),
        h30._placement_image(source, inside, decoded, "masked_contained"),
    )
    save_error_heatmap(
        str(artifact_dir / "outside_support.png"),
        np.repeat((coverage * (~inside))[..., None], 3, axis=2),
        scale=args.error_scale,
    )
    np.savez_compressed(
        artifact_dir / "analysis.npz",
        mask=inside,
        sdf=geometry["sdf"],
        ridge=geometry["ridge"],
        isotropic_unreachable=geometry["isotropic_unreachable"],
        reconstruction_raw=cold_np,
        unit_coverage=coverage,
        error_raw=cold_np - target,
        worst_crop_bounds=np.asarray(bounds, dtype=np.int32),
        hair_crop_bounds=np.asarray(hair_bounds, dtype=np.int32),
    )
    _write_json(
        artifact_dir / "fit_history.json",
        {"schema": REPORT_SCHEMA, "history": method.get("history", [])},
    )
    _write_json(
        artifact_dir / "projection_history.json",
        {
            "schema": REPORT_SCHEMA,
            "applied": bool(method.get("projection_history")),
            "history": method.get("projection_history", []),
        },
    )
    _write_json(
        artifact_dir / "geometry_history.json",
        {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "renderer": renderer,
            "count_neutral": decoded.n == CAPACITY,
            "micro_scale_px": MICRO_SCALE if "micro" in arm else None,
            "metadata": method.get("metadata", {}),
        },
    )
    _write_json(
        artifact_dir / "scaling_readiness.json",
        {"schema": REPORT_SCHEMA, **scaling},
    )
    scales = np.exp(decoded.log_scales.detach().cpu().numpy())
    row: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "image": "C0001",
        "seed": args.seed,
        "mode": "masked_contained",
        "arm": arm,
        "renderer": renderer,
        "target_gaussians": CAPACITY,
        "n_gaussians": decoded.n,
        "artifact_dir": str(artifact_dir.relative_to(output_root)),
        "field_file_sha256": h22.report_utils._sha256(field_path),
        "field_file_bytes": field_path.stat().st_size,
        "field_npz_keys": field_keys,
        "four_array_endpoint_exact": set(field_keys) == FOUR_ARRAY_KEYS,
        "masked_mse": foreground_metrics["masked_mse"],
        "psnr_db": foreground_metrics["psnr_db"],
        "ms_ssim": foreground_metrics["ms_ssim"],
        "lpips": foreground_metrics["lpips"],
        "ssim": foreground_metrics["ssim"],
        "pixel_rmse_max": foreground_metrics["artifact_pixel_rmse_max"],
        "patch7_rmse_max": foreground_metrics["artifact_patch_rmse_max_7"],
        "full_psnr_db": full_metrics["psnr_db"],
        "maintained_render_parity_max_abs": maintained_parity,
        "repeated_render_parity_max_abs": repeated_parity,
        "centres_inside_mask": int(centres_inside.sum()),
        "centres_outside_mask": int((~centres_inside).sum()),
        "unit_coverage_outside_abs_max": float(outside_coverage.max(initial=0.0)),
        "reconstruction_outside_abs_max": float(
            outside_reconstruction.max(initial=0.0)
        ),
        "containment_pass": containment_pass,
        "scale_min_px": float(scales.min()),
        "scale_q01_px": float(np.quantile(scales, 0.01)),
        "scale_median_px": float(np.median(scales)),
        "scale_max_px": float(scales.max()),
        "micro_rows_le_0081": int(np.count_nonzero(np.max(scales, axis=1) <= 0.081)),
        "coefficient_abs_max": float(decoded.colors.detach().abs().max()),
        "coefficient_abs_q99": float(
            np.quantile(np.abs(decoded.colors.detach().cpu().numpy()), 0.99)
        ),
        **regional,
        **detail,
        **holes,
        "tile_sse_cv": scaling["tile_sse_cv"],
        "tile_mse_cv": scaling["tile_mse_cv"],
        "tile_complexity_normalized_error_cv": scaling[
            "tile_complexity_normalized_error_cv"
        ],
        "tile_gaussian_density_cv": scaling["tile_gaussian_density_cv"],
        "tile_next_row_gain_proxy_cv": scaling["tile_next_row_gain_proxy_cv"],
        "tile_max_sse_fraction": scaling["tile_max_sse_fraction"],
        "method_seconds": float(method.get("seconds", 0.0)),
    }
    _write_json(artifact_dir / "row.json", row)
    return row


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: ARMS.index(str(row["arm"])))
    _write_json(
        output_root / "metrics.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "rows": rows},
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        str(row.get(key))
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def _refresh_coverage_extrema(
    output_root: Path,
    rows: list[dict[str, object]],
) -> None:
    """Repair derived extrema after a resumed run without rerendering endpoints."""
    for row in rows:
        artifact_dir = output_root / str(row["artifact_dir"])
        with np.load(artifact_dir / "analysis.npz", allow_pickle=False) as analysis:
            coverage = np.asarray(analysis["unit_coverage"])
            inside = np.asarray(analysis["mask"], dtype=bool)
        active = coverage[inside]
        row["coverage_inside_min"] = float(active.min())
        row["coverage_inside_max"] = float(active.max())
        _write_json(artifact_dir / "row.json", row)


def _decision(
    rows: list[dict[str, object]],
    feasibility: dict[str, object],
) -> dict[str, object]:
    by_arm = {str(row["arm"]): row for row in rows}
    control = by_arm.get("hier030_cold_additive_n7000")
    candidates = [row for row in rows if row is not control]
    best_holes = min(candidates, key=lambda row: int(row["raw_hole_pixels"])) if candidates else None
    best_psnr = max(candidates, key=lambda row: float(row["psnr_db"])) if candidates else None
    best_detail = (
        min(candidates, key=lambda row: float(row["detail_highpass_mse"]))
        if candidates
        else None
    )
    qualifying = [
        row
        for row in candidates
        if int(row["raw_hole_pixels"]) == 0
        and bool(row["containment_pass"])
        and int(row["n_gaussians"]) == CAPACITY
        and (
            control is None
            or float(row["interior_gt4_psnr_db"])
            >= float(control["interior_gt4_psnr_db"]) - 0.05
        )
    ]
    selected = (
        min(
            qualifying,
            key=lambda row: (
                float(row["detail_highpass_mse"]),
                -float(row["psnr_db"]),
            ),
        )
        if qualifying
        else None
    )

    def delta(row: dict[str, object] | None, key: str) -> float | None:
        if row is None or control is None:
            return None
        return float(row[key]) - float(control[key])

    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "complete": len(rows) == len(ARMS),
        "formal_claim_ready": False,
        "selected_arm": None if selected is None else selected["arm"],
        "selection_reason": (
            "no candidate met the frozen topology/interior guard"
            if selected is None
            else "zero raw holes, exact containment/count, interior guard, then lowest deep high-pass MSE"
        ),
        "best_hole_arm": None if best_holes is None else best_holes["arm"],
        "best_psnr_arm": None if best_psnr is None else best_psnr["arm"],
        "best_detail_arm": None if best_detail is None else best_detail["arm"],
        "selected_psnr_delta_db": delta(selected, "psnr_db"),
        "selected_boundary_psnr_delta_db": delta(selected, "boundary_le4_psnr_db"),
        "selected_interior_psnr_delta_db": delta(selected, "interior_gt4_psnr_db"),
        "selected_highpass_mse_delta": delta(selected, "detail_highpass_mse"),
        "control_raw_holes": None if control is None else control["raw_hole_pixels"],
        "selected_raw_holes": None if selected is None else selected["raw_hole_pixels"],
        "feasibility": {
            "isotropic_unreachable_pixels": feasibility["isotropic_unreachable_pixels"],
            "centreless_component_pixels": feasibility["centreless_component_pixels"],
            "micro_all_mask_centres_certified": feasibility[
                "micro_all_mask_centres_certified"
            ],
        },
        "limits": [
            "one exposed Janelle image and seed zero",
            "max-side-1200 diagnostic, not native-camera resolution",
            "dirty-source self-review; no default or publication claim",
            "the next-row gain metric is a deterministic allocation proxy, not a fitted oracle",
            "mask topology identifies silhouette thinness; it does not label every internal hair strand",
        ],
    }


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return escape(str(value))


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    feasibility: dict[str, object],
    decision: dict[str, object],
) -> None:
    row_html = []
    cards = []
    finalization_link = (
        " · <a href='presentation_finalization.json'>presentation finalization</a>"
        if (output_root / "presentation_finalization.json").is_file()
        else ""
    )
    for row in sorted(rows, key=lambda item: ARMS.index(str(item["arm"]))):
        artifact = str(row["artifact_dir"])
        row_html.append(
            "<tr>"
            f"<td>{escape(str(row['arm']))}</td>"
            f"<td>{_fmt(row['n_gaussians'], 0)}</td>"
            f"<td>{_fmt(row['psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['boundary_le4_psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['interior_gt4_psnr_db'], 3)}</td>"
            f"<td>{_fmt(row['ms_ssim'], 5)}</td>"
            f"<td>{_fmt(row['lpips'], 5)}</td>"
            f"<td>{_fmt(row['raw_hole_pixels'], 0)}</td>"
            f"<td>{_fmt(row['coverage_lt_005_pixels'], 0)}</td>"
            f"<td>{float(row['detail_highpass_mse']):.3e}</td>"
            f"<td>{float(row['detail_laplacian_mse']):.3e}</td>"
            f"<td>{_fmt(row['tile_next_row_gain_proxy_cv'], 3)}</td>"
            f"<td>{_fmt(row['micro_rows_le_0081'], 0)}</td>"
            f"<td>{float(row['unit_coverage_outside_abs_max']):.1e}</td>"
            f"<td>{float(row['reconstruction_outside_abs_max']):.1e}</td>"
            "</tr>"
        )
        cards.append(
            f"<section class='card'><h3>{escape(str(row['arm']))}</h3>"
            f"<p>N={row['n_gaussians']:,}; foreground {float(row['psnr_db']):.3f} dB; "
            f"boundary {float(row['boundary_le4_psnr_db']):.3f} dB; "
            f"raw holes {row['raw_hole_pixels']:,}; &lt;0.05 {row['coverage_lt_005_pixels']:,}.</p>"
            f"<div class='figs'><figure><a href='{artifact}/objective_source.png'><img src='{artifact}/objective_source.png'></a><figcaption>black-matted target</figcaption></figure>"
            f"<figure><a href='{artifact}/objective_reconstruction.png'><img src='{artifact}/objective_reconstruction.png'></a><figcaption>masked reconstruction</figcaption></figure>"
            f"<figure><a href='{artifact}/objective_error.png'><img src='{artifact}/objective_error.png'></a><figcaption>masked-objective error ×4</figcaption></figure>"
            f"<figure><a href='{artifact}/unit_coverage.png'><img src='{artifact}/unit_coverage.png'></a><figcaption>unit coverage</figcaption></figure>"
            f"<figure><a href='{artifact}/holes.png'><img src='{artifact}/holes.png'></a><figcaption>orange &lt;0.05; red zero</figcaption></figure>"
            f"<figure><a href='{artifact}/placement.png'><img src='{artifact}/placement.png'></a><figcaption>centres</figcaption></figure></div>"
            f"<h4>Worst-error crop</h4><div class='figs'><figure><a href='{artifact}/source_crop.png'><img src='{artifact}/source_crop.png'></a></figure>"
            f"<figure><a href='{artifact}/reconstruction_crop.png'><img src='{artifact}/reconstruction_crop.png'></a></figure>"
            f"<figure><a href='{artifact}/error_crop.png'><img src='{artifact}/error_crop.png'></a></figure></div>"
            f"<h4>Fixed head/hair crop</h4><div class='figs'><figure><a href='{artifact}/hair_source_crop.png'><img src='{artifact}/hair_source_crop.png'></a></figure>"
            f"<figure><a href='{artifact}/hair_reconstruction_crop.png'><img src='{artifact}/hair_reconstruction_crop.png'></a></figure>"
            f"<figure><a href='{artifact}/hair_error_crop.png'><img src='{artifact}/hair_error_crop.png'></a></figure></div>"
            f"<p><a href='{artifact}/field.gaussian.npz'>field</a> · "
            f"<a href='{artifact}/analysis.npz'>raw analysis</a> · "
            f"<a href='{artifact}/scaling_readiness.json'>scaling readiness</a> · "
            f"<a href='{artifact}/source.png'>unmasked source</a> · "
            f"<a href='{artifact}/reconstruction.png'>raw reconstruction</a> · "
            f"<a href='{artifact}/error.png'>unmasked-reference error</a> · "
            f"<a href='{artifact}/reconstruction_crop.png'>reconstruction crop</a></p></section>"
        )
    errors = [attempt for attempt in attempts if attempt.get("status") != "ok"]
    error_html = (
        "<p>No execution errors.</p>"
        if not errors
        else "<ul>"
        + "".join(
            f"<li>{escape(str(item.get('arm')))}: {escape(str(item.get('error')))}</li>"
            for item in errors
        )
        + "</ul>"
    )
    selected = decision.get("selected_arm") or "none"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>HIER-031 exact-7k masked boundary/detail allocation</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#11151b;color:#e9eef5;line-height:1.45}}
main{{max-width:1500px;margin:auto;padding:24px}}a{{color:#79c4ff}}table{{border-collapse:collapse;width:100%;font-size:13px;overflow:auto;display:block}}
th,td{{border:1px solid #3b4654;padding:6px 8px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
.callout{{background:#1b2430;border-left:5px solid #68d391;padding:14px 18px;margin:16px 0}}.warn{{border-color:#f6ad55}}
.card{{background:#18202a;border:1px solid #33404e;border-radius:10px;padding:16px;margin:18px 0}}.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}}
figure{{margin:0;background:#0d1117;padding:6px}}img{{width:100%;height:auto;display:block}}figcaption{{font-size:12px;color:#b9c4d0;padding-top:4px}}
code{{background:#0d1117;padding:2px 5px;border-radius:4px}}
</style></head><body><main><h1>HIER-031 — exact-7k masked boundary and thin-detail allocation</h1>
<div class='callout'><strong>Disposition:</strong> selected arm <code>{escape(str(selected))}</code>. This is a one-image, seed-0, dirty-source diagnostic—not a default or publication claim.</div>
<p>The test holds the endpoint at exactly 7,000 ordinary four-array Gaussians. It asks whether rows should be moved from redundant interior coverage to mask holes and deep high-frequency residuals before any later count scaling.</p>
<h2>Feasibility result</h2><div class='callout warn'><p>The 0.35 px model has {feasibility['isotropic_unreachable_pixels']:,} mask pixels outside the reach of an isotropic minimum-scale row from an admissible centre. Certified tangent elongation can reach some of these, so this is an upper bound. The hard lower bound is {feasibility['centreless_component_pixels']:,} pixels in {feasibility['centreless_components']} disconnected components with no admissible 0.35 px centre. A 0.08 px support has margin+radius {float(feasibility['micro_certificate_radius_px']):.2f} px and can be centred on every mask pixel.</p></div>
<p><a href='feasibility.json'>Full feasibility ledger</a> · <a href='research_context.md'>Research context</a></p>
<h2>Metrics</h2><p>“Next-row CV” is a spatial starvation proxy: lower means the estimated value of one common additional compact row is more even across occupied tiles. It is not a demand for equal raw pixel error.</p>
<table><tr><th>arm</th><th>N</th><th>PSNR</th><th>boundary PSNR</th><th>interior PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>raw holes</th><th>coverage&lt;.05</th><th>high-pass MSE</th><th>Laplacian MSE</th><th>next-row CV</th><th>micro rows</th><th>support outside</th><th>recon outside</th></tr>{''.join(row_html)}</table>
<p>Machine-readable: <a href='manifest.json'>manifest</a> · <a href='metrics.json'>JSON</a> · <a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> · <a href='decision.json'>decision</a> · <a href='attempts.json'>errors ledger</a>{finalization_link}.</p>
<h2>Execution errors</h2>{error_html}<h2>Visual comparisons and errors</h2>{''.join(cards)}
</main></body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": h22.report_utils._sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": files},
    )


def _method_for_arm(
    arm: str,
    source: np.ndarray,
    inside: np.ndarray,
    geometry: dict[str, np.ndarray],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    target = source * inside[..., None].astype(np.float32)
    base = _pure_field(
        GaussianField.load(str(args.base_bundle / BASE_FIELD_REL), device=args.device)
    )
    started = time.perf_counter()
    if arm == "hier030_cold_additive_n7000":
        return {
            "field": base,
            "renderer": "cuda_additive",
            "history": [],
            "projection_history": [],
            "metadata": {
                "source": str(args.base_bundle / BASE_FIELD_REL),
                "source_sha256": h22.report_utils._sha256(args.base_bundle / BASE_FIELD_REL),
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "micro_hole_reallocation_n7000":
        field, history = _micro_reallocation(
            base, target, inside, geometry["sdf"], args, torch
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": history,
            "projection_history": [row["projection"] for row in history if "projection" in row],
            "metadata": {
                "operator": "count-neutral low-column-energy donor exchange to raw hole sites",
                "micro_certificate_radius_px": MICRO_CERTIFICATE_RADIUS,
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "detail_reallocation_n7000":
        field, history = _detail_reallocation(
            base, target, inside, geometry["sdf"], args, torch
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": history,
            "projection_history": [row["projection"] for row in history],
            "metadata": {
                "operator": "six count-neutral 128-row exact-site deep high-pass waves",
                "detail_rows": DETAIL_ROWS,
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "combined_micro_detail_n7000":
        field, micro_before = _micro_reallocation(
            base, target, inside, geometry["sdf"], args, torch
        )
        field, detail_history = _detail_reallocation(
            field, target, inside, geometry["sdf"], args, torch
        )
        field, micro_after = _micro_reallocation(
            field, target, inside, geometry["sdf"], args, torch
        )
        history = [
            {"stage": "micro_before", **row} for row in micro_before
        ] + [
            {"stage": "detail", **row} for row in detail_history
        ] + [
            {"stage": "micro_after", **row} for row in micro_after
        ]
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": history,
            "projection_history": [row["projection"] for row in history if "projection" in row],
            "metadata": {
                "operator": "micro hole closure, 768-row deep high-pass exchange, micro closure",
                "detail_rows": DETAIL_ROWS,
            },
            "seconds": time.perf_counter() - started,
        }
    if arm in ("merge_funded_micro_n7000", "merge_funded_micro_exempt_n7000"):
        field, history = _merge_funded_micro_reallocation(
            base, target, inside, geometry["sdf"], args, torch
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": history,
            "projection_history": [row["projection"] for row in history if "projection" in row],
            "metadata": {
                "operator": "function-preserving envelope merge funded micro hole closure",
                "stage": "post-first-screen mechanistic rescue",
                "ordinary_recertificate_exempts_existing_micro_rows": True,
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "merge_micro_geometry_recovery_n7000":
        corrected_path = (
            args.out
            / "artifacts/merge_funded_micro_exempt_n7000/field.gaussian.npz"
        )
        if corrected_path.is_file():
            corrected = GaussianField.load(str(corrected_path), device=args.device)
            source_record = {
                "path": str(corrected_path.resolve()),
                "sha256": h22.report_utils._sha256(corrected_path),
            }
        else:
            corrected, corrected_history = _merge_funded_micro_reallocation(
                base, target, inside, geometry["sdf"], args, torch
            )
            source_record = {
                "path": None,
                "recomputed": True,
                "funding_history": corrected_history,
            }
        field, fit_history, metadata = _geometry_recovery(
            corrected, target, inside, args, torch
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": fit_history,
            "projection_history": [metadata["projection"]],
            "metadata": {
                "operator": "500-step ordinary-row geometry recovery around fixed micro reserve",
                "base": source_record,
                **metadata,
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "geometry_recovery_terminal_closure_n7000":
        recovery_path = (
            args.out
            / "artifacts/merge_micro_geometry_recovery_n7000/field.gaussian.npz"
        )
        if not recovery_path.is_file():
            raise RuntimeError(
                "terminal closure requires the completed geometry-recovery arm"
            )
        recovery = GaussianField.load(str(recovery_path), device=args.device)
        field, history = _merge_funded_micro_reallocation(
            recovery, target, inside, geometry["sdf"], args, torch
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": history,
            "projection_history": [row["projection"] for row in history if "projection" in row],
            "metadata": {
                "operator": "terminal merge-funded closure after ordinary geometry recovery",
                "base": {
                    "path": str(recovery_path.resolve()),
                    "sha256": h22.report_utils._sha256(recovery_path),
                },
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "coverage_constrained_geometry_recovery_n7000":
        corrected_path = (
            args.out
            / "artifacts/merge_funded_micro_exempt_n7000/field.gaussian.npz"
        )
        if not corrected_path.is_file():
            raise RuntimeError(
                "coverage-constrained recovery requires the corrected merge-funded arm"
            )
        corrected = GaussianField.load(str(corrected_path), device=args.device)
        field, fit_history, metadata = _geometry_recovery(
            corrected,
            target,
            inside,
            args,
            torch,
            coverage_constrained=True,
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": fit_history,
            "projection_history": [metadata["projection"]],
            "metadata": {
                "operator": "boundary-undercoverage-constrained ordinary geometry recovery",
                "base": {
                    "path": str(corrected_path.resolve()),
                    "sha256": h22.report_utils._sha256(corrected_path),
                },
                **metadata,
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "deep_only_geometry_recovery_n7000":
        corrected_path = (
            args.out
            / "artifacts/merge_funded_micro_exempt_n7000/field.gaussian.npz"
        )
        if not corrected_path.is_file():
            raise RuntimeError(
                "deep-only recovery requires the corrected merge-funded arm"
            )
        corrected = GaussianField.load(str(corrected_path), device=args.device)
        field, fit_history, metadata = _geometry_recovery(
            corrected,
            target,
            inside,
            args,
            torch,
            deep_only=True,
            sdf=geometry["sdf"],
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": fit_history,
            "projection_history": [metadata["projection"]],
            "metadata": {
                "operator": "deep-interior-only geometry recovery around fixed boundary support",
                "base": {
                    "path": str(corrected_path.resolve()),
                    "sha256": h22.report_utils._sha256(corrected_path),
                },
                **metadata,
            },
            "seconds": time.perf_counter() - started,
        }
    if arm == "deep_only_terminal_closure_n7000":
        recovery_path = (
            args.out
            / "artifacts/deep_only_geometry_recovery_n7000/field.gaussian.npz"
        )
        if not recovery_path.is_file():
            raise RuntimeError(
                "deep-only terminal closure requires the completed deep-only recovery arm"
            )
        recovery = GaussianField.load(str(recovery_path), device=args.device)
        field, history = _merge_funded_micro_reallocation(
            recovery, target, inside, geometry["sdf"], args, torch
        )
        return {
            "field": field,
            "renderer": "cuda_additive",
            "history": history,
            "projection_history": [
                row["projection"] for row in history if "projection" in row
            ],
            "metadata": {
                "operator": "terminal merge-funded closure after deep-only recovery",
                "base": {
                    "path": str(recovery_path.resolve()),
                    "sha256": h22.report_utils._sha256(recovery_path),
                },
            },
            "seconds": time.perf_counter() - started,
        }
    if arm in ("pipeline_fixed_n7000", "pipeline_boundary_recycle_n7000"):
        field, history, metadata = _pipeline_arm(
            source,
            inside,
            args,
            torch,
            boundary_recycle=arm == "pipeline_boundary_recycle_n7000",
        )
        return {
            "field": field,
            "renderer": "cuda",
            "history": history,
            "projection_history": [],
            "metadata": metadata,
            "seconds": time.perf_counter() - started,
        }
    raise ValueError(f"unknown HIER-031 arm {arm!r}")


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if (args.out / "COMPLETED").is_file():
        raise SystemExit(f"completed HIER-031 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-031 diagnostic requires CUDA")
    source, inside, raster = h22.report_utils._load_evaluation_raster(
        args.image, args.mask, max_side=args.max_side, mask_threshold=0.5
    )
    if inside is None or source.shape[:2] != EVALUATION_SHAPE:
        raise RuntimeError(
            f"expected masked evaluation raster {EVALUATION_SHAPE}, got {source.shape}"
        )
    if (raster["original_height"], raster["original_width"]) != NATIVE_SHAPE:
        raise RuntimeError(f"native Janelle shape differs: {raster!r}")
    feasibility, geometry = _feasibility_audit(inside)
    _write_json(args.out / "feasibility.json", feasibility)
    np.savez_compressed(args.out / "feasibility.npz", **geometry)
    input_dir = args.out / "input"
    input_dir.mkdir(exist_ok=True)
    save_image(str(input_dir / "source.png"), source)
    save_image(str(input_dir / "mask.png"), inside.astype(np.float32))
    save_image(
        str(input_dir / "foreground_black_matted.png"),
        source * inside[..., None].astype(np.float32),
    )
    snapshots = _snapshot_sources(args.out)
    _write_json(args.out / "environment.json", h22._environment(torch))
    _write_json(
        args.out / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": _command(),
            "git": h22._git_record(),
            "source_snapshots": snapshots,
            "arguments": vars(args),
            "source": {
                "path": str(args.image.resolve()),
                "sha256": SOURCE_SHA256,
                "native_shape": list(NATIVE_SHAPE),
            },
            "mask": {
                "path": str(args.mask.resolve()),
                "sha256": MASK_SHA256,
                "active_pixels": int(inside.sum()),
                "margin_px": MASK_MARGIN,
            },
            "raster": raster,
            "arms": list(ARMS),
            "capacity": CAPACITY,
            "ordinary_min_scale_px": ORDINARY_MIN_SCALE,
            "micro_scale_px": MICRO_SCALE,
            "detail_rows": DETAIL_ROWS,
            "base_field": {
                "path": str((args.base_bundle / BASE_FIELD_REL).resolve()),
                "sha256": h22.report_utils._sha256(args.base_bundle / BASE_FIELD_REL),
            },
            "claim_limits": [
                "one exposed image/seed/device",
                "dirty-source self-reviewed diagnostic",
                "no default, native-resolution, 57.6k, or publication claim",
            ],
        },
    )
    (args.out / "NATURAL_STARTED").write_text(
        "HIER-031 hash-bound source/mask decoded after the protocol and arms were frozen.\n",
        encoding="utf-8",
    )
    (args.out / "research_context.md").write_text(
        "# Research context\n\n"
        "The experiment separates exact mask topology from photographic detail. The topology "
        "arm uses certified compact supports at uncovered mask pixels; the detail arm uses "
        "deep high-pass residual pursuit. This follows the general thin-structure lesson of "
        "skeleton-aware allocation (Prior-Enhanced Gaussian Splatting, SIGGRAPH Asia 2025) and "
        "uses clDice (CVPR 2021) as motivation for measuring skeleton connectivity rather than "
        "only average overlap. Primary sources:\n\n"
        "- https://arxiv.org/abs/2512.11356\n"
        "- https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_"
        "Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html\n\n"
        "The cited work motivates diagnostics; HIER-031 does not claim those methods as novel.\n",
        encoding="utf-8",
    )
    with (args.out / "git.diff").open("wb") as stream:
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=stream,
        )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    metrics_path = args.out / "metrics.json"
    attempts_path = args.out / "attempts.json"
    if args.resume and metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
    if args.resume and attempts_path.is_file():
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")).get("attempts", [])
    complete_arms = {str(row["arm"]) for row in rows}
    for arm in args.arms:
        if arm in complete_arms:
            continue
        started = time.perf_counter()
        try:
            method = _method_for_arm(arm, source, inside, geometry, args, torch)
            row = _write_arm(
                output_root=args.out,
                arm=arm,
                method=method,
                source=source,
                inside=inside,
                geometry=geometry,
                args=args,
                torch=torch,
            )
            rows.append(row)
            complete_arms.add(arm)
            attempts.append(
                {
                    "arm": arm,
                    "status": "ok",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "arm": arm,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if not args.quiet:
                print(f"{arm}: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            _write_tables(args.out, rows)
            _write_json(
                attempts_path,
                {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
            )
            torch.cuda.empty_cache()

    _refresh_coverage_extrema(args.out, rows)
    _write_tables(args.out, rows)
    decision = _decision(rows, feasibility)
    _write_json(args.out / "decision.json", decision)
    _write_report(args.out, rows, attempts, feasibility, decision)
    all_ok = set(ARMS) <= complete_arms
    if all_ok:
        (args.out / "COMPLETED").write_text(
            "HIER-031 exact-7k masked boundary/detail diagnostic complete; do not overwrite.\n",
            encoding="utf-8",
        )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
