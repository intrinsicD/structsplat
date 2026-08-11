#!/usr/bin/env python3
"""Run HIER-018's source-bound counted-background coverage diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from html import escape
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from scripts.experiments import hier013_global_projection_development as h13  # noqa: E402
from scripts.experiments import hier015_geometry_escape as h15  # noqa: E402
from scripts.experiments import hier016_normalized_tail_repair as h16  # noqa: E402
from scripts.experiments import hier017_normalization_epsilon as h17  # noqa: E402
from structsplat.config import StructureTensorConfig  # noqa: E402
from structsplat.fit import _normalized_color_denominator  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.init import build_field  # noqa: E402
from structsplat.pixel_contraction import contract_image  # noqa: E402


REPORT_SCHEMA = "structsplat.hier018_counted_background.diagnostic.v1"
DEVELOPMENT_BINDINGS = {
    "COCO_train2014_000000402844.jpg": (
        "c9894417161c13b10e9df7c3cca75471d5a0ef5d801036aa5097330296879412"
    ),
    "COCO_train2014_000000210071.jpg": (
        "609b781d3a3baa8c939f84f777d00fb147f21a1769a087bb7dc26c67ba0c1ba2"
    ),
    "COCO_train2014_000000091348.jpg": (
        "6a8b9aae88e9b40e73aad18135737c68220b49fd1727f91b5c79a8e8a04c4670"
    ),
    "COCO_train2014_000000165574.jpg": (
        "fbc83f705db6519111056875a4cdc76dd9feee01a3995d8bd10e0f88f4ee4205"
    ),
}
SELECTION_DIGESTS = {
    "COCO_train2014_000000402844.jpg": (
        "0000071156a02cf7316b9402a234e5f81ea4d719479c28b8b2b80ce8760141d6"
    ),
    "COCO_train2014_000000210071.jpg": (
        "00006c4b471f11776ed96529eb99610add61c929598149ade8c22d34c644f2b1"
    ),
    "COCO_train2014_000000091348.jpg": (
        "0000ea0b8b93ed69e58fd3a1f1f2280318999ec7978e24a469e049584c4a260b"
    ),
    "COCO_train2014_000000165574.jpg": (
        "0000ebdadce300565df93731db6bb0197eee321e282e322524a3104edf738cb0"
    ),
}
CONTROL_ARM = "h005_control"
BASELINE_ARM = "direct_no_background"
BACKGROUND_ARM = "direct_bg64_grid8"
DEVELOPMENT_ARMS = (CONTROL_ARM, BASELINE_ARM, BACKGROUND_ARM)
REPLAY_ARMS = (BASELINE_ARM, BACKGROUND_ARM)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "development",
            "replay_h15",
            "replay_h16",
            "replay_h17",
            "replay_tests",
        ),
        required=True,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--control-metrics", type=Path)
    parser.add_argument("--recover-from", type=Path)
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direct-fit-steps", type=int, default=750)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additive-renderer", default="cuda_additive")
    parser.add_argument("--direct-renderer", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--error-scale", type=float, default=4.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "target_gaussians": 7000,
        "max_side": 512,
        "seed": 0,
        "direct_fit_steps": 750,
        "device": "cuda",
        "additive_renderer": "cuda_additive",
        "direct_renderer": "cuda",
        "render_chunk": 256,
        "lpips": True,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-018 protocol requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    replay = args.phase != "development"
    if replay != (args.development_decision is not None):
        raise SystemExit("replay requires --development-decision; development rejects it")
    needs_controls = args.phase in ("replay_h15", "replay_h16", "replay_h17")
    if needs_controls != (args.control_metrics is not None):
        raise SystemExit("COCO replays require --control-metrics; other phases reject it")
    if args.recover_from is not None and args.phase != "development":
        raise SystemExit("--recover-from is valid only for development")


def _validate_development_decision(args: argparse.Namespace) -> None:
    if args.phase == "development":
        return
    decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    if (
        decision.get("schema") != REPORT_SCHEMA
        or decision.get("phase") != "development"
        or BACKGROUND_ARM not in decision.get("numeric_candidates", [])
    ):
        raise SystemExit("the HIER-018 background arm did not pass development")


def _bound_paths(directory: Path, bindings: dict[str, str]) -> list[Path]:
    paths = [directory / name for name in bindings]
    actual = {
        path.name: report_utils._sha256(path)
        for path in paths
        if path.is_file()
    }
    if actual != bindings:
        raise SystemExit(f"source binding mismatch: expected {bindings}, got {actual}")
    return [path.resolve() for path in paths]


def _discover_images(args: argparse.Namespace) -> list[Path]:
    if args.phase == "development":
        return _bound_paths(args.images, DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h15":
        return _bound_paths(args.images, h15.DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h16":
        return _bound_paths(args.images, h16.DEVELOPMENT_BINDINGS)
    if args.phase == "replay_h17":
        return _bound_paths(args.images, h17.DEVELOPMENT_BINDINGS)
    return h13._discover_sources([args.images])


def _configs(args: argparse.Namespace):
    init_config, fit_config = h15._direct_configs(args)
    fit_config = replace(fit_config, normalization_eps=1e-8)
    background_config = replace(
        init_config,
        background_fraction=0.05,
        background_grid=8,
    )
    return init_config, background_config, fit_config


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "init.py",
        ROOT / "src" / "structsplat" / "gaussians.py",
        ROOT / "src" / "structsplat" / "fit.py",
        ROOT / "src" / "structsplat" / "render.py",
        ROOT / "scripts" / "experiments" / "hier017_normalization_epsilon.py",
        ROOT / "tasks" / "HIER-018-counted-background-coverage.md",
    )
    records: list[dict[str, object]] = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": report_utils._sha256(destination),
            }
        )
    return records


def _background_geometry_hash(field: GaussianField) -> str:
    mask = field.background_mask
    if mask is None:
        return hashlib.sha256(b"no-background").hexdigest()
    digest = hashlib.sha256()
    for name, tensor in (
        ("means", field.means[mask]),
        ("log_scales", field.log_scales[mask]),
        ("rotations", field.rotations[mask]),
    ):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _background_geometry_delta(before: GaussianField, after: GaussianField) -> dict[str, object]:
    mask = before.background_mask
    if mask is None or after.background_mask is None:
        return {
            "background_geometry_bit_exact": before.background_count == after.background_count == 0,
            "background_mean_shift_max": 0.0,
            "background_log_scale_shift_max": 0.0,
            "background_rotation_shift_max": 0.0,
        }
    return {
        "background_geometry_bit_exact": (
            torch_equal(before.means[mask], after.means[after.background_mask])
            and torch_equal(before.log_scales[mask], after.log_scales[after.background_mask])
            and torch_equal(before.rotations[mask], after.rotations[after.background_mask])
        ),
        "background_mean_shift_max": float(
            (before.means[mask] - after.means[after.background_mask]).abs().max().detach().cpu()
        ),
        "background_log_scale_shift_max": float(
            (before.log_scales[mask] - after.log_scales[after.background_mask])
            .abs().max().detach().cpu()
        ),
        "background_rotation_shift_max": float(
            (before.rotations[mask] - after.rotations[after.background_mask])
            .abs().max().detach().cpu()
        ),
    }


def torch_equal(left, right) -> bool:
    import torch

    return bool(torch.equal(left.detach(), right.detach()))


def _background_coverage(
    field: GaussianField,
    fit_config,
    image: np.ndarray,
    reconstruction: np.ndarray,
    artifact_dir: Path,
) -> dict[str, object]:
    if field.background_mask is None or field.background_count == 0:
        return {
            "background_denominator_min": 0.0,
            "background_denominator_q01": 0.0,
            "background_denominator_median": 0.0,
        }
    height, width = image.shape[:2]
    background = field.subset(field.background_mask)
    denominator = (
        _normalized_color_denominator(background, fit_config, height, width)
        .reshape(height, width).detach().cpu().numpy().astype(np.float64)
    )
    pixel_rmse = np.sqrt(
        np.mean(
            (reconstruction.astype(np.float64) - image.astype(np.float64)) ** 2,
            axis=2,
        )
    )
    bands = {
        "zero": denominator == 0.0,
        "positive_below_1e12": (denominator > 0.0) & (denominator < h17.EPS_CANDIDATE),
        "1e12_to_1e8": (
            (denominator >= h17.EPS_CANDIDATE) & (denominator < h17.EPS_BASELINE)
        ),
        "at_least_1e8": denominator >= h17.EPS_BASELINE,
    }
    error_by_band: dict[str, dict[str, float | int | None]] = {}
    for name, active in bands.items():
        values = pixel_rmse[active]
        error_by_band[name] = {
            "count": int(active.sum()),
            "mean_pixel_rmse": float(values.mean()) if values.size else None,
            "max_pixel_rmse": float(values.max()) if values.size else None,
        }
    np.savez_compressed(
        artifact_dir / "background_denominator.npz",
        denominator=denominator.astype(np.float32),
        pixel_rmse=pixel_rmse.astype(np.float32),
    )
    record = {
        "background_denominator_min": float(denominator.min()),
        "background_denominator_q01": float(np.quantile(denominator, 0.01)),
        "background_denominator_median": float(np.median(denominator)),
        "error_by_background_denominator_band": error_by_band,
    }
    report_utils._write_json(artifact_dir / "background_denominator.json", record)
    return record


def _coverage_improvement(
    baseline_field: GaussianField,
    fit_config,
    image: np.ndarray,
    baseline_render: np.ndarray,
    candidate_render: np.ndarray,
) -> dict[str, object]:
    denominator = (
        _normalized_color_denominator(
            baseline_field, fit_config, image.shape[0], image.shape[1]
        ).reshape(image.shape[:2]).detach().cpu().numpy()
    )
    low = denominator < 1e-8
    baseline_error = h17._display_pixel_rmse(baseline_render, image)
    candidate_error = h17._display_pixel_rmse(candidate_render, image)
    improvement = baseline_error - candidate_error
    if low.any():
        masked = np.where(low, improvement, -np.inf)
        y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
        maximum = float(masked[y, x])
        coordinate: list[int] | None = [int(y), int(x)]
    else:
        maximum, coordinate = 0.0, None
    return {
        "baseline_low_coverage_pixel_count": int(low.sum()),
        "low_coverage_display_error_improvement_max": maximum,
        "low_coverage_best_yx": coordinate,
    }


def _pairs(rows: list[dict[str, object]], arm: str, control_arm: str) -> list[dict[str, object]]:
    controls = {str(row["image"]): row for row in rows if row["arm"] == control_arm}
    pairs: list[dict[str, object]] = []
    for row in rows:
        control = controls.get(str(row["image"]))
        if row["arm"] != arm or control is None:
            continue
        pairs.append(
            {
                "image": row["image"],
                "mse_ratio": float(row["masked_mse"]) / float(control["masked_mse"]),
                "psnr_delta_db": float(row["psnr_db"]) - float(control["psnr_db"]),
                "ms_ssim_delta": float(row["ms_ssim"]) - float(control["ms_ssim"]),
                "lpips_delta": float(row["lpips"]) - float(control["lpips"]),
                "pixel_max_delta": float(row["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "patch7_max_delta": float(row["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
                "time_ratio": float(row["pipeline_algorithm_seconds"])
                / max(float(control["pipeline_algorithm_seconds"]), 1e-12),
                "n_gaussians": row["n_gaussians"],
                "background_count": row.get("background_count"),
                "detail_count": row.get("detail_count"),
                "background_geometry_bit_exact": row.get("background_geometry_bit_exact"),
                "background_geometry_persistence_bit_exact": row.get(
                    "background_geometry_persistence_bit_exact"
                ),
                "denominator_min": row.get("denominator_min"),
                "baseline_low_coverage_pixel_count": row.get(
                    "baseline_low_coverage_pixel_count", 0
                ),
                "low_coverage_display_error_improvement_max": row.get(
                    "low_coverage_display_error_improvement_max", 0.0
                ),
                "maintained_render_parity_max_abs": row[
                    "maintained_render_parity_max_abs"
                ],
                "repeated_render_parity_max_abs": row[
                    "repeated_render_parity_max_abs"
                ],
            }
        )
    return pairs


def _finite_pairs(pairs: list[dict[str, object]]) -> bool:
    keys = (
        "mse_ratio", "psnr_delta_db", "ms_ssim_delta", "lpips_delta",
        "pixel_max_delta", "patch7_max_delta", "time_ratio",
    )
    try:
        return all(math.isfinite(float(pair[key])) for pair in pairs for key in keys)
    except (TypeError, ValueError):
        return False


def _development_decision(
    rows: list[dict[str, object]], attempts: list[dict[str, object]]
) -> dict[str, object]:
    h005_pairs = _pairs(rows, BACKGROUND_ARM, CONTROL_ARM)
    baseline_pairs = _pairs(rows, BACKGROUND_ARM, BASELINE_ARM)
    gates = {
        "complete_four_h005_pairs": len(h005_pairs) == 4,
        "complete_four_baseline_pairs": len(baseline_pairs) == 4,
        "all_finite": _finite_pairs(h005_pairs + baseline_pairs),
        "all_exact_count": all(int(pair["n_gaussians"]) == 7000 for pair in baseline_pairs),
        "all_exact_64_background_6936_detail": all(
            int(pair["background_count"]) == 64 and int(pair["detail_count"]) == 6936
            for pair in baseline_pairs
        ),
        "all_background_geometry_bit_exact": all(
            bool(pair["background_geometry_bit_exact"])
            and bool(pair["background_geometry_persistence_bit_exact"])
            for pair in baseline_pairs
        ),
        "all_denominator_min_ge_1e8": all(
            float(pair["denominator_min"]) >= 1e-8 for pair in baseline_pairs
        ),
        "all_parity_le_2e_5": all(
            float(pair["maintained_render_parity_max_abs"]) <= 2e-5
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
            for pair in baseline_pairs
        ),
        "all_psnr_gain_vs_h005_ge_2_db": all(
            float(pair["psnr_delta_db"]) >= 2.0 for pair in h005_pairs
        ),
        "all_pixel_max_noninferior_vs_h005": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in h005_pairs
        ),
        "all_patch7_max_noninferior_vs_h005": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in h005_pairs
        ),
        "mean_ms_ssim_noninferior_vs_h005": (
            float(np.mean([pair["ms_ssim_delta"] for pair in h005_pairs])) >= -1e-7
            if h005_pairs else False
        ),
        "mean_lpips_noninferior_vs_h005": (
            float(np.mean([pair["lpips_delta"] for pair in h005_pairs])) <= 1e-7
            if h005_pairs else False
        ),
        "all_mse_noninferior_vs_baseline": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in baseline_pairs
        ),
        "all_pixel_max_noninferior_vs_baseline": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in baseline_pairs
        ),
        "all_patch7_max_noninferior_vs_baseline": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in baseline_pairs
        ),
        "mean_ms_ssim_delta_vs_baseline_ge_neg_0_001": (
            float(np.mean([pair["ms_ssim_delta"] for pair in baseline_pairs])) >= -0.001
            if baseline_pairs else False
        ),
        "mean_lpips_delta_vs_baseline_le_0_002": (
            float(np.mean([pair["lpips_delta"] for pair in baseline_pairs])) <= 0.002
            if baseline_pairs else False
        ),
        "median_algorithm_time_ratio_le_1_10": (
            float(np.median([pair["time_ratio"] for pair in baseline_pairs])) <= 1.10
            if baseline_pairs else False
        ),
        "at_least_one_low_coverage_improvement_ge_1_255": any(
            int(pair["baseline_low_coverage_pixel_count"]) > 0
            and float(pair["low_coverage_display_error_improvement_max"]) >= 1.0 / 255.0
            for pair in baseline_pairs
        ),
    }
    candidates = [BACKGROUND_ARM] if all(gates.values()) else []
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "gates": gates,
        "h005_pairs": h005_pairs,
        "baseline_pairs": baseline_pairs,
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "numeric_candidates": candidates,
        "numeric_disposition": candidates[0] if candidates else "no_background_candidate",
        "visual_review_required": True,
        "interpretation": (
            "Numeric candidate requires visual review before replay."
            if candidates else "The counted background misses a frozen gate; do not replay."
        ),
    }


def _external_pairs(rows: list[dict[str, object]], path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    controls = {
        str(row["image"]): row for row in payload["rows"] if row["arm"] == CONTROL_ARM
    }
    candidates = {
        str(row["image"]): row for row in rows if row["arm"] == BACKGROUND_ARM
    }
    return [
        {
            "image": image,
            "psnr_delta_db": float(candidates[image]["psnr_db"]) - float(row["psnr_db"]),
            "pixel_max_delta": float(candidates[image]["artifact_pixel_rmse_max"])
            - float(row["artifact_pixel_rmse_max"]),
            "patch7_max_delta": float(candidates[image]["artifact_patch_rmse_max_7"])
            - float(row["artifact_patch_rmse_max_7"]),
        }
        for image, row in sorted(controls.items())
        if image in candidates
    ]


def _replay_decision(rows, attempts, args):
    pairs = _pairs(rows, BACKGROUND_ARM, BASELINE_ARM)
    expected = 16 if args.phase == "replay_tests" else 4
    clauses = {
        "complete_pairs": len(pairs) == expected,
        "all_finite": _finite_pairs(pairs),
        "all_exact_count_and_partition": all(
            int(pair["n_gaussians"]) == 7000
            and int(pair["background_count"]) == 64
            and int(pair["detail_count"]) == 6936
            for pair in pairs
        ),
        "all_background_geometry_bit_exact": all(
            bool(pair["background_geometry_bit_exact"])
            and bool(pair["background_geometry_persistence_bit_exact"])
            for pair in pairs
        ),
        "all_denominator_min_ge_1e8": all(
            float(pair["denominator_min"]) >= 1e-8 for pair in pairs
        ),
        "all_mse_noninferior": all(float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in pairs),
        "all_pixel_max_noninferior": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_patch7_max_noninferior": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_parity_le_2e_5": all(
            float(pair["maintained_render_parity_max_abs"]) <= 2e-5
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
            for pair in pairs
        ),
    }
    external = _external_pairs(rows, args.control_metrics)
    if args.phase != "replay_tests":
        clauses.update(
            {
                "complete_external_h005_pairs": len(external) == 4,
                "all_psnr_gain_vs_h005_ge_2_db": all(
                    pair["psnr_delta_db"] >= 2.0 for pair in external
                ),
                "all_pixel_max_noninferior_vs_h005": all(
                    pair["pixel_max_delta"] <= 1e-12 for pair in external
                ),
                "all_patch7_max_noninferior_vs_h005": all(
                    pair["patch7_max_delta"] <= 1e-12 for pair in external
                ),
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "pairs": pairs,
        "external_h005_pairs": external,
        "clauses": clauses,
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "bounded_bank_pass": all(clauses.values()),
        "interpretation": "Consumed reporting replay; no retuning or held-out claim.",
    }


def _write_report(output_root, rows, decision, command):
    table, cards = [], []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        background = row.get("background_count")
        denominator = row.get("denominator_min")
        coverage = (
            f"<a href='{artifact}/denominator.json'>total</a>"
            if denominator is not None else "—"
        )
        if int(background or 0) > 0:
            coverage += f" · <a href='{artifact}/background_denominator.json'>background</a>"
        table.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{escape(str(row['arm']))}</td>"
            f"<td>{int(background or 0)}</td><td>{int(row['n_gaussians'])}</td>"
            f"<td>{float(row['psnr_db']):.3f}</td><td>{float(row['ms_ssim']):.5f}</td>"
            f"<td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td>{coverage}</td><td><a href='{artifact}/reconstruction.png'>full</a> · "
            f"<a href='{artifact}/reconstruction_crop.png'>crop</a></td></tr>"
        )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} — {escape(str(row['arm']))}</h3>"
            f"<img src='{artifact}/source.png'><img src='{artifact}/reconstruction.png'>"
            f"<img src='{artifact}/error.png'><img src='{artifact}/reconstruction_crop.png'>"
            "</section>"
        )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>HIER-018</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1600px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}</style>
</head><body><h1>HIER-018 counted background — {escape(str(decision['phase']))}</h1>
<p>Dirty-source diagnostic; every background row counts against exact N=7,000.</p>
<p><code>{escape(command)}</code></p><p><a href='config.json'>config</a> ·
<a href='decision.json'>decision</a> · <a href='metrics.json'>JSON</a> ·
<a href='metrics.jsonl'>JSONL</a> · <a href='metrics.csv'>CSV</a> ·
<a href='attempts.json'>attempts</a> · <a href='manifest.json'>manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>arm</th><th>bg</th><th>N</th><th>PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>coverage</th>
<th>visuals</th></tr>{''.join(table)}</table><h2>Visual audit</h2>{''.join(cards)}
</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _recover(args, images, output_root, command) -> bool:
    if args.recover_from is None:
        return False
    source_root = args.recover_from.resolve()
    paths = [source_root / name for name in ("metrics.json", "attempts.json", "decision.json")]
    if not source_root.is_dir() or not all(path.is_file() for path in paths):
        raise SystemExit("recovery source is missing its ledgers")
    rows = json.loads(paths[0].read_text(encoding="utf-8")).get("rows", [])
    attempts = json.loads(paths[1].read_text(encoding="utf-8")).get("attempts", [])
    decision = json.loads(paths[2].read_text(encoding="utf-8"))
    expected = len(DEVELOPMENT_BINDINGS) * len(DEVELOPMENT_ARMS)
    if len(rows) != expected or len(attempts) != expected or any(
        record.get("status") != "ok" for record in attempts
    ):
        raise SystemExit("recovery source is not a complete successful HIER-018 run")
    shutil.copytree(source_root, output_root)
    snapshot = output_root / "recovery_source_snapshot" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), snapshot)
    decision.update({"recovered_from_complete_raw_run": True, "cell_computation_rerun": False})
    report_utils._write_json(output_root / "decision.json", decision)
    report_utils._write_json(
        output_root / "recovery.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": command,
            "source_path": str(source_root),
            "source_metrics_sha256": report_utils._sha256(paths[0]),
            "source_attempts_sha256": report_utils._sha256(paths[1]),
            "source_decision_sha256": report_utils._sha256(paths[2]),
            "cell_computation_rerun": False,
            "recovery_driver_snapshot": str(snapshot.relative_to(output_root)),
            "recovery_driver_sha256": report_utils._sha256(snapshot),
            "source_count": len(images),
        },
    )
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    _write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    _validate_development_decision(args)
    images = _discover_images(args)
    output_root = args.out.resolve()

    import torch

    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])])
    if _recover(args, images, output_root, command):
        return 0
    output_root.mkdir(parents=True, exist_ok=False)
    contraction_config = h15._contraction_config(args)
    baseline_init, background_init, fit_config = _configs(args)
    arms = DEVELOPMENT_ARMS if args.phase == "development" else REPLAY_ARMS
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "command": command,
        "args": vars(args),
        "arms": list(arms),
        "development_selection_digests": SELECTION_DIGESTS,
        "sources": [{"path": str(path), "sha256": report_utils._sha256(path)} for path in images],
        "contraction": asdict(contraction_config),
        "baseline_init": asdict(baseline_init),
        "background_init": asdict(background_init),
        "fit": asdict(fit_config),
        "source_snapshots": _snapshot_sources(output_root),
        "git": report_utils._git_record(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "limitations": [
            "Dirty-source one-seed diagnostic without distinct review.",
            "CUDA accumulation is numerically, not bit, reproducible.",
            "Consumed replays cannot tune the fixed CORE-009 recipe.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record(image_path, arm, started, error=None):
        item = {
            "image": image_path.stem, "arm": arm,
            "status": "ok" if error is None else "error",
            "elapsed_seconds": time.perf_counter() - started,
        }
        if error is not None:
            item["error"] = f"{type(error).__name__}: {error}"[:1000]
        attempts.append(item)
        report_utils._write_json(
            output_root / "attempts.json",
            {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
        )

    def persist():
        h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)

    run_started = time.perf_counter()
    for image_path in images:
        load_started = time.perf_counter()
        try:
            image, loaded_mask, raster = report_utils._load_evaluation_raster(
                image_path, None, max_side=args.max_side, mask_threshold=0.5
            )
            if loaded_mask is not None:
                raise RuntimeError("HIER-018 requires a generated full-frame mask")
        except Exception as exc:
            for arm in arms:
                record(image_path, arm, load_started, exc)
            continue
        mask = np.ones(image.shape[:2], dtype=bool)
        control_reconstruction = image
        if CONTROL_ARM in arms:
            started = time.perf_counter()
            try:
                h17._seed_everything(args.seed)
                torch.cuda.reset_peak_memory_stats()
                control = contract_image(image, contraction_config, mask=mask)
                seconds = time.perf_counter() - started
                row = h15._write_observation_cell(
                    output_root=output_root, image_path=image_path, image=image, mask=mask,
                    raster=raster, arm=CONTROL_ARM, field=control.field,
                    control_field=control.field, control_reconstruction=control.reconstruction,
                    expected=control.reconstruction, contraction_seconds=seconds,
                    method_seconds=0.0, projection=None, alternating=None,
                    peak_cuda_bytes=int(torch.cuda.max_memory_allocated()), args=args,
                    schema=REPORT_SCHEMA,
                )
                rows.append(row)
                control_reconstruction = control.reconstruction
                record(image_path, CONTROL_ARM, started)
                persist()
            except Exception as exc:
                record(image_path, CONTROL_ARM, started, exc)

        baseline_field = baseline_render = None
        started = time.perf_counter()
        try:
            h17._seed_everything(args.seed)
            init_started = time.perf_counter()
            initial = build_field(
                image, baseline_init, StructureTensorConfig(), device=args.device
            )
            init_seconds = time.perf_counter() - init_started
            init_hash = h15._gaussian_content_hash(initial)
            baseline_field, result, peak = h17._run_fit(initial, image, fit_config)
            extra = {
                "background_count": int(result["background_count"]),
                "detail_count": int(result["detail_count"]),
                "background_geometry_bit_exact": True,
                "background_geometry_persistence_bit_exact": True,
                "background_mean_shift_max": 0.0,
                "background_log_scale_shift_max": 0.0,
                "background_rotation_shift_max": 0.0,
            }
            row, baseline_render = h17._write_cell(
                output_root=output_root, image_path=image_path, image=image, mask=mask,
                raster=raster, arm=BASELINE_ARM, field=baseline_field, fit_result=result,
                init_seconds=init_seconds, control_reconstruction=control_reconstruction,
                peak_cuda_bytes=peak, fit_config=fit_config, init_hash=init_hash, args=args,
                extra_row=extra, schema=REPORT_SCHEMA,
            )
            rows.append(row)
            record(image_path, BASELINE_ARM, started)
            persist()
        except Exception as exc:
            record(image_path, BASELINE_ARM, started, exc)

        started = time.perf_counter()
        try:
            h17._seed_everything(args.seed)
            init_started = time.perf_counter()
            initial_bg = build_field(
                image, background_init, StructureTensorConfig(), device=args.device
            )
            init_seconds = time.perf_counter() - init_started
            init_hash = h15._gaussian_content_hash(initial_bg)
            geometry_hash_before = _background_geometry_hash(initial_bg)
            candidate_field, result, peak = h17._run_fit(initial_bg, image, fit_config)
            geometry = _background_geometry_delta(initial_bg, candidate_field)
            geometry["background_geometry_hash_before"] = geometry_hash_before
            geometry["background_geometry_hash_after"] = _background_geometry_hash(candidate_field)
            extra = {
                "background_count": int(result["background_count"]),
                "detail_count": int(result["detail_count"]),
                **geometry,
            }
            row, candidate_render = h17._write_cell(
                output_root=output_root, image_path=image_path, image=image, mask=mask,
                raster=raster, arm=BACKGROUND_ARM, field=candidate_field, fit_result=result,
                init_seconds=init_seconds,
                control_reconstruction=(
                    baseline_render if baseline_render is not None else control_reconstruction
                ),
                peak_cuda_bytes=peak, fit_config=fit_config, init_hash=init_hash, args=args,
                extra_row=extra, schema=REPORT_SCHEMA,
            )
            artifact_dir = output_root / str(row["artifact_dir"])
            persisted_field = GaussianField.load(
                str(artifact_dir / "field.gaussian.npz"), device=args.device
            )
            persisted_geometry = _background_geometry_delta(initial_bg, persisted_field)
            row.update(
                {
                    "background_geometry_persistence_bit_exact": bool(
                        persisted_geometry["background_geometry_bit_exact"]
                    ),
                    "background_persisted_mean_shift_max": persisted_geometry[
                        "background_mean_shift_max"
                    ],
                    "background_persisted_log_scale_shift_max": persisted_geometry[
                        "background_log_scale_shift_max"
                    ],
                    "background_persisted_rotation_shift_max": persisted_geometry[
                        "background_rotation_shift_max"
                    ],
                    "background_geometry_hash_persisted": _background_geometry_hash(
                        persisted_field
                    ),
                }
            )
            row.update(
                _background_coverage(
                    candidate_field, fit_config, image, candidate_render, artifact_dir
                )
            )
            if baseline_field is not None and baseline_render is not None:
                row.update(
                    _coverage_improvement(
                        baseline_field, fit_config, image, baseline_render, candidate_render
                    )
                )
            report_utils._write_json(artifact_dir / "row.json", row)
            rows.append(row)
            record(image_path, BACKGROUND_ARM, started)
            persist()
        except Exception as exc:
            record(image_path, BACKGROUND_ARM, started, exc)

    decision = (
        _development_decision(rows, attempts)
        if args.phase == "development" else _replay_decision(rows, attempts, args)
    )
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    _write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
