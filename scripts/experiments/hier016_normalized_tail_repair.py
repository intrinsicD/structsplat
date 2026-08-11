#!/usr/bin/env python3
"""Run HIER-016's source-bound normalized exact-7k tail-repair diagnostic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from structsplat.normalized_refinement import (  # noqa: E402
    NormalizedTailRefinementConfig,
    NormalizedTailRefinementResult,
    refine_normalized_color_tail,
)
from structsplat.pixel_contraction import contract_image  # noqa: E402


REPORT_SCHEMA = "structsplat.hier016_normalized_tail_repair.diagnostic.v1"
DEVELOPMENT_BINDINGS = {
    "COCO_train2014_000000229559.jpg": (
        "12909150daa0ca6162a5d2fa7cd7c87b5526c85ca583563952eb06d241397972"
    ),
    "COCO_train2014_000000160926.jpg": (
        "0de693701b819d1a58fe8c4a84745029e52ccea8010ef637bee2181a36a73321"
    ),
    "COCO_train2014_000000380591.jpg": (
        "1a534dc62ff9d0c91bfdf68fcae720e0e9ee234120fd6b8e1b2be98e38f78511"
    ),
    "COCO_train2014_000000198396.jpg": (
        "c61f811798a871c8867c524f51ddc32f40ceff7f81bc6859c4da91dd5d27e0dd"
    ),
}
SELECTION_DIGESTS = {
    "COCO_train2014_000000229559.jpg": (
        "00015a0657e9b73b9234a76e72fde86f4ab361f0f1e1423a9603165973780c1c"
    ),
    "COCO_train2014_000000160926.jpg": (
        "00016d4361597b41788f3f0ae46c1a771633f453754701f9269c55357a92d4cc"
    ),
    "COCO_train2014_000000380591.jpg": (
        "0001e9e9b17208099d3b766d6f8ed96323844fe1de3320b5aace17fa585eb438"
    ),
    "COCO_train2014_000000198396.jpg": (
        "00021264a64ee08587d2fc0d4c330674dd999c5aca64524bc9b0957e958be088"
    ),
}
DIRECT_ARM = h15.DIRECT_ARM
TAIL_ARMS = ("tail_top1pct", "tail_top0_1pct")
DEVELOPMENT_ARMS = ("h005_control", DIRECT_ARM, *TAIL_ARMS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("development", "replay_h15", "replay_tests"),
        required=True,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--disposition", choices=TAIL_ARMS)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--h15-control-metrics", type=Path)
    parser.add_argument(
        "--recover-from",
        type=Path,
        help="copy a complete pre-decision development run and regenerate only reports",
    )
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direct-fit-steps", type=int, default=750)
    parser.add_argument("--tail-steps", type=int, default=100)
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
        "tail_steps": 100,
        "device": "cuda",
        "additive_renderer": "cuda_additive",
        "direct_renderer": "cuda",
        "render_chunk": 256,
        "lpips": True,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-016 protocol requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    replay = args.phase != "development"
    if replay and (args.disposition is None or args.development_decision is None):
        raise SystemExit("replay requires --disposition and --development-decision")
    if not replay and (args.disposition is not None or args.development_decision is not None):
        raise SystemExit("development does not accept replay disposition arguments")
    if args.phase == "replay_h15" and args.h15_control_metrics is None:
        raise SystemExit("replay_h15 requires --h15-control-metrics")
    if args.phase != "replay_h15" and args.h15_control_metrics is not None:
        raise SystemExit("--h15-control-metrics is valid only for replay_h15")
    if args.recover_from is not None and args.phase != "development":
        raise SystemExit("--recover-from is valid only for development")


def _validate_development_decision(args: argparse.Namespace) -> dict[str, object] | None:
    if args.phase == "development":
        return None
    decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    if decision.get("schema") != REPORT_SCHEMA or decision.get("phase") != "development":
        raise SystemExit("--development-decision is not a HIER-016 development decision")
    if args.disposition not in decision.get("numeric_candidates", []):
        raise SystemExit("replay disposition did not pass the frozen numeric gate")
    return decision


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
    if args.phase in ("development", "replay_h15"):
        bindings = DEVELOPMENT_BINDINGS if args.phase == "development" else h15.DEVELOPMENT_BINDINGS
        return _bound_paths(args.images, bindings)
    return h13._discover_sources([args.images])


def _tail_config(args: argparse.Namespace, arm: str) -> NormalizedTailRefinementConfig:
    fractions = {"tail_top1pct": 0.01, "tail_top0_1pct": 0.001}
    return NormalizedTailRefinementConfig(
        steps=args.tail_steps,
        checkpoint_every=5,
        learning_rate=0.01,
        tail_fraction=fractions[arm],
        tail_weight=4.0,
        max_color_shift=1.0,
        color_abs_limit=8.0,
        sse_relative_tolerance=1e-8,
        display_absolute_tolerance=1e-12,
    )


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "structsplat" / "normalized_refinement.py",
        ROOT / "src" / "structsplat" / "fit.py",
        ROOT / "src" / "structsplat" / "init.py",
        ROOT / "scripts" / "experiments" / "hier015_geometry_escape.py",
        ROOT / "tasks" / "HIER-016-normalized-tail-safe-refinement.md",
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


def _tail_fit_result(result: NormalizedTailRefinementResult) -> dict[str, object]:
    import torch

    last_step = max(checkpoint.step for checkpoint in result.checkpoints)
    return {
        "render": torch.as_tensor(np.array(result.reconstruction_raw, copy=True)),
        "history": {
            "kind": "normalized_color_tail",
            "checkpoints": result.checkpoint_records(),
        },
        "fit_seconds": result.elapsed_seconds,
        "iterations_run": last_step,
        "selected_iter": result.selected_step,
    }


def _write_tail_cell(
    *,
    output_root: Path,
    image_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    raster: dict[str, object],
    arm: str,
    result: NormalizedTailRefinementResult,
    direct_seconds: float,
    control_reconstruction: np.ndarray,
    peak_cuda_bytes: int,
    fit_config,
    tail_config: NormalizedTailRefinementConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    selected = next(
        checkpoint for checkpoint in result.checkpoints if checkpoint.selected
    )
    return h15._write_direct_cell(
        output_root=output_root,
        image_path=image_path,
        image=image,
        mask=mask,
        raster=raster,
        field=result.field,
        fit_result=_tail_fit_result(result),
        init_seconds=direct_seconds,
        control_reconstruction=control_reconstruction,
        peak_cuda_bytes=peak_cuda_bytes,
        fit_config=fit_config,
        args=args,
        arm=arm,
        schema=REPORT_SCHEMA,
        extra_row={
            "tail_fraction": tail_config.tail_fraction,
            "tail_weight": tail_config.tail_weight,
            "tail_count": result.tail_count,
            "selected_tail_step": result.selected_step,
            "tail_transaction_safe": selected.eligible,
            "tail_internal_render_parity_max_abs": (
                result.maintained_render_parity_max_abs
            ),
            "tail_initial_sse": result.initial_sse,
            "tail_final_sse": result.final_sse,
            "tail_initial_display_pixel_rmse_max": (
                result.initial_display_pixel_rmse_max
            ),
            "tail_final_display_pixel_rmse_max": (
                result.final_display_pixel_rmse_max
            ),
            "tail_initial_display_patch7_rmse_max": (
                result.initial_display_patch7_rmse_max
            ),
            "tail_final_display_patch7_rmse_max": (
                result.final_display_patch7_rmse_max
            ),
            "color_shift_max": result.color_shift_max,
            "non_color_arrays_bit_exact": result.non_color_arrays_bit_exact,
            "direct_algorithm_seconds": direct_seconds,
            "tail_seconds": result.elapsed_seconds,
            "method_overhead_ratio": result.elapsed_seconds / max(direct_seconds, 1e-12),
            "geometry_changed": False,
            "selected_geometry_steps": 0,
        },
    )


def _pairs(
    rows: list[dict[str, object]], arm: str, control_arm: str
) -> list[dict[str, object]]:
    controls = {
        str(row["image"]): row for row in rows if row["arm"] == control_arm
    }
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row["arm"] != arm or str(row["image"]) not in controls:
            continue
        control = controls[str(row["image"])]
        pairs.append(
            {
                "image": row["image"],
                "mse_ratio": float(row["masked_mse"]) / float(control["masked_mse"]),
                "psnr_delta_db": float(row["psnr_db"]) - float(control["psnr_db"]),
                "ms_ssim_delta": float(row["ms_ssim"]) - float(control["ms_ssim"]),
                "lpips_delta": float(row["lpips"]) - float(control["lpips"]),
                "pixel_max_ratio": float(row["artifact_pixel_rmse_max"])
                / max(float(control["artifact_pixel_rmse_max"]), 1e-12),
                "pixel_max_delta": float(row["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "patch7_max_delta": float(row["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
                "n_gaussians": row["n_gaussians"],
                "coefficient_abs_max": row["coefficient_abs_max"],
                "color_shift_max": row.get("color_shift_max"),
                "selected_tail_step": row.get("selected_tail_step"),
                "tail_transaction_safe": row.get("tail_transaction_safe"),
                "non_color_arrays_bit_exact": row.get("non_color_arrays_bit_exact"),
                "tail_internal_render_parity_max_abs": row.get(
                    "tail_internal_render_parity_max_abs"
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


def _geometric_mean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def _tail_gate(
    h005_pairs: list[dict[str, object]],
    direct_pairs: list[dict[str, object]],
) -> tuple[dict[str, bool], dict[str, float]]:
    finite_keys = (
        "mse_ratio",
        "psnr_delta_db",
        "ms_ssim_delta",
        "lpips_delta",
        "pixel_max_delta",
        "patch7_max_delta",
    )
    gate = {
        "complete_four_h005_pairs": len(h005_pairs) == 4,
        "complete_four_direct_pairs": len(direct_pairs) == 4,
        "all_finite": all(
            math.isfinite(float(pair[key]))
            for pair in (*h005_pairs, *direct_pairs)
            for key in finite_keys
        ),
        "all_exact_count": all(
            int(pair["n_gaussians"]) == 7000 for pair in direct_pairs
        ),
        "all_non_color_bit_exact": all(
            bool(pair["non_color_arrays_bit_exact"]) for pair in direct_pairs
        ),
        "all_color_abs_le_8": all(
            float(pair["coefficient_abs_max"]) <= 8.0 + 1e-7 for pair in direct_pairs
        ),
        "all_color_shift_le_1": all(
            float(pair["color_shift_max"]) <= 1.0 + 1e-7 for pair in direct_pairs
        ),
        "all_transactions_safe": all(
            bool(pair["tail_transaction_safe"]) for pair in direct_pairs
        ),
        "all_parity_le_2e_5": all(
            float(pair["tail_internal_render_parity_max_abs"]) <= 2e-5
            and float(pair["maintained_render_parity_max_abs"]) <= 2e-5
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
            for pair in direct_pairs
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
            float(np.mean([float(pair["ms_ssim_delta"]) for pair in h005_pairs])) >= -1e-7
            if h005_pairs
            else False
        ),
        "mean_lpips_noninferior_vs_h005": (
            float(np.mean([float(pair["lpips_delta"]) for pair in h005_pairs])) <= 1e-7
            if h005_pairs
            else False
        ),
        "all_mse_noninferior_vs_direct": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in direct_pairs
        ),
        "all_pixel_max_noninferior_vs_direct": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in direct_pairs
        ),
        "all_patch7_max_noninferior_vs_direct": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in direct_pairs
        ),
        "mean_ms_ssim_delta_vs_direct_ge_neg_0_001": (
            float(np.mean([float(pair["ms_ssim_delta"]) for pair in direct_pairs])) >= -0.001
            if direct_pairs
            else False
        ),
        "mean_lpips_delta_vs_direct_le_0_002": (
            float(np.mean([float(pair["lpips_delta"]) for pair in direct_pairs])) <= 0.002
            if direct_pairs
            else False
        ),
        "at_least_one_nonzero_refinement": any(
            int(pair["selected_tail_step"]) > 0 for pair in direct_pairs
        ),
    }
    score = {
        "maximum_pixel_max_ratio_vs_h005": max(
            (float(pair["pixel_max_ratio"]) for pair in h005_pairs), default=math.inf
        ),
        "geometric_mean_mse_ratio_vs_direct": _geometric_mean(
            [float(pair["mse_ratio"]) for pair in direct_pairs]
        )
        if direct_pairs
        else math.inf,
        "mean_ms_ssim_delta_vs_direct": float(
            np.mean([float(pair["ms_ssim_delta"]) for pair in direct_pairs])
        )
        if direct_pairs
        else 0.0,
        "mean_lpips_delta_vs_direct": float(
            np.mean([float(pair["lpips_delta"]) for pair in direct_pairs])
        )
        if direct_pairs
        else 0.0,
    }
    return gate, score


def _development_decision(
    rows: list[dict[str, object]], attempts: list[dict[str, object]]
) -> dict[str, object]:
    arms: dict[str, object] = {}
    candidates: list[str] = []
    for arm in TAIL_ARMS:
        h005_pairs = _pairs(rows, arm, "h005_control")
        direct_pairs = _pairs(rows, arm, DIRECT_ARM)
        gate, score = _tail_gate(h005_pairs, direct_pairs)
        arms[arm] = {
            "h005_pairs": h005_pairs,
            "direct_pairs": direct_pairs,
            "gate": gate,
            "score": score,
        }
        if all(gate.values()):
            candidates.append(arm)
    preference = {"tail_top0_1pct": 0, "tail_top1pct": 1}
    candidates.sort(
        key=lambda arm: (
            float(arms[arm]["score"]["maximum_pixel_max_ratio_vs_h005"]),
            float(arms[arm]["score"]["geometric_mean_mse_ratio_vs_direct"]),
            preference[arm],
        )
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "development",
        "arms": arms,
        "attempt_count": len(attempts),
        "failure_count": sum(record["status"] != "ok" for record in attempts),
        "numeric_candidates": candidates,
        "numeric_disposition": candidates[0] if candidates else "no_tail_candidate",
        "visual_review_required": True,
        "interpretation": (
            "Numeric candidates require frozen full-frame/worst-crop visual review before replay."
            if candidates
            else "No tail arm clears the frozen numeric gate; do not access consumed banks."
        ),
    }


def _replay_decision(
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    pairs = _pairs(rows, str(args.disposition), DIRECT_ARM)
    expected = 4 if args.phase == "replay_h15" else 16
    clauses = {
        "complete_pairs": len(pairs) == expected,
        "all_exact_count": all(int(pair["n_gaussians"]) == 7000 for pair in pairs),
        "all_non_color_bit_exact": all(
            bool(pair["non_color_arrays_bit_exact"]) for pair in pairs
        ),
        "all_transactions_safe": all(
            bool(pair["tail_transaction_safe"]) for pair in pairs
        ),
        "all_mse_noninferior_vs_direct": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in pairs
        ),
        "all_pixel_max_noninferior_vs_direct": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_patch7_max_noninferior_vs_direct": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in pairs
        ),
        "all_parity_le_2e_5": all(
            float(pair["tail_internal_render_parity_max_abs"]) <= 2e-5
            and float(pair["maintained_render_parity_max_abs"]) <= 2e-5
            and float(pair["repeated_render_parity_max_abs"]) <= 2e-5
            for pair in pairs
        ),
    }
    h15_comparison: dict[str, object] | None = None
    if args.phase == "replay_h15":
        report = json.loads(args.h15_control_metrics.read_text(encoding="utf-8"))
        controls = {
            str(row["image"]): row
            for row in report["rows"]
            if row["arm"] == "h005_control"
        }
        tail_rows = {
            str(row["image"]): row
            for row in rows
            if row["arm"] == args.disposition
        }
        comparisons = []
        for image in sorted(controls):
            if image not in tail_rows:
                continue
            control = controls[image]
            tail = tail_rows[image]
            comparisons.append(
                {
                    "image": image,
                    "psnr_delta_db": float(tail["psnr_db"]) - float(control["psnr_db"]),
                    "pixel_max_delta": float(tail["artifact_pixel_rmse_max"])
                    - float(control["artifact_pixel_rmse_max"]),
                    "patch7_max_delta": float(tail["artifact_patch_rmse_max_7"])
                    - float(control["artifact_patch_rmse_max_7"]),
                }
            )
        h15_clauses = {
            "complete_four": len(comparisons) == 4,
            "all_psnr_gain_ge_2_db": all(
                row["psnr_delta_db"] >= 2.0 for row in comparisons
            ),
            "all_pixel_max_noninferior": all(
                row["pixel_max_delta"] <= 1e-12 for row in comparisons
            ),
            "all_patch7_max_noninferior": all(
                row["patch7_max_delta"] <= 1e-12 for row in comparisons
            ),
        }
        h15_comparison = {"rows": comparisons, "clauses": h15_clauses}
        clauses["known_h15_counterexample_repaired"] = all(h15_clauses.values())
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "disposition": args.disposition,
        "pairs": pairs,
        "clauses": clauses,
        "h15_comparison": h15_comparison,
        "attempt_count": len(attempts),
        "failure_count": sum(record["status"] != "ok" for record in attempts),
        "bounded_bank_pass": all(clauses.values()),
        "interpretation": "Consumed reporting replay; no retuning or held-out claim.",
    }


def _write_report(
    output_root: Path,
    rows: list[dict[str, object]],
    decision: dict[str, object],
    command: str,
) -> None:
    table = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{escape(str(row['arm']))}</td>"
            f"<td>{int(row['n_gaussians'])}</td><td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td>{row.get('selected_tail_step', '')}</td>"
            f"<td><a href='{artifact}/reconstruction.png'>full</a> · "
            f"<a href='{artifact}/reconstruction_crop.png'>crop</a></td></tr>"
        )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} — {escape(str(row['arm']))}</h3>"
            f"<img src='{artifact}/source.png'><img src='{artifact}/reconstruction.png'>"
            f"<img src='{artifact}/error.png'><img src='{artifact}/reconstruction_crop.png'>"
            "</section>"
        )
    decision_text = escape(json.dumps(decision, indent=2, sort_keys=True))
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>HIER-016 normalized tail repair</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1600px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-016 normalized tail repair — {escape(str(decision['phase']))}</h1>
<p>Dirty-source diagnostic. Visual review is mandatory before replay or bounded disposition.</p>
<p><code>{escape(command)}</code></p><p><a href='config.json'>config</a> ·
<a href='decision.json'>decision</a> · <a href='metrics.json'>metrics</a> ·
<a href='attempts.json'>attempts</a> · <a href='manifest.json'>manifest</a></p>
<h2>Decision</h2><pre>{decision_text}</pre><h2>Cells</h2><table><tr><th>image</th>
<th>arm</th><th>N</th><th>PSNR</th><th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th>
<th>7x7 max</th><th>tail step</th><th>artifacts</th></tr>{''.join(table)}</table>
<h2>Visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    _validate_development_decision(args)
    images = _discover_images(args)
    output_root = args.out.resolve()

    import torch

    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])])
    if args.recover_from is not None:
        source_root = args.recover_from.resolve()
        if not source_root.is_dir():
            raise SystemExit(f"recovery source is not a directory: {source_root}")
        source_metrics = source_root / "metrics.json"
        source_attempts = source_root / "attempts.json"
        if not source_metrics.is_file() or not source_attempts.is_file():
            raise SystemExit("recovery source lacks metrics.json or attempts.json")
        shutil.copytree(source_root, output_root)
        rows = json.loads(source_metrics.read_text(encoding="utf-8"))["rows"]
        attempts = json.loads(source_attempts.read_text(encoding="utf-8"))["attempts"]
        expected_cells = len(DEVELOPMENT_BINDINGS) * len(DEVELOPMENT_ARMS)
        if len(rows) != expected_cells or len(attempts) != expected_cells:
            raise SystemExit(
                f"recovery source is incomplete: rows={len(rows)}, attempts={len(attempts)}, "
                f"expected={expected_cells}"
            )
        if any(record.get("status") != "ok" for record in attempts):
            raise SystemExit("recovery source contains failed cells")
        recovery_snapshot = output_root / "recovery_source_snapshot" / Path(__file__).name
        recovery_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), recovery_snapshot)
        decision = _development_decision(rows, attempts)
        decision["elapsed_seconds"] = None
        decision["recovered_from_complete_raw_run"] = True
        report_utils._write_json(output_root / "decision.json", decision)
        report_utils._write_json(
            output_root / "recovery.json",
            {
                "schema": REPORT_SCHEMA,
                "status": "diagnostic",
                "command": command,
                "source_path": str(source_root),
                "source_metrics_sha256": report_utils._sha256(source_metrics),
                "source_attempts_sha256": report_utils._sha256(source_attempts),
                "cell_computation_rerun": False,
                "recovery_action": "copy complete raw run; regenerate pure decision/report/manifest",
                "recovery_driver_snapshot": str(recovery_snapshot.relative_to(output_root)),
                "recovery_driver_sha256": report_utils._sha256(recovery_snapshot),
            },
        )
        h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
        _write_report(output_root, rows, decision, command)
        h15._write_manifest(output_root, schema=REPORT_SCHEMA)
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    contraction_config = h15._contraction_config(args)
    init_config, fit_config = h15._direct_configs(args)
    tail_configs = {arm: _tail_config(args, arm) for arm in TAIL_ARMS}
    arms = DEVELOPMENT_ARMS if args.phase == "development" else (DIRECT_ARM, str(args.disposition))
    h15_control_record = None
    if args.h15_control_metrics is not None:
        h15_control_record = {
            "path": str(args.h15_control_metrics.resolve()),
            "sha256": report_utils._sha256(args.h15_control_metrics),
        }
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": args.phase,
        "command": command,
        "args": vars(args),
        "arms": list(arms),
        "development_selection_digests": SELECTION_DIGESTS,
        "sources": [
            {"path": str(path), "sha256": report_utils._sha256(path)} for path in images
        ],
        "contraction": asdict(contraction_config),
        "direct_init": asdict(init_config),
        "direct_fit": asdict(fit_config),
        "tail": {arm: asdict(cfg) for arm, cfg in tail_configs.items()},
        "h15_control_metrics": h15_control_record,
        "source_snapshots": _snapshot_sources(output_root),
        "git": report_utils._git_record(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "limitations": [
            "Dirty-source, one-seed diagnostic without distinct prospective review.",
            "CUDA accumulation is numerically, not bit, reproducible.",
            "Consumed replays are reporting-only and cannot tune the recipe.",
            "Lossless NPZ artifacts are reference persistence, not production codec rate.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record(image_path: Path, arm: str, started: float, error: Exception | None = None) -> None:
        item: dict[str, object] = {
            "image": image_path.stem,
            "arm": arm,
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

    def persist() -> None:
        h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)

    run_started = time.perf_counter()
    for image_path in images:
        load_started = time.perf_counter()
        try:
            image, loaded_mask, raster = report_utils._load_evaluation_raster(
                image_path, None, max_side=args.max_side, mask_threshold=0.5
            )
            if loaded_mask is not None:
                raise RuntimeError("HIER-016 requires a generated full-frame mask")
        except Exception as exc:
            for arm in arms:
                record(image_path, arm, load_started, exc)
            continue
        mask = np.ones(image.shape[:2], dtype=bool)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        control = None
        contraction_seconds = 0.0
        contraction_peak = 0
        if "h005_control" in arms:
            contraction_started = time.perf_counter()
            try:
                torch.cuda.reset_peak_memory_stats()
                control = contract_image(image, contraction_config, mask=mask)
                contraction_seconds = time.perf_counter() - contraction_started
                contraction_peak = int(torch.cuda.max_memory_allocated())
                cell_started = time.perf_counter()
                row = h15._write_observation_cell(
                    output_root=output_root,
                    image_path=image_path,
                    image=image,
                    mask=mask,
                    raster=raster,
                    arm="h005_control",
                    field=control.field,
                    control_field=control.field,
                    control_reconstruction=control.reconstruction,
                    expected=control.reconstruction,
                    contraction_seconds=contraction_seconds,
                    method_seconds=0.0,
                    projection=None,
                    alternating=None,
                    peak_cuda_bytes=contraction_peak,
                    args=args,
                    schema=REPORT_SCHEMA,
                )
                rows.append(row)
                record(image_path, "h005_control", cell_started)
                persist()
            except Exception as exc:
                record(image_path, "h005_control", contraction_started, exc)

        direct_field = None
        direct_result = None
        direct_fit_config = fit_config
        direct_seconds = 0.0
        direct_peak = 0
        direct_started = time.perf_counter()
        try:
            direct_field, direct_result, init_seconds, direct_fit_config, direct_peak = (
                h15._run_direct(image, args)
            )
            direct_seconds = init_seconds + float(direct_result["fit_seconds"])
            row = h15._write_direct_cell(
                output_root=output_root,
                image_path=image_path,
                image=image,
                mask=mask,
                raster=raster,
                field=direct_field,
                fit_result=direct_result,
                init_seconds=init_seconds,
                control_reconstruction=(image if control is None else control.reconstruction),
                peak_cuda_bytes=direct_peak,
                fit_config=direct_fit_config,
                args=args,
                schema=REPORT_SCHEMA,
                extra_row={"direct_checkpoint_zero": True},
            )
            rows.append(row)
            record(image_path, DIRECT_ARM, direct_started)
            persist()
        except Exception as exc:
            record(image_path, DIRECT_ARM, direct_started, exc)

        requested_tail_arms = TAIL_ARMS if args.phase == "development" else (str(args.disposition),)
        if direct_field is None or direct_result is None:
            failure = RuntimeError("shared direct checkpoint zero failed")
            for arm in requested_tail_arms:
                record(image_path, arm, direct_started, failure)
            continue
        for arm in requested_tail_arms:
            tail_started = time.perf_counter()
            try:
                torch.cuda.reset_peak_memory_stats()
                result = refine_normalized_color_tail(
                    direct_field,
                    image,
                    mask,
                    direct_fit_config,
                    config=tail_configs[arm],
                )
                peak = max(direct_peak, int(torch.cuda.max_memory_allocated()))
                row = _write_tail_cell(
                    output_root=output_root,
                    image_path=image_path,
                    image=image,
                    mask=mask,
                    raster=raster,
                    arm=arm,
                    result=result,
                    direct_seconds=direct_seconds,
                    control_reconstruction=direct_result["render"].detach().cpu().numpy(),
                    peak_cuda_bytes=peak,
                    fit_config=direct_fit_config,
                    tail_config=tail_configs[arm],
                    args=args,
                )
                rows.append(row)
                record(image_path, arm, tail_started)
                persist()
            except Exception as exc:
                record(image_path, arm, tail_started, exc)

    decision = (
        _development_decision(rows, attempts)
        if args.phase == "development"
        else _replay_decision(rows, attempts, args)
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
