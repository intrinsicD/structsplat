#!/usr/bin/env python3
"""Run CORE-019's frozen four-arm coherent-depth development protocol.

This driver is intentionally tied to ``tasks/CORE-019-calibrated-coherent-depth-fusion.md``. It
creates one packet set from construction cameras, cold-reloads those bytes for every arm, excludes
the four preregistered reporting cameras from every construction operation, and persists complete
quality/convergence/rate/time/visual receipts. Source RGB is unavailable to coherent-depth
construction after packet creation.

Reproduce from the StructSplat root with::

    PYTHONPATH=/home/alex/Documents/vggt:/home/alex/Documents/realtime-gs/src:src \
      /home/alex/Documents/realtime-gs/.venv/bin/python \
      scripts/experiments/core019_coherent_depth_downstream.py \
      --frame /home/alex/Dropbox/Work/Janelle/karate/frame_00005 \
      --weights /home/alex/.cache/huggingface/hub/models--facebook--VGGT-1B/blobs/f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e.repairing-20260807 \
      --out results/core019_coherent_depth_karate_frame00005_2026-08-07_v4

The output directory is immutable: this script refuses to overwrite it.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
from html import escape
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rtgs.data.calibrated import load_calibrated_scene
from rtgs.render.base import get_rasterizer

from structsplat.realtime_gs_coherent_depth import (
    CoherentDepthConfig,
    CoherentDepthField,
    VGGT_MODEL_BYTES,
    VGGT_MODEL_REVISION,
    VGGT_MODEL_SHA256,
    VGGT_SOURCE_REVISION,
    infer_coherent_depth_field,
    initialize_calibrated_coherent_depth,
)


STRUCTSPLAT_ROOT = Path(__file__).resolve().parents[2]
BASE_DRIVER_PATH = Path(__file__).with_name("core018_ray_posterior_downstream.py")
_BASE_SPEC = importlib.util.spec_from_file_location("_core018_driver", BASE_DRIVER_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("CORE-018 control driver cannot be loaded")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)
BASE_CARVE_CONFIG = base._carve_config

DEFAULT_RTGS_ROOT = Path("/home/alex/Documents/realtime-gs")
DEFAULT_FRAME = Path("/home/alex/Dropbox/Work/Janelle/karate/frame_00005")
ALL_CAMERA_IDS = (
    "C0004",
    "C0005",
    "C0006",
    "C0007",
    "C0008",
    "C0010",
    "C0012",
    "C0013",
    "C0014",
    "C0016",
    "C0018",
    "C0020",
    "C0021",
    "C0022",
    "C0024",
    "C0025",
    "C0026",
    "C0028",
    "C0029",
    "C0030",
    "C0031",
    "C0034",
    "C0037",
    "C0038",
    "C0039",
    "C1000",
    "C1001",
    "C1002",
    "C1004",
    "C1005",
)
REPORT_CAMERA_IDS = ("C0024", "C0010", "C1004", "C0022")
TRAIN_CAMERA_IDS = tuple(value for value in ALL_CAMERA_IDS if value not in REPORT_CAMERA_IDS)
ARMS = (
    "interior",
    "posterior_no_reciprocal",
    "vggt_raw_known_ray",
    "vggt_coherent_wse",
)
ARM_LABELS = {
    "interior": "ordinary interior consensus + generic surface cover",
    "posterior_no_reciprocal": "CORE-018 independent ray posterior negative control",
    "vggt_raw_known_ray": "raw calibrated coherent depth + balanced exact budget",
    "vggt_coherent_wse": "projective support + anchored WSE + bounded contraction",
}
PACKET_DOWNSCALE = 4
EVALUATION_DOWNSCALE = 8
INITIAL_GAUSSIANS = 10_000
MAX_GAUSSIANS = 30_000
ITERATIONS = 1_500
FIXED_PREFIX_STEPS = 500
EVAL_EVERY = 100
SEED = 0


def _coherent_config() -> CoherentDepthConfig:
    return CoherentDepthConfig(
        target_count=INITIAL_GAUSSIANS,
        seed=SEED,
        max_anchor_rounds=12,
    )


def _interior_carve_config() -> Any:
    """Mechanical capacity correction; support thresholds remain the CORE-018 control's."""

    return dataclasses.replace(BASE_CARVE_CONFIG(), candidate_multiplier=4)


def _save_field_bundle(field: CoherentDepthField, out: Path) -> dict[str, Any]:
    root = out / "coherent_depth_field"
    root.mkdir(parents=True, exist_ok=True)
    arrays = root / "fused_depth.npz"
    np.savez_compressed(
        arrays,
        depth=field.depth.numpy(),
        uncertainty=field.uncertainty.numpy(),
        confidence=field.confidence.numpy(),
        normals=field.normals.numpy(),
    )
    receipt = root / "receipt.json"
    base._write_json(receipt, field.diagnostics)

    selected = [0, len(field.depth) // 3, 2 * len(field.depth) // 3, len(field.depth) - 1]
    figure, axes = plt.subplots(len(selected), 4, figsize=(15, 3.5 * len(selected)))
    for row, view_index in enumerate(selected):
        axes[row, 0].imshow(field.images[view_index].permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"packet view {view_index}")
        axes[row, 1].imshow(field.depth[view_index].numpy(), cmap="turbo")
        axes[row, 1].set_title("fused depth")
        axes[row, 2].imshow(field.uncertainty[view_index].numpy(), cmap="magma")
        axes[row, 2].set_title("depth uncertainty")
        axes[row, 3].imshow(field.confidence[view_index].numpy(), vmin=0.0, vmax=1.0)
        axes[row, 3].set_title("confidence")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    sheet = root / "construction_field_contact_sheet.png"
    figure.savefig(sheet, dpi=150)
    plt.close(figure)
    return {
        "arrays": base._artifact(arrays, out),
        "receipt": base._artifact(receipt, out),
        "contact_sheet": base._artifact(sheet, out),
    }


def _run_coherent_arm(
    arm: str,
    inputs: Any,
    views: list[Any],
    input_record: dict[str, Any],
    field: CoherentDepthField,
    scene: Any,
    renderer: Any,
    out: Path,
) -> dict[str, Any]:
    root = out / "arms" / arm
    root.mkdir(parents=True, exist_ok=True)
    config = _coherent_config()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    lift_started = time.perf_counter()
    result = initialize_calibrated_coherent_depth(
        inputs,
        views,
        None,
        config,
        field=field,
        apply_projective_support=arm == "vggt_coherent_wse",
    )
    torch.cuda.synchronize()
    lift_seconds = time.perf_counter() - lift_started
    lift_peak_allocated = int(torch.cuda.max_memory_allocated())
    lift_peak_reserved = int(torch.cuda.max_memory_reserved())
    initialization = result.initialization
    initial = initialization.gaussians.to("cpu")
    init_npz = root / "gaussians_init.npz"
    init_ply = root / "gaussians_init.ply"
    initial.save_npz(init_npz)
    initial.save_ply(init_ply)
    curves = [
        {
            "step": 0,
            "optimization_elapsed_seconds": 0.0,
            "n_gaussians": initial.n,
            "metrics": base._evaluate_model(scene, initial.to("cuda"), renderer),
        }
    ]

    def checkpoint(snapshot: Any, step: int) -> None:
        print(f"  [eval] {arm} step={step} N={snapshot.n}", flush=True)
        curves.append(
            {
                "step": step,
                "optimization_elapsed_seconds": None,
                "n_gaussians": snapshot.n,
                "metrics": base._evaluate_model(scene, snapshot, renderer),
            }
        )

    train_config = base._train_config()
    print(f"[train] {arm} N={initial.n} steps={ITERATIONS}", flush=True)
    train_started = time.perf_counter()
    final, history = base.Trainer(train_config).train(
        scene,
        initialization.gaussians,
        checkpoint_callback=checkpoint,
    )
    torch.cuda.synchronize()
    train_wall_seconds = time.perf_counter() - train_started
    elapsed = {int(step): float(value) for step, value in history["elapsed"]}
    for record in curves:
        if record["step"] in elapsed:
            record["optimization_elapsed_seconds"] = elapsed[record["step"]]
    final_cpu = final.to("cpu")
    final_npz = root / "gaussians_final.npz"
    final_ply = root / "gaussians_final.ply"
    final_cpu.save_npz(final_npz)
    final_cpu.save_ply(final_ply)
    history_path = root / "training_history.json"
    curves_path = root / "checkpoint_metrics.json"
    base._write_json(history_path, history)
    base._write_json(curves_path, curves)
    visuals = base._save_visuals(arm, scene, initial, final_cpu, renderer, out)
    query = []
    for name, view in zip(inputs.view_names, views, strict=True):
        backend = view.query_backend
        query.append(
            {
                "view_name": name,
                "index_entries": backend.n_entries,
                "index_payload_bytes": backend.payload_bytes,
                "pairs_evaluated": backend.total_pairs_evaluated,
                "peak_pair_chunk": backend.peak_pair_chunk,
            }
        )
    return {
        "arm": arm,
        "label": ARM_LABELS[arm],
        "status": "ok",
        "input": input_record,
        "coherent_config": dataclasses.asdict(config),
        "train_config": dataclasses.asdict(train_config),
        "lift_seconds": lift_seconds,
        "lift_peak_cuda_allocated_bytes": lift_peak_allocated,
        "lift_peak_cuda_reserved_bytes": lift_peak_reserved,
        "lift_diagnostics": result.diagnostics,
        "query_backends": query,
        "initial_n_gaussians": initial.n,
        "final_n_gaussians": final_cpu.n,
        "training_wall_seconds_including_metrics": train_wall_seconds,
        "training_native_seconds": float(history["elapsed"][-1][1]),
        "peak_vram_gb": float(history["peak_vram_gb"]),
        "initial_metrics": curves[0]["metrics"],
        "final_metrics": curves[-1]["metrics"],
        "curve_rows": curves,
        "curves": base._artifact(curves_path, out),
        "history": base._artifact(history_path, out),
        "models": {
            "initial_npz": base._artifact(init_npz, out),
            "initial_ply": base._artifact(init_ply, out),
            "final_npz": base._artifact(final_npz, out),
            "final_ply": base._artifact(final_ply, out),
        },
        "visuals": visuals,
    }


def _terminal_row(record: dict[str, Any]) -> dict[str, Any]:
    if record["status"] != "ok":
        return {"arm": record["arm"], "status": record["status"], "error": record["error"]}
    metrics = record["final_metrics"]["reporting"]["aggregate"]
    initial = record["initial_metrics"]["reporting"]["aggregate"]
    packet_bytes = record["input"]["complete_packet_bytes"]
    source_bytes = record["input"]["original_source_bytes"]
    model_bytes = record["models"]["final_npz"]["bytes"]
    feature_seconds = (
        float(record["lift_diagnostics"].get("feature_seconds_shared", 0.0))
        if record["arm"] == "posterior_no_reciprocal"
        else 0.0
    )
    field_seconds = (
        float(record["lift_diagnostics"]["coherent_depth_field"].get("total_seconds", 0.0))
        if record["arm"].startswith("vggt_")
        else 0.0
    )
    return {
        "arm": record["arm"],
        "status": "ok",
        "input_bytes": packet_bytes,
        "original_source_bytes": source_bytes,
        "final_model_bytes": model_bytes,
        "encoder_checkpoint_bytes_separate": VGGT_MODEL_BYTES if field_seconds else 0,
        "original_over_packets": source_bytes / packet_bytes,
        "original_over_model": source_bytes / model_bytes,
        "original_over_packets_plus_model": source_bytes / (packet_bytes + model_bytes),
        "initial_n_gaussians": record["initial_n_gaussians"],
        "final_n_gaussians": record["final_n_gaussians"],
        "initial_reporting_psnr": initial["psnr"],
        "initial_reporting_ms_ssim": initial["ms_ssim"],
        "initial_reporting_lpips": initial.get("lpips"),
        "initial_reporting_gradient_mae": initial["gradient_mae"],
        "reporting_psnr": metrics["psnr"],
        "reporting_ssim": metrics["ssim"],
        "reporting_ms_ssim": metrics["ms_ssim"],
        "reporting_lpips": metrics.get("lpips"),
        "reporting_gradient_mae": metrics["gradient_mae"],
        "reporting_p99_abs": metrics["p99_abs"],
        "lift_seconds": record["lift_seconds"],
        "shared_feature_seconds_charged": feature_seconds,
        "shared_coherent_field_seconds_charged": field_seconds,
        "pretraining_seconds": (
            record["input"]["cold_decode_seconds"]
            + record["input"]["adapter_seconds"]
            + record["lift_seconds"]
            + feature_seconds
            + field_seconds
        ),
        "training_native_seconds": record["training_native_seconds"],
        "peak_vram_gb": record["peak_vram_gb"],
    }


def _metric_at(record: dict[str, Any], step: int) -> dict[str, Any]:
    for row in record.get("curve_rows", []):
        if int(row["step"]) == step:
            return row["metrics"]["reporting"]["aggregate"]
    raise KeyError(f"{record['arm']} has no checkpoint at step {step}")


def _first_target(record: dict[str, Any], target: float) -> tuple[int | None, float | None]:
    for row in record.get("curve_rows", []):
        if row["metrics"]["reporting"]["aggregate"]["psnr"] >= target:
            return int(row["step"]), row["optimization_elapsed_seconds"]
    return None, None


def _spacing_tail(record: dict[str, Any]) -> float | None:
    cover = record.get("lift_diagnostics", {}).get("surface_cover")
    if not cover:
        return None
    spacing = cover.get("spacing", {})
    median, p90 = spacing.get("median"), spacing.get("p90")
    if median is None or p90 is None or median <= 0.0:
        return None
    return float(p90 / median)


def _decision(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["status"] != "ok" for row in rows):
        return {
            "advance": False,
            "scalar_pass": False,
            "manual_visual_review_required": True,
            "reason": "one or more arms failed",
        }
    by_arm = {row["arm"]: row for row in rows}
    record_by_arm = {record["arm"]: record for record in records}
    complete = by_arm["vggt_coherent_wse"]
    raw = by_arm["vggt_raw_known_ray"]
    controls = [by_arm[name] for name in ARMS if name != "vggt_coherent_wse"]
    strongest = max(controls, key=lambda row: row["reporting_psnr"])
    interior_initial = by_arm["interior"]
    full_step500 = _metric_at(record_by_arm["vggt_coherent_wse"], FIXED_PREFIX_STEPS)
    control_step500_rows = [
        (row, _metric_at(record_by_arm[row["arm"]], FIXED_PREFIX_STEPS)) for row in controls
    ]
    strongest500_row, strongest500 = max(
        control_step500_rows, key=lambda item: item[1]["psnr"]
    )
    control_step, control_seconds = _first_target(
        record_by_arm[strongest["arm"]], strongest["reporting_psnr"]
    )
    candidate_step, candidate_seconds = _first_target(
        record_by_arm["vggt_coherent_wse"], strongest["reporting_psnr"]
    )

    protected_metrics = ("reporting_psnr", "reporting_ms_ssim")
    inverse_metrics = ("reporting_lpips", "reporting_gradient_mae")
    dominated = False
    for other in controls:
        no_worse = all(other[name] >= complete[name] for name in protected_metrics)
        no_worse &= all(
            other[name] is not None
            and complete[name] is not None
            and other[name] <= complete[name]
            for name in inverse_metrics
        )
        strict = any(other[name] > complete[name] for name in protected_metrics) or any(
            other[name] < complete[name] for name in inverse_metrics
        )
        dominated |= no_worse and strict

    raw_spacing_tail = _spacing_tail(record_by_arm["vggt_raw_known_ray"])
    full_spacing_tail = _spacing_tail(record_by_arm["vggt_coherent_wse"])
    geometry_tail_better = (
        raw_spacing_tail is not None
        and full_spacing_tail is not None
        and full_spacing_tail < raw_spacing_tail
    )
    quality_or_convergence_better = (
        complete["reporting_psnr"] > raw["reporting_psnr"]
        or (
            complete["reporting_lpips"] is not None
            and raw["reporting_lpips"] is not None
            and complete["reporting_lpips"] < raw["reporting_lpips"]
        )
        or complete["training_native_seconds"] < raw["training_native_seconds"]
    )
    gates = {
        "step0_psnr_plus_2db_over_interior": (
            complete["initial_reporting_psnr"]
            >= interior_initial["initial_reporting_psnr"] + 2.0
        ),
        "step0_gradient_mae_no_worse_than_interior": (
            complete["initial_reporting_gradient_mae"]
            <= interior_initial["initial_reporting_gradient_mae"]
        ),
        "step0_lpips_no_worse_than_interior": (
            complete["initial_reporting_lpips"] is not None
            and interior_initial["initial_reporting_lpips"] is not None
            and complete["initial_reporting_lpips"] <= interior_initial["initial_reporting_lpips"]
        ),
        "step500_psnr_within_0_1db_of_strongest_control": (
            full_step500["psnr"] >= strongest500["psnr"] - 0.1
        ),
        "step500_ms_ssim_within_0_01_of_strongest_control": (
            full_step500["ms_ssim"] >= strongest500["ms_ssim"] - 0.01
        ),
        "step500_lpips_no_worse_than_strongest_control": (
            full_step500.get("lpips") is not None
            and strongest500.get("lpips") is not None
            and full_step500["lpips"] <= strongest500["lpips"]
        ),
        "step500_gradient_mae_no_worse_than_strongest_control": (
            full_step500["gradient_mae"] <= strongest500["gradient_mae"]
        ),
        "control_terminal_psnr_reached_no_later": (
            candidate_step is not None
            and control_step is not None
            and candidate_step <= control_step
            and candidate_seconds is not None
            and control_seconds is not None
            and candidate_seconds <= control_seconds
        ),
        "terminal_pareto_nondominated": not dominated,
        "full_beats_raw_geometry_tail": geometry_tail_better,
        "full_beats_raw_quality_or_convergence": quality_or_convergence_better,
        "complete_scene_representation_smaller_than_original_jpegs": (
            complete["original_over_packets_plus_model"] > 1.0
        ),
        "all_initial_counts_exact": all(
            row["initial_n_gaussians"] == INITIAL_GAUSSIANS for row in rows
        ),
        "all_final_counts_within_cap": all(
            row["final_n_gaussians"] <= MAX_GAUSSIANS for row in rows
        ),
        "all_arms_reuse_identical_packets": len(
            {tuple(record["input"]["packet_hashes"]) for record in records}
        )
        == 1,
    }
    return {
        "advance": False,
        "scalar_pass": all(gates.values()),
        "manual_visual_review_required": True,
        "artifact_gate": (
            "no smear, duplicate shell, trail, floater/sheet, grid imprint, boundary hole, "
            "or thin-feature deletion in any native reporting view"
        ),
        "strongest_terminal_control": strongest["arm"],
        "strongest_step500_control": strongest500_row["arm"],
        "gates": gates,
        "candidate_minus_terminal_control": {
            "reporting_psnr_db": complete["reporting_psnr"] - strongest["reporting_psnr"],
            "reporting_ms_ssim": complete["reporting_ms_ssim"] - strongest["reporting_ms_ssim"],
            "reporting_lpips": (
                None
                if complete["reporting_lpips"] is None or strongest["reporting_lpips"] is None
                else complete["reporting_lpips"] - strongest["reporting_lpips"]
            ),
            "reporting_gradient_mae": (
                complete["reporting_gradient_mae"] - strongest["reporting_gradient_mae"]
            ),
        },
        "full_vs_raw": {
            "spacing_p90_over_median": full_spacing_tail,
            "raw_spacing_p90_over_median": raw_spacing_tail,
            "reporting_psnr_db": complete["reporting_psnr"] - raw["reporting_psnr"],
        },
        "convergence_to_control_terminal": {
            "control_step": control_step,
            "control_seconds": control_seconds,
            "candidate_step": candidate_step,
            "candidate_seconds": candidate_seconds,
        },
    }


def _write_html(
    out: Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    decision: dict[str, Any],
    curves: Path,
    field_bundle: dict[str, Any],
) -> None:
    table = []
    for row in rows:
        if row["status"] != "ok":
            table.append(
                f"<tr><td>{escape(row['arm'])}</td><td colspan='10'>{escape(row['error'])}</td></tr>"
            )
            continue
        table.append(
            "<tr>"
            f"<td>{escape(row['arm'])}</td><td>{row['final_n_gaussians']:,}</td>"
            f"<td>{row['initial_reporting_psnr']:.3f}</td><td>{row['reporting_psnr']:.3f}</td>"
            f"<td>{row['reporting_ms_ssim']:.4f}</td><td>{row['reporting_lpips']:.4f}</td>"
            f"<td>{row['reporting_gradient_mae']:.4f}</td>"
            f"<td>{row['pretraining_seconds']:.1f}</td><td>{row['training_native_seconds']:.1f}</td>"
            f"<td>{row['original_over_packets_plus_model']:.2f}×</td></tr>"
        )
    cards = []
    for record in records:
        if record["status"] != "ok":
            continue
        sheet = record["visuals"]["contact_sheet"]["path"]
        cards.append(
            f"<section><h2>{escape(record['label'])}</h2>"
            f"<a href='{escape(sheet)}'><img src='{escape(sheet)}' loading='lazy'></a>"
            f"<p><a href='{escape(record['curves']['path'])}'>checkpoint metrics</a> · "
            f"<a href='{escape(record['models']['final_npz']['path'])}'>final NPZ</a></p></section>"
        )
    field_sheet = field_bundle["contact_sheet"]["path"]
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CORE-019 coherent-depth development diagnostic</title>
<style>body{{font-family:sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.4rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;height:auto}}pre{{white-space:pre-wrap;background:#f4f4f4;padding:1rem}}</style></head>
<body><h1>CORE-019 calibrated coherent-depth development diagnostic</h1>
<p><strong>Scope:</strong> one exposed development scene and seed. Scalar success cannot override
native visual failure; reporting cameras were excluded from construction.</p>
<p><a href="manifest.json">manifest</a> · <a href="plan.json">plan</a> ·
<a href="metrics.json">metrics JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="decision.json">decision</a></p>
<table><thead><tr><th>Arm</th><th>Final N</th><th>Step-0 PSNR</th><th>Final PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>Grad MAE</th><th>Pretrain s</th><th>Train s</th>
<th>Original/(packet+model)</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<h2>Construction-only coherent field</h2><a href="{escape(field_sheet)}"><img src="{escape(field_sheet)}"></a>
<h2>All reporting curves</h2><a href="{escape(curves.name)}"><img src="{escape(curves.name)}"></a>
<h2>Fail-closed scalar decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
{''.join(cards)}</body></html>"""
    (out / "index.html").write_text(html)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--realtime-gs-root", type=Path, default=DEFAULT_RTGS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = args.out.expanduser().resolve()
    frame = args.frame.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    rtgs_root = args.realtime_gs_root.expanduser().resolve()
    calibration = frame.parent / "calibration_dome.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    if not frame.is_dir() or not calibration.is_file():
        raise FileNotFoundError("karate frame or calibration is missing")
    if not weights.is_file():
        raise FileNotFoundError("pinned VGGT weights are missing")
    if not torch.cuda.is_available():
        raise RuntimeError("CORE-019 diagnostic requires CUDA")
    command = "PYTHONPATH=/home/alex/Documents/vggt:/home/alex/Documents/realtime-gs/src:src "
    command += shlex.join([sys.executable, *sys.argv])

    base.ARM_LABELS = ARM_LABELS
    base.SOURCE_SNAPSHOTS = (
        "src/structsplat/codec_native_field.py",
        "src/structsplat/realtime_gs_adapter.py",
        "src/structsplat/realtime_gs_ray_posterior.py",
        "src/structsplat/realtime_gs_coherent_depth.py",
        "scripts/experiments/core018_ray_posterior_downstream.py",
        "scripts/experiments/core019_coherent_depth_downstream.py",
    )
    out.mkdir(parents=True)
    started = time.perf_counter()
    repositories = {
        "structsplat": base._repository_record(STRUCTSPLAT_ROOT),
        "realtime_gs": base._repository_record(rtgs_root),
    }
    snapshots = base._snapshot_sources(out, rtgs_root)
    packet_scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=PACKET_DOWNSCALE,
        view_ids=TRAIN_CAMERA_IDS,
        test_every=0,
        load_masks=False,
        undistort=True,
    )
    scene = load_calibrated_scene(
        frame,
        calibration_path=calibration,
        downscale=EVALUATION_DOWNSCALE,
        view_ids=TRAIN_CAMERA_IDS + REPORT_CAMERA_IDS,
        test_every=0,
        load_masks=False,
        undistort=True,
    )
    if packet_scene.bounds_hint is None or scene.bounds_hint is None:
        raise RuntimeError("the calibrated loader must provide explicit shared bounds")
    scene.train_indices = list(range(len(TRAIN_CAMERA_IDS)))
    scene.test_indices = list(range(len(TRAIN_CAMERA_IDS), len(ALL_CAMERA_IDS)))
    scene.name = "core019-karate-frame00005"
    scene.validate()
    plan = {
        "schema": "core019.coherent_depth.plan.v1",
        "scope": "single_exposed_development_scene_single_seed_reduced_resolution_diagnostic",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "command": command,
        "repositories": repositories,
        "environment": base._environment(),
        "source_snapshots": snapshots,
        "frame": str(frame),
        "calibration": {
            "path": str(calibration),
            "bytes": calibration.stat().st_size,
            "sha256": base._sha256(calibration),
        },
        "train_camera_ids": list(TRAIN_CAMERA_IDS),
        "report_camera_ids": list(REPORT_CAMERA_IDS),
        "report_camera_selection": "calibration_only_before_frame_open",
        "train_view_names": list(packet_scene.view_names),
        "report_view_names": list(scene.view_names[len(TRAIN_CAMERA_IDS) :]),
        "packet_downscale": PACKET_DOWNSCALE,
        "evaluation_downscale": EVALUATION_DOWNSCALE,
        "bounds_hint": {
            "center": packet_scene.bounds_hint[0].tolist(),
            "extent": packet_scene.bounds_hint[1],
        },
        "arms": list(ARMS),
        "interior_carve_config": dataclasses.asdict(_interior_carve_config()),
        "posterior_carve_config": dataclasses.asdict(BASE_CARVE_CONFIG()),
        "posterior_no_reciprocal": dataclasses.asdict(base._posterior_config(False)),
        "coherent_config": dataclasses.asdict(_coherent_config()),
        "train_config": dataclasses.asdict(base._train_config()),
        "fixed_topology_prefix_steps": FIXED_PREFIX_STEPS,
        "vggt": {
            "weights_path": str(weights),
            "source_revision": VGGT_SOURCE_REVISION,
            "model_revision": VGGT_MODEL_REVISION,
            "checkpoint_bytes": VGGT_MODEL_BYTES,
            "checkpoint_sha256": VGGT_MODEL_SHA256,
            "license": "CC-BY-NC-4.0",
            "scene_payload_includes_encoder": False,
        },
        "decision_gate": {
            "manual_native_visual_review_required": True,
            "step0_psnr_over_interior_db": 2.0,
            "step500_psnr_control_margin_db": -0.1,
            "step500_ms_ssim_control_margin": -0.01,
            "terminal_pareto_nondominated": True,
            "original_over_packets_plus_model_min": 1.0,
        },
    }
    base._write_json(out / "plan.json", plan)

    print("[packets] build shared construction packets", flush=True)
    packet_record = base._build_shared_packets(packet_scene, frame, out)
    packet_record["schema"] = "core019.shared_packets.v1"
    base._write_json(out / "shared_packets.json", packet_record)

    print("[features] build shared CORE-018 control features", flush=True)
    feature_inputs, feature_views, _ = base._reload_packets(packet_record, packet_scene, out)
    features = base.build_packet_feature_pyramids(
        feature_views,
        base._posterior_config(False),
        progress_callback=lambda completed, total: print(
            f"  [features] {completed}/{total}", flush=True
        ),
    )
    base._write_json(out / "feature_receipt.json", features.diagnostics)
    del feature_inputs, feature_views
    gc.collect()
    torch.cuda.empty_cache()

    print("[coherent-depth] infer shared packet-only field", flush=True)
    field_inputs, field_views, _ = base._reload_packets(packet_record, packet_scene, out)
    field = infer_coherent_depth_field(
        field_inputs,
        field_views,
        weights,
        _coherent_config(),
    )
    field_bundle = _save_field_bundle(field, out)
    del field_inputs, field_views
    gc.collect()
    torch.cuda.empty_cache()

    renderer = get_rasterizer(
        "gsplat", device=torch.device("cuda"), packed=False, antialiased=True
    )
    records = []
    for arm in ARMS:
        inputs = views = None
        try:
            print(f"[arm] {arm}: cold reload", flush=True)
            inputs, views, input_record = base._reload_packets(packet_record, packet_scene, out)
            if arm in {"interior", "posterior_no_reciprocal"}:
                base._carve_config = (
                    _interior_carve_config if arm == "interior" else BASE_CARVE_CONFIG
                )
                record = base._run_arm(
                    arm,
                    inputs,
                    views,
                    input_record,
                    features,
                    scene,
                    renderer,
                    out,
                )
            else:
                record = _run_coherent_arm(
                    arm,
                    inputs,
                    views,
                    input_record,
                    field,
                    scene,
                    renderer,
                    out,
                )
        except Exception as error:
            record = {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(record["traceback"], file=sys.stderr, flush=True)
        records.append(record)
        base._write_json(out / "partial_records.json", records)
        del inputs, views
        gc.collect()
        torch.cuda.empty_cache()

    rows = [_terminal_row(record) for record in records]
    base._write_metrics(out, rows)
    decision = _decision(records, rows)
    base._write_json(out / "decision.json", decision)
    curves = base._plot_curves(records, out)
    _write_html(out, rows, records, decision, curves, field_bundle)
    manifest_records = []
    for record in records:
        retained = dict(record)
        retained.pop("curve_rows", None)
        manifest_records.append(retained)
    manifest = {
        "schema": "core019.coherent_depth.manifest.v1",
        "status": "ok" if all(record["status"] == "ok" for record in records) else "partial",
        "claim_ready": False,
        "command": command,
        "scope": plan["scope"],
        "plan": base._artifact(out / "plan.json", out),
        "shared_packets": base._artifact(out / "shared_packets.json", out),
        "feature_receipt": base._artifact(out / "feature_receipt.json", out),
        "coherent_depth_field": field_bundle,
        "records": manifest_records,
        "decision": decision,
        "metrics": {
            name: base._artifact(out / name, out)
            for name in ("metrics.json", "metrics.jsonl", "metrics.csv")
        },
        "plots": {"all_metric_curves": base._artifact(curves, out)},
        "report": base._artifact(out / "index.html", out),
        "total_wall_seconds": time.perf_counter() - started,
    }
    base._write_json(out / "manifest.json", manifest)
    print(json.dumps({"decision": decision, "rows": rows}, indent=2), flush=True)
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
