#!/usr/bin/env python3
"""FIT-050 exposed-image color-ray screen, with source-bound parent artifacts.

Formal: python scripts/experiments/fit050_color_ray.py --out OUT --approved-protocol-digest DIGEST
Protocol only: same driver with --protocol-only (prints identity; writes nothing).
Wiring only: --out NEW_OUT --smoke (one procedural CPU fixture, never COCO).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.fit050_controls import (  # noqa: E402
    ARMS, IMAGE_IDS, SEEDS, fit_parent, full_frame_context, parent_configs, run_arm,
)
from benchmarks.hier_research_report import (  # noqa: E402
    ResearchBundle, protocol_digest, save_rgb, sha256, write_json,
)

_INIT, _FIT = parent_configs(0)
PROTOCOL = {
    "task": "FIT-050", "version": 1, "device": "cuda", "gpu": "NVIDIA GeForce RTX 3050",
    "dtype": "float32", "threads": 1, "torch": "2.9.0+cu128",
    "image_ids": IMAGE_IDS, "seeds": SEEDS, "arms": ARMS,
    "image_paths": {str(i): f"tests/test_images/COCO_train2014_{i:012d}.jpg" for i in IMAGE_IDS},
    "max_side": 512, "resize": "Pillow RGB LANCZOS; Python round(scale*dimension), max(1,...)",
    "parent_init": asdict(_INIT), "parent_fit": asdict(_FIT), "n_gaussians": 2000,
    "parent_steps": 750, "adam_steps": 32, "ray_max_trials": 6, "cg_maxiter": 32,
    "ridge": 1e-4, "coverage_tau": 0.05, "quality_backends": "reference coverage and tail",
    "data_role": "four previously exposed COCO development images; seeds clustered within image",
    "order": "one new worker per image/seed; shared exact parent; rotate six arms by workload index",
    "warmup": "procedural64 seed77, 3 parent steps then every arm (Adam2), LPIPS only for formal",
    "parent_selection": "terminal750, no topology edits or earlier-checkpoint selection",
    "adam_control": "32 full-parameter updates, inherited moments, then same transaction gate; unequal work",
    "ray_selection": "first safe nonzero coefficient change among alpha*2^-k, k=0..5; replay failure rolls back without retry",
    "cg_control": "independent legacyCG32 endpoint per logical arm; no shared proposal work",
    "objective": "whole-image raw RGB L2; ray/CG ridge pulls toward parent; signed unclamped colors",
    "metrics": "float64 raw PSNR/MSE; display-clamped builtin MS-SSIM and LPIPS; complete protected vector",
    "curves": "unaltered native parent/Adam histories and every ray trial protected vector; perceptual endpoints",
    "timing": "whole transaction including all construction, checks, failed trials and selected replay; scoring/serialization separate",
    "occupancy": "parent process samples nvidia-smi about once per second; foreign/query-failure makes speed evidence ineligible",
    "minimum_median_image_gain_db": 0.1, "maximum_cell_loss_db": 0.01,
    "maximum_ms_ssim_loss": 0.001, "maximum_lpips_increase": 0.002,
    "cold_parity_max_abs": 2e-5, "worker_timeout_seconds": 3600,
    "missing_policy": "all errors retained; any missing/error cell prohibits positive utility verdict; no selective rerun",
    "forbidden": ["default change", "threshold rescue", "repeated ray rescue", "sealed data", "in-place repair"],
}
SOURCES = [
    "scripts/experiments/fit050_color_ray.py", "benchmarks/fit050_controls.py",
    "benchmarks/hier_research_report.py", "scripts/check_report_bundle.py",
    *sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "src/structsplat").rglob("*.py")),
    "src/structsplat/cuda/render_ext.cpp", "src/structsplat/cuda/render_ext.cu",
    *PROTOCOL["image_paths"].values(),
]


def workloads(diagnostic=False, protocol=PROTOCOL):
    return [(-1, 77)] if diagnostic else [(i, s) for i in protocol["image_ids"] for s in protocol["seeds"]]


def parent_id(image_id, seed):
    return f"procedural_s{seed}" if image_id < 0 else f"coco{image_id:012d}_s{seed}"


def expected_cells(protocol=PROTOCOL, diagnostic=False):
    return [f"{parent_id(i, s)}_{arm}" for i, s in workloads(diagnostic, protocol) for arm in protocol["arms"]]


def procedural_target():
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    return np.stack([0.4 + 0.2 * np.sin(xx / 9), 0.5 + 0.2 * np.cos(yy / 11),
                     0.4 + 0.15 * np.sin((xx + yy) / 7)], -1).astype(np.float32)


def load_target(image_id, *, smoke=False):
    if smoke:
        return procedural_target(), {"role": "procedural wiring only", "source_path": None}
    source = ROOT / PROTOCOL["image_paths"][str(image_id)]
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        original_shape = [image.height, image.width]
        scale = min(1.0, PROTOCOL["max_side"] / max(image.size))
        if scale < 1:
            image = image.resize(tuple(max(1, round(value * scale)) for value in image.size), Image.Resampling.LANCZOS)
        target = np.array(image, dtype=np.float32) / 255
    return target, {"role": PROTOCOL["data_role"], "source_path": source.relative_to(ROOT).as_posix(),
                    "source_sha256": sha256(source), "original_shape": original_shape,
                    "shape": list(target.shape)}


def score(raw, target, *, smoke=False):
    from structsplat.metrics import LPIPS, ms_ssim
    mse = float((raw.double() - target.double()).square().mean())
    display = raw.clamp(0, 1)
    lpips = None if smoke else LPIPS.distance(display, target)
    if lpips is None and not smoke:
        raise RuntimeError("required LPIPS unavailable")
    return {"raw_mse": mse, "psnr": -10 * math.log10(max(mse, 1e-12)),
            "ms_ssim": float(ms_ssim(display, target)), "lpips": lpips}


def _curves(path, parent_history, metadata, adam_history, parent_metrics, selected_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, grid = plt.subplots(2, 2, figsize=(10, 6))
    axes = grid.reshape(-1)
    axes[0].plot(parent_history["iter"], parent_history["psnr"])
    axes[0].set(xlabel="Parent native pre-step iteration", ylabel="Raw PSNR (dB)")
    if adam_history:
        axes[1].plot(adam_history["iter"], adam_history["psnr"], marker=".")
        axes[1].set(xlabel="Adam continuation native pre-step iteration", ylabel="Raw PSNR (dB)")
    else:
        values = [metadata["parent_metrics"]["foreground_mse"]]
        values += [r["surrogate_metrics"]["foreground_mse"] for r in metadata.get("trials", [])]
        if len(values) == 1:
            values += [metadata["selected_metrics"]["foreground_mse"]]
        axes[1].plot(range(len(values)), [-10 * math.log10(max(v, 1e-12)) for v in values], marker=".")
        axes[1].set(xlabel="Parent then proposal/trial order (not optimizer steps)", ylabel="Raw PSNR (dB)")
    for axis, key in zip(axes[2:], ("ms_ssim", "lpips")):
        values = [parent_metrics[key], selected_metrics[key]]
        if all(value is not None for value in values):
            axis.plot([0, 1], values, marker="o")
        else:
            axis.text(0.5, 0.5, "Not computed in procedural smoke", ha="center", va="center")
        axis.set(xticks=[0, 1], xticklabels=["Parent", "Selected"], ylabel=key,
                 xlabel="Measured endpoints only")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _cpu_state(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    return value


def worker(request_path):
    import torch
    from structsplat.config import FitConfig
    from structsplat.gaussians import GaussianField
    from structsplat.safe_schedule import evaluate_quality
    request = json.loads(Path(request_path).read_text())
    root = Path(request["root"])
    parent_dir = Path(request_path).parent
    smoke = request["smoke"]
    device = "cpu" if smoke else "cuda"
    torch.set_num_threads(PROTOCOL["threads"])
    if not smoke and (torch.__version__ != PROTOCOL["torch"] or not torch.cuda.is_available()
                      or torch.cuda.get_device_name() != PROTOCOL["gpu"]):
        raise RuntimeError("frozen GPU unavailable")
    # Warm only a disposable procedural fixture, never an additional natural-image outcome.
    warm = fit_parent(procedural_target(), 77, smoke=True, device=device)
    warm_target = torch.as_tensor(procedural_target(), device=device)
    warm_cfg = FitConfig(**warm["fit_config"])
    warm_context = full_frame_context(warm_target, warm_cfg, warm["field"].n)
    for arm in ARMS:
        run_arm(warm["field"], warm_target, warm_cfg, *warm_context, arm, warm["optimizer_state"], smoke=True)
    score(warm["render"], warm_target, smoke=smoke)
    del warm, warm_target, warm_context
    target_np, source = load_target(request["image_id"], smoke=smoke)
    parent = fit_parent(target_np, request["seed"], smoke=smoke, device=device)
    parent["initial_field"].save(parent_dir / "initial_field.npz")
    parent["field"].save(parent_dir / "field.npz")
    np.save(parent_dir / "target.npy", target_np)
    np.save(parent_dir / "reconstruction.npy", parent["render"].detach().cpu().numpy())
    save_rgb(parent_dir / "target.png", target_np)
    save_rgb(parent_dir / "reconstruction.png", parent["render"].detach().cpu().numpy())
    torch.save(_cpu_state(parent["optimizer_state"]), parent_dir / "optimizer_state.pt")
    write_json(parent_dir / "history.json", parent["history"])
    config = {"init": parent["init_config"], "fit": parent["fit_config"], "tensor": parent["tensor_config"],
              "image_id": request["image_id"], "seed": request["seed"], "source": source,
              "initial_field_sha256": sha256(parent_dir / "initial_field.npz"),
              "field_sha256": sha256(parent_dir / "field.npz"),
              "optimizer_state_sha256": sha256(parent_dir / "optimizer_state.pt"),
              "target_sha256": sha256(parent_dir / "target.npy"),
              "init_seconds": parent["init_seconds"], "fit_seconds": parent["fit_seconds"],
              "parent_total_seconds": parent["parent_total_seconds"],
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "numpy": np.__version__, "pillow": pillow_version,
                              "python": sys.version, "worker_pid": os.getpid(),
                              "gpu": None if smoke else torch.cuda.get_device_name()}}
    write_json(parent_dir / "config.json", config)
    cfg = FitConfig(**parent["fit_config"])
    target = torch.as_tensor(target_np, device=device)
    parent_metrics = score(parent["render"], target, smoke=smoke)
    write_json(parent_dir / "metrics.json", parent_metrics)
    context = full_frame_context(target, cfg, parent["field"].n)
    rows = []
    for arm in request["arm_order"]:
        cell_id = f'{request["parent_id"]}_{arm}'
        directory = root / "cells" / cell_id
        directory.mkdir(parents=True, exist_ok=False)
        common = {"cell_id": cell_id, "parent_id": request["parent_id"], "image_id": request["image_id"],
                  "seed": request["seed"], "method": arm, "smoke": smoke,
                  "parent_field_sha256": config["field_sha256"],
                  "parent_optimizer_sha256": config["optimizer_state_sha256"],
                  "target_sha256": config["target_sha256"]}
        write_json(directory / "request.json", common)
        try:
            if not smoke:
                torch.cuda.reset_peak_memory_stats()
            cell_started = time.perf_counter()
            selected, quality, metadata, adam_history = run_arm(
                parent["field"], target, cfg, *context, arm, parent["optimizer_state"], smoke=smoke)
            peak = 0 if smoke else torch.cuda.max_memory_allocated()
            reserved = 0 if smoke else torch.cuda.max_memory_reserved()
            selected.save(directory / "field.npz")
            shutil.copy2(parent_dir / "field.npz", directory / "input_field.npz")
            # Actual-reader artifacts and cold replay are separately charged reporting work.
            scored_quality, raw = evaluate_quality(selected, target, context[0], cfg,
                                                   context[1], context[2].coverage_tau)
            cold = GaussianField.load(directory / "field.npz", device=device)
            for name in ("means", "log_scales", "rotations", "colors", "opacities", "scale_max",
                         "color_grads", "background_mask", "filter_variance"):
                expected, decoded = getattr(selected, name), getattr(cold, name)
                if (expected is None) != (decoded is None) or (
                        expected is not None and not torch.equal(expected, decoded)):
                    raise RuntimeError(f"cold field parameter mismatch: {name}")
            cold_quality, cold_raw = evaluate_quality(cold, target, context[0], cfg,
                                                      context[1], context[2].coverage_tau)
            parity = float((cold_raw - raw).abs().max())
            if parity > PROTOCOL["cold_parity_max_abs"] or selected.n != parent["field"].n:
                raise RuntimeError("cold replay or exact count failed")
            metrics = score(raw, target, smoke=smoke)
            np.save(directory / "reconstruction.npy", raw.detach().cpu().numpy())
            save_rgb(directory / "target.png", target_np)
            save_rgb(directory / "reconstruction.png", raw.detach().cpu().numpy())
            save_rgb(directory / "error.png", (raw - target).abs().detach().cpu().numpy() * 4)
            write_json(directory / "history.json", {"transaction": metadata, "adam_fit": adam_history,
                "parent_history": f'../../parents/{request["parent_id"]}/history.json',
                "reporting_quality": scored_quality.to_dict(), "cold_quality": cold_quality.to_dict(),
                "reporting_work": {"quality_evaluations": 2, "gaussian_renders": 2, "coverage_passes": 2}})
            write_json(directory / "config.json", {"request": common, "fit": asdict(cfg),
                "schedule": asdict(context[2]), "parent_config": f'../../parents/{request["parent_id"]}/config.json',
                "parent_field_sha256": config["field_sha256"], "environment": config["environment"]})
            _curves(directory / "curves.png", parent["history"], metadata, adam_history, parent_metrics, metrics)
            filenames = ("request.json", "field.npz", "input_field.npz", "history.json", "config.json",
                         "reconstruction.npy", "target.png", "reconstruction.png", "error.png", "curves.png")
            row = {**common, **metrics, "status": "ok", "n_gaussians": selected.n,
                "iterations_run": (2 if smoke else 32) if arm == "adam32" else 0,
                "selected_iteration": (2 if smoke else 32) if arm == "adam32" and metadata["accepted"] else 0,
                "total_seconds": metadata["transaction_seconds"],
                "transaction_seconds": metadata["transaction_seconds"],
                "cell_total_seconds": time.perf_counter() - cell_started,
                "accepted": metadata["accepted"], "coefficients_changed": metadata["coefficients_changed"],
                "selected_fraction": metadata["selected_fraction"], "rollback_reason": metadata["rollback_reason"],
                "counts": metadata["counts"], "protected_metrics": quality.to_dict(),
                "reporting_metrics": scored_quality.to_dict(),
                "noncolor_changed_fields": metadata["noncolor_changed_fields"],
                "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
                "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cold_render_max_abs": parity,
                "cold_parameters_exact": True,
                "artifacts": {name: f"cells/{cell_id}/{name}" for name in filenames}}
            for filename in ("target.npy", "field.npz", "history.json", "config.json", "optimizer_state.pt", "metrics.json"):
                row["artifacts"]["parent_" + filename] = f'parents/{request["parent_id"]}/{filename}'
        except Exception as exc:
            row = {**common, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "artifacts": {"request": f"cells/{cell_id}/request.json"}}
        write_json(directory / "row.json", row)
        rows.append(row)
        print(json.dumps({"cell_id": cell_id, "status": row["status"]}), flush=True)
    write_json(parent_dir / "rows.json", rows)


def validate_rows(rows, protocol, problems, *, diagnostic=False):
    expected = expected_cells(protocol, diagnostic)
    identifiers = [row.get("cell_id") for row in rows]
    if any(not isinstance(identifier, str) for identifier in identifiers) or sorted(identifiers) != sorted(expected):
        problems.append("FIT-050 matrix has missing, duplicate or unexpected cells")
    for row in rows:
        pair = (row.get("image_id"), row.get("seed"))
        if pair not in workloads(diagnostic, protocol) or row.get("method") not in protocol["arms"]:
            problems.append(f'{row.get("cell_id")}: row workload/method identity differs')
        elif row.get("parent_id") != parent_id(*pair) or row.get("cell_id") != f'{parent_id(*pair)}_{row["method"]}':
            problems.append(f'{row.get("cell_id")}: cell and parent identity differ')
        if row.get("status") != "ok":
            continue
        count = 16 if diagnostic else protocol["n_gaussians"]
        if row.get("n_gaussians") != count:
            problems.append(f'{row.get("cell_id")}: exact count differs')
        steps = (2 if diagnostic else protocol["adam_steps"]) if row.get("method") == "adam32" else 0
        selected_steps = steps if row.get("method") == "adam32" and row.get("accepted") else 0
        if row.get("iterations_run") != steps or row.get("selected_iteration") != selected_steps:
            problems.append(f'{row.get("cell_id")}: attempted/selected horizon differs')
        if row.get("cold_parameters_exact") is not True:
            problems.append(f'{row.get("cell_id")}: cold parameter equality is not established')
        quality = row.get("protected_metrics", {})
        try:
            _quality_record(quality, count)
            _quality_record(row.get("reporting_metrics", {}), count)
        except (ValueError, TypeError, KeyError) as exc:
            problems.append(f'{row.get("cell_id")}: {exc}')
        counts = row.get("counts", {})
        required_counts = {"quality_evaluations", "gaussian_renders", "raw_coverage_passes",
                           "basis_denominator_passes", "basis_apply_calls", "basis_transpose_calls"}
        if not isinstance(counts, dict) or not required_counts <= counts.keys() or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts.values()):
            problems.append(f'{row.get("cell_id")}: missing/invalid exact work counts')
        elif row.get("method") in ("noop", "legacy_cg32", "adam32"):
            method = row["method"]
            quality_calls = 1 if method == "noop" else 2
            expected_work = {"quality_evaluations": quality_calls,
                "gaussian_renders": quality_calls + (steps + 1 if method == "adam32" else 0),
                "raw_coverage_passes": quality_calls, "basis_denominator_passes": int(method == "legacy_cg32"),
                "basis_apply_calls": counts["basis_apply_calls"] if method == "legacy_cg32" else 0,
                "basis_transpose_calls": counts["basis_apply_calls"] + 1 if method == "legacy_cg32" else 0,
                "gradient_evaluations": steps if method == "adam32" else 0}
            if counts != expected_work or (method == "legacy_cg32" and not 1 <= counts["basis_apply_calls"] <= 33):
                problems.append(f'{row.get("cell_id")}: endpoint work differs from exact control')
        elif counts.get("quality_evaluations") not in (1, 2) or counts["gaussian_renders"] != counts["quality_evaluations"]:
            problems.append(f'{row.get("cell_id")}: ray render/evaluation counts differ')
        fraction = row.get("selected_fraction")
        if not row.get("accepted") and fraction != 0:
            problems.append(f'{row.get("cell_id")}: rollback retained a selected fraction')
        if row.get("accepted") and fraction not in ([2.0**(-k) for k in range(6)] if row.get("method", "").endswith("_ray") else [1.0]):
            problems.append(f'{row.get("cell_id")}: selected fraction differs from frozen ladder')
        for key in ("parent_field_sha256", "parent_optimizer_sha256", "target_sha256"):
            if not isinstance(row.get(key), str) or not row[key]:
                problems.append(f'{row.get("cell_id")}: missing paired input binding {key}')
        for key in ("psnr", "ms_ssim", "raw_mse", "transaction_seconds", "cold_render_max_abs"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                problems.append(f'{row.get("cell_id")}: invalid {key}')
        if not diagnostic and (not isinstance(row.get("lpips"), (int, float)) or not math.isfinite(row["lpips"])):
            problems.append(f'{row.get("cell_id")}: required LPIPS missing or invalid')
        if row.get("transaction_seconds", 0) <= 0:
            problems.append(f'{row.get("cell_id")}: nonpositive transaction time')
        if not row.get("accepted") and row.get("coefficients_changed"):
            problems.append(f'{row.get("cell_id")}: rejected transaction changed coefficients')
        if row.get("cold_render_max_abs", math.inf) > protocol["cold_parity_max_abs"]:
            problems.append(f'{row.get("cell_id")}: cold replay failed')
        if row["method"] != "adam32" and row.get("noncolor_changed_fields"):
            problems.append(f'{row.get("cell_id")}: RGB-only arm changed geometry')
    for image_id, seed in workloads(diagnostic, protocol):
        group = [r for r in rows if r.get("image_id") == image_id and r.get("seed") == seed]
        for key in ("parent_field_sha256", "parent_optimizer_sha256", "target_sha256"):
            values = {r.get(key) for r in group if r.get("status") == "ok"}
            if len(values) > 1:
                problems.append(f"FIT-050 paired inputs differ for image{image_id}/seed{seed}: {key}")


def summarize(rows, protocol=PROTOCOL, *, diagnostic=False):
    problems = []
    validate_rows(rows, protocol, problems, diagnostic=diagnostic)
    complete = not problems and len(rows) == len(expected_cells(protocol, diagnostic)) and all(r["status"] == "ok" for r in rows)
    lookup = {(r["image_id"], r["seed"], r["method"]): r for r in rows if r["status"] == "ok"}
    records = []
    for method in protocol["arms"]:
        if method == "noop":
            continue
        pairs = []
        for image_id, seed in workloads(diagnostic, protocol):
            candidate, baseline = lookup.get((image_id, seed, method)), lookup.get((image_id, seed, "noop"))
            if candidate is None or baseline is None:
                continue
            pairs.append({"image_id": image_id, "seed": seed, "gain_db": candidate["psnr"] - baseline["psnr"],
                "ms_ssim_difference": candidate["ms_ssim"] - baseline["ms_ssim"],
                "lpips_difference": None if candidate["lpips"] is None or baseline["lpips"] is None
                    else candidate["lpips"] - baseline["lpips"],
                "accepted": candidate["accepted"], "coefficients_changed": candidate["coefficients_changed"],
                "selected_fraction": candidate["selected_fraction"], "transaction_seconds": candidate["transaction_seconds"],
                "comparators": {control: {"gain_db": candidate["psnr"] - lookup[(image_id, seed, control)]["psnr"],
                    "transaction_seconds_ratio": candidate["transaction_seconds"] / max(lookup[(image_id, seed, control)]["transaction_seconds"], 1e-12)}
                    for control in ("legacy_cg32", "cg_ray", "adam32") if (image_id, seed, control) in lookup}})
        image_gains = [statistics.mean(p["gain_db"] for p in pairs if p["image_id"] == image_id)
                       for image_id in sorted({p["image_id"] for p in pairs})]
        median = statistics.median(image_gains) if image_gains else None
        useful = complete and not diagnostic and median >= protocol["minimum_median_image_gain_db"] and all(
            p["gain_db"] >= -protocol["maximum_cell_loss_db"]
            and p["ms_ssim_difference"] >= -protocol["maximum_ms_ssim_loss"]
            and p["lpips_difference"] is not None and p["lpips_difference"] <= protocol["maximum_lpips_increase"] for p in pairs)
        useful = useful and any(p["accepted"] and p["coefficients_changed"] for p in pairs)
        records.append({"method": method, "pairs": pairs, "median_image_averaged_gain_db": median,
                        "passes_utility_gate": bool(useful), "complete_matrix": complete})
    return {"records": records, "problems": problems, "data_role": protocol["data_role"],
            "speed_claim": False, "default_promotion": False, "diagnostic": diagnostic}


def _quality_record(value, count):
    from dataclasses import fields
    from structsplat.safe_schedule import QualityMetrics
    names = {item.name for item in fields(QualityMetrics)}
    if not isinstance(value, dict) or not names <= value.keys():
        raise ValueError("incomplete protected quality vector")
    if value["finite"] is not True or value["n_gaussians"] != count:
        raise ValueError("nonfinite or wrong-count protected quality vector")
    for name in names - {"finite", "n_gaussians"}:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError(f"invalid protected metric {name}")
    result = QualityMetrics(**{name: value[name] for name in names})
    if result.to_dict() != value:
        raise ValueError("derived protected PSNR fields disagree")
    return result


def _json_form(value):
    return json.loads(json.dumps(value))


def _field_equal(first, second, *, colors=True):
    import torch
    names = ["means", "log_scales", "rotations", "opacities", "scale_max", "color_grads",
             "background_mask", "filter_variance"] + (["colors"] if colors else [])
    for name in names:
        left, right = getattr(first, name), getattr(second, name)
        if (left is None) != (right is None) or (left is not None and not torch.equal(left, right)):
            return False
    return True


def _validate_transaction(metadata, row, schedule, count, steps):
    from structsplat.safe_schedule import safe_commit_decision
    before = _quality_record(metadata["parent_metrics"], count)
    selected = _quality_record(metadata["selected_metrics"], count)
    if metadata["selected_metrics"] != row["protected_metrics"]:
        raise ValueError("row protected quality differs from transaction selection")
    for key in ("accepted", "coefficients_changed", "selected_fraction", "rollback_reason", "counts",
                "noncolor_changed_fields"):
        if metadata[key] != row[key]:
            raise ValueError(f"row differs from transaction {key}")
    if metadata["transaction_seconds"] != row["transaction_seconds"] or row["total_seconds"] != row["transaction_seconds"]:
        raise ValueError("transaction timing fields disagree")
    if metadata["quality_coverage_backend"] != "reference" or metadata["quality_tail_backend"] != "reference":
        raise ValueError("transaction quality backend differs from reference")
    counts = metadata["counts"]
    method = row["method"]
    def gate(value):
        return safe_commit_decision(before, _quality_record(value, count), schedule.tolerances,
                                    schedule.hole_regression_budget)
    if method.endswith("_ray"):
        if metadata["direction"] != method.removesuffix("_ray") or metadata["max_trials"] != 6 or metadata["ridge"] != 1e-4:
            raise ValueError("ray direction/trial/ridge config differs")
        trials = metadata["trials"]
        if len(trials) > 6:
            raise ValueError("ray exceeded six trials")
        for index, trial in enumerate(trials):
            if trial["index"] != index or trial["fraction"] != 2.0 ** (-index):
                raise ValueError("ray trial order/fraction differs")
            if trial["alpha"] != metadata["alpha_star"] * trial["fraction"]:
                raise ValueError("ray trial alpha differs")
            accepted, reasons = gate(trial["surrogate_metrics"])
            if not trial["finite"]:
                accepted, reasons = False, reasons + ["nonfinite_trial"]
            if not trial["coefficients_changed"]:
                accepted, reasons = False, reasons + ["unchanged_coefficients"]
            if trial["surrogate_accepted"] != accepted or trial["reasons"] != reasons:
                raise ValueError("surrogate safe gate was misreported")
            if accepted and index != len(trials) - 1:
                raise ValueError("ray continued after its first safe trial")
        replay = metadata["replay_metrics"]
        if replay is not None:
            accepted, reasons = gate(replay)
            if not isinstance(metadata["replay_max_abs_error"], (float, int)) or metadata["replay_max_abs_error"] > 2e-5:
                accepted = False
                reasons.append("replay_parity_failed")
            if metadata["replay_reasons"] != reasons or metadata["accepted"] != accepted:
                raise ValueError("actual replay safe gate was misreported")
            if not trials or not trials[-1]["surrogate_accepted"]:
                raise ValueError("replay lacks a selected safe trial")
            if trials[-1].get("replay_metrics") != replay or trials[-1].get("replay_accepted") != accepted:
                raise ValueError("selected trial replay record differs")
        elif metadata["accepted"]:
            raise ValueError("accepted ray lacks actual replay")
        if metadata["accepted"]:
            if metadata["selected_fraction"] != trials[-1]["fraction"] or metadata["selected_trial_index"] != trials[-1]["index"]:
                raise ValueError("selected ray fraction/index differs")
        elif metadata["selected_fraction"] != 0 or metadata["selected_alpha"] != 0:
            raise ValueError("rolled-back ray retained a selected fraction")
        quality_calls = 1 + int(replay is not None)
        basis_ok = metadata["basis_parent_max_abs_error"] is not None and metadata["basis_parent_max_abs_error"] <= 2e-5
        expected = {"quality_evaluations": quality_calls, "gaussian_renders": quality_calls,
            "raw_coverage_passes": quality_calls + int(basis_ok), "basis_denominator_passes": 1,
            "basis_apply_calls": 1 + int("direction_render" in metadata["phase_seconds"]),
            "basis_transpose_calls": int(basis_ok and method != "cg_ray"),
            "basis_diagonal_passes": int(basis_ok and method == "jacobi_ray"),
            "interpolated_quality_evaluations": len(trials), "legacy_cg_iterations": 0}
    else:
        expected = {"quality_evaluations": 1 if method == "noop" else 2,
            "gaussian_renders": 1 if method == "noop" else 2 + (steps + 1 if method == "adam32" else 0),
            "raw_coverage_passes": 1 if method == "noop" else 2,
            "basis_denominator_passes": 0, "basis_apply_calls": 0, "basis_transpose_calls": 0,
            "gradient_evaluations": steps if method == "adam32" else 0}
        if method != "noop":
            accepted, reasons = gate(metadata["candidate_metrics"])
            if metadata["accepted"] != accepted or metadata["candidate_reasons"] != reasons:
                raise ValueError("endpoint safe gate was misreported")
        elif metadata["accepted"] or metadata["selected_metrics"] != metadata["parent_metrics"]:
            raise ValueError("no-op is not exact unchanged selection")
    if method in ("legacy_cg32", "cg_ray") and "legacy_cg" in metadata:
        cg = metadata["legacy_cg"]
        iterations, calls = cg["iterations"], cg["normal_matvec_calls"]
        if not 0 <= iterations <= 32 or calls not in (iterations + 1, iterations + 2) or calls > 33:
            raise ValueError("legacy CG horizon/matvec work differs")
        if cg["basis_apply_calls"] != calls or cg["basis_transpose_calls"] != calls + 1 or cg["denominator_calls"] != 1:
            raise ValueError("legacy CG operator counters disagree")
        expected["basis_apply_calls"] += calls
        expected["basis_transpose_calls"] += calls + 1
        expected["basis_denominator_passes"] += 1
        if method == "cg_ray":
            expected["legacy_cg_iterations"] = iterations
    if counts != expected:
        raise ValueError(f"exact operator/render counters disagree: expected {expected}")
    if not metadata["accepted"] and metadata["selected_metrics"] != metadata["parent_metrics"]:
        raise ValueError("rejected transaction did not restore parent quality")
    if metadata["accepted"] and not gate(metadata["selected_metrics"])[0]:
        raise ValueError("selected field does not pass the unchanged safe gate")
    if metadata["foreground_mse_improved"] != (selected.foreground_mse < before.foreground_mse):
        raise ValueError("reported foreground improvement differs")


def validate_artifacts(root, rows, protocol, problems, *, diagnostic=False):
    """Recompute bounded FIT-050 artifact semantics on CPU; hashes alone are insufficient."""
    import torch
    from structsplat.config import StructureTensorConfig
    from structsplat.gaussians import GaussianField
    from structsplat.safe_schedule import SafeScheduleConfig
    root = Path(root)
    expected_count, horizon = (16, 3) if diagnostic else (protocol["n_gaussians"], protocol["parent_steps"])
    parent_cache = {}
    all_samples = []
    for image_id, seed in workloads(diagnostic, protocol):
        name = parent_id(image_id, seed)
        directory = root / "parents" / name
        successful = [r for r in rows if r.get("parent_id") == name and r.get("status") == "ok"]
        if not successful:
            continue
        try:
            config = json.loads((directory / "config.json").read_text())
            init, fit_cfg = parent_configs(seed, smoke=diagnostic, device="cpu" if diagnostic else "cuda")
            expected_init = asdict(init) if diagnostic else {**protocol["parent_init"], "seed": seed}
            expected_fit = asdict(fit_cfg) if diagnostic else protocol["parent_fit"]
            if config["init"] != _json_form(expected_init) or config["fit"] != _json_form(expected_fit):
                raise ValueError("parent resolved initialization/fitter differs from frozen protocol")
            if config["tensor"] != _json_form(asdict(StructureTensorConfig())):
                raise ValueError("parent structure-tensor config differs")
            if config["image_id"] != image_id or config["seed"] != seed:
                raise ValueError("parent image/seed identity differs")
            initial = GaussianField.load(directory / "initial_field.npz", device="cpu")
            field = GaussianField.load(directory / "field.npz", device="cpu")
            if initial.n != expected_count or field.n != expected_count:
                raise ValueError("parent initial/terminal exact count differs")
            for candidate in (initial, field):
                for parameter in (candidate.means, candidate.log_scales, candidate.rotations, candidate.colors):
                    if not bool(torch.isfinite(parameter).all()):
                        raise ValueError("parent initial/terminal parameters are not finite")
            history = json.loads((directory / "history.json").read_text())
            expected_iterations = sorted(set(range(0, horizon, fit_cfg.log_every)) | {horizon - 1})
            if history["iter"] != expected_iterations or history["n_gaussians"] != [expected_count] * len(expected_iterations):
                raise ValueError("parent native history horizon/count differs")
            state = torch.load(directory / "optimizer_state.pt", map_location="cpu", weights_only=True)
            if len(state["state"]) != 4 or len(state["param_groups"]) != 4:
                raise ValueError("parent Adam group/state inventory differs")
            for group, tensor in zip(state["param_groups"], (field.means, field.log_scales, field.rotations, field.colors)):
                entry = state["state"][group["params"][0]]
                if float(entry["step"]) != horizon:
                    raise ValueError("parent Adam step differs from exact horizon")
                for key in ("exp_avg", "exp_avg_sq"):
                    if entry[key].shape != tensor.shape or not bool(torch.isfinite(entry[key]).all()):
                        raise ValueError("parent Adam moment shape/finiteness differs")
            bindings = {"initial_field_sha256": "initial_field.npz", "field_sha256": "field.npz",
                        "optimizer_state_sha256": "optimizer_state.pt", "target_sha256": "target.npy"}
            if any(config[key] != sha256(directory / filename) for key, filename in bindings.items()):
                raise ValueError("parent payload identity differs from resolved config")
            target = np.load(directory / "target.npy", allow_pickle=False)
            expected_target, source = load_target(image_id, smoke=diagnostic)
            if not np.array_equal(target, expected_target) or config["source"] != source:
                raise ValueError("parent target/source differs from frozen fixture preprocessing")
            samples = [json.loads(line) for line in (directory / "occupancy.jsonl").read_text().splitlines() if line]
            if not diagnostic and not samples:
                raise ValueError("formal parent lacks occupancy samples")
            previous = None
            for sample in samples:
                timestamp = datetime.fromisoformat(sample["utc"])
                if timestamp.tzinfo is None or (previous is not None and timestamp < previous):
                    raise ValueError("occupancy timestamps are not ordered timezone-aware values")
                previous = timestamp
                if sample["owned_worker_pid"] != config["environment"]["worker_pid"]:
                    raise ValueError("occupancy owner differs from parent worker")
                if sample["ok"] is True:
                    foreign = [p["pid"] for p in sample["processes"] if p["pid"] != sample["owned_worker_pid"]]
                    if sample["foreign_pids"] != foreign:
                        raise ValueError("occupancy foreign-process qualification differs")
                elif sample["ok"] is not False or not sample.get("error"):
                    raise ValueError("invalid occupancy query failure record")
            all_samples.extend(samples)
            parent_cache[name] = (field, config, target)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, IndexError) as exc:
            problems.append(f"FIT-050 parent {name}: {exc}")
    for row in rows:
        if row.get("status") != "ok" or row.get("parent_id") not in parent_cache:
            continue
        try:
            name = row["cell_id"]
            directory = root / "cells" / name
            parent, parent_config, target = parent_cache[row["parent_id"]]
            config = json.loads((directory / "config.json").read_text())
            request = json.loads((directory / "request.json").read_text())
            common = {key: row[key] for key in ("cell_id", "parent_id", "image_id", "seed", "method", "smoke",
                      "parent_field_sha256", "parent_optimizer_sha256", "target_sha256")}
            if config["request"] != common or request != common or row["smoke"] != diagnostic:
                raise ValueError("saved request identity differs from row")
            if config["fit"] != parent_config["fit"] or config["environment"] != parent_config["environment"]:
                raise ValueError("cell fitter/environment differs from its frozen parent")
            if config["parent_field_sha256"] != row["parent_field_sha256"]:
                raise ValueError("cell config parent-field binding differs")
            for row_key, parent_key in (("parent_field_sha256", "field_sha256"),
                                       ("parent_optimizer_sha256", "optimizer_state_sha256"),
                                       ("target_sha256", "target_sha256")):
                if row[row_key] != parent_config[parent_key]:
                    raise ValueError("cell does not bind exact parent field/state/target")
            if sha256(directory / "input_field.npz") != row["parent_field_sha256"]:
                raise ValueError("cell copied input field differs from parent")
            schedule = SafeScheduleConfig(capacity=expected_count, coverage_target_gaussians=expected_count,
                                           detail_target_gaussians=expected_count)
            if config["schedule"] != _json_form(asdict(schedule)):
                raise ValueError("cell safe schedule/tolerances differ from frozen defaults")
            history = json.loads((directory / "history.json").read_text())
            if history["parent_history"] != f'../../parents/{row["parent_id"]}/history.json':
                raise ValueError("cell parent-history link differs")
            metadata = history["transaction"]
            steps = (2 if diagnostic else protocol["adam_steps"]) if row["method"] == "adam32" else 0
            _validate_transaction(metadata, row, schedule, expected_count, steps)
            if row["method"] == "adam32":
                if history["adam_fit"]["iter"] != list(range(steps)) or metadata["iterations_run"] != steps:
                    raise ValueError("Adam continuation native history/horizon differs")
                expected_continuation = {**parent_config["fit"], "iters": steps, "log_every": 1}
                if metadata["continuation_config"] != expected_continuation:
                    raise ValueError("Adam continuation resolved config differs")
            elif history["adam_fit"]:
                raise ValueError("non-Adam arm contains Adam continuation history")
            selected = GaussianField.load(directory / "field.npz", device="cpu")
            if selected.n != expected_count:
                raise ValueError("saved selected field count differs")
            if row["method"] != "adam32" and not _field_equal(parent, selected, colors=False):
                raise ValueError("saved RGB-only field changed non-color state")
            if not row["accepted"] and not _field_equal(parent, selected):
                raise ValueError("saved rejected transaction is not exact parent rollback")
            if (not torch.equal(parent.colors, selected.colors)) != row["coefficients_changed"]:
                raise ValueError("saved coefficient-change flag differs")
            raw = np.load(directory / "reconstruction.npy", allow_pickle=False)
            if raw.shape != target.shape or not np.isfinite(raw).all():
                raise ValueError("stored reconstruction shape/finiteness differs")
            mse = float(np.square(raw.astype(np.float64) - target.astype(np.float64)).mean())
            psnr = -10 * math.log10(max(mse, 1e-12))
            if not math.isclose(mse, row["raw_mse"], rel_tol=1e-11, abs_tol=1e-14) or abs(psnr - row["psnr"]) > 1e-9:
                raise ValueError("raw reconstruction metrics differ from reported MSE/PSNR")
            if history["reporting_quality"] != row["reporting_metrics"]:
                raise ValueError("reporting quality differs between row and history")
            _quality_record(history["cold_quality"], expected_count)
            if history["reporting_work"] != {"quality_evaluations": 2, "gaussian_renders": 2, "coverage_passes": 2}:
                raise ValueError("reporting work counters differ")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, IndexError) as exc:
            problems.append(f'FIT-050 cell {row.get("cell_id")}: {exc}')
    try:
        occupancy = json.loads((root / "occupancy.json").read_text())
        timing_eligible = bool(all_samples) and all(s["ok"] and not s["foreign_pids"] for s in all_samples)
        if occupancy["samples"] != all_samples or occupancy["timing_eligible"] != timing_eligible:
            raise ValueError("root occupancy does not reproduce strict parent sample qualification")
        expected_decision = summarize(rows, protocol, diagnostic=diagnostic)
        expected_decision["timing_eligible"] = timing_eligible
        if json.loads((root / "decision.json").read_text()) != expected_decision:
            raise ValueError("decision.json differs from complete rows and occupancy qualification")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        problems.append(f"FIT-050 disposition: {exc}")


def gpu_snapshot(owned_pid):
    result = {"utc": datetime.now(timezone.utc).isoformat(), "owned_worker_pid": owned_pid}
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits"], text=True, timeout=5)
        processes = []
        for line in output.splitlines():
            pid, name, memory = line.split(",", 2)
            processes.append({"pid": int(pid.strip()), "name": name.strip(), "memory_mib": memory.strip()})
        result.update({"ok": True, "processes": processes,
                       "foreign_pids": [p["pid"] for p in processes if p["pid"] != owned_pid]})
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        result.update({"ok": False, "error": str(exc), "foreign_pids": []})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--worker")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--print-protocol-digest", action="store_true")
    parser.add_argument("--approved-protocol-digest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    digest = protocol_digest(PROTOCOL, SOURCES)
    if args.protocol_only:
        print(json.dumps({"protocol": PROTOCOL, "sources": SOURCES, "digest": digest}, indent=2))
        return
    if args.print_protocol_digest:
        print(digest)
        return
    if not args.out or (not args.smoke and args.approved_protocol_digest != digest):
        parser.error("--out and exact approved digest required for formal execution")
    bundle = ResearchBundle(args.out, task="FIT-050", protocol=PROTOCOL, digest=digest,
        expected_cells=expected_cells(diagnostic=args.smoke), diagnostic=args.smoke, source_paths=SOURCES)
    rows, samples = [], []
    for index, (image_id, seed) in enumerate(workloads(args.smoke)):
        pid = parent_id(image_id, seed)
        directory = bundle.root / "parents" / pid
        directory.mkdir(parents=True)
        shift = index % len(ARMS)
        request = {"root": str(bundle.root), "image_id": image_id, "seed": seed,
                   "parent_id": pid, "smoke": args.smoke, "arm_order": ARMS[shift:] + ARMS[:shift]}
        write_json(directory / "request.json", request)
        with (directory / "worker.log").open("w") as log, (directory / "occupancy.jsonl").open("w") as occupancy:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory / "request.json")],
                                       stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while process.poll() is None:
                if not args.smoke:
                    snapshot = gpu_snapshot(process.pid)
                    samples.append(snapshot)
                    occupancy.write(json.dumps(snapshot) + "\n")
                    occupancy.flush()
                if time.monotonic() - started > PROTOCOL["worker_timeout_seconds"]:
                    process.terminate()
                    process.wait(timeout=30)
                    break
                time.sleep(1)
        if process.returncode == 0 and (directory / "rows.json").exists():
            new_rows = json.loads((directory / "rows.json").read_text())
        else:
            new_rows = [{"cell_id": f"{pid}_{arm}", "parent_id": pid, "image_id": image_id,
                         "seed": seed, "method": arm, "status": "error",
                         "error": f"parent worker exit {process.returncode}",
                         "artifacts": {"worker.log": f"parents/{pid}/worker.log"}} for arm in ARMS]
        rows.extend(new_rows)
        print(json.dumps({"parent_id": pid, "statuses": [r["status"] for r in new_rows]}), flush=True)
    timing_eligible = bool(samples) and all(s["ok"] and not s["foreign_pids"] for s in samples)
    write_json(bundle.root / "occupancy.json", {"samples": samples, "timing_eligible": timing_eligible,
        "qualification": "point samples only; absence of observed foreign work is not continuous exclusivity proof"})
    decision = summarize(rows, diagnostic=args.smoke)
    decision["timing_eligible"] = timing_eligible
    write_json(bundle.root / "decision.json", decision)
    for row in rows:
        row.setdefault("artifacts", {}).update({"decision": "decision.json", "occupancy": "occupancy.json"})
    bundle.finish(rows, title="FIT-050 — Safeguarded normalized color rays",
        interpretation="Four exposed development images, paired seeds; single fixed-geometry RGB transactions versus independently charged CG and full-parameter Adam continuation. Defaults unchanged. Native parent/Adam curves and every ray trial are retained; perceptual metrics are endpoints. Transaction timing includes checks, not subsequent reporting.")


if __name__ == "__main__":
    main()
