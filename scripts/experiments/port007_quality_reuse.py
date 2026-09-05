#!/usr/bin/env python3
"""PORT-007 paired, same-state and complete-pipeline quality-reuse experiment.

Freeze: python scripts/experiments/port007_quality_reuse.py --protocol-only
Formal: python scripts/experiments/port007_quality_reuse.py OUT \
    --parent-bundle FIT050_OUT --approved-protocol-digest DIGEST

Only the complete frozen matrix is exposed. Unit tests use procedural arrays, not formal images.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
import traceback

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmarks.hier_research_report import (  # noqa: E402
    ResearchBundle, protocol_digest, repository_state, save_rgb, sha256, validate_bundle, write_json,
)
from benchmarks.port007_controls import (  # noqa: E402
    ARMS, BACKENDS, IMAGE_HASHES, PROTOCOL, SOURCES, coefficient_of_variation,
    counterbalanced_orders, discrete_projection, ellipse_mask, expected_cells,
    pipeline_order, signature, summarize,
)


def load_image(image_id):
    path = ROOT / "tests" / "test_images" / f"COCO_train2014_{image_id:012d}.jpg"
    if sha256(path) != IMAGE_HASHES[image_id]:
        raise RuntimeError(f"frozen JPEG hash mismatch: {image_id}")
    image = Image.open(path).convert("RGB")
    scale = min(1.0, PROTOCOL["max_side"] / max(image.size))
    if scale < 1:
        image = image.resize(tuple(max(1, round(v * scale)) for v in image.size),
                             Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def gpu_snapshot():
    result = {"monotonic_seconds": time.perf_counter(), "pid": os.getpid(),
              "compute_processes": [], "error": None}
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        result["raw_compute_processes"] = raw
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if parts and parts[0].isdigit():
                result["compute_processes"].append({"pid": int(parts[0]), "name": parts[1],
                                                     "memory_mib": parts[2]})
        result["gpu_state"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    result["timing_eligible"] = (result["error"] is None and all(
        process["pid"] == os.getpid() for process in result["compute_processes"]))
    return result


def copy_field(field, device):
    from structsplat.gaussians import GaussianField
    names = ("means", "log_scales", "rotations", "colors", "opacities", "scale_max",
             "color_grads", "background_mask", "filter_variance")
    return GaussianField(*(None if getattr(field, name) is None else
                           getattr(field, name).detach().to(device).clone() for name in names))


def score(raw, target):
    import torch
    from structsplat.metrics import LPIPS, ms_ssim, ssim
    with torch.no_grad():
        mse = float((raw.double() - target.double()).square().mean())
        display = raw.clamp(0, 1)
        lpips = LPIPS.distance(display, target)
        if lpips is None or not math.isfinite(lpips):
            raise RuntimeError("required LPIPS is unavailable/nonfinite")
        return {"psnr": -10 * math.log10(max(mse, 1e-12)), "mse": mse,
                "mae": float((raw - target).abs().mean()), "ssim": float(ssim(display, target)),
                "ms_ssim": float(ms_ssim(display, target)), "lpips": lpips}


def plot_curves(directory, trace, *, same_state=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ("psnr", "ssim", "ms_ssim", "lpips", "mse", "mae", "cvar99_mse", "p99_mse",
            "interior_hole_fraction", "boundary_hole_fraction", "elapsed_seconds", "n_gaussians")
    fig, axes = plt.subplots(4, 3, figsize=(12, 11))
    xkey = "round" if same_state else "attempted_steps"
    for ax, key in zip(axes.flat, keys):
        values = [(point[xkey], point.get(key)) for point in trace if point.get(key) is not None]
        if values:
            ax.plot([p[0] for p in values], [p[1] for p in values], marker=".", markersize=3)
        ax.set(xlabel="Repeated same-state call" if same_state else "Attempted schedule steps",
               ylabel=key)
    fig.suptitle("Fixed field: repeated quality calls, not convergence" if same_state else
                 "Selected state at every native event (including rejected transactions)")
    fig.tight_layout()
    fig.savefig(directory / "curves.png", dpi=120)
    plt.close(fig)


def write_cell_artifacts(directory, field, target, raw, history, config, *, same_state=False):
    directory.mkdir(parents=True, exist_ok=True)
    field.save(directory / "field.npz")
    target_np, raw_np = target.detach().cpu().numpy(), raw.detach().cpu().numpy()
    np.save(directory / "target.npy", target_np)
    np.save(directory / "reconstruction.npy", raw_np)
    save_rgb(directory / "target.png", target_np)
    save_rgb(directory / "reconstruction.png", raw_np)
    save_rgb(directory / "error.png", np.abs(target_np - raw_np) * 4)
    write_json(directory / "history.json", history)
    write_json(directory / "config.json", config)
    plot_curves(directory, history["checkpoints"], same_state=same_state)
    return {path.name: f"cells/{directory.name}/{path.name}" for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "row.json"}


def warmup(*, pipeline=False):
    import torch
    from benchmarks.fit050_controls import full_frame_context, parent_configs
    from structsplat.gaussians import GaussianField
    from structsplat.pipeline import PipelineConfig, run_pipeline
    from structsplat.safe_schedule import evaluate_quality
    rng = np.random.default_rng(812)
    field = GaussianField.from_numpy(rng.uniform(6, 26, (5, 2)).astype(np.float32),
        np.full((5, 2), 3, dtype=np.float32), np.zeros(5, dtype=np.float32),
        rng.uniform(0.1, 0.9, (5, 3)).astype(np.float32), device="cuda")
    target_np = rng.uniform(0, 1, (32, 32, 3)).astype(np.float32)
    target = torch.as_tensor(target_np, device="cuda")
    _, cfg = parent_configs(77, smoke=True)
    mask, constraint, schedule = full_frame_context(target, cfg, field.n)
    for arm in ARMS:
        coverage, tail = BACKENDS[arm]
        arm_cfg = replace(cfg, quality_coverage_backend=coverage, quality_tail_backend=tail)
        for _ in range(2):
            evaluate_quality(field, target, mask, arm_cfg, constraint, schedule.coverage_tau)
    if pipeline:
        run_pipeline(target_np, cfg=PipelineConfig(capacity=16, step_scale=0.001,
            device="cuda", renderer="cuda", seed=77), verbose=False)
    torch.cuda.synchronize()


def instrumented_probe(field, target, mask, cfg, constraint, tau):
    """Untimed replay only; capture same-call denominator and count reference fallbacks."""
    import structsplat.safe_schedule as safe
    original_inputs = safe._quality_render_inputs
    original_reference = safe._raw_weight_map_field
    captured = {"reference_calls": 0}

    def reference(*args, **kwargs):
        captured["reference_calls"] += 1
        return original_reference(*args, **kwargs)

    def inputs(*args, **kwargs):
        rgb, den = original_inputs(*args, **kwargs)
        captured["denominator"] = den
        return rgb, den

    safe._quality_render_inputs = inputs
    safe._raw_weight_map_field = reference
    try:
        metrics, rendered = safe.evaluate_quality(field, target, mask, cfg, constraint, tau)
    finally:
        safe._quality_render_inputs = original_inputs
        safe._raw_weight_map_field = original_reference
    return metrics, rendered, captured["denominator"], captured["reference_calls"]


def decision(before, after, tolerance):
    from structsplat.safe_schedule import QualityMetrics, safe_commit_decision
    keys = QualityMetrics.__dataclass_fields__
    left = QualityMetrics(**{key: before[key] for key in keys})
    right = QualityMetrics(**{key: after[key] for key in keys})
    accepted, reasons = safe_commit_decision(left, right, tolerance)
    return {"accepted": accepted, "reasons": reasons}


def same_worker(request, bundle_root):
    import torch
    from benchmarks.fit050_controls import full_frame_context, parent_configs
    from structsplat.gaussians import GaussianField
    from structsplat.safe_schedule import CommitTolerances, evaluate_quality

    image, seed = request["image"], request["seed"]
    parent_id = f"coco{image:012d}_s{seed}"
    parent = bundle_root / "parents" / parent_id
    target_np = np.load(parent / "target.npy", allow_pickle=False)
    if not np.array_equal(target_np, load_image(image)):
        raise RuntimeError("transferred parent target differs from frozen decoded image")
    target = torch.as_tensor(target_np, device="cuda")
    parent_config = json.loads((parent / "config.json").read_text())
    init_cfg, cfg = parent_configs(seed)
    if parent_config["init"] != asdict(init_cfg) or parent_config["fit"] != asdict(cfg):
        raise RuntimeError("transferred parent config differs from frozen parent configuration")
    fields = {"initial": GaussianField.load(parent / "initial_field.npz", device="cuda"),
              "terminal": GaussianField.load(parent / "field.npz", device="cuda")}
    if any(field.n != 2000 for field in fields.values()):
        raise RuntimeError("transferred parent violates frozen exact count")
    mask, constraint, schedule = full_frame_context(target, cfg, 2000)
    samples = {(state, arm): [] for state in fields for arm in ARMS}
    snapshots = [gpu_snapshot()]
    for state, field in fields.items():
        configs = {arm: replace(cfg, quality_coverage_backend=BACKENDS[arm][0],
                                quality_tail_backend=BACKENDS[arm][1]) for arm in ARMS}
        for arm in ARMS:
            for _ in range(PROTOCOL["same_state"]["warmups_per_arm_and_state"]):
                evaluate_quality(field, target, mask, configs[arm], constraint, schedule.coverage_tau)
        for round_index, order in enumerate(counterbalanced_orders()):
            for position, arm in enumerate(order):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                quality, raw = evaluate_quality(field, target, mask, configs[arm], constraint,
                                               schedule.coverage_tau)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated()
                raw_np = raw.cpu().numpy().copy()
                replay_quality, replay_raw, den, reference_calls = instrumented_probe(
                    field, target, mask, configs[arm], constraint, schedule.coverage_tau)
                samples[state, arm].append({"round": round_index, "order_position": position,
                    "seconds": elapsed, "peak_allocated_bytes": peak, "quality": quality.to_dict(),
                    "replay_quality": replay_quality.to_dict(), "render": raw_np,
                    "replay_render": replay_raw.cpu().numpy().copy(),
                    "denominator": den.cpu().numpy().copy(),
                    "holes": (den < schedule.coverage_tau).cpu().numpy().copy(),
                    "reference_calls_in_untimed_replay": reference_calls})
        snapshots.append(gpu_snapshot())
    tolerance = CommitTolerances()
    for state, field in fields.items():
        other = "terminal" if state == "initial" else "initial"
        for arm in ARMS:
            cell_id = f"same_{parent_id}_{state}_{arm}"
            directory = bundle_root / "cells" / cell_id
            directory.mkdir(parents=True, exist_ok=False)
            points, checks, quality_deltas, decisions = [], [], [], []
            records = samples[state, arm]
            baseline = samples[state, "legacy_a"]
            for r, (record, base) in enumerate(zip(records, baseline)):
                null = decision(base["quality"], record["quality"], tolerance)
                null_control = decision(base["quality"], base["quality"], tolerance)
                changed_before = samples[other, "legacy_a"][r]["quality"]
                changed = decision(changed_before, record["quality"], tolerance)
                changed_control = decision(changed_before, base["quality"], tolerance)
                decisions.append({"round": r, "null": null, "null_control": null_control,
                                  "changed": changed, "changed_control": changed_control,
                                  "changed_direction": f"{other}_to_{state}"})
                rgb_error = float(np.max(np.abs(record["render"] - base["render"])))
                replay_error = float(np.max(np.abs(record["replay_render"] - base["replay_render"])))
                exact_holes = bool(np.array_equal(record["holes"], base["holes"]))
                finite = (record["quality"]["finite"] and record["replay_quality"]["finite"]
                          and all(np.isfinite(record[key]).all() for key in
                                  ("render", "replay_render", "denominator")))
                checks.append({"round": r, "max_rgb_error": rgb_error,
                    "max_replay_rgb_error": replay_error, "hole_mask_equal": exact_holes,
                    "null_decision_equal": null == null_control,
                    "changed_decision_equal": changed == changed_control, "finite": bool(finite),
                    "pass": bool(finite and rgb_error <= 2e-5 and replay_error <= 2e-5
                                 and exact_holes and null == null_control and changed == changed_control)})
                quality_deltas.append({key: record["quality"][key] - base["quality"][key]
                                       for key in record["quality"] if
                                       isinstance(record["quality"][key], (float, int))
                                       and not isinstance(record["quality"][key], bool)})
                scores = score(torch.as_tensor(record["render"], device="cuda"), target)
                points.append({**scores, **record["quality"], "round": r,
                    "attempted_steps": 0 if state == "initial" else 750,
                    "elapsed_seconds": sum(x["seconds"] for x in records[:r + 1])})
            np.savez_compressed(directory / "measurements.npz",
                renders=np.stack([r["render"] for r in records]),
                replay_renders=np.stack([r["replay_render"] for r in records]),
                raw_denominators=np.stack([r["denominator"] for r in records]),
                hole_masks=np.stack([r["holes"] for r in records]))
            measurements = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in records]
            write_json(directory / "measurements.json", measurements)
            write_json(directory / "parity.json", {"checks": checks, "quality_deltas": quality_deltas,
                                                     "decisions": decisions})
            final = torch.as_tensor(records[-1]["render"], device="cuda")
            history = {"checkpoints": points, "kind": "repeated fixed state, not optimization",
                       "parent_history": json.loads((parent / "history.json").read_text())}
            config = {"request": {**request, "cell_id": cell_id, "state": state, "method": arm},
                      "fit": asdict(replace(cfg, quality_coverage_backend=BACKENDS[arm][0],
                                            quality_tail_backend=BACKENDS[arm][1])),
                      "parent_id": parent_id, "parent_config": parent_config,
                      "parent_files": {p.name: sha256(p) for p in sorted(parent.iterdir()) if p.is_file()},
                      "gpu_snapshots": snapshots, "torch": torch.__version__, "cuda": torch.version.cuda,
                      "numpy": np.__version__, "worker_pid": os.getpid(), "gpu": torch.cuda.get_device_name()}
            artifacts = write_cell_artifacts(directory, field, target, final, history, config, same_state=True)
            times = [r["seconds"] for r in records]
            row = {"cell_id": cell_id, "kind": "same", "status": "ok", "image": image, "seed": seed,
                "state": state, "method": arm, "n_gaussians": field.n,
                "iterations_run": 0 if state == "initial" else 750,
                "selected_iteration": 0 if state == "initial" else 750,
                **score(final, target), "quality": records[-1]["quality"],
                "total_seconds": sum(times), "call_seconds": times,
                "median_call_seconds": statistics.median(times), "call_time_cv": coefficient_of_variation(times),
                "parity_pass": all(check["pass"] for check in checks),
                "timing_eligible": all(snap["timing_eligible"] for snap in snapshots),
                "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
                "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "untimed_reference_coverage_calls": sum(r["reference_calls_in_untimed_replay"] for r in records),
                "artifacts": artifacts}
            write_json(directory / "row.json", row)


def pipeline_worker(request, bundle_root, *, device="cuda"):
    import torch
    from structsplat.fit import _MaskConstraint
    from structsplat.pipeline import PipelineConfig, build_fit_config, render_field, run_pipeline

    image, seed, arm = request["image"], request["seed"], request["method"]
    directory = bundle_root / "cells" / request["cell_id"]
    directory.mkdir(parents=True, exist_ok=False)
    image_np = load_image(image)
    mask_np = None if image == 9 else ellipse_mask(*image_np.shape[:2])
    coverage, tail = BACKENDS[arm]
    cfg = PipelineConfig(**{**PROTOCOL["pipeline"]["config"], "seed": seed, "device": device,
                            "renderer": "cuda" if device == "cuda" else "normalized",
                            "quality_coverage_backend": coverage, "quality_tail_backend": tail})
    fit_cfg = build_fit_config(cfg, device)
    observed, observer_seconds = [], 0.0

    def observer(field, event):
        nonlocal observer_seconds
        started = time.perf_counter()
        selected_cpu = copy_field(field, "cpu")
        duration = time.perf_counter() - started
        observer_seconds += duration
        observed.append((selected_cpu, event, observer_seconds))

    before = gpu_snapshot()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = run_pipeline(image_np, mask=mask_np, cfg=cfg, observer=observer, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    after = gpu_snapshot()
    target, raw = output["target"], output["render"].detach()
    actual_mask = torch.ones(target.shape[:2], device=device, dtype=torch.bool) if mask_np is None else torch.as_tensor(mask_np, device=device)
    constraint = _MaskConstraint.from_mask(actual_mask.cpu().numpy(), device, target.dtype,
        fit_cfg.sigma_cutoff, fit_cfg.mask_margin, aa_dilation=fit_cfg.aa_dilation,
        cap_mode="anisotropic", undercoverage_band=cfg.boundary_band)
    reference_cfg = replace(fit_cfg, quality_coverage_backend="reference", quality_tail_backend="reference")
    quality, reference_raw, reference_den, _ = instrumented_probe(
        output["field"], target, actual_mask, reference_cfg, constraint, cfg.coverage_tau)
    final_parity = float((reference_raw - raw).abs().max())
    np.save(directory / "reference_reconstruction.npy", reference_raw.cpu().numpy())
    np.save(directory / "reference_denominator.npy", reference_den.cpu().numpy())
    points = []
    snapshot_directory = directory / "snapshots"
    snapshot_directory.mkdir()
    for index, (selected, event, observed_seconds) in enumerate(observed):
        selected.save(snapshot_directory / f"field_{index:04d}.npz")
        gpu_field = copy_field(selected, device)
        selected_raw = render_field(gpu_field, target, reference_cfg).detach()
        point = {**score(selected_raw, target), **event["selected"],
                 "event_index": index, "attempted_steps": event["global_attempted_steps"],
                 "accepted_steps": event["global_accepted_steps"],
                 "elapsed_seconds": event["elapsed_seconds"],
                 "observer_cumulative_seconds": observed_seconds,
                 "phase": event["phase"], "event": event["event"], "accepted": event["accepted"]}
        points.append(point)
        if index in (0, len(observed) - 1):
            save_rgb(directory / f"checkpoint_{index:04d}.png", selected_raw.cpu().numpy())
    trajectory = discrete_projection(output["history"])
    write_json(directory / "trajectory.json", trajectory)
    np.save(directory / "mask.npy", actual_mask.cpu().numpy())
    history = {"checkpoints": points, "native_events": output["history"],
               "attempted_steps": output["attempted_steps"], "accepted_steps": output["accepted_steps"],
               "nominal_phase_ceilings": {phase["name"]: phase["max_steps"]
                   for phase in output["schedule"].values() if isinstance(phase, dict) and "max_steps" in phase},
               "selected_snapshot_scope": "every native event; rejected trial fields not exposed"}
    config = {"request": request, "pipeline": asdict(cfg), "fit": output["fit_config"],
        "schedule": output["schedule"], "recipe": output["recipe"], "init": output["init"],
        "storage": output["storage"], "native_timing": output["timing"],
        "observer_seconds": observer_seconds, "instrumented_total_seconds": elapsed,
        "gpu_snapshots": [before, after], "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "worker_pid": os.getpid(),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "source_jpeg_sha256": IMAGE_HASHES[image], "mask_sha256": sha256(directory / "mask.npy")}
    artifacts = write_cell_artifacts(directory, output["field"], target, raw, history, config)
    row = {**request, "status": "ok", "n_gaussians": output["field"].n,
        "iterations_run": output["attempted_steps"],
        "selected_iteration": max((event["global_attempted_steps"] for event in output["history"]
                                   if event["accepted"]), default=0),
        "accepted_steps": output["accepted_steps"],
        **score(raw, target), "quality": quality.to_dict(), "final_reference_rgb_max_error": final_parity,
        "total_seconds": elapsed, "observer_seconds": observer_seconds,
        "exploratory_observer_subtracted_seconds": elapsed - observer_seconds,
        "native_pipeline_seconds": output["timing"]["total_seconds"],
        "timing_eligible": before["timing_eligible"] and after["timing_eligible"],
        "trajectory_sha256": signature(trajectory), "event_count": len(output["history"]),
        "peak_allocated_bytes": peak,
        "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "artifacts": artifacts}
    write_json(directory / "row.json", row)


def worker(request_path):
    import torch
    torch.set_num_threads(PROTOCOL["threads"])
    if (not torch.cuda.is_available() or torch.cuda.get_device_name() != PROTOCOL["gpu"]
            or torch.__version__ != PROTOCOL["torch"]):
        raise RuntimeError("frozen GPU/PyTorch environment unavailable")
    request = json.loads(Path(request_path).read_text())
    bundle_root = Path(request["bundle_root"])
    warmup(pipeline=request["kind"] == "pipeline")
    if request["kind"] == "same":
        same_worker(request, bundle_root)
    else:
        pipeline_worker(request, bundle_root)


def copy_parents(source, destination):
    """Require a completed, hash-valid prospective FIT-050 bundle, not selected parent exports."""
    problems = validate_bundle(source)
    if problems:
        raise RuntimeError("FIT-050 parent bundle invalid: " + "; ".join(problems))
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["task"] != "FIT-050" or manifest["repository"]["commit"] != repository_state()["commit"]:
        raise RuntimeError("parent bundle must be FIT-050 from this exact clean source commit")
    names = ("initial_field.npz", "field.npz", "history.json", "config.json", "target.npy", "optimizer_state.pt")
    copied = {}
    for image in PROTOCOL["images"]:
        for seed in PROTOCOL["same_state"]["seeds"]:
            parent_id = f"coco{image:012d}_s{seed}"
            output = destination / "parents" / parent_id
            output.mkdir(parents=True)
            for name in names:
                relative = f"parents/{parent_id}/{name}"
                original = source / relative
                if manifest["files"].get(relative) != sha256(original):
                    raise RuntimeError("parent file is not bound by completed FIT-050 manifest")
                shutil.copy2(original, output / name)
                copied[relative] = sha256(output / name)
    write_json(destination / "parent_source.json", {
        "source_bundle": str(source), "manifest_sha256": sha256(source / "manifest.json"),
        "protocol_digest": manifest["protocol_digest"], "repository": manifest["repository"],
        "files": copied, "source_identity": manifest,
        "transfer": "all eight prescribed parents, initial and terminal, without outcome selection"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?")
    parser.add_argument("--parent-bundle", type=Path)
    parser.add_argument("--approved-protocol-digest")
    parser.add_argument("--protocol-only", "--print-protocol", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    digest = protocol_digest(PROTOCOL, SOURCES)
    if args.protocol_only:
        print(json.dumps({"protocol_digest": digest, "protocol": PROTOCOL, "sources": SOURCES},
                         indent=2, sort_keys=True))
        return
    if not args.outdir or args.parent_bundle is None or args.approved_protocol_digest != digest:
        parser.error("OUT, --parent-bundle and the exact --approved-protocol-digest are required")
    source = args.parent_bundle.resolve()
    # Check before spending any compute or creating a nominally successful output bundle.
    problems = validate_bundle(source)
    if problems:
        raise RuntimeError("parent bundle validation failed: " + "; ".join(problems))
    bundle = ResearchBundle(args.outdir, task="PORT-007", protocol=PROTOCOL, digest=digest,
                            expected_cells=expected_cells(), source_paths=SOURCES)
    copy_parents(source, bundle.root)
    requests = []
    for image in PROTOCOL["images"]:
        for seed in PROTOCOL["same_state"]["seeds"]:
            requests.append({"kind": "same", "image": image, "seed": seed,
                             "group_id": f"same_coco{image:012d}_s{seed}"})
    for image_index, image in enumerate(PROTOCOL["pipeline"]["images"]):
        for seed in PROTOCOL["pipeline"]["seeds"]:
            for arm in pipeline_order(image_index, seed):
                cell_id = f"pipeline_coco{image:012d}_s{seed}_{arm}"
                requests.append({"kind": "pipeline", "image": image, "seed": seed,
                                 "method": arm, "cell_id": cell_id, "group_id": cell_id})
    rows = []
    requests_dir = bundle.root / "requests"
    requests_dir.mkdir()
    for index, request in enumerate(requests):
        request = {**request, "bundle_root": str(bundle.root)}
        request_path = requests_dir / (request["group_id"] + ".json")
        write_json(request_path, request)
        log_path = requests_dir / (request["group_id"] + ".log")
        error = None
        try:
            with log_path.open("w") as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                    timeout=PROTOCOL["worker_timeout_seconds"])
        except (subprocess.SubprocessError, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            (requests_dir / (request["group_id"] + ".exception.txt")).write_text(traceback.format_exc())
        ids = [request["cell_id"]] if request["kind"] == "pipeline" else [
            f'{request["group_id"]}_{state}_{arm}' for state in ("initial", "terminal") for arm in ARMS]
        for cell_id in ids:
            row_path = bundle.root / "cells" / cell_id / "row.json"
            if row_path.exists():
                rows.append(json.loads(row_path.read_text()))
            else:
                rows.append({"cell_id": cell_id, "status": "error", "kind": request["kind"],
                    "image": request["image"], "seed": request["seed"],
                    "method": request["method"] if request["kind"] == "pipeline"
                    else next(arm for arm in ARMS if cell_id.endswith("_" + arm)),
                    "error": error or "worker produced no row", "artifacts": {
                        "worker_log": log_path.relative_to(bundle.root).as_posix(),
                        "request": request_path.relative_to(bundle.root).as_posix()}})
        print(f"[{index + 1}/{len(requests)}] {request['group_id']}: {error or 'complete'}", flush=True)
    write_json(bundle.root / "summary.json", summarize(rows))
    bundle.finish(rows, title="PORT-007: shared quality measurements",
        interpretation="Exposed-development 110-cell study: repeated fixed-state quality evaluation plus instrumented bounded complete pipelines. Both optimizations are the preregistered primary; singles are explanatory. CUDA A/A and exact decision/coverage guards determine eligibility; no default or 11000-row claim.")
    issues = validate_bundle(bundle.root)
    print(json.dumps({"output": str(bundle.root), "validation_problems": issues}, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
