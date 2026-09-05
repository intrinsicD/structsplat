#!/usr/bin/env python3
"""HIER-035 fixed-count additive convergence controls.

Formal: python scripts/experiments/hier035_convergence.py OUT --approved-protocol-digest DIGEST
Wiring: same driver with --smoke, using only synthetic seed77 and three steps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmarks.hier_additive_controls import (  # noqa: E402
    ControlConfig, additive_render, fit_control, pack, unpack,
)
from benchmarks.hier_research_report import (  # noqa: E402
    ResearchBundle, protocol_digest, save_rgb, sha256, write_json,
)

SOURCES = [
    "scripts/experiments/hier035_convergence.py", "benchmarks/hier_additive_controls.py",
    "benchmarks/hier_research_report.py", "scripts/check_report_bundle.py",
    "src/structsplat/pixel_gradient.py", "src/structsplat/gaussians.py",
    "src/structsplat/render.py", "src/structsplat/cuda_render.py",
    "src/structsplat/cuda/render_ext.cpp", "src/structsplat/cuda/render_ext.cu",
    "src/structsplat/metrics.py",
]
PROTOCOL = {
    "task": "HIER-035", "version": 1, "device": "cuda", "gpu": "NVIDIA GeForce RTX 3050",
    "dtype": "float32", "threads": 1, "renderer": "cuda_additive", "sigma_cutoff": 3,
    "support_fade_alpha": 1, "shape": [64, 64], "n_gaussians": 16,
    "families": ["translated", "anisotropic", "overlap", "texture"], "seeds": [0, 1, 2],
    "arms": ["adam_0p3", "adam_1", "adam_3", "diagonal", "block"],
    "order": "rotate arm order by seed; one separate warmed process per cell",
    "config": asdict(ControlConfig()),
    "adam_multipliers": {"adam_0p3": 0.3, "adam_1": 1.0, "adam_3": 3.0},
    "adam_parameter_lrs": [0.1, 0.03, 0.03, 0.03],
    "objective": "raw 0.5 mean squared RGB error, whole image; no mask or display clamp in fitting",
    "terminal": "exact 160 attempted updates; terminal state, no best-checkpoint selection",
    "scoring": "raw PSNR floor MSE1e-12; display-clamped MS-SSIM/LPIPS",
    "warmup": "seed77 translated, 2 steps each Adam, diagonal, block",
    "worker_timeout_seconds": 600, "max_backtracks": 6,
    "primary": "terminal 160-step PSNR against per-seed best-of-three Adam envelope",
    "minimum_median_gain_db": 0.5, "max_seed_loss_db": 0.1,
    "max_ms_ssim_loss": 0.005, "max_lpips_increase": 0.01,
    "cold_parity_max_abs": 2e-5,
    "timing": "complete fit including gradients, curvature, trial renders and scalar trace; descriptive only",
    "timing_scope": "shared workstation; record GPU process snapshots, no speed or isolated-time claim",
    "matched_time": "last trace sample at or before common minimum terminal elapsed time within family/seed",
    "data_role": "new procedural development mechanisms, not natural-image or held-out evidence",
    "missing_policy": "retain all errors; incomplete family has no positive verdict; no selective rerun",
    "forbidden": ["default changes", "threshold rescue", "in-place repair", "held-out access"],
}


def fixture(family, seed):
    import torch
    from structsplat.gaussians import GaussianField

    rng = np.random.default_rng(seed)
    height, width = PROTOCOL["shape"]
    coordinates = np.linspace(10, 54, 4, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    means = np.stack((xx.ravel(), yy.ravel()), 1)
    means += rng.normal(0, 0.2, means.shape).astype(np.float32)
    scale = (5.0, 1.5) if family == "anisotropic" else (8.0, 5.0) if family in ("overlap", "texture") else (4.0, 4.0)
    scales = np.tile(scale, (16, 1)).astype(np.float32)
    angles = rng.uniform(-0.6, 0.6, 16).astype(np.float32)
    colors = rng.uniform(0.08, 0.4, (16, 3)).astype(np.float32)
    truth = GaussianField.from_numpy(means, scales, angles, colors)
    target = additive_render(truth, height, width).detach()
    truth.colors *= 0.85 / target.max().clamp_min(0.85)
    target = additive_render(truth, height, width).detach()
    initial = truth.detached()
    if family == "translated":
        initial.means += torch.from_numpy(rng.normal(0, 1.8, means.shape).astype(np.float32))
    elif family == "anisotropic":
        initial.means += torch.from_numpy(rng.normal(0, 0.5, means.shape).astype(np.float32))
        initial.log_scales += torch.from_numpy(rng.normal(0, 0.3, means.shape).astype(np.float32))
        initial.rotations += 0.35
        initial.colors *= 0.8
    elif family == "overlap":
        initial.means += torch.from_numpy(rng.normal(0, 1.5, means.shape).astype(np.float32))
        initial.log_scales += torch.from_numpy(rng.normal(0, 0.15, means.shape).astype(np.float32))
        initial.colors *= 0.6
    elif family == "texture":
        py, px = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        phase = seed * 0.31
        target = torch.stack((0.4 + 0.2 * torch.sin(px / 5 + phase) * torch.cos(py / 7),
                              0.4 + 0.2 * torch.sin((px + py) / 8 + phase),
                              0.4 + 0.2 * torch.cos(px / 11 - py / 9)), -1)
    else:
        raise ValueError("unknown fixture family")
    return initial, target.to(torch.float32)


def gpu_process_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True).strip()


def resolved_config(method, smoke=False):
    cfg = ControlConfig(**PROTOCOL["config"])
    return replace(cfg, arm="adam" if method.startswith("adam") else method,
                   adam_multiplier=PROTOCOL["adam_multipliers"].get(method, 1.0),
                   steps=3 if smoke else cfg.steps)


def worker(request_path):
    import torch
    from structsplat.gaussians import GaussianField
    from structsplat.metrics import LPIPS, ms_ssim

    torch.set_num_threads(PROTOCOL["threads"])
    request = json.loads(Path(request_path).read_text())
    directory = Path(request_path).parent
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != PROTOCOL["gpu"]:
        raise RuntimeError("frozen GPU unavailable")
    before_processes = gpu_process_snapshot()
    warm, warm_target = fixture("translated", 77)
    warm, warm_target = unpack(pack(warm).cuda()), warm_target.cuda()
    for arm in ("adam", "diagonal", "block"):
        fit_control(warm, warm_target, ControlConfig(arm=arm, steps=2), renderer=PROTOCOL["renderer"])
    initial, target = fixture(request["family"], request["seed"])
    initial.save(directory / "input_field.npz")
    np.save(directory / "target.npy", target.numpy())
    initial, target = unpack(pack(initial).cuda()), target.cuda()
    cfg = resolved_config(request["method"], request["smoke"])
    torch.cuda.reset_peak_memory_stats()
    with (directory / "progress.jsonl").open("w") as progress:
        def callback(row):
            progress.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            progress.flush()
        field, raw, history, elapsed = fit_control(initial, target, cfg,
            renderer=PROTOCOL["renderer"], callback=callback)
    peak = torch.cuda.max_memory_allocated()
    after_processes = gpu_process_snapshot()
    field.save(directory / "field.npz")
    cold = GaussianField.load(directory / "field.npz", device="cuda")
    cold_render = additive_render(cold, *PROTOCOL["shape"], renderer=PROTOCOL["renderer"])
    cold_error = float((cold_render - raw).abs().max())
    if field.n != PROTOCOL["n_gaussians"] or not torch.equal(pack(cold), pack(field)):
        raise RuntimeError("decoded parameter or exact count mismatch")
    if cold_error > PROTOCOL["cold_parity_max_abs"]:
        raise RuntimeError("cold render parity failed")
    raw_np, target_np = raw.cpu().numpy(), target.cpu().numpy()
    np.save(directory / "reconstruction.npy", raw_np)
    save_rgb(directory / "target.png", target_np)
    save_rgb(directory / "reconstruction.png", raw_np)
    save_rgb(directory / "error.png", np.abs(raw_np - target_np) * 4)
    write_json(directory / "history.json", {"checkpoints": history, "nominal_iterations": cfg.steps})
    write_json(directory / "config.json", {"request": request, "control": asdict(cfg),
        "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(), "worker_pid": os.getpid(),
        "gpu_processes_before": before_processes, "gpu_processes_after": after_processes,
        "input_field_sha256": sha256(directory / "input_field.npz"),
        "target_sha256": sha256(directory / "target.npy")})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for axis, key, label in zip(axes, ("iteration", "elapsed_seconds"),
                               ("Attempted updates", "Elapsed fit seconds (shared workstation)")):
        axis.plot([r[key] for r in history], [r["psnr"] for r in history])
        axis.set(xlabel=label, ylabel="Raw PSNR (dB)")
    fig.tight_layout()
    fig.savefig(directory / "curves.png", dpi=130)
    plt.close(fig)
    display = raw.clamp(0, 1)
    lpips = LPIPS.distance(display, target)
    if lpips is None:
        raise RuntimeError("required LPIPS unavailable")
    mse = float((raw.double() - target.double()).square().mean())
    row = {**request, "status": "ok", "n_gaussians": field.n,
           "iterations_run": len(history) - 1, "selected_iteration": len(history) - 1,
           "psnr": -10 * math.log10(max(mse, 1e-12)),
           "initial_psnr": history[0]["psnr"], "ms_ssim": float(ms_ssim(display, target)),
           "lpips": lpips, "raw_mse": mse, "total_seconds": elapsed,
           "forward_evaluations": history[-1]["forward_evaluations"],
           "gradient_evaluations": history[-1]["gradient_evaluations"],
           "rejected_updates": sum(not r["accepted"] for r in history[1:]),
           "peak_allocated_bytes": peak, "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           "cold_render_max_abs": cold_error,
           "artifacts": {name: f'cells/{request["cell_id"]}/{name}' for name in (
               "input_field.npz", "field.npz", "target.npy", "reconstruction.npy", "target.png",
               "reconstruction.png", "error.png", "curves.png", "history.json", "config.json", "progress.jsonl")}}
    write_json(directory / "row.json", row)


def evaluate_results(root, rows):
    records = []
    for family in sorted({r["family"] for r in rows}):
        for method in ("diagonal", "block"):
            pairs = []
            for seed in PROTOCOL["seeds"]:
                group = [r for r in rows if r["family"] == family and r["seed"] == seed]
                arms = {r["method"]: r for r in group if r["status"] == "ok"}
                if set(arms) != set(PROTOCOL["arms"]):
                    pairs.append({"seed": seed, "complete": False, "reason": "missing/error arm"})
                    continue
                controls = [arms[name] for name in PROTOCOL["adam_multipliers"]]
                candidate = arms[method]
                baseline = max(controls, key=lambda r: (r["psnr"], r["method"]))
                directories = [root / "cells" / r["cell_id"] for r in arms.values()]
                same_inputs = all(len({sha256(d / name) for d in directories}) == 1
                                  for name in ("input_field.npz", "target.npy"))
                integrity = same_inputs and all(
                    r["n_gaussians"] == PROTOCOL["n_gaussians"]
                    and r["iterations_run"] == r["selected_iteration"] == PROTOCOL["config"]["steps"]
                    and r["cold_render_max_abs"] <= PROTOCOL["cold_parity_max_abs"]
                    for r in arms.values())
                common_time = min(r["total_seconds"] for r in arms.values())
                timed = {}
                for name, row in arms.items():
                    trace = json.loads((root / "cells" / row["cell_id"] / "history.json").read_text())["checkpoints"]
                    eligible = [h for h in trace if h["elapsed_seconds"] <= common_time]
                    timed[name] = eligible[-1]["psnr"] if eligible else None
                matched_gain = (timed[method] - max(timed[name] for name in PROTOCOL["adam_multipliers"])
                                if all(value is not None for value in timed.values()) else None)
                pairs.append({"seed": seed, "complete": True, "integrity": integrity,
                    "adam_envelope_method": baseline["method"],
                    "gain_db": candidate["psnr"] - baseline["psnr"],
                    "ms_ssim_difference": candidate["ms_ssim"] - baseline["ms_ssim"],
                    "lpips_difference": candidate["lpips"] - baseline["lpips"],
                    "common_seconds": common_time, "descriptive_matched_time_gain_db": matched_gain})
            complete = all(p["complete"] and p.get("integrity", False) for p in pairs)
            gains = [p["gain_db"] for p in pairs if p["complete"]]
            median = statistics.median(gains) if len(gains) == len(PROTOCOL["seeds"]) else None
            passes = complete and median >= PROTOCOL["minimum_median_gain_db"] and all(
                p["gain_db"] >= -PROTOCOL["max_seed_loss_db"]
                and p["ms_ssim_difference"] >= -PROTOCOL["max_ms_ssim_loss"]
                and p["lpips_difference"] <= PROTOCOL["max_lpips_increase"] for p in pairs)
            records.append({"family": family, "method": method, "pairs": pairs,
                            "median_terminal_gain_db": median, "passes_iteration_quality_gate": passes})
    decision = {"records": records, "scope": "procedural fixed-count 160-step development only",
                "speed_claim": False, "default_promotion": False}
    write_json(root / "decision.json", decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?")
    parser.add_argument("--worker")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--approved-protocol-digest")
    parser.add_argument("--print-protocol-digest", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    digest = protocol_digest(PROTOCOL, SOURCES)
    if args.print_protocol_digest:
        print(digest)
        return
    if not args.outdir or (not args.smoke and args.approved_protocol_digest != digest):
        parser.error("outdir and exact approved digest required for formal execution")
    workloads = [(f, s) for f in PROTOCOL["families"] for s in PROTOCOL["seeds"]]
    if args.smoke:
        workloads = [("translated", 77)]
    requests = []
    for family, seed in workloads:
        methods = PROTOCOL["arms"]
        shift = seed % len(methods)
        for method in methods[shift:] + methods[:shift]:
            requests.append({"family": family, "seed": seed, "method": method,
                             "cell_id": f"{family}_s{seed}_{method}", "smoke": args.smoke})
    bundle = ResearchBundle(args.outdir, task="HIER-035", protocol=PROTOCOL, digest=digest,
                            expected_cells=[r["cell_id"] for r in requests], diagnostic=args.smoke,
                            source_paths=SOURCES)
    rows = []
    for request in requests:
        directory = bundle.root / "cells" / request["cell_id"]
        directory.mkdir(parents=True)
        path = directory / "request.json"
        write_json(path, request)
        try:
            with (directory / "worker.log").open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                               check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=PROTOCOL["worker_timeout_seconds"])
            row = json.loads((directory / "row.json").read_text())
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            row = {**request, "status": "error", "error": str(exc),
                   "artifacts": {"worker.log": f'cells/{request["cell_id"]}/worker.log'}}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("cell_id", "status", "total_seconds") if k in row}), flush=True)
    evaluate_results(bundle.root, rows)
    for row in rows:
        row.setdefault("artifacts", {})["decision"] = "decision.json"
    bundle.finish(rows, title="HIER-035 — Fixed-count additive convergence controls",
                  interpretation="Procedural development: terminal 160-step quality versus a three-learning-rate Adam envelope. Complete fit times are descriptive on a shared workstation, not speed claims.")


if __name__ == "__main__":
    main()
