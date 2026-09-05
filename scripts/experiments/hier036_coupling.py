#!/usr/bin/env python3
"""HIER-036 bounded cross-Gaussian coupling/cap factorial.

Formal: python scripts/experiments/hier036_coupling.py OUT --approved-protocol-digest DIGEST
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

from benchmarks.hier_coupling_oracle import (  # noqa: E402
    MODES, MAX_JACOBIAN_BYTES, MAX_PARAMETERS, fit_coupling,
)
from scripts.experiments.hier035_convergence import fixture  # noqa: E402

SOURCES = [
    "scripts/experiments/hier036_coupling.py", "scripts/experiments/hier035_convergence.py",
    "benchmarks/hier_coupling_oracle.py", "benchmarks/hier_additive_controls.py",
    "benchmarks/hier_research_report.py", "scripts/check_report_bundle.py",
    "src/structsplat/pixel_gradient.py", "src/structsplat/gaussians.py",
    "src/structsplat/render.py", "src/structsplat/cuda_render.py",
    "src/structsplat/cuda/render_ext.cpp", "src/structsplat/cuda/render_ext.cu",
    "src/structsplat/metrics.py",
]
PROTOCOL = {
    "task": "HIER-036", "version": 1, "device": "cuda", "gpu": "NVIDIA GeForce RTX 3050",
    "dtype": "float32", "threads": 1, "renderer": "cuda_additive", "sigma_cutoff": 3,
    "support_fade_alpha": 1, "shape": [64, 64], "n_gaussians": 16,
    "families": ["overlap", "texture"], "seeds": [0, 1, 2, 3, 4, 5],
    "strata": {"exposed": [0, 1, 2], "additional_procedural": [3, 4, 5]},
    "arms": ["adam_0p3", "adam_1", "adam_3", *MODES],
    "order": "rotate seven-arm order by seed; separate warmed process per cell",
    "config": asdict(ControlConfig(arm="block")),
    "adam_multipliers": {"adam_0p3": 0.3, "adam_1": 1.0, "adam_3": 3.0},
    "adam_parameter_lrs": [0.1, 0.03, 0.03, 0.03],
    "objective": "raw 0.5 mean squared RGB error, whole image, no mask/display clamp in fitting",
    "terminal": "exact160 attempted updates; terminal state, no best selection",
    "scoring": "raw MSE; PSNR floor1e-12 and uncapped counterpart; display-clamped MS-SSIM/LPIPS",
    "warmup": "translated77,2updates each of all7arms",
    "precision": {"float32_matmul_precision": "highest", "allow_tf32": False},
    "dense_limits": {"max_jacobian_bytes": MAX_JACOBIAN_BYTES, "max_parameters": MAX_PARAMETERS},
    "dense_cap_scope": "retained J only; CUDA peak includes Gram/solve temporaries and live worker tensors",
    "system": "same dense J,g=JTr/numel,H=JTJ/numel; only cross-row mask and cap scope differ",
    "ridge": "per-row .01*max scaled diagonal, diagonal floor1e-12; identical across full/block",
    "bridge_cpu": {"rtol": 1e-6, "atol": 1e-8},
    "bridge_cuda_system": {"rtol": 1e-5, "atol": 1e-7},
    "bridge_cuda_step": {"rtol": 1e-3, "atol": 1e-3},
    "worker_timeout_seconds": 600,
    "primary": "full_shared vs block_shared, per family and exposure stratum",
    "secondary": "full_row vs block_row separately; full arms vs strongest Adam separately",
    "minimum_coupling_gain_db": 1.0, "minimum_adam_gain_db": 0.5,
    "max_seed_loss_db": 0.1, "max_ms_ssim_loss": 0.005, "max_lpips_increase": 0.01,
    "adam_tie": "maximum (terminal PSNR, lexical method name)",
    "cold_parity_max_abs": 2e-5,
    "timing": "complete fit and all inner work, descriptive only on shared GPU",
    "data_role": "procedural mechanism evidence; exposed and additional conditions separate",
    "positive_gate": "all84cells complete and integrity-valid before any stratum passes",
    "missing_policy": "retain all failures including solve errors; no fallback or selective rerun",
    "forbidden": ["default promotion", "speed claim", "natural/held-out claim", "threshold rescue", "in-place repair"],
}


def gpu_process_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True).strip()


def resolved_config(method, smoke=False):
    cfg = ControlConfig(**PROTOCOL["config"])
    return replace(cfg, arm="adam" if method.startswith("adam") else "block",
                   adam_multiplier=PROTOCOL["adam_multipliers"].get(method, 1.0),
                   steps=3 if smoke else cfg.steps)


def run_fit(initial, target, cfg, method, callback=None):
    if method.startswith("adam"):
        return fit_control(initial, target, cfg, renderer=PROTOCOL["renderer"], callback=callback)
    return fit_coupling(initial, target, cfg, mode=method, renderer=PROTOCOL["renderer"], callback=callback)


def worker(request_path):
    import torch
    from structsplat.gaussians import GaussianField
    from structsplat.metrics import LPIPS, ms_ssim

    torch.set_num_threads(PROTOCOL["threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    request = json.loads(Path(request_path).read_text())
    directory = Path(request_path).parent
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != PROTOCOL["gpu"]:
        raise RuntimeError("frozen GPU unavailable")
    before_processes = gpu_process_snapshot()
    warm, warm_target = fixture("translated", 77)
    warm, warm_target = unpack(pack(warm).cuda()), warm_target.cuda()
    warm_forwards = 0
    for method in PROTOCOL["arms"]:
        _, _, trace, _ = run_fit(warm, warm_target, replace(resolved_config(method), steps=2), method)
        warm_forwards += trace[-1]["forward_evaluations"]
    initial, target = fixture(request["family"], request["seed"])
    initial.save(directory / "input_field.npz")
    np.save(directory / "target.npy", target.numpy())
    initial, target = unpack(pack(initial).cuda()), target.cuda()
    cfg = resolved_config(request["method"], request["smoke"])
    initial_raw = additive_render(initial, *PROTOCOL["shape"], renderer=PROTOCOL["renderer"])
    np.save(directory / "initial_reconstruction.npy", initial_raw.cpu().numpy())
    torch.cuda.reset_peak_memory_stats()
    with (directory / "progress.jsonl").open("w") as progress:
        def callback(row):
            progress.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            progress.flush()
        field, raw, history, elapsed = run_fit(initial, target, cfg, request["method"], callback)
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
        "dense_mode": request["method"] if request["method"] in MODES else None,
        "dense_limits": PROTOCOL["dense_limits"], "precision": PROTOCOL["precision"],
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
           "psnr_uncapped": -10 * math.log10(mse) if mse > 0 else None,
           "psnr_ceiling_applied": mse < 1e-12,
           "stratum": next((k for k, v in PROTOCOL["strata"].items() if request["seed"] in v), "diagnostic"),
           "initial_psnr": history[0]["psnr"], "ms_ssim": float(ms_ssim(display, target)),
           "lpips": lpips, "raw_mse": mse, "total_seconds": elapsed,
           "forward_evaluations": history[-1]["forward_evaluations"],
           "gradient_evaluations": history[-1]["gradient_evaluations"],
           "rejected_updates": sum(not r["accepted"] for r in history[1:]),
           "jacobian_constructions": cfg.steps if request["method"] in MODES else 0,
           "linear_solves": cfg.steps if request["method"] in MODES else 0,
           "retained_jacobian_bytes": target.numel() * field.n * 8 * target.element_size() if request["method"] in MODES else 0,
           "retained_gram_bytes": (field.n * 8) ** 2 * target.element_size() if request["method"] in MODES else 0,
           "warmup_forward_evaluations": warm_forwards, "warmup_gradient_evaluations": 14,
           "warmup_jacobian_constructions": 8, "warmup_linear_solves": 8,
           "fixture_render_evaluations": 4, "cold_forward_evaluations": 1,
           "initial_diagnostic_forward_evaluations": 1,
           "worker_forward_evaluations": warm_forwards + 6 + history[-1]["forward_evaluations"],
           "peak_allocated_bytes": peak, "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
           "cold_render_max_abs": cold_error,
           "artifacts": {name: f'cells/{request["cell_id"]}/{name}' for name in (
               "input_field.npz", "field.npz", "target.npy", "reconstruction.npy", "initial_reconstruction.npy", "target.png",
               "reconstruction.png", "error.png", "curves.png", "history.json", "config.json", "progress.jsonl")}}
    write_json(directory / "row.json", row)


def evaluate_results(root, rows):
    from benchmarks.hier_research_report import _validate_coupling_cell

    expected = {f"{f}_s{s}_{m}" for f in PROTOCOL["families"] for s in PROTOCOL["seeds"]
                for m in PROTOCOL["arms"]}
    ids = [r["cell_id"] for r in rows]
    whole_valid = len(ids) == len(expected) and set(ids) == expected
    problems = []
    canonical = json.loads(json.dumps(PROTOCOL))
    for row in rows:
        if row["status"] != "ok":
            problems.append(f"failed cell: {row['cell_id']}")
            continue
        try:
            _validate_coupling_cell(root, row, canonical, row["smoke"], problems)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"invalid cell {row['cell_id']}: {exc}")
    whole_valid = whole_valid and not problems
    by_key = {(r["family"], r["seed"], r["method"]): r for r in rows if r["status"] == "ok"}
    records = []
    for family in PROTOCOL["families"]:
        for stratum, seeds in PROTOCOL["strata"].items():
            for cap in ("shared", "row"):
                for comparator in ("coupling", "adam_envelope"):
                    pairs = []
                    for seed in seeds:
                        group = {m: by_key.get((family, seed, m)) for m in PROTOCOL["arms"]}
                        if any(r is None for r in group.values()):
                            pairs.append({"seed": seed, "complete": False})
                            continue
                        candidate = group[f"full_{cap}"]
                        baseline = (group[f"block_{cap}"] if comparator == "coupling"
                                    else max((group[m] for m in PROTOCOL["adam_multipliers"]),
                                             key=lambda r: (r["psnr"], r["method"])))
                        directories = [root / "cells" / r["cell_id"] for r in group.values()]
                        same_inputs = all(len({sha256(d / name) for d in directories}) == 1
                                          for name in ("input_field.npz", "target.npy"))
                        if not same_inputs:
                            whole_valid = False
                        pairs.append({"seed": seed, "complete": True, "same_inputs": same_inputs,
                            "baseline_method": baseline["method"],
                            "gain_db": candidate["psnr"] - baseline["psnr"],
                            "candidate_raw_mse": candidate["raw_mse"], "baseline_raw_mse": baseline["raw_mse"],
                            "candidate_ceiling": candidate["psnr_ceiling_applied"],
                            "baseline_ceiling": baseline["psnr_ceiling_applied"],
                            "ms_ssim_difference": candidate["ms_ssim"] - baseline["ms_ssim"],
                            "lpips_difference": candidate["lpips"] - baseline["lpips"]})
                    complete = all(p["complete"] and p["same_inputs"] for p in pairs)
                    median = statistics.median(p["gain_db"] for p in pairs) if complete else None
                    threshold = PROTOCOL["minimum_coupling_gain_db" if comparator == "coupling" else "minimum_adam_gain_db"]
                    local_pass = complete and median >= threshold and all(
                        p["gain_db"] >= -PROTOCOL["max_seed_loss_db"]
                        and p["ms_ssim_difference"] >= -PROTOCOL["max_ms_ssim_loss"]
                        and p["lpips_difference"] <= PROTOCOL["max_lpips_increase"] for p in pairs)
                    records.append({"family": family, "stratum": stratum, "cap": cap, "comparator": comparator,
                                    "pairs": pairs, "median_gain_db": median, "local_predicates_pass": local_pass})
    for record in records:
        record["passes_gate"] = whole_valid and record["local_predicates_pass"]
    decision = {"whole_matrix_valid": whole_valid, "integrity_problems": problems, "records": records,
                "scope": "bounded procedural dense GN coupling/cap factorial, not a scalable optimizer",
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
    bundle = ResearchBundle(args.outdir, task="HIER-036", protocol=PROTOCOL, digest=digest,
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
    bundle.finish(rows, title="HIER-036 — Dense cross-Gaussian coupling oracle",
                  interpretation="Bounded dense GN factorial: cross-Gaussian coupling and row/shared trust caps, with separate strongest-Adam controls. Exposed and additional procedural conditions are separate; near-ceiling gains are numerical polish. Shared-GPU times are descriptive, not speed claims.")


if __name__ == "__main__":
    main()
