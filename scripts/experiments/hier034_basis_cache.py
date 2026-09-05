#!/usr/bin/env python3
"""HIER-034 fixed-geometry cache assay; see tasks/HIER-034-fixed-geometry-basis-cache.md.

Run from reviewed clean commit:
  python scripts/experiments/hier034_basis_cache.py OUT --base-bundle BASE \
    --approved-protocol-digest DIGEST
Wiring only: same command with --smoke (new diagnostic directory, synthetic seed 77).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import itertools
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from benchmarks.hier_research_report import (  # noqa: E402
    ResearchBundle, protocol_digest, save_rgb, sha256, write_json,
)
from structsplat.contraction_refinement import (  # noqa: E402
    CoefficientProjectionConfig, project_contracted_coefficients,
)

SOURCES = [
    "scripts/experiments/hier034_basis_cache.py", "benchmarks/hier_research_report.py",
    "src/structsplat/additive_basis.py", "src/structsplat/contraction_refinement.py",
    "src/structsplat/observation_field.py", "src/structsplat/pixel_contraction.py",
    "src/structsplat/progressive_residual_quadtree.py", "src/structsplat/gaussians.py",
    "src/structsplat/render.py", "src/structsplat/cuda_render.py", "src/structsplat/metrics.py",
    "scripts/check_report_bundle.py",
    "src/structsplat/cuda/render_ext.cpp", "src/structsplat/cuda/render_ext.cu",
]
PROTOCOL = {
    "task": "HIER-034", "version": 1, "device": "cuda", "gpu": "NVIDIA GeForce RTX 3050",
    "dtype": "float32", "threads": 1, "renderer": "cuda_additive", "render_chunk": 256,
    "families": ["smooth", "thin", "irregular"], "seeds": [0, 1, 2],
    "repetitions": 6, "backend_order": "all lexicographic permutations of off/scatter/csr",
    "backends": ["off", "scatter", "csr"], "synthetic_shape": [128, 128],
    "synthetic_grid": 12, "synthetic_center_jitter": 0.7,
    "scale_xy": {"smooth": [6.0, 6.0], "thin": [8.0, 1.8], "irregular": [6.0, 3.0]},
    "synthetic_color_init": "target sampled at rounded centers divided by 2; Gaussian noise std0.02",
    "synthetic_objective": "bounded analytic RGB patterns; irregular mask only selects loss pixels",
    "natural_family": "hier031", "natural_seed": 0, "natural_role": "exposed development",
    "natural_color_multiplier": 0.97, "natural_shape": [1038, 1200], "natural_n": 7000,
    "natural_files": {
        "artifacts/deep_only_terminal_closure_n7000/field.gaussian.npz":
            "a0a080ccbd255ce51f11489cd504956a1c5181a495bbca2b4bf74ecb0995c1db",
        "input/source.png": "612a6dd3249304e47fc9b96936601c51b8924a54f9a594bdfe17d6c1612bf14f",
        "input/mask.png": "be6bac5acb2cd3d41846c317432ccbc2015dddd0a347b46f47fc321e9e57b078",
    },
    "solver": asdict(CoefficientProjectionConfig()),
    "scope": "complete fixed-geometry projection including build, PCG, checkpoints and replay",
    "worker_timeout_seconds": 600,
    "warmup": "synthetic seed77 projections for off/scatter/csr in that order, 2 iterations each",
    "metrics": ["raw masked PSNR/SSE", "display black-matted MS-SSIM/LPIPS", "paired wall time",
                "cache build/retained bytes", "peak allocated/reserved VRAM", "worker peak RSS",
                "attempted/selected iterations", "forward/transpose calls", "all checkpoints"],
    "parity_max_abs": 2e-4, "sse_relative_tolerance": 1e-4, "sse_absolute_floor": 1e-6,
    "required_median_speedup": 1.1, "minimum_iterations": 2,
    "checkpoint_violation_tolerance": 1e-6,
    "decision": "workload-specific projection acceleration only; every repeat must pass parity",
    "missing_policy": "retain error row; incomplete field has no positive verdict",
    "forbidden": ["default promotion", "threshold retuning", "in-place rerun", "sealed images"],
}


def fixture(family, seed, base_bundle=None):
    import torch
    from structsplat.gaussians import GaussianField
    from structsplat.observation_field import CanvasCropTransform, adapt_factorized_additive_gaussian_field

    if family == "hier031":
        root = Path(base_bundle)
        for name, digest in PROTOCOL["natural_files"].items():
            if sha256(root / name) != digest:
                raise ValueError(f"immutable natural input hash mismatch: {name}")
        with np.load(root / "artifacts/deep_only_terminal_closure_n7000/field.gaussian.npz",
                     allow_pickle=False) as data:
            field = GaussianField(*(torch.from_numpy(data[key].copy()) for key in
                                    ("means", "log_scales", "rotations", "colors")))
        field.colors *= PROTOCOL["natural_color_multiplier"]
        target = np.asarray(Image.open(root / "input/source.png").convert("RGB"), np.float32) / 255
        mask = np.asarray(Image.open(root / "input/mask.png").convert("L")) > 0
        if list(mask.shape) != PROTOCOL["natural_shape"] or field.n != PROTOCOL["natural_n"]:
            raise ValueError("natural dimensions or count differ from frozen protocol")
    else:
        rng = np.random.default_rng(seed)
        height, width = PROTOCOL["synthetic_shape"]
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        target = np.stack((0.45 + 0.3 * np.sin(xx / 17) * np.cos(yy / 21),
                           0.45 + 0.25 * np.sin((xx + yy) / (4 if family == "thin" else 23)),
                           0.4 + 0.25 * np.cos(xx / 31 - yy / 19)), -1).astype(np.float32)
        mask = np.ones((height, width), bool)
        if family == "irregular":
            mask = ((xx - 64)**2 / 58**2 + (yy - 64)**2 / 53**2 < 1)
            mask &= ~((xx > 61) & (xx < 68) & (yy < 52))
        grid = np.linspace(5, 122, PROTOCOL["synthetic_grid"], dtype=np.float32)
        gy, gx = np.meshgrid(grid, grid, indexing="ij")
        means = np.stack((gx.ravel(), gy.ravel()), 1)
        means += rng.normal(0, PROTOCOL["synthetic_center_jitter"], means.shape).astype(np.float32)
        ix, iy = np.rint(means).astype(int).T
        colors = target[iy, ix] / 2 + rng.normal(0, 0.02, (len(means), 3)).astype(np.float32)
        scales = np.tile(PROTOCOL["scale_xy"][family], (len(means), 1)).astype(np.float32)
        angles = rng.uniform(-0.8, 0.8, len(means)).astype(np.float32)
        field = GaussianField.from_numpy(means, scales, angles, colors)
    h, w = mask.shape
    observation = adapt_factorized_additive_gaussian_field(
        field, canvas_crop=CanvasCropTransform(w, h, 0, 0, w, h),
        coefficient_domain="signed", sigma_cutoff=3, support_fade_alpha=1,
    ).require_pixel_exact()
    return observation, target, mask


def worker(request_path):
    import torch
    from structsplat.metrics import LPIPS, ms_ssim
    from structsplat.observation_field import save_observation_field, load_observation_field
    from structsplat.pixel_contraction import render_observation_field

    request = json.loads(Path(request_path).read_text())
    directory = Path(request_path).parent
    torch.set_num_threads(PROTOCOL["threads"])
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != PROTOCOL["gpu"]:
        raise RuntimeError("frozen GPU unavailable")
    warm_field, warm_target, warm_mask = fixture("smooth", 77)
    for backend in PROTOCOL["backends"]:
        project_contracted_coefficients(warm_field, warm_target, warm_mask,
            np.ones(warm_field.n, bool),
            config=CoefficientProjectionConfig(max_iterations=2, basis_cache=backend),
            device="cuda", renderer="cuda_additive", render_chunk=256)
    field, target, mask = fixture(request["family"], request["seed"], request.get("base_bundle"))
    cfg = CoefficientProjectionConfig(**PROTOCOL["solver"])
    cfg = replace(cfg, basis_cache=request["backend"],
                  max_iterations=3 if request.get("smoke") else cfg.max_iterations)
    save_observation_field(field, directory / "input_field.npz")
    np.save(directory / "target.npy", target)
    np.save(directory / "mask.npy", mask)
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = project_contracted_coefficients(field, target, mask, np.ones(field.n, bool),
        config=cfg, device="cuda", renderer="cuda_additive", render_chunk=256)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    save_observation_field(result.field, directory / "field.npz")
    cold = load_observation_field(directory / "field.npz")
    cold_render = render_observation_field(cold, device="cuda", renderer="cuda_additive", render_chunk=256)
    cold_difference = float(np.max(np.abs(cold_render - result.reconstruction_raw)))
    if not np.array_equal(cold.rgb_coeff, result.field.rgb_coeff):
        raise RuntimeError("decoded coefficients differ from saved state")
    raw = result.reconstruction_raw
    np.save(directory / "reconstruction.npy", raw)
    save_rgb(directory / "target.png", np.where(mask[..., None], target, 0))
    save_rgb(directory / "reconstruction.png", np.where(mask[..., None], raw, 0))
    save_rgb(directory / "error.png", np.where(mask[..., None], np.abs(raw - target) * 4, 0))
    pred = torch.from_numpy(np.where(mask[..., None], np.clip(raw, 0, 1), 0)).cuda()
    truth = torch.from_numpy(np.where(mask[..., None], target, 0)).cuda()
    masked_sse = float(np.square(raw[mask].astype(np.float64) - target[mask]).sum())
    checkpoints = result.checkpoint_records()
    history = {"checkpoints": checkpoints, "nominal_iterations": cfg.max_iterations}
    write_json(directory / "history.json", history)
    write_json(directory / "config.json", {"solver": asdict(cfg), "request": request,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "numpy": np.__version__,
               "target_sha256": sha256(directory / "target.npy"),
               "input_field_sha256": sha256(directory / "input_field.npz")})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    steps = [c["iteration"] for c in checkpoints]
    psnrs = [-10 * math.log10(max(c["raw_sse"] / (mask.sum() * 3), 1e-12)) for c in checkpoints]
    axes[0].plot(steps, psnrs)
    axes[0].set(xlabel="Attempted PCG iterations", ylabel="Raw masked PSNR (dB)")
    axes[1].plot([c["elapsed_seconds"] for c in checkpoints], psnrs)
    axes[1].set(xlabel="Elapsed projection seconds", ylabel="Raw masked PSNR (dB)")
    fig.tight_layout()
    fig.savefig(directory / "curves.png", dpi=130)
    plt.close(fig)
    row = {"cell_id": request["cell_id"], "status": "ok", "family": request["family"],
           "seed": request["seed"], "repeat": request["repeat"], "backend": request["backend"],
           "n_gaussians": field.n, "iterations_run": len(checkpoints) - 1,
           "selected_iteration": result.selected_iteration, "forward_calls": result.forward_applications,
           "transpose_calls": result.transpose_applications, "raw_sse": masked_sse,
           "psnr": -10 * math.log10(max(masked_sse / (mask.sum() * 3), 1e-12)),
           "ms_ssim": float(ms_ssim(pred, truth)), "lpips": LPIPS.distance(pred, truth),
           "total_seconds": elapsed, "cache_bytes": result.basis_cache_bytes,
           "cache_build_seconds": result.basis_cache_build_seconds, "cache_nnz": result.basis_cache_nnz,
           "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
           "baseline_allocated_bytes": baseline_allocated, "worker_peak_rss_bytes": peak_rss,
           "baseline_reserved_bytes": baseline_reserved,
           "cold_render_max_abs": cold_difference,
           "maintained_parity_max_abs": result.maintained_render_parity_max_abs,
           "adjoint_relative_error": result.adjoint_relative_error,
           "selected_violation": checkpoints[result.selected_iteration]["display_normalized_violation"],
           "artifacts": {name: f'cells/{request["cell_id"]}/{name}' for name in (
               "target.png", "reconstruction.png", "error.png", "curves.png", "input_field.npz",
               "field.npz", "history.json", "config.json", "target.npy", "mask.npy", "reconstruction.npy")}}
    if row["lpips"] is None:
        raise RuntimeError("required LPIPS unavailable")
    write_json(directory / "row.json", row)


def checkpoint_agreement(base, candidate):
    """Expose every transaction decision, not just the winning iteration."""
    disagreements = []
    if len(base) != len(candidate):
        disagreements.append({"kind": "trace_length", "off": len(base), "candidate": len(candidate)})
    for a, b in zip(base, candidate):
        for key in ("iteration", "selectable", "transaction_safe", "bounded"):
            if a[key] != b[key]:
                disagreements.append({"iteration": a["iteration"], "key": key,
                                      "off": a[key], "candidate": b[key]})
        delta = abs(a["display_normalized_violation"] - b["display_normalized_violation"])
        if delta > PROTOCOL["checkpoint_violation_tolerance"]:
            disagreements.append({"iteration": a["iteration"], "key": "violation", "difference": delta})
    return disagreements


def speed_gate(pairs, *, timing_eligible=True):
    ratios = [p["paired_speedup"] for p in pairs if "paired_speedup" in p]
    median = statistics.median(ratios) if ratios else None
    eligible = (len(pairs) == len(ratios) == PROTOCOL["repetitions"]
                and all(p["pass"] for p in pairs))
    return {"median_paired_speedup": median,
            "speedup_min": min(ratios) if ratios else None,
            "speedup_max": max(ratios) if ratios else None,
            "integrity_eligible": eligible,
            "passes_interchangeability_gate": eligible,
            "timing_eligible": timing_eligible,
            "descriptive_ratio_threshold_passed": eligible and median >= PROTOCOL["required_median_speedup"],
            "passes_speed_gate": timing_eligible and eligible and median >= PROTOCOL["required_median_speedup"]}


def analyze_bundle(root, rows, *, timing_eligible=True):
    """Recompute the preregistered decisions from saved states and raw float images."""
    from structsplat.observation_field import load_observation_field

    records = []
    for family, seed in sorted({(r["family"], r["seed"]) for r in rows}):
        group = [r for r in rows if r["family"] == family and r["seed"] == seed]
        for backend in ("scatter", "csr"):
            pairs = []
            for repeat in range(PROTOCOL["repetitions"]):
                base = next((r for r in group if r["backend"] == "off" and r["repeat"] == repeat), None)
                candidate = next((r for r in group if r["backend"] == backend and r["repeat"] == repeat), None)
                if not base or not candidate or base["status"] != "ok" or candidate["status"] != "ok":
                    pairs.append({"repeat": repeat, "pass": False, "reason": "missing/error pair"})
                    continue
                base_dir, cand_dir = (root / "cells" / r["cell_id"] for r in (base, candidate))
                off_field = load_observation_field(base_dir / "field.npz")
                field = load_observation_field(cand_dir / "field.npz")
                incoming = load_observation_field(cand_dir / "input_field.npz")
                mask = np.load(base_dir / "mask.npy", allow_pickle=False)
                target = np.load(base_dir / "target.npy", allow_pickle=False)
                off_render = np.load(base_dir / "reconstruction.npy", allow_pickle=False)
                render = np.load(cand_dir / "reconstruction.npy", allow_pickle=False)
                off_sse = float(np.square(off_render[mask].astype(np.float64) - target[mask]).sum())
                sse = float(np.square(render[mask].astype(np.float64) - target[mask]).sum())
                maximum = float(np.max(np.abs(render[mask] - off_render[mask])))
                coeff_diff = (float(np.max(np.abs(field.rgb_coeff - off_field.rgb_coeff)))
                              if field.n == off_field.n else None)
                exact_geometry = field.n == incoming.n == off_field.n and all(
                    np.array_equal(getattr(field, name), getattr(incoming, name))
                    and np.array_equal(getattr(off_field, name), getattr(incoming, name))
                    for name in ("means_xy", "log_scales_xy", "rotations_rad")
                )
                same_inputs = all(sha256(base_dir / name) == sha256(cand_dir / name)
                                  for name in ("input_field.npz", "mask.npy", "target.npy"))
                decisions = checkpoint_agreement(
                    json.loads((base_dir / "history.json").read_text())["checkpoints"],
                    json.loads((cand_dir / "history.json").read_text())["checkpoints"])
                clauses = {
                    "same_inputs": same_inputs, "exact_count_geometry": exact_geometry,
                    "coefficient_bound": bool(np.max(np.abs(field.rgb_coeff)) <= 16.0),
                    "cold_and_maintained_parity": all(
                        r[key] <= PROTOCOL["parity_max_abs"] for r in (base, candidate)
                        for key in ("cold_render_max_abs", "maintained_parity_max_abs")),
                    "paired_pixel_parity": maximum <= PROTOCOL["parity_max_abs"],
                    "paired_sse_parity": abs(sse - off_sse) <= PROTOCOL["sse_relative_tolerance"]
                        * max(off_sse, PROTOCOL["sse_absolute_floor"]),
                    "same_selected_iteration": candidate["selected_iteration"] == base["selected_iteration"],
                    "same_checkpoint_decisions": not decisions,
                    "nontrivial_work": min(candidate["iterations_run"], base["iterations_run"])
                        >= PROTOCOL["minimum_iterations"],
                    "finite_positive_time": min(base["total_seconds"], candidate["total_seconds"]) > 0,
                    "retained_memory_bound": candidate["cache_bytes"] <= PROTOCOL["solver"]["basis_cache_max_bytes"],
                }
                pairs.append({"repeat": repeat, "pass": all(clauses.values()), "clauses": clauses,
                              "pixel_max_abs": maximum, "coefficient_max_abs": coeff_diff,
                              "raw_sse_difference": sse - off_sse,
                              "checkpoint_disagreements": decisions,
                              "paired_speedup": base["total_seconds"] / candidate["total_seconds"]})
            records.append({"family": family, "seed": seed, "backend": backend, "pairs": pairs,
                            **speed_gate(pairs, timing_eligible=timing_eligible)})
    decision = {"scope": "workload-specific exposed/synthetic projection only",
                "records": records, "default_promotion": False,
                "timing_eligible": timing_eligible,
                "performance_disposition": ("workload-specific timing assay" if timing_eligible else
                                            "inconclusive—shared-GPU contention")}
    write_json(root / "decision.json", decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?")
    parser.add_argument("--base-bundle")
    parser.add_argument("--worker")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-protocol-digest", action="store_true")
    parser.add_argument("--approved-protocol-digest")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    digest = protocol_digest(PROTOCOL, SOURCES)
    if args.print_protocol_digest:
        print(digest)
        return
    if not args.outdir:
        parser.error("outdir is required")
    if not args.smoke and (args.approved_protocol_digest != digest or not args.base_bundle):
        parser.error("formal run requires exact approved protocol digest and base bundle")
    workloads = [(family, seed) for family in PROTOCOL["families"] for seed in PROTOCOL["seeds"]]
    workloads += [("hier031", 0)]
    orders = list(itertools.permutations(PROTOCOL["backends"]))
    if args.smoke:
        workloads, orders = [("smooth", 77)], [tuple(PROTOCOL["backends"])]
    requests = []
    for family, seed in workloads:
        for repeat, order in enumerate(orders):
            for backend in order:
                cell_id = f"{family}_s{seed}_r{repeat}_{backend}"
                requests.append({"family": family, "seed": seed, "repeat": repeat,
                    "backend": backend, "cell_id": cell_id, "base_bundle": args.base_bundle,
                    "smoke": args.smoke})
    bundle = ResearchBundle(args.outdir, task="HIER-034", protocol=PROTOCOL, digest=digest,
                            expected_cells=[r["cell_id"] for r in requests], diagnostic=args.smoke,
                            source_paths=SOURCES)
    rows = []
    for request in requests:
        directory = bundle.root / "cells" / request["cell_id"]
        directory.mkdir(parents=True)
        request_path = directory / "request.json"
        write_json(request_path, request)
        try:
            with (directory / "worker.log").open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)],
                               check=True, stdout=log, stderr=subprocess.STDOUT,
                               timeout=PROTOCOL["worker_timeout_seconds"])
            row = json.loads((directory / "row.json").read_text())
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            row = {**request, "status": "error", "error": str(exc),
                   "artifacts": {"worker.log": f'cells/{request["cell_id"]}/worker.log'}}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("cell_id", "status", "total_seconds") if k in row}), flush=True)
    analyze_bundle(bundle.root, rows)
    for row in rows:
        row.setdefault("artifacts", {})["decision"] = "decision.json"
    bundle.finish(rows, title="HIER-034 — Fixed-geometry projection cache",
                  interpretation="Exposed/synthetic fixed-geometry projection assay. Includes cache build and checkpoint work. No whole-pipeline speed or default claim.")


if __name__ == "__main__":
    main()
