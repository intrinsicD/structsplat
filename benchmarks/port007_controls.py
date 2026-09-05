"""Frozen, bounded PORT-007 quality-evaluation reuse assay (not a generic benchmark)."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from structsplat.pipeline import PipelineConfig


ARMS = ("legacy_a", "legacy_b", "coverage", "tail", "both")
BACKENDS = {
    "legacy_a": ("reference", "reference"),
    "legacy_b": ("reference", "reference"),
    "coverage": ("renderer", "reference"),
    "tail": ("reference", "shared"),
    "both": ("renderer", "shared"),
}
IMAGE_HASHES = {
    9: "35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
    25: "d8f12a26d8803701cabac80494b080f998e5ed9bafaf61a2825ce6212c85487a",
    30: "0444b10826d376ad9075805061405f6071a62b80eda29c5f284ed77b093d5b1d",
    34: "2c46871034fa901ae795a8bb916ba7f2f728507cab9e511cced0986bd083d193",
}
ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([
    "scripts/experiments/port007_quality_reuse.py", "benchmarks/port007_controls.py",
    "scripts/experiments/fit050_color_ray.py",
    "benchmarks/fit050_controls.py", "benchmarks/hier_research_report.py",
    "scripts/check_report_bundle.py", "src/structsplat/cuda/render_ext.cpp",
    "src/structsplat/cuda/render_ext.cu",
] + [path.relative_to(ROOT).as_posix() for path in (ROOT / "src" / "structsplat").rglob("*.py")])
PROTOCOL = {
    "task": "PORT-007", "version": 1, "device": "cuda", "dtype": "float32",
    "artifact_metric_rtol": 5e-6, "artifact_metric_atol": 1e-8,
    "artifact_metric_scope": "CPU consistency check of saved float32 arrays, never a substitute for recorded CUDA decisions",
    "gpu": "NVIDIA GeForce RTX 3050", "torch": "2.9.0+cu128", "threads": 1,
    "images": list(IMAGE_HASHES), "jpeg_sha256": {str(k): v for k, v in IMAGE_HASHES.items()},
    "max_side": 512, "resize": "Pillow RGB LANCZOS, round scaled width and height, no upscale",
    "data_role": "already exposed COCO development fixtures; no held-out or confirmation claim",
    "arms": list(ARMS), "backends": {k: list(v) for k, v in BACKENDS.items()},
    "primary_arm": "both", "control": "legacy_a", "repeat_control": "legacy_b",
    "secondary_arms": ["coverage", "tail"],
    "same_state": {
        "seeds": [0, 1], "states": ["initial", "terminal"], "n_gaussians": 2000,
        "parent_steps": 750, "rounds": 10, "warmups_per_arm_and_state": 2,
        "parent_source": "required completed clean FIT-050 bundle at the same source commit; copy and hash initial/terminal fields, target, configs, history and optimizer state; no selection/refitting",
        "parent_config": "benchmarks.fit050_controls.parent_configs(seed), exact full dictionaries in each parent config",
        "order": "five cyclic rotations of arms, then reversals of those five rotations",
        "timing": "one complete uninstrumented evaluate_quality per round/arm; CUDA synchronize immediately before and after; serialization, parity replay and perceptual metrics excluded",
        "parity": "all ten measured RGB images and quality vectors retained; separate untimed instrumented replay captures same-call RGB/denominator, hole mask and fallback count; coverage parity is replay-to-replay, not a bitwise claim about another atomic call",
        "decisions": "each round: legacy_a state to each arm same state (null); initial legacy_a to terminal arm; terminal legacy_a to initial arm; compare exact boolean/reasons to legacy_a counterpart",
        "aggregation": "median ten paired per-round baseline/candidate ratios within state/seed; median four state/seed ratios within image; median across four images",
        "max_rgb_absolute_error": 2e-5, "minimum_speed_ratio": 1.1,
        "minimum_per_image_speed_ratio": 1 / 1.05,
        "maximum_call_time_cv": 0.25,
        "aa_runtime_ratio_interval": [0.9, 1.1],
        "aa_timing_gate": "within every image/state/seed, median ten paired legacy_a/legacy_b call-time ratios must lie in [0.9,1.1]; all sixteen image/state/seed pairs must pass; per-image AA medians are descriptive only",
        "correctness_gate": "all cells finite; all measured and replay RGB errors <=2e-5; exact replay hole masks and exact all decision/reason vectors; scalar quality deltas all reported without dropping fields",
    },
    "pipeline": {
        "images": [9, 25], "seeds": [0, 1, 2],
        "mask": "COCO9 full frame; COCO25 ellipse ((x-(W-1)/2)/(0.42W))^2+((y-(H-1)/2)/(0.42H))^2<=1, fixed pixel centers; not semantic segmentation",
        "config": asdict(PipelineConfig(capacity=1000, step_scale=0.025, block_steps=25,
                                         pareto_checkpoint_every=25, device="cuda", renderer="cuda")),
        "order": "rotate five arms by (image index*3+seed) modulo five; new warmed process per cell",
        "timing": "whole run_pipeline including initialization, all scheduled stages, final render and CPU field-snapshot observer; synchronized; reporting/perceptual rescoring afterward excluded",
        "observation": "every native observer event: copy selected field to CPU and record complete native event including rejected candidates; offline score selected snapshots; rejected trial fields are not exposed by native API",
        "primary_runtime": "instrumented_total_seconds (observer cost included)",
        "exploratory_runtime": "total minus measured observer seconds is not a causal uninstrumented runtime estimate",
        "minimum_speed_ratio": 1.05, "minimum_per_image_speed_ratio": 1 / 1.05,
        "aa_runtime_ratio_interval": [0.9, 1.1],
        "maximum_psnr_loss_db": 0.05, "maximum_ms_ssim_loss": 0.001,
        "maximum_lpips_increase": 0.002,
        "max_reference_rgb_absolute_error": 2e-5,
        "finite_gate": "baseline, A/A and candidate require finite PSNR/MS-SSIM/LPIPS, all finite native quality scalars, quality.finite=True, and final reference RGB error in [0,2e-5]",
        "trajectory_gate": "exact event/phase/reason/accepted/count/attempted/selected-index discrete projection for candidate and A/A; no final-count or selected-step mismatch; selected_iteration is the global attempted index of the last accepted native event",
        "aggregation": "median paired baseline/candidate runtime across three seeds per image, then median across the two images; all six quality guardrails must hold",
        "scope": "instrumented bounded complete pipeline, not the default 11000-row/full-step regime",
    },
    "scoring": "raw float64-accumulated PSNR/MSE; display-clamped SSIM/MS-SSIM/LPIPS, raw MAE and complete native safe-quality vector",
    "resource_scope": "same-state call GPU allocated peaks; pipeline complete-call GPU allocated peaks and worker process peak RSS; parent and reporting costs separate",
    "gpu_monitor": "nvidia-smi compute-process and utilization snapshots before/after each group and each measured pipeline; foreign compute PID or unreadable monitor makes timing ineligible; point samples do not prove continuous exclusivity",
    "worker_timeout_seconds": 1800,
    "warmup": "new worker: tiny procedural 32x32 five-Gaussian normalized CUDA field, all five quality modes twice; pipeline worker additionally runs complete capacity16 step_scale0.001 full-frame pipeline once",
    "expected_cells": 110,
    "missing_policy": "preserve every exception/error cell, no retry or in-place repair; incomplete matrix cannot support a positive decision",
    "forbidden": ["default changes", "post-outcome threshold rescue", "sealed data", "same-state speed described as end-to-end speed"],
}


def expected_cells():
    same = [f"same_coco{image:012d}_s{seed}_{state}_{arm}"
            for image in PROTOCOL["images"] for seed in PROTOCOL["same_state"]["seeds"]
            for state in PROTOCOL["same_state"]["states"] for arm in ARMS]
    pipeline = [f"pipeline_coco{image:012d}_s{seed}_{arm}"
                for image in PROTOCOL["pipeline"]["images"]
                for seed in PROTOCOL["pipeline"]["seeds"] for arm in ARMS]
    return same + pipeline


def counterbalanced_orders():
    rotated = [ARMS[i:] + ARMS[:i] for i in range(len(ARMS))]
    return rotated + [tuple(reversed(order)) for order in rotated]


def pipeline_order(image_index, seed):
    offset = (image_index * 3 + seed) % len(ARMS)
    return ARMS[offset:] + ARMS[:offset]


def ellipse_mask(height, width):
    import numpy as np
    y, x = np.mgrid[:height, :width]
    return (((x - (width - 1) / 2) / (0.42 * width)) ** 2
            + ((y - (height - 1) / 2) / (0.42 * height)) ** 2) <= 1


def discrete_projection(value):
    """Keep actual decisions/topology/selection, not CUDA-sensitive scalar metrics or time."""
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, list):
        return [discrete_projection(v) for v in value]
    if isinstance(value, dict):
        return {k: discrete_projection(v) for k, v in value.items()
                if not isinstance(v, float)}
    if isinstance(value, float):
        return None
    raise TypeError(f"unsupported event scalar {type(value)}")


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def coefficient_of_variation(values):
    return statistics.pstdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pipeline_numeric_valid(row):
    quality = row.get("quality", {})
    required = {"n_gaussians", "foreground_mse", "boundary_mse", "cvar99_mse", "p99_mse",
                "interior_hole_fraction", "boundary_hole_fraction", "outside_max_abs",
                "outside_coverage_max", "finite"}
    error = row.get("final_reference_rgb_max_error")
    return (all(_finite_number(row.get(key)) for key in ("psnr", "ms_ssim", "lpips"))
            and required.issubset(quality) and quality.get("finite") is True
            and all(_finite_number(value) for key, value in quality.items() if key != "finite")
            and _finite_number(error)
            and 0 <= error <= PROTOCOL["pipeline"]["max_reference_rgb_absolute_error"])


def summarize(rows):
    """Apply frozen gates with image as aggregation unit; keep A/A failures visible."""
    problems = []
    validate_rows(rows, PROTOCOL, problems)
    if problems:
        return {"complete": False, "primary_arm": "both", "validation_problems": problems,
                "candidates": {arm: {"role": "primary" if arm == "both" else "explanatory predeclared",
                    "same_state_correctness_pass": False, "same_state_timing_eligible": False,
                    "same_state_images": [], "component_speed_pass": False,
                    "pipeline_images": [], "pipeline_quality_pass": False,
                    "pipeline_trajectory_pass": False, "pipeline_timing_eligible": False,
                    "pipeline_speed_pass": False} for arm in ("coverage", "tail", "both")}}
    lookup = {r["cell_id"]: r for r in rows}
    complete = (len(rows) == len(lookup) == len(expected_cells())
                and set(lookup) == set(expected_cells())
                and all(r.get("status") == "ok" for r in rows))
    result = {"complete": complete, "primary_arm": "both", "candidates": {}}
    for arm in ("coverage", "tail", "both"):
        component_images, component_checks, timing_checks = [], [], []
        for image in PROTOCOL["images"]:
            pairs, aa_pairs = [], []
            for seed in PROTOCOL["same_state"]["seeds"]:
                for state in PROTOCOL["same_state"]["states"]:
                    prefix = f"same_coco{image:012d}_s{seed}_{state}_"
                    group = [lookup.get(prefix + name) for name in ("legacy_a", "legacy_b", arm)]
                    if not all(r and r.get("status") == "ok" for r in group):
                        component_checks.append(False)
                        continue
                    baseline, aa, candidate = group
                    pairs.append(statistics.median(a / b for a, b in zip(
                        baseline["call_seconds"], candidate["call_seconds"])))
                    aa_ratio = statistics.median(a / b for a, b in zip(
                        baseline["call_seconds"], aa["call_seconds"]))
                    aa_stable = (PROTOCOL["same_state"]["aa_runtime_ratio_interval"][0]
                                 <= aa_ratio <= PROTOCOL["same_state"]["aa_runtime_ratio_interval"][1])
                    aa_pairs.append({"seed": seed, "state": state, "speed_ratio": aa_ratio,
                                     "timing_stable": aa_stable})
                    timing_checks.append(aa_stable)
                    component_checks.extend((aa["parity_pass"], candidate["parity_pass"]))
                    timing_checks.extend(r["timing_eligible"] and r["call_time_cv"] <=
                                         PROTOCOL["same_state"]["maximum_call_time_cv"]
                                         for r in group)
            component_images.append({"image": image, "speed_ratio": statistics.median(pairs)
                                     if len(pairs) == 4 else None,
                                     "aa_speed_ratio": statistics.median(pair["speed_ratio"] for pair in aa_pairs)
                                     if len(aa_pairs) == 4 else None, "aa_pairs": aa_pairs})
        cs = [r["speed_ratio"] for r in component_images if r["speed_ratio"] is not None]
        component_pass = (complete and all(component_checks) and all(timing_checks)
                          and len(cs) == 4 and statistics.median(cs) >= 1.1
                          and min(cs) >= PROTOCOL["same_state"]["minimum_per_image_speed_ratio"])
        pipeline_images, quality, trajectories, isolated = [], [], [], []
        for image in PROTOCOL["pipeline"]["images"]:
            ratios, pairs = [], []
            for seed in PROTOCOL["pipeline"]["seeds"]:
                prefix = f"pipeline_coco{image:012d}_s{seed}_"
                group = [lookup.get(prefix + name) for name in ("legacy_a", "legacy_b", arm)]
                if not all(r and r.get("status") == "ok" for r in group):
                    quality.append(False)
                    continue
                base, aa, candidate = group
                ratios.append(base["total_seconds"] / candidate["total_seconds"])
                numeric_valid = all(_pipeline_numeric_valid(r) for r in group)
                q = numeric_valid and all(r["psnr"] >= base["psnr"] - 0.05
                        and r["ms_ssim"] >= base["ms_ssim"] - 0.001
                        and r["lpips"] <= base["lpips"] + 0.002
                        and r["quality"]["outside_max_abs"] <= base["quality"]["outside_max_abs"]
                        and r["quality"]["outside_coverage_max"] <= base["quality"]["outside_coverage_max"]
                        for r in (aa, candidate))
                same = all(r["trajectory_sha256"] == base["trajectory_sha256"]
                           and r["n_gaussians"] == base["n_gaussians"]
                           and r["selected_iteration"] == base["selected_iteration"] for r in (aa, candidate))
                quality.append(q)
                trajectories.append(same)
                aa_ratio = base["total_seconds"] / aa["total_seconds"]
                isolated.extend(r["timing_eligible"] for r in group)
                isolated.append(0.9 <= aa_ratio <= 1.1)
                pairs.append({"seed": seed, "quality_pass": q,
                              "reference_rgb_and_finite_pass": numeric_valid,
                              "aa_and_candidate_trajectory_equal": same,
                              "aa_speed_ratio": aa_ratio,
                              "psnr_gain_db": candidate["psnr"] - base["psnr"]})
            pipeline_images.append({"image": image, "speed_ratio": statistics.median(ratios)
                                    if len(ratios) == 3 else None, "pairs": pairs})
        ps = [r["speed_ratio"] for r in pipeline_images if r["speed_ratio"] is not None]
        pipeline_pass = (complete and all(component_checks) and all(quality)
                         and all(trajectories) and all(isolated) and len(ps) == 2
                         and statistics.median(ps) >= 1.05
                         and min(ps) >= PROTOCOL["pipeline"]["minimum_per_image_speed_ratio"])
        result["candidates"][arm] = {
            "role": "primary" if arm == "both" else "explanatory predeclared",
            "same_state_correctness_pass": complete and all(component_checks),
            "same_state_timing_eligible": complete and all(timing_checks),
            "same_state_images": component_images, "component_speed_pass": component_pass,
            "pipeline_images": pipeline_images, "pipeline_quality_pass": complete and all(quality),
            "pipeline_trajectory_pass": complete and all(trajectories),
            "pipeline_timing_eligible": complete and all(isolated),
            "pipeline_speed_pass": pipeline_pass,
        }
    return result


def validate_rows(rows, protocol, problems):
    """Additional task-local structural checks used by the maintained bundle validator."""
    if protocol != PROTOCOL:
        problems.append("PORT-007 protocol differs from executable protocol")
    ids = [r.get("cell_id") for r in rows]
    if len(rows) != 110 or len(ids) != len(set(ids)) or set(ids) != set(expected_cells()):
        problems.append("PORT-007 matrix is not the frozen 110 cells")
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("method") not in ARMS or row.get("image") not in PROTOCOL["images"]:
            problems.append("PORT-007 invalid arm/image identity")
        kind, image, seed, arm = (row.get(key) for key in ("kind", "image", "seed", "method"))
        if type(image) is not int or type(seed) is not int:
            problems.append(f"PORT-007 noninteger image/seed identity: {row['cell_id']}")
            continue
        if not _finite_number(row.get("total_seconds")) or row["total_seconds"] <= 0:
            problems.append(f"PORT-007 invalid total runtime: {row['cell_id']}")
            continue
        if any(type(row.get(key)) is not int or row[key] < 0
               for key in ("n_gaussians", "iterations_run", "selected_iteration")):
            problems.append(f"PORT-007 invalid integer count/step fields: {row['cell_id']}")
            continue
        if row.get("kind") == "same":
            state = row.get("state")
            if (seed not in PROTOCOL["same_state"]["seeds"]
                    or state not in PROTOCOL["same_state"]["states"]
                    or row["cell_id"] != f"same_coco{image:012d}_s{seed}_{state}_{arm}"):
                problems.append(f"PORT-007 row identity disagrees with fixed-state matrix: {row['cell_id']}")
            timings = row.get("call_seconds", [])
            valid_times = len(timings) == 10 and all(_finite_number(v) and v > 0 for v in timings)
            if not valid_times:
                problems.append(f"PORT-007 missing ten complete-call timings: {row['cell_id']}")
            elif (not _finite_number(row.get("call_time_cv"))
                  or not math.isclose(row["call_time_cv"], coefficient_of_variation(timings), abs_tol=1e-12)
                  or not math.isclose(row["total_seconds"], sum(timings), rel_tol=1e-12, abs_tol=1e-12)):
                problems.append(f"PORT-007 timing summaries disagree with ten calls: {row['cell_id']}")
            horizon = 0 if state == "initial" else 750
            if (row.get("n_gaussians") != 2000 or row.get("iterations_run") != horizon
                    or row.get("selected_iteration") != horizon):
                problems.append(f"PORT-007 parent count/horizon mismatch: {row['cell_id']}")
        elif row.get("kind") == "pipeline":
            if (image not in PROTOCOL["pipeline"]["images"] or seed not in PROTOCOL["pipeline"]["seeds"]
                    or row["cell_id"] != f"pipeline_coco{image:012d}_s{seed}_{arm}"):
                problems.append(f"PORT-007 row identity disagrees with pipeline matrix: {row['cell_id']}")
            if (not 1 <= row["n_gaussians"] <= 1000
                    or row["selected_iteration"] > row["iterations_run"]
                    or not _finite_number(row.get("observer_seconds"))
                    or not 0 <= row["observer_seconds"] <= row["total_seconds"]):
                problems.append(f"PORT-007 pipeline budget/observer mismatch: {row['cell_id']}")
        else:
            problems.append(f"PORT-007 unknown assay kind: {row['cell_id']}")
