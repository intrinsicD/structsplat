#!/usr/bin/env python3
"""FIT-051: actual-render transactions on all eight immutable FIT-050 parents.

Freeze: python scripts/experiments/fit051_actual_color_ray.py --protocol-only
Formal: python scripts/experiments/fit051_actual_color_ray.py --out NEW_OUT \
    --parent-bundle FIT050_OUT --approved-protocol-digest DIGEST
Diagnostic: --out NEW_OUT --smoke (one procedural CPU fixture, never formal parents).
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
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmarks.fit050_controls import (  # noqa: E402
    _changed_noncolors, fit_parent, full_frame_context, parent_configs,
)
from benchmarks.fit051_controls import (  # noqa: E402
    ARMS, DIRECTIONS, IMAGE_IDS, PARENT_FILES, PARENT_MANIFEST_SHA256, PARENT_PROTOCOL_DIGEST,
    PARENT_SOURCE_COMMIT, SEEDS, compare_rows, expected_cells, parent_id, quality_inputs,
    run_endpoint, summarize_rows,
)
from benchmarks.hier_research_report import (  # noqa: E402
    ResearchBundle, protocol_digest, save_rgb, sha256, validate_bundle, write_json,
)
from scripts.experiments.fit050_color_ray import (  # noqa: E402
    _cpu_state, _field_equal, _json_form, _quality_record, gpu_snapshot, procedural_target,
    score as _fit050_endpoint_score,
)

_INIT, _FIT = parent_configs(0)
PROTOCOL = {
    "task": "FIT-051", "version": 1, "arms": ARMS, "directions": DIRECTIONS,
    "image_ids": IMAGE_IDS, "seeds": SEEDS, "n_gaussians": 2000, "parent_steps": 750,
    "device": "cuda", "gpu": "NVIDIA GeForce RTX 3050", "torch": "2.9.0+cu128",
    "dtype": "float32", "threads": 1, "parent_init": asdict(_INIT), "parent_fit": asdict(_FIT),
    "parent_manifest_sha256": PARENT_MANIFEST_SHA256, "parent_protocol_digest": PARENT_PROTOCOL_DIGEST,
    "parent_source_commit": PARENT_SOURCE_COMMIT, "parent_file_sha256": PARENT_FILES,
    "parent_transfer": "all eight original FIT-050 initial/terminal fields, configs, target arrays, optimizer states and histories; cross-commit transfer is explicit, all hashes frozen here; no refits, selection or new natural images",
    "data_role": "already exposed COCO development parents; new mechanism after FIT-050's recorded compatibility aborts; original outcome untouched",
    "ridge": 1e-4, "fractions": [2.0 ** -k for k in range(6)], "cg_maxiter": 32,
    "adam_steps": 32, "coverage_tau": 0.05, "quality_backends": "reference coverage and tail",
    "selection": "one transaction; first actual-render safe nonzero change, then actual selected replay; no retry after replay failure",
    "streaming_directions": "approximate cross-backend proposals, not exact maintained CUDA Jacobians; only actual images enter the gate",
    "native_direction": "maintained-renderer color VJP with target-minus-measured-parent residual; no claimed specialized color-only backward kernel; geometry/opacity/support frozen",
    "alpha": "non-CG: q is actual maintained render of signed direction coefficients; alpha=(residual*q).sum/(q.square().sum+ridge*direction.square().sum); alpha and every trial are charged",
    "cg_endpoint": "fraction1 copies independently solved legacy endpoint colors exactly; later fractions use parent colors + fraction*(endpoint-parent), avoiding endpoint subtract/add roundoff",
    "controls": "task-local observable wrappers use unchanged maintained _solve_colors_normalized and fit; all candidate fields/images, including rejected endpoints, retained; no mutated old globals",
    "order": "new process per prescribed parent; rotate seven arms by image-index*2+seed modulo seven",
    "warmup": "procedural64 seed77, three-step16-row CPU-or-CUDA parent; all seven arms (Adam2), canonical CPU LPIPS warmup; excluded from transaction cost",
    "timing": "complete transaction, directions, actual trials/replay and owned capture; reporting/cold replay/serialization separately recorded; shared workstation descriptive only",
    "occupancy": "parent process samples nvidia-smi about once per second; foreign/query-failure makes timing ineligible; no continuous exclusivity claim",
    "curves": "every in-transaction actual quality evaluation (parent, trials/candidate and selected replay) plus a separately labeled reporting endpoint: raw PSNR/MSE/MAE, display-clamped SSIM/MS-SSIM/LPIPS and complete protected vector; cold reader check is retained in arrays/history but not plotted; original parent/Adam histories linked; evaluation index is not optimizer steps",
    "reporting_scorer": "canonical CPU float32 copies of retained raw RGB and target, CPU threads1, for every endpoint/curve PSNR/MSE/MAE/SSIM/MS-SSIM/LPIPS; reporting/copy cost outside instrumented transaction; portable checker uses the identical helper; protected quality and safe decisions remain on the original renderer device",
    "reporting_scope": "all seven FIT-051 arms use canonical CPU reporting; no cross-source comparison with old FIT-050 perceptual metric values",
    "minimum_median_image_gain_db": 0.1, "maximum_cell_loss_db": 0.01,
    "maximum_ms_ssim_loss": 0.001, "maximum_lpips_increase": 0.002,
    "cold_parity_max_abs": 2e-5, "selected_parity_max_abs": 2e-5,
    "artifact_metric_relative_tolerance": 5e-6, "artifact_metric_absolute_tolerance": 1e-10,
    "artifact_alpha_relative_tolerance": 5e-6, "artifact_alpha_absolute_tolerance": 1e-8,
    "artifact_tolerances": "CPU recomputation of saved CUDA reductions only; never alter the exact recorded safe-commit predicates",
    "artifact_tensor_inventory": "canonical float32 method/stage operands with exact shapes; declared method and attained stage determine work, never optional payload presence; irrelevant operands forbidden",
    "worker_timeout_seconds": 3600,
    "missing_policy": "complete56 required for positive utility; retain all errors/partial artifacts; no selective rerun or in-place repair",
    "diagnostic": "one procedural64 CPU fixture,16 rows, parent3/Adam2; no imported parent/data, no scientific verdict",
    "forbidden": ["default promotion", "threshold rescue", "repeated ray", "new natural parents", "sealed data"],
}
SOURCES = sorted([
    "scripts/experiments/fit051_actual_color_ray.py", "benchmarks/fit051_controls.py",
    "scripts/experiments/fit050_color_ray.py", "benchmarks/fit050_controls.py",
    "benchmarks/hier_research_report.py", "scripts/check_report_bundle.py",
    "src/structsplat/cuda/render_ext.cpp", "src/structsplat/cuda/render_ext.cu",
] + [path.relative_to(ROOT).as_posix() for path in (ROOT / "src/structsplat").rglob("*.py")])


def validate_rows(rows, protocol, problems, *, diagnostic=False):
    if protocol != _json_form(PROTOCOL):
        problems.append("FIT-051 protocol differs from the frozen executable")
    compare_rows(rows, protocol, problems, diagnostic=diagnostic)


def summarize(rows, protocol=PROTOCOL, *, diagnostic=False):
    return summarize_rows(rows, protocol, diagnostic=diagnostic)


def endpoint_score(raw, target, *, smoke=False):
    """Canonical portable reporting, separate from the unchanged device-native safe gate."""
    import torch
    from structsplat.metrics import ssim
    prior_threads = torch.get_num_threads()
    if prior_threads != 1:
        torch.set_num_threads(1)
    try:
        image, truth = raw.detach().to("cpu", copy=True), target.detach().to("cpu", copy=True)
        return {**_fit050_endpoint_score(image, truth, smoke=smoke),
                "mae": float((image - truth).abs().mean()), "ssim": float(ssim(image.clamp(0, 1), truth))}
    finally:
        if prior_threads != 1:
            torch.set_num_threads(prior_threads)


def scored_point(raw, target, quality, index, elapsed, *, diagnostic=False, stage="actual"):
    return {**endpoint_score(raw, target, smoke=diagnostic), **quality,
            "evaluation_index": index, "elapsed_seconds": elapsed, "stage": stage}


def save_arrays(directory, arrays):
    flattened, inventory = {}, {}
    for name, value in arrays.items():
        if value is None:
            inventory[name] = None
        elif isinstance(value, list):
            keys = [f"{name}_{index}" for index in range(len(value))]
            inventory[name] = keys
            flattened.update({key: tensor.detach().cpu().numpy() for key, tensor in zip(keys, value)})
        else:
            inventory[name] = name
            flattened[name] = value.detach().cpu().numpy()
    np.savez_compressed(directory / "transaction_arrays.npz", **flattened)
    write_json(directory / "tensor_inventory.json", inventory)


def plot(directory, points, parent_history, adam_history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ("psnr", "ssim", "ms_ssim", "lpips", "raw_mse", "mae", "cvar99_mse", "p99_mse",
            "interior_hole_fraction", "boundary_hole_fraction", "outside_coverage_max", "elapsed_seconds")
    fig, axes = plt.subplots(4, 3, figsize=(12, 10))
    for axis, key in zip(axes.flat, keys):
        values = [(point["evaluation_index"], point[key]) for point in points if point[key] is not None]
        if values:
            axis.plot([p[0] for p in values], [p[1] for p in values], marker=".")
        axis.set(xlabel="Quality evaluation + reporting (not optimizer steps)", ylabel=key)
    fig.tight_layout()
    fig.savefig(directory / "curves.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for axis, history, label in zip(axes, (parent_history, adam_history), ("Original parent", "Adam continuation")):
        if history:
            axis.plot(history["iter"], history["psnr"])
        axis.set(xlabel="Native pre-step iteration", ylabel="Raw PSNR (dB)", title=label)
    fig.tight_layout()
    fig.savefig(directory / "optimizer_curves.png", dpi=120)
    plt.close(fig)


def transfer_parents(source, destination):
    if sha256(source / "manifest.json") != PARENT_MANIFEST_SHA256:
        raise ValueError("source is not the exact frozen FIT-050 manifest")
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["task"] != "FIT-050" or manifest["protocol_digest"] != PARENT_PROTOCOL_DIGEST
            or manifest["repository"]["commit"] != PARENT_SOURCE_COMMIT or manifest["repository"]["dirty"]):
        raise ValueError("source FIT-050 identity differs")
    for name, expected in PARENT_FILES.items():
        if manifest["files"].get(name) != expected or sha256(source / name) != expected:
            raise ValueError(f"frozen imported parent differs: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(source / "manifest.json", destination / "parent_source_manifest.json")
    write_json(destination / "parent_transfer.json", {"source_bundle": str(source),
        "source_manifest_sha256": PARENT_MANIFEST_SHA256, "source_protocol_digest": PARENT_PROTOCOL_DIGEST,
        "source_commit": PARENT_SOURCE_COMMIT, "files": PARENT_FILES})


def make_diagnostic_parent(directory):
    import torch
    target = procedural_target()
    parent = fit_parent(target, 77, smoke=True, device="cpu")
    directory.mkdir(parents=True, exist_ok=False)
    parent["initial_field"].save(directory / "initial_field.npz")
    parent["field"].save(directory / "field.npz")
    np.save(directory / "target.npy", target)
    torch.save(_cpu_state(parent["optimizer_state"]), directory / "optimizer_state.pt")
    write_json(directory / "history.json", parent["history"])
    write_json(directory / "config.json", {"init": parent["init_config"], "fit": parent["fit_config"],
        "tensor": parent["tensor_config"], "image_id": -1, "seed": 77, "diagnostic": True,
        "field_sha256": sha256(directory / "field.npz"), "target_sha256": sha256(directory / "target.npy"),
        "optimizer_state_sha256": sha256(directory / "optimizer_state.pt")})


def worker(request_path):
    import torch
    from structsplat.actual_color_ray import refine_actual_color_ray
    from structsplat.config import FitConfig
    from structsplat.gaussians import GaussianField

    request = json.loads(Path(request_path).read_text())
    root, diagnostic = Path(request["root"]), request["smoke"]
    device = "cpu" if diagnostic else "cuda"
    torch.set_num_threads(1)
    if not diagnostic and (torch.__version__ != PROTOCOL["torch"] or not torch.cuda.is_available()
                           or torch.cuda.get_device_name() != PROTOCOL["gpu"]):
        raise RuntimeError("frozen GPU/torch unavailable")
    warm = fit_parent(procedural_target(), 77, smoke=True, device=device)
    warm_target = torch.as_tensor(procedural_target(), device=device)
    warm_cfg = FitConfig(**warm["fit_config"])
    context = full_frame_context(warm_target, warm_cfg, warm["field"].n)
    for method in ARMS:
        if method in DIRECTIONS:
            refine_actual_color_ray(warm["field"], warm_target, warm_cfg, *context, direction=DIRECTIONS[method])
        else:
            run_endpoint(warm["field"], warm_target, warm_cfg, *context, method, warm["optimizer_state"], diagnostic=True)
    endpoint_score(warm["render"], warm_target, smoke=diagnostic)
    del warm, warm_target, context
    parent_dir = root / "parents" / request["parent_id"]
    parent_config = json.loads((parent_dir / "config.json").read_text())
    cfg = FitConfig(**parent_config["fit"])
    parent = GaussianField.load(parent_dir / "field.npz", device=device)
    target_np = np.load(parent_dir / "target.npy", allow_pickle=False)
    target = torch.as_tensor(target_np, device=device)
    optimizer_state = torch.load(parent_dir / "optimizer_state.pt", map_location=device, weights_only=True)
    context = full_frame_context(target, cfg, parent.n)
    parent_history = json.loads((parent_dir / "history.json").read_text())
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
                   "python": sys.version, "worker_pid": os.getpid(),
                   "gpu": None if diagnostic else torch.cuda.get_device_name()}
    rows = []
    for method in request["arm_order"]:
        cell_id = f"{request['parent_id']}_{method}"
        directory = root / "cells" / cell_id
        directory.mkdir(parents=True, exist_ok=False)
        common = {key: request[key] for key in ("parent_id", "image_id", "seed", "smoke")}
        common.update({"cell_id": cell_id, "method": method,
            "parent_field_sha256": sha256(parent_dir / "field.npz"),
            "parent_optimizer_sha256": sha256(parent_dir / "optimizer_state.pt"),
            "target_sha256": sha256(parent_dir / "target.npy")})
        write_json(directory / "request.json", common)
        try:
            torch.manual_seed(request["seed"])
            if not diagnostic:
                torch.cuda.reset_peak_memory_stats()
            cell_started = time.perf_counter()
            if method in DIRECTIONS:
                selected, protected, metadata, arrays = refine_actual_color_ray(
                    parent, target, cfg, *context, direction=DIRECTIONS[method], max_trials=6, cg_maxiter=32)
                metadata.update({"arm": method, "noncolor_changed_fields": _changed_noncolors(parent, selected)})
                metadata["transaction_seconds"] = metadata["elapsed_seconds"]
                adam_history, candidate_field = {}, None
            else:
                selected, protected, metadata, arrays, adam_history, candidate_field = run_endpoint(
                    parent, target, cfg, *context, method, optimizer_state, diagnostic=diagnostic)
            peak = 0 if diagnostic else torch.cuda.max_memory_allocated()
            reserved = 0 if diagnostic else torch.cuda.max_memory_reserved()
            selected.save(directory / "field.npz")
            shutil.copy2(parent_dir / "field.npz", directory / "input_field.npz")
            if candidate_field is not None:
                candidate_field.save(directory / "candidate_field.npz")
            reporting, raw, den = quality_inputs(selected, target, cfg, context[0], context[1], context[2].coverage_tau)
            cold = GaussianField.load(directory / "field.npz", device=device)
            cold_quality, cold_raw, cold_den = quality_inputs(cold, target, cfg, context[0], context[1], context[2].coverage_tau)
            if metadata["accepted"]:
                selected_raw = arrays["replay_render"] if method in DIRECTIONS else arrays["candidate_render"]
            else:
                selected_raw = arrays["parent_render"]
            replay_error = float((raw - selected_raw).abs().max())
            cold_error = float((raw - cold_raw).abs().max())
            arrays.update({"reporting_render": raw, "reporting_denominator": den,
                           "cold_render": cold_raw, "cold_denominator": cold_den})
            save_arrays(directory, arrays)
            np.save(directory / "reconstruction.npy", raw.detach().cpu().numpy())
            save_rgb(directory / "target.png", target_np)
            save_rgb(directory / "reconstruction.png", raw.detach().cpu().numpy())
            save_rgb(directory / "error.png", (raw - target).abs().detach().cpu().numpy() * 4)
            points = [scored_point(arrays["parent_render"], target, metadata["parent_metrics"], 0, 0,
                                   diagnostic=diagnostic, stage="parent")]
            if method in DIRECTIONS:
                for index, trial in enumerate(metadata["trials"]):
                    points.append(scored_point(arrays["trial_renders"][index], target, trial["actual_metrics"],
                        index + 1, trial["transaction_elapsed_seconds"], diagnostic=diagnostic, stage="actual_trial"))
                    save_rgb(directory / f"trial_{index:02d}.png", arrays["trial_renders"][index].cpu().numpy())
                if metadata["replay_metrics"] is not None:
                    points.append(scored_point(arrays["replay_render"], target, metadata["replay_metrics"],
                        len(points), metadata["transaction_seconds"], diagnostic=diagnostic, stage="selected_actual_replay"))
            elif method != "noop":
                points.append(scored_point(arrays["candidate_render"], target, metadata["candidate_metrics"],
                    1, metadata["transaction_seconds"], diagnostic=diagnostic, stage="endpoint_candidate"))
            points.append(scored_point(raw, target, reporting.to_dict(), len(points),
                                       metadata["transaction_seconds"], diagnostic=diagnostic, stage="reporting"))
            write_json(directory / "history.json", {"transaction": metadata, "adam_fit": adam_history,
                "checkpoints": points, "parent_history": f"../../parents/{request['parent_id']}/history.json",
                "reporting_quality": reporting.to_dict(), "cold_quality": cold_quality.to_dict(),
                "reporting_work": {"quality_evaluations": 2, "gaussian_renders": 2, "raw_coverage_passes": 2}})
            write_json(directory / "config.json", {"request": common, "fit": asdict(cfg),
                "schedule": asdict(context[2]), "environment": environment,
                "parent_config": f"../../parents/{request['parent_id']}/config.json"})
            plot(directory, points, parent_history, adam_history)
            exact = _field_equal(selected, cold)
            if not exact or cold_error > 2e-5 or replay_error > 2e-5:
                raise RuntimeError("selected/cold reader replay failed; raw arrays retained")
            row = {**common, **endpoint_score(raw, target, smoke=diagnostic), "status": "ok",
                "n_gaussians": selected.n, "iterations_run": (2 if diagnostic else 32) if method == "adam32" else 0,
                "selected_iteration": ((2 if diagnostic else 32) if method == "adam32" and metadata["accepted"] else 0),
                "accepted": metadata["accepted"], "coefficients_changed": metadata["coefficients_changed"],
                "selected_fraction": metadata["selected_fraction"], "rollback_reason": metadata["rollback_reason"],
                "noncolor_changed_fields": metadata["noncolor_changed_fields"], "counts": metadata["counts"],
                "parent_protected_metrics": metadata["parent_metrics"], "protected_metrics": protected.to_dict(),
                "reporting_metrics": reporting.to_dict(), "cold_metrics": cold_quality.to_dict(),
                "cold_parameters_exact": exact, "cold_render_max_abs": cold_error,
                "selected_replay_max_abs": replay_error, "transaction_seconds": metadata["transaction_seconds"],
                "total_seconds": metadata["transaction_seconds"], "cell_total_seconds": time.perf_counter() - cell_started,
                "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
                "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        except Exception as exc:
            row = {**common, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        row["artifacts"] = {path.name: f"cells/{cell_id}/{path.name}" for path in directory.iterdir()
                            if path.is_file() and path.name != "row.json"}
        for filename in ("config.json", "field.npz", "optimizer_state.pt", "target.npy", "history.json"):
            row["artifacts"]["parent_" + filename] = f"parents/{request['parent_id']}/{filename}"
        write_json(directory / "row.json", row)
        rows.append(row)
        print(json.dumps({"cell_id": cell_id, "status": row["status"]}), flush=True)
    write_json(parent_dir / "rows.json", rows)


def _close(actual, expected, protocol, *, alpha=False):
    prefix = "artifact_alpha" if alpha else "artifact_metric"
    return math.isclose(float(actual), float(expected),
        rel_tol=protocol[prefix + "_relative_tolerance"], abs_tol=protocol[prefix + "_absolute_tolerance"])


def _raw_quality(raw, den, target, cfg, context, expected, protocol):
    import torch
    from structsplat.safe_schedule import _quality_from_render
    if (raw.shape != target.shape or den.shape != target.shape[:2]
            or raw.dtype != np.float32 or den.dtype != np.float32
            or not np.isfinite(raw).all() or not np.isfinite(den).all()):
        raise ValueError("raw RGB/coverage shape/dtype/finiteness differs")
    _quality_record(expected, context[2].capacity)
    result = _quality_from_render(torch.from_numpy(raw), target, torch.from_numpy(den),
        context[0], context[1], context[2].coverage_tau, context[2].capacity,
        tail_backend="reference").to_dict()
    for key, value in result.items():
        if isinstance(value, bool) or key == "n_gaussians":
            if value != expected[key]:
                raise ValueError(f"raw quality {key} differs")
        elif not _close(value, expected[key], protocol):
            raise ValueError(f"raw quality {key} differs")


def _validate_ray_inventory(parent, target, direction, metadata, arrays, inventory):
    """Require every operand produced by the declared method and attained stage."""
    rgb, den, colors = tuple(target.shape), tuple(target.shape[:2]), tuple(parent.colors.shape)
    shapes = {"parent_render": rgb, "parent_denominator": den, "gradient": colors,
        "diagonal": (parent.n,), "direction": colors, "direction_render": rgb,
        "native_gradient_render": rgb, "cg_endpoint_colors": colors,
        "replay_render": rgb, "replay_denominator": den}
    core = set(shapes) | {"trial_renders", "trial_denominators"}
    reporting = {"reporting_render", "reporting_denominator", "cold_render", "cold_denominator"}
    if not core <= set(inventory) or set(inventory) - core - reporting:
        raise ValueError("canonical ray tensor inventory keys differ")
    declared = [entry for value in inventory.values() if value is not None
                for entry in (value if isinstance(value, list) else [value])]
    if len(declared) != len(set(declared)) or set(declared) != set(arrays):
        raise ValueError("canonical ray tensor payload inventory differs")

    def require(name, present, shape):
        if not present:
            if inventory[name] is not None or name in arrays:
                raise ValueError(f"irrelevant or unexecuted ray operand {name} is present")
        elif (inventory[name] != name or name not in arrays
              or arrays[name].shape != shape or arrays[name].dtype != np.float32):
            raise ValueError(f"required ray operand {name} missing or shape/dtype differs")

    for name in ("parent_render", "parent_denominator", "direction"):
        require(name, True, shapes[name])
    for name, present in (("gradient", direction != "cg"),
                          ("diagonal", direction == "streaming_jacobi"),
                          ("native_gradient_render", direction == "native_gradient"),
                          ("cg_endpoint_colors", direction == "cg")):
        require(name, present, shapes[name])
    if ("legacy_cg" in metadata) != (direction == "cg"):
        raise ValueError("declared method legacy CG ledger inventory differs")
    vector = arrays["direction"]
    q_stage = bool(np.isfinite(vector).all() and np.any(vector != 0))
    require("direction_render", q_stage, rgb)
    replay_stage = metadata["replay_metrics"] is not None
    for name in ("replay_render", "replay_denominator"):
        require(name, replay_stage, shapes[name])
    trials = metadata["trials"]
    if not isinstance(trials, list) or len(trials) > 6:
        raise ValueError("actual trial inventory differs from frozen ladder")
    for prefix, shape in (("trial_renders", rgb), ("trial_denominators", den)):
        if inventory[prefix] != [f"{prefix}_{i}" for i in range(len(trials))]:
            raise ValueError("actual trial tensor list differs from ladder")
        for name in inventory[prefix]:
            if arrays[name].shape != shape or arrays[name].dtype != np.float32:
                raise ValueError(f"actual trial operand {name} shape/dtype differs")
    if trials and (not q_stage or not np.isfinite(arrays["direction_render"]).all()
                   or metadata["alpha_star"] is None or not math.isfinite(metadata["alpha_star"])
                   or metadata["alpha_star"] <= 0):
        raise ValueError("nonempty actual ladder lacks finite direction/image/positive scalar")
    phases = {"initial_actual_quality"}
    phases.update({"cg": {"legacy_cg"}, "native_gradient": {"native_color_vjp"},
        "streaming_gradient": {"streaming_denominator", "streaming_transpose"},
        "streaming_jacobi": {"streaming_denominator", "streaming_transpose", "streaming_diagonal"}}[direction])
    if q_stage:
        phases.add("actual_direction_render")
    if trials:
        phases.add("actual_trial_quality")
    if replay_stage:
        phases.add("selected_actual_replay")
    if (set(metadata["phase_seconds"]) != phases
            or any(not math.isfinite(value) or not 0 <= value <= metadata["transaction_seconds"]
                   for value in metadata["phase_seconds"].values())):
        raise ValueError("declared method/stage phase inventory differs")
    return q_stage


def _validate_mechanism(parent, selected, target, cfg, context, row, metadata, arrays, inventory,
                        candidate_field, protocol, diagnostic):
    import torch
    from structsplat.safe_schedule import safe_commit_decision
    count, method = parent.n, row["method"]
    before = _quality_record(metadata["parent_metrics"], count)
    gate = lambda value: safe_commit_decision(before, _quality_record(value, count),
                                              context[2].tolerances, context[2].hole_regression_budget)
    for key in ("accepted", "coefficients_changed", "selected_fraction", "rollback_reason", "counts",
                "noncolor_changed_fields", "transaction_seconds"):
        if row[key] != metadata[key]:
            raise ValueError(f"row/transaction {key} differs")
    if (row["parent_protected_metrics"] != metadata["parent_metrics"]
            or row["protected_metrics"] != metadata["selected_metrics"]
            or metadata["quality_coverage_backend"] != "reference"
            or metadata["quality_tail_backend"] != "reference"):
        raise ValueError("row quality/backend differs from transaction")
    _raw_quality(arrays["parent_render"], arrays["parent_denominator"], target, cfg, context,
                 metadata["parent_metrics"], protocol)
    if not _field_equal(parent, selected, colors=False) and method != "adam32":
        raise ValueError("color-only transaction modified non-color state")
    if _changed_noncolors(parent, selected) != row["noncolor_changed_fields"]:
        raise ValueError("non-color mutation flag differs from saved fields")
    if row["coefficients_changed"] != (not torch.equal(parent.colors, selected.colors)):
        raise ValueError("coefficient-change flag differs from saved fields")
    if not row["accepted"] and (not _field_equal(parent, selected)
                               or metadata["selected_metrics"] != metadata["parent_metrics"]):
        raise ValueError("rejected transaction was not exact rollback")
    if metadata["foreground_mse_improved"] != (
            metadata["selected_metrics"]["foreground_mse"] < before.foreground_mse):
        raise ValueError("reported foreground improvement differs")
    if method not in DIRECTIONS:
        steps = (2 if diagnostic else 32) if method == "adam32" else 0
        expected_counts = {"quality_evaluations": 1 if method == "noop" else 2,
            "gaussian_renders": 1 if method == "noop" else 2 + (steps + 1 if method == "adam32" else 0),
            "raw_coverage_passes": 1 if method == "noop" else 2, "basis_denominator_passes": 0,
            "basis_apply_calls": 0, "basis_transpose_calls": 0, "gradient_evaluations": steps}
        if method == "noop":
            if row["accepted"] or candidate_field is not None or metadata["selected_metrics"] != metadata["parent_metrics"]:
                raise ValueError("noop did not retain exact parent")
        else:
            _raw_quality(arrays["candidate_render"], arrays["candidate_denominator"], target, cfg, context,
                         metadata["candidate_metrics"], protocol)
            accepted, reasons = gate(metadata["candidate_metrics"])
            if accepted != metadata["accepted"] or reasons != metadata["candidate_reasons"]:
                raise ValueError("endpoint complete safe decision differs")
            if candidate_field is None or candidate_field.n != count:
                raise ValueError("endpoint candidate field missing/wrong count")
            if not _field_equal(selected, candidate_field if accepted else parent):
                raise ValueError("endpoint selected field differs from safe choice")
            if (metadata["selected_metrics"] != (metadata["candidate_metrics"] if accepted else metadata["parent_metrics"])
                    or metadata["rollback_reason"] != (None if accepted else "candidate_rejected")
                    or metadata["selected_alpha"] != (1.0 if accepted else 0.0)):
                raise ValueError("endpoint selected metrics/alpha/rollback disposition differs")
            candidate_cfg = replace(cfg, iters=steps, log_every=1) if method == "adam32" else replace(cfg, color_solve_maxiter=32)
            if metadata["candidate_config"] != _json_form(asdict(candidate_cfg)):
                raise ValueError("endpoint proposal configuration changed")
        if method == "legacy_cg32":
            if not _field_equal(parent, candidate_field, colors=False):
                raise ValueError("legacy CG changed geometry")
            cg = metadata["legacy_cg"]
            calls, iters = cg["normal_matvec_calls"], cg["iterations"]
            if not 0 <= iters <= 32 or calls not in (iters + 1, iters + 2) or calls > 33:
                raise ValueError("legacy CG work horizon differs")
            expected_counts.update({"basis_denominator_passes": 1, "basis_apply_calls": calls,
                                    "basis_transpose_calls": calls + 1})
            if (cg["denominator_calls"] != 1 or cg["basis_apply_calls"] != calls
                    or cg["basis_transpose_calls"] != calls + 1):
                raise ValueError("legacy CG operator ledger differs")
    else:
        direction = DIRECTIONS[method]
        if (metadata["direction"] != direction or metadata["max_trials"] != 6
                or metadata["cg_maxiter"] != 32 or metadata["ridge"] != 1e-4
                or metadata["image_interpolation"] is not False
                or metadata["cg_endpoint_exact_alpha1"] != (direction == "cg")):
            raise ValueError("actual-render direction/ladder/ridge semantics changed")
        q_stage = _validate_ray_inventory(parent, target, direction, metadata, arrays, inventory)
        if candidate_field is not None:
            raise ValueError("actual ray unexpectedly stores an endpoint-control candidate")
        vector = arrays["direction"]
        if direction == "cg":
            expected_vector = torch.from_numpy(arrays["cg_endpoint_colors"]) - parent.colors
            if not np.array_equal(vector, expected_vector.numpy(), equal_nan=True):
                raise ValueError("CG direction differs from stored endpoint")
        else:
            gradient = torch.from_numpy(arrays["gradient"])
            expected_vector = gradient
            if direction == "streaming_jacobi":
                divisor = torch.from_numpy(arrays["diagonal"]) + 1e-4
                valid = torch.isfinite(divisor) & (divisor > 0)
                safe = torch.where(valid, divisor, torch.ones_like(divisor))
                expected_vector = torch.where(valid[:, None], gradient / safe[:, None], torch.zeros_like(gradient))
            if not np.allclose(vector, expected_vector.numpy(), rtol=2e-7, atol=1e-10, equal_nan=True):
                raise ValueError("gradient/Jacobi direction differs from recorded operands")
        q = arrays.get("direction_render")
        if q is not None and np.isfinite(q).all() and direction != "cg":
            residual = (target.numpy() - arrays["parent_render"]).astype(np.float64)
            numerator = float(np.sum(residual * q.astype(np.float64)))
            denominator = float(np.square(q.astype(np.float64)).sum() + 1e-4 * np.square(vector.astype(np.float64)).sum())
            if not _close(numerator, metadata["numerator"], protocol, alpha=True) or not _close(
                    denominator, metadata["denominator"], protocol, alpha=True):
                raise ValueError("line minimizer operands differ from raw direction image")
        trials = metadata["trials"]
        if len(trials) > 6 or len(inventory["trial_renders"]) != len(trials) or len(inventory["trial_denominators"]) != len(trials):
            raise ValueError("actual trial inventory differs from ladder")
        previous_elapsed, selected_colors = 0.0, None
        for index, trial in enumerate(trials):
            if (trial["index"] != index or trial["fraction"] != 2.0 ** -index
                    or trial["alpha"] != metadata["alpha_star"] * trial["fraction"]):
                raise ValueError("actual trial order/fraction/alpha differs")
            if (not previous_elapsed <= trial["transaction_elapsed_seconds"] <= metadata["transaction_seconds"]
                    or not 0 <= trial["elapsed_seconds"] <= trial["transaction_elapsed_seconds"]):
                raise ValueError("actual trial temporal accounting differs")
            previous_elapsed = trial["transaction_elapsed_seconds"]
            colors = torch.from_numpy(arrays["cg_endpoint_colors"]) if direction == "cg" and index == 0 else parent.colors + trial["alpha"] * torch.from_numpy(vector)
            raw, den = arrays[f"trial_renders_{index}"], arrays[f"trial_denominators_{index}"]
            _raw_quality(raw, den, target, cfg, context, trial["actual_metrics"], protocol)
            accepted, reasons = gate(trial["actual_metrics"])
            changed = not torch.equal(colors, parent.colors)
            finite = bool(torch.isfinite(colors).all()) and trial["actual_metrics"]["finite"]
            if not finite:
                accepted, reasons = False, reasons + ["nonfinite_trial"]
            if not changed:
                accepted, reasons = False, reasons + ["unchanged_coefficients"]
            if (trial["accepted"] != accepted or trial["reasons"] != reasons
                    or trial["coefficients_changed"] != changed or trial["finite"] != finite):
                raise ValueError("actual trial complete gate/change flags differ")
            if accepted and index != len(trials) - 1:
                raise ValueError("continued after the first actual safe trial")
            checks = {"coefficient_max_abs_change": float((colors - parent.colors).abs().max()),
                "image_max_abs_change": float(np.max(np.abs(raw - arrays["parent_render"]))),
                "raw_sse": float(np.square(raw.astype(np.float64) - target.numpy().astype(np.float64)).sum()),
                "ridge_penalty": float(1e-4 * np.square((colors - parent.colors).numpy().astype(np.float64)).sum())}
            if any(not _close(value, trial[key], protocol, alpha=True) for key, value in checks.items()):
                raise ValueError("actual trial raw SSE/ridge/field-change diagnostics differ")
            selected_colors = colors
        replay_present = metadata["replay_metrics"] is not None
        if replay_present:
            if not trials or not trials[-1]["accepted"]:
                raise ValueError("replay lacks first safe actual trial")
            _raw_quality(arrays["replay_render"], arrays["replay_denominator"], target, cfg, context,
                         metadata["replay_metrics"], protocol)
            error = float(np.max(np.abs(arrays["replay_render"] - arrays[f"trial_renders_{len(trials)-1}"])))
            accepted, reasons = gate(metadata["replay_metrics"])
            if error > 2e-5:
                accepted, reasons = False, reasons + ["replay_parity_failed"]
            if (error != metadata["replay_max_abs_error"] or accepted != metadata["accepted"]
                    or reasons != metadata["replay_reasons"] or trials[-1]["replay_accepted"] != accepted
                    or trials[-1]["replay_metrics"] != metadata["replay_metrics"]):
                raise ValueError("selected actual replay gate/parity differs")
        elif metadata["accepted"] or (trials and trials[-1]["accepted"]):
            raise ValueError("accepted actual trial lacks selected replay")
        if metadata["accepted"]:
            if (not torch.equal(selected.colors, selected_colors)
                    or metadata["selected_fraction"] != trials[-1]["fraction"]
                    or metadata["selected_alpha"] != trials[-1]["alpha"]
                    or metadata["selected_trial_index"] != trials[-1]["index"]
                    or metadata["selected_metrics"] != metadata["replay_metrics"]
                    or metadata["rollback_reason"] is not None):
                raise ValueError("selected field does not implement the selected actual trial")
        elif (metadata["selected_fraction"] != 0 or metadata["selected_alpha"] != 0
              or metadata["selected_trial_index"] is not None):
            raise ValueError("rollback retained a selected coefficient change/index")
        elif replay_present and metadata["rollback_reason"] != "selected_replay_failed":
            raise ValueError("rejected replay has incorrect rollback disposition")
        elif trials and not replay_present and metadata["rollback_reason"] != "all_trials_rejected":
            raise ValueError("exhausted actual ladder has incorrect rollback disposition")
        if not trials:
            if metadata["accepted"] or metadata["alpha_star"] is not None:
                raise ValueError("empty actual ladder has a selected scalar/acceptance")
            reason = metadata["rollback_reason"]
            if reason == "invalid_or_zero_direction":
                if vector is None or (np.isfinite(vector).all() and np.any(vector != 0)) or q is not None:
                    raise ValueError("zero/invalid direction abort is not supported by raw direction")
            elif reason == "invalid_line_minimizer":
                if q is None or (metadata["numerator"] is not None and metadata["denominator"] is not None
                                 and metadata["numerator"] > 0 and metadata["denominator"] > 0):
                    raise ValueError("line-minimizer abort lacks a failed operand")
            elif reason == "nonfinite_direction_render":
                if q is None or np.isfinite(q).all():
                    raise ValueError("nonfinite direction-render abort lacks raw evidence")
            else:
                raise ValueError("empty actual ladder lacks an admissible early-abort reason")
        if trials:
            expected_alpha = 1.0 if direction == "cg" else metadata["numerator"] / metadata["denominator"]
            if metadata["alpha_star"] != expected_alpha:
                raise ValueError("actual ray scalar minimizer differs")
            if not trials[-1]["accepted"] and len(trials) != 6:
                raise ValueError("actual ray abandoned its frozen ladder early")
        quality_calls = 1 + len(trials) + int(replay_present)
        q_calls = int(q_stage)
        native = int(direction == "native_gradient")
        expected_counts = {"quality_evaluations": quality_calls, "raw_coverage_passes": quality_calls,
            "gaussian_renders": quality_calls + q_calls + native, "actual_direction_render_calls": q_calls,
            "native_gradient_forward_calls": native, "native_color_vjp_calls": native,
            "basis_denominator_passes": int(direction.startswith("streaming")),
            "basis_transpose_calls": int(direction.startswith("streaming")),
            "basis_diagonal_passes": int(direction == "streaming_jacobi"),
            "basis_apply_calls": 0, "legacy_cg_iterations": 0}
        if direction == "cg":
            cg = metadata["legacy_cg"]
            calls, iters = cg["normal_matvec_calls"], cg["iterations"]
            if not 0 <= iters <= 32 or calls not in (iters + 1, iters + 2) or calls > 33:
                raise ValueError("actual CG work horizon differs")
            if cg["denominator_calls"] != 1 or cg["basis_apply_calls"] != calls or cg["basis_transpose_calls"] != calls + 1:
                raise ValueError("actual CG operator ledger differs")
            expected_counts.update({"basis_denominator_passes": 1, "basis_apply_calls": calls,
                                    "basis_transpose_calls": calls + 1, "legacy_cg_iterations": iters})
    if expected_counts != metadata["counts"]:
        raise ValueError(f"exact work ledger differs: expected {expected_counts}")


def validate_artifacts(root, rows, protocol, problems, *, diagnostic=False):
    """CPU replay of parent identities, raw-quality gates, field deltas, work and disposition."""
    import torch
    from datetime import datetime
    from structsplat.config import FitConfig, StructureTensorConfig
    from structsplat.gaussians import GaussianField
    root = Path(root)
    parents, samples = {}, []
    try:
        if not diagnostic:
            if sha256(root / "parent_source_manifest.json") != PARENT_MANIFEST_SHA256:
                raise ValueError("source manifest no longer matches the frozen import")
            transfer = json.loads((root / "parent_transfer.json").read_text())
            if (transfer["files"] != PARENT_FILES or transfer["source_commit"] != PARENT_SOURCE_COMMIT
                    or transfer["source_protocol_digest"] != PARENT_PROTOCOL_DIGEST
                    or transfer["source_manifest_sha256"] != PARENT_MANIFEST_SHA256):
                raise ValueError("parent transfer receipt differs")
            for name, expected in PARENT_FILES.items():
                if sha256(root / name) != expected:
                    raise ValueError(f"imported parent payload differs: {name}")
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"FIT-051 imported source: {exc}")
    workloads = [(-1, 77)] if diagnostic else [(image, seed) for image in IMAGE_IDS for seed in SEEDS]
    for image, seed in workloads:
        pid = "procedural_s77" if diagnostic else parent_id(image, seed)
        directory = root / "parents" / pid
        try:
            config = json.loads((directory / "config.json").read_text())
            init, fit_cfg = parent_configs(seed, smoke=diagnostic, device="cpu" if diagnostic else "cuda")
            if (config["init"] != _json_form(asdict(init)) or config["fit"] != _json_form(asdict(fit_cfg))
                    or config["tensor"] != _json_form(asdict(StructureTensorConfig()))
                    or config["image_id"] != image or config["seed"] != seed):
                raise ValueError("imported parent resolved config differs")
            parent = GaussianField.load(directory / "field.npz", device="cpu")
            target_np = np.load(directory / "target.npy", allow_pickle=False)
            count, steps = (16, 3) if diagnostic else (2000, 750)
            if parent.n != count or target_np.dtype != np.float32 or target_np.ndim != 3 or target_np.shape[2] != 3 or not np.isfinite(target_np).all():
                raise ValueError("imported parent target/count/dtype differs")
            target = torch.from_numpy(target_np)
            context = full_frame_context(target, fit_cfg, parent.n)
            hist = json.loads((directory / "history.json").read_text())
            expected_iter = sorted(set(range(0, steps, fit_cfg.log_every)) | {steps-1})
            if hist["iter"] != expected_iter or hist["n_gaussians"] != [count] * len(expected_iter):
                raise ValueError("imported parent horizon/count differs")
            state = torch.load(directory / "optimizer_state.pt", map_location="cpu", weights_only=True)
            if len(state["state"]) != 4 or len(state["param_groups"]) != 4:
                raise ValueError("imported parent Adam state inventory differs")
            for group, tensor in zip(state["param_groups"], (parent.means, parent.log_scales, parent.rotations, parent.colors)):
                entry = state["state"][group["params"][0]]
                if float(entry["step"]) != steps or any(entry[key].shape != tensor.shape or not bool(torch.isfinite(entry[key]).all()) for key in ("exp_avg", "exp_avg_sq")):
                    raise ValueError("imported parent Adam horizon/moments differ")
            occupancy = [json.loads(line) for line in (directory / "occupancy.jsonl").read_text().splitlines() if line]
            if not diagnostic and not occupancy:
                raise ValueError("new worker lacks occupancy sampling")
            previous = None
            for sample in occupancy:
                timestamp = datetime.fromisoformat(sample["utc"])
                if timestamp.tzinfo is None or (previous is not None and timestamp < previous):
                    raise ValueError("occupancy timestamp ordering differs")
                previous = timestamp
                if sample["ok"]:
                    if sample["foreign_pids"] != [p["pid"] for p in sample["processes"] if p["pid"] != sample["owned_worker_pid"]]:
                        raise ValueError("occupancy qualification differs from raw processes")
                elif not sample.get("error"):
                    raise ValueError("occupancy failure lacks error")
            samples.extend(occupancy)
            parents[pid] = (parent, target, FitConfig(**config["fit"]), context, config, occupancy)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
            problems.append(f"FIT-051 parent {pid}: {exc}")
    for row in rows:
        if row.get("status") != "ok" or row.get("parent_id") not in parents:
            continue
        try:
            directory = root / "cells" / row["cell_id"]
            parent, target, cfg, context, parent_config, occupancy = parents[row["parent_id"]]
            config = json.loads((directory / "config.json").read_text())
            request = json.loads((directory / "request.json").read_text())
            expected_request = {key: row[key] for key in ("cell_id", "parent_id", "image_id", "seed", "smoke", "method", "parent_field_sha256", "parent_optimizer_sha256", "target_sha256")}
            if request != expected_request or config["request"] != request or row["smoke"] != diagnostic:
                raise ValueError("row/request/config identity differs")
            if (config["fit"] != parent_config["fit"] or config["schedule"] != _json_form(asdict(context[2]))
                    or config["parent_config"] != f"../../parents/{row['parent_id']}/config.json"):
                raise ValueError("candidate fitter/schedule/tolerances differ")
            if not diagnostic and (config["environment"]["torch"] != protocol["torch"]
                    or config["environment"]["gpu"] != protocol["gpu"]
                    or any(s["owned_worker_pid"] != config["environment"]["worker_pid"] for s in occupancy)):
                raise ValueError("candidate worker environment/occupancy owner differs")
            if sha256(directory / "input_field.npz") != row["parent_field_sha256"]:
                raise ValueError("copied input field differs from imported parent")
            selected = GaussianField.load(directory / "field.npz", device="cpu")
            if selected.n != parent.n:
                raise ValueError("saved selected count differs")
            history = json.loads((directory / "history.json").read_text())
            metadata = history["transaction"]
            if history["parent_history"] != f"../../parents/{row['parent_id']}/history.json":
                raise ValueError("parent history link differs")
            with np.load(directory / "transaction_arrays.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            inventory = json.loads((directory / "tensor_inventory.json").read_text())
            declared_arrays = [entry for value in inventory.values() if value is not None
                               for entry in (value if isinstance(value, list) else [value])]
            if len(declared_arrays) != len(set(declared_arrays)) or set(declared_arrays) != set(arrays):
                raise ValueError("raw tensor inventory differs")
            candidate_path = directory / "candidate_field.npz"
            candidate = GaussianField.load(candidate_path, device="cpu") if candidate_path.exists() else None
            _validate_mechanism(parent, selected, target, cfg, context, row, metadata, arrays, inventory,
                                candidate, protocol, diagnostic)
            for prefix, quality in (("reporting", row["reporting_metrics"]), ("cold", row["cold_metrics"])):
                _raw_quality(arrays[prefix + "_render"], arrays[prefix + "_denominator"], target, cfg, context, quality, protocol)
                if history[prefix + "_quality"] != quality:
                    raise ValueError("reader quality disagrees with history")
            reference_key = ("replay_render" if row["method"] in DIRECTIONS else "candidate_render") if row["accepted"] else "parent_render"
            selected_error = float(np.max(np.abs(arrays["reporting_render"] - arrays[reference_key])))
            cold_error = float(np.max(np.abs(arrays["reporting_render"] - arrays["cold_render"])))
            if selected_error != row["selected_replay_max_abs"] or cold_error != row["cold_render_max_abs"]:
                raise ValueError("selected/cold raw replay error differs")
            raw = np.load(directory / "reconstruction.npy", allow_pickle=False)
            if not np.array_equal(raw, arrays["reporting_render"]):
                raise ValueError("reported reconstruction differs from selected raw array")
            metrics = endpoint_score(torch.from_numpy(raw), target, smoke=diagnostic)
            if any(not _close(value, row[key], protocol) for key, value in metrics.items() if value is not None):
                raise ValueError("raw/display endpoint metrics differ from saved reconstruction")
            evaluations = [("parent", arrays["parent_render"], metadata["parent_metrics"], 0.0)]
            if row["method"] in DIRECTIONS:
                evaluations.extend(("actual_trial", arrays[f"trial_renders_{index}"], trial["actual_metrics"],
                                    trial["transaction_elapsed_seconds"])
                                   for index, trial in enumerate(metadata["trials"]))
                if metadata["replay_metrics"] is not None:
                    evaluations.append(("selected_actual_replay", arrays["replay_render"],
                                        metadata["replay_metrics"], metadata["transaction_seconds"]))
            elif row["method"] != "noop":
                evaluations.append(("endpoint_candidate", arrays["candidate_render"],
                                    metadata["candidate_metrics"], metadata["transaction_seconds"]))
            evaluations.append(("reporting", arrays["reporting_render"], row["reporting_metrics"],
                                metadata["transaction_seconds"]))
            checkpoints = history["checkpoints"]
            if len(checkpoints) != len(evaluations):
                raise ValueError("curve omitted or invented an actual quality evaluation")
            for index, (point, (stage, image, quality, elapsed)) in enumerate(zip(checkpoints, evaluations)):
                if (point["evaluation_index"] != index or point["stage"] != stage
                        or point["elapsed_seconds"] != elapsed
                        or any(point[key] != value for key, value in quality.items())):
                    raise ValueError("curve stage/index/time/protected metrics differ")
                rescored = scored_point(torch.from_numpy(image), target, quality, index, elapsed,
                                         diagnostic=diagnostic, stage=stage)
                for key in ("psnr", "raw_mse", "mae", "ssim", "ms_ssim", "lpips"):
                    if rescored[key] is None:
                        if point[key] is not None:
                            raise ValueError("diagnostic curve invented an unavailable metric")
                    elif not _close(rescored[key], point[key], protocol):
                        raise ValueError(f"curve {key} differs from its saved actual image")
            if history["reporting_work"] != {"quality_evaluations": 2, "gaussian_renders": 2, "raw_coverage_passes": 2}:
                raise ValueError("reader work accounting differs")
            if row["method"] == "adam32":
                steps = 2 if diagnostic else 32
                if history["adam_fit"]["iter"] != list(range(steps)) or history["adam_fit"]["n_gaussians"] != [parent.n] * steps:
                    raise ValueError("Adam continuation native horizon/count differs")
            elif history["adam_fit"]:
                raise ValueError("non-Adam arm has optimizer continuation history")
        except (OSError, KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
            problems.append(f"FIT-051 cell {row.get('cell_id')}: {exc}")
    try:
        occupancy = json.loads((root / "occupancy.json").read_text())
        eligible = bool(samples) and all(s["ok"] and not s["foreign_pids"] for s in samples)
        if occupancy["samples"] != samples or occupancy["timing_eligible"] != eligible:
            raise ValueError("complete occupancy ledger differs")
        decision = summarize(rows, protocol, diagnostic=diagnostic)
        decision["timing_eligible"] = eligible
        if decision != json.loads((root / "decision.json").read_text()):
            raise ValueError("saved decision does not replay from complete rows and occupancy")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        problems.append(f"FIT-051 decision: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parent-bundle", type=Path)
    parser.add_argument("--approved-protocol-digest")
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--print-protocol-digest", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", type=Path)
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
    if not args.out or (not args.smoke and (args.parent_bundle is None or args.approved_protocol_digest != digest)):
        parser.error("--out, --parent-bundle and exact approved digest required for formal execution")
    if not args.smoke:
        issues = validate_bundle(args.parent_bundle.resolve())
        if issues:
            raise RuntimeError("frozen source bundle is invalid: " + "; ".join(issues))
    bundle = ResearchBundle(args.out, task="FIT-051", protocol=PROTOCOL, digest=digest,
        expected_cells=expected_cells(PROTOCOL, args.smoke), source_paths=SOURCES, diagnostic=args.smoke)
    if args.smoke:
        make_diagnostic_parent(bundle.root / "parents/procedural_s77")
    else:
        transfer_parents(args.parent_bundle.resolve(), bundle.root)
    rows, samples = [], []
    workloads = [(-1, 77)] if args.smoke else [(image, seed) for image in IMAGE_IDS for seed in SEEDS]
    for index, (image, seed) in enumerate(workloads):
        pid = "procedural_s77" if args.smoke else parent_id(image, seed)
        directory = bundle.root / "parents" / pid
        shift = index % len(ARMS)
        request = {"root": str(bundle.root), "parent_id": pid, "image_id": image, "seed": seed,
                   "smoke": args.smoke, "arm_order": ARMS[shift:] + ARMS[:shift]}
        write_json(directory / "request.json", request)
        with (directory / "worker.log").open("w") as log, (directory / "occupancy.jsonl").open("w") as occupancy:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(directory / "request.json")],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while process.poll() is None:
                if not args.smoke:
                    sample = gpu_snapshot(process.pid)
                    samples.append(sample)
                    occupancy.write(json.dumps(sample) + "\n")
                    occupancy.flush()
                if time.monotonic() - started > PROTOCOL["worker_timeout_seconds"]:
                    process.terminate()
                    process.wait(timeout=30)
                    break
                time.sleep(1)
        for arm in ARMS:
            cell = bundle.root / "cells" / f"{pid}_{arm}" / "row.json"
            if cell.exists():
                rows.append(json.loads(cell.read_text()))
            else:
                rows.append({"cell_id": f"{pid}_{arm}", "parent_id": pid, "image_id": image, "seed": seed,
                    "method": arm, "status": "error", "error": f"worker exit {process.returncode}; row missing",
                    "artifacts": {"worker.log": f"parents/{pid}/worker.log"}})
        print(json.dumps({"parent_id": pid, "completed_workers": index + 1}), flush=True)
    eligible = bool(samples) and all(sample["ok"] and not sample["foreign_pids"] for sample in samples)
    write_json(bundle.root / "occupancy.json", {"samples": samples, "timing_eligible": eligible,
        "qualification": "sampled qualification only, not continuous exclusivity; no speed claim"})
    decision = summarize(rows, diagnostic=args.smoke)
    decision["timing_eligible"] = eligible
    write_json(bundle.root / "decision.json", decision)
    for row in rows:
        row.setdefault("artifacts", {}).update({"decision": "decision.json", "occupancy": "occupancy.json"})
    bundle.finish(rows, title="FIT-051 — Actual-render color transactions",
        interpretation="All eight original exposed FIT-050 parents, seven independently charged one-transaction arms. Actual trial images replace cross-backend interpolation; protected gate unchanged. Original evidence is retained. No default or speed claim; raw trial arrays and original histories are linked.")
    problems = validate_bundle(bundle.root, allow_dirty=args.smoke)
    if problems:
        print(json.dumps({"bundle_validation_errors": problems}), flush=True)
        raise SystemExit(1)
    print(json.dumps({"bundle_validated": str(bundle.root)}), flush=True)


if __name__ == "__main__":
    main()
