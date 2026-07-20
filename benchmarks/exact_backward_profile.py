"""Source-bound CUDA-event profile of StructSplat's exact normalized backward.

This follow-on experiment compares the unchanged atomic baseline with an opt-in warp/shared-memory
block reduction. It separates exact normalized forward, exact renderer backward, the complete
loss-backward phase, Adam update, and a representative fitter step on a deterministic synthetic
field. Timings are current-stream CUDA-event medians; they exclude Python launch overhead,
extension compilation, input construction, and synchronization wait time. PyTorch peak allocator
memory is reported separately.

The optimized gate is frozen in :data:`PREREGISTERED_GATE` before candidate timing: established
parity, at least 10% exact-backward reduction, at least 3% representative-step reduction, no more
than 5% peak allocator-memory regression, stable repeats at the named representative cell, and
nonnegative exact-backward and representative-step direction at every frozen grid cell. Passing
supports only this source/device-bound microprofile claim and keeps the selector opt-in.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import torch

from structsplat import metrics as M
from structsplat.gaussians import GaussianField
from structsplat.render import _tile_bounds, render_field


PROTOCOL = "exact-normalized-CUDA-backward-block-reduce-v1"
WORK_SCOPE = "synthetic_microfit_cuda_event_device_timeline_only"
DEFAULT_RESOLUTIONS = (128, 256)
DEFAULT_COUNTS = (512, 2048)
DEFAULT_OVERLAPS = (4.0, 16.0)
DEFAULT_MODES = ("cuda", "cuda_block_reduce")
SIGMA_CUTOFF = 3.0
SUPPORT_FADE = True
PARITY_ATOL = 5e-4
PARITY_RTOL = 5e-4

# Frozen before the first optimized timing run. Do not tune these fields after seeing output.
PREREGISTERED_GATE: dict[str, Any] = {
    "registered_before_optimized_measurement": True,
    "representative_cell": {
        "height": 256,
        "width": 256,
        "n_gaussians": 2048,
        "requested_support_overlap": 16.0,
    },
    "baseline_renderer": "cuda",
    "candidate_renderer": "cuda_block_reduce",
    "parity_atol": PARITY_ATOL,
    "parity_rtol": PARITY_RTOL,
    "minimum_exact_backward_reduction_fraction": 0.10,
    "minimum_representative_step_reduction_fraction": 0.03,
    "minimum_grid_direction_reduction_fraction": 0.0,
    "maximum_peak_incremental_allocator_memory_regression_fraction": 0.05,
    "stability_statistic": "population standard deviation / mean over raw CUDA-event repeats",
    "maximum_representative_timing_cv": 0.05,
    "pass_scope": "source_device_bound_microprofile_only_selector_remains_opt_in",
    "default_flip_authorized": False,
    "quality_or_compression_claim_authorized": False,
}

_CRITICAL_SOURCE_PATHS = (
    "benchmarks/exact_backward_profile.py",
    "src/structsplat/cuda_render.py",
    "src/structsplat/cuda/render_ext.cpp",
    "src/structsplat/cuda/render_ext.cu",
    "src/structsplat/render.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/fit.py",
    "src/structsplat/metrics.py",
)


@dataclass(frozen=True)
class CellSpec:
    height: int
    width: int
    n_gaussians: int
    requested_support_overlap: float
    seed: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_path(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        kind = "symlink_target"
    else:
        payload = path.read_bytes()
        kind = "file_bytes"
    return {"kind": kind, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def collect_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Hash exact worktree sources independently of their tracked/untracked status."""
    root = repo_root or Path(__file__).resolve().parents[1]
    status = _git_bytes(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked_diff = _git_bytes(root, "diff", "--binary", "HEAD", "--", ".")
    sources = {relative: _hash_path(root / relative) for relative in _CRITICAL_SOURCE_PATHS}
    return {
        "git_commit": _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip(),
        "git_branch": (
            _git_bytes(root, "branch", "--show-current").decode("utf-8").strip() or None
        ),
        "git_dirty": bool(status),
        "git_status_porcelain_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "tracked_diff_size_bytes": len(tracked_diff),
        "critical_source_hash_mode": "raw_worktree_bytes_independent_of_git_index",
        "critical_sources": sources,
    }


def inspect_exact_backward_source(source: str) -> dict[str, Any]:
    """Extract baseline atomic and candidate reduction structure from live CUDA source."""
    baseline_start = source.index("__global__ void backward_kernel(")
    candidate_start = source.index("__global__ void backward_block_reduce_kernel(")
    tiled_start = source.index("__global__ void tiled_backward_kernel(", candidate_start)
    baseline_kernel = source[baseline_start:candidate_start]
    candidate_kernel = source[candidate_start:tiled_start]
    baseline_launch_start = source.index("structsplat_render_backward_cuda(", tiled_start)
    candidate_launch_start = source.index(
        "structsplat_render_backward_block_reduce_cuda(", baseline_launch_start
    )
    tiled_launch_start = source.index(
        "structsplat_render_backward_tiled_cuda(", candidate_launch_start
    )
    baseline_launcher = source[baseline_launch_start:candidate_launch_start]
    candidate_launcher = source[candidate_launch_start:tiled_launch_start]
    baseline_thread_match = re.search(
        r"constexpr int threads\s*=\s*(\d+)\s*;", baseline_launcher
    )
    candidate_thread_match = re.search(
        r"constexpr int threads\s*=\s*(\d+)\s*;", candidate_launcher
    )
    atomic_targets = re.findall(r"atomicAdd\(&grad_(\w+)\[", baseline_kernel)
    target_counts = {name: atomic_targets.count(name) for name in sorted(set(atomic_targets))}
    baseline_threads = int(baseline_thread_match.group(1)) if baseline_thread_match else None
    candidate_threads = int(candidate_thread_match.group(1)) if candidate_thread_match else None
    expected_counts = {"means": 2, "conics": 3, "colors": 3, "opacities": 1}
    baseline_one_block = "int i = blockIdx.x;" in baseline_kernel
    baseline_local_loop = (
        "for (int t = threadIdx.x; t < total; t += blockDim.x)" in baseline_kernel
    )
    same_gaussian_addresses = all(
        fragment in baseline_kernel
        for fragment in (
            "grad_means[i * 2",
            "grad_conics[i * 3",
            "grad_colors[i * 3",
            "grad_opacities[i]",
        )
    )
    baseline_pattern = bool(
        baseline_one_block
        and baseline_local_loop
        and baseline_threads == 256
        and target_counts == expected_counts
        and same_gaussian_addresses
    )
    candidate_one_block = "int i = blockIdx.x;" in candidate_kernel
    candidate_local_loop = (
        "for (int t = threadIdx.x; t < total; t += blockDim.x)" in candidate_kernel
    )
    candidate_atomic_count = len(re.findall(r"atomicAdd\(", candidate_kernel))
    candidate_direct_writes = all(
        fragment in candidate_kernel
        for fragment in (
            "grad_means[i * 2 + 0] =",
            "grad_means[i * 2 + 1] =",
            "grad_conics[i * 3 + 0] =",
            "grad_conics[i * 3 + 1] =",
            "grad_conics[i * 3 + 2] =",
            "grad_colors[i * 3 + 0] =",
            "grad_colors[i * 3 + 1] =",
            "grad_colors[i * 3 + 2] =",
            "grad_opacities[i] =",
        )
    )
    candidate_pattern = bool(
        candidate_one_block
        and candidate_local_loop
        and candidate_threads == 256
        and "warp_reduce_sum" in candidate_kernel
        and "__shared__ float warp_sums" in candidate_kernel
        and candidate_atomic_count == 0
        and candidate_direct_writes
    )
    return {
        "baseline": {
            "kernel": "backward_kernel",
            "one_block_per_gaussian": baseline_one_block,
            "threads_per_block": baseline_threads,
            "thread_local_pixel_loop": baseline_local_loop,
            "atomic_callsite_count": len(atomic_targets),
            "atomic_callsite_counts_by_gradient": target_counts,
            "same_gaussian_gradient_addresses_across_threads": same_gaussian_addresses,
            "actionable_within_block_reduction_pattern": baseline_pattern,
        },
        "block_reduce": {
            "kernel": "backward_block_reduce_kernel",
            "one_block_per_gaussian": candidate_one_block,
            "threads_per_block": candidate_threads,
            "thread_local_pixel_loop": candidate_local_loop,
            "uses_warp_shuffle_reduction": "warp_reduce_sum" in candidate_kernel,
            "uses_shared_warp_sums": "__shared__ float warp_sums" in candidate_kernel,
            "atomic_callsite_count": candidate_atomic_count,
            "direct_gradient_component_writes": candidate_direct_writes,
            "implemented_pattern_matches_experiment": candidate_pattern,
        },
        "comparison_ready": baseline_pattern and candidate_pattern,
        "evidence_kind": "executed_source_structure_not_hardware_counter",
    }


def _run_text(command: list[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
    }


def probe_ncu(device_index: int = 0) -> dict[str, Any]:
    """Check whether Nsight Compute counters are usable without privilege escalation."""
    executable = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
    if executable is None:
        return {
            "status": "unavailable_binary_missing",
            "available_noninteractive": False,
            "executable": None,
        }
    version = _run_text([executable, "--version"])
    query = _run_text([executable, "--query-metrics", "--devices", str(device_index)])
    combined = f"{query['stdout']}\n{query['stderr']}"
    if "ERR_NVGPUCTRPERM" in combined or query["returncode"] != 0:
        reason = (
            "gpu_performance_counter_permission_denied"
            if "ERR_NVGPUCTRPERM" in combined
            else "query_metrics_failed"
        )
        return {
            "status": f"unavailable_{reason}",
            "available_noninteractive": False,
            "executable": executable,
            "version": version,
            "query": query,
        }
    return {
        "status": "available_noninteractive",
        "available_noninteractive": True,
        "executable": executable,
        "version": version,
        "query_metrics_sha256": _sha256_bytes(combined.encode("utf-8")),
    }


def collect_ncu_evidence(
    probe: dict[str, Any], outdir: Path, representative: dict[str, Any]
) -> dict[str, Any]:
    """Collect one bounded backward-kernel report when counters are accessible."""
    if not probe.get("available_noninteractive"):
        return {
            "status": probe["status"],
            "collected": False,
            "reason": "Nsight Compute counters were not accessible noninteractively.",
            "probe": probe,
        }
    raw_report = (outdir / "ncu_backward_block_reduce").resolve()
    executable = str(probe["executable"])
    cell = representative
    command = [
        executable,
        "--force-overwrite",
        "--export",
        str(raw_report),
        "--target-processes",
        "application-only",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        "regex:backward_block_reduce_kernel",
        "--launch-count",
        "1",
        "--section",
        "MemoryWorkloadAnalysis",
        "--section",
        "InstructionStats",
        sys.executable,
        "-m",
        "benchmarks.exact_backward_profile",
        "--ncu-target",
        "--height",
        str(cell["height"]),
        "--width",
        str(cell["width"]),
        "--n-gaussians",
        str(cell["n_gaussians"]),
        "--requested-overlap",
        str(cell["requested_support_overlap"]),
    ]
    result = _run_text(command, timeout=120)
    report_path = raw_report.with_suffix(".ncu-rep")
    return {
        "status": "collected" if result["returncode"] == 0 and report_path.exists() else "failed",
        "collected": bool(result["returncode"] == 0 and report_path.exists()),
        "command": command,
        "result": result,
        "report_path": str(report_path) if report_path.exists() else None,
        "report_sha256": _hash_path(report_path)["sha256"] if report_path.exists() else None,
        "scope": (
            "one representative block-reduced untiled backward launch; "
            "memory/instruction sections"
        ),
    }


def collect_environment(device_index: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device_index)
    sm_count = getattr(properties, "multi_processor_count", None)
    driver = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,uuid,memory.total",
            "--format=csv,noheader,nounits",
            "--id",
            str(device_index),
        ]
    )
    nvcc_path = shutil.which("nvcc")
    nvcc = _run_text([nvcc_path, "--version"]) if nvcc_path else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "multiprocessor_count": sm_count,
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "nvidia_smi": driver,
        "nvcc": nvcc,
    }


def requested_scale(spec: CellSpec, sigma_cutoff: float = SIGMA_CUTOFF) -> float:
    """Choose a support scale from requested mean AABB visits per pixel."""
    area_per_gaussian = (
        spec.requested_support_overlap * spec.height * spec.width / spec.n_gaussians
    )
    requested_radius = max(1.0, (math.sqrt(area_per_gaussian) - 1.0) / 2.0)
    return max(0.45, requested_radius / sigma_cutoff)


def make_field_arrays(spec: CellSpec) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    base_scale = requested_scale(spec)
    means = np.stack(
        [
            rng.uniform(0.0, max(0.0, spec.width - 1.0), spec.n_gaussians),
            rng.uniform(0.0, max(0.0, spec.height - 1.0), spec.n_gaussians),
        ],
        axis=1,
    ).astype(np.float32)
    anisotropy = rng.uniform(0.85, 1.25, spec.n_gaussians).astype(np.float32)
    scales = np.stack(
        [base_scale * anisotropy, base_scale / anisotropy], axis=1
    ).astype(np.float32)
    angles = rng.uniform(0.0, math.pi, spec.n_gaussians).astype(np.float32)
    colors = rng.uniform(0.05, 0.95, (spec.n_gaussians, 3)).astype(np.float32)
    opacity_values = rng.uniform(0.35, 0.9, spec.n_gaussians).astype(np.float32)
    opacity_logits = np.log(opacity_values) - np.log1p(-opacity_values)
    return {
        "means": means,
        "scales": scales,
        "angles": angles,
        "colors": colors,
        "opacity_logits": opacity_logits.astype(np.float32),
    }


def make_field(arrays: dict[str, np.ndarray], device: torch.device | str) -> GaussianField:
    return GaussianField.from_numpy(
        arrays["means"],
        arrays["scales"],
        arrays["angles"],
        arrays["colors"],
        opacities=arrays["opacity_logits"],
        device=device,
    ).trainable()


def actual_support_statistics(spec: CellSpec, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    field = make_field(arrays, "cpu")
    radii = field.radii(SIGMA_CUTOFF)
    _, _, _, visits = _tile_bounds(field.means, radii, spec.height, spec.width)
    total_visits = int(visits.sum().item())
    return {
        "support_aabb_visits": total_visits,
        "actual_support_overlap": total_visits / float(spec.height * spec.width),
        "mean_support_aabb_area": total_visits / float(spec.n_gaussians),
        "mean_radius_x": float(radii[:, 0].float().mean()),
        "mean_radius_y": float(radii[:, 1].float().mean()),
        "evidence_kind": "deterministic_support_work_proxy_not_hardware_counter",
    }


def make_target(spec: CellSpec, device: torch.device | str) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(spec.height, device=device, dtype=torch.float32),
        torch.arange(spec.width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    nx = xx / max(1.0, float(spec.width - 1))
    ny = yy / max(1.0, float(spec.height - 1))
    return torch.stack(
        (
            0.5 + 0.25 * torch.sin(2.0 * math.pi * nx),
            0.5 + 0.25 * torch.cos(2.0 * math.pi * ny),
            0.25 + 0.5 * nx * ny,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def render_exact(field: GaussianField, spec: CellSpec, mode: str) -> torch.Tensor:
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(SIGMA_CUTOFF),
        spec.height,
        spec.width,
        mode=mode,
        opacities=field.opacity_values(),
        support_fade=SUPPORT_FADE,
        sigma_cutoff=SIGMA_CUTOFF,
    )


def representative_loss(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mirror the default FitConfig L1/SSIM mixture without unrelated fit controls."""
    pixel = (image - target).abs().mean()
    similarity = M.ssim(image, target, backend="builtin")
    return 0.7 * pixel + 0.3 * (1.0 - similarity)


def make_optimizer(field: GaussianField) -> torch.optim.Optimizer:
    groups = field.parameter_groups(5e-2, 3e-2, 1e-2, 3e-2, 1e-2)
    return torch.optim.Adam(groups)


def _parameter_snapshots(field: GaussianField) -> list[tuple[torch.Tensor, torch.Tensor]]:
    values = [field.means, field.log_scales, field.rotations, field.colors]
    if field.opacities is not None:
        values.append(field.opacities)
    return [(value, value.detach().clone()) for value in values]


@torch.no_grad()
def _restore_parameters(snapshots: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for parameter, frozen in snapshots:
        parameter.copy_(frozen)


def _event_ms(action: Callable[[], Any]) -> tuple[float, Any]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = action()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), value


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("timing sample list must not be empty")
    return float(statistics.median(values))


def _population_cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if mean <= 0.0:
        return math.inf
    return float(statistics.pstdev(values) / mean)


def _profile_mode(
    spec: CellSpec,
    arrays: dict[str, np.ndarray],
    target: torch.Tensor,
    mode: str,
    *,
    warmup: int,
    repeats: int,
    device_index: int,
    atomic_structure: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = torch.device(f"cuda:{device_index}")
    field = make_field(arrays, device)
    optimizer = make_optimizer(field)
    snapshots = _parameter_snapshots(field)
    grad_out = (target - 0.5).contiguous()

    def full_step() -> torch.Tensor:
        image = render_exact(field, spec, mode)
        loss = representative_loss(image, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(warmup):
        full_step()
        torch.cuda.synchronize()
        _restore_parameters(snapshots)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    samples: dict[str, list[float]] = {
        "exact_forward": [],
        "exact_backward": [],
        "fit_backward": [],
        "optimizer_update": [],
        "representative_fit_step": [],
    }

    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        elapsed, image = _event_ms(lambda: render_exact(field, spec, mode))
        samples["exact_forward"].append(elapsed)
        del image

    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        image = render_exact(field, spec, mode)
        torch.cuda.synchronize()
        elapsed, _ = _event_ms(lambda current=image: current.backward(grad_out))
        samples["exact_backward"].append(elapsed)
        del image

    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        image = render_exact(field, spec, mode)
        loss = representative_loss(image, target)
        torch.cuda.synchronize()
        elapsed, _ = _event_ms(loss.backward)
        samples["fit_backward"].append(elapsed)
        del image, loss

    # Populate steady optimizer state and a representative gradient once outside the update timer.
    optimizer.zero_grad(set_to_none=True)
    update_image = render_exact(field, spec, mode)
    representative_loss(update_image, target).backward()
    torch.cuda.synchronize()
    del update_image
    for _ in range(repeats):
        elapsed, _ = _event_ms(optimizer.step)
        samples["optimizer_update"].append(elapsed)
        _restore_parameters(snapshots)

    for _ in range(repeats):
        elapsed, loss = _event_ms(full_step)
        samples["representative_fit_step"].append(elapsed)
        _restore_parameters(snapshots)
        del loss

    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    medians = {f"{phase}_median_ms": _median(values) for phase, values in samples.items()}
    stability = {f"{phase}_cv": _population_cv(values) for phase, values in samples.items()}
    backward_share = (
        medians["exact_backward_median_ms"]
        / medians["representative_fit_step_median_ms"]
    )
    opacity_components = 1 if field.opacities is not None else 0
    source_atomic_model = None
    if mode == "cuda" and atomic_structure["baseline"][
        "actionable_within_block_reduction_pattern"
    ]:
        source_atomic_model = (
            spec.n_gaussians
            * int(atomic_structure["baseline"]["threads_per_block"])
            * (8 + opacity_components)
        )
    elif mode == "cuda_block_reduce" and atomic_structure["block_reduce"][
        "implemented_pattern_matches_experiment"
    ]:
        source_atomic_model = 0
    row: dict[str, Any] = {
        **asdict(spec),
        "mode": mode,
        "status": "ok",
        "work_scope": WORK_SCOPE,
        "warmup": warmup,
        "repeats": repeats,
        **medians,
        **stability,
        "exact_backward_share_of_representative_step": backward_share,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_incremental_allocated_bytes": max(0, peak_allocated - baseline_allocated),
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "source_atomic_instruction_model": source_atomic_model,
        "source_atomic_instruction_model_scope": (
            "untiled source-issued call model, not a hardware counter or speedup estimate"
            if source_atomic_model is not None
            else None
        ),
        "microprofile_speedup_claim_authorized": False,
    }
    raw_rows = [
        {
            **asdict(spec),
            "mode": mode,
            "phase": phase,
            "repeat": repeat,
            "cuda_event_ms": value,
            "work_scope": WORK_SCOPE,
        }
        for phase, values in samples.items()
        for repeat, value in enumerate(values)
    ]
    gc.collect()
    torch.cuda.empty_cache()
    return row, raw_rows


def _run_field_once(
    spec: CellSpec,
    arrays: dict[str, np.ndarray],
    target: torch.Tensor,
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    field = make_field(arrays, device)
    image = render_exact(field, spec, mode)
    representative_loss(image, target).backward()
    torch.cuda.synchronize()
    parameters = [field.means, field.log_scales, field.rotations, field.colors]
    if field.opacities is not None:
        parameters.append(field.opacities)
    gradients = [parameter.grad.detach().clone() for parameter in parameters]
    return image.detach().clone(), gradients


def parity_between_modes(
    spec: CellSpec,
    arrays: dict[str, np.ndarray],
    target: torch.Tensor,
    first_mode: str,
    second_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    first_image, first_gradients = _run_field_once(spec, arrays, target, first_mode, device)
    second_image, second_gradients = _run_field_once(spec, arrays, target, second_mode, device)
    image_max_abs = float((first_image - second_image).abs().max().detach().cpu())
    gradient_max_abs = max(
        float((first - second).abs().max().detach().cpu())
        for first, second in zip(first_gradients, second_gradients, strict=True)
    )
    passed = bool(
        torch.allclose(first_image, second_image, atol=PARITY_ATOL, rtol=PARITY_RTOL)
        and all(
            torch.allclose(first, second, atol=PARITY_ATOL, rtol=PARITY_RTOL)
            for first, second in zip(first_gradients, second_gradients, strict=True)
        )
    )
    del first_image, second_image, first_gradients, second_gradients
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "first_mode": first_mode,
        "second_mode": second_mode,
        "passed": passed,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "image_max_abs": image_max_abs,
        "gradient_max_abs": gradient_max_abs,
        "comparison_scope": "same normalized equation, constant RGB, opacity, support fade",
    }


def reference_parity_calibration(device_index: int) -> dict[str, Any]:
    spec = CellSpec(64, 64, 64, 4.0, 991)
    arrays = make_field_arrays(spec)
    device = torch.device(f"cuda:{device_index}")
    target = make_target(spec, device)
    return {
        "scope": "small correctness calibration only; reference timing is not compared",
        "reference_vs_cuda": parity_between_modes(
            spec, arrays, target, "normalized", "cuda", device
        ),
        "reference_vs_cuda_block_reduce": parity_between_modes(
            spec, arrays, target, "normalized", "cuda_block_reduce", device
        ),
    }


def attach_pair_ratios(
    rows: list[dict[str, Any]], parity: dict[tuple[int, int, int, float], dict[str, Any]]
) -> None:
    by_cell: dict[tuple[int, int, int, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["height"],
            row["width"],
            row["n_gaussians"],
            row["requested_support_overlap"],
        )
        by_cell.setdefault(key, {})[row["mode"]] = row
    for key, pair in by_cell.items():
        valid = bool(parity[key]["passed"])
        baseline = pair.get("cuda")
        candidate = pair.get("cuda_block_reduce")
        for row in pair.values():
            row["baseline_vs_block_reduce_semantic_parity_passed"] = valid
            row["candidate_over_baseline_exact_backward_time_ratio"] = None
            row["exact_backward_reduction_fraction"] = None
            row["candidate_over_baseline_representative_step_time_ratio"] = None
            row["representative_step_reduction_fraction"] = None
            row["peak_incremental_allocator_memory_regression_fraction"] = None
            row["microprofile_speedup_claim_authorized"] = False
        if valid and baseline is not None and candidate is not None:
            backward_ratio = (
                candidate["exact_backward_median_ms"] / baseline["exact_backward_median_ms"]
            )
            step_ratio = (
                candidate["representative_fit_step_median_ms"]
                / baseline["representative_fit_step_median_ms"]
            )
            baseline_memory = baseline["peak_incremental_allocated_bytes"]
            memory_regression = (
                candidate["peak_incremental_allocated_bytes"] / baseline_memory - 1.0
                if baseline_memory > 0
                else math.inf
            )
            for row in pair.values():
                row["candidate_over_baseline_exact_backward_time_ratio"] = backward_ratio
                row["exact_backward_reduction_fraction"] = 1.0 - backward_ratio
                row["candidate_over_baseline_representative_step_time_ratio"] = step_ratio
                row["representative_step_reduction_fraction"] = 1.0 - step_ratio
                row["peak_incremental_allocator_memory_regression_fraction"] = memory_regression


def evaluate_preregistered_gate(
    rows: list[dict[str, Any]],
    atomic_structure: dict[str, Any],
    parity: list[dict[str, Any]],
    reference_calibration: dict[str, Any],
) -> dict[str, Any]:
    cell = PREREGISTERED_GATE["representative_cell"]
    matching = [
        row
        for row in rows
        if all(row[key] == value for key, value in cell.items()) and row["status"] == "ok"
    ]
    by_mode = {row["mode"]: row for row in matching}
    baseline = by_mode.get(PREREGISTERED_GATE["baseline_renderer"])
    candidate = by_mode.get(PREREGISTERED_GATE["candidate_renderer"])
    if baseline is None or candidate is None or len(matching) != 2:
        return {
            "pass": False,
            "status": "representative_cell_missing_or_ambiguous",
            "matching_rows": len(matching),
            "microprofile_speedup_claim_authorized": False,
            "default_flip_authorized": False,
        }
    backward_reduction = 1.0 - (
        candidate["exact_backward_median_ms"] / baseline["exact_backward_median_ms"]
    )
    step_reduction = 1.0 - (
        candidate["representative_fit_step_median_ms"]
        / baseline["representative_fit_step_median_ms"]
    )
    baseline_memory = int(baseline["peak_incremental_allocated_bytes"])
    memory_regression = (
        int(candidate["peak_incremental_allocated_bytes"]) / baseline_memory - 1.0
        if baseline_memory > 0
        else math.inf
    )
    stability_values = {
        "baseline_exact_backward_cv": baseline["exact_backward_cv"],
        "candidate_exact_backward_cv": candidate["exact_backward_cv"],
        "baseline_representative_step_cv": baseline["representative_fit_step_cv"],
        "candidate_representative_step_cv": candidate["representative_fit_step_cv"],
    }
    max_cv = max(stability_values.values())
    parity_pass = bool(
        all(item["passed"] for item in parity)
        and reference_calibration["reference_vs_cuda"]["passed"]
        and reference_calibration["reference_vs_cuda_block_reduce"]["passed"]
    )
    required_grid = {
        (resolution, resolution, count, overlap)
        for resolution in DEFAULT_RESOLUTIONS
        for count in DEFAULT_COUNTS
        for overlap in DEFAULT_OVERLAPS
    }
    grid_pairs: dict[tuple[int, int, int, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (
            int(row["height"]),
            int(row["width"]),
            int(row["n_gaussians"]),
            float(row["requested_support_overlap"]),
        )
        if key in required_grid:
            grid_pairs.setdefault(key, {})[str(row["mode"])] = row
    grid_cells = []
    grid_complete = set(grid_pairs) == required_grid and all(
        set(pair) == {
            PREREGISTERED_GATE["baseline_renderer"],
            PREREGISTERED_GATE["candidate_renderer"],
        }
        for pair in grid_pairs.values()
    )
    if grid_complete:
        for key in sorted(grid_pairs):
            pair = grid_pairs[key]
            grid_baseline = pair[PREREGISTERED_GATE["baseline_renderer"]]
            grid_candidate = pair[PREREGISTERED_GATE["candidate_renderer"]]
            exact_reduction = 1.0 - (
                grid_candidate["exact_backward_median_ms"]
                / grid_baseline["exact_backward_median_ms"]
            )
            grid_step_reduction = 1.0 - (
                grid_candidate["representative_fit_step_median_ms"]
                / grid_baseline["representative_fit_step_median_ms"]
            )
            grid_cells.append({
                "cell": {
                    "height": key[0],
                    "width": key[1],
                    "n_gaussians": key[2],
                    "requested_support_overlap": key[3],
                },
                "exact_backward_reduction_fraction": exact_reduction,
                "representative_step_reduction_fraction": grid_step_reduction,
                "pass": min(exact_reduction, grid_step_reduction)
                >= PREREGISTERED_GATE["minimum_grid_direction_reduction_fraction"],
            })
    grid_direction_pass = bool(
        grid_complete and all(item["pass"] for item in grid_cells)
    )
    checks = {
        "all_reference_and_grid_parity": {
            "value": parity_pass,
            "threshold": "all pass at established atol/rtol",
            "pass": parity_pass,
        },
        "exact_backward_reduction": {
            "value": backward_reduction,
            "threshold": (
                f">={PREREGISTERED_GATE['minimum_exact_backward_reduction_fraction']}"
            ),
            "pass": backward_reduction
            >= PREREGISTERED_GATE["minimum_exact_backward_reduction_fraction"],
        },
        "representative_step_reduction": {
            "value": step_reduction,
            "threshold": (
                f">={PREREGISTERED_GATE['minimum_representative_step_reduction_fraction']}"
            ),
            "pass": step_reduction
            >= PREREGISTERED_GATE["minimum_representative_step_reduction_fraction"],
        },
        "peak_incremental_allocator_memory_regression": {
            "value": memory_regression,
            "threshold": (
                "<="
                f"{PREREGISTERED_GATE['maximum_peak_incremental_allocator_memory_regression_fraction']}"
            ),
            "pass": memory_regression
            <= PREREGISTERED_GATE[
                "maximum_peak_incremental_allocator_memory_regression_fraction"
            ],
        },
        "repeat_stability": {
            "value": max_cv,
            "threshold": f"<={PREREGISTERED_GATE['maximum_representative_timing_cv']}",
            "pass": max_cv <= PREREGISTERED_GATE["maximum_representative_timing_cv"],
            "components": stability_values,
        },
        "all_grid_direction_retention": {
            "value": grid_direction_pass,
            "threshold": (
                "complete frozen grid; exact-backward and representative-step reduction "
                f">={PREREGISTERED_GATE['minimum_grid_direction_reduction_fraction']} "
                "at every cell"
            ),
            "pass": grid_direction_pass,
            "grid_complete": grid_complete,
            "cells": grid_cells,
        },
        "executed_source_structure": {
            "value": atomic_structure["comparison_ready"],
            "threshold": "baseline atomic and candidate block-reduction patterns both present",
            "pass": bool(atomic_structure["comparison_ready"]),
        },
    }
    passed = all(check["pass"] for check in checks.values())
    for row in rows:
        row["microprofile_speedup_claim_authorized"] = passed
    return {
        "pass": passed,
        "status": (
            "opt_in_microprofile_gate_pass"
            if passed
            else "keep_benchmark_only_optimization_gate_failed"
        ),
        "checks": checks,
        "representative_baseline_row": baseline,
        "representative_candidate_row": candidate,
        "exact_backward_reduction_fraction": backward_reduction,
        "representative_step_reduction_fraction": step_reduction,
        "peak_incremental_allocator_memory_regression_fraction": memory_regression,
        "all_grid_direction_retention_passed": grid_direction_pass,
        "microprofile_speedup_claim_authorized": passed,
        "selector_remains_opt_in": True,
        "default_flip_authorized": False,
        "meaning": (
            "A pass supports only a source/device-bound microprofile speedup claim and continued "
            "opt-in testing. It does not authorize a default flip or any quality, convergence, "
            "end-to-end fit, cross-GPU, or compression claim."
        ),
    }


def profile_grid(
    *,
    resolutions: tuple[int, ...] = DEFAULT_RESOLUTIONS,
    counts: tuple[int, ...] = DEFAULT_COUNTS,
    overlaps: tuple[float, ...] = DEFAULT_OVERLAPS,
    modes: tuple[str, ...] = DEFAULT_MODES,
    warmup: int = 5,
    repeats: int = 15,
    seed: int = 0,
    device_index: int = 0,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("exact backward profile requires CUDA")
    if warmup < 1 or repeats < 3:
        raise ValueError("warmup must be >=1 and repeats must be >=3")
    if tuple(modes) != DEFAULT_MODES:
        raise ValueError(
            "this frozen optimization profile compares exactly cuda and cuda_block_reduce"
        )
    if any(value <= 0 for value in resolutions + counts) or any(value <= 0 for value in overlaps):
        raise ValueError("resolutions, counts, and overlaps must all be positive")

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    repo_root = Path(__file__).resolve().parents[1]
    cuda_source = (repo_root / "src/structsplat/cuda/render_ext.cu").read_text()
    atomic_structure = inspect_exact_backward_source(cuda_source)
    # Compile/load once before all warmup and peak-memory accounting.
    calibration = reference_parity_calibration(device_index)

    rows: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    parity_by_key: dict[tuple[int, int, int, float], dict[str, Any]] = {}
    cell_index = 0
    for resolution in resolutions:
        for count in counts:
            for overlap in overlaps:
                spec = CellSpec(resolution, resolution, count, float(overlap), seed + cell_index)
                cell_index += 1
                arrays = make_field_arrays(spec)
                support = actual_support_statistics(spec, arrays)
                target = make_target(spec, device)
                parity = parity_between_modes(
                    spec, arrays, target, "cuda", "cuda_block_reduce", device
                )
                key = (spec.height, spec.width, spec.n_gaussians, overlap)
                parity_by_key[key] = {**asdict(spec), **parity}
                # Alternate arm order by frozen cell index to limit first-arm thermal/cache bias.
                ordered_modes = modes if cell_index % 2 == 1 else tuple(reversed(modes))
                for order_index, mode in enumerate(ordered_modes):
                    row, samples = _profile_mode(
                        spec,
                        arrays,
                        target,
                        mode,
                        warmup=warmup,
                        repeats=repeats,
                        device_index=device_index,
                        atomic_structure=atomic_structure,
                    )
                    row.update(support)
                    row["within_cell_measurement_order"] = order_index
                    rows.append(row)
                    for sample in samples:
                        sample.update(
                            {
                                "actual_support_overlap": support["actual_support_overlap"],
                                "support_aabb_visits": support["support_aabb_visits"],
                                "within_cell_measurement_order": order_index,
                            }
                        )
                    raw_samples.extend(samples)
                del target
                torch.cuda.empty_cache()

    attach_pair_ratios(rows, parity_by_key)
    parity_rows = list(parity_by_key.values())
    gate = evaluate_preregistered_gate(rows, atomic_structure, parity_rows, calibration)
    return {
        "protocol": PROTOCOL,
        "work_scope": WORK_SCOPE,
        "timing_method": (
            "current-stream torch.cuda.Event(enable_timing=True), synchronized per sample, "
            "median over repeats after warmup"
        ),
        "memory_method": (
            "PyTorch CUDA allocator max_memory_allocated/max_memory_reserved after warmup; "
            "not whole-device VRAM"
        ),
        "loss_scope": "FitConfig-default 0.7*L1 + 0.3*(1-builtin SSIM)",
        "optimizer_scope": "Adam with FitConfig default learning rates; parameters restored after step",
        "grid": {
            "resolutions": list(resolutions),
            "counts": list(counts),
            "requested_support_overlaps": list(overlaps),
            "modes": list(modes),
            "warmup": warmup,
            "repeats": repeats,
            "seed": seed,
            "arm_order": "alternated_by_frozen_cell_index",
        },
        "preregistered_gate": PREREGISTERED_GATE,
        "atomic_structure": atomic_structure,
        "reference_parity_calibration": calibration,
        "cell_parity": parity_rows,
        "rows": rows,
        "raw_samples": raw_samples,
        "decision": gate,
        "microprofile_speedup_claim_authorized": gate[
            "microprofile_speedup_claim_authorized"
        ],
        "default_flip_authorized": False,
        "quality_claim_authorized": False,
        "compression_claim_authorized": False,
        "limitations": [
            "This is a deterministic synthetic micro-fit, not an end-to-end fit or image benchmark.",
            (
                "CUDA events measure device work on the current stream; Python launch overhead, "
                "extension compilation, input construction, and synchronization wait are excluded."
            ),
            (
                "Support AABB visits and source atomic counts are explanatory proxies; only paired "
                "CUDA-event deltas enter the microprofile speedup gate."
            ),
            "Peak memory is the PyTorch allocator peak, not total process or device VRAM.",
            "Medians describe this device/source grid; no confidence interval or cross-GPU claim is made.",
            (
                "Baseline/candidate timing ratios are emitted only after same-equation "
                "forward/backward parity; the PyTorch reference is correctness calibration and "
                "is not timed."
            ),
            "The candidate remains opt-in regardless of this microprofile decision.",
        ],
    }


def _strict_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    config = {
        key: result[key]
        for key in (
            "protocol",
            "work_scope",
            "timing_method",
            "memory_method",
            "loss_scope",
            "optimizer_scope",
            "grid",
            "preregistered_gate",
            "environment",
            "provenance",
            "ncu_evidence",
        )
    }
    aggregate = {key: value for key, value in result.items() if key != "raw_samples"}
    (outdir / "config.json").write_text(_strict_json(config))
    (outdir / "aggregate.json").write_text(_strict_json(aggregate))
    (outdir / "rows.json").write_text(_strict_json(result["rows"]))
    _write_csv(outdir / "rows.csv", result["rows"])
    _write_csv(outdir / "samples.csv", result["raw_samples"])
    (outdir / "ncu.json").write_text(_strict_json(result["ncu_evidence"]))

    lines = [
        "# Exact normalized CUDA block-reduction experiment",
        "",
        f"- Work scope: `{result['work_scope']}`",
        f"- Device: {result['environment']['device_name']}",
        f"- Timing: {result['timing_method']}",
        f"- Nsight Compute: `{result['ncu_evidence']['status']}`",
        "- Source/device-bound microprofile speedup claim authorized: "
        f"**{'yes' if result['decision']['microprofile_speedup_claim_authorized'] else 'no'}**",
        "- Default flip authorized: **no**",
        "",
        "## Preregistered decision",
        "",
        f"**{result['decision']['status']}** — pass={result['decision']['pass']}.",
    ]
    if "exact_backward_reduction_fraction" in result["decision"]:
        lines.append(
            "Representative reductions: exact backward "
            f"{100.0 * result['decision']['exact_backward_reduction_fraction']:.2f}%, "
            "full step "
            f"{100.0 * result['decision']['representative_step_reduction_fraction']:.2f}%, "
            "peak allocator-memory regression "
            f"{100.0 * result['decision']['peak_incremental_allocator_memory_regression_fraction']:.2f}%."
        )
    lines.extend(
        (
            "",
            "A pass supports only this source/device-bound microprofile comparison. The candidate "
            "remains opt-in and no default, end-to-end fit, quality, or cross-GPU claim follows.",
            "",
            "## Measured rows",
            "",
            "| H | N | Requested overlap | Actual overlap | Mode | Forward ms | Exact backward ms | "
            "Fit backward ms | Adam ms | Full step ms | Backward CV | Step CV | Peak +MiB |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in result["rows"]:
        lines.append(
            f"| {row['height']} | {row['n_gaussians']} | "
            f"{row['requested_support_overlap']:.1f} | {row['actual_support_overlap']:.2f} | "
            f"{row['mode']} | {row['exact_forward_median_ms']:.3f} | "
            f"{row['exact_backward_median_ms']:.3f} | {row['fit_backward_median_ms']:.3f} | "
            f"{row['optimizer_update_median_ms']:.3f} | "
            f"{row['representative_fit_step_median_ms']:.3f} | "
            f"{100.0 * row['exact_backward_cv']:.2f}% | "
            f"{100.0 * row['representative_fit_step_cv']:.2f}% | "
            f"{row['peak_incremental_allocated_bytes'] / 2**20:.1f} |"
        )
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {limitation}" for limitation in result["limitations"])
    lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines))


def _ncu_target(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        return 2
    device = torch.device(f"cuda:{args.device_index}")
    spec = CellSpec(
        args.height,
        args.width,
        args.n_gaussians,
        args.requested_overlap,
        args.seed,
    )
    arrays = make_field_arrays(spec)
    field = make_field(arrays, device)
    target = make_target(spec, device)
    image = render_exact(field, spec, "cuda_block_reduce")
    representative_loss(image, target).backward()
    torch.cuda.synchronize()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/bench010_exact_backward_block_reduce")
    parser.add_argument("--resolutions", type=int, nargs="+", default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--overlaps", type=float, nargs="+", default=list(DEFAULT_OVERLAPS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--ncu-target", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--height", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--n-gaussians", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--requested-overlap", type=float, default=16.0, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ncu_target:
        return _ncu_target(args)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for exact backward profiling")
    outdir = Path(args.outdir)
    torch.set_num_threads(1)
    result = profile_grid(
        resolutions=tuple(args.resolutions),
        counts=tuple(args.counts),
        overlaps=tuple(args.overlaps),
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        device_index=args.device_index,
    )
    result["environment"] = collect_environment(args.device_index)
    result["provenance"] = collect_provenance()
    ncu_probe = probe_ncu(args.device_index)
    result["ncu_evidence"] = collect_ncu_evidence(
        ncu_probe, outdir, PREREGISTERED_GATE["representative_cell"]
    )
    write_outputs(result, outdir)
    print(
        _strict_json(
            {
                "outdir": str(outdir),
                "device": result["environment"]["device_name"],
                "rows": len(result["rows"]),
                "ncu_status": result["ncu_evidence"]["status"],
                "decision": result["decision"],
                "microprofile_speedup_claim_authorized": result["decision"][
                    "microprofile_speedup_claim_authorized"
                ],
                "default_flip_authorized": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
