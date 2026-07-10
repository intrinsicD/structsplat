"""Native external-repository benchmark with centrally recomputed metrics.

This harness is intentionally separate from ``fair_density_control_compare``. The latter
isolates policies inside one StructSplat fitter; this file executes real external code in
subprocesses so incompatible ``gsplat`` builds cannot contaminate one another. ``matched_axes``
means the image, resolution, count cap, requested steps, and seed are matched. It does *not*
mean the native renderer, optimizer, loss, or growth implementation are the same.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from benchmarks.common import (
    json_safe_rows,
    load_image,
    psnr_auc,
    resolve_seeds,
    run_config,
    save_image,
    target_tensor,
    write_config,
    write_csv,
    write_json,
)
from structsplat import metrics as M


SCHEMA_VERSION = 2
ADAPTER_REVISION = "gaussianimage_plus_v2"
METRIC_PROTOCOL_REVISION = 2
METHOD = "gaussianimage_plus_native"
METHOD_LABEL = "GaussianImage++ native"
PROTOCOL = "matched_axes"
PAIRED_METRICS = {
    "psnr": 1.0,
    "ms_ssim": 1.0,
    "lpips": -1.0,
    "auc_psnr": 1.0,
    "fit_seconds": -1.0,
    "total_seconds": -1.0,
}


def _repo_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _tracked_diff_sha256(repo: Path) -> str:
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
    )
    return hashlib.sha256(diff).hexdigest()


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkout_fingerprint(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    gsplat = (repo / "gsplat").resolve()
    repo_status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    gsplat_status = _git(gsplat, "status", "--porcelain", "--untracked-files=all")
    untracked_python = [
        line[3:]
        for line in (repo_status + "\n" + gsplat_status).splitlines()
        if line.startswith("?? ") and line[3:].endswith(".py")
    ]
    if untracked_python:
        raise RuntimeError(
            "GaussianImage++ checkout contains untracked Python sources: "
            + ", ".join(sorted(untracked_python))
        )
    empty_diff_sha256 = hashlib.sha256(b"").hexdigest()
    repo_diff = _tracked_diff_sha256(repo)
    gsplat_diff = _tracked_diff_sha256(gsplat)
    if repo_diff != empty_diff_sha256 or gsplat_diff != empty_diff_sha256:
        raise RuntimeError(
            "GaussianImage++ native evidence requires clean tracked repo and gsplat sources"
        )
    return {
        "repo_root": str(repo),
        "repo_commit": _repo_commit(repo),
        "repo_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "repo_tracked_diff_sha256": repo_diff,
        "repo_python_source_sha256": _python_tree_sha256(repo),
        "repo_untracked_files": [
            line[3:] for line in repo_status.splitlines() if line.startswith("?? ")
        ],
        "gsplat_root": str(gsplat),
        "gsplat_commit": _repo_commit(gsplat),
        "gsplat_tree": _git(gsplat, "rev-parse", "HEAD^{tree}"),
        "gsplat_tracked_diff_sha256": gsplat_diff,
        "gsplat_python_source_sha256": _python_tree_sha256(gsplat / "gsplat"),
        "gsplat_untracked_files": [
            line[3:] for line in gsplat_status.splitlines() if line.startswith("?? ")
        ],
    }


def _environment_fingerprint(device: str) -> dict[str, Any]:
    dependencies = {}
    for name in ("lpips", "pillow", "pytorch-msssim", "torchvision"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
                "compute_capability": [
                    torch.cuda.get_device_properties(index).major,
                    torch.cuda.get_device_properties(index).minor,
                ],
            }
            for index in range(torch.cuda.device_count())
        ],
        "metric_dependency_versions": dependencies,
        "metric_device": device,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: np.ndarray) -> str:
    """Hash the exact decoded uint8 target pixels used by every comparison harness."""
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype="<i8").tobytes())
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _artifact_source_id(path: Path, source_sha256: str) -> str:
    """Return a readable, collision-resistant directory name for one source file."""
    canonical = path.resolve()
    digest = hashlib.sha256()
    digest.update(str(canonical).encode("utf-8"))
    digest.update(b"\0")
    digest.update(source_sha256.encode("ascii"))
    return f"{canonical.stem}_{digest.hexdigest()[:16]}"


def _bundled_extension(repo: Path) -> tuple[Path, str]:
    matches = sorted((repo / "gsplat" / "gsplat").glob("csrc*.so"))
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one GaussianImage++ csrc extension under "
            f"{repo / 'gsplat' / 'gsplat'}, found {len(matches)}"
        )
    extension = matches[0].resolve()
    return extension, _sha256(extension)


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("schema_version", 0) or 0),
        row.get("adapter_revision"),
        int(row.get("metric_protocol_revision", 0) or 0),
        row.get("protocol"),
        row.get("method"),
        row.get("repo_commit"),
        row.get("repo_tree"),
        row.get("repo_tracked_diff_sha256"),
        row.get("repo_python_source_sha256"),
        json.dumps(row.get("repo_untracked_files"), sort_keys=True),
        row.get("gsplat_commit"),
        row.get("gsplat_tree"),
        row.get("gsplat_tracked_diff_sha256"),
        row.get("gsplat_python_source_sha256"),
        json.dumps(row.get("gsplat_untracked_files"), sort_keys=True),
        row.get("gsplat_csrc_sha256"),
        row.get("native_adapter_source_sha256"),
        row.get("native_harness_source_sha256"),
        row.get("metric_protocol_source_sha256"),
        row.get("python_executable"),
        row.get("python_version"),
        row.get("torch_version"),
        row.get("torch_cuda_version"),
        row.get("numpy_version"),
        json.dumps(row.get("cuda_devices"), sort_keys=True),
        json.dumps(row.get("metric_dependency_versions"), sort_keys=True),
        row.get("metric_device"),
        row.get("source_path"),
        row.get("source_sha256"),
        row.get("target_sha256"),
        row.get("target_pixel_sha256"),
        int(row.get("max_side", 0)),
        int(row.get("budget_cap", 0)),
        int(row.get("start_gaussians", 0)),
        int(row.get("requested_iters", 0)),
        int(row.get("seed", 0)),
        int(row.get("growth_waves_requested", 0) or 0),
        int(row.get("render_warmup", 0) or 0),
        int(row.get("render_repeats", 0) or 0),
        bool(row.get("lpips_requested", False)),
        row.get("device"),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(f"expected a JSON row list in {path}")


def _comparison_key(row: dict[str, Any], *, native: bool) -> tuple[Any, ...]:
    source = Path(str(row.get("source_path"))).resolve()
    return (
        str(source),
        int(row.get("max_side", 0)),
        int(row.get("budget_cap" if native else "final_budget", 0)),
        int(row.get("requested_iters" if native else "iters", 0)),
        int(row.get("seed", 0)),
    )


def _target_pixel_sha256(row: dict[str, Any]) -> str | None:
    recorded = row.get("target_pixel_sha256")
    return recorded if isinstance(recorded, str) and recorded else None


def _comparison_start_count(row: dict[str, Any], *, native: bool) -> int | None:
    names = (
        ("requested_start_gaussians", "start_gaussians")
        if native else ("start_budget", "start_gaussians")
    )
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _bootstrap_image_mean_ci(
    values: list[float],
    *,
    samples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None, None, None
    center = float(array.mean())
    if array.size == 1 or samples <= 0:
        return center, float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return center, float(low), float(high)


def _paired_native_vs_structsplat(
    native_rows: list[dict[str, Any]],
    structsplat_rows: list[dict[str, Any]],
    *,
    native_method: str = METHOD,
    baseline_method: str = "structsplat_best_default",
    relation_metrics: tuple[str, ...] | None = None,
    require_start_match: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if relation_metrics is None:
        relation_metrics = tuple(metric for metric in PAIRED_METRICS if metric != "lpips")
    if not relation_metrics:
        raise ValueError("paired relation metrics must not be empty")
    unknown = set(relation_metrics) - set(PAIRED_METRICS)
    if unknown:
        raise ValueError(f"unknown paired relation metrics: {sorted(unknown)}")
    structsplat = {
        _comparison_key(row, native=False): row
        for row in structsplat_rows
        if row.get("status") == "ok" and row.get("method") == baseline_method
    }
    pairs = []
    for native_row in native_rows:
        if native_row.get("status") != "ok" or native_row.get("method") != native_method:
            continue
        base = structsplat.get(_comparison_key(native_row, native=True))
        if base is None:
            continue
        native_target_sha256 = _target_pixel_sha256(native_row)
        structsplat_target_sha256 = _target_pixel_sha256(base)
        if (
            native_target_sha256 is None
            or structsplat_target_sha256 is None
            or native_target_sha256 != structsplat_target_sha256
        ):
            continue
        native_start = _comparison_start_count(native_row, native=True)
        structsplat_start = _comparison_start_count(base, native=False)
        start_verified = (
            native_start is not None
            and structsplat_start is not None
            and native_start == structsplat_start
        )
        if require_start_match and not start_verified:
            continue
        pair = {
            "source_path": native_row["source_path"],
            "source_id": native_row.get("source_id"),
            "image": native_row.get("source_id")
            or Path(str(native_row["source_path"])).stem,
            "max_side": native_row["max_side"],
            "budget": native_row["budget_cap"],
            "iters": native_row["requested_iters"],
            "seed": native_row["seed"],
            "native_method": native_row["method"],
            "structsplat_method": base["method"],
            "target_pixel_sha256": native_target_sha256,
            "target_pixels_verified": True,
            "native_start_gaussians": native_start,
            "structsplat_start_gaussians": structsplat_start,
            "start_gaussians_verified": start_verified,
            "native_growth_waves": native_row.get("growth_waves_requested"),
            "structsplat_growth_waves": base.get("growth_waves"),
        }
        for metric, direction in PAIRED_METRICS.items():
            native_value = native_row.get(metric)
            base_value = base.get(metric)
            pair[f"native_{metric}"] = native_value
            pair[f"structsplat_{metric}"] = base_value
            pair[f"native_gain_{metric}"] = (
                None
                if native_value is None or base_value is None
                else direction * (float(native_value) - float(base_value))
            )
        pairs.append(pair)
    if not pairs:
        return [], None

    core_metrics = [metric for metric in PAIRED_METRICS if metric != "lpips"]
    complete_pairs = [
        pair for pair in pairs
        if all(
            pair.get(f"native_gain_{metric}") is not None
            and np.isfinite(float(pair[f"native_gain_{metric}"]))
            for metric in core_metrics
        )
    ]

    aggregate: dict[str, Any] = {
        "native_method": native_method,
        "baseline_method": baseline_method,
        "pairs": len(complete_pairs),
        "images": len({pair["source_path"] for pair in complete_pairs}),
        "relation_metrics": list(relation_metrics),
    }
    intervals = {}
    familywise_intervals = {}
    for metric_idx, metric in enumerate(PAIRED_METRICS):
        by_image: dict[str, list[float]] = {}
        metric_pairs = pairs if metric == "lpips" else complete_pairs
        for pair in metric_pairs:
            value = pair.get(f"native_gain_{metric}")
            if value is not None:
                by_image.setdefault(str(pair["source_path"]), []).append(float(value))
        image_means = [mean(values) for values in by_image.values()]
        center, low, high = _bootstrap_image_mean_ci(
            image_means,
            seed=metric_idx,
        )
        aggregate[f"gain_{metric}"] = center
        aggregate[f"ci_low_{metric}"] = low
        aggregate[f"ci_high_{metric}"] = high
        aggregate[f"pairs_{metric}"] = sum(len(values) for values in by_image.values())
        aggregate[f"images_{metric}"] = len(image_means)
        if metric in relation_metrics and center is not None and low is not None and high is not None:
            intervals[metric] = (center, low, high)
            fw_center, fw_low, fw_high = _bootstrap_image_mean_ci(
                image_means,
                seed=metric_idx,
                alpha=0.05 / len(relation_metrics),
            )
            familywise_intervals[metric] = (fw_center, fw_low, fw_high)

    core = list(intervals.values())
    if len(core) != len(relation_metrics):
        aggregate["sample_relation"] = "incomplete"
        aggregate["supported_relation_95ci"] = "incomplete"
    elif all(center > 0.0 for center, _low, _high in core):
        aggregate["sample_relation"] = "native_dominates"
        aggregate["supported_relation_95ci"] = (
            "native_dominates"
            if len(familywise_intervals) == len(relation_metrics)
            and all(low > 0.0 for _center, low, _high in familywise_intervals.values())
            else "inconclusive"
        )
    elif all(center < 0.0 for center, _low, _high in core):
        aggregate["sample_relation"] = "structsplat_dominates"
        aggregate["supported_relation_95ci"] = (
            "structsplat_dominates"
            if len(familywise_intervals) == len(relation_metrics)
            and all(high < 0.0 for _center, _low, high in familywise_intervals.values())
            else "inconclusive"
        )
    else:
        aggregate["sample_relation"] = "tradeoff"
        aggregate["supported_relation_95ci"] = "tradeoff"
    return pairs, aggregate


def _format_gain_ci(row: dict[str, Any], metric: str, digits: int) -> str:
    center = row.get(f"gain_{metric}")
    low = row.get(f"ci_low_{metric}")
    high = row.get(f"ci_high_{metric}")
    if center is None or low is None or high is None:
        return "-"
    return f"{float(center):+.{digits}f} [{float(low):+.{digits}f}, {float(high):+.{digits}f}]"


def _target_hits(history: dict[str, Any], targets: list[float]) -> dict[str, int | None]:
    iterations = history.get("iter") or []
    psnrs = history.get("psnr") or []
    result = {}
    for target in targets:
        hit = next(
            (int(iteration) for iteration, psnr in zip(iterations, psnrs)
             if float(psnr) >= float(target)),
            None,
        )
        result[str(float(target))] = hit
    return result


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    repo: Path,
    repo_commit: str,
    target_path: Path,
    target_shape: tuple[int, int],
    budget: int,
    start_gaussians: int,
    requested_iters: int,
    seed: int,
    growth_waves: int,
    expected_extension: Path,
    expected_extension_sha256: str,
    environment_fingerprint: dict[str, Any] | None = None,
) -> np.ndarray:
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported native result schema {manifest.get('schema_version')!r}")
    if manifest.get("method") != METHOD:
        raise ValueError(f"unexpected native method {manifest.get('method')!r}")
    if manifest.get("adapter_revision") != ADAPTER_REVISION:
        raise ValueError(f"unexpected native adapter revision {manifest.get('adapter_revision')!r}")
    if Path(manifest["repo_root"]).resolve() != repo.resolve():
        raise ValueError("native manifest repository root does not match the requested checkout")
    if str(manifest.get("repo_commit")) != repo_commit:
        raise ValueError(
            "native manifest commit does not match the requested checkout: "
            f"{manifest.get('repo_commit')!r} != {repo_commit!r}"
        )
    if environment_fingerprint is not None:
        runner_python = manifest.get("python_executable")
        if (
            not runner_python
            or Path(str(runner_python)).resolve()
            != Path(str(environment_fingerprint["python_executable"])).resolve()
        ):
            raise ValueError("native manifest Python executable does not match the harness")
        for name in (
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "numpy_version",
        ):
            if manifest.get(name) != environment_fingerprint.get(name):
                raise ValueError(
                    f"native manifest environment {name}={manifest.get(name)!r} does not "
                    f"match {environment_fingerprint.get(name)!r}"
                )
        if json.dumps(manifest.get("cuda_devices"), sort_keys=True) != json.dumps(
            environment_fingerprint.get("cuda_devices"), sort_keys=True
        ):
            raise ValueError("native manifest CUDA devices do not match the harness")
    extension = Path(manifest["gsplat_csrc"]).resolve()
    try:
        extension.relative_to((repo / "gsplat").resolve())
    except ValueError as exc:
        raise ValueError(f"native extension {extension} is outside the requested checkout") from exc
    if not extension.is_file():
        raise ValueError(f"native extension does not exist: {extension}")
    if extension != expected_extension.resolve():
        raise ValueError("native manifest resolved a different extension than the harness fingerprint")
    actual_extension_hash = _sha256(extension)
    if (
        str(manifest.get("gsplat_csrc_sha256")) != actual_extension_hash
        or actual_extension_hash != expected_extension_sha256
    ):
        raise ValueError("native extension hash does not match the loaded checkout artifact")
    expected_axes = {
        "source_path": str(target_path.resolve()),
        "height": int(target_shape[0]),
        "width": int(target_shape[1]),
        "seed": int(seed),
        "requested_iters": int(requested_iters),
        "budget_cap": int(budget),
        "start_gaussians": int(start_gaussians),
        "growth_waves_requested": int(growth_waves),
    }
    for name, expected_value in expected_axes.items():
        value = manifest.get(name)
        if name == "source_path":
            matches = value is not None and Path(str(value)).resolve() == target_path.resolve()
        else:
            try:
                matches = int(value) == expected_value
            except (TypeError, ValueError):
                matches = False
        if not matches:
            raise ValueError(
                f"native manifest axis {name}={value!r} does not match {expected_value!r}"
            )
    n_gaussians = int(manifest.get("n_gaussians", budget + 1))
    if n_gaussians <= 0 or n_gaussians > int(budget):
        raise ValueError(
            f"native result exceeded count cap: {manifest.get('n_gaussians')} > {budget}"
        )
    iterations_run = int(manifest.get("iterations_run", -1))
    if iterations_run <= 0 or iterations_run > requested_iters:
        raise ValueError(
            f"invalid native iteration count {iterations_run} for request {requested_iters}"
        )
    reconstruction = np.load(Path(manifest["reconstruction_npy"]), allow_pickle=False)
    expected = (*target_shape, 3)
    if reconstruction.shape != expected:
        raise ValueError(
            f"native reconstruction shape {reconstruction.shape} does not match target {expected}"
        )
    if not np.isfinite(reconstruction).all():
        raise ValueError("native reconstruction contains non-finite values")

    for name in (
        "init_seconds",
        "fit_seconds",
        "total_seconds",
        "render_ms_mean",
        "render_ms_median",
        "render_ms_p10",
        "render_ms_p90",
        "render_fps_median",
        "native_reported_psnr",
    ):
        try:
            value = float(manifest[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"native manifest lacks finite {name}") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"native manifest has invalid {name}={value!r}")
    history = manifest.get("history")
    if not isinstance(history, dict):
        raise ValueError("native manifest lacks per-step history")
    iterations = history.get("iter")
    psnrs = history.get("psnr")
    elapsed = history.get("elapsed")
    counts = history.get("n_gaussians")
    if not all(isinstance(values, list) for values in (iterations, psnrs, elapsed, counts)):
        raise ValueError("native per-step history fields must be lists")
    if not all(len(values) == iterations_run for values in (iterations, psnrs, elapsed, counts)):
        raise ValueError("native per-step history length does not match iterations_run")
    if [int(value) for value in iterations] != list(range(iterations_run)):
        raise ValueError("native per-step iteration indices are not contiguous")
    if not all(np.isfinite(float(value)) for value in psnrs):
        raise ValueError("native per-step PSNR history contains non-finite values")
    elapsed_values = [float(value) for value in elapsed]
    if (
        not all(np.isfinite(value) and value >= 0.0 for value in elapsed_values)
        or any(right < left for left, right in zip(elapsed_values, elapsed_values[1:]))
    ):
        raise ValueError("native elapsed history must be finite and nondecreasing")
    if not all(0 < int(value) <= budget for value in counts):
        raise ValueError("native Gaussian-count history violates the requested cap")
    selected_iter = int(manifest.get("selected_iter", 0))
    if not 1 <= selected_iter <= iterations_run:
        raise ValueError(f"native selected_iter {selected_iter} is outside the trajectory")
    if manifest.get("selection_policy") != "best_training_psnr_checkpoint":
        raise ValueError(f"unexpected native selection policy {manifest.get('selection_policy')!r}")
    return reconstruction.astype(np.float32, copy=False)


def _central_metrics(
    reconstruction: np.ndarray,
    target: np.ndarray,
    *,
    device: str,
    want_lpips: bool,
) -> dict[str, float | None]:
    # Display-referred comparison: external image codecs cannot preserve out-of-range values.
    render = target_tensor(np.clip(reconstruction, 0.0, 1.0), device)
    target_t = target_tensor(target, device)
    lpips = None
    if want_lpips:
        lpips = M.LPIPS.distance(render, target_t)
    return {
        "psnr": M.psnr(render, target_t),
        "ssim": float(M.ssim(render, target_t)),
        "ms_ssim": M.ms_ssim(render, target_t),
        "lpips": lpips,
    }


def _native_command(
    *,
    repo: Path,
    image: Path,
    cell_dir: Path,
    start_gaussians: int,
    budget: int,
    iters: int,
    seed: int,
    growth_waves: int,
    render_warmup: int,
    render_repeats: int,
) -> list[str]:
    runner = Path(__file__).resolve().parent / "native_runners" / "gaussianimage_plus.py"
    return [
        sys.executable,
        str(runner),
        "--repo", str(repo),
        "--image", str(image),
        "--outdir", str(cell_dir),
        "--initial-gaussians", str(start_gaussians),
        "--max-gaussians", str(budget),
        "--iterations", str(iters),
        "--seed", str(seed),
        "--growth-waves", str(growth_waves),
        "--render-warmup", str(render_warmup),
        "--render-repeats", str(render_repeats),
    ]


def _run_cell(
    *,
    repo: Path,
    repo_commit: str,
    checkout_fingerprint: dict[str, Any],
    source_fingerprint: dict[str, Any],
    environment_fingerprint: dict[str, Any],
    expected_extension: Path,
    expected_extension_sha256: str,
    source_path: Path,
    source_sha256: str,
    target_path: Path,
    target_sha256: str,
    target_pixel_sha256: str,
    target: np.ndarray,
    max_side: int,
    budget: int,
    start_gaussians: int,
    iters: int,
    seed: int,
    growth_waves: int,
    render_warmup: int,
    render_repeats: int,
    device: str,
    want_lpips: bool,
    cell_dir: Path,
) -> dict[str, Any]:
    command = _native_command(
        repo=repo,
        image=target_path,
        cell_dir=cell_dir,
        start_gaussians=start_gaussians,
        budget=budget,
        iters=iters,
        seed=seed,
        growth_waves=growth_waves,
        render_warmup=render_warmup,
        render_repeats=render_repeats,
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "gsplat")])
    cell_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = cell_dir / "stdout.log"
    stderr_path = cell_dir / "stderr.log"
    wall_start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    wall_seconds = time.perf_counter() - wall_start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    base = {
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
        "protocol": PROTOCOL,
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "implementation_kind": "native_external",
        **checkout_fingerprint,
        **source_fingerprint,
        **environment_fingerprint,
        "gsplat_csrc": str(expected_extension),
        "gsplat_csrc_sha256": expected_extension_sha256,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_id": _artifact_source_id(source_path, source_sha256),
        "target_path": str(target_path),
        "target_sha256": target_sha256,
        "target_pixel_sha256": target_pixel_sha256,
        "image": source_path.stem,
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        "max_side": max_side,
        "seed": seed,
        "requested_iters": iters,
        "budget_cap": budget,
        "start_gaussians": start_gaussians,
        "start_fraction": start_gaussians / budget,
        "growth_waves_requested": growth_waves,
        "render_warmup": render_warmup,
        "render_repeats": render_repeats,
        "lpips_requested": want_lpips,
        "device": device,
        "command": command,
        "subprocess_wall_seconds": wall_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "actual_codec_bytes": None,
        "actual_bpp": None,
        "metric_protocol": "display-clamped StructSplat central metrics",
    }
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1:] or completed.stdout.strip().splitlines()[-1:]
        return {
            **base,
            "status": "error",
            "error": f"native subprocess exit {completed.returncode}: {' '.join(tail)}",
        }

    manifest_path = cell_dir / "result.json"
    if not manifest_path.exists():
        return {**base, "status": "error", "error": "native subprocess wrote no result.json"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reconstruction = _validate_manifest(
            manifest,
            repo=repo,
            repo_commit=repo_commit,
            target_path=target_path,
            target_shape=target.shape[:2],
            budget=budget,
            start_gaussians=start_gaussians,
            requested_iters=iters,
            seed=seed,
            growth_waves=growth_waves,
            expected_extension=expected_extension,
            expected_extension_sha256=expected_extension_sha256,
            environment_fingerprint=environment_fingerprint,
        )
        central = _central_metrics(
            reconstruction,
            target,
            device=device,
            want_lpips=want_lpips,
        )
    except Exception as exc:
        return {**base, "status": "error", "error": f"invalid native result: {type(exc).__name__}: {exc}"}

    history = manifest.get("history") or {}
    return {
        **base,
        **manifest,
        # Preserve the canonical parent-side values after validation. In particular,
        # ``sys.executable`` may be a symlink in the subprocess; retaining that raw path
        # would make the saved row's resume key differ from the next requested key.
        **environment_fingerprint,
        **central,
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
        "protocol": PROTOCOL,
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "repo_root": str(repo),
        "repo_commit": repo_commit,
        "gsplat_csrc": str(expected_extension),
        "gsplat_csrc_sha256": expected_extension_sha256,
        "source_path": str(source_path),
        "source_id": _artifact_source_id(source_path, source_sha256),
        "target_path": str(target_path),
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        "max_side": max_side,
        "seed": seed,
        "requested_iters": iters,
        "budget_cap": budget,
        "start_gaussians": start_gaussians,
        "start_fraction": start_gaussians / budget,
        "growth_waves_requested": growth_waves,
        "render_warmup": render_warmup,
        "render_repeats": render_repeats,
        "lpips_requested": want_lpips,
        "device": device,
        "auc_psnr": psnr_auc(history, nominal_iters=iters),
        "iters_to_targets": _target_hits(history, [22.0, 24.0, 26.0, 28.0, 30.0, 32.0]),
        "command": command,
        "subprocess_wall_seconds": wall_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "metric_protocol": "display-clamped StructSplat central metrics",
        "status": "ok",
        "error": "",
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    outdir: Path,
    structsplat_rows: list[dict[str, Any]] | None = None,
) -> None:
    # Keep the resume journal scoped to the exact requested/current keys.  This prevents old
    # source pixels or superseded harness revisions from being attributed to the current config.
    with (outdir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=str) + "\n")
    json_rows = json_safe_rows(rows, skip={"history", "upstream_args"})
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        fieldnames = sorted(
            key for row in json_rows for key in row
            if key not in {"command", "iters_to_targets"}
        )
        write_csv(
            outdir / "metrics.csv",
            json_rows,
            fieldnames=sorted(set(fieldnames)),
            extrasaction="ignore",
        )
    paired_rows, paired_summary = _paired_native_vs_structsplat(
        rows,
        structsplat_rows or [],
        require_start_match=True,
    )
    if paired_rows and paired_summary is not None:
        write_csv(
            outdir / "paired_native_vs_structsplat.csv",
            paired_rows,
            fieldnames=sorted({key for row in paired_rows for key in row}),
        )
        write_csv(
            outdir / "paired_native_vs_structsplat_summary.csv",
            [paired_summary],
            fieldnames=list(paired_summary),
        )
    ok = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# Native Reference Comparison",
        "",
        "`matched_axes` aligns input image, resolution, Gaussian cap, requested optimization "
        "steps, and seed. Native renderer/loss/optimizer/growth semantics remain repository-specific; "
        "this is not a same-hyperparameter or matched-policy claim.",
        "",
        "Metrics are centrally recomputed from each exported float reconstruction. Native-reported "
        "metrics and synchronized renderer timing are retained separately. Codec bpp remains blank "
        "unless a real native encoded stream is produced.",
        "",
        "GaussianImage++ natively restores its best training-PSNR checkpoint before export, whereas "
        "the paired StructSplat row exports its terminal field. The selected iteration is recorded "
        "per native row; convergence AUC still covers the complete optimization trajectory. This "
        "selection-policy asymmetry is preserved and must be considered when interpreting final "
        "quality.",
        "",
        "| Method | Image | Side | Cap | Start | Seed | PSNR | MS-SSIM | LPIPS | AUC | Fit s | Render FPS | Param bpp | Commit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ok:
        lpips = "-" if row.get("lpips") is None else f"{float(row['lpips']):.4f}"
        lines.append(
            f"| {row['method_label']} | {row['image']} | {row['max_side']} | "
            f"{row['budget_cap']} | {row['start_gaussians']} | {row['seed']} | "
            f"{float(row['psnr']):.4f} | {float(row['ms_ssim']):.5f} | {lpips} | "
            f"{float(row['auc_psnr']):.4f} | {float(row['fit_seconds']):.3f} | "
            f"{float(row['render_fps_median']):.1f} | "
            f"{float(row['parameter_bpp_float32']):.3f} | {str(row['repo_commit'])[:12]} |"
        )
    if paired_summary is not None:
        lines += [
            "",
            "## Paired Native vs StructSplat Default",
            "",
            "Positive is a native GaussianImage++ gain; for time and LPIPS, positive means lower "
            "is better. Confidence intervals bootstrap source images after averaging seeds within "
            "each image. Displayed intervals are marginal 95% intervals; a same-direction "
            "dominance relation requires Bonferroni-adjusted 95% familywise bounds across all five "
            "core metrics. This matched-axis proxy does not substitute for either repository's "
            "native-authentic/full-resolution protocol.",
            "",
            "| Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Sample relation | Familywise 95% relation |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|",
            f"| {paired_summary['pairs']} / {paired_summary['images']} | "
            f"{_format_gain_ci(paired_summary, 'psnr', 4)} | "
            f"{_format_gain_ci(paired_summary, 'ms_ssim', 5)} | "
            f"{_format_gain_ci(paired_summary, 'lpips', 4)} | "
            f"{_format_gain_ci(paired_summary, 'auc_psnr', 4)} | "
            f"{_format_gain_ci(paired_summary, 'fit_seconds', 4)} | "
            f"{_format_gain_ci(paired_summary, 'total_seconds', 4)} | "
            f"{str(paired_summary['sample_relation']).replace('_', ' ')} | "
            f"{str(paired_summary['supported_relation_95ci']).replace('_', ' ')} |",
        ]
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- `{row.get('image')}`: {row.get('error')}" for row in errors)
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    html = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<title>Native Reference Comparison</title>",
        "<style>body{font-family:system-ui;margin:24px;max-width:1100px;line-height:1.45}.note{background:#f5f5f5;border-left:4px solid #666;padding:10px}img{max-width:480px;border:1px solid #ddd}</style></head><body>",
        "<h1>Native Reference Comparison</h1>",
        "<p class=\"note\"><b>Matched axes, not same hyperparameters.</b> External code runs in an isolated subprocess; shared metrics are recomputed from float reconstructions.</p>",
        "<p><a href=\"summary.md\">summary.md</a> · <a href=\"metrics.csv\">metrics.csv</a> · <a href=\"metrics.json\">metrics.json</a> · <a href=\"paired_native_vs_structsplat.csv\">paired cells</a> · <a href=\"paired_native_vs_structsplat_summary.csv\">paired summary</a> · <a href=\"config.json\">config.json</a></p>",
    ]
    for row in ok:
        try:
            rel = Path(row["reconstruction_png"]).relative_to(outdir)
        except ValueError:
            continue
        html.append(
            f"<h2>{row['method_label']} — {row['image']} — {row['budget_cap']}G</h2>"
            f"<p>PSNR {row['psnr']:.4f}, MS-SSIM {row['ms_ssim']:.5f}, "
            f"fit {row['fit_seconds']:.3f}s, render {row['render_fps_median']:.1f} FPS</p>"
            f"<a href=\"{rel}\"><img src=\"{rel}\" alt=\"native reconstruction\"></a>"
        )
    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo = args.gaussianimage_plus_repo.resolve()
    checkout_fingerprint = _checkout_fingerprint(repo)
    repo_commit = checkout_fingerprint["repo_commit"]
    expected_extension, expected_extension_sha256 = _bundled_extension(repo)
    source_fingerprint = {
        "native_adapter_source_sha256": _sha256(
            Path(__file__).resolve().parent / "native_runners" / "gaussianimage_plus.py"
        ),
        "native_harness_source_sha256": _sha256(Path(__file__).resolve()),
        "metric_protocol_source_sha256": _sha256(Path(M.__file__).resolve()),
    }
    seeds = resolve_seeds(args.seed, args.seeds)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    environment_fingerprint = _environment_fingerprint(device)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    structsplat_rows = (
        _load_json_rows(args.structsplat_metrics)
        if args.structsplat_metrics is not None
        else []
    )
    write_config(str(outdir), run_config({
        "protocol": PROTOCOL,
        "method": METHOD,
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
        "repo_root": str(repo),
        **checkout_fingerprint,
        "gsplat_csrc": str(expected_extension),
        "gsplat_csrc_sha256": expected_extension_sha256,
        **source_fingerprint,
        **environment_fingerprint,
        "images": [str(path) for path in args.images],
        "max_sides": args.max_sides,
        "budgets": args.budgets,
        "start_fraction": args.start_fraction,
        "iters": args.iters,
        "seeds": seeds,
        "growth_waves": args.growth_waves,
        "render_warmup": args.render_warmup,
        "render_repeats": args.render_repeats,
        "lpips": args.lpips,
        "structsplat_metrics": (
            str(args.structsplat_metrics) if args.structsplat_metrics is not None else None
        ),
        "resume": args.resume,
        "max_new_cells": args.max_new_cells,
    }, device=device))

    jsonl_path = outdir / "metrics.jsonl"
    existing = _load_jsonl(jsonl_path) if args.resume else []
    if not args.resume and jsonl_path.exists():
        jsonl_path.unlink()
    existing_by_key = {
        _cell_key(row): row for row in existing if row.get("status") == "ok"
    }
    rows: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    total = len(args.images) * len(args.max_sides) * len(args.budgets) * len(seeds)
    cell_index = 0
    new_cells = 0
    for source in args.images:
        source = source.resolve()
        source_sha256 = _sha256(source)
        source_id = _artifact_source_id(source, source_sha256)
        for max_side in args.max_sides:
            target = load_image(source, max_side=max_side)
            target_path = outdir / "targets" / f"{source_id}_s{max_side}.png"
            save_image(target, target_path)
            target_sha256 = _sha256(target_path)
            target_pixel_sha256 = _pixel_sha256(target)
            for budget in args.budgets:
                start_gaussians = max(1, min(budget, int(round(budget * args.start_fraction))))
                for seed in seeds:
                    cell_index += 1
                    key_row = {
                        "schema_version": SCHEMA_VERSION,
                        "adapter_revision": ADAPTER_REVISION,
                        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
                        "protocol": PROTOCOL,
                        "method": METHOD,
                        **checkout_fingerprint,
                        "gsplat_csrc_sha256": expected_extension_sha256,
                        **source_fingerprint,
                        **environment_fingerprint,
                        "source_path": str(source),
                        "source_sha256": source_sha256,
                        "target_sha256": target_sha256,
                        "target_pixel_sha256": target_pixel_sha256,
                        "max_side": max_side,
                        "budget_cap": budget,
                        "start_gaussians": start_gaussians,
                        "requested_iters": args.iters,
                        "seed": seed,
                        "growth_waves_requested": args.growth_waves,
                        "render_warmup": args.render_warmup,
                        "render_repeats": args.render_repeats,
                        "lpips_requested": args.lpips,
                        "device": device,
                    }
                    key = _cell_key(key_row)
                    if key in selected_keys:
                        print(
                            f"[{cell_index}/{total}] skip duplicate requested cell "
                            f"{source.stem} s={max_side} n={budget} seed={seed}",
                            flush=True,
                        )
                        continue
                    cached = existing_by_key.get(key)
                    cache_valid = False
                    if cached is not None:
                        try:
                            manifest_path = (
                                Path(str(cached["reconstruction_npy"])).parent / "result.json"
                            )
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            reconstruction = _validate_manifest(
                                manifest,
                                repo=repo,
                                repo_commit=repo_commit,
                                target_path=target_path.resolve(),
                                target_shape=target.shape[:2],
                                budget=budget,
                                start_gaussians=start_gaussians,
                                requested_iters=args.iters,
                                seed=seed,
                                growth_waves=args.growth_waves,
                                expected_extension=expected_extension,
                                expected_extension_sha256=expected_extension_sha256,
                                environment_fingerprint=environment_fingerprint,
                            )
                            cached.update(
                                _central_metrics(
                                    reconstruction,
                                    target,
                                    device=device,
                                    want_lpips=args.lpips,
                                )
                            )
                            history = manifest.get("history") or {}
                            cached["auc_psnr"] = psnr_auc(
                                history, nominal_iters=args.iters
                            )
                            cached["iters_to_targets"] = _target_hits(
                                history, [22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
                            )
                            cached.update(key_row)
                            cache_valid = True
                        except (
                            OSError,
                            EOFError,
                            KeyError,
                            RuntimeError,
                            TypeError,
                            ValueError,
                            json.JSONDecodeError,
                        ):
                            cache_valid = False
                    if cache_valid:
                        rows.append(cached)
                        selected_keys.add(key)
                        print(f"[{cell_index}/{total}] skip existing {source.stem} s={max_side} n={budget} seed={seed}", flush=True)
                        continue
                    print(f"[{cell_index}/{total}] native GI++ {source.stem} s={max_side} n={budget} seed={seed}", flush=True)
                    cell_dir = outdir / "cells" / source_id / f"s{max_side}_n{budget}_seed{seed}"
                    row = _run_cell(
                        repo=repo,
                        repo_commit=repo_commit,
                        checkout_fingerprint=checkout_fingerprint,
                        source_fingerprint=source_fingerprint,
                        environment_fingerprint=environment_fingerprint,
                        expected_extension=expected_extension,
                        expected_extension_sha256=expected_extension_sha256,
                        source_path=source,
                        source_sha256=source_sha256,
                        target_path=target_path.resolve(),
                        target_sha256=target_sha256,
                        target_pixel_sha256=target_pixel_sha256,
                        target=target,
                        max_side=max_side,
                        budget=budget,
                        start_gaussians=start_gaussians,
                        iters=args.iters,
                        seed=seed,
                        growth_waves=args.growth_waves,
                        render_warmup=args.render_warmup,
                        render_repeats=args.render_repeats,
                        device=device,
                        want_lpips=args.lpips,
                        cell_dir=cell_dir.resolve(),
                    )
                    with jsonl_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, default=str) + "\n")
                    rows.append(row)
                    selected_keys.add(key)
                    new_cells += 1
                    if row.get("status") == "ok":
                        print(
                            f"  psnr={row['psnr']:.3f} ms={row['ms_ssim']:.5f} "
                            f"auc={row['auc_psnr']:.3f} fit={row['fit_seconds']:.2f}s "
                            f"render={row['render_fps_median']:.1f}fps",
                            flush=True,
                        )
                    else:
                        print(f"  ERROR {row['error']}", flush=True)
                    if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                        _write_outputs(rows, outdir, structsplat_rows)
                        return rows
    _write_outputs(rows, outdir, structsplat_rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Native external-reference comparison")
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/native_reference_compare"))
    parser.add_argument("--gaussianimage-plus-repo", type=Path, required=True)
    parser.add_argument("--max-sides", nargs="+", type=int, default=[160])
    parser.add_argument("--budgets", nargs="+", type=int, default=[640])
    parser.add_argument("--start-fraction", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--growth-waves", type=int, default=4)
    parser.add_argument("--render-warmup", type=int, default=10)
    parser.add_argument("--render-repeats", type=int, default=50)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument(
        "--structsplat-metrics",
        type=Path,
        default=None,
        help="optional fair-harness metrics.json containing structsplat_best_default rows",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-cells", type=int, default=None)
    parser.add_argument("--device", default=None)
    from benchmarks.common import add_seed_args

    add_seed_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
