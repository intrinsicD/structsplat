"""Native Image-GS benchmark with centrally recomputed StructSplat metrics.

The upstream repository runs in its own Python environment and subprocess. Four profiles are
explicitly separated because current release defaults, the compression quick-start, the SIGGRAPH
paper protocol, and a short fixed-N matched-step comparison are not the same experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks import native_reference_compare as shared
from benchmarks.common import (
    json_safe_rows,
    load_image,
    psnr_auc,
    resolve_seeds,
    run_config,
    save_image,
    write_config,
    write_csv,
    write_json,
)


SCHEMA_VERSION = 1
ADAPTER_REVISION = "image_gs_v2"
METRIC_PROTOCOL_REVISION = 2
METHOD = "image_gs_native"
METHOD_LABEL = "Image-GS native"
FUSED_SSIM_COMMIT = "b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3"
IMAGE_GS_COMMIT = "03088368d42684fb54225c981cfd94b58cc0393a"
OFFICIAL_ENVIRONMENT = {
    "python": "3.11.10",
    "torch": "2.4.1",
    "cuda": "12.4",
}
OFFICIAL_PINNED_DEPENDENCIES = {
    "numpy": "2.0.2",
    "scipy": "1.13.1",
    "opencv-python": "4.12.0.88",
    "scikit-image": "0.24.0",
    "torchvision": "0.19.1",
    "lpips": "0.1.4",
    "pytorch-msssim": "1.0.0",
    "torchmetrics": "1.5.2",
}
DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "opencv-python",
    "scikit-image",
    "torchvision",
    "pillow",
    "lpips",
    "pytorch-msssim",
    "torchmetrics",
)
PROFILES = {
    "matched_steps_fixed_n": {
        "protocol": "matched_steps_fixed_n",
        "description": (
            "Common final N and requested steps; Image-GS progressive allocation is disabled, "
            "so it starts at full N while StructSplat retains its pinned growth policy."
        ),
    },
    "siggraph25": {
        "protocol": "siggraph25_algorithm_profile",
        "description": (
            "Paper-aligned 5000-step, constant-LR, 16-bit analytical-payload algorithm profile "
            "with native progressive allocation, applied at the requested benchmark resolution."
        ),
    },
    "release_quickstart": {
        "protocol": "release_quickstart_algorithm_profile",
        "description": (
            "Current release defaults plus --quantize: 16-bit analytical payload, progressive "
            "allocation, and current LR-decay/early-stop schedule."
        ),
    },
    "release_default_float": {
        "protocol": "release_float_algorithm_profile",
        "description": (
            "Current bare-config float32 profile with progressive allocation and current "
            "LR-decay/early-stop schedule."
        ),
    },
}


def _repo_state(repo: Path) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo, text=True
    )
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    remote = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return {
        "repo_commit": commit,
        "repo_tree": tree,
        "repo_dirty": bool(status.strip()),
        "repo_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "repo_status": status.splitlines(),
        "repo_origin_url": remote,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _pixel_sha256(image: np.ndarray) -> str:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype="<i8").tobytes())
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _environment_fingerprint(
    python: Path,
    repo: Path,
    preload: Path | None,
) -> dict[str, Any]:
    script = r"""
import hashlib, importlib.metadata, json, pathlib, sys
import torch, fused_ssim, fused_ssim_cuda, gsplat, gsplat.csrc

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def python_tree_sha(root):
    digest = hashlib.sha256()
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()

def direct(module_path, name):
    site = next(p for p in pathlib.Path(module_path).parents if p.name == "site-packages")
    paths = sorted(site.glob(f"{name}-*.dist-info/direct_url.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} direct_url, found {len(paths)}")
    return str(paths[0].resolve()), json.loads(paths[0].read_text())

gp, fp = pathlib.Path(gsplat.__file__).resolve(), pathlib.Path(fused_ssim.__file__).resolve()
gd_path, gd = direct(gp, "gsplat")
fd_path, fd = direct(fp, "fused_ssim")
result = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "environment_root": str(pathlib.Path(sys.prefix).resolve()),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_devices": [
        {
            "index": idx,
            "name": torch.cuda.get_device_properties(idx).name,
            "total_memory": torch.cuda.get_device_properties(idx).total_memory,
            "compute_capability": [
                torch.cuda.get_device_properties(idx).major,
                torch.cuda.get_device_properties(idx).minor,
            ],
        }
        for idx in range(torch.cuda.device_count())
    ],
    "gsplat_module": str(gp),
    "gsplat_csrc": str(pathlib.Path(gsplat.csrc.__file__).resolve()),
    "gsplat_csrc_sha256": sha(gsplat.csrc.__file__),
    "gsplat_python_source_sha256": python_tree_sha(gp.parent),
    "gsplat_direct_url_path": gd_path,
    "gsplat_direct_url": gd,
    "fused_ssim_module": str(fp),
    "fused_ssim_csrc": str(pathlib.Path(fused_ssim_cuda.__file__).resolve()),
    "fused_ssim_csrc_sha256": sha(fused_ssim_cuda.__file__),
    "fused_ssim_python_sha256": python_tree_sha(fp.parent),
    "native_dependency_versions": {
        name: importlib.metadata.version(name)
        for name in (
            "numpy", "scipy", "opencv-python", "scikit-image", "torchvision",
            "pillow", "lpips", "pytorch-msssim", "torchmetrics"
        )
    },
    "fused_ssim_direct_url_path": fd_path,
    "fused_ssim_direct_url": fd,
}
print(json.dumps(result))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["PYTHONNOUSERSITE"] = "1"
    if preload is not None:
        env["LD_PRELOAD"] = str(preload)
    completed = subprocess.run(
        [str(python), "-c", script],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Image-GS environment preflight failed: "
            + (completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown")
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_url = (repo / "gsplat").resolve().as_uri()
    if payload.get("gsplat_direct_url", {}).get("url") != expected_url:
        raise RuntimeError(
            f"Image-GS gsplat build source mismatch: "
            f"{payload.get('gsplat_direct_url', {}).get('url')!r} != {expected_url!r}"
        )
    repo_gsplat_python_sha256 = _python_tree_sha256(repo / "gsplat" / "gsplat")
    if payload.get("gsplat_python_source_sha256") != repo_gsplat_python_sha256:
        raise RuntimeError(
            "installed Image-GS gsplat Python sources do not match the requested checkout"
        )
    payload["repo_gsplat_python_source_sha256"] = repo_gsplat_python_sha256
    fused_commit = payload.get("fused_ssim_direct_url", {}).get("vcs_info", {}).get("commit_id")
    if fused_commit != FUSED_SSIM_COMMIT:
        raise RuntimeError(
            f"Image-GS fused-ssim commit mismatch: {fused_commit!r} != {FUSED_SSIM_COMMIT!r}"
        )
    payload["fused_ssim_commit"] = str(fused_commit)
    payload["libstdcxx_preload"] = None if preload is None else str(preload)
    payload["libstdcxx_sha256"] = None if preload is None else _sha256(preload)
    payload["official_environment_expected"] = OFFICIAL_ENVIRONMENT
    payload["official_pinned_dependencies_expected"] = OFFICIAL_PINNED_DEPENDENCIES
    payload["official_environment_match"] = bool(
        str(payload["python_version"]).startswith(OFFICIAL_ENVIRONMENT["python"])
        and str(payload["torch_version"]).split("+")[0] == OFFICIAL_ENVIRONMENT["torch"]
        and str(payload["torch_cuda_version"]) == OFFICIAL_ENVIRONMENT["cuda"]
        and all(
            payload["native_dependency_versions"].get(name) == version
            for name, version in OFFICIAL_PINNED_DEPENDENCIES.items()
        )
    )
    return payload


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("schema_version", 0) or 0),
        row.get("adapter_revision"),
        int(row.get("metric_protocol_revision", 0) or 0),
        row.get("protocol"),
        row.get("profile"),
        row.get("method"),
        row.get("repo_commit"),
        row.get("repo_tree"),
        row.get("repo_diff_sha256"),
        row.get("gsplat_csrc_sha256"),
        row.get("fused_ssim_csrc_sha256"),
        row.get("fused_ssim_commit"),
        row.get("fused_ssim_python_sha256"),
        json.dumps(row.get("native_dependency_versions"), sort_keys=True),
        row.get("python_executable"),
        row.get("python_version"),
        row.get("environment_root"),
        row.get("torch_version"),
        row.get("torch_cuda_version"),
        json.dumps(row.get("cuda_devices"), sort_keys=True),
        row.get("libstdcxx_preload"),
        row.get("libstdcxx_sha256"),
        row.get("native_adapter_source_sha256"),
        row.get("shared_compare_source_sha256"),
        row.get("structsplat_metrics_source_sha256"),
        row.get("metric_python_version"),
        row.get("metric_torch_version"),
        row.get("metric_numpy_version"),
        row.get("metric_lpips_version"),
        row.get("metric_torchvision_version"),
        row.get("source_path"),
        row.get("source_sha256"),
        row.get("target_sha256"),
        row.get("target_pixel_sha256"),
        int(row.get("max_side", 0) or 0),
        int(row.get("budget_cap", 0) or 0),
        int(row.get("requested_start_gaussians", 0) or 0),
        int(row.get("requested_iters", 0) or 0),
        int(row.get("seed", 0) or 0),
        int(row.get("growth_waves_requested", 0) or 0),
        int(row.get("render_warmup", 0) or 0),
        int(row.get("render_repeats", 0) or 0),
        bool(row.get("lpips_requested", False)),
        row.get("native_device"),
        row.get("metric_device"),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            if line.strip():
                rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print(f"Ignoring torn final journal line in {path}", flush=True)
                break
            raise
    return rows


def _native_command(
    *,
    python: Path,
    repo: Path,
    target_path: Path,
    cell_dir: Path,
    profile: str,
    start_gaussians: int,
    budget: int,
    iters: int,
    seed: int,
    growth_waves: int,
    render_warmup: int,
    render_repeats: int,
    native_device: str,
) -> list[str]:
    runner = Path(__file__).resolve().parent / "native_runners" / "image_gs.py"
    return [
        str(python),
        str(runner),
        "--repo", str(repo),
        "--image", str(target_path),
        "--outdir", str(cell_dir),
        "--profile", profile,
        "--initial-gaussians", str(start_gaussians),
        "--max-gaussians", str(budget),
        "--iterations", str(iters),
        "--seed", str(seed),
        "--growth-waves", str(growth_waves),
        "--render-warmup", str(render_warmup),
        "--render-repeats", str(render_repeats),
        "--device", native_device,
    ]


def _finite_nonnegative(manifest: dict[str, Any], name: str) -> float:
    try:
        value = float(manifest[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"native manifest lacks finite {name}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"native manifest has invalid {name}={value!r}")
    return value


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    repo: Path,
    repo_state: dict[str, Any],
    environment: dict[str, Any],
    target_path: Path,
    target_shape: tuple[int, int],
    profile: str,
    budget: int,
    requested_start: int,
    requested_iters: int,
    seed: int,
    growth_waves: int,
    target_sha256: str,
    native_device: str,
) -> np.ndarray:
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "method": METHOD,
        "profile": profile,
        "repo_commit": repo_state["repo_commit"],
        "repo_tree": repo_state["repo_tree"],
        "repo_diff_sha256": repo_state["repo_diff_sha256"],
        "repo_origin_url": repo_state["repo_origin_url"],
        "adapter_source_sha256": environment["native_adapter_source_sha256"],
        "seed": seed,
        "requested_iters": requested_iters,
        "effective_max_steps": requested_iters,
        "budget_cap": budget,
        "requested_start_gaussians": requested_start,
        "growth_waves_requested": growth_waves,
        "target_sha256": target_sha256,
        "native_device": native_device,
        "height": target_shape[0],
        "width": target_shape[1],
    }
    for name, expected in expected_scalars.items():
        value = manifest.get(name)
        if value != expected:
            raise ValueError(f"native manifest {name}={value!r} does not match {expected!r}")
    if Path(manifest["repo_root"]).resolve() != repo.resolve() or manifest.get("repo_dirty"):
        raise ValueError("native manifest repository provenance is dirty or mismatched")
    if Path(manifest["source_path"]).resolve() != target_path.resolve():
        raise ValueError("native manifest source path does not match the benchmark target")
    for name in (
        "environment_root",
        "python_executable",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "gsplat_module",
        "gsplat_csrc",
        "gsplat_csrc_sha256",
        "gsplat_python_source_sha256",
        "repo_gsplat_python_source_sha256",
        "gsplat_direct_url_path",
        "gsplat_direct_url",
        "fused_ssim_module",
        "fused_ssim_csrc",
        "fused_ssim_csrc_sha256",
        "fused_ssim_python_sha256",
        "native_dependency_versions",
        "fused_ssim_direct_url_path",
        "fused_ssim_direct_url",
        "fused_ssim_commit",
    ):
        if manifest.get(name) != environment.get(name):
            raise ValueError(f"native environment field {name} does not match preflight")
    try:
        native_index = int(native_device.split(":", 1)[1]) if ":" in native_device else 0
        expected_gpu_name = environment["cuda_devices"][native_index]["name"]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid native CUDA device {native_device!r}") from exc
    if manifest.get("native_gpu_name") != expected_gpu_name:
        raise ValueError("native GPU name does not match the preflight device")
    expected_patch = min(target_shape) <= 160
    if manifest.get("terminal_metrics_patch") is not expected_patch:
        raise ValueError("native terminal-metric patch state does not match the image dimensions")

    iterations_run = int(manifest.get("iterations_run", -1))
    if not 0 < iterations_run <= requested_iters:
        raise ValueError(
            f"invalid native iteration count {iterations_run} for request {requested_iters}"
        )
    if profile in {"matched_steps_fixed_n", "siggraph25"} and iterations_run != requested_iters:
        raise ValueError(f"fixed-horizon profile stopped at {iterations_run}/{requested_iters}")
    n_gaussians = int(manifest.get("n_gaussians", 0))
    if n_gaussians != budget:
        raise ValueError(f"native result did not reach the exact count: {n_gaussians}/{budget}")
    if int(manifest.get("selected_iter", -1)) != iterations_run:
        raise ValueError("Image-GS result is not labeled as a terminal-step selection")
    if manifest.get("selection_policy") != "terminal_step":
        raise ValueError("unexpected Image-GS selection policy")

    reconstruction_path = Path(manifest["reconstruction_npy"])
    reconstruction = np.load(reconstruction_path, allow_pickle=False)
    expected_shape = (*target_shape, 3)
    if (
        reconstruction.shape != expected_shape
        or reconstruction.dtype != np.float32
        or not np.isfinite(reconstruction).all()
    ):
        raise ValueError(
            f"invalid native reconstruction shape/values: {reconstruction.shape}, "
            f"expected {expected_shape}"
        )
    if manifest.get("reconstruction_dtype") != "float32":
        raise ValueError("native reconstruction must be recorded as float32")
    if manifest.get("reconstruction_shape") != list(expected_shape):
        raise ValueError("native reconstruction shape metadata does not match its contents")
    if manifest.get("reconstruction_sha256") != _sha256(reconstruction_path):
        raise ValueError("native reconstruction hash does not match its contents")

    for name in (
        "fit_seconds",
        "init_seconds",
        "total_seconds",
        "native_internal_optimization_seconds",
        "native_internal_render_seconds",
        "render_ms_mean",
        "render_ms_median",
        "render_ms_p10",
        "render_ms_p90",
        "render_fps_median",
        "native_reported_psnr",
        "native_reported_ssim",
        "analytical_payload_bytes",
        "analytical_bpp",
    ):
        _finite_nonnegative(manifest, name)
    if manifest.get("actual_codec_bytes") is not None or manifest.get("actual_bpp") is not None:
        raise ValueError("Image-GS analytical payload was mislabeled as an actual codec stream")

    upstream_args = manifest.get("upstream_args")
    if not isinstance(upstream_args, dict):
        raise ValueError("native manifest lacks resolved upstream arguments")
    expected_profile_args = {
        "disable_prog_optim": profile == "matched_steps_fixed_n",
        "disable_lr_schedule": profile in {"matched_steps_fixed_n", "siggraph25"},
        "quantize": profile in {"siggraph25", "release_quickstart"},
        "topk": 10,
        "disable_tiles": False,
        "num_gaussians": budget,
        "max_steps": requested_iters,
    }
    for name, expected in expected_profile_args.items():
        if upstream_args.get(name) != expected:
            raise ValueError(
                f"native upstream argument {name}={upstream_args.get(name)!r} "
                f"does not match {expected!r}"
            )

    history = manifest.get("history")
    if not isinstance(history, dict):
        raise ValueError("native manifest lacks evaluation trajectory")
    fields = [history.get(name) for name in ("iter", "psnr", "ssim", "elapsed", "n_gaussians")]
    if not fields or not all(isinstance(values, list) and values for values in fields):
        raise ValueError("native trajectory fields must be non-empty lists")
    if len({len(values) for values in fields}) != 1:
        raise ValueError("native trajectory fields have inconsistent lengths")
    steps = [int(value) for value in fields[0]]
    if steps[0] != 0 or steps[-1] != iterations_run or any(
        right <= left for left, right in zip(steps, steps[1:])
    ):
        raise ValueError("native trajectory steps must increase from zero to terminal step")
    if not all(math.isfinite(float(value)) for values in fields[1:4] for value in values):
        raise ValueError("native trajectory contains non-finite values")
    counts = [int(value) for value in fields[4]]
    if counts[0] != requested_start or counts[-1] != n_gaussians:
        raise ValueError("native trajectory start/final counts do not match requested/result counts")
    if any(not 0 < count <= budget for count in counts):
        raise ValueError("native trajectory count violates cap")
    return reconstruction.astype(np.float32, copy=False)


def _run_cell(
    *,
    python: Path,
    repo: Path,
    repo_state: dict[str, Any],
    environment: dict[str, Any],
    preload: Path | None,
    source_path: Path,
    target_path: Path,
    target: np.ndarray,
    max_side: int,
    profile: str,
    budget: int,
    start_gaussians: int,
    iters: int,
    seed: int,
    growth_waves: int,
    render_warmup: int,
    render_repeats: int,
    device: str,
    native_device: str,
    want_lpips: bool,
    cell_dir: Path,
) -> dict[str, Any]:
    source_sha256 = _sha256(source_path)
    target_sha256 = _sha256(target_path)
    target_pixel_sha256 = _pixel_sha256(target)
    command = _native_command(
        python=python,
        repo=repo,
        target_path=target_path,
        cell_dir=cell_dir,
        profile=profile,
        start_gaussians=start_gaussians,
        budget=budget,
        iters=iters,
        seed=seed,
        growth_waves=growth_waves,
        render_warmup=render_warmup,
        render_repeats=render_repeats,
        native_device=native_device,
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(repo)
    if preload is not None:
        env["LD_PRELOAD"] = str(preload)
    cell_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = cell_dir / "stdout.log", cell_dir / "stderr.log"
    wall_start = time.perf_counter()
    completed = subprocess.run(
        command, cwd=repo, env=env, text=True, capture_output=True, check=False
    )
    subprocess_wall_seconds = time.perf_counter() - wall_start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    base = {
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
        "protocol": PROFILES[profile]["protocol"],
        "profile": profile,
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "implementation_kind": "native_external",
        "repo_root": str(repo),
        **repo_state,
        **environment,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "target_path": str(target_path),
        "target_sha256": target_sha256,
        "target_pixel_sha256": target_pixel_sha256,
        "image": source_path.stem,
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        "max_side": int(max_side),
        "seed": int(seed),
        "requested_iters": int(iters),
        "budget_cap": int(budget),
        "requested_start_gaussians": int(start_gaussians),
        "growth_waves_requested": int(growth_waves),
        "render_warmup": int(render_warmup),
        "render_repeats": int(render_repeats),
        "lpips_requested": bool(want_lpips),
        "device": device,
        "metric_device": device,
        "native_device": native_device,
        "command": command,
        "libstdcxx_preload": None if preload is None else str(preload),
        "subprocess_wall_seconds": subprocess_wall_seconds,
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
            repo_state=repo_state,
            environment=environment,
            target_path=target_path,
            target_shape=target.shape[:2],
            profile=profile,
            budget=budget,
            requested_start=start_gaussians,
            requested_iters=iters,
            seed=seed,
            growth_waves=growth_waves,
            target_sha256=target_sha256,
            native_device=native_device,
        )
        central = shared._central_metrics(
            reconstruction, target, device=device, want_lpips=want_lpips
        )
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error": f"invalid native result: {type(exc).__name__}: {exc}",
        }
    history = manifest["history"]
    return {
        **base,
        **manifest,
        **central,
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
        "protocol": PROFILES[profile]["protocol"],
        "profile": profile,
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "repo_root": str(repo),
        **repo_state,
        **environment,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "target_pixel_sha256": target_pixel_sha256,
        "image": source_path.stem,
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        "max_side": int(max_side),
        "seed": int(seed),
        "requested_iters": int(iters),
        "budget_cap": int(budget),
        "requested_start_gaussians": int(start_gaussians),
        "growth_waves_requested": int(growth_waves),
        "render_warmup": int(render_warmup),
        "render_repeats": int(render_repeats),
        "lpips_requested": bool(want_lpips),
        "device": device,
        "metric_device": device,
        "native_device": native_device,
        "auc_psnr": psnr_auc(history, nominal_iters=iters),
        "iters_to_targets": shared._target_hits(
            history, [22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
        ),
        "command": command,
        "libstdcxx_preload": None if preload is None else str(preload),
        "subprocess_wall_seconds": subprocess_wall_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "metric_protocol": "display-clamped StructSplat central metrics",
        "status": "ok",
        "error": "",
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    outdir: Path,
    structsplat_rows: list[dict[str, Any]],
    *,
    baseline_methods: list[str] | None = None,
    compact_journal: bool = False,
) -> None:
    if compact_journal:
        with (outdir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, default=str) + "\n")
    json_rows = json_safe_rows(
        rows,
        skip={"history", "upstream_args", "gsplat_direct_url", "fused_ssim_direct_url"},
    )
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        fields = sorted(
            key for row in json_rows for key in row
            if key not in {"command", "iters_to_targets", "repo_status", "attribute_bits"}
        )
        write_csv(outdir / "metrics.csv", json_rows, fieldnames=fields, extrasaction="ignore")
    ok = [row for row in rows if row.get("status") == "ok"]
    profile = str(ok[0]["profile"]) if ok else ""
    paired_results = []
    for baseline_method in baseline_methods or ["structsplat_best_default"]:
        paired_rows, paired_summary = shared._paired_native_vs_structsplat(
            rows,
            structsplat_rows,
            native_method=METHOD,
            baseline_method=baseline_method,
            relation_metrics=("psnr", "ms_ssim"),
            require_start_match=profile != "matched_steps_fixed_n",
        )
        if paired_summary is None:
            continue
        paired_summary["timing_protocol_comparable"] = False
        paired_summary["strict_implementation_dominance_tested"] = False
        paired_results.append((baseline_method, paired_rows, paired_summary))
        suffix = baseline_method.replace("/", "_")
        write_csv(
            outdir / f"paired_native_vs_{suffix}.csv",
            paired_rows,
            fieldnames=sorted({key for row in paired_rows for key in row}),
        )
        write_csv(
            outdir / f"paired_native_vs_{suffix}_summary.csv",
            [paired_summary],
            fieldnames=list(paired_summary),
        )
        if baseline_method == "structsplat_best_default":
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

    if profile == "matched_steps_fixed_n":
        comparison_axis_note = (
            "Final N and requested steps match, but Image-GS starts at full N while the pinned "
            "StructSplat row starts at half N and grows."
        )
    else:
        comparison_axis_note = (
            "Start N, final N, requested steps, and target pixels match; native loss, renderer, "
            "growth policy/wave count, and final 16-bit Image-GS quantization remain algorithm-"
            "specific. Analytical bpp is not a byte-matched rate constraint."
        )
    lines = [
        "# Native Image-GS Comparison",
        "",
        "This artifact executes the pinned official Image-GS checkout in an isolated environment. "
        "Metrics are centrally recomputed from float reconstructions; upstream-reported metrics, analytical "
        "payload estimates, synchronized wall timing, and synchronized end-to-end render timing "
        "are retained separately.",
        "",
        f"Profile `{ok[0]['profile']}`: {PROFILES[ok[0]['profile']]['description']}" if ok else "",
        "",
        "Image-GS emits no packed codec stream. `analytical_bpp` follows its documented attribute-"
        "bit formula and omits headers/min-max metadata; `actual_bpp` remains blank. Native "
        "trajectory samples use Image-GS's evaluation cadence rather than adding per-step GPU "
        "synchronization. Target hits are interval-censored at that cadence. Final Image-GS "
        "fields are terminal-step selections. `proxy_ms_ssim` is the shared small-image adaptive "
        "proxy, not the paper's fixed five-scale native MS-SSIM.",
        "",
        (
            "Official environment reproduction: yes."
            if ok and ok[0].get("official_environment_match")
            else "Official environment reproduction: no; algorithm/build provenance is pinned, "
            "but the recorded Python/Torch/CUDA versions differ from the official environment."
        ),
        "",
        "| Image | Profile | Side | Cap | Start | Seed | Steps | PSNR | Proxy MS-SSIM | LPIPS | AUC | Sync fit s | Native self s | Render FPS | Analytical bpp | Commit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ok:
        lpips = "-" if row.get("lpips") is None else f"{float(row['lpips']):.4f}"
        lines.append(
            f"| {row['image']} | {row['profile']} | {row['max_side']} | {row['budget_cap']} | "
            f"{row['start_gaussians']} | {row['seed']} | {row['iterations_run']} | "
            f"{float(row['psnr']):.4f} | {float(row['ms_ssim']):.5f} | {lpips} | "
            f"{float(row['auc_psnr']):.4f} | {float(row['fit_seconds']):.3f} | "
            f"{float(row['native_internal_optimization_seconds']):.3f} | "
            f"{float(row['render_fps_median']):.1f} | {float(row['analytical_bpp']):.3f} | "
            f"{str(row['repo_commit'])[:12]} |"
        )
    for baseline_method, _paired_rows, paired_summary in paired_results:
        lines += [
            "",
            f"## Paired Image-GS vs `{baseline_method}`",
            "",
            "Positive is an Image-GS gain; timing and LPIPS signs are inverted so positive always "
            "means better. Displayed intervals are marginal 95% image-bootstrap intervals; a "
            "final-quality relation uses PSNR and proxy MS-SSIM with Bonferroni-adjusted 95% "
            "familywise bounds. LPIPS is reported separately. AUC is diagnostic only because "
            "the native histories use different render clamping/cadence semantics. Paired rows "
            "require identical run-recorded decoded-pixel hashes. "
            f"{comparison_axis_note} Image-GS synchronized fit wall includes its terminal image logging and "
            "checkpoint write, while StructSplat fit timing does not; timing deltas are therefore "
            "diagnostic and the displayed relation is not a strict implementation-dominance test.",
            "",
            "| Pairs / images | PSNR gain [95% CI] | Proxy MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | Diagnostic AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Final-quality relation | Familywise 95% relation |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|",
            f"| {paired_summary['pairs']} / {paired_summary['images']} | "
            f"{shared._format_gain_ci(paired_summary, 'psnr', 4)} | "
            f"{shared._format_gain_ci(paired_summary, 'ms_ssim', 5)} | "
            f"{shared._format_gain_ci(paired_summary, 'lpips', 4)} | "
            f"{shared._format_gain_ci(paired_summary, 'auc_psnr', 4)} | "
            f"{shared._format_gain_ci(paired_summary, 'fit_seconds', 4)} | "
            f"{shared._format_gain_ci(paired_summary, 'total_seconds', 4)} | "
            f"{str(paired_summary['sample_relation']).replace('_', ' ')} | "
            f"{str(paired_summary['supported_relation_95ci']).replace('_', ' ')} |",
        ]
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- `{row.get('image')}`: {row.get('error')}" for row in errors)
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    links = [
        '<a href="summary.md">summary.md</a>',
        '<a href="metrics.csv">metrics.csv</a>',
        '<a href="metrics.json">metrics.json</a>',
        '<a href="config.json">config.json</a>',
    ]
    for baseline_method, _paired_rows, _paired_summary in paired_results:
        suffix = baseline_method.replace("/", "_")
        links += [
            f'<a href="paired_native_vs_{suffix}.csv">paired {baseline_method}</a>',
            f'<a href="paired_native_vs_{suffix}_summary.csv">summary {baseline_method}</a>',
        ]
    html = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        "<title>Native Image-GS Comparison</title>",
        '<style>body{font-family:system-ui;margin:24px;max-width:1100px;line-height:1.45}.note{background:#f5f5f5;border-left:4px solid #666;padding:10px}img{max-width:480px;border:1px solid #ddd}</style></head><body>',
        "<h1>Native Image-GS Comparison</h1>",
        '<p class="note"><b>Pinned native code, explicit profile.</b> Analytical payload is not a packed codec stream; central metrics come from float reconstructions. MS-SSIM is the shared small-image proxy.</p>',
        "<p>" + " · ".join(links) + "</p>",
    ]
    for row in ok:
        try:
            relative = Path(row["reconstruction_png"]).relative_to(outdir)
        except ValueError:
            continue
        html.append(
            f"<h2>{row['image']} — {row['profile']} — {row['budget_cap']}G</h2>"
            f"<p>PSNR {row['psnr']:.4f}, MS-SSIM {row['ms_ssim']:.5f}, "
            f"fit {row['fit_seconds']:.3f}s, render {row['render_fps_median']:.1f} FPS</p>"
            f'<a href="{relative}"><img src="{relative}" alt="Image-GS reconstruction"></a>'
        )
    html.append("</body></html>")
    (outdir / "index.html").write_text("\n".join(html) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo = args.image_gs_repo.resolve()
    # Do not resolve a venv interpreter symlink to the base Python; invoking the symlink is
    # what activates the environment-specific ``sys.prefix`` and site-packages.
    python = args.image_gs_python.expanduser().absolute()
    preload = args.libstdcxx_preload.resolve() if args.libstdcxx_preload is not None else None
    if preload is not None and not preload.is_file():
        raise FileNotFoundError(f"libstdc++ preload does not exist: {preload}")
    repo_state = _repo_state(repo)
    if repo_state["repo_dirty"]:
        raise RuntimeError(f"Image-GS checkout is dirty: {repo_state['repo_status']!r}")
    if repo_state["repo_commit"] != args.expected_image_gs_commit:
        raise RuntimeError(
            f"Image-GS commit {repo_state['repo_commit']} does not match the required "
            f"commit {args.expected_image_gs_commit}"
        )
    environment = _environment_fingerprint(python, repo, preload)
    runner_source = Path(__file__).resolve().parent / "native_runners" / "image_gs.py"
    environment.update({
        "native_adapter_source_sha256": _sha256(runner_source),
        "native_harness_source_sha256": _sha256(Path(__file__).resolve()),
        "shared_compare_source_sha256": _sha256(Path(shared.__file__).resolve()),
        "structsplat_metrics_source_sha256": _sha256(Path(shared.M.__file__).resolve()),
        "metric_python_version": sys.version,
        "metric_torch_version": torch.__version__,
        "metric_numpy_version": np.__version__,
        "metric_lpips_version": importlib.metadata.version("lpips"),
        "metric_torchvision_version": importlib.metadata.version("torchvision"),
    })
    if not args.native_device.startswith("cuda"):
        raise ValueError("--native-device must name a CUDA device, for example cuda:0")
    try:
        native_index = int(args.native_device.split(":", 1)[1]) if ":" in args.native_device else 0
    except ValueError as exc:
        raise ValueError(f"invalid --native-device {args.native_device!r}") from exc
    if not 0 <= native_index < int(environment["cuda_device_count"]):
        raise ValueError(
            f"native CUDA device index {native_index} is unavailable; "
            f"found {environment['cuda_device_count']} device(s)"
        )
    if any(side < 0 for side in args.max_sides):
        raise ValueError("--max-sides values must be nonnegative (0 means native resolution)")
    if any(budget <= 0 for budget in args.budgets) or args.iters <= 0:
        raise ValueError("budgets and iterations must be positive")
    if args.render_warmup < 0 or args.render_repeats <= 0:
        raise ValueError("render-warmup must be nonnegative and render-repeats must be positive")
    if args.profile == "siggraph25" and args.iters != 5000:
        raise ValueError("siggraph25 requires --iters 5000")
    if args.profile in {"release_quickstart", "release_default_float"} and args.iters != 10000:
        raise ValueError(f"{args.profile} requires --iters 10000")
    if args.profile != "matched_steps_fixed_n":
        if args.growth_waves != 4 or args.start_fraction != 0.5:
            raise ValueError(
                f"{args.profile} requires --growth-waves 4 and --start-fraction 0.5"
            )
    seeds = resolve_seeds(args.seed, args.seeds)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    structsplat_rows = (
        shared._load_json_rows(args.structsplat_metrics)
        if args.structsplat_metrics is not None
        else []
    )
    write_config(str(outdir), run_config({
        "protocol": PROFILES[args.profile]["protocol"],
        "profile": args.profile,
        "profile_description": PROFILES[args.profile]["description"],
        "method": METHOD,
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
        "repo_root": str(repo),
        "expected_image_gs_commit": args.expected_image_gs_commit,
        **repo_state,
        **environment,
        "image_gs_python": str(python),
        "libstdcxx_preload": None if preload is None else str(preload),
        "native_device": args.native_device,
        "metric_device": device,
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
            None if args.structsplat_metrics is None else str(args.structsplat_metrics)
        ),
        "structsplat_methods": args.structsplat_methods,
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
        source_id = f"{source.stem}_{source_sha256[:12]}"
        for max_side in args.max_sides:
            resize_limit = None if max_side == 0 else max_side
            target = load_image(source, max_side=resize_limit)
            side_label = "native" if max_side == 0 else str(max_side)
            target_path = outdir / "targets" / f"{source_id}_s{side_label}.png"
            save_image(target, target_path)
            target_sha256 = _sha256(target_path)
            target_pixel_sha256 = _pixel_sha256(target)
            for budget in args.budgets:
                start = (
                    int(budget)
                    if args.profile == "matched_steps_fixed_n"
                    else (int(budget) + 1) // 2
                )
                for seed in seeds:
                    cell_index += 1
                    key_row = {
                        "schema_version": SCHEMA_VERSION,
                        "adapter_revision": ADAPTER_REVISION,
                        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
                        "protocol": PROFILES[args.profile]["protocol"],
                        "profile": args.profile,
                        "method": METHOD,
                        **repo_state,
                        **environment,
                        "source_path": str(source),
                        "source_sha256": source_sha256,
                        "target_sha256": target_sha256,
                        "target_pixel_sha256": target_pixel_sha256,
                        "max_side": int(max_side),
                        "budget_cap": int(budget),
                        "requested_start_gaussians": int(start),
                        "requested_iters": int(args.iters),
                        "seed": int(seed),
                        "growth_waves_requested": int(args.growth_waves),
                        "render_warmup": int(args.render_warmup),
                        "render_repeats": int(args.render_repeats),
                        "lpips_requested": bool(args.lpips),
                        "native_device": args.native_device,
                        "metric_device": device,
                    }
                    key = _cell_key(key_row)
                    key_digest = hashlib.sha256(
                        json.dumps(key, default=str).encode("utf-8")
                    ).hexdigest()[:12]
                    cell_dir = (
                        outdir / "cells" / source_id
                        / (
                            f"{args.profile}_s{max_side}_n{budget}_i{args.iters}_seed{seed}"
                            f"_k{key_digest}"
                        )
                    )
                    if key in selected_keys:
                        print(
                            f"[{cell_index}/{total}] skip duplicate requested cell {source.stem} "
                            f"{args.profile} s={max_side} n={budget} seed={seed}", flush=True
                        )
                        continue
                    cached = existing_by_key.get(key)
                    cache_valid = False
                    if cached is not None:
                        try:
                            cached_manifest_path = (
                                Path(str(cached["reconstruction_npy"])).parent / "result.json"
                            )
                            manifest = json.loads(
                                cached_manifest_path.read_text(encoding="utf-8")
                            )
                            reconstruction = _validate_manifest(
                                manifest,
                                repo=repo,
                                repo_state=repo_state,
                                environment=environment,
                                target_path=target_path.resolve(),
                                target_shape=target.shape[:2],
                                profile=args.profile,
                                budget=budget,
                                requested_start=start,
                                requested_iters=args.iters,
                                seed=seed,
                                growth_waves=args.growth_waves,
                                target_sha256=target_sha256,
                                native_device=args.native_device,
                            )
                            cached.update(
                                shared._central_metrics(
                                    reconstruction,
                                    target,
                                    device=device,
                                    want_lpips=args.lpips,
                                )
                            )
                            cached["auc_psnr"] = psnr_auc(
                                manifest["history"], nominal_iters=args.iters
                            )
                            cached["iters_to_targets"] = shared._target_hits(
                                manifest["history"], [22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
                            )
                            for name, value in key_row.items():
                                cached[name] = value
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
                        print(
                            f"[{cell_index}/{total}] skip existing {source.stem} "
                            f"{args.profile} s={max_side} n={budget} seed={seed}", flush=True
                        )
                        continue
                    print(
                        f"[{cell_index}/{total}] native Image-GS {source.stem} {args.profile} "
                        f"s={max_side} n={budget} seed={seed}", flush=True
                    )
                    row = _run_cell(
                        python=python,
                        repo=repo,
                        repo_state=repo_state,
                        environment=environment,
                        preload=preload,
                        source_path=source,
                        target_path=target_path.resolve(),
                        target=target,
                        max_side=max_side,
                        profile=args.profile,
                        budget=budget,
                        start_gaussians=start,
                        iters=args.iters,
                        seed=seed,
                        growth_waves=args.growth_waves,
                        render_warmup=args.render_warmup,
                        render_repeats=args.render_repeats,
                        device=device,
                        native_device=args.native_device,
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
                            f"render={row['render_fps_median']:.1f}fps", flush=True
                        )
                    else:
                        print(f"  ERROR {row['error']}", flush=True)
                    if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                        _write_outputs(
                            rows,
                            outdir,
                            structsplat_rows,
                            baseline_methods=args.structsplat_methods,
                        )
                        return rows
    _write_outputs(
        rows,
        outdir,
        structsplat_rows,
        baseline_methods=args.structsplat_methods,
        compact_journal=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Native Image-GS comparison")
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/native_image_gs_compare"))
    parser.add_argument("--image-gs-repo", type=Path, required=True)
    parser.add_argument("--image-gs-python", type=Path, required=True)
    parser.add_argument("--expected-image-gs-commit", default=IMAGE_GS_COMMIT)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="matched_steps_fixed_n")
    parser.add_argument("--max-sides", nargs="+", type=int, default=[160])
    parser.add_argument("--budgets", nargs="+", type=int, default=[640])
    parser.add_argument("--start-fraction", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--growth-waves", type=int, default=4)
    parser.add_argument("--render-warmup", type=int, default=10)
    parser.add_argument("--render-repeats", type=int, default=50)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--structsplat-metrics", type=Path, default=None)
    parser.add_argument(
        "--structsplat-methods",
        nargs="+",
        default=["structsplat_best_default"],
        help="StructSplat methods in the supplied metrics artifact to pair separately",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-cells", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--native-device", default="cuda:0")
    default_preload = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    parser.add_argument(
        "--libstdcxx-preload",
        type=Path,
        default=default_preload if default_preload.is_file() else None,
    )
    from benchmarks.common import add_seed_args

    add_seed_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
