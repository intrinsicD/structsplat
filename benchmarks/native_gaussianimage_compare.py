"""Official GaussianImage representation benchmark with central metric recomputation.

The native subprocess uses GaussianImage's exact renderer, Cholesky/RS parameterization, L2
loss, Adan optimizer, and terminal-state policy.  This parent process owns preprocessing,
artifact validation, central metrics, target hashes, cache keys, and paired comparison against a
StructSplat fair-harness artifact.  ``matched_axes`` aligns image pixels, fixed Gaussian count,
requested steps, and seed; it is not a same-hyperparameter claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
ADAPTER_REVISION = "gaussianimage_v1"
METRIC_PROTOCOL_REVISION = 2
METHOD = "gaussianimage_native"
METHOD_LABEL = "GaussianImage native"
PROTOCOL = "matched_axes_fixed_n"
PROFILES = (
    "matched_steps_fixed_n",
    "release_cholesky",
    "release_rs",
    "paper_cholesky_50k_n70k",
)
OFFICIAL_GAUSSIANIMAGE_COMMIT = "d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1"
OFFICIAL_GSPLAT_COMMIT = "bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: np.ndarray) -> str:
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(pixels.shape, dtype="<i8").tobytes())
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


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
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint_checkout(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    submodule = (repo / "gsplat").resolve()
    model_sources = {
        name: _sha256(repo / name)
        for name in ("gaussianimage_cholesky.py", "gaussianimage_rs.py")
    }
    return {
        "repo_root": str(repo),
        "repo_commit": _git(repo, "rev-parse", "HEAD"),
        "repo_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "repo_tracked_diff_sha256": _tracked_diff_sha256(repo),
        "gsplat_root": str(submodule),
        "gsplat_commit": _git(submodule, "rev-parse", "HEAD"),
        "gsplat_tree": _git(submodule, "rev-parse", "HEAD^{tree}"),
        "gsplat_tracked_diff_sha256": _tracked_diff_sha256(submodule),
        "repo_gsplat_python_source_sha256": _python_tree_sha256(submodule / "gsplat"),
        "model_source_sha256": model_sources,
    }


def _environment_fingerprint(
    python: Path,
    repo: Path,
    preload: Path | None,
) -> dict[str, Any]:
    script = r"""
import hashlib, importlib.metadata, json, pathlib, sys
import torch, gsplat, gsplat.csrc

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def python_tree_sha(root):
    digest = hashlib.sha256()
    root = pathlib.Path(root)
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()

versions = {}
for name in ("numpy", "pillow", "pytorch-msssim", "torchvision"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None

print(json.dumps({
    "python_executable": sys.executable,
    "python_version": sys.version,
    "environment_root": str(pathlib.Path(sys.prefix).resolve()),
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
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
    "gsplat_module": str(pathlib.Path(gsplat.__file__).resolve()),
    "gsplat_csrc": str(pathlib.Path(gsplat.csrc.__file__).resolve()),
    "gsplat_csrc_sha256": sha(gsplat.csrc.__file__),
    "gsplat_version": importlib.metadata.version("gsplat"),
    "gsplat_python_source_sha256": python_tree_sha(pathlib.Path(gsplat.__file__).parent),
    "native_dependency_versions": versions,
}))
"""
    env = os.environ.copy()
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
        detail = completed.stderr.strip().splitlines()[-1:] or ["unknown failure"]
        raise RuntimeError(f"GaussianImage environment preflight failed: {detail[0]}")
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    extension = Path(payload["gsplat_csrc"]).resolve()
    module = Path(payload["gsplat_module"]).resolve()
    expected_root = Path(payload["environment_root"]).resolve()
    for name, path in (("module", module), ("extension", extension)):
        try:
            path.relative_to(expected_root)
        except ValueError as exc:
            raise RuntimeError(
                f"GaussianImage installed gsplat {name} provenance failure: {path} is outside "
                f"{expected_root}"
            ) from exc
    repo_python_sha256 = _python_tree_sha256(repo / "gsplat" / "gsplat")
    if payload["gsplat_python_source_sha256"] != repo_python_sha256:
        raise RuntimeError(
            "installed GaussianImage gsplat Python sources do not match the pinned submodule"
        )
    wheel_paths = sorted((expected_root.parent / "wheels").glob("gsplat-*.whl"))
    if len(wheel_paths) != 1:
        raise RuntimeError(f"expected one retained gsplat wheel, found {len(wheel_paths)}")
    payload["gsplat_wheel"] = str(wheel_paths[0].resolve())
    payload["gsplat_wheel_sha256"] = _sha256(wheel_paths[0])
    payload["libstdcxx_preload"] = None if preload is None else str(preload)
    payload["libstdcxx_sha256"] = None if preload is None else _sha256(preload)
    return payload


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("schema_version", 0) or 0),
        row.get("adapter_revision"),
        int(row.get("metric_protocol_revision", 0) or 0),
        row.get("protocol"),
        row.get("profile"),
        row.get("repo_commit"),
        row.get("gsplat_commit"),
        row.get("gsplat_csrc_sha256"),
        row.get("native_adapter_source_sha256"),
        row.get("native_harness_source_sha256"),
        row.get("metric_protocol_source_sha256"),
        row.get("shared_compare_source_sha256"),
        row.get("environment_root"),
        row.get("torch_version"),
        row.get("torch_cuda_version"),
        row.get("source_path"),
        row.get("target_pixel_sha256"),
        int(row.get("max_side", 0) or 0),
        int(row.get("budget_cap", 0) or 0),
        int(row.get("requested_iters", 0) or 0),
        int(row.get("seed", 0) or 0),
        int(row.get("render_warmup", 0) or 0),
        int(row.get("render_repeats", 0) or 0),
        bool(row.get("lpips_requested", False)),
        row.get("native_python"),
        row.get("libstdcxx_sha256"),
        row.get("device"),
    )


def _native_command(
    *,
    native_python: Path,
    repo: Path,
    image: Path,
    cell_dir: Path,
    profile: str,
    budget: int,
    iters: int,
    seed: int,
    render_warmup: int,
    render_repeats: int,
    device: str,
) -> list[str]:
    runner = Path(__file__).resolve().parent / "native_runners" / "gaussianimage.py"
    return [
        str(native_python),
        str(runner),
        "--repo", str(repo),
        "--image", str(image),
        "--outdir", str(cell_dir),
        "--profile", profile,
        "--num-gaussians", str(budget),
        "--iterations", str(iters),
        "--seed", str(seed),
        "--render-warmup", str(render_warmup),
        "--render-repeats", str(render_repeats),
        "--device", device,
    ]


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    fingerprint: dict[str, Any],
    target_path: Path,
    target_shape: tuple[int, int],
    profile: str,
    budget: int,
    iters: int,
    seed: int,
) -> np.ndarray:
    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "method": METHOD,
        "profile": profile,
        "repo_commit": fingerprint["repo_commit"],
        "gsplat_commit": fingerprint["gsplat_commit"],
        "seed": seed,
        "requested_iters": iters,
        "iterations_run": iters,
        "budget_cap": budget,
        "start_gaussians": budget,
        "n_gaussians": budget,
        "selected_n_gaussians": budget,
        "selected_iter": iters,
    }
    for name, expected in expected_scalars.items():
        value = manifest.get(name)
        if isinstance(expected, int):
            try:
                matches = int(value) == expected
            except (TypeError, ValueError):
                matches = False
        else:
            matches = value == expected
        if not matches:
            raise ValueError(f"manifest {name}={value!r} does not match {expected!r}")
    if Path(str(manifest.get("repo_root"))).resolve() != Path(
        fingerprint["repo_root"]
    ).resolve():
        raise ValueError("manifest repository root does not match requested checkout")
    if Path(str(manifest.get("source_path"))).resolve() != target_path.resolve():
        raise ValueError("manifest source path does not match benchmark target")
    if manifest.get("selection_policy") != "terminal":
        raise ValueError("GaussianImage representation must export its terminal state")
    states = (
        ("repo_state", "repo_tracked_diff_sha256"),
        ("gsplat_state", "gsplat_tracked_diff_sha256"),
    )
    for state_name, fingerprint_name in states:
        state = manifest.get(state_name)
        if not isinstance(state, dict) or (
            state.get("tracked_diff_sha256") != fingerprint[fingerprint_name]
        ):
            raise ValueError(f"manifest {state_name} does not match the preflight source state")
    if (int(manifest.get("height", -1)), int(manifest.get("width", -1))) != target_shape:
        raise ValueError("manifest reconstruction shape does not match target")

    extension = Path(str(manifest.get("gsplat_csrc"))).resolve()
    if extension != Path(fingerprint["gsplat_csrc"]).resolve():
        raise ValueError("manifest loaded a different gsplat extension")
    if not extension.is_file() or _sha256(extension) != fingerprint["gsplat_csrc_sha256"]:
        raise ValueError("loaded gsplat extension hash changed")
    if manifest.get("gsplat_csrc_sha256") != fingerprint["gsplat_csrc_sha256"]:
        raise ValueError("manifest gsplat hash does not match harness fingerprint")

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
        value = float(manifest.get(name, float("nan")))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"manifest has invalid {name}={value!r}")

    history = manifest.get("history")
    if not isinstance(history, dict):
        raise ValueError("manifest lacks trajectory history")
    fields = [history.get(name) for name in ("iter", "psnr", "elapsed", "n_gaussians")]
    if not all(isinstance(values, list) and len(values) == iters for values in fields):
        raise ValueError("trajectory fields do not match requested iteration count")
    if [int(value) for value in fields[0]] != list(range(iters)):
        raise ValueError("trajectory iteration indices are not contiguous")
    if not all(np.isfinite(float(value)) for value in fields[1]):
        raise ValueError("trajectory contains non-finite PSNR")
    elapsed = [float(value) for value in fields[2]]
    if any(right < left for left, right in zip(elapsed, elapsed[1:])):
        raise ValueError("trajectory elapsed values are not nondecreasing")
    if any(int(value) != budget for value in fields[3]):
        raise ValueError("GaussianImage fixed-N trajectory changed Gaussian count")

    reconstruction_path = Path(str(manifest.get("reconstruction_npy")))
    reconstruction = np.load(reconstruction_path, allow_pickle=False)
    expected_shape = (*target_shape, 3)
    if reconstruction.shape != expected_shape or not np.isfinite(reconstruction).all():
        raise ValueError(
            f"invalid float reconstruction {reconstruction.shape}; expected {expected_shape}"
        )
    checkpoint = Path(str(manifest.get("checkpoint_path")))
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError("manifest terminal checkpoint is missing")
    return reconstruction.astype(np.float32, copy=False)


def _run_cell(
    *,
    native_python: Path,
    repo: Path,
    fingerprint: dict[str, Any],
    preload: Path | None,
    source_path: Path,
    target_path: Path,
    target: np.ndarray,
    target_hash: str,
    max_side: int,
    profile: str,
    budget: int,
    iters: int,
    seed: int,
    render_warmup: int,
    render_repeats: int,
    device: str,
    want_lpips: bool,
    cell_dir: Path,
) -> dict[str, Any]:
    command = _native_command(
        native_python=native_python,
        repo=repo,
        image=target_path,
        cell_dir=cell_dir,
        profile=profile,
        budget=budget,
        iters=iters,
        seed=seed,
        render_warmup=render_warmup,
        render_repeats=render_repeats,
        device=device,
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    if preload is not None:
        env["LD_PRELOAD"] = str(preload)
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
        "profile": profile,
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "implementation_kind": "native_external",
        **fingerprint,
        "source_path": str(source_path),
        "target_path": str(target_path),
        "target_pixel_sha256": target_hash,
        "image": source_path.stem,
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        "max_side": max_side,
        "seed": seed,
        "requested_iters": iters,
        "budget_cap": budget,
        "start_gaussians": budget,
        "start_fraction": 1.0,
        "growth_waves_requested": 0,
        "render_warmup": render_warmup,
        "render_repeats": render_repeats,
        "lpips_requested": want_lpips,
        "native_python": str(native_python),
        "libstdcxx_preload": None if preload is None else str(preload),
        "libstdcxx_sha256": None if preload is None else _sha256(preload),
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
            fingerprint=fingerprint,
            target_path=target_path,
            target_shape=target.shape[:2],
            profile=profile,
            budget=budget,
            iters=iters,
            seed=seed,
        )
        central = shared._central_metrics(
            reconstruction,
            target,
            device=device,
            want_lpips=want_lpips,
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
        **base,
        "auc_psnr": psnr_auc(history, nominal_iters=iters),
        "iters_to_targets": shared._target_hits(
            history, [22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
        ),
        "status": "ok",
        "error": "",
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    outdir: Path,
    structsplat_rows: list[dict[str, Any]],
    baseline_methods: list[str] | None = None,
) -> None:
    # Compact the execution journal to the exact current request.  Resume may read an older
    # journal, but stale source pixels or evaluator revisions must not survive into new evidence.
    with (outdir / "metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, default=str) + "\n")
    json_rows = json_safe_rows(rows, skip={"history", "repo_state", "gsplat_state"})
    write_json(outdir / "metrics.json", json_rows)
    if json_rows:
        fieldnames = sorted(
            {
                key
                for row in json_rows
                for key in row
                if key not in {"command", "iters_to_targets", "dependencies"}
            }
        )
        write_csv(
            outdir / "metrics.csv",
            json_rows,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
    paired_results = []
    for baseline_method in baseline_methods or ["structsplat_best_default"]:
        pairs, paired = shared._paired_native_vs_structsplat(
            rows,
            structsplat_rows,
            native_method=METHOD,
            baseline_method=baseline_method,
        )
        if paired is None:
            continue
        paired_results.append((baseline_method, pairs, paired))
        suffix = baseline_method.replace("/", "_")
        write_csv(
            outdir / f"paired_native_vs_{suffix}.csv",
            pairs,
            fieldnames=sorted({key for row in pairs for key in row}),
        )
        write_csv(
            outdir / f"paired_native_vs_{suffix}_summary.csv",
            [paired],
            fieldnames=list(paired),
        )
        if baseline_method == "structsplat_best_default":
            write_csv(
                outdir / "paired_native_vs_structsplat.csv",
                pairs,
                fieldnames=sorted({key for row in pairs for key in row}),
            )
            write_csv(
                outdir / "paired_native_vs_structsplat_summary.csv",
                [paired],
                fieldnames=list(paired),
            )

    ok = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# Native GaussianImage Comparison",
        "",
        "`matched_axes_fixed_n` aligns decoded target pixels, fixed Gaussian count, requested "
        "steps, and seed. GaussianImage retains its native renderer, Cholesky/RS "
        "parameterization, L2 loss, Adan optimizer, and scheduler.",
        "",
        "Shared PSNR, SSIM, proxy MS-SSIM, and LPIPS are centrally recomputed from exported "
        "float pixels. Fit/render timings are explicitly CUDA-synchronized. GaussianImage's "
        "representation path exports the terminal state and does not produce a codec bitstream.",
        "",
        "| Profile | Image | Side | N | Seed | PSNR | MS-SSIM | LPIPS | AUC | Fit s | Render FPS | Param bpp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ok:
        lpips = "-" if row.get("lpips") is None else f"{float(row['lpips']):.4f}"
        lines.append(
            f"| {row['profile']} | {row['image']} | {row['max_side']} | "
            f"{row['budget_cap']} | {row['seed']} | {float(row['psnr']):.4f} | "
            f"{float(row['ms_ssim']):.5f} | {lpips} | {float(row['auc_psnr']):.4f} | "
            f"{float(row['fit_seconds']):.3f} | {float(row['render_fps_median']):.1f} | "
            f"{float(row['parameter_bpp_float32']):.3f} |"
        )
    for baseline_method, _pairs, paired in paired_results:
        lines += [
            "",
            f"## Paired Native vs `{baseline_method}`",
            "",
            "Positive is a native GaussianImage gain; positive time/LPIPS gains mean lower is "
            "better. Intervals bootstrap source images after averaging correlated seeds. The "
            "familywise relation uses Bonferroni-adjusted bounds across the five core metrics.",
            "",
            "| Pairs / images | PSNR gain | MS-SSIM gain | LPIPS gain | AUC gain | Fit gain s | Total gain s | Sample relation | Familywise relation |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|",
            f"| {paired['pairs']} / {paired['images']} | "
            f"{shared._format_gain_ci(paired, 'psnr', 4)} | "
            f"{shared._format_gain_ci(paired, 'ms_ssim', 5)} | "
            f"{shared._format_gain_ci(paired, 'lpips', 4)} | "
            f"{shared._format_gain_ci(paired, 'auc_psnr', 4)} | "
            f"{shared._format_gain_ci(paired, 'fit_seconds', 4)} | "
            f"{shared._format_gain_ci(paired, 'total_seconds', 4)} | "
            f"{str(paired['sample_relation']).replace('_', ' ')} | "
            f"{str(paired['supported_relation_95ci']).replace('_', ' ')} |",
        ]
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        lines += ["", "## Errors", ""]
        lines.extend(f"- `{row.get('image')}`: {row.get('error')}" for row in errors)
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo = args.gaussianimage_repo.resolve()
    # Do not resolve a venv/Conda interpreter symlink to its base executable: invoking the
    # environment path is what preserves environment-specific sys.prefix and site-packages.
    native_python = args.native_python.expanduser().absolute()
    if not native_python.is_file():
        raise FileNotFoundError(f"native Python does not exist: {native_python}")
    preload = args.libstdcxx_preload.resolve() if args.libstdcxx_preload else None
    if preload is not None and not preload.is_file():
        raise FileNotFoundError(f"libstdc++ preload does not exist: {preload}")
    fingerprint = _fingerprint_checkout(repo)
    if fingerprint["repo_commit"] != args.expected_gaussianimage_commit:
        raise RuntimeError(
            f"GaussianImage commit {fingerprint['repo_commit']} does not match required "
            f"{args.expected_gaussianimage_commit}"
        )
    if fingerprint["gsplat_commit"] != args.expected_gsplat_commit:
        raise RuntimeError(
            f"GaussianImage gsplat commit {fingerprint['gsplat_commit']} does not match "
            f"required {args.expected_gsplat_commit}"
        )
    empty_diff_sha256 = hashlib.sha256(b"").hexdigest()
    for name in ("repo_tracked_diff_sha256", "gsplat_tracked_diff_sha256"):
        if fingerprint[name] != empty_diff_sha256:
            raise RuntimeError(f"official GaussianImage checkout has tracked changes: {name}")
    environment = _environment_fingerprint(native_python, repo, preload)
    source_fingerprint = {
        "native_adapter_source_sha256": _sha256(
            Path(__file__).resolve().parent / "native_runners" / "gaussianimage.py"
        ),
        "native_harness_source_sha256": _sha256(Path(__file__).resolve()),
        "metric_protocol_source_sha256": _sha256(Path(shared.M.__file__).resolve()),
        "shared_compare_source_sha256": _sha256(Path(shared.__file__).resolve()),
    }
    fingerprint.update(environment)
    fingerprint.update(source_fingerprint)
    seeds = resolve_seeds(args.seed, args.seeds)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    structsplat_rows = (
        shared._load_json_rows(args.structsplat_metrics)
        if args.structsplat_metrics is not None
        else []
    )
    write_config(
        str(outdir),
        run_config(
            {
                "protocol": PROTOCOL,
                "profile": args.profile,
                "method": METHOD,
                "schema_version": SCHEMA_VERSION,
                "adapter_revision": ADAPTER_REVISION,
                "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
                **fingerprint,
                "native_python": str(native_python),
                "libstdcxx_preload": None if preload is None else str(preload),
                "libstdcxx_sha256": None if preload is None else _sha256(preload),
                "images": [str(path) for path in args.images],
                "max_sides": args.max_sides,
                "budgets": args.budgets,
                "iters": args.iters,
                "seeds": seeds,
                "render_warmup": args.render_warmup,
                "render_repeats": args.render_repeats,
                "lpips": args.lpips,
                "structsplat_metrics": (
                    None if args.structsplat_metrics is None else str(args.structsplat_metrics)
                ),
                "structsplat_methods": args.structsplat_methods,
                "resume": args.resume,
                "max_new_cells": args.max_new_cells,
            },
            device=device,
        ),
    )

    jsonl_path = outdir / "metrics.jsonl"
    existing = shared._load_jsonl(jsonl_path) if args.resume else []
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
        for max_side in args.max_sides:
            target = load_image(source, max_side=max_side)
            target_hash = _pixel_sha256(target)
            target_path = outdir / "targets" / f"{source.stem}_s{max_side}.png"
            save_image(target, target_path)
            for budget in args.budgets:
                for seed in seeds:
                    cell_index += 1
                    key_row = {
                        "schema_version": SCHEMA_VERSION,
                        "adapter_revision": ADAPTER_REVISION,
                        "metric_protocol_revision": METRIC_PROTOCOL_REVISION,
                        "protocol": PROTOCOL,
                        "profile": args.profile,
                        **fingerprint,
                        "source_path": str(source),
                        "target_pixel_sha256": target_hash,
                        "max_side": max_side,
                        "budget_cap": budget,
                        "requested_iters": args.iters,
                        "seed": seed,
                        "render_warmup": args.render_warmup,
                        "render_repeats": args.render_repeats,
                        "lpips_requested": args.lpips,
                        "native_python": str(native_python),
                        "libstdcxx_sha256": None if preload is None else _sha256(preload),
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
                                fingerprint=fingerprint,
                                target_path=target_path.resolve(),
                                target_shape=target.shape[:2],
                                profile=args.profile,
                                budget=budget,
                                iters=args.iters,
                                seed=seed,
                            )
                            cached.update(
                                shared._central_metrics(
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
                            cached["iters_to_targets"] = shared._target_hits(
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
                        print(
                            f"[{cell_index}/{total}] skip {source.stem} s={max_side} "
                            f"n={budget} seed={seed}",
                            flush=True,
                        )
                        continue
                    print(
                        f"[{cell_index}/{total}] native GaussianImage {source.stem} "
                        f"s={max_side} n={budget} seed={seed}",
                        flush=True,
                    )
                    cell_dir = (
                        outdir
                        / "cells"
                        / source.stem
                        / f"s{max_side}_n{budget}_seed{seed}"
                    ).resolve()
                    row = _run_cell(
                        native_python=native_python,
                        repo=repo,
                        fingerprint=fingerprint,
                        preload=preload,
                        source_path=source,
                        target_path=target_path.resolve(),
                        target=target,
                        target_hash=target_hash,
                        max_side=max_side,
                        profile=args.profile,
                        budget=budget,
                        iters=args.iters,
                        seed=seed,
                        render_warmup=args.render_warmup,
                        render_repeats=args.render_repeats,
                        device=device,
                        want_lpips=args.lpips,
                        cell_dir=cell_dir,
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
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Official GaussianImage native comparison")
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--outdir", type=Path, default=Path("results/native_gaussianimage_compare")
    )
    parser.add_argument("--gaussianimage-repo", type=Path, required=True)
    parser.add_argument("--native-python", type=Path, required=True)
    parser.add_argument(
        "--expected-gaussianimage-commit",
        default=OFFICIAL_GAUSSIANIMAGE_COMMIT,
    )
    parser.add_argument("--expected-gsplat-commit", default=OFFICIAL_GSPLAT_COMMIT)
    parser.add_argument("--profile", choices=PROFILES, default="matched_steps_fixed_n")
    parser.add_argument("--max-sides", nargs="+", type=int, default=[160])
    parser.add_argument("--budgets", nargs="+", type=int, default=[640])
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--render-warmup", type=int, default=10)
    parser.add_argument("--render-repeats", type=int, default=50)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--structsplat-metrics", type=Path, default=None)
    parser.add_argument(
        "--structsplat-methods",
        nargs="+",
        default=["structsplat_best_default"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-cells", type=int, default=None)
    parser.add_argument("--device", default=None)
    default_preload = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    parser.add_argument(
        "--libstdcxx-preload",
        type=Path,
        default=default_preload if default_preload.exists() else None,
    )
    from benchmarks.common import add_seed_args

    add_seed_args(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
