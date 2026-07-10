"""Single-image native Image-GS adapter.

This module deliberately imports no StructSplat code. It instantiates the upstream
``GaussianSplatting2D`` implementation from an exact checkout and records enough build and
trajectory provenance for the parent harness to reject cross-repository ``gsplat`` contamination.
Training renderer, loss, optimizer, initialization, growth, quantization, and stopping semantics
remain upstream-native for the selected profile.
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
from statistics import mean, median
from typing import Any


SCHEMA_VERSION = 1
ADAPTER_REVISION = "image_gs_v2"
METHOD = "image_gs_native"
FUSED_SSIM_COMMIT = "b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3"
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
PROFILES = (
    "matched_steps_fixed_n",
    "siggraph25",
    "release_quickstart",
    "release_default_float",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Image-GS single-image runner")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--initial-gaussians", type=int, required=True)
    parser.add_argument("--max-gaussians", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--growth-waves", type=int, default=4)
    parser.add_argument("--render-warmup", type=int, default=10)
    parser.add_argument("--render-repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _git(repo: Path, *args: str, text: bool = True):
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=text, stderr=subprocess.DEVNULL
    )


def _git_state(repo: Path) -> dict[str, Any]:
    commit = str(_git(repo, "rev-parse", "HEAD")).strip()
    tree = str(_git(repo, "rev-parse", "HEAD^{tree}")).strip()
    status = str(_git(repo, "status", "--porcelain", "--untracked-files=normal"))
    diff = _git(repo, "diff", "--binary", "HEAD", text=False)
    try:
        remote = str(_git(repo, "config", "--get", "remote.origin.url")).strip()
    except subprocess.CalledProcessError:
        remote = ""
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


def _direct_url_for(module_path: Path, distribution: str) -> tuple[Path, dict[str, Any]]:
    site_packages = next(
        parent for parent in module_path.parents if parent.name == "site-packages"
    )
    matches = sorted(site_packages.glob(f"{distribution}-*.dist-info/direct_url.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {distribution} direct_url.json in {site_packages}, found {len(matches)}"
        )
    path = matches[0].resolve()
    return path, json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _configure_profile(args: argparse.Namespace, cli: argparse.Namespace) -> None:
    args.seed = int(cli.seed)
    args.device = cli.device
    args.num_gaussians = int(cli.max_gaussians)
    args.max_steps = int(cli.iterations)
    args.initial_ratio = float(cli.initial_gaussians) / float(cli.max_gaussians)
    args.add_times = int(cli.growth_waves)
    args.add_steps = 500
    args.eval_steps = min(
        25 if cli.profile == "matched_steps_fixed_n" else 100,
        max(1, int(cli.iterations)),
    )
    args.save_image_steps = max(100_000, int(cli.iterations) + 1)
    args.save_ckpt_steps = max(100_000, int(cli.iterations) + 1)
    args.save_image_format = "png"
    args.vis_gaussians = False

    if cli.profile == "matched_steps_fixed_n":
        if cli.initial_gaussians != cli.max_gaussians:
            raise ValueError("matched_steps_fixed_n requires initial-gaussians == max-gaussians")
        args.disable_prog_optim = True
        args.disable_lr_schedule = True
        args.quantize = False
    else:
        canonical_steps = 5000 if cli.profile == "siggraph25" else 10000
        if cli.iterations != canonical_steps:
            raise ValueError(
                f"profile {cli.profile} requires exactly {canonical_steps} steps, "
                f"got {cli.iterations}"
            )
        if cli.growth_waves != 4 or cli.initial_gaussians != (cli.max_gaussians + 1) // 2:
            raise ValueError(
                f"profile {cli.profile} requires four growth waves and an exact 0.5 "
                "initial ratio"
            )
        args.initial_ratio = 0.5
        min_steps = args.add_steps * args.add_times + int(args.post_min_steps)
        if cli.iterations < min_steps:
            raise ValueError(
                f"profile {cli.profile} requires at least {min_steps} steps; "
                "the adapter will not silently rescale or extend native growth"
            )
        args.disable_prog_optim = False
        args.quantize = cli.profile != "release_default_float"
        args.disable_lr_schedule = cli.profile == "siggraph25"
    if args.quantize:
        args.pos_bits = args.scale_bits = args.rot_bits = args.feat_bits = 16
    else:
        args.pos_bits = args.scale_bits = args.rot_bits = args.feat_bits = 32


def main() -> None:
    cli = _parse_args()
    repo = cli.repo.resolve()
    image_path = cli.image.resolve()
    outdir = cli.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if cli.initial_gaussians <= 0 or cli.max_gaussians < cli.initial_gaussians:
        raise ValueError("expected 0 < initial-gaussians <= max-gaussians")
    if cli.iterations <= 0 or cli.growth_waves <= 0:
        raise ValueError("iterations and growth-waves must be positive")
    if cli.render_warmup < 0 or cli.render_repeats <= 0:
        raise ValueError("render-warmup must be nonnegative and render-repeats must be positive")
    if not cli.device.startswith("cuda"):
        raise ValueError("native Image-GS fitting requires a CUDA device such as cuda:0")

    state = _git_state(repo)
    if state["repo_dirty"]:
        raise RuntimeError(
            "Image-GS checkout must be clean for a native result; "
            f"status={state['repo_status']!r}"
        )

    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    import torch
    from PIL import Image

    import fused_ssim
    import fused_ssim_cuda
    import gsplat
    import gsplat.csrc as gsplat_csrc
    import main as upstream_main
    import model as upstream_model

    native_device = torch.device(cli.device)
    torch.cuda.set_device(native_device)

    # ``venv`` commonly symlinks its interpreter to the base Python, so the resolved
    # executable path is not the environment boundary. ``sys.prefix`` is authoritative.
    environment_root = Path(sys.prefix).resolve()
    model_path = Path(upstream_model.__file__).resolve()
    if model_path != (repo / "model.py").resolve():
        raise RuntimeError(f"Image-GS model provenance failure: {model_path}")

    gsplat_path = Path(gsplat.__file__).resolve()
    csrc_path = Path(gsplat_csrc.__file__).resolve()
    fused_package_path = Path(fused_ssim.__file__).resolve()
    fused_binary_path = Path(fused_ssim_cuda.__file__).resolve()
    for name, path in {
        "gsplat package": gsplat_path,
        "gsplat extension": csrc_path,
        "fused-ssim package": fused_package_path,
        "fused-ssim extension": fused_binary_path,
    }.items():
        try:
            path.relative_to(environment_root)
        except ValueError as exc:
            raise RuntimeError(f"{name} is outside isolated environment {environment_root}: {path}") from exc

    gsplat_direct_path, gsplat_direct = _direct_url_for(gsplat_path, "gsplat")
    fused_direct_path, fused_direct = _direct_url_for(fused_package_path, "fused_ssim")
    expected_gsplat_url = (repo / "gsplat").resolve().as_uri()
    if gsplat_direct.get("url") != expected_gsplat_url:
        raise RuntimeError(
            f"gsplat build provenance mismatch: {gsplat_direct.get('url')!r} != "
            f"{expected_gsplat_url!r}"
        )
    installed_gsplat_python_sha256 = _python_tree_sha256(gsplat_path.parent)
    repo_gsplat_python_sha256 = _python_tree_sha256(repo / "gsplat" / "gsplat")
    if installed_gsplat_python_sha256 != repo_gsplat_python_sha256:
        raise RuntimeError(
            "installed gsplat Python sources do not match the requested Image-GS checkout"
        )
    fused_commit = fused_direct.get("vcs_info", {}).get("commit_id")
    if fused_commit != FUSED_SSIM_COMMIT:
        raise RuntimeError(
            f"fused-ssim build provenance mismatch: {fused_commit!r} != "
            f"{FUSED_SSIM_COMMIT!r}"
        )
    fused_ssim_python_sha256 = _python_tree_sha256(fused_package_path.parent)
    dependency_versions = {
        name: importlib.metadata.version(name) for name in DEPENDENCY_DISTRIBUTIONS
    }

    parser = argparse.ArgumentParser()
    parser = upstream_main.load_cfg(cfg_path="cfgs/default.yaml", parser=parser)
    args = parser.parse_args([])
    _configure_profile(args, cli)
    # Preserve the upstream CLI's bit-field mutation/validation even though the adapter uses a
    # collision-free cell directory instead of its derived experiment name.
    upstream_main.get_gaussian_cfg(args)
    if int(args.topk) != 10 or bool(args.disable_tiles):
        raise ValueError(
            "the verified native adapter requires compiled TOP_K=10 and tiled rendering"
        )
    args.data_root = str(image_path.parent)
    args.input_path = image_path.name
    args.log_dir = str(outdir / "upstream")
    args.log_root = str(outdir)
    args.exp_name = "upstream"

    torch.hub.set_dir(str(outdir / "torch_models"))
    torch.cuda.synchronize(native_device)
    init_wall_start = time.perf_counter()
    trainer = upstream_model.GaussianSplatting2D(args)
    torch.cuda.synchronize(native_device)
    init_wall_seconds = time.perf_counter() - init_wall_start

    history_by_step: dict[int, dict[str, Any]] = {}
    trajectory_start = time.perf_counter()
    original_evaluate = trainer._evaluate

    def tracked_evaluate(log=True, upsample=False):
        psnr, ssim = original_evaluate(log=log, upsample=upsample)
        if not upsample:
            history_by_step[int(trainer.step)] = {
                "psnr": float(psnr),
                "ssim": float(ssim),
                "elapsed": time.perf_counter() - trajectory_start,
                "n_gaussians": int(trainer.num_gaussians),
            }
        return psnr, ssim

    trainer._evaluate = tracked_evaluate

    # Standard five-scale MS-SSIM is undefined for the small matched proxy. Preserve the
    # upstream terminal quantization/checkpoint path while leaving optional native extras blank;
    # the parent harness centrally recomputes shared metrics from the float reconstruction.
    terminal_metrics_patch = min(int(trainer.img_h), int(trainer.img_w)) <= 160
    if terminal_metrics_patch:
        def metric_safe_extra():
            trainer.lpips_final = float("nan")
            trainer.flip_final = float("nan")
            trainer.msssim_final = float("nan")

        trainer._evaluate_extra = metric_safe_extra

    torch.cuda.synchronize(native_device)
    fit_wall_start = time.perf_counter()
    trainer.optimize()
    torch.cuda.synchronize(native_device)
    fit_wall_seconds = time.perf_counter() - fit_wall_start

    with torch.no_grad():
        raw_render = trainer._render_images(upsample=False).float()
        display_render = torch.pow(
            torch.clamp(raw_render, 0.0, 1.0), 1.0 / float(trainer.gamma)
        )
    reconstruction = display_render.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    reconstruction_npy = outdir / "reconstruction.npy"
    reconstruction_png = outdir / "reconstruction.png"
    np.save(reconstruction_npy, reconstruction)
    Image.fromarray(np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0).astype(np.uint8)).save(
        reconstruction_png
    )

    timings_ms = []
    with torch.no_grad():
        for _ in range(max(0, cli.render_warmup)):
            trainer._render_images(upsample=False)
        torch.cuda.synchronize(native_device)
        for _ in range(max(1, cli.render_repeats)):
            torch.cuda.synchronize(native_device)
            start = time.perf_counter()
            trainer._render_images(upsample=False)
            torch.cuda.synchronize(native_device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    timings_ms.sort()

    history_steps = sorted(history_by_step)
    history = {
        "iter": history_steps,
        "psnr": [history_by_step[step]["psnr"] for step in history_steps],
        "ssim": [history_by_step[step]["ssim"] for step in history_steps],
        "elapsed": [history_by_step[step]["elapsed"] for step in history_steps],
        "n_gaussians": [history_by_step[step]["n_gaussians"] for step in history_steps],
    }
    checkpoint_path = Path(trainer.ckpt_dir) / f"ckpt_step-{trainer.step:d}.pt"
    analytical_bits = (
        2 * int(args.pos_bits)
        + 2 * int(args.scale_bits)
        + int(args.rot_bits)
        + int(trainer.feat_dim) * int(args.feat_bits)
    ) * int(trainer.total_num_gaussians)
    height, width = reconstruction.shape[:2]
    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_source_sha256": _sha256(Path(__file__).resolve()),
        "method": METHOD,
        "method_label": "Image-GS native",
        "implementation_kind": "native_external",
        "profile": cli.profile,
        "repo_root": str(repo),
        **state,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment_root": str(environment_root),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "native_device": str(native_device),
        "native_gpu_name": torch.cuda.get_device_name(native_device),
        "model_module": str(model_path),
        "gsplat_module": str(gsplat_path),
        "gsplat_csrc": str(csrc_path),
        "gsplat_csrc_sha256": _sha256(csrc_path),
        "gsplat_python_source_sha256": installed_gsplat_python_sha256,
        "repo_gsplat_python_source_sha256": repo_gsplat_python_sha256,
        "gsplat_direct_url_path": str(gsplat_direct_path),
        "gsplat_direct_url": gsplat_direct,
        "fused_ssim_module": str(fused_package_path),
        "fused_ssim_csrc": str(fused_binary_path),
        "fused_ssim_csrc_sha256": _sha256(fused_binary_path),
        "fused_ssim_python_sha256": fused_ssim_python_sha256,
        "native_dependency_versions": dependency_versions,
        "fused_ssim_direct_url_path": str(fused_direct_path),
        "fused_ssim_direct_url": fused_direct,
        "fused_ssim_commit": str(fused_commit),
        "source_path": str(image_path),
        "target_sha256": _sha256(image_path),
        "image": image_path.stem,
        "height": int(height),
        "width": int(width),
        "seed": int(cli.seed),
        "requested_iters": int(cli.iterations),
        "effective_max_steps": int(trainer.max_steps),
        "iterations_run": int(trainer.step),
        "stopped_early": int(trainer.step) < int(cli.iterations),
        "selection_policy": "terminal_step",
        "selected_iter": int(trainer.step),
        "budget_cap": int(cli.max_gaussians),
        "start_gaussians": int(history["n_gaussians"][0]),
        "requested_start_gaussians": int(cli.initial_gaussians),
        "n_gaussians": int(trainer.num_gaussians),
        "growth_waves_requested": int(cli.growth_waves),
        "growth_interval": None if args.disable_prog_optim else int(args.add_steps),
        "growth_policy": (
            "disabled; full-N native optimization"
            if args.disable_prog_optim
            else "upstream squared-error progressive allocation with Adam reset"
        ),
        "renderer": "Image-GS bundled gsplat top-k sum renderer",
        "optimizer": "Adam (upstream recreation after every growth wave)",
        "loss": "L1 + 0.1*(1-fused_ssim)",
        "quantized": bool(args.quantize),
        "attribute_bits": {
            "position": int(args.pos_bits),
            "scale": int(args.scale_bits),
            "rotation": int(args.rot_bits),
            "feature": int(args.feat_bits),
        },
        "native_reported_psnr": float(history["psnr"][-1]),
        "native_reported_ssim": float(history["ssim"][-1]),
        "native_periodic_reported_psnr": float(trainer.psnr_curr),
        "native_periodic_reported_ssim": float(trainer.ssim_curr),
        "native_reported_ms_ssim": (
            None if terminal_metrics_patch else float(trainer.msssim_final)
        ),
        "native_reported_lpips": None if terminal_metrics_patch else float(trainer.lpips_final),
        "native_reported_flip": None if terminal_metrics_patch else float(trainer.flip_final),
        "terminal_metrics_patch": bool(terminal_metrics_patch),
        "terminal_metrics_patch_reason": (
            "upstream fixed-five-scale MS-SSIM rejects images with a side <= 160; "
            "native optional extras left blank; shared PSNR/proxy-MS-SSIM and optional LPIPS "
            "are recomputed centrally (FLIP is not)"
            if terminal_metrics_patch
            else None
        ),
        "native_internal_optimization_seconds": float(trainer.total_time_accum),
        "native_internal_render_seconds": float(trainer.render_time_accum),
        "fit_seconds": float(fit_wall_seconds),
        "init_seconds": float(init_wall_seconds),
        "total_seconds": float(init_wall_seconds + fit_wall_seconds),
        "render_ms_mean": mean(timings_ms),
        "render_ms_median": median(timings_ms),
        "render_ms_p10": float(np.quantile(timings_ms, 0.10)),
        "render_ms_p90": float(np.quantile(timings_ms, 0.90)),
        "render_fps_median": 1000.0 / max(median(timings_ms), 1e-12),
        "parameter_count": int(sum(parameter.numel() for parameter in trainer.parameters())),
        "analytical_payload_bits": int(analytical_bits),
        "analytical_payload_bytes": float(analytical_bits / 8.0),
        "analytical_bpp": float(analytical_bits / (height * width)),
        "actual_codec_bytes": None,
        "actual_bpp": None,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "reconstruction_npy": str(reconstruction_npy),
        "reconstruction_png": str(reconstruction_png),
        "reconstruction_sha256": _sha256(reconstruction_npy),
        "reconstruction_dtype": str(reconstruction.dtype),
        "reconstruction_shape": list(reconstruction.shape),
        "history": history,
        "trajectory_eval_every": int(args.eval_steps),
        "target_hit_semantics": "first sampled native evaluation; interval-censored",
        "upstream_args": _json_safe(vars(args)),
    }
    result_path = outdir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
