"""Single-image adapter for the official GaussianImage representation code.

This module deliberately imports no StructSplat code.  It runs an exact external checkout in
an isolated Python process, keeps GaussianImage's renderer, parameterization, L2 loss, Adan
optimizer, and scheduler intact, and exports float pixels plus a provenance-rich manifest for a
parent benchmark to validate and score centrally.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any


SCHEMA_VERSION = 1
ADAPTER_REVISION = "gaussianimage_v1"
METHOD = "gaussianimage_native"
PROFILES = (
    "matched_steps_fixed_n",
    "release_cholesky",
    "release_rs",
    "paper_cholesky_50k_n70k",
)
DEPENDENCY_DISTRIBUTIONS = (
    "numpy",
    "pillow",
    "pytorch-msssim",
    "torchvision",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official GaussianImage single-image runner")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="matched_steps_fixed_n")
    parser.add_argument("--num-gaussians", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--render-warmup", type=int, default=10)
    parser.add_argument("--render-repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str, text: bool = True):
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=text, stderr=subprocess.DEVNULL
    )


def _git_state(repo: Path) -> dict[str, Any]:
    try:
        remote = str(_git(repo, "config", "--get", "remote.origin.url")).strip()
    except subprocess.CalledProcessError:
        remote = ""
    return {
        "commit": str(_git(repo, "rev-parse", "HEAD")).strip(),
        "tree": str(_git(repo, "rev-parse", "HEAD^{tree}")).strip(),
        "remote": remote,
        "status_porcelain": str(
            _git(repo, "status", "--porcelain", "--untracked-files=normal")
        ).splitlines(),
        "tracked_diff_sha256": hashlib.sha256(
            _git(repo, "diff", "--binary", "HEAD", text=False)
        ).hexdigest(),
    }


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _profile_model(profile: str) -> tuple[str, str]:
    if profile == "release_rs":
        return "GaussianImage_RS", "gaussianimage_rs.py"
    return "GaussianImage_Cholesky", "gaussianimage_cholesky.py"


def _validate_profile(profile: str, num_gaussians: int, iterations: int) -> None:
    if num_gaussians <= 0:
        raise ValueError("num-gaussians must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if profile == "paper_cholesky_50k_n70k" and (
        num_gaussians != 70_000 or iterations != 50_000
    ):
        raise ValueError(
            "paper_cholesky_50k_n70k requires --num-gaussians 70000 "
            "and --iterations 50000"
        )


def main() -> None:
    cli = _parse_args()
    repo = cli.repo.resolve()
    image_path = cli.image.resolve()
    outdir = cli.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    _validate_profile(cli.profile, cli.num_gaussians, cli.iterations)

    # Import the isolated environment's wheel before exposing the repository source tree. The
    # official setup builds the pinned submodule from a clean archive and deliberately leaves no
    # binary inside the checkout.
    os.chdir(repo)

    import numpy as np
    import torch
    from PIL import Image

    import gsplat
    import gsplat.csrc as gsplat_csrc

    sys.path.insert(0, str(repo))

    model_name, model_source_name = _profile_model(cli.profile)
    if model_name == "GaussianImage_Cholesky":
        from gaussianimage_cholesky import GaussianImage_Cholesky as Model
    else:
        from gaussianimage_rs import GaussianImage_RS as Model

    csrc_path = Path(gsplat_csrc.__file__).resolve()
    expected_extension_root = Path(gsplat.__file__).resolve().parent
    try:
        csrc_path.relative_to(expected_extension_root)
    except ValueError as exc:
        raise RuntimeError(
            f"GaussianImage extension provenance failure: {csrc_path} is not under "
            f"{expected_extension_root}"
        ) from exc

    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official GaussianImage gsplat execution requires CUDA")

    torch.manual_seed(cli.seed)
    random.seed(cli.seed)
    torch.cuda.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cli.seed)

    target_u8 = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    target = torch.from_numpy(target_u8.copy()).to(device=device, dtype=torch.float32)
    target = target.permute(2, 0, 1).unsqueeze(0) / 255.0
    height, width = target_u8.shape[:2]

    torch.cuda.synchronize(device)
    init_start = time.perf_counter()
    model = Model(
        loss_type="L2",
        opt_type="adan",
        num_points=int(cli.num_gaussians),
        H=height,
        W=width,
        BLOCK_H=16,
        BLOCK_W=16,
        device=device,
        lr=1e-3,
        quantize=False,
    ).to(device)
    torch.cuda.synchronize(device)
    init_seconds = time.perf_counter() - init_start

    history: dict[str, list[Any]] = {
        "iter": [],
        "psnr": [],
        "elapsed": [],
        "n_gaussians": [],
        "lr": [],
    }
    model.train()
    torch.cuda.synchronize(device)
    fit_start = time.perf_counter()
    for iteration in range(cli.iterations):
        _loss, psnr = model.train_iter(target)
        history["iter"].append(iteration)
        history["psnr"].append(float(psnr))
        history["elapsed"].append(time.perf_counter() - fit_start)
        history["n_gaussians"].append(int(cli.num_gaussians))
        history["lr"].append(float(model.optimizer.param_groups[0]["lr"]))
    torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - fit_start

    model.eval()
    with torch.no_grad():
        final_output = model()["render"].float()
    torch.cuda.synchronize(device)
    mse = torch.nn.functional.mse_loss(final_output, target).clamp_min(1e-12)
    native_psnr = float(10.0 * torch.log10(1.0 / mse))

    with torch.no_grad():
        for _ in range(max(0, cli.render_warmup)):
            model()
        torch.cuda.synchronize(device)
        timings_ms = []
        for _ in range(max(1, cli.render_repeats)):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            model()
            torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    timings_ms.sort()

    reconstruction = (
        final_output.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
    )
    reconstruction_npy = outdir / "reconstruction.npy"
    reconstruction_png = outdir / "reconstruction.png"
    np.save(reconstruction_npy, reconstruction)
    Image.fromarray(
        np.rint(np.clip(reconstruction, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB"
    ).save(reconstruction_png)

    checkpoint_path = outdir / "gaussian_model_terminal.pth.tar"
    torch.save(model.state_dict(), checkpoint_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_source = repo / model_source_name
    submodule_state = _git_state(repo / "gsplat")
    repo_state = _git_state(repo)
    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "method": METHOD,
        "method_label": "GaussianImage native",
        "implementation_kind": "native_external",
        "profile": cli.profile,
        "repo_root": str(repo),
        "repo_commit": repo_state["commit"],
        "repo_state": repo_state,
        "gsplat_commit": submodule_state["commit"],
        "gsplat_state": submodule_state,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "dependencies": _distribution_versions(),
        "gsplat_module": str(Path(gsplat.__file__).resolve()),
        "gsplat_csrc": str(csrc_path),
        "gsplat_csrc_sha256": _sha256(csrc_path),
        "model_source": str(model_source.resolve()),
        "model_source_sha256": _sha256(model_source),
        "image": image_path.stem,
        "source_path": str(image_path),
        "height": height,
        "width": width,
        "seed": cli.seed,
        "requested_iters": cli.iterations,
        "iterations_run": cli.iterations,
        "stopped_early": False,
        "budget_cap": cli.num_gaussians,
        "start_gaussians": cli.num_gaussians,
        "n_gaussians": cli.num_gaussians,
        "growth_waves_requested": 0,
        "growth_policy": "fixed random count",
        "renderer": "GaussianImage bundled gsplat sum renderer",
        "parameterization": model_name,
        "optimizer": "Adan",
        "learning_rate": 1e-3,
        "lr_schedule": "StepLR(step_size=20000,gamma=0.5)",
        "loss": "L2",
        "native_reported_psnr": native_psnr,
        "native_reported_ms_ssim": None,
        "selection_policy": "terminal",
        "selected_iter": cli.iterations,
        "selected_training_psnr": history["psnr"][-1],
        "trajectory_final_training_psnr": history["psnr"][-1],
        "selected_n_gaussians": cli.num_gaussians,
        "init_seconds": init_seconds,
        "fit_seconds": fit_seconds,
        "total_seconds": init_seconds + fit_seconds,
        "render_ms_mean": mean(timings_ms),
        "render_ms_median": median(timings_ms),
        "render_ms_p10": float(np.quantile(timings_ms, 0.10)),
        "render_ms_p90": float(np.quantile(timings_ms, 0.90)),
        "render_fps_median": 1000.0 / max(median(timings_ms), 1e-12),
        "parameter_count": int(parameter_count),
        "parameter_bpp_float32": float(parameter_count * 32 / (height * width)),
        "actual_codec_bytes": None,
        "actual_bpp": None,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "reconstruction_npy": str(reconstruction_npy),
        "reconstruction_png": str(reconstruction_png),
        "history": history,
    }
    result_path = outdir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
