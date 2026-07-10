"""Single-image native GaussianImage++ adapter.

This file deliberately does not import StructSplat. It instantiates the upstream
``SimpleTrainer2d`` directly because the public GaussianImage++ CLI hard-codes complete
Kodak/DIV2K filename loops. The adapter only supplies orchestration and instrumentation:
the optimizer, renderer, loss, pruning, and residual growth remain upstream-native.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any


SCHEMA_VERSION = 2
ADAPTER_REVISION = "gaussianimage_plus_v2"


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native GaussianImage++ single-image runner")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--initial-gaussians", type=int, required=True)
    parser.add_argument("--max-gaussians", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--growth-waves", type=int, default=4)
    parser.add_argument("--render-warmup", type=int, default=10)
    parser.add_argument("--render-repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    cli = _parse_args()
    repo = cli.repo.resolve()
    image_path = cli.image.resolve()
    outdir = cli.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if cli.initial_gaussians <= 0 or cli.max_gaussians < cli.initial_gaussians:
        raise ValueError("expected 0 < initial-gaussians <= max-gaussians")
    if cli.iterations <= 0:
        raise ValueError("iterations must be positive")

    # Put this exact checkout and its bundled extension first. The base environment may
    # contain an editable gsplat from a different reference repository.
    sys.path.insert(0, str(repo / "gsplat"))
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    import torch
    from PIL import Image

    import gsplat
    import gsplat.csrc as gsplat_csrc
    import train as upstream

    csrc_path = Path(gsplat_csrc.__file__).resolve()
    expected_extension_root = (repo / "gsplat").resolve()
    try:
        csrc_path.relative_to(expected_extension_root)
    except ValueError as exc:
        raise RuntimeError(
            f"GaussianImage++ extension provenance failure: {csrc_path} is not under "
            f"{expected_extension_root}"
        ) from exc

    args = upstream.parse_args([])
    args.model_name = "GaussianImage_Covariance"
    args.num_points = int(cli.initial_gaussians)
    args.max_num_points = int(cli.max_gaussians)
    args.iterations = int(cli.iterations)
    args.seed = int(cli.seed)
    args.adaptive_add = cli.max_gaussians > cli.initial_gaussians
    args.grow_iter = max(1, cli.iterations // max(1, cli.growth_waves + 1))
    args.prune_iter = max(1, min(int(args.prune_iter), cli.iterations))
    args.save_imgs = False
    args.print = False
    args.quantize = False
    args.data_name = "structsplat_native_adapter"

    torch.manual_seed(cli.seed)
    random.seed(cli.seed)
    torch.cuda.manual_seed(cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cli.seed)

    torch.cuda.synchronize()
    init_start = time.perf_counter()
    trainer = upstream.SimpleTrainer2d(
        image_path=image_path,
        log_dir=str(outdir / "native_logs"),
        num_points=cli.initial_gaussians,
        iterations=cli.iterations,
        model_path=None,
        args=args,
    )
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_start

    history: dict[str, list[Any]] = {
        "iter": [],
        "psnr": [],
        "elapsed": [],
        "n_gaussians": [],
    }
    original_train_iter = trainer.gaussian_model.train_iter
    trajectory_start = time.perf_counter()

    def tracked_train_iter(*call_args, **call_kwargs):
        result = original_train_iter(*call_args, **call_kwargs)
        history["iter"].append(len(history["iter"]))
        history["psnr"].append(float(result[1]))
        history["elapsed"].append(time.perf_counter() - trajectory_start)
        history["n_gaussians"].append(int(trainer.gaussian_model.cur_num_points))
        return result

    trainer.gaussian_model.train_iter = tracked_train_iter

    render_stats: dict[str, Any] = {}
    final_render = None

    # Upstream pytorch_msssim requires a minimum side >160 for its fixed five-scale
    # protocol. The fair proxy uses smaller targets, so replace only the terminal metric
    # helper; training itself remains upstream-native and all final metrics are recomputed
    # centrally from the exported float reconstruction.
    def instrumented_test():
        nonlocal final_render
        trainer.gaussian_model.eval()
        with torch.no_grad():
            output = trainer.gaussian_model(H=trainer.H, W=trainer.W)["render"].float()
            final_render = output.detach().cpu()
            mse = torch.nn.functional.mse_loss(output, trainer.gt_image.float()).clamp_min(1e-12)
            native_psnr = float(10.0 * torch.log10(1.0 / mse))
            for _ in range(max(0, cli.render_warmup)):
                trainer.gaussian_model(H=trainer.H, W=trainer.W)
            torch.cuda.synchronize()
            timings_ms = []
            for _ in range(max(1, cli.render_repeats)):
                torch.cuda.synchronize()
                start = time.perf_counter()
                trainer.gaussian_model(H=trainer.H, W=trainer.W)
                torch.cuda.synchronize()
                timings_ms.append((time.perf_counter() - start) * 1000.0)
        timings_ms.sort()
        render_stats.update({
            "render_ms_mean": mean(timings_ms),
            "render_ms_median": median(timings_ms),
            "render_ms_p10": float(np.quantile(timings_ms, 0.10)),
            "render_ms_p90": float(np.quantile(timings_ms, 0.90)),
            "render_fps_median": 1000.0 / max(median(timings_ms), 1e-12),
        })
        return native_psnr, float("nan"), mean(timings_ms) / 1000.0, 1000.0 / mean(timings_ms)

    trainer.test = instrumented_test
    native_psnr, _native_ms_ssim, fit_seconds, _eval_seconds, _native_fps = trainer.train()
    if final_render is None:
        instrumented_test()
    assert final_render is not None

    reconstruction = final_render.squeeze(0).permute(1, 2, 0).numpy().astype(np.float32)
    reconstruction_npy = outdir / "reconstruction.npy"
    reconstruction_png = outdir / "reconstruction.png"
    np.save(reconstruction_npy, reconstruction)
    display = np.clip(reconstruction, 0.0, 1.0)
    Image.fromarray(np.rint(display * 255.0).astype(np.uint8), mode="RGB").save(
        reconstruction_png
    )

    checkpoint_path = trainer.log_dir / "gaussian_model.pth.tar"
    parameter_count = sum(parameter.numel() for parameter in trainer.gaussian_model.parameters())
    height, width = reconstruction.shape[:2]
    selected_history_index = int(np.argmax(np.asarray(history["psnr"], dtype=np.float64)))
    selected_iter = int(history["iter"][selected_history_index]) + 1
    result = {
        "schema_version": SCHEMA_VERSION,
        "adapter_revision": ADAPTER_REVISION,
        "method": "gaussianimage_plus_native",
        "method_label": "GaussianImage++ native",
        "implementation_kind": "native_external",
        "repo_root": str(repo),
        "repo_commit": _git_commit(repo),
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
        "gsplat_module": str(Path(gsplat.__file__).resolve()),
        "gsplat_csrc": str(csrc_path),
        "gsplat_csrc_sha256": _sha256(csrc_path),
        "image": image_path.stem,
        "source_path": str(image_path),
        "height": height,
        "width": width,
        "seed": cli.seed,
        "requested_iters": cli.iterations,
        "iterations_run": len(history["iter"]),
        "stopped_early": len(history["iter"]) < cli.iterations,
        "budget_cap": cli.max_gaussians,
        "start_gaussians": cli.initial_gaussians,
        "n_gaussians": int(trainer.gaussian_model.cur_num_points),
        "growth_interval": int(args.grow_iter),
        "growth_waves_requested": int(cli.growth_waves),
        "growth_policy": "upstream residual top-k add (native batch size/cap)",
        "renderer": "GaussianImage++ bundled gsplat covariance/sum renderer",
        "optimizer": str(args.opt_type),
        "loss": str(args.loss_type),
        "native_reported_psnr": float(native_psnr),
        "native_reported_ms_ssim": None,
        "selection_policy": "best_training_psnr_checkpoint",
        "selected_iter": selected_iter,
        "selected_training_psnr": float(history["psnr"][selected_history_index]),
        "trajectory_final_training_psnr": float(history["psnr"][-1]),
        "selected_n_gaussians": int(history["n_gaussians"][selected_history_index]),
        "init_seconds": init_seconds,
        "fit_seconds": float(fit_seconds),
        "total_seconds": init_seconds + float(fit_seconds),
        **render_stats,
        "parameter_count": int(parameter_count),
        "parameter_bpp_float32": float(parameter_count * 32 / (height * width)),
        "actual_codec_bytes": None,
        "actual_bpp": None,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "reconstruction_npy": str(reconstruction_npy),
        "reconstruction_png": str(reconstruction_png),
        "history": history,
        "upstream_args": _json_safe(vars(args)),
    }
    result_path = outdir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
