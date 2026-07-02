"""Command line entry: `structsplat fit ...` and `structsplat ablation ...`.

Torch is imported lazily inside the commands so `--help` and the NumPy modules work without it.
"""
from __future__ import annotations
import argparse
import os
import numpy as np


def load_image(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_image(path: str, arr: np.ndarray):
    from PIL import Image
    a = np.clip(arr, 0, 1) * 255.0
    Image.fromarray(a.astype(np.uint8)).save(path)


def cmd_fit(args):
    import torch
    from .config import InitConfig, FitConfig, PyramidConfig
    from . import init as _init
    from .fit import fit
    from .pyramid import fit_pyramid

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image(args.image)
    target = torch.as_tensor(img, device=device)

    icfg = InitConfig(strategy=args.strategy, num_gaussians=args.num_gaussians, seed=args.seed)
    fcfg = FitConfig(iters=args.iters, target_psnr=args.target_psnr, render_chunk=args.chunk)

    if args.pyramid:
        out = fit_pyramid(img, target, icfg, fcfg, PyramidConfig())
    else:
        field = _init.build_field(img, icfg, device=device)
        out = fit(field, target, fcfg)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    save_image(os.path.join(args.outdir, f"{base}_{args.strategy}.png"),
               out["render"].detach().cpu().numpy())
    out["field"].save(os.path.join(args.outdir, f"{base}_{args.strategy}.npz"))
    print(f"\n{base}: {out['n_gaussians']} gaussians | PSNR {out['psnr']:.2f} | "
          f"SSIM {out['ssim']:.4f} | MS-SSIM {out['ms_ssim']:.4f} | "
          f"iters_to_target {out['iters_to_target']}")


def cmd_ablation(args):
    from benchmarks.ablation import run_ablation
    run_ablation(args.images, budgets=args.budgets, iters=args.iters,
                 target_psnr=args.target_psnr, outdir=args.outdir, device=args.device)


def main():
    p = argparse.ArgumentParser(prog="structsplat")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a single image")
    f.add_argument("image")
    f.add_argument("--strategy", default="aniso_flanking")
    f.add_argument("--num-gaussians", type=int, default=20000, dest="num_gaussians")
    f.add_argument("--iters", type=int, default=2000)
    f.add_argument("--pyramid", action="store_true")
    f.add_argument("--target-psnr", type=float, default=None, dest="target_psnr")
    f.add_argument("--chunk", type=int, default=4096)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--outdir", default="runs")
    f.add_argument("--device", default=None)
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("ablation", help="run the init-strategy sweep (ABL-001)")
    a.add_argument("images", nargs="+", help="image files or a directory")
    a.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000, 10000, 20000])
    a.add_argument("--iters", type=int, default=1500)
    a.add_argument("--target-psnr", type=float, default=35.0, dest="target_psnr")
    a.add_argument("--outdir", default="results")
    a.add_argument("--device", default=None)
    a.set_defaults(func=cmd_ablation)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
