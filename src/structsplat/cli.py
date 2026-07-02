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

    icfg = InitConfig(strategy=args.strategy, num_gaussians=args.num_gaussians, seed=args.seed,
                      flank_offset_frac=args.flank_offset,
                      max_axis_ratio=args.max_axis_ratio,
                      coherence_power=args.coherence_power)
    fcfg = FitConfig(iters=args.iters, target_psnr=args.target_psnr, render_chunk=args.chunk,
                     pixel_loss=args.pixel_loss, ssim_weight=args.ssim_weight,
                     compute_lpips=args.lpips,
                     lr_decay_every=args.lr_decay_every, lr_decay_gamma=args.lr_decay_gamma,
                     prune_every=args.prune_every, prune_min_activity=args.prune_min_activity,
                     prune_keep_min=args.prune_keep_min,
                     split_every=args.split_every, split_count=args.split_count,
                     split_scale=args.split_scale, max_gaussians=args.max_gaussians)

    if args.pyramid:
        pcfg = PyramidConfig(levels=args.pyramid_levels,
                             level_fractions=args.level_fractions,
                             iters_per_level=args.iters_per_level)
        out = fit_pyramid(img, target, icfg, fcfg, pcfg)
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
    run_ablation(args.images, budgets=args.budgets, strategies=args.strategies,
                 seeds=args.seeds, iters=args.iters, target_psnr=args.target_psnr,
                 target_psnrs=args.target_psnrs, flank_offsets=args.flank_offsets,
                 flat_fracs=args.flat_fracs, corner_fracs=args.corner_fracs,
                 max_axis_ratios=args.max_axis_ratios,
                 coherence_powers=args.coherence_powers, render_chunk=args.chunk,
                 pixel_loss=args.pixel_loss, ssim_weight=args.ssim_weight,
                 compute_lpips=args.lpips,
                 outdir=args.outdir, device=args.device, write_plots=not args.no_plots)


def main():
    p = argparse.ArgumentParser(prog="structsplat")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a single image")
    f.add_argument("image")
    f.add_argument("--strategy", default="aniso_flanking")
    f.add_argument("--num-gaussians", type=int, default=20000, dest="num_gaussians")
    f.add_argument("--iters", type=int, default=2000)
    f.add_argument("--pyramid", action="store_true")
    f.add_argument("--pyramid-levels", type=int, default=4)
    f.add_argument("--level-fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    f.add_argument("--iters-per-level", type=int, default=500)
    f.add_argument("--target-psnr", type=float, default=None, dest="target_psnr")
    f.add_argument("--chunk", type=int, default=512)
    f.add_argument("--pixel-loss", choices=["l1", "l2"], default="l1")
    f.add_argument("--ssim-weight", type=float, default=0.3)
    f.add_argument("--lpips", action="store_true", help="compute LPIPS after fitting")
    f.add_argument("--flank-offset", type=float, default=0.5)
    f.add_argument("--max-axis-ratio", type=float, default=6.0)
    f.add_argument("--coherence-power", type=float, default=1.0)
    f.add_argument("--lr-decay-every", type=int, default=None)
    f.add_argument("--lr-decay-gamma", type=float, default=0.5)
    f.add_argument("--prune-every", type=int, default=None)
    f.add_argument("--prune-min-activity", type=float, default=0.0)
    f.add_argument("--prune-keep-min", type=int, default=16)
    f.add_argument("--split-every", type=int, default=None)
    f.add_argument("--split-count", type=int, default=0)
    f.add_argument("--split-scale", type=float, default=0.7)
    f.add_argument("--max-gaussians", type=int, default=None)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--outdir", default="runs")
    f.add_argument("--device", default=None)
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("ablation", help="run the init-strategy sweep (ABL-001)")
    a.add_argument("images", nargs="+", help="image files or a directory")
    a.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000, 10000, 20000])
    a.add_argument("--strategies", nargs="*", default=None)
    a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a.add_argument("--iters", type=int, default=1500)
    a.add_argument("--target-psnr", type=float, default=35.0, dest="target_psnr")
    a.add_argument("--target-psnrs", type=float, nargs="*", default=[])
    a.add_argument("--flank-offsets", type=float, nargs="+", default=[0.5])
    a.add_argument("--flat-fracs", type=float, nargs="+", default=[0.02])
    a.add_argument("--corner-fracs", type=float, nargs="+", default=[0.15])
    a.add_argument("--max-axis-ratios", type=float, nargs="+", default=[6.0])
    a.add_argument("--coherence-powers", type=float, nargs="+", default=[1.0])
    a.add_argument("--chunk", type=int, default=512)
    a.add_argument("--pixel-loss", choices=["l1", "l2"], default="l1")
    a.add_argument("--ssim-weight", type=float, default=0.3)
    a.add_argument("--lpips", action="store_true", help="compute LPIPS for each run")
    a.add_argument("--no-plots", action="store_true")
    a.add_argument("--outdir", default="results")
    a.add_argument("--device", default=None)
    a.set_defaults(func=cmd_ablation)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
