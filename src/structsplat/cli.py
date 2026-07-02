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
    from .config import InitConfig, FitConfig, PyramidConfig, StructureTensorConfig
    from . import init as _init
    from .fit import fit
    from .pyramid import fit_pyramid

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image(args.image)
    target = torch.as_tensor(img, device=device)

    icfg = InitConfig(strategy=args.strategy, num_gaussians=args.num_gaussians, seed=args.seed,
                      density_mode=args.density_mode,
                      sampling_mode=args.sampling_mode,
                      orientation_mode=args.orientation_mode,
                      flank_offset_frac=args.flank_offset,
                      max_axis_ratio=args.max_axis_ratio,
                      coherence_power=args.coherence_power,
                      scale_mode=args.scale_mode,
                      init_scale_mult=args.init_scale_mult,
                      color_mode=args.color_mode,
                      color_radius=args.color_radius,
                      opacity_mode=args.opacity_mode,
                      init_opacity=args.init_opacity)
    scfg = StructureTensorConfig(grad_sigma=args.grad_sigma,
                                 tensor_sigma=args.tensor_sigma,
                                 gradient_operator=args.tensor_operator,
                                 color_space=args.tensor_color,
                                 flat_frac=args.flat_frac,
                                 corner_frac=args.corner_frac)
    fcfg = FitConfig(iters=args.iters, target_psnr=args.target_psnr, render_chunk=args.chunk,
                     optimizer=args.optimizer,
                     pixel_loss=args.pixel_loss, ssim_weight=args.ssim_weight,
                     loss_warmup_iters=args.loss_warmup_iters,
                     loss_warmup_pixel_loss=args.loss_warmup_pixel_loss,
                     compute_lpips=args.lpips,
                     renderer=args.renderer,
                     lr_schedule=args.lr_schedule,
                     lr_decay_every=args.lr_decay_every, lr_decay_gamma=args.lr_decay_gamma,
                     prune_every=args.prune_every, prune_min_activity=args.prune_min_activity,
                     prune_keep_min=args.prune_keep_min,
                     split_every=args.split_every, split_count=args.split_count,
                     split_mode=args.split_mode,
                     split_scale=args.split_scale, max_gaussians=args.max_gaussians)

    if args.pyramid:
        # honor --iters when --iters-per-level is not given, so the pyramid spends the
        # same total optimization budget the user asked for
        per_level = args.iters_per_level or max(1, args.iters // max(1, args.pyramid_levels))
        pcfg = PyramidConfig(levels=args.pyramid_levels,
                             level_fractions=args.level_fractions,
                             iters_per_level=per_level)
        out = fit_pyramid(img, target, icfg, fcfg, pcfg, scfg)
    else:
        field = _init.build_field(img, icfg, scfg, device=device)
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


def cmd_stage_search(args):
    from benchmarks.stage_search import run_stage_search
    run_stage_search(
        args.images, budgets=args.budgets, seeds=args.seeds, iters=args.iters, mode=args.mode,
        max_side=args.max_side, strategies=args.strategies,
        tensor_operators=args.tensor_operators, tensor_colors=args.tensor_colors,
        density_modes=args.density_modes,
        sampling_modes=args.sampling_modes, orientation_modes=args.orientation_modes,
        color_modes=args.color_modes,
        scale_modes=args.scale_modes, opacity_modes=args.opacity_modes, renderers=args.renderers,
        pixel_losses=args.pixel_losses, optimizers=args.optimizers,
        lr_schedules=args.lr_schedules, refine_modes=args.refine_modes,
        pyramid_modes=args.pyramid_modes, render_chunk=args.chunk,
        ssim_weight=args.ssim_weight, split_every=args.split_every,
        split_count=args.split_count, prune_every=args.prune_every,
        prune_min_activity=args.prune_min_activity, max_gaussians=args.max_gaussians,
        pyramid_levels=args.pyramid_levels, pyramid_fractions=args.pyramid_fractions,
        pyramid_iters_per_level=args.pyramid_iters_per_level, compute_lpips=args.lpips,
        target_psnr=args.target_psnr, target_psnrs=args.target_psnrs,
        log_every=args.log_every, dedupe=not args.no_dedupe,
        max_configs=args.max_configs, shuffle_configs=args.shuffle_configs,
        config_seed=args.config_seed, outdir=args.outdir, device=args.device, verbose=args.verbose,
    )


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
    f.add_argument("--iters-per-level", type=int, default=None,
                   help="default: --iters / --pyramid-levels")
    f.add_argument("--target-psnr", type=float, default=None, dest="target_psnr")
    f.add_argument("--chunk", type=int, default=512)
    f.add_argument("--tensor-operator", choices=["central", "sobel", "scharr"], default="central")
    f.add_argument("--tensor-color", choices=["luma", "rgb"], default="luma")
    f.add_argument("--grad-sigma", type=float, default=1.0)
    f.add_argument("--tensor-sigma", type=float, default=2.0)
    f.add_argument("--flat-frac", type=float, default=0.02)
    f.add_argument("--corner-frac", type=float, default=0.15)
    f.add_argument("--density-mode", choices=["structure", "gradient", "variance", "hybrid", "uniform"],
                   default="structure")
    f.add_argument("--sampling-mode",
                   choices=["wse", "density_random", "jittered_grid", "dart_throwing", "halton",
                            "farthest_point", "cvt"],
                   default="wse")
    f.add_argument("--orientation-mode", choices=["tensor", "random", "zero"], default="tensor")
    f.add_argument("--color-mode", choices=["bilinear", "local_mean", "two_sided"], default="bilinear")
    f.add_argument("--color-radius", type=float, default=1.5)
    f.add_argument("--scale-mode", choices=["spacing", "uniform", "knn"], default="spacing")
    f.add_argument("--init-scale-mult", type=float, default=1.0)
    f.add_argument("--opacity-mode", choices=["none", "constant"], default="none")
    f.add_argument("--init-opacity", type=float, default=0.9)
    f.add_argument("--renderer", choices=["normalized", "additive"], default="normalized")
    f.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    f.add_argument("--pixel-loss", choices=["l1", "l2", "charbonnier"], default="l1")
    f.add_argument("--loss-warmup-iters", type=int, default=0)
    f.add_argument("--loss-warmup-pixel-loss", choices=["l1", "l2", "charbonnier"], default="l2")
    f.add_argument("--ssim-weight", type=float, default=0.3)
    f.add_argument("--lpips", action="store_true", help="compute LPIPS after fitting")
    f.add_argument("--flank-offset", type=float, default=0.5)
    f.add_argument("--max-axis-ratio", type=float, default=6.0)
    f.add_argument("--coherence-power", type=float, default=1.0)
    f.add_argument("--lr-schedule", choices=["none", "step", "cosine"], default="none")
    f.add_argument("--lr-decay-every", type=int, default=None)
    f.add_argument("--lr-decay-gamma", type=float, default=0.5)
    f.add_argument("--prune-every", type=int, default=None)
    f.add_argument("--prune-min-activity", type=float, default=0.0)
    f.add_argument("--prune-keep-min", type=int, default=16)
    f.add_argument("--split-every", type=int, default=None)
    f.add_argument("--split-count", type=int, default=0)
    f.add_argument("--split-mode", choices=["duplicate", "residual_add"], default="duplicate")
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

    s = sub.add_parser("stage-search", help="search complete StructSplat stage combinations")
    s.add_argument("images", nargs="+", help="image files or a directory")
    s.add_argument("--mode", choices=["factorial", "influence"], default="factorial",
                   help="factorial: full product; influence: one-factor-at-a-time deltas "
                        "around the baseline (first value of each stage axis)")
    s.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
    s.add_argument("--seeds", type=int, nargs="+", default=[0])
    s.add_argument("--iters", type=int, default=300)
    s.add_argument("--max-side", type=int, default=320)
    s.add_argument("--strategies", nargs="+", default=None)
    s.add_argument("--tensor-operators", nargs="+", default=None)
    s.add_argument("--tensor-colors", nargs="+", default=None)
    s.add_argument("--density-modes", nargs="+", default=None)
    s.add_argument("--sampling-modes", nargs="+", default=None)
    s.add_argument("--orientation-modes", nargs="+", default=None)
    s.add_argument("--color-modes", nargs="+", default=None)
    s.add_argument("--scale-modes", nargs="+", default=None)
    s.add_argument("--opacity-modes", nargs="+", default=None)
    s.add_argument("--renderers", nargs="+", default=None)
    s.add_argument("--pixel-losses", nargs="+", default=None)
    s.add_argument("--optimizers", nargs="+", default=None)
    s.add_argument("--lr-schedules", nargs="+", default=None)
    s.add_argument("--refine-modes", nargs="+", default=None)
    s.add_argument("--pyramid-modes", nargs="+", default=None)
    s.add_argument("--chunk", type=int, default=512)
    s.add_argument("--ssim-weight", type=float, default=0.3)
    s.add_argument("--target-psnr", type=float, default=None, dest="target_psnr",
                   help="record iters/seconds-to-target (convergence-rate comparisons)")
    s.add_argument("--target-psnrs", type=float, nargs="*", default=[])
    s.add_argument("--log-every", type=int, default=None)
    s.add_argument("--no-dedupe", action="store_true")
    s.add_argument("--split-every", type=int, default=None)
    s.add_argument("--split-count", type=int, default=64)
    s.add_argument("--prune-every", type=int, default=None)
    s.add_argument("--prune-min-activity", type=float, default=0.0)
    s.add_argument("--max-gaussians", type=int, default=None)
    s.add_argument("--pyramid-levels", type=int, default=2)
    s.add_argument("--pyramid-fractions", type=float, nargs="+", default=[0.35, 0.65])
    s.add_argument("--pyramid-iters-per-level", type=int, default=None)
    s.add_argument("--lpips", action="store_true")
    s.add_argument("--max-configs", type=int, default=None)
    s.add_argument("--shuffle-configs", action="store_true")
    s.add_argument("--config-seed", type=int, default=0)
    s.add_argument("--outdir", default="results/stage_search")
    s.add_argument("--device", default=None)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_stage_search)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
