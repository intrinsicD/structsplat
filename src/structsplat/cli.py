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
    # round, don't truncate: astype(uint8) floors, biasing every saved pixel down by up to 1/255
    a = np.rint(np.clip(arr, 0, 1) * 255.0)
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
                      scale_cap_mode=args.scale_cap_mode,
                      scale_cap_max=args.scale_cap_max,
                      background_fraction=args.background_fraction,
                      background_grid=args.background_grid,
                      color_mode=args.color_mode,
                      color_radius=args.color_radius,
                      opacity_mode=args.opacity_mode,
                      init_opacity=args.init_opacity,
                      predictor_checkpoint=args.predictor_checkpoint,
                      predictor_fallback_strategy=args.predictor_fallback_strategy)
    scfg = StructureTensorConfig(grad_sigma=args.grad_sigma,
                                 tensor_sigma=args.tensor_sigma,
                                 gradient_operator=args.tensor_operator,
                                 color_space=args.tensor_color,
                                 flat_frac=args.flat_frac,
                                 corner_frac=args.corner_frac)
    fcfg = FitConfig(iters=args.iters, target_psnr=args.target_psnr,
                     target_ms_ssim=args.target_ms_ssim, target_bpp=args.target_bpp,
                     render_chunk=args.chunk,
                     optimizer=args.optimizer,
                     pixel_loss=args.pixel_loss, ssim_weight=args.ssim_weight,
                     loss_weighting=args.loss_weighting,
                     loss_weight_beta=args.loss_weight_beta,
                     ssim_backend=args.ssim_backend,
                     loss_warmup_iters=args.loss_warmup_iters,
                     loss_warmup_pixel_loss=args.loss_warmup_pixel_loss,
                     compute_lpips=args.lpips,
                     renderer=args.renderer,
                     color_basis=args.color_basis,
                     color_grad_l2=args.color_grad_l2,
                     color_solve_every=args.color_solve_every,
                     color_solve_schedule=args.color_solve_schedule,
                     color_solve_lambda=args.color_solve_lambda,
                     color_solve_maxiter=args.color_solve_maxiter,
                     qat_mode=args.qat_mode,
                     lambda_rate=args.lambda_rate,
                     qat_bits_means=args.qat_bits_means,
                     qat_bits_scales=args.qat_bits_scales,
                     qat_bits_rot=args.qat_bits_rot,
                     qat_bits_colors=args.qat_bits_colors,
                     qat_bits_opacity=args.qat_bits_opacity,
                     support_fade=args.support_fade,
                     support_fade_until_frac=args.support_fade_until_frac,
                     support_fade_crossfade_iters=args.support_fade_crossfade_iters,
                     aa_dilation=args.aa_dilation,
                     lr_schedule=args.lr_schedule,
                     lr_decay_every=args.lr_decay_every, lr_decay_gamma=args.lr_decay_gamma,
                     prune_every=args.prune_every, prune_min_activity=args.prune_min_activity,
                     prune_keep_min=args.prune_keep_min,
                     split_every=args.split_every, split_count=args.split_count,
                     split_mode=args.split_mode,
                     split_scale=args.split_scale,
                     split_oversample=args.split_oversample,
                     split_min_spacing=args.split_min_spacing,
                     split_color_init=args.split_color_init,
                     seed_new_row_optimizer_state=args.seed_new_row_optimizer_state,
                     new_row_temper_iters=args.new_row_temper_iters,
                     new_row_temper_start=args.new_row_temper_start,
                     absgrad_decay=args.absgrad_decay,
                     relocate_every=args.relocate_every,
                     relocate_at_split=args.relocate_at_split,
                     relocate_count=args.relocate_count,
                     relocate_init_opacity=args.relocate_init_opacity,
                     relocate_residual_downsample=args.relocate_residual_downsample,
                     max_gaussians=args.max_gaussians,
                     adaptive_count=args.adaptive_count,
                     adaptive_growth_every=args.adaptive_growth_every,
                     adaptive_growth_count=args.adaptive_growth_count,
                     adaptive_split_mode=args.adaptive_split_mode,
                     adaptive_min_delta_psnr=args.adaptive_min_delta_psnr,
                     adaptive_patience=args.adaptive_patience)

    if args.pyramid:
        # honor --iters when --iters-per-level is not given, so the pyramid spends the
        # same total optimization budget the user asked for
        per_level = args.iters_per_level or max(1, args.iters // max(1, args.pyramid_levels))
        pcfg = PyramidConfig(levels=args.pyramid_levels,
                             level_fractions=args.level_fractions,
                             iters_per_level=per_level,
                             level_iters=args.level_iters)
        out = fit_pyramid(img, target, icfg, fcfg, pcfg, scfg)
    else:
        field = _init.build_field(img, icfg, scfg, device=device)
        out = fit(field, target, fcfg)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    save_image(os.path.join(args.outdir, f"{base}_{args.strategy}.png"),
               out["render"].detach().cpu().numpy())
    out["field"].save(os.path.join(args.outdir, f"{base}_{args.strategy}.npz"))
    line = (f"\n{base}: {out['n_gaussians']} gaussians | PSNR {out['psnr']:.2f} | "
            f"SSIM {out['ssim']:.4f} | MS-SSIM {out['ms_ssim']:.4f}")
    if out.get("lpips") is not None:  # --lpips loaded AlexNet but the value was never surfaced
        line += f" | LPIPS {out['lpips']:.4f}"
    line += f" | iters_to_target {out['iters_to_target']}"
    print(line)


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
        scale_modes=args.scale_modes, scale_cap_modes=args.scale_cap_modes,
        background_modes=args.background_modes,
        opacity_modes=args.opacity_modes, renderers=args.renderers,
        aa_dilations=args.aa_dilations,
        color_basis_modes=args.color_basis_modes,
        color_solve_modes=args.color_solve_modes,
        pixel_losses=args.pixel_losses, loss_weight_modes=args.loss_weight_modes,
        optimizers=args.optimizers,
        lr_schedules=args.lr_schedules, refine_modes=args.refine_modes,
        state_seed_modes=args.state_seed_modes,
        row_temper_modes=args.row_temper_modes,
        support_fade_modes=args.support_fade_modes,
        pyramid_modes=args.pyramid_modes, render_chunk=args.chunk,
        ssim_weight=args.ssim_weight, ssim_backend=args.ssim_backend,
        split_every=args.split_every,
        split_count=args.split_count, prune_every=args.prune_every,
        prune_min_activity=args.prune_min_activity, max_gaussians=args.max_gaussians,
        pyramid_levels=args.pyramid_levels, pyramid_fractions=args.pyramid_fractions,
        pyramid_iters_per_level=args.pyramid_iters_per_level,
        pyramid_level_iters=args.pyramid_level_iters,
        compute_lpips=args.lpips,
        target_psnr=args.target_psnr, target_psnrs=args.target_psnrs,
        target_ms_ssim=args.target_ms_ssim, target_bpp=args.target_bpp,
        adaptive_count=args.adaptive_count,
        adaptive_growth_every=args.adaptive_growth_every,
        adaptive_growth_count=args.adaptive_growth_count,
        adaptive_split_mode=args.adaptive_split_mode,
        adaptive_min_delta_psnr=args.adaptive_min_delta_psnr,
        adaptive_patience=args.adaptive_patience,
        log_every=args.log_every, dedupe=not args.no_dedupe,
        max_configs=args.max_configs, shuffle_configs=args.shuffle_configs,
        config_seed=args.config_seed, outdir=args.outdir, device=args.device, verbose=args.verbose,
    )


def cmd_generate(args):
    from .config import GenConfig
    from .generate import generate

    cfg = GenConfig(
        model_id=args.model_id,
        height=args.height,
        width=args.width,
        num_gaussians=args.num_gaussians,
        sample_steps=args.sample_steps,
        fit_iters=args.fit_iters,
        refine_steps=args.steps,
        sample_guidance_scale=args.sample_guidance_scale,
        sds_guidance_scale=args.sds_guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        init_strategy=args.strategy,
        init_opacity=args.init_opacity,
        renderer=args.renderer,
        render_chunk=args.chunk,
        sigma_cutoff=args.sigma_cutoff,
        support_fade=args.support_fade,
        aa_dilation=args.aa_dilation,
        lr_means=args.lr_means,
        lr_scales=args.lr_scales,
        lr_rot=args.lr_rot,
        lr_color=args.lr_color,
        lr_opacity=args.lr_opacity,
        sds_t_min=args.sds_t_min,
        sds_t_max=args.sds_t_max,
        grad_clip_norm=args.grad_clip_norm,
        save_resolutions=args.resolutions,
    )
    sample = load_image(args.sample_image) if args.sample_image else None
    out = generate(
        args.prompt,
        cfg,
        sample_image=sample,
        outdir=args.outdir,
        verbose=not args.quiet,
    )
    paths = out.get("paths", {})
    field_path = paths.get("field.npz", os.path.join(args.outdir, "field.npz"))
    print(
        f"\n{out['field'].n} gaussians | prompt {args.prompt!r} | "
        f"saved {field_path}"
    )


def main():
    from .config import GenConfig, DEFAULT_INIT_STRATEGY, DEFAULT_PREDICTOR_FALLBACK_STRATEGY

    p = argparse.ArgumentParser(prog="structsplat")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a single image")
    f.add_argument("image")
    f.add_argument("--strategy", default=DEFAULT_INIT_STRATEGY)
    f.add_argument("--num-gaussians", type=int, default=20000, dest="num_gaussians")
    f.add_argument("--predictor-checkpoint", default=None,
                   help="saved GaussianField or learned .pt checkpoint used by strategy=feedforward")
    f.add_argument("--predictor-fallback-strategy", default=DEFAULT_PREDICTOR_FALLBACK_STRATEGY,
                   help="tensor-prior strategy used by strategy=feedforward when no saved "
                        "checkpoint is provided or padding is needed")
    f.add_argument("--iters", type=int, default=2000)
    f.add_argument("--pyramid", action="store_true")
    f.add_argument("--pyramid-levels", type=int, default=4)
    f.add_argument("--level-fractions", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    f.add_argument("--iters-per-level", type=int, default=None,
                   help="default: --iters / --pyramid-levels")
    f.add_argument("--level-iters", type=int, nargs="+", default=None,
                   help="explicit coarse-to-fine pyramid iteration counts")
    f.add_argument("--target-psnr", type=float, default=None, dest="target_psnr")
    f.add_argument("--target-ms-ssim", type=float, default=None, dest="target_ms_ssim",
                   help="adaptive-count stop target for MS-SSIM")
    f.add_argument("--target-bpp", type=float, default=None, dest="target_bpp",
                   help="adaptive-count raw-attribute bpp cap/target")
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
    f.add_argument("--color-mode", choices=["bilinear", "local_mean", "two_sided", "aggregate"],
                   default="bilinear")
    f.add_argument("--color-radius", type=float, default=1.5)
    f.add_argument("--scale-mode", choices=["spacing", "uniform", "knn"], default="spacing")
    f.add_argument("--init-scale-mult", type=float, default=1.0)
    f.add_argument("--scale-cap-mode", choices=["none", "hard", "feature", "feature_rel"],
                   default="none")
    f.add_argument("--scale-cap-max", type=float, default=None)
    f.add_argument("--background-fraction", type=float, default=0.0,
                   help="fraction of total Gaussian budget reserved for frozen broad bg rows")
    f.add_argument("--background-grid", type=int, default=0,
                   help="max side of jittered-grid background layer; 0 disables it")
    f.add_argument("--opacity-mode", choices=["none", "constant"], default="none")
    f.add_argument("--init-opacity", type=float, default=0.9)
    f.add_argument("--renderer",
                   choices=[
                       "normalized", "additive", "cuda", "cuda_additive",
                       "cuda_tiled", "cuda_tiled_additive", "gsplat",
                   ],
                   default="normalized")
    f.add_argument("--color-solve-every", type=int, default=None,
                   help="periodically solve fixed-geometry RGB colors with CG; normalized renderer only")
    f.add_argument("--color-solve-schedule", default="every",
                   help="none/every/init/final/on_split, composable with +")
    f.add_argument("--color-basis", choices=["constant", "affine"], default="constant",
                   help="per-Gaussian color model")
    f.add_argument("--color-grad-l2", type=float, default=1e-4,
                   help="L2 regularization for affine color coefficients")
    f.add_argument("--color-solve-lambda", type=float, default=1e-4,
                   help="Tikhonov pull toward pre-solve colors for --color-solve-every")
    f.add_argument("--color-solve-maxiter", type=int, default=32,
                   help="maximum CG iterations for each color solve")
    f.add_argument("--qat-mode", choices=["off", "ste", "noise"], default="off",
                   help="fit-time quantization-aware render mode")
    f.add_argument("--lambda-rate", type=float, default=0.0,
                   help="weight for differentiable fit-time rate proxy")
    f.add_argument("--qat-bits-means", type=int, default=16)
    f.add_argument("--qat-bits-scales", type=int, default=8)
    f.add_argument("--qat-bits-rot", type=int, default=8)
    f.add_argument("--qat-bits-colors", type=int, default=8)
    f.add_argument("--qat-bits-opacity", type=int, default=8)
    f.add_argument("--aa-dilation", type=float, default=0.0)
    f.add_argument("--support-fade", action="store_true",
                   help="subtract the Gaussian tail at sigma_cutoff for C0 compact support")
    f.add_argument("--support-fade-until-frac", type=float, default=None,
                   help="enable support fade for the first fraction of fit iterations")
    f.add_argument("--support-fade-crossfade-iters", type=int, default=10,
                   help="iterations used to ramp scheduled support fade off")
    f.add_argument("--optimizer", choices=["adam", "adamw", "adan"], default="adam")
    f.add_argument("--pixel-loss", choices=["l1", "l2", "charbonnier"], default="l1")
    f.add_argument("--loss-weighting", choices=["none", "tensor"], default="none")
    f.add_argument("--loss-weight-beta", type=float, default=1.0)
    f.add_argument("--loss-warmup-iters", type=int, default=0)
    f.add_argument("--loss-warmup-pixel-loss", choices=["l1", "l2", "charbonnier"], default="l2")
    f.add_argument("--ssim-weight", type=float, default=0.3)
    f.add_argument("--ssim-backend", choices=["builtin", "fused", "auto"], default="builtin")
    f.add_argument("--lpips", action="store_true", help="compute LPIPS after fitting")
    f.add_argument("--flank-offset", type=float, default=None,
                   help="edge-center offset fraction; default is strategy-aware")
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
    f.add_argument("--split-mode",
                   choices=[
                       "duplicate", "fp_duplicate", "moment_preserving", "support_duplicate",
                       "residual_add", "residual_tensor_add", "ranked_wave", "absgrad_wave",
                       "freq_violation",
                   ],
                   default="duplicate")
    f.add_argument("--split-scale", type=float, default=0.7)
    f.add_argument("--split-oversample", type=float, default=1.0,
                   help="candidate multiplier for residual-add spacing suppression")
    f.add_argument("--split-min-spacing", type=float, default=0.0,
                   help="residual-add spacing radius as a multiple of the densify base scale")
    f.add_argument("--split-color-init", choices=["target", "residual"], default="target",
                   help="color initialization for normalized residual-add children")
    f.add_argument("--seed-new-row-optimizer-state", action="store_true",
                   help="seed new-row optimizer moments from split parents or carried-row median")
    f.add_argument("--new-row-temper-iters", type=int, default=0,
                   help="post-insert update-ramp length for new/relocated rows")
    f.add_argument("--new-row-temper-start", type=float, default=0.25,
                   help="first-step update multiplier for --new-row-temper-iters")
    f.add_argument("--absgrad-decay", type=float, default=1.0)
    f.add_argument("--relocate-every", type=int, default=None)
    f.add_argument("--relocate-at-split", action="store_true",
                   help="relocate low-activity Gaussians on split/growth iterations")
    f.add_argument("--relocate-count", type=int, default=0)
    f.add_argument("--relocate-init-opacity", type=float, default=0.05)
    f.add_argument("--relocate-residual-downsample", type=int, default=1,
                   help="max-pool residual by this factor before relocation candidate search")
    f.add_argument("--max-gaussians", type=int, default=None)
    f.add_argument("--adaptive-count", action="store_true",
                   help="grow until target/max-N/stall instead of using a fixed final count")
    f.add_argument("--adaptive-growth-every", type=int, default=50)
    f.add_argument("--adaptive-growth-count", type=int, default=64)
    f.add_argument("--adaptive-split-mode",
                   choices=[
                       "residual_add", "residual_tensor_add", "ranked_wave",
                       "freq_violation", "fp_duplicate", "moment_preserving",
                       "support_duplicate",
                   ],
                   default="residual_tensor_add")
    f.add_argument("--adaptive-min-delta-psnr", type=float, default=0.02)
    f.add_argument("--adaptive-patience", type=int, default=2)
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
    s.add_argument("--scale-cap-modes", nargs="+", default=None)
    s.add_argument("--background-modes", nargs="+", default=None,
                   help="off or frac<F>_grid<N>, e.g. frac0.10_grid16")
    s.add_argument("--opacity-modes", nargs="+", default=None)
    s.add_argument("--renderers", nargs="+", default=None)
    s.add_argument("--aa-dilations", type=float, nargs="+", default=None)
    s.add_argument("--color-basis-modes", nargs="+", default=None)
    s.add_argument("--color-solve-modes", nargs="+", default=None)
    s.add_argument("--pixel-losses", nargs="+", default=None)
    s.add_argument("--loss-weight-modes", nargs="+", default=None)
    s.add_argument("--optimizers", nargs="+", default=None)
    s.add_argument("--lr-schedules", nargs="+", default=None)
    s.add_argument("--refine-modes", nargs="+", default=None)
    s.add_argument("--state-seed-modes", nargs="+", default=None,
                   help="off or on; seed new-row optimizer moments from parent/median")
    s.add_argument("--row-temper-modes", nargs="+", default=None,
                   help="off or warmup<N>; post-insert update ramp")
    s.add_argument("--support-fade-modes", nargs="+", default=None,
                   help="off, on, or until<F>; scheduled compact-support fade")
    s.add_argument("--pyramid-modes", nargs="+", default=None)
    s.add_argument("--chunk", type=int, default=512)
    s.add_argument("--ssim-weight", type=float, default=0.3)
    s.add_argument("--ssim-backend", choices=["builtin", "fused", "auto"], default="builtin")
    s.add_argument("--target-psnr", type=float, default=None, dest="target_psnr",
                   help="record iters/seconds-to-target (convergence-rate comparisons)")
    s.add_argument("--target-psnrs", type=float, nargs="*", default=[])
    s.add_argument("--target-ms-ssim", type=float, default=None, dest="target_ms_ssim")
    s.add_argument("--target-bpp", type=float, default=None, dest="target_bpp")
    s.add_argument("--log-every", type=int, default=None)
    s.add_argument("--no-dedupe", action="store_true")
    s.add_argument("--split-every", type=int, default=None)
    s.add_argument("--split-count", type=int, default=64)
    s.add_argument("--prune-every", type=int, default=None)
    s.add_argument("--prune-min-activity", type=float, default=0.0)
    s.add_argument("--max-gaussians", type=int, default=None)
    s.add_argument("--adaptive-count", action="store_true",
                   help="enable FIT-008 self-adaptive Gaussian count controller")
    s.add_argument("--adaptive-growth-every", type=int, default=50)
    s.add_argument("--adaptive-growth-count", type=int, default=64)
    s.add_argument("--adaptive-split-mode",
                   choices=[
                       "residual_add", "residual_tensor_add", "ranked_wave",
                       "freq_violation", "fp_duplicate", "moment_preserving",
                       "support_duplicate",
                   ],
                   default="residual_tensor_add")
    s.add_argument("--adaptive-min-delta-psnr", type=float, default=0.02)
    s.add_argument("--adaptive-patience", type=int, default=2)
    s.add_argument("--pyramid-levels", type=int, default=2)
    s.add_argument("--pyramid-fractions", type=float, nargs="+", default=[0.35, 0.65])
    s.add_argument("--pyramid-iters-per-level", type=int, default=None)
    s.add_argument("--pyramid-level-iters", type=int, nargs="+", default=None)
    s.add_argument("--lpips", action="store_true")
    s.add_argument("--max-configs", type=int, default=None)
    s.add_argument("--shuffle-configs", action="store_true")
    s.add_argument("--config-seed", type=int, default=0)
    s.add_argument("--outdir", default="results/stage_search")
    s.add_argument("--device", default=None)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_stage_search)

    g = sub.add_parser("generate", help="generate a GaussianField from a text prompt")
    g.add_argument("prompt")
    g.add_argument("--model-id", default=GenConfig.model_id)
    g.add_argument("--n", type=int, default=GenConfig.num_gaussians, dest="num_gaussians")
    g.add_argument("--steps", type=int, default=GenConfig.refine_steps,
                   help="SDS refinement steps")
    g.add_argument("--sample-steps", type=int, default=GenConfig.sample_steps)
    g.add_argument("--fit-iters", type=int, default=GenConfig.fit_iters)
    g.add_argument("--height", type=int, default=GenConfig.height)
    g.add_argument("--width", type=int, default=GenConfig.width)
    g.add_argument("--sample-guidance-scale", type=float, default=GenConfig.sample_guidance_scale)
    g.add_argument("--sds-guidance-scale", type=float, default=GenConfig.sds_guidance_scale)
    g.add_argument("--negative-prompt", default=GenConfig.negative_prompt)
    g.add_argument("--sample-image", default=None,
                   help="optional raster init image; SDS still uses the text prompt")
    g.add_argument("--strategy", default=GenConfig.init_strategy)
    g.add_argument("--init-opacity", type=float, default=GenConfig.init_opacity)
    g.add_argument("--renderer",
                   choices=["additive", "cuda_additive", "cuda_tiled_additive", "gsplat"],
                   default=GenConfig.renderer)
    g.add_argument("--aa-dilation", type=float, default=GenConfig.aa_dilation)
    g.add_argument("--support-fade", action="store_true")
    g.add_argument("--sigma-cutoff", type=float, default=GenConfig.sigma_cutoff)
    g.add_argument("--chunk", type=int, default=GenConfig.render_chunk)
    g.add_argument("--lr-means", type=float, default=GenConfig.lr_means)
    g.add_argument("--lr-scales", type=float, default=GenConfig.lr_scales)
    g.add_argument("--lr-rot", type=float, default=GenConfig.lr_rot)
    g.add_argument("--lr-color", type=float, default=GenConfig.lr_color)
    g.add_argument("--lr-opacity", type=float, default=GenConfig.lr_opacity)
    g.add_argument("--sds-t-min", type=int, default=GenConfig.sds_t_min)
    g.add_argument("--sds-t-max", type=int, default=GenConfig.sds_t_max)
    g.add_argument("--grad-clip-norm", type=float, default=GenConfig.grad_clip_norm)
    g.add_argument("--resolutions", type=int, nargs="*", default=GenConfig().save_resolutions,
                   help="long-side PNG sizes rendered from the final field")
    g.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=GenConfig.dtype)
    g.add_argument("--seed", type=int, default=GenConfig.seed)
    g.add_argument("--device", default=None)
    g.add_argument("--outdir", default="runs/generate")
    g.add_argument("--quiet", action="store_true")
    g.set_defaults(func=cmd_generate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
