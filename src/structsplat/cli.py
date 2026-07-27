"""Command line entry for fitting, rendering, diagnostics, and benchmarks.

Torch is imported lazily inside the commands so `--help` and the NumPy modules work without it.
"""
from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
import sys
import numpy as np


def _benchmark_symbol(module_name: str, symbol: str):
    """Import a repo-level benchmark module from the console-script entry point."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        top_level = module_name.partition(".")[0]
        if exc.name not in (top_level, module_name):
            raise
        repo_root = Path(__file__).resolve().parents[2]
        if (repo_root / top_level).is_dir() and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
            # A failed lookup may have cached an unrelated installed namespace package
            # (notably ``benchmarks``) before the repository root became importable.
            sys.modules.pop(top_level, None)
            importlib.invalidate_caches()
            module = importlib.import_module(module_name)
        else:
            raise ModuleNotFoundError(
                f"{module_name!r} is required for this command. Run from an editable "
                "StructSplat checkout, or install the repository with benchmark modules."
            ) from exc
    return getattr(module, symbol)


def load_image(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    """Load a mask as (H, W) float in [0, 1]: RGBA/LA alpha if present, else grayscale."""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        a = np.asarray(im.convert("RGBA"), dtype=np.float32)[..., 3]
    else:
        a = np.asarray(im.convert("L"), dtype=np.float32)
    return a / 255.0


def save_image(path: str, arr: np.ndarray):
    from PIL import Image
    # round, don't truncate: astype(uint8) floors, biasing every saved pixel down by up to 1/255
    a = np.rint(np.clip(arr, 0, 1) * 255.0)
    Image.fromarray(a.astype(np.uint8)).save(path)


def save_rgba(path: str, rgb: np.ndarray, alpha: np.ndarray):
    """Save an RGBA PNG with a hard alpha channel (for visualizing mask containment)."""
    from PIL import Image
    rgb_u8 = np.rint(np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    a_u8 = np.rint(np.clip(alpha, 0, 1) * 255.0).astype(np.uint8)
    rgba = np.dstack([rgb_u8, a_u8])
    Image.fromarray(rgba, mode="RGBA").save(path)


def save_error_heatmap(path: str, signed_error: np.ndarray, scale: float = 4.0) -> None:
    """Save a fixed-scale heatmap of mean absolute RGB reconstruction error."""
    from PIL import Image
    from .visualize import scalar_heatmap

    if scale <= 0.0:
        raise ValueError("error heatmap scale must be positive")
    error = np.asarray(signed_error, dtype=np.float32)
    if error.ndim != 3 or error.shape[2] != 3:
        raise ValueError(f"signed_error must have shape (H,W,3), got {error.shape}")
    display = np.clip(np.mean(np.abs(error), axis=2) * float(scale), 0.0, 1.0)
    Image.fromarray(scalar_heatmap(display), mode="RGB").save(path)


def build_fit_configs(args):
    """Init / structure-tensor / fit configs from parsed fit-CLI options.

    Shared by `fit` and `batch-fit` (PORT-005) so both encode paths are reproducible from the
    same logged option surface.
    """
    from .config import InitConfig, FitConfig, StructureTensorConfig

    icfg = InitConfig(strategy=args.strategy, num_gaussians=args.num_gaussians, seed=args.seed,
                      density_mode=args.density_mode,
                      sampling_mode=args.sampling_mode,
                      wse_progressive_order=args.wse_progressive_order,
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
                     render_checkpoint=args.render_checkpoint,
                     optimizer=args.optimizer,
                     pixel_loss=args.pixel_loss, ssim_weight=args.ssim_weight,
                     loss_weighting=args.loss_weighting,
                     loss_weight_beta=args.loss_weight_beta,
                     mask_contain=args.mask_contain,
                     mask_margin=args.mask_margin,
                     mask_coverage_weight=args.mask_coverage_weight,
                     mask_cap_mode=args.mask_cap_mode,
                     mask_cap_refresh_every=args.mask_cap_refresh_every,
                     mask_undercoverage_weight=args.mask_undercoverage_weight,
                     mask_interior_undercoverage_weight=(
                         args.mask_interior_undercoverage_weight),
                     mask_undercoverage_band=args.mask_undercoverage_band,
                     mask_undercoverage_tau=args.mask_undercoverage_tau,
                     mask_undercoverage_every=args.mask_undercoverage_every,
                     mask_boundary_add_every=args.mask_boundary_add_every,
                     mask_boundary_add_count=args.mask_boundary_add_count,
                     mask_boundary_add_band=args.mask_boundary_add_band,
                     mask_boundary_add_spacing=args.mask_boundary_add_spacing,
                     geometry_loss_weight=args.geometry_loss_weight,
                     geometry_loss_every=args.geometry_loss_every,
                     ssim_backend=args.ssim_backend,
                     loss_warmup_iters=args.loss_warmup_iters,
                     loss_warmup_pixel_loss=args.loss_warmup_pixel_loss,
                     loss_target_downsample=args.loss_target_downsample,
                     loss_target_full_frac=args.loss_target_full_frac,
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
                     covariance_filter_mode=args.covariance_filter_mode,
                     covariance_filter_alpha=args.covariance_filter_alpha,
                     covariance_filter_max_variance=args.covariance_filter_max_variance,
                     checkpoint_policy=args.checkpoint_policy,
                     lr_schedule=args.lr_schedule,
                     lr_cosine_start_frac=args.lr_cosine_start_frac,
                     lr_decay_every=args.lr_decay_every, lr_decay_gamma=args.lr_decay_gamma,
                     prune_every=args.prune_every, prune_min_activity=args.prune_min_activity,
                     prune_keep_min=args.prune_keep_min,
                     split_every=args.split_every, split_count=args.split_count,
                     split_mode=args.split_mode,
                     refine_site=args.refine_site,
                     refine_primitive=args.refine_primitive,
                     refine_nms=args.refine_nms,
                     split_scale=args.split_scale,
                     split_oversample=args.split_oversample,
                     split_min_spacing=args.split_min_spacing,
                     split_color_init=args.split_color_init,
                     sampled_add_score=args.sampled_add_score,
                     responsibility_mass_alpha=args.responsibility_mass_alpha,
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
                     adaptive_patience=args.adaptive_patience,
                     triage_every=getattr(args, "triage_every", None),
                     triage_spawn_count=getattr(args, "triage_spawn_count", 64),
                     triage_split_count=getattr(args, "triage_split_count", 16),
                     triage_merge_count=getattr(args, "triage_merge_count", 16),
                     triage_park_count=getattr(args, "triage_park_count", 16),
                     target_file_bytes=(
                         None if getattr(args, "target_file_kb", None) is None
                         else int(round(float(args.target_file_kb) * 1000))
                     ),
                     pool_capacity=getattr(args, "pool_capacity", None),
                     coverage_match_weight=getattr(args, "coverage_match_weight", 0.0),
                     coverage_match_target=getattr(args, "coverage_match_target", "tensor"),
                     coverage_match_beta=getattr(args, "coverage_match_beta", 1.0),
                     coverage_match_boundary_boost=getattr(
                         args, "coverage_match_boundary_boost", 2.0),
                     coverage_match_boundary_band=getattr(
                         args, "coverage_match_boundary_band", 3.0),
                     coverage_match_error_alpha=getattr(
                         args, "coverage_match_error_alpha", 0.5),
                     coverage_match_decay_frac=getattr(
                         args, "coverage_match_decay_frac", 0.0))
    return icfg, scfg, fcfg


def cmd_convert(args):
    """Run the current best pipeline (ADR-0025). Masked when --mask is given, else full frame."""
    import json

    import torch

    from .pipeline import PipelineConfig, run_pipeline

    img = load_image(args.image)
    mask_np = load_mask(args.mask) if args.mask else None
    if mask_np is not None and args.mask_invert:
        mask_np = 1.0 - mask_np

    cfg = PipelineConfig(
        capacity=args.capacity,
        initial_gaussians=args.initial_gaussians,
        boundary_gaussians=args.boundary_gaussians,
        coverage_target=args.coverage_target,
        detail_target=args.detail_target,
        seed=args.seed,
        device=args.device,
        renderer=args.renderer,
        step_scale=args.step_scale,
        block_steps=args.block_steps,
        storage_policy=args.storage_policy,
        pareto_safe_checkpoints=not args.no_checkpoints,
        pareto_checkpoint_every=args.checkpoint_every,
        event_color_solve=args.event_color_solve,
        mask_margin=args.mask_margin,
        boundary_band=args.boundary_band,
        coverage_tau=args.coverage_tau,
        hole_regression_budget=args.hole_regression_budget,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    result = run_pipeline(img, mask_np, cfg, verbose=not args.quiet)

    field = result["field"]
    field.save(str(outdir / f"{stem}.npz"))
    H, W = img.shape[:2]
    with torch.no_grad():
        from .fit import _render

        fit_cfg_dict = result["fit_config"]
        from .config import FitConfig

        render_cfg = FitConfig(**{
            k: v for k, v in fit_cfg_dict.items() if k in FitConfig.__dataclass_fields__
        })
        rendered = _render(field, render_cfg, H, W, support_fade_alpha=1.0)
    recon = rendered.detach().cpu().numpy()
    save_image(str(outdir / f"{stem}_reconstruction.png"), recon)

    report = {k: v for k, v in result.items() if k not in ("field", "optimizer_state")}
    (outdir / f"{stem}_pipeline.json").write_text(json.dumps(report, indent=2, default=str))

    metrics = result["metrics"]
    print(f"arm={result['arm']} recipe={result['recipe']['name']}@{result['recipe']['version']}")
    print(f"n={field.n} device={result['device']} seconds={result['seconds']:.1f}")
    for key in ("foreground_psnr_db", "boundary_psnr_db", "interior_hole_fraction",
                "boundary_hole_fraction"):
        if key in metrics:
            print(f"{key}={metrics[key]}")
    print(f"wrote {outdir}/{stem}.npz")


def cmd_fit(args):
    import torch
    from .config import PyramidConfig
    from . import init as _init
    from .fit import fit
    from .pyramid import fit_pyramid

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = load_image(args.image)
    target = torch.as_tensor(img, device=device)
    icfg, scfg, fcfg = build_fit_configs(args)

    mask_np = load_mask(args.mask) if args.mask else None
    if mask_np is not None and args.mask_invert:
        mask_np = 1.0 - mask_np
    uses_mask = mask_np is not None or fcfg.mask_contain or fcfg.mask_coverage_weight > 0.0 \
        or fcfg.mask_undercoverage_weight > 0.0 \
        or fcfg.mask_interior_undercoverage_weight > 0.0 \
        or (fcfg.mask_boundary_add_every is not None and fcfg.mask_boundary_add_count > 0) \
        or fcfg.loss_weighting == "mask"
    if uses_mask and mask_np is None:
        raise SystemExit("--mask is required for --mask-contain / --mask-coverage-weight / "
                         "--mask-undercoverage-weight / "
                         "--mask-interior-undercoverage-weight / "
                         "--mask-boundary-add-every / --loss-weighting mask")
    if uses_mask and args.pyramid:
        raise SystemExit("mask-contained fitting does not support --pyramid yet")
    if fcfg.triage_every is not None and args.pyramid:
        raise SystemExit("pooled triage fitting (--triage-every) does not support --pyramid")
    if uses_mask and fcfg.mask_contain and not fcfg.support_fade:
        print("note: exact zero outside the mask needs --support-fade; without it the render is "
              "contained to the sigma_cutoff ellipse but weight can leak inside its bounding box")

    viewer = None
    if args.live:
        from .viewer import LiveFitViewer

        viewer = LiveFitViewer((img.shape[0], img.shape[1]),
                               host=args.live_host, port=args.live_port,
                               token=args.live_token,
                               name=os.path.splitext(os.path.basename(args.image))[0])
        viewer.start()
        print(f"live viewer: {viewer.url}")
    observer = viewer.observer if viewer is not None else None

    if args.pyramid:
        # honor --iters when --iters-per-level is not given, so the pyramid spends the
        # same total optimization budget the user asked for
        per_level = args.iters_per_level or max(1, args.iters // max(1, args.pyramid_levels))
        pcfg = PyramidConfig(levels=args.pyramid_levels,
                             level_fractions=args.level_fractions,
                             iters_per_level=per_level,
                             level_iters=args.level_iters)
        out = fit_pyramid(img, target, icfg, fcfg, pcfg, scfg,
                          iteration_observer=observer, observer_every=args.live_every)
    elif uses_mask:
        field = _init.build_masked_field(img, mask_np, icfg, scfg, device=device,
                                          sigma_cutoff=fcfg.sigma_cutoff,
                                          mask_margin=fcfg.mask_margin,
                                          contain=fcfg.mask_contain,
                                          cap_mode=fcfg.mask_cap_mode)
        out = fit(field, target, fcfg, mask=mask_np,
                  iteration_observer=observer, observer_every=args.live_every)
    else:
        field = _init.build_field(img, icfg, scfg, device=device)
        out = fit(field, target, fcfg,
                  iteration_observer=observer, observer_every=args.live_every)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]
    render = out["render"].detach().cpu().numpy()
    save_image(os.path.join(args.outdir, f"{base}_{args.strategy}.png"), render)
    if uses_mask:
        from . import mask as _maskmod
        inside = _maskmod.as_bool_mask(mask_np).astype(np.float32)
        save_rgba(os.path.join(args.outdir, f"{base}_{args.strategy}_masked.png"), render, inside)
        outside_energy = float(np.abs(render * (1.0 - inside)[..., None]).mean())
        print(f"  out-of-mask mean |render| = {outside_energy:.3e}")
        # Boundary-band PSNR (<= 2 px inside the mask edge): the metric the whole-image PSNR
        # hides; the CORE-010/011 boundary work is judged here.
        sdf = _maskmod.signed_distance(inside > 0.5)
        band = (sdf > 0.0) & (sdf <= 2.0)
        if band.any():
            band_mse = float(((render - img) ** 2)[band].mean())
            band_psnr = 10.0 * float(np.log10(1.0 / max(band_mse, 1e-12)))
            print(f"  boundary-band (<=2 px) PSNR = {band_psnr:.2f} dB over {int(band.sum())} px")
    out["field"].save(os.path.join(args.outdir, f"{base}_{args.strategy}.npz"))
    if fcfg.triage_every is not None:
        # Budgeted pooled fit: also write the SSPL1 the byte budget was derived against, with
        # the alpha map in-container for masked inputs, and report bytes against the budget.
        from . import codec

        blob = codec.encode(out["field"], img.shape[0], img.shape[1], fcfg=fcfg,
                            alpha=mask_np)
        sspl_path = os.path.join(args.outdir, f"{base}_{args.strategy}.sspl")
        with open(sspl_path, "wb") as fh:
            fh.write(blob)
        pool_info = out.get("pool") or {}
        pool_line = (f"  pooled: live {pool_info.get('live_n')}/{pool_info.get('capacity')} "
                     f"(started {pool_info.get('initial_live')}) | {sspl_path} {len(blob)} B")
        if fcfg.target_file_bytes is not None:
            state = "OK" if len(blob) <= fcfg.target_file_bytes else "OVER BUDGET"
            pool_line += f" / budget {fcfg.target_file_bytes} B [{state}]"
        print(pool_line)
    line = (f"\n{base}: {out['n_gaussians']} gaussians | PSNR {out['psnr']:.2f} | "
            f"SSIM {out['ssim']:.4f} | MS-SSIM {out['ms_ssim']:.4f}")
    if out.get("lpips") is not None:  # --lpips loaded AlexNet but the value was never surfaced
        line += f" | LPIPS {out['lpips']:.4f}"
    line += f" | iters_to_target {out['iters_to_target']}"
    print(line)

    if viewer is not None:
        import time

        viewer.publish(out["field"], iteration=args.iters, psnr=float(out["psnr"]))
        print("live viewer still serving the final field; Ctrl-C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.stop()


def cmd_batch_fit(args):
    from .batch import run_batch

    rows = run_batch(args)
    if any(row.get("error") for row in rows):
        raise SystemExit(1)


def cmd_render(args):
    """Render a native GaussianField NPZ or a self-describing SSPL1 stream."""
    import json
    import torch

    from . import metrics as M
    from .gaussians import GaussianField
    from .render import render_field

    source = Path(args.field)
    if not source.is_file():
        raise SystemExit(f"Gaussian field does not exist: {source}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reference = load_image(args.reference) if args.reference else None
    with source.open("rb") as stream:
        magic = stream.read(5)
    is_sspl1 = magic == b"SSPL1"
    blob = source.read_bytes() if is_sspl1 else b""

    if reference is None and (args.error_out or args.raw_error_out or args.metrics_out):
        raise SystemExit("--error-out, --raw-error-out, and --metrics-out require --reference")
    if args.height is not None and args.height <= 0:
        raise SystemExit("--height must be positive")
    if args.width is not None and args.width <= 0:
        raise SystemExit("--width must be positive")
    if args.chunk is not None and args.chunk <= 0:
        raise SystemExit("--chunk must be positive")
    if args.aa_dilation is not None and args.aa_dilation < 0.0:
        raise SystemExit("--aa-dilation must be non-negative")
    if args.sigma_cutoff is not None and args.sigma_cutoff <= 0.0:
        raise SystemExit("--sigma-cutoff must be positive")
    if args.error_out and args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be positive")
    if args.gaussians_out and args.ellipse_limit <= 0:
        raise SystemExit("--ellipse-limit must be positive")

    if is_sspl1:
        from . import codec

        header = codec.blob_header(blob)
        height, width = int(header["H"]), int(header["W"])
        if args.height is not None and args.height != height:
            raise SystemExit(f"--height {args.height} disagrees with SSPL1 height {height}")
        if args.width is not None and args.width != width:
            raise SystemExit(f"--width {args.width} disagrees with SSPL1 width {width}")
        if any(
            value is not None
            for value in (args.renderer, args.aa_dilation, args.sigma_cutoff, args.chunk)
        ) or args.support_fade:
            raise SystemExit(
                "SSPL1 stores its renderer settings; do not pass renderer/support options"
            )
        field = codec.decode(blob, device=device)
        with torch.no_grad():
            reconstruction_t = codec.decode_and_render(blob, device=device)
        format_name = "SSPL1"
    else:
        if reference is None and (args.height is None or args.width is None):
            raise SystemExit("NPZ rendering requires --reference or both --height and --width")
        height = int(reference.shape[0]) if reference is not None else int(args.height)
        width = int(reference.shape[1]) if reference is not None else int(args.width)
        if args.height is not None and args.height != height:
            raise SystemExit(f"--height {args.height} disagrees with reference height {height}")
        if args.width is not None and args.width != width:
            raise SystemExit(f"--width {args.width} disagrees with reference width {width}")
        field = GaussianField.load(str(source), device=device)
        renderer = args.renderer or "normalized"
        aa_dilation = 0.0 if args.aa_dilation is None else float(args.aa_dilation)
        sigma_cutoff = 3.0 if args.sigma_cutoff is None else float(args.sigma_cutoff)
        chunk = 512 if args.chunk is None else int(args.chunk)
        scales = field.effective_scales(
            aa_dilation if renderer in ("gsplat", "cuda_gsplat") else 0.0
        )
        with torch.no_grad():
            reconstruction_t = render_field(
                field.means,
                field.conics(aa_dilation),
                field.colors,
                field.radii(sigma_cutoff, aa_dilation),
                height,
                width,
                chunk=chunk,
                mode=renderer,
                opacities=field.opacity_values(),
                scales=scales,
                rotations=field.rotations,
                support_fade=args.support_fade,
                sigma_cutoff=sigma_cutoff,
                color_grads=field.color_grads,
            )
        format_name = "GaussianField NPZ"

    if reference is not None and reference.shape[:2] != (height, width):
        raise SystemExit(
            f"reference is {reference.shape[1]}x{reference.shape[0]}, "
            f"but the render is {width}x{height}"
        )
    if not bool(torch.isfinite(reconstruction_t).all()):
        raise SystemExit("renderer produced non-finite pixels")
    reconstruction = reconstruction_t.detach().cpu().numpy().astype(np.float32, copy=False)
    display = np.clip(reconstruction, 0.0, 1.0)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(str(output), display)

    if args.gaussians_out:
        from .visualize import gaussian_overlay

        overlay_path = Path(args.gaussians_out)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        background = reference if reference is not None else display
        gaussian_overlay(background, field, limit=args.ellipse_limit).save(overlay_path)

    metric_values = None
    if reference is not None:
        target_t = torch.as_tensor(reference, device=reconstruction_t.device)
        display_t = reconstruction_t.clamp(0.0, 1.0)
        signed_error = display - reference
        metric_values = {
            "field": str(source),
            "format": format_name,
            "reference": str(args.reference),
            "width": width,
            "height": height,
            "n_gaussians": int(field.n),
            "comparison": "display-referred reconstruction clamped to [0,1]",
            "mse": float(np.mean(np.square(signed_error), dtype=np.float64)),
            "mae": float(np.mean(np.abs(signed_error), dtype=np.float64)),
            "max_abs_error": float(np.max(np.abs(signed_error))),
            "psnr_db": M.psnr(display_t, target_t),
            "ssim": float(M.ssim(display_t, target_t)),
            "ms_ssim": None,
        }
        try:
            metric_values["ms_ssim"] = M.ms_ssim(display_t, target_t)
        except ValueError:
            # Images below the SSIM window still have well-defined pixel/PSNR/SSIM metrics.
            pass
        if args.error_out:
            error_path = Path(args.error_out)
            error_path.parent.mkdir(parents=True, exist_ok=True)
            save_error_heatmap(str(error_path), signed_error, args.error_scale)
        if args.raw_error_out:
            raw_error_path = Path(args.raw_error_out)
            raw_error_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_error_path.open("wb") as stream:
                np.save(stream, signed_error.astype(np.float32, copy=False))
        if args.metrics_out:
            metrics_path = Path(args.metrics_out)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(metric_values, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    line = f"rendered {field.n} Gaussians from {format_name} to {output} ({width}x{height})"
    if metric_values is not None:
        line += (
            f" | PSNR {metric_values['psnr_db']:.2f} dB"
            f" | SSIM {metric_values['ssim']:.4f}"
            f" | MAE {metric_values['mae']:.6f}"
        )
    print(line)


def cmd_ablation(args):
    run_ablation = _benchmark_symbol("benchmarks.ablation", "run_ablation")
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
    run_stage_search = _benchmark_symbol("benchmarks.stage_search", "run_stage_search")
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
        refine_sites=args.refine_sites,
        refine_primitives=args.refine_primitives,
        refine_nms_modes=args.refine_nms_modes,
        refine_color_inits=args.refine_color_inits,
        refine_score_modes=args.refine_score_modes,
        responsibility_mass_alpha=args.responsibility_mass_alpha,
        refine_prune_modes=args.refine_prune_modes,
        refine_relocate_modes=args.refine_relocate_modes,
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
        early_exit=args.early_exit,
        early_exit_window=args.early_exit_window,
        early_exit_min_delta=args.early_exit_min_delta,
        early_exit_min_iters=args.early_exit_min_iters,
        log_every=args.log_every, dedupe=not args.no_dedupe,
        max_configs=args.max_configs, resume=args.resume, max_new_cells=args.max_new_cells,
        shuffle_configs=args.shuffle_configs,
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
    from .config import (
        DEFAULT_INIT_STRATEGY,
        DEFAULT_PREDICTOR_FALLBACK_STRATEGY,
        FitConfig,
        GenConfig,
    )
    # Torch-free by construction (see structsplat.pipeline), so `--help` still works without torch.
    from .pipeline import PipelineConfig

    p = argparse.ArgumentParser(prog="structsplat")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared fit-option surface: consumed positionally by `fit` (one image) and `batch-fit`
    # (PORT-005, many images across worker processes). Keeping one definition site keeps the
    # two encode paths reproducible from the same logged config surface.
    f = argparse.ArgumentParser(add_help=False)
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
    f.add_argument("--render-checkpoint", action="store_true",
                   help="checkpoint reference-renderer chunks to reduce backward memory")
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
    f.add_argument("--wse-progressive-order", action="store_true",
                   help="order a pure-WSE detail set into blue-noise-like nested prefixes")
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
    f.add_argument("--live", action="store_true",
                   help="stream fitting to the igsv browser viewer (diagnostic; ADR-0018)")
    f.add_argument("--live-host", default="127.0.0.1")
    f.add_argument("--live-port", type=int, default=8890)
    f.add_argument("--live-token", default=None)
    f.add_argument("--live-every", type=int, default=25,
                   help="publish a snapshot every N fit iterations")
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
    f.add_argument(
        "--covariance-filter-mode",
        choices=["none", "generation_density"],
        default="none",
        help="fixed birth-cohort covariance filter (experimental)",
    )
    f.add_argument(
        "--covariance-filter-alpha",
        type=float,
        default=FitConfig.covariance_filter_alpha,
        help="density divisor alpha in H*W/(alpha*N_after)",
    )
    f.add_argument(
        "--covariance-filter-max-variance",
        type=float,
        default=FitConfig.covariance_filter_max_variance,
        help="maximum added isotropic variance in px^2",
    )
    f.add_argument("--support-fade", action="store_true",
                   help="subtract the Gaussian tail at sigma_cutoff for C0 compact support")
    f.add_argument("--support-fade-until-frac", type=float, default=None,
                   help="enable support fade for the first fraction of fit iterations")
    f.add_argument("--support-fade-crossfade-iters", type=int, default=10,
                   help="iterations used to ramp scheduled support fade off")
    f.add_argument("--optimizer", choices=["adam", "adamw", "adan"], default="adam")
    f.add_argument("--pixel-loss", choices=["l1", "l2", "charbonnier"], default="l1")
    f.add_argument("--loss-weighting", choices=["none", "tensor", "mask"], default="none",
                   help="mask: drop out-of-mask pixels from the loss (requires --mask)")
    f.add_argument("--loss-weight-beta", type=float, default=1.0)
    f.add_argument("--mask", default=None,
                   help="binary/alpha mask image (CORE-010); white=inside. Enables mask fitting")
    f.add_argument("--mask-invert", action="store_true",
                   help="treat black as inside (invert the mask) for white=background masks")
    f.add_argument("--mask-contain", action="store_true",
                   help="project means + cap scales so the sigma_cutoff support stays in the mask")
    f.add_argument("--mask-margin", type=float, default=1.5,
                   help="px safety margin for mask erosion/caps (must exceed ~0.71)")
    f.add_argument("--mask-coverage-weight", type=float, default=0.0,
                   help="soft penalty weight on the out-of-mask unnormalized weight sum")
    f.add_argument("--mask-cap-mode", choices=["isotropic", "anisotropic"], default="isotropic",
                   help="anisotropic: certify longer tangent caps near the boundary (ADR-0019)")
    f.add_argument("--mask-cap-refresh-every", type=int, default=10,
                   help="anisotropic cap recertification cadence in iterations")
    f.add_argument("--mask-undercoverage-weight", type=float, default=0.0,
                   help="hinge penalty weight on uncovered boundary-band pixels (CORE-011)")
    f.add_argument("--mask-interior-undercoverage-weight", type=float, default=0.0,
                   help="one-sided raw coverage-floor penalty in the reachable mask interior")
    f.add_argument("--mask-undercoverage-band", type=float, default=3.0,
                   help="px band beyond the margin supervised by the under-coverage hinge")
    f.add_argument("--mask-undercoverage-tau", type=float, default=0.1,
                   help="raw weight sum below which a band pixel counts as uncovered")
    f.add_argument("--mask-undercoverage-every", type=int, default=1,
                   help="evaluate one-sided coverage-floor losses every N optimizer steps")
    f.add_argument("--mask-boundary-add-every", type=int, default=None,
                   help="spawn tangent-aligned boundary Gaussians every N iters (CORE-011)")
    f.add_argument("--mask-boundary-add-count", type=int, default=0,
                   help="max boundary Gaussians spawned per event")
    f.add_argument("--mask-boundary-add-band", type=float, default=4.0,
                   help="px SDF band whose residual peaks seed boundary spawns")
    f.add_argument("--mask-boundary-add-spacing", type=float, default=5.0,
                   help="px spacing between boundary spawns along the contour")
    f.add_argument(
        "--geometry-loss-weight",
        type=float,
        default=0.0,
        help="weight for ground-truth-gradient-weighted Sobel consistency",
    )
    f.add_argument(
        "--geometry-loss-every",
        type=int,
        default=1,
        help="evaluate Sobel consistency every N steps and scale its active weight by N",
    )
    f.add_argument("--loss-warmup-iters", type=int, default=0)
    f.add_argument("--loss-warmup-pixel-loss", choices=["l1", "l2", "charbonnier"], default="l2")
    f.add_argument(
        "--loss-target-downsample",
        type=int,
        default=1,
        help="area-downsample factor for optional coarse-to-full loss-target curriculum",
    )
    f.add_argument(
        "--loss-target-full-frac",
        type=float,
        default=0.0,
        help="global fit fraction by which the loss target becomes the full image",
    )
    f.add_argument("--ssim-weight", type=float, default=0.3)
    f.add_argument("--ssim-backend", choices=["builtin", "fused", "auto"], default="builtin")
    f.add_argument("--lpips", action="store_true", help="compute LPIPS after fitting")
    f.add_argument(
        "--checkpoint-policy",
        choices=["terminal", "best_psnr_final_count"],
        default="terminal",
        help="terminal field or best post-transition PSNR state at the terminal Gaussian count",
    )
    f.add_argument("--flank-offset", type=float, default=None,
                   help="edge-center offset fraction; default is strategy-aware")
    f.add_argument("--max-axis-ratio", type=float, default=6.0)
    f.add_argument("--coherence-power", type=float, default=1.0)
    f.add_argument("--lr-schedule", choices=["none", "step", "cosine"], default="none")
    f.add_argument(
        "--lr-cosine-start-frac",
        type=float,
        default=0.0,
        help="hold full LR through this fraction of a cosine schedule, then decay",
    )
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
    f.add_argument(
        "--refine-site",
        choices=[
            "none", "residual", "residual_tensor", "support", "responsibility",
            "absgrad", "ranked", "freq_violation",
        ],
        default=None,
        help="factored refinement site; overrides the site implied by --split-mode",
    )
    f.add_argument(
        "--refine-primitive",
        choices=["duplicate", "fp", "moment_preserving", "sampled_add"],
        default=None,
        help="factored refinement primitive; overrides the primitive implied by --split-mode",
    )
    f.add_argument(
        "--refine-nms",
        choices=["off", "on"],
        default=None,
        help="factored sampled-add NMS mode; overrides the mode implied by --split-mode",
    )
    f.add_argument("--split-scale", type=float, default=0.7)
    f.add_argument("--split-oversample", type=float, default=1.0,
                   help="candidate multiplier for residual-add spacing suppression")
    f.add_argument("--split-min-spacing", type=float, default=0.0,
                   help="residual-add spacing radius as a multiple of the densify base scale")
    f.add_argument("--split-color-init", choices=["target", "residual"], default="target",
                   help="color initialization for normalized residual-add children")
    f.add_argument("--sampled-add-score",
                   choices=["legacy_abs", "gaussian_abs", "signed_gaussian"],
                   default="legacy_abs",
                   help="sampled-add site score; signed_gaussian is the FIT-017 experiment")
    f.add_argument(
        "--responsibility-mass-alpha",
        type=float,
        default=0.7,
        help="ownership-mass exponent for --refine-site responsibility (0 < alpha <= 1)",
    )
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
    f.add_argument("--triage-every", type=int, default=None,
                   help="FIT-021 pooled triage cadence; enables the pooled row lifecycle and "
                        "replaces --split-every/--relocate-every/--prune-every")
    f.add_argument("--triage-spawn-count", type=int, default=64)
    f.add_argument("--triage-split-count", type=int, default=16)
    f.add_argument("--triage-merge-count", type=int, default=16)
    f.add_argument("--triage-park-count", type=int, default=16)
    f.add_argument("--target-file-kb", type=float, default=None,
                   help="encoded SSPL1(+alpha) budget in decimal KB (168 -> 168,000 bytes, the "
                        "same convention as the realtime-gs .rtgsv cap); derives the pool "
                        "capacity and writes the budgeted .sspl next to the NPZ")
    f.add_argument("--pool-capacity", type=int, default=None,
                   help="explicit pooled capacity in rows (overrides --target-file-kb)")
    f.add_argument("--coverage-match-weight", type=float, default=0.0,
                   help="FIT-022 coverage-matching regularizer weight; 0 disables")
    f.add_argument("--coverage-match-target",
                   choices=["tensor", "tensor_boundary", "error_blend"], default="tensor")
    f.add_argument("--coverage-match-beta", type=float, default=1.0)
    f.add_argument("--coverage-match-boundary-boost", type=float, default=2.0)
    f.add_argument("--coverage-match-boundary-band", type=float, default=3.0)
    f.add_argument("--coverage-match-error-alpha", type=float, default=0.5)
    f.add_argument("--coverage-match-decay-frac", type=float, default=0.0,
                   help="cosine-decay the coverage weight to zero at this schedule fraction")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--outdir", default="runs")
    f.add_argument("--device", default=None)

    c = sub.add_parser(
        "convert",
        help="convert an image into a GaussianField with the current best pipeline (ADR-0025)",
        description=(
            "The maintained best-known pipeline. Pass --mask for alpha-matted/dome inputs to get "
            "the masked arm (containment, boundary initialization and closure); omit it for the "
            "full-frame arm, which runs the identical schedule with the mask stages inert. "
            "`structsplat fit` remains the knob-level research command."
        ),
    )
    c.add_argument("image")
    c.add_argument("--mask", default=None,
                   help="mask image (grayscale or RGBA alpha); selects the masked arm")
    c.add_argument("--mask-invert", action="store_true",
                   help="treat the dark side of the mask as the object")
    c.add_argument("--capacity", type=int, default=PipelineConfig.capacity,
                   help="row budget; the phase targets scale with it")
    c.add_argument("--initial-gaussians", type=int, default=None,
                   help="rows before growth (default: capacity * 5/11)")
    c.add_argument("--boundary-gaussians", type=int, default=None,
                   help="masked arm: tangent-aligned boundary rows in the initial field "
                        "(default: 10%% of the initial rows)")
    c.add_argument("--coverage-target", type=int, default=None,
                   help="row count the coverage phase grows to (default: capacity * 8/11)")
    c.add_argument("--detail-target", type=int, default=None,
                   help="row count the detail phase grows to (default: capacity * 10/11)")
    c.add_argument("--step-scale", type=float, default=PipelineConfig.step_scale,
                   help="scale every phase's step ceiling (0.1 for a quick look)")
    c.add_argument("--block-steps", type=int, default=None,
                   help="commit-gate granularity in steps")
    c.add_argument("--storage-policy",
                   choices=["dynamic", "fixed_capacity", "geometric"],
                   default=PipelineConfig.storage_policy)
    c.add_argument("--no-checkpoints", action="store_true",
                   help="disable Pareto-safe state checkpoints (FIT-023 found them a win; "
                        "this reverts to the matched control)")
    c.add_argument("--checkpoint-every", type=int,
                   default=PipelineConfig.pareto_checkpoint_every)
    c.add_argument("--event-color-solve", action="store_true",
                   help="post-topology color solve; FIT-023 measured it worse, default off")
    c.add_argument("--mask-margin", type=float, default=PipelineConfig.mask_margin)
    c.add_argument("--boundary-band", type=float, default=PipelineConfig.boundary_band)
    c.add_argument("--coverage-tau", type=float, default=PipelineConfig.coverage_tau)
    c.add_argument("--hole-regression-budget", type=float,
                   default=PipelineConfig.hole_regression_budget,
                   help="ADR-0026: interior hole fraction a block may add and still commit "
                        "(default 0.0 = strict). Boundary holes are never budgeted.")
    c.add_argument("--renderer", default=None,
                   choices=["normalized", "cuda", "cuda_tiled"],
                   help="default: cuda on a CUDA device, else normalized")
    c.add_argument("--seed", type=int, default=PipelineConfig.seed)
    c.add_argument("--device", default=None)
    c.add_argument("--outdir", default="runs/convert")
    c.add_argument("--quiet", action="store_true")
    c.set_defaults(func=cmd_convert)

    fit_parser = sub.add_parser(
        "fit",
        aliases=["image-to-gaussians2d"],
        help="fit a single image into a native 2D GaussianField (knob-level research command)",
        parents=[f],
    )
    fit_parser.add_argument("image")
    fit_parser.set_defaults(func=cmd_fit)

    render_parser = sub.add_parser(
        "render",
        aliases=["gaussians2d-to-image"],
        help="render a GaussianField NPZ or self-describing SSPL1 stream",
    )
    render_parser.add_argument("field", help="GaussianField .npz or SSPL1 stream")
    render_parser.add_argument("--out", required=True, help="output reconstruction image")
    render_parser.add_argument(
        "--reference",
        help="original image; supplies NPZ dimensions and enables errors/metrics",
    )
    render_parser.add_argument("--height", type=int, help="NPZ canvas height without --reference")
    render_parser.add_argument("--width", type=int, help="NPZ canvas width without --reference")
    render_parser.add_argument(
        "--renderer",
        choices=[
            "normalized", "additive", "cuda", "cuda_additive",
            "cuda_tiled", "cuda_tiled_additive", "gsplat",
        ],
        default=None,
        help="NPZ render mode (default: normalized); SSPL1 uses its stored mode",
    )
    render_parser.add_argument("--chunk", type=int, default=None)
    render_parser.add_argument("--aa-dilation", type=float, default=None)
    render_parser.add_argument("--sigma-cutoff", type=float, default=None)
    render_parser.add_argument("--support-fade", action="store_true")
    render_parser.add_argument(
        "--error-out",
        help="fixed-scale heatmap of mean absolute RGB error (requires --reference)",
    )
    render_parser.add_argument(
        "--error-scale",
        type=float,
        default=4.0,
        help="multiplier used only for --error-out display (default: 4)",
    )
    render_parser.add_argument(
        "--raw-error-out",
        help="signed float32 HxWx3 reconstruction-minus-reference .npy",
    )
    render_parser.add_argument(
        "--metrics-out",
        help="JSON reconstruction metrics computed before image encoding",
    )
    render_parser.add_argument(
        "--gaussians-out",
        help="one-sigma fitted-Gaussian ellipse overlay on the reference/reconstruction",
    )
    render_parser.add_argument(
        "--ellipse-limit",
        type=int,
        default=500,
        help="maximum evenly sampled ellipses in --gaussians-out",
    )
    render_parser.add_argument("--device", default=None)
    render_parser.set_defaults(func=cmd_render)

    b = sub.add_parser(
        "batch-fit",
        help="fit many images in parallel worker processes (PORT-005 encode throughput)",
        parents=[f],
    )
    b.add_argument("images", nargs="+", help="image files, directories, or glob patterns")
    b.add_argument("--devices", default=None,
                   help="comma-separated device list to round-robin workers over "
                        "(e.g. cuda:0,cuda:1); default: cuda if available else cpu")
    b.add_argument("--workers", type=int, default=None,
                   help="worker processes; default: one per device (CPU default: 1)")
    b.add_argument("--force", action="store_true",
                   help="re-fit images whose output .npz already exists instead of skipping")
    b.add_argument("--torch-threads", type=int, default=None,
                   help="torch intra-op threads per worker (CPU oversubscription guard)")
    b.set_defaults(func=cmd_batch_fit)

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
    s.add_argument("--refine-sites", nargs="+", default=None,
                   help="none, residual, residual_tensor, support, responsibility, ranked, "
                        "absgrad, freq_violation")
    s.add_argument("--refine-primitives", nargs="+", default=None,
                   help="duplicate, fp, moment_preserving, sampled_add")
    s.add_argument("--refine-nms-modes", nargs="+", default=None,
                   help="off or on")
    s.add_argument("--refine-color-inits", nargs="+", default=None,
                   help="target or residual")
    s.add_argument("--refine-score-modes", nargs="+", default=None,
                   help="legacy_abs, gaussian_abs, or signed_gaussian; sampled-add only")
    s.add_argument(
        "--responsibility-mass-alpha",
        type=float,
        default=0.7,
        help="fixed ownership-mass exponent for responsibility refinement cells",
    )
    s.add_argument("--refine-prune-modes", nargs="+", default=None,
                   help="off or on")
    s.add_argument("--refine-relocate-modes", nargs="+", default=None,
                   help="off or on")
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
    s.add_argument("--early-exit", action="store_true",
                   help="opt-in plateau early exit; AUC holds last PSNR to the nominal horizon")
    s.add_argument("--early-exit-window", type=int, default=150)
    s.add_argument("--early-exit-min-delta", type=float, default=0.02)
    s.add_argument("--early-exit-min-iters", type=int, default=None)
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
    s.add_argument("--resume", action="store_true",
                   help="resume from stage_search.jsonl in --outdir and skip completed cells")
    s.add_argument("--max-new-cells", type=int, default=None,
                   help="stop after this many newly executed cells; useful for GPU shards")
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
