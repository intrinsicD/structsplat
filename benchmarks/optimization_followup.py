"""Follow-up optimization checks for the COCO stage-search results.

This script runs exact candidate configurations rather than a factorial product:

1. validate the current baseline and top sampled configs on held-out COCO images;
2. benchmark the reference renderer against a tile-local forward prototype;
3. sweep WSE candidate oversampling and opt-in early stopping;
4. revisit pyramid/refinement variants after the above checks.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import torch

from benchmarks.common import add_seed_args, load_image as _load_image, psnr_auc
from benchmarks.common import resolve_seeds, run_config, write_config, write_rows as _write_rows
from structsplat.config import FitConfig, InitConfig, PyramidConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.gaussians import GaussianField
from structsplat.init import build_field
from structsplat.pyramid import fit_pyramid
from structsplat.render import render_field


STAGE_KEYS = [
    "strategy", "tensor", "tensor_color", "density", "sampling", "orientation", "color",
    "scale", "opacity", "renderer", "loss", "optimizer", "lr_schedule", "refine", "pyramid",
]

PREVIOUS_FOUR = {
    "COCO_train2014_000000000009.jpg",
    "COCO_train2014_000000000025.jpg",
    "COCO_train2014_000000000030.jpg",
    "COCO_train2014_000000000034.jpg",
}


@dataclass(frozen=True)
class Candidate:
    label: str
    strategy: str = "aniso_flanking"
    tensor: str = "central"
    tensor_color: str = "luma"
    density: str = "structure"
    sampling: str = "wse"
    orientation: str = "tensor"
    color: str = "bilinear"
    scale: str = "spacing"
    opacity: str = "none"
    renderer: str = "normalized"
    loss: str = "l1"
    optimizer: str = "adam"
    lr_schedule: str = "none"
    refine: str = "none"
    pyramid: str = "single"
    candidate_oversample: float = 6.0
    scale_cap_mode: str = "none"
    scale_cap_max: float | None = None
    scale_feature_sigma: float = 3.0
    scale_feature_min: float = 0.75
    scale_feature_energy_frac: float = 0.25
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
    early_stop_min_iters: int = 0

    @property
    def config_label(self) -> str:
        return "|".join(f"{k}={getattr(self, k)}" for k in STAGE_KEYS)


BASELINE = Candidate("current_baseline")

TOP1 = Candidate(
    "stage_top1",
    density="variance",
    opacity="constant",
    loss="charbonnier",
)

TOP2 = Candidate(
    "stage_top2",
    strategy="aniso_onedge",
    tensor="scharr",
    density="structure",
    opacity="constant",
    loss="charbonnier",
    refine="prune_residual_add",
)

TOP3 = Candidate(
    "stage_top3",
    strategy="aniso_onedge",
    tensor="scharr",
    tensor_color="rgb",
    density="variance",
    color="local_mean",
    loss="l1",
    optimizer="adamw",
    refine="residual_add",
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _select_images(dataset_dir: Path, count: int) -> list[Path]:
    files = sorted(p for p in dataset_dir.glob("COCO_train2014_*.jpg")
                   if p.name not in PREVIOUS_FOUR)
    if len(files) < count:
        raise SystemExit(f"only found {len(files)} held-out images, need {count}")
    return files[:count]


def _refine_kwargs(c: Candidate, iters: int, split_count: int, max_gaussians: int | None) -> dict:
    if c.refine == "none":
        return {}
    split_every = max(1, iters // 2)
    prune_every = max(1, iters // 2)
    if c.refine == "residual_add":
        return {
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": "residual_add",
            "max_gaussians": max_gaussians,
        }
    if c.refine == "prune_residual_add":
        return {
            "prune_every": prune_every,
            "prune_min_activity": 1e-2,
            "split_every": split_every,
            "split_count": split_count,
            "split_mode": "residual_add",
            "max_gaussians": max_gaussians,
        }
    raise ValueError(f"unknown refine mode {c.refine!r}")


def _make_configs(c: Candidate, budget: int, seed: int, iters: int, render_chunk: int,
                  split_count: int, max_gaussians: int | None) -> tuple[InitConfig, FitConfig,
                                                                        StructureTensorConfig]:
    scfg = StructureTensorConfig(gradient_operator=c.tensor, color_space=c.tensor_color)
    # equal-budget arms (BENCH-002): a refine arm caps at the cell budget and starts below it so
    # its planned additions land AT budget, instead of running with up to +split_count more
    # capacity than the baselines it is ranked against.
    adds = c.refine != "none"
    if adds and max_gaussians is None:
        max_gaussians = budget
    init_budget = max(1, budget - split_count) if adds else budget
    icfg = InitConfig(
        strategy=c.strategy,
        num_gaussians=init_budget,
        candidate_oversample=c.candidate_oversample,
        density_mode=c.density,
        sampling_mode=c.sampling,
        orientation_mode=c.orientation,
        color_mode=c.color,
        scale_mode=c.scale,
        scale_cap_mode=c.scale_cap_mode,
        scale_cap_max=c.scale_cap_max,
        scale_feature_sigma=c.scale_feature_sigma,
        scale_feature_min=c.scale_feature_min,
        scale_feature_energy_frac=c.scale_feature_energy_frac,
        opacity_mode=c.opacity,
        init_opacity=0.9,
        seed=seed,
    )
    fcfg = FitConfig(
        iters=iters,
        render_chunk=render_chunk,
        target_psnr=30.0,
        target_psnrs=[30.0],
        log_every=max(1, iters // 20),
        pixel_loss=c.loss,
        optimizer=c.optimizer,
        lr_schedule=c.lr_schedule,
        renderer=c.renderer,
        compute_lpips=False,
        early_stop_patience=c.early_stop_patience,
        early_stop_min_delta=c.early_stop_min_delta,
        early_stop_min_iters=c.early_stop_min_iters,
        **_refine_kwargs(c, iters, split_count, max_gaussians),
    )
    return icfg, fcfg, scfg


def run_candidate(
    image_path: Path,
    c: Candidate,
    *,
    budget: int,
    seed: int,
    iters: int,
    max_side: int,
    render_chunk: int,
    split_count: int,
    max_gaussians: int | None,
    device: str,
    return_field: bool = False,
) -> tuple[dict, GaussianField | None, FitConfig | None, torch.Tensor | None]:
    img = _load_image(image_path, max_side)
    target = torch.as_tensor(img, dtype=torch.float32, device=device)
    icfg, fcfg, scfg = _make_configs(c, budget, seed, iters, render_chunk, split_count,
                                     max_gaussians)
    start = time.time()
    if c.pyramid == "single":
        field = build_field(img, icfg, scfg, device=device)
        init_seconds = time.time() - start
        out = fit(field, target, fcfg, verbose=False)
        total_seconds = time.time() - start
    elif c.pyramid == "pyramid":
        pcfg = PyramidConfig(levels=2, level_fractions=[0.35, 0.65],
                             iters_per_level=max(1, iters // 2))
        out = fit_pyramid(img, target, icfg, fcfg, pcfg, scfg, verbose=False)
        field = out["field"]
        total_seconds = time.time() - start
        init_seconds = max(total_seconds - float(out.get("fit_seconds", 0.0)), 0.0)
    else:
        raise ValueError(f"unknown pyramid mode {c.pyramid!r}")

    hist = out.get("history", {})
    iterations_run = int(out.get("iterations_run") or ((max(hist.get("iter", [iters - 1])) + 1)
                                                       if hist else iters))
    row = {
        "image": image_path.stem,
        "source_path": str(image_path),
        "candidate": c.label,
        "budget": budget,
        "seed": seed,
        **{k: getattr(c, k) for k in STAGE_KEYS},
        "candidate_oversample": c.candidate_oversample,
        "early_stop_patience": c.early_stop_patience,
        "early_stop_min_delta": c.early_stop_min_delta,
        "early_stop_min_iters": c.early_stop_min_iters,
        "config_label": c.config_label,
        "width": img.shape[1],
        "height": img.shape[0],
        "psnr": round(float(out["psnr"]), 4),
        "ssim": round(float(out["ssim"]), 5),
        "ms_ssim": round(float(out["ms_ssim"]), 5),
        "auc_psnr": psnr_auc(hist, round_digits=4),
        "iters_to_target": out.get("iters_to_target"),
        "n_gaussians": int(out["n_gaussians"]),
        "init_seconds": round(float(init_seconds), 4),
        "fit_seconds": round(float(out.get("fit_seconds", 0.0)), 4),
        "total_seconds": round(float(total_seconds), 4),
        "iterations_run": iterations_run,
        "stopped_early": bool(out.get("stopped_early", False)),
    }
    return row, (field if return_field else None), (fcfg if return_field else None), (
        target if return_field else None)


def _run_matrix(images: list[Path], candidates: Iterable[Candidate], section: str, args) -> list[dict]:
    rows = []
    candidates = list(candidates)
    for image in images:
        for seed in args.seeds:
            for c in candidates:
                print(f"[{section}] {image.stem} seed={seed} {c.label}", flush=True)
                row, _, _, _ = run_candidate(
                    image, c, budget=args.budget, seed=seed, iters=args.iters,
                    max_side=args.max_side, render_chunk=args.render_chunk,
                    split_count=args.split_count, max_gaussians=args.max_gaussians,
                    device=args.device,
                )
                rows.append({"section": section, **row})
                print(f"  psnr={row['psnr']:.3f} n={row['n_gaussians']} "
                      f"fit={row['fit_seconds']:.2f}s total={row['total_seconds']:.2f}s "
                      f"iters={row['iterations_run']}", flush=True)
    return rows


@torch.no_grad()
def render_tiled_forward(field: GaussianField, cfg: FitConfig, H: int, W: int,
                         tile: int = 16) -> torch.Tensor:
    means = field.means
    conics = field.conics(cfg.aa_dilation)
    colors = field.colors
    radii = field.radii(cfg.sigma_cutoff, cfg.aa_dilation)
    opacities = field.opacity_values()
    dev, dt = means.device, means.dtype

    ix = torch.round(means[:, 0].detach()).long()
    iy = torch.round(means[:, 1].detach()).long()
    gx0 = (ix - radii[:, 0]).clamp(min=0)
    gx1 = (ix + radii[:, 0]).clamp(max=W - 1)
    gy0 = (iy - radii[:, 1]).clamp(min=0)
    gy1 = (iy + radii[:, 1]).clamp(max=H - 1)

    out = torch.zeros(H, W, 3, device=dev, dtype=dt)
    for ty0 in range(0, H, tile):
        ty1 = min(H, ty0 + tile)
        yy = torch.arange(ty0, ty1, device=dev, dtype=dt)
        for tx0 in range(0, W, tile):
            tx1 = min(W, tx0 + tile)
            overlap = (gx1 >= tx0) & (gx0 < tx1) & (gy1 >= ty0) & (gy0 < ty1)
            idx = torch.nonzero(overlap, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            xx = torch.arange(tx0, tx1, device=dev, dtype=dt)
            py, px = torch.meshgrid(yy, xx, indexing="ij")
            px = px.reshape(1, -1)
            py = py.reshape(1, -1)
            m = means[idx]
            dx = px - m[:, 0:1]
            dy = py - m[:, 1:2]
            a = conics[idx, 0:1]
            b = conics[idx, 1:2]
            c = conics[idx, 2:3]
            q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
            w = torch.exp(-0.5 * q)
            support = ((px >= gx0[idx, None]) & (px <= gx1[idx, None])
                       & (py >= gy0[idx, None]) & (py <= gy1[idx, None]))
            w = w * support.to(dt)
            if opacities is not None:
                w = w * opacities[idx, None]
            num = w.transpose(0, 1) @ colors[idx]
            if cfg.renderer == "normalized":
                den = w.sum(dim=0).unsqueeze(1)
                tile_img = num / (den + 1e-8)
            elif cfg.renderer == "additive":
                tile_img = num
            else:
                raise ValueError(f"unknown renderer {cfg.renderer!r}")
            out[ty0:ty1, tx0:tx1] = tile_img.view(ty1 - ty0, tx1 - tx0, 3)
    return out


def _time_call(fn, repeats: int = 8) -> tuple[float, torch.Tensor]:
    for _ in range(2):
        result = fn()
    _sync()
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn()
    _sync()
    return (time.perf_counter() - start) / repeats, result


def run_spatial_benchmark(images: list[Path], args) -> list[dict]:
    rows = []
    for image in images[:args.spatial_images]:
        for seed in args.seeds:
            print(f"[spatial] {image.stem} seed={seed} fitting {TOP1.label}", flush=True)
            fit_row, field, fcfg, target = run_candidate(
                image, TOP1, budget=args.budget, seed=seed, iters=args.iters,
                max_side=args.max_side, render_chunk=args.render_chunk,
                split_count=args.split_count, max_gaussians=args.max_gaussians,
                device=args.device, return_field=True,
            )
            assert field is not None and fcfg is not None and target is not None
            H, W = target.shape[:2]
            # symmetric timing (BENCH-002): both closures under the SAME grad mode. render_tiled_forward
            # is @torch.no_grad, so timing the reference WITH autograd graph construction inflated the
            # reported speedup; a forward-speed benchmark measures both forwards under no_grad.
            with torch.no_grad():
                ref_s, ref_img = _time_call(
                    lambda: render_field(field.means, field.conics(fcfg.aa_dilation), field.colors,
                                         field.radii(fcfg.sigma_cutoff, fcfg.aa_dilation), H, W,
                                         fcfg.render_chunk, fcfg.renderer, field.opacity_values()),
                    repeats=args.render_repeats,
                )
                tile_s, tile_img = _time_call(
                    lambda: render_tiled_forward(field, fcfg, H, W, args.tile_size),
                    repeats=args.render_repeats,
                )
            max_abs = float((ref_img - tile_img).abs().max().detach().cpu())
            rows.append({
                "section": "spatial_render",
                "image": image.stem,
                "seed": seed,
                "width": W,
                "height": H,
                "n_gaussians": fit_row["n_gaussians"],
                "reference_forward_seconds": round(ref_s, 6),
                "tile_forward_seconds": round(tile_s, 6),
                "tile_size": args.tile_size,
                "speedup_vs_reference": round(ref_s / tile_s, 4) if tile_s > 0 else None,
                "max_abs_diff": round(max_abs, 8),
            })
            print(f"  ref={ref_s:.5f}s tile={tile_s:.5f}s speedup={ref_s / tile_s:.3f} "
                  f"diff={max_abs:.2e}", flush=True)
    return rows


def _aggregate(rows: list[dict], key: str = "candidate") -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    out = []
    for label, vals in groups.items():
        ps = [float(v["psnr"]) for v in vals]
        out.append({
            key: label,
            "runs": len(vals),
            "mean_psnr": mean(ps),
            "std_psnr": pstdev(ps) if len(ps) > 1 else 0.0,
            "mean_ms_ssim": mean(float(v["ms_ssim"]) for v in vals),
            "mean_auc_psnr": mean(float(v["auc_psnr"]) for v in vals if v["auc_psnr"] is not None),
            "mean_fit_seconds": mean(float(v["fit_seconds"]) for v in vals),
            "mean_total_seconds": mean(float(v["total_seconds"]) for v in vals),
            "mean_iterations_run": mean(float(v["iterations_run"]) for v in vals),
            "mean_n_gaussians": mean(float(v["n_gaussians"]) for v in vals),
            "stopped_early": sum(1 for v in vals if v.get("stopped_early")),
        })
    return sorted(out, key=lambda r: r["mean_psnr"], reverse=True)


def _markdown_table(rows: list[dict], label_key: str = "candidate") -> list[str]:
    lines = [
        f"| {label_key} | Runs | Mean PSNR | Std | MS-SSIM | AUC | Fit s | Total s | Iters | Gaussians | Early stops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r[label_key]}` | {r['runs']} | {r['mean_psnr']:.4f} | "
            f"{r['std_psnr']:.4f} | {r['mean_ms_ssim']:.5f} | "
            f"{r['mean_auc_psnr']:.4f} | {r['mean_fit_seconds']:.3f} | "
            f"{r['mean_total_seconds']:.3f} | {r['mean_iterations_run']:.1f} | "
            f"{r['mean_n_gaussians']:.1f} | {r['stopped_early']} |"
        )
    return lines


def write_summary(outdir: Path, sections: dict[str, list[dict]], images: list[Path], args) -> None:
    lines = [
        "# Optimization Follow-up",
        "",
        f"- Held-out images: {len(images)}",
        f"- Budget: {args.budget}",
        f"- Iterations: {args.iters}",
        f"- Seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- Max side: {args.max_side}",
        f"- Render chunk: {args.render_chunk}",
        f"- Device: {args.device}",
        "- Aggregate rows report mean and population std over image x seed runs.",
        "",
        "## Images",
        "",
    ]
    lines += [f"- `{p.name}`" for p in images]
    for name in ("validation", "oversample", "early_stop", "refine_pyramid"):
        rows = sections.get(name, [])
        if not rows:
            continue
        lines += ["", f"## {name.replace('_', ' ').title()}", ""]
        lines += _markdown_table(_aggregate(rows))
    spatial = sections.get("spatial_render", [])
    if spatial:
        lines += [
            "",
            "## Spatial Render Prototype",
            "",
            "| Image | Seed | N | Ref forward s | Tile forward s | Speedup | Max abs diff |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in spatial:
            lines.append(
                f"| `{r['image']}` | {r['seed']} | {r['n_gaussians']} | "
                f"{r['reference_forward_seconds']:.6f} | {r['tile_forward_seconds']:.6f} | "
                f"{r['speedup_vs_reference']:.4f} | {r['max_abs_diff']:.2e} |"
            )
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run bounded optimization follow-up experiments")
    p.add_argument("--dataset-dir", type=Path, default=None,
                   help="COCO train2014 directory (required; no personal-path default, BENCH-002)")
    p.add_argument("--outdir", type=Path, default=Path("results/optimization_followup"))
    p.add_argument("--image-count", type=int, default=8)
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--max-side", type=int, default=160)
    add_seed_args(p)
    p.add_argument("--render-chunk", type=int, default=512)
    p.add_argument("--split-count", type=int, default=128)
    p.add_argument("--max-gaussians", type=int, default=None)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument("--spatial-images", type=int, default=3)
    p.add_argument("--render-repeats", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.dataset_dir is None:
        raise SystemExit("--dataset-dir is required (path to the COCO train2014 images)")
    args.seeds = resolve_seeds(args.seed, args.seeds)
    args.seed = args.seeds[0]
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_config(str(args.outdir), run_config(vars(args), device=args.device))
    images = _select_images(args.dataset_dir, args.image_count)

    sections: dict[str, list[dict]] = {}
    sections["validation"] = _run_matrix(images, [BASELINE, TOP1, TOP2, TOP3],
                                         "validation", args)
    _write_rows(args.outdir / "validation", sections["validation"])

    oversample = [replace(TOP1, label=f"top1_wse_oversample_{v:g}", candidate_oversample=v)
                  for v in (2.0, 3.0, 4.0, 6.0)]
    sections["oversample"] = _run_matrix(images, oversample, "oversample", args)
    _write_rows(args.outdir / "oversample", sections["oversample"])

    early = [
        replace(TOP1, label="top1_fixed_80"),
        replace(TOP1, label="top1_es_p3_d001_m24", early_stop_patience=3,
                early_stop_min_delta=0.01, early_stop_min_iters=24),
        replace(TOP1, label="top1_es_p2_d002_m24", early_stop_patience=2,
                early_stop_min_delta=0.02, early_stop_min_iters=24),
    ]
    sections["early_stop"] = _run_matrix(images, early, "early_stop", args)
    _write_rows(args.outdir / "early_stop", sections["early_stop"])

    refine = [
        replace(TOP1, label="top1_single_none", refine="none", pyramid="single"),
        replace(TOP1, label="top1_single_residual_add", refine="residual_add", pyramid="single"),
        replace(TOP1, label="top1_single_prune_residual_add", refine="prune_residual_add",
                pyramid="single"),
        replace(TOP1, label="top1_pyramid_none", refine="none", pyramid="pyramid"),
        replace(TOP1, label="top1_pyramid_residual_add", refine="residual_add", pyramid="pyramid"),
    ]
    sections["refine_pyramid"] = _run_matrix(images, refine, "refine_pyramid", args)
    _write_rows(args.outdir / "refine_pyramid", sections["refine_pyramid"])

    sections["spatial_render"] = run_spatial_benchmark(images, args)
    _write_rows(args.outdir / "spatial_render", sections["spatial_render"])

    all_rows = [row for name, rows in sections.items() if name != "spatial_render" for row in rows]
    _write_rows(args.outdir / "all_fit_runs", all_rows)
    write_summary(args.outdir, sections, images, args)
    print(f"\nwrote optimization follow-up outputs to {args.outdir}")


if __name__ == "__main__":
    main()
