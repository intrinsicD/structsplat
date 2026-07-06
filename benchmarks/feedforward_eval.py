"""Evaluate FF-001 learned warm starts against equal-N baselines.

The comparison is deliberately narrow: every row uses the same final Gaussian budget and the same
short-refinement iteration count. It reports quality and wall time so learned checkpoints can be
screened before running larger amortized-prediction studies.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from benchmarks.ablation import _iter_images
from benchmarks.common import load_image, run_config, target_tensor, write_config, write_csv, write_json


DEFAULT_METHODS = ("learned", "tensor_prior", "scratch")


def _psnr_auc(history: dict) -> float | None:
    psnr = history.get("psnr", [])
    iters = history.get("iter", [])
    if not psnr:
        return None
    if len(psnr) == 1 or len(iters) < 2:
        return float(psnr[0])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(psnr, iters) / max(iters[-1] - iters[0], 1))


def _method_init_config(
    method: str,
    *,
    budget: int,
    checkpoint: str,
    prior_strategy: str,
    scratch_strategy: str,
    seed: int,
):
    from structsplat.config import InitConfig

    if method == "learned":
        return InitConfig(
            strategy="feedforward",
            num_gaussians=budget,
            predictor_checkpoint=checkpoint,
            predictor_fallback_strategy=prior_strategy,
            seed=seed,
        )
    if method == "tensor_prior":
        return InitConfig(strategy=prior_strategy, num_gaussians=budget, seed=seed)
    if method == "scratch":
        return InitConfig(strategy=scratch_strategy, num_gaussians=budget, seed=seed)
    raise ValueError(
        f"unknown method {method!r}; expected one of {', '.join(DEFAULT_METHODS)}"
    )


def evaluate_feedforward_predictor(
    images,
    *,
    checkpoint: str,
    outdir: str | Path = "results/feedforward_eval",
    budget: int = 512,
    iters: int = 80,
    max_side: int | None = 160,
    render_chunk: int = 512,
    seed: int = 0,
    device: str | None = None,
    prior_strategy: str = "aniso_flanking",
    scratch_strategy: str = "random",
    methods=DEFAULT_METHODS,
    target_psnr: float | None = None,
) -> list[dict]:
    """Run learned/tensor-prior/scratch equal-N short-refinement comparisons."""

    import torch

    from structsplat import init as _init
    from structsplat.config import FitConfig, StructureTensorConfig
    from structsplat.fit import fit

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = _iter_images(images)
    if not files:
        raise SystemExit("no images found")
    methods = tuple(methods)
    unknown = set(methods) - set(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = run_config({
        "images": files,
        "checkpoint": checkpoint,
        "budget": int(budget),
        "iters": int(iters),
        "max_side": max_side,
        "render_chunk": int(render_chunk),
        "seed": int(seed),
        "prior_strategy": prior_strategy,
        "scratch_strategy": scratch_strategy,
        "methods": list(methods),
        "target_psnr": target_psnr,
    }, device=device)
    write_config(str(out), cfg)

    rows = []
    scfg = StructureTensorConfig()
    for path in files:
        img = load_image(path, max_side)
        target = target_tensor(img, device)
        for method in methods:
            icfg = _method_init_config(
                method,
                budget=int(budget),
                checkpoint=checkpoint,
                prior_strategy=prior_strategy,
                scratch_strategy=scratch_strategy,
                seed=int(seed),
            )
            init_start = time.time()
            field = _init.build_field(img, icfg, scfg, device=device)
            init_seconds = time.time() - init_start
            fcfg = FitConfig(
                iters=int(iters),
                render_chunk=int(render_chunk),
                log_every=max(1, int(iters) // 10) if int(iters) > 0 else 1,
                target_psnr=target_psnr,
            )
            fit_out = fit(field, target, fcfg, verbose=False)
            row = {
                "image": Path(path).stem,
                "source_path": str(path),
                "method": method,
                "budget": int(budget),
                "n_gaussians": int(fit_out["field"].n),
                "iters": int(iters),
                "iterations_run": int(fit_out.get("iterations_run", 0)),
                "psnr": round(float(fit_out["psnr"]), 4),
                "ms_ssim": round(float(fit_out["ms_ssim"]), 5),
                "psnr_auc": None if _psnr_auc(fit_out["history"]) is None
                else round(float(_psnr_auc(fit_out["history"])), 4),
                "init_seconds": round(float(init_seconds), 6),
                "fit_seconds": round(float(fit_out.get("fit_seconds", 0.0)), 6),
                "total_seconds": round(float(init_seconds + fit_out.get("fit_seconds", 0.0)), 6),
                "iters_to_target": fit_out.get("iters_to_target"),
                "estimated_bpp": round(float(fit_out.get("estimated_bpp", 0.0)), 6),
            }
            rows.append(row)
            print(row)

    write_json(out / "feedforward_eval.json", {"config": cfg, "rows": rows})
    write_csv(out / "feedforward_eval.csv", rows)
    _write_summary(out, rows, target_psnr=target_psnr)
    return rows


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return None if not vals else sum(vals) / len(vals)


def _write_summary(out: Path, rows: list[dict], *, target_psnr: float | None) -> None:
    methods = sorted({r["method"] for r in rows})
    lines = [
        "# Feed-Forward Predictor Evaluation",
        "",
        f"Rows: {len(rows)}",
        "",
        "| method | rows | mean PSNR | mean MS-SSIM | mean AUC | mean total s | target hits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        hits = sum(r.get("iters_to_target") is not None for r in subset)
        lines.append(
            f"| {method} | {len(subset)} | {_mean(r['psnr'] for r in subset):.4f} | "
            f"{_mean(r['ms_ssim'] for r in subset):.5f} | "
            f"{_mean(r['psnr_auc'] for r in subset):.4f} | "
            f"{_mean(r['total_seconds'] for r in subset):.6f} | {hits} |"
        )
    lines.extend([
        "",
        f"Target PSNR: {target_psnr if target_psnr is not None else 'not set'}",
        "",
        "Verdict: equal-N evaluator completed. Interpret quality/speed only at the scope of the "
        "image split and checkpoint used for this run.",
    ])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate FF-001 learned warm starts")
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", default="results/feedforward_eval")
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--max-side", type=int, default=160)
    p.add_argument("--render-chunk", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--prior-strategy", default="aniso_flanking")
    p.add_argument("--scratch-strategy", default="random")
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                   choices=list(DEFAULT_METHODS))
    p.add_argument("--target-psnr", type=float, default=None)
    args = p.parse_args()
    evaluate_feedforward_predictor(
        args.images,
        checkpoint=args.checkpoint,
        outdir=args.outdir,
        budget=args.budget,
        iters=args.iters,
        max_side=args.max_side,
        render_chunk=args.render_chunk,
        seed=args.seed,
        device=args.device,
        prior_strategy=args.prior_strategy,
        scratch_strategy=args.scratch_strategy,
        methods=args.methods,
        target_psnr=args.target_psnr,
    )


if __name__ == "__main__":
    main()
