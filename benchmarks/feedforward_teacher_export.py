"""FF-001 teacher-field export for feed-forward warm-start training.

Runs the current StructSplat teacher pipeline, saves fitted `GaussianField` NPZ files, and writes a
manifest/config that a future learned predictor trainer can consume. This is intentionally a data
exporter, not a claim that a trained feed-forward model exists.
"""
from __future__ import annotations

from pathlib import Path

from benchmarks.common import load_image, run_config, target_tensor, write_config, write_csv, write_json
from benchmarks.ablation import _iter_images
from structsplat.config import DEFAULT_INIT_STRATEGY


def export_teacher_fields(
    images,
    *,
    outdir="results/feedforward_teacher_export",
    budget=512,
    strategy=DEFAULT_INIT_STRATEGY,
    seed=0,
    iters=80,
    max_side: int | None = 160,
    render_chunk=512,
    device=None,
):
    import torch

    from structsplat import init as _init
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
    from structsplat.fit import fit

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(outdir)
    field_dir = out / "fields"
    field_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_images(images)
    if not files:
        raise SystemExit("no images found")

    cfg = run_config({
        "images": files,
        "budget": budget,
        "strategy": strategy,
        "seed": seed,
        "iters": iters,
        "max_side": max_side,
        "render_chunk": render_chunk,
    }, device=device)
    write_config(str(out), cfg)

    rows = []
    scfg = StructureTensorConfig()
    for path in files:
        img = load_image(path, max_side)
        target = target_tensor(img, device)
        name = Path(path).stem
        icfg = InitConfig(strategy=strategy, num_gaussians=budget, seed=seed)
        fcfg = FitConfig(iters=iters, render_chunk=render_chunk, log_every=max(1, iters // 10))
        field = _init.build_field(img, icfg, scfg, device=device)
        fit_out = fit(field, target, fcfg, verbose=False)
        field_path = field_dir / f"{name}_n{budget}_seed{seed}.npz"
        fit_out["field"].save(str(field_path))
        row = {
            "image": name,
            "source_path": str(path),
            "field_path": str(field_path),
            "budget": int(budget),
            "n_gaussians": int(fit_out["field"].n),
            "strategy": strategy,
            "seed": int(seed),
            "iters": int(iters),
            "psnr": round(float(fit_out["psnr"]), 4),
            "ms_ssim": round(float(fit_out["ms_ssim"]), 5),
            "fit_seconds": round(float(fit_out.get("fit_seconds", 0.0)), 6),
        }
        rows.append(row)
        print(row)

    write_json(out / "teacher_manifest.json", {"config": cfg, "rows": rows})
    write_csv(out / "teacher_manifest.csv", rows)
    lines = [
        "# Feed-Forward Teacher Export",
        "",
        f"Rows: {len(rows)}",
        "",
        "| image | budget | PSNR | MS-SSIM | field |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['image']} | {r['budget']} | {r['psnr']:.4f} | "
            f"{r['ms_ssim']:.5f} | `{r['field_path']}` |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    import argparse

    p = argparse.ArgumentParser(description="Export fitted teacher GaussianFields for FF-001")
    p.add_argument("images", nargs="+")
    p.add_argument("--outdir", default="results/feedforward_teacher_export")
    p.add_argument("--budget", type=int, default=512)
    p.add_argument("--strategy", default=DEFAULT_INIT_STRATEGY)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--max-side", type=int, default=160)
    p.add_argument("--render-chunk", type=int, default=512)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    export_teacher_fields(
        args.images,
        outdir=args.outdir,
        budget=args.budget,
        strategy=args.strategy,
        seed=args.seed,
        iters=args.iters,
        max_side=args.max_side,
        render_chunk=args.render_chunk,
        device=args.device,
    )


if __name__ == "__main__":
    main()
