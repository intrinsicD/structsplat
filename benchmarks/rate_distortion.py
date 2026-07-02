"""COMP-001: rate-distortion sweep — fit once per image x budget, then sweep codec bit mixes.

Each row: (image, budget, bit mix, qat) -> bpp (actual bitstream), PSNR, MS-SSIM. QAT rows
fine-tune a copy of the fitted field through the straight-through-quantized renderer before
encoding. Emits JSON + CSV + markdown alongside the ablation outputs. Torch imported lazily.
"""
from __future__ import annotations
import csv
import json
import os

DEFAULT_BIT_MIXES = ((16, 8, 8, 8), (12, 8, 6, 8), (12, 6, 6, 6), (10, 5, 5, 5))


def run_rd(images, budgets=(2000, 5000), strategy="aniso_flanking", seeds=(0,),
           iters=1500, bit_mixes=DEFAULT_BIT_MIXES, qat_iters=150,
           render_chunk=512, lr_means=None, lr_decay_every=None,
           outdir="results_rd", device=None):
    import torch
    from structsplat.cli import load_image
    from structsplat import init as _init, codec
    from structsplat.config import InitConfig, FitConfig
    from structsplat.fit import fit
    from benchmarks.ablation import _iter_images

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir, exist_ok=True)
    files = _iter_images(images)
    if not files:
        raise SystemExit("no images found")

    rows = []
    for path in files:
        img = load_image(path)
        target = torch.as_tensor(img, device=device)
        name = os.path.splitext(os.path.basename(path))[0]
        for budget in budgets:
            for seed in seeds:
                icfg = InitConfig(strategy=strategy, num_gaussians=budget, seed=seed)
                fkw = {"iters": iters, "render_chunk": render_chunk}
                if lr_means is not None:
                    fkw["lr_means"] = lr_means
                if lr_decay_every is not None:
                    fkw["lr_decay_every"] = lr_decay_every
                fcfg = FitConfig(**fkw)
                field = _init.build_field(img, icfg, device=device)
                out = fit(field, target, fcfg, verbose=False)
                field = out["field"]
                base = {"image": name, "budget": budget, "seed": seed,
                        "fit_psnr": round(out["psnr"], 4)}
                for mix in bit_mixes:
                    ccfg = codec.CodecConfig(bits_means=mix[0], bits_scales=mix[1],
                                             bits_rot=mix[2], bits_colors=mix[3])
                    r = codec.rd_point(field, target, fcfg, ccfg)
                    rows.append({**base, "qat": False, **_row(r)})
                    print(rows[-1])
                    if qat_iters > 0:
                        copy = field.detached()
                        ccfg_q = codec.qat_finetune(copy, target, fcfg, ccfg, iters=qat_iters)
                        rq = codec.rd_point(copy, target, fcfg, ccfg_q)
                        rows.append({**base, "qat": True, **_row(rq)})
                        print(rows[-1])

    _write(rows, outdir)
    return rows


def _row(r):
    return {"bits": "/".join(str(b) for b in r["bits"]), "bpp": round(r["bpp"], 4),
            "raw_bpp": round(r["raw_bpp"], 4), "bytes": r["bytes"],
            "psnr": round(r["psnr"], 4), "ms_ssim": round(r["ms_ssim"], 5)}


def _write(rows, outdir):
    with open(os.path.join(outdir, "rate_distortion.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(os.path.join(outdir, "rate_distortion.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    lines = ["# Rate-distortion (bpp vs quality)\n",
             "| image | budget | bits (mu/s/rot/c) | qat | bpp | PSNR | MS-SSIM |",
             "|---|---:|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['image']} | {r['budget']} | {r['bits']} | {'y' if r['qat'] else 'n'} "
                     f"| {r['bpp']:.3f} | {r['psnr']:.2f} | {r['ms_ssim']:.4f} |")
    with open(os.path.join(outdir, "rate_distortion.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote rate_distortion.json / .csv / .md to {outdir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="StructSplat rate-distortion benchmark (COMP-001)")
    p.add_argument("images", nargs="+")
    p.add_argument("--budgets", type=int, nargs="+", default=[2000, 5000])
    p.add_argument("--strategy", default="aniso_flanking")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--qat-iters", type=int, default=150)
    p.add_argument("--lr-means", type=float, default=None)
    p.add_argument("--lr-decay-every", type=int, default=None)
    p.add_argument("--render-chunk", type=int, default=512)
    p.add_argument("--outdir", default="results_rd")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    run_rd(a.images, budgets=a.budgets, strategy=a.strategy, seeds=a.seeds, iters=a.iters,
           qat_iters=a.qat_iters, lr_means=a.lr_means, lr_decay_every=a.lr_decay_every,
           render_chunk=a.render_chunk, outdir=a.outdir, device=a.device)
