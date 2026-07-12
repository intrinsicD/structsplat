"""FIT-017 shared-start comparison of sampled-add residual scores.

Each image/seed pair is fitted once to the pre-growth count. The resulting field is cloned for
three equal-count interventions: the historical tensor-smoothed absolute score, a same-width
Gaussian magnitude score, and a same-width signed Gaussian score. This isolates signed residual
coherence from both initialization noise and the effect of using a wider smoothing kernel.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmarks.common import load_image, run_config, target_tensor, write_config, write_csv
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import _add_from_residual, _render, fit
from structsplat.init import build_field


DEFAULT_IMAGES = (
    "tests/test_images/COCO_train2014_000000000009.jpg",
    "tests/test_images/COCO_train2014_000000000025.jpg",
    "tests/test_images/COCO_train2014_000000000030.jpg",
    "tests/test_images/COCO_train2014_000000000034.jpg",
)
SCORE_MODES = ("legacy_abs", "gaussian_abs", "signed_gaussian")
BASELINE_MODE = "legacy_abs"


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    by_pair = {(r["image"], r["seed"], r["score_mode"]): r for r in rows}
    for row in rows:
        grouped[row["score_mode"]].append(row)

    summary: dict[str, dict[str, float]] = {}
    for mode, mode_rows in grouped.items():
        out = {
            "n": float(len(mode_rows)),
            "immediate_psnr": float(np.mean([r["immediate_psnr"] for r in mode_rows])),
            "post20_psnr": float(np.mean([r["post20_psnr"] for r in mode_rows])),
            "post100_psnr": float(np.mean([r["post100_psnr"] for r in mode_rows])),
            "score_seconds": float(np.mean([r["score_seconds"] for r in mode_rows])),
            "total100_seconds": float(np.mean([r["total100_seconds"] for r in mode_rows])),
        }
        if mode != BASELINE_MODE:
            pairs = []
            for row in mode_rows:
                base = by_pair[(row["image"], row["seed"], BASELINE_MODE)]
                pairs.append((row, base))
            for metric in ("immediate_psnr", "post20_psnr", "post100_psnr"):
                deltas = [candidate[metric] - base[metric] for candidate, base in pairs]
                out[f"delta_{metric}"] = float(np.mean(deltas))
            post20 = [candidate["post20_psnr"] - base["post20_psnr"]
                      for candidate, base in pairs]
            out["post20_positive_fraction"] = float(np.mean(np.asarray(post20) > 0.0))
            base_time = float(np.mean([base["total100_seconds"] for _, base in pairs]))
            out["total100_overhead_fraction"] = out["total100_seconds"] / base_time - 1.0
        summary[mode] = out
    return summary


def _decision(summary: dict[str, dict[str, float]]) -> dict[str, object]:
    signed = summary["signed_gaussian"]
    checks = {
        "post20_gain_ge_0.10_db": signed["delta_post20_psnr"] >= 0.10,
        "post20_positive_fraction_ge_0.75": signed["post20_positive_fraction"] >= 0.75,
        "post100_delta_ge_minus_0.05_db": signed["delta_post100_psnr"] >= -0.05,
        "total100_overhead_le_0.15": signed["total100_overhead_fraction"] <= 0.15,
    }
    return {"guard_survives": all(checks.values()), "checks": checks}


def _write_summary(outdir: Path, aggregate: dict[str, dict[str, float]],
                   decision: dict[str, object]) -> None:
    lines = [
        "# FIT-017 sampled-add score guard",
        "",
        "All arms share the same fitted pre-growth field for each image/seed and add the same",
        "number of tensor-aligned, target-colored children. Positive deltas favor the score arm.",
        "",
        "| Score | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Δpost-20 | Δpost-100 | "
        "Positive pairs | Total-100 overhead |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in SCORE_MODES:
        row = aggregate[mode]
        lines.append(
            f"| `{mode}` | {row['immediate_psnr']:.4f} | {row['post20_psnr']:.4f} | "
            f"{row['post100_psnr']:.4f} | "
            f"{row.get('delta_post20_psnr', 0.0):+.4f} | "
            f"{row.get('delta_post100_psnr', 0.0):+.4f} | "
            f"{row.get('post20_positive_fraction', 0.0):.1%} | "
            f"{row.get('total100_overhead_fraction', 0.0):+.1%} |"
        )
    lines.extend([
        "",
        f"Preregistered guard survives: **{decision['guard_survives']}**.",
        "",
        "The signed score is a coherent-error heuristic. It is not an exact loss-reduction score",
        "for the normalized anisotropic renderer; promotion requires the larger held-out protocol.",
        "",
    ])
    outdir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(images: list[str], *, outdir: str, seeds: tuple[int, ...] = (0, 1),
        max_side: int = 64, start_count: int = 64, add_count: int = 16,
        pre_iters: int = 40, device: str = "cpu", render_chunk: int = 512) -> list[dict]:
    if start_count <= 0 or add_count <= 0:
        raise ValueError("start_count and add_count must be positive")
    if pre_iters <= 0:
        raise ValueError("pre_iters must be positive")
    paths = [str(Path(path)) for path in images]
    if not paths or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("every FIT-017 input must be an existing image")

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    write_config(str(output), run_config({
        "protocol": "fit017_shared_start_guard",
        "images": paths,
        "seeds": list(seeds),
        "max_side": max_side,
        "start_count": start_count,
        "add_count": add_count,
        "pre_iters": pre_iters,
        "recovery_horizons": [20, 100],
        "score_modes": list(SCORE_MODES),
        "renderer": "normalized",
        "strategy": "quadtree_wse",
        "refine_site": "residual_tensor",
        "refine_primitive": "sampled_add",
        "split_color_init": "target",
        "render_chunk": render_chunk,
    }, device=device))

    rows: list[dict] = []
    scfg = StructureTensorConfig()
    for image_path in paths:
        img = load_image(image_path, max_side=max_side)
        target = target_tensor(img, device)
        H, W = img.shape[:2]
        for seed in seeds:
            torch.manual_seed(seed)
            initial = build_field(
                img,
                InitConfig(strategy="quadtree_wse", num_gaussians=start_count, seed=seed),
                scfg,
                device=device,
            )
            pre_cfg = FitConfig(
                iters=pre_iters,
                renderer="normalized",
                render_chunk=render_chunk,
                log_every=max(1, pre_iters),
            )
            pre = fit(initial, target, pre_cfg, verbose=False)
            base = pre["field"].detached()

            for score_mode in SCORE_MODES:
                grow_cfg = FitConfig(
                    iters=1,
                    renderer="normalized",
                    render_chunk=render_chunk,
                    split_count=add_count,
                    split_mode="residual_tensor_add",
                    sampled_add_score=score_mode,
                    split_color_init="target",
                    max_gaussians=start_count + add_count,
                )
                before = _render(base, grow_cfg, H, W)
                _sync(device)
                start = time.perf_counter()
                grown, added = _add_from_residual(
                    base.detached(), target, before, grow_cfg, tensor_aligned=True
                )
                _sync(device)
                score_seconds = time.perf_counter() - start
                if added != add_count:
                    raise RuntimeError(f"expected {add_count} additions, got {added}")
                immediate = M.psnr(_render(grown, grow_cfg, H, W), target)

                recovered = {}
                fit_seconds = {}
                for horizon in (20, 100):
                    recover_cfg = FitConfig(
                        iters=horizon,
                        renderer="normalized",
                        render_chunk=render_chunk,
                        log_every=horizon,
                    )
                    result = fit(grown.detached(), target, recover_cfg, verbose=False)
                    recovered[horizon] = float(result["psnr"])
                    fit_seconds[horizon] = float(result["fit_seconds"])
                rows.append({
                    "image": Path(image_path).stem,
                    "seed": int(seed),
                    "score_mode": score_mode,
                    "start_count": start_count,
                    "final_count": int(grown.n),
                    "pre_psnr": round(float(pre["psnr"]), 6),
                    "immediate_psnr": round(float(immediate), 6),
                    "post20_psnr": round(recovered[20], 6),
                    "post100_psnr": round(recovered[100], 6),
                    "score_seconds": round(score_seconds, 6),
                    "fit20_seconds": round(fit_seconds[20], 6),
                    "fit100_seconds": round(fit_seconds[100], 6),
                    "total100_seconds": round(score_seconds + fit_seconds[100], 6),
                })

    aggregate = _aggregate(rows)
    decision = _decision(aggregate)
    write_csv(output / "rows.csv", rows)
    output.joinpath("aggregate.json").write_text(
        json.dumps({"scores": aggregate, "decision": decision}, indent=2),
        encoding="utf-8",
    )
    _write_summary(output, aggregate, decision)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", default=list(DEFAULT_IMAGES))
    parser.add_argument("--outdir", default="results/fit017_sampled_add_score_guard")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-side", type=int, default=64)
    parser.add_argument("--start-count", type=int, default=64)
    parser.add_argument("--add-count", type=int, default=16)
    parser.add_argument("--pre-iters", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-chunk", type=int, default=512)
    args = parser.parse_args()
    run(
        args.images,
        outdir=args.outdir,
        seeds=tuple(args.seeds),
        max_side=args.max_side,
        start_count=args.start_count,
        add_count=args.add_count,
        pre_iters=args.pre_iters,
        device=args.device,
        render_chunk=args.render_chunk,
    )


if __name__ == "__main__":
    main()
