"""FIT-018 shared-start guard for responsibility error-density split sites.

Each image/seed pair is fitted once at N=64, then cloned into four one-shot,
moment-preserving 64->80 split arms. Recovery at 20 and 100 steps is independently replayed
from the same post-split field so the only intervention is parent-site ranking.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

import numpy as np
import torch

from benchmarks.common import load_image, run_config, target_tensor, write_config, write_csv
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import _render, _split_from_residual_with_stats, fit
from structsplat.init import build_field


DEFAULT_IMAGES = (
    "tests/test_images/COCO_train2014_000000000009.jpg",
    "tests/test_images/COCO_train2014_000000000025.jpg",
    "tests/test_images/COCO_train2014_000000000030.jpg",
    "tests/test_images/COCO_train2014_000000000034.jpg",
)
ARMS = (
    "residual",
    "support",
    "responsibility_alpha1",
    "responsibility_alpha0.7",
)
EXISTING_ARMS = ("residual", "support")
DONOR_ARM = "responsibility_alpha0.7"
SOURCE_PATHS = (
    "benchmarks/responsibility_split_compare.py",
    "benchmarks/common.py",
    "src/structsplat/config.py",
    "src/structsplat/fit.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/init.py",
    "src/structsplat/metrics.py",
    "src/structsplat/render.py",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_provenance(root: Path) -> dict[str, object]:
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in SOURCE_PATHS:
        value = _sha256(root / relative)
        files[relative] = value
        combined.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return {"combined_sha256": combined.hexdigest(), "files": files}


def _reproduction_command(
    images: list[str], *, outdir: str, seeds: tuple[int, ...], max_side: int,
    start_count: int, split_count: int, pre_iters: int, device: str, render_chunk: int,
) -> str:
    command = [
        "python", "-m", "benchmarks.responsibility_split_compare",
        "--images", *images,
        "--outdir", outdir,
        "--seeds", *[str(seed) for seed in seeds],
        "--max-side", str(max_side),
        "--start-count", str(start_count),
        "--split-count", str(split_count),
        "--pre-iters", str(pre_iters),
        "--device", device,
        "--render-chunk", str(render_chunk),
    ]
    prefixes = [
        f"{name}={shlex.quote(os.environ[name])}"
        for name in ("LD_PRELOAD", "PYTHONPATH") if os.environ.get(name)
    ]
    return " ".join([*prefixes, *map(shlex.quote, command)])


def _arm_config(arm: str) -> tuple[str, float | None]:
    if arm in EXISTING_ARMS:
        return arm, None
    if arm == "responsibility_alpha1":
        return "responsibility", 1.0
    if arm == DONOR_ARM:
        return "responsibility", 0.7
    raise ValueError(f"unknown FIT-018 arm {arm!r}")


def _jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    if not union:
        return 1.0
    return len(set(left) & set(right)) / len(union)


def _aggregate(rows: list[dict]) -> dict[str, object]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    by_pair = {(r["image"], int(r["seed"]), r["arm"]): r for r in rows}
    for row in rows:
        grouped[row["arm"]].append(row)

    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_rows = grouped.get(arm, [])
        if not arm_rows:
            arms[arm] = {"n": 0.0}
            continue
        arms[arm] = {
            "n": float(len(arm_rows)),
            "immediate_psnr": float(np.mean([r["immediate_psnr"] for r in arm_rows])),
            "post20_psnr": float(np.mean([r["post20_psnr"] for r in arm_rows])),
            "post100_psnr": float(np.mean([r["post100_psnr"] for r in arm_rows])),
            "score_seconds": float(np.mean([r["score_seconds"] for r in arm_rows])),
            "total100_seconds": float(np.mean([r["total100_seconds"] for r in arm_rows])),
            "jaccard_vs_residual": float(np.mean([
                r["parent_jaccard_vs_residual"] for r in arm_rows
            ])),
            "jaccard_vs_support": float(np.mean([
                r["parent_jaccard_vs_support"] for r in arm_rows
            ])),
        }

    stronger = max(
        EXISTING_ARMS,
        key=lambda arm: float(arms.get(arm, {}).get("post20_psnr", -float("inf"))),
    )
    pairs: list[tuple[dict, dict]] = []
    for donor in grouped.get(DONOR_ARM, []):
        baseline = by_pair.get((donor["image"], int(donor["seed"]), stronger))
        if baseline is not None:
            pairs.append((donor, baseline))
    post20 = [donor["post20_psnr"] - base["post20_psnr"] for donor, base in pairs]
    post100 = [donor["post100_psnr"] - base["post100_psnr"] for donor, base in pairs]
    immediate = [
        donor["immediate_psnr"] - base["immediate_psnr"] for donor, base in pairs
    ]
    donor_time = float(np.mean([donor["total100_seconds"] for donor, _ in pairs])) \
        if pairs else float("nan")
    base_time = float(np.mean([base["total100_seconds"] for _, base in pairs])) \
        if pairs else float("nan")
    comparison = {
        "donor_arm": DONOR_ARM,
        "stronger_existing_arm": stronger,
        "n_pairs": len(pairs),
        "delta_immediate_psnr": float(np.mean(immediate)) if immediate else float("nan"),
        "delta_post20_psnr": float(np.mean(post20)) if post20 else float("nan"),
        "delta_post100_psnr": float(np.mean(post100)) if post100 else float("nan"),
        "post20_positive_count": int(sum(delta > 0.0 for delta in post20)),
        "post20_positive_fraction": float(np.mean(np.asarray(post20) > 0.0))
        if post20 else float("nan"),
        "total100_overhead_fraction": donor_time / base_time - 1.0
        if pairs and base_time > 0.0 else float("nan"),
    }
    return {"arms": arms, "comparison": comparison}


def _decision(rows: list[dict], aggregate: dict[str, object]) -> dict[str, object]:
    comparison = aggregate["comparison"]
    assert isinstance(comparison, dict)
    checks = {
        "pair_count_eq_8": comparison["n_pairs"] == 8,
        "post20_gain_ge_0.10_db": comparison["delta_post20_psnr"] >= 0.10,
        "post20_positive_count_ge_6": comparison["post20_positive_count"] >= 6,
        "post100_delta_ge_minus_0.05_db": comparison["delta_post100_psnr"] >= -0.05,
        "total100_overhead_le_0.15": comparison["total100_overhead_fraction"] <= 0.15,
        "all_final_counts_eq_80": bool(rows) and all(r["final_count"] == 80 for r in rows),
        "all_scores_finite": bool(rows) and all(r["scores_finite"] for r in rows),
        "all_renders_finite": bool(rows) and all(r["renders_finite"] for r in rows),
    }
    return {
        "guard_survives": all(checks.values()),
        "stronger_existing_arm": comparison["stronger_existing_arm"],
        "checks": checks,
    }


def _write_summary(outdir: Path, aggregate: dict[str, object],
                   decision: dict[str, object]) -> None:
    arms = aggregate["arms"]
    comparison = aggregate["comparison"]
    assert isinstance(arms, dict) and isinstance(comparison, dict)
    lines = [
        "# FIT-018 responsibility split guard",
        "",
        "Every arm starts from the same fitted field and performs the same moment-preserving",
        "split. Recovery horizons are independently replayed from that post-split field.",
        "",
        "| Site score | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Score s | Total-100 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = arms[arm]
        lines.append(
            f"| `{arm}` | {row.get('immediate_psnr', float('nan')):.4f} | "
            f"{row.get('post20_psnr', float('nan')):.4f} | "
            f"{row.get('post100_psnr', float('nan')):.4f} | "
            f"{row.get('score_seconds', float('nan')):.6f} | "
            f"{row.get('total100_seconds', float('nan')):.6f} |"
        )
    lines.extend([
        "",
        f"Comparator selected before donor deltas: `{comparison['stronger_existing_arm']}` "
        "(higher mean post-20 PSNR of residual/support).",
        "",
        f"Donor deltas: post-20 {comparison['delta_post20_psnr']:+.4f} dB; "
        f"post-100 {comparison['delta_post100_psnr']:+.4f} dB; "
        f"positive post-20 pairs {comparison['post20_positive_count']}/8; "
        f"total-100 overhead {comparison['total100_overhead_fraction']:+.1%}.",
        "",
        f"Preregistered guard survives: **{decision['guard_survives']}**.",
        "",
        "This reused-fixture mechanism smoke does not support default, SOTA, or compression claims.",
        "",
    ])
    outdir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    images: list[str],
    *,
    outdir: str,
    seeds: tuple[int, ...] = (0, 1),
    max_side: int = 64,
    start_count: int = 64,
    split_count: int = 16,
    pre_iters: int = 40,
    device: str = "cpu",
    render_chunk: int = 512,
) -> list[dict]:
    if device != "cpu":
        raise ValueError("FIT-018's preregistered mechanism guard requires device='cpu'")
    if start_count <= 0 or split_count <= 0 or pre_iters <= 0:
        raise ValueError("start_count, split_count, and pre_iters must be positive")
    # The preregistration calls for deterministic CPU replay. Parallel CPU reductions produced
    # sub-millidecibel post-100 drift on this machine, so make that requirement executable rather
    # than treating "cpu" as sufficient provenance.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = [str(Path(path)) for path in images]
    if not paths or any(not Path(path).is_file() for path in paths):
        raise FileNotFoundError("every FIT-018 input must be an existing image")

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    config = run_config({
        "protocol": "fit018_shared_start_responsibility_guard",
        "images": paths,
        "image_sha256": {path: _sha256(path) for path in paths},
        "seeds": list(seeds),
        "max_side": max_side,
        "start_count": start_count,
        "split_count": split_count,
        "final_count": start_count + split_count,
        "pre_iters": pre_iters,
        "recovery_horizons": [20, 100],
        "arms": list(ARMS),
        "responsibility_mass_alphas": [1.0, 0.7],
        "renderer": "normalized",
        "strategy": "quadtree_wse",
        "refine_primitive": "moment_preserving",
        "render_chunk": render_chunk,
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }, device=device)
    config["source"] = _source_provenance(root)
    config["reproduction_command"] = _reproduction_command(
        paths,
        outdir=outdir,
        seeds=seeds,
        max_side=max_side,
        start_count=start_count,
        split_count=split_count,
        pre_iters=pre_iters,
        device=device,
        render_chunk=render_chunk,
    )
    write_config(str(output), config)

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
                log_every=pre_iters,
            )
            pre = fit(initial, target, pre_cfg, verbose=False)
            base = pre["field"].detached()
            before = _render(base, pre_cfg, H, W)
            pair_rows: list[dict] = []

            for arm in ARMS:
                site, alpha = _arm_config(arm)
                grow_cfg = FitConfig(
                    iters=1,
                    renderer="normalized",
                    render_chunk=render_chunk,
                    split_count=split_count,
                    split_mode="duplicate",
                    refine_site=site,
                    refine_primitive="moment_preserving",
                    refine_nms="off",
                    responsibility_mass_alpha=0.7 if alpha is None else alpha,
                    max_gaussians=start_count + split_count,
                )
                start = time.perf_counter()
                grown, added, score_stats = _split_from_residual_with_stats(
                    base.detached(), target, before, grow_cfg
                )
                intervention_seconds = time.perf_counter() - start
                score_seconds = float(score_stats.get("site_score_seconds", float("nan")))
                if added != split_count:
                    raise RuntimeError(f"expected {split_count} additions, got {added}")
                parent_tensor = getattr(grown, "_new_parent_idx", None)
                if parent_tensor is None:
                    raise RuntimeError("split did not expose selected parent indices")
                parent_ids = [int(value) for value in parent_tensor.detach().cpu().tolist()]
                immediate_render = _render(grown, grow_cfg, H, W)
                immediate_psnr = float(M.psnr(immediate_render, target))

                recovered: dict[int, float] = {}
                fit_seconds: dict[int, float] = {}
                recovery_finite = True
                for horizon in (20, 100):
                    torch.manual_seed(seed)
                    recover_cfg = FitConfig(
                        iters=horizon,
                        renderer="normalized",
                        render_chunk=render_chunk,
                        log_every=horizon,
                    )
                    result = fit(grown.detached(), target, recover_cfg, verbose=False)
                    recovered[horizon] = float(result["psnr"])
                    fit_seconds[horizon] = float(result["fit_seconds"])
                    recovery_finite = recovery_finite and bool(
                        torch.isfinite(result["render"]).all().detach().cpu()
                    )
                row = {
                    "image": Path(image_path).stem,
                    "seed": int(seed),
                    "arm": arm,
                    "refine_site": site,
                    "responsibility_mass_alpha": alpha,
                    "start_count": start_count,
                    "final_count": int(grown.n),
                    "pre_psnr": round(float(pre["psnr"]), 6),
                    "immediate_psnr": round(immediate_psnr, 6),
                    "post20_psnr": round(recovered[20], 6),
                    "post100_psnr": round(recovered[100], 6),
                    "score_seconds": round(score_seconds, 6),
                    "intervention_seconds": round(intervention_seconds, 6),
                    "fit20_seconds": round(fit_seconds[20], 6),
                    "fit100_seconds": round(fit_seconds[100], 6),
                    "total100_seconds": round(intervention_seconds + fit_seconds[100], 6),
                    "scores_finite": bool(score_stats.get("site_scores_finite", False)),
                    "renders_finite": bool(torch.isfinite(immediate_render).all().detach().cpu())
                    and recovery_finite,
                    "site_score_mean": score_stats.get("site_score_mean"),
                    "site_score_max": score_stats.get("site_score_max"),
                    "responsibility_mass_mean": score_stats.get("responsibility_mass_mean"),
                    "responsibility_mass_min": score_stats.get("responsibility_mass_min"),
                    "responsibility_mass_max": score_stats.get("responsibility_mass_max"),
                    "responsibility_error_mean": score_stats.get("responsibility_error_mean"),
                    "parent_ids": json.dumps(parent_ids, separators=(",", ":")),
                    "_parent_ids": parent_ids,
                }
                pair_rows.append(row)

            parents = {row["arm"]: row["_parent_ids"] for row in pair_rows}
            for row in pair_rows:
                row["parent_jaccard_vs_residual"] = round(
                    _jaccard(row["_parent_ids"], parents["residual"]), 6
                )
                row["parent_jaccard_vs_support"] = round(
                    _jaccard(row["_parent_ids"], parents["support"]), 6
                )
                row.pop("_parent_ids")
                rows.append(row)

    aggregate = _aggregate(rows)
    decision = _decision(rows, aggregate)
    write_csv(output / "rows.csv", rows)
    output.joinpath("aggregate.json").write_text(
        json.dumps({**aggregate, "decision": decision}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_summary(output, aggregate, decision)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", default=list(DEFAULT_IMAGES))
    parser.add_argument("--outdir", default="results/fit018_responsibility_split_guard")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-side", type=int, default=64)
    parser.add_argument("--start-count", type=int, default=64)
    parser.add_argument("--split-count", type=int, default=16)
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
        split_count=args.split_count,
        pre_iters=args.pre_iters,
        device=args.device,
        render_chunk=args.render_chunk,
    )


if __name__ == "__main__":
    main()
