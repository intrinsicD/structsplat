"""INIT-009 audit of candidate-index versus progressive WSE survivor ordering."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.common import run_config, write_config, write_csv
from structsplat import sampling as sa


ORDERS = ("candidate_index", "progressive")


def _prefix_metrics(domain: np.ndarray, prefix: np.ndarray) -> tuple[float, float]:
    k = len(prefix)
    if k < 2:
        return 0.0, 0.0
    pair_d2 = ((prefix[:, None, :] - prefix[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(pair_d2, np.inf)
    min_spacing = float(np.sqrt(pair_d2.min())) * np.sqrt(k)

    # Candidate-set covering radius is an exact, shared finite-domain comparison. Normalize by
    # expected spacing so prefix sizes remain commensurate; invert so higher is better.
    nearest_d2 = np.full(len(domain), np.inf)
    for start in range(0, len(domain), 512):
        block = domain[start:start + 512]
        d2 = ((block[:, None, :] - prefix[None, :, :]) ** 2).sum(axis=2)
        nearest_d2[start:start + len(block)] = d2.min(axis=1)
    normalized_hole = float(np.sqrt(nearest_d2.max())) * np.sqrt(k)
    inverse_hole = 1.0 / max(normalized_hole, 1e-12)
    return min_spacing, inverse_hole


def _aggregate(rows: list[dict], timings: list[dict]) -> tuple[dict, dict]:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    by_pair = {}
    for row in rows:
        grouped[(row["prefix_count"], row["order"])].append(row)
        by_pair[(row["seed"], row["prefix_count"], row["order"])] = row

    prefixes = sorted({row["prefix_count"] for row in rows})
    summary = {}
    for prefix in prefixes:
        summary[str(prefix)] = {}
        for order in ORDERS:
            values = grouped[(prefix, order)]
            summary[str(prefix)][order] = {
                "normalized_min_spacing": float(np.mean([
                    row["normalized_min_spacing"] for row in values
                ])),
                "inverse_normalized_coverage_hole": float(np.mean([
                    row["inverse_normalized_coverage_hole"] for row in values
                ])),
            }
        for metric in ("normalized_min_spacing", "inverse_normalized_coverage_hole"):
            summary[str(prefix)]["progressive"][f"delta_{metric}"] = (
                summary[str(prefix)]["progressive"][metric]
                - summary[str(prefix)]["candidate_index"][metric]
            )

    wins = []
    for seed in sorted({row["seed"] for row in rows}):
        for prefix in prefixes:
            legacy = by_pair[(seed, prefix, "candidate_index")]
            progressive = by_pair[(seed, prefix, "progressive")]
            wins.append(
                progressive["normalized_min_spacing"]
                > legacy["normalized_min_spacing"]
                and progressive["inverse_normalized_coverage_hole"]
                > legacy["inverse_normalized_coverage_hole"]
            )
    overhead = float(np.mean([row["order_seconds"] for row in timings])) / max(
        float(np.mean([row["selection_seconds"] for row in timings])), 1e-12
    )
    checks = {
        "terminal_sets_identical": all(row["terminal_sets_identical"] for row in timings),
        "mean_spacing_improves_every_prefix": all(
            summary[str(prefix)]["progressive"]["delta_normalized_min_spacing"] > 0.0
            for prefix in prefixes
        ),
        "mean_coverage_improves_every_prefix": all(
            summary[str(prefix)]["progressive"]
            ["delta_inverse_normalized_coverage_hole"] > 0.0
            for prefix in prefixes
        ),
        "joint_pair_win_fraction_ge_0.75": float(np.mean(wins)) >= 0.75,
        "ordering_overhead_fraction_le_0.25": overhead <= 0.25,
    }
    decision = {
        "accepted": all(checks.values()),
        "checks": checks,
        "joint_pair_win_fraction": float(np.mean(wins)),
        "ordering_overhead_fraction": overhead,
    }
    return summary, decision


def _write_summary(outdir: Path, summary: dict, decision: dict) -> None:
    lines = [
        "# INIT-009 progressive WSE prefix audit",
        "",
        "Higher is better for both normalized metrics. The progressive order is a permutation of",
        "the identical terminal WSE survivor set.",
        "",
        "| Prefix | Legacy spacing | Progressive spacing | Δ spacing | Legacy inv. hole | "
        "Progressive inv. hole | Δ inv. hole |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prefix, group in sorted(summary.items(), key=lambda item: int(item[0])):
        legacy = group["candidate_index"]
        progressive = group["progressive"]
        lines.append(
            f"| {prefix} | {legacy['normalized_min_spacing']:.4f} | "
            f"{progressive['normalized_min_spacing']:.4f} | "
            f"{progressive['delta_normalized_min_spacing']:+.4f} | "
            f"{legacy['inverse_normalized_coverage_hole']:.4f} | "
            f"{progressive['inverse_normalized_coverage_hole']:.4f} | "
            f"{progressive['delta_inverse_normalized_coverage_hole']:+.4f} |"
        )
    lines.extend([
        "",
        f"Joint pair win fraction: {decision['joint_pair_win_fraction']:.1%}.",
        f"Ordering/selection time: {decision['ordering_overhead_fraction']:.1%}.",
        f"Preregistered repair accepted: **{decision['accepted']}**.",
        "",
    ])
    outdir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(*, outdir: str, seeds: tuple[int, ...] = tuple(range(8)),
        candidates: int = 2048, terminal: int = 256,
        prefixes: tuple[int, ...] = (16, 32, 64, 128)) -> list[dict]:
    if candidates < terminal or terminal <= 0:
        raise ValueError("require candidates >= terminal > 0")
    if not prefixes or any(prefix <= 1 or prefix > terminal for prefix in prefixes):
        raise ValueError("prefixes must be in [2, terminal]")
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    write_config(str(output), run_config({
        "protocol": "init009_uniform_prefix_audit",
        "seeds": list(seeds),
        "candidates": candidates,
        "terminal": terminal,
        "prefixes": list(prefixes),
        "metric": "euclidean_uniform",
    }, device="cpu"))

    rows = []
    timings = []
    radius = np.sqrt(1.0 / (np.pi * terminal))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        points = rng.random((candidates, 2))
        radii = np.full(candidates, radius)
        start = time.perf_counter()
        legacy = sa.eliminate(points, terminal, radii)
        selection_seconds = time.perf_counter() - start
        start = time.perf_counter()
        permutation = sa.progressive_order(points[legacy], radii[legacy])
        order_seconds = time.perf_counter() - start
        progressive = legacy[permutation]
        timings.append({
            "seed": seed,
            "selection_seconds": selection_seconds,
            "order_seconds": order_seconds,
            "terminal_sets_identical": bool(np.array_equal(np.sort(progressive), legacy)),
        })
        for prefix in prefixes:
            for name, indices in (
                ("candidate_index", legacy[:prefix]),
                ("progressive", progressive[:prefix]),
            ):
                spacing, inverse_hole = _prefix_metrics(points, points[indices])
                rows.append({
                    "seed": seed,
                    "prefix_count": prefix,
                    "order": name,
                    "normalized_min_spacing": spacing,
                    "inverse_normalized_coverage_hole": inverse_hole,
                })

    summary, decision = _aggregate(rows, timings)
    write_csv(output / "prefix_rows.csv", rows)
    write_csv(output / "timings.csv", timings)
    output.joinpath("aggregate.json").write_text(
        json.dumps({"prefixes": summary, "decision": decision}, indent=2),
        encoding="utf-8",
    )
    _write_summary(output, summary, decision)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/init009_wse_prefix_audit")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--terminal", type=int, default=256)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[16, 32, 64, 128])
    args = parser.parse_args()
    run(
        outdir=args.outdir,
        seeds=tuple(args.seeds),
        candidates=args.candidates,
        terminal=args.terminal,
        prefixes=tuple(args.prefixes),
    )


if __name__ == "__main__":
    main()
