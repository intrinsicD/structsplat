"""ABL-004 confirmation runner and paired analysis.

This module wraps the existing resumable ``benchmarks.ablation`` harness with the predeclared
confirmation protocol that followed the 8-image screen:

    28 images x seeds {0,1,2} x budgets {2k,5k,10k} x 6 finalist/control arms.

It deliberately keeps measurement rows in the ablation format. The added value here is the
decision-grade bookkeeping around a long run: an expected-cell manifest, resumable shards, missing
cell reporting, paired deltas with bootstrap confidence intervals, and rank-stability tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

import numpy as np

from benchmarks._util import run_config, write_config
from benchmarks.ablation import run_ablation
from benchmarks.common import HEADLINE_TARGET_PSNRS, target_hit_stats, target_label
from benchmarks.common import write_csv, write_json


DEFAULT_KODAK_DIR = Path("results/datasets/abl004/kodak24")
DEFAULT_COCO_DIR = Path("tests/test_images")
DEFAULT_OUTDIR = Path("results/abl004_confirmation")
DEFAULT_STRATEGIES = [
    "aniso_onedge",
    "aniso_flanking",
    "quadtree_wse",
    "quadtree_hybrid",
    "iso_blue_noise",
    "floyd_steinberg",
]
DEFAULT_BUDGETS = [2000, 5000, 10000]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TARGET_PSNRS = [22.0, 24.0, 26.0, 28.0, 30.0, 32.0]
HALVING_DECISIONS = "abl006_elimination_decisions.json"


@dataclass(frozen=True)
class ConfirmationSpec:
    images: tuple[str, ...]
    budgets: tuple[int, ...] = tuple(DEFAULT_BUDGETS)
    seeds: tuple[int, ...] = tuple(DEFAULT_SEEDS)
    strategies: tuple[str, ...] = tuple(DEFAULT_STRATEGIES)
    max_side: int = 768
    iters: int = 1500
    renderer: str = "cuda"
    render_chunk: int = 512
    target_psnr: float | None = 35.0
    target_psnrs: tuple[float, ...] = tuple(DEFAULT_TARGET_PSNRS)
    baseline: str = "aniso_onedge"


def default_confirmation_images(
    kodak_dir: Path = DEFAULT_KODAK_DIR,
    coco_dir: Path = DEFAULT_COCO_DIR,
) -> list[str]:
    """Return the default ABL-004 confirmation image list: Kodak-24 + pinned COCO4."""
    images = sorted(str(p) for p in kodak_dir.glob("*.png"))
    images.extend(sorted(str(p) for p in coco_dir.glob("*.jpg")))
    return images


def make_spec(args: argparse.Namespace) -> ConfirmationSpec:
    images = list(args.images or default_confirmation_images(args.kodak_dir, args.coco_dir))
    if not images:
        raise SystemExit(
            "no confirmation images found; pass --images or run the ABL-004 image prep first"
        )
    unknown = sorted(set(args.strategies) - set(DEFAULT_STRATEGIES))
    if unknown:
        valid = ", ".join(DEFAULT_STRATEGIES)
        raise SystemExit(f"unknown confirmation strategy {unknown}; expected one of: {valid}")
    if args.baseline not in args.strategies:
        raise SystemExit("--baseline must be one of --strategies")
    return ConfirmationSpec(
        images=tuple(images),
        budgets=tuple(int(v) for v in args.budgets),
        seeds=tuple(int(v) for v in args.seeds),
        strategies=tuple(args.strategies),
        max_side=int(args.max_side),
        iters=int(args.iters),
        renderer=args.renderer,
        render_chunk=int(args.render_chunk),
        target_psnr=args.target_psnr,
        target_psnrs=tuple(float(v) for v in args.target_psnrs),
        baseline=args.baseline,
    )


def expected_cells(spec: ConfirmationSpec) -> list[dict[str, Any]]:
    cells = []
    for source_path in spec.images:
        image = Path(source_path).stem
        for budget in spec.budgets:
            for seed in spec.seeds:
                for strategy in spec.strategies:
                    cells.append({
                        "image": image,
                        "source_path": source_path,
                        "budget": int(budget),
                        "seed": int(seed),
                        "strategy": strategy,
                        "max_side": int(spec.max_side),
                        "iters": int(spec.iters),
                        "renderer": spec.renderer,
                    })
    return cells


def write_plan(spec: ConfirmationSpec, outdir: Path) -> list[dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    cells = expected_cells(spec)
    write_config(
        str(outdir),
        run_config({
            "protocol": "ABL-004 confirmation",
            "images": list(spec.images),
            "budgets": list(spec.budgets),
            "seeds": list(spec.seeds),
            "strategies": list(spec.strategies),
            "expected_cells": len(cells),
            "max_side": spec.max_side,
            "iters": spec.iters,
            "renderer": spec.renderer,
            "render_chunk": spec.render_chunk,
            "target_psnr": spec.target_psnr,
            "target_psnrs": list(spec.target_psnrs),
            "baseline": spec.baseline,
        }),
        name="confirmation_config.json",
    )
    write_csv(outdir / "confirmation_plan.csv", cells)
    return cells


def default_halving_decisions(spec: ConfirmationSpec) -> dict[str, Any]:
    return {
        "protocol": "ABL-006 successive halving",
        "version": 1,
        "rule": {
            "metric": "psnr",
            "ci_method": "bootstrap_mean_ci",
            "confidence": 0.95,
            "pairing_key": ["image", "seed", "budget"],
            "leader": "highest mean PSNR among complete paired units in the stage",
            "survivor": "CI for arm-minus-leader overlaps zero",
            "elimination": "CI for arm-minus-leader is strictly below zero",
            "frozen_before_stage1_complete": True,
        },
        "stages": {
            "stage1": {
                "budget": int(spec.budgets[0]),
                "seeds": [int(s) for s in spec.seeds[:2]],
                "survivors": None,
                "eliminated": [],
            },
            "stage2": {
                "budget": int(spec.budgets[1]) if len(spec.budgets) > 1 else None,
                "seeds": [int(s) for s in spec.seeds[:2]],
                "survivors": None,
                "eliminated": [],
            },
            "stage3": {
                "budget": int(spec.budgets[2]) if len(spec.budgets) > 2 else None,
                "seeds": [int(s) for s in spec.seeds[:2]],
                "extra_seed": int(spec.seeds[2]) if len(spec.seeds) > 2 else None,
                "finalists": None,
                "eliminated": [],
            },
        },
    }


def _halving_decisions_path(outdir: Path, path: Path | None) -> Path:
    return path if path is not None else outdir / HALVING_DECISIONS


def load_halving_decisions(path: Path, spec: ConfirmationSpec) -> dict[str, Any]:
    if not path.exists():
        decisions = default_halving_decisions(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
        return decisions
    return json.loads(path.read_text(encoding="utf-8"))


def _survivors(decisions: dict[str, Any], stage: str) -> list[str] | None:
    raw = (decisions.get("stages", {}).get(stage, {}) or {}).get("survivors")
    if raw is None:
        raw = (decisions.get("stages", {}).get(stage, {}) or {}).get("finalists")
    if raw is None:
        return None
    return [str(v) for v in raw]


def _validate_halving_decisions(spec: ConfirmationSpec, decisions: dict[str, Any]) -> None:
    allowed = set(spec.strategies)
    for stage in ("stage1", "stage2", "stage3"):
        selected = _survivors(decisions, stage)
        if selected is None:
            continue
        unknown = sorted(set(selected) - allowed)
        if unknown:
            raise ValueError(
                f"{stage} contains unknown survivor/finalist arm(s) {unknown}; "
                f"expected one of {sorted(allowed)}"
            )


def _validate_halving_spec(spec: ConfirmationSpec) -> None:
    if len(spec.budgets) < 3:
        raise ValueError("ABL-006 requires at least three budgets")
    if len(spec.seeds) < 3:
        raise ValueError("ABL-006 requires at least three seeds")


def successive_halving_cells(
    spec: ConfirmationSpec,
    decisions: dict[str, Any],
) -> list[dict[str, Any]]:
    _validate_halving_spec(spec)
    _validate_halving_decisions(spec, decisions)
    cells: list[dict[str, Any]] = []

    def add(stage: int, run_group: str, budgets: Iterable[int],
            seeds: Iterable[int], strategies: Iterable[str]) -> None:
        for source_path in spec.images:
            image = Path(source_path).stem
            for budget in budgets:
                for seed in seeds:
                    for strategy in strategies:
                        cells.append({
                            "stage": stage,
                            "run_group": run_group,
                            "image": image,
                            "source_path": source_path,
                            "budget": int(budget),
                            "seed": int(seed),
                            "strategy": strategy,
                            "max_side": int(spec.max_side),
                            "iters": int(spec.iters),
                            "renderer": spec.renderer,
                        })

    stage1_seeds = [int(s) for s in spec.seeds[:2]]
    add(
        1,
        f"stage1_budget{int(spec.budgets[0])}",
        [int(spec.budgets[0])],
        stage1_seeds,
        spec.strategies,
    )

    stage1_survivors = _survivors(decisions, "stage1")
    if not stage1_survivors:
        return cells
    add(
        2,
        f"stage2_budget{int(spec.budgets[1])}",
        [int(spec.budgets[1])],
        stage1_seeds,
        stage1_survivors,
    )

    stage2_survivors = _survivors(decisions, "stage2")
    if not stage2_survivors:
        return cells
    add(
        3,
        f"stage3_budget{int(spec.budgets[2])}",
        [int(spec.budgets[2])],
        stage1_seeds,
        stage2_survivors,
    )
    add(3, "stage3_seed2_all_budgets", spec.budgets, [int(spec.seeds[2])], stage2_survivors)
    return cells


def halving_run_groups(
    spec: ConfirmationSpec,
    decisions: dict[str, Any],
) -> list[dict[str, Any]]:
    cells = successive_halving_cells(spec, decisions)
    groups: dict[str, dict[str, Any]] = {}
    for cell in cells:
        group = groups.setdefault(cell["run_group"], {
            "run_group": cell["run_group"],
            "stage": cell["stage"],
            "budgets": set(),
            "seeds": set(),
            "strategies": set(),
        })
        group["budgets"].add(int(cell["budget"]))
        group["seeds"].add(int(cell["seed"]))
        group["strategies"].add(str(cell["strategy"]))
    out = []
    for group in sorted(groups.values(), key=lambda g: (g["stage"], g["run_group"])):
        out.append({
            "run_group": group["run_group"],
            "stage": int(group["stage"]),
            "budgets": sorted(group["budgets"]),
            "seeds": sorted(group["seeds"]),
            "strategies": sorted(group["strategies"]),
        })
    return out


def _eliminated_rows(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage, entry in sorted((decisions.get("stages") or {}).items()):
        for item in entry.get("eliminated") or []:
            rec = {"stage": stage}
            if isinstance(item, dict):
                rec.update(item)
            else:
                rec["strategy"] = str(item)
            rows.append(rec)
    return rows


def write_halving_plan(
    spec: ConfirmationSpec,
    outdir: Path,
    *,
    decisions_path: Path | None = None,
) -> list[dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    path = _halving_decisions_path(outdir, decisions_path)
    decisions = load_halving_decisions(path, spec)
    cells = successive_halving_cells(spec, decisions)
    groups = halving_run_groups(spec, decisions)
    write_config(
        str(outdir),
        run_config({
            "protocol": "ABL-006 successive halving",
            "images": list(spec.images),
            "budgets": list(spec.budgets),
            "seeds": list(spec.seeds),
            "strategies": list(spec.strategies),
            "planned_cells": len(cells),
            "run_groups": groups,
            "decisions_path": str(path),
            "rule": decisions.get("rule", {}),
            "max_side": spec.max_side,
            "iters": spec.iters,
            "renderer": spec.renderer,
            "render_chunk": spec.render_chunk,
            "target_psnr": spec.target_psnr,
            "target_psnrs": list(spec.target_psnrs),
            "baseline": spec.baseline,
        }),
        name="abl006_config.json",
    )
    write_csv(outdir / "abl006_plan.csv", cells)
    elim = _eliminated_rows(decisions)
    _write_rows(
        outdir / "abl006_elimination_trail.csv",
        elim,
        ["stage", "strategy", "reason", "mean_delta_psnr", "ci95_low_psnr", "ci95_high_psnr"],
    )
    write_json(outdir / "abl006_run_groups.json", groups)
    return cells


def load_ablation_rows(outdir: Path) -> list[dict[str, Any]]:
    jsonl = outdir / "ablation.jsonl"
    if jsonl.exists():
        rows = []
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    json_path = outdir / "ablation.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return []


def _row_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (str(row["image"]), int(row["budget"]), int(row["seed"]), str(row["strategy"]))


def _cell_key(cell: dict[str, Any]) -> tuple[str, int, int, str]:
    return (str(cell["image"]), int(cell["budget"]), int(cell["seed"]), str(cell["strategy"]))


def _missing_from_plan(
    rows_by_key: dict[tuple, dict[str, Any]],
    cells: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    completed = set(rows_by_key)
    return [cell for cell in cells if _cell_key(cell) not in completed]


def _expected_runs_by_budget_strategy(
    cells: Iterable[dict[str, Any]],
) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    seen: set[tuple[str, int, int, str]] = set()
    for cell in cells:
        key = _cell_key(cell)
        if key in seen:
            continue
        seen.add(key)
        budget = int(cell["budget"])
        strategy = str(cell["strategy"])
        counts[(budget, strategy)] = counts.get((budget, strategy), 0) + 1
    return counts


def _dedupe_rows(rows: Iterable[dict[str, Any]], spec: ConfirmationSpec) -> dict[tuple, dict[str, Any]]:
    allowed = set(spec.strategies)
    out = {}
    for row in rows:
        if row.get("strategy") not in allowed:
            continue
        if int(row.get("budget", -1)) not in spec.budgets:
            continue
        if int(row.get("seed", -1)) not in spec.seeds:
            continue
        if int(row.get("iters", spec.iters)) != spec.iters:
            continue
        if row.get("renderer", spec.renderer) != spec.renderer:
            continue
        out[_row_key(row)] = row
    return out


def _mean_or_none(vals: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    return mean(clean) if clean else None


def _std_or_none(vals: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not clean:
        return None
    return pstdev(clean) if len(clean) > 1 else 0.0


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(
    values: Iterable[float],
    samples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None, None, None
    center = float(arr.mean())
    if arr.size == 1 or samples <= 0:
        return center, float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(samples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return center, float(lo), float(hi)


def _iters_to_target(row: dict[str, Any], target: float | None) -> int | None:
    if target is None:
        return row.get("iters_to_target")
    targets = row.get("iters_to_targets", {})
    if isinstance(targets, dict):
        return targets.get(str(float(target))) or targets.get(str(target))
    return row.get("iters_to_target")


def _headline_targets(spec: ConfirmationSpec) -> list[float]:
    spec_targets = {float(t) for t in spec.target_psnrs}
    selected = [float(t) for t in HEADLINE_TARGET_PSNRS if not spec_targets or float(t) in spec_targets]
    return selected or list(HEADLINE_TARGET_PSNRS)


def _target_field(target: float, suffix: str) -> str:
    return f"target_{target_label(target).replace('.', '_')}_{suffix}"


def _target_fieldnames(targets: list[float]) -> list[str]:
    out: list[str] = []
    for target in targets:
        out.append(_target_field(target, "reached"))
        out.append(_target_field(target, "mean_iter"))
    return out


def leaderboard(
    rows_by_key: dict[tuple, dict[str, Any]],
    spec: ConfirmationSpec,
    *,
    expected_runs: dict[tuple[int, str], int] | None = None,
) -> list[dict[str, Any]]:
    rows = list(rows_by_key.values())
    targets = _headline_targets(spec)
    out = []
    for budget in spec.budgets:
        scored = []
        for strategy in spec.strategies:
            vals = [r for r in rows if int(r["budget"]) == budget and r["strategy"] == strategy]
            total_seconds = [
                float(r.get("init_seconds", 0.0)) + float(r.get("fit_seconds", 0.0) or 0.0)
                for r in vals
            ]
            rec = {
                "budget": int(budget),
                "strategy": strategy,
                "runs": len(vals),
                "expected_runs": (
                    expected_runs.get((int(budget), strategy), 0)
                    if expected_runs is not None
                    else len(spec.images) * len(spec.seeds)
                ),
                "mean_psnr": _mean_or_none(r.get("psnr") for r in vals),
                "std_psnr": _std_or_none(r.get("psnr") for r in vals),
                "mean_ms_ssim": _mean_or_none(r.get("ms_ssim") for r in vals),
                "std_ms_ssim": _std_or_none(r.get("ms_ssim") for r in vals),
                "mean_total_seconds": _mean_or_none(total_seconds),
                "reached_target": sum(
                    1 for r in vals if _iters_to_target(r, spec.target_psnr) is not None
                ),
            }
            for target in targets:
                stats = target_hit_stats(vals, target)
                rec[_target_field(target, "reached")] = stats["hits"]
                rec[_target_field(target, "mean_iter")] = stats["mean_iter"]
            scored.append(rec)
        ranked = sorted(
            scored,
            key=lambda r: (-math.inf if r["mean_psnr"] is None else r["mean_psnr"]),
            reverse=True,
        )
        for rank, rec in enumerate(ranked, 1):
            rec["rank_psnr"] = rank
            out.append(rec)
    return out


def paired_deltas(
    rows_by_key: dict[tuple, dict[str, Any]],
    spec: ConfirmationSpec,
    *,
    baseline: str | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows_by_key.values())
    by_unit = {
        (r["image"], int(r["budget"]), int(r["seed"]), r["strategy"]): r
        for r in rows
    }
    pairwise = []
    unit_rows = []
    pairs = []
    if baseline is None:
        for idx, left in enumerate(spec.strategies):
            for right in spec.strategies[idx + 1:]:
                pairs.append((left, right))
    else:
        pairs = [(strategy, baseline) for strategy in spec.strategies if strategy != baseline]

    for budget in spec.budgets:
        for left, right in pairs:
            deltas_psnr = []
            deltas_ms = []
            for image in [Path(p).stem for p in spec.images]:
                for seed in spec.seeds:
                    a = by_unit.get((image, int(budget), int(seed), left))
                    b = by_unit.get((image, int(budget), int(seed), right))
                    if a is None or b is None:
                        continue
                    d_psnr = float(a["psnr"]) - float(b["psnr"])
                    d_ms = float(a["ms_ssim"]) - float(b["ms_ssim"])
                    deltas_psnr.append(d_psnr)
                    deltas_ms.append(d_ms)
                    unit_rows.append({
                        "budget": int(budget),
                        "image": image,
                        "seed": int(seed),
                        "left": left,
                        "right": right,
                        "delta_psnr": d_psnr,
                        "delta_ms_ssim": d_ms,
                        "left_psnr": float(a["psnr"]),
                        "right_psnr": float(b["psnr"]),
                        "left_ms_ssim": float(a["ms_ssim"]),
                        "right_ms_ssim": float(b["ms_ssim"]),
                    })
            mean_delta, ci_low, ci_high = bootstrap_mean_ci(
                deltas_psnr,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            mean_ms, ms_low, ms_high = bootstrap_mean_ci(
                deltas_ms,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            pairwise.append({
                "budget": int(budget),
                "left": left,
                "right": right,
                "paired_units": len(deltas_psnr),
                "mean_delta_psnr": mean_delta,
                "ci95_low_psnr": ci_low,
                "ci95_high_psnr": ci_high,
                "wins_psnr": sum(1 for v in deltas_psnr if v > 0.0),
                "losses_psnr": sum(1 for v in deltas_psnr if v < 0.0),
                "ties_psnr": sum(1 for v in deltas_psnr if v == 0.0),
                "mean_delta_ms_ssim": mean_ms,
                "ci95_low_ms_ssim": ms_low,
                "ci95_high_ms_ssim": ms_high,
                "wins_ms_ssim": sum(1 for v in deltas_ms if v > 0.0),
                "losses_ms_ssim": sum(1 for v in deltas_ms if v < 0.0),
                "ties_ms_ssim": sum(1 for v in deltas_ms if v == 0.0),
            })
    return pairwise, unit_rows


def rank_stability(
    rows_by_key: dict[tuple, dict[str, Any]],
    spec: ConfirmationSpec,
    *,
    expected_runs: dict[tuple[int, str], int] | None = None,
) -> list[dict[str, Any]]:
    rows = list(rows_by_key.values())
    by_group: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault((row["image"], int(row["budget"]), int(row["seed"])), []).append(row)
    rank_values: dict[tuple[int, str], list[int]] = {}
    wins: dict[tuple[int, str], int] = {}
    complete_groups: dict[int, int] = {int(b): 0 for b in spec.budgets}
    expected_by_budget: dict[int, set[str]] = {}
    for budget in spec.budgets:
        budget_int = int(budget)
        if expected_runs is None:
            expected_by_budget[budget_int] = set(spec.strategies)
        else:
            expected_by_budget[budget_int] = {
                strategy
                for strategy in spec.strategies
                if expected_runs.get((budget_int, strategy), 0) > 0
            }
    for (_image, budget, _seed), vals in by_group.items():
        expected_strategies = expected_by_budget.get(int(budget), set(spec.strategies))
        if not expected_strategies:
            continue
        vals = [v for v in vals if v["strategy"] in expected_strategies]
        if {v["strategy"] for v in vals} != expected_strategies:
            continue
        complete_groups[budget] += 1
        ranked = sorted(vals, key=lambda r: float(r["psnr"]), reverse=True)
        for rank, row in enumerate(ranked, 1):
            key = (budget, row["strategy"])
            rank_values.setdefault(key, []).append(rank)
            if rank == 1:
                wins[key] = wins.get(key, 0) + 1

    out = []
    for budget in spec.budgets:
        for strategy in spec.strategies:
            vals = rank_values.get((int(budget), strategy), [])
            out.append({
                "budget": int(budget),
                "strategy": strategy,
                "complete_image_seed_groups": complete_groups[int(budget)],
                "wins": wins.get((int(budget), strategy), 0),
                "mean_rank": _mean_or_none(vals),
                "median_rank": (median(vals) if vals else None),
                "rank_std": _std_or_none(vals),
            })
    return out


def missing_cells(rows_by_key: dict[tuple, dict[str, Any]], spec: ConfirmationSpec) -> list[dict[str, Any]]:
    completed = set(rows_by_key)
    return [cell for cell in expected_cells(spec) if _cell_key(cell) not in completed]


def _write_summary(
    outdir: Path,
    spec: ConfirmationSpec,
    *,
    title: str = "ABL-004 Confirmation Analysis",
    completed: int,
    expected: int,
    missing: list[dict[str, Any]],
    board: list[dict[str, Any]],
    baseline_pairs: list[dict[str, Any]],
    ranks: list[dict[str, Any]],
    plan_file: str = "confirmation_plan.csv",
    extra_files: list[str] | None = None,
) -> None:
    targets = _headline_targets(spec)
    target_header = "".join(
        f" | Hit {target_label(t)} | Iter {target_label(t)}"
        for t in targets
    )
    target_align = "|---:" * (2 * len(targets))
    lines = [
        f"# {title}",
        "",
        f"Expected cells: {expected}",
        f"Completed cells: {completed}",
        f"Missing cells: {len(missing)}",
        f"Baseline for paired deltas: `{spec.baseline}`",
        "",
        "## Leaderboard",
        "",
        "| Budget | Rank | Strategy | Runs | PSNR | PSNR std | MS-SSIM"
        f"{target_header} | Mean total s |",
        "|---:|---:|---|---:|---:|---:|---:"
        f"{target_align}|---:|",
    ]
    for row in sorted(board, key=lambda r: (r["budget"], r["rank_psnr"])):
        target_cells = []
        for target in targets:
            target_cells.append(f"{row[_target_field(target, 'reached')]}/{row['runs']}")
            target_cells.append(_fmt(row[_target_field(target, "mean_iter")], 1))
        lines.append(
            f"| {row['budget']} | {row['rank_psnr']} | {row['strategy']} | "
            f"{row['runs']}/{row['expected_runs']} | {_fmt(row['mean_psnr'])} | "
            f"{_fmt(row['std_psnr'])} | {_fmt(row['mean_ms_ssim'], 5)} | "
            f"{' | '.join(target_cells)} | {_fmt(row['mean_total_seconds'], 2)} |"
        )

    lines += [
        "",
        "## Paired Deltas Vs Baseline",
        "",
        "Positive means `left - baseline` is better. Confidence intervals bootstrap image x seed units.",
        "",
        "| Budget | Left | Right | Pairs | PSNR delta | 95% CI | PSNR wins | MS-SSIM delta | 95% CI |",
        "|---:|---|---|---:|---:|---|---:|---:|---|",
    ]
    for row in baseline_pairs:
        lines.append(
            f"| {row['budget']} | {row['left']} | {row['right']} | {row['paired_units']} | "
            f"{_fmt(row['mean_delta_psnr'])} | "
            f"[{_fmt(row['ci95_low_psnr'])}, {_fmt(row['ci95_high_psnr'])}] | "
            f"{row['wins_psnr']}/{row['paired_units']} | "
            f"{_fmt(row['mean_delta_ms_ssim'], 5)} | "
            f"[{_fmt(row['ci95_low_ms_ssim'], 5)}, {_fmt(row['ci95_high_ms_ssim'], 5)}] |"
        )

    lines += [
        "",
        "## Rank Stability",
        "",
        "| Budget | Strategy | Complete groups | Wins | Mean rank | Median rank | Rank std |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(ranks, key=lambda r: (r["budget"], r["mean_rank"] or 999, r["strategy"])):
        lines.append(
            f"| {row['budget']} | {row['strategy']} | {row['complete_image_seed_groups']} | "
            f"{row['wins']} | {_fmt(row['mean_rank'], 3)} | {_fmt(row['median_rank'], 3)} | "
            f"{_fmt(row['rank_std'], 3)} |"
        )

    lines += [
        "",
        "## Files",
        "",
        f"- `{plan_file}`: expected cells.",
        "- `missing_cells.csv`: cells not yet present in `ablation.jsonl` / `ablation.json`.",
        "- `leaderboard.csv`: aggregate budget x strategy means.",
        "- `paired_deltas_vs_baseline.csv`: strategy minus baseline paired deltas.",
        "- `pairwise_deltas.csv`: all pairwise strategy deltas.",
        "- `paired_units_vs_baseline.csv`: per image x seed paired rows for failure inspection.",
        "- `rank_stability.csv`: per-budget winner counts and rank distribution.",
    ]
    for item in extra_files or []:
        lines.append(item)
    (outdir / "confirmation_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_index(
    outdir: Path,
    *,
    title: str = "ABL-004 Confirmation Overview",
    note: str | None = None,
    config_file: str = "confirmation_config.json",
    plan_file: str = "confirmation_plan.csv",
    extra_links: list[tuple[str, str]] | None = None,
) -> None:
    note = note or (
        "Decision-grade confirmation artifacts. Results may be partial while "
        "bounded shards are still running; see missing_cells.csv for remaining cells. "
        "This overview embeds scalar plots only because the ablation confirmation harness does "
        "not persist per-cell reconstruction images."
    )
    links = [
        ("Confirmation analysis", "confirmation_analysis.md"),
        ("Ablation summary", "summary.md"),
        ("Confirmation config", config_file),
        ("Confirmation plan", plan_file),
        ("Missing cells", "missing_cells.csv"),
        ("Leaderboard", "leaderboard.csv"),
        ("Paired deltas vs baseline", "paired_deltas_vs_baseline.csv"),
        ("All pairwise deltas", "pairwise_deltas.csv"),
        ("Paired units vs baseline", "paired_units_vs_baseline.csv"),
        ("Rank stability", "rank_stability.csv"),
        ("Raw JSONL rows", "ablation.jsonl"),
    ]
    links.extend(extra_links or [])
    plots = ["psnr_vs_budget.png", "psnr_vs_iters.png"]
    html = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;line-height:1.45;color:#1f2328;background:#fff}",
        "h1{margin-bottom:0.25rem} .note{color:#57606a;max-width:80ch}",
        "ul{padding-left:1.25rem} a{color:#0969da;text-decoration:none} a:hover{text-decoration:underline}",
        ".plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;margin-top:18px}",
        "figure{margin:0} img{max-width:100%;border:1px solid #d0d7de;border-radius:6px;background:#fff}",
        "figcaption{font-size:0.9rem;color:#57606a;margin-top:4px}",
        "</style></head><body>",
        f"<h1>{escape(title)}</h1>",
        f'<p class="note">{escape(note)}</p>',
        "<h2>Files</h2>",
        "<ul>",
    ]
    for label, rel in links:
        if (outdir / rel).exists():
            html.append(f'<li><a href="{escape(rel)}">{escape(label)}</a></li>')
    html += ["</ul>", "<h2>Plots</h2>", '<div class="plots">']
    for rel in plots:
        if (outdir / rel).exists():
            html.append(
                f'<figure><a href="{escape(rel)}"><img src="{escape(rel)}" alt="{escape(rel)}"></a>'
                f"<figcaption>{escape(rel)}</figcaption></figure>"
            )
    html += ["</div>", "</body></html>"]
    (outdir / "index.html").write_text("\n".join(html) + "\n", encoding="utf-8")


def write_confirmation_analysis(
    outdir: Path,
    spec: ConfirmationSpec,
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    plan = write_plan(spec, outdir)
    return _write_analysis(
        outdir,
        spec,
        plan,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def _write_analysis(
    outdir: Path,
    spec: ConfirmationSpec,
    plan: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
    analysis_title: str = "ABL-004 Confirmation Analysis",
    index_title: str = "ABL-004 Confirmation Overview",
    index_note: str | None = None,
    config_file: str = "confirmation_config.json",
    plan_file: str = "confirmation_plan.csv",
    extra_links: list[tuple[str, str]] | None = None,
    extra_summary_files: list[str] | None = None,
) -> dict[str, Any]:
    rows_by_key = _dedupe_rows(load_ablation_rows(outdir), spec)
    missing = _missing_from_plan(rows_by_key, plan)
    expected_runs = _expected_runs_by_budget_strategy(plan)
    board = leaderboard(
        rows_by_key,
        spec,
        expected_runs=expected_runs,
    )
    baseline_pairs, baseline_units = paired_deltas(
        rows_by_key,
        spec,
        baseline=spec.baseline,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    pairwise, _ = paired_deltas(
        rows_by_key,
        spec,
        baseline=None,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    ranks = rank_stability(rows_by_key, spec, expected_runs=expected_runs)
    target_fields = _target_fieldnames(_headline_targets(spec))

    cell_fields = list(plan[0].keys()) if plan else []
    _write_rows(outdir / "missing_cells.csv", missing, cell_fields)
    _write_rows(outdir / "leaderboard.csv", board, [
        "budget", "strategy", "runs", "expected_runs", "mean_psnr", "std_psnr",
        "mean_ms_ssim", "std_ms_ssim", *target_fields, "mean_total_seconds",
        "reached_target", "rank_psnr",
    ])
    pair_fields = [
        "budget", "left", "right", "paired_units", "mean_delta_psnr", "ci95_low_psnr",
        "ci95_high_psnr", "wins_psnr", "losses_psnr", "ties_psnr", "mean_delta_ms_ssim",
        "ci95_low_ms_ssim", "ci95_high_ms_ssim", "wins_ms_ssim", "losses_ms_ssim",
        "ties_ms_ssim",
    ]
    _write_rows(outdir / "paired_deltas_vs_baseline.csv", baseline_pairs, pair_fields)
    _write_rows(outdir / "pairwise_deltas.csv", pairwise, pair_fields)
    _write_rows(
        outdir / "paired_units_vs_baseline.csv",
        sorted(
            baseline_units,
            key=lambda r: (r["budget"], r["left"], r["delta_psnr"], r["image"], r["seed"]),
        ),
        [
            "budget", "image", "seed", "left", "right", "delta_psnr", "delta_ms_ssim",
            "left_psnr", "right_psnr", "left_ms_ssim", "right_ms_ssim",
        ],
    )
    _write_rows(outdir / "rank_stability.csv", ranks, [
        "budget", "strategy", "complete_image_seed_groups", "wins", "mean_rank",
        "median_rank", "rank_std",
    ])
    summary = {
        "expected_cells": len(plan),
        "completed_cells": len(rows_by_key),
        "missing_cells": len(missing),
        "baseline": spec.baseline,
        "budgets": list(spec.budgets),
        "seeds": list(spec.seeds),
        "strategies": list(spec.strategies),
    }
    write_json(outdir / "confirmation_analysis.json", summary)
    _write_summary(
        outdir,
        spec,
        title=analysis_title,
        completed=len(rows_by_key),
        expected=len(plan),
        missing=missing,
        board=board,
        baseline_pairs=baseline_pairs,
        ranks=ranks,
        plan_file=plan_file,
        extra_files=extra_summary_files,
    )
    _write_index(
        outdir,
        title=index_title,
        note=index_note,
        config_file=config_file,
        plan_file=plan_file,
        extra_links=extra_links,
    )
    return summary


def write_halving_analysis(
    outdir: Path,
    spec: ConfirmationSpec,
    *,
    decisions_path: Path | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    plan = write_halving_plan(spec, outdir, decisions_path=decisions_path)
    return _write_analysis(
        outdir,
        spec,
        plan,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        analysis_title="ABL-006 Successive-Halving Analysis",
        index_title="ABL-006 Successive-Halving Overview",
        index_note=(
            "Successive-halving confirmation artifacts. Missing cells are measured against the "
            "staged plan only; dropped arms are listed in the elimination trail instead of being "
            "counted as unrun high-budget cells."
        ),
        config_file="abl006_config.json",
        plan_file="abl006_plan.csv",
        extra_links=[
            ("ABL-006 decisions", HALVING_DECISIONS),
            ("ABL-006 run groups", "abl006_run_groups.json"),
            ("ABL-006 elimination trail", "abl006_elimination_trail.csv"),
        ],
        extra_summary_files=[
            "- `abl006_elimination_decisions.json`: frozen rule and stage survivor decisions.",
            "- `abl006_elimination_trail.csv`: eliminated arms and decision statistics.",
            "- `abl006_run_groups.json`: cartesian shards executed by `halving-run`.",
        ],
    )


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    spec = make_spec(args)
    outdir = args.outdir
    write_plan(spec, outdir)
    run_ablation(
        list(spec.images),
        budgets=list(spec.budgets),
        strategies=list(spec.strategies),
        seeds=list(spec.seeds),
        iters=spec.iters,
        target_psnr=spec.target_psnr,
        target_psnrs=list(spec.target_psnrs),
        flank_offsets=[0.5],
        flat_fracs=[0.02],
        corner_fracs=[0.15],
        max_axis_ratios=[6.0],
        coherence_powers=[1.0],
        render_chunk=spec.render_chunk,
        renderer=spec.renderer,
        pixel_loss=args.pixel_loss,
        ssim_weight=args.ssim_weight,
        compute_lpips=args.lpips,
        early_exit=args.early_exit,
        early_exit_window=args.early_exit_window,
        early_exit_min_delta=args.early_exit_min_delta,
        early_exit_min_iters=args.early_exit_min_iters,
        max_side=spec.max_side,
        resume=args.resume,
        max_new_cells=args.max_new_cells,
        outdir=str(outdir),
        device=args.device,
        write_plots=not args.no_plots,
    )
    if args.no_analyze:
        return {"analyzed": False}
    return write_confirmation_analysis(
        outdir,
        spec,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


def _completed_ablation_keys(outdir: Path, spec: ConfirmationSpec) -> set[tuple]:
    return set(_dedupe_rows(load_ablation_rows(outdir), spec))


def run_halving(args: argparse.Namespace) -> dict[str, Any]:
    spec = make_spec(args)
    outdir = args.outdir
    decisions_path = _halving_decisions_path(outdir, args.decisions)
    write_halving_plan(spec, outdir, decisions_path=decisions_path)
    decisions = load_halving_decisions(decisions_path, spec)
    groups = halving_run_groups(spec, decisions)
    remaining = args.max_new_cells

    for idx, group in enumerate(groups):
        if remaining is not None and remaining <= 0:
            break
        group_resume = bool(args.resume or idx > 0)
        before = _completed_ablation_keys(outdir, spec) if group_resume else set()
        print(
            "running "
            f"{group['run_group']} budgets={group['budgets']} seeds={group['seeds']} "
            f"strategies={group['strategies']} resume={group_resume}",
            flush=True,
        )
        run_ablation(
            list(spec.images),
            budgets=list(group["budgets"]),
            strategies=list(group["strategies"]),
            seeds=list(group["seeds"]),
            iters=spec.iters,
            target_psnr=spec.target_psnr,
            target_psnrs=list(spec.target_psnrs),
            flank_offsets=[0.5],
            flat_fracs=[0.02],
            corner_fracs=[0.15],
            max_axis_ratios=[6.0],
            coherence_powers=[1.0],
            render_chunk=spec.render_chunk,
            renderer=spec.renderer,
            pixel_loss=args.pixel_loss,
            ssim_weight=args.ssim_weight,
            compute_lpips=args.lpips,
            early_exit=args.early_exit,
            early_exit_window=args.early_exit_window,
            early_exit_min_delta=args.early_exit_min_delta,
            early_exit_min_iters=args.early_exit_min_iters,
            max_side=spec.max_side,
            resume=group_resume,
            max_new_cells=remaining,
            outdir=str(outdir),
            device=args.device,
            write_plots=not args.no_plots,
        )
        if remaining is not None:
            after = _completed_ablation_keys(outdir, spec)
            remaining -= len(after - before)

    if args.no_analyze:
        return {"analyzed": False, "run_groups": len(groups)}
    summary = write_halving_analysis(
        outdir,
        spec,
        decisions_path=decisions_path,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["run_groups"] = len(groups)
    return summary


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--kodak-dir", type=Path, default=DEFAULT_KODAK_DIR)
    parser.add_argument("--coco-dir", type=Path, default=DEFAULT_COCO_DIR)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--baseline", default="aniso_onedge")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--renderer", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=512)
    parser.add_argument("--target-psnr", type=float, default=35.0)
    parser.add_argument("--target-psnrs", type=float, nargs="*", default=DEFAULT_TARGET_PSNRS)
    parser.add_argument("--device", default=None)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-cells", type=int, default=None)
    parser.add_argument("--pixel-loss", choices=["l1", "l2", "charbonnier"], default="l1")
    parser.add_argument("--ssim-weight", type=float, default=0.3)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--early-exit", action="store_true",
                        help="opt-in plateau early exit; raw rows record stopped_at")
    parser.add_argument("--early-exit-window", type=int, default=150)
    parser.add_argument("--early-exit-min-delta", type=float, default=0.02)
    parser.add_argument("--early-exit-min-iters", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="ABL-004 confirmation runner and analysis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan_p = sub.add_parser("plan", help="write the expected confirmation cell manifest")
    _add_common_args(plan_p)

    run_p = sub.add_parser("run", help="run a resumable confirmation shard")
    _add_common_args(run_p)
    _add_run_args(run_p)

    halving_plan_p = sub.add_parser("halving-plan", help="write the ABL-006 staged manifest")
    _add_common_args(halving_plan_p)
    halving_plan_p.add_argument("--decisions", type=Path, default=None)

    halving_run_p = sub.add_parser("halving-run", help="run ABL-006 staged confirmation shards")
    _add_common_args(halving_run_p)
    halving_run_p.add_argument("--decisions", type=Path, default=None)
    _add_run_args(halving_run_p)

    analyze_p = sub.add_parser("analyze", help="analyze existing ablation rows")
    _add_common_args(analyze_p)
    analyze_p.add_argument("--bootstrap-samples", type=int, default=5000)
    analyze_p.add_argument("--bootstrap-seed", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "plan":
        spec = make_spec(args)
        cells = write_plan(spec, args.outdir)
        print(f"wrote {len(cells)} expected cells to {args.outdir / 'confirmation_plan.csv'}")
    elif args.cmd == "run":
        summary = run_confirmation(args)
        print(json.dumps(summary, indent=2, default=str))
    elif args.cmd == "halving-plan":
        spec = make_spec(args)
        cells = write_halving_plan(spec, args.outdir, decisions_path=args.decisions)
        print(f"wrote {len(cells)} staged cells to {args.outdir / 'abl006_plan.csv'}")
    elif args.cmd == "halving-run":
        summary = run_halving(args)
        print(json.dumps(summary, indent=2, default=str))
    elif args.cmd == "analyze":
        spec = make_spec(args)
        summary = write_confirmation_analysis(
            args.outdir,
            spec,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
