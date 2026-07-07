"""BENCH-004 proxy-regime calibration.

Compare a cheap proxy screen against a committed full-regime screen. The proxy is accepted only
when it preserves the strategy ranking and the signs of meaningful paired deltas.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from benchmarks.common import run_config, write_config, write_csv, write_json


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return data["rows"]
        raise ValueError(f"{path} does not contain a row list")
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"unsupported result file extension: {path}")


def _as_int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def _as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _budget(row: dict[str, Any]) -> int:
    raw = row.get("budget", row.get("final_budget"))
    if raw is None:
        raise KeyError("row has neither budget nor final_budget")
    return int(float(raw))


def _strategy(row: dict[str, Any]) -> str:
    return str(row.get("strategy") or row.get("method"))


def aggregate(rows: list[dict[str, Any]], metric: str = "psnr") -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if row.get("status") == "error":
            continue
        if metric not in row or row.get(metric) in (None, ""):
            continue
        groups.setdefault((_budget(row), _strategy(row)), []).append(_as_float(row, metric))
    out = []
    for (budget, strategy), vals in sorted(groups.items()):
        out.append({
            "budget": budget,
            "strategy": strategy,
            f"mean_{metric}": mean(vals),
            "runs": len(vals),
        })
    return out


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = avg
        pos = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    rx = np.asarray(_average_ranks(x), dtype=np.float64)
    ry = np.asarray(_average_ranks(y), dtype=np.float64)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def calibrate(
    full_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    *,
    metric: str = "psnr",
    min_delta: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    full = aggregate(full_rows, metric)
    proxy = aggregate(proxy_rows, metric)
    value_key = f"mean_{metric}"
    full_by = {(r["budget"], r["strategy"]): r for r in full}
    proxy_by = {(r["budget"], r["strategy"]): r for r in proxy}
    budgets = sorted({b for b, _s in full_by} & {b for b, _s in proxy_by})

    rank_rows = []
    sign_rows = []
    for budget in budgets:
        strategies = sorted(
            {s for b, s in full_by if b == budget}
            & {s for b, s in proxy_by if b == budget}
        )
        full_vals = [float(full_by[(budget, s)][value_key]) for s in strategies]
        proxy_vals = [float(proxy_by[(budget, s)][value_key]) for s in strategies]
        rho = spearman(full_vals, proxy_vals)
        rank_rows.append({
            "budget": budget,
            "metric": metric,
            "strategies": len(strategies),
            "spearman_rho": rho,
            "full_top1": strategies[int(np.argmax(full_vals))] if strategies else None,
            "proxy_top1": strategies[int(np.argmax(proxy_vals))] if strategies else None,
        })

        agreed = 0
        compared = 0
        skipped_small = 0
        for left, right in combinations(strategies, 2):
            full_delta = (
                float(full_by[(budget, left)][value_key])
                - float(full_by[(budget, right)][value_key])
            )
            proxy_delta = (
                float(proxy_by[(budget, left)][value_key])
                - float(proxy_by[(budget, right)][value_key])
            )
            if abs(full_delta) <= min_delta:
                skipped_small += 1
                continue
            compared += 1
            if np.sign(full_delta) == np.sign(proxy_delta):
                agreed += 1
        sign_rows.append({
            "budget": budget,
            "metric": metric,
            "min_full_delta": min_delta,
            "compared_pairs": compared,
            "agreeing_pairs": agreed,
            "skipped_small_delta_pairs": skipped_small,
            "sign_agreement": (agreed / compared if compared else None),
        })
    return {
        "full_aggregate": full,
        "proxy_aggregate": proxy,
        "rank_correlation": rank_rows,
        "sign_agreement": sign_rows,
    }


def write_report(
    outdir: Path,
    result: dict[str, list[dict[str, Any]]],
    *,
    full_path: Path,
    proxy_path: Path,
    metric: str,
    min_delta: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "calibration.json", result)
    write_csv(outdir / "full_aggregate.csv", result["full_aggregate"])
    write_csv(outdir / "proxy_aggregate.csv", result["proxy_aggregate"])
    write_csv(outdir / "rank_correlation.csv", result["rank_correlation"])
    write_csv(outdir / "sign_agreement.csv", result["sign_agreement"])

    lines = [
        "# BENCH-004 Proxy Calibration",
        "",
        f"Full regime: `{full_path}`",
        f"Proxy regime: `{proxy_path}`",
        f"Metric: `{metric}`",
        f"Sign agreement threshold: `|full delta| > {min_delta:g}` dB",
        "",
        "## Rank Correlation",
        "",
        "| Budget | Strategies | Spearman rho | Full top-1 | Proxy top-1 |",
        "|---:|---:|---:|---|---|",
    ]
    for row in result["rank_correlation"]:
        rho = "-" if row["spearman_rho"] is None else f"{row['spearman_rho']:.4f}"
        lines.append(
            f"| {row['budget']} | {row['strategies']} | {rho} | "
            f"{row['full_top1']} | {row['proxy_top1']} |"
        )
    lines += [
        "",
        "## Paired-Delta Sign Agreement",
        "",
        "| Budget | Compared pairs | Agreeing pairs | Skipped small deltas | Agreement |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["sign_agreement"]:
        agreement = "-" if row["sign_agreement"] is None else f"{100.0 * row['sign_agreement']:.1f}%"
        lines.append(
            f"| {row['budget']} | {row['compared_pairs']} | {row['agreeing_pairs']} | "
            f"{row['skipped_small_delta_pairs']} | {agreement} |"
        )
    lines += [
        "",
        "Acceptance target: Spearman rho >= 0.9 and sign agreement >= 90% at each budget.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a cheap proxy screen against a full screen")
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--metric", default="psnr")
    parser.add_argument("--min-delta", type=float, default=0.1)
    args = parser.parse_args()

    full_rows = load_rows(args.full)
    proxy_rows = load_rows(args.proxy)
    result = calibrate(full_rows, proxy_rows, metric=args.metric, min_delta=args.min_delta)
    write_config(
        str(args.outdir),
        run_config({
            "full": str(args.full),
            "proxy": str(args.proxy),
            "metric": args.metric,
            "min_delta": args.min_delta,
        }),
    )
    write_report(
        args.outdir,
        result,
        full_path=args.full,
        proxy_path=args.proxy,
        metric=args.metric,
        min_delta=args.min_delta,
    )


if __name__ == "__main__":
    main()
