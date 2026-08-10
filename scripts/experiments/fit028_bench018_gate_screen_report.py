#!/usr/bin/env python3
"""Cross-arm comparison report for the FIT-028 / FIT-029 / BENCH-018 masked-arm screens.

`scripts/stage_search.py` already writes one maintained card per cell: per-run curves over
attempted steps, native-resolution target/reconstruction/error images, intermediate accepted
states, and the `gate_telemetry` commit-gate accounting added for these tasks. What it does not
write is the *cross-arm* view these three tasks actually decide on:

* paired per-seed deltas against the registered `current` baseline arm;
* quality against **wall-clock**, which is BENCH-018's estimand (a smaller block buys fewer wasted
  steps at the price of more gate evaluations, so equal-step curves answer the wrong question);
* terminal interior/boundary hole fractions side by side, which is FIT-028's guardrail; and
* per-phase acceptance across arms, which is how FIT-029 separates a vetoed phase from a
  miscalibrated one.

This driver only reads and serializes an executed bundle. It fits nothing, changes no default, and
cannot turn a development diagnostic into a promotion.

Reproduction:

    python scripts/experiments/fit028_bench018_gate_screen_report.py \\
        results/fit028_hole_budget_janelle_frame00008_2026-08-08 \\
        --baseline current --out-name comparison.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

# Terminal responses reported for every arm. `lower_is_better` drives the delta arrows only.
RESPONSES: tuple[tuple[str, str, bool], ...] = (
    ("psnr", "PSNR (dB)", False),
    ("ms_ssim", "MS-SSIM", False),
    ("lpips", "LPIPS", True),
    ("mae", "MAE", True),
    ("n_gaussians", "Gaussians", False),
    ("attempted_steps", "attempted", True),
    ("accepted_steps", "accepted", False),
    ("fit_seconds", "fit s", True),
)
# Curves overlaid across arms. The elapsed-seconds x-axis is the BENCH-018 reading.
OVERLAYS: tuple[tuple[str, str, str, str], ...] = (
    ("psnr", "attempted_steps", "PSNR over attempted steps", "#e65f2b"),
    ("psnr", "elapsed_seconds", "PSNR over wall-clock seconds", "#345995"),
    ("ms_ssim", "elapsed_seconds", "MS-SSIM over wall-clock seconds", "#147d72"),
    ("interior_hole_fraction", "attempted_steps", "Interior holes over steps", "#2f6b3c"),
    ("boundary_hole_fraction", "attempted_steps", "Boundary holes over steps", "#99582a"),
    ("n_gaussians", "attempted_steps", "Gaussians over steps", "#7a5195"),
)
PALETTE = ("#1d2528", "#e65f2b", "#345995", "#147d72", "#b58416", "#8f2d56")


def _load_rows(bundle: Path) -> list[dict[str, Any]]:
    payload = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{bundle / 'metrics.json'} is not a row list")
    return payload


def _arm_of(row: dict[str, Any]) -> str:
    return str(row.get("variant") or row.get("method") or "unknown")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _paired_delta(
    arm_rows: dict[int, dict[str, Any]],
    base_rows: dict[int, dict[str, Any]],
    key: str,
) -> tuple[float | None, int]:
    """Mean paired delta on the seeds both arms completed. Unpaired seeds are dropped, not filled."""

    deltas = []
    for seed in sorted(set(arm_rows) & set(base_rows)):
        arm_value = _finite(arm_rows[seed].get(key))
        base_value = _finite(base_rows[seed].get(key))
        if arm_value is None or base_value is None:
            continue
        deltas.append(arm_value - base_value)
    if not deltas:
        return None, 0
    return statistics.fmean(deltas), len(deltas)


def _svg_overlay(
    series: list[tuple[str, str, list[tuple[float, float]]]],
    title: str,
) -> str:
    points = [point for _, _, curve in series for point in curve]
    if len(points) < 2:
        return (
            f"<div class='chart'><strong>{html.escape(title)}</strong>"
            "<p class='empty'>no data</p></div>"
        )
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1e-9
    width, height, pad = 460.0, 210.0, 34.0

    def place(x: float, y: float) -> tuple[float, float]:
        px = pad + (x - x0) / (x1 - x0) * (width - 2 * pad)
        py = height - pad - (y - y0) / (y1 - y0) * (height - 2 * pad)
        return px, py

    polylines = []
    legend = []
    for label, colour, curve in series:
        if len(curve) < 2:
            continue
        coordinates = " ".join(
            f"{px:.2f},{py:.2f}" for px, py in (place(x, y) for x, y in curve)
        )
        polylines.append(
            f"<polyline points='{coordinates}' stroke='{colour}'></polyline>"
        )
        legend.append(
            f"<span><i style='background:{colour}'></i>{html.escape(label)}</span>"
        )
    return (
        f"<div class='chart'><strong>{html.escape(title)}</strong>"
        f"<div class='legend'>{''.join(legend)}</div>"
        f"<svg viewBox='0 0 {width:.0f} {height:.0f}' role='img'>"
        f"<path d='M{pad},{height - pad} L{width - pad},{height - pad}'></path>"
        f"<path d='M{pad},{pad} L{pad},{height - pad}'></path>"
        f"{''.join(polylines)}"
        f"<text x='{pad}' y='{height - 10:.0f}'>{x0:.4g}</text>"
        f"<text x='{width - pad:.0f}' y='{height - 10:.0f}' text-anchor='end'>{x1:.4g}</text>"
        f"<text x='4' y='{pad}'>{y1:.4g}</text>"
        f"<text x='4' y='{height - pad:.0f}'>{y0:.4g}</text>"
        "</svg></div>"
    )


def _curve_points(row: dict[str, Any], y_key: str, x_key: str) -> list[tuple[float, float]]:
    out = []
    for point in row.get("curves") or ():
        x = _finite(point.get(x_key))
        y = _finite(point.get(y_key))
        if x is None or y is None:
            continue
        out.append((x, y))
    out.sort(key=lambda item: item[0])
    return out


def _block_level_reasons(bundle: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Count *rejected blocks*, and blocks citing each reason, from the cell's history.

    `gate_telemetry.rejection_reasons` counts reason **occurrences**, because one rejected block
    can cite several at once. BENCH-017's published figure ("82 of 110 rejected blocks died on
    `interior_holes_regressed`") is a **block** count, so comparing the two directly would
    overstate any reason that co-occurs. This recomputes the block-level view from the persisted
    history so the two are on the same footing. It is read-only and post-hoc: the running
    experiment's telemetry is not changed mid-grid.
    """

    link = row.get("history_json")
    if not link:
        return {}
    path = Path(str(link))
    if not path.is_absolute():
        path = bundle / path
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("schedule_history") or []
    rejected_blocks = 0
    blocks_citing: dict[str, int] = {}
    blocks_citing_only: dict[str, int] = {}
    for record in history:
        reasons = record.get("reasons") or []
        if not reasons:
            continue
        rejected_blocks += 1
        unique = {str(name) for name in reasons}
        for reason in unique:
            blocks_citing[reason] = blocks_citing.get(reason, 0) + 1
        if len(unique) == 1:
            sole = next(iter(unique))
            blocks_citing_only[sole] = blocks_citing_only.get(sole, 0) + 1
    return {
        "rejected_blocks": rejected_blocks,
        "blocks_citing": blocks_citing,
        # The revivable set. A block vetoed by several terms at once cannot be recovered by
        # relaxing one of them, so `blocks_citing_only[r]` upper-bounds what loosening `r` can
        # possibly return — which is the quantity FIT-028's budget ladder actually acts on.
        "blocks_citing_only": blocks_citing_only,
    }


def _terminal_holes(row: dict[str, Any]) -> tuple[float | None, float | None]:
    curves = row.get("curves") or ()
    if not curves:
        return None, None
    last = max(curves, key=lambda point: float(point.get("attempted_steps") or 0))
    return (
        _finite(last.get("interior_hole_fraction")),
        _finite(last.get("boundary_hole_fraction")),
    )


def _relative(bundle: Path, value: Any) -> str | None:
    if not value:
        return None
    try:
        return Path(str(value)).resolve().relative_to(bundle.resolve()).as_posix()
    except ValueError:
        # Already relative inside the bundle, or genuinely outside it.
        candidate = Path(str(value))
        return candidate.as_posix() if not candidate.is_absolute() else None


def build_report(bundle: Path, baseline: str) -> tuple[str, dict[str, Any]]:
    rows = _load_rows(bundle)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]

    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for row in ok_rows:
        by_arm.setdefault(_arm_of(row), {})[int(row.get("seed", 0))] = row
    if baseline not in by_arm:
        raise ValueError(
            f"baseline arm {baseline!r} has no completed cell; found {sorted(by_arm)}"
        )
    # Baseline first, then registry order.
    arms = [baseline] + sorted(arm for arm in by_arm if arm != baseline)

    summary: dict[str, Any] = {
        "bundle": bundle.name,
        "baseline": baseline,
        "arms": {},
        "error_cells": [
            {"variant": _arm_of(row), "seed": row.get("seed"), "error": row.get("error")}
            for row in error_rows
        ],
    }

    matrix_rows = []
    for index, arm in enumerate(arms):
        seeds = by_arm[arm]
        record: dict[str, Any] = {"seeds": sorted(seeds), "n_cells": len(seeds)}
        cells = []
        for key, _label, _lower in RESPONSES:
            values = [v for v in (_finite(r.get(key)) for r in seeds.values()) if v is not None]
            mean = statistics.fmean(values) if values else None
            record[key] = mean
            delta, paired = (None, 0)
            if arm != baseline:
                delta, paired = _paired_delta(seeds, by_arm[baseline], key)
                record[f"delta_{key}"] = delta
                record[f"paired_seeds_{key}"] = paired
            text = "n/a" if mean is None else f"{mean:,.4f}"
            if delta is not None:
                text += f"<br><small>{delta:+,.4f} ({paired} paired)</small>"
            cells.append(f"<td class='n'>{text}</td>")
        interiors, boundaries = [], []
        for row in seeds.values():
            interior, boundary = _terminal_holes(row)
            if interior is not None:
                interiors.append(interior)
            if boundary is not None:
                boundaries.append(boundary)
        record["terminal_interior_hole_fraction"] = (
            statistics.fmean(interiors) if interiors else None
        )
        record["terminal_boundary_hole_fraction"] = (
            statistics.fmean(boundaries) if boundaries else None
        )
        cells.append(
            "<td class='n'>"
            + (
                "n/a"
                if not interiors
                else f"{statistics.fmean(interiors):.4%}"
            )
            + "</td>"
        )
        cells.append(
            "<td class='n'>"
            + (
                "n/a"
                if not boundaries
                else f"{statistics.fmean(boundaries):.4%}"
            )
            + "</td>"
        )
        marker = " (baseline)" if arm == baseline else ""
        matrix_rows.append(
            f"<tr><td><b style='color:{PALETTE[index % len(PALETTE)]}'>&#9632;</b> "
            f"{html.escape(arm)}{marker}</td>"
            f"<td class='n'>{len(seeds)}</td>{''.join(cells)}</tr>"
        )
        summary["arms"][arm] = record

    # Per-phase acceptance across arms: FIT-029 reads safe_polish here.
    phases: list[str] = []
    for arm in arms:
        for row in by_arm[arm].values():
            for phase in (row.get("gate_telemetry") or {}).get("phases", {}):
                if phase not in phases:
                    phases.append(phase)
    gate_rows = []
    for index, arm in enumerate(arms):
        cells = []
        for phase in phases:
            attempted, accepted = 0, 0
            for row in by_arm[arm].values():
                stats = ((row.get("gate_telemetry") or {}).get("phases") or {}).get(phase)
                if stats:
                    attempted += int(stats.get("attempted_steps") or 0)
                    accepted += int(stats.get("accepted_steps") or 0)
            summary["arms"][arm].setdefault("phase_acceptance", {})[phase] = {
                "attempted_steps": attempted,
                "accepted_steps": accepted,
                "step_acceptance": None if attempted <= 0 else accepted / attempted,
            }
            cells.append(
                "<td class='n'>"
                + ("—" if attempted <= 0 else f"{accepted / attempted:.1%}<br><small>{accepted:,}/{attempted:,}</small>")
                + "</td>"
            )
        gate_rows.append(
            f"<tr><td><b style='color:{PALETTE[index % len(PALETTE)]}'>&#9632;</b> "
            f"{html.escape(arm)}</td>{''.join(cells)}</tr>"
        )

    reasons: dict[str, dict[str, int]] = {}
    for arm in arms:
        for row in by_arm[arm].values():
            for name, count in ((row.get("gate_telemetry") or {}).get("rejection_reasons") or {}).items():
                reasons.setdefault(arm, {})[name] = reasons.setdefault(arm, {}).get(name, 0) + int(count)
    blocks: dict[str, dict[str, Any]] = {}
    for arm in arms:
        rejected = 0
        citing: dict[str, int] = {}
        only: dict[str, int] = {}
        for row in by_arm[arm].values():
            level = _block_level_reasons(bundle, row)
            rejected += int(level.get("rejected_blocks") or 0)
            for name, count in (level.get("blocks_citing") or {}).items():
                citing[name] = citing.get(name, 0) + int(count)
            for name, count in (level.get("blocks_citing_only") or {}).items():
                only[name] = only.get(name, 0) + int(count)
        blocks[arm] = {
            "rejected_blocks": rejected,
            "blocks_citing": citing,
            "blocks_citing_only": only,
        }
        summary["arms"][arm]["rejected_blocks"] = rejected
        summary["arms"][arm]["blocks_citing"] = citing
        summary["arms"][arm]["blocks_citing_only"] = only

    reason_names = sorted({name for per_arm in reasons.values() for name in per_arm})
    reason_rows = []
    for arm in arms:
        rejected = blocks[arm]["rejected_blocks"]
        cells = []
        for name in reason_names:
            occurrences = reasons.get(arm, {}).get(name, 0)
            citing = blocks[arm]["blocks_citing"].get(name, 0)
            sole = blocks[arm]["blocks_citing_only"].get(name, 0)
            share = (
                ""
                if rejected <= 0
                else f"<br><small>{citing:,}/{rejected:,} blocks · {sole:,} alone</small>"
            )
            cells.append(f"<td class='n'>{occurrences:,}{share}</td>")
        reason_rows.append(
            f"<tr><td>{html.escape(arm)}</td>"
            f"<td class='n'>{rejected:,}</td>{''.join(cells)}</tr>"
        )
    summary["rejection_reasons"] = reasons
    summary["block_level_reasons"] = blocks

    charts = []
    for y_key, x_key, title, _colour in OVERLAYS:
        series = []
        for index, arm in enumerate(arms):
            seed = sorted(by_arm[arm])[0]
            curve = _curve_points(by_arm[arm][seed], y_key, x_key)
            if curve:
                series.append((f"{arm} (seed {seed})", PALETTE[index % len(PALETTE)], curve))
        charts.append(_svg_overlay(series, title))

    galleries = []
    for arm in arms:
        seed = sorted(by_arm[arm])[0]
        row = by_arm[arm][seed]
        figures = []
        for label, key in (
            ("target", "target_png"),
            ("reconstruction", "reconstruction_png"),
            ("absolute error x4", "error_png"),
        ):
            link = _relative(bundle, row.get(key))
            if link:
                figures.append(
                    f"<figure><a href='{html.escape(link)}'>"
                    f"<img src='{html.escape(link)}' loading='lazy' alt='{html.escape(label)}'>"
                    f"</a><figcaption>{html.escape(label)}</figcaption></figure>"
                )
        galleries.append(
            f"<section><h3>{html.escape(arm)} · seed {seed}</h3>"
            f"<div class='gallery'>{''.join(figures)}</div></section>"
        )

    error_html = ""
    if error_rows:
        error_html = (
            "<h2 class='errors'>Error cells</h2><ul class='errors'>"
            + "".join(
                f"<li>{html.escape(_arm_of(row))} seed {row.get('seed')}: "
                f"{html.escape(str(row.get('error')))}</li>"
                for row in error_rows
            )
            + "</ul>"
        )

    response_headers = "".join(f"<th>{html.escape(label)}</th>" for _k, label, _l in RESPONSES)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(bundle.name)} — cross-arm comparison</title>
<style>
:root{{--line:#d8d2c6;--panel:#fffdf7;--muted:#6b6b66}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f1ea;color:#1d2528;
font-family:"Liberation Sans",system-ui,sans-serif}}
header,main{{width:min(100% - 40px,1500px);margin:0 auto}}header{{padding:38px 0 10px}}
h1{{margin:.1em 0}}h2{{margin:1.4em 0 .3em}}h3{{margin:.2em 0}}
p.scope{{max-width:1000px;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;background:var(--panel);font-size:.84rem;
font-family:"Liberation Mono",monospace}}
th,td{{border:1px solid var(--line);padding:7px;text-align:left}}
td.n,th.n{{text-align:right}}small{{color:var(--muted)}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.chart{{background:#fff;border:1px solid var(--line);padding:9px}}
.chart svg{{width:100%;height:auto}}.chart path{{fill:none;stroke:#c7c1b5;stroke-width:1}}
.chart polyline{{fill:none;stroke-width:2.4}}
.chart text{{font:10px "Liberation Mono",monospace;fill:#6b6b66}}
.legend{{display:flex;flex-wrap:wrap;gap:9px;font-size:.74rem;margin:5px 0}}
.legend i{{display:inline-block;width:9px;height:9px;margin-right:4px}}
.gallery{{display:flex;gap:12px;overflow:auto;padding:6px 0}}
figure{{margin:0;min-width:260px;flex:1}}img{{width:100%;height:auto;display:block;
border:1px solid #777;background:#111}}
figcaption{{font-size:.8rem;color:var(--muted);margin-top:4px}}
.errors{{color:#8a251a}}.empty{{color:var(--muted)}}
</style></head><body>
<header><h1>{html.escape(bundle.name)}</h1>
<p class="scope">Cross-arm view of one executed stage-search bundle. Baseline arm
<code>{html.escape(baseline)}</code>. Deltas are means over the seeds both arms completed; unpaired
seeds are dropped, never filled. This is a development diagnostic on exposed data: it cannot
promote a default.</p></header><main>
<h2>Terminal responses</h2>
<table><thead><tr><th>arm</th><th class="n">cells</th>{response_headers}
<th class="n">interior holes</th><th class="n">boundary holes</th></tr></thead>
<tbody>{''.join(matrix_rows)}</tbody></table>
<h2>Per-phase step acceptance</h2>
<table><thead><tr><th>arm</th>
{''.join(f'<th class="n">{html.escape(p)}</th>' for p in phases)}</tr></thead>
<tbody>{''.join(gate_rows)}</tbody></table>
<h2>Rejection reasons</h2>
<p class="scope">Top number is reason <b>occurrences</b>; a rejected block can cite several at once.
The line below it is the count of <b>rejected blocks</b> citing that reason, which is the footing
BENCH-017's "82 of 110 rejected blocks" figure uses. Do not compare the two directly. The
<b>alone</b> count is the subset vetoed by that reason and nothing else: relaxing one term cannot
revive a block that several terms rejected, so it upper-bounds what loosening that term can
return.</p>
<table><thead><tr><th>arm</th><th class="n">rejected blocks</th>
{''.join(f'<th class="n">{html.escape(n)}</th>' for n in reason_names)}</tr></thead>
<tbody>{''.join(reason_rows)}</tbody></table>
<h2>Curves</h2>
<div class="charts">{''.join(charts)}</div>
<h2>Reconstructions</h2>
{''.join(galleries)}
{error_html}
</main></body></html>"""
    return page, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="executed stage_search output directory")
    parser.add_argument("--baseline", default="current", help="registered baseline arm")
    parser.add_argument("--out-name", default="comparison.html")
    args = parser.parse_args(argv)

    bundle = args.bundle.expanduser().resolve()
    page, summary = build_report(bundle, args.baseline)
    (bundle / args.out_name).write_text(page, encoding="utf-8")
    (bundle / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"wrote {bundle / args.out_name}")
    print(f"wrote {bundle / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
