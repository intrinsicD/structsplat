#!/usr/bin/env python3
"""Build one audit-oriented HTML comparison for Janelle safe-schedule runs."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any


COLORS = ("#7dd3fc", "#fbbf24", "#a7f3d0", "#f0abfc", "#fb7185")
FACTOR_FIELDS = (
    "pareto_safe_checkpoints",
    "event_color_solve",
)


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run must contain a label and path")
    return label.strip(), Path(raw_path).expanduser().resolve()


def _relative(out: Path, path: Path) -> str:
    return Path(os.path.relpath(path, out)).as_posix()


def _load(label: str, run: Path, out: Path) -> dict[str, Any]:
    required = (
        "summary.json",
        "run_config.json",
        "schedule_history.json",
        "index.html",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run}: missing {', '.join(missing)}")
    summary = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "run_config.json").read_text())
    schedule = json.loads((run / "schedule_history.json").read_text())
    final = summary["final"]
    native = final.get("native") or {}
    reconstruction = native.get("reconstruction", final["reconstruction"])
    error = native.get("absolute_error_x4", final["absolute_error_x4"])
    events = Counter(
        record["event"]
        for record in schedule["history"]
        if record["accepted"]
    )
    rejected = Counter(
        reason
        for record in schedule["history"]
        if not record["accepted"]
        for reason in record["reasons"]
    )
    committed_recoveries: list[dict[str, Any]] = []
    for record in schedule["history"]:
        if not record["accepted"]:
            continue
        metadata = record.get("metadata") or {}
        if "checkpoint_candidates" in metadata:
            committed_recoveries.append(metadata)
        winner = metadata.get("winner_metadata") or {}
        recovery = winner.get("recovery")
        if isinstance(recovery, dict):
            committed_recoveries.append(recovery)
    curve = []
    for record in schedule["history"]:
        if not record["accepted"] or record["event"] == "phase_end":
            continue
        selected = record["selected"]
        curve.append({
            "x": int(record["global_attempted_steps"]),
            "fg": float(selected["foreground_psnr_db"]),
            "boundary": float(selected["boundary_psnr_db"]),
            "interior_holes": float(selected["interior_hole_fraction"]),
            "boundary_holes": float(selected["boundary_hole_fraction"]),
            "cvar99": float(selected["cvar99_mse"]),
            "p99": float(selected["p99_mse"]),
        })
    schedule_config = config["schedule"]
    return {
        "label": label,
        "path": str(run),
        "index": _relative(out, run / "index.html"),
        "reconstruction": _relative(out, run / reconstruction),
        "error": _relative(out, run / error),
        "metrics": summary["safe_metrics"],
        "initial": summary["initial"],
        "final": final,
        "attempted_steps": int(schedule["attempted_steps"]),
        "accepted_steps": int(schedule["accepted_steps"]),
        "converged": bool(schedule["converged"]),
        "seconds_schedule": float(schedule["seconds"]),
        "seconds_total": float(config["timing"]["total_seconds"]),
        "peak_gpu_memory_bytes": int(config["timing"]["peak_gpu_memory_bytes"]),
        "refinement_policy": schedule_config.get("refinement_policy", "global"),
        "local_start_phase": schedule_config.get("local_start_phase", "coverage"),
        "local_seed_count": schedule_config.get("local_seed_count", 0),
        "local_neighbor_count": schedule_config.get("local_neighbor_count", 0),
        "topology_neighbor_count": schedule_config.get("topology_neighbor_count", 0),
        "boundary_recycle_at_capacity": schedule_config.get(
            "boundary_recycle_at_capacity", False
        ),
        "pareto_safe_checkpoints": schedule_config.get(
            "pareto_safe_checkpoints", False
        ),
        "pareto_checkpoint_every": schedule_config.get(
            "pareto_checkpoint_every", 50
        ),
        "event_color_solve": schedule_config.get("event_color_solve", False),
        "earlier_checkpoint_commits": sum(
            bool(metadata.get("selected_from_pareto_checkpoint"))
            for metadata in committed_recoveries
        ),
        "event_color_solve_commits": sum(
            bool(metadata.get("event_color_solve_applied"))
            for metadata in committed_recoveries
        ),
        "guard_p99": schedule_config["tolerances"].get("guard_p99", False),
        "accepted_events": dict(events),
        "rejected_reasons": dict(rejected),
        "curve": curve,
        "source": config["source"],
        "repository": config["repository"],
        "initialization": {
            key: value
            for key, value in config["initialization"].items()
            if key != "metrics"
        },
        "schedule_config": schedule_config,
        "fit_config": config["fit_config"],
        "environment": config["environment"],
    }


def _source_hashes(run: dict[str, Any]) -> dict[str, str]:
    return {
        str(record["source"]): str(record["sha256"])
        for record in run["repository"]["executed_source_snapshots"]
    }


def _validate_factorial(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 4:
        raise ValueError("--require-factorial requires exactly four runs")
    expected = {(False, False), (True, False), (False, True), (True, True)}
    combinations = {
        (
            bool(run["pareto_safe_checkpoints"]),
            bool(run["event_color_solve"]),
        )
        for run in runs
    }
    if combinations != expected:
        raise ValueError(
            "factorial runs must cover each checkpoint/color-solve combination exactly once"
        )
    baseline = runs[0]
    baseline_schedule = {
        key: value
        for key, value in baseline["schedule_config"].items()
        if key not in FACTOR_FIELDS
    }
    for run in runs[1:]:
        schedule = {
            key: value
            for key, value in run["schedule_config"].items()
            if key not in FACTOR_FIELDS
        }
        checks = (
            ("source", run["source"], baseline["source"]),
            ("initialization", run["initialization"], baseline["initialization"]),
            ("fit config", run["fit_config"], baseline["fit_config"]),
            ("schedule excluding factors", schedule, baseline_schedule),
            ("environment", run["environment"], baseline["environment"]),
            ("executed source hashes", _source_hashes(run), _source_hashes(baseline)),
        )
        for name, actual, expected_value in checks:
            if actual != expected_value:
                raise ValueError(
                    f"factorial fairness check failed for {run['label']}: {name}"
                )
    return {
        "valid": True,
        "arms": {
            f"checkpoint={str(run['pareto_safe_checkpoints']).lower()},"
            f"event_color={str(run['event_color_solve']).lower()}": run["label"]
            for run in runs
        },
        "controlled": [
            "source files and seed",
            "initialization configuration",
            "fit configuration",
            "schedule configuration excluding the two boolean factors",
            "runtime environment",
            "executed source snapshots",
        ],
    }


def _factorial_effects(runs: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {
        (
            bool(run["pareto_safe_checkpoints"]),
            bool(run["event_color_solve"]),
        ): run
        for run in runs
    }
    fields = (
        "foreground_mse",
        "boundary_mse",
        "cvar99_mse",
        "p99_mse",
        "interior_hole_fraction",
        "boundary_hole_fraction",
        "outside_max_abs",
        "outside_coverage_max",
    )
    effects: dict[str, Any] = {}
    for field in fields:
        v00 = float(arms[(False, False)]["metrics"][field])
        v10 = float(arms[(True, False)]["metrics"][field])
        v01 = float(arms[(False, True)]["metrics"][field])
        v11 = float(arms[(True, True)]["metrics"][field])
        effects[field] = {
            "control": v00,
            "checkpoint_without_event_color": v10 - v00,
            "event_color_without_checkpoint": v01 - v00,
            "combined_vs_control": v11 - v00,
            "interaction": v11 - v10 - v01 + v00,
        }
    control = arms[(False, False)]["metrics"]
    combined = arms[(True, True)]["metrics"]
    protected = (
        "foreground_mse",
        "boundary_mse",
        "cvar99_mse",
        "interior_hole_fraction",
        "boundary_hole_fraction",
        "outside_max_abs",
        "outside_coverage_max",
    )
    nonworse = {
        field: float(combined[field]) <= float(control[field])
        for field in protected
    }
    material = any(
        float(combined[field]) < float(control[field])
        for field in protected
    )
    return {
        "metrics": effects,
        "combined_decision_rule": {
            "all_protected_metrics_nonworse": all(nonworse.values()),
            "per_metric_nonworse": nonworse,
            "at_least_one_strict_improvement": material,
            "passes": all(nonworse.values()) and material,
        },
    }


def _svg_chart(
    runs: list[dict[str, Any]],
    field: str,
    title: str,
    *,
    percent: bool = False,
) -> str:
    width, height = 920, 300
    left, right, top, bottom = 66, 20, 38, 48
    all_points = [
        point
        for run in runs
        for point in run["curve"]
        if point.get(field) is not None
    ]
    max_x = max(point["x"] for point in all_points)
    values = [point[field] * (100.0 if percent else 1.0) for point in all_points]
    min_y, max_y = min(values), max(values)
    padding = max((max_y - min_y) * 0.08, 1e-6)
    min_y -= padding
    max_y += padding

    def px(value: float) -> float:
        return left + value / max(max_x, 1) * (width - left - right)

    def py(value: float) -> float:
        return top + (max_y - value) / max(max_y - min_y, 1e-9) * (
            height - top - bottom
        )

    lines = [
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{html.escape(title)}'>",
        f"<text x='{left}' y='22' class='chart-title'>{html.escape(title)}</text>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}'/>",
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' "
        f"y2='{height-bottom}'/>",
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = min_y + fraction * (max_y - min_y)
        y = py(value)
        suffix = "%" if percent else ""
        lines.append(
            f"<line class='grid' x1='{left}' y1='{y:.2f}' "
            f"x2='{width-right}' y2='{y:.2f}'/>"
        )
        lines.append(
            f"<text class='tick' x='{left-8}' y='{y+4:.2f}' "
            f"text-anchor='end'>{value:.3f}{suffix}</text>"
        )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = fraction * max_x
        x = px(value)
        lines.append(
            f"<text class='tick' x='{x:.2f}' y='{height-bottom+20}' "
            f"text-anchor='middle'>{int(value):,}</text>"
        )
    lines.append(
        f"<text class='tick' x='{(left+width-right)/2:.2f}' y='{height-8}' "
        "text-anchor='middle'>attempted optimizer steps</text>"
    )
    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        points = " ".join(
            f"{px(point['x']):.2f},{py(point[field] * (100.0 if percent else 1.0)):.2f}"
            for point in run["curve"]
        )
        lines.append(
            f"<polyline points='{points}' style='stroke:{color}'/>"
        )
        legend_x = left + index * 190
        lines.append(
            f"<line class='legend-line' x1='{legend_x}' y1='{height-27}' "
            f"x2='{legend_x+22}' y2='{height-27}' style='stroke:{color}'/>"
        )
        lines.append(
            f"<text class='tick' x='{legend_x+28}' y='{height-23}'>"
            f"{html.escape(run['label'])}</text>"
        )
    lines.append("</svg>")
    return "".join(lines)


def _metric_rows(runs: list[dict[str, Any]]) -> str:
    metrics = (
        ("FG PSNR", "foreground_psnr_db", " dB", 3),
        ("Boundary PSNR", "boundary_psnr_db", " dB", 3),
        ("FG MSE", "foreground_mse", "", 6),
        ("Boundary MSE", "boundary_mse", "", 6),
        ("CVaR99 MSE", "cvar99_mse", "", 6),
        ("p99 MSE", "p99_mse", "", 6),
        ("Interior undercoverage", "interior_hole_fraction", "%", 3),
        ("Boundary undercoverage", "boundary_hole_fraction", "%", 3),
        ("Outside render max", "outside_max_abs", "", 6),
    )
    rows = []
    for label, key, suffix, digits in metrics:
        values = []
        for run in runs:
            value = float(run["metrics"][key])
            if suffix == "%":
                value *= 100.0
            values.append(f"<td>{value:.{digits}f}{suffix}</td>")
        rows.append(f"<tr><th>{html.escape(label)}</th>{''.join(values)}</tr>")
    return "\n".join(rows)


def _run_cards(runs: list[dict[str, Any]]) -> str:
    cards = []
    for run in runs:
        accepted_local = run["accepted_events"].get("local_residual_fit", 0)
        cards.append(
            "<article class='card'>"
            f"<h3>{html.escape(run['label'])}</h3>"
            f"<p><a href='{html.escape(run['index'])}'>vollständige Timeline</a></p>"
            f"<p>policy <code>{html.escape(run['refinement_policy'])}</code>; "
            f"Pareto checkpoints <code>{str(run['pareto_safe_checkpoints']).lower()}</code> "
            f"(every {run['pareto_checkpoint_every']}); event color solve "
            f"<code>{str(run['event_color_solve']).lower()}</code>; "
            f"local start <code>{html.escape(run['local_start_phase'])}</code>; "
            f"local/topology neighbours {run['local_neighbor_count']}/"
            f"{run['topology_neighbor_count']}; p99 guard "
            f"{str(run['guard_p99']).lower()}; boundary recycle "
            f"{str(run['boundary_recycle_at_capacity']).lower()}</p>"
            f"<p>{run['attempted_steps']:,} attempted / "
            f"{run['accepted_steps']:,} accepted optimizer steps; "
            f"{run['earlier_checkpoint_commits']} earlier-checkpoint commits; "
            f"{run['event_color_solve_commits']} event-color commits; "
            f"{accepted_local} accepted standalone local fits; "
            f"{run['seconds_total']:.1f} s total</p>"
            f"<img src='{html.escape(run['reconstruction'])}' "
            f"alt='{html.escape(run['label'])} reconstruction'>"
            f"<img src='{html.escape(run['error'])}' "
            f"alt='{html.escape(run['label'])} error x4'>"
            "</article>"
        )
    return "\n".join(cards)


def _factor_rows(runs: list[dict[str, Any]]) -> str:
    return "\n".join(
        "<tr>"
        f"<th>{html.escape(run['label'])}</th>"
        f"<td>{str(run['pareto_safe_checkpoints']).lower()}</td>"
        f"<td>{str(run['event_color_solve']).lower()}</td>"
        f"<td>{run['earlier_checkpoint_commits']}</td>"
        f"<td>{run['event_color_solve_commits']}</td>"
        "</tr>"
        for run in runs
    )


def _effect_rows(factorial: dict[str, Any]) -> str:
    labels = (
        ("FG MSE", "foreground_mse", 8, 1.0),
        ("Boundary MSE", "boundary_mse", 8, 1.0),
        ("CVaR99 MSE", "cvar99_mse", 8, 1.0),
        ("p99 MSE", "p99_mse", 8, 1.0),
        ("Interior holes", "interior_hole_fraction", 5, 100.0),
        ("Boundary holes", "boundary_hole_fraction", 5, 100.0),
        ("Outside render max", "outside_max_abs", 8, 1.0),
        ("Outside coverage max", "outside_coverage_max", 8, 1.0),
    )
    rows = []
    for label, key, digits, scale in labels:
        values = factorial["metrics"][key]
        suffix = " pp" if scale == 100.0 else ""

        def value(name: str) -> str:
            return f"{scale * float(values[name]):+.{digits}f}{suffix}"

        rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{value('checkpoint_without_event_color')}</td>"
            f"<td>{value('event_color_without_checkpoint')}</td>"
            f"<td>{value('combined_vs_control')}</td>"
            f"<td>{value('interaction')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build(args: argparse.Namespace) -> None:
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = [_load(label, path, out) for label, path in args.run]
    if len(runs) < 2:
        raise ValueError("comparison needs at least two --run entries")
    fairness = None
    factorial = None
    if getattr(args, "require_factorial", False):
        fairness = _validate_factorial(runs)
        factorial = _factorial_effects(runs)
    target = Path(runs[0]["path"]) / "images/target_foreground.jpg"
    target_relative = _relative(out, target)
    header = "".join(f"<th>{html.escape(run['label'])}</th>" for run in runs)
    factorial_section = ""
    if factorial is not None:
        decision = factorial["combined_decision_rule"]
        decision_class = "ok" if decision["passes"] else "bad"
        decision_text = (
            "besteht"
            if decision["passes"]
            else "besteht nicht"
        )
        factorial_section = f"""
<h2>Kontrollierte Faktoren und tatsächlich committed Mechanismen</h2>
<table><thead><tr><th>Arm</th><th>Pareto-Checkpoints</th>
<th>Event-Color-Solve</th><th>frühere Checkpoints committed</th>
<th>Event-Color-Solves committed</th></tr></thead>
<tbody>{_factor_rows(runs)}</tbody></table>
<p class="ok">Fairness-Prüfung bestanden: Quelle/Seed, Initialisierung, Fit-Konfiguration,
Budgets, Umgebung und ausgeführte Source-Snapshots stimmen überein.</p>
<h2>2×2-Effekte (Delta; bei allen gelisteten Fehlermetriken ist kleiner besser)</h2>
<table><thead><tr><th>Metrik</th><th>Checkpoint bei Color=off</th>
<th>Event-Color bei Checkpoint=off</th><th>Kombiniert vs. Kontrolle</th>
<th>Interaktion</th></tr></thead><tbody>{_effect_rows(factorial)}</tbody></table>
<p class="{decision_class}">Die vorab definierte Combined-vs-Control-Regel
{decision_text}: alle geschützten Metriken nonworse =
<code>{str(decision['all_protected_metrics_nonworse']).lower()}</code>,
mindestens eine strikt besser =
<code>{str(decision['at_least_one_strict_improvement']).lower()}</code>.
Das wissenschaftliche Verdict mit Bild- und Event-Audit steht im
<a href="report.md">Versuchsbericht</a>.</p>
"""
    document = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Janelle C0001 — Safe-Schedule-Vergleich</title>
<style>
:root{{--bg:#0d1016;--panel:#171b24;--line:#343b4a;--text:#edf2f7;--muted:#aeb8c7}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);
font:15px/1.45 system-ui,sans-serif}} main{{max-width:1680px;margin:auto;padding:24px}}
h1,h2,h3{{line-height:1.15}} p{{color:var(--muted)}} a{{color:#7dd3fc}}
code{{color:#fde68a}} .target{{width:min(1200px,100%);background:#000}}
table{{border-collapse:collapse;width:100%;background:var(--panel);margin:18px 0}}
th,td{{border:1px solid var(--line);padding:8px;text-align:right}}
th:first-child{{text-align:left}} .grid-cards{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);padding:14px}}
.card img{{width:100%;display:block;background:#000;margin-top:10px}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(620px,1fr));gap:16px}}
svg{{width:100%;background:var(--panel);border:1px solid var(--line)}}
svg line{{stroke:#718096;stroke-width:1}} svg .grid{{stroke:#303746}}
svg polyline{{fill:none;stroke-width:2}} .chart-title{{fill:var(--text);font-size:16px}}
.tick{{fill:var(--muted);font-size:11px}} .legend-line{{stroke-width:3}}
.scope{{border-left:4px solid #fbbf24;padding:8px 14px;background:#211d12}}
.ok{{color:#a7f3d0}} .bad{{color:#fda4af}}
</style></head><body><main>
<h1>Janelle frame_00008 / C0001 — Safe-Schedule-Vergleich</h1>
<p class="scope">Ein Bild und ein Seed auf CUDA. Wegen atomarer GPU-Akkumulation können
Residual-Rangfolgen und damit spätere Birth-Entscheidungen zwischen Läufen verzweigen.
Die Seite belegt Mechanismusverhalten und diesen konkreten Lauf, keine allgemeine Dominanz.</p>
{factorial_section}
<h2>Ziel</h2><img class="target" src="{html.escape(target_relative)}" alt="target">
<h2>Finale Safe-Metriken</h2>
<table><thead><tr><th>Metrik</th>{header}</tr></thead>
<tbody>{_metric_rows(runs)}</tbody></table>
<section class="charts">
{_svg_chart(runs, "fg", "Foreground PSNR")}
{_svg_chart(runs, "boundary", "Boundary PSNR")}
{_svg_chart(runs, "interior_holes", "Interior undercoverage", percent=True)}
{_svg_chart(runs, "boundary_holes", "Boundary undercoverage", percent=True)}
</section>
<h2>Rekonstruktionen und Fehler ×4</h2>
<section class="grid-cards">{_run_cards(runs)}</section>
</main></body></html>
"""
    payload = {
        "schema": "structsplat.janelle.safe_schedule_comparison.v2",
        "runs": runs,
        "factorial_fairness": fairness,
        "factorial_effects": factorial,
    }
    _atomic_text(out / "comparison.json", json.dumps(payload, indent=2, sort_keys=True))
    _atomic_text(out / "index.html", document)
    digest = hashlib.sha256(document.encode()).hexdigest()
    print(f"wrote {out / 'index.html'} sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-factorial",
        action="store_true",
        help="require and audit one run for every 2x2 factor combination",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
