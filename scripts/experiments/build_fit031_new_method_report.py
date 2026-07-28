#!/usr/bin/env python3
"""Build the canonical report artifact for the FIT-031 stage comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_ROOT = REPOSITORY_ROOT / "ara/evidence/fit031-new-method-stages-janelle-2026-07-28"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _dataset_source(
    dataset: str,
    label: str,
    description: str,
    metric_definitions: list[str],
) -> dict:
    path = f"ara/evidence/fit031-new-method-stages-janelle-2026-07-28/datasets/{dataset}.json"
    return {
        "id": f"{dataset}_source",
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{path}')",
            "description": description,
            "tables_used": [path],
            "metric_definitions": metric_definitions,
        },
    }


def _same_base_rows(comparison: dict) -> list[dict]:
    same_base = comparison["published_same_base_comparison"]
    labels = {
        "error_tail": "FIT031 error-only tail",
        "orthogonal_pursuit": "Orthogonal pursuit",
    }
    return [
        {
            "method": labels[key],
            "added_rows": int(values["added_rows"]),
            "deep_rows": int(values["deep_rows"]),
            "foreground_psnr_gain_db": float(values["foreground_psnr_gain_db"]),
            "highpass_reduction": float(values["sigma_1_5_highpass_reduction"]),
            "laplacian_reduction": float(values["laplacian_reduction"]),
            "lpips_reduction": float(values["lpips_reduction"]),
            "seconds": float(values["seconds"]),
        }
        for key, values in same_base.items()
        if key in labels
    ]


def build_artifact(comparison: dict) -> dict:
    title = "StructSplat Fine-Detail Stage Comparison"
    generated_at = comparison["created_utc"]
    endpoints = comparison["stage_endpoints"]
    same_base_rows = _same_base_rows(comparison)
    pursuit = next(row for row in same_base_rows if row["method"] == "Orthogonal pursuit")
    error_tail = next(row for row in same_base_rows if row["method"] == "FIT031 error-only tail")

    stage_endpoint_long = []
    for endpoint in endpoints:
        for metric, label in (
            ("sigma_1_5_highpass_reduction", "Sigma-1.5 high-pass"),
            ("laplacian_reduction", "Laplacian"),
        ):
            stage_endpoint_long.append(
                {
                    "stage_order": int(endpoint["stage_order"]),
                    "task": endpoint["task"],
                    "method": endpoint["method"],
                    "metric": label,
                    "reduction": float(endpoint[metric]),
                    "added_rows": int(endpoint["added_rows"]),
                    "gate_status": endpoint["gate_status"],
                    "evidence_class": endpoint["evidence_class"],
                    "target_reached": bool(endpoint["target_reached"]),
                }
            )

    stage_table = [
        {
            "stage_order": int(endpoint["stage_order"]),
            "task": endpoint["task"],
            "method": endpoint["method"],
            "added_rows": int(endpoint["added_rows"]),
            "highpass_reduction": float(endpoint["sigma_1_5_highpass_reduction"]),
            "laplacian_reduction": float(endpoint["laplacian_reduction"]),
            "foreground_psnr_gain_db": float(endpoint["foreground_psnr_gain_db"]),
            "gate_status": endpoint["gate_status"],
            "evidence_class": endpoint["evidence_class"],
        }
        for endpoint in endpoints
    ]
    stage_curves = [
        {
            "task": row["task"],
            "added_rows": int(row["added_rows"]),
            "highpass_reduction": float(row["sigma_1_5_highpass_reduction"]),
            "laplacian_reduction": float(row["laplacian_reduction"]),
            "protected_safe": bool(row["protected_safe"]),
        }
        for row in comparison["stage_curves"]
    ]
    same_base_detail_long = []
    for row in same_base_rows:
        for metric, field in (
            ("Sigma-1.5 high-pass", "highpass_reduction"),
            ("Laplacian", "laplacian_reduction"),
            ("LPIPS", "lpips_reduction"),
        ):
            same_base_detail_long.append(
                {
                    "method": row["method"],
                    "metric": metric,
                    "reduction": row[field],
                    "added_rows": row["added_rows"],
                    "deep_rows": row["deep_rows"],
                    "seconds": row["seconds"],
                }
            )
    audit_checks = [
        {
            "check": check.replace("_", " "),
            "passed": bool(passed),
        }
        for check, passed in comparison["checks"].items()
    ]
    summary_metrics = [
        {
            "audit_checks_passed": sum(1 for value in comparison["checks"].values() if value),
            "audit_checks_total": len(comparison["checks"]),
            "rerun_cells": len(
                comparison["evidence_inventory"]["rerun_due_to_missing_durable_machine_results"]
            ),
            "pursuit_added_rows": pursuit["added_rows"],
            "pursuit_highpass_reduction": pursuit["highpass_reduction"],
            "pursuit_laplacian_reduction": pursuit["laplacian_reduction"],
            "error_tail_added_rows": error_tail["added_rows"],
            "error_tail_foreground_psnr_gain_db": error_tail["foreground_psnr_gain_db"],
            "pursuit_foreground_psnr_gain_db": pursuit["foreground_psnr_gain_db"],
        }
    ]

    evidence_sources = [
        {
            "id": "comparison_audit",
            "label": "FIT031 new-method stage audit",
            "path": ("ara/evidence/fit031-new-method-stages-janelle-2026-07-28/comparison.json"),
        },
        {
            "id": "fit031_audit",
            "label": "FIT031 original crop audit",
            "path": ("ara/evidence/fit031-error-only-tail-janelle-2026-07-27/audit.json"),
        },
        {
            "id": "fit040_audit",
            "label": "FIT040/FIT041 exact-same-base audit",
            "path": ("ara/evidence/fit040-orthogonal-detail-pursuit-janelle-2026-07-28/audit.json"),
        },
    ]
    dataset_sources = [
        _dataset_source(
            "summary_metrics",
            "Reviewed headline metrics",
            "Loads the reviewed headline metrics shown in report cards.",
            [
                "Reduction fields are relative reductions, stored as fractions.",
                "Foreground PSNR gain is candidate minus baseline in decibels.",
            ],
        ),
        _dataset_source(
            "stage_endpoint_long",
            "Reviewed stage endpoints",
            "Loads tidy deep-detail endpoint metrics for each stage.",
            [
                "Reduction = 1 - candidate MSE / baseline MSE.",
                "Added rows count accepted terminal capacity only.",
            ],
        ),
        _dataset_source(
            "stage_curves",
            "Reviewed FIT037/FIT038 curves",
            "Loads protected-safe row-budget curves for static and iterative pursuit.",
            [
                "High-pass reduction uses the deep sigma-1.5 residual MSE.",
                "Laplacian reduction uses deep Laplacian residual MSE.",
            ],
        ),
        _dataset_source(
            "same_base_detail_long",
            "Reviewed exact-same-base detail metrics",
            "Loads tidy FIT031 and pursuit detail/perceptual reductions.",
            [
                "LPIPS reduction uses raw unclamped LPIPS.",
                "Both methods start from the same 11,000-row field.",
            ],
        ),
        _dataset_source(
            "same_base_rows",
            "Reviewed exact-same-base method rows",
            "Loads method-level rows, timings, and quality metrics.",
            [
                "Tail seconds exclude the shared base fit.",
                "Deep rows have support on the declared deep-detail region.",
            ],
        ),
        _dataset_source(
            "stage_table",
            "Reviewed stage endpoint ledger",
            "Loads the exact stage ledger shown in the report table.",
            [
                "Cross-tier endpoint values are descriptive only.",
                "Gate status is the predeclared stage disposition.",
            ],
        ),
        _dataset_source(
            "audit_checks",
            "Independent audit checks",
            "Loads the source, binding, metric, and gate audit outcomes.",
            ["Passed is true only when the independent check succeeds."],
        ),
    ]
    sources = [*evidence_sources, *dataset_sources]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "A source-bound technical comparison of FIT032--041 against "
            "FIT031, with explicit evidence tiers and stage judgments."
        ),
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "evidence_card",
                "description": "Independent validation and rerun scope.",
                "dataset": "summary_metrics",
                "sourceId": "summary_metrics_source",
                "metrics": [
                    {
                        "label": "Audit checks passed",
                        "field": "audit_checks_passed",
                        "format": "number",
                    },
                    {
                        "label": "Rerun cells",
                        "field": "rerun_cells",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "pursuit_card",
                "description": ("Published exact-same-base fine-detail result."),
                "dataset": "summary_metrics",
                "sourceId": "summary_metrics_source",
                "metrics": [
                    {
                        "label": "Pursuit high-pass reduction",
                        "field": "pursuit_highpass_reduction",
                        "format": "percent",
                    },
                    {
                        "label": "Pursuit rows",
                        "field": "pursuit_added_rows",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "error_tail_card",
                "description": ("Published exact-same-base FIT031 global-quality result."),
                "dataset": "summary_metrics",
                "sourceId": "summary_metrics_source",
                "metrics": [
                    {
                        "label": "FIT031 FG PSNR gain",
                        "field": "error_tail_foreground_psnr_gain_db",
                        "format": "number",
                    },
                    {
                        "label": "FIT031 rows",
                        "field": "error_tail_added_rows",
                        "format": "number",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "stage_endpoint_chart",
                "title": "Deep-detail reduction at each stage endpoint",
                "subtitle": (
                    "Iterative reselection improves the curve; exact-site "
                    "exclusion is the only tested stage that reaches both "
                    "fine-detail thresholds."
                ),
                "type": "bar",
                "dataset": "stage_endpoint_long",
                "sourceId": "stage_endpoint_long_source",
                "valueFormat": "percent",
                "encodings": {
                    "x": {
                        "field": "task",
                        "type": "nominal",
                        "label": "Stage",
                    },
                    "y": {
                        "field": "reduction",
                        "type": "quantitative",
                        "label": "Relative MSE reduction",
                        "format": "percent",
                    },
                    "color": {
                        "field": "metric",
                        "type": "nominal",
                        "label": "Metric",
                    },
                    "tooltip": [
                        {
                            "field": "added_rows",
                            "type": "quantitative",
                            "label": "Added rows",
                            "format": "number",
                        },
                        {
                            "field": "gate_status",
                            "type": "nominal",
                            "label": "Gate",
                        },
                    ],
                },
            },
            {
                "id": "stage_curve_chart",
                "title": ("Sigma-1.5 high-pass reduction over added rows"),
                "subtitle": (
                    "Iterative FIT038 stays above static FIT037 after the "
                    "first wave but plateaus below the 25% target."
                ),
                "type": "line",
                "dataset": "stage_curves",
                "sourceId": "stage_curves_source",
                "valueFormat": "percent",
                "encodings": {
                    "x": {
                        "field": "added_rows",
                        "type": "quantitative",
                        "label": "Added rows",
                    },
                    "y": {
                        "field": "highpass_reduction",
                        "type": "quantitative",
                        "label": "High-pass reduction",
                        "format": "percent",
                    },
                    "color": {
                        "field": "task",
                        "type": "nominal",
                        "label": "Stage",
                    },
                    "tooltip": [
                        {
                            "field": "laplacian_reduction",
                            "type": "quantitative",
                            "label": "Laplacian reduction",
                            "format": "percent",
                        }
                    ],
                },
            },
            {
                "id": "same_base_detail_chart",
                "title": ("Exact-same-base detail and perceptual reductions"),
                "subtitle": (
                    "Orthogonal pursuit dominates the deep-detail and LPIPS "
                    "objectives while using fewer rows."
                ),
                "type": "bar",
                "dataset": "same_base_detail_long",
                "sourceId": "same_base_detail_long_source",
                "valueFormat": "percent",
                "encodings": {
                    "x": {
                        "field": "metric",
                        "type": "nominal",
                        "label": "Metric",
                    },
                    "y": {
                        "field": "reduction",
                        "type": "quantitative",
                        "label": "Relative reduction",
                        "format": "percent",
                    },
                    "color": {
                        "field": "method",
                        "type": "nominal",
                        "label": "Method",
                    },
                    "tooltip": [
                        {
                            "field": "added_rows",
                            "type": "quantitative",
                            "label": "Added rows",
                            "format": "number",
                        },
                        {
                            "field": "deep_rows",
                            "type": "quantitative",
                            "label": "Deep rows",
                            "format": "number",
                        },
                    ],
                },
            },
            {
                "id": "same_base_psnr_chart",
                "title": "Exact-same-base foreground PSNR gain",
                "subtitle": (
                    "FIT031 wins the global foreground metric; pursuit wins "
                    "the separately measured fine-detail objectives."
                ),
                "type": "bar",
                "dataset": "same_base_rows",
                "sourceId": "same_base_rows_source",
                "valueFormat": "number",
                "encodings": {
                    "x": {
                        "field": "method",
                        "type": "nominal",
                        "label": "Method",
                    },
                    "y": {
                        "field": "foreground_psnr_gain_db",
                        "type": "quantitative",
                        "label": "Foreground PSNR gain (dB)",
                        "format": "number",
                    },
                    "tooltip": [
                        {
                            "field": "added_rows",
                            "type": "quantitative",
                            "label": "Added rows",
                            "format": "number",
                        },
                        {
                            "field": "seconds",
                            "type": "quantitative",
                            "label": "Tail seconds",
                            "format": "number",
                        },
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "stage_table",
                "title": "Stage endpoint ledger",
                "subtitle": ("Exact values, evidence class, and gate disposition."),
                "dataset": "stage_table",
                "sourceId": "stage_table_source",
                "defaultSort": {
                    "field": "stage_order",
                    "direction": "asc",
                },
                "density": "dense",
                "layout": "full",
                "columns": [
                    {
                        "field": "stage_order",
                        "label": "Order",
                        "type": "number",
                    },
                    {"field": "task", "label": "Task", "type": "text"},
                    {
                        "field": "method",
                        "label": "Method",
                        "type": "text",
                    },
                    {
                        "field": "added_rows",
                        "label": "Added rows",
                        "format": "number",
                    },
                    {
                        "field": "highpass_reduction",
                        "label": "High-pass",
                        "format": "percent",
                    },
                    {
                        "field": "laplacian_reduction",
                        "label": "Laplacian",
                        "format": "percent",
                    },
                    {
                        "field": "foreground_psnr_gain_db",
                        "label": "FG PSNR gain (dB)",
                        "format": "number",
                    },
                    {
                        "field": "gate_status",
                        "label": "Gate",
                        "type": "text",
                    },
                    {
                        "field": "evidence_class",
                        "label": "Evidence class",
                        "type": "text",
                    },
                ],
            },
            {
                "id": "same_base_table",
                "title": "Exact-same-base FIT031 and pursuit results",
                "subtitle": (
                    "Published full-frame control values used for the final method judgment."
                ),
                "dataset": "same_base_rows",
                "sourceId": "same_base_rows_source",
                "defaultSort": {
                    "field": "added_rows",
                    "direction": "asc",
                },
                "columns": [
                    {
                        "field": "method",
                        "label": "Method",
                        "type": "text",
                    },
                    {
                        "field": "added_rows",
                        "label": "Added rows",
                        "format": "number",
                    },
                    {
                        "field": "deep_rows",
                        "label": "Deep rows",
                        "format": "number",
                    },
                    {
                        "field": "highpass_reduction",
                        "label": "High-pass",
                        "format": "percent",
                    },
                    {
                        "field": "laplacian_reduction",
                        "label": "Laplacian",
                        "format": "percent",
                    },
                    {
                        "field": "lpips_reduction",
                        "label": "LPIPS",
                        "format": "percent",
                    },
                    {
                        "field": "foreground_psnr_gain_db",
                        "label": "FG PSNR gain (dB)",
                        "format": "number",
                    },
                    {
                        "field": "seconds",
                        "label": "Tail seconds",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "audit_table",
                "title": "Independent audit checks",
                "subtitle": "All source, binding, metric, and gate checks.",
                "dataset": "audit_checks",
                "sourceId": "audit_checks_source",
                "defaultSort": {
                    "field": "check",
                    "direction": "asc",
                },
                "columns": [
                    {
                        "field": "check",
                        "label": "Check",
                        "type": "text",
                    },
                    {
                        "field": "passed",
                        "label": "Passed",
                        "type": "text",
                    },
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": (
                    f"# {title}\n\n"
                    "A source-bound comparison of the newly added "
                    "fine-detail stages against FIT031."
                ),
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    "Only FIT032–038 were rerun because their durable "
                    "machine-readable results were missing. FIT039–041 were "
                    "reused after their hashes and all 17 committed audit "
                    "checks passed. The new 18-check comparison audit passes "
                    "in full."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "evidence_card",
                    "pursuit_card",
                    "error_tail_card",
                ],
            },
            {
                "id": "stage_judgment",
                "type": "markdown",
                "body": (
                    "## Stage judgment\n\n"
                    "**Keep FIT032–037 as negative or diagnostic stages. "
                    "Keep FIT038 as the useful iterative mechanism. Retain "
                    "FIT039 exact-site exclusion and FIT040 integration as "
                    "the fine-detail path.** No stage authorizes a default "
                    "change yet."
                ),
            },
            {
                "id": "stage_endpoint_chart_block",
                "type": "chart",
                "chartId": "stage_endpoint_chart",
            },
            {
                "id": "stage_endpoint_explanation",
                "type": "markdown",
                "body": (
                    "FIT033 establishes that high-pass-targeted placement "
                    "and a partial color solve are useful. FIT034–036 add "
                    "spectral mixing, affine colors, and ridge geometry, but "
                    "none clears its stage target. Static FIT037 saturates "
                    "early; iterative FIT038 improves the endpoint but still "
                    "misses 25% high-pass and 20% Laplacian at 2,048 rows."
                ),
            },
            {
                "id": "method_progression",
                "type": "markdown",
                "body": (
                    "## Static versus iterative pursuit\n\n"
                    "FIT037 and FIT038 use the same target, base field, "
                    "protected gate, and row budgets. The difference is "
                    "reselection: FIT038 recomputes residual sites after "
                    "each accepted 128-row wave."
                ),
            },
            {
                "id": "stage_curve_chart_block",
                "type": "chart",
                "chartId": "stage_curve_chart",
            },
            {
                "id": "stage_curve_explanation",
                "type": "markdown",
                "body": (
                    "The curves coincide at 128 rows. By 2,048 rows, FIT038 "
                    "reaches 20.90% high-pass reduction versus 15.73% for "
                    "FIT037, confirming that iterative reselection matters. "
                    "Exact-site exclusion then removes repeated placements "
                    "and reaches the published target at 768 rows."
                ),
            },
            {
                "id": "fit031_comparison",
                "type": "markdown",
                "body": (
                    "## FIT031 versus orthogonal pursuit\n\n"
                    "The valid direct comparison is FIT041's full-frame "
                    "exact-same-base control, not FIT031's original crop. "
                    "On that control, the methods optimize different "
                    "objectives: FIT031 improves global foreground PSNR; "
                    "orthogonal pursuit improves deep detail and LPIPS."
                ),
            },
            {
                "id": "same_base_detail_chart_block",
                "type": "chart",
                "chartId": "same_base_detail_chart",
            },
            {
                "id": "same_base_detail_explanation",
                "type": "markdown",
                "body": (
                    "Orthogonal pursuit uses 768 deep rows and reduces "
                    "high-pass, Laplacian, and raw LPIPS by 25.93%, 27.32%, "
                    "and 10.46%. FIT031 uses 2,777 rows, places zero rows in "
                    "the deep region, and changes the two deep-detail MSEs "
                    "by effectively zero."
                ),
            },
            {
                "id": "same_base_psnr_chart_block",
                "type": "chart",
                "chartId": "same_base_psnr_chart",
            },
            {
                "id": "same_base_psnr_explanation",
                "type": "markdown",
                "body": (
                    "FIT031 gains 0.321 dB foreground PSNR versus 0.034 dB "
                    "for pursuit. This is not a contradiction: the first "
                    "method spends capacity on foreground MAE and boundary "
                    "error, while pursuit spends capacity on deep "
                    "high-frequency residuals."
                ),
            },
            {
                "id": "same_base_table_block",
                "type": "table",
                "tableId": "same_base_table",
                "layout": "full",
            },
            {
                "id": "stage_ledger",
                "type": "markdown",
                "body": (
                    "## Stage ledger\n\n"
                    "The endpoint table keeps evidence tier, row count, "
                    "metrics, and gate outcome together. Cross-tier values "
                    "are descriptive; only the FIT031/FIT041 control is an "
                    "exact-base method comparison."
                ),
            },
            {
                "id": "stage_table_block",
                "type": "table",
                "tableId": "stage_table",
                "layout": "full",
            },
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## Scope, data, and definitions\n\n"
                    "Source: masked Janelle frame_00008/C0001 at 1200×1038, "
                    "seed 0. The primary objective is relative reduction in "
                    "sigma-1.5 high-pass RGB residual MSE on pixels deeper "
                    "than mask margin + 6; the orthogonal objective is "
                    "Laplacian residual MSE. Protected safety covers "
                    "foreground, boundary, tail-risk, holes, finiteness, and "
                    "outside-mask containment."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "body": (
                    "## Methodology\n\n"
                    "The published PNG target was regenerated with Pillow "
                    "12.3 and verified byte-for-byte. A Pillow 11 resize was "
                    "quarantined. FIT032–038 share one 10,816-row RTX4090 "
                    "base; their reported reductions were independently "
                    "recomputed from before/after MSE. FIT039–041 use the "
                    "published 11,000-row RTX3050 base and its cold/audit "
                    "artifacts."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and robustness\n\n"
                    "This remains one exposed image, one seed per evidence "
                    "tier, and two GPU/software trajectories. The RTX4090 "
                    "replication is same-target and common-base within its "
                    "tier, but not an exact-base replay of the RTX3050 "
                    "published tier. Results are not rate-matched, and no "
                    "default, generality, or equal-rate claim is authorized."
                ),
            },
            {
                "id": "fit035_correction",
                "type": "markdown",
                "body": (
                    "## FIT035 evidence correction\n\n"
                    "The first rerun exposed a harness error: an identity "
                    "A/A render was rejected solely because the strict gate "
                    "requires material gain. The check now permits exactly "
                    "`no_material_gain` while still failing on any protected "
                    "regression. The corrected rerun passes A/A and remains "
                    "scientifically negative."
                ),
            },
            {
                "id": "audit_checks_block",
                "type": "table",
                "tableId": "audit_table",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Treat FIT039/040 as the sole fine-detail candidate.\n"
                    "2. Keep FIT031 only when global/boundary cleanup is the "
                    "declared objective.\n"
                    "3. Run a preregistered multi-image, multi-seed, "
                    "equal-rate comparison before changing defaults.\n"
                    "4. If a hybrid is tested, allocate separate boundary "
                    "and deep-detail budgets rather than merging objectives "
                    "post hoc."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "Does exact-site exclusion generalize beyond this frame? "
                    "At equal encoded bytes, is a boundary/deep hybrid "
                    "Pareto-superior to either tail alone? Are the 768 "
                    "pursuit rows stable across seeds and renderer hardware?"
                ),
            },
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "summary_metrics": summary_metrics,
            "stage_endpoint_long": stage_endpoint_long,
            "stage_curves": stage_curves,
            "same_base_detail_long": same_base_detail_long,
            "same_base_rows": same_base_rows,
            "stage_table": stage_table,
            "audit_checks": audit_checks,
        },
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
        "package_info": {
            "originUrl": ("artifact://structsplat-fit031-new-method-stages"),
            "controls": {
                "edit": False,
                "refresh": False,
                "persistence": False,
                "copyAsImage": False,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "comparison.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EVIDENCE_ROOT / "artifact.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    artifact = build_artifact(_read(arguments.comparison))
    for dataset, rows in artifact["snapshot"]["datasets"].items():
        _atomic_json(
            arguments.out.parent / "datasets" / f"{dataset}.json",
            rows,
        )
    _atomic_json(arguments.out, artifact)
    print(
        json.dumps(
            {
                "out": str(arguments.out),
                "blocks": len(artifact["manifest"]["blocks"]),
                "datasets": len(artifact["snapshot"]["datasets"]),
            },
            indent=2,
        )
    )
