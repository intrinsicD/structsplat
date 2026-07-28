#!/usr/bin/env python3
"""Independently audit the Janelle cross-view two-tail diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPOSITORY_ROOT / "runs/janelle_cross_view_tail_diagnostic_20260728"
DEFAULT_REPORT = REPOSITORY_ROOT / "ara/evidence/janelle-cross-view-tail-diagnostic-2026-07-28"
SCHEMA = "structsplat.janelle_cross_view_tail_diagnostic.v1"
DETAIL_KEYS = (
    "detail_highpass_sigma_0_75_mse",
    "detail_highpass_sigma_1_5_mse",
    "detail_highpass_sigma_3_mse",
    "detail_laplacian_mse",
    "detail_residual_mse",
    "detail_sobel_mse",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _reduction(before: float, after: float) -> float:
    return 0.0 if before <= 0.0 else 1.0 - after / before


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []
        self.cards = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if "view-card" in classes:
            self.cards += 1
        for key in ("src", "href"):
            value = values.get(key)
            if value and not value.startswith(("#", "http:", "https:", "data:")):
                self.paths.append(value)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _expected_winner(arms: dict[str, dict[str, Any]]) -> str:
    pursuit = arms["pursuit"]
    error = arms["error_only"]
    pursuit_pass = bool(pursuit["target_reached_common_25hp_20lap"])
    error_pass = bool(error["target_reached_common_25hp_20lap"])
    if pursuit_pass and not error_pass:
        return "pursuit"
    if error_pass and not pursuit_pass:
        return "error_only"
    if pursuit_pass and error_pass:
        pursuit_rows = int(pursuit["activated_rows"])
        error_rows = int(error["activated_rows"])
        if pursuit_rows < error_rows:
            return "pursuit"
        if error_rows < pursuit_rows:
            return "error_only"
        return "tie"
    return "neither"


def run(args: argparse.Namespace) -> None:
    run_root = args.run.resolve()
    report_root = args.report.resolve()
    checks: list[dict[str, Any]] = []
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    requested = [(row["frame"], row["view_id"]) for row in manifest["requested_cells"]]
    _check(
        checks,
        "manifest_complete",
        manifest["status"] == "complete",
        manifest["status"],
    )
    _check(
        checks,
        "manifest_scope",
        (
            len(requested) == 51
            and len(set(requested)) == 51
            and ("frame_00008", "C0001") not in requested
        ),
        {"requested": len(requested), "unique": len(set(requested))},
    )
    _check(
        checks,
        "run_summary_hash",
        _sha256(run_root / "summary.json") == manifest["summary_sha256"],
        manifest["summary_sha256"],
    )

    cells: list[dict[str, Any]] = []
    per_cell_failures: list[dict[str, Any]] = []
    for frame, view in requested:
        cell_dir = run_root / "cells" / frame / view
        result_path = cell_dir / "result.json"
        if not result_path.is_file():
            per_cell_failures.append({"cell": f"{frame}/{view}", "reason": "missing result"})
            continue
        cell = json.loads(result_path.read_text(encoding="utf-8"))
        cell_errors = []
        if cell.get("schema") != f"{SCHEMA}.cell":
            cell_errors.append("schema")
        if not cell.get("eligible"):
            cell_errors.append("ineligible")
        if not all(cell["source"]["binding_checks"].values()):
            cell_errors.append("source_binding")
        source_files = {
            "archive": (
                Path(cell["source"]["archive"]),
                cell["source"]["archive_sha256"],
            ),
            "image": (
                Path(cell["source"]["image"]),
                cell["source"]["image_sha256"],
            ),
            "mask": (
                Path(cell["source"]["mask"]),
                cell["source"]["mask_sha256"],
            ),
            "target": (
                Path(cell["source"]["target"]),
                cell["source"]["target_sha256"],
            ),
            "materialized_mask": (
                Path(cell["source"]["materialized_mask"]),
                cell["source"]["materialized_mask_sha256"],
            ),
            "base": (Path(cell["base"]["path"]), cell["base"]["sha256"]),
        }
        for label, (path, digest) in source_files.items():
            if not path.is_file() or _sha256(path) != digest:
                cell_errors.append(f"{label}_hash")
        fixed_point = cell["base"]["constraint_resolution_adapter"]["fixed_point_check"]
        if any(float(value) != 0.0 for value in fixed_point.values()):
            cell_errors.append("base_constraint_not_fixed")
        if int(cell["base"]["baseline"]["detail_deep_pixels"]) < 4096:
            cell_errors.append("deep_pixels")

        for arm_name in ("pursuit", "error_only"):
            arm = cell["arms"][arm_name]
            if arm.get("schema") != f"{SCHEMA}.arm":
                cell_errors.append(f"{arm_name}_schema")
            expected_bindings = {
                "base_sha256": cell["base"]["sha256"],
                "target_sha256": cell["source"]["target_sha256"],
                "mask_sha256": cell["source"]["materialized_mask_sha256"],
            }
            for key, value in expected_bindings.items():
                if arm["source"].get(key) != value:
                    cell_errors.append(f"{arm_name}_{key}")
            field_path = Path(arm["field"]["path"])
            if not field_path.is_file() or _sha256(field_path) != arm["field"]["sha256"]:
                cell_errors.append(f"{arm_name}_field_hash")
            for image_path in arm["images"].values():
                if not Path(image_path).is_file():
                    cell_errors.append(f"{arm_name}_image")
            if not arm["protected_safe"] or not arm["outside_exact_zero"]:
                cell_errors.append(f"{arm_name}_protection")
            if arm_name == "pursuit" and not arm["inherited_prefix_exact"]:
                cell_errors.append("pursuit_prefix")
            for key in DETAIL_KEYS:
                expected = _reduction(
                    float(arm["baseline"][key]),
                    float(arm["final"][key]),
                )
                if not _close(expected, arm["relative_reductions"][key]):
                    cell_errors.append(f"{arm_name}_{key}_reduction")
            expected_target = bool(
                arm["protected_safe"]
                and arm["relative_reductions"]["detail_highpass_sigma_1_5_mse"] >= 0.25
                and arm["relative_reductions"]["detail_laplacian_mse"] >= 0.20
            )
            if expected_target != arm["target_reached_common_25hp_20lap"]:
                cell_errors.append(f"{arm_name}_target_rule")
            expected_rows = int(arm["field"]["rows"]) - int(cell["base"]["rows"])
            if expected_rows != int(arm["activated_rows"]):
                cell_errors.append(f"{arm_name}_row_accounting")
            if arm_name == "pursuit" and (bool(arm["tail"]["target_reached"]) != expected_target):
                cell_errors.append("pursuit_native_target")
        if _expected_winner(cell["arms"]) != cell["comparison"]["winner"]:
            cell_errors.append("winner")
        if cell_errors:
            per_cell_failures.append({"cell": f"{frame}/{view}", "reasons": cell_errors})
        cells.append(cell)

    _check(
        checks,
        "all_51_cell_artifacts_valid",
        len(cells) == 51 and not per_cell_failures,
        per_cell_failures,
    )

    pursuit = [cell["arms"]["pursuit"] for cell in cells]
    error = [cell["arms"]["error_only"] for cell in cells]
    winners = {
        name: sum(cell["comparison"]["winner"] == name for cell in cells)
        for name in ("pursuit", "error_only", "tie", "neither")
    }
    recomputed = {
        "completed": len(cells),
        "pursuit_target_reached": sum(row["target_reached_common_25hp_20lap"] for row in pursuit),
        "error_target_reached": sum(row["target_reached_common_25hp_20lap"] for row in error),
        "pursuit_protected_safe": sum(row["protected_safe"] for row in pursuit),
        "error_protected_safe": sum(row["protected_safe"] for row in error),
        "pursuit_median_rows": float(np.median([row["activated_rows"] for row in pursuit])),
        "error_median_rows": float(np.median([row["activated_rows"] for row in error])),
        "pursuit_median_highpass": float(
            np.median(
                [row["relative_reductions"]["detail_highpass_sigma_1_5_mse"] for row in pursuit]
            )
        ),
        "error_median_highpass": float(
            np.median(
                [row["relative_reductions"]["detail_highpass_sigma_1_5_mse"] for row in error]
            )
        ),
        "pursuit_median_laplacian": float(
            np.median([row["relative_reductions"]["detail_laplacian_mse"] for row in pursuit])
        ),
        "error_median_laplacian": float(
            np.median([row["relative_reductions"]["detail_laplacian_mse"] for row in error])
        ),
        "pursuit_total_rows": sum(row["activated_rows"] for row in pursuit),
        "error_total_rows": sum(row["activated_rows"] for row in error),
        "winners": winners,
    }
    aggregate_expected = {
        "completed": summary["scope"]["completed_cells"],
        "pursuit_target_reached": summary["arms"]["pursuit"]["target_reached"],
        "error_target_reached": summary["arms"]["error_only"]["target_reached"],
        "pursuit_protected_safe": summary["arms"]["pursuit"]["protected_safe"],
        "error_protected_safe": summary["arms"]["error_only"]["protected_safe"],
        "pursuit_median_rows": summary["arms"]["pursuit"]["added_rows"]["median"],
        "error_median_rows": summary["arms"]["error_only"]["added_rows"]["median"],
        "pursuit_median_highpass": summary["arms"]["pursuit"]["highpass_reduction"]["median"],
        "error_median_highpass": summary["arms"]["error_only"]["highpass_reduction"]["median"],
        "pursuit_median_laplacian": summary["arms"]["pursuit"]["laplacian_reduction"]["median"],
        "error_median_laplacian": summary["arms"]["error_only"]["laplacian_reduction"]["median"],
        "pursuit_total_rows": summary["arms"]["pursuit"]["total_added_rows"],
        "error_total_rows": summary["arms"]["error_only"]["total_added_rows"],
        "winners": summary["winners"],
    }
    aggregate_match = all(
        (
            _close(recomputed[key], aggregate_expected[key])
            if isinstance(recomputed[key], float)
            else recomputed[key] == aggregate_expected[key]
        )
        for key in recomputed
    )
    _check(
        checks,
        "independent_aggregate_reconciliation",
        aggregate_match,
        {"recomputed": recomputed, "reported": aggregate_expected},
    )

    with (report_root / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_keys = {(row["frame"], row["view_id"]) for row in csv_rows}
    _check(
        checks,
        "tidy_csv_complete",
        len(csv_rows) == 51 and csv_keys == set(requested),
        {"rows": len(csv_rows), "unique_keys": len(csv_keys)},
    )
    report_summary = json.loads((report_root / "summary.json").read_text(encoding="utf-8"))
    _check(
        checks,
        "report_summary_matches_run",
        report_summary == summary,
        {
            "run_sha256": _sha256(run_root / "summary.json"),
            "report_sha256": _sha256(report_root / "summary.json"),
        },
    )
    parser = _Links()
    parser.feed((report_root / "index.html").read_text(encoding="utf-8"))
    missing_links = []
    for relative in parser.paths:
        path = (report_root / relative).resolve()
        if not path.exists():
            missing_links.append(relative)
    _check(
        checks,
        "html_cards_and_links",
        parser.cards == 51 and not missing_links,
        {
            "cards": parser.cards,
            "links": len(parser.paths),
            "missing": missing_links,
        },
    )
    copied_images = list((report_root / "images").glob("*/*/*.png"))
    _check(
        checks,
        "report_crop_inventory",
        len(copied_images) == 51 * 7,
        {"images": len(copied_images), "expected": 51 * 7},
    )
    artifact = json.loads((report_root / "artifact.json").read_text(encoding="utf-8"))
    artifact_checks = {
        "summary": (artifact["summary_sha256"] == _sha256(report_root / "summary.json")),
        "csv": (artifact["comparison_csv_sha256"] == _sha256(report_root / "comparison.csv")),
        "index": (artifact["index_sha256"] == _sha256(report_root / "index.html")),
        "run": (artifact["run_markdown_sha256"] == _sha256(report_root / "run.md")),
        "source_manifest": (
            artifact["source_run_manifest_sha256"] == _sha256(run_root / "manifest.json")
        ),
    }
    _check(
        checks,
        "artifact_hashes",
        all(artifact_checks.values()),
        artifact_checks,
    )

    passed = all(check["passed"] for check in checks)
    payload = {
        "schema": f"{SCHEMA}.audit",
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed": recomputed,
        "interpretation_guardrails": {
            "independent_scene_confirmation": False,
            "same_capture_correlated": True,
            "archived_mask_contained_bases": True,
            "natural_not_equal_row_budgets": True,
        },
        "audit_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    audit_path = report_root / "audit.json"
    _atomic_json(audit_path, payload)
    artifact_path = report_root / "artifact.json"
    artifact["audit_sha256"] = _sha256(audit_path)
    artifact["audit_script_sha256"] = payload["audit_script"]["sha256"]
    _atomic_json(artifact_path, artifact)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
