#!/usr/bin/env python3
"""Audit and package the FIT-032--FIT-041 Janelle stage comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "runs/fit031_new_methods_comparison_20260728"
DEFAULT_FIT031_ROOT = REPOSITORY_ROOT / "ara/evidence/fit031-error-only-tail-janelle-2026-07-27"
DEFAULT_PUBLISHED_ROOT = (
    REPOSITORY_ROOT / "ara/evidence/fit040-orthogonal-detail-pursuit-janelle-2026-07-28"
)
DEFAULT_OUT = (
    REPOSITORY_ROOT / "ara/evidence/fit031-new-method-stages-janelle-2026-07-28" / "comparison.json"
)
EXPECTED_TARGET_SHA256 = "b11b3a3b063e5630581f6a15ee09527216522b19bee1002a641ddbcc39443db3"
EXPECTED_MASK_SHA256 = "94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3"
RESULT_SCHEMAS = {
    "fit032": "structsplat.fit032.janelle-dipole-screen.v1",
    "fit033": "structsplat.fit033.janelle-highpass-partial-solve.v1",
    "fit034": "structsplat.fit034.janelle-weight-screen.v1",
    "fit035": "structsplat.fit035.janelle-affine-screen.v1",
    "fit036": "structsplat.fit036.janelle-ridge-screen.v1",
    "fit037": "structsplat.fit037.janelle-minimum-detail-rows.v1",
    "fit038": "structsplat.fit038.janelle-detail-pursuit.v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _base_hash(payload: dict) -> str:
    if "base_field" in payload:
        return str(payload["base_field"]["sha256"])
    return str(payload["base_field_sha256"])


def _find_row(rows: list[dict], **values) -> dict:
    matches = [row for row in rows if all(row.get(key) == value for key, value in values.items())]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {values}, found {len(matches)}")
    return matches[0]


def _reductions_match(payload: dict) -> bool:
    baseline = payload["baseline"]
    highpass_before = float(baseline["detail_highpass_sigma_1_5_mse"])
    laplacian_before = float(baseline["detail_laplacian_mse"])
    for row in payload["rows"]:
        if "sigma_1_5_reduction" not in row:
            continue
        metrics = row["metrics"]
        expected_highpass = 1.0 - (
            float(metrics["detail_highpass_sigma_1_5_mse"]) / highpass_before
        )
        expected_laplacian = 1.0 - (float(metrics["detail_laplacian_mse"]) / laplacian_before)
        if not math.isclose(
            float(row["sigma_1_5_reduction"]),
            expected_highpass,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        if not math.isclose(
            float(row["laplacian_reduction"]),
            expected_laplacian,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
    return True


def _detail_reductions(payload: dict, row: dict) -> tuple[float, float]:
    baseline = payload["baseline"]
    metrics = row["metrics"]
    highpass = 1.0 - (
        float(metrics["detail_highpass_sigma_1_5_mse"])
        / float(baseline["detail_highpass_sigma_1_5_mse"])
    )
    laplacian = 1.0 - (
        float(metrics["detail_laplacian_mse"]) / float(baseline["detail_laplacian_mse"])
    )
    return highpass, laplacian


def _endpoint(
    *,
    order: int,
    task: str,
    method: str,
    evidence_class: str,
    row: dict | None,
    added_rows: int,
    highpass_reduction: float,
    laplacian_reduction: float,
    gate_status: str,
    target_reached: bool,
    source_artifact: str,
) -> dict:
    if row is None:
        foreground_psnr_gain_db = 0.0
        protected_safe = True
    else:
        baseline_psnr = float(row["_baseline_foreground_psnr_db"])
        foreground_psnr_gain_db = float(row["metrics"]["foreground_psnr_db"]) - baseline_psnr
        protected_safe = bool(row["protected_safe"])
    return {
        "stage_order": order,
        "task": task,
        "method": method,
        "evidence_class": evidence_class,
        "added_rows": added_rows,
        "sigma_1_5_highpass_reduction": highpass_reduction,
        "laplacian_reduction": laplacian_reduction,
        "foreground_psnr_gain_db": foreground_psnr_gain_db,
        "protected_safe": protected_safe,
        "target_reached": target_reached,
        "gate_status": gate_status,
        "source_artifact": source_artifact,
    }


def _with_baseline(row: dict, payload: dict) -> dict:
    copied = dict(row)
    copied["_baseline_foreground_psnr_db"] = float(payload["baseline"]["foreground_psnr_db"])
    return copied


def _source_snapshot_matches(payload: dict) -> bool:
    entries = payload.get("sources")
    if entries is None:
        entries = payload.get("repository", {}).get("source_snapshot", [])
    if not entries:
        return False
    return all(
        (REPOSITORY_ROOT / entry["path"]).is_file()
        and _sha256(REPOSITORY_ROOT / entry["path"]) == entry["sha256"]
        for entry in entries
    )


def _baselines_match(
    reference: dict,
    candidate: dict,
    keys: tuple[str, ...],
) -> bool:
    for key in keys:
        expected = reference[key]
        actual = candidate[key]
        if key == "n_gaussians":
            if int(actual) != int(expected):
                return False
        elif not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            return False
    return True


def _archive_inputs(archive_dir: Path, inputs: dict[str, Path]) -> dict:
    archive_dir.mkdir(parents=True, exist_ok=True)
    ledger = {}
    for name, source in inputs.items():
        destination = archive_dir / f"{name}.json"
        shutil.copy2(source, destination)
        ledger[name] = {
            "source_path": _relative(source),
            "source_sha256": _sha256(source),
            "archived_path": _relative(destination),
            "archived_sha256": _sha256(destination),
        }
    return ledger


def run(args: argparse.Namespace) -> None:
    base_job = args.run_root / "base_exact/runs/current/C0001/seed_0"
    result_paths = {
        f"fit{number:03d}": (
            args.run_root
            / ("fit035_final3" if number == 35 else f"fit{number:03d}")
            / "result.json"
        )
        for number in range(32, 39)
    }
    support_paths = {
        "base_config": base_job / "config.json",
        "base_result": base_job / "result.json",
        "fit031_audit": args.fit031_root / "audit.json",
        "fit040_audit": args.published_root / "audit.json",
        "fit040_production_result": (args.published_root / "production_result.json"),
        "fit041_equal_base_result": (args.published_root / "equal_base_error_tail_result.json"),
    }
    all_inputs = {**support_paths, **result_paths}
    payloads = {name: _read(path) for name, path in all_inputs.items()}
    results = {name: payloads[name] for name in result_paths}
    base_config = payloads["base_config"]
    base_result = payloads["base_result"]
    fit031_audit = payloads["fit031_audit"]
    fit040_audit = payloads["fit040_audit"]
    production = payloads["fit040_production_result"]

    base_field_path = base_job / "field.npz"
    target_path = base_job / "target.png"
    common_base_hash = _sha256(base_field_path)
    baseline_keys = (
        "n_gaussians",
        "foreground_mse",
        "boundary_mse",
        "cvar99_mse",
        "p99_mse",
        "interior_hole_fraction",
        "boundary_hole_fraction",
        "outside_max_abs",
        "outside_coverage_max",
        "fine_detail_highpass_mse",
    )
    reference_baseline = {key: results["fit032"]["baseline"][key] for key in baseline_keys}

    fit033 = results["fit033"]
    fit033_row = _find_row(fit033["rows"], arm="highpass_solved", budget=128)
    fit033_highpass, fit033_laplacian = _detail_reductions(
        fit033,
        fit033_row,
    )
    fit034 = results["fit034"]
    fit034_row = _find_row(
        fit034["rows"],
        raw_weight=fit034["selected_raw_weight"],
    )
    fit035 = results["fit035"]
    fit035_row = _find_row(
        fit035["rows"],
        scale=fit035["selected"]["scale"],
        gradient_ridge=fit035["selected"]["gradient_ridge"],
        objective=fit035["selected"]["objective"],
    )
    fit036 = results["fit036"]
    fit036_row = _find_row(
        fit036["rows"],
        max_long_scale=fit036["selected"]["max_long_scale"],
        coherence_power=fit036["selected"]["coherence_power"],
    )
    fit037 = results["fit037"]
    fit037_row = fit037["rows"][-1]
    fit038 = results["fit038"]
    fit038_row = fit038["rows"][-1]

    replication_class = "same-target RTX4090 replication; common 10,816-row base"
    stage_endpoints = [
        _endpoint(
            order=32,
            task="FIT-032",
            method="Gauge-lifted residual dipoles",
            evidence_class=replication_class,
            row=None,
            added_rows=0,
            highpass_reduction=0.0,
            laplacian_reduction=0.0,
            gate_status="rejected: no protected recovery accepted",
            target_reached=False,
            source_artifact="raw/fit032.json",
        ),
        _endpoint(
            order=33,
            task="FIT-033",
            method="High-pass births + partial color solve",
            evidence_class=replication_class,
            row=_with_baseline(fit033_row, fit033),
            added_rows=128,
            highpass_reduction=fit033_highpass,
            laplacian_reduction=fit033_laplacian,
            gate_status="advanced to independent confirmation only",
            target_reached=False,
            source_artifact="raw/fit033.json",
        ),
        _endpoint(
            order=34,
            task="FIT-034",
            method="Spectral/raw mixed partial solve",
            evidence_class=replication_class,
            row=_with_baseline(fit034_row, fit034),
            added_rows=128,
            highpass_reduction=float(fit034_row["sigma_1_5_reduction"]),
            laplacian_reduction=float(fit034_row["laplacian_reduction"]),
            gate_status="negative: selected raw weight 0",
            target_reached=False,
            source_artifact="raw/fit034.json",
        ),
        _endpoint(
            order=35,
            task="FIT-035",
            method="Sparse affine detail births",
            evidence_class=replication_class,
            row=_with_baseline(fit035_row, fit035),
            added_rows=128,
            highpass_reduction=float(fit035_row["sigma_1_5_reduction"]),
            laplacian_reduction=float(fit035_row["laplacian_reduction"]),
            gate_status="negative: affine target missed",
            target_reached=False,
            source_artifact="raw/fit035.json",
        ),
        _endpoint(
            order=36,
            task="FIT-036",
            method="High-pass residual ridge births",
            evidence_class=replication_class,
            row=_with_baseline(fit036_row, fit036),
            added_rows=128,
            highpass_reduction=float(fit036_row["sigma_1_5_reduction"]),
            laplacian_reduction=float(fit036_row["laplacian_reduction"]),
            gate_status="negative: ridge target missed",
            target_reached=False,
            source_artifact="raw/fit036.json",
        ),
        _endpoint(
            order=37,
            task="FIT-037",
            method="Static nested deep-detail rows",
            evidence_class=replication_class,
            row=_with_baseline(fit037_row, fit037),
            added_rows=int(fit037_row["budget"]),
            highpass_reduction=float(fit037_row["sigma_1_5_reduction"]),
            laplacian_reduction=float(fit037_row["laplacian_reduction"]),
            gate_status="negative: target missed at 2,048 rows",
            target_reached=False,
            source_artifact="raw/fit037.json",
        ),
        _endpoint(
            order=38,
            task="FIT-038",
            method="Iterative orthogonal pursuit, radius 2",
            evidence_class=replication_class,
            row=_with_baseline(fit038_row, fit038),
            added_rows=int(fit038_row["added_rows"]),
            highpass_reduction=float(fit038_row["sigma_1_5_reduction"]),
            laplacian_reduction=float(fit038_row["laplacian_reduction"]),
            gate_status="negative: target missed at 2,048 rows",
            target_reached=False,
            source_artifact="raw/fit038.json",
        ),
        {
            "stage_order": 39,
            "task": "FIT-039/040",
            "method": "Orthogonal pursuit + exact-site exclusion",
            "evidence_class": ("published same-target RTX3050 result; 11,000-row base"),
            "added_rows": int(production["pursuit_tail"]["activated_rows"]),
            "sigma_1_5_highpass_reduction": float(production["pursuit_tail"]["highpass_reduction"]),
            "laplacian_reduction": float(production["pursuit_tail"]["laplacian_reduction"]),
            "foreground_psnr_gain_db": float(production["pursuit_tail"]["foreground_psnr_gain_db"]),
            "protected_safe": bool(production["all_acceptance_checks_passed"]),
            "target_reached": bool(production["pursuit_tail"]["target_reached"]),
            "gate_status": "fine-detail target reached at 768 rows",
            "source_artifact": "raw/fit040_production_result.json",
        },
    ]

    stage_curves = []
    for task, rows, row_key in (
        ("FIT-037", fit037["rows"], "budget"),
        ("FIT-038", fit038["rows"], "added_rows"),
    ):
        for row in rows:
            stage_curves.append(
                {
                    "task": task,
                    "added_rows": int(row[row_key]),
                    "sigma_1_5_highpass_reduction": float(row["sigma_1_5_reduction"]),
                    "laplacian_reduction": float(row["laplacian_reduction"]),
                    "protected_safe": bool(row["protected_safe"]),
                }
            )

    published_same_base = fit040_audit["same_base_result"]
    renderer_aa = fit035["renderer_aa"]
    checks = {
        "expected_schemas": all(
            results[name]["schema"] == schema for name, schema in RESULT_SCHEMAS.items()
        ),
        "exact_published_target_file_reused": (
            _sha256(target_path) == EXPECTED_TARGET_SHA256
            and _sha256(args.run_root / "protocol_inputs/C0001.png") == EXPECTED_TARGET_SHA256
            and production["source"]["target_pixel_sha256"] == EXPECTED_TARGET_SHA256
        ),
        "mask_binding_matches_published": (
            base_config["source"]["mask_sha256"] == EXPECTED_MASK_SHA256
            and production["source"]["mask_sha256"] == EXPECTED_MASK_SHA256
        ),
        "all_replications_share_base_field": all(
            _base_hash(result) == common_base_hash for result in results.values()
        ),
        "all_replications_share_baseline": all(
            _baselines_match(
                reference_baseline,
                result["baseline"],
                baseline_keys,
            )
            for result in results.values()
        ),
        "base_result_field_binding": (
            base_result["field_sha256"] == common_base_hash
            and int(base_result["n_gaussians"]) == 10816
        ),
        "captured_sources_match": all(
            _source_snapshot_matches(result) for result in results.values()
        ),
        "reported_reductions_recompute": all(
            _reductions_match(results[name])
            for name in (
                "fit033",
                "fit034",
                "fit035",
                "fit036",
                "fit037",
                "fit038",
            )
        ),
        "fit032_rejection_replicated": (
            not results["fit032"]["decision"]["promote"]
            and int(results["fit032"]["decision"]["budgets_passed"]) == 0
            and not any(row["recovery_accepted"] for row in results["fit032"]["rows"])
        ),
        "fit033_advance_only_replicated": (
            bool(fit033["decision"]["advance_to_independent_confirmation"])
            and not bool(fit033["decision"]["production_promotion_authorized"])
            and all(
                row["protected_safe"] for row in fit033["rows"] if row["arm"] == "highpass_solved"
            )
        ),
        "fit034_negative_replicated": (
            float(fit034["selected_raw_weight"]) == 0.0
            and bool(fit034_row["protected_safe"])
            and (
                float(fit034_row["sigma_1_5_reduction"]) < 0.15
                or float(fit034_row["laplacian_reduction"]) < 0.10
            )
        ),
        "fit035_renderer_aa_valid": (
            float(renderer_aa["max_abs_render_delta"]) <= 2e-6
            and float(renderer_aa["foreground_mse_delta"]) == 0.0
            and bool(renderer_aa["protected_nondegrading"])
            and renderer_aa["protected_reasons"] == ["no_material_gain"]
        ),
        "fit035_negative_replicated": (
            bool(fit035_row["protected_safe"])
            and (
                float(fit035_row["sigma_1_5_reduction"]) < 0.15
                or float(fit035_row["laplacian_reduction"]) < 0.10
            )
        ),
        "fit036_negative_replicated": (
            bool(fit036_row["protected_safe"])
            and (
                float(fit036_row["sigma_1_5_reduction"]) < 0.15
                or float(fit036_row["laplacian_reduction"]) < 0.10
            )
        ),
        "fit037_target_miss_replicated": (
            not bool(fit037["decision"]["target_reached"])
            and not bool(fit037["decision"]["production_promotion_authorized"])
            and all(row["protected_safe"] for row in fit037["rows"])
        ),
        "fit038_target_miss_replicated": (
            not bool(fit038["decision"]["target_reached"])
            and not bool(fit038["decision"]["production_promotion_authorized"])
            and all(row["protected_safe"] for row in fit038["rows"])
        ),
        "fit031_original_audit_valid": bool(fit031_audit["valid"]),
        "published_equal_base_audit_passed": (
            bool(fit040_audit["passed"])
            and len(fit040_audit["checks"]) == 17
            and all(fit040_audit["checks"].values())
        ),
    }

    archive = _archive_inputs(args.out.parent / "raw", all_inputs)
    payload = {
        "schema": "structsplat.fit031-new-method-stages-audit.v1",
        "created_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "passed": all(checks.values()),
        "checks": checks,
        "scope": {
            "source": "Janelle frame_00008/C0001 full masked frame",
            "fit_size_wh": [1200, 1038],
            "seed": 0,
            "replication_device": "NVIDIA GeForce RTX 4090",
            "replication_base_rows": 10816,
            "replication_base_field_sha256": common_base_hash,
            "published_control_device": fit040_audit["scope"]["device"],
            "published_control_base_rows": int(fit040_audit["scope"]["base_rows"]),
            "comparability": (
                "FIT-032--038 are a common-base, same-target RTX4090 "
                "replication. FIT-031 versus FIT-039/040 uses the separately "
                "published exact-same-base RTX3050 control. Cross-tier "
                "endpoint values are descriptive, not exact-base effects."
            ),
        },
        "evidence_inventory": {
            "rerun_due_to_missing_durable_machine_results": [
                "FIT-032",
                "FIT-033",
                "FIT-034",
                "FIT-035",
                "FIT-036",
                "FIT-037",
                "FIT-038",
            ],
            "reused_after_hash_and_audit_validation": [
                "FIT-031 original crop bundle",
                "FIT-039 exclusion result",
                "FIT-040 production replay",
                "FIT-041 exact-same-base FIT-031 control",
            ],
            "quarantined": {
                "path": _relative(args.run_root / "base"),
                "reason": (
                    "Pillow 11 Lanczos pixels did not match the published "
                    "Pillow 12.3 target; no result from this base is used."
                ),
            },
        },
        "stage_endpoints": stage_endpoints,
        "stage_curves": stage_curves,
        "published_same_base_comparison": published_same_base,
        "original_fit031_context": {
            "evidence_class": (
                "original 1200x437 crop, one image/seed/RTX4090; not a "
                "direct control for the full-frame stages"
            ),
            "added_rows": int(fit031_audit["results"]["tail"]["activated_rows"]),
            "foreground_psnr_gain_db": float(
                fit031_audit["within_run_deltas"]["foreground_psnr_db"]
            ),
            "boundary_psnr_gain_db": float(fit031_audit["within_run_deltas"]["boundary_psnr_db"]),
            "cvar99_relative_reduction": -float(
                fit031_audit["within_run_deltas"]["cvar99_relative"]
            ),
            "p99_relative_reduction": -float(fit031_audit["within_run_deltas"]["p99_relative"]),
        },
        "fit035_audit_correction": {
            "issue": (
                "The original harness treated an identity A/A result as "
                "invalid because the strict gate requires material gain."
            ),
            "fix": (
                "A/A now accepts either a full gate pass or exactly the "
                "no_material_gain reason; any protected regression still "
                "fails closed."
            ),
            "outcome": (
                "The corrected A/A check passes and the scientific FIT-035 "
                "decision remains negative."
            ),
        },
        "decision": {
            "fine_detail_winner": "orthogonal_pursuit_with_exact_site_exclusion",
            "global_foreground_psnr_winner_on_exact_same_base": ("fit031_error_only_tail"),
            "recommended_stage_judgment": (
                "Keep FIT-032--037 as negative/diagnostic stages; keep "
                "FIT-038 as the useful iterative mechanism; retain "
                "FIT-039 exact-site exclusion and FIT-040 integration as "
                "the fine-detail path."
            ),
            "default_change_authorized": False,
            "generality_authorized": False,
            "equal_rate_claim_authorized": False,
        },
        "inputs": archive,
    }
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "checks": len(checks),
                "stage_endpoints": len(stage_endpoints),
                "out": _relative(args.out),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    parser.add_argument(
        "--fit031-root",
        type=Path,
        default=DEFAULT_FIT031_ROOT,
    )
    parser.add_argument(
        "--published-root",
        type=Path,
        default=DEFAULT_PUBLISHED_ROOT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
