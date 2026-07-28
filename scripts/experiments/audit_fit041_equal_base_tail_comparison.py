#!/usr/bin/env python3
"""Adversarial JSON audit of the equal-base Janelle terminal-tail comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PURSUIT = (
    REPOSITORY_ROOT / "runs/fit040_janelle_production_pursuit_20260728/result.json"
)
DEFAULT_ERROR = (
    REPOSITORY_ROOT / "runs/fit041_janelle_equal_base_error_tail_20260728/result.json"
)
DEFAULT_PURSUIT_AUDIT = (
    REPOSITORY_ROOT
    / "runs/fit039_janelle_exclusion_screen_20260728/audit/audit.json"
)
DEFAULT_PURSUIT_PERCEPTUAL = (
    REPOSITORY_ROOT
    / "runs/fit039_janelle_exclusion_screen_20260728/audit/perceptual.json"
)
DEFAULT_ERROR_PERCEPTUAL = (
    REPOSITORY_ROOT
    / "runs/fit041_janelle_equal_base_error_tail_20260728/perceptual.json"
)
DEFAULT_SPATIAL = (
    REPOSITORY_ROOT
    / "runs/fit039_janelle_exclusion_screen_20260728/audit/spatial.json"
)
DEFAULT_OUT = (
    REPOSITORY_ROOT / "runs/fit041_janelle_equal_base_error_tail_20260728/audit.json"
)
SOURCE_FILES = (
    "src/structsplat/detail_pursuit.py",
    "src/structsplat/safe_schedule.py",
    "src/structsplat/pipeline.py",
    "src/structsplat/workflows.py",
    "scripts/experiments/fit040_janelle_production_pursuit.py",
    "scripts/experiments/fit041_janelle_equal_base_error_tail.py",
    "scripts/experiments/audit_fit041_equal_base_tail_comparison.py",
    "tests/test_detail_pursuit.py",
    "tests/test_pursuit_schedule.py",
    "tests/test_pipeline.py",
    "tests/test_pipeline_workflows.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _relative_lpips_reduction(payload: dict) -> float:
    baseline = float(payload["baseline"]["raw"]["lpips"])
    candidate = float(payload["candidate"]["raw"]["lpips"])
    return 1.0 - candidate / baseline


def run(args: argparse.Namespace) -> None:
    pursuit = _read(args.pursuit)
    error = _read(args.error)
    pursuit_audit = _read(args.pursuit_audit)
    pursuit_perceptual = _read(args.pursuit_perceptual)
    error_perceptual = _read(args.error_perceptual)
    spatial = _read(args.spatial)

    pursuit_tail = pursuit["pursuit_tail"]
    error_tail = error["error_tail"]
    comparison = error["comparison"]
    pursuit_lpips = _relative_lpips_reduction(pursuit_perceptual)
    error_lpips = _relative_lpips_reduction(error_perceptual)
    checks = {
        "same_base_binding": bool(comparison["same_base"]),
        "production_replay_acceptance": bool(
            pursuit["all_acceptance_checks_passed"]
        ),
        "prototype_cold_audit": bool(pursuit_audit["passed"]),
        "pursuit_first_target_at_768": (
            int(pursuit_tail["activated_rows"]) == 768
            and not any(
                bool(wave["target_reached"])
                for wave in pursuit_tail["waves"][:-1]
            )
            and bool(pursuit_tail["waves"][-1]["target_reached"])
        ),
        "pursuit_rows_unique": (
            int(pursuit_tail["unique_sites"])
            == int(pursuit_tail["activated_rows"])
        ),
        "pursuit_inherited_rows_frozen": all(
            bool(wave["inherited_rows_frozen"])
            for wave in pursuit_tail["waves"]
            if wave["accepted"]
        ),
        "pursuit_protected_safe": not any(
            wave["protected_reasons"]
            for wave in pursuit_tail["waves"]
            if wave["accepted"]
        ),
        "pursuit_detail_targets": (
            float(pursuit_tail["highpass_reduction"]) >= 0.25
            and float(pursuit_tail["laplacian_reduction"]) >= 0.20
        ),
        "error_tail_all_requested_rows_activated": (
            int(error_tail["activated_rows"]) == int(error_tail["requested_rows"])
        ),
        "error_tail_fixed_point": (
            error_tail["convergence_termination_reason"]
            == "deterministic_fixed_point"
        ),
        "both_outside_exact_zero": (
            float(error_tail["after"]["outside_max_abs"]) == 0.0
            and float(error_tail["after"]["outside_coverage_max"]) == 0.0
            and float(pursuit["metrics"]["outside_max_abs"]) == 0.0
            and float(pursuit["metrics"]["outside_coverage_max"]) == 0.0
        ),
        "placement_separation": (
            int(comparison["error_tail_deep_rows"]) == 0
            and int(comparison["pursuit_deep_rows"]) == 768
        ),
        "pursuit_uses_fewer_rows": bool(comparison["pursuit_uses_fewer_rows"]),
        "pursuit_larger_highpass_reduction": bool(
            comparison["pursuit_has_larger_highpass_reduction"]
        ),
        "pursuit_larger_laplacian_reduction": bool(
            comparison["pursuit_has_larger_laplacian_reduction"]
        ),
        "pursuit_larger_lpips_reduction": pursuit_lpips > error_lpips,
        "error_tail_larger_global_psnr_gain": (
            float(comparison["error_tail_foreground_psnr_gain_db"])
            > float(comparison["pursuit_foreground_psnr_gain_db"])
        ),
    }
    payload = {
        "schema": "structsplat.fit041.equal-base-tail-audit.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "source": "Janelle frame_00008/C0001 full masked frame",
            "fit_size_wh": [1200, 1038],
            "base_rows": int(pursuit_tail["start_n"]),
            "seed": 0,
            "device": pursuit["environment"]["gpu"],
            "latest_commit_screen_is_not_direct_control": (
                "FIT-031's committed 1200x437 crop differs from this 1200x1038 target; "
                "FIT-041 supplies the equal-base control"
            ),
        },
        "inputs": {
            "pursuit": {
                "path": str(args.pursuit.resolve()),
                "sha256": _sha256(args.pursuit),
            },
            "error_tail": {
                "path": str(args.error.resolve()),
                "sha256": _sha256(args.error),
            },
            "pursuit_audit": {
                "path": str(args.pursuit_audit.resolve()),
                "sha256": _sha256(args.pursuit_audit),
            },
            "pursuit_perceptual": {
                "path": str(args.pursuit_perceptual.resolve()),
                "sha256": _sha256(args.pursuit_perceptual),
            },
            "error_perceptual": {
                "path": str(args.error_perceptual.resolve()),
                "sha256": _sha256(args.error_perceptual),
            },
            "spatial": {
                "path": str(args.spatial.resolve()),
                "sha256": _sha256(args.spatial),
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
        "same_base_result": {
            "error_tail": {
                "added_rows": int(error_tail["activated_rows"]),
                "foreground_psnr_gain_db": float(
                    comparison["error_tail_foreground_psnr_gain_db"]
                ),
                "sigma_1_5_highpass_reduction": float(
                    comparison["error_tail_highpass_reduction"]
                ),
                "laplacian_reduction": float(
                    comparison["error_tail_laplacian_reduction"]
                ),
                "lpips_reduction": error_lpips,
                "deep_rows": int(comparison["error_tail_deep_rows"]),
                "seconds": float(comparison["error_tail_seconds"]),
            },
            "orthogonal_pursuit": {
                "added_rows": int(pursuit_tail["activated_rows"]),
                "foreground_psnr_gain_db": float(
                    comparison["pursuit_foreground_psnr_gain_db"]
                ),
                "sigma_1_5_highpass_reduction": float(
                    comparison["pursuit_highpass_reduction"]
                ),
                "laplacian_reduction": float(
                    comparison["pursuit_laplacian_reduction"]
                ),
                "lpips_reduction": pursuit_lpips,
                "deep_rows": int(comparison["pursuit_deep_rows"]),
                "seconds": float(comparison["pursuit_tail_seconds"]),
            },
            "error_to_pursuit_row_ratio": float(
                comparison["error_tail_to_pursuit_row_ratio"]
            ),
        },
        "spatial_boundary": {
            "deep_pixels_improved_fraction": float(
                spatial["pixel_attribution"]["improved_fraction"]
            ),
            "deep_pixels_worsened_fraction": float(
                spatial["pixel_attribution"]["worsened_fraction"]
            ),
            "positive_32px_tile_fraction": float(
                spatial["tiles_32"]["positive_fraction"]
            ),
            "interpretation": (
                "The net fine-detail effect is large but spatially concentrated; this is "
                "not a uniform-detail or general-image claim."
            ),
        },
        "decision": {
            "fine_detail_winner": "orthogonal_pursuit",
            "global_foreground_psnr_winner": "error_tail",
            "default_change_authorized": False,
            "generality_authorized": False,
            "equal_rate_claim_authorized": False,
        },
        "sources": [
            {
                "path": relative,
                "sha256": _sha256(REPOSITORY_ROOT / relative),
            }
            for relative in SOURCE_FILES
        ],
    }
    _atomic_json(args.out, payload)
    print(json.dumps({"passed": payload["passed"], **payload["decision"]}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pursuit", type=Path, default=DEFAULT_PURSUIT)
    parser.add_argument("--error", type=Path, default=DEFAULT_ERROR)
    parser.add_argument(
        "--pursuit-audit",
        type=Path,
        default=DEFAULT_PURSUIT_AUDIT,
    )
    parser.add_argument(
        "--pursuit-perceptual",
        type=Path,
        default=DEFAULT_PURSUIT_PERCEPTUAL,
    )
    parser.add_argument(
        "--error-perceptual",
        type=Path,
        default=DEFAULT_ERROR_PERCEPTUAL,
    )
    parser.add_argument("--spatial", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
