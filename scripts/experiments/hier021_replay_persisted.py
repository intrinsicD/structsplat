#!/usr/bin/env python3
"""Replay frozen HIER-021 SPT1 on 40 persisted HIER-015--020/test fields."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import platform
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier005_pixel_contraction as report_utils  # noqa: E402
from scripts.experiments import hier015_geometry_escape as h15  # noqa: E402
from scripts.experiments import hier017_normalization_epsilon as h17  # noqa: E402
from scripts.experiments import hier020_sparse_pixel_safe_tail as h20  # noqa: E402
from scripts.experiments import hier021_source_patch_tail as h21  # noqa: E402
from structsplat.gaussians import GaussianField  # noqa: E402


REPORT_SCHEMA = h21.REPORT_SCHEMA
PHASE = "replay_persisted_h15_h20_and_tests"
SOURCE_SPECS = (
    {
        "group": "consumed_h15_h19",
        "phase": "replay_consumed_h15_h19",
        "direct_count": 20,
        "requires_bounded_pass": True,
    },
    {
        "group": "hier020_fresh",
        "phase": "development",
        "direct_count": 4,
        "requires_bounded_pass": True,
    },
    {
        "group": "tests_test_images",
        "phase": "replay_tests",
        "direct_count": 16,
        "requires_bounded_pass": False,
    },
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-results", type=Path, nargs=3)
    source.add_argument("--review-from", type=Path)
    parser.add_argument("--development-decision", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--visual-disposition", choices=("pending", "pass", "fail"), default="pending"
    )
    parser.add_argument("--target-gaussians", type=int, default=7000)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direct-fit-steps", type=int, default=750)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additive-renderer", default="cuda_additive")
    parser.add_argument("--direct-renderer", default="cuda")
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--coverage-threshold", type=float, default=1e-8)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.set_defaults(phase="development")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "target_gaussians": 7000,
        "max_side": 512,
        "seed": 0,
        "direct_fit_steps": 750,
        "device": "cuda",
        "additive_renderer": "cuda_additive",
        "direct_renderer": "cuda",
        "render_chunk": 256,
        "lpips": True,
        "patch_radius": 3,
        "coverage_threshold": 1e-8,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-021 replay requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if args.source_results is not None:
        if args.development_decision is None:
            raise SystemExit("--source-results requires --development-decision")
        if args.visual_disposition != "pending":
            raise SystemExit("record the visual verdict only with --review-from")
    elif args.development_decision is not None:
        raise SystemExit("--review-from rejects --development-decision")


def _development_receipt(path: Path) -> dict[str, object]:
    source = path.resolve()
    if not source.is_file():
        raise SystemExit(f"development decision does not exist: {source}")
    decision = json.loads(source.read_text(encoding="utf-8"))
    if (
        decision.get("schema") != REPORT_SCHEMA
        or decision.get("phase") != "development"
        or not decision.get("bounded_bank_pass")
        or decision.get("visual_disposition") != "pass"
    ):
        raise SystemExit("HIER-021 development decision did not pass numeric and visual gates")
    return {"path": str(source), "sha256": report_utils._sha256(source)}


def _source_records(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    all_images: set[str] = set()
    for source_path, spec in zip(paths, SOURCE_SPECS):
        root = source_path.resolve()
        required = [root / name for name in ("metrics.json", "attempts.json", "decision.json", "manifest.json")]
        if not root.is_dir() or not all(path.is_file() for path in required):
            raise SystemExit(f"persisted source bundle is incomplete: {root}")
        payload = json.loads(required[0].read_text(encoding="utf-8"))
        attempts = json.loads(required[1].read_text(encoding="utf-8"))
        decision = json.loads(required[2].read_text(encoding="utf-8"))
        if payload.get("schema") != h20.REPORT_SCHEMA:
            raise SystemExit(f"{root} is not a HIER-020 persisted bundle")
        if decision.get("phase") != spec["phase"]:
            raise SystemExit(
                f"{root} has phase {decision.get('phase')!r}; expected {spec['phase']!r}"
            )
        if spec["requires_bounded_pass"] and not decision.get("bounded_bank_pass"):
            raise SystemExit(f"{root} did not pass its HIER-020 bounded gate")
        rows = payload.get("rows", [])
        controls = [row for row in rows if row.get("arm") == h20.CONTROL_ARM]
        direct = [row for row in rows if row.get("arm") == h20.DIRECT_ARM]
        expected = int(spec["direct_count"])
        if len(controls) != expected or len(direct) != expected:
            raise SystemExit(
                f"{root} must expose {expected} controls and {expected} direct fields"
            )
        if len(attempts.get("attempts", [])) != 2 * expected or any(
            item.get("status") != "ok" for item in attempts.get("attempts", [])
        ):
            raise SystemExit(f"{root} does not have a complete successful attempt ledger")
        controls_by_image = {str(row["image"]): row for row in controls}
        direct_images = {str(row["image"]) for row in direct}
        if set(controls_by_image) != direct_images:
            raise SystemExit(f"{root} control/direct image keys differ")
        overlap = all_images & direct_images
        if overlap:
            raise SystemExit(f"persisted source banks overlap on image keys: {sorted(overlap)}")
        all_images |= direct_images
        records.append(
            {
                **spec,
                "root": root,
                "metrics_sha256": report_utils._sha256(required[0]),
                "attempts_sha256": report_utils._sha256(required[1]),
                "decision_sha256": report_utils._sha256(required[2]),
                "manifest_sha256": report_utils._sha256(required[3]),
                "controls": controls_by_image,
                "direct": direct,
            }
        )
    if len(all_images) != 40:
        raise SystemExit(f"frozen HIER-021 replay requires 40 unique fields, got {len(all_images)}")
    return records


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier021_source_patch_tail.py",
        ROOT / "src" / "structsplat" / "source_patch_tail.py",
        ROOT / "tasks" / "HIER-021-low-coverage-rgb-patch-tail.md",
    )
    records: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": report_utils._sha256(destination),
            }
        )
    return records


def _row_lineage(row: dict[str, object], group: str) -> str:
    if group == "consumed_h15_h19":
        value = row.get("source_lineage")
        if value not in {"hier015", "hier016", "hier017", "hier018", "hier019"}:
            raise RuntimeError(f"invalid consumed lineage {value!r}")
        return str(value)
    return group


def _copy_cell(
    *,
    source_root: Path,
    source_row: dict[str, object],
    output_root: Path,
    group: str,
    lineage: str,
    arm: str,
) -> dict[str, object]:
    source_dir = source_root / str(source_row["artifact_dir"])
    destination = (
        output_root
        / "artifacts"
        / f"{group}__{lineage}__{source_row['image']}__{arm}__n7000"
    )
    shutil.copytree(source_dir, destination)
    row = dict(source_row)
    row.update(
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "phase": PHASE,
            "arm": arm,
            "artifact_dir": str(destination.relative_to(output_root)),
            "source_replay_group": group,
            "source_lineage": lineage,
            "source_result_schema": source_row["schema"],
            "source_result_arm": source_row["arm"],
            "persisted_field_reused_without_refit": True,
        }
    )
    report_utils._write_json(destination / "row.json", row)
    return row


def _pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    controls = {
        (str(row["source_lineage"]), str(row["image"])): row
        for row in rows
        if row.get("arm") == h21.CONTROL_ARM
    }
    pairs: list[dict[str, object]] = []
    for row in rows:
        if row.get("arm") != h21.DIRECT_ARM:
            continue
        key = (str(row["source_lineage"]), str(row["image"]))
        control = controls.get(key)
        if control is None:
            continue
        selected = row["spt1_selected_metrics"]
        pairs.append(
            {
                "source_replay_group": row["source_replay_group"],
                "source_lineage": row["source_lineage"],
                "image": row["image"],
                "selected_mode": row["spt1_selected_mode"],
                "selected_count": row["spt1_selected_count"],
                "psnr_delta_db": float(selected["psnr_db"]) - float(control["psnr_db"]),
                "mse_ratio": float(selected["masked_mse"])
                / max(float(control["masked_mse"]), 1e-30),
                "ms_ssim_delta": float(selected["ms_ssim"])
                - float(control["ms_ssim"]),
                "lpips_delta": float(selected["lpips"]) - float(control["lpips"]),
                "pixel_max_delta": float(selected["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "patch7_max_delta": float(selected["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
                "baseline_pixel_max_delta": float(row["artifact_pixel_rmse_max"])
                - float(control["artifact_pixel_rmse_max"]),
                "baseline_patch7_max_delta": float(row["artifact_patch_rmse_max_7"])
                - float(control["artifact_patch_rmse_max_7"]),
            }
        )
    return pairs


def _decision(
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    *,
    visual_disposition: str,
) -> dict[str, object]:
    direct = [row for row in rows if row.get("arm") == h21.DIRECT_ARM]
    controls = [row for row in rows if row.get("arm") == h21.CONTROL_ARM]
    pairs = _pairs(rows)
    tests_pairs = [
        pair for pair in pairs if pair["source_replay_group"] == "tests_test_images"
    ]
    baseline_local_failures = [
        pair
        for pair in pairs
        if float(pair["baseline_pixel_max_delta"]) > 1e-12
        or float(pair["baseline_patch7_max_delta"]) > 1e-12
    ]

    def selected_safe(row: dict[str, object], key: str, *, upper: float) -> bool:
        if row["spt1_selected_mode"] == h21.BASELINE_MODE:
            return True
        return float(row["spt1_metric_deltas_vs_baseline"][key]) <= upper

    gates = {
        "complete_forty_direct_rows": len(direct) == 40,
        "complete_forty_controls": len(controls) == 40,
        "complete_forty_pairs": len(pairs) == 40,
        "complete_sixteen_test_pairs": len(tests_pairs) == 16,
        "complete_attempt_ledger": len(attempts) == 80,
        "zero_failures": all(item.get("status") == "ok" for item in attempts),
        "all_persisted_without_refit": all(
            bool(row["persisted_field_reused_without_refit"]) for row in direct
        ),
        "all_exact_count": all(int(row["n_gaussians"]) == 7000 for row in direct),
        "all_field_bytes_unchanged": all(
            row["spt1_field_file_sha256_before"] == row["spt1_field_file_sha256_after"]
            for row in direct
        ),
        "all_payloads_canonical": all(
            bool(row["spt1_payload_roundtrip_exact"]) for row in direct
        ),
        "all_candidates_finite": all(bool(row["spt1_candidate_finite"]) for row in direct),
        "all_outside_payload_bit_exact": all(
            float(row["spt1_outside_identity_max_abs"]) == 0.0 for row in direct
        ),
        "all_render_and_decode_parity_le_2e_5": all(
            float(row["spt1_baseline_cold_parity_max_abs"]) <= 2e-5
            and float(row["spt1_candidate_repeated_parity_max_abs"]) <= 2e-5
            for row in direct
        ),
        "all_selected_transactions_safe": all(
            row["spt1_selected_mode"] == h21.BASELINE_MODE
            or all(bool(value) for value in row["spt1_selection_clauses"].values())
            for row in direct
        ),
        "all_selected_mse_noninferior_vs_baseline": all(
            selected_safe(row, "mse_ratio", upper=1.0 + 1e-8) for row in direct
        ),
        "all_selected_ms_ssim_noninferior_vs_baseline": all(
            row["spt1_selected_mode"] == h21.BASELINE_MODE
            or float(row["spt1_metric_deltas_vs_baseline"]["ms_ssim_delta"]) >= -1e-7
            for row in direct
        ),
        "all_selected_lpips_noninferior_vs_baseline": all(
            selected_safe(row, "lpips_delta", upper=1e-7) for row in direct
        ),
        "all_selected_pixel_noninferior_vs_baseline": all(
            selected_safe(row, "pixel_max_delta", upper=1e-12) for row in direct
        ),
        "all_selected_patch_noninferior_vs_baseline": all(
            selected_safe(row, "patch7_max_delta", upper=1e-12) for row in direct
        ),
        "all_recorded_h005_local_failures_repaired": all(
            float(pair["pixel_max_delta"]) <= 1e-12
            and float(pair["patch7_max_delta"]) <= 1e-12
            for pair in baseline_local_failures
        ),
        "all_tests_psnr_gain_vs_h005_ge_2_db": all(
            float(pair["psnr_delta_db"]) >= 2.0 for pair in tests_pairs
        ),
        "all_tests_mse_noninferior_vs_h005": all(
            float(pair["mse_ratio"]) <= 1.0 + 1e-8 for pair in tests_pairs
        ),
        "all_tests_ms_ssim_noninferior_vs_h005": all(
            float(pair["ms_ssim_delta"]) >= -1e-7 for pair in tests_pairs
        ),
        "all_tests_lpips_noninferior_vs_h005": all(
            float(pair["lpips_delta"]) <= 1e-7 for pair in tests_pairs
        ),
        "all_tests_pixel_noninferior_vs_h005": all(
            float(pair["pixel_max_delta"]) <= 1e-12 for pair in tests_pairs
        ),
        "all_tests_patch_noninferior_vs_h005": all(
            float(pair["patch7_max_delta"]) <= 1e-12 for pair in tests_pairs
        ),
        "median_pipeline_time_ratio_le_1_25": (
            float(np.median([row["spt1_pipeline_time_ratio"] for row in direct])) <= 1.25
            if direct
            else False
        ),
        "median_selected_decode_ratio_le_2": (
            float(np.median([row["spt1_decode_time_ratio"] for row in direct])) <= 2.0
            if direct
            else False
        ),
    }
    numeric_pass = all(gates.values())
    selected_by_group = {
        str(spec["group"]): sum(
            row["source_replay_group"] == spec["group"]
            and row["spt1_selected_mode"] == h21.CANDIDATE_MODE
            for row in direct
        )
        for spec in SOURCE_SPECS
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": PHASE,
        "gates": gates,
        "selected_vs_h005_pairs": pairs,
        "baseline_local_failures": baseline_local_failures,
        "baseline_local_failure_count": len(baseline_local_failures),
        "selected_image_count": sum(
            row["spt1_selected_mode"] == h21.CANDIDATE_MODE for row in direct
        ),
        "selected_images_by_group": selected_by_group,
        "proposed_pixel_count": sum(int(row["spt1_selected_count"]) for row in direct),
        "selected_pixel_count": sum(
            int(row["spt1_selected_count"])
            for row in direct
            if row["spt1_selected_mode"] == h21.CANDIDATE_MODE
        ),
        "candidate_side_bytes": sum(
            int(row["spt1_candidate_payload_bytes"]) for row in direct
        ),
        "selected_side_bytes": sum(
            int(row["spt1_selected_payload_bytes"]) for row in direct
        ),
        "attempt_count": len(attempts),
        "failure_count": sum(item.get("status") != "ok" for item in attempts),
        "numeric_bank_pass": numeric_pass,
        "visual_review_required": True,
        "visual_disposition": visual_disposition,
        "bounded_bank_pass": numeric_pass and visual_disposition == "pass",
        "numeric_candidates": [h21.CANDIDATE_MODE] if numeric_pass else [],
        "interpretation": (
            "Frozen 40-field replay passed numeric and recorded visual gates."
            if numeric_pass and visual_disposition == "pass"
            else (
                "Frozen 40-field replay is numerically safe and awaits native visual review."
                if numeric_pass
                else "A frozen 40-field replay gate failed; retain the counterexample."
            )
        ),
    }


def _review(args: argparse.Namespace, output_root: Path, command: str) -> bool:
    if args.review_from is None:
        return False
    source_root = args.review_from.resolve()
    required = [source_root / name for name in ("metrics.json", "attempts.json", "decision.json")]
    if not source_root.is_dir() or not all(path.is_file() for path in required):
        raise SystemExit("--review-from is missing complete HIER-021 replay ledgers")
    rows = json.loads(required[0].read_text(encoding="utf-8"))["rows"]
    attempts = json.loads(required[1].read_text(encoding="utf-8"))["attempts"]
    prior = json.loads(required[2].read_text(encoding="utf-8"))
    if (
        prior.get("schema") != REPORT_SCHEMA
        or prior.get("phase") != PHASE
        or not prior.get("numeric_bank_pass")
        or prior.get("visual_disposition") != "pending"
    ):
        raise SystemExit("--review-from is not a pending numeric-pass HIER-021 replay")
    shutil.copytree(source_root, output_root)
    decision = _decision(rows, attempts, visual_disposition=args.visual_disposition)
    decision.update(
        {
            "reviewed_from": str(source_root),
            "review_only": True,
            "cell_computation_rerun": False,
            "source_decision_sha256": report_utils._sha256(required[2]),
        }
    )
    report_utils._write_json(output_root / "decision.json", decision)
    config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    config["command"] = command
    config["visual_review"] = {
        "source_path": str(source_root),
        "source_decision_sha256": report_utils._sha256(required[2]),
        "disposition": args.visual_disposition,
        "cell_computation_rerun": False,
    }
    report_utils._write_json(output_root / "config.json", config)
    report_utils._write_json(
        output_root / "visual_review.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", **config["visual_review"]},
    )
    h21._write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    output_root = args.out.resolve()
    command = shlex.join(
        [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    )
    if _review(args, output_root, command):
        return 0

    import torch

    assert args.source_results is not None
    assert args.development_decision is not None
    development = _development_receipt(args.development_decision)
    sources = _source_records(args.source_results)
    output_root.mkdir(parents=True, exist_ok=False)
    init_config, fit_config, patch_config = h21._configs(args)
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": PHASE,
        "command": command,
        "args": vars(args),
        "development_decision": development,
        "direct_init": asdict(init_config),
        "fit": asdict(fit_config),
        "source_patch": asdict(patch_config),
        "spt1": {
            "magic": "SPT1",
            "header_bytes": 16,
            "record_bytes": 7,
            "records": "strictly increasing raster-flat uint32le plus RGB8",
        },
        "persisted_source_results": [
            {
                "group": record["group"],
                "path": str(record["root"]),
                "phase": record["phase"],
                "direct_count": record["direct_count"],
                "metrics_sha256": record["metrics_sha256"],
                "attempts_sha256": record["attempts_sha256"],
                "decision_sha256": record["decision_sha256"],
                "manifest_sha256": record["manifest_sha256"],
            }
            for record in sources
        ],
        "source_snapshots": _snapshot_sources(output_root),
        "git": report_utils._git_record(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "limitations": [
            "Consumed one-seed dirty-source replay; it is not independent confirmation.",
            "All 40 direct fields and their H005 controls were copied without refitting.",
            "SPT1 is an explicit source-RGB residual layer outside the 7,000-Gaussian field.",
            "Target-known SPT1 construction and whole-image selection are encoder-side RDO.",
            "NPZ plus SPT1 is reference accounting, not a production codec rate.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record(
        *,
        group: str,
        lineage: str,
        source_row: dict[str, object],
        arm: str,
        started: float,
        error: Exception | None = None,
    ) -> None:
        item: dict[str, object] = {
            "source_replay_group": group,
            "source_lineage": lineage,
            "image": source_row["image"],
            "arm": arm,
            "status": "ok" if error is None else "error",
            "elapsed_seconds": time.perf_counter() - started,
        }
        if error is not None:
            item["error"] = f"{type(error).__name__}: {error}"[:1000]
        attempts.append(item)
        report_utils._write_json(
            output_root / "attempts.json",
            {"schema": REPORT_SCHEMA, "status": "diagnostic", "attempts": attempts},
        )

    run_started = time.perf_counter()
    for source in sources:
        group = str(source["group"])
        source_root = source["root"]
        controls = source["controls"]
        direct_rows = source["direct"]
        assert isinstance(source_root, Path)
        assert isinstance(controls, dict)
        assert isinstance(direct_rows, list)
        for source_direct in direct_rows:
            source_control = controls[str(source_direct["image"])]
            lineage = _row_lineage(source_direct, group)
            started = time.perf_counter()
            try:
                control = _copy_cell(
                    source_root=source_root,
                    source_row=source_control,
                    output_root=output_root,
                    group=group,
                    lineage=lineage,
                    arm=h21.CONTROL_ARM,
                )
                rows.append(control)
                record(
                    group=group,
                    lineage=lineage,
                    source_row=source_control,
                    arm=h21.CONTROL_ARM,
                    started=started,
                )
            except Exception as exc:
                record(
                    group=group,
                    lineage=lineage,
                    source_row=source_control,
                    arm=h21.CONTROL_ARM,
                    started=started,
                    error=exc,
                )

            started = time.perf_counter()
            try:
                row = _copy_cell(
                    source_root=source_root,
                    source_row=source_direct,
                    output_root=output_root,
                    group=group,
                    lineage=lineage,
                    arm=h21.DIRECT_ARM,
                )
                artifact_dir = output_root / str(row["artifact_dir"])
                field_path = artifact_dir / "field.gaussian.npz"
                if report_utils._sha256(field_path) != row["field_file_sha256"]:
                    raise RuntimeError("persisted field hash differs after copy")
                image_path = Path(str(row["source_path"]))
                image, loaded_mask, _ = report_utils._load_evaluation_raster(
                    image_path, None, max_side=args.max_side, mask_threshold=0.5
                )
                if loaded_mask is not None or image.shape[:2] != (
                    int(row["height"]),
                    int(row["width"]),
                ):
                    raise RuntimeError("persisted source raster contract differs")
                mask = np.ones(image.shape[:2], dtype=bool)
                field = GaussianField.load(str(field_path), device=args.device)
                baseline_render = h17._render_numpy(
                    field, fit_config, image.shape[0], image.shape[1]
                )
                row = h21._augment_direct_row(
                    output_root=output_root,
                    row=row,
                    image=image,
                    mask=mask,
                    baseline_render=baseline_render,
                    fit_config=fit_config,
                    patch_config=patch_config,
                    args=args,
                )
                rows.append(row)
                record(
                    group=group,
                    lineage=lineage,
                    source_row=source_direct,
                    arm=h21.DIRECT_ARM,
                    started=started,
                )
            except Exception as exc:
                record(
                    group=group,
                    lineage=lineage,
                    source_row=source_direct,
                    arm=h21.DIRECT_ARM,
                    started=started,
                    error=exc,
                )
            h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)

    decision = _decision(rows, attempts, visual_disposition="pending")
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    h21._write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
