#!/usr/bin/env python3
"""Replay HIER-020 once on the 20 persisted HIER-015--019 direct fields."""
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
from structsplat.gaussians import GaussianField  # noqa: E402
from structsplat.tail_recovery import (  # noqa: E402
    SparseTailPayload,
    apply_sparse_tail_payload,
    render_confidence_gated_self_prior,
    render_sparse_tail_payload,
)


REPORT_SCHEMA = h20.REPORT_SCHEMA
PHASE = "replay_consumed_h15_h19"
LINEAGES = (
    (
        "hier015",
        "structsplat.hier015_geometry_escape.diagnostic.v1",
        "direct_normalized_fixed7k",
    ),
    (
        "hier016",
        "structsplat.hier016_normalized_tail_repair.diagnostic.v1",
        "direct_normalized_fixed7k",
    ),
    (
        "hier017",
        "structsplat.hier017_normalization_epsilon.diagnostic.v1",
        "direct_eps1e8",
    ),
    (
        "hier018",
        "structsplat.hier018_counted_background.diagnostic.v1",
        "direct_no_background",
    ),
    (
        "hier019",
        "structsplat.hier019_confidence_tail.diagnostic.v1",
        "direct_no_recovery",
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-results", type=Path, nargs=5)
    source.add_argument("--review-from", type=Path)
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
    parser.add_argument("--tail-scale-multiplier", type=float, default=2.0)
    parser.add_argument("--tail-coverage-threshold", type=float, default=1e-8)
    parser.add_argument("--error-scale", type=float, default=4.0)
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
        "tail_scale_multiplier": 2.0,
        "tail_coverage_threshold": 1e-8,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-020 replay requires --{name.replace('_', '-')} {expected}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if args.source_results is not None and args.visual_disposition != "pending":
        raise SystemExit("record the visual verdict only with --review-from")


def _source_records(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path, (lineage, expected_schema, direct_arm) in zip(paths, LINEAGES):
        root = path.resolve()
        metrics_path = root / "metrics.json"
        manifest_path = root / "manifest.json"
        if not metrics_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"{root} is missing metrics.json or manifest.json")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("schema") != expected_schema:
            raise SystemExit(
                f"{root} has schema {payload.get('schema')!r}; expected {expected_schema!r}"
            )
        controls = [row for row in payload["rows"] if row.get("arm") == h20.CONTROL_ARM]
        direct = [row for row in payload["rows"] if row.get("arm") == direct_arm]
        if len(controls) != 4 or len(direct) != 4:
            raise SystemExit(
                f"{root} must expose four H005 controls and four {direct_arm!r} fields"
            )
        if {row["image"] for row in controls} != {row["image"] for row in direct}:
            raise SystemExit(f"{root} control/direct image keys differ")
        records.append(
            {
                "lineage": lineage,
                "root": root,
                "schema": expected_schema,
                "direct_arm": direct_arm,
                "metrics_path": metrics_path,
                "metrics_sha256": report_utils._sha256(metrics_path),
                "manifest_sha256": report_utils._sha256(manifest_path),
                "controls": controls,
                "direct": direct,
            }
        )
    return records


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier020_sparse_pixel_safe_tail.py",
        ROOT / "src" / "structsplat" / "tail_recovery.py",
        ROOT / "tasks" / "HIER-020-sparse-pixel-safe-tail.md",
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


def _copy_cell(
    *,
    source_root: Path,
    source_row: dict[str, object],
    output_root: Path,
    lineage: str,
    arm: str,
) -> dict[str, object]:
    source_dir = source_root / str(source_row["artifact_dir"])
    destination = (
        output_root
        / "artifacts"
        / f"{lineage}__{source_row['image']}__{arm}__n7000"
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
            "source_lineage": lineage,
            "source_result_schema": source_row["schema"],
            "source_result_arm": source_row["arm"],
            "persisted_field_reused_without_refit": True,
        }
    )
    report_utils._write_json(destination / "row.json", row)
    return row


def _retime_selected_decode(
    *,
    output_root: Path,
    row: dict[str, object],
    fit_config,
    tail_config,
    device: str,
) -> dict[str, object]:
    import torch

    from structsplat.fit import _render

    artifact_dir = output_root / str(row["artifact_dir"])
    field_path = artifact_dir / "field.gaussian.npz"
    height, width = int(row["height"]), int(row["width"])
    field = GaussianField.load(str(field_path), device=device)
    torch.cuda.synchronize()
    baseline_started = time.perf_counter()
    baseline = _render(field, fit_config, height, width)
    torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started

    if row["sst1_selected_mode"] == h20.CANDIDATE_MODE:
        payload = SparseTailPayload.from_bytes((artifact_dir / "candidate.sst1").read_bytes())
    else:
        payload = SparseTailPayload(height, width)
    selected_field = GaussianField.load(str(field_path), device=device)
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    decoded = render_sparse_tail_payload(
        selected_field, fit_config, height, width, payload, tail_config
    )
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    if payload.count:
        full = render_confidence_gated_self_prior(
            field, fit_config, height, width, tail_config
        )
        expected = apply_sparse_tail_payload(full, payload)
    else:
        expected = baseline
    torch.cuda.synchronize()
    parity = float((decoded - expected).abs().max().detach().cpu())
    row.update(
        {
            "sst1_full_frame_decode_seconds_v1": row["sst1_selected_tail_decode_seconds"],
            "sst1_full_frame_decode_time_ratio_v1": row["sst1_decode_time_ratio"],
            "sst1_ordinary_render_seconds": baseline_seconds,
            "sst1_selected_tail_decode_seconds": decode_seconds,
            "sst1_decode_time_ratio": decode_seconds / max(baseline_seconds, 1e-12),
            "sst1_selected_decode_payload_count": payload.count,
            "sst1_optimized_decode_parity_max_abs": parity,
            "sst1_decode_implementation": "ordinary_plus_coordinate_only_tail_v2",
        }
    )
    report_utils._write_json(artifact_dir / "row.json", row)
    return row


def _decision(
    rows: list[dict[str, object]],
    attempts: list[dict[str, object]],
    *,
    visual_disposition: str,
) -> dict[str, object]:
    direct = [row for row in rows if row.get("arm") == h20.DIRECT_ARM]
    controls = [row for row in rows if row.get("arm") == h20.CONTROL_ARM]
    pairs = h20._selected_pairs(rows)
    baseline_local_failures = [
        pair
        for pair in pairs
        if float(pair["baseline_pixel_max_delta"]) > 1e-12
        or float(pair["baseline_patch7_max_delta"]) > 1e-12
    ]
    selected_lineages = {
        str(row["image"]): row["source_lineage"] for row in direct
    }
    for pair in pairs:
        pair["source_lineage"] = selected_lineages[str(pair["image"])]

    def safe_delta(row: dict[str, object], key: str, *, upper: float) -> bool:
        if row["sst1_selected_mode"] == h20.BASELINE_MODE:
            return True
        return float(row["sst1_metric_deltas_vs_baseline"][key]) <= upper

    gates = {
        "complete_twenty_direct_rows": len(direct) == 20,
        "complete_twenty_controls": len(controls) == 20,
        "complete_twenty_pairs": len(pairs) == 20,
        "complete_attempt_ledger": len(attempts) == 40,
        "zero_failures": all(record.get("status") == "ok" for record in attempts),
        "all_persisted_without_refit": all(
            bool(row["persisted_field_reused_without_refit"]) for row in direct
        ),
        "all_exact_count": all(int(row["n_gaussians"]) == 7000 for row in direct),
        "all_field_bytes_unchanged": all(
            row["sst1_field_file_sha256_before"] == row["sst1_field_file_sha256_after"]
            for row in direct
        ),
        "all_payloads_canonical": all(
            bool(row["sst1_payload_roundtrip_exact"]) for row in direct
        ),
        "all_candidates_finite": all(bool(row["sst1_candidate_finite"]) for row in direct),
        "all_outside_payload_bit_exact": all(
            float(row["sst1_outside_identity_max_abs"]) == 0.0 for row in direct
        ),
        "all_render_and_decode_parity_le_2e_5": all(
            float(row["sst1_baseline_cold_parity_max_abs"]) <= 2e-5
            and float(row["sst1_candidate_repeated_parity_max_abs"]) <= 2e-5
            and float(row["sst1_optimized_decode_parity_max_abs"]) <= 2e-5
            for row in direct
        ),
        "all_selected_transactions_safe": all(
            row["sst1_selected_mode"] == h20.BASELINE_MODE
            or all(bool(value) for value in row["sst1_selection_clauses"].values())
            for row in direct
        ),
        "all_selected_mse_noninferior_vs_baseline": all(
            safe_delta(row, "mse_ratio", upper=1.0 + 1e-8) for row in direct
        ),
        "all_selected_ms_ssim_noninferior_vs_baseline": all(
            row["sst1_selected_mode"] == h20.BASELINE_MODE
            or float(row["sst1_metric_deltas_vs_baseline"]["ms_ssim_delta"]) >= -1e-7
            for row in direct
        ),
        "all_selected_lpips_noninferior_vs_baseline": all(
            safe_delta(row, "lpips_delta", upper=1e-7) for row in direct
        ),
        "all_selected_pixel_noninferior_vs_baseline": all(
            safe_delta(row, "pixel_max_delta", upper=1e-12) for row in direct
        ),
        "all_selected_patch_noninferior_vs_baseline": all(
            safe_delta(row, "patch7_max_delta", upper=1e-12) for row in direct
        ),
        "all_recorded_h005_local_failures_repaired": all(
            float(pair["pixel_max_delta"]) <= 1e-12
            and float(pair["patch7_max_delta"]) <= 1e-12
            for pair in baseline_local_failures
        ),
    }
    numeric_pass = all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": PHASE,
        "gates": gates,
        "selected_vs_h005_pairs": pairs,
        "baseline_local_failures": baseline_local_failures,
        "baseline_local_failure_count": len(baseline_local_failures),
        "selected_image_count": sum(
            row["sst1_selected_mode"] == h20.CANDIDATE_MODE for row in direct
        ),
        "proposed_safe_pixel_count": sum(
            int(row["sst1_selected_count"]) for row in direct
        ),
        "selected_pixel_count": sum(
            int(row["sst1_selected_count"])
            for row in direct
            if row["sst1_selected_mode"] == h20.CANDIDATE_MODE
        ),
        "candidate_side_bytes": sum(
            int(row["sst1_candidate_payload_bytes"]) for row in direct
        ),
        "selected_side_bytes": sum(
            int(row["sst1_selected_payload_bytes"]) for row in direct
        ),
        "attempt_count": len(attempts),
        "failure_count": sum(record.get("status") != "ok" for record in attempts),
        "numeric_bank_pass": numeric_pass,
        "visual_review_required": True,
        "visual_disposition": visual_disposition,
        "bounded_bank_pass": numeric_pass and visual_disposition == "pass",
        "numeric_candidates": [h20.CANDIDATE_MODE] if numeric_pass else [],
        "interpretation": (
            "Consumed replay passed numeric and recorded visual gates."
            if numeric_pass and visual_disposition == "pass"
            else (
                "Consumed replay is numerically safe and awaits native visual review."
                if numeric_pass
                else "A frozen consumed-field replay gate failed; do not access tests/test_images."
            )
        ),
    }


def _review(args: argparse.Namespace, output_root: Path, command: str) -> bool:
    if args.review_from is None:
        return False
    source_root = args.review_from.resolve()
    required = [source_root / name for name in ("metrics.json", "attempts.json", "decision.json")]
    if not source_root.is_dir() or not all(path.is_file() for path in required):
        raise SystemExit("--review-from is missing its complete replay ledgers")
    rows = json.loads(required[0].read_text(encoding="utf-8")).get("rows", [])
    attempts = json.loads(required[1].read_text(encoding="utf-8")).get("attempts", [])
    prior = json.loads(required[2].read_text(encoding="utf-8"))
    if (
        prior.get("schema") != REPORT_SCHEMA
        or prior.get("phase") != PHASE
        or not prior.get("numeric_bank_pass")
        or prior.get("visual_disposition") != "pending"
    ):
        raise SystemExit("--review-from is not a pending numeric-pass consumed replay")
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
    h20._write_report(output_root, rows, decision, command)
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
    sources = _source_records(args.source_results)
    output_root.mkdir(parents=True, exist_ok=False)
    init_config, fit_config, tail_config = h20._configs(args)
    config = {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": PHASE,
        "command": command,
        "args": vars(args),
        "direct_init": asdict(init_config),
        "fit": asdict(fit_config),
        "tail": asdict(tail_config),
        "persisted_source_results": [
            {
                "lineage": record["lineage"],
                "path": str(record["root"]),
                "schema": record["schema"],
                "direct_arm": record["direct_arm"],
                "metrics_sha256": record["metrics_sha256"],
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
            "Consumed one-seed dirty-source replay; it is not confirmation evidence.",
            "The 20 direct fields and their H005 controls were copied without refitting.",
            "Target-known SST1 selection and the whole-image transaction are encoder-side RDO.",
        ],
    }
    report_utils._write_json(output_root / "config.json", config)
    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []

    def record(
        lineage: str,
        source_row: dict[str, object],
        arm: str,
        started: float,
        error: Exception | None = None,
    ) -> None:
        item: dict[str, object] = {
            "lineage": lineage,
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
        lineage = str(source["lineage"])
        source_root = source["root"]
        assert isinstance(source_root, Path)
        controls = {str(row["image"]): row for row in source["controls"]}
        direct_rows = source["direct"]
        assert isinstance(direct_rows, list)
        for source_direct in direct_rows:
            source_control = controls[str(source_direct["image"])]
            started = time.perf_counter()
            try:
                control = _copy_cell(
                    source_root=source_root,
                    source_row=source_control,
                    output_root=output_root,
                    lineage=lineage,
                    arm=h20.CONTROL_ARM,
                )
                rows.append(control)
                record(lineage, source_control, h20.CONTROL_ARM, started)
            except Exception as exc:
                record(lineage, source_control, h20.CONTROL_ARM, started, exc)

            started = time.perf_counter()
            try:
                row = _copy_cell(
                    source_root=source_root,
                    source_row=source_direct,
                    output_root=output_root,
                    lineage=lineage,
                    arm=h20.DIRECT_ARM,
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
                row = h20._augment_direct_row(
                    output_root=output_root,
                    row=row,
                    image=image,
                    mask=mask,
                    baseline_render=baseline_render,
                    fit_config=fit_config,
                    tail_config=tail_config,
                    args=args,
                )
                row = _retime_selected_decode(
                    output_root=output_root,
                    row=row,
                    fit_config=fit_config,
                    tail_config=tail_config,
                    device=args.device,
                )
                rows.append(row)
                record(lineage, source_direct, h20.DIRECT_ARM, started)
            except Exception as exc:
                record(lineage, source_direct, h20.DIRECT_ARM, started, exc)
            h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)

    decision = _decision(rows, attempts, visual_disposition="pending")
    decision["elapsed_seconds"] = time.perf_counter() - run_started
    report_utils._write_json(output_root / "decision.json", decision)
    h15._write_tables(output_root, rows, schema=REPORT_SCHEMA)
    h20._write_report(output_root, rows, decision, command)
    h15._write_manifest(output_root, schema=REPORT_SCHEMA)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
