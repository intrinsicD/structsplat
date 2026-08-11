#!/usr/bin/env python3
"""Run HIER-028's frozen residual-pursuit pure-additive confirmation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from html import escape
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.experiments import hier022_additive_continuation as h22  # noqa: E402
from scripts.experiments import hier023_unit_gauge_continuation as h23  # noqa: E402
from scripts.experiments import hier024_gauge_geometry_projection as h24  # noqa: E402
from scripts.experiments import hier026_progressive_additive_capacity as h26  # noqa: E402
from scripts.experiments import hier027_cold_additive_capacity as h27  # noqa: E402
from structsplat.config import StructureTensorConfig  # noqa: E402
from structsplat.residual_pursuit_additive import (  # noqa: E402
    ResidualPursuitAdditiveConfig,
    append_residual_pursuit_gaussians,
)


REPORT_SCHEMA = "structsplat.hier028_residual_pursuit_additive.confirmation.v1"
ARMS = (
    "normalized_plain_n640",
    "cold_additive_projected_n960",
    "residual_pursuit_additive_n1024",
    "cold_additive_projected_n1024",
)
PROJECTED_ENDPOINT_ARMS = frozenset(
    ("cold_additive_projected_n960", "cold_additive_projected_n1024")
)
PURE_ADDITIVE_ARMS = frozenset(set(ARMS) - {"normalized_plain_n640"})
COUNT_BY_ARM = {
    "normalized_plain_n640": 640,
    "cold_additive_projected_n960": 960,
    "residual_pursuit_additive_n1024": 1024,
    "cold_additive_projected_n1024": 1024,
}
GAUSSIAN_ROW_UPDATES_BY_ARM = {
    "normalized_plain_n640": 640 * 500,
    "cold_additive_projected_n960": 960 * 500,
    "residual_pursuit_additive_n1024": 960 * 500,
    "cold_additive_projected_n1024": 1024 * 500,
}
SELECTION_SALT = "HIER-028-confirm-v1:"
SELECTION_ORDER = (
    "0804.png",
    "0830.png",
    "0822.png",
    "0812.png",
    "0810.png",
    "0862.png",
    "0803.png",
    "0826.png",
)
SELECTION_BINDINGS = {
    "0804.png": "0686f57768896183a307e62c52b53806515c65b82856225f0053c3b51c7da0c3",
    "0830.png": "0c84c4de7ca7ce6cfb42573327b2c34933b88bc53c939e0b5a403f747e5bca5f",
    "0822.png": "130cdf4d4c1a67dab7b4ce502044a2ecc5f6f1b8bd01365dce3ffc4f11311db3",
    "0812.png": "132e21bc39e02a6cde90ba28d3a64c12d575bb6d8e2a001c5154924edda6a63c",
    "0810.png": "1704a6e1b96ad30381b0dfba6e4ab8a5d3ee7a61df23689ac625c9fe46a996fd",
    "0862.png": "18240279a254669300683c105df63f9584d1a396417d783ef5db734a05eb2313",
    "0803.png": "1a48bfa234e74bd95c2f7875565809acaedf73de995088c0b532c105f1eb0e06",
    "0826.png": "1ac09ff808f01c4e326025121790ba7aa336e7889bf4ad34437fc1dc7042729c",
}
SOURCE_BINDINGS = {
    "0804.png": "16b5fdbe808b868bed0be32f235208a1716d44e271a37b79cbc77ab53d2f6bdb",
    "0830.png": "4eb18566ab01447a06daf0314a3711aa78cea5ca0eaa47cfedafbceeb6dd0a3e",
    "0822.png": "a1d308fd62adecb1ea8b0fa8d0c687c92d3cf0d3358e598c8b97aca1b9cf8ad0",
    "0812.png": "49e45b8922872b44ece90db047756f3a5356612bb6ee30bdc23df2bd208ec861",
    "0810.png": "6940c660b97d2c5f1113101c3e6360d1d6886743c5796cad52224b8076b903f8",
    "0862.png": "31a02d7392ee9dadd4b8a2c1b5b9d670943135d0d40e85d4178ab77923c75548",
    "0803.png": "4b0148a9a1ff877ad9f76e65736a50cc36e10822b5d8ccd2abb2988ff4e1782b",
    "0826.png": "b0f675a14e8fe9f2ec0b705bee98d75f8a22478eafdc1a0a0afc0f820bc5ab4d",
}
ARCHIVE_SHA256 = h26.ARCHIVE_SHA256
ARCHIVE_BYTES = h26.ARCHIVE_BYTES
COEFFICIENT_LIMIT = 16.0
PARITY_LIMIT = 2e-5
FOUR_ARRAY_KEYS = frozenset(("means", "log_scales", "rotations", "colors"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-side", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-chunk", type=int, default=256)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "max_side": 160,
        "seeds": [0, 1],
        "device": "cuda",
        "lpips": True,
        "render_chunk": 256,
    }
    for name, expected in frozen.items():
        if getattr(args, name) != expected:
            raise SystemExit(
                f"frozen HIER-028 protocol requires {name}={expected!r}, "
                f"got {getattr(args, name)!r}"
            )
    if not math.isfinite(args.error_scale) or args.error_scale <= 0.0:
        raise SystemExit("--error-scale must be finite and positive")
    if not args.images.is_dir():
        raise SystemExit(f"image directory does not exist: {args.images}")
    args.iters = 500
    args.budgets = [640]


def _command() -> str:
    return " ".join(shlex.quote(value) for value in sys.argv)


def _write_json(path: Path, value: object) -> None:
    h22._write_json(path, value)


def _discover_sources(root: Path) -> list[Path]:
    actual_names = sorted(path.name for path in root.iterdir() if path.is_file())
    if actual_names != sorted(SELECTION_ORDER):
        raise SystemExit(
            "HIER-028 extraction root must contain exactly the eight bound members: "
            f"got {actual_names!r}"
        )
    paths = [root / name for name in SELECTION_ORDER]
    hashes = {path.name: h22.report_utils._sha256(path) for path in paths}
    if hashes != SOURCE_BINDINGS:
        raise SystemExit(f"HIER-028 source hash binding differs: {hashes!r}")
    return [path.resolve() for path in paths]


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier023_unit_gauge_continuation.py",
        ROOT / "scripts" / "experiments" / "hier024_gauge_geometry_projection.py",
        ROOT / "scripts" / "experiments" / "hier026_progressive_additive_capacity.py",
        ROOT / "scripts" / "experiments" / "hier027_cold_additive_capacity.py",
        ROOT / "src" / "structsplat" / "residual_pursuit_additive.py",
        ROOT / "src" / "structsplat" / "endpoint_appearance_projection.py",
        ROOT / "tests" / "test_residual_pursuit_additive.py",
        ROOT / "tests" / "test_residual_pursuit_confirmation.py",
        ROOT / "tasks" / "HIER-028-residual-pursuit-additive-confirmation.md",
        ROOT / "scripts" / "check_report_bundle.py",
    )
    records = []
    for source in paths:
        relative = source.relative_to(ROOT)
        destination = output_root / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "repository_path": str(relative),
                "snapshot_path": str(destination.relative_to(output_root)),
                "bytes": destination.stat().st_size,
                "sha256": h22.report_utils._sha256(destination),
            }
        )
    return records


def _tail_config(args: argparse.Namespace) -> ResidualPursuitAdditiveConfig:
    return ResidualPursuitAdditiveConfig(
        tail_gaussians=64,
        scale_px=0.35,
        coefficient_abs_limit=COEFFICIENT_LIMIT,
        sigma_cutoff=3.0,
        render_chunk=args.render_chunk,
        renderer="cuda_additive",
    )


def _pursuit_method(
    base_method: dict[str, object], target: np.ndarray, args: argparse.Namespace, torch
) -> dict[str, object]:
    result = append_residual_pursuit_gaussians(
        base_method["field"], target, config=_tail_config(args)
    )
    method = dict(base_method)
    method.update(
        {
            "field": result.field,
            "expected": result.reconstruction_raw,
            "coverage": h26._coverage(result.field, target, args, torch),
            "endpoint_parity": max(
                float(base_method["endpoint_parity"]),
                result.analytic_render_parity_max_abs,
            ),
            "completed": result.completed,
            "method_status": result.status,
            "pursuit_result": result,
            "pursuit_seconds": result.elapsed_seconds,
            "base_projection_final_digest": base_method["final_field_digest"],
            "final_field_digest": result.endpoint_field_digest,
            "preprojection_endpoint_digest": result.endpoint_field_digest,
            "shared_endpoint_group": "projected_n960_base_plus_tail64",
        }
    )
    return method


def _save_shared_audit(
    output_root: Path,
    image_stem: str,
    seed: int,
    initial640,
    normalized: dict[str, object],
    cold960: dict[str, object],
    projected960: dict[str, object],
    pursuit: dict[str, object],
    cold1024: dict[str, object],
    projected1024: dict[str, object],
) -> dict[str, object]:
    directory = output_root / "shared" / f"{image_stem}__s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    pursuit_result = pursuit["pursuit_result"]
    fields = {
        "n640_initial": initial640,
        "normalized_endpoint": normalized["field"],
        "n960_initial": cold960["audit_initial_field"],
        "n960_training": cold960["audit_training_field"],
        "n960_preprojection": cold960["field"],
        "n960_projected_base": projected960["field"],
        "pursuit_tail64": pursuit_result.tail_field,
        "pursuit_n1024_endpoint": pursuit_result.field,
        "cold_n1024_initial": cold1024["audit_initial_field"],
        "cold_n1024_training": cold1024["audit_training_field"],
        "cold_n1024_preprojection": cold1024["field"],
        "cold_n1024_projected": projected1024["field"],
    }
    records = {}
    for name, field in fields.items():
        record = h26._save_field(directory / f"{name}.field.gaussian.npz", field)
        record["path"] = str(Path(record["path"]).relative_to(output_root))
        records[name] = record
    receipt = {
        "schema": REPORT_SCHEMA,
        "image": image_stem,
        "seed": seed,
        "fields": records,
        "counts": {"normalized": 640, "base": 960, "tail": 64, "total": 1024},
        "steps": {"normalized": 500, "base": 500, "cold_control": 500, "tail": 0},
        "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM,
        "residual_scan_pixel_evaluations": pursuit_result.residual_scan_pixel_evaluations,
        "tail_kernel_pixel_updates": pursuit_result.tail_kernel_pixel_updates,
        "base_prefix_bit_exact": pursuit_result.base_prefix_bit_exact,
        "analytic_render_parity_max_abs": pursuit_result.analytic_render_parity_max_abs,
    }
    _write_json(directory / "receipt.json", receipt)
    return {
        "dir": str(directory.relative_to(output_root)),
        "receipt_path": str((directory / "receipt.json").relative_to(output_root)),
        "receipt_sha256": h22.report_utils._sha256(directory / "receipt.json"),
        "fields": records,
    }


def _projection_record(method: dict[str, object]) -> dict[str, object]:
    return h26._projection_record(method)


def _write_cell(
    *,
    output_root: Path,
    image_path: Path,
    target: np.ndarray,
    raster: dict[str, object],
    seed: int,
    arm: str,
    method: dict[str, object],
    initial_file_sha256: str,
    shared_audit: dict[str, object],
    args: argparse.Namespace,
    torch,
) -> dict[str, object]:
    count = COUNT_BY_ARM[arm]
    row = h23._write_cell(
        output_root=output_root,
        image_path=image_path,
        target=target,
        raster=raster,
        seed=seed,
        budget=count,
        arm=arm,
        initial_field_sha256=initial_file_sha256,
        init_seconds=float(method["init_seconds"]),
        method=method,
        args=args,
        torch=torch,
    )
    artifact_dir = output_root / str(row["artifact_dir"])
    incoming_path = artifact_dir / "incoming.field.gaussian.npz"
    proposal_path = artifact_dir / "proposal.field.gaussian.npz"
    method["incoming_field"].save(str(incoming_path))
    method["proposal_field"].save(str(proposal_path))
    projection = _projection_record(method)
    projection["scope"] = (
        "base_n960" if arm == "residual_pursuit_additive_n1024" else "endpoint"
    )
    _write_json(artifact_dir / "projection_history.json", projection)
    pursuit_result = method.get("pursuit_result")
    pursuit_path = artifact_dir / "pursuit_history.json"
    if pursuit_result is None:
        pursuit_payload = {"schema": REPORT_SCHEMA, "applied": False, "trajectory": []}
    else:
        tail_path = artifact_dir / "tail.field.gaussian.npz"
        pursuit_result.tail_field.save(str(tail_path))
        pursuit_payload = {
            "schema": REPORT_SCHEMA,
            "applied": True,
            "config": asdict(_tail_config(args)),
            "base_count": pursuit_result.base_count,
            "tail_count": pursuit_result.tail_count,
            "total_count": pursuit_result.total_count,
            "base_field_digest": pursuit_result.base_field_digest,
            "tail_field_digest": pursuit_result.tail_field_digest,
            "endpoint_field_digest": pursuit_result.endpoint_field_digest,
            "base_prefix_bit_exact": pursuit_result.base_prefix_bit_exact,
            "fixed_tail_geometry": pursuit_result.fixed_tail_geometry,
            "training_payload_removed": pursuit_result.training_payload_removed,
            "residual_scan_pixel_evaluations": (
                pursuit_result.residual_scan_pixel_evaluations
            ),
            "tail_kernel_pixel_updates": pursuit_result.tail_kernel_pixel_updates,
            "analytic_render_parity_max_abs": (
                pursuit_result.analytic_render_parity_max_abs
            ),
            "initial_pixel_rmse_max": pursuit_result.initial_pixel_rmse_max,
            "final_pixel_rmse_max": pursuit_result.final_pixel_rmse_max,
            "coefficient_abs_max": pursuit_result.coefficient_abs_max,
            "trajectory": pursuit_result.trajectory_records(),
            "tail_file": "tail.field.gaussian.npz",
            "tail_file_sha256": h22.report_utils._sha256(tail_path),
        }
    _write_json(pursuit_path, pursuit_payload)
    _write_json(
        artifact_dir / "geometry_history.json",
        {
            "base_projection_final_digest": method.get(
                "base_projection_final_digest", method["final_field_digest"]
            ),
            "final_field_digest": method["final_field_digest"],
            "base_prefix_bit_exact": (
                None if pursuit_result is None else pursuit_result.base_prefix_bit_exact
            ),
            "fixed_tail_geometry": (
                None if pursuit_result is None else pursuit_result.fixed_tail_geometry
            ),
            "training_payload_stripped": arm in PURE_ADDITIVE_ARMS,
        },
    )
    init_count = 960 if arm == "residual_pursuit_additive_n1024" else count
    fit_config = (
        asdict(h23._fit_config(args, "normalized_plain"))
        if arm == "normalized_plain_n640"
        else asdict(h26._cold_fit_config(args, init_count))
    )
    _write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "seed": seed,
            "count": count,
            "init": asdict(h22._init_config(init_count, seed)),
            "fit": fit_config,
            "projection": (
                asdict(h24._projection_config(args))
                if arm != "normalized_plain_n640"
                else None
            ),
            "safety": (
                asdict(h24._safety_config())
                if arm != "normalized_plain_n640"
                else None
            ),
            "pursuit": (
                asdict(_tail_config(args))
                if arm == "residual_pursuit_additive_n1024"
                else None
            ),
            "shared_audit_receipt": shared_audit["receipt_path"],
        },
    )
    pure = arm in PURE_ADDITIVE_ARMS
    field_keys = row["field_npz_keys"]
    base_projection_final = method.get(
        "base_projection_final_digest", method["final_field_digest"]
    )
    row.update(
        {
            "schema": REPORT_SCHEMA,
            "phase": "untouched_confirmation",
            "renderer": method["renderer"],
            "source_rank": SELECTION_ORDER.index(image_path.name) + 1,
            "source_sha256": SOURCE_BINDINGS[image_path.name],
            "selection_salt": SELECTION_SALT,
            "selection_sha256": SELECTION_BINDINGS[image_path.name],
            "archive_sha256": ARCHIVE_SHA256,
            "archive_bytes": ARCHIVE_BYTES,
            "count_ratio_vs_normalized_n640": count / 640.0,
            "pure_additive_endpoint": pure,
            "four_array_endpoint_exact": not pure or set(field_keys) == FOUR_ARRAY_KEYS,
            "training_payload_present": pure and set(field_keys) != FOUR_ARRAY_KEYS,
            "selected_lambda": 0.0 if pure else None,
            "attempted_steps": 500,
            "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM[arm],
            "endpoint_internal_parity_max_abs": method["endpoint_parity"],
            "diagnostic_renderer_calls_fit": method["diagnostic_renderer_calls"],
            "projection_applied": method["projection_applied"],
            "projection_scope": projection["scope"],
            "projection_selected": method["projection_selected"],
            "projection_reason": method["projection_reason"],
            "projection_clauses": method["projection_clauses"],
            "projection_seconds": method["projection_seconds"],
            "projection_metric_seconds": method["projection_metric_seconds"],
            "projection_selected_iteration": projection["selected_iteration"],
            "projection_initial_sse": projection["initial_sse"],
            "projection_final_sse": projection["final_sse"],
            "projection_forward_applications": projection["forward_applications"],
            "projection_transpose_applications": projection["transpose_applications"],
            "projection_relative_normal_residual_max": projection[
                "relative_normal_residual_max"
            ],
            "projection_adjoint_relative_error": projection["adjoint_relative_error"],
            "projection_initial_operator_parity_max_abs": projection[
                "initial_operator_parity_max_abs"
            ],
            "projection_maintained_render_parity_max_abs": projection[
                "maintained_render_parity_max_abs"
            ],
            "projection_geometry_exact": projection["geometry_exact"],
            "incoming_field_digest": method["incoming_field_digest"],
            "proposal_field_digest": method["proposal_field_digest"],
            "base_projection_final_digest": base_projection_final,
            "final_field_digest": method["final_field_digest"],
            "incoming_field_file_sha256": h22.report_utils._sha256(incoming_path),
            "proposal_field_file_sha256": h22.report_utils._sha256(proposal_path),
            "initial_field_digest": method["initial_field_digest"],
            "preprojection_endpoint_digest": method["preprojection_endpoint_digest"],
            "shared_endpoint_group": method["shared_endpoint_group"],
            "shared_audit_dir": shared_audit["dir"],
            "shared_audit_receipt": shared_audit["receipt_path"],
            "shared_audit_receipt_sha256": shared_audit["receipt_sha256"],
            "pursuit_applied": pursuit_result is not None,
            "pursuit_base_count": None if pursuit_result is None else pursuit_result.base_count,
            "pursuit_tail_count": None if pursuit_result is None else pursuit_result.tail_count,
            "pursuit_base_field_digest": (
                None if pursuit_result is None else pursuit_result.base_field_digest
            ),
            "pursuit_tail_field_digest": (
                None if pursuit_result is None else pursuit_result.tail_field_digest
            ),
            "pursuit_base_prefix_bit_exact": (
                None if pursuit_result is None else pursuit_result.base_prefix_bit_exact
            ),
            "pursuit_fixed_tail_geometry": (
                None if pursuit_result is None else pursuit_result.fixed_tail_geometry
            ),
            "pursuit_analytic_render_parity_max_abs": (
                None
                if pursuit_result is None
                else pursuit_result.analytic_render_parity_max_abs
            ),
            "pursuit_residual_scan_pixel_evaluations": (
                0
                if pursuit_result is None
                else pursuit_result.residual_scan_pixel_evaluations
            ),
            "pursuit_tail_kernel_pixel_updates": (
                0 if pursuit_result is None else pursuit_result.tail_kernel_pixel_updates
            ),
            "pursuit_renderer_calls": (
                0 if pursuit_result is None else pursuit_result.renderer_calls
            ),
            "pursuit_seconds": (
                0.0 if pursuit_result is None else pursuit_result.elapsed_seconds
            ),
            "pursuit_history_path": str(pursuit_path.relative_to(output_root)),
            "pursuit_history_sha256": h22.report_utils._sha256(pursuit_path),
        }
    )
    for prefix, values in (
        ("incoming", method["incoming_selection_metrics"]),
        ("proposal", method["proposal_selection_metrics"]),
    ):
        for key in ("raw_mse", "ms_ssim", "lpips", "pixel_max", "patch7_max"):
            row[f"{prefix}_{key}"] = None if values is None else values[key]
    row["pipeline_algorithm_seconds"] = (
        float(row["pipeline_algorithm_seconds"])
        + float(method["projection_seconds"])
        + float(row["pursuit_seconds"])
    )
    row["total_seconds"] = (
        float(row["total_seconds"])
        + float(method["projection_seconds"])
        + float(method["projection_metric_seconds"])
        + float(row["pursuit_seconds"])
    )
    _write_json(artifact_dir / "row.json", row)
    return row


def _write_tables(output_root: Path, rows: list[dict[str, object]]) -> None:
    _write_json(
        output_root / "metrics.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "rows": rows},
    )
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    columns = sorted({key for row in rows for key in row})
    with (output_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _integrity(rows: list[dict[str, object]], arm: str) -> bool:
    count = COUNT_BY_ARM[arm]
    pure = arm in PURE_ADDITIVE_ARMS
    return bool(
        len(rows) == len(SOURCE_BINDINGS) * 2
        and all(
            row["completed"]
            and row["method_status"] == "completed"
            and row["n_gaussians"] == row["target_gaussians"] == count
            and row["finite_reconstruction"]
            and float(row["coefficient_abs_max"]) <= COEFFICIENT_LIMIT
            and float(row["endpoint_internal_parity_max_abs"]) <= PARITY_LIMIT
            and float(row["maintained_render_parity_max_abs"]) <= PARITY_LIMIT
            and float(row["repeated_render_parity_max_abs"]) <= PARITY_LIMIT
            and (
                not pure
                or (
                    row["selected_lambda"] == 0.0
                    and row["semantic_family"] == "additive_rgb_peak_one_v1"
                    and row["renderer"] == "cuda_additive"
                    and row["four_array_endpoint_exact"]
                    and not row["mass_payload_present"]
                    and not row["denominator_payload_present"]
                    and not row["optimizer_payload_present"]
                    and not row["auxiliary_rgb_payload_present"]
                    and not row["training_payload_present"]
                )
            )
            for row in rows
        )
    )


def _decision(rows: list[dict[str, object]]) -> dict[str, object]:
    expected_count = len(SOURCE_BINDINGS) * 2
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    complete = all(len(by_arm[arm]) == expected_count for arm in ARMS)
    aggregates = {
        arm: {
            "cell_count": len(by_arm[arm]),
            "count": COUNT_BY_ARM[arm],
            "mean_psnr_db": _mean(by_arm[arm], "psnr_db") if by_arm[arm] else None,
            "mean_ms_ssim": _mean(by_arm[arm], "ms_ssim") if by_arm[arm] else None,
            "mean_lpips": _mean(by_arm[arm], "lpips") if by_arm[arm] else None,
            "mean_pixel_max": _mean(by_arm[arm], "artifact_pixel_rmse_max")
            if by_arm[arm]
            else None,
            "mean_patch7_max": _mean(by_arm[arm], "artifact_patch_rmse_max_7")
            if by_arm[arm]
            else None,
            "mean_fit_seconds": _mean(by_arm[arm], "fit_seconds")
            if by_arm[arm]
            else None,
            "mean_gaussian_row_updates": _mean(by_arm[arm], "gaussian_row_updates")
            if by_arm[arm]
            else None,
            "mean_pursuit_seconds": _mean(by_arm[arm], "pursuit_seconds")
            if by_arm[arm]
            else None,
            "projection_selected_count": sum(
                bool(row["projection_selected"]) for row in by_arm[arm]
            ),
        }
        for arm in ARMS
    }
    integrity = {
        arm: _integrity(by_arm[arm], arm) if complete else False for arm in ARMS
    }
    projection_fail_closed = complete and all(
        (row["projection_selected"] and all(row["projection_clauses"].values()))
        or (
            not row["projection_selected"]
            and row["base_projection_final_digest"] == row["incoming_field_digest"]
        )
        for arm in PURE_ADDITIVE_ARMS
        for row in by_arm[arm]
    )
    pursuit_rows = by_arm["residual_pursuit_additive_n1024"]
    pursuit_contract = complete and all(
        row["pursuit_applied"]
        and row["pursuit_base_count"] == 960
        and row["pursuit_tail_count"] == 64
        and row["pursuit_base_prefix_bit_exact"]
        and row["pursuit_fixed_tail_geometry"]
        and float(row["pursuit_analytic_render_parity_max_abs"]) <= PARITY_LIMIT
        and row["pursuit_base_field_digest"] == row["base_projection_final_digest"]
        for row in pursuit_rows
    )
    shared_base = complete
    base_by_key = {
        (row["image"], row["seed"]): row["final_field_digest"]
        for row in by_arm["cold_additive_projected_n960"]
    }
    for row in pursuit_rows:
        shared_base = shared_base and (
            row["pursuit_base_field_digest"]
            == base_by_key.get((row["image"], row["seed"]))
        )
    work_exact = complete and all(
        row["attempted_steps"] == 500
        and row["gaussian_row_updates"] == GAUSSIAN_ROW_UPDATES_BY_ARM[arm]
        for arm in ARMS
        for row in by_arm[arm]
    )
    quality: dict[str, object] = {}
    if complete:
        normalized = by_arm["normalized_plain_n640"]
        for arm in (
            "cold_additive_projected_n960",
            "residual_pursuit_additive_n1024",
            "cold_additive_projected_n1024",
        ):
            quality[arm] = h26._quality_gate(by_arm[arm], normalized)
            quality[arm]["integrity_pass"] = integrity[arm]
            quality[arm]["numeric_quality_capable"] = bool(
                integrity[arm]
                and projection_fail_closed
                and quality[arm]["numeric_pass"]
            )
    base_local_nonregression = False
    if complete:
        base_rows = {
            (row["image"], row["seed"]): row
            for row in by_arm["cold_additive_projected_n960"]
        }
        base_local_nonregression = all(
            float(row["artifact_pixel_rmse_max"])
            <= float(base_rows[(row["image"], row["seed"])]["artifact_pixel_rmse_max"])
            and float(row["artifact_patch_rmse_max_7"])
            <= float(base_rows[(row["image"], row["seed"])]["artifact_patch_rmse_max_7"])
            for row in pursuit_rows
        )
    pursuit_quality = quality.get("residual_pursuit_additive_n1024", {})
    robust_pursuit = bool(
        pursuit_quality.get("numeric_quality_capable", False)
        and pursuit_quality["mean_psnr_delta_db"] >= 0.50
        and pursuit_quality["minimum_psnr_delta_db"] >= 0.0
        and base_local_nonregression
    )
    gates = {
        "all_cells_present": complete,
        "all_arm_integrity": complete and all(integrity.values()),
        "projection_transactions_fail_closed": projection_fail_closed,
        "pursuit_contract_exact": pursuit_contract,
        "shared_projected_n960_base_exact": shared_base,
        "work_accounting_exact": work_exact,
        "pursuit_local_nonregression_vs_base": base_local_nonregression,
    }
    numeric_pass = robust_pursuit and all(gates.values())
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "untouched_confirmation",
        "aggregates": aggregates,
        "integrity": integrity,
        "quality": quality,
        "gates": gates,
        "pursuit_robust_numeric": robust_pursuit,
        "same_count_cold_numeric": bool(
            quality.get("cold_additive_projected_n1024", {}).get(
                "numeric_quality_capable", False
            )
        ),
        "normalization_not_required_for_fidelity_numeric": numeric_pass,
        "numeric_selected_arm": (
            "residual_pursuit_additive_n1024" if numeric_pass else None
        ),
        "numeric_pass": numeric_pass,
        "visual_review": "pending_native_audit",
        "overall_pass": False,
        "formal_claim_ready": False,
        "interpretation": (
            "Residual pursuit passes numerically; audit native visuals."
            if numeric_pass
            else "Frozen residual-pursuit candidate fails; retain without tuning."
        ),
    }


def _write_report(
    output_root: Path, rows: list[dict[str, object]], decision: dict[str, object]
) -> None:
    table_rows = []
    cards = []
    for row in rows:
        artifact = escape(str(row["artifact_dir"]))
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['image']))}</td><td>{int(row['seed'])}</td>"
            f"<td>{escape(str(row['arm']))}</td><td>{int(row['n_gaussians'])}</td>"
            f"<td>{float(row['psnr_db']):.3f}</td>"
            f"<td>{float(row['ms_ssim']):.5f}</td><td>{float(row['lpips']):.5f}</td>"
            f"<td>{float(row['artifact_pixel_rmse_max']):.4f}</td>"
            f"<td>{float(row['artifact_patch_rmse_max_7']):.4f}</td>"
            f"<td><a href='{artifact}/source.png'>source</a> · "
            f"<a href='{artifact}/reconstruction.png'>full</a> · "
            f"<a href='{artifact}/reconstruction_crop.png'>crop</a> · "
            f"<a href='{artifact}/error.png'>error</a> · "
            f"<a href='{artifact}/learning_curve.svg'>curve</a></td></tr>"
        )
        cards.append(
            f"<section><h3>{escape(str(row['image']))} · seed {int(row['seed'])} · "
            f"{escape(str(row['arm']))}</h3>"
            f"<a href='{artifact}/source.png'><img src='{artifact}/source.png'></a>"
            f"<a href='{artifact}/reconstruction.png'><img src='{artifact}/reconstruction.png'></a>"
            f"<a href='{artifact}/error.png'><img src='{artifact}/error.png'></a>"
            f"<a href='{artifact}/reconstruction_crop.png'>"
            f"<img src='{artifact}/reconstruction_crop.png'></a></section>"
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>HIER-028 residual-pursuit additive confirmation</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1900px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-028 residual-pursuit pure-additive confirmation</h1>
<p><strong>Untouched-data producer confirmation.</strong> Protocol, filenames, archive, and member
hashes preceded selected-image decode; dirty source and producer review keep this provisional.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="manifest.json">manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>seed</th><th>arm</th><th>N</th><th>PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>artifacts</th>
</tr>{''.join(table_rows)}</table><h2>Native visual audit</h2>{''.join(cards)}</body></html>"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def _write_manifest(output_root: Path) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": h22.report_utils._sha256(path),
                }
            )
    _write_json(
        output_root / "manifest.json",
        {"schema": REPORT_SCHEMA, "status": "diagnostic", "files": files},
    )


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if (args.out / "COMPLETED").is_file():
        raise SystemExit(f"completed HIER-028 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-028 protocol requires CUDA")
    sources = _discover_sources(args.images)
    _write_json(args.out / "environment.json", h22._environment(torch))
    snapshots = _snapshot_sources(args.out)
    _write_json(
        args.out / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "status": "diagnostic",
            "command": _command(),
            "git": h22._git_record(),
            "source_snapshots": snapshots,
            "archive": {
                "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
                "bytes": ARCHIVE_BYTES,
                "sha256": ARCHIVE_SHA256,
            },
            "source_selection": {
                "salt": SELECTION_SALT,
                "order": list(SELECTION_ORDER),
                "excluded_hier026_hier027": sorted(
                    ["0847.png", "0860.png", "0895.png", "0898.png"]
                    + list(h27.SELECTION_ORDER)
                ),
                "selection_bindings": SELECTION_BINDINGS,
                "source_bindings": SOURCE_BINDINGS,
                "decoded_before_protocol_freeze": False,
            },
            "arguments": vars(args),
            "arms": list(ARMS),
            "counts": COUNT_BY_ARM,
            "gaussian_row_updates": GAUSSIAN_ROW_UPDATES_BY_ARM,
            "structure_tensor": asdict(StructureTensorConfig()),
            "fit_n640": asdict(h23._fit_config(args, "normalized_plain")),
            "fit_n960": asdict(h26._cold_fit_config(args, 960)),
            "fit_n1024": asdict(h26._cold_fit_config(args, 1024)),
            "projection": asdict(h24._projection_config(args)),
            "safety": asdict(h24._safety_config()),
            "pursuit": asdict(_tail_config(args)),
            "claim_limits": [
                "max-side-160 count/work exchange only",
                "dirty-source producer confirmation",
                "no equal-byte, codec, production, default, or novelty claim",
            ],
        },
    )
    with (args.out / "git.diff").open("wb") as handle:
        subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=handle,
        )
    (args.out / "NATURAL_STARTED").write_text(
        "HIER-028 untouched source pixels decoded; no in-place tuning or replay.\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    metrics_path = args.out / "metrics.json"
    attempts_path = args.out / "attempts.json"
    if args.resume and metrics_path.is_file():
        rows = json.loads(metrics_path.read_text(encoding="utf-8")).get("rows", [])
    if args.resume and attempts_path.is_file():
        attempts = json.loads(attempts_path.read_text(encoding="utf-8")).get(
            "attempts", []
        )
    row_keys = {(row["image"], row["seed"], row["arm"]) for row in rows}
    tensor_config = StructureTensorConfig()
    for image_path in sources:
        target, mask, raster = h22.report_utils._load_evaluation_raster(
            image_path, None, max_side=args.max_side, mask_threshold=0.5
        )
        if mask is not None:
            raise RuntimeError("HIER-028 requires an unmasked full-frame source")
        for seed in args.seeds:
            expected_keys = {(image_path.stem, seed, arm) for arm in ARMS}
            if expected_keys <= row_keys:
                continue
            methods = {}
            initial_hashes: dict[str, str] = {}
            shared_audit = None
            fit_error = None
            try:
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                init_started = time.perf_counter()
                initial640 = h22.build_field(
                    target,
                    h22._init_config(640, seed),
                    tensor_config,
                    device=args.device,
                )
                init640_seconds = time.perf_counter() - init_started
                normalized = h23._run_method(
                    initial640, target, "normalized_plain", args, torch
                )
                normalized.update(
                    {
                        "attempted_steps": 500,
                        "gaussian_row_updates": 640 * 500,
                        "diagnostic_renderer_calls": 0,
                        "init_seconds": init640_seconds,
                        "initial_field_digest": h24._field_digest(initial640),
                        "preprojection_endpoint_digest": h24._field_digest(
                            normalized["field"]
                        ),
                        "shared_endpoint_group": "normalized_n640",
                    }
                )
                cold960 = h27._run_cold(target, seed, 960, args, torch)
                cold1024 = h27._run_cold(target, seed, 1024, args, torch)
                cold960["shared_endpoint_group"] = "cold_additive_n960"
                cold1024["shared_endpoint_group"] = "cold_additive_n1024"
                projected960 = h24._project_method(cold960, target, args)
                projected1024 = h24._project_method(cold1024, target, args)
                pursuit = _pursuit_method(projected960, target, args, torch)
                shared_audit = _save_shared_audit(
                    args.out,
                    image_path.stem,
                    seed,
                    initial640,
                    normalized,
                    cold960,
                    projected960,
                    pursuit,
                    cold1024,
                    projected1024,
                )
                initial_hashes = {
                    "normalized_plain_n640": shared_audit["fields"]["n640_initial"][
                        "sha256"
                    ],
                    "cold_additive_projected_n960": shared_audit["fields"][
                        "n960_initial"
                    ]["sha256"],
                    "residual_pursuit_additive_n1024": shared_audit["fields"][
                        "n960_initial"
                    ]["sha256"],
                    "cold_additive_projected_n1024": shared_audit["fields"][
                        "cold_n1024_initial"
                    ]["sha256"],
                }
                methods = {
                    "normalized_plain_n640": h24._base_method(normalized),
                    "cold_additive_projected_n960": projected960,
                    "residual_pursuit_additive_n1024": pursuit,
                    "cold_additive_projected_n1024": projected1024,
                }
            except Exception as exc:
                fit_error = exc
            for arm in ARMS:
                stable_key = (image_path.stem, seed, arm)
                if stable_key in row_keys:
                    continue
                cell_started = time.perf_counter()
                try:
                    if fit_error is not None:
                        raise RuntimeError(f"paired execution failed: {fit_error}")
                    if shared_audit is None:
                        raise RuntimeError("shared audit receipt was not created")
                    row = _write_cell(
                        output_root=args.out,
                        image_path=image_path,
                        target=target,
                        raster=raster,
                        seed=seed,
                        arm=arm,
                        method=methods[arm],
                        initial_file_sha256=initial_hashes[arm],
                        shared_audit=shared_audit,
                        args=args,
                        torch=torch,
                    )
                    rows.append(row)
                    row_keys.add(stable_key)
                    attempts.append(
                        {
                            "image": image_path.stem,
                            "seed": seed,
                            "arm": arm,
                            "status": "ok",
                            "elapsed_seconds": time.perf_counter() - cell_started,
                        }
                    )
                except Exception as exc:
                    attempts.append(
                        {
                            "image": image_path.stem,
                            "seed": seed,
                            "arm": arm,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:1000],
                            "elapsed_seconds": time.perf_counter() - cell_started,
                        }
                    )
                finally:
                    _write_tables(args.out, rows)
                    _write_json(
                        attempts_path,
                        {
                            "schema": REPORT_SCHEMA,
                            "status": "diagnostic",
                            "attempts": attempts,
                        },
                    )
                    torch.cuda.empty_cache()

    decision = _decision(rows)
    _write_json(args.out / "decision.json", decision)
    _write_report(args.out, rows, decision)
    (args.out / "COMPLETED").write_text(
        "HIER-028 untouched producer confirmation complete; do not overwrite.\n",
        encoding="utf-8",
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
