#!/usr/bin/env python3
"""Run HIER-027's frozen cold pure-additive capacity confirmation."""
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
from structsplat.config import StructureTensorConfig  # noqa: E402


REPORT_SCHEMA = "structsplat.hier027_cold_additive_capacity.confirmation.v1"
ARMS = (
    "normalized_plain_n640",
    "additive_plain_n640",
    "additive_projected_n640",
    "cold_additive_projected_n1024",
    "cold_additive_plain_n1088",
    "cold_additive_projected_n1088",
    "cold_additive_projected_n1152",
)
PROJECTED_ARMS = frozenset(
    (
        "additive_projected_n640",
        "cold_additive_projected_n1024",
        "cold_additive_projected_n1088",
        "cold_additive_projected_n1152",
    )
)
PURE_ADDITIVE_ARMS = frozenset(set(ARMS) - {"normalized_plain_n640"})
COUNT_BY_ARM = {
    "normalized_plain_n640": 640,
    "additive_plain_n640": 640,
    "additive_projected_n640": 640,
    "cold_additive_projected_n1024": 1024,
    "cold_additive_plain_n1088": 1088,
    "cold_additive_projected_n1088": 1088,
    "cold_additive_projected_n1152": 1152,
}
SELECTION_SALT = "HIER-027-confirm-v1:"
SELECTION_ORDER = (
    "0859.png",
    "0833.png",
    "0874.png",
    "0880.png",
    "0802.png",
    "0808.png",
    "0815.png",
    "0889.png",
)
SELECTION_BINDINGS = {
    "0859.png": "03488568c8031c428e16d4365ce5c3241276d460b4eb944204aec6dbe1cdfe42",
    "0833.png": "03f45d5a4ad1a7e29466b4bf012b4b4ba1ae96cbf1bcecc07cff36ac3c98e8ce",
    "0874.png": "08dafd50533c303e3375e55fa7cb1b04f36067caa8694ffd175506b10c5cc5a3",
    "0880.png": "0a05d43823a705d32c5b2daf099b7901d9ad0a8d1c62d32a976a52c296a02f5b",
    "0802.png": "0a31d512d0f0b526a503c3a51eb2f0c274984e156e6bf4eac75479b564cefd99",
    "0808.png": "0e0d3b42d9d4ee8fbe42f756119af883e9d47ec6ec58e6825c65ae99c2530824",
    "0815.png": "1225e9713eb595e0f3482a4fc07b26459f50c929ea94d48a5ce6648bd7bdebf8",
    "0889.png": "151d9fb642f2afc1b96797072e537accdbfe2798591498e1ff09a59952edfe9d",
}
SOURCE_BINDINGS = {
    "0859.png": "3ada872de7c5def1d408920385db278b1ff3a5a0cfcab83105a789ff540a1827",
    "0833.png": "2e9668b3a318284ec90c9bbdd940317ecd2f7b95314e68c48c94d2380fad679a",
    "0874.png": "11cb511247d70d84adad5557a720254e5f73e3786dbcd399c6053a1982ce1784",
    "0880.png": "db5773c6e460824c5132c23917492fda7acd370c87e9ae6293a0103fee2b642d",
    "0802.png": "4ad6f3ca8bf740192042978121f05ec493ddbe5a3da5584eaf0d9699c25ee431",
    "0808.png": "956528ab3e0fadad1ed8ce93f93a30bf9f58c36ffa9dd775e2ad362ffdcf5ace",
    "0815.png": "c8f278e51f2bc9be7a696935b7e386eb4adafde24572d8ecdd4edf8adf4b4108",
    "0889.png": "a8f73c42065e3193c4deb883dcb3bc432a3f838e9be5bacea708ee39eb2c6e04",
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
                f"frozen HIER-027 protocol requires {name}={expected!r}, "
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
            "HIER-027 extraction root must contain exactly the eight bound members: "
            f"got {actual_names!r}"
        )
    paths = [root / name for name in SELECTION_ORDER]
    hashes = {path.name: h22.report_utils._sha256(path) for path in paths}
    if hashes != SOURCE_BINDINGS:
        raise SystemExit(f"HIER-027 source hash binding differs: {hashes!r}")
    return [path.resolve() for path in paths]


def _snapshot_sources(output_root: Path) -> list[dict[str, object]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts" / "experiments" / "hier022_additive_continuation.py",
        ROOT / "scripts" / "experiments" / "hier023_unit_gauge_continuation.py",
        ROOT / "scripts" / "experiments" / "hier024_gauge_geometry_projection.py",
        ROOT / "scripts" / "experiments" / "hier026_progressive_additive_capacity.py",
        ROOT / "src" / "structsplat" / "endpoint_appearance_projection.py",
        ROOT / "tests" / "test_cold_additive_capacity_confirmation.py",
        ROOT / "tests" / "test_endpoint_appearance_projection.py",
        ROOT / "tasks" / "HIER-027-cold-additive-capacity-confirmation.md",
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


def _compat_cold(method: dict[str, object]) -> dict[str, object]:
    result = dict(method)
    result.update(
        {
            "hold_psnr_db": None,
            "optimizer_reset_count": 0,
            "optimizer_reset_step": None,
            "hold_optimizer_reset_count": 0,
        }
    )
    return result


def _run_cold(target, seed: int, count: int, args, torch) -> dict[str, object]:
    return _compat_cold(h26._run_cold_additive(target, seed, count, args, torch))


def _save_shared_audit(
    output_root: Path,
    image_stem: str,
    seed: int,
    normalized: dict[str, object],
    methods_by_count: dict[int, dict[str, object]],
) -> dict[str, object]:
    directory = output_root / "shared" / f"{image_stem}__s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    fields = {"normalized_endpoint": normalized["field"]}
    for count, method in methods_by_count.items():
        fields[f"n{count}_initial"] = method["audit_initial_field"]
        fields[f"n{count}_training"] = method["audit_training_field"]
        fields[f"n{count}_endpoint"] = method["field"]
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
        "counts": sorted(methods_by_count),
        "steps": 500,
        "gaussian_row_updates": {
            str(count): count * 500 for count in methods_by_count
        },
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
    _write_json(artifact_dir / "projection_history.json", projection)
    _write_json(
        artifact_dir / "geometry_history.json",
        {
            "initial_field_digest": method["initial_field_digest"],
            "preprojection_endpoint_digest": method["preprojection_endpoint_digest"],
            "shared_endpoint_group": method["shared_endpoint_group"],
            "training_payload_stripped": arm in PURE_ADDITIVE_ARMS,
        },
    )
    fit_config = (
        asdict(h23._fit_config(args, "normalized_plain"))
        if arm == "normalized_plain_n640"
        else asdict(h26._cold_fit_config(args, count))
    )
    _write_json(
        artifact_dir / "config.json",
        {
            "schema": REPORT_SCHEMA,
            "arm": arm,
            "seed": seed,
            "count": count,
            "init": asdict(h22._init_config(count, seed)),
            "fit": fit_config,
            "projection": (
                asdict(h24._projection_config(args)) if arm in PROJECTED_ARMS else None
            ),
            "safety": asdict(h24._safety_config()) if arm in PROJECTED_ARMS else None,
            "shared_audit_receipt": shared_audit["receipt_path"],
        },
    )
    pure = arm in PURE_ADDITIVE_ARMS
    field_keys = row["field_npz_keys"]
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
            "gaussian_row_updates": count * 500,
            "endpoint_internal_parity_max_abs": method["endpoint_parity"],
            "diagnostic_renderer_calls_fit": method["diagnostic_renderer_calls"],
            "projection_applied": method["projection_applied"],
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
            "final_field_digest": method["final_field_digest"],
            "incoming_field_file_sha256": h22.report_utils._sha256(incoming_path),
            "proposal_field_file_sha256": h22.report_utils._sha256(proposal_path),
            "initial_field_digest": method["initial_field_digest"],
            "preprojection_endpoint_digest": method["preprojection_endpoint_digest"],
            "shared_endpoint_group": method["shared_endpoint_group"],
            "shared_audit_dir": shared_audit["dir"],
            "shared_audit_receipt": shared_audit["receipt_path"],
            "shared_audit_receipt_sha256": shared_audit["receipt_sha256"],
        }
    )
    for prefix, values in (
        ("incoming", method["incoming_selection_metrics"]),
        ("proposal", method["proposal_selection_metrics"]),
    ):
        for key in ("raw_mse", "ms_ssim", "lpips", "pixel_max", "patch7_max"):
            row[f"{prefix}_{key}"] = None if values is None else values[key]
    row["pipeline_algorithm_seconds"] = float(row["pipeline_algorithm_seconds"]) + float(
        method["projection_seconds"]
    )
    row["total_seconds"] = float(row["total_seconds"]) + float(
        method["projection_seconds"]
    ) + float(method["projection_metric_seconds"])
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
            and row["final_field_digest"] == row["incoming_field_digest"]
        )
        for arm in PROJECTED_ARMS
        for row in by_arm[arm]
    )
    shared_endpoints = complete
    for plain_arm, projected_arm in (
        ("additive_plain_n640", "additive_projected_n640"),
        ("cold_additive_plain_n1088", "cold_additive_projected_n1088"),
    ):
        grouped: dict[tuple[object, object], set[object]] = {}
        for arm in (plain_arm, projected_arm):
            for row in by_arm[arm]:
                grouped.setdefault((row["image"], row["seed"]), set()).add(
                    row["preprojection_endpoint_digest"]
                )
        shared_endpoints = shared_endpoints and all(
            None not in values and len(values) == 1 for values in grouped.values()
        )
    work_exact = complete and all(
        row["attempted_steps"] == 500
        and row["gaussian_row_updates"] == COUNT_BY_ARM[arm] * 500
        for arm in ARMS
        for row in by_arm[arm]
    )
    quality: dict[str, object] = {}
    if complete:
        normalized = by_arm["normalized_plain_n640"]
        for arm in (
            "additive_plain_n640",
            "additive_projected_n640",
            "cold_additive_projected_n1024",
            "cold_additive_projected_n1088",
            "cold_additive_projected_n1152",
        ):
            quality[arm] = h26._quality_gate(by_arm[arm], normalized)
            quality[arm]["integrity_pass"] = integrity[arm]
            quality[arm]["numeric_quality_capable"] = bool(
                integrity[arm]
                and projection_fail_closed
                and quality[arm]["numeric_pass"]
            )
    primary_n1088 = bool(
        quality.get("cold_additive_projected_n1088", {}).get(
            "numeric_quality_capable", False
        )
    )
    fallback_n1152 = bool(
        quality.get("cold_additive_projected_n1152", {}).get(
            "numeric_quality_capable", False
        )
        and quality["cold_additive_projected_n1152"]["mean_psnr_delta_db"] >= 0.50
        and quality["cold_additive_projected_n1152"]["minimum_psnr_delta_db"] >= 0.0
    )
    gates = {
        "all_cells_present": complete,
        "all_arm_integrity": complete and all(integrity.values()),
        "projection_transactions_fail_closed": projection_fail_closed,
        "shared_preprojection_endpoints_exact": shared_endpoints,
        "work_accounting_exact": work_exact,
    }
    selected_arm = None
    selector_reason = "neither selectable pure-additive capacity rung passes"
    if primary_n1088 and all(gates.values()):
        selected_arm = "cold_additive_projected_n1088"
        selector_reason = "primary N=1088 passes every frozen numeric clause"
    elif fallback_n1152 and all(gates.values()):
        selected_arm = "cold_additive_projected_n1152"
        selector_reason = "primary fails; predeclared robust N=1152 fallback passes"
    numeric_pass = selected_arm is not None
    return {
        "schema": REPORT_SCHEMA,
        "status": "diagnostic",
        "phase": "untouched_confirmation",
        "aggregates": aggregates,
        "integrity": integrity,
        "quality": quality,
        "gates": gates,
        "boundary_n1024_numeric_only": bool(
            quality.get("cold_additive_projected_n1024", {}).get(
                "numeric_quality_capable", False
            )
        ),
        "primary_n1088_numeric": primary_n1088,
        "fallback_n1152_numeric": fallback_n1152,
        "same_count_additive_better_numeric": bool(
            quality.get("additive_plain_n640", {}).get(
                "numeric_quality_capable", False
            )
        ),
        "normalization_not_required_for_fidelity_numeric": numeric_pass,
        "numeric_selected_arm": selected_arm,
        "selector_reason": selector_reason,
        "numeric_pass": numeric_pass,
        "visual_review": "pending_native_audit",
        "overall_pass": False,
        "formal_claim_ready": False,
        "interpretation": (
            "A selectable cold pure-additive rung passes numerically; audit native visuals."
            if numeric_pass
            else "No selectable rung passes; retain without tuning."
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
            f"<td>{'yes' if row['projection_selected'] else 'no'}</td>"
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
<title>HIER-027 cold additive capacity</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1900px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.35rem}}img{{max-width:300px;max-height:240px;margin:.25rem}}
section{{border-top:1px solid #aaa;margin-top:1rem}}pre,code{{white-space:pre-wrap}}
</style></head><body><h1>HIER-027 cold pure-additive capacity threshold</h1>
<p><strong>Untouched-data producer confirmation.</strong> Protocol, filenames, archive, and member
hashes preceded selected-image decode; dirty source and producer review keep this provisional.</p>
<p><code>{escape(_command())}</code></p>
<p><a href="config.json">config</a> · <a href="decision.json">decision</a> ·
<a href="metrics.json">JSON</a> · <a href="metrics.jsonl">JSONL</a> ·
<a href="metrics.csv">CSV</a> · <a href="attempts.json">attempts</a> ·
<a href="manifest.json">manifest</a></p>
<h2>Decision</h2><pre>{escape(json.dumps(decision, indent=2, sort_keys=True))}</pre>
<h2>Cells</h2><table><tr><th>image</th><th>seed</th><th>arm</th><th>N</th><th>PSNR</th>
<th>MS-SSIM</th><th>LPIPS</th><th>pixel max</th><th>7x7 max</th><th>projection</th>
<th>artifacts</th></tr>{''.join(table_rows)}</table>
<h2>Native visual audit</h2>{''.join(cards)}</body></html>"""
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
        raise SystemExit(f"completed HIER-027 bundle is immutable: {args.out}")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; use --resume: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("frozen HIER-027 protocol requires CUDA")
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
                "excluded_hier026": ["0847.png", "0860.png", "0895.png", "0898.png"],
                "selection_bindings": SELECTION_BINDINGS,
                "source_bindings": SOURCE_BINDINGS,
                "decoded_before_protocol_freeze": False,
            },
            "arguments": vars(args),
            "arms": list(ARMS),
            "counts": COUNT_BY_ARM,
            "structure_tensor": asdict(StructureTensorConfig()),
            "fit_n640": asdict(h26._cold_fit_config(args, 640)),
            "fit_n1024": asdict(h26._cold_fit_config(args, 1024)),
            "fit_n1088": asdict(h26._cold_fit_config(args, 1088)),
            "fit_n1152": asdict(h26._cold_fit_config(args, 1152)),
            "projection": asdict(h24._projection_config(args)),
            "safety": asdict(h24._safety_config()),
            "shared_fit_reuse": (
                "plain/projected N=640 share one fit; plain/projected N=1088 share one fit"
            ),
            "claim_limits": [
                "max-side-160 count/work exchange only",
                "N=1024 is non-selectable because HIER-026 retains a counterexample",
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
        "HIER-027 untouched source pixels decoded; no in-place tuning or replay.\n",
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
    for image_path in sources:
        target, mask, raster = h22.report_utils._load_evaluation_raster(
            image_path, None, max_side=args.max_side, mask_threshold=0.5
        )
        if mask is not None:
            raise RuntimeError("HIER-027 requires an unmasked full-frame source")
        for seed in args.seeds:
            expected_keys = {(image_path.stem, seed, arm) for arm in ARMS}
            if expected_keys <= row_keys:
                continue
            methods = {}
            initial_hashes: dict[int, str] = {}
            shared_audit = None
            fit_error = None
            try:
                cold = {
                    count: _run_cold(target, seed, count, args, torch)
                    for count in (640, 1024, 1088, 1152)
                }
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                normalized = h23._run_method(
                    cold[640]["audit_initial_field"],
                    target,
                    "normalized_plain",
                    args,
                    torch,
                )
                normalized.update(
                    {
                        "attempted_steps": 500,
                        "gaussian_row_updates": 640 * 500,
                        "diagnostic_renderer_calls": 0,
                        "init_seconds": 0.0,
                        "initial_field_digest": cold[640]["initial_field_digest"],
                        "preprojection_endpoint_digest": h24._field_digest(
                            normalized["field"]
                        ),
                        "shared_endpoint_group": "normalized_n640",
                    }
                )
                for count, method in cold.items():
                    method["shared_endpoint_group"] = f"additive_n{count}"
                shared_audit = _save_shared_audit(
                    args.out, image_path.stem, seed, normalized, cold
                )
                for count in cold:
                    initial_hashes[count] = shared_audit["fields"][
                        f"n{count}_initial"
                    ]["sha256"]
                methods = {
                    "normalized_plain_n640": h24._base_method(normalized),
                    "additive_plain_n640": h24._base_method(cold[640]),
                    "additive_projected_n640": h24._project_method(
                        cold[640], target, args
                    ),
                    "cold_additive_projected_n1024": h24._project_method(
                        cold[1024], target, args
                    ),
                    "cold_additive_plain_n1088": h24._base_method(cold[1088]),
                    "cold_additive_projected_n1088": h24._project_method(
                        cold[1088], target, args
                    ),
                    "cold_additive_projected_n1152": h24._project_method(
                        cold[1152], target, args
                    ),
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
                        initial_file_sha256=initial_hashes[COUNT_BY_ARM[arm]],
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
        "HIER-027 untouched producer confirmation complete; do not overwrite.\n",
        encoding="utf-8",
    )
    _write_manifest(args.out)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
