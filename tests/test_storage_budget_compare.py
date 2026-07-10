from __future__ import annotations

import json
from pathlib import Path

from benchmarks import fair_density_control_compare as F
from benchmarks import storage_budget as SB
from benchmarks import storage_budget_compare as C


def test_frozen_storage_suite_covers_every_registered_method() -> None:
    assert C.STORAGE_BENCHMARK_METHODS == tuple(F.DEFAULT_METHODS)
    assert len(C.STORAGE_BENCHMARK_METHODS) == 41


def test_fair_args_pin_exact_capacity_and_convergence_protocol(tmp_path: Path) -> None:
    args = C.build_parser().parse_args(["--outdir", str(tmp_path)])
    resolved = C.fair_args(args)

    assert resolved.storage_budget_bytes == 172_032
    assert resolved.budgets == [5_376]
    assert resolved.methods == list(C.STORAGE_BENCHMARK_METHODS)
    assert resolved.iters == 10_000
    assert resolved.log_every == 100
    assert resolved.early_stop_patience == 6
    assert resolved.early_stop_min_delta == 0.005
    assert resolved.early_stop_min_iters == 6_500
    assert resolved.max_side == 160
    assert resolved.lpips is True
    assert resolved.compute_codec_metrics is True
    assert resolved.resume is True


def test_report_manifest_counts_partial_run_and_exposes_byte_links(tmp_path: Path) -> None:
    for name in (
        "index.html",
        "image_storage.csv",
        "image_metrics.csv",
        "storage_method_summary.csv",
        "convergence_curves.csv",
        "config.json",
    ):
        (tmp_path / name).write_text("test\n", encoding="utf-8")
    rows = [
        {
            "status": "ok",
            "storage_exact_budget": True,
            "run_protocol_sha256": "abc",
        },
        {
            "status": "ok",
            "storage_exact_budget": False,
            "run_protocol_sha256": "abc",
        },
        {"status": "error", "run_protocol_sha256": "abc"},
    ]
    path = C.write_report_manifest(
        tmp_path,
        rows,
        images=[Path("a.jpg"), Path("b.jpg")],
        methods=["left", "right"],
        seeds=[0, 1],
        max_side=160,
        iters=10_000,
        log_every=100,
        early_stop_patience=6,
        early_stop_min_delta=0.005,
        early_stop_min_iters=6_500,
        lpips_enabled=True,
        codec_metrics_enabled=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["completeness"] == {
        "expected_cells": 8,
        "completed_cells": 2,
        "error_cells": 1,
    }
    assert payload["coverage"]["exact_capacity_cells"] == 1
    assert payload["coverage"]["plateau_converged_cells"] == 0
    assert payload["coverage"]["max_horizon_cells"] == 0
    assert payload["storage"] == {
        "budget_bytes": SB.FIXED_STORAGE_BUDGET_BYTES,
        "bytes_per_gaussian": SB.CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN,
        "gaussian_cap": SB.FIXED_STORAGE_GAUSSIAN_CAP,
        "accounting": SB.STORAGE_ACCOUNTING_NAME,
    }
    assert payload["protocol_sha256"] == "abc"
    assert {link["path"] for link in payload["links"]} >= {
        "index.html",
        "image_storage.csv",
        "image_metrics.csv",
        "convergence_curves.csv",
    }
