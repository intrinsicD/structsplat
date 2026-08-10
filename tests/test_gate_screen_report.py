"""Focused tests for the FIT-028/FIT-029/BENCH-018 cross-arm comparison driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "fit028_bench018_gate_screen_report.py"
)
_spec = importlib.util.spec_from_file_location("_gate_screen_report", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def _cell(variant, seed, psnr, *, status="ok", fit_seconds=10.0, holes=(0.0, 0.1), **extra):
    row = {
        "variant": variant,
        "seed": seed,
        "status": status,
        "psnr": psnr,
        "ms_ssim": 0.9,
        "lpips": None,
        "mae": 0.001,
        "n_gaussians": 11_000,
        "attempted_steps": 1_000,
        "accepted_steps": 100,
        "fit_seconds": fit_seconds,
        "curves": [
            {
                "attempted_steps": 0,
                "elapsed_seconds": 0.0,
                "psnr": 10.0,
                "ms_ssim": 0.5,
                "interior_hole_fraction": 0.2,
                "boundary_hole_fraction": 0.3,
                "n_gaussians": 5_000,
            },
            {
                "attempted_steps": 1_000,
                "elapsed_seconds": fit_seconds,
                "psnr": psnr,
                "ms_ssim": 0.9,
                "interior_hole_fraction": holes[0],
                "boundary_hole_fraction": holes[1],
                "n_gaussians": 11_000,
            },
        ],
        "gate_telemetry": {
            "step_acceptance": 0.1,
            "attempted_steps": 1_000,
            "accepted_steps": 100,
            "rejection_reasons": {"interior_holes_regressed": 3},
            "phases": {
                "detail_growth": {
                    "attempted_steps": 800,
                    "accepted_steps": 100,
                    "blocks": 4,
                    "accepted_blocks": 1,
                    "step_acceptance": 0.125,
                    "rejection_reasons": {"interior_holes_regressed": 3},
                },
                "safe_polish": {
                    "attempted_steps": 200,
                    "accepted_steps": 0,
                    "blocks": 1,
                    "accepted_blocks": 0,
                    "step_acceptance": 0.0,
                    "rejection_reasons": {"interior_holes_regressed": 1},
                },
            },
        },
    }
    row.update(extra)
    return row


def _bundle(tmp_path: Path, rows) -> Path:
    (tmp_path / "metrics.json").write_text(json.dumps(rows), encoding="utf-8")
    return tmp_path


def test_paired_deltas_use_only_seeds_both_arms_completed(tmp_path):
    bundle = _bundle(
        tmp_path,
        [
            _cell("current", 0, 30.0),
            _cell("current", 1, 32.0),
            _cell("current", 2, 34.0),
            # The arm is missing seed 2, so its delta must average seeds 0 and 1 only.
            _cell("budget5e4", 0, 31.0),
            _cell("budget5e4", 1, 31.0),
        ],
    )

    _page, summary = report.build_report(bundle, "current")

    arm = summary["arms"]["budget5e4"]
    assert arm["n_cells"] == 2
    assert arm["paired_seeds_psnr"] == 2
    # (31-30) + (31-32) averaged = 0.0, not the -1.0 an unpaired mean-of-means would give.
    assert arm["delta_psnr"] == pytest.approx(0.0)
    assert summary["arms"]["current"]["psnr"] == pytest.approx(32.0)


def test_terminal_hole_fractions_come_from_the_last_curve_point(tmp_path):
    bundle = _bundle(
        tmp_path,
        [
            _cell("current", 0, 30.0, holes=(0.0, 0.10)),
            _cell("budget2e3", 0, 31.0, holes=(0.05, 0.12)),
        ],
    )

    _page, summary = report.build_report(bundle, "current")

    assert summary["arms"]["current"]["terminal_interior_hole_fraction"] == pytest.approx(0.0)
    # FIT-028's guardrail: a PSNR gain bought with interior coverage must stay visible.
    assert summary["arms"]["budget2e3"]["terminal_interior_hole_fraction"] == pytest.approx(0.05)
    assert summary["arms"]["budget2e3"]["delta_psnr"] == pytest.approx(1.0)


def test_error_cells_are_reported_and_never_silently_dropped(tmp_path):
    bundle = _bundle(
        tmp_path,
        [
            _cell("current", 0, 30.0),
            _cell("block25", 0, 0.0, status="error", error="CUDA out of memory"),
        ],
    )

    page, summary = report.build_report(bundle, "current")

    assert summary["error_cells"] == [
        {"variant": "block25", "seed": 0, "error": "CUDA out of memory"}
    ]
    assert "CUDA out of memory" in page
    assert "block25" not in summary["arms"]


def test_phase_acceptance_exposes_a_phase_that_never_commits(tmp_path):
    bundle = _bundle(tmp_path, [_cell("current", 0, 30.0), _cell("current", 1, 30.0)])

    _page, summary = report.build_report(bundle, "current")

    phases = summary["arms"]["current"]["phase_acceptance"]
    # FIT-029 reads exactly this: the phase attempted work and kept none of it.
    assert phases["safe_polish"]["attempted_steps"] == 400
    assert phases["safe_polish"]["accepted_steps"] == 0
    assert phases["safe_polish"]["step_acceptance"] == 0.0
    assert phases["detail_growth"]["step_acceptance"] == pytest.approx(0.125)


def test_missing_baseline_arm_is_an_explicit_error(tmp_path):
    bundle = _bundle(tmp_path, [_cell("block50", 0, 30.0)])

    with pytest.raises(ValueError, match="baseline arm 'current' has no completed cell"):
        report.build_report(bundle, "current")


def test_report_renders_wall_clock_and_step_axes(tmp_path):
    bundle = _bundle(
        tmp_path,
        [_cell("current", 0, 30.0, fit_seconds=10.0), _cell("block25", 0, 30.5, fit_seconds=40.0)],
    )

    page, _summary = report.build_report(bundle, "current")

    # BENCH-018 decides on quality-per-second, so both axes must be present.
    assert "PSNR over wall-clock seconds" in page
    assert "PSNR over attempted steps" in page
    assert "Per-phase step acceptance" in page


def test_block_level_reasons_are_counted_per_block_not_per_occurrence(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "schedule_history": [
                    {"phase": "detail", "reasons": []},
                    # One rejected block citing two reasons must count as ONE block for each.
                    {
                        "phase": "detail",
                        "reasons": ["interior_holes_regressed", "cvar99_mse_regressed"],
                    },
                    {"phase": "detail", "reasons": ["cvar99_mse_regressed"]},
                    # A duplicated reason inside one block still counts that block once.
                    {
                        "phase": "polish",
                        "reasons": ["interior_holes_regressed", "interior_holes_regressed"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    bundle = _bundle(tmp_path, [_cell("current", 0, 30.0, history_json=str(history))])

    _page, summary = report.build_report(bundle, "current")

    arm = summary["arms"]["current"]
    assert arm["rejected_blocks"] == 3
    assert arm["blocks_citing"]["interior_holes_regressed"] == 2
    assert arm["blocks_citing"]["cvar99_mse_regressed"] == 2
    # The occurrence view stays separate and is not silently replaced.
    assert summary["rejection_reasons"]["current"]["interior_holes_regressed"] == 3


def test_missing_history_file_degrades_without_failing(tmp_path):
    bundle = _bundle(
        tmp_path,
        [_cell("current", 0, 30.0, history_json=str(tmp_path / "absent.json"))],
    )

    _page, summary = report.build_report(bundle, "current")

    assert summary["arms"]["current"]["rejected_blocks"] == 0


def test_sole_reason_blocks_bound_what_relaxing_one_term_can_revive(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "schedule_history": [
                    # Co-vetoed: relaxing the hole budget alone cannot revive this block.
                    {"reasons": ["interior_holes_regressed", "cvar99_mse_regressed"]},
                    {"reasons": ["interior_holes_regressed", "cvar99_mse_regressed"]},
                    # Vetoed by the hole term alone: this one is revivable.
                    {"reasons": ["interior_holes_regressed"]},
                    {"reasons": ["cvar99_mse_regressed"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    bundle = _bundle(tmp_path, [_cell("current", 0, 30.0, history_json=str(history))])

    page, summary = report.build_report(bundle, "current")

    arm = summary["arms"]["current"]
    assert arm["rejected_blocks"] == 4
    assert arm["blocks_citing"]["interior_holes_regressed"] == 3
    # Only one of those three is recoverable by moving the budget.
    assert arm["blocks_citing_only"]["interior_holes_regressed"] == 1
    assert arm["blocks_citing_only"]["cvar99_mse_regressed"] == 1
    assert "alone" in page
