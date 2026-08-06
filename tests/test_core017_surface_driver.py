from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_driver():
    pytest.importorskip("rtgs")
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "core016_multiview_downstream.py"
    )
    spec = importlib.util.spec_from_file_location("_core017_surface_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_rows(driver):
    baseline = {
        "heldout_psnr_fg": 20.0,
        "heldout_ms_ssim": 0.80,
        "heldout_lpips": 0.20,
        "heldout_gradient_mae_fg": 0.020,
        "heldout_alpha_iou": 0.90,
        "heldout_alpha_outside": 0.030,
        "lift_seconds": 2.0,
        "training_native_seconds": 5.0,
        "final_n_gaussians": 5_000,
        "input_bytes": 123_456,
    }
    rows = []
    for arm in driver.PROFILES["surface2x2"].arms:
        row = {**baseline, "arm": arm, "status": "ok"}
        if arm == "dual_shell_cover":
            row.update(
                {
                    "heldout_psnr_fg": 20.5,
                    "heldout_gradient_mae_fg": 0.019,
                    "heldout_alpha_iou": 0.89,
                    "heldout_alpha_outside": 0.04,
                }
            )
        rows.append(row)
    return rows


def test_surface_factorial_profile_is_frozen_and_fail_closed_for_visual_review() -> None:
    driver = _load_driver()
    profile = driver.PROFILES["surface2x2"]

    assert profile.arms == tuple(driver.SURFACE_ARM_RECIPES)
    assert profile.n_init_3d == profile.density_max_gaussians == 5_000
    assert profile.iterations == 1_000
    assert profile.eval_every == 100
    assert profile.densify is False

    decision = driver._decision(profile, _passing_rows(driver))

    assert decision["scalar_pass"] is True
    assert decision["manual_visual_review_required"] is True
    assert decision["advance"] is False
    assert all(decision["gates"].values())


def test_surface_factorial_decision_rejects_a_single_protected_regression() -> None:
    driver = _load_driver()
    profile = driver.PROFILES["surface2x2"]
    rows = _passing_rows(driver)
    combined = next(row for row in rows if row["arm"] == "dual_shell_cover")
    combined["heldout_alpha_outside"] = 0.0401

    decision = driver._decision(profile, rows)

    assert decision["scalar_pass"] is False
    assert decision["gates"]["combined_alpha_outside_within_0_01"] is False
    assert decision["advance"] is False
