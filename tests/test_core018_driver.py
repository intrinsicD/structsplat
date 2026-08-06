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
        / "core018_ray_posterior_downstream.py"
    )
    spec = importlib.util.spec_from_file_location("_core018_ray_posterior_driver", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _passing_rows(driver):
    common = {
        "status": "ok",
        "input_bytes": 100,
        "original_source_bytes": 1_000,
        "final_model_bytes": 200,
        "original_over_packets": 10.0,
        "original_over_packets_plus_model": 1_000 / 300,
        "initial_n_gaussians": driver.INITIAL_GAUSSIANS,
        "final_n_gaussians": 20_000,
        "initial_reporting_psnr": 15.0,
        "reporting_ssim": 0.90,
        "reporting_ms_ssim": 0.95,
        "reporting_lpips": 0.10,
        "reporting_gradient_mae": 0.02,
        "reporting_p99_abs": 0.10,
        "lift_seconds": 2.0,
        "feature_seconds": 0.0,
        "training_native_seconds": 10.0,
        "peak_vram_gb": 1.0,
    }
    return [
        {**common, "arm": "interior", "reporting_psnr": 20.0, "pretraining_seconds": 5.0},
        {
            **common,
            "arm": "posterior_no_reciprocal",
            "reporting_psnr": 20.2,
            "pretraining_seconds": 8.0,
        },
        {
            **common,
            "arm": "posterior_reciprocal",
            "reporting_psnr": 20.2,
            "pretraining_seconds": 8.0,
        },
    ]


def _records(rows):
    targets = {
        "interior": [(0, 15.0), (500, 20.0), (1_500, 20.0)],
        "posterior_no_reciprocal": [(0, 16.0), (800, 20.2), (1_500, 20.2)],
        "posterior_reciprocal": [(0, 17.0), (500, 20.2), (1_500, 20.2)],
    }
    return [
        {
            "arm": row["arm"],
            "status": "ok",
            "curve_rows": [
                {
                    "step": step,
                    "optimization_elapsed_seconds": float(step),
                    "metrics": {"reporting": {"aggregate": {"psnr": psnr}}},
                }
                for step, psnr in targets[row["arm"]]
            ],
        }
        for row in rows
    ]


def test_core018_protocol_is_frozen_and_reporting_views_are_disjoint() -> None:
    driver = _load_driver()

    assert driver.REPORT_CAMERA_IDS == ("C0004", "C0025", "C1004", "C1005")
    assert set(driver.TRAIN_CAMERA_IDS).isdisjoint(driver.REPORT_CAMERA_IDS)
    assert set(driver.TRAIN_CAMERA_IDS) | set(driver.REPORT_CAMERA_IDS) == set(
        driver.ALL_CAMERA_IDS
    )
    assert driver.ARMS == (
        "interior",
        "posterior_no_reciprocal",
        "posterior_reciprocal",
    )
    assert driver.INITIAL_GAUSSIANS == 10_000
    assert driver.MAX_GAUSSIANS == 30_000
    assert driver.ITERATIONS == 1_500
    assert driver.FIXED_PREFIX_STEPS == 500
    assert driver.EVAL_EVERY == 100

    train = driver._train_config()
    assert train.density.start_iter == 600
    assert train.density.stop_iter == 1_400
    assert train.density.max_gaussians == driver.MAX_GAUSSIANS
    assert train.random_background is False
    assert train.use_masks is False

    no_reciprocal = driver._posterior_config(False)
    reciprocal = driver._posterior_config(True)
    assert no_reciprocal.apply_reciprocal is False
    assert reciprocal.apply_reciprocal is True
    assert no_reciprocal.target_views == reciprocal.target_views == 4
    assert no_reciprocal.best_view_count == reciprocal.best_view_count == 2


def test_core018_scalar_pass_remains_fail_closed_for_visual_review() -> None:
    driver = _load_driver()
    rows = _passing_rows(driver)

    decision = driver._decision(_records(rows), rows)

    assert decision["scalar_pass"] is True
    assert all(decision["gates"].values())
    assert decision["manual_visual_review_required"] is True
    assert decision["advance"] is False
    assert decision["strongest_control"] == "posterior_no_reciprocal"


def test_core018_decision_rejects_one_protected_regression() -> None:
    driver = _load_driver()
    rows = _passing_rows(driver)
    candidate = next(row for row in rows if row["arm"] == "posterior_reciprocal")
    candidate["reporting_gradient_mae"] = 0.0201

    decision = driver._decision(_records(rows), rows)

    assert decision["scalar_pass"] is False
    assert decision["gates"]["terminal_reporting_gradient_mae_no_worse"] is False
    assert decision["advance"] is False
