"""Procedural protocol and pipeline wiring tests; no formal COCO outcomes consumed."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json

import numpy as np
import pytest

from benchmarks import port007_controls as controls


def synthetic_rows():
    rows = []
    for cell_id in controls.expected_cells():
        kind, coco, seed, *rest = cell_id.split("_")
        method = "_".join(rest[1:] if kind == "same" else rest)
        candidate = method not in ("legacy_a", "legacy_b")
        rows.append({"cell_id": cell_id, "kind": kind, "status": "ok", "image": int(coco[4:]),
            "seed": int(seed[1:]), "state": rest[0] if kind == "same" else None,
            "method": method, "call_seconds": [0.7 if candidate else 1.0] * 10,
            "total_seconds": 7.0 if candidate else 10.0, "call_time_cv": 0.0,
            "parity_pass": True, "timing_eligible": True, "trajectory_sha256": "same",
            "n_gaussians": 2000 if kind == "same" else 1000,
            "iterations_run": (0 if rest[0] == "initial" else 750) if kind == "same" else 20,
            "selected_iteration": (0 if rest[0] == "initial" else 750) if kind == "same" else 20,
            "observer_seconds": 0.1,
            "psnr": 30.0, "ms_ssim": 0.9, "lpips": 0.1,
            "final_reference_rgb_max_error": 0.0,
            "quality": {"n_gaussians": 1000, "foreground_mse": 0.001,
                        "boundary_mse": 0.001, "cvar99_mse": 0.01, "p99_mse": 0.009,
                        "interior_hole_fraction": 0.0, "boundary_hole_fraction": 0.0,
                        "outside_max_abs": 0.0, "outside_coverage_max": 0.0, "finite": True}})
    return rows


def test_matrix_and_counterbalancing():
    cells = controls.expected_cells()
    assert len(cells) == len(set(cells)) == 110
    assert sum(cell.startswith("same_") for cell in cells) == 80
    orders = controls.counterbalanced_orders()
    assert len(orders) == 10
    for position in range(5):
        for arm in controls.ARMS:
            assert sum(order[position] == arm for order in orders) == 2
    for seed in range(3):
        assert set(controls.pipeline_order(1, seed)) == set(controls.ARMS)


def test_ellipse_is_symmetric_nonempty_and_not_fullframe():
    mask = controls.ellipse_mask(31, 48)
    assert mask.dtype == bool and mask.any() and not mask.all()
    assert np.array_equal(mask, mask[::-1])
    assert np.array_equal(mask, mask[:, ::-1])


def test_discrete_signature_preserves_decisions_and_ignores_float_noise():
    event = {"selected": {"n_gaussians": 9, "mse": 0.1}, "accepted": True,
             "reasons": [], "metadata": {"selected_iteration": 5}, "elapsed_seconds": 1.0}
    changed_time = copy.deepcopy(event)
    changed_time["selected"]["mse"] = 0.100001
    changed_time["elapsed_seconds"] = 9.0
    assert controls.signature(controls.discrete_projection(event)) == controls.signature(
        controls.discrete_projection(changed_time))
    changed_time["metadata"]["selected_iteration"] = 4
    assert controls.signature(controls.discrete_projection(event)) != controls.signature(
        controls.discrete_projection(changed_time))


def test_summary_image_aggregation_and_complete_matrix_gate():
    rows = synthetic_rows()
    output = controls.summarize(rows)
    assert output["complete"]
    assert all(r["component_speed_pass"] and r["pipeline_speed_pass"]
               for r in output["candidates"].values())
    assert output["candidates"]["both"]["same_state_images"][0]["speed_ratio"] == pytest.approx(1 / 0.7)
    assert not controls.summarize(rows[:-1])["candidates"]["both"]["pipeline_speed_pass"]


@pytest.mark.parametrize("mutation,expected", [
    ("aa_parity", "same_state_correctness_pass"),
    ("foreign_gpu", "same_state_timing_eligible"),
    ("timing_noise", "same_state_timing_eligible"),
    ("trajectory", "pipeline_trajectory_pass"),
    ("aa_runtime", "pipeline_timing_eligible"),
    ("quality", "pipeline_quality_pass"),
])
def test_summary_killing_gates_are_not_rescued(mutation, expected):
    rows = synthetic_rows()
    if mutation in ("aa_parity", "foreign_gpu", "timing_noise"):
        selected = next(r for r in rows if r["kind"] == "same" and r["method"] == "legacy_b")
        selected[{"aa_parity": "parity_pass", "foreign_gpu": "timing_eligible", "timing_noise": "call_time_cv"}[mutation]] = 0.4 if mutation == "timing_noise" else False
    else:
        selected = next(r for r in rows if r["kind"] == "pipeline"
                        and r["method"] == ("legacy_b" if mutation == "aa_runtime" else "both"))
        if mutation == "trajectory":
            selected["trajectory_sha256"] = "different"
        elif mutation == "quality":
            selected["psnr"] -= 0.06
        else:
            selected["total_seconds"] = 20
    assert not controls.summarize(rows)["candidates"]["both"][expected]


def test_validator_rejects_missing_repeats_and_wrong_capacity():
    rows = synthetic_rows()
    problems = []
    controls.validate_rows(rows, controls.PROTOCOL, problems)
    assert problems == []
    rows[0]["call_seconds"] = [1.0]
    rows[-1]["n_gaussians"] = 1001
    controls.validate_rows(rows, controls.PROTOCOL, problems)
    assert len(problems) == 2


@pytest.mark.parametrize("method", ["legacy_a", "legacy_b", "both"])
@pytest.mark.parametrize("mutation", ["rgb_error", "rgb_nan", "quality_finite", "quality_nan",
                                      "psnr_nan", "ms_ssim_inf", "lpips_inf", "missing_metric"])
def test_pipeline_replay_and_nonfinite_guards_fail_closed(method, mutation):
    rows = synthetic_rows()
    row = next(r for r in rows if r["kind"] == "pipeline" and r["method"] == method)
    if mutation == "rgb_error":
        row["final_reference_rgb_max_error"] = 2.01e-5
    elif mutation == "rgb_nan":
        row["final_reference_rgb_max_error"] = float("nan")
    elif mutation == "quality_finite":
        row["quality"]["finite"] = False
    elif mutation == "quality_nan":
        row["quality"]["cvar99_mse"] = float("nan")
    elif mutation == "missing_metric":
        del row["lpips"]
    else:
        row[mutation.rsplit("_", 1)[0]] = float("nan" if mutation.endswith("nan") else "inf")
    result = controls.summarize(rows)["candidates"]["both"]
    assert not result["pipeline_quality_pass"]
    assert not result["pipeline_speed_pass"]


def test_error_and_duplicate_rows_cannot_pass_complete_gates():
    rows = synthetic_rows()
    rows[-1]["status"] = "error"
    result = controls.summarize(rows)
    assert not result["complete"]
    assert not result["candidates"]["both"]["pipeline_speed_pass"]
    duplicated = synthetic_rows()
    duplicated.append(copy.deepcopy(duplicated[-1]))
    assert not controls.summarize(duplicated)["complete"]


@pytest.mark.parametrize("mutation", ["wrong_arm", "wrong_seed", "wrong_state", "parent_horizon",
                                      "parent_selected", "wrong_count", "missing_calls"])
def test_summary_validates_identity_budget_and_raw_repeats_before_gating(mutation):
    rows = synthetic_rows()
    row = rows[0]
    key, value = {"wrong_arm": ("method", "tail"), "wrong_seed": ("seed", 1),
                  "wrong_state": ("state", "terminal"), "parent_horizon": ("iterations_run", 750),
                  "parent_selected": ("selected_iteration", 1), "wrong_count": ("n_gaussians", 1999),
                  "missing_calls": ("call_seconds", [1.0])}[mutation]
    row[key] = value
    result = controls.summarize(rows)
    assert not result["complete"]
    assert result["validation_problems"]
    assert not result["candidates"]["both"]["pipeline_speed_pass"]


def test_source_inventory_binds_all_python_modules_and_parent_validator():
    assert "src/structsplat/triage.py" in controls.SOURCES
    assert "scripts/experiments/fit050_color_ray.py" in controls.SOURCES
    assert controls.SOURCES == sorted(set(controls.SOURCES))


def test_stable_biased_same_state_aa_timing_cannot_be_averaged_away():
    rows = synthetic_rows()
    row = next(r for r in rows if r["kind"] == "same" and r["method"] == "legacy_b")
    row["call_seconds"] = [0.8] * 10
    row["total_seconds"] = 8.0
    result = controls.summarize(rows)
    assert result["complete"]
    for candidate in result["candidates"].values():
        assert candidate["same_state_correctness_pass"]
        assert not candidate["same_state_timing_eligible"]
        assert not candidate["component_speed_pass"]
        first_image = candidate["same_state_images"][0]
        assert first_image["aa_speed_ratio"] == 1.0
        assert first_image["aa_pairs"][0]["speed_ratio"] == 1.25
        assert not first_image["aa_pairs"][0]["timing_stable"]


def test_cpu_pipeline_worker_complete_artifact_wiring(tmp_path, monkeypatch):
    """Exercise the actual schedule and report path on one procedural CPU field."""
    import torch
    from scripts.experiments import port007_quality_reuse as driver
    from structsplat.pipeline import PipelineConfig

    torch.set_num_threads(1)
    rng = np.random.default_rng(41)
    source = rng.uniform(0.1, 0.9, (32, 32, 3)).astype(np.float32)
    monkeypatch.setattr(driver, "load_image", lambda _: source)
    monkeypatch.setattr(driver, "gpu_snapshot", lambda: {"timing_eligible": False, "error": "CPU test"})
    monkeypatch.setitem(driver.PROTOCOL["pipeline"], "config", asdict(PipelineConfig(
        capacity=16, step_scale=0.001, device="cpu", renderer="normalized")))

    def simple_scores(raw, target):
        mse = float((raw - target).square().mean())
        return {"psnr": -10 * np.log10(max(mse, 1e-12)), "mse": mse,
                "mae": float((raw - target).abs().mean()), "ssim": 0.5,
                "ms_ssim": 0.5, "lpips": 0.1}

    monkeypatch.setattr(driver, "score", simple_scores)
    request = {"cell_id": "pipeline_procedural_s0_legacy_a", "kind": "pipeline",
               "image": 9, "seed": 0, "method": "legacy_a"}
    driver.pipeline_worker(request, tmp_path, device="cpu")
    directory = tmp_path / "cells" / request["cell_id"]
    row = json.loads((directory / "row.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    assert row["status"] == "ok" and row["n_gaussians"] <= 16
    assert row["total_seconds"] > row["observer_seconds"] >= 0
    assert len(history["checkpoints"]) == len(history["native_events"]) == row["event_count"]
    assert len(list((directory / "snapshots").glob("*.npz"))) == row["event_count"]
    assert set(("psnr", "ssim", "ms_ssim", "lpips", "mse", "mae", "cvar99_mse",
                "p99_mse", "interior_hole_fraction", "boundary_hole_fraction", "elapsed_seconds")) <= set(history["checkpoints"][0])
    for name in ("field.npz", "target.npy", "reconstruction.npy", "reference_reconstruction.npy", "config.json", "history.json",
                 "target.png", "reconstruction.png", "error.png", "curves.png", "trajectory.json"):
        assert (directory / name).is_file()
