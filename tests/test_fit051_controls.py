"""FIT-051 protocol and adversarial artifact tests use only procedural CPU fixtures."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch

from benchmarks import fit051_controls as controls
from scripts.experiments import fit051_actual_color_ray as driver
from structsplat.safe_schedule import QualityMetrics


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def _rows():
    rows = []
    before = QualityMetrics(2000, 0.001, 0.001, 0.002, 0.002, 0., 0., 0., 0., True).to_dict()
    for image in controls.IMAGE_IDS:
        for seed in controls.SEEDS:
            pid = controls.parent_id(image, seed)
            for arm in controls.ARMS:
                accepted = arm != "noop"
                mse = 10 ** (-(30.2 if accepted else 30.0) / 10)
                after = QualityMetrics(2000, mse, mse, mse * 2, mse * 2, 0., 0., 0., 0., True).to_dict()
                rows.append({"cell_id": f"{pid}_{arm}", "parent_id": pid, "image_id": image, "seed": seed,
                    "method": arm, "status": "ok", "n_gaussians": 2000, "smoke": False,
                    "psnr": -10 * math.log10(mse), "raw_mse": mse, "ms_ssim": 0.9, "lpips": 0.1,
                    "transaction_seconds": 0.1, "total_seconds": 0.1, "selected_replay_max_abs": 0.,
                    "cold_render_max_abs": 0., "cold_parameters_exact": True,
                    "accepted": accepted, "coefficients_changed": accepted, "selected_fraction": 1. if accepted else 0.,
                    "iterations_run": 32 if arm == "adam32" else 0,
                    "selected_iteration": 32 if arm == "adam32" else 0,
                    "noncolor_changed_fields": [], "parent_protected_metrics": before,
                    "protected_metrics": after if accepted else before, "reporting_metrics": after,
                    "counts": {"quality_evaluations": 1, "gaussian_renders": 1,
                               "raw_coverage_passes": 1, "basis_denominator_passes": 0,
                               "basis_apply_calls": 0, "basis_transpose_calls": 0},
                    "parent_field_sha256": controls.PARENT_FILES[f"parents/{pid}/field.npz"],
                    "parent_optimizer_sha256": controls.PARENT_FILES[f"parents/{pid}/optimizer_state.pt"],
                    "target_sha256": controls.PARENT_FILES[f"parents/{pid}/target.npy"]})
    return rows


def test_exact_parent_inventory_and_complete_matrix():
    assert len(driver.expected_cells()) == len(set(driver.expected_cells())) == 56
    assert len(driver.expected_cells(diagnostic=True)) == 7
    assert len(controls.PARENT_FILES) == 48
    assert len({name.split("/")[1] for name in controls.PARENT_FILES}) == 8
    assert driver.PROTOCOL["parent_source_commit"] == "e2bf6ae6e06ca8050d9aa8a93d713679a0c9c150"
    assert driver.PROTOCOL["parent_file_sha256"] == controls.PARENT_FILES
    assert driver.SOURCES == sorted(set(driver.SOURCES))
    assert "src/structsplat/triage.py" in driver.SOURCES
    assert "src/structsplat/actual_color_ray.py" in driver.SOURCES
    assert "scripts/experiments/fit050_color_ray.py" in driver.SOURCES


def test_summary_reports_every_control_without_promoting_defaults():
    decision = driver.summarize(_rows())
    assert decision["complete_matrix"]
    assert all(row["passes_utility_gate"] for row in decision["records"])
    assert all(row["median_image_averaged_gain_db"] == pytest.approx(0.2) for row in decision["records"])
    assert set(decision["records"][0]["pairs"][0]["comparators"]) == {"legacy_cg32", "actual_cg_ray", "adam32"}
    assert not decision["speed_claim"] and not decision["default_promotion"]


@pytest.mark.parametrize("fault", ["duplicate", "missing", "error", "parent_hash", "count", "cold",
                                   "selected_replay", "nan", "missing_metric", "protected", "zero_work"])
def test_summary_cannot_bypass_identity_finiteness_or_gate_faults(fault):
    rows = _rows()
    row = next(row for row in rows if row["method"] == "native_gradient_ray")
    if fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "missing":
        rows.pop()
    elif fault == "error":
        row.update(status="error", error="retained worker failure")
    elif fault == "parent_hash":
        row["parent_field_sha256"] = "0" * 64
    elif fault == "count":
        row["n_gaussians"] = 1999
    elif fault == "cold":
        row["cold_render_max_abs"] = 2.1e-5
    elif fault == "selected_replay":
        row["selected_replay_max_abs"] = 2.1e-5
    elif fault == "nan":
        row["lpips"] = float("nan")
    elif fault == "missing_metric":
        del row["raw_mse"]
    elif fault == "protected":
        row["protected_metrics"] = QualityMetrics(2000, .002, .002, .004, .004, 0., 0., 0., 0., True).to_dict()
    else:
        row["counts"] = {}
    decision = driver.summarize(rows)
    assert not decision["complete_matrix"]
    assert not any(record["passes_utility_gate"] for record in decision["records"])


def _actual_case(method):
    from structsplat.actual_color_ray import refine_actual_color_ray
    from structsplat.config import FitConfig
    from structsplat.fit import _render
    from structsplat.gaussians import GaussianField
    from benchmarks.fit050_controls import full_frame_context, _changed_noncolors

    parent = GaussianField(torch.tensor([[2., 3.], [6., 5.]]),
        torch.full((2, 2), math.log(1.2)), torch.tensor([.2, -.4]), torch.full((2, 3), .03),
        torch.tensor([.3, -.2]), torch.full((2, 2), 4.),
        background_mask=torch.tensor([False, False]), filter_variance=torch.full((2,), .08))
    cfg = FitConfig(renderer="normalized", pixel_loss="l2", ssim_weight=0., support_fade=True,
                    aa_dilation=.04, color_solve_lambda=1e-4, render_chunk=1)
    truth = parent.detached()
    truth.colors.copy_(torch.tensor([[.2, .7, .3], [.8, .2, .6]]))
    target = _render(truth, cfg, 9, 9, support_fade_alpha=1.)
    context = full_frame_context(target, cfg, parent.n)
    selected, _, metadata, tensors = refine_actual_color_ray(parent, target, cfg, *context,
                                                            direction=controls.DIRECTIONS[method])
    metadata["transaction_seconds"] = metadata["elapsed_seconds"]
    metadata["noncolor_changed_fields"] = _changed_noncolors(parent, selected)
    row = {key: metadata[key] for key in ("accepted", "coefficients_changed", "selected_fraction",
            "rollback_reason", "counts", "noncolor_changed_fields", "transaction_seconds")}
    row.update({"method": method, "parent_protected_metrics": metadata["parent_metrics"],
                "protected_metrics": metadata["selected_metrics"]})
    arrays, inventory = {}, {}
    for key, value in tensors.items():
        if value is None:
            inventory[key] = None
        elif isinstance(value, list):
            inventory[key] = [f"{key}_{i}" for i in range(len(value))]
            arrays.update({name: item.numpy().copy() for name, item in zip(inventory[key], value)})
        else:
            inventory[key] = key
            arrays[key] = value.numpy().copy()
    return parent, selected, target, cfg, context, row, metadata, arrays, inventory


@pytest.mark.parametrize("method", list(controls.DIRECTIONS))
def test_cpu_artifact_replay_agrees_with_every_actual_direction(method):
    case = _actual_case(method)
    assert case[5]["accepted"]
    driver._validate_mechanism(*case, None, driver.PROTOCOL, True)


@pytest.mark.parametrize("fault", ["trial_acceptance", "replay_acceptance", "raw_trial", "raw_denominator",
                                   "direction", "selected_colors", "work", "fraction", "alpha", "timing"])
def test_cpu_artifact_replay_rejects_false_trial_field_gate_and_work_claims(fault):
    case = list(_actual_case("native_gradient_ray"))
    row, metadata, arrays = case[5], case[6], case[7]
    if fault == "trial_acceptance":
        metadata["trials"][0]["accepted"] = False
    elif fault == "replay_acceptance":
        metadata["trials"][-1]["replay_accepted"] = False
    elif fault == "raw_trial":
        arrays["trial_renders_0"] += .2
    elif fault == "raw_denominator":
        arrays["parent_denominator"] *= 0
    elif fault == "direction":
        arrays["direction"] += 0.1
    elif fault == "selected_colors":
        case[1].colors += .01
    elif fault == "work":
        metadata["counts"]["native_color_vjp_calls"] += 1
        row["counts"] = copy.deepcopy(metadata["counts"])
    elif fault == "fraction":
        metadata["trials"][0]["fraction"] = .25
    elif fault == "alpha":
        metadata["numerator"] *= 2
    else:
        metadata["trials"][0]["transaction_elapsed_seconds"] = metadata["transaction_seconds"] + 1
    with pytest.raises(ValueError):
        driver._validate_mechanism(*case, None, driver.PROTOCOL, True)


def test_cpu_artifact_replay_checks_exact_cg_endpoint_convention():
    case = list(_actual_case("actual_cg_ray"))
    case[7]["cg_endpoint_colors"][0, 0] += .001
    with pytest.raises(ValueError, match="CG direction"):
        driver._validate_mechanism(*case, None, driver.PROTOCOL, True)


@pytest.mark.parametrize("method,operand", [
    ("native_gradient_ray", "native_gradient_render"),
    ("native_gradient_ray", "gradient"),
    ("actual_gradient_ray", "gradient"),
    ("actual_jacobi_ray", "gradient"),
    ("actual_jacobi_ray", "diagonal"),
    ("actual_cg_ray", "cg_endpoint_colors"),
] + [(method, operand) for method in controls.DIRECTIONS
     for operand in ("direction", "direction_render")])
@pytest.mark.parametrize("inventory_fault", ["delete", "null"])
def test_declared_method_operands_cannot_be_deleted_with_consistently_lowered_work(method, operand, inventory_fault):
    case = list(_actual_case(method))
    row, metadata, arrays, inventory = case[5:]
    del arrays[operand]
    if inventory_fault == "delete":
        del inventory[operand]
    else:
        inventory[operand] = None
    counts = metadata["counts"]
    if operand == "direction_render":
        counts["actual_direction_render_calls"] = 0
        counts["gaussian_renders"] -= 1
        metadata["phase_seconds"].pop("actual_direction_render")
    elif operand == "native_gradient_render" or (operand == "gradient" and method == "native_gradient_ray"):
        counts["native_gradient_forward_calls"] = counts["native_color_vjp_calls"] = 0
        counts["gaussian_renders"] -= 1
        metadata["phase_seconds"].pop("native_color_vjp")
    elif operand == "gradient":
        counts["basis_denominator_passes"] = counts["basis_transpose_calls"] = 0
        metadata["phase_seconds"].pop("streaming_denominator")
        metadata["phase_seconds"].pop("streaming_transpose")
    elif operand == "diagonal":
        counts["basis_diagonal_passes"] = 0
        metadata["phase_seconds"].pop("streaming_diagonal")
    elif operand == "cg_endpoint_colors":
        for key in ("basis_denominator_passes", "basis_apply_calls", "basis_transpose_calls", "legacy_cg_iterations"):
            counts[key] = 0
        del metadata["legacy_cg"]
        metadata["phase_seconds"].pop("legacy_cg")
    row["counts"] = copy.deepcopy(counts)
    with pytest.raises(ValueError, match="inventory|operand"):
        driver._validate_mechanism(*case, None, driver.PROTOCOL, True)


@pytest.mark.parametrize("operand", ["gradient", "direction", "direction_render", "native_gradient_render",
                                    "replay_render", "replay_denominator", "trial_renders_0", "trial_denominators_0"])
@pytest.mark.parametrize("fault", ["shape", "dtype"])
def test_declared_ray_operand_shapes_and_dtypes_are_not_inferred_from_payloads(operand, fault):
    case = list(_actual_case("native_gradient_ray"))
    case[7][operand] = case[7][operand].reshape(-1) if fault == "shape" else case[7][operand].astype(np.float64)
    with pytest.raises(ValueError, match="shape/dtype"):
        driver._validate_mechanism(*case, None, driver.PROTOCOL, True)


@pytest.mark.parametrize("fault", ["irrelevant_diagonal", "irrelevant_cg", "replay_render", "replay_denominator",
                                   "trial_list", "cg_ledger"])
def test_method_stage_inventory_rejects_extra_work_and_missing_replay_or_trials(fault):
    case = list(_actual_case("actual_cg_ray" if fault == "cg_ledger" else "native_gradient_ray"))
    row, metadata, arrays, inventory = case[5:]
    if fault == "irrelevant_diagonal":
        inventory["diagonal"] = "diagonal"
        arrays["diagonal"] = np.ones(case[0].n, dtype=np.float32)
        metadata["counts"]["basis_diagonal_passes"] = 1
    elif fault == "irrelevant_cg":
        inventory["cg_endpoint_colors"] = "cg_endpoint_colors"
        arrays["cg_endpoint_colors"] = arrays["direction"].copy()
    elif fault.startswith("replay_"):
        del arrays[fault]
        inventory[fault] = None
    elif fault == "trial_list":
        del arrays["trial_renders_0"]
        inventory["trial_renders"] = []
    else:
        del metadata["legacy_cg"]
        for key in ("basis_denominator_passes", "basis_apply_calls", "basis_transpose_calls", "legacy_cg_iterations"):
            metadata["counts"][key] = 0
    row["counts"] = copy.deepcopy(metadata["counts"])
    with pytest.raises(ValueError, match="inventory|operand|tensor list"):
        driver._validate_mechanism(*case, None, driver.PROTOCOL, True)


@pytest.mark.cuda
@pytest.mark.parametrize("close_to_target", [False, True])
def test_procedural_cuda_inputs_use_identical_canonical_cpu_endpoint_and_curve_scores(close_to_target):
    """CPU/CUDA LPIPS agreement failed prospectively; reporting now uses one CPU backend.

    This checks GPU-sourced inputs, not cross-device perceptual-network equivalence. Protected
    quality remains device-native and still meets the unchanged portable replay tolerance.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from structsplat.config import FitConfig
    from structsplat.safe_schedule import _quality_from_render
    from benchmarks.fit050_controls import full_frame_context

    target = torch.from_numpy(driver.procedural_target())
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    perturbation = torch.stack((torch.sin(xx / 3.), torch.cos(yy / 5.), torch.sin((xx - yy) / 4.)), -1)
    raw = target + 1e-4 * perturbation if close_to_target else 1.7 * target - .35 + .22 * perturbation
    if not close_to_target:
        assert bool((raw < 0).any()) and bool((raw > 1).any())
    endpoint, points = {}, {}
    for device in ("cpu", "cuda"):
        image, truth = raw.to(device), target.to(device)
        cfg = FitConfig(renderer="normalized" if device == "cpu" else "cuda")
        context = full_frame_context(truth, cfg, 2)
        quality = _quality_from_render(image, truth, torch.ones_like(truth[..., 0]),
            context[0], context[1], context[2].coverage_tau, 2, tail_backend="reference").to_dict()
        endpoint[device] = driver.endpoint_score(image, truth, smoke=False)
        points[device] = driver.scored_point(image, truth, quality, 1, .1, stage="actual_trial")
    assert endpoint["cpu"] == endpoint["cuda"]
    for key in endpoint["cpu"]:
        assert points["cpu"][key] == points["cuda"][key] == endpoint["cpu"][key]
    for records in (endpoint, points):
        assert records["cpu"].keys() == records["cuda"].keys()
        for key, value in records["cpu"].items():
            observed = records["cuda"][key]
            if isinstance(value, (str, bool)):
                assert observed == value
            else:
                assert math.isfinite(value) and math.isfinite(observed)
                assert driver._close(value, observed, driver.PROTOCOL), (close_to_target, key, value, observed)


@pytest.mark.integration
def test_procedural_cli_bundle_passes_complete_artifact_checker(tmp_path, monkeypatch):
    from benchmarks.hier_research_report import validate_bundle
    out = tmp_path / "fit051-diagnostic"
    monkeypatch.setattr("sys.argv", ["fit051_actual_color_ray.py", "--out", str(out), "--smoke"])
    driver.main()
    assert validate_bundle(out, allow_dirty=True) == []
    rows = json.loads((out / "metrics.json").read_text())
    assert len(rows) == 7 and all(row["status"] == "ok" for row in rows)
    assert (out / "index.html").is_file()
    problems = []
    driver.validate_artifacts(out, rows, driver.PROTOCOL, problems, diagnostic=True)
    assert problems == []
