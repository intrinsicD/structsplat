"""FIT-050 orchestration tests use only deterministic procedural CPU data."""
from dataclasses import asdict
import copy
import json
import shutil

import pytest
import torch

from benchmarks.fit050_controls import ARMS, fit_parent, full_frame_context, parent_configs, run_arm
from scripts.experiments import fit050_color_ray as driver
from structsplat.config import FitConfig
from structsplat.gaussians import GaussianField
from structsplat.safe_schedule import QualityMetrics


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture(scope="module")
def parent():
    return fit_parent(driver.procedural_target(), 77, smoke=True, device="cpu")


def test_parent_config_is_exactly_protocol_bound():
    init, cfg = parent_configs(0)
    assert asdict(init) == driver.PROTOCOL["parent_init"]
    assert asdict(cfg) == driver.PROTOCOL["parent_fit"]
    assert cfg.iters == 750 and cfg.log_every == 25
    assert cfg.quality_coverage_backend == cfg.quality_tail_backend == "reference"
    assert init.num_gaussians == 2000
    assert len(driver.expected_cells()) == len(set(driver.expected_cells())) == 48
    assert len(driver.expected_cells(diagnostic=True)) == 6


@pytest.mark.parametrize("arm", ARMS)
def test_controls_preserve_parent_and_report_complete_transaction(parent, arm):
    target = torch.from_numpy(driver.procedural_target())
    cfg = FitConfig(**parent["fit_config"])
    context = full_frame_context(target, cfg, parent["field"].n)
    snapshot = parent["field"].detached()
    steps_before = [state["step"].clone() for state in parent["optimizer_state"]["state"].values()]
    selected, quality, metadata, history = run_arm(parent["field"], target, cfg, *context,
                                                   arm, parent["optimizer_state"], smoke=True)
    assert selected.n == 16
    assert metadata["transaction_seconds"] > 0
    assert metadata["quality_coverage_backend"] == metadata["quality_tail_backend"] == "reference"
    if arm != "adam32":
        assert not metadata["noncolor_changed_fields"]
    else:
        assert metadata["iterations_run"] == 2
        assert metadata["counts"]["gradient_evaluations"] == 2
        assert len(history["iter"]) == 2
    for name in ("means", "log_scales", "rotations", "colors"):
        torch.testing.assert_close(getattr(parent["field"], name), getattr(snapshot, name), rtol=0, atol=0)
    for before, state in zip(steps_before, parent["optimizer_state"]["state"].values()):
        torch.testing.assert_close(before, state["step"], rtol=0, atol=0)
    assert quality.finite
    json.dumps(metadata, allow_nan=False)


def _rows():
    rows = []
    for image_id, seed in driver.workloads():
        for arm in ARMS:
            quality = QualityMetrics(2000, .001, .001, .002, .002, 0., 0., 0., 0., True).to_dict()
            rows.append({"cell_id": f"{driver.parent_id(image_id, seed)}_{arm}",
                         "parent_id": driver.parent_id(image_id, seed),
                         "image_id": image_id, "seed": seed, "method": arm, "status": "ok",
                         "n_gaussians": 2000, "psnr": 30.0 if arm == "noop" else 30.2,
                         "raw_mse": 0.001, "ms_ssim": 0.9, "lpips": 0.1,
                         "transaction_seconds": 0.1, "cold_render_max_abs": 0.0,
                         "noncolor_changed_fields": [], "accepted": arm != "noop",
                         "coefficients_changed": arm != "noop", "selected_fraction": 0.0 if arm == "noop" else 1.0,
                         "parent_field_sha256": "a", "parent_optimizer_sha256": "b",
                         "target_sha256": "c", "iterations_run": 32 if arm == "adam32" else 0,
                         "selected_iteration": 32 if arm == "adam32" else 0,
                         "cold_parameters_exact": True, "protected_metrics": quality,
                         "reporting_metrics": quality,
                         "counts": {"quality_evaluations": 1 if arm == "noop" else 2,
                                    "gaussian_renders": 1 if arm == "noop" else 35 if arm == "adam32" else 2,
                                    "raw_coverage_passes": 1 if arm == "noop" else 2,
                                    "basis_denominator_passes": int(arm == "legacy_cg32"),
                                    "basis_apply_calls": int(arm == "legacy_cg32"),
                                    "basis_transpose_calls": 2 if arm == "legacy_cg32" else 0,
                                    "gradient_evaluations": 32 if arm == "adam32" else 0}})
    return rows


def test_summary_clusters_seeds_within_images_and_retains_comparators():
    result = driver.summarize(_rows())
    assert all(record["passes_utility_gate"] for record in result["records"])
    assert all(record["median_image_averaged_gain_db"] == pytest.approx(0.2) for record in result["records"])
    assert set(result["records"][0]["pairs"][0]["comparators"]) == {"legacy_cg32", "cg_ray", "adam32"}
    assert not result["default_promotion"] and not result["speed_claim"]


@pytest.mark.parametrize("corruption", ["missing", "error", "count", "input", "geometry", "parity",
                                         "horizon", "cold_parameters", "quality_finite", "counts"])
def test_invalid_complete_matrix_cannot_pass_utility(corruption):
    rows = _rows()
    if corruption == "missing":
        rows.pop()
    elif corruption == "error":
        rows[0]["status"] = "error"
    elif corruption == "count":
        rows[0]["n_gaussians"] = 2001
    elif corruption == "input":
        rows[0]["parent_field_sha256"] = "changed"
    elif corruption == "geometry":
        rows[0]["noncolor_changed_fields"] = ["means"]
    elif corruption == "parity":
        rows[0]["cold_render_max_abs"] = 0.1
    elif corruption == "horizon":
        rows[0]["iterations_run"] = 999
    elif corruption == "cold_parameters":
        rows[0]["cold_parameters_exact"] = False
    elif corruption == "quality_finite":
        rows[0]["protected_metrics"]["finite"] = False
    else:
        rows[0]["counts"] = {}
    assert not any(r["passes_utility_gate"] for r in driver.summarize(rows)["records"])


def test_full_procedural_worker_writes_parent_and_six_cells(tmp_path):
    parent_dir = tmp_path / "parents" / "procedural_s77"
    parent_dir.mkdir(parents=True)
    request = {"root": str(tmp_path), "image_id": -1, "seed": 77, "parent_id": "procedural_s77",
               "smoke": True, "arm_order": ARMS}
    (parent_dir / "request.json").write_text(json.dumps(request))
    driver.worker(parent_dir / "request.json")
    rows = json.loads((parent_dir / "rows.json").read_text())
    assert len(rows) == 6
    assert all(row["status"] == "ok" for row in rows), rows
    for filename in ("initial_field.npz", "field.npz", "optimizer_state.pt", "target.npy",
                     "history.json", "config.json", "metrics.json"):
        assert (parent_dir / filename).is_file()
    state = torch.load(parent_dir / "optimizer_state.pt", weights_only=True)
    assert state["state"]
    problems = []
    driver.validate_rows(rows, driver.PROTOCOL, problems, diagnostic=True)
    assert not problems
    for row in rows:
        for relative in row["artifacts"].values():
            assert (tmp_path / relative).is_file()
        history = json.loads((tmp_path / "cells" / row["cell_id"] / "history.json").read_text())
        assert "reporting_work" in history
    assert not any(record["passes_utility_gate"] for record in driver.summarize(rows, diagnostic=True)["records"])


@pytest.fixture(scope="module")
def artifact_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("fit050-artifacts")
    parent_dir = root / "parents" / "procedural_s77"
    parent_dir.mkdir(parents=True)
    request = {"root": str(root), "image_id": -1, "seed": 77, "parent_id": "procedural_s77",
               "smoke": True, "arm_order": ARMS}
    (parent_dir / "request.json").write_text(json.dumps(request))
    driver.worker(parent_dir / "request.json")
    rows = json.loads((parent_dir / "rows.json").read_text())
    assert all(row["status"] == "ok" for row in rows)
    (parent_dir / "occupancy.jsonl").write_text("")
    (root / "occupancy.json").write_text(json.dumps({"samples": [], "timing_eligible": False}))
    decision = driver.summarize(rows, diagnostic=True)
    decision["timing_eligible"] = False
    (root / "decision.json").write_text(json.dumps(decision))
    problems = []
    driver.validate_rows(rows, driver.PROTOCOL, problems, diagnostic=True)
    driver.validate_artifacts(root, rows, driver.PROTOCOL, problems, diagnostic=True)
    assert not problems
    return root, rows


def _rehash(root):
    hashes = {path.relative_to(root).as_posix(): driver.sha256(path)
              for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    (root / "manifest.json").write_text(json.dumps({"files": hashes}))
    assert all(driver.sha256(root / name) == digest for name, digest in hashes.items())


@pytest.mark.parametrize("corruption", ["parent_config", "parent_history", "parent_adam_step", "parent_count",
    "cell_request", "cell_config", "metadata_counts", "trial_gate", "rollback_field", "noncolor_field",
    "reconstruction", "decision", "occupancy"])
def test_rehashed_artifact_corruption_is_rejected(artifact_fixture, tmp_path, corruption):
    original, original_rows = artifact_fixture
    root = tmp_path / "bundle"
    shutil.copytree(original, root)
    rows = copy.deepcopy(original_rows)
    parent_dir = root / "parents" / "procedural_s77"
    ray_row = next(row for row in rows if row["method"] == "jacobi_ray")
    ray_dir = root / "cells" / ray_row["cell_id"]
    def alter(path, function):
        value = json.loads(path.read_text())
        function(value)
        path.write_text(json.dumps(value))
    if corruption == "parent_config":
        alter(parent_dir / "config.json", lambda value: value["fit"].update(lr_color=0.7))
    elif corruption == "parent_history":
        alter(parent_dir / "history.json", lambda value: value["iter"].__setitem__(-1, 999))
    elif corruption == "parent_adam_step":
        path = parent_dir / "optimizer_state.pt"
        state = torch.load(path, weights_only=True)
        next(iter(state["state"].values()))["step"].fill_(999)
        torch.save(state, path)
    elif corruption == "parent_count":
        path = parent_dir / "initial_field.npz"
        field = GaussianField.load(path)
        field.subset(torch.arange(field.n - 1)).save(path)
    elif corruption == "cell_request":
        alter(ray_dir / "request.json", lambda value: value.update(seed=999))
    elif corruption == "cell_config":
        alter(ray_dir / "config.json", lambda value: value["fit"].update(ssim_weight=0.3))
    elif corruption == "metadata_counts":
        alter(ray_dir / "history.json", lambda value: value["transaction"]["counts"].update(gaussian_renders=999))
    elif corruption == "trial_gate":
        def corrupt_gate(value):
            trial = value["transaction"]["trials"][0]
            trial["surrogate_accepted"] = not trial["surrogate_accepted"]
        alter(ray_dir / "history.json", corrupt_gate)
    elif corruption == "rollback_field":
        row = next(row for row in rows if row["method"] == "noop")
        path = root / "cells" / row["cell_id"] / "field.npz"
        field = GaussianField.load(path)
        field.colors[0, 0] += 0.01
        field.save(path)
    elif corruption == "noncolor_field":
        path = ray_dir / "field.npz"
        field = GaussianField.load(path)
        field.means[0, 0] += 0.01
        field.save(path)
    elif corruption == "reconstruction":
        import numpy as np
        path = ray_dir / "reconstruction.npy"
        raw = np.load(path, allow_pickle=False)
        raw[0, 0, 0] += 0.2
        np.save(path, raw)
    elif corruption == "decision":
        alter(root / "decision.json", lambda value: value.update(default_promotion=True))
    else:
        alter(root / "occupancy.json", lambda value: value.update(timing_eligible=True))
    _rehash(root)
    problems = []
    driver.validate_artifacts(root, rows, driver.PROTOCOL, problems, diagnostic=True)
    assert problems, corruption
