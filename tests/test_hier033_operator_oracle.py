from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from benchmarks.hier_additive_controls import ControlConfig, additive_render
from benchmarks.hier_operator_oracle import SHAPE, fixture
from benchmarks.hier_research_report import _validate_oracle_cell, sha256, write_json
from scripts.experiments.hier033_operator_oracle import PROTOCOL, evaluate_results


def rows_for_decision(root):
    rows = []
    for family in PROTOCOL["families"]:
        for seed in PROTOCOL["seeds"]:
            for action in PROTOCOL["actions"]:
                identity = f"{family}_s{seed}_{action}"
                directory = root / "cells" / identity
                directory.mkdir(parents=True)
                # Decision arithmetic uses file identity; serialization has separate tests below.
                (directory / "base_field.npz").write_bytes(b"identical decision-test base")
                (directory / "target.npy").write_bytes(b"identical decision-test target")
                gain = 0.01 if action == "move_m5" else 0.0
                rows.append({"family": family, "seed": seed, "action": action,
                    "cell_id": identity, "status": "ok", "n_gaussians": 3,
                    "iterations_run": 20, "selected_iteration": 20, "cold_render_max_abs": 0,
                    "action_family": action.split("_")[0], "predicted_gain": gain,
                    "base_objective": 0.02, "immediate_gain": gain, "recovered_gain": gain,
                    "position_activity": 0.001, "position_coherence": 0.001})
    return rows


@pytest.mark.parametrize("failure", ["none", "missing", "count", "horizon", "parity", "bad_prediction"])
def test_oracle_gate_requires_entire_atlas_and_detects_prediction_regret(tmp_path, failure):
    rows = rows_for_decision(tmp_path)
    if failure == "missing":
        rows.pop()
    elif failure == "count":
        rows[0]["n_gaussians"] = 4
    elif failure == "horizon":
        rows[0]["iterations_run"] = 19
    elif failure == "parity":
        rows[0]["cold_render_max_abs"] = 0.001
    elif failure == "bad_prediction":
        for row in rows:
            if row["action"] == "color_m5":
                row["predicted_gain"] = 0.02
    decision = evaluate_results(tmp_path, rows)
    assert decision["passes_local_selector_gate"] == (failure == "none")
    assert not decision["speed_claim"] and not decision["default_promotion"]
    if failure == "none":
        assert all(c["cancellation_counterexample"] for c in decision["cases"])
        assert decision["low_regret_fractions"] == {"immediate": 1, "recovered": 1}


def noop_cell(root):
    field, target = fixture("translation", 77)
    raw = additive_render(field, *SHAPE).detach()
    value = float(0.5 * (raw.double() - target.double()).square().mean())
    psnr = -10 * math.log10(max(2 * value, 1e-12))
    request = {"cell_id": "translation_s77_noop", "family": "translation", "seed": 77,
               "action": "noop", "smoke": True}
    directory = root / "cells" / request["cell_id"]
    directory.mkdir(parents=True)
    for name in ("base_field", "input_field", "field"):
        field.save(directory / (name + ".npz"))
    np.save(directory / "target.npy", target.numpy())
    for name in ("base_reconstruction", "immediate_reconstruction", "reconstruction"):
        np.save(directory / (name + ".npy"), raw.numpy())
    config = {"request": request, "recovery": asdict(ControlConfig(steps=2)),
              "action_family": "noop", "donor": None, "magnitude": 0.0, "predicted_gain": 0.0}
    for name in ("base_field", "input_field", "target"):
        config[name + "_sha256"] = sha256(directory / (name + (".npy" if name == "target" else ".npz")))
    write_json(directory / "request.json", request)
    write_json(directory / "config.json", config)
    trace = [{"iteration": i, "objective": value, "psnr": psnr, "elapsed_seconds": 0.1 * i,
              "gradient_evaluations": i, "forward_evaluations": i + 1} for i in range(3)]
    write_json(directory / "history.json", {"checkpoints": trace, "nominal_iterations": 2})
    row = {**request, "action_family": "noop", "donor": None, "magnitude": 0.0,
        "predicted_gain": 0.0, "iterations_run": 2, "selected_iteration": 2,
        "gradient_evaluations": 2, "forward_evaluations": 3, "total_seconds": 0.21,
        "counter_scope": "recovery only", "shared_case_render_evaluations": 6,
        "extra_cell_render_evaluations": 2, "total_cell_render_evaluations": 5,
        "base_objective": value, "immediate_objective": value, "terminal_objective": value,
        "immediate_gain": 0.0, "recovered_gain": 0.0, "psnr": psnr, "n_gaussians": 3,
        "lpips": 0.01, "cold_render_max_abs": 0.0, "position_activity": 0.0,
        "position_coherence": 0.0, "proposal_seconds": 0.01}
    return directory, row, field


def test_disjoint_phase_failures_cannot_pass_joint_condition_gate(tmp_path):
    rows = rows_for_decision(tmp_path)
    cases = [(family, seed) for family in PROTOCOL["families"] for seed in PROTOCOL["seeds"]]
    for row in rows:
        index = cases.index((row["family"], row["seed"]))
        if row["action"] == "color_m5":
            if index < 3:
                row["immediate_gain"] = 0.02
            elif index < 6:
                row["recovered_gain"] = 0.02
    decision = evaluate_results(tmp_path, rows)
    assert decision["low_regret_fractions"] == {"immediate": 15 / 18, "recovered": 15 / 18}
    assert decision["joint_low_regret_fraction"] == 12 / 18
    assert not decision["passes_local_selector_gate"]


@pytest.mark.parametrize("failure", ["none", "gain", "raw", "work", "count", "untouched", "identity"])
def test_oracle_contract_checks_actual_artifacts_and_untouched_rows(tmp_path, failure):
    directory, row, field = noop_cell(tmp_path)
    if failure == "gain":
        row["immediate_gain"] = 0.01
    elif failure == "raw":
        np.save(directory / "immediate_reconstruction.npy", np.zeros((64, 64, 3), np.float32))
    elif failure == "work":
        row["forward_evaluations"] = 2
    elif failure == "count":
        row["n_gaussians"] = 4
    elif failure == "untouched":
        field.means[1, 0] += 1
        field.save(directory / "input_field.npz")
        config = json.loads((directory / "config.json").read_text())
        config["input_field_sha256"] = sha256(directory / "input_field.npz")
        write_json(directory / "config.json", config)
    elif failure == "identity":
        row["seed"] = 0
    problems = []
    _validate_oracle_cell(tmp_path, row, json.loads(json.dumps(PROTOCOL)), True, problems)
    assert bool(problems) == (failure != "none"), problems
