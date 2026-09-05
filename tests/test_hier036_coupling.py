from dataclasses import asdict
import json
import math

import numpy as np
import pytest

from benchmarks.hier_research_report import _validate_coupling_cell, sha256, write_json
from scripts.experiments.hier035_convergence import fixture
from scripts.experiments.hier036_coupling import PROTOCOL, evaluate_results, resolved_config
from benchmarks.hier_additive_controls import additive_render


def policy_rows(root, monkeypatch):
    import benchmarks.hier_research_report as report
    # Gate arithmetic unit test: semantic corruption is covered separately with real artifacts.
    def fake_validate(root, row, protocol, diagnostic, problems):
        if row.get("invalid"):
            problems.append("deliberate invalid cell")
    monkeypatch.setattr(report, "_validate_coupling_cell", fake_validate)
    rows = []
    for family in PROTOCOL["families"]:
        for seed in PROTOCOL["seeds"]:
            for method in PROTOCOL["arms"]:
                identity = f"{family}_s{seed}_{method}"
                directory = root / "cells" / identity
                directory.mkdir(parents=True)
                for name in ("input_field.npz", "target.npy"):
                    (directory / name).write_bytes(f"{family}:{seed}".encode())
                psnr = 21.5 if method.startswith("full") else 20.2 if method.startswith("adam") else 20.
                rows.append({"cell_id": identity, "family": family, "seed": seed, "method": method,
                             "status": "ok", "smoke": False, "psnr": psnr, "raw_mse": 10 ** (-psnr / 10),
                             "psnr_ceiling_applied": False, "ms_ssim": .9, "lpips": .1})
    return rows


@pytest.mark.parametrize("failure", ["none", "missing", "duplicate", "invalid", "input"])
def test_whole_matrix_gate_fails_closed_for_every_stratum(tmp_path, monkeypatch, failure):
    rows = policy_rows(tmp_path, monkeypatch)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = dict(rows[0])
    elif failure == "invalid":
        rows[-1]["invalid"] = True
    elif failure == "input":
        (tmp_path / "cells" / rows[-1]["cell_id"] / "target.npy").write_bytes(b"changed")
    decision = evaluate_results(tmp_path, rows)
    assert decision["whole_matrix_valid"] == (failure == "none")
    assert all(r["passes_gate"] == (failure == "none") for r in decision["records"])
    assert not decision["speed_claim"] and not decision["default_promotion"]
    for record in decision["records"]:
        if record["comparator"] == "adam_envelope":
            assert all(p.get("baseline_method", "adam_3") == "adam_3" for p in record["pairs"])


def test_coupling_and_adam_gates_cannot_substitute_for_each_other(tmp_path, monkeypatch):
    rows = policy_rows(tmp_path, monkeypatch)
    for row in rows:
        if row["method"].startswith("adam"):
            row["psnr"] = 25
    decision = evaluate_results(tmp_path, rows)
    assert all(r["passes_gate"] == (r["comparator"] == "coupling") for r in decision["records"])


@pytest.mark.parametrize("failure", ["weak", "seed_loss", "ssim", "lpips"])
def test_local_guard_failure_does_not_hide_other_cap_or_stratum(tmp_path, monkeypatch, failure):
    rows = policy_rows(tmp_path, monkeypatch)
    selected = [r for r in rows if r["family"] == "texture" and r["seed"] < 3 and r["method"] == "full_shared"]
    if failure == "weak":
        for row in selected:
            row["psnr"] = 20.4
    elif failure == "seed_loss":
        selected[0]["psnr"] = 19.8
    elif failure == "ssim":
        selected[0]["ms_ssim"] = .89
    else:
        selected[0]["lpips"] = .12
    decision = evaluate_results(tmp_path, rows)
    for record in decision["records"]:
        affected = record["family"] == "texture" and record["stratum"] == "exposed" and record["cap"] == "shared"
        assert record["passes_gate"] == (not affected)


def real_smoke_cell(root, method="full_shared"):
    field, target = fixture("translated", 77)
    raw = additive_render(field, 64, 64).detach()
    identity = f"translated_s77_{method}"
    directory = root / "cells" / identity
    directory.mkdir(parents=True)
    field.save(directory / "input_field.npz")
    field.save(directory / "field.npz")
    for name, array in (("target.npy", target.numpy()), ("reconstruction.npy", raw.numpy()),
                        ("initial_reconstruction.npy", raw.numpy())):
        np.save(directory / name, array)
    mse = float((raw.double() - target.double()).square().mean())
    psnr = -10 * math.log10(max(mse, 1e-12))
    request = {"family": "translated", "seed": 77, "method": method, "cell_id": identity, "smoke": True}
    config = {"request": request, "control": asdict(resolved_config(method, True)),
              "dense_mode": method, "dense_limits": PROTOCOL["dense_limits"], "precision": PROTOCOL["precision"],
              "target_sha256": sha256(directory / "target.npy"),
              "input_field_sha256": sha256(directory / "input_field.npz")}
    write_json(directory / "request.json", request)
    write_json(directory / "config.json", config)
    trace = [{"iteration": i, "elapsed_seconds": .1 * i, "psnr": psnr, "objective": mse * .5,
              "forward_evaluations": i + 1, "gradient_evaluations": i, "accepted": True,
              "line_search_trials": int(i > 0), "jacobian_constructions": i, "linear_solves": i,
              "directional_derivative": 0, "cross_gram_fraction": .1} for i in range(4)]
    write_json(directory / "history.json", {"checkpoints": trace, "nominal_iterations": 3})
    (directory / "progress.jsonl").write_text("\n".join(json.dumps(r) for r in trace) + "\n")
    row = {**request, "iterations_run": 3, "selected_iteration": 3, "gradient_evaluations": 3,
           "forward_evaluations": 4, "jacobian_constructions": 3, "linear_solves": 3, "rejected_updates": 0,
           "total_seconds": .31, "psnr": psnr, "initial_psnr": psnr, "raw_mse": mse,
           "psnr_uncapped": -10 * math.log10(mse), "psnr_ceiling_applied": False, "stratum": "diagnostic",
           "n_gaussians": 16, "lpips": .1, "cold_render_max_abs": 0,
           "retained_jacobian_bytes": 64*64*3*128*4, "retained_gram_bytes": 128*128*4,
           "peak_allocated_bytes": 10**7, "warmup_forward_evaluations": 21,
           "warmup_gradient_evaluations": 14, "warmup_jacobian_constructions": 8, "warmup_linear_solves": 8,
           "fixture_render_evaluations": 4, "cold_forward_evaluations": 1,
           "initial_diagnostic_forward_evaluations": 1, "worker_forward_evaluations": 31}
    return directory, row


@pytest.mark.parametrize("failure", ["none", "mode", "precision", "ledger", "trace", "ceiling", "initial", "stratum", "memory", "bounds"])
def test_real_artifact_contract_rejects_semantic_corruption(tmp_path, failure):
    directory, row = real_smoke_cell(tmp_path)
    if failure in ("mode", "precision"):
        config = json.loads((directory / "config.json").read_text())
        config["dense_mode" if failure == "mode" else "precision"] = "wrong"
        write_json(directory / "config.json", config)
    elif failure == "ledger":
        row["worker_forward_evaluations"] += 1
    elif failure == "trace":
        row["linear_solves"] -= 1
    elif failure == "ceiling":
        row["psnr_ceiling_applied"] = True
    elif failure == "initial":
        np.save(directory / "initial_reconstruction.npy", np.zeros((64, 64, 3), np.float32))
    elif failure == "stratum":
        row["stratum"] = "exposed"
    elif failure == "memory":
        row["retained_jacobian_bytes"] -= 1
    elif failure == "bounds":
        field, _ = fixture("translated", 77)
        field.means[0, 0] = -1
        field.save(directory / "field.npz")
    problems = []
    _validate_coupling_cell(tmp_path, row, json.loads(json.dumps(PROTOCOL)), True, problems)
    assert bool(problems) == (failure != "none"), problems
