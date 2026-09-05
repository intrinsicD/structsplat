import copy
from dataclasses import asdict
import json
import math

import numpy as np
import pytest
import torch

from benchmarks.hier_additive_controls import additive_render, pack
from benchmarks.hier_research_report import _validate_convergence_cell, sha256, write_json
from scripts.experiments.hier035_convergence import (
    PROTOCOL, evaluate_results, fixture, resolved_config,
)


@pytest.mark.parametrize("family", PROTOCOL["families"])
@pytest.mark.parametrize("seed", PROTOCOL["seeds"] + [77])
def test_convergence_fixture_is_finite_bounded_owned_and_reproducible(family, seed):
    first, target = fixture(family, seed)
    second, again = fixture(family, seed)
    assert first.n == 16 and target.shape == (64, 64, 3)
    assert bool(torch.isfinite(pack(first)).all()) and bool(((target >= 0) & (target <= 1)).all())
    torch.testing.assert_close(pack(first), pack(second), rtol=0, atol=0)
    torch.testing.assert_close(target, again, rtol=0, atol=0)
    first.colors.zero_()
    assert bool(second.colors.abs().sum() > 0)


def make_rows(root):
    rows = []
    for seed in PROTOCOL["seeds"]:
        initial, target = fixture("translated", seed)
        for index, method in enumerate(PROTOCOL["arms"]):
            identity = f"translated_s{seed}_{method}"
            directory = root / "cells" / identity
            directory.mkdir(parents=True)
            initial.save(directory / "input_field.npz")
            np.save(directory / "target.npy", target.numpy())
            psnr = 20 + (index * 0.1 if method.startswith("adam") else 0.7)
            history = [{"iteration": step, "elapsed_seconds": step / 160 + 0.001,
                        "psnr": psnr, "objective": 0.5 * 10 ** (-psnr / 10),
                        "forward_evaluations": step + 1, "gradient_evaluations": step}
                       for step in range(161)]
            write_json(directory / "history.json", {"checkpoints": history, "nominal_iterations": 160})
            rows.append({"cell_id": identity, "family": "translated", "seed": seed,
                         "method": method, "status": "ok", "psnr": psnr, "ms_ssim": 0.9,
                         "lpips": 0.1, "n_gaussians": 16, "iterations_run": 160,
                         "selected_iteration": 160, "cold_render_max_abs": 0, "total_seconds": 1.1})
    return rows


@pytest.mark.parametrize("failure", ["none", "missing", "count", "horizon", "parity", "perceptual", "weak"])
def test_family_gate_compares_best_adam_and_requires_all_cells(tmp_path, failure):
    rows = make_rows(tmp_path)
    candidate = next(r for r in rows if r["method"] == "diagonal")
    if failure == "missing":
        rows.remove(next(r for r in rows if r["method"] == "adam_0p3"))
    elif failure == "count":
        candidate["n_gaussians"] = 17
    elif failure == "horizon":
        candidate["iterations_run"] = 159
    elif failure == "parity":
        candidate["cold_render_max_abs"] = 0.001
    elif failure == "perceptual":
        candidate["lpips"] += 0.02
    elif failure == "weak":
        candidate["psnr"] -= 1.0
    decision = evaluate_results(tmp_path, rows)
    record = next(r for r in decision["records"] if r["method"] == "diagonal")
    assert record["passes_iteration_quality_gate"] == (failure == "none")
    assert not decision["speed_claim"] and not decision["default_promotion"]
    for pair in record["pairs"]:
        if pair["complete"]:
            assert pair["adam_envelope_method"] == "adam_3"


def smoke_cell(root):
    initial, target = fixture("translated", 77)
    raw = additive_render(initial, 64, 64).detach()
    directory = root / "cells" / "translated_s77_diagonal"
    directory.mkdir(parents=True)
    initial.save(directory / "input_field.npz")
    initial.save(directory / "field.npz")
    np.save(directory / "target.npy", target.numpy())
    np.save(directory / "reconstruction.npy", raw.numpy())
    mse = float((raw.double() - target.double()).square().mean())
    psnr = -10 * math.log10(max(mse, 1e-12))
    request = {"family": "translated", "seed": 77, "method": "diagonal",
               "cell_id": "translated_s77_diagonal", "smoke": True}
    cfg = asdict(resolved_config("diagonal", True))
    write_json(directory / "request.json", request)
    write_json(directory / "config.json", {"request": request, "control": cfg,
        "target_sha256": sha256(directory / "target.npy"),
        "input_field_sha256": sha256(directory / "input_field.npz")})
    trace = [{"iteration": i, "elapsed_seconds": i * 0.1, "psnr": psnr,
              "objective": mse * 0.5, "forward_evaluations": i + 1, "gradient_evaluations": i}
             for i in range(4)]
    write_json(directory / "history.json", {"checkpoints": trace, "nominal_iterations": 3})
    row = {**request, "iterations_run": 3, "selected_iteration": 3, "gradient_evaluations": 3,
           "total_seconds": 0.31, "psnr": psnr, "initial_psnr": psnr, "raw_mse": mse,
           "n_gaussians": 16, "lpips": 0.1, "cold_render_max_abs": 0}
    return directory, row


@pytest.mark.parametrize("failure", ["none", "raw", "config", "history", "field", "identity"])
def test_convergence_contract_rejects_semantic_corruption(tmp_path, failure):
    directory, row = smoke_cell(tmp_path)
    if failure == "raw":
        np.save(directory / "reconstruction.npy", np.zeros((64, 64, 3), np.float32))
    elif failure == "config":
        config = json.loads((directory / "config.json").read_text())
        config["control"]["steps"] = 4
        write_json(directory / "config.json", config)
    elif failure == "history":
        history = json.loads((directory / "history.json").read_text())
        history["checkpoints"] = history["checkpoints"][:-1]
        write_json(directory / "history.json", history)
    elif failure == "field":
        initial, _target = fixture("translated", 77)
        initial.subset(torch.arange(15)).save(directory / "field.npz")
    elif failure == "identity":
        row["seed"] = 0
    # JSON round trip is the manifest's canonical tuple/list conversion.
    protocol = json.loads(json.dumps(copy.deepcopy(PROTOCOL)))
    problems = []
    _validate_convergence_cell(tmp_path, row, protocol, True, problems)
    assert bool(problems) == (failure != "none"), problems

