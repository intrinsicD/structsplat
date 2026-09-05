"""Byte-consistent synthetic corruption tests; never execute formal image outcomes."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from benchmarks.fit050_controls import parent_configs
from benchmarks.hier_research_report import _validate_port_artifacts, sha256, write_json
from benchmarks.port007_controls import PROTOCOL, summarize
from structsplat.safe_schedule import QualityMetrics


def port_tree(root):
    target = np.full((3, 4, 3), .1, dtype=np.float32)
    raw = np.zeros_like(target)
    means = np.zeros((2000, 2), dtype=np.float32)
    files = {}
    for image in PROTOCOL["images"]:
        for seed in (0, 1):
            parent = root / "parents" / f"coco{image:012d}_s{seed}"
            parent.mkdir(parents=True)
            init, fit = parent_configs(seed)
            write_json(parent / "config.json", {"init": asdict(init), "fit": asdict(fit)})
            write_json(parent / "history.json", {"iter": [0, 749]})
            (parent / "optimizer_state.pt").write_bytes(b"synthetic fixture only")
            np.save(parent / "target.npy", target)
            np.savez(parent / "field.npz", means=means)
            np.savez(parent / "initial_field.npz", means=means)
            files.update({p.relative_to(root).as_posix(): sha256(p) for p in parent.iterdir()})
    repository = {"commit": "a" * 40}
    manifest = {"task": "FIT-050", "diagnostic": False, "repository": repository, "files": files}
    write_json(root / "parent_manifest.json", manifest)
    write_json(root / "parent_source.json", {"source_identity": manifest,
        "manifest_sha256": sha256(root / "parent_manifest.json"), "repository": repository, "files": files})
    write_json(root / "RUNNING.json", {"repository": repository})
    quality = QualityMetrics(2000, .01, .01, .01, .01, 0, 0, 0, 0, True).to_dict()
    raw_psnr = -10 * np.log10(np.square(raw.astype(np.float64) - target.astype(np.float64)).mean())
    records = [{"round": i, "seconds": 1., "quality": quality, "replay_quality": quality}
               for i in range(10)]
    label = "same_coco000000000009_s0_initial_legacy_a"
    directory = root / "cells" / label
    directory.mkdir(parents=True)
    other = root / "cells" / "same_coco000000000009_s0_terminal_legacy_a"
    other.mkdir()
    write_json(other / "measurements.json", records)
    write_json(directory / "measurements.json", records)
    np.savez(directory / "measurements.npz", renders=np.stack([raw] * 10),
        replay_renders=np.stack([raw] * 10), raw_denominators=np.ones((10, 3, 4)),
        hole_masks=np.zeros((10, 3, 4), dtype=bool))
    np.savez(directory / "field.npz", means=means)
    np.save(directory / "target.npy", target)
    np.save(directory / "reconstruction.npy", raw)
    null = {"accepted": False, "reasons": ["no_material_gain"]}
    checks = [{"round": i, "max_rgb_error": 0., "max_replay_rgb_error": 0.,
               "hole_mask_equal": True, "null_decision_equal": True,
               "changed_decision_equal": True, "finite": True, "pass": True} for i in range(10)]
    decisions = [{"round": i, "null": null, "null_control": null, "changed": null,
                  "changed_control": null, "changed_direction": "terminal_to_initial"} for i in range(10)]
    deltas = [{key: 0 for key, value in quality.items() if not isinstance(value, bool)} for _ in range(10)]
    write_json(directory / "parity.json", {"checks": checks, "decisions": decisions, "quality_deltas": deltas})
    write_json(directory / "history.json", {"checkpoints": [{"psnr": raw_psnr} for _ in range(10)]})
    request = {"cell_id": label, "kind": "same", "image": 9, "seed": 0, "method": "legacy_a"}
    parent = root / "parents" / "coco000000000009_s0"
    parent_config = json.loads((parent / "config.json").read_text())
    write_json(directory / "config.json", {"request": request, "fit": parent_config["fit"],
        "parent_id": parent.name, "parent_config": parent_config,
        "gpu_snapshots": [{"error": None, "pid": 123, "compute_processes": []}]})
    row = {**request, "state": "initial", "status": "ok", "n_gaussians": 2000,
           "mse": float(np.square(raw.astype(np.float64) - target.astype(np.float64)).mean()),
           "mae": float(np.abs(raw.astype(np.float64) - target.astype(np.float64)).mean()),
           "psnr": raw_psnr, "timing_eligible": True, "quality": quality, "parity_pass": True,
           "call_seconds": [1.] * 10, "total_seconds": 10., "call_time_cv": 0.}
    write_json(directory / "row.json", row)
    write_json(root / "summary.json", summarize([row]))
    return [row], directory


@pytest.mark.parametrize("corruption", ["none", "parity", "timing", "config", "holes", "raw", "decision", "parent"])
def test_port_recomputes_artifacts_not_success_flags(tmp_path, corruption):
    rows, directory = port_tree(tmp_path)
    if corruption in {"parity", "config", "decision"}:
        path = directory / {"parity": "parity.json", "config": "config.json", "decision": "parity.json"}[corruption]
        value = json.loads(path.read_text())
        if corruption == "parity":
            value["checks"][0]["pass"] = False
        elif corruption == "config":
            value["fit"]["quality_tail_backend"] = "shared"
        else:
            value["decisions"][0]["null"]["accepted"] = True
        write_json(path, value)
    elif corruption == "timing":
        rows[0]["call_seconds"][0] = 2.
        write_json(directory / "row.json", rows[0])
        write_json(tmp_path / "summary.json", summarize(rows))
    elif corruption == "holes":
        with np.load(directory / "measurements.npz") as data:
            arrays = {key: data[key] for key in data.files}
        arrays["hole_masks"][0, 0, 0] = True
        np.savez(directory / "measurements.npz", **arrays)
    elif corruption == "raw":
        np.save(directory / "reconstruction.npy", np.ones((3, 4, 3), dtype=np.float32))
    elif corruption == "parent":
        (tmp_path / "parents" / "coco000000000009_s0" / "optimizer_state.pt").write_bytes(b"changed")
    problems = []
    _validate_port_artifacts(tmp_path, rows, PROTOCOL, problems)
    assert bool(problems) == (corruption != "none"), problems


@pytest.mark.parametrize("corruption", ["none", "reference", "mask", "trajectory", "work"])
def test_port_pipeline_artifacts_against_actual_procedural_worker(tmp_path, monkeypatch, corruption):
    import torch
    from scripts.experiments import port007_quality_reuse as driver
    from structsplat.pipeline import PipelineConfig

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        port_tree(tmp_path)
        target = np.random.default_rng(51).uniform(.1, .9, (32, 32, 3)).astype(np.float32)
        monkeypatch.setattr(driver, "load_image", lambda _: target)
        monkeypatch.setattr(driver, "gpu_snapshot", lambda: {"error": "CPU fixture", "timing_eligible": False})
        monkeypatch.setitem(PROTOCOL["pipeline"], "config", asdict(PipelineConfig(
            capacity=16, step_scale=.001, device="cpu", renderer="normalized")))
        def scores(raw, target):
            mse = float((raw.double() - target.double()).square().mean())
            return {"psnr": -10 * np.log10(max(mse, 1e-12)), "mse": mse,
                    "mae": float((raw - target).abs().mean()),
                    "ssim": .5, "ms_ssim": .5, "lpips": .1}
        monkeypatch.setattr(driver, "score", scores)
        request = {"cell_id": "pipeline_coco000000000009_s0_legacy_a", "kind": "pipeline",
                   "image": 9, "seed": 0, "method": "legacy_a"}
        driver.pipeline_worker(request, tmp_path, device="cpu")
        directory = tmp_path / "cells" / request["cell_id"]
        row = json.loads((directory / "row.json").read_text())
        if corruption == "reference":
            np.save(directory / "reference_reconstruction.npy", np.zeros_like(target))
        elif corruption == "mask":
            np.save(directory / "mask.npy", np.zeros((32, 32), dtype=bool))
        elif corruption == "trajectory":
            write_json(directory / "trajectory.json", [])
        elif corruption == "work":
            row["iterations_run"] += 1
            write_json(directory / "row.json", row)
        write_json(tmp_path / "summary.json", summarize([row]))
        problems = []
        _validate_port_artifacts(tmp_path, [row], PROTOCOL, problems)
        assert bool(problems) == (corruption != "none"), problems
    finally:
        torch.set_num_threads(original_threads)
