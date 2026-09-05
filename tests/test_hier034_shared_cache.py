import copy
import json
import time

import pytest

from benchmarks.hier_research_report import _validate_shared_cache_scope, sha256, write_json
from scripts.experiments import hier034_shared_cache as shared
from scripts.experiments.hier034_basis_cache import PROTOCOL as ORIGINAL, speed_gate


def test_amendment_preserves_numerical_protocol_and_forbids_speed():
    for key in ("families", "seeds", "backends", "repetitions", "solver", "natural_files",
                "parity_max_abs", "sse_relative_tolerance", "checkpoint_violation_tolerance",
                "worker_timeout_seconds", "minimum_iterations", "warmup"):
        assert shared.PROTOCOL[key] == ORIGINAL[key]
    pairs = [{"pass": True, "paired_speedup": 100.} for _ in range(6)]
    original = speed_gate(pairs)
    amended = speed_gate(pairs, timing_eligible=False)
    assert original["passes_speed_gate"]
    assert amended["passes_interchangeability_gate"] and amended["descriptive_ratio_threshold_passed"]
    assert amended["timing_eligible"] is False and amended["passes_speed_gate"] is False
    assert shared.PROTOCOL["timing_eligible"] is False


def test_shared_gate_still_requires_every_numerical_pair():
    pairs = [{"pass": True, "paired_speedup": 100.} for _ in range(6)]
    pairs[-1]["pass"] = False
    assert not speed_gate(pairs, timing_eligible=False)["passes_interchangeability_gate"]
    assert not speed_gate(pairs[:-1], timing_eligible=False)["passes_interchangeability_gate"]


def snapshot(tick, phase=None):
    value = {"wall_time_utc": "2026-09-05T01:00:00+00:00", "monotonic_ns": tick,
             "status": "ok", "processes": "123, python, 692"}
    if phase:
        value["phase"] = phase
    return value


def scope_bundle(root):
    rows = []
    for backend in ORIGINAL["backends"]:
        identity = f"smooth_s77_r0_{backend}"
        directory = root / "cells" / identity
        directory.mkdir(parents=True)
        config = {"execution_profile": "shared_correctness", "timing_eligible": False,
                  "solver": {**ORIGINAL["solver"], "basis_cache": backend, "max_iterations": 3}}
        for label, tick in (("before", 2), ("after", 3)):
            path = directory / f"occupancy_{label}.json"
            write_json(path, snapshot(tick))
            config[f"occupancy_{label}_sha256"] = sha256(path)
        write_json(directory / "config.json", config)
        rows.append({"cell_id": identity, "family": "smooth", "seed": 77, "repeat": 0,
                     "backend": backend, "status": "ok", "timing_eligible": False,
                     "execution_profile": "shared_correctness"})
    log = [snapshot(1, "start"), snapshot(4, "end")]
    (root / "gpu_occupancy.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    decision = {"timing_eligible": False, "performance_disposition": shared.PROTOCOL["performance_disposition"],
                "records": [{"family": "smooth", "seed": 77, "backend": b,
                             "timing_eligible": False, "passes_speed_gate": False,
                             "integrity_eligible": False, "passes_interchangeability_gate": False}
                            for b in ("scatter", "csr")]}
    write_json(root / "decision.json", decision)
    manifest = {"protocol": json.loads(json.dumps(shared.PROTOCOL)), "diagnostic": True}
    return manifest, rows


@pytest.mark.parametrize("failure", ["none", "speed", "disposition", "row", "solver", "hash", "interval", "empty_monitor", "coverage"])
def test_shared_scope_cannot_silently_promote_or_lose_provenance(tmp_path, failure):
    manifest, rows = scope_bundle(tmp_path)
    if failure in ("speed", "disposition", "coverage"):
        decision = json.loads((tmp_path / "decision.json").read_text())
        if failure == "speed":
            decision["records"][0]["passes_speed_gate"] = True
        elif failure == "disposition":
            decision["performance_disposition"] = "speed confirmed"
        else:
            decision["records"].pop()
        write_json(tmp_path / "decision.json", decision)
    elif failure == "row":
        rows[0]["timing_eligible"] = True
    elif failure in ("solver", "hash", "interval"):
        directory = tmp_path / "cells" / rows[0]["cell_id"]
        path = directory / "config.json"
        config = json.loads(path.read_text())
        if failure == "solver":
            config["solver"]["max_iterations"] = 4
        elif failure == "hash":
            config["occupancy_before_sha256"] = "bad"
        else:
            write_json(directory / "occupancy_after.json", snapshot(9))
            config["occupancy_after_sha256"] = sha256(directory / "occupancy_after.json")
        write_json(path, config)
    elif failure == "empty_monitor":
        (tmp_path / "gpu_occupancy.jsonl").write_text("")
    problems = []
    _validate_shared_cache_scope(tmp_path, manifest, rows, problems)
    assert bool(problems) == (failure != "none"), problems


def test_monitor_stops_before_context_completion_and_preserves_error_status(tmp_path, monkeypatch):
    counter = 0
    def fake_snapshot():
        nonlocal counter
        counter += 1
        return {"wall_time_utc": "2026-09-05T01:00:00+00:00", "monotonic_ns": counter,
                "status": "error", "error": "deliberate unavailable telemetry"}
    monkeypatch.setattr(shared, "gpu_snapshot", fake_snapshot)
    protocol = copy.deepcopy(shared.PROTOCOL)
    protocol["occupancy_interval_seconds"] = .001
    monkeypatch.setattr(shared, "PROTOCOL", protocol)
    path = tmp_path / "monitor.jsonl"
    with pytest.raises(ValueError):
        with shared.monitor_gpu(path):
            time.sleep(.01)
            raise ValueError("deliberate worker-controller error")
    before = path.read_bytes()
    time.sleep(.01)
    assert path.read_bytes() == before
    rows = [json.loads(line) for line in before.splitlines()]
    assert rows[0]["phase"] == "start" and rows[-1]["phase"] == "end"
    assert any(r["phase"] == "sample" for r in rows)
    assert all(r["status"] == "error" for r in rows)
