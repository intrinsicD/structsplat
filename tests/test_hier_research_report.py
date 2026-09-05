import copy
import json

import numpy as np
import pytest

from benchmarks.hier_research_report import (
    ResearchBundle, protocol_digest, sha256, validate_bundle, write_json,
    has_approved_protocol,
)
from scripts.experiments.hier034_basis_cache import (
    PROTOCOL, analyze_bundle, checkpoint_agreement, speed_gate,
)
from structsplat.observation_field import save_observation_field
from structsplat.overlap_elimination import lattice_observation_field


def test_prospective_receipt_requires_distinct_approved_exact_digest_and_no_outcomes():
    text = "### Protocol review\n\n#### Reviewer\nindependent-reviewer\n\n#### Verdict\nApproved\n\n#### Protocol digest\n" + "a" * 64 + "\n\n#### Digest scope\nsource and protocol\n\n#### Outcomes accessed\nNo\n\n#### Review focus\ncontrols\n"
    assert has_approved_protocol(text, "a" * 64)
    assert not has_approved_protocol(text, "b" * 64)
    for old, new in (("independent-reviewer", "codex-root"), ("Approved", "Rejected"), ("\nNo\n", "\nYes\n")):
        assert not has_approved_protocol(text.replace(old, new), "a" * 64)


def checkpoint(iteration=0):
    return {"iteration": iteration, "selectable": True, "transaction_safe": True,
            "bounded": True, "display_normalized_violation": 0.5}


def test_checkpoint_gate_checks_hidden_transactions_with_tolerance():
    base = [checkpoint(i) for i in range(3)]
    candidate = copy.deepcopy(base)
    assert not checkpoint_agreement(base, candidate)
    candidate[1]["display_normalized_violation"] += 5e-7
    assert not checkpoint_agreement(base, candidate)
    candidate[1]["selectable"] = False
    assert checkpoint_agreement(base, candidate)[0]["key"] == "selectable"
    assert checkpoint_agreement(base, candidate[:-1])[0]["kind"] == "trace_length"


def test_speed_gate_exact_threshold_and_missing_or_failed_pair():
    pairs = [{"pass": True, "paired_speedup": 1.1} for _ in range(6)]
    assert speed_gate(pairs)["passes_speed_gate"]
    assert not speed_gate(pairs[:-1])["passes_speed_gate"]
    pairs[0]["pass"] = False
    assert not speed_gate(pairs)["integrity_eligible"]
    pairs = [{"pass": True, "paired_speedup": 1.099999} for _ in range(6)]
    assert not speed_gate(pairs)["passes_speed_gate"]


def make_pair_tree(root):
    mask = np.ones((5, 7), bool)
    basis = np.zeros_like(mask)
    basis[2, 3] = True
    field = lattice_observation_field(mask, basis, np.ones((1, 3), np.float32) * 0.2,
                                      scale_px=1.5, sigma_cutoff=3)
    rows = []
    for repeat in range(6):
        for backend in ("off", "scatter", "csr"):
            identity = f"smooth_s0_r{repeat}_{backend}"
            directory = root / "cells" / identity
            directory.mkdir(parents=True)
            for name in ("field.npz", "input_field.npz"):
                save_observation_field(field, directory / name)
            np.save(directory / "mask.npy", mask)
            np.save(directory / "target.npy", np.full((5, 7, 3), 0.1, np.float32))
            np.save(directory / "reconstruction.npy", np.zeros((5, 7, 3), np.float32))
            write_json(directory / "history.json", {"checkpoints": [checkpoint(i) for i in range(3)]})
            rows.append({"cell_id": identity, "family": "smooth", "seed": 0, "repeat": repeat,
                         "backend": backend, "status": "ok", "selected_iteration": 2,
                         "iterations_run": 2, "cold_render_max_abs": 0,
                         "maintained_parity_max_abs": 0, "cache_bytes": 100,
                         "total_seconds": 1.1 if backend == "off" else 1.0})
    return rows, field


@pytest.mark.parametrize("failure", ["none", "selection", "work", "cold", "sse", "geometry", "missing"])
def test_saved_state_pair_gates(tmp_path, failure):
    from dataclasses import replace

    rows, field = make_pair_tree(tmp_path)
    candidate = next(row for row in rows if row["backend"] == "scatter")
    directory = tmp_path / "cells" / candidate["cell_id"]
    if failure == "selection":
        candidate["selected_iteration"] = 1
    elif failure == "work":
        candidate["iterations_run"] = 1
    elif failure == "cold":
        candidate["cold_render_max_abs"] = 0.01
    elif failure == "sse":
        np.save(directory / "reconstruction.npy", np.full((5, 7, 3), 0.0001, np.float32))
    elif failure == "geometry":
        save_observation_field(replace(field, means_xy=field.means_xy + 0.01), directory / "field.npz")
    elif failure == "missing":
        rows.remove(candidate)
    decision = analyze_bundle(tmp_path, rows)
    record = next(r for r in decision["records"] if r["backend"] == "scatter")
    assert record["passes_speed_gate"] == (failure == "none")


def diagnostic_error_bundle(root):
    sources = ["benchmarks/hier_research_report.py"]
    identities = [f"smooth_s77_r0_{b}" for b in PROTOCOL["backends"]]
    bundle = ResearchBundle(root, task="HIER-034", protocol=PROTOCOL,
                            digest=protocol_digest(PROTOCOL, sources), expected_cells=identities,
                            diagnostic=True, source_paths=sources)
    rows = [{"cell_id": identity, "status": "error", "error": "intentional fixture failure"}
            for identity in identities]
    bundle.finish(rows, title="Report contract fixture", interpretation="Not scientific evidence")
    return bundle


def rehash(root):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"] = {p.relative_to(root).as_posix(): sha256(p)
                         for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    write_json(root / "manifest.json", manifest)


@pytest.mark.parametrize("corruption", ["none", "digest", "matrix", "receipt", "source", "success"])
def test_report_contract_detects_internally_rehashed_corruption(tmp_path, corruption):
    root = tmp_path / "bundle"
    diagnostic_error_bundle(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if corruption == "digest":
        manifest["protocol_digest"] = "a" * 64
    elif corruption == "matrix":
        manifest["expected_cells"] = manifest["expected_cells"][:-1]
    elif corruption == "receipt":
        manifest["command"] = "different command"
    elif corruption == "source":
        (root / "source_snapshot" / "benchmarks" / "hier_research_report.py").write_text("changed")
    elif corruption == "success":
        rows = json.loads((root / "metrics.json").read_text())
        rows[0]["status"] = "ok"
        write_json(root / "metrics.json", rows)
    write_json(root / "manifest.json", manifest)
    rehash(root)
    problems = validate_bundle(root, allow_dirty=True, allow_error_cells=True)
    assert bool(problems) == (corruption != "none"), problems
    assert "intentional fixture failure" in (root / "index.html").read_text()


def test_diagnostic_and_error_cells_require_explicit_flags(tmp_path):
    root = tmp_path / "bundle"
    diagnostic_error_bundle(root)
    assert validate_bundle(root)
    assert validate_bundle(root, allow_dirty=True)
