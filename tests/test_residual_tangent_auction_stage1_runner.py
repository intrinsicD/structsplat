# ruff: noqa: E402
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks import residual_tangent_auction_stage1_runner as R
from benchmarks.residual_tangent_auction_stage1_core import (
    affine_six_columns,
    checkerboard_fold_masks,
    packet_linear_realization,
    realize_carrier_six,
    render_normalized,
)
from structsplat.gaussians import GaussianField
from structsplat.fit import fit
from structsplat.init import build_field
from structsplat.config import StructureTensorConfig


def _write_fake_preflight(root, *, passing=True, development=False):
    root.mkdir(parents=True, exist_ok=True)
    protocol = "toy Stage-1 preflight"
    repo_root = Path(R.__file__).resolve().parents[1]
    critical_sources = {
        relative: {"sha256": hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()}
        for relative in R.manifest_module.CRITICAL_SOURCE_PATHS
    }
    config = {
        "protocol": protocol,
        "development_run_authorized": False,
        "provenance": {"critical_sources": critical_sources},
    }
    manifest = {
        "protocol": protocol,
        "target_manifest_sha256": "1" * 64,
        "development_run_authorized": False,
        "scientific_cells_present": False,
    }
    config_path = root / "config.json"
    manifest_path = root / "manifest.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    preflight = {
        "protocol": protocol,
        "preflight_pass": passing,
        "development_run_authorized": development,
        "scientific_cells_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (root / "preflight.json").write_text(json.dumps(preflight, sort_keys=True) + "\n")
    return root


def _toy_target(size=64):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    u = (xx + 0.5) / float(size)
    v = (yy + 0.5) / float(size)
    return np.stack(
        (
            0.12 + 0.68 * u,
            0.16 + 0.62 * v,
            0.72 - 0.31 * u + 0.08 * np.sin(2.0 * np.pi * v),
        ),
        axis=2,
    ).astype(np.float32)


def _toy_field(size=32):
    return GaussianField.from_numpy(
        means=np.asarray(
            [[6.2, 6.4], [24.1, 7.3], [8.5, 23.2], [23.0, 24.4]], dtype=np.float64
        ),
        scales=np.asarray(
            [[5.1, 4.2], [4.4, 5.0], [5.2, 4.0], [4.6, 5.3]], dtype=np.float64
        ),
        angles=np.asarray([0.18, -0.42, 0.63, -0.71], dtype=np.float64),
        colors=np.asarray(
            [
                [0.18, 0.52, 0.74],
                [0.79, 0.27, 0.39],
                [0.31, 0.72, 0.25],
                [0.66, 0.61, 0.16],
            ],
            dtype=np.float64,
        ),
        dtype=torch.float64,
    )


def test_carrier_banks_and_provisional_fields_are_frozen_and_distinct():
    assert len(R.SMALL_CARRIER_BANK) == 2 * 4 == 8
    assert len(R.FULL_CARRIER_BANK) == 4 * 8 == 32
    assert [item.index for item in R.SMALL_CARRIER_BANK] == list(range(8))
    assert [item.index for item in R.FULL_CARRIER_BANK] == list(range(32))
    assert {item.magnitude for item in R.SMALL_CARRIER_BANK} == {0.18, 0.32}
    assert {item.orientation_degrees for item in R.FULL_CARRIER_BANK} == {
        22.5 * index for index in range(8)
    }
    first = R.SMALL_CARRIER_BANK[0]
    assert first.frequency_xy == pytest.approx((0.18, 0.0))
    manifest = R.carrier_bank_manifest()
    assert manifest["small"]["coefficient_basis"] == "sin_cos_rgb"
    assert manifest["small"]["optimized_dofs"] == 6
    assert manifest["small"]["provisional_packet_bytes"] == 30
    assert sum(item["width_bytes"] for item in R.TANGENT_SELECTOR_FIELDS) == 37
    assert sum(item["width_bytes"] for item in R.AFFINE_SELECTOR_FIELDS) == 27
    assert sum(item["width_bytes"] for item in R.BIRTH_SELECTOR_FIELDS) == 65
    assert manifest["small"]["price_scope"] == R.PRICE_SCOPE
    assert len(manifest["small"]["bank_sha256"]) == 64
    assert manifest["small"]["bank_sha256"] != manifest["full"]["bank_sha256"]


def test_preflight_validation_requires_pass_and_preserves_development_false(tmp_path):
    good = _write_fake_preflight(tmp_path / "good")
    record = R.validate_preflight_artifact(good)
    assert record["preflight_pass"] is True
    assert record["development_run_authorized"] is False
    assert record["scientific_cells_present"] is False

    bad_pass = _write_fake_preflight(tmp_path / "bad_pass", passing=False)
    with pytest.raises(RuntimeError, match="preflight_pass"):
        R.validate_preflight_artifact(bad_pass)
    bad_development = _write_fake_preflight(
        tmp_path / "bad_development", passing=True, development=True
    )
    with pytest.raises(RuntimeError, match="preflight_development_false"):
        R.validate_preflight_artifact(bad_development)

    stale = _write_fake_preflight(tmp_path / "stale")
    stale_config_path = stale / "config.json"
    stale_config = json.loads(stale_config_path.read_text())
    stale_config["provenance"]["critical_sources"][
        "benchmarks/residual_tangent_auction_stage1_runner.py"
    ]["sha256"] = "0" * 64
    stale_config_path.write_text(json.dumps(stale_config, sort_keys=True) + "\n")
    stale_preflight_path = stale / "preflight.json"
    stale_preflight = json.loads(stale_preflight_path.read_text())
    stale_preflight["config_sha256"] = hashlib.sha256(
        stale_config_path.read_bytes()
    ).hexdigest()
    stale_preflight_path.write_text(json.dumps(stale_preflight, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="critical_source_hashes_live"):
        R.validate_preflight_artifact(stale)


def test_parent_protocol_is_one_trajectory_and_matches_ordinary_fit_at_both_steps(tmp_path):
    image = _toy_target()
    target_record = {
        "target_id": "toy_generated_not_frozen",
        "family": "toy",
        "image": image,
        "pixel_sha256": R._tensor_sha256(image),
    }
    field_paths = {24: tmp_path / "toy_24.npz", 160: tmp_path / "toy_160.npz"}
    trajectory, records = R.fit_parent_trajectory(
        target_record,
        seed=101,
        field_paths=field_paths,
        binding_sha256="a" * 64,
    )
    assert trajectory["checkpoint_steps"] == [24, 160]
    assert trajectory["optimizer_state"] == "continuous_steps_1_through_160"
    assert len(records) == 2
    assert {record["checkpoint_step"] for record in records} == {24, 160}
    assert len({record["trajectory_id"] for record in records}) == 1

    init_config, fit_config = R.resolved_parent_configs(101)
    torch.manual_seed(101)
    initial = build_field(
        image,
        init_config,
        StructureTensorConfig(),
        device="cpu",
    )
    target = torch.as_tensor(image)
    ordinary = {}
    for step in (24, 160):
        result = fit(
            initial.detached(),
            target,
            replace(fit_config, iters=step, log_every=step),
            verbose=False,
        )
        ordinary[step] = R.field_sha256(result["field"].detached())
    for record in records:
        step = record["checkpoint_step"]
        assert field_paths[step].is_file()
        assert record["field_sha256"] == ordinary[step]
        assert record["independent_horizon_run"] is False
        assert record["optimizer_state_from_step1_continuous"] is True
        assert record["n_gaussians"] == 64
        assert record["opacity_present"] is False
        assert record["color_grads_present"] is False
        assert record["topology_events"] == "disabled"
        assert record["fit_config"]["iters"] == 160
        assert math.isfinite(record["mse"]) and record["mse"] >= 0.0


def test_affine_and_carrier_selectors_fit_discovery_and_survive_heldout_perturbation():
    field = _toy_field()
    height = width = 32
    base = render_normalized(field, height, width)
    affine_coefficients = torch.tensor(
        [0.035, -0.021, 0.016, -0.018, 0.027, 0.013], dtype=torch.float64
    )
    affine_design = affine_six_columns(field, 1, height, width)
    affine_target = packet_linear_realization(base, affine_design, affine_coefficients)
    affine = R.select_affine_candidate(field, affine_target, 0)
    assert affine.identity == {"parent_row": 1}
    assert affine.candidate_count == field.n
    assert affine.optimized_dofs == 6
    assert affine.provisional_packet_bytes == 27
    assert len(affine.discovery_input_sha256) == 64
    assert affine.numerical_repeatability_tolerance == 1e-12
    assert affine.discovery_post_mse <= 1e-25

    frequency = R.SMALL_CARRIER_BANK[3]
    carrier_coefficients = torch.tensor(
        [0.027, -0.019, 0.013, -0.022, 0.017, 0.025], dtype=torch.float64
    )
    carrier_target = realize_carrier_six(
        field,
        2,
        frequency.frequency_xy,
        carrier_coefficients,
        height,
        width,
    )
    carrier = R.select_carrier_candidate(
        field, carrier_target, 1, bank_name="small"
    )
    assert carrier.identity["parent_row"] == 2
    assert carrier.identity["frequency_index"] == frequency.index
    assert carrier.identity["bank_name"] == "small"
    assert carrier.candidate_count == field.n * len(R.SMALL_CARRIER_BANK)
    assert carrier.provisional_packet_bytes == 30
    assert len(carrier.discovery_input_sha256) == 64
    assert carrier.discovery_post_mse <= 1e-25


def test_two_birth_selector_uses_unordered_discovery_geometry_pairs_only():
    field = _toy_field()
    size = 32
    base = render_normalized(field, size, size)
    target = base.clone()
    # Peaks on both checkerboard parities produce a deterministic discovery-only NMS bank.
    for y, x, color in (
        (2, 2, (0.8, -0.3, 0.4)),
        (2, 14, (-0.5, 0.7, 0.2)),
        (2, 26, (0.4, 0.2, -0.6)),
        (14, 2, (0.6, -0.2, 0.5)),
        (14, 14, (-0.4, 0.6, 0.3)),
        (14, 26, (0.5, 0.4, -0.3)),
        (26, 2, (0.7, 0.1, -0.4)),
        (26, 26, (-0.3, 0.5, 0.6)),
    ):
        target[y, x] += torch.tensor(color, dtype=torch.float64)
    selection = R.select_two_birth_candidate(field, target, 0)
    pair = selection.identity["unordered_site_pair"]
    assert pair == sorted(pair)
    assert pair[0] < pair[1]
    assert selection.identity["geometry_source"] == "discovery_rgb_squared_residual_nms"
    assert selection.identity["geometry_bank_site_count"] == 8
    assert selection.candidate_count == math.comb(8, 2)
    assert selection.optimized_dofs == 6
    assert selection.provisional_packet_bytes == 65
    assert len(selection.discovery_input_sha256) == 64


def test_leakage_guard_fails_closed_if_selector_reads_heldout():
    field = _toy_field()
    target = render_normalized(field, 32, 32)
    original = R.select_affine_candidate(field, target, 0, verify_leakage=False)
    _discovery, heldout = checkerboard_fold_masks(32, 32, 0)

    def malicious(changed_target, _verify):
        checksum = float(torch.as_tensor(changed_target).reshape(-1)[heldout].sum())
        return replace(original, identity={"heldout_checksum": checksum})

    with pytest.raises(RuntimeError, match="held-out perturbation changed"):
        R._verify_heldout_invariance(malicious, target, heldout, original)

    def malicious_hash(_changed_target, _verify):
        return replace(original, discovery_input_sha256="0" * 64)

    with pytest.raises(RuntimeError, match="discovery_hash_changed=True"):
        R._verify_heldout_invariance(malicious_hash, target, heldout, original)


def test_jsonl_resume_reader_rejects_duplicates_and_cli_refuses_missing_preflight(tmp_path):
    path = tmp_path / "records.jsonl"
    R._append_jsonl(path, {"record_id": "one", "value": 1})
    assert R._load_jsonl(path, "record_id")["one"]["value"] == 1
    R._append_jsonl(path, {"record_id": "one", "value": 2})
    with pytest.raises(ValueError, match="duplicate record_id"):
        R._load_jsonl(path, "record_id")

    with pytest.raises(FileNotFoundError, match="preflight artifacts"):
        R.main(
            [
                "--phase",
                "parents",
                "--preflight-dir",
                str(tmp_path / "missing"),
                "--outdir",
                str(tmp_path / "out"),
                "--max-parents",
                "1",
            ]
        )
