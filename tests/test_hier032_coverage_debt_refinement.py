from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import check_report_bundle as report_checker
from scripts.experiments import hier032_coverage_debt_refinement as h32
from structsplat.gaussians import GaussianField


def test_component_detection_is_deterministic_and_eight_connected():
    weak = np.zeros((6, 7), dtype=bool)
    weak[1, 1] = True
    weak[2, 2] = True  # diagonal joins under the frozen 8-connectivity rule
    weak[4, 5] = True

    labels_a, records_a = h32._label_components8(weak)
    labels_b, records_b = h32._label_components8(weak.copy())

    assert np.array_equal(labels_a, labels_b)
    assert records_a == records_b
    assert [record["pixels"] for record in records_a] == [2, 1]
    assert records_a[0]["first_flat"] < records_a[1]["first_flat"]


def test_fallback_bank_guarantees_each_weak_pixel_is_individually_satisfied():
    inside = np.zeros((9, 9), dtype=bool)
    inside[1:8, 1:8] = True
    coverage = np.full((9, 9), 0.2, dtype=np.float32)
    coverage[3, 3] = 0.01
    coverage[5, 6] = 0.0

    bank = h32._fallback_candidates(coverage, inside)

    assert len(bank.candidates) == 2
    assert len(bank.weak_flats) == 2
    for index, candidate in enumerate(bank.candidates):
        assert candidate.kind == "fallback"
        assert candidate.weak_indices == (index,)
        assert candidate.weights[0] > h32.COVERAGE_THRESHOLD
        assert candidate.weights[0] >= bank.deficits[index]


def test_greedy_cover_uses_new_satisfaction_before_mass_and_is_complete():
    candidates = (
        h32.Candidate(
            0,
            1,
            "mass_only",
            0,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            appearance_variance=0.0,
            weak_indices=(0, 1),
            weights=(0.02, 0.02),
        ),
        h32.Candidate(
            1,
            1,
            "satisfies_one",
            0,
            0.0,
            0.0,
            0.08,
            0.08,
            0.0,
            appearance_variance=10.0,
            weak_indices=(0,),
            weights=(0.04,),
        ),
        h32.Candidate(
            2,
            1,
            "fallback",
            1,
            1.0,
            0.0,
            0.08,
            0.08,
            0.0,
            appearance_variance=10.0,
            weak_indices=(1,),
            weights=(0.04,),
        ),
    )
    bank = h32.CandidateBank(
        candidates=candidates,
        weak_flats=np.asarray([0, 1], dtype=np.int64),
        deficits=np.asarray([0.04, 0.04]),
        labels=np.ones((1, 2), dtype=np.int32),
        component_records=({"label": 1, "pixels": 2},),
        detector_seconds=0.0,
        incidence_seconds=0.0,
        incidence_edges=4,
    )

    selected, record = h32._greedy_cover(bank)

    assert selected[0].candidate_id == 1
    assert [candidate.candidate_id for candidate in selected] == [1, 2]
    assert record["complete"] is True
    assert record["selected_count"] == 2


def test_station_ball_tangent_candidate_has_no_outside_pixel_support():
    inside = np.zeros((31, 31), dtype=bool)
    inside[3:28, 3:28] = True
    means, scales, angles, valid = h32._certify_tangent_geometries(
        inside,
        np.asarray([[15.0, 5.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    assert valid.tolist() == [True]
    assert scales[0, 0] > scales[0, 1]
    yy, xx = np.mgrid[: inside.shape[0], : inside.shape[1]]
    weights = h32._row_weights(means[0], scales[0], float(angles[0]), xx, yy)
    assert float(weights[~inside].max(initial=0.0)) == 0.0


def test_local_least_squares_merge_ranks_reconstructable_pair_first():
    inside = np.ones((21, 21), dtype=bool)
    yy, xx = np.mgrid[:21, :21]
    color = np.asarray([0.2, 0.4, 0.6], dtype=np.float64)
    scales = np.asarray([1.0, 1.0])

    good_w = h32._row_weights(np.asarray([10.0, 10.0]), scales, 0.0, xx, yy)
    good_target = 2.0 * good_w[..., None] * color
    good = h32._local_merge_fit(
        good_target,
        good_target,
        inside,
        (np.asarray([10.0, 10.0]), scales, 0.0, color),
        (np.asarray([10.0, 10.0]), scales, 0.0, color),
        (np.asarray([10.0, 10.0]), scales, 0.0, color),
    )

    wa = h32._row_weights(np.asarray([6.0, 10.0]), scales, 0.0, xx, yy)
    wb = h32._row_weights(np.asarray([14.0, 10.0]), scales, 0.0, xx, yy)
    bad_target = (wa + wb)[..., None] * color
    bad = h32._local_merge_fit(
        bad_target,
        bad_target,
        inside,
        (np.asarray([6.0, 10.0]), scales, 0.0, color),
        (np.asarray([14.0, 10.0]), scales, 0.0, color),
        (np.asarray([10.0, 10.0]), np.asarray([4.0, 1.0]), 0.0, color),
    )
    order = h32._contribution_order(
        {
            "merge_sse": np.asarray([bad["merge_sse"], good["merge_sse"]]),
            "delta_sse": np.asarray([bad["delta_sse"], good["delta_sse"]]),
        }
    )

    assert good["merge_sse"] == pytest.approx(0.0, abs=1e-12)
    assert bad["merge_sse"] > good["merge_sse"]
    assert order.tolist() == [1, 0]


def test_funding_assembly_preserves_exact_count(monkeypatch):
    monkeypatch.setattr(h32, "CAPACITY", 4)
    inside = np.zeros((20, 20), dtype=bool)
    inside[2:18, 2:18] = True
    field = GaussianField.from_numpy(
        np.asarray([[6, 6], [7, 6], [12, 12], [14, 14]], dtype=np.float32),
        np.full((4, 2), 0.5, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.zeros((4, 3), dtype=np.float32),
    )
    pair_data = {
        "first": torch.tensor([0]),
        "second": torch.tensor([1]),
        "means": torch.tensor([[6.5, 6.0]]),
        "scales": torch.tensor([[0.5, 0.5]]),
        "rotations": torch.tensor([0.0]),
    }
    placement = h32.Candidate(
        0, 1, "fallback", 8 * 20 + 8, 8.0, 8.0, 0.08, 0.08, 0.0
    )
    monkeypatch.setattr(
        h32,
        "_render",
        lambda _field, shape, _args, _torch: torch.zeros((*shape, 3)),
    )

    class Projection:
        selected_iteration = 0
        initial_sse = 0.0
        final_sse = 0.0
        forward_applications = 0
        transpose_applications = 0
        relative_normal_residual_max = 0.0
        adjoint_relative_error = 0.0
        maintained_render_parity_max_abs = 0.0
        elapsed_seconds = 0.0

        @staticmethod
        def checkpoint_records():
            return []

    monkeypatch.setattr(
        h32,
        "project_additive_endpoint",
        lambda proposal, *_args, **_kwargs: SimpleNamespace(
            field=proposal, projection=Projection()
        ),
    )
    args = SimpleNamespace(render_chunk=16, device="cpu")

    output, record = h32._assemble_funded_field(
        field,
        pair_data,
        np.asarray([0]),
        np.zeros((1, 3), dtype=np.float32),
        [placement],
        np.zeros((20, 20, 3), dtype=np.float32),
        inside,
        args,
        torch,
    )

    assert output.n == 4
    assert len(record["absorbed_rows"]) == 1
    assert output.means[-1].tolist() == pytest.approx([8.0, 8.0])


def _row(arm: str, **overrides):
    row = {
        "arm": arm,
        "n_gaussians": 7_000,
        "four_array_endpoint_exact": True,
        "raw_hole_pixels": 0,
        "coverage_lt_005_pixels": 0,
        "unit_coverage_outside_abs_max": 0.0,
        "reconstruction_outside_abs_max": 0.0,
        "maintained_render_parity_max_abs": 0.0,
        "boundary_le4_psnr_db": 21.0,
        "hair_psnr_db": 22.0,
        "interior_gt4_psnr_db": 35.4,
        "psnr_db": 24.0,
    }
    row.update(overrides)
    return row


def test_decision_gate_selects_only_complete_acceptance():
    control = _row(
        h32.ARMS[0],
        coverage_lt_005_pixels=743,
        boundary_le4_psnr_db=20.0,
        hair_psnr_db=20.0,
    )
    candidates = [_row(arm, psnr_db=24.0 - 0.1 * index) for index, arm in enumerate(h32.ARMS[1:])]
    attempts = [{"arm": arm, "status": "ok"} for arm in h32.ARMS]

    accepted = h32._decision([control, *candidates], attempts)
    rejected_rows = [
        control.copy(),
        *[
            _row(
                arm,
                psnr_db=24.0 - 0.1 * index,
                hair_psnr_db=19.0,
            )
            for index, arm in enumerate(h32.ARMS[1:])
        ],
    ]
    rejected = h32._decision(
        rejected_rows, attempts
    )

    assert accepted["selected_arm"] == h32.ARMS[1]
    assert candidates[0]["acceptance_pass"] is True
    assert rejected["selected_arm"] is None
    assert rejected["selected_method"] is False


def test_decision_fails_closed_when_any_frozen_arm_errors():
    control = _row(
        h32.ARMS[0],
        coverage_lt_005_pixels=743,
        boundary_le4_psnr_db=20.0,
        hair_psnr_db=20.0,
    )
    rows = [control] + [_row(arm) for arm in h32.ARMS[1:]]
    attempts = [{"arm": arm, "status": "ok"} for arm in h32.ARMS]
    attempts[2]["status"] = "error"

    decision = h32._decision(rows, attempts)

    assert decision["complete"] is False
    assert decision["all_arms_succeeded"] is False
    assert decision["selected_arm"] is None
    assert decision["selected_method"] is False
    assert decision["selection_reason"] == "matrix incomplete; no selection authorized"


def test_field_state_hash_covers_decoded_four_array_payload(tmp_path):
    field = GaussianField.from_numpy(
        np.asarray([[2.0, 3.0]], dtype=np.float32),
        np.asarray([[0.5, 0.75]], dtype=np.float32),
        np.asarray([0.25], dtype=np.float32),
        np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
    )
    identical = field.detached()
    changed = field.detached()
    changed.colors[0, 0] += 1e-4

    assert h32._field_state_sha256(field) == h32._field_state_sha256(identical)
    assert h32._field_state_max_abs(field, identical) == 0.0
    assert h32._field_state_sha256(field) != h32._field_state_sha256(changed)
    assert h32._field_state_max_abs(field, changed) == pytest.approx(1e-4, abs=1e-8)
    path = tmp_path / "field.gaussian.npz"
    field.save(str(path))
    assert report_checker._gaussian_four_array_state_sha256(path) == h32._field_state_sha256(
        field
    )


def test_report_checker_recomputes_gates_from_raw_metrics():
    control = _row(
        h32.ARMS[0],
        coverage_lt_005_pixels=743,
        boundary_le4_psnr_db=20.0,
        hair_psnr_db=20.0,
    )
    candidate = _row(h32.ARMS[2], hair_psnr_db=19.0)

    gates = report_checker._hier032_expected_gates(candidate, control)

    assert gates["hair_improved"] is False
    assert all(value for key, value in gates.items() if key != "hair_improved")


def test_protocol_digest_binds_frozen_execution_contract():
    encoded = json.dumps(
        h32.PROTOCOL, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()

    assert digest == h32.FROZEN_PROTOCOL_DIGEST
    assert digest == report_checker.HIER032_PROTOCOL_DIGEST
    assert h32.PROTOCOL["environment"]["gpu_name"] == h32.EXPECTED_GPU_NAME
    assert h32.PROTOCOL["decision"]["report_gate_requires_all_five_arms_ok"] is True
    assert h32.PROTOCOL["representation"]["decoded_field_hash_required"] is True


def test_report_schema_is_registered_with_the_bundle_checker():
    assert h32.REPORT_SCHEMA == report_checker.HIER032_COVERAGE_DEBT_REFINEMENT_SCHEMA
    assert h32.REPORT_SCHEMA in report_checker.HIER015_PLUS_REPORT_SCHEMAS
    assert h32.ARMS == tuple(h32.PROTOCOL["arms"])
