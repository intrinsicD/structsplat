from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from benchmarks.residual_tangent_auction_stage1_analysis import load_jsonl
from benchmarks.residual_tangent_auction_stage1_core import BirthGeometry
from benchmarks.residual_tangent_auction_stage1_recovery import (
    ACTION_NAMES,
    BASELINE_ACTION,
    EXPECTED_TRAJECTORIES,
    RecoveryContext,
    RecoveryParentKey,
    build_recovery_plan,
    capture_immediate_action_state,
    immediate_state_from_payload,
    packet_for_action,
    run_recovery,
)
from structsplat.gaussians import GaussianField


def _field() -> GaussianField:
    return GaussianField(
        means=torch.tensor([[1.4, 2.0], [3.4, 2.2]], dtype=torch.float32),
        log_scales=torch.log(
            torch.tensor([[1.7, 1.2], [1.4, 1.8]], dtype=torch.float32)
        ),
        rotations=torch.tensor([0.2, -0.3], dtype=torch.float32),
        colors=torch.tensor(
            [[0.2, 0.4, 0.65], [0.8, 0.3, 0.15]], dtype=torch.float32
        ),
    )


def _target_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing="ij")
    target = torch.stack(
        (
            0.15 + 0.12 * xx,
            0.2 + 0.08 * yy,
            0.65 - 0.05 * xx + 0.03 * yy,
        ),
        dim=-1,
    ).to(torch.float32)
    heldout_pixels = (xx + yy) % 2 == 0
    heldout = heldout_pixels.unsqueeze(-1).expand(5, 5, 3).reshape(-1)
    return target, ~heldout, heldout


def _packet(action: str):
    coefficients = torch.tensor(
        [0.03, -0.02, 0.01, -0.01, 0.025, -0.015], dtype=torch.float32
    )
    if action in (BASELINE_ACTION, "tangent6", "full_j"):
        return packet_for_action(action)
    if "affine" in action:
        return packet_for_action(action, coefficients, identity={"parent_row": 0})
    if "carrier" in action:
        return packet_for_action(
            action,
            coefficients,
            identity={"parent_row": 1, "frequency_xy": [0.2, -0.1]},
        )
    return packet_for_action(
        action,
        coefficients,
        identity={
            "geometries": [
                {
                    "mean_xy": [1.0, 1.0],
                    "scale_xy": [0.8, 0.7],
                    "angle_radians": 0.1,
                },
                {
                    "mean_xy": [3.0, 3.0],
                    "scale_xy": [0.7, 0.9],
                    "angle_radians": -0.2,
                },
            ]
        },
    )


@pytest.mark.parametrize(
    "action",
    [BASELINE_ACTION, "tangent6", "full_j", "affine6", "carrier6", "birth2_rgb6"],
)
def test_native_recovery_covers_baseline_tangent_full_and_packets(action: str) -> None:
    parent = _field()
    start = parent.detached()
    if action in ("tangent6", "full_j"):
        start.colors[0, 0] += 0.02
    target, discovery, heldout = _target_masks()
    source = capture_immediate_action_state(
        source_cell_id=f"source:{action}",
        action=action,
        base_field=start,
        packet=_packet(action),
        target=target,
        discovery_mask=discovery,
        heldout_mask=heldout,
        predicted_gain=0.01,
        residual_energy_reduction=0.02,
        render_chunk=16,
    )
    result = run_recovery(
        parent=parent,
        source=source,
        target=target,
        discovery_mask=discovery,
        heldout_mask=heldout,
        context=RecoveryContext(
            binding_sha256="a" * 64,
            family="toy",
            seed=101,
            parent_checkpoint=160,
            fold=0,
            trust_radius=None if action == BASELINE_ACTION else 0.25,
            parent_id="toy-parent",
        ),
        render_chunk=16,
    )

    assert [row["logical_checkpoint"] for row in result.checkpoint_records] == [0, 20, 100]
    assert [row["newly_rendered_recovery_checkpoint"] for row in result.checkpoint_records] == [
        False,
        True,
        True,
    ]
    step0, step20, step100 = result.checkpoint_records
    assert step0["state_sha256"] == source.expected_state_sha256
    assert step0["packet_sha256"] == source.expected_packet_sha256
    assert step0["native_render_sha256"] == source.expected_render_sha256
    assert step0["native_support_sha256"] == source.expected_support_sha256
    assert step20["support_relative_previous_checkpoint"] is not None
    assert step100["support_relative_previous_checkpoint"] is not None
    assert step20["optimizer_steps_completed"] == 20
    assert step100["optimizer_steps_completed"] == 100
    assert result.terminal_record["status"] == "ok"
    assert result.terminal_record["optimizer"]["fresh_state"] is True
    assert result.terminal_record["optimizer"]["betas"] == [0.9, 0.999]
    assert len(result.evidence_records) == (2 if action in ("tangent6", "affine6", "carrier6", "birth2_rgb6") else 0)


def test_step0_verification_fails_before_recovery_on_packet_tamper() -> None:
    parent = _field()
    target, discovery, heldout = _target_masks()
    source = capture_immediate_action_state(
        source_cell_id="source:affine",
        action="affine6",
        base_field=parent,
        packet=_packet("affine6"),
        target=target,
        discovery_mask=discovery,
        heldout_mask=heldout,
    )
    assert source.packet.coefficients is not None
    source.packet.coefficients[0] += 1.0
    with pytest.raises(RuntimeError, match="source state hash mismatch"):
        run_recovery(
            parent=parent,
            source=source,
            target=target,
            discovery_mask=discovery,
            heldout_mask=heldout,
            context=RecoveryContext(
                binding_sha256="b" * 64,
                family="toy",
                seed=101,
                parent_checkpoint=24,
                fold=0,
                trust_radius=0.25,
                parent_id="parent",
            ),
            render_chunk=16,
        )


def test_generic_immediate_payload_loader_reconstructs_dimensionless_step() -> None:
    parent = _field()
    target, discovery, heldout = _target_masks()
    base_step = torch.zeros(parent.n * 8)
    base_step[-1] = 0.02
    extension = torch.tensor([0.01, 0.02, 0.03, -0.01, -0.02, -0.03])
    # Capture once to obtain the source-side hashes that a real immediate artifact must persist.
    from benchmarks.residual_tangent_auction_stage1_core import make_no_opacity_parameterization

    parameterization = make_no_opacity_parameterization(parent, 5, 5)
    updated = parameterization.field_at(base_step)
    packet = packet_for_action("affine6", extension, identity={"parent_row": 0})
    captured = capture_immediate_action_state(
        source_cell_id="source-loader",
        action="affine6",
        base_field=updated,
        packet=packet,
        target=target,
        discovery_mask=discovery,
        heldout_mask=heldout,
    )
    payload = {
        "cell_id": "source-loader",
        "action": "affine6",
        "height": 5,
        "width": 5,
        "base_parameter_step": base_step.tolist(),
        "extension_coefficients": extension.tolist(),
        "identity": {"parent_row": 0},
        "source_hashes": {
            "state_sha256": captured.expected_state_sha256,
            "packet_sha256": captured.expected_packet_sha256,
            "native_render_sha256": captured.expected_render_sha256,
            "native_support_sha256": captured.expected_support_sha256,
            "metric_sha256": captured.expected_metric_sha256,
        },
    }
    loaded = immediate_state_from_payload(parent, payload)
    assert loaded.expected_state_sha256 == captured.expected_state_sha256
    assert torch.equal(loaded.base_field.colors, captured.base_field.colors)
    assert loaded.packet.kind == "affine"


def test_frozen_plan_has_816_unique_trajectories_and_exact_axes() -> None:
    families = (
        "ramp_shading",
        "step_junction",
        "thin_curve",
        "stationary_sinusoid",
        "chirp",
        "natural_texture",
    )
    parents = [
        RecoveryParentKey(family, seed, checkpoint, f"{family}:{seed}:{checkpoint}")
        for family in families
        for seed in (101, 211)
        for checkpoint in (24, 160)
    ]
    links: dict[tuple[str, int, float | None, str], str] = {}
    for parent in parents:
        for fold in (0, 1):
            links[(parent.parent_id, fold, None, BASELINE_ACTION)] = (
                f"baseline:{parent.parent_id}:{fold}"
            )
            for radius in (0.25, 0.75):
                for action in ACTION_NAMES:
                    links[(parent.parent_id, fold, radius, action)] = (
                        f"source:{parent.parent_id}:{fold}:{radius}:{action}"
                    )
    plan = build_recovery_plan(parents, links)
    assert len(plan) == EXPECTED_TRAJECTORIES == 816
    assert len({entry.trajectory_id for entry in plan}) == 816
    assert sum(entry.action == BASELINE_ACTION for entry in plan) == 48
    assert sum(entry.action != BASELINE_ACTION for entry in plan) == 768


def test_matched_evidence_rows_are_strict_analyzer_compatible(tmp_path) -> None:
    parent = _field()
    target, discovery, heldout = _target_masks()
    source = capture_immediate_action_state(
        source_cell_id="source:tangent",
        action="tangent6",
        base_field=parent,
        packet=_packet("tangent6"),
        target=target,
        discovery_mask=discovery,
        heldout_mask=heldout,
    )
    result = run_recovery(
        parent=parent,
        source=source,
        target=target,
        discovery_mask=discovery,
        heldout_mask=heldout,
        context=RecoveryContext(
            binding_sha256="c" * 64,
            family="toy",
            seed=101,
            parent_checkpoint=160,
            fold=0,
            trust_radius=0.25,
            parent_id="toy-parent",
        ),
        render_chunk=16,
    )
    path = tmp_path / "evidence.jsonl"
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in result.evidence_records),
        encoding="utf-8",
    )
    loaded = load_jsonl(
        path,
        expected_ids={row["cell_id"] for row in result.evidence_records},
        expected_binding_sha256="c" * 64,
    )
    assert loaded.status == "available"
    assert loaded.observed_count == 2


def test_birth_geometry_is_fixed_in_packet_protocol() -> None:
    packet = _packet("birth2_rgb6")
    assert packet.birth_geometries == (
        BirthGeometry((1.0, 1.0), (0.8, 0.7), 0.1),
        BirthGeometry((3.0, 3.0), (0.7, 0.9), -0.2),
    )
    with pytest.raises(ValueError, match="action and packet kind disagree"):
        replace(
            capture_immediate_action_state(
                source_cell_id="bad",
                action="birth2_rgb6",
                base_field=_field(),
                packet=packet,
                target=_target_masks()[0],
                discovery_mask=_target_masks()[1],
                heldout_mask=_target_masks()[2],
            ),
            packet=packet_for_action("tangent6"),
        ).validate(_field())
