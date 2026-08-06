from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

import structsplat.artifact_first_quadtree as hierarchy
from structsplat.artifact_first_quadtree import (
    ArtifactFirstQuadtreeConfig,
    build_artifact_first_quadtree,
    frontier_partition_valid,
    initialize_artifact_first_quadtree,
    rank_frontier_splits,
    support_overlapping_keys,
)
from structsplat.observation_field import ObservationField2D
from structsplat.pixel_contraction import render_observation_field
from structsplat.progressive_residual_quadtree import _base_keys, _child_keys


ROOT = Path(__file__).resolve().parents[1]


def _image(height: int = 8, width: int = 9, seed: int = 13) -> np.ndarray:
    return np.random.default_rng(seed).random((height, width, 3), dtype=np.float32)


def _small_config(**kwargs) -> ArtifactFirstQuadtreeConfig:
    values = {
        "start_level": 2,
        "max_gaussians": 20,
        "max_child_rows_per_stage": 8,
        "base_steps": 2,
        "layer_steps": 2,
        "checkpoint_every": 1,
        "learning_rate": 0.02,
        "error_smoothing_sigma_px": 0.5,
        "device": "cpu",
        "renderer": "additive",
        "render_chunk": 16,
        "milestone_counts": (12,),
        "max_stages": 2,
    }
    values.update(kwargs)
    return ArtifactFirstQuadtreeConfig(**values)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"selection_mode": "bad"}, ValueError),
        ({"reconciliation_scope": "global"}, ValueError),
        ({"overlap_margin_px": -1}, ValueError),
        ({"error_weight_floor": 1.1}, ValueError),
        ({"error_weight_ceiling": 0.9}, ValueError),
        ({"max_child_rows_per_stage": 0}, ValueError),
        ({"renderer": "cuda_additive", "device": "cpu"}, ValueError),
        ({"milestone_counts": [4]}, TypeError),
    ],
)
def test_config_fails_closed(kwargs, error):
    with pytest.raises(error):
        ArtifactFirstQuadtreeConfig(**kwargs)


def test_frontier_parent_replacement_preserves_antichain_partition():
    mask = np.ones((7, 9), dtype=bool)
    base = _base_keys(mask, 2)
    assert frontier_partition_valid(mask, base)
    parent = base[0]
    children = _child_keys(mask, parent)
    replaced = tuple(key for key in base if key != parent) + children
    assert frontier_partition_valid(mask, replaced)
    assert len(replaced) == len(base) - 1 + len(children)
    assert not frontier_partition_valid(mask, base + children)


def test_artifact_first_prioritizes_isolated_peak_while_energy_prioritizes_extent():
    mask = np.ones((4, 8), dtype=bool)
    target = np.zeros((4, 8, 3), dtype=np.float32)
    reconstruction = np.zeros_like(target)
    reconstruction[0, 0] = 1.0
    reconstruction[:, 4:] = 0.4
    frontier = ((2, 0, 0), (2, 0, 1))
    common = {
        "start_level": 2,
        "max_gaussians": 16,
        "error_smoothing_sigma_px": 0.0,
        "pixel_rmse_threshold": 0.02,
        "patch7_rmse_threshold": 0.01,
        "milestone_counts": (),
    }
    artifact = rank_frontier_splits(
        frontier,
        set(),
        mask,
        reconstruction,
        target,
        ArtifactFirstQuadtreeConfig(selection_mode="artifact_first", **common),
    )
    energy = rank_frontier_splits(
        frontier,
        set(),
        mask,
        reconstruction,
        target,
        ArtifactFirstQuadtreeConfig(selection_mode="energy", **common),
    )
    assert artifact[0].parent_key == (2, 0, 0)
    assert energy[0].parent_key == (2, 0, 1)
    assert artifact[0].artifact_score > artifact[1].artifact_score
    assert energy[0].energy_score > energy[1].energy_score


def test_support_overlap_includes_near_neighbor_and_excludes_far_row():
    mask = np.ones((4, 32), dtype=bool)
    selected = ((2, 0, 0),)
    active = ((2, 0, 0), (2, 0, 1), (2, 0, 6))
    overlap = support_overlapping_keys(
        mask,
        active,
        selected,
        leaf_scale_px=0.18,
        sigma_cutoff=3.0,
        margin_px=1,
    )
    assert (2, 0, 1) in overlap
    assert (2, 0, 6) not in overlap
    assert selected[0] not in overlap


def test_shared_base_is_axis_independent_and_cpu_build_is_deterministic():
    image = _image()
    mask = np.ones(image.shape[:2], dtype=bool)
    artifact_config = _small_config(
        selection_mode="artifact_first",
        reconciliation_scope="overlap",
    )
    energy_config = _small_config(
        selection_mode="energy",
        reconciliation_scope="new_only",
    )
    base = initialize_artifact_first_quadtree(image, artifact_config, mask=mask)
    first = build_artifact_first_quadtree(
        image, artifact_config, mask=mask, start_state=base
    )
    second = build_artifact_first_quadtree(
        image, artifact_config, mask=mask, start_state=base
    )
    control = build_artifact_first_quadtree(
        image, energy_config, mask=mask, start_state=base
    )
    assert first.stages[0] == control.stages[0]
    assert first.field.canonical_hash() == second.field.canonical_hash()
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    assert [stage.status for stage in first.stages] == [
        stage.status for stage in second.stages
    ]


def test_builder_replaces_complete_groups_freezes_nonlocal_rows_and_respects_cap():
    image = _image(9, 11, seed=19)
    mask = np.ones(image.shape[:2], dtype=bool)
    config = _small_config(max_gaussians=19, max_stages=3)
    result = build_artifact_first_quadtree(image, config, mask=mask)
    assert result.final_count <= config.max_gaussians
    assert frontier_partition_valid(mask, result.active_keys)
    assert result.stored_node_count - result.final_count == result.inactive_node_count
    assert result.coefficient_event_rows >= result.stored_node_count
    for stage in result.stages[1:]:
        assert stage.proposed_child_rows == len(stage.child_keys)
        assert stage.proposed_net_rows == len(stage.child_keys) - len(stage.parent_keys)
        assert stage.proposed_child_rows <= config.max_child_rows_per_stage
        assert stage.frontier_partition_valid
        if stage.status == "accepted":
            assert stage.count_after == stage.count_before + stage.proposed_net_rows
            assert stage.accepted_net_rows == stage.proposed_net_rows
            assert stage.untouched_coefficients_bit_exact
        else:
            assert stage.count_after == stage.count_before
            assert stage.rollback_bit_exact


def test_rejected_trials_roll_back_field_topology_and_coefficients(monkeypatch):
    image = _image(7, 8, seed=23)
    mask = np.ones(image.shape[:2], dtype=bool)
    config = _small_config(max_gaussians=16, max_stages=1)
    base = initialize_artifact_first_quadtree(image, config, mask=mask)

    def reject(**kwargs):
        count = int(kwargs["initial_colors"].shape[0])
        return hierarchy._ColorOptimization(
            colors=None,
            reconstruction_raw=None,
            selected_step=-1,
            selected_metrics=(0.0, 0.0, 0.0, 0.0),
            checkpoints=(),
            elapsed_seconds=0.0,
            attribution_seconds=0.0,
            attempted_steps=0,
            weight_telemetry=hierarchy._unit_weight_telemetry(count),
        )

    monkeypatch.setattr(hierarchy, "_optimize_color_block", reject)
    result = build_artifact_first_quadtree(
        image, config, mask=mask, start_state=base
    )
    assert result.field.canonical_hash() == base.field.canonical_hash()
    assert np.array_equal(result.reconstruction_raw, base.reconstruction_raw)
    rejected = result.stages[1:]
    assert rejected
    assert all(stage.status.startswith("rolled_back") for stage in rejected)
    assert all(stage.rollback_bit_exact for stage in rejected)
    assert result.accepted_split_count == 0


def test_start_state_rejects_different_image_or_base_configuration():
    image = _image(seed=31)
    mask = np.ones(image.shape[:2], dtype=bool)
    config = _small_config()
    base = initialize_artifact_first_quadtree(image, config, mask=mask)
    changed = image.copy()
    changed[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="different executed image"):
        build_artifact_first_quadtree(changed, config, mask=mask, start_state=base)
    with pytest.raises(ValueError, match="base configuration"):
        build_artifact_first_quadtree(
            image,
            _small_config(learning_rate=0.03),
            mask=mask,
            start_state=base,
        )


def test_snapshot_roundtrip_and_cold_render_parity(tmp_path: Path):
    image = _image(7, 9, seed=37)
    mask = np.ones(image.shape[:2], dtype=bool)
    mask[:2, :2] = False
    result = build_artifact_first_quadtree(image, _small_config(), mask=mask)
    for index, snapshot in enumerate(result.snapshots):
        path = tmp_path / f"snapshot_{index}.npz"
        snapshot.field.save_lossless(path)
        loaded = ObservationField2D.load_lossless(path)
        assert loaded.canonical_hash() == snapshot.field.canonical_hash()
        cold = render_observation_field(
            loaded,
            device="cpu",
            renderer="additive",
            render_chunk=16,
            apply_declared_alpha=False,
        )
        assert np.max(np.abs(cold - snapshot.reconstruction_raw)) < 2e-6


def test_module_import_does_not_require_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "import structsplat.artifact_first_quadtree; print('ok')"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_four_arm_report_writes_portable_visual_bundle(tmp_path: Path):
    height, width = 24, 32
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        [
            xx / (width - 1),
            yy / (height - 1),
            ((xx // 3 + yy // 3) % 2).astype(np.float64),
        ],
        axis=2,
    )
    mask = ((xx - 16) / 13) ** 2 + ((yy - 12) / 9) ** 2 <= 1.0
    image_path = tmp_path / "tiny.jpg"
    mask_path = tmp_path / "tiny_mask.png"
    Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB").save(image_path)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    output_root = tmp_path / "report"
    diagnostic_context = tmp_path / "diagnostic-context"
    diagnostic_context.mkdir()
    (diagnostic_context / "metrics.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "status": "diagnostic",
                        "method": "context_control",
                        "variant": "n16",
                        "n_gaussians": 16,
                        "psnr_db": 30.0,
                        "ms_ssim": 0.9,
                        "lpips": 0.1,
                        "artifact_pixel_rmse_max": 0.2,
                        "artifact_patch_rmse_max_7": 0.1,
                        "artifact_gate_pass": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    script = ROOT / "scripts" / "experiments" / "hier007_artifact_first_quadtree.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--images",
            str(image_path),
            "--mask",
            str(mask_path),
            "--out",
            str(output_root),
            "--max-side",
            "32",
            "--start-level",
            "3",
            "--max-gaussians",
            "20",
            "--max-child-rows-per-stage",
            "8",
            "--base-steps",
            "2",
            "--layer-steps",
            "2",
            "--checkpoint-every",
            "1",
            "--max-stages",
            "1",
            "--device",
            "cpu",
            "--renderer",
            "additive",
            "--render-chunk",
            "16",
            "--milestone-counts",
            "16",
            "--hier005-report",
            str(diagnostic_context),
            "--hier006-report",
            str(tmp_path / "missing-hier006"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote diagnostic report" in completed.stdout
    from scripts.check_report_bundle import check_bundle

    assert check_bundle(output_root, allow_dirty=True) == []
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    context = json.loads((output_root / "context.json").read_text(encoding="utf-8"))
    assert manifest["task"] == "HIER-007"
    expected_arms = [
        "energy__new_only",
        "artifact_first__new_only",
        "energy__overlap",
        "artifact_first__overlap",
    ]
    assert manifest["arms"] == expected_arms
    assert {row["arm"] for row in rows} == set(expected_arms)
    assert context == [
        {
            "artifact_gate_pass": False,
            "artifact_patch_rmse_max_7": 0.1,
            "artifact_pixel_rmse_max": 0.2,
            "context": "HIER-005 contextual",
            "lpips": 0.1,
            "method": "context_control",
            "ms_ssim": 0.9,
            "n_gaussians": 16,
            "psnr_db": 30.0,
            "variant": "n16",
        }
    ]
    base_hashes = {
        row["field_canonical_sha256"]
        for row in rows
        if row["snapshot_label"] == "base"
    }
    assert len(base_hashes) == 1
    assert all(Path(output_root / row["field_npz"]).is_file() for row in rows)
    assert (output_root / "curves" / "catalog.json").is_file()
    assert (
        output_root
        / "source_snapshot"
        / "src"
        / "structsplat"
        / "artifact_first_quadtree.py"
    ).is_file()
