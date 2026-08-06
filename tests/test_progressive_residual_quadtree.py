from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from structsplat.observation_field import ObservationField2D
from structsplat.pixel_contraction import render_observation_field
from structsplat.progressive_residual_quadtree import (
    _base_keys,
    _child_keys,
    _lexicographic_improves,
    _node_geometry,
    ProgressiveResidualConfig,
    build_progressive_residual_quadtree,
    progressive_artifact_metrics,
    progressive_prefix_field,
)


ROOT = Path(__file__).resolve().parents[1]


def _image(height: int = 9, width: int = 11, seed: int = 17) -> np.ndarray:
    return np.random.default_rng(seed).random((height, width, 3), dtype=np.float32)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"start_level": True}, TypeError),
        ({"max_gaussians": 0}, ValueError),
        ({"leaf_scale_px": 0.0}, ValueError),
        ({"error_smoothing_sigma_px": -0.1}, ValueError),
        ({"tail_fraction": 0.0}, ValueError),
        ({"tail_fraction": 1.1}, ValueError),
        ({"support_fade_alpha": 1.1}, ValueError),
        ({"renderer": "cuda_additive", "device": "cpu"}, ValueError),
        ({"milestone_counts": [4]}, TypeError),
        ({"milestone_counts": (True,)}, TypeError),
        ({"milestone_counts": ("4",)}, TypeError),
    ],
)
def test_config_fails_closed(kwargs, error):
    with pytest.raises(error):
        ProgressiveResidualConfig(**kwargs)


def test_config_canonicalizes_milestones_within_cap():
    config = ProgressiveResidualConfig(
        max_gaussians=12,
        milestone_counts=(12, 4, 16, 4),
    )
    assert config.milestone_counts == (4, 12)


def test_lexicographic_comparator_treats_float32_roundoff_as_a_tie():
    reference_violation = 18.83538928817966
    candidate_violation = np.float32(reference_violation).item()
    assert candidate_violation > reference_violation
    assert _lexicographic_improves(
        candidate_violation,
        192.0,
        reference_violation,
        217.0,
    )
    assert not _lexicographic_improves(
        reference_violation + 0.01,
        100.0,
        reference_violation,
        217.0,
    )


def test_quadtree_keys_are_mask_present_canonical_and_children_partition_parent():
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:6, 2:8] = True
    keys = _base_keys(mask, 2)
    assert keys == ((2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1))
    children = _child_keys(mask, (2, 0, 0))
    assert children == ((1, 0, 1), (1, 1, 1))
    covered = np.zeros_like(mask)
    for child in children:
        side = 1 << child[0]
        y0 = child[1] * side
        x0 = child[2] * side
        covered[y0 : y0 + side, x0 : x0 + side] |= mask[
            y0 : y0 + side, x0 : x0 + side
        ]
    assert np.array_equal(covered[:4, :4], mask[:4, :4])


def test_node_geometry_uses_mask_moments_and_leaf_variance():
    mask = np.zeros((5, 6), dtype=bool)
    mask[2, 3] = True
    mean, log_scales, angle = _node_geometry(mask, (3, 0, 0), 0.25)
    assert np.array_equal(mean, np.array([3.0, 2.0], dtype=np.float32))
    assert np.allclose(np.exp(log_scales), 0.25, atol=1e-7)
    assert float(angle) == pytest.approx(0.0)


def test_displayed_artifact_metrics_quantize_both_images_exactly():
    target = np.zeros((3, 3, 3), dtype=np.float32)
    reconstruction = target.copy()
    target[1, 1] = 0.5001
    reconstruction[1, 1] = 0.5018
    mask = np.ones((3, 3), dtype=bool)
    raw = progressive_artifact_metrics(
        reconstruction,
        target,
        mask,
        pixel_threshold=0.001,
        patch7_threshold=0.001,
        displayed=False,
    )
    displayed = progressive_artifact_metrics(
        reconstruction,
        target,
        mask,
        pixel_threshold=0.001,
        patch7_threshold=0.001,
        displayed=True,
    )
    assert float(raw["pixel_rmse_max"]) > 0.001
    assert float(displayed["pixel_rmse_max"]) == pytest.approx(0.0)


def test_builder_preserves_prefix_and_is_deterministic():
    image = _image()
    base_config = ProgressiveResidualConfig(
        start_level=3,
        max_gaussians=4,
        max_rows_per_stage=8,
        base_steps=2,
        layer_steps=2,
        checkpoint_every=1,
        render_chunk=16,
        milestone_counts=(),
    )
    base = build_progressive_residual_quadtree(image, base_config)
    full_config = ProgressiveResidualConfig(
        start_level=3,
        max_gaussians=20,
        max_rows_per_stage=8,
        base_steps=2,
        layer_steps=2,
        checkpoint_every=1,
        render_chunk=16,
        milestone_counts=(12,),
    )
    first = build_progressive_residual_quadtree(image, full_config)
    second = build_progressive_residual_quadtree(image, full_config)
    assert first.field.canonical_hash() == second.field.canonical_hash()
    assert np.array_equal(first.reconstruction_raw, second.reconstruction_raw)
    assert [stage.status for stage in first.stages] == [stage.status for stage in second.stages]
    assert first.prefix_bit_exact
    assert first.final_count <= full_config.max_gaussians
    assert first.base_count == base.final_count
    for base_array, final_array in (
        (base.field.means_xy, first.field.means_xy),
        (base.field.log_scales_xy, first.field.log_scales_xy),
        (base.field.rotations_rad, first.field.rotations_rad),
        (base.field.rgb_coeff, first.field.rgb_coeff),
    ):
        assert np.array_equal(base_array, final_array[: base.final_count])
    for stage in first.stages:
        assert stage.count_after == stage.count_before + stage.accepted_rows
        if stage.kind == "residual_children":
            assert stage.proposed_rows == len(stage.child_keys)
            assert stage.proposed_rows <= full_config.max_rows_per_stage
            if stage.status == "accepted":
                assert _lexicographic_improves(
                    stage.raw_violation_after,
                    stage.sse_after,
                    stage.raw_violation_before,
                    stage.sse_before,
                )
            else:
                assert stage.count_after == stage.count_before
    accepted_children = [
        child
        for stage in first.stages
        if stage.status == "accepted" and stage.kind == "residual_children"
        for child in stage.child_keys
    ]
    assert len(accepted_children) == len(set(accepted_children))


def test_prefix_roundtrip_and_cold_render_match(tmp_path: Path):
    image = _image(7, 9, seed=29)
    mask = np.ones((7, 9), dtype=bool)
    mask[:2, :3] = False
    result = build_progressive_residual_quadtree(
        image,
        ProgressiveResidualConfig(
            start_level=2,
            max_gaussians=18,
            max_rows_per_stage=6,
            base_steps=2,
            layer_steps=2,
            checkpoint_every=1,
            render_chunk=16,
            milestone_counts=(),
        ),
        mask=mask,
    )
    assert result.field.packed_alpha is not None
    prefix = progressive_prefix_field(result.field, result.base_count)
    assert prefix.n == result.base_count
    for prefix_array, full_array in (
        (prefix.means_xy, result.field.means_xy),
        (prefix.log_scales_xy, result.field.log_scales_xy),
        (prefix.rotations_rad, result.field.rotations_rad),
        (prefix.rgb_coeff, result.field.rgb_coeff),
    ):
        assert np.array_equal(prefix_array, full_array[: result.base_count])

    field_path = tmp_path / "field.npz"
    result.field.save_lossless(field_path)
    loaded = ObservationField2D.load_lossless(field_path)
    assert loaded.canonical_hash() == result.field.canonical_hash()
    cold = render_observation_field(loaded, render_chunk=16, apply_declared_alpha=False)
    assert np.max(np.abs(cold - result.reconstruction_raw)) < 2e-6
    assert np.array_equal(result.reconstruction[~mask], np.zeros((int((~mask).sum()), 3)))


def test_module_import_does_not_require_torch():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "import structsplat.progressive_residual_quadtree; print('ok')"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "ok"


def test_report_driver_writes_a_valid_generic_bundle(tmp_path: Path):
    height, width = 32, 40
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack(
        [
            xx / (width - 1),
            yy / (height - 1),
            ((xx // 4 + yy // 4) % 2).astype(np.float64),
        ],
        axis=2,
    )
    mask = ((xx - 20) / 17) ** 2 + ((yy - 16) / 13) ** 2 <= 1.0
    image_path = tmp_path / "tiny.jpg"
    mask_path = tmp_path / "tiny_mask.png"
    Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), mode="RGB").save(image_path)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    output_root = tmp_path / "report"
    missing_comparison = tmp_path / "missing-comparison"
    script = ROOT / "scripts" / "experiments" / "hier006_progressive_residual_quadtree.py"
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
            "40",
            "--start-level",
            "4",
            "--max-gaussians",
            "24",
            "--max-rows-per-stage",
            "8",
            "--base-steps",
            "2",
            "--layer-steps",
            "2",
            "--checkpoint-every",
            "1",
            "--device",
            "cpu",
            "--renderer",
            "additive",
            "--render-chunk",
            "16",
            "--milestone-counts",
            "16",
            "--comparison-report",
            str(missing_comparison),
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
    assert manifest["schema"] == "structsplat.current_pipeline.workflow.v1"
    assert manifest["method"] == "progressive_residual_quadtree"
    assert len(manifest["variants"]) == len(rows) >= 2
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["prefix_bit_exact"] for row in rows)
    assert all(Path(output_root / row["field_npz"]).is_file() for row in rows)
    assert (output_root / "curves" / "catalog.json").is_file()
    assert (
        output_root
        / "source_snapshot"
        / "src"
        / "structsplat"
        / "progressive_residual_quadtree.py"
    ).is_file()
