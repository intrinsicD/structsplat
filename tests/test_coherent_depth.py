from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from structsplat.realtime_gs_coherent_depth import (
    CoherentDepthConfig,
    _compatible_surface_cover,
    build_calibration_groups,
    classify_projective_depth,
    contract_candidates,
    dynamic_weighted_sample_elimination,
    fuse_overlapping_depths,
    select_feature_anchors,
    umeyama_similarity,
)


def test_coherent_depth_module_imports_when_optional_dependencies_are_forbidden() -> None:
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "sys.modules['torch']=None; sys.modules['scipy']=None; "
        "sys.modules['safetensors']=None; sys.modules['vggt']=None; sys.modules['rtgs']=None; "
        "import structsplat.realtime_gs_coherent_depth as m; "
        "assert m.VGGT_MODEL_BYTES == 5026367224"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"group_size": 3}, "group_size"),
        ({"inference_max_side": 390}, "divisible"),
        ({"structural_fraction": 0.0}, "structural_fraction"),
        ({"min_primary_fraction": 1.1}, "min_primary_fraction"),
        ({"checkpoint_sha256": "bad"}, "checkpoint_sha256"),
        ({"vggt_source_revision": "bad"}, "vggt_source_revision"),
        ({"surface_cover_target_alpha": 0.0}, "surface_cover_target_alpha"),
        ({"surface_cover_max_opacity": 1.0}, "opacity bounds"),
        ({"surface_cover_neighbors": 2, "surface_cover_spacing_neighbors": 3}, "cannot exceed"),
    ],
)
def test_coherent_depth_config_rejects_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CoherentDepthConfig(**changes)


def test_umeyama_recovers_exact_positive_similarity() -> None:
    source = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.2, 0.3, 1.0]],
        dtype=np.float64,
    )
    angle = np.deg2rad(31.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    scale = 2.75
    translation = np.asarray([0.4, -1.2, 3.0])
    target = scale * (source @ rotation.T) + translation

    result = umeyama_similarity(source, target)

    assert result.scale == pytest.approx(scale, rel=1e-12)
    assert np.allclose(result.rotation, rotation, atol=1e-12)
    assert np.allclose(result.translation, translation, atol=1e-12)
    assert result.diagnostics["center_rmse_over_diameter"] < 1e-12


def test_projective_depth_classifies_support_occlusion_contradiction_and_invalid() -> None:
    projected = torch.full((4,), 2.0)
    target = torch.tensor([2.02, 1.50, 2.50, 0.0])
    uncertainty = torch.full((4,), 0.01)
    valid = torch.ones(4, dtype=torch.bool)

    result = classify_projective_depth(
        projected,
        target,
        uncertainty,
        uncertainty,
        valid,
        relative_tolerance=0.05,
        uncertainty_multiplier=2.0,
    )

    assert result.support.tolist() == [True, False, False, False]
    assert result.occluded.tolist() == [False, True, False, False]
    assert result.contradiction.tolist() == [False, False, True, False]
    assert result.invalid.tolist() == [False, False, False, True]


def test_overlapping_depth_fusion_rejects_one_scaled_outlier() -> None:
    config = CoherentDepthConfig(
        target_count=8,
        candidate_multiplier=2,
        inference_max_side=56,
        oom_fallback_max_side=42,
    )
    estimates = [
        [
            (torch.full((3, 4), 2.00), torch.ones(3, 4)),
            (torch.full((3, 4), 2.02), torch.ones(3, 4)),
            (torch.full((3, 4), 5.00), torch.ones(3, 4)),
        ]
    ]

    depth, uncertainty, confidence, diagnostics = fuse_overlapping_depths(estimates, config)

    assert torch.allclose(depth, torch.full_like(depth, 2.01), atol=1e-6)
    assert torch.allclose(uncertainty, torch.full_like(uncertainty, 0.01005), atol=1e-6)
    assert torch.all(confidence > 0.0)
    assert diagnostics["views"][0]["estimate_count"] == 3


def test_dynamic_wse_is_deterministic_exact_and_preserves_view_floor_and_unique_feature() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(scale=0.01, size=(30, 3))
    second = np.asarray([1.0, 0.0, 0.0])[None] + rng.normal(scale=0.01, size=(30, 3))
    unique = np.asarray([[0.5, 0.8, 0.0]])
    points = np.concatenate([base, second, unique], axis=0)
    colors = np.zeros_like(points)
    colors[:30, 0] = 0.2
    colors[30:60, 1] = 0.2
    colors[-1] = [1.0, 0.0, 0.0]
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(points), 1))
    importance = np.ones(len(points), dtype=np.float64)
    importance[-1] = 0.0
    views = np.concatenate([np.zeros(30), np.ones(31)]).astype(np.int64)
    protected = np.zeros(len(points), dtype=bool)
    protected[-1] = True

    first = dynamic_weighted_sample_elimination(
        points,
        colors,
        normals,
        importance,
        views,
        12,
        view_floor_fraction=0.5,
        protected=protected,
    )
    second_result = dynamic_weighted_sample_elimination(
        points,
        colors,
        normals,
        importance,
        views,
        12,
        view_floor_fraction=0.5,
        protected=protected,
    )

    assert len(first.selected_indices) == 12
    assert np.array_equal(first.selected_indices, second_result.selected_indices)
    assert 60 in first.selected_indices
    assert np.sum(views[first.selected_indices] == 0) >= 3
    assert np.sum(views[first.selected_indices] == 1) >= 3
    assert first.diagnostics["protected_survivor_count"] == 1


def test_feature_anchor_nms_preserves_both_sides_of_color_and_normal_barriers() -> None:
    points = np.asarray(
        [
            [0.000, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.003, 0.0, 0.0],
            [1.000, 0.0, 0.0],
            [2.000, 0.0, 0.0],
            [2.002, 0.0, 0.0],
            [2.003, 0.0, 0.0],
            [3.000, 0.0, 0.0],
        ]
    )
    colors = np.zeros((8, 3), dtype=np.float64)
    colors[[2, 6], 0] = 1.0
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (8, 1))
    normals[[1, 5]] = [1.0, 0.0, 0.0]
    feature = np.asarray([1.0, 0.9, 0.8, 0.1, 1.0, 0.9, 0.8, 0.1])
    importance = np.ones(8)
    views = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    protected, diagnostics = select_feature_anchors(
        points,
        colors,
        normals,
        feature,
        importance,
        views,
        4,
        fraction=1.0,
        radius=0.01,
        rgb_barrier=0.15,
        normal_cosine=0.70,
    )

    assert np.flatnonzero(protected).tolist() == [0, 1, 4, 5]
    assert diagnostics["nms_selected_count"] == 4
    assert diagnostics["forced_fill_count"] == 0


def test_post_wse_contraction_is_cross_view_bounded_and_does_not_blend_appearance() -> None:
    candidates = {
        "means": torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.008, 0.0, 0.0], [1.008, 0.0, 0.0]]
        ),
        "colors": torch.tensor(
            [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1.0, 0.0, 0.0]]
        ),
        "normals": torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1),
        "importance": torch.ones(4),
        "source_view": torch.tensor([0, 0, 1, 1]),
        "source_kind": torch.zeros(4, dtype=torch.long),
        "support": torch.full((4,), 2, dtype=torch.long),
        "contradiction": torch.zeros(4, dtype=torch.long),
    }
    config = CoherentDepthConfig(
        target_count=2,
        candidate_multiplier=2,
        inference_max_side=56,
        oom_fallback_max_side=42,
    )

    output, diagnostics = contract_candidates(candidates, np.asarray([0, 1]), 10.0, config)

    assert output["means"].shape == (2, 3)
    assert 0.0 < float(output["means"][0, 0]) <= diagnostics["max_displacement"]
    assert float(output["means"][1, 0]) == pytest.approx(1.0)
    assert torch.equal(output["colors"], candidates["colors"][:2])
    assert diagnostics["absorbed_proposal_count"] == 1
    assert diagnostics["appearance_changed"] is False


def test_compatible_surface_cover_does_not_bridge_color_or_normal_discontinuities() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.00],
            [1.0, 0.0, 0.00],
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [0.0, 0.0, 0.02],
            [1.0, 0.0, 0.02],
        ]
    )
    colors = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    uncertainty = torch.full((6,), 0.01)
    pixel_footprint = torch.full((6,), 0.1)
    config = CoherentDepthConfig(
        target_count=6,
        candidate_multiplier=1,
        inference_max_side=56,
        oom_fallback_max_side=42,
        surface_cover_neighbors=5,
        surface_cover_spacing_neighbors=1,
        surface_cover_max_pixel_sigma=10.0,
    )

    covariance, opacity, spacing, sigma_tangent, _sigma_normal, diagnostics = (
        _compatible_surface_cover(
            points,
            colors,
            normals,
            uncertainty,
            pixel_footprint,
            10.0,
            config,
        )
    )

    assert torch.allclose(spacing, torch.ones_like(spacing), atol=1e-6)
    assert torch.allclose(sigma_tangent, torch.full_like(sigma_tangent, 0.5), atol=1e-6)
    assert torch.all(opacity > 0.5)
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0.0)
    assert diagnostics["fallback_nearest_geometry_count"] == 0


def test_single_component_surface_cover_is_finite_without_a_neighbor() -> None:
    config = CoherentDepthConfig(
        target_count=1,
        candidate_multiplier=1,
        inference_max_side=56,
        oom_fallback_max_side=42,
    )
    covariance, opacity, spacing, sigma_tangent, _sigma_normal, diagnostics = (
        _compatible_surface_cover(
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([0.01]),
            torch.tensor([0.1]),
            10.0,
            config,
        )
    )

    assert torch.isfinite(covariance).all()
    assert torch.isfinite(opacity).all()
    assert torch.isfinite(spacing).all()
    assert torch.isfinite(sigma_tangent).all()
    assert diagnostics["neighbor_k"] == 0


def test_calibration_groups_cover_all_cameras_deterministically() -> None:
    pytest.importorskip("rtgs")
    from rtgs.core.camera import Camera

    cameras = [
        Camera.look_at(torch.tensor([float(index), float(index % 2), -3.0]), torch.zeros(3))
        for index in range(6)
    ]

    first = build_calibration_groups(cameras)
    second = build_calibration_groups(cameras)

    assert first == second
    assert len(first) == len(cameras)
    assert all(len(group) == 4 for group in first)
    assert {index for group in first for index in group} == set(range(len(cameras)))
