from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

pytest.importorskip("rtgs")

from rtgs.core.camera import Camera
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.compact_carve import CompactCarveConfig

from structsplat.codec_native_field import (
    CodecNativeFieldConfig,
    build_codec_native_field,
)
from structsplat.realtime_gs_adapter import make_realtime_gs_view
from structsplat.realtime_gs_ray_posterior import (
    PacketFeatureSet,
    PacketFeatureView,
    RayPosteriorConfig,
    _candidate_reciprocal_support,
    build_packet_feature_pyramids,
    initialize_occlusion_aware_ray_posterior,
)


class _PatchMeanModel(torch.nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward_features(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = functional.avg_pool2d(value, self.patch_size, self.patch_size)
        tokens = pooled.flatten(2).transpose(1, 2)
        return {"x_norm_patchtokens": functional.normalize(tokens, dim=-1)}


def _codec_config(seed: int, count: int = 96) -> CodecNativeFieldConfig:
    return CodecNativeFieldConfig(
        appearance_codec="webp_lossless",
        appearance_quality=100,
        lattice_sigma_px=0.45,
        lattice_radius_px=3,
        lattice_prefilter_steps=8,
        structural_count=count,
        structural_seed=seed,
    )


def _camera(x: float, y: float, size: int = 48) -> Camera:
    return Camera.look_at(
        torch.tensor([x, y, -3.0]),
        torch.zeros(3),
        width=size,
        height=size,
        fov_x_deg=48.0,
    )


def _plane_fixture(size: int = 48):
    cameras = [
        _camera(-0.70, 0.00, size),
        _camera(-0.25, 0.15, size),
        _camera(0.25, -0.10, size),
        _camera(0.70, 0.05, size),
    ]
    views = []
    feature_views = []
    for index, camera in enumerate(cameras):
        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32) + 0.5,
            torch.arange(size, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        uv = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        origin, direction = camera.pixel_rays(uv)
        depth = -origin[2] / direction[:, 2]
        world = origin + depth[:, None] * direction
        world_x, world_y = world[:, 0], world[:, 1]
        rgb = torch.stack(
            [
                (torch.sin(3.0 * world_x) + 1.0) / 2.0,
                (torch.sin(4.0 * world_y + 0.4) + 1.0) / 2.0,
                (torch.sin(2.0 * world_x + 3.0 * world_y) + 1.0) / 2.0,
            ],
            dim=-1,
        ).reshape(size, size, 3)
        packet = build_codec_native_field(
            rgb.numpy().astype(np.float32),
            config=_codec_config(index),
        )
        views.append(make_realtime_gs_view(packet, device="cpu", query_device="cpu"))
        descriptor = torch.stack(
            [
                torch.sin(2.0 * world_x),
                torch.cos(2.0 * world_x),
                torch.sin(2.0 * world_y),
                torch.cos(2.0 * world_y),
                torch.sin(3.0 * (world_x + world_y)),
                torch.cos(3.0 * (world_x + world_y)),
                torch.sin(5.0 * world_x - 2.0 * world_y),
                torch.cos(5.0 * world_x - 2.0 * world_y),
            ],
            dim=0,
        ).reshape(8, size, size)
        descriptor = functional.normalize(descriptor, dim=0)
        feature_views.append(
            PacketFeatureView(
                semantic=descriptor,
                detail=descriptor.clone(),
                alpha=torch.ones(1, size, size),
                crop=(0, 0, size, size),
                canvas=(size, size),
                semantic_input_shape=(size, size),
            )
        )
    inputs = ReconstructionInputs(
        observations=[view.structural_field for view in views],
        cameras=cameras,
        view_names=[f"v{index}" for index in range(len(cameras))],
        bounds_hint=(torch.zeros(3), 2.0),
        name="ray-posterior-plane",
    )
    return inputs, views, PacketFeatureSet(tuple(feature_views), {"seconds": 0.0})


def test_feature_builder_queries_packet_appearance_without_touching_structural_index() -> None:
    image = np.linspace(0.0, 1.0, 16 * 20 * 3, dtype=np.float32).reshape(16, 20, 3)
    packet = build_codec_native_field(image, config=_codec_config(3, count=32))
    view = make_realtime_gs_view(packet, device="cpu", query_device="cpu")
    pairs_before = view.query_backend.structural_backend.total_pairs_evaluated
    config = RayPosteriorConfig(
        feature_device="cpu",
        feature_storage_dtype="float32",
        feature_patch_size=4,
        feature_max_side=32,
        target_views=1,
        best_view_count=1,
        min_evidence_views=1,
        min_reciprocal_views=0,
    )

    result = build_packet_feature_pyramids(
        [view], config, model=_PatchMeanModel(config.feature_patch_size)
    )
    query_xy = torch.tensor([[0.5, 0.5], [19.5, 15.5]], dtype=torch.float32)
    color, alpha, valid = view.query_backend.query_appearance(query_xy)

    assert result.diagnostics["source_rgb_opened"] is False
    assert result.diagnostics["model"]["injected_model"] is True
    assert result.views[0].detail.shape == (8, 16, 20)
    assert result.views[0].semantic.shape[0] == 3
    assert torch.isfinite(result.views[0].semantic).all()
    assert color.device == alpha.device == valid.device == query_xy.device
    assert view.query_backend.structural_backend.total_pairs_evaluated == pairs_before


def test_textured_plane_lifts_exact_count_near_true_surface() -> None:
    inputs, views, features = _plane_fixture()
    carve = CompactCarveConfig(
        n_init_3d=24,
        candidate_multiplier=3,
        samples_per_ray=16,
        query_batch_size=1_024,
        seed=3,
        bounds_scale=0.5,
        min_views=2,
    )
    config = RayPosteriorConfig(
        feature_device="cpu",
        feature_storage_dtype="float32",
        target_views=3,
        target_baseline_deg=15.0,
        min_baseline_deg=0.0,
        max_baseline_deg=90.0,
        best_view_count=2,
        min_evidence_views=2,
        dustbin_cost=0.8,
        posterior_temperature=0.05,
        dino_weight=1.0,
        detail_weight=0.0,
        reciprocal_pixel_radius=20.0,
        reciprocal_depth_extent_fraction=0.1,
        reciprocal_world_extent_fraction=0.2,
        min_reciprocal_views=0,
        apply_reciprocal=False,
        min_primary_fraction=1.0,
        apply_surface_cover=False,
    )

    first = initialize_occlusion_aware_ray_posterior(
        inputs, views, carve, features, config
    )
    second = initialize_occlusion_aware_ray_posterior(
        inputs, views, carve, features, config
    )
    absolute_z = first.initialization.gaussians.means[:, 2].abs()

    assert first.initialization.gaussians.n == 24
    assert float(absolute_z.median()) < 0.01
    assert float(torch.quantile(absolute_z, 0.75)) < 0.04
    assert first.diagnostics["structural_index_pairs_evaluated"] == 0
    assert first.diagnostics["fallback_selected_count"] == 0
    assert torch.equal(
        first.initialization.gaussians.means,
        second.initialization.gaussians.means,
    )
    assert torch.isfinite(first.initialization.gaussians.covariance()).all()


def test_lift_rejects_a_view_order_that_does_not_match_inputs() -> None:
    inputs, views, features = _plane_fixture()
    carve = CompactCarveConfig(n_init_3d=8, candidate_multiplier=2, samples_per_ray=8)
    config = RayPosteriorConfig(
        feature_device="cpu",
        feature_storage_dtype="float32",
        target_views=2,
        best_view_count=2,
        min_evidence_views=2,
        min_reciprocal_views=0,
        apply_reciprocal=False,
        apply_surface_cover=False,
    )

    with pytest.raises(ValueError, match="does not own"):
        initialize_occlusion_aware_ray_posterior(
            inputs,
            [views[1], views[0], *views[2:]],
            carve,
            features,
            config,
        )


def test_reciprocal_support_rejects_an_unmatched_floater() -> None:
    cameras = (_camera(-0.5, 0.0), _camera(0.5, 0.0))
    shared = torch.tensor([[-0.2, 0.0, 0.0], [0.2, 0.1, 0.0]])
    floater = torch.tensor([[0.0, -0.2, 0.7]])
    means = torch.cat([shared, floater, shared, torch.tensor([[0.0, 0.2, 0.0]])])
    view_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    xy_parts = []
    depths = []
    for view in range(2):
        indices = (view_ids == view).nonzero(as_tuple=True)[0]
        projected, depth = cameras[view].project(means[indices])
        xy_parts.append(projected)
        depths.append(depth)
    xy = torch.cat(xy_parts)
    depth = torch.cat(depths)
    config = RayPosteriorConfig(
        feature_device="cpu",
        target_views=1,
        best_view_count=1,
        min_evidence_views=1,
        reciprocal_pixel_radius=6.0,
        reciprocal_depth_extent_fraction=0.03,
        reciprocal_world_extent_fraction=0.08,
        min_reciprocal_views=1,
    )

    support = _candidate_reciprocal_support(
        means,
        depth,
        torch.ones(6, dtype=torch.bool),
        view_ids,
        xy,
        cameras,
        ((1,), (0,)),
        2.0,
        config,
    )

    assert support[:2].tolist() == [1, 1]
    assert support[2] == 0
