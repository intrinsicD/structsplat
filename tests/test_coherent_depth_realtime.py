from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("rtgs")

from rtgs.core.camera import Camera
from rtgs.data.reconstruction_inputs import ReconstructionInputs

from structsplat.codec_native_field import CodecNativeFieldConfig, build_codec_native_field
from structsplat.realtime_gs_adapter import make_realtime_gs_view
from structsplat.realtime_gs_coherent_depth import (
    CoherentDepthConfig,
    build_packet_inference_images,
    infer_coherent_depth_field,
    initialize_calibrated_coherent_depth,
)


def _camera(x: float, y: float, size: int) -> Camera:
    return Camera.look_at(
        torch.tensor([x, y, -3.0]),
        torch.zeros(3),
        width=size,
        height=size,
        fov_x_deg=50.0,
    )


def _fixture(size: int = 42):
    cameras = [
        _camera(-0.8, -0.25, size),
        _camera(0.7, -0.35, size),
        _camera(-0.6, 0.50, size),
        _camera(0.8, 0.45, size),
    ]
    views = []
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
        color = torch.stack(
            [
                (torch.sin(3.0 * world[:, 0]) + 1.0) / 2.0,
                (torch.sin(4.0 * world[:, 1] + 0.2) + 1.0) / 2.0,
                (torch.sin(2.0 * world[:, 0] - 3.0 * world[:, 1]) + 1.0) / 2.0,
            ],
            dim=-1,
        ).reshape(size, size, 3)
        packet = build_codec_native_field(
            color.numpy().astype(np.float32),
            config=CodecNativeFieldConfig(
                appearance_codec="webp_lossless",
                appearance_quality=100,
                lattice_sigma_px=0.45,
                lattice_radius_px=3,
                lattice_prefilter_steps=8,
                structural_count=96,
                structural_seed=index,
            ),
        )
        views.append(make_realtime_gs_view(packet, device="cpu", query_device="cpu"))
    inputs = ReconstructionInputs(
        observations=[view.structural_field for view in views],
        cameras=cameras,
        view_names=[f"v{index}" for index in range(len(views))],
        bounds_hint=(torch.zeros(3), 2.0),
        name="coherent-depth-plane",
    )
    return inputs, views


class _ExactPlanePredictor:
    def __init__(self, cameras: list[Camera]) -> None:
        self.cameras = cameras

    def __call__(self, images: torch.Tensor, group: tuple[int, ...]):
        height, width = images.shape[-2:]
        depths = []
        extrinsics = []
        intrinsics = []
        for view_index in group:
            camera = self.cameras[view_index]
            yy, xx = torch.meshgrid(
                torch.arange(height, dtype=torch.float32),
                torch.arange(width, dtype=torch.float32),
                indexing="ij",
            )
            uv = torch.stack(
                [
                    (xx.reshape(-1) + 0.5) * camera.width / width,
                    (yy.reshape(-1) + 0.5) * camera.height / height,
                ],
                dim=-1,
            )
            origin, direction = camera.pixel_rays(uv)
            depths.append((-origin[2] / direction[:, 2]).reshape(height, width))
            extrinsics.append(camera.viewmat[:3].numpy())
            scale_x = width / camera.width
            scale_y = height / camera.height
            intrinsic = camera.K.numpy().copy()
            intrinsic[0] *= scale_x
            intrinsic[1] *= scale_y
            intrinsics.append(intrinsic)
        return {
            "depth": torch.stack(depths).numpy(),
            "confidence": np.ones((4, height, width), dtype=np.float32),
            "extrinsic": np.stack(extrinsics),
            "intrinsic": np.stack(intrinsics),
            "seconds": 0.0,
        }


def _config() -> CoherentDepthConfig:
    return CoherentDepthConfig(
        target_count=48,
        candidate_multiplier=3,
        structural_fraction=0.8,
        seed=7,
        inference_max_side=56,
        oom_fallback_max_side=42,
        max_group_center_rmse_fraction=1e-4,
        max_group_loo_median_fraction=1e-4,
        max_group_orientation_median_deg=0.05,
        max_group_focal_relative_median=1e-3,
        support_views=3,
        support_relative_tolerance=0.05,
        min_source_confidence=0.0,
        min_target_confidence=0.0,
        min_support_views=1,
        max_contradictions=0,
        min_normal_cosine=0.8,
        min_primary_fraction=0.9,
        contraction_radius_diameter_fraction=0.002,
        wse_neighbors=12,
        apply_surface_cover=True,
    )


def test_packet_inference_images_never_query_structural_index() -> None:
    _inputs, views = _fixture()
    before = [view.query_backend.structural_backend.total_pairs_evaluated for view in views]

    images, diagnostics = build_packet_inference_images(views, _config())

    after = [view.query_backend.structural_backend.total_pairs_evaluated for view in views]
    assert images.shape == (4, 3, 56, 56)
    assert torch.isfinite(images).all()
    assert diagnostics["source_rgb_opened"] is False
    assert before == after


def test_exact_coherent_plane_fuses_and_lifts_deterministic_finite_exact_budget() -> None:
    inputs, views = _fixture()
    config = _config()
    predictor = _ExactPlanePredictor(inputs.cameras)

    field = infer_coherent_depth_field(
        inputs,
        views,
        None,
        config,
        predictor=predictor,
    )
    first = initialize_calibrated_coherent_depth(
        inputs,
        views,
        None,
        config,
        field=field,
        apply_projective_support=True,
    )
    second = initialize_calibrated_coherent_depth(
        inputs,
        views,
        None,
        config,
        field=field,
        apply_projective_support=True,
    )

    assert field.diagnostics["accepted_group_count"] == 4
    assert field.diagnostics["source_rgb_opened"] is False
    assert first.initialization.gaussians.n == config.target_count
    assert torch.equal(
        first.initialization.gaussians.means,
        second.initialization.gaussians.means,
    )
    assert float(first.initialization.gaussians.means[:, 2].abs().max()) < 1e-4
    covariance = first.initialization.gaussians.covariance()
    assert torch.isfinite(covariance).all()
    assert torch.all(torch.linalg.eigvalsh(covariance) > 0.0)
    assert first.diagnostics["selected_fallback_count"] == 0
    assert first.diagnostics["wse"]["applied"] is True
    assert first.diagnostics["contraction"]["applied"] is True
    assert first.diagnostics["wse"]["removed_count"] > 0
    assert min(first.diagnostics["selected_per_view"].values()) >= 6
    assert first.diagnostics["feature_anchors"]["selected_count"] == 7
    assert first.diagnostics["surface_cover"]["orientation"] == "fused_depth_normal"


def test_raw_known_ray_arm_uses_balanced_selection_without_wse_or_support() -> None:
    inputs, views = _fixture()
    config = _config()
    field = infer_coherent_depth_field(
        inputs,
        views,
        None,
        config,
        predictor=_ExactPlanePredictor(inputs.cameras),
    )

    result = initialize_calibrated_coherent_depth(
        inputs,
        views,
        None,
        config,
        field=field,
        apply_projective_support=False,
    )

    assert result.initialization.gaussians.n == config.target_count
    assert result.diagnostics["wse"]["applied"] is False
    assert result.diagnostics["contraction"]["applied"] is False
    assert result.diagnostics["selected_fallback_count"] == 0
