from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from structsplat.codec_native_field import (
    CodecNativeField,
    CodecNativeFieldConfig,
    build_codec_native_field,
    decode_appearance,
    encode_appearance,
    packet_byte_ledger,
)
from structsplat.observation_field import CanvasCropTransform


def _fixture(height: int = 18, width: int = 24) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    image = np.stack(
        [
            xx / max(width - 1, 1),
            yy / max(height - 1, 1),
            ((xx // 3 + yy // 2) % 2).astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    mask = np.zeros((height, width), dtype=bool)
    mask[2 : height - 2, 3 : width - 3] = True
    return image, mask


def _config(**changes) -> CodecNativeFieldConfig:
    values = {
        "appearance_codec": "jpeg",
        "appearance_quality": 92,
        "structural_count": 48,
        "structural_seed": 7,
    }
    values.update(changes)
    return CodecNativeFieldConfig(**values)


def test_structure_is_exact_count_deterministic_and_separate_from_appearance() -> None:
    image, mask = _fixture()
    first = build_codec_native_field(image, config=_config(), mask=mask)
    second = build_codec_native_field(image, config=_config(), mask=mask)

    assert first.structure.n == 48
    assert first.structure.canonical_hash() == second.structure.canonical_hash()
    assert first.appearance_payload == second.appearance_payload
    assert np.all(first.structure.rgb_coeff == 0.0)
    assert first.structure.structural_mass is not None
    assert np.all(first.structure.structural_mass >= 0.0)
    assert np.isclose(first.structure.structural_mass.mean(), 1.0, atol=1e-6)
    assert np.all(mask[
        np.rint(first.structure.means_xy[:, 1]).astype(int),
        np.rint(first.structure.means_xy[:, 0]).astype(int),
    ])


def test_implicit_gaussian_lattice_replays_decoded_pixel_centers() -> None:
    image, _mask = _fixture()
    packet = build_codec_native_field(image, config=_config(structural_count=24))
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    queried = packet.query_appearance(points, apply_alpha=False).reshape(image.shape)

    # The finite normalized Gaussian lattice is deliberately near-cardinal, but still provides a
    # smooth continuous extension between pixels.
    assert np.max(np.abs(queried - packet.decoded_appearance)) < 0.002
    midpoint = packet.query_appearance(np.asarray([[7.5, 8.5]]), apply_alpha=False)
    assert np.isfinite(midpoint).all()
    assert np.all((midpoint >= 0.0) & (midpoint <= 1.0))


def test_prefiltered_wider_gaussian_lattice_interpolates_pixel_centers() -> None:
    image, _mask = _fixture()
    packet = build_codec_native_field(
        image,
        config=_config(
            structural_count=24,
            lattice_sigma_px=0.5,
            lattice_radius_px=3,
            lattice_prefilter_steps=16,
        ),
    )
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    queried = packet.query_appearance(points, apply_alpha=False).reshape(image.shape)

    assert np.max(np.abs(queried - packet.decoded_appearance)) < 2e-6
    assert float(packet.appearance_coefficients.min()) < 0.0
    assert float(packet.appearance_coefficients.max()) > 1.0


def test_prefilter_rejects_non_diagonally_dominant_kernel_at_config_boundary() -> None:
    with pytest.raises(ValueError, match="strictly diagonal-dominant"):
        _config(
            lattice_sigma_px=1.0,
            lattice_radius_px=3,
            lattice_prefilter_steps=8,
        )


def test_lossless_webp_is_exact_and_constant_field_query_is_reproducing() -> None:
    image, _mask = _fixture()
    config = _config(
        appearance_codec="webp_lossless",
        appearance_quality=100,
        structural_count=24,
    )
    payload = encode_appearance(image, config)
    decoded = decode_appearance(payload, "webp_lossless", image.shape[:2])
    expected = np.rint(image * 255.0).astype(np.uint8).astype(np.float32) / 255.0
    assert np.array_equal(decoded, expected)

    constant = np.full((12, 16, 3), [0.2, 0.4, 0.8], dtype=np.float32)
    packet = build_codec_native_field(constant, config=config)
    points = np.asarray([[0.0, 0.0], [5.25, 7.75], [15.0, 11.0]])
    queried = packet.query_appearance(points, apply_alpha=False)
    expected_constant = np.rint(constant[0, 0] * 255.0) / 255.0
    assert np.allclose(queried, expected_constant, atol=1e-7)


def test_query_rejects_malformed_points_and_zeroes_outside_boundary() -> None:
    image, _mask = _fixture()
    packet = build_codec_native_field(image, config=_config(structural_count=24))

    for malformed in (
        np.zeros((3,), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.asarray([[np.nan, 0.0]], dtype=np.float32),
        np.asarray([["x", "y"]]),
    ):
        with pytest.raises(ValueError):
            packet.query_appearance(malformed)
    with pytest.raises(ValueError):
        packet.query_appearance(np.zeros((1, 2)), coordinate_space="invalid")

    points = np.asarray([[-0.49, 0.0], [-0.51, 0.0], [23.49, 17.0], [23.51, 17.0]])
    result = packet.query(points)
    assert result.valid.tolist() == [True, False, True, False]
    assert np.all(result.color[[1, 3]] == 0.0)
    assert np.all(result.structural_density[[1, 3]] == 0.0)


def test_alpha_gates_both_planes_and_canvas_coordinates() -> None:
    image, mask = _fixture()
    transform = CanvasCropTransform(60, 50, 11, 13, image.shape[1], image.shape[0])
    packet = build_codec_native_field(
        image,
        config=_config(),
        mask=mask,
        canvas_crop=transform,
    )
    crop_points = np.asarray([[4.0, 3.0], [0.0, 0.0], [-1.0, 0.0]])
    canvas_points = crop_points + np.asarray([11.0, 13.0])
    crop = packet.query(crop_points)
    canvas = packet.query(canvas_points, coordinate_space="canvas")

    assert np.allclose(crop.color, canvas.color)
    assert np.allclose(crop.structural_density, canvas.structural_density)
    assert crop.alpha.tolist() == [True, False, False]
    assert np.all(crop.color[1:] == 0.0)
    assert np.all(crop.structural_density[1:] == 0.0)
    assert crop.valid.tolist() == [True, True, False]


def test_packet_is_canonical_cold_decodable_and_fully_accounted(tmp_path: Path) -> None:
    image, mask = _fixture()
    source = b"exact supplied source bytes"
    packet = build_codec_native_field(
        image,
        config=_config(appearance_codec="webp", appearance_quality=80),
        mask=mask,
        source_payload=source,
    )
    first_path = tmp_path / "first.sgdp"
    second_path = tmp_path / "second.sgdp"
    first_ledger = packet.save(first_path)
    second_ledger = packet.save(second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_ledger == second_ledger == packet_byte_ledger(first_path)
    assert first_ledger.complete_bytes == first_path.stat().st_size
    assert first_ledger.complete_bytes == (
        first_ledger.manifest_bytes
        + first_ledger.appearance_bytes
        + first_ledger.structure_bytes
        + first_ledger.container_overhead_bytes
    )
    loaded = CodecNativeField.load(first_path)
    assert loaded.source_bytes == len(source)
    assert loaded.source_sha256 == packet.source_sha256
    assert loaded.structure.canonical_hash() == packet.structure.canonical_hash()
    assert np.array_equal(loaded.decoded_appearance, packet.decoded_appearance)
    assert np.array_equal(loaded.render(), packet.render())


def _rewrite_packet(path: Path, mutate) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        members = {info.filename: archive.read(info.filename) for info in archive.infolist()}
    mutate(members)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in sorted(members.items()):
            compression = zipfile.ZIP_STORED if name == "appearance.bin" else zipfile.ZIP_DEFLATED
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            archive.writestr(info, payload, compress_type=compression, compresslevel=9)


@pytest.mark.parametrize("corruption", ["extra", "appearance"])
def test_packet_rejects_unknown_or_integrity_broken_members(
    tmp_path: Path, corruption: str
) -> None:
    image, _mask = _fixture()
    packet = build_codec_native_field(image, config=_config())
    path = tmp_path / "corrupt.sgdp"
    packet.save(path)

    if corruption == "extra":
        _rewrite_packet(path, lambda members: members.__setitem__("extra.bin", b"x"))
    else:
        def corrupt_appearance(members: dict[str, bytes]) -> None:
            payload = bytearray(members["appearance.bin"])
            payload[len(payload) // 2] ^= 1
            members["appearance.bin"] = bytes(payload)

        _rewrite_packet(path, corrupt_appearance)
    with pytest.raises(ValueError):
        CodecNativeField.load(path)


def test_base_module_imports_when_torch_is_forbidden() -> None:
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; sys.modules['torch'] = None; "
        "import structsplat.codec_native_field as m; "
        "import structsplat.realtime_gs_adapter; "
        "import structsplat.realtime_gs_surface_lift; "
        "assert m.PACKET_SCHEMA.endswith('.v2')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_realtime_adapter_query_matches_numpy_when_optional_dependency_is_present() -> None:
    pytest.importorskip("rtgs")
    torch = pytest.importorskip("torch")
    from structsplat.realtime_gs_adapter import make_realtime_gs_view

    image, mask = _fixture()
    transform = CanvasCropTransform(6000, 4608, 274, 1964, image.shape[1], image.shape[0])
    packet = build_codec_native_field(
        image,
        config=_config(structural_count=24),
        mask=mask,
        canvas_crop=transform,
    )
    view = make_realtime_gs_view(packet)
    local = np.asarray([[4.0, 3.0], [0.0, 0.0], [10.25, 8.75]], dtype=np.float32)
    offset = np.asarray([274.5, 1964.5], dtype=np.float32)
    full_numpy = local + offset
    roundtrip_local = full_numpy.astype(np.float64) - offset.astype(np.float64)
    full = torch.from_numpy(full_numpy)
    result = view.query_backend.query(full)
    expected = packet.query_appearance(roundtrip_local)

    assert np.allclose(result.color.detach().cpu().numpy(), expected, atol=2e-6)
    assert result.weight_sum[0] > 0
    assert result.weight_sum[1] == 0
    assert view.structural_field.n == packet.structure.n
    assert view.alpha_crop.shape == mask.shape
    assert np.allclose(
        view.structural_field.local_means().detach().cpu().numpy(),
        packet.structure.means_xy,
        atol=2e-7,
    )


def test_alpha_support_backend_is_calibrated_and_skips_structural_index() -> None:
    pytest.importorskip("rtgs")
    torch = pytest.importorskip("torch")
    from structsplat.realtime_gs_adapter import (
        make_alpha_support_backend,
        make_realtime_gs_view,
    )

    image, mask = _fixture()
    packet = build_codec_native_field(
        image,
        config=_config(structural_count=24),
        mask=mask,
    )
    view = make_realtime_gs_view(packet)
    support = make_alpha_support_backend(
        view,
        coverage_scale=4.0,
        soft_coverage=0.8,
    )
    local = np.asarray([[4.0, 3.0], [0.0, 0.0], [10.25, 8.75]], dtype=np.float32)
    canvas = torch.from_numpy(local + 0.5)
    pairs_before = view.query_backend.structural_backend.total_pairs_evaluated
    result = support.query(canvas)

    assert np.isclose(support.reconstructed_soft_coverage(), 0.8, atol=1e-12)
    assert result.weight_sum[0] == pytest.approx(support.weight_inside)
    assert result.weight_sum[1] == 0.0
    assert np.allclose(
        result.color.detach().cpu().numpy(),
        packet.query_appearance(local),
        atol=2e-6,
    )
    assert support.n_entries == 0
    assert support.payload_bytes == 0
    assert support.total_queries == 1
    assert support.total_points == 3
    assert view.query_backend.structural_backend.total_pairs_evaluated == pairs_before


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"coverage_scale": 0.0}, "coverage_scale"),
        ({"soft_coverage": 1.0}, "soft_coverage"),
    ],
)
def test_alpha_support_backend_rejects_invalid_calibration(changes, error: str) -> None:
    pytest.importorskip("rtgs")
    pytest.importorskip("torch")
    from structsplat.realtime_gs_adapter import (
        make_alpha_support_backend,
        make_realtime_gs_view,
    )

    image, mask = _fixture()
    packet = build_codec_native_field(image, config=_config(structural_count=24), mask=mask)
    view = make_realtime_gs_view(packet)
    with pytest.raises(ValueError, match=error):
        make_alpha_support_backend(view, **changes)


def test_realtime_adapter_supports_cpu_metadata_with_cuda_queries() -> None:
    pytest.importorskip("rtgs")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA query-backend integration requires a CUDA device")
    from structsplat.realtime_gs_adapter import (
        make_alpha_support_backend,
        make_realtime_gs_view,
    )

    image, mask = _fixture()
    packet = build_codec_native_field(
        image,
        config=_config(structural_count=24),
        mask=mask,
    )
    view = make_realtime_gs_view(packet, device="cpu", query_device="cuda")
    local = torch.tensor([[4.0, 3.0], [10.25, 8.75]], dtype=torch.float32)
    canvas = local + 0.5
    result = view.query_backend.query(canvas)
    expected = packet.query_appearance(local.numpy())

    assert view.structural_field.device.type == "cpu"
    assert view.query_backend.structural_backend.device.type == "cuda"
    assert result.color.device.type == "cpu"
    assert result.weight_sum.device.type == "cpu"
    assert view.query_backend.total_pairs_evaluated > 0
    assert view.query_backend.peak_pair_chunk > 0
    assert np.allclose(result.color.numpy(), expected, atol=2e-6)

    structural_pairs = view.query_backend.structural_backend.total_pairs_evaluated
    support = make_alpha_support_backend(view)
    support_result = support.query(canvas)
    assert support_result.color.device.type == "cpu"
    assert support_result.weight_sum.device.type == "cpu"
    assert support_result.weight_sum.min() > 0
    assert view.query_backend.structural_backend.total_pairs_evaluated == structural_pairs


def test_realtime_adapter_runs_existing_compact_2d_to_3d_initializer() -> None:
    pytest.importorskip("rtgs")
    torch = pytest.importorskip("torch")
    from rtgs.core.camera import Camera
    from rtgs.data.reconstruction_inputs import ReconstructionInputs
    from rtgs.lift.compact_carve import CompactCarveConfig, CompactCarveInitializer

    from structsplat.realtime_gs_adapter import make_realtime_gs_view

    image = np.full((32, 32, 3), [0.3, 0.6, 0.2], dtype=np.float32)
    config = _config(
        appearance_codec="webp_lossless",
        appearance_quality=75,
        lattice_sigma_px=0.45,
        lattice_radius_px=3,
        lattice_prefilter_steps=8,
        structural_count=32,
        structural_seed=5,
    )
    packets = [build_codec_native_field(image, config=config) for _ in range(2)]
    views = [make_realtime_gs_view(packet, device="cpu") for packet in packets]
    cameras = [
        Camera.look_at(
            torch.tensor([x, 0.0, -3.0]),
            torch.zeros(3),
            width=32,
            height=32,
            fov_x_deg=55.0,
        )
        for x in (-0.75, 0.75)
    ]
    inputs = ReconstructionInputs(
        observations=[view.structural_field for view in views],
        cameras=cameras,
        view_names=["left", "right"],
        bounds_hint=(torch.zeros(3), 1.2),
        name="codec-native-lift-smoke",
    )
    carve = CompactCarveConfig(
        n_init_3d=4,
        candidate_multiplier=8,
        samples_per_ray=16,
        query_batch_size=64,
        seed=17,
        min_views=2,
        hull_fraction=1.0,
        coverage_scale=4.0,
        coverage_threshold=0.01,
        color_std_sigma=0.25,
        min_score=0.001,
    )
    backends = [view.query_backend for view in views]
    first = CompactCarveInitializer(carve).initialize(inputs, backends=backends)
    repeated = CompactCarveInitializer(carve).initialize(inputs, backends=backends)

    assert first.gaussians.n == carve.n_init_3d
    assert bool(torch.isfinite(first.gaussians.means).all())
    assert torch.equal(first.gaussians.means, repeated.gaussians.means)
    assert first.diagnostics["teacher_backend_kinds"] == [
        "CodecNativeObservationBackend",
        "CodecNativeObservationBackend",
    ]


def _surface_lift_fixture(*, camera_offsets: tuple[float, ...]):
    torch = pytest.importorskip("torch")
    from rtgs.core.camera import Camera
    from rtgs.data.reconstruction_inputs import ReconstructionInputs

    from structsplat.realtime_gs_adapter import make_realtime_gs_view

    image = np.full((32, 32, 3), [0.3, 0.6, 0.2], dtype=np.float32)
    mask = np.ones((32, 32), dtype=bool)
    packets = [
        build_codec_native_field(
            image,
            config=_config(
                appearance_codec="webp_lossless",
                appearance_quality=100,
                structural_count=40,
                structural_seed=index + 3,
            ),
            mask=mask,
        )
        for index in range(len(camera_offsets))
    ]
    views = [make_realtime_gs_view(packet) for packet in packets]
    cameras = [
        Camera.look_at(
            torch.tensor([offset, 0.0, -3.0]),
            torch.zeros(3),
            width=32,
            height=32,
            fov_x_deg=55.0,
        )
        for offset in camera_offsets
    ]
    inputs = ReconstructionInputs(
        observations=[view.structural_field for view in views],
        cameras=cameras,
        view_names=[f"view-{index}" for index in range(len(views))],
        bounds_hint=(torch.zeros(3), 1.2),
        name="visibility-ordered-surface-lift-test",
    )
    return inputs, views


def test_visibility_ordered_surface_uses_first_maximum_and_preserves_means() -> None:
    pytest.importorskip("rtgs")
    torch = pytest.importorskip("torch")
    from rtgs.lift.compact_carve import CompactCarveConfig

    from structsplat.realtime_gs_surface_lift import initialize_visibility_ordered_surface

    inputs, views = _surface_lift_fixture(camera_offsets=(0.0, 0.0))
    carve = CompactCarveConfig(
        n_init_3d=8,
        candidate_multiplier=8,
        samples_per_ray=16,
        query_batch_size=128,
        seed=17,
        min_views=2,
        hull_fraction=1.0,
        coverage_scale=1.0,
        coverage_threshold=0.4,
        color_std_sigma=0.25,
        min_score=0.01,
    )
    first = initialize_visibility_ordered_surface(inputs, views, carve)
    repeated = initialize_visibility_ordered_surface(inputs, views, carve)

    assert first.initialization.gaussians.n == carve.n_init_3d
    assert first.diagnostics["selected_depth_index"]["max"] == 0.0
    assert first.diagnostics["support_index_entries"] == 0
    assert first.diagnostics["support_new_payload_bytes"] == 0
    assert first.diagnostics["surface_cover"] is not None
    assert first.diagnostics["structural_pairs_before"] == first.diagnostics[
        "structural_pairs_after"
    ]
    assert torch.equal(
        first.initialization.gaussians.means,
        first.raw_initialization.gaussians.means,
    )
    assert torch.equal(
        first.initialization.gaussians.sh,
        first.raw_initialization.gaussians.sh,
    )
    assert not torch.equal(
        first.initialization.gaussians.log_scales,
        first.raw_initialization.gaussians.log_scales,
    )
    assert torch.equal(
        first.initialization.gaussians.means,
        repeated.initialization.gaussians.means,
    )
    assert torch.equal(
        first.initialization.gaussians.log_scales,
        repeated.initialization.gaussians.log_scales,
    )
    assert bool(torch.isfinite(first.initialization.gaussians.covariance()).all())


def test_visibility_ordered_surface_differs_from_interior_consensus() -> None:
    pytest.importorskip("rtgs")
    torch = pytest.importorskip("torch")
    from rtgs.lift.compact_carve import CompactCarveConfig, CompactCarveInitializer

    from structsplat.realtime_gs_surface_lift import (
        VisibilityOrderedSurfaceLiftConfig,
        initialize_visibility_ordered_surface,
    )

    inputs, views = _surface_lift_fixture(camera_offsets=(-0.75, 0.0, 0.75))
    carve = CompactCarveConfig(
        n_init_3d=8,
        candidate_multiplier=8,
        samples_per_ray=24,
        query_batch_size=192,
        seed=23,
        min_views=2,
        hull_fraction=2.0 / 3.0,
        coverage_scale=1.0,
        coverage_threshold=0.1,
        color_std_sigma=0.25,
        min_score=0.001,
    )
    interior = CompactCarveInitializer(carve).initialize(
        inputs,
        backends=[view.query_backend for view in views],
    )
    shell = initialize_visibility_ordered_surface(
        inputs,
        views,
        carve,
        VisibilityOrderedSurfaceLiftConfig(apply_surface_cover=False),
    )

    assert shell.initialization.gaussians.n == interior.gaussians.n == carve.n_init_3d
    assert not torch.equal(shell.initialization.depths, interior.depths)
    assert shell.initialization.diagnostics["teacher_backend_kinds"] == [
        "CodecNativeAlphaSupportBackend",
        "CodecNativeAlphaSupportBackend",
        "CodecNativeAlphaSupportBackend",
    ]


def test_visibility_ordered_surface_rejects_mismatched_views() -> None:
    pytest.importorskip("rtgs")
    pytest.importorskip("torch")
    from rtgs.lift.compact_carve import CompactCarveConfig

    from structsplat.realtime_gs_surface_lift import initialize_visibility_ordered_surface

    inputs, views = _surface_lift_fixture(camera_offsets=(-0.5, 0.5))
    with pytest.raises(ValueError, match="one paired codec-native view"):
        initialize_visibility_ordered_surface(
            inputs,
            views[:1],
            CompactCarveConfig(n_init_3d=4),
        )
