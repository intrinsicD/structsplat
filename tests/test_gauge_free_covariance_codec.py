import math

import numpy as np
import pytest
import torch

from benchmarks import gauge_free_covariance_codec as codec
from benchmarks import gauge_free_covariance_core as core
from structsplat.gaussians import GaussianField


def _field(n: int = 19) -> GaussianField:
    rng = np.random.default_rng(23)
    means = rng.uniform([-5.0, -3.0], [68.0, 51.0], size=(n, 2))
    scales = np.exp(rng.uniform(-1.4, 1.7, size=(n, 2)))
    rotations = rng.uniform(-4.0 * math.pi, 4.0 * math.pi, size=n)
    colors = rng.normal(0.45, 0.35, size=(n, 3))
    return GaussianField.from_numpy(means, scales, rotations, colors)


def _ordered_reference(field: GaussianField, height: int = 48, width: int = 64) -> dict[str, np.ndarray]:
    means = field.means.detach().numpy().astype(np.float64)
    log_scales = field.effective_log_scales().detach().numpy().astype(np.float64)
    rotations = field.rotations.detach().numpy().astype(np.float64)
    colors = field.colors.detach().numpy().astype(np.float64)
    order = codec.morton_order(means, height, width)
    return {
        "means": means[order],
        "log_scales": log_scales[order],
        "rotations": rotations[order],
        "colors": colors[order],
    }


@pytest.mark.parametrize("chart", ["current_rs", "canonical_rs", "log_spd"])
@pytest.mark.parametrize("predictor", ["absolute", "modular_delta"])
@pytest.mark.parametrize("coder_name", ["zlib9", "zstd9"])
def test_cold_round_trip_all_arms(chart: str, predictor: str, coder_name: str) -> None:
    if coder_name == "zstd9":
        pytest.importorskip("zstandard")
    field = _field()
    config = codec.CodecConfig(
        chart=chart,
        geometry_bits=(3, 6, 9),
        predictor=predictor,
        coder=coder_name,
    )
    blob = codec.encode(field, 48, 64, config)
    decoded = codec.decode(blob)
    header = codec.blob_header(blob)
    reference = _ordered_reference(field)

    assert isinstance(decoded, GaussianField)
    assert decoded.n == field.n
    assert decoded.means.device.type == "cpu"
    assert header["chart"] == chart
    assert header["predictor"] == predictor
    assert header["coder"] == coder_name
    assert header["geometry_bits"] == [3, 6, 9]
    assert header["geometry_total_bits"] == 18
    assert torch.isfinite(decoded.means).all()
    assert torch.isfinite(decoded.log_scales).all()
    assert torch.isfinite(decoded.rotations).all()
    assert torch.isfinite(decoded.colors).all()

    # Non-geometry streams are identical decisions in all arms and stay within one uniform bin.
    means_step = (reference["means"].max(axis=0) - reference["means"].min(axis=0)) / (2**12 - 1)
    colors_step = (reference["colors"].max(axis=0) - reference["colors"].min(axis=0)) / (2**8 - 1)
    assert np.all(
        np.abs(decoded.means.numpy() - reference["means"]) <= means_step[None, :] / 2 + 1e-6
    )
    assert np.all(
        np.abs(decoded.colors.numpy() - reference["colors"]) <= colors_step[None, :] / 2 + 1e-6
    )

    original_covariance = core.covariance_entries(
        reference["log_scales"], reference["rotations"]
    )
    decoded_covariance = core.covariance_entries(
        decoded.log_scales.numpy(), decoded.rotations.numpy()
    )
    assert np.isfinite(decoded_covariance).all()
    # This is a serialization test, not an RD gate: coarse asymmetric geometry bits can be lossy.
    assert np.max(np.abs(decoded_covariance - original_covariance)) < 500.0


def test_header_is_fixed_size_and_component_accounting_is_exact_across_charts() -> None:
    field = _field(13)
    blobs = {
        chart: codec.encode(
            field,
            48,
            64,
            codec.CodecConfig(
                chart=chart,
                geometry_bits=(4, 6, 8),
                predictor="modular_delta",
                coder="zlib9",
            ),
        )
        for chart in ("current_rs", "canonical_rs", "log_spd")
    }
    for chart, blob in blobs.items():
        header = codec.blob_header(blob)
        components = codec.blob_components(blob)
        streams = components["streams"]
        assert header["chart"] == chart
        assert header["header_bytes"] == codec.HEADER_BYTES == 100
        assert header["range_bytes"] == codec.RANGE_BYTES == 64
        assert header["range_scalar_bytes"] == 4
        assert components["total_bytes"] == len(blob)
        assert components["header_bytes"] + components["framed_stream_bytes"] == len(blob)
        assert components["framing_bytes"] == 3 * codec.FRAME_BYTES
        assert components["stream_payload_bytes"] == sum(
            stream["payload_bytes"] for stream in streams.values()
        )
        assert streams["means"]["raw_bytes"] == 2 * math.ceil(13 * 12 / 8)
        assert streams["geometry"]["raw_bytes"] == sum(
            math.ceil(13 * bits / 8) for bits in (4, 6, 8)
        )
        assert streams["colors"]["raw_bytes"] == 3 * math.ceil(13 * 8 / 8)


@pytest.mark.parametrize("chart", ["current_rs", "canonical_rs", "log_spd"])
@pytest.mark.parametrize("coder_name", ["zlib9", "zstd9"])
def test_decoded_cold_reencode_is_byte_identical(chart: str, coder_name: str) -> None:
    if coder_name == "zstd9":
        pytest.importorskip("zstandard")
    for predictor in ("absolute", "modular_delta"):
        blob = codec.encode(
            _field(17),
            48,
            64,
            codec.CodecConfig(
                chart=chart,
                geometry_bits=(10, 7, 7),
                predictor=predictor,
                coder=coder_name,
            ),
        )
        assert codec.canonical_reencode(blob) == blob
        assert codec.decoded_reencode(blob) == blob


def test_all_frozen_allocations_are_decoded_field_fixed_points() -> None:
    field = _field(31)
    allocations = [
        (first, second, third)
        for total in (12, 18, 24)
        for first in range(3, 11)
        for second in range(3, 11)
        for third in range(3, 11)
        if first + second + third == total
    ]
    assert len(allocations) == 84
    for chart in ("current_rs", "canonical_rs", "log_spd"):
        for predictor in ("absolute", "modular_delta"):
            for allocation in allocations:
                blob = codec.encode(
                    field,
                    48,
                    64,
                    codec.CodecConfig(
                        chart=chart,
                        geometry_bits=allocation,
                        predictor=predictor,
                        coder="zlib9",
                    ),
                )
                assert codec.decoded_reencode(blob) == blob


def test_log_spd_cycle_search_handles_a_long_float32_transient() -> None:
    rng = np.random.default_rng(404)
    scales = angles = None
    for _ in range(148):
        n = int(rng.integers(1, 65))
        scales = rng.uniform(-8.0, 8.0, (n, 2)).astype(np.float32).astype(np.float64)
        angles = rng.uniform(-20.0, 20.0, n).astype(np.float32).astype(np.float64)
        if n > 3:
            scales[0, 1] = scales[0, 0]
            scales[1, 1] = np.nextafter(
                np.float32(scales[1, 0]), np.float32(np.inf)
            ).astype(np.float64)
    assert scales is not None and angles is not None and scales.shape == (11, 2)

    seen: set[tuple[bytes, bytes, bytes]] = set()
    projected_scales, projected_angles = scales, angles
    for projection_count in range(2_000):
        symbols, lower, upper = codec._quantize_geometry(  # noqa: SLF001
            projected_scales, projected_angles, "log_spd", (6, 3, 3)
        )
        key = (
            symbols.tobytes(),
            np.asarray(lower, dtype="<f4").tobytes(),
            np.asarray(upper, dtype="<f4").tobytes(),
        )
        if key in seen:
            break
        seen.add(key)
        projected_scales, projected_angles = codec._dequantize_geometry_grid(  # noqa: SLF001
            "log_spd", symbols, lower, upper, (6, 3, 3)
        )
        projected_scales = projected_scales.astype(np.float32).astype(np.float64)
        projected_angles = projected_angles.astype(np.float32).astype(np.float64)
    assert projection_count == 853

    stabilized = codec._stabilize_geometry(scales, angles, "log_spd", (6, 3, 3))  # noqa: SLF001
    replay_scales, replay_angles = codec._dequantize_geometry_grid(  # noqa: SLF001
        "log_spd", *stabilized, (6, 3, 3)
    )
    replay = codec._stabilize_geometry(replay_scales, replay_angles, "log_spd", (6, 3, 3))  # noqa: SLF001
    assert all(np.array_equal(left, right) for left, right in zip(stabilized, replay, strict=True))


def test_geometry_predictor_does_not_change_frozen_mean_or_color_symbols() -> None:
    field = _field(31)
    blobs = [
        codec.encode(
            field,
            48,
            64,
            codec.CodecConfig(chart="canonical_rs", predictor=predictor, coder="zlib9"),
        )
        for predictor in ("absolute", "modular_delta")
    ]
    raw_streams = [
        codec._read_frames(blob, codec._decode_header(blob), decompress=True)[0]  # noqa: SLF001
        for blob in blobs
    ]
    assert raw_streams[0][0] == raw_streams[1][0]
    assert raw_streams[0][2] == raw_streams[1][2]
    assert raw_streams[0][1] != raw_streams[1][1]


@pytest.mark.parametrize("chart", ["current_rs", "canonical_rs", "log_spd"])
@pytest.mark.parametrize("predictor", ["absolute", "modular_delta"])
def test_constant_field_round_trip(chart: str, predictor: str) -> None:
    n = 9
    field = GaussianField.from_numpy(
        np.full((n, 2), [7.25, -1.5]),
        np.full((n, 2), [2.0, 2.0]),
        np.full(n, 1.234),
        np.full((n, 3), [0.1, 0.2, 0.3]),
    )
    blob = codec.encode(
        field,
        32,
        32,
        codec.CodecConfig(chart=chart, geometry_bits=(4, 4, 4), predictor=predictor),
    )
    decoded = codec.decode(blob)
    np.testing.assert_array_equal(decoded.means.numpy(), field.means.numpy())
    np.testing.assert_array_equal(decoded.colors.numpy(), field.colors.numpy())
    np.testing.assert_allclose(
        core.covariance_entries(decoded.log_scales.numpy(), decoded.rotations.numpy()),
        core.covariance_entries(field.log_scales.numpy(), field.rotations.numpy()),
        rtol=0.0,
        atol=5e-7,
    )
    header = codec.blob_header(blob)
    assert header["means_lo"] == header["means_hi"]
    assert header["colors_lo"] == header["colors_hi"]
    assert codec.decoded_reencode(blob) == blob


def test_fixed_morton_order_is_stable_for_ties_and_shared_by_all_charts() -> None:
    means = np.array([[1.0, 1.0], [1.0, 1.0], [31.0, 0.0], [0.0, 31.0]])
    order = codec.morton_order(means, 32, 32)
    assert order.tolist()[:2] == [0, 1]
    field = GaussianField.from_numpy(
        means,
        np.array([[2.0, 1.0], [3.0, 0.8], [1.2, 1.1], [0.7, 2.5]]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.arange(12, dtype=np.float64).reshape(4, 3),
    )
    decoded_means = []
    for chart in ("current_rs", "canonical_rs", "log_spd"):
        blob = codec.encode(field, 32, 32, codec.CodecConfig(chart=chart))
        decoded_means.append(codec.decode(blob).means.numpy())
    np.testing.assert_array_equal(decoded_means[0], decoded_means[1])
    np.testing.assert_array_equal(decoded_means[0], decoded_means[2])


def test_corruption_and_truncation_are_rejected() -> None:
    blob = codec.encode(_field(7), 48, 64, codec.CodecConfig())

    bad_magic = bytearray(blob)
    bad_magic[0] ^= 1
    with pytest.raises(ValueError, match="not a gauge-free"):
        codec.decode(bytes(bad_magic))

    bad_header = bytearray(blob)
    bad_header[20] ^= 1
    with pytest.raises(ValueError, match="header CRC"):
        codec.decode(bytes(bad_header))

    bad_payload = bytearray(blob)
    bad_payload[-1] ^= 1
    with pytest.raises(ValueError, match="invalid zlib|CRC mismatch"):
        codec.decode(bytes(bad_payload))

    with pytest.raises(ValueError, match="truncated"):
        codec.decode(blob[:20])
    with pytest.raises(ValueError, match="truncated"):
        codec.decode(blob[:-3])
    with pytest.raises(ValueError, match="trailing"):
        codec.decode(blob + b"x")


@pytest.mark.parametrize(
    "geometry_bits",
    [(3, 3, 3), (2, 5, 5), (3, 3), (11, 3, 4), (3, 4, 6)],
)
def test_invalid_geometry_allocations_are_rejected(geometry_bits: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        codec.CodecConfig(geometry_bits=geometry_bits)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"bits_means": 16}, {"bits_colors": 7}])
def test_frozen_mean_and_color_depths_are_enforced(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="12-bit means and 8-bit colors"):
        codec.CodecConfig(**kwargs)  # type: ignore[arg-type]


def test_unsupported_field_payloads_are_rejected() -> None:
    base = _field(3)
    with_opacity = GaussianField(
        base.means, base.log_scales, base.rotations, base.colors, torch.zeros(3)
    )
    with_gradients = GaussianField(
        base.means,
        base.log_scales,
        base.rotations,
        base.colors,
        color_grads=torch.zeros(3, 2, 3),
    )
    with pytest.raises(ValueError, match="opacity"):
        codec.encode(with_opacity, 48, 64)
    with pytest.raises(ValueError, match="affine"):
        codec.encode(with_gradients, 48, 64)
