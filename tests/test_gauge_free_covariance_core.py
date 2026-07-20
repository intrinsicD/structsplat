import math

import numpy as np
import pytest

from benchmarks import gauge_free_covariance_core as C


def test_log_spd_chart_is_invariant_to_covariance_gauges() -> None:
    scales = np.array([[1.2, -0.4], [0.1, 0.1], [-0.7, 0.8]], dtype=np.float64)
    theta = np.array([-0.3, 7.2, 1.8], dtype=np.float64)
    base = C.rs_to_log_spd(scales, theta)
    swapped = C.rs_to_log_spd(scales[:, ::-1], theta + math.pi / 2.0)
    wrapped = C.rs_to_log_spd(scales, theta + 5.0 * math.pi)
    np.testing.assert_allclose(swapped, base, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(wrapped, base, rtol=0.0, atol=4e-15)


def test_log_spd_round_trip_preserves_covariance_and_canonicalizes() -> None:
    rng = np.random.default_rng(7)
    scales = rng.uniform(-2.0, 3.0, size=(1000, 2))
    theta = rng.uniform(-20.0, 20.0, size=1000)
    decoded_scales, decoded_theta = C.log_spd_to_rs(C.rs_to_log_spd(scales, theta))
    np.testing.assert_allclose(
        C.covariance_entries(decoded_scales, decoded_theta),
        C.covariance_entries(scales, theta),
        rtol=2e-14,
        atol=2e-12,
    )
    assert np.all(decoded_scales[:, 0] >= decoded_scales[:, 1])
    assert np.all((0.0 <= decoded_theta) & (decoded_theta < math.pi))


def test_isotropy_maps_to_one_point_and_zero_decoder_angle() -> None:
    scales = np.full((32, 2), 0.75)
    theta = np.linspace(-100.0, 100.0, 32)
    coords = C.rs_to_log_spd(scales, theta)
    np.testing.assert_array_equal(coords[:, 0], np.full(32, 1.5))
    np.testing.assert_allclose(coords[:, 1:], 0.0, rtol=0.0, atol=0.0)
    decoded_scales, decoded_theta = C.log_spd_to_rs(coords)
    np.testing.assert_array_equal(decoded_scales, scales)
    np.testing.assert_array_equal(decoded_theta, np.zeros(32))


def test_chart_entries_are_exact_log_covariance_coordinates() -> None:
    scales = np.array([[0.2, -0.7], [1.0, 1.0], [-1.2, 0.4]])
    theta = np.array([0.3, 2.4, -0.8])
    chart = C.rs_to_log_spd(scales, theta)
    entries = C.log_covariance_entries(chart)
    for index in range(len(scales)):
        c, s = math.cos(theta[index]), math.sin(theta[index])
        rotation = np.array([[c, -s], [s, c]])
        expected = rotation @ np.diag(2.0 * scales[index]) @ rotation.T
        np.testing.assert_allclose(
            entries[index], [expected[0, 0], expected[0, 1], expected[1, 1]], atol=2e-15
        )


@pytest.mark.parametrize("bits", [1, 4, 8, 9, 16, 24])
def test_bitpack_and_modular_prediction_round_trip(bits: int) -> None:
    rng = np.random.default_rng(bits)
    symbols = rng.integers(0, 1 << bits, size=(137, 3), dtype=np.uint32)
    deltas = C.modular_delta(symbols, bits)
    np.testing.assert_array_equal(C.modular_undelta(deltas, bits), symbols)
    payload = C.bitpack(deltas.T, bits)
    unpacked = C.bitunpack(payload, bits, deltas.size).reshape(3, 137).T
    np.testing.assert_array_equal(C.modular_undelta(unpacked, bits), symbols)
    assert len(payload) == math.ceil(symbols.size * bits / 8)


def test_linear_quantizer_handles_constant_coordinate_exactly() -> None:
    values = np.array([[1.0, -2.0], [3.0, -2.0], [2.0, -2.0]])
    q, lo, hi = C.quantize_linear(values, 5)
    restored = C.dequantize_linear(q, lo, hi, 5)
    np.testing.assert_array_equal(q[:, 1], np.zeros(3, dtype=np.uint32))
    np.testing.assert_array_equal(restored[:, 1], values[:, 1])


def test_shape_and_nonfinite_guards() -> None:
    with pytest.raises(ValueError):
        C.rs_to_log_spd(np.zeros((3, 3)), np.zeros(3))
    with pytest.raises(ValueError):
        C.log_spd_to_rs(np.array([[0.0, math.nan, 0.0]]))
    with pytest.raises(ValueError):
        C.bitunpack(b"", 6, 1)
