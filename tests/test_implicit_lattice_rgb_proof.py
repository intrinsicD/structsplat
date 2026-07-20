from __future__ import annotations

import hashlib

import numpy as np
import pytest

from benchmarks import implicit_lattice_rgb_proof as proof


@pytest.mark.parametrize(
    ("kind", "expected_sha256", "expected_unique"),
    [
        (
            "correlated",
            "5c16fe619aec37e49dbf8e29ab1d6a8e467a6c31d0eb678c279c913ea18a003a",
            8019,
        ),
        (
            "palette",
            "94df89cfc735d982b4a630b4e5f44955c2ec0e8ada47a727b86081f48d135332",
            7435,
        ),
        (
            "uniform",
            "072ca733f038da0cee0db9eb946ade0ca9640d2af2dc348ab97b23db686b8853",
            8192,
        ),
    ],
)
def test_synthetic_fixtures_are_frozen_and_target_free(
    kind: str, expected_sha256: str, expected_unique: int
) -> None:
    first = proof.synthetic_fixture(kind)
    second = proof.synthetic_fixture(kind)
    assert first.shape == (proof.ROW_COUNT, 3)
    assert first.dtype == np.uint8
    assert not first.flags.writeable
    assert np.array_equal(first, second)
    assert hashlib.sha256(first.tobytes(order="C")).hexdigest() == expected_sha256
    assert len(np.unique(first, axis=0)) == expected_unique


def test_integer_ycocg_is_exactly_reversible_on_all_fixture_rows() -> None:
    for kind in ("correlated", "palette", "uniform"):
        rgb = proof.synthetic_fixture(kind)
        transformed = proof.ycocg(rgb)
        reconstructed = proof.inverse_ycocg(transformed[:, 0], transformed[:, 1], transformed[:, 2])
        assert np.array_equal(reconstructed, rgb)


def test_general_empirical_frequency_rule_is_positive_exact_and_deterministic() -> None:
    values = np.arange(proof.ROW_COUNT, dtype=np.uint32) % 17
    first = proof.empirical_frequencies(values, 17)
    second = proof.empirical_frequencies(values, 17)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint32
    assert np.all(first > 0)
    assert int(first.sum()) == proof.TOTAL_FREQUENCY


def test_lattice_descriptor_is_exact_and_constant_channels_keep_binary_alphabet() -> None:
    assert proof.LATTICE_DESCRIPTOR.size == 26
    values = np.zeros(proof.ROW_COUNT, dtype=np.uint32)
    frequencies = proof.empirical_frequencies(values, 2)
    assert frequencies.tolist() == [32767, 1]


@pytest.mark.parametrize("kind", ["missing", "", "CORRELATED"])
def test_unknown_fixture_fails_closed(kind: str) -> None:
    with pytest.raises(proof.LatticeProofError, match="unknown fixture"):
        proof.synthetic_fixture(kind)
