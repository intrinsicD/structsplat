import numpy as np
import pytest

from benchmarks import topology_policy_value_core as T


def _rectangle(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    image = np.zeros((T.SIZE, T.SIZE), dtype=np.float32)
    image[y0:y1, x0:x1] = 1.0
    return image


def test_frozen_target_suite_is_complete_hashed_and_analytically_valid():
    first = T.target_suite()
    second = T.target_suite()

    assert len(first) == 24
    assert tuple(first) == tuple(
        f"{family}_v{variant}"
        for family in T.TARGET_FAMILIES
        for variant in T.VARIANTS
    )
    assert T.EXPECTED_SUITE_SHA256 == (
        "52eb4469f1eaa7343d0bc6c9028c1bcd94fe396cedb414e87ab6544610cbe975"
    )
    assert T.suite_sha256(first) == T.EXPECTED_SUITE_SHA256
    assert len(T.EXPECTED_TARGET_HASHES) == 24
    assert len(set(T.EXPECTED_TARGET_HASHES.values())) == 24

    family_counts = {family: 0 for family in T.TARGET_FAMILIES}
    for name, record in first.items():
        family_counts[record["family"]] += 1
        image = record["image"]
        assert image.shape == (64, 64)
        assert image.dtype == np.float32
        assert image.flags.c_contiguous
        assert image.flags.writeable is False
        assert set(np.unique(image)) <= {0.0, 1.0}
        assert record["pixel_sha256"] == T.EXPECTED_TARGET_HASHES[name]
        assert record["pixel_sha256"] == second[name]["pixel_sha256"]
        assert np.array_equal(image, second[name]["image"])
        assert image is not second[name]["image"]
        assert record["split"] == "development"
        assert record["target_id"] == f"bench012_{name}"

        foreground = np.argwhere(image > 0.5)
        assert int(foreground.min()) >= 6
        assert int(foreground[:, 0].max()) <= 57
        assert int(foreground[:, 1].max()) <= 57
        for connectivity in T.FOREGROUND_CONNECTIVITIES:
            observed = T.betti_numbers(image > 0.5, connectivity)
            assert observed == record["expected_betti"]
            assert observed == record["expected_betti_by_foreground_connectivity"][
                connectivity
            ]

    assert family_counts == {family: 4 for family in T.TARGET_FAMILIES}


def test_betti_numbers_use_dual_connectivity_and_one_exterior_component():
    diagonal = np.zeros((5, 5), dtype=np.bool_)
    diagonal[1, 1] = True
    diagonal[2, 2] = True
    assert T.betti_numbers(diagonal, 4) == (2, 0)
    assert T.betti_numbers(diagonal, 8) == (1, 0)

    ring = np.zeros((9, 9), dtype=np.bool_)
    ring[2:7, 2] = True
    ring[2:7, 6] = True
    ring[2, 2:7] = True
    ring[6, 2:7] = True
    assert T.betti_numbers(ring, 4) == (1, 1)
    assert T.betti_numbers(ring, 8) == (1, 1)

    # A foreground wall divides the in-image background, but both sides reach the same explicit
    # zero-valued exterior and therefore do not form a hole.
    boundary_wall = np.zeros((9, 9), dtype=np.bool_)
    boundary_wall[:, 4] = True
    assert T.betti_numbers(boundary_wall, 4) == (1, 0)
    assert T.betti_numbers(boundary_wall, 8) == (1, 0)


def test_anchor_lattice_is_frozen_row_major_and_read_only():
    anchors = T.anchor_lattice()
    assert anchors.shape == (256, 2)
    assert anchors.dtype == np.int16
    assert anchors.flags.writeable is False
    assert tuple(np.unique(anchors[:, 0])) == T.ANCHOR_COORDINATES
    assert tuple(np.unique(anchors[:, 1])) == T.ANCHOR_COORDINATES
    assert tuple(anchors[0]) == (1, 1)
    assert tuple(anchors[-1]) == (61, 61)


@pytest.mark.parametrize("foreground_connectivity", [4, 8])
def test_acpd_identity_preserves_foreground_and_background_evidence(
    foreground_connectivity,
):
    target = T.target_suite()["closed_frame_v0"]["image"]
    target_rgb = np.repeat(target[:, :, None], 3, axis=2)
    result = T.acpd(target, target_rgb, 0.5, foreground_connectivity)

    assert result["total"] == 0.0
    assert result["foreground"]["distance"] == 0.0
    assert result["background"]["distance"] == 0.0
    assert result["foreground"]["xor_count"] == 0
    assert result["background"]["xor_count"] == 0
    assert result["foreground"]["union_count"] > 0
    assert result["background"]["union_count"] > 0
    assert result["target_betti"] == (1, 1)
    assert result["candidate_betti"] == (1, 1)
    assert result["background_connectivity"] == (8 if foreground_connectivity == 4 else 4)


@pytest.mark.parametrize("foreground_connectivity", [4, 8])
def test_acpd_detects_spatial_mismatch_when_global_betti_numbers_tie(
    foreground_connectivity,
):
    target = _rectangle(8, 24, 8, 24)
    displaced = _rectangle(40, 56, 40, 56)

    assert T.betti_numbers(target > 0.5, foreground_connectivity) == (1, 0)
    assert T.betti_numbers(displaced > 0.5, foreground_connectivity) == (1, 0)
    result = T.acpd(target, displaced, 0.5, foreground_connectivity)
    assert result["target_betti"] == result["candidate_betti"] == (1, 0)
    assert result["foreground"]["distance"] > 0.0
    assert result["background"]["distance"] > 0.0
    assert result["total"] > 0.0


def test_stable_selector_requires_one_unique_winner_in_all_six_settings():
    target = T.target_suite()["connected_double_frame_v0"]["image"]
    candidates = {
        7: np.zeros_like(target),
        2: np.roll(target, shift=4, axis=1),
        5: target.copy(),
    }
    result = T.stable_acpd_select(target, candidates)

    assert result["selected_index"] == 5
    assert result["stable"] is True
    assert result["fallback_required"] is False
    assert len(result["settings"]) == 6
    assert all(setting["winner_index"] == 5 for setting in result["settings"])
    assert all(setting["minimum_total"] == 0.0 for setting in result["settings"])
    assert all(setting["unique_margin"] > 0.0 for setting in result["settings"])
    assert [score["candidate_index"] for score in result["settings"][0]["scores"]] == [
        2,
        5,
        7,
    ]

    reversed_result = T.stable_acpd_select(target, dict(reversed(tuple(candidates.items()))))
    assert reversed_result == result


def test_stable_selector_falls_back_on_exact_ties():
    target = T.target_suite()["open_frame_v0"]["image"]
    result = T.stable_acpd_select(target, {11: target.copy(), 3: target.copy()})

    assert result["selected_index"] is None
    assert result["stable"] is False
    assert result["fallback_required"] is True
    assert all(setting["winner_index"] is None for setting in result["settings"])
    assert all(setting["unique_margin"] is None for setting in result["settings"])


def test_stable_selector_rejects_a_better_total_that_trades_foreground_for_background():
    target = T.target_suite()["closed_frame_v0"]["image"]

    # Opening the ring between anchors leaves every foreground anchor relation unchanged but
    # merges the interior and exterior background partitions.  Adding a remote foreground island
    # does the opposite trade: it perturbs foreground relations but is much better on background.
    foreground_match = target.copy()
    foreground_match[10:15, 19] = 0.0
    lower_total_trade = target.copy()
    lower_total_trade[4:7, 4:7] = 1.0

    result = T.stable_acpd_select(
        target,
        {1: foreground_match, 2: lower_total_trade},
    )

    assert result["selected_index"] is None
    assert result["fallback_required"] is True
    for setting in result["settings"]:
        assert setting["unique_total_winner_index"] == 2
        assert setting["foreground_minimizers"] == [1]
        assert setting["background_minimizers"] == [2]
        assert setting["no_class_trade"] is False
        assert setting["winner_index"] is None
        assert setting["rejection_reason"] == "foreground_background_trade"
        assert setting["unique_total_margin"] > 0.0
        assert setting["unique_margin"] is None


def test_stable_selector_falls_back_when_threshold_winner_changes():
    target = _rectangle(8, 24, 8, 24)
    low_threshold_candidate = target * np.float32(0.49)
    high_threshold_candidate = target * np.float32(0.70)
    high_threshold_candidate[40:56, 40:56] = np.float32(0.45)

    result = T.stable_acpd_select(
        target,
        {10: low_threshold_candidate, 20: high_threshold_candidate},
    )
    winners_by_threshold = {
        threshold: {
            setting["winner_index"]
            for setting in result["settings"]
            if setting["threshold"] == threshold
        }
        for threshold in T.THRESHOLDS
    }

    assert winners_by_threshold == {0.4: {10}, 0.5: {20}, 0.6: {20}}
    assert result["selected_index"] is None
    assert result["fallback_required"] is True


def test_stable_selector_falls_back_when_connectivity_winner_changes():
    target = np.zeros((T.SIZE, T.SIZE), dtype=np.float32)
    target[4:8, 4:8] = 1.0
    target[8:12, 8:12] = 1.0

    four_connected_match = target.copy()
    four_connected_match[7, 7] = 0.0
    eight_connected_match = target.copy()
    eight_connected_match[7, 8] = 1.0

    result = T.stable_acpd_select(
        target,
        {4: four_connected_match, 8: eight_connected_match},
    )
    winners_by_connectivity = {
        connectivity: {
            setting["winner_index"]
            for setting in result["settings"]
            if setting["foreground_connectivity"] == connectivity
        }
        for connectivity in T.FOREGROUND_CONNECTIVITIES
    }

    assert winners_by_connectivity == {4: {4}, 8: {8}}
    assert result["selected_index"] is None
    assert result["fallback_required"] is True


def test_topology_public_apis_reject_ambiguous_inputs():
    image = np.zeros((T.SIZE, T.SIZE), dtype=np.float32)
    with pytest.raises(ValueError, match="4 or 8"):
        T.betti_numbers(image, 6)
    with pytest.raises(ValueError, match="two-dimensional"):
        T.betti_numbers(image[:, :, None], 4)
    with pytest.raises(ValueError, match="shape"):
        T.acpd(image[:32], image[:32], 0.5, 4)
    with pytest.raises(ValueError, match="strictly between"):
        T.acpd(image, image, 1.0, 4)
    with pytest.raises(ValueError, match="at least one"):
        T.stable_acpd_select(image, {})
    with pytest.raises(TypeError, match="indices"):
        T.stable_acpd_select(image, {"zero": image})
