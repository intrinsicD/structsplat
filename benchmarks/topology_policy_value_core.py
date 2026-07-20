"""Deterministic data and digital-topology core for BENCH-012.

This module intentionally contains no fitter, renderer, or policy-recovery code.  It freezes the
synthetic target suite and the spatial-connectivity statistic that the benchmark runner consumes.
All topology operations are implemented with NumPy so the reference path does not depend on a
platform-specific connected-component library.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import math
from typing import Any

import numpy as np


SIZE = 64
THRESHOLDS = (0.4, 0.5, 0.6)
FOREGROUND_CONNECTIVITIES = (4, 8)
ANCHOR_COORDINATES = tuple(range(1, SIZE, 4))
TARGET_FAMILIES = (
    "connected_dumbbell",
    "separated_pair",
    "closed_frame",
    "open_frame",
    "connected_double_frame",
    "theta_frame",
)
VARIANTS = (0, 1, 2, 3)

ANALYTIC_BETTI_BY_FAMILY = {
    "connected_dumbbell": (1, 0),
    "separated_pair": (2, 0),
    "closed_frame": (1, 1),
    "open_frame": (1, 0),
    "connected_double_frame": (1, 2),
    "theta_frame": (1, 2),
}

# Filled after the generator was frozen.  These hashes include dtype and shape in addition to the
# decoded pixel bytes, so a silent representation change is treated as target drift.
EXPECTED_TARGET_HASHES = {
    "connected_dumbbell_v0": "f39f3f8b0e6ce35334a3d2563766ece46fbbaa4731123b0287573e087239ad9a",
    "connected_dumbbell_v1": "0ff74ff99b6fa71852c5810f26c9848959e21d24364e532a13904a775ff17d77",
    "connected_dumbbell_v2": "38740b47807489e428fb90edd8a664454b7ebd2861a8917bce0ac5fc07c70427",
    "connected_dumbbell_v3": "97b36cd1356a6753697bd18c1982d8cdc34aa5bfc950841e5f614e2356a9c4d2",
    "separated_pair_v0": "438bac2bb047915714587ec7deb082f4003c9de7dd5f3975fb429b7a3cae8d95",
    "separated_pair_v1": "038e6e678bd7b2a97906e1409440f75b87591ce3d9b2125d43166e20156f187b",
    "separated_pair_v2": "76bf387b0eb4d2bf8ca290a0a933c13752bc98e919dc5d4c0fb1759964012e1e",
    "separated_pair_v3": "b2ede53dfdda042009ed1890392961e2075e03b45bbe32a6dd211ba51c75cf7c",
    "closed_frame_v0": "534cd1e4d20f992c76192a4ad38481c093af391fadc666c8190f326def5d632c",
    "closed_frame_v1": "791f1c52d089bc2c322257bcb5d39e9b49c8502e53753e5f0ebc776b9cd9ae35",
    "closed_frame_v2": "6a0fe1ca7eeb6526e93d81d8e1f5e67b428bdda6b5fb9fd1de6428ab1e563b52",
    "closed_frame_v3": "95b960a92a7a418be37b723ae94cecf5b99fc58de0be3c42e29bcff58bb8f4cd",
    "open_frame_v0": "3cc5709bcd934b1239fd7d7b578a2097f6dc5f97cc467516cac7b5ae8bb21f4f",
    "open_frame_v1": "ab0e22693d6dc04c8235a830cdf5f4dfe88721e8c65503175c33fd6eda31e4d8",
    "open_frame_v2": "eef162edbaeddf4af9c0e5345a81e77479784260b2c415f1d82577cfdc9e29ac",
    "open_frame_v3": "293c963643a991c6c2b6818822af96a931fbfc1b526adb2ffc1d121a7d017ef6",
    "connected_double_frame_v0": "b2bc8408bc076722a461a5f0f742c9c5821b8a6ac52daf2bbdf341136815c23b",
    "connected_double_frame_v1": "cc89445a8fe3090bbf0f0f9073f6b62a30fe607d11a15cb12966547ec30f8c45",
    "connected_double_frame_v2": "0a95e8dbb45bf87647e305162cf96ec19113faa3cdea4a18dca24f09f5053905",
    "connected_double_frame_v3": "b294b23fa9382af3c5603878b89b1b1483c4535dc75a6a96103ea4ba75c67e94",
    "theta_frame_v0": "96f4503a31d43546ed72ef0b64df11d254592ad2be2ab8e49a011544dbe1f8b9",
    "theta_frame_v1": "b42e2700bb33131196d25aec16e2325b33d0e068cdc1dcf5d53e2d50af0782a6",
    "theta_frame_v2": "24f2677c3d021ef88d90ca120b2ed5d9b6f20c27bc411add7eb8f46b0ac772ef",
    "theta_frame_v3": "1c11cca35012aee953dee9503f7a8ead41f1c5870e20d17fa366cc811659eb56",
}
EXPECTED_SUITE_SHA256 = "52eb4469f1eaa7343d0bc6c9028c1bcd94fe396cedb414e87ab6544610cbe975"


def _validate_connectivity(connectivity: int) -> int:
    if isinstance(connectivity, bool) or connectivity not in FOREGROUND_CONNECTIVITIES:
        raise ValueError("connectivity must be the integer 4 or 8")
    return int(connectivity)


def _dual_connectivity(foreground_connectivity: int) -> int:
    return 8 if _validate_connectivity(foreground_connectivity) == 4 else 4


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def array_sha256(image: np.ndarray) -> str:
    """Hash an array's decoded values together with its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(image))
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)}))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def anchor_lattice() -> np.ndarray:
    """Return the frozen 16x16 row-major ``(y, x)`` anchor lattice."""

    anchors = np.asarray(
        [(y, x) for y in ANCHOR_COORDINATES for x in ANCHOR_COORDINATES],
        dtype=np.int16,
    )
    anchors.setflags(write=False)
    return anchors


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if _validate_connectivity(connectivity) == 4:
        return ((-1, 0), (0, -1), (0, 1), (1, 0))
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def _component_labels(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=np.bool_)
    if binary.ndim != 2:
        raise ValueError("component masks must be two-dimensional")
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    offsets = _neighbor_offsets(connectivity)
    component_count = 0

    for start_y, start_x in np.argwhere(binary):
        y = int(start_y)
        x = int(start_x)
        if labels[y, x] != 0:
            continue
        component_count += 1
        labels[y, x] = component_count
        queue: deque[tuple[int, int]] = deque([(y, x)])
        while queue:
            current_y, current_x = queue.popleft()
            for delta_y, delta_x in offsets:
                neighbor_y = current_y + delta_y
                neighbor_x = current_x + delta_x
                if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                    continue
                if not binary[neighbor_y, neighbor_x] or labels[neighbor_y, neighbor_x] != 0:
                    continue
                labels[neighbor_y, neighbor_x] = component_count
                queue.append((neighbor_y, neighbor_x))
    return labels, component_count


def betti_numbers(mask: np.ndarray, foreground_connectivity: int) -> tuple[int, int]:
    """Return ``(beta0, beta1)`` under a legal dual digital-topology convention.

    Foreground uses ``foreground_connectivity`` and background uses the complementary 4/8
    connectivity.  A one-pixel zero-valued exterior is included, so every background region that
    reaches any image boundary belongs to the single exterior component.  ``beta1`` is therefore
    the number of remaining background components.
    """

    binary = np.asarray(mask, dtype=np.bool_)
    if binary.ndim != 2:
        raise ValueError("Betti masks must be two-dimensional")
    foreground_connectivity = _validate_connectivity(foreground_connectivity)
    _foreground_labels, beta0 = _component_labels(binary, foreground_connectivity)

    padded_foreground = np.pad(binary, 1, mode="constant", constant_values=False)
    padded_background = np.logical_not(padded_foreground)
    _background_labels, background_components = _component_labels(
        padded_background,
        _dual_connectivity(foreground_connectivity),
    )
    if background_components < 1:
        raise RuntimeError("the explicit exterior background component is missing")
    return int(beta0), int(background_components - 1)


def _empty_target() -> np.ndarray:
    return np.zeros((SIZE, SIZE), dtype=np.bool_)


def _fill_rect(
    mask: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    value: bool = True,
) -> None:
    if not (0 <= y0 < y1 <= SIZE and 0 <= x0 < x1 <= SIZE):
        raise ValueError("invalid half-open rectangle")
    mask[y0:y1, x0:x1] = value


def _frame(y0: int, y1: int, x0: int, x1: int, thickness: int) -> np.ndarray:
    if 2 * thickness >= min(y1 - y0, x1 - x0):
        raise ValueError("frame thickness leaves no interior")
    mask = _empty_target()
    _fill_rect(mask, y0, y1, x0, x1)
    _fill_rect(
        mask,
        y0 + thickness,
        y1 - thickness,
        x0 + thickness,
        x1 - thickness,
        value=False,
    )
    return mask


def _connected_dumbbell(variant: int) -> np.ndarray:
    rectangles = (
        ((16, 38, 8, 23), (18, 40, 41, 56), (25, 31, 21, 43)),
        ((12, 33, 10, 26), (20, 43, 40, 54), (25, 31, 24, 42)),
        ((8, 23, 19, 42), (41, 56, 21, 44), (21, 43, 28, 34)),
        ((22, 46, 8, 25), (12, 36, 39, 56), (28, 34, 23, 41)),
    )[variant]
    mask = _empty_target()
    for rectangle in rectangles:
        _fill_rect(mask, *rectangle)
    return mask


def _separated_pair(variant: int) -> np.ndarray:
    rectangles = (
        ((16, 39, 8, 25), (18, 41, 39, 56)),
        ((8, 24, 18, 42), (40, 56, 20, 44)),
        ((9, 27, 8, 27), (37, 56, 37, 56)),
        ((12, 33, 10, 29), (31, 52, 39, 56)),
    )[variant]
    mask = _empty_target()
    for rectangle in rectangles:
        _fill_rect(mask, *rectangle)
    return mask


def _closed_frame(variant: int) -> np.ndarray:
    parameters = (
        (10, 54, 12, 52, 5),
        (8, 48, 16, 56, 4),
        (14, 56, 8, 48, 6),
        (9, 55, 9, 55, 5),
    )[variant]
    return _frame(*parameters)


def _open_frame(variant: int) -> np.ndarray:
    frame_parameters, opening = (
        ((10, 54, 12, 52, 5), (10, 15, 28, 36)),
        ((12, 52, 10, 57, 5), (27, 37, 52, 57)),
        ((8, 56, 14, 50, 6), (50, 56, 28, 36)),
        ((11, 53, 8, 55, 4), (26, 36, 8, 12)),
    )[variant]
    mask = _frame(*frame_parameters)
    _fill_rect(mask, *opening, value=False)
    return mask


def _connected_double_frame(variant: int) -> np.ndarray:
    left, right, bridge = (
        ((14, 50, 8, 29, 4), (14, 50, 35, 56, 4), (29, 35, 27, 37)),
        ((9, 43, 10, 31, 5), (19, 55, 35, 56, 5), (27, 33, 29, 37)),
        ((8, 29, 15, 51, 4), (35, 56, 15, 51, 4), (27, 37, 30, 36)),
        ((12, 53, 8, 30, 4), (10, 55, 36, 56, 6), (30, 36, 28, 38)),
    )[variant]
    mask = np.logical_or(_frame(*left), _frame(*right))
    _fill_rect(mask, *bridge)
    return mask


def _theta_frame(variant: int) -> np.ndarray:
    frame_parameters, divider = (
        ((10, 54, 10, 54, 5), (29, 35, 10, 54)),
        ((8, 56, 14, 50, 4), (25, 31, 14, 50)),
        ((12, 52, 8, 56, 5), (12, 52, 29, 35)),
        ((9, 55, 9, 55, 6), (34, 40, 9, 55)),
    )[variant]
    mask = _frame(*frame_parameters)
    _fill_rect(mask, *divider)
    return mask


_TARGET_GENERATORS = {
    "connected_dumbbell": _connected_dumbbell,
    "separated_pair": _separated_pair,
    "closed_frame": _closed_frame,
    "open_frame": _open_frame,
    "connected_double_frame": _connected_double_frame,
    "theta_frame": _theta_frame,
}


def _build_target_suite_unchecked() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for family in TARGET_FAMILIES:
        expected_betti = ANALYTIC_BETTI_BY_FAMILY[family]
        for variant in VARIANTS:
            name = f"{family}_v{variant}"
            binary = _TARGET_GENERATORS[family](variant)
            image = np.ascontiguousarray(binary.astype(np.float32))
            observed = {
                connectivity: betti_numbers(binary, connectivity)
                for connectivity in FOREGROUND_CONNECTIVITIES
            }
            if any(value != expected_betti for value in observed.values()):
                raise RuntimeError(
                    f"analytic topology mismatch for {name}: "
                    f"expected={expected_betti}, observed={observed}"
                )
            image.setflags(write=False)
            records[name] = {
                "target_id": f"bench012_{name}",
                "family": family,
                "variant": variant,
                "split": "development",
                "image": image,
                "expected_betti": expected_betti,
                "expected_betti_by_foreground_connectivity": observed,
                "pixel_sha256": array_sha256(image),
            }
    return records


def _suite_manifest(records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "target_id": str(record["target_id"]),
            "family": str(record["family"]),
            "variant": int(record["variant"]),
            "split": str(record["split"]),
            "expected_betti": list(record["expected_betti"]),
            "pixel_sha256": str(record["pixel_sha256"]),
        }
        for name, record in records.items()
    ]


def suite_sha256(records: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """Return a canonical hash of target identity, topology metadata, and pixel hashes."""

    selected = _build_target_suite_unchecked() if records is None else records
    return hashlib.sha256(_canonical_json(_suite_manifest(selected))).hexdigest()


def target_suite() -> dict[str, dict[str, Any]]:
    """Return the frozen 24-target BENCH-012 development suite."""

    records = _build_target_suite_unchecked()
    actual_hashes = {name: record["pixel_sha256"] for name, record in records.items()}
    if actual_hashes != EXPECTED_TARGET_HASHES:
        raise RuntimeError(
            "BENCH-012 target hashes drifted: "
            + json.dumps(
                {"actual": actual_hashes, "expected": EXPECTED_TARGET_HASHES},
                sort_keys=True,
            )
        )
    actual_suite_hash = suite_sha256(records)
    if actual_suite_hash != EXPECTED_SUITE_SHA256:
        raise RuntimeError(
            "BENCH-012 suite manifest hash drifted: "
            f"expected={EXPECTED_SUITE_SHA256}, actual={actual_suite_hash}"
        )
    return records


def _as_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array.astype(np.float64, copy=False)
    elif array.ndim == 3 and array.shape[2] in (1, 3):
        gray = array.astype(np.float64, copy=False).mean(axis=2)
    else:
        raise ValueError("images must have shape (H,W), (H,W,1), or (H,W,3)")
    if gray.shape != (SIZE, SIZE):
        raise ValueError(f"BENCH-012 images must have shape ({SIZE},{SIZE})")
    if not np.isfinite(gray).all():
        raise ValueError("images must contain only finite values")
    return np.ascontiguousarray(gray)


def _threshold_mask(image: np.ndarray, threshold: float) -> np.ndarray:
    numeric_threshold = float(threshold)
    if not math.isfinite(numeric_threshold) or not 0.0 < numeric_threshold < 1.0:
        raise ValueError("threshold must be finite and strictly between zero and one")
    return np.ascontiguousarray(_as_gray(image) >= numeric_threshold)


def _anchor_relations(
    foreground: np.ndarray,
    foreground_connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    foreground_connectivity = _validate_connectivity(foreground_connectivity)
    anchors = anchor_lattice().astype(np.int64, copy=False)
    anchor_y = anchors[:, 0]
    anchor_x = anchors[:, 1]

    foreground_labels, _count = _component_labels(foreground, foreground_connectivity)
    foreground_anchor_labels = foreground_labels[anchor_y, anchor_x]
    foreground_membership = foreground[anchor_y, anchor_x]
    foreground_relations = np.logical_and(
        np.logical_and(
            foreground_membership[:, None],
            foreground_membership[None, :],
        ),
        foreground_anchor_labels[:, None] == foreground_anchor_labels[None, :],
    )

    padded_foreground = np.pad(foreground, 1, mode="constant", constant_values=False)
    padded_background = np.logical_not(padded_foreground)
    background_labels, _count = _component_labels(
        padded_background,
        _dual_connectivity(foreground_connectivity),
    )
    background_anchor_labels = background_labels[anchor_y + 1, anchor_x + 1]
    background_membership = np.logical_not(foreground_membership)
    background_relations = np.logical_and(
        np.logical_and(
            background_membership[:, None],
            background_membership[None, :],
        ),
        background_anchor_labels[:, None] == background_anchor_labels[None, :],
    )
    return foreground_relations, background_relations


def _relation_difference(
    target_relation: np.ndarray,
    candidate_relation: np.ndarray,
) -> dict[str, int | float]:
    xor_count = int(np.logical_xor(target_relation, candidate_relation).sum())
    union_count = int(np.logical_or(target_relation, candidate_relation).sum())
    distance = 0.0 if union_count == 0 else xor_count / union_count
    return {
        "distance": float(distance),
        "xor_count": xor_count,
        "union_count": union_count,
        "target_relation_count": int(target_relation.sum()),
        "candidate_relation_count": int(candidate_relation.sum()),
    }


def _acpd_from_masks(
    target_foreground: np.ndarray,
    candidate_foreground: np.ndarray,
    foreground_connectivity: int,
) -> dict[str, Any]:
    target_fg, target_bg = _anchor_relations(target_foreground, foreground_connectivity)
    candidate_fg, candidate_bg = _anchor_relations(candidate_foreground, foreground_connectivity)
    foreground = _relation_difference(target_fg, candidate_fg)
    background = _relation_difference(target_bg, candidate_bg)
    total = float(foreground["distance"] + background["distance"])
    return {
        "foreground": foreground,
        "background": background,
        "total": total,
        "target_betti": betti_numbers(target_foreground, foreground_connectivity),
        "candidate_betti": betti_numbers(candidate_foreground, foreground_connectivity),
    }


def acpd(
    target_rgb_or_gray: np.ndarray,
    candidate_rgb_or_gray: np.ndarray,
    threshold: float,
    foreground_connectivity: int,
) -> dict[str, Any]:
    """Compute foreground/background anchored connectivity partition distance."""

    foreground_connectivity = _validate_connectivity(foreground_connectivity)
    target_foreground = _threshold_mask(target_rgb_or_gray, threshold)
    candidate_foreground = _threshold_mask(candidate_rgb_or_gray, threshold)
    result = _acpd_from_masks(
        target_foreground,
        candidate_foreground,
        foreground_connectivity,
    )
    return {
        "threshold": float(threshold),
        "foreground_connectivity": foreground_connectivity,
        "background_connectivity": _dual_connectivity(foreground_connectivity),
        **result,
    }


def _exact_total_score(score: Mapping[str, Any]) -> Fraction:
    total = Fraction(0, 1)
    for class_name in ("foreground", "background"):
        values = score[class_name]
        denominator = int(values["union_count"])
        if denominator:
            total += Fraction(int(values["xor_count"]), denominator)
    return total


def _exact_class_score(score: Mapping[str, Any], class_name: str) -> Fraction:
    values = score[class_name]
    denominator = int(values["union_count"])
    if denominator == 0:
        return Fraction(0, 1)
    return Fraction(int(values["xor_count"]), denominator)


def stable_acpd_select(
    target: np.ndarray,
    candidate_images_by_index: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """Select only a unique ACPD winner shared by all six frozen evaluations.

    A tied setting, a threshold-dependent winner, or a connectivity-dependent winner returns
    ``selected_index=None``.  The benchmark runner is responsible for applying the preregistered
    immediate-objective fallback.
    """

    if not candidate_images_by_index:
        raise ValueError("at least one candidate image is required")
    candidate_indices = tuple(sorted(candidate_images_by_index))
    if any(isinstance(index, bool) or not isinstance(index, int) for index in candidate_indices):
        raise TypeError("candidate indices must be integers")

    target_gray = _as_gray(target)
    candidate_gray = {
        index: _as_gray(candidate_images_by_index[index]) for index in candidate_indices
    }
    settings: list[dict[str, Any]] = []
    winners: list[int | None] = []

    for threshold in THRESHOLDS:
        target_foreground = np.ascontiguousarray(target_gray >= threshold)
        candidate_foregrounds = {
            index: np.ascontiguousarray(candidate_gray[index] >= threshold)
            for index in candidate_indices
        }
        for foreground_connectivity in FOREGROUND_CONNECTIVITIES:
            score_records: list[dict[str, Any]] = []
            exact_scores: dict[int, Fraction] = {}
            exact_foreground_scores: dict[int, Fraction] = {}
            exact_background_scores: dict[int, Fraction] = {}
            for index in candidate_indices:
                score = _acpd_from_masks(
                    target_foreground,
                    candidate_foregrounds[index],
                    foreground_connectivity,
                )
                exact_scores[index] = _exact_total_score(score)
                exact_foreground_scores[index] = _exact_class_score(score, "foreground")
                exact_background_scores[index] = _exact_class_score(score, "background")
                score_records.append({"candidate_index": index, **score})

            minimum = min(exact_scores.values())
            total_minimizers = [
                index for index in candidate_indices if exact_scores[index] == minimum
            ]
            minimum_foreground = min(exact_foreground_scores.values())
            foreground_minimizers = [
                index
                for index in candidate_indices
                if exact_foreground_scores[index] == minimum_foreground
            ]
            minimum_background = min(exact_background_scores.values())
            background_minimizers = [
                index
                for index in candidate_indices
                if exact_background_scores[index] == minimum_background
            ]
            unique_total_winner = (
                total_minimizers[0] if len(total_minimizers) == 1 else None
            )
            no_class_trade = bool(
                unique_total_winner is not None
                and unique_total_winner in foreground_minimizers
                and unique_total_winner in background_minimizers
            )
            winner = unique_total_winner if no_class_trade else None
            winners.append(winner)
            sorted_exact = sorted(set(exact_scores.values()))
            unique_total_margin = None
            if unique_total_winner is not None and len(sorted_exact) > 1:
                unique_total_margin = float(sorted_exact[1] - sorted_exact[0])
            if unique_total_winner is None:
                rejection_reason = "total_tie"
            elif not no_class_trade:
                rejection_reason = "foreground_background_trade"
            else:
                rejection_reason = None
            settings.append(
                {
                    "threshold": threshold,
                    "foreground_connectivity": foreground_connectivity,
                    "background_connectivity": _dual_connectivity(foreground_connectivity),
                    "winner_index": winner,
                    "unique_total_winner_index": unique_total_winner,
                    "no_class_trade": no_class_trade,
                    "rejection_reason": rejection_reason,
                    "minimum_total": float(minimum),
                    "minimum_foreground": float(minimum_foreground),
                    "minimum_background": float(minimum_background),
                    "total_minimizers": total_minimizers,
                    "foreground_minimizers": foreground_minimizers,
                    "background_minimizers": background_minimizers,
                    "unique_total_margin": unique_total_margin,
                    "unique_margin": unique_total_margin if winner is not None else None,
                    "scores": score_records,
                }
            )

    stable_winner: int | None = None
    if all(winner is not None for winner in winners) and len(set(winners)) == 1:
        stable_winner = winners[0]
    return {
        "selected_index": stable_winner,
        "stable": stable_winner is not None,
        "fallback_required": stable_winner is None,
        "thresholds": list(THRESHOLDS),
        "foreground_connectivities": list(FOREGROUND_CONNECTIVITIES),
        "anchor_coordinates": list(ANCHOR_COORDINATES),
        "target_sha256": array_sha256(target_gray),
        "candidate_sha256_by_index": {
            index: array_sha256(candidate_gray[index]) for index in candidate_indices
        },
        "settings": settings,
    }
