"""BENCH-009 Stage-1 manifest and numerical preflight.

This module freezes the development workload and validates the numerical paths needed by the
residual tangent auction.  It deliberately does *not* execute scientific cells or recovery.  A
passing preflight therefore keeps ``development_run_authorized`` false until a separate runner is
implemented and audited.

The manifest maintains two ledgers:

* a causal ledger comparing the unpriced full current tangent with joint ``J + extension``
  actions; and
* a matched ledger comparing four six-coefficient actions.

Literal unregularized projector energy is kept separate from damped, clipped trust-step
predictions.  Finite births use an affine-action gain because their denominator offset means they
are not a zero-offset linear subspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable

import numpy as np
import torch

from benchmarks.residual_tangent_auction import (
    _render_field_reference,
    _residualize,
    _scaled_svd,
    _solve_svd,
)
from structsplat.gaussians import GaussianField


PROTOCOL = "BENCH-009 residual tangent auction Stage 1 manifest/numerical preflight"
SCOPE = "stage1_development_killing_screen_manifest_preflight_only"
SIZE = 64
PARENT_COUNT = 64
SEEDS = (101, 211)
PARENT_HORIZONS = (("underfit", 24), ("near_plateau", 160))
TILE_SIZE = 8
FOLDS = (0, 1)
DAMPINGS = (1e-3, 1e-2)
RCONDS = (1e-5, 1e-4)
TRUST_RADII = (0.25, 0.75)
TRUST_NORM = "linf"
DAMPING_COORDINATE_SYSTEM = "dimensionless_parent_chart"
RECOVERY_STEPS = (20, 100)
PRIMARY_DAMPING = DAMPINGS[0]
PRIMARY_RCOND = RCONDS[0]
SIGMA_CUTOFF = 3.0
JVP_EPSILON = 1e-4
JVP_RELATIVE_TOLERANCE = 1e-3
RENDER_ABSOLUTE_TOLERANCE = 1e-10
NUMERICAL_REPEATABILITY_ABSOLUTE_TOLERANCE = 1e-12
GAIN_RATIO_BOUNDS = (0.5, 2.0)
SPEARMAN_MINIMUM = 0.8
PSNR_SURVIVAL_DB = 0.15
ENERGY_SURVIVAL_RATIO = 1.25
MINIMUM_POSITIVE_FAMILIES = 4
PRICE_SCOPE = "diagnostic_fixed_width_fields_not_actual_bytes_or_compression"

TARGET_FAMILIES = (
    "ramp_shading",
    "step_junction",
    "thin_curve",
    "stationary_sinusoid",
    "chirp",
    "natural_texture",
)

CAUSAL_ACTIONS = (
    "full_j",
    "j_plus_affine6",
    "j_plus_carrier6",
    "j_plus_birth2_rgb6",
)
MATCHED_ACTIONS = (
    "tangent6",
    "affine6",
    "carrier6",
    "birth2_rgb6",
)
ACTION_LEDGERS = {
    "causal": CAUSAL_ACTIONS,
    "matched": MATCHED_ACTIONS,
}
SELECTORS = ("tangent6", "affine", "carrier", "birth_pair")
CARRIER_ACTIONS = ("j_plus_carrier6", "carrier6")

# Filled from the exact generators below.  These constants turn accidental target drift into a
# preflight failure rather than silently creating a new development screen.
EXPECTED_TARGET_HASHES: dict[str, str] = {
    "ramp_shading": "45933d8c4bbce7e9086c8557f65cb81b746d36420645ad289ca7214d85eeefcf",
    "step_junction": "02fbf7769e3b82435dde3714b82f0c49396b8ffe6ecbf4c813b8f12911c1d8a9",
    "thin_curve": "3009297cb74d0877152d7243f81ff5505ec487916c22107be81a529e0ba4bc50",
    "stationary_sinusoid": (
        "f6d0fc49b0ede2798c030f69b4a3a42ad208521b9586b2281b433d9a718cbc2b"
    ),
    "chirp": "fc7c70032557fca3b1ea58d05334e980610e2cbbcf667b2e7910a1659e06072e",
    "natural_texture": "4ea876339579f8e12a7c60a64d7cf3c69cfa6d14b1bb71a0fd6867d8c804465b",
}

FIT020_MANIFEST_SHA256 = "4e51518b9c2b1816311f1b2856c19ea77acaea913b534987fa776cf160740ebe"
COMP006_MANIFEST_SHA256 = "a1ea1ea5be41e36c3e4a8557d01ce721167545d680a6945522c935b832f60f0e"

CRITICAL_SOURCE_PATHS = (
    "benchmarks/residual_tangent_auction_stage1.py",
    "benchmarks/residual_tangent_auction_stage1_core.py",
    "benchmarks/residual_tangent_auction_stage1_runner.py",
    "benchmarks/residual_tangent_auction_stage1_science.py",
    "benchmarks/residual_tangent_auction_stage1_analysis.py",
    "benchmarks/residual_tangent_auction_stage1_recovery.py",
    "benchmarks/residual_tangent_auction_matrixfree.py",
    "benchmarks/residual_tangent_auction.py",
    "benchmarks/perturb_recover_spectroscopy.py",
    "benchmarks/marginal_cold_stream_rd.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/render.py",
    "tasks/BENCH-009-residual-tangent-auction.md",
    "tests/test_residual_tangent_auction_stage1.py",
    "tests/test_residual_tangent_auction_stage1_core.py",
    "tests/test_residual_tangent_auction_stage1_runner.py",
    "tests/test_residual_tangent_auction_stage1_science.py",
    "tests/test_residual_tangent_auction_stage1_analysis.py",
    "tests/test_residual_tangent_auction_stage1_recovery.py",
    "tests/test_residual_tangent_auction_matrixfree.py",
    "pyproject.toml",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    array = np.ascontiguousarray(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    )
    prefix = _canonical_json({"dtype": str(array.dtype), "shape": list(array.shape)})
    return _sha256_bytes(prefix + b"\0" + array.tobytes(order="C"))


def _hash_worktree_entry(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        kind = "symlink_target"
    else:
        payload = path.read_bytes()
        kind = "file_bytes"
    return {"kind": kind, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _collect_provenance() -> dict[str, Any]:
    """Bind the preflight to raw critical sources and the complete dirty worktree."""
    repo_root = Path(__file__).resolve().parents[1]
    critical_sources = {
        relative: _hash_worktree_entry(repo_root / relative)
        for relative in CRITICAL_SOURCE_PATHS
    }
    status = _git_bytes(
        repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    tracked_diff = _git_bytes(repo_root, "diff", "--binary", "HEAD", "--", ".")
    untracked_raw = _git_bytes(
        repo_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_paths = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked_raw.split(b"\0")
        if item
    )
    untracked_files = {
        relative: _hash_worktree_entry(repo_root / relative)
        for relative in untracked_paths
    }
    return {
        "git_commit": _git_bytes(repo_root, "rev-parse", "HEAD").decode("ascii").strip(),
        "git_branch": (
            _git_bytes(repo_root, "branch", "--show-current").decode("utf-8").strip()
            or None
        ),
        "git_dirty": bool(status),
        "git_status_porcelain_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "tracked_diff_size_bytes": len(tracked_diff),
        "critical_source_hash_mode": "raw_worktree_bytes_independent_of_git_index",
        "critical_sources": critical_sources,
        "untracked_file_count": len(untracked_files),
        "untracked_manifest_sha256": _sha256_bytes(_canonical_json(untracked_files)),
        "untracked_files": untracked_files,
    }


def _target_coordinates() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    return (xx + 0.5) / float(SIZE), (yy + 0.5) / float(SIZE)


def _thin_bezier_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 257, dtype=np.float64)
    omt = 1.0 - t
    x = omt * omt * 0.09 + 2.0 * omt * t * 0.83 + t * t * 0.91
    y = omt * omt * 0.76 + 2.0 * omt * t * 0.08 + t * t * 0.66
    distance = np.full_like(u, np.inf, dtype=np.float64)
    for start in range(0, len(t), 32):
        dx = u[..., None] - x[start : start + 32]
        dy = v[..., None] - y[start : start + 32]
        distance = np.minimum(distance, np.sqrt(np.min(dx * dx + dy * dy, axis=2)))
    return distance


def _natural_texture() -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import skimage
        from skimage import data
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "BENCH-009 Stage 1 requires scikit-image for its frozen natural crop"
        ) from exc
    source = np.asarray(data.astronaut())
    if source.shape != (512, 512, 3) or source.dtype != np.uint8:
        raise RuntimeError(
            f"unexpected skimage astronaut payload {source.shape} {source.dtype}"
        )
    y0, y1, x0, x1 = 176, 240, 208, 272
    crop = np.ascontiguousarray(source[y0:y1, x0:x1])
    if crop.shape != (SIZE, SIZE, 3):
        raise RuntimeError("frozen natural crop has the wrong shape")
    image = (crop.astype(np.float32) / np.float32(255.0)).astype(np.float32)
    return image, {
        "provider": "skimage.data.astronaut",
        "skimage_version": skimage.__version__,
        "source_shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "source_sha256": _tensor_sha256(source),
        "crop_yx": [y0, y1, x0, x1],
        "resize": None,
        "conversion": "uint8_rgb_divide_255_float32",
    }


def _build_targets_unchecked() -> dict[str, dict[str, Any]]:
    u, v = _target_coordinates()
    targets: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    ramp = np.stack(
        (
            0.12 + 0.67 * u + 0.09 * v + 0.08 * (u - 0.5) * (v - 0.5),
            0.18 + 0.19 * u + 0.56 * v + 0.06 * (u - 0.5) ** 2,
            0.78 - 0.43 * u + 0.11 * v + 0.07 * (v - 0.5) ** 2,
        ),
        axis=2,
    )
    targets["ramp_shading"] = (
        np.clip(ramp, 0.0, 1.0).astype(np.float32),
        {"generator": "fixed_affine_plus_quadratic_shading_v1"},
    )

    x = u - 0.46
    y = v - 0.53
    angle = np.arctan2(y, x)
    region = np.floor(np.mod(angle + 0.23, 2.0 * np.pi) / (2.0 * np.pi / 3.0)).astype(int)
    palette = np.asarray(
        [[0.10, 0.24, 0.81], [0.86, 0.22, 0.16], [0.19, 0.76, 0.31]],
        dtype=np.float64,
    )
    junction = palette[region]
    boundary_softener = np.clip(1.0 - 5.0 * np.sqrt(x * x + y * y), 0.0, 1.0)
    junction = (1.0 - 0.12 * boundary_softener[..., None]) * junction + (
        0.12 * boundary_softener[..., None] * np.asarray([0.58, 0.52, 0.44])
    )
    targets["step_junction"] = (
        np.clip(junction, 0.0, 1.0).astype(np.float32),
        {"generator": "fixed_three_region_angular_junction_v1"},
    )

    distance = _thin_bezier_distance(u, v)
    width = 0.0175
    coverage = np.clip((width + 1.3 / SIZE - distance) * SIZE / 1.3, 0.0, 1.0)
    background = np.stack((0.08 + 0.15 * u, 0.11 + 0.12 * v, 0.16 + 0.08 * u), axis=2)
    stroke = np.asarray([0.94, 0.73, 0.17], dtype=np.float64)
    curve = background * (1.0 - coverage[..., None]) + stroke * coverage[..., None]
    targets["thin_curve"] = (
        np.clip(curve, 0.0, 1.0).astype(np.float32),
        {"generator": "fixed_antialiased_quadratic_bezier_v1"},
    )

    theta = math.radians(27.0)
    along = math.cos(theta) * (u - 0.5) + math.sin(theta) * (v - 0.5)
    phase = 2.0 * np.pi * (4.35 * along + 0.37)
    sinusoid = np.stack(
        (
            0.50 + 0.38 * np.sin(phase),
            0.48 + 0.34 * np.cos(phase + 0.71),
            0.52 + 0.31 * np.sin(phase - 1.13),
        ),
        axis=2,
    )
    targets["stationary_sinusoid"] = (
        np.clip(sinusoid, 0.0, 1.0).astype(np.float32),
        {"generator": "fixed_oriented_rgb_sinusoid_v1", "cycles": 4.35},
    )

    theta = math.radians(61.0)
    along = math.cos(theta) * (u - 0.5) + math.sin(theta) * (v - 0.5)
    phase = 2.0 * np.pi * (2.35 * along + 3.85 * along * along + 0.19)
    envelope = 0.72 + 0.24 * np.cos(2.0 * np.pi * (v - 0.5))
    chirp = np.stack(
        (
            0.49 + 0.39 * envelope * np.sin(phase),
            0.51 + 0.32 * envelope * np.sin(phase + 1.17),
            0.47 + 0.35 * envelope * np.cos(phase - 0.63),
        ),
        axis=2,
    )
    targets["chirp"] = (
        np.clip(chirp, 0.0, 1.0).astype(np.float32),
        {"generator": "fixed_quadratic_phase_rgb_chirp_v1"},
    )

    natural, natural_meta = _natural_texture()
    targets["natural_texture"] = (natural, natural_meta)

    records: dict[str, dict[str, Any]] = {}
    for family in TARGET_FAMILIES:
        image, metadata = targets[family]
        records[family] = {
            "target_id": f"bench009_stage1_{family}_v0",
            "family": family,
            "split": "development",
            "image": np.ascontiguousarray(image),
            "pixel_sha256": _tensor_sha256(image),
            "metadata": metadata,
        }
    return records


def target_suite() -> dict[str, dict[str, Any]]:
    records = _build_targets_unchecked()
    actual = {family: item["pixel_sha256"] for family, item in records.items()}
    if actual != EXPECTED_TARGET_HASHES:
        raise RuntimeError(
            "BENCH-009 Stage-1 target hashes drifted: "
            + json.dumps({"expected": EXPECTED_TARGET_HASHES, "actual": actual}, sort_keys=True)
        )
    return records


def _prior_target_hashes() -> dict[str, set[str]]:
    from benchmarks import marginal_cold_stream_rd as comp006
    from benchmarks import perturb_recover_spectroscopy as fit020

    fit_records = fit020._target_suite(48)
    fit_hashes = {_tensor_sha256(item["image"]) for item in fit_records.values()}
    comp_records = comp006.target_suite(64)
    comp_hashes = {str(item["pixel_sha256"]) for item in comp_records.values()}
    return {"FIT-020": fit_hashes, "COMP-006": comp_hashes}


def fold_masks() -> dict[int, np.ndarray]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    assignment = ((yy // TILE_SIZE) + (xx // TILE_SIZE)) % 2
    return {fold: np.ascontiguousarray(assignment == fold) for fold in FOLDS}


def _target_manifest_records(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in records[family].items() if key != "image"}
        for family in TARGET_FAMILIES
    ]


def _parent_axes(records: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for family in TARGET_FAMILIES:
        target = records[family]
        for seed in SEEDS:
            for horizon_name, horizon_iters in PARENT_HORIZONS:
                yield {
                    "target_id": target["target_id"],
                    "target_pixel_sha256": target["pixel_sha256"],
                    "family": family,
                    "seed": seed,
                    "parent_horizon": horizon_name,
                    "parent_iters": horizon_iters,
                    "parent_count": PARENT_COUNT,
                }


def _with_cell_id(cell: dict[str, Any]) -> dict[str, Any]:
    result = dict(cell)
    result["cell_id"] = _sha256_bytes(_canonical_json(cell))
    return result


def _action_cells(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    actions = [(ledger, action) for ledger, values in ACTION_LEDGERS.items() for action in values]
    for parent in _parent_axes(records):
        for fold in FOLDS:
            for damping in DAMPINGS:
                for rcond in RCONDS:
                    for radius in TRUST_RADII:
                        for ledger, action in actions:
                            cells.append(
                                _with_cell_id(
                                    {
                                        **parent,
                                        "cell_kind": "core_immediate",
                                        "selection_scope": "crossfit",
                                        "heldout_fold": fold,
                                        "ledger": ledger,
                                        "action": action,
                                        "carrier_bank": "full" if action in CARRIER_ACTIONS else None,
                                        "damping": damping,
                                        "rcond": rcond,
                                        "trust_radius": radius,
                                        "trust_norm": TRUST_NORM,
                                    }
                                )
                            )
        for damping in DAMPINGS:
            for rcond in RCONDS:
                for radius in TRUST_RADII:
                    for ledger, action in actions:
                        cells.append(
                            _with_cell_id(
                                {
                                    **parent,
                                    "cell_kind": "core_immediate",
                                    "selection_scope": "all_pixel_identity_and_coefficient_oracle",
                                    "heldout_fold": None,
                                    "ledger": ledger,
                                    "action": action,
                                    "carrier_bank": "full" if action in CARRIER_ACTIONS else None,
                                    "damping": damping,
                                    "rcond": rcond,
                                    "trust_radius": radius,
                                    "trust_norm": TRUST_NORM,
                                }
                            )
                        )

        for fold in FOLDS:
            cells.append(
                _with_cell_id(
                    {
                        **parent,
                        "cell_kind": "no_action_baseline",
                        "selection_scope": "crossfit",
                        "heldout_fold": fold,
                        "ledger": "baseline",
                        "action": "no_action",
                        "carrier_bank": None,
                        "damping": None,
                        "rcond": None,
                        "trust_radius": None,
                        "trust_norm": TRUST_NORM,
                    }
                )
            )
        cells.append(
            _with_cell_id(
                {
                    **parent,
                    "cell_kind": "no_action_baseline",
                    "selection_scope": "all_pixel_identity_and_coefficient_oracle",
                    "heldout_fold": None,
                    "ledger": "baseline",
                    "action": "no_action",
                    "carrier_bank": None,
                    "damping": None,
                    "rcond": None,
                    "trust_radius": None,
                    "trust_norm": TRUST_NORM,
                }
            )
        )

        for selection_scope, folds in (
            ("crossfit", FOLDS),
            ("all_pixel_identity_and_coefficient_oracle", (None,)),
        ):
            for fold in folds:
                for radius in TRUST_RADII:
                    for ledger, action in (
                        ("causal", "j_plus_carrier6"),
                        ("matched", "carrier6"),
                    ):
                        cells.append(
                            _with_cell_id(
                                {
                                    **parent,
                                    "cell_kind": "carrier_bank_sensitivity",
                                    "selection_scope": selection_scope,
                                    "heldout_fold": fold,
                                    "ledger": ledger,
                                    "action": action,
                                    "carrier_bank": "small",
                                    "damping": PRIMARY_DAMPING,
                                    "rcond": PRIMARY_RCOND,
                                    "trust_radius": radius,
                                    "trust_norm": TRUST_NORM,
                                }
                            )
                        )
    return cells


def _selection_plan(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for parent in _parent_axes(records):
        for scope, folds in (
            ("crossfit", FOLDS),
            ("all_pixel_identity_and_coefficient_oracle", (None,)),
        ):
            for fold in folds:
                for selector in SELECTORS:
                    rows.append(
                        _with_cell_id(
                            {
                                **parent,
                                "selection_scope": scope,
                                "heldout_fold": fold,
                                "selector": selector,
                            }
                        )
                    )
    return rows


def _projector_plan(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    actions = [(ledger, action) for ledger, values in ACTION_LEDGERS.items() for action in values]
    for parent in _parent_axes(records):
        for scope, folds in (
            ("crossfit", FOLDS),
            ("all_pixel_identity_and_coefficient_oracle", (None,)),
        ):
            for fold in folds:
                for rcond in RCONDS:
                    for ledger, action in actions:
                        if "birth" in action and scope == "crossfit":
                            statistic = "finite_affine_action_gain_discovery"
                        elif "birth" in action:
                            statistic = (
                                "finite_affine_action_gain_all_pixel_identity_and_"
                                "coefficient_oracle"
                            )
                        elif scope == "crossfit":
                            statistic = "unregularized_projector_energy_discovery"
                        else:
                            statistic = "unregularized_projector_energy_all_pixel_oracle"
                        rows.append(
                            _with_cell_id(
                                {
                                    **parent,
                                    "selection_scope": scope,
                                    "heldout_fold": fold,
                                    "ledger": ledger,
                                    "action": action,
                                    "rcond": rcond,
                                    "rank_threshold": rcond,
                                    "statistic": statistic,
                                    "damping": None,
                                    "trust_radius": None,
                                }
                            )
                        )
    return rows


def _recovery_plan(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    actions = [(ledger, action) for ledger, values in ACTION_LEDGERS.items() for action in values]
    for parent in _parent_axes(records):
        for fold in FOLDS:
            for radius in TRUST_RADII:
                for ledger, action in actions:
                    rows.append(
                        _with_cell_id(
                            {
                                **parent,
                                "heldout_fold": fold,
                                "ledger": ledger,
                                "action": action,
                                "damping": PRIMARY_DAMPING,
                                "rcond": PRIMARY_RCOND,
                                "trust_radius": radius,
                                "optimizer_state_policy": "fresh_zero_state",
                                "trajectory_steps": 100,
                                "newly_rendered_checkpoint_steps": list(RECOVERY_STEPS),
                                "logical_checkpoint_steps": [0, *RECOVERY_STEPS],
                                "step0_record_policy": "link_immediate_action_record",
                                "topology_events": "disabled",
                            }
                        )
                    )
            rows.append(
                _with_cell_id(
                    {
                        **parent,
                        "heldout_fold": fold,
                        "ledger": "baseline",
                        "action": "no_action",
                        "damping": None,
                        "rcond": None,
                        "trust_radius": None,
                        "optimizer_state_policy": "fresh_zero_state",
                        "trajectory_steps": 100,
                        "newly_rendered_checkpoint_steps": list(RECOVERY_STEPS),
                        "logical_checkpoint_steps": [0, *RECOVERY_STEPS],
                        "step0_record_policy": "link_parent_baseline_record",
                        "topology_events": "disabled",
                    }
                )
            )
    return rows


def expected_counts() -> dict[str, int]:
    parents = len(TARGET_FAMILIES) * len(SEEDS) * len(PARENT_HORIZONS)
    numeric_grid = len(DAMPINGS) * len(RCONDS) * len(TRUST_RADII)
    actions = len(CAUSAL_ACTIONS) + len(MATCHED_ACTIONS)
    crossfit_core = parents * len(FOLDS) * numeric_grid * actions
    oracle_core = parents * numeric_grid * actions
    crossfit_base = parents * len(FOLDS)
    oracle_base = parents
    bank_crossfit = parents * len(FOLDS) * len(TRUST_RADII) * len(CARRIER_ACTIONS)
    bank_oracle = parents * len(TRUST_RADII) * len(CARRIER_ACTIONS)
    recovery = parents * len(FOLDS) * (
        len(TRUST_RADII) * actions + 1
    )
    crossfit_nonbirth_scores = parents * len(FOLDS) * len(RCONDS) * (actions - 2)
    oracle_nonbirth_scores = parents * len(RCONDS) * (actions - 2)
    crossfit_birth_scores = parents * len(FOLDS) * len(RCONDS) * 2
    oracle_birth_scores = parents * len(RCONDS) * 2
    projector = (
        crossfit_nonbirth_scores
        + oracle_nonbirth_scores
        + crossfit_birth_scores
        + oracle_birth_scores
    )
    return {
        "target_images": len(TARGET_FAMILIES),
        "parent_trajectories": len(TARGET_FAMILIES) * len(SEEDS),
        "parent_checkpoints": parents,
        "heldout_parent_fold_units": parents * len(FOLDS),
        "primary_discrete_selections_crossfit": parents * len(FOLDS) * len(SELECTORS),
        "primary_discrete_selections_oracle": parents * len(SELECTORS),
        "crossfit_immediate_action_cells": crossfit_core,
        "all_pixel_oracle_action_cells": oracle_core,
        "crossfit_no_action_baselines": crossfit_base,
        "all_pixel_no_action_baselines": oracle_base,
        "carrier_bank_sensitivity_crossfit": bank_crossfit,
        "carrier_bank_sensitivity_oracle": bank_oracle,
        "recovery_trajectories": recovery,
        "new_recovery_checkpoint_renders": recovery * len(RECOVERY_STEPS),
        "logical_recovery_checkpoint_records": recovery * (len(RECOVERY_STEPS) + 1),
        "primary_component_cells": crossfit_core + oracle_core + recovery,
        "crossfit_unregularized_nonbirth_linear_score_rows": crossfit_nonbirth_scores,
        "oracle_unregularized_nonbirth_linear_score_rows": oracle_nonbirth_scores,
        "unregularized_nonbirth_linear_score_rows": (
            crossfit_nonbirth_scores + oracle_nonbirth_scores
        ),
        "crossfit_finite_birth_affine_action_score_rows": crossfit_birth_scores,
        "oracle_finite_birth_affine_action_score_rows": oracle_birth_scores,
        "finite_birth_affine_action_score_rows": (
            crossfit_birth_scores + oracle_birth_scores
        ),
        "total_diagnostic_score_rows": projector,
    }


def build_manifest() -> dict[str, Any]:
    records = target_suite()
    action_cells = _action_cells(records)
    selections = _selection_plan(records)
    score_rows = _projector_plan(records)
    recoveries = _recovery_plan(records)
    counts = expected_counts()
    expected_action_rows = sum(
        counts[key]
        for key in (
            "crossfit_immediate_action_cells",
            "all_pixel_oracle_action_cells",
            "crossfit_no_action_baselines",
            "all_pixel_no_action_baselines",
            "carrier_bank_sensitivity_crossfit",
            "carrier_bank_sensitivity_oracle",
        )
    )
    if len(action_cells) != expected_action_rows:
        raise RuntimeError("action-cell manifest count drifted")
    if len(score_rows) != counts["total_diagnostic_score_rows"]:
        raise RuntimeError("score manifest count drifted")
    if len(recoveries) != counts["recovery_trajectories"]:
        raise RuntimeError("recovery manifest count drifted")
    for label, rows in (
        ("action", action_cells),
        ("selection", selections),
        ("score", score_rows),
        ("recovery", recoveries),
    ):
        ids = [str(row["cell_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate {label} cell key")
    masks = fold_masks()
    return {
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "split": "development",
        "target_manifest": _target_manifest_records(records),
        "target_manifest_sha256": _sha256_bytes(
            _canonical_json(_target_manifest_records(records))
        ),
        "excluded_prior_manifests": {
            "FIT-020": FIT020_MANIFEST_SHA256,
            "COMP-006": COMP006_MANIFEST_SHA256,
        },
        "axes": {
            "size": SIZE,
            "parent_count": PARENT_COUNT,
            "seeds": list(SEEDS),
            "parent_horizons": [
                {"name": name, "iterations": iterations}
                for name, iterations in PARENT_HORIZONS
            ],
            "tile_size": TILE_SIZE,
            "folds": list(FOLDS),
            "fold_pixel_counts": {str(k): int(v.sum()) for k, v in masks.items()},
            "dampings": list(DAMPINGS),
            "rconds": list(RCONDS),
            "rank_thresholds": list(RCONDS),
            "trust_radii": list(TRUST_RADII),
            "trust_norm": TRUST_NORM,
            "damping_coordinate_system": DAMPING_COORDINATE_SYSTEM,
            "causal_actions": list(CAUSAL_ACTIONS),
            "matched_actions": list(MATCHED_ACTIONS),
            "carrier_banks": ["small", "full"],
            "recovery_steps": list(RECOVERY_STEPS),
        },
        "statistics": {
            "zero_offset_extensions_discovery": "unregularized_projector_energy",
            "crossfit_heldout": "transported_gain_without_heldout_refit",
            "finite_birth": "finite_affine_action_gain_not_literal_projector_delta",
            "regularized_action": "predicted_trust_gain",
            "damping_coordinate_system": DAMPING_COORDINATE_SYSTEM,
            "column_normalized_sensitivity_ridge": "diagnostic_only",
            "realization": "single_joint_normalized_render",
            "crossfit_fit_policy": (
                "select_identity_and_fit_six_coefficients_on_discovery_then_score_heldout_"
                "without_refit"
            ),
            "post_selection_all_pixel_refit": "excluded_oracle_or_recovery_vector",
            "all_pixel_oracle_scope_label": (
                "all_pixel_identity_and_coefficient_oracle"
            ),
            "price_scope": PRICE_SCOPE,
        },
        "gates": {
            "jvp_relative_max": JVP_RELATIVE_TOLERANCE,
            "spearman_min": SPEARMAN_MINIMUM,
            "median_realized_predicted_ratio": list(GAIN_RATIO_BOUNDS),
            "maximum_unstable_target_families": 1,
            "stability_gate_axes": ["rank_threshold", "trust_radius"],
            "energy_survival_ratio": ENERGY_SURVIVAL_RATIO,
            "psnr_survival_db": PSNR_SURVIVAL_DB,
            "minimum_positive_target_families": MINIMUM_POSITIVE_FAMILIES,
            "recovery_steps": list(RECOVERY_STEPS),
            "missing_or_failed_required_cell": "invalidate",
        },
        "expected_counts": counts,
        "action_cells": action_cells,
        "selection_plan": selections,
        "score_plan": score_rows,
        "recovery_plan": recoveries,
        "scientific_cells_present": False,
        "recovery_results_present": False,
        "development_run_authorized": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }


def build_no_opacity_parent(
    target: np.ndarray | torch.Tensor,
    *,
    seed: int = SEEDS[0],
    dtype: torch.dtype = torch.float64,
) -> GaussianField:
    image = torch.as_tensor(target, dtype=dtype)
    if tuple(image.shape) != (SIZE, SIZE, 3):
        raise ValueError("preflight parent target must be 64x64 RGB")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    y, x = torch.meshgrid(
        torch.arange(8, dtype=dtype), torch.arange(8, dtype=dtype), indexing="ij"
    )
    means = torch.stack((4.0 + 8.0 * x.reshape(-1), 4.0 + 8.0 * y.reshape(-1)), dim=1)
    jitter = 0.18 * torch.randn(means.shape, generator=generator, dtype=dtype)
    means = (means + jitter).clamp(1.0, SIZE - 2.0)
    ids = torch.arange(PARENT_COUNT, dtype=dtype)
    scales = torch.stack((3.10 + 0.23 * torch.sin(ids), 2.75 + 0.19 * torch.cos(ids)), dim=1)
    rotations = 0.43 * torch.sin(0.37 * ids)
    px = means[:, 0].round().long().clamp(0, SIZE - 1)
    py = means[:, 1].round().long().clamp(0, SIZE - 1)
    sampled = image[py, px]
    color_jitter = 0.015 * torch.randn(sampled.shape, generator=generator, dtype=dtype)
    colors = (sampled + color_jitter).clamp(0.02, 0.98)
    return GaussianField(
        means,
        torch.log(scales),
        rotations,
        colors,
        None,
    )


def _pack_no_opacity(field: GaussianField) -> torch.Tensor:
    if field.opacities is not None or field.n != PARENT_COUNT:
        raise ValueError("Stage-1 primary parent must be no-opacity N=64")
    pieces = (
        field.means.reshape(-1),
        field.log_scales.reshape(-1),
        field.rotations.reshape(-1),
        field.colors.reshape(-1),
    )
    return torch.cat(pieces).detach().clone()


def _parent_mean_chart_independent(field: GaussianField) -> torch.Tensor:
    """Freeze ``R(theta_parent) diag(sx_parent, sy_parent)`` independently."""

    scales = field.scales().detach()
    cosine = torch.cos(field.rotations.detach())
    sine = torch.sin(field.rotations.detach())
    chart = torch.zeros(
        field.n,
        2,
        2,
        dtype=field.means.dtype,
        device=field.means.device,
    )
    chart[:, 0, 0] = cosine * scales[:, 0]
    chart[:, 1, 0] = sine * scales[:, 0]
    chart[:, 0, 1] = -sine * scales[:, 1]
    chart[:, 1, 1] = cosine * scales[:, 1]
    return chart


def _dimensionless_updated_field(
    parent: GaussianField,
    step: torch.Tensor,
    *,
    parent_mean_chart: torch.Tensor | None = None,
) -> GaussianField:
    """Apply the frozen unit-fair chart without using the Stage-1 core implementation."""

    if parent.opacities is not None:
        raise ValueError("dimensionless Stage-1 chart forbids opacity")
    n = parent.n
    expected = 8 * n
    if step.ndim != 1 or int(step.numel()) != expected:
        raise ValueError(f"dimensionless step must contain {expected} coordinates")
    chart = (
        _parent_mean_chart_independent(parent)
        if parent_mean_chart is None
        else parent_mean_chart
    )
    if tuple(chart.shape) != (n, 2, 2):
        raise ValueError("parent mean chart has the wrong shape")
    cursor = 0
    local_means = step[cursor : cursor + 2 * n].reshape(n, 2)
    cursor += 2 * n
    log_scales = step[cursor : cursor + 2 * n].reshape(n, 2)
    cursor += 2 * n
    rotations = step[cursor : cursor + n]
    cursor += n
    colors = step[cursor : cursor + 3 * n].reshape(n, 3)
    cursor += 3 * n
    if cursor != expected:
        raise RuntimeError("internal dimensionless coordinate layout mismatch")
    return GaussianField(
        parent.means + torch.einsum("nij,nj->ni", chart, local_means),
        parent.log_scales + log_scales,
        parent.rotations + rotations,
        parent.colors + colors,
        None,
        parent.scale_max,
        None,
        parent.background_mask,
        parent.filter_variance,
    )


def _unpack_no_opacity(vector: torch.Tensor, n: int = PARENT_COUNT) -> GaussianField:
    cursor = 0

    def take(count: int) -> torch.Tensor:
        nonlocal cursor
        result = vector[cursor : cursor + count]
        cursor += count
        return result

    field = GaussianField(
        take(2 * n).reshape(n, 2),
        take(2 * n).reshape(n, 2),
        take(n),
        take(3 * n).reshape(n, 3),
        None,
    )
    if cursor != int(vector.numel()):
        raise ValueError("unexpected no-opacity parameter-vector length")
    return field


def _render_reference(field: GaussianField, *, color_grads: torch.Tensor | None = None) -> torch.Tensor:
    return _render_field_reference(
        field,
        SIZE,
        SIZE,
        sigma_cutoff=SIGMA_CUTOFF,
        color_grads=color_grads,
    )


def no_opacity_jvp_preflight(field: GaussianField) -> dict[str, Any]:
    vector = _pack_no_opacity(field)
    origin = torch.zeros_like(vector)
    parent_mean_chart = _parent_mean_chart_independent(field)

    def render(step: torch.Tensor) -> torch.Tensor:
        updated = _dimensionless_updated_field(
            field,
            step,
            parent_mean_chart=parent_mean_chart,
        )
        return _independent_render(
            updated,
            support_field=field,
        ).reshape(-1)

    generator = torch.Generator(device="cpu").manual_seed(20260715)
    direction = torch.randn(vector.shape, generator=generator, dtype=vector.dtype)
    direction = direction / direction.norm().clamp_min(1e-12)
    _, jvp = torch.autograd.functional.jvp(
        render, origin, direction, create_graph=False, strict=True
    )
    positive = render(JVP_EPSILON * direction)
    negative = render(-JVP_EPSILON * direction)
    finite = (positive - negative) / (2.0 * JVP_EPSILON)
    error = finite - jvp
    relative = float(error.norm() / finite.norm().clamp_min(1e-12))
    maximum = float(error.abs().max())
    return {
        "parameter_columns": int(vector.numel()),
        "opacity_present": field.opacities is not None,
        "mean_chart": "R_parent_diag_parent_scales",
        "log_scale_units": 1.0,
        "rotation_radians": 1.0,
        "display_linear_rgb": 1.0,
        "support": "exact_parent_pixel_membership",
        "direction_seed": 20260715,
        "epsilon": JVP_EPSILON,
        "relative_l2": relative,
        "max_abs": maximum,
        "pass": bool(relative <= JVP_RELATIVE_TOLERANCE),
    }


def coordinate_chart_preflight(field: GaussianField) -> dict[str, Any]:
    """Cross-check the independently implemented chart against the Stage-1 core."""

    from benchmarks.residual_tangent_auction_stage1_core import (
        make_no_opacity_parameterization,
    )

    parameterization = make_no_opacity_parameterization(
        field,
        SIZE,
        SIZE,
        sigma_cutoff=SIGMA_CUTOFF,
        render_chunk=1,
    )
    chart = _parent_mean_chart_independent(field)
    chart_error = float((chart - parameterization.parent_mean_chart).abs().max())
    n = field.n
    gid = 11
    x_step = torch.zeros(8 * n, dtype=field.means.dtype, device=field.means.device)
    y_step = torch.zeros_like(x_step)
    x_step[2 * gid] = 1.0
    y_step[2 * gid + 1] = 1.0
    x_field = _dimensionless_updated_field(field, x_step, parent_mean_chart=chart)
    y_field = _dimensionless_updated_field(field, y_step, parent_mean_chart=chart)
    x_delta = x_field.means[gid] - field.means[gid]
    y_delta = y_field.means[gid] - field.means[gid]
    x_axis_error = float((x_delta - chart[gid, :, 0]).abs().max())
    y_axis_error = float((y_delta - chart[gid, :, 1]).abs().max())

    unit_step = torch.zeros_like(x_step)
    unit_step[2 * n] = 1.0
    unit_step[4 * n] = 1.0
    unit_step[5 * n] = 1.0
    unit_field = _dimensionless_updated_field(field, unit_step, parent_mean_chart=chart)
    log_scale_delta = float(unit_field.log_scales[0, 0] - field.log_scales[0, 0])
    rotation_delta = float(unit_field.rotations[0] - field.rotations[0])
    rgb_delta = float(unit_field.colors[0, 0] - field.colors[0, 0])

    generator = torch.Generator(device=field.means.device).manual_seed(271815)
    comparison_step = 0.03 * torch.randn(
        (8 * n,),
        generator=generator,
        dtype=field.means.dtype,
        device=field.means.device,
    )
    independent_field = _dimensionless_updated_field(
        field,
        comparison_step,
        parent_mean_chart=chart,
    )
    core_field = parameterization.field_at(comparison_step)
    field_error = max(
        float((left - right).abs().max())
        for left, right in (
            (independent_field.means, core_field.means),
            (independent_field.log_scales, core_field.log_scales),
            (independent_field.rotations, core_field.rotations),
            (independent_field.colors, core_field.colors),
        )
    )
    independent_render = _independent_render(
        independent_field,
        support_field=field,
    )
    core_render = parameterization.render_step(comparison_step)
    frozen_render_error = float((independent_render - core_render).abs().max())
    rotated_anisotropic = bool(
        abs(float(field.rotations[gid])) > 1e-2
        and abs(float(field.scales()[gid, 0] - field.scales()[gid, 1])) > 1e-2
    )
    tolerance = RENDER_ABSOLUTE_TOLERANCE
    passed = bool(
        rotated_anisotropic
        and chart_error <= tolerance
        and x_axis_error <= tolerance
        and y_axis_error <= tolerance
        and abs(log_scale_delta - 1.0) <= tolerance
        and abs(rotation_delta - 1.0) <= tolerance
        and abs(rgb_delta - 1.0) <= tolerance
        and field_error <= tolerance
        and frozen_render_error <= tolerance
    )
    return {
        "mean_chart": "R_parent_diag_parent_scales",
        "mean_test_gid": gid,
        "mean_test_is_rotated_anisotropic": rotated_anisotropic,
        "chart_core_max_abs": chart_error,
        "local_x_axis_max_abs": x_axis_error,
        "local_y_axis_max_abs": y_axis_error,
        "log_scale_unit_delta": log_scale_delta,
        "rotation_unit_delta_radians": rotation_delta,
        "display_linear_rgb_unit_delta": rgb_delta,
        "independent_core_field_max_abs": field_error,
        "independent_core_frozen_render_max_abs": frozen_render_error,
        "support": "exact_parent_pixel_membership",
        "pass": passed,
    }


def _raw_support_membership(field: GaussianField) -> torch.Tensor:
    """Independent native AABB membership, including rounded centers and radii."""

    dtype = field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(SIZE, dtype=dtype, device=field.means.device),
        torch.arange(SIZE, dtype=dtype, device=field.means.device),
        indexing="ij",
    )
    radii = field.radii(SIGMA_CUTOFF)
    ix = torch.round(field.means[:, 0].detach()).long()
    iy = torch.round(field.means[:, 1].detach()).long()
    return (
        (xx.unsqueeze(0) >= (ix - radii[:, 0])[:, None, None])
        & (xx.unsqueeze(0) <= (ix + radii[:, 0])[:, None, None])
        & (yy.unsqueeze(0) >= (iy - radii[:, 1])[:, None, None])
        & (yy.unsqueeze(0) <= (iy + radii[:, 1])[:, None, None])
    )


def _support_membership_change_count(
    parent: GaussianField,
    updated: GaussianField,
) -> int:
    return int(_support_membership_change_summary(parent, updated)["xor_count"])


def _support_membership_change_summary(
    parent: GaussianField,
    updated: GaussianField,
) -> dict[str, int | str]:
    if parent.n != updated.n:
        raise ValueError("support-change comparison requires matching field sizes")
    parent_membership = _raw_support_membership(parent)
    native_membership = _raw_support_membership(updated)
    added = native_membership & ~parent_membership
    removed = parent_membership & ~native_membership
    return {
        "added_count": int(added.sum()),
        "removed_count": int(removed.sum()),
        "xor_count": int((added | removed).sum()),
        "parent_membership_sha256": _tensor_sha256(parent_membership),
        "native_membership_sha256": _tensor_sha256(native_membership),
    }


def _raw_weight_images(
    field: GaussianField,
    *,
    support_field: GaussianField | None = None,
) -> torch.Tensor:
    """Independent weights with either native or exact parent-frozen membership."""

    if support_field is not None and support_field.n != field.n:
        raise ValueError("support field must have the same Gaussian count")
    dtype = field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(SIZE, dtype=dtype, device=field.means.device),
        torch.arange(SIZE, dtype=dtype, device=field.means.device),
        indexing="ij",
    )
    conics = field.conics()
    membership = _raw_support_membership(field if support_field is None else support_field)
    weights = []
    for gid in range(field.n):
        dx = xx - field.means[gid, 0]
        dy = yy - field.means[gid, 1]
        a, b, c = conics[gid]
        q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        weight = torch.where(
            membership[gid], torch.exp(-0.5 * q), torch.zeros_like(q)
        )
        weights.append(weight)
    return torch.stack(weights, dim=0)


def _independent_render(
    field: GaussianField,
    *,
    support_field: GaussianField | None = None,
    color_grads: torch.Tensor | None = None,
    carrier_gid: int | None = None,
    frequency: tuple[float, float] = (0.39, 0.21),
    carrier_coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    if field.opacities is not None:
        raise ValueError("Stage-1 independent renderer accepts no-opacity fields only")
    weights = _raw_weight_images(field, support_field=support_field)
    denominator = weights.sum(dim=0)
    numerator = torch.einsum("nhw,nc->hwc", weights, field.colors)
    dtype = field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(SIZE, dtype=dtype),
        torch.arange(SIZE, dtype=dtype),
        indexing="ij",
    )
    if color_grads is not None:
        for gid in range(field.n):
            dx = xx - field.means[gid, 0]
            dy = yy - field.means[gid, 1]
            angle = field.rotations[gid]
            cosine, sine = torch.cos(angle), torch.sin(angle)
            scales = field.scales()[gid]
            local_x = (cosine * dx + sine * dy) / scales[0].clamp_min(1e-6)
            local_y = (-sine * dx + cosine * dy) / scales[1].clamp_min(1e-6)
            variation = (
                color_grads[gid, 0][None, None, :] * local_x[..., None]
                + color_grads[gid, 1][None, None, :] * local_y[..., None]
            )
            numerator = numerator + weights[gid, ..., None] * variation
    if carrier_gid is not None:
        if carrier_coefficients is None or tuple(carrier_coefficients.shape) != (2, 3):
            raise ValueError("carrier coefficients must have shape (2,3)")
        dx = xx - field.means[carrier_gid, 0]
        dy = yy - field.means[carrier_gid, 1]
        angle = field.rotations[carrier_gid]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        scales = field.scales()[carrier_gid]
        local_x = (cosine * dx + sine * dy) / scales[0].clamp_min(1e-6)
        local_y = (-sine * dx + cosine * dy) / scales[1].clamp_min(1e-6)
        phase = 2.0 * math.pi * (
            float(frequency[0]) * local_x + float(frequency[1]) * local_y
        )
        variation = (
            torch.sin(phase)[..., None] * carrier_coefficients[0]
            + torch.cos(phase)[..., None] * carrier_coefficients[1]
        )
        numerator = numerator + weights[carrier_gid, ..., None] * variation
    return numerator / (denominator[..., None] + 1e-8)


def _append_no_opacity_births(
    field: GaussianField,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    colors: torch.Tensor,
) -> GaussianField:
    if field.opacities is not None:
        raise ValueError("birth preflight requires a no-opacity parent")
    return GaussianField(
        torch.cat((field.means, means), dim=0),
        torch.cat((field.log_scales, torch.log(scales)), dim=0),
        torch.cat((field.rotations, rotations), dim=0),
        torch.cat((field.colors, colors), dim=0),
        None,
    )


def joint_two_birth_preflight(field: GaussianField) -> dict[str, Any]:
    # Exercise the exact birth solve after a nonzero dimensionless current-grammar update.  This
    # prevents unchanged-base parity from standing in for the required joint action path.
    frozen_parent = field
    field = _joint_updated_field(frozen_parent, TRUST_RADII[-1])
    dtype = field.means.dtype
    means = torch.tensor([[13.35, 47.20], [51.10, 15.65]], dtype=dtype)
    scales = torch.tensor([[2.25, 1.45], [1.55, 2.35]], dtype=dtype)
    rotations = torch.tensor([0.37, -0.64], dtype=dtype)
    true_colors = torch.tensor([[0.87, 0.16, 0.64], [0.14, 0.82, 0.31]], dtype=dtype)

    parent_weights = _raw_weight_images(field)
    denominator = parent_weights.sum(dim=0)
    base_numerator = torch.einsum("nhw,nc->hwc", parent_weights, field.colors)
    base = base_numerator / (denominator[..., None] + 1e-8)
    births_zero = _append_no_opacity_births(
        field, means, scales, rotations, torch.zeros_like(true_colors)
    )
    birth_weights = _raw_weight_images(births_zero)[-2:]
    birth_membership = _raw_support_membership(births_zero)[-2:]
    total_denominator = denominator + birth_weights.sum(dim=0) + 1e-8
    a1 = birth_weights[0] / total_denominator
    a2 = birth_weights[1] / total_denominator
    offset_render = (1.0 - a1 - a2)[..., None] * base
    formula_render = (
        offset_render
        + a1[..., None] * true_colors[0]
        + a2[..., None] * true_colors[1]
    )
    direct_field = _append_no_opacity_births(
        field, means, scales, rotations, true_colors
    )
    direct_render = _render_reference(direct_field)
    direct_error = float((formula_render - direct_render).abs().max())

    design = torch.stack((a1.reshape(-1), a2.reshape(-1)), dim=1)
    solved_channels = []
    for channel in range(3):
        rhs = (direct_render[..., channel] - offset_render[..., channel]).reshape(-1)
        solved_channels.append(torch.linalg.lstsq(design, rhs).solution)
    solved_colors = torch.stack(solved_channels, dim=1)
    solved_render = (
        offset_render
        + a1[..., None] * solved_colors[0]
        + a2[..., None] * solved_colors[1]
    )
    solve_render_error = float((solved_render - direct_render).abs().max())
    color_error = float((solved_colors - true_colors).abs().max())
    independent_base_error = float((base - _render_reference(field)).abs().max())
    frozen_parent_weights = _raw_weight_images(field, support_field=frozen_parent)
    frozen_denominator = frozen_parent_weights.sum(dim=0)
    frozen_base = torch.einsum(
        "nhw,nc->hwc", frozen_parent_weights, field.colors
    ) / (frozen_denominator[..., None] + 1e-8)
    frozen_native_base_difference = float((frozen_base - base).abs().max())
    support_change = _support_membership_change_summary(frozen_parent, field)
    return {
        "optimized_rgb_coefficients": 6,
        "birth_count": 2,
        "nonzero_base_update_trust_radius": TRUST_RADII[-1],
        "parent_opacity_present": field.opacities is not None,
        "common_denominator": True,
        "birth_geometry": "fixed_before_rgb_solve",
        "finite_realization_support": "native_recentered_recomputed_aabb",
        "linearization_support": "exact_original_parent_membership",
        "parent_support_membership_changes": support_change["xor_count"],
        "parent_support_change": support_change,
        "birth_support": {
            "row_count": 2,
            "fixed_geometry": True,
            "membership_sha256": _tensor_sha256(birth_membership),
        },
        "frozen_support_native_base_max_abs": frozen_native_base_difference,
        "independent_base_render_max_abs": independent_base_error,
        "formula_direct_render_max_abs": direct_error,
        "solved_render_max_abs": solve_render_error,
        "solved_color_max_abs": color_error,
        "pass": bool(
            independent_base_error <= RENDER_ABSOLUTE_TOLERANCE
            and direct_error <= RENDER_ABSOLUTE_TOLERANCE
            and solve_render_error <= RENDER_ABSOLUTE_TOLERANCE
            and color_error <= RENDER_ABSOLUTE_TOLERANCE
        ),
    }


def _joint_updated_field(field: GaussianField, radius: float) -> GaussianField:
    vector = _pack_no_opacity(field)
    generator = torch.Generator(device="cpu").manual_seed(314159)
    direction = torch.randn(vector.shape, generator=generator, dtype=vector.dtype)
    direction = direction / direction.abs().max().clamp_min(1e-12)
    return _dimensionless_updated_field(
        field,
        float(radius) * direction,
        parent_mean_chart=_parent_mean_chart_independent(field),
    )


def joint_realization_preflight(field: GaussianField) -> dict[str, Any]:
    by_radius: dict[str, Any] = {}
    for radius in TRUST_RADII:
        updated = _joint_updated_field(field, radius)
        gradients = torch.zeros(PARENT_COUNT, 2, 3, dtype=field.colors.dtype)
        gradients[17] = float(radius) * torch.tensor(
            [[0.021, -0.014, 0.018], [-0.013, 0.017, 0.011]],
            dtype=field.colors.dtype,
        )
        affine_direct = _render_reference(updated, color_grads=gradients)
        affine_independent = _independent_render(updated, color_grads=gradients)
        affine_error = float((affine_direct - affine_independent).abs().max())
        affine_frozen = _independent_render(
            updated,
            support_field=field,
            color_grads=gradients,
        )
        affine_frozen_native_difference = float(
            (affine_frozen - affine_independent).abs().max()
        )

        carrier_coefficients = float(radius) * torch.tensor(
            [[0.030, -0.018, 0.022], [-0.016, 0.024, 0.019]],
            dtype=field.colors.dtype,
        )
        base_direct = _render_reference(updated)
        one_hot = torch.zeros_like(updated.colors)
        one_hot[17] = 1.0
        probe = GaussianField(
            updated.means,
            updated.log_scales,
            updated.rotations,
            one_hot,
            None,
        )
        responsibility = _render_reference(probe)[..., 0]
        yy, xx = torch.meshgrid(
            torch.arange(SIZE, dtype=field.colors.dtype),
            torch.arange(SIZE, dtype=field.colors.dtype),
            indexing="ij",
        )
        dx = xx - updated.means[17, 0]
        dy = yy - updated.means[17, 1]
        angle = updated.rotations[17]
        cosine, sine = torch.cos(angle), torch.sin(angle)
        local_x = (cosine * dx + sine * dy) / updated.scales()[17, 0]
        local_y = (-sine * dx + cosine * dy) / updated.scales()[17, 1]
        phase = 2.0 * math.pi * (0.39 * local_x + 0.21 * local_y)
        variation = (
            torch.sin(phase)[..., None] * carrier_coefficients[0]
            + torch.cos(phase)[..., None] * carrier_coefficients[1]
        )
        carrier_direct = base_direct + responsibility[..., None] * variation
        carrier_independent = _independent_render(
            updated,
            carrier_gid=17,
            frequency=(0.39, 0.21),
            carrier_coefficients=carrier_coefficients,
        )
        carrier_error = float((carrier_direct - carrier_independent).abs().max())
        carrier_frozen = _independent_render(
            updated,
            support_field=field,
            carrier_gid=17,
            frequency=(0.39, 0.21),
            carrier_coefficients=carrier_coefficients,
        )
        carrier_frozen_native_difference = float(
            (carrier_frozen - carrier_independent).abs().max()
        )
        support_change = _support_membership_change_summary(field, updated)
        by_radius[str(radius)] = {
            "updated_opacity_present": updated.opacities is not None,
            "support_membership_changes": support_change["xor_count"],
            "support_change": support_change,
            "affine_joint_max_abs": affine_error,
            "affine_frozen_support_native_max_abs": affine_frozen_native_difference,
            "carrier_joint_max_abs": carrier_error,
            "carrier_frozen_support_native_max_abs": carrier_frozen_native_difference,
            "pass": bool(
                affine_error <= RENDER_ABSOLUTE_TOLERANCE
                and carrier_error <= RENDER_ABSOLUTE_TOLERANCE
            ),
        }
    return {
        "trust_norm": TRUST_NORM,
        "dimensionless_action_scaling": "one_uniform_linf_scale",
        "linearization_support": "exact_parent_pixel_membership",
        "finite_realization_support": "native_recentered_recomputed_aabb",
        "radii": by_radius,
        "pass": all(bool(item["pass"]) for item in by_radius.values()),
    }


def _deterministic_small_least_squares(
    design: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Solve a tiny full-rank system with a fixed scalar reduction/elimination order.

    The leakage preflight compares two fits whose discovery inputs are bit-identical.  LAPACK's
    rank-revealing least-squares drivers may return last-bit differences across repeated calls,
    which would make a bit-exact non-leakage assertion depend on thread scheduling.  This helper
    is intentionally limited to the at-most-three-column synthetic preflight and uses ``fsum``
    plus deterministic pivoting; it is not a scientific-cell solver.
    """
    if design.ndim != 2 or target.ndim != 1 or design.shape[0] != target.shape[0]:
        raise ValueError("least-squares inputs have invalid shape")
    rows, columns = design.shape
    if rows < columns or columns == 0 or columns > 3:
        raise ValueError("deterministic preflight solve expects one to three columns")
    x = design.detach().cpu().to(torch.float64)
    y = target.detach().cpu().to(torch.float64).tolist()
    x_columns = [x[:, column].tolist() for column in range(columns)]
    gram = [
        [
            math.fsum(
                left * right
                for left, right in zip(x_columns[i], x_columns[j], strict=True)
            )
            for j in range(columns)
        ]
        for i in range(columns)
    ]
    rhs = [
        math.fsum(left * right for left, right in zip(x_columns[i], y, strict=True))
        for i in range(columns)
    ]
    augmented = [gram[row] + [rhs[row]] for row in range(columns)]
    scale = max(abs(value) for row in gram for value in row)
    for pivot in range(columns):
        pivot_row = max(
            range(pivot, columns),
            key=lambda row: (abs(augmented[row][pivot]), -row),
        )
        if abs(augmented[pivot_row][pivot]) <= 1e-14 * max(scale, 1.0):
            raise RuntimeError("synthetic leakage dictionary became rank deficient")
        augmented[pivot], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot],
        )
        pivot_value = augmented[pivot][pivot]
        for row in range(pivot + 1, columns):
            factor = augmented[row][pivot] / pivot_value
            for column in range(pivot, columns + 1):
                augmented[row][column] -= factor * augmented[pivot][column]
    solution = [0.0] * columns
    for row in range(columns - 1, -1, -1):
        remainder = math.fsum(
            augmented[row][column] * solution[column]
            for column in range(row + 1, columns)
        )
        solution[row] = (augmented[row][-1] - remainder) / augmented[row][row]
    return torch.tensor(solution, dtype=design.dtype, device=design.device)


def _select_omp_columns(
    design: torch.Tensor,
    residual: torch.Tensor,
    discovery_mask: torch.Tensor,
    *,
    count: int,
) -> tuple[int, ...]:
    if design.ndim != 2 or residual.ndim != 1 or discovery_mask.ndim != 1:
        raise ValueError("OMP inputs have invalid rank")
    if design.shape[0] != residual.shape[0] or residual.shape[0] != discovery_mask.shape[0]:
        raise ValueError("OMP inputs have mismatched rows")
    if count <= 0 or count > design.shape[1]:
        raise ValueError("invalid OMP count")
    x = design[discovery_mask]
    y = residual[discovery_mask]
    selected: list[int] = []
    for _ in range(count):
        best: tuple[float, int] | None = None
        for candidate in range(design.shape[1]):
            if candidate in selected:
                continue
            columns = [*selected, candidate]
            coefficients = _deterministic_small_least_squares(x[:, columns], y)
            remaining = y - x[:, columns] @ coefficients
            score = float(remaining.square().sum())
            key = (score, candidate)
            if best is None or key < best:
                best = key
        if best is None:
            raise RuntimeError("OMP exhausted its candidate dictionary")
        selected.append(best[1])
    return tuple(selected)


def _discovery_input_sha256(
    design: torch.Tensor,
    residual: torch.Tensor,
    discovery_mask: torch.Tensor,
) -> str:
    """Hash the exact bytes that identity selection and coefficient fitting may inspect."""

    selected = torch.as_tensor(discovery_mask, dtype=torch.bool, device=design.device)
    payload = {
        "mask": _tensor_sha256(selected),
        "design": _tensor_sha256(design[selected]),
        "residual": _tensor_sha256(residual[selected]),
    }
    return _sha256_bytes(_canonical_json(payload))


def crossfit_leakage_preflight() -> dict[str, Any]:
    masks = fold_masks()
    u, v = _target_coordinates()
    columns_np = np.stack(
        (
            np.ones_like(u),
            u,
            v,
            u * v,
            np.sin(2.0 * np.pi * u),
            np.cos(2.0 * np.pi * v),
            np.sin(2.0 * np.pi * (u + v)),
            np.cos(4.0 * np.pi * (u - v)),
        ),
        axis=2,
    ).reshape(-1, 8)
    design = torch.as_tensor(columns_np, dtype=torch.float64)
    residual = torch.as_tensor(
        (0.7 * columns_np[:, 4] - 0.35 * columns_np[:, 2] + 0.11 * columns_np[:, 6]),
        dtype=torch.float64,
    )
    by_fold = {}
    for fold, heldout_np in masks.items():
        heldout = torch.as_tensor(heldout_np.reshape(-1), dtype=torch.bool)
        discovery = ~heldout
        discovery_input_sha256 = _discovery_input_sha256(
            design, residual, discovery
        )
        selected = _select_omp_columns(design, residual, discovery, count=3)
        coefficients = _deterministic_small_least_squares(
            design[discovery][:, list(selected)], residual[discovery]
        )
        heldout_prediction = design[heldout][:, list(selected)] @ coefficients
        perturbed = residual.clone()
        perturbed[heldout] = 1000.0 * torch.sin(
            torch.arange(int(heldout.sum()), dtype=torch.float64)
        )
        discovery_residual_exactly_equal = torch.equal(
            residual[discovery], perturbed[discovery]
        )
        perturbed_discovery_input_sha256 = _discovery_input_sha256(
            design, perturbed, discovery
        )
        selected_perturbed = _select_omp_columns(design, perturbed, discovery, count=3)
        coefficients_perturbed = _deterministic_small_least_squares(
            design[discovery][:, list(selected_perturbed)], perturbed[discovery]
        )
        heldout_prediction_perturbed = (
            design[heldout][:, list(selected_perturbed)] @ coefficients_perturbed
        )
        heldout_sse = float((residual[heldout] - heldout_prediction).square().sum())
        heldout_sse_perturbed = float(
            (perturbed[heldout] - heldout_prediction_perturbed).square().sum()
        )
        identity_unchanged = selected == selected_perturbed
        coefficient_change = float((coefficients - coefficients_perturbed).abs().max())
        prediction_change = float(
            (heldout_prediction - heldout_prediction_perturbed).abs().max()
        )
        discovery_hash_unchanged = (
            discovery_input_sha256 == perturbed_discovery_input_sha256
        )
        coefficients_repeatable = (
            coefficient_change <= NUMERICAL_REPEATABILITY_ABSOLUTE_TOLERANCE
        )
        prediction_repeatable = (
            prediction_change <= NUMERICAL_REPEATABILITY_ABSOLUTE_TOLERANCE
        )
        score_changed = heldout_sse != heldout_sse_perturbed
        by_fold[str(fold)] = {
            "selected": list(selected),
            "selected_after_heldout_perturbation": list(selected_perturbed),
            "discovery_input_sha256": discovery_input_sha256,
            "discovery_input_sha256_after_heldout_perturbation": (
                perturbed_discovery_input_sha256
            ),
            "discovery_input_hash_exactly_equal": discovery_hash_unchanged,
            "coefficient_max_abs_change_after_heldout_perturbation": coefficient_change,
            "prediction_max_abs_change_after_heldout_perturbation": prediction_change,
            "heldout_sse": heldout_sse,
            "heldout_sse_after_heldout_perturbation": heldout_sse_perturbed,
            "heldout_score_changed": score_changed,
            "heldout_pixels": int(heldout.sum()),
            "discovery_pixels": int(discovery.sum()),
            "fit_policy": "identity_and_coefficients_fit_on_discovery_only",
            "heldout_refit": False,
            "discovery_residual_exactly_equal": discovery_residual_exactly_equal,
            "numerical_repeatability_absolute_tolerance": (
                NUMERICAL_REPEATABILITY_ABSOLUTE_TOLERANCE
            ),
            "unchanged": bool(
                identity_unchanged
                and discovery_hash_unchanged
                and discovery_residual_exactly_equal
                and coefficients_repeatable
                and prediction_repeatable
                and score_changed
            ),
        }
    return {
        "folds": by_fold,
        "pass": all(bool(item["unchanged"]) for item in by_fold.values()),
    }


def projector_trust_preflight() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(271828)
    design = torch.randn(180, 20, generator=generator, dtype=torch.float64)
    design[:, -1] = design[:, 0] - 0.5 * design[:, 3]
    extension = torch.randn(180, 6, generator=generator, dtype=torch.float64)
    residual = torch.randn(180, generator=generator, dtype=torch.float64)
    by_rcond = {}
    all_finite = True
    for rcond in RCONDS:
        base = _scaled_svd(design, rcond=rcond)
        joint = _scaled_svd(torch.cat((design, extension), dim=1), rcond=rcond)
        p_base = base.u @ (base.u.T @ residual)
        p_joint = joint.u @ (joint.u.T @ residual)
        delta = float(p_joint.square().sum() - p_base.square().sum())
        residualized = _residualize(base, extension)
        orthogonality = float((base.u.T @ residualized).abs().max())
        by_rcond[str(rcond)] = {
            "base_rank": base.rank,
            "joint_rank": joint.rank,
            "unregularized_delta": delta,
            "orthogonality_max_abs": orthogonality,
        }
        for damping in DAMPINGS:
            coefficients, _ = _solve_svd(joint, residual, ridge=damping)
            for radius in TRUST_RADII:
                maximum = float(coefficients.abs().max())
                fraction = min(1.0, float(radius) / max(maximum, 1e-30))
                predicted = joint.design @ (coefficients * fraction)
                all_finite = all_finite and bool(torch.isfinite(predicted).all())
    return {
        "literal_projector_has_no_damping_or_radius": True,
        "regularized_grid_size": len(DAMPINGS) * len(RCONDS) * len(TRUST_RADII),
        "by_rcond": by_rcond,
        "pass": bool(
            all_finite
            and all(
                float(item["orthogonality_max_abs"]) <= RENDER_ABSOLUTE_TOLERANCE
                for item in by_rcond.values()
            )
        ),
    }


def run_preflight() -> tuple[dict[str, Any], dict[str, Any]]:
    records = target_suite()
    manifest = build_manifest()
    prior = _prior_target_hashes()
    target_hashes = {item["pixel_sha256"] for item in records.values()}
    collisions = {
        name: sorted(target_hashes & hashes)
        for name, hashes in prior.items()
    }
    target_check = {
        "count": len(records),
        "families": list(records),
        "all_shapes_64_rgb": all(
            tuple(item["image"].shape) == (SIZE, SIZE, 3) for item in records.values()
        ),
        "all_float32": all(item["image"].dtype == np.float32 for item in records.values()),
        "all_finite": all(np.isfinite(item["image"]).all() for item in records.values()),
        "all_in_unit_range": all(
            float(item["image"].min()) >= 0.0 and float(item["image"].max()) <= 1.0
            for item in records.values()
        ),
        "unique_hashes": len(target_hashes) == len(records),
        "collisions": collisions,
        "fit020_manifest_sha256": FIT020_MANIFEST_SHA256,
        "comp006_manifest_sha256": COMP006_MANIFEST_SHA256,
    }
    target_check["pass"] = bool(
        target_check["count"] == len(TARGET_FAMILIES)
        and target_check["all_shapes_64_rgb"]
        and target_check["all_float32"]
        and target_check["all_finite"]
        and target_check["all_in_unit_range"]
        and target_check["unique_hashes"]
        and all(not values for values in collisions.values())
    )

    masks = fold_masks()
    fold_check = {
        "tile_size": TILE_SIZE,
        "fold_pixel_counts": {str(k): int(v.sum()) for k, v in masks.items()},
        "pairwise_overlap": int(np.logical_and(masks[0], masks[1]).sum()),
        "union_pixels": int(np.logical_or(masks[0], masks[1]).sum()),
    }
    fold_check["pass"] = bool(
        fold_check["pairwise_overlap"] == 0
        and fold_check["union_pixels"] == SIZE * SIZE
        and set(fold_check["fold_pixel_counts"].values()) == {SIZE * SIZE // 2}
    )

    parent = build_no_opacity_parent(records["natural_texture"]["image"])
    parent_check = {
        "n": parent.n,
        "opacity_present": parent.opacities is not None,
        "field_sha256": _tensor_sha256(_pack_no_opacity(parent)),
        "parameter_columns": int(_pack_no_opacity(parent).numel()),
    }
    parent_check["pass"] = bool(
        parent.n == PARENT_COUNT
        and parent.opacities is None
        and parent_check["parameter_columns"] == 8 * PARENT_COUNT
    )

    counts = expected_counts()
    action_rows_expected = sum(
        counts[key]
        for key in (
            "crossfit_immediate_action_cells",
            "all_pixel_oracle_action_cells",
            "crossfit_no_action_baselines",
            "all_pixel_no_action_baselines",
            "carrier_bank_sensitivity_crossfit",
            "carrier_bank_sensitivity_oracle",
        )
    )
    count_check = {
        "expected": counts,
        "actual": {
            "action_manifest_rows": len(manifest["action_cells"]),
            "recovery_trajectories": len(manifest["recovery_plan"]),
            "total_diagnostic_score_rows": len(manifest["score_plan"]),
        },
    }
    count_check["expected_action_manifest_rows"] = action_rows_expected
    count_check["pass"] = bool(
        count_check["actual"]["action_manifest_rows"] == action_rows_expected
        and count_check["actual"]["recovery_trajectories"]
        == counts["recovery_trajectories"]
        and count_check["actual"]["total_diagnostic_score_rows"]
        == counts["total_diagnostic_score_rows"]
    )

    checks = {
        "target_manifest_and_prior_disjointness": target_check,
        "checkerboard_fold_partition": fold_check,
        "manifest_cell_completeness": count_check,
        "no_opacity_n64_parent": parent_check,
        "dimensionless_coordinate_chart": coordinate_chart_preflight(parent),
        "centered_fd_jvp": no_opacity_jvp_preflight(parent),
        "projector_vs_regularized_trust": projector_trust_preflight(),
        "joint_two_birth_common_denominator": joint_two_birth_preflight(parent),
        "joint_base_extension_realization": joint_realization_preflight(parent),
        "crossfit_heldout_leakage": crossfit_leakage_preflight(),
    }
    preflight_pass = all(bool(item["pass"]) for item in checks.values())
    result = {
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "checks": checks,
        "preflight_pass": preflight_pass,
        "scientific_cells_present": False,
        "recovery_runner_present": False,
        "development_runner_present": False,
        "development_run_authorized": False,
        "authorization_reason": (
            "preflight plumbing passed but no scientific/recovery runner exists"
            if preflight_pass
            else "one or more fail-closed preflight checks failed"
        ),
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
        "price_scope": PRICE_SCOPE,
        "damping_coordinate_system": DAMPING_COORDINATE_SYSTEM,
        "column_normalized_sensitivity_ridge": "diagnostic_only",
    }
    return manifest, result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_preflight(outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    manifest, result = run_preflight()
    provenance = _collect_provenance()
    config = {
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float64_preflight_float32_targets",
        },
        "constants": {
            "size": SIZE,
            "parent_count": PARENT_COUNT,
            "seeds": list(SEEDS),
            "parent_horizons": [list(item) for item in PARENT_HORIZONS],
            "tile_size": TILE_SIZE,
            "folds": list(FOLDS),
            "dampings": list(DAMPINGS),
            "rconds": list(RCONDS),
            "trust_radii": list(TRUST_RADII),
            "trust_norm": TRUST_NORM,
            "damping_coordinate_system": DAMPING_COORDINATE_SYSTEM,
            "recovery_steps": list(RECOVERY_STEPS),
        },
        "expected_counts": expected_counts(),
        "provenance": provenance,
        "development_run_authorized": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }
    _write_json(outdir / "config.json", config)
    _write_json(outdir / "manifest.json", manifest)
    result["config_sha256"] = _sha256_bytes((outdir / "config.json").read_bytes())
    result["manifest_sha256"] = _sha256_bytes((outdir / "manifest.json").read_bytes())
    _write_json(outdir / "preflight.json", result)

    lines = [
        "# BENCH-009 Stage-1 manifest and numerical preflight",
        "",
        f"**Preflight:** {'pass' if result['preflight_pass'] else 'fail'}",
        "",
        "**Development run authorized:** no — the scientific and recovery runners do not exist.",
        "",
        "**Scope:** development killing-screen plumbing only; no quality, convergence, speed, "
        "rate, compression, or method claim.",
        "",
        "## Checks",
        "",
        "| Check | Pass |",
        "|---|:---:|",
    ]
    for name, values in result["checks"].items():
        lines.append(f"| {name} | {'yes' if values['pass'] else 'no'} |")
    lines.extend(
        (
            "",
            "## Frozen counts",
            "",
            "| Unit | Count |",
            "|---|---:|",
        )
    )
    for name, value in expected_counts().items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        (
            "",
            "All-pixel oracle cells are descriptive and excluded from future gates. Diagnostic "
            "packet fields are not actual bytes or BPP.",
            "",
        )
    )
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="results/bench009_tangent_auction_stage1_preflight",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = write_preflight(Path(args.outdir))
    print(
        json.dumps(
            {
                "outdir": str(args.outdir),
                "preflight_pass": result["preflight_pass"],
                "development_run_authorized": result["development_run_authorized"],
                "method_promotion_authorized": False,
                "compression_claim_authorized": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if result["preflight_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
