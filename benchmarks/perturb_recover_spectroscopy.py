"""FIT-020 ranked-deduplication perturb--recover response spectroscopy.

This is a frozen, benchmark-only mechanism assay.  It asks whether one early response-shape
descriptor predicts paired step-200 utility on held-out procedural target variants.  It does not
implement a production allocator or claim pure coverage, optimizer-state continuation, rendering
speed, compression, or expressiveness gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from benchmarks._util import run_config, write_config
from benchmarks.common import write_csv
from benchmarks.gauge_equivalence_audit import (
    _action_hash,
    _field_sha256,
    _sequential_moment_action,
    _sha256,
    _stable_topk,
    _target_suite as fit019_target_suite,
    _tensor_sha256,
)
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import _render, _responsibility_error_density_scores, fit
from structsplat.init import build_field


RECOVERY_STEPS = (0, 1, 2, 5, 10, 20, 40, 60, 100, 200)
COVERAGES = (5, 6, 7, 8)
TRAIN_VARIANTS = (0, 2, 3, 5)
HELDOUT_VARIANTS = (1, 4)
SEEDS = (0, 1, 2)
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TARGET_MANIFEST_SHA256 = "4e51518b9c2b1816311f1b2856c19ea77acaea913b534987fa776cf160740ebe"
TARGET_FAMILIES = (
    "sinusoid",
    "chirp",
    "oriented_ramp",
    "opposing_ramps",
    "soft_ridge",
    "lattice",
)


TARGET_PARAMETER_TABLE: dict[str, tuple[tuple[float, ...], ...]] = {
    "sinusoid": (
        (2.75, 12.0, 0.25),
        (3.80, 38.0, 1.10),
        (4.90, 71.0, 2.05),
        (6.10, 109.0, 2.75),
        (5.55, 143.0, 0.70),
        (7.25, 166.0, 1.85),
    ),
    "chirp": (
        (1.40, 3.20, 8.0, 0.30),
        (2.10, 4.60, 34.0, 1.20),
        (1.80, 6.10, 67.0, 2.10),
        (2.80, 3.80, 101.0, 2.85),
        (2.45, 5.35, 137.0, 0.75),
        (3.20, 6.70, 169.0, 1.65),
    ),
    "oriented_ramp": (
        (5.0, 0.85, 0.70, 0.15),
        (31.0, 1.10, 1.25, 0.80),
        (63.0, 1.35, 1.70, 1.45),
        (97.0, 0.95, 2.10, 2.10),
        (132.0, 1.20, 0.90, 2.75),
        (164.0, 1.45, 1.40, 3.40),
    ),
    "opposing_ramps": (
        (18.0, 0.90, 0.80, 0.25),
        (45.0, 1.15, 1.30, 0.95),
        (76.0, 1.40, 0.65, 1.55),
        (112.0, 1.05, 1.80, 2.20),
        (146.0, 1.30, 1.10, 2.85),
        (173.0, 1.50, 1.55, 3.50),
    ),
    "soft_ridge": (
        (10.0, 0.055, -0.18, 0.10),
        (42.0, 0.080, -0.08, 0.70),
        (74.0, 0.105, 0.02, 1.30),
        (108.0, 0.065, 0.12, 1.90),
        (139.0, 0.090, -0.14, 2.50),
        (168.0, 0.120, 0.16, 3.10),
    ),
    "lattice": (
        (7.0, 2.5, 3.5, 1.2, 0.2),
        (29.0, 3.5, 5.0, 1.8, 0.8),
        (58.0, 4.5, 6.0, 2.4, 1.4),
        (93.0, 5.5, 3.0, 3.0, 2.0),
        (127.0, 6.5, 4.5, 2.0, 2.6),
        (161.0, 7.0, 6.5, 2.8, 3.2),
    ),
}


SOURCE_PATHS = (
    "benchmarks/perturb_recover_spectroscopy.py",
    "benchmarks/gauge_equivalence_audit.py",
    "benchmarks/common.py",
    "benchmarks/_util.py",
    "tasks/FIT-020-perturb-recover-spectroscopy.md",
    "tests/test_perturb_recover_spectroscopy.py",
    "pyproject.toml",
)


def _source_paths(root: Path) -> tuple[str, ...]:
    production = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.joinpath("src", "structsplat").glob("*.py"))
    )
    return tuple(dict.fromkeys((*SOURCE_PATHS, *production)))


def _source_provenance(root: Path) -> dict[str, Any]:
    paths = _source_paths(root)
    missing = [path for path in paths if not root.joinpath(path).is_file()]
    if missing:
        raise FileNotFoundError(f"FIT-020 source provenance is missing: {missing}")
    hashes = {path: _sha256(root / path) for path in paths}
    joined = "\n".join(f"{path}:{hashes[path]}" for path in paths).encode("utf-8")
    return {
        "files": hashes,
        "combined_sha256": hashlib.sha256(joined).hexdigest(),
        "snapshot_policy": "all listed benchmark/task/test files plus top-level package modules",
    }


def _write_source_snapshot(root: Path, outdir: Path, source: dict[str, Any]) -> None:
    for relative, expected_hash in source["files"].items():
        payload = root.joinpath(relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise RuntimeError(f"source changed while snapshotting {relative}")
        destination = outdir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _rotated_coordinates(
    u: np.ndarray, v: np.ndarray, angle_degrees: float
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(float(angle_degrees))
    x = u - 0.5
    y = v - 0.5
    along = math.cos(angle) * x + math.sin(angle) * y
    across = -math.sin(angle) * x + math.cos(angle) * y
    return along, across


def _hue_endpoints(hue: float) -> tuple[np.ndarray, np.ndarray]:
    phases = float(hue) + np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    low = 0.5 + 0.38 * np.cos(phases)
    high = 0.5 + 0.38 * np.cos(phases + np.pi)
    return low.astype(np.float32), high.astype(np.float32)


def _make_target(
    family: str, params: tuple[float, ...], u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    if family == "sinusoid":
        cycles, angle, phase = params
        along, _ = _rotated_coordinates(u, v, angle)
        carrier = 2.0 * np.pi * cycles * (along + 0.5) + phase
        image = np.stack(
            [
                0.5 + 0.43 * np.sin(carrier),
                0.5 + 0.43 * np.sin(carrier + 2.0 * np.pi / 3.0),
                0.5 + 0.43 * np.sin(carrier + 4.0 * np.pi / 3.0),
            ],
            axis=2,
        )
    elif family == "chirp":
        base_cycles, sweep, angle, phase = params
        along, across = _rotated_coordinates(u, v, angle)
        position = along + 0.5
        carrier = (
            2.0 * np.pi * (base_cycles * position + 0.5 * sweep * position * position)
            + phase
            + 0.35 * np.pi * across
        )
        image = np.stack(
            [
                0.5 + 0.42 * np.sin(carrier),
                0.5 + 0.42 * np.sin(carrier + 1.7),
                0.5 + 0.42 * np.cos(carrier - 0.8),
            ],
            axis=2,
        )
    elif family == "oriented_ramp":
        angle, slope, gamma, hue = params
        along, _ = _rotated_coordinates(u, v, angle)
        value = np.clip(0.5 + slope * along, 0.0, 1.0) ** gamma
        low, high = _hue_endpoints(hue)
        image = (1.0 - value[..., None]) * low + value[..., None] * high
    elif family == "opposing_ramps":
        angle, slope, gamma, hue = params
        along, across = _rotated_coordinates(u, v, angle)
        first = np.clip(0.5 + slope * along, 0.0, 1.0) ** gamma
        second = np.clip(0.5 + slope * across, 0.0, 1.0) ** (1.0 / gamma)
        low, high = _hue_endpoints(hue)
        chroma = (1.0 - second[..., None]) * low + second[..., None] * high
        opposing = np.stack([first, 1.0 - first, 0.15 + 0.70 * second], axis=2)
        image = 0.58 * opposing + 0.42 * chroma
    elif family == "soft_ridge":
        angle, width, offset, hue = params
        along, _ = _rotated_coordinates(u, v, angle)
        ridge = np.exp(-0.5 * ((along - offset) / width) ** 2)
        low, high = _hue_endpoints(hue)
        image = (1.0 - ridge[..., None]) * (0.65 * low) + ridge[..., None] * high
    elif family == "lattice":
        angle, freq_u, freq_v, softness, hue = params
        along, across = _rotated_coordinates(u, v, angle)
        carrier = np.sin(2.0 * np.pi * freq_u * (along + 0.5)) * np.sin(
            2.0 * np.pi * freq_v * (across + 0.5)
        )
        value = 0.5 + 0.5 * np.tanh(softness * carrier)
        low, high = _hue_endpoints(hue)
        image = (1.0 - value[..., None]) * low + value[..., None] * high
    else:
        raise ValueError(f"unknown FIT-020 target family {family!r}")
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _target_suite(size: int = 48) -> dict[str, dict[str, Any]]:
    if size < 8:
        raise ValueError("procedural target size must be at least 8")
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    u = x / float(size - 1)
    v = y / float(size - 1)
    result: dict[str, dict[str, Any]] = {}
    for family in TARGET_FAMILIES:
        params_by_variant = TARGET_PARAMETER_TABLE[family]
        if len(params_by_variant) != 6:
            raise RuntimeError(f"frozen family {family} does not have six variants")
        for variant, params in enumerate(params_by_variant):
            target = f"{family}_v{variant}"
            result[target] = {
                "family": family,
                "variant": variant,
                "split": "train" if variant in TRAIN_VARIANTS else "heldout",
                "params": tuple(float(value) for value in params),
                "image": _make_target(family, params, u, v),
            }
    return result


def _coverage_actions(ranking: Iterable[int]) -> dict[int, list[int]]:
    ranking = [int(value) for value in ranking]
    if len(ranking) < 8:
        raise ValueError("FIT-020 coverage actions require at least eight ranked groups")
    ranking = ranking[:8]
    if len(set(ranking)) != 8:
        raise ValueError("FIT-020's first eight ranked groups must be unique")
    g1, g2, g3, g4, g5, g6, g7, g8 = ranking
    actions = {
        5: [g1, g2, g3, g4, g5, g3, g2, g1],
        6: [g1, g2, g3, g4, g5, g6, g2, g1],
        7: [g1, g2, g3, g4, g5, g6, g7, g1],
        8: [g1, g2, g3, g4, g5, g6, g7, g8],
    }
    for coverage in COVERAGES:
        if len(actions[coverage]) != 8 or len(set(actions[coverage])) != coverage:
            raise RuntimeError("invalid frozen coverage action")
    return actions


def _terminal_auc(steps: list[int], values: list[float]) -> float:
    if len(steps) != len(values):
        raise ValueError("AUC step/value length mismatch")
    if len(steps) < 2:
        raise ValueError("AUC requires at least two steps")
    step_array = np.asarray(steps, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(step_array).all() or not np.isfinite(value_array).all():
        raise ValueError("AUC inputs must be finite")
    if np.any(np.diff(step_array) <= 0.0):
        raise ValueError("AUC steps must be strictly increasing")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    area = float(trapezoid(value_array, step_array))
    return area / float(step_array[-1] - step_array[0])


def _extract_trajectory(result: dict[str, Any]) -> dict[str, Any]:
    if int(result.get("iterations_run", -1)) != 200:
        raise ValueError("FIT-020 requires exactly 200 recovery iterations")
    if int(result.get("trajectory_terminal_iter", -1)) != 200:
        raise ValueError("FIT-020 requires terminal iteration 200")
    history = result.get("history") or {}
    iterations = list(history.get("iter") or [])
    psnrs = list(history.get("psnr") or [])
    losses = list(history.get("loss") or [])
    if iterations != list(range(200)) or len(psnrs) != 200 or len(losses) != 200:
        raise ValueError("FIT-020 history must contain aligned iterations 0..199")
    terminal_psnr = float(result["trajectory_terminal_psnr"])
    terminal_loss = float(result["trajectory_terminal_loss"])
    full_steps = [*iterations, 200]
    full_psnr = [*map(float, psnrs), terminal_psnr]
    full_loss = [*map(float, losses), terminal_loss]
    if not np.isfinite(full_psnr).all() or not np.isfinite(full_loss).all():
        raise ValueError("FIT-020 trajectory values must be finite")
    return {
        "psnr_by_step": {step: full_psnr[step] for step in RECOVERY_STEPS},
        "loss_by_step": {step: full_loss[step] for step in RECOVERY_STEPS},
        "full_steps": full_steps,
        "full_psnr": full_psnr,
        "full_loss": full_loss,
    }


def _objective_loss(field, target: torch.Tensor, cfg: FitConfig) -> float:
    with torch.no_grad():
        image = _render(field, cfg, target.shape[0], target.shape[1])
        pixel = (image - target).abs().mean()
        loss = (1.0 - float(cfg.ssim_weight)) * pixel
        if cfg.ssim_weight != 0.0:
            loss = loss + float(cfg.ssim_weight) * (
                1.0 - M.ssim(image, target, backend=cfg.ssim_backend)
            )
    return float(loss.detach().cpu())


def _residual_stats(render: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    residual = (render - target).abs().mean(dim=2).detach().cpu().numpy().reshape(-1)
    total = float(residual.sum())
    if total <= 0.0:
        return {"residual_entropy": 0.0, "residual_top_decile_mass": 0.0}
    probability = residual.astype(np.float64) / total
    positive = probability > 0.0
    entropy = -float(np.sum(probability[positive] * np.log(probability[positive])))
    entropy /= math.log(float(probability.size))
    count = max(1, int(math.ceil(0.10 * probability.size)))
    top_mass = float(np.partition(probability, -count)[-count:].sum())
    return {
        "residual_entropy": entropy,
        "residual_top_decile_mass": top_mass,
    }


def _identity_tuple(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["family"]), int(row["variant"]), str(row["split"])


def _validate_raw_cells(rows: list[dict[str, Any]]) -> None:
    expected_targets = {
        f"{family}_v{variant}"
        for family in TARGET_FAMILIES
        for variant in range(6)
    }
    actual_targets = {str(row["target"]) for row in rows}
    if actual_targets != expected_targets:
        missing = sorted(expected_targets - actual_targets)
        extra = sorted(actual_targets - expected_targets)
        raise ValueError(f"FIT-020 target table mismatch; missing={missing}, extra={extra}")
    mandatory_names = {
        "target",
        "family",
        "variant",
        "split",
        "seed",
        "coverage",
        "target_sha256",
        "source_combined_sha256",
        "canonical_count",
        "canonical_field_sha256",
        "ranking_groups",
        "ranking_hash",
        "action_groups",
        "action_hash",
        "unique_action_groups",
        "repeated_action_count",
        "pre_psnr",
        "pre_loss",
        "residual_entropy",
        "residual_top_decile_mass",
        "intervention_rms",
        "trajectory_auc",
        "peak_gain",
        "peak_step",
        "late_asymptote",
        "psnr_curve",
        "checkpoint_psnr",
        "checkpoint_loss",
        "history_complete",
        "final_count",
        "renders_finite",
        "action_seconds",
        "fit_seconds",
        "total_seconds",
        *(f"psnr_step{step}" for step in RECOVERY_STEPS),
        *(f"loss_step{step}" for step in RECOVERY_STEPS),
    }
    for row in rows:
        missing_names = sorted(mandatory_names - set(row))
        if missing_names:
            raise ValueError(
                f"FIT-020 integrity row {row.get('target')}/seed{row.get('seed')}/"
                f"C{row.get('coverage')} is missing {missing_names}"
            )
        if not bool(row["history_complete"]):
            raise ValueError("FIT-020 integrity requires every step-0..200 history")
        if int(row["canonical_count"]) != 32 or int(row["final_count"]) != 40:
            raise ValueError("FIT-020 integrity requires exact N=32->40")
        if not bool(row["renders_finite"]):
            raise ValueError("FIT-020 integrity requires finite renders")
        numeric_names = {
            "pre_psnr",
            "pre_loss",
            "residual_entropy",
            "residual_top_decile_mass",
            "intervention_rms",
            "trajectory_auc",
            "peak_gain",
            "peak_step",
            "late_asymptote",
            "action_seconds",
            "fit_seconds",
            "total_seconds",
            *(f"psnr_step{step}" for step in RECOVERY_STEPS),
            *(f"loss_step{step}" for step in RECOVERY_STEPS),
        }
        if not all(math.isfinite(float(row[name])) for name in numeric_names):
            raise ValueError("FIT-020 integrity metrics must be finite")
        ranking = json.loads(row["ranking_groups"])
        action = json.loads(row["action_groups"])
        expected_action = _coverage_actions(ranking)[int(row["coverage"])]
        if action != expected_action:
            raise ValueError("FIT-020 integrity action does not match the frozen nested chain")
        if row["ranking_hash"] != _action_hash(ranking):
            raise ValueError("FIT-020 integrity ranking hash mismatch")
        if row["action_hash"] != _action_hash(action):
            raise ValueError("FIT-020 integrity action hash mismatch")
        coverage = int(row["coverage"])
        if int(row["unique_action_groups"]) != coverage:
            raise ValueError("FIT-020 integrity unique action count does not match coverage")
        if int(row["repeated_action_count"]) != 8 - coverage:
            raise ValueError("FIT-020 integrity repeated action count does not match coverage")
        curve = np.asarray(json.loads(row["psnr_curve"]), dtype=np.float64)
        if curve.shape != (201,) or not np.isfinite(curve).all():
            raise ValueError("FIT-020 integrity PSNR curve must contain finite steps 0..200")
        checkpoint_psnr = {
            int(step): float(value) for step, value in json.loads(row["checkpoint_psnr"]).items()
        }
        checkpoint_loss = {
            int(step): float(value) for step, value in json.loads(row["checkpoint_loss"]).items()
        }
        if tuple(checkpoint_psnr) != RECOVERY_STEPS or tuple(checkpoint_loss) != RECOVERY_STEPS:
            raise ValueError("FIT-020 integrity checkpoint maps do not match frozen steps")
        for step in RECOVERY_STEPS:
            if float(row[f"psnr_step{step}"]) != checkpoint_psnr[step]:
                raise ValueError("FIT-020 integrity PSNR checkpoint mismatch")
            if float(row[f"loss_step{step}"]) != checkpoint_loss[step]:
                raise ValueError("FIT-020 integrity loss checkpoint mismatch")
            if float(row[f"psnr_step{step}"]) != float(curve[step]):
                raise ValueError("FIT-020 integrity full/checkpoint curve mismatch")
        if not math.isclose(
            float(row["trajectory_auc"]),
            _terminal_auc(list(range(201)), curve.tolist()),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("FIT-020 integrity terminal-inclusive AUC mismatch")
    for target in sorted(expected_targets):
        target_rows = [row for row in rows if row["target"] == target]
        identities = {_identity_tuple(row) for row in target_rows}
        if len(identities) != 1:
            raise ValueError(f"FIT-020 target {target} has mixed identity/family metadata")
        family, variant, split = next(iter(identities))
        expected_split = "train" if variant in TRAIN_VARIANTS else "heldout"
        if target != f"{family}_v{variant}" or split != expected_split:
            raise ValueError(f"FIT-020 target {target} has invalid identity/split metadata")
        for seed in SEEDS:
            seed_rows = [row for row in target_rows if int(row["seed"]) == seed]
            control_names = (
                "target_sha256",
                "source_combined_sha256",
                "canonical_count",
                "canonical_field_sha256",
                "ranking_groups",
                "ranking_hash",
                "pre_psnr",
                "pre_loss",
                "residual_entropy",
                "residual_top_decile_mass",
            )
            if any(len({row[name] for row in seed_rows}) != 1 for name in control_names):
                raise ValueError(
                    f"FIT-020 integrity controls differ across arms for {target}/seed{seed}"
                )
            for coverage in COVERAGES:
                cell = [
                    row
                    for row in target_rows
                    if int(row["seed"]) == seed and int(row["coverage"]) == coverage
                ]
                if len(cell) != 1:
                    raise ValueError(
                        f"FIT-020 requires exactly one row for {target}/seed{seed}/C{coverage}"
                    )
    target_hashes = {
        target: {row["target_sha256"] for row in rows if row["target"] == target}
        for target in expected_targets
    }
    if not all(len(hashes) == 1 for hashes in target_hashes.values()):
        raise ValueError("FIT-020 integrity requires one target hash per target")
    if len({next(iter(hashes)) for hashes in target_hashes.values()}) != 36:
        raise ValueError("FIT-020 integrity requires 36 unique target hashes")
    if len({row["source_combined_sha256"] for row in rows}) != 1:
        raise ValueError("FIT-020 integrity requires one source snapshot hash")


def _aggregate_seed_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each concentrated arm with C8, then average seeds as repeated measures."""
    by_target_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target_seed[(str(row["target"]), int(row["seed"]))].append(row)

    seed_pairs: list[dict[str, Any]] = []
    for (target, seed), values in sorted(by_target_seed.items()):
        identities = {_identity_tuple(row) for row in values}
        if len(identities) != 1:
            raise ValueError(f"FIT-020 target {target} has mixed identity/family metadata")
        by_coverage: dict[int, dict[str, Any]] = {}
        for coverage in COVERAGES:
            matches = [row for row in values if int(row["coverage"]) == coverage]
            if len(matches) != 1:
                raise ValueError(
                    f"FIT-020 requires exactly one row for {target}/seed{seed}/C{coverage}"
                )
            by_coverage[coverage] = matches[0]
        reference = by_coverage[8]
        family, variant, split = next(iter(identities))
        for coverage in (5, 6, 7):
            row = by_coverage[coverage]
            paired: dict[str, Any] = {
                "target": target,
                "family": family,
                "variant": variant,
                "split": split,
                "seed": seed,
                "coverage": coverage,
                "coverage_deficit": 8 - coverage,
                "coverage_deficit_sq": (8 - coverage) ** 2,
                "pre_psnr": float(row["pre_psnr"]),
                "residual_entropy": float(row["residual_entropy"]),
                "residual_top_decile_mass": float(row["residual_top_decile_mass"]),
                "intervention_rms": float(row["intervention_rms"]),
                "c8_gain_0_10": float(reference["psnr_step10"])
                - float(reference["psnr_step0"]),
            }
            for step in RECOVERY_STEPS:
                key = f"psnr_step{step}"
                if key in row and key in reference:
                    paired[f"d{step}"] = float(row[key]) - float(reference[key])
            for required_step in (0, 2, 10, 200):
                if f"d{required_step}" not in paired:
                    raise ValueError(
                        f"FIT-020 row {target}/seed{seed}/C{coverage} lacks step {required_step}"
                    )
            paired["y"] = paired["d200"]
            paired["bend"] = (
                (paired["d10"] - paired["d2"]) / 8.0
                - (paired["d2"] - paired["d0"]) / 2.0
            )
            if all(f"d{step}" in paired for step in (0, 1, 2, 5, 10)):
                paired["early_auc_delta"] = _terminal_auc(
                    [0, 1, 2, 5, 10],
                    [paired[f"d{step}"] for step in (0, 1, 2, 5, 10)],
                )
            for own_name, paired_name in (
                ("trajectory_auc", "auc_delta"),
                ("late_asymptote", "late_asymptote_delta"),
                ("peak_gain", "peak_gain_delta"),
            ):
                if own_name in row and own_name in reference:
                    paired[paired_name] = float(row[own_name]) - float(reference[own_name])
            if "psnr_curve" in row and "psnr_curve" in reference:
                arm_curve = np.asarray(json.loads(row["psnr_curve"]), dtype=np.float64)
                ref_curve = np.asarray(json.loads(reference["psnr_curve"]), dtype=np.float64)
                if arm_curve.shape != (201,) or ref_curve.shape != (201,):
                    raise ValueError("FIT-020 stored curves must contain steps 0..200")
                delta_curve = arm_curve - ref_curve
                paired["paired_peak_delta"] = float(delta_curve.max())
                paired["paired_peak_step"] = int(np.argmax(delta_curve))
                signs = np.sign(delta_curve)
                nonzero = np.flatnonzero(signs != 0.0)
                crossovers = []
                if nonzero.size:
                    previous = int(signs[nonzero[0]])
                    for index in nonzero[1:]:
                        current = int(signs[index])
                        if current != previous:
                            crossovers.append(int(index))
                            previous = current
                paired["crossover_count"] = len(crossovers)
                paired["first_crossover_step"] = crossovers[0] if crossovers else 201
            seed_pairs.append(paired)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_pairs:
        grouped[(row["target"], int(row["coverage"]))].append(row)
    aggregated: list[dict[str, Any]] = []
    family_order = {family: idx for idx, family in enumerate(TARGET_FAMILIES)}
    for (target, coverage), values in sorted(
        grouped.items(),
        key=lambda item: (
            family_order[item[1][0]["family"]],
            int(item[1][0]["variant"]),
            item[0][1],
        ),
    ):
        seeds = {int(row["seed"]) for row in values}
        if seeds != set(SEEDS) or len(values) != len(SEEDS):
            raise ValueError(
                f"FIT-020 requires exactly one seed pair for {target}/C{coverage}/seed0,1,2"
            )
        identities = {_identity_tuple(row) for row in values}
        if len(identities) != 1:
            raise ValueError(f"FIT-020 target {target} has mixed identity/family metadata")
        family, variant, split = next(iter(identities))
        output: dict[str, Any] = {
            "target": target,
            "family": family,
            "variant": variant,
            "split": split,
            "coverage": coverage,
            "seed_count": len(values),
        }
        numeric_names = sorted(
            {
                name
                for row in values
                for name, value in row.items()
                if name not in {"seed", "variant", "coverage", "seed_count"}
                and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for name in numeric_names:
            output[name] = float(np.mean([float(row[name]) for row in values]))
        output["coverage"] = coverage
        output["coverage_deficit"] = 8 - coverage
        output["coverage_deficit_sq"] = (8 - coverage) ** 2
        aggregated.append(output)
    return aggregated


FAMILY_FEATURES = tuple(f"family_{family}" for family in TARGET_FAMILIES[:-1])
INTERVENTION_FEATURES = (
    *FAMILY_FEATURES,
    "coverage_deficit",
    "coverage_deficit_sq",
    "pre_psnr",
    "residual_entropy",
    "residual_top_decile_mass",
    "d0",
    "intervention_rms",
)
EARLY_FEATURES = (*INTERVENTION_FEATURES, "d10", "c8_gain_0_10")
RESPONSE_FEATURES = (*EARLY_FEATURES, "bend")


def _feature_value(row: dict[str, Any], name: str) -> float:
    if name.startswith("family_"):
        return float(row["family"] == name.removeprefix("family_"))
    return float(row[name])


def _design_matrix(
    rows: list[dict[str, Any]], feature_names: tuple[str, ...] | list[str]
) -> np.ndarray:
    return np.asarray(
        [[_feature_value(row, name) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def _fit_ridge_model(
    rows: list[dict[str, Any]], feature_names: tuple[str, ...], alpha: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("ridge fit requires rows")
    if alpha < 0.0 or not math.isfinite(float(alpha)):
        raise ValueError("ridge alpha must be finite and nonnegative")
    features = tuple(feature_names)
    x = _design_matrix(rows, features)
    y = np.asarray([float(row["y"]) for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (x - mean) / scale
    y_mean = float(y.mean())
    centered_y = y - y_mean
    gram = standardized.T @ standardized
    rhs = standardized.T @ centered_y
    coefficient = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    return {
        "features": list(features),
        "alpha": float(alpha),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": y_mean,
        "fit_rows": len(rows),
        "fit_targets": len({row["target"] for row in rows}),
    }


def _predict_ridge(model: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    features = tuple(model["features"])
    x = _design_matrix(rows, features)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    prediction = float(model["intercept"]) + ((x - mean) / scale) @ coefficient
    return np.asarray(prediction, dtype=np.float64)


def _grouped_cv_ridge(
    rows: list[dict[str, Any]], feature_names: tuple[str, ...]
) -> dict[str, Any]:
    targets = sorted({str(row["target"]) for row in rows})
    if len(targets) < 2:
        raise ValueError("grouped ridge CV requires at least two target variants")
    scores: dict[str, float] = {}
    fold_counts: dict[str, int] = {}
    for alpha in RIDGE_ALPHAS:
        actual: list[float] = []
        predicted: list[float] = []
        for target in targets:
            train_rows = [row for row in rows if row["target"] != target]
            validation_rows = [row for row in rows if row["target"] == target]
            model = _fit_ridge_model(train_rows, feature_names, alpha)
            actual.extend(float(row["y"]) for row in validation_rows)
            predicted.extend(_predict_ridge(model, validation_rows).tolist())
        error = np.asarray(predicted) - np.asarray(actual)
        scores[f"{alpha:g}"] = float(np.sqrt(np.mean(error * error)))
        fold_counts[f"{alpha:g}"] = len(targets)
    selected = min(RIDGE_ALPHAS, key=lambda value: (scores[f"{value:g}"], value))
    return {
        "alpha": float(selected),
        "cv_rmse_by_alpha": scores,
        "cv_fold_count_by_alpha": fold_counts,
        "grouping": "leave_one_target_variant_out_all_coverages_together",
        "model": _fit_ridge_model(rows, feature_names, selected),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if actual.size < 2:
        return None
    left = _rankdata(actual)
    right = _rankdata(predicted)
    if float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _regression_metrics(
    rows: list[dict[str, Any]],
    prediction: np.ndarray,
    *,
    majority_positive: bool,
) -> dict[str, Any]:
    actual = np.asarray([float(row["y"]) for row in rows], dtype=np.float64)
    error = prediction - actual
    eligible = np.abs(actual) >= 0.05
    if eligible.any():
        sign_accuracy = float(np.mean((prediction[eligible] > 0.0) == (actual[eligible] > 0.0)))
        majority_accuracy = float(np.mean((actual[eligible] > 0.0) == majority_positive))
    else:
        sign_accuracy = None
        majority_accuracy = None
    calibration_slope = None
    calibration_intercept = None
    if prediction.size >= 2 and float(prediction.std()) > 1e-12:
        design = np.column_stack([np.ones(prediction.size), prediction])
        calibration_intercept, calibration_slope = np.linalg.lstsq(
            design, actual, rcond=None
        )[0].tolist()
    per_family_rmse = {}
    for family in TARGET_FAMILIES:
        mask = np.asarray([row["family"] == family for row in rows], dtype=bool)
        family_error = error[mask]
        per_family_rmse[family] = float(np.sqrt(np.mean(family_error * family_error)))
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "spearman": _spearman(actual, prediction),
        "eligible_sign_cells": int(eligible.sum()),
        "sign_accuracy": sign_accuracy,
        "majority_accuracy": majority_accuracy,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "per_family_rmse": per_family_rmse,
    }


def _choose_coverage(values: dict[int, float]) -> int:
    return max(COVERAGES, key=lambda coverage: (float(values[coverage]), coverage))


def _selection_metrics(
    heldout: list[dict[str, Any]],
    early_prediction: np.ndarray,
    response_prediction: np.ndarray,
) -> dict[str, Any]:
    by_target: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(heldout):
        by_target[row["target"]].append(idx)
    details = []
    for target in sorted(by_target):
        indices = by_target[target]
        rows = [heldout[idx] for idx in indices]
        actual = {8: 0.0, **{int(row["coverage"]): float(row["y"]) for row in rows}}
        step10 = {8: 0.0, **{int(row["coverage"]): float(row["d10"]) for row in rows}}
        early = {8: 0.0, **{int(heldout[idx]["coverage"]): float(early_prediction[idx]) for idx in indices}}
        response = {
            8: 0.0,
            **{
                int(heldout[idx]["coverage"]): float(response_prediction[idx])
                for idx in indices
            },
        }
        oracle = _choose_coverage(actual)
        choices = {
            "early": _choose_coverage(early),
            "response": _choose_coverage(response),
            "step10": _choose_coverage(step10),
            "c8": 8,
        }
        details.append(
            {
                "target": target,
                "family": rows[0]["family"],
                "oracle_coverage": oracle,
                "oracle_delta": actual[oracle],
                **{f"{name}_coverage": coverage for name, coverage in choices.items()},
                **{f"{name}_delta": actual[coverage] for name, coverage in choices.items()},
                **{
                    f"{name}_regret": actual[oracle] - actual[coverage]
                    for name, coverage in choices.items()
                },
            }
        )
    result: dict[str, Any] = {"targets": len(details), "details": details}
    for name in ("early", "response", "step10", "c8"):
        result[f"{name}_regret"] = float(np.mean([row[f"{name}_regret"] for row in details]))
        result[f"{name}_selected_delta"] = float(
            np.mean([row[f"{name}_delta"] for row in details])
        )
    return result


def _bootstrap_intervals(
    heldout: list[dict[str, Any]],
    early_prediction: np.ndarray,
    response_prediction: np.ndarray,
    *,
    replicates: int = 2000,
) -> dict[str, list[float]]:
    indexed = [
        {**row, "early_prediction": float(early_prediction[idx]),
         "response_prediction": float(response_prediction[idx])}
        for idx, row in enumerate(heldout)
    ]
    by_family_target: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in indexed:
        by_family_target[row["family"]][row["target"]].append(row)
    rng = np.random.default_rng(20260715)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        draw: list[dict[str, Any]] = []
        for family in TARGET_FAMILIES:
            targets = sorted(by_family_target[family])
            selected = rng.choice(targets, size=len(targets), replace=True)
            for duplicate, target in enumerate(selected.tolist()):
                draw.extend(
                    {**row, "target": f"{target}#boot{duplicate}"}
                    for row in by_family_target[family][target]
                )
        actual = np.asarray([row["y"] for row in draw], dtype=np.float64)
        early = np.asarray([row["early_prediction"] for row in draw], dtype=np.float64)
        response = np.asarray([row["response_prediction"] for row in draw], dtype=np.float64)
        early_rmse = float(np.sqrt(np.mean((early - actual) ** 2)))
        response_rmse = float(np.sqrt(np.mean((response - actual) ** 2)))
        samples["rmse_ratio"].append(response_rmse / max(early_rmse, 1e-12))
        samples["bias"].append(float(np.mean(response - actual)))
        selection = _selection_metrics(draw, early, response)
        samples["response_regret"].append(selection["response_regret"])
        samples["response_selected_delta"].append(selection["response_selected_delta"])
    return {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for name, values in samples.items()
    }


def aggregate_and_decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_raw_cells(rows)
    paired = _aggregate_seed_pairs(rows)
    train = [row for row in paired if row["split"] == "train"]
    heldout = [row for row in paired if row["split"] == "heldout"]
    if len(train) != 72 or len(heldout) != 36:
        raise ValueError("FIT-020 paired table must contain 72 train and 36 held-out rows")

    hashes_by_target = {
        target: {row["target_sha256"] for row in rows if row["target"] == target}
        for target in {row["target"] for row in rows}
    }
    integrity_checks = {
        "raw_rows_eq_432": len(rows) == 432,
        "paired_rows_eq_108": len(paired) == 108,
        "train_rows_eq_72": len(train) == 72,
        "heldout_rows_eq_36": len(heldout) == 36,
        "all_histories_complete": all(bool(row["history_complete"]) for row in rows),
        "all_final_counts_eq_40": all(int(row["final_count"]) == 40 for row in rows),
        "all_renders_finite": all(bool(row["renders_finite"]) for row in rows),
        "all_actions_match_frozen_chain": all(
            json.loads(row["action_groups"])
            == _coverage_actions(json.loads(row["ranking_groups"]))[int(row["coverage"])]
            for row in rows
        ),
        "all_unique_counts_match_coverage": all(
            int(row["unique_action_groups"]) == int(row["coverage"]) for row in rows
        ),
        "one_hash_per_target": all(len(hashes) == 1 for hashes in hashes_by_target.values())
        and len({next(iter(hashes)) for hashes in hashes_by_target.values()}) == 36,
    }
    integrity = {"checks": integrity_checks, "passes": all(integrity_checks.values())}

    eligible_train = [row for row in train if abs(float(row["y"])) >= 0.05]
    majority_positive = (
        sum(float(row["y"]) > 0.0 for row in eligible_train) * 2 >= len(eligible_train)
        if eligible_train
        else True
    )
    model_specs = {
        "intervention": INTERVENTION_FEATURES,
        "early": EARLY_FEATURES,
        "response": RESPONSE_FEATURES,
    }
    fitted = {
        name: _grouped_cv_ridge(train, features) for name, features in model_specs.items()
    }
    predictions = {
        name: _predict_ridge(payload["model"], heldout) for name, payload in fitted.items()
    }
    metrics = {
        name: _regression_metrics(
            heldout, prediction, majority_positive=majority_positive
        )
        for name, prediction in predictions.items()
    }
    actual = np.asarray([float(row["y"]) for row in heldout], dtype=np.float64)
    signal_sd = float(actual.std(ddof=1))
    signal_large = int(np.sum(np.abs(actual) >= 0.10))
    signal = {
        "heldout_cells": len(heldout),
        "sd": signal_sd,
        "abs_ge_0.10_cells": signal_large,
        "checks": {
            "sd_ge_0.10": signal_sd >= 0.10,
            "at_least_9_abs_ge_0.10": signal_large >= 9,
        },
    }
    signal["passes"] = all(signal["checks"].values())

    early = metrics["early"]
    response = metrics["response"]
    family_wins = sum(
        response["per_family_rmse"][family] < early["per_family_rmse"][family]
        for family in TARGET_FAMILIES
    )
    eligible_count = int(response["eligible_sign_cells"])
    sign_accuracy = response["sign_accuracy"]
    early_sign = early["sign_accuracy"]
    majority_sign = response["majority_accuracy"]
    prediction_checks = {
        "integrity_guard": bool(integrity["passes"]),
        "signal_guard": bool(signal["passes"]),
        "response_rmse_le_0.85_early": response["rmse"] <= 0.85 * early["rmse"],
        "at_least_18_sign_cells": eligible_count >= 18,
        "response_sign_ge_0.70": sign_accuracy is not None and sign_accuracy >= 0.70,
        "response_sign_ge_early_plus_0.10": (
            sign_accuracy is not None
            and early_sign is not None
            and sign_accuracy >= early_sign + 0.10
        ),
        "response_sign_ge_majority_plus_0.10": (
            sign_accuracy is not None
            and majority_sign is not None
            and sign_accuracy >= majority_sign + 0.10
        ),
        "response_better_at_least_4_families": family_wins >= 4,
        "abs_bias_le_0.10": abs(float(response["bias"])) <= 0.10,
    }
    prediction = {
        "intervention": metrics["intervention"],
        "early": early,
        "response": response,
        "intervention_rmse": metrics["intervention"]["rmse"],
        "early_rmse": early["rmse"],
        "response_rmse": response["rmse"],
        "response_rmse_ratio": response["rmse"] / max(early["rmse"], 1e-12),
        "eligible_sign_cells": eligible_count,
        "families_response_better": family_wins,
        "train_majority_positive": majority_positive,
        "checks": prediction_checks,
        "models": fitted,
    }
    prediction["passes"] = all(prediction_checks.values())

    selection = _selection_metrics(heldout, predictions["early"], predictions["response"])
    screening_checks = {
        "prediction_claim": bool(prediction["passes"]),
        "response_regret_lt_early": selection["response_regret"] < selection["early_regret"],
        "response_regret_lt_step10": selection["response_regret"] < selection["step10_regret"],
        "response_regret_lt_c8": selection["response_regret"] < selection["c8_regret"],
        "response_selected_delta_positive": selection["response_selected_delta"] > 0.0,
    }
    screening = {
        **selection,
        "checks": screening_checks,
    }
    screening["passes"] = all(screening_checks.values())

    for idx, row in enumerate(heldout):
        row["prediction_intervention"] = float(predictions["intervention"][idx])
        row["prediction_early"] = float(predictions["early"][idx])
        row["prediction_response"] = float(predictions["response"][idx])

    bootstrap = _bootstrap_intervals(
        heldout, predictions["early"], predictions["response"], replicates=2000
    )
    decision = "pass" if integrity["passes"] and screening["passes"] else "stop"
    return {
        "protocol": "fit020_ranked_deduplication_response_spectroscopy_v1",
        "integrity": integrity,
        "signal": signal,
        "prediction": prediction,
        "screening": screening,
        "bootstrap_95pct": bootstrap,
        "decision": decision,
        "paired_rows": paired,
    }


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for coverage in COVERAGES:
        values = [row for row in rows if int(row["coverage"]) == coverage]
        summary[str(coverage)] = {
            "trajectories": len(values),
            "psnr_step0": float(np.mean([row["psnr_step0"] for row in values])),
            "psnr_step10": float(np.mean([row["psnr_step10"] for row in values])),
            "psnr_step200": float(np.mean([row["psnr_step200"] for row in values])),
            "trajectory_auc": float(np.mean([row["trajectory_auc"] for row in values])),
            "late_asymptote": float(np.mean([row["late_asymptote"] for row in values])),
            "mean_fit_seconds": float(np.mean([row["fit_seconds"] for row in values])),
        }
    return summary


def _write_summary(outdir: Path, aggregate: dict[str, Any]) -> None:
    signal = aggregate["signal"]
    prediction = aggregate["prediction"]
    screening = aggregate["screening"]
    coverage = aggregate["coverage_summary"]
    lines = [
        "# FIT-020 ranked deduplication response spectroscopy",
        "",
        f"**Decision:** `{aggregate['decision']}`",
        "",
        "This frozen procedural assay tests a ranked one-ticket deduplication path, not abstract "
        "coverage. All arms have eight births and N=40; the result cannot establish compression, "
        "expressiveness, renderer speed, or optimizer-state continuation.",
        "",
        "## Held-out decision",
        "",
        f"- signal SD: `{signal['sd']:.6f} dB`; |y| >= 0.10: "
        f"`{signal['abs_ge_0.10_cells']}/36`; signal pass: `{signal['passes']}`",
        f"- early / response RMSE: `{prediction['early_rmse']:.6f}` / "
        f"`{prediction['response_rmse']:.6f}` dB; ratio "
        f"`{prediction['response_rmse_ratio']:.6f}`",
        f"- eligible sign cells: `{prediction['eligible_sign_cells']}`; response sign accuracy: "
        f"`{prediction['response']['sign_accuracy']}`",
        f"- response-better families: `{prediction['families_response_better']}/6`; "
        f"response bias: `{prediction['response']['bias']:.6f} dB`",
        f"- mean selection regret, early / response / step10 / C8: "
        f"`{screening['early_regret']:.6f}` / `{screening['response_regret']:.6f}` / "
        f"`{screening['step10_regret']:.6f}` / `{screening['c8_regret']:.6f} dB`",
        f"- response selected late delta over C8: "
        f"`{screening['response_selected_delta']:+.6f} dB`",
        f"- prediction / screening pass: `{prediction['passes']}` / `{screening['passes']}`",
        "",
        "## Coverage-arm descriptive means",
        "",
        "| Arm | Step 0 PSNR | Step 10 PSNR | Step 200 PSNR | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in COVERAGES:
        row = coverage[str(arm)]
        lines.append(
            f"| C{arm} | {row['psnr_step0']:.4f} | {row['psnr_step10']:.4f} | "
            f"{row['psnr_step200']:.4f} | {row['trajectory_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Peak, crossover, asymptote, and full AUC are descriptive because they use post-step-10 "
            "information. Timings include dense benchmark instrumentation and are not production "
            "performance evidence.",
        ]
    )
    outdir.joinpath("summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paired_outputs(output: Path, paired_rows: list[dict[str, Any]]) -> None:
    # Held-out rows carry frozen predictions while training rows intentionally do not. Build the
    # CSV header from the union instead of the first (training) row.
    write_csv(output / "paired_rows.csv", paired_rows, sort_fieldnames=True)
    output.joinpath("paired_rows.json").write_text(
        json.dumps(paired_rows, indent=2, allow_nan=False), encoding="utf-8"
    )


def _reproduction_command(
    *,
    outdir: str,
    size: int,
    seeds: tuple[int, ...],
    start_count: int,
    add_count: int,
    pre_iters: int,
    recovery_iters: int,
    device: str,
    render_chunk: int,
) -> str:
    command = [
        "python",
        "-m",
        "benchmarks.perturb_recover_spectroscopy",
        "--outdir",
        outdir,
        "--size",
        str(size),
        "--seeds",
        *map(str, seeds),
        "--start-count",
        str(start_count),
        "--add-count",
        str(add_count),
        "--pre-iters",
        str(pre_iters),
        "--recovery-iters",
        str(recovery_iters),
        "--device",
        device,
        "--render-chunk",
        str(render_chunk),
    ]
    prefixes = [
        f"{name}={shlex.quote(os.environ[name])}"
        for name in ("LD_PRELOAD", "PYTHONPATH")
        if os.environ.get(name)
    ]
    return " ".join([*prefixes, *map(shlex.quote, command)])


def run(
    *,
    outdir: str,
    size: int = 48,
    seeds: tuple[int, ...] = SEEDS,
    start_count: int = 32,
    add_count: int = 8,
    pre_iters: int = 40,
    recovery_iters: int = 200,
    device: str = "cpu",
    render_chunk: int = 512,
) -> list[dict[str, Any]]:
    if device != "cpu":
        raise ValueError("FIT-020's frozen protocol requires device='cpu'")
    if tuple(seeds) != SEEDS:
        raise ValueError("FIT-020's frozen protocol requires seeds=(0, 1, 2)")
    if size != 48:
        raise ValueError("FIT-020's frozen protocol requires size=48")
    if pre_iters != 40:
        raise ValueError("FIT-020's frozen protocol requires pre_iters=40")
    if (start_count, add_count, recovery_iters) != (32, 8, 200):
        raise ValueError("FIT-020 requires N/add/recovery=(32,8,200)")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"FIT-020 requires a fresh output directory, found files in {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[1]
    source = _source_provenance(root)
    targets = _target_suite(size)
    target_hashes = {
        target: _tensor_sha256(record["image"]) for target, record in targets.items()
    }
    target_manifest = "\n".join(
        f"{target}:{target_hashes[target]}" for target in targets
    ).encode("utf-8")
    target_manifest_sha256 = hashlib.sha256(target_manifest).hexdigest()
    if target_manifest_sha256 != TARGET_MANIFEST_SHA256:
        raise RuntimeError(
            "FIT-020 generated target manifest differs from the frozen preregistered hash"
        )
    if len(set(target_hashes.values())) != 36:
        raise RuntimeError("FIT-020 target hashes are not unique")
    old_hashes = {_tensor_sha256(image) for image in fit019_target_suite(size).values()}
    if set(target_hashes.values()) & old_hashes:
        raise RuntimeError("FIT-020 target suite collides with FIT-019")

    pre_cfg = FitConfig(
        iters=pre_iters,
        renderer="normalized",
        render_chunk=render_chunk,
        log_every=pre_iters,
        lr_schedule="none",
        checkpoint_policy="terminal",
    )
    action_cfg = FitConfig(
        renderer="normalized",
        render_chunk=render_chunk,
        refine_primitive="moment_preserving",
        split_scale=0.7,
    )
    recovery_cfg = FitConfig(
        iters=recovery_iters,
        renderer="normalized",
        render_chunk=render_chunk,
        log_every=1,
        lr_schedule="none",
        checkpoint_policy="terminal",
    )
    config = run_config(
        {
            "protocol": "fit020_ranked_deduplication_response_spectroscopy_v1",
            "frozen_task": "tasks/FIT-020-perturb-recover-spectroscopy.md",
            "size": size,
            "seeds": list(seeds),
            "start_count": start_count,
            "add_count": add_count,
            "final_count": start_count + add_count,
            "pre_iters": pre_iters,
            "recovery_iters": recovery_iters,
            "recovery_steps": list(RECOVERY_STEPS),
            "coverages": list(COVERAGES),
            "train_variants": list(TRAIN_VARIANTS),
            "heldout_variants": list(HELDOUT_VARIANTS),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "bootstrap_replicates": 2000,
            "targets": {
                target: {
                    "family": record["family"],
                    "variant": record["variant"],
                    "split": record["split"],
                    "params": list(record["params"]),
                    "sha256": target_hashes[target],
                }
                for target, record in targets.items()
            },
            "target_manifest_sha256": target_manifest_sha256,
            "target_parameter_table": TARGET_PARAMETER_TABLE,
            "renderer": "normalized",
            "strategy": "quadtree_wse",
            "opacity_mode": "constant_explicit_0.9",
            "ranking": "stable canonical alpha1 responsibility top8",
            "coverage_actions": {
                "C5": "g1,g2,g3,g4,g5,g3,g2,g1",
                "C6": "g1,g2,g3,g4,g5,g6,g2,g1",
                "C7": "g1,g2,g3,g4,g5,g6,g7,g1",
                "C8": "g1,g2,g3,g4,g5,g6,g7,g8",
            },
            "refine_primitive": "sequential_moment_preserving_on_canonical_field",
            "recovery_optimizer_state": "fresh Adam restart per arm",
            "history_semantics": "step t is pre-update render after t updates; terminal appended",
            "models": {
                "intervention_features": list(INTERVENTION_FEATURES),
                "early_features": list(EARLY_FEATURES),
                "response_features": list(RESPONSE_FEATURES),
                "cv": "leave_one_training_target_variant_out",
                "seed_policy": "pair within seed then average seeds before modeling",
            },
            "resolved_component_configs": {
                "structure_tensor": asdict(StructureTensorConfig()),
                "init_seed0": asdict(
                    InitConfig(
                        strategy="quadtree_wse",
                        num_gaussians=start_count,
                        seed=0,
                        opacity_mode="constant",
                        init_opacity=0.9,
                    )
                ),
                "pre_fit": asdict(pre_cfg),
                "action": asdict(action_cfg),
                "recovery": asdict(recovery_cfg),
            },
            "render_chunk": render_chunk,
            "claim_scope": (
                "procedural ranked-deduplication response assay; quality/convergence screening "
                "only; no production/compression/expressiveness/performance claim"
            ),
        },
        device=device,
    )
    config["cpu_environment"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "libc": list(platform.libc_ver()),
        "cpu_count": os.cpu_count(),
    }
    config["source"] = source
    config["original_invocation"] = _reproduction_command(
        outdir=outdir,
        size=size,
        seeds=seeds,
        start_count=start_count,
        add_count=add_count,
        pre_iters=pre_iters,
        recovery_iters=recovery_iters,
        device=device,
        render_chunk=render_chunk,
    )
    config["reproduction_command"] = _reproduction_command(
        outdir="{fresh_outdir}",
        size=size,
        seeds=seeds,
        start_count=start_count,
        add_count=add_count,
        pre_iters=pre_iters,
        recovery_iters=recovery_iters,
        device=device,
        render_chunk=render_chunk,
    )
    _write_source_snapshot(root, output, source)
    write_config(str(output), config)

    rows: list[dict[str, Any]] = []
    journal = output / "journal.jsonl"
    scfg = StructureTensorConfig()
    for target_index, (target_name, record) in enumerate(targets.items(), start=1):
        target_np = record["image"]
        target = torch.as_tensor(target_np, device=device, dtype=torch.float32)
        H, W = target.shape[:2]
        for seed in seeds:
            torch.manual_seed(seed)
            initial = build_field(
                target_np,
                InitConfig(
                    strategy="quadtree_wse",
                    num_gaussians=start_count,
                    seed=seed,
                    opacity_mode="constant",
                    init_opacity=0.9,
                ),
                scfg,
                device=device,
            )
            pre = fit(initial, target, pre_cfg, verbose=False)
            base = pre["field"].detached()
            if base.n != start_count:
                raise RuntimeError(f"FIT-020 pre-fit count changed to {base.n}")
            base_render = _render(base, pre_cfg, H, W)
            pre_loss = _objective_loss(base, target, pre_cfg)
            residual = _residual_stats(base_render, target)
            score_cfg = FitConfig(
                renderer="normalized",
                render_chunk=render_chunk,
                responsibility_mass_alpha=1.0,
            )
            score, _ = _responsibility_error_density_scores(
                base, target, base_render, score_cfg
            )
            ranking = _stable_topk(score, add_count)
            actions = _coverage_actions(ranking)
            score_sum = float(score.detach().sum().cpu())
            score_top8 = float(score[torch.as_tensor(ranking)].detach().sum().cpu())
            common = {
                "target": target_name,
                "family": record["family"],
                "variant": int(record["variant"]),
                "split": record["split"],
                "target_sha256": target_hashes[target_name],
                "source_combined_sha256": source["combined_sha256"],
                "seed": int(seed),
                "canonical_count": int(base.n),
                "canonical_field_sha256": _field_sha256(base),
                "ranking_groups": json.dumps(ranking, separators=(",", ":")),
                "ranking_hash": _action_hash(ranking),
                "pre_psnr": float(pre["trajectory_terminal_psnr"]),
                "pre_loss": pre_loss,
                "score_top8_mass_fraction": score_top8 / max(score_sum, 1e-12),
                **residual,
            }
            for coverage in COVERAGES:
                action = actions[coverage]
                action_start = time.perf_counter()
                grown = _sequential_moment_action(
                    base,
                    action,
                    target=target,
                    render_img=base_render,
                    cfg=action_cfg,
                )
                action_seconds = time.perf_counter() - action_start
                if grown.n != start_count + add_count:
                    raise RuntimeError(f"FIT-020 action C{coverage} produced N={grown.n}")
                grown_render = _render(grown, action_cfg, H, W)
                intervention_rms = float(
                    torch.sqrt(torch.mean((grown_render - base_render) ** 2)).detach().cpu()
                )
                torch.manual_seed(seed)
                recovered = fit(grown.detached(), target, recovery_cfg, verbose=False)
                recovered["trajectory_terminal_loss"] = _objective_loss(
                    recovered["field"], target, recovery_cfg
                )
                trajectory = _extract_trajectory(recovered)
                curve = np.asarray(trajectory["full_psnr"], dtype=np.float64)
                peak_index = int(np.argmax(curve))
                opacity0 = grown.opacity_values()
                opacity200 = recovered["field"].opacity_values()
                row: dict[str, Any] = {
                    **common,
                    "coverage": coverage,
                    "action_groups": json.dumps(action, separators=(",", ":")),
                    "action_hash": _action_hash(action),
                    "unique_action_groups": len(set(action)),
                    "repeated_action_count": add_count - len(set(action)),
                    "intervention_rms": intervention_rms,
                    "action_seconds": float(action_seconds),
                    "fit_seconds": float(recovered["fit_seconds"]),
                    "total_seconds": float(action_seconds) + float(recovered["fit_seconds"]),
                    "trajectory_auc": _terminal_auc(
                        trajectory["full_steps"], trajectory["full_psnr"]
                    ),
                    "peak_gain": float(curve[peak_index] - curve[0]),
                    "peak_step": peak_index,
                    "late_asymptote": float(curve[160:201].mean()),
                    "psnr_curve": json.dumps(
                        trajectory["full_psnr"], separators=(",", ":"), allow_nan=False
                    ),
                    "checkpoint_psnr": json.dumps(
                        trajectory["psnr_by_step"], separators=(",", ":"), allow_nan=False
                    ),
                    "checkpoint_loss": json.dumps(
                        trajectory["loss_by_step"], separators=(",", ":"), allow_nan=False
                    ),
                    "history_complete": trajectory["full_steps"] == list(range(201)),
                    "final_count": int(recovered["trajectory_terminal_n_gaussians"]),
                    "renders_finite": bool(
                        torch.isfinite(grown_render).all().detach().cpu()
                        and torch.isfinite(recovered["render"]).all().detach().cpu()
                    ),
                    "new_opacity_mean_step0": (
                        None
                        if opacity0 is None
                        else float(opacity0[start_count:].detach().mean().cpu())
                    ),
                    "new_opacity_mean_step200": (
                        None
                        if opacity200 is None
                        else float(opacity200[start_count:].detach().mean().cpu())
                    ),
                }
                if (
                    row["new_opacity_mean_step0"] is not None
                    and row["new_opacity_mean_step200"] is not None
                ):
                    row["new_opacity_delta200"] = (
                        row["new_opacity_mean_step200"] - row["new_opacity_mean_step0"]
                    )
                for step in RECOVERY_STEPS:
                    row[f"psnr_step{step}"] = float(trajectory["psnr_by_step"][step])
                    row[f"loss_step{step}"] = float(trajectory["loss_by_step"][step])
                rows.append(row)
                with journal.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"FIT-020 target {target_index:02d}/36 {target_name} seed {seed} complete",
                flush=True,
            )

    aggregate = aggregate_and_decide(rows)
    aggregate["coverage_summary"] = _coverage_summary(rows)
    paired_rows = aggregate.pop("paired_rows")
    write_csv(output / "rows.csv", rows)
    output.joinpath("rows.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_paired_outputs(output, paired_rows)
    output.joinpath("aggregate.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_summary(output, aggregate)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/fit020_response_spectroscopy")
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--start-count", type=int, default=32)
    parser.add_argument("--add-count", type=int, default=8)
    parser.add_argument("--pre-iters", type=int, default=40)
    parser.add_argument("--recovery-iters", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-chunk", type=int, default=512)
    args = parser.parse_args()
    run(
        outdir=args.outdir,
        size=args.size,
        seeds=tuple(args.seeds),
        start_count=args.start_count,
        add_count=args.add_count,
        pre_iters=args.pre_iters,
        recovery_iters=args.recovery_iters,
        device=args.device,
        render_chunk=args.render_chunk,
    )


if __name__ == "__main__":
    main()
