"""FIT-019 opacity-split gauge-equivalence and recovery audit.

The audit deliberately keeps equivalence-group IDs outside ``GaussianField``. Exact opacity
copies are used only to ask an allocator for an ordered action; every action is then replayed on
the same canonical checkpoint so duplicated optimizer state and raw row count cannot explain the
recovery result.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import time
from typing import Any

import numpy as np
import torch

from benchmarks.common import psnr_auc, run_config, write_config, write_csv
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import (
    _moment_preserving_duplicate_indices,
    _render,
    _responsibility_error_density_scores,
    _support_residual_scores,
    fit,
)
from structsplat.gaussians import GaussianField
from structsplat.init import build_field


TARGET_NAMES = (
    "vertical_step",
    "diagonal_step",
    "corner",
    "disk",
    "diagonal_line",
    "sinusoid",
    "checkerboard",
    "opposing_ramps",
)
ARMS = (
    "support_canonical",
    "support_gauge_row",
    "responsibility_alpha0.7_canonical",
    "responsibility_alpha0.7_gauge_row",
    "responsibility_alpha1_canonical",
    "responsibility_alpha1_gauge_row",
    "quotient_alpha1_canonical",
    "quotient_alpha1_gauge",
)
SOURCE_PATHS = (
    "benchmarks/gauge_equivalence_audit.py",
    "benchmarks/common.py",
    "benchmarks/_util.py",
    "tasks/FIT-019-opacity-gauge-equivalence.md",
    "tests/test_gauge_equivalence_audit.py",
    "pyproject.toml",
)
_EPS = 1e-8


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: np.ndarray | torch.Tensor) -> str:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _field_sha256(field: GaussianField) -> str:
    digest = hashlib.sha256()
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "opacities",
        "scale_max",
        "color_grads",
        "background_mask",
        "filter_variance",
    ):
        value = getattr(field, name)
        digest.update(name.encode("ascii") + b"\0")
        if value is None:
            digest.update(b"NONE\n")
        else:
            digest.update(_tensor_sha256(value).encode("ascii") + b"\n")
    return digest.hexdigest()


def _source_paths(root: Path) -> tuple[str, ...]:
    package_sources = tuple(
        str(path.relative_to(root))
        for path in sorted(root.joinpath("src/structsplat").glob("*.py"))
    )
    return tuple(sorted(set((*SOURCE_PATHS, *package_sources))))


def _source_provenance(root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in _source_paths(root):
        value = _sha256(root / relative)
        files[relative] = value
        combined.update(relative.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return {"combined_sha256": combined.hexdigest(), "files": files}


def _write_source_snapshot(root: Path, outdir: Path, source: dict[str, Any]) -> None:
    snapshot = outdir / "source_snapshot"
    for relative, expected_hash in source["files"].items():
        payload = root.joinpath(relative).read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"source changed while snapshotting {relative}")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _exact_logit(probability: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("opacity probabilities must be finite")
    if bool(((probability <= 0.0) | (probability >= 1.0)).any()):
        raise ValueError("exact opacity refinement requires probabilities strictly in (0, 1)")
    return torch.log(probability) - torch.log1p(-probability)


def exact_opacity_refinement(
    field: GaussianField,
    *,
    split_groups: set[int],
    fractions: tuple[float, ...] = (0.5, 0.5),
) -> tuple[GaussianField, torch.Tensor, torch.Tensor]:
    """Replace selected canonical rows by exact opacity-mass copies.

    Returns the refined field, external canonical group IDs, and the opacity fraction carried by
    each output row. The input is intentionally restricted to one row per canonical group.
    """
    if field.opacities is None:
        raise ValueError("exact opacity refinement requires explicit opacity logits")
    if not fractions or any(not math.isfinite(v) or v <= 0.0 for v in fractions):
        raise ValueError("fractions must be finite and positive")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("opacity fractions must sum to one")
    unknown = split_groups - set(range(field.n))
    if unknown:
        raise ValueError(f"unknown canonical group IDs: {sorted(unknown)}")

    row_ids: list[int] = []
    row_fractions: list[float] = []
    group_ids: list[int] = []
    for group in range(field.n):
        values = fractions if group in split_groups else (1.0,)
        for fraction in values:
            row_ids.append(group)
            row_fractions.append(float(fraction))
            group_ids.append(group)
    idx = torch.as_tensor(row_ids, device=field.means.device, dtype=torch.long)
    fraction_t = torch.as_tensor(row_fractions, device=field.means.device, dtype=field.means.dtype)
    opacity = field.opacity_values()
    assert opacity is not None
    refined_opacity = opacity.detach()[idx] * fraction_t

    def take(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach()[idx].clone()

    refined = GaussianField(
        field.means.detach()[idx].clone(),
        field.log_scales.detach()[idx].clone(),
        field.rotations.detach()[idx].clone(),
        field.colors.detach()[idx].clone(),
        _exact_logit(refined_opacity),
        take(field.scale_max),
        take(field.color_grads),
        take(field.background_mask),
        take(field.filter_variance),
    )
    return (
        refined,
        torch.as_tensor(group_ids, device=field.means.device, dtype=torch.long),
        fraction_t,
    )


def _assert_exact_group_members(field: GaussianField, group_ids: torch.Tensor) -> None:
    if group_ids.ndim != 1 or int(group_ids.numel()) != field.n:
        raise ValueError("group_ids must contain one label per field row")
    if group_ids.dtype != torch.long:
        raise ValueError("group_ids must use torch.long")
    if bool((group_ids < 0).any()):
        raise ValueError("group_ids must be non-negative")
    labels = torch.unique(group_ids, sorted=True)
    attributes = (
        ("means", field.means),
        ("log_scales", field.log_scales),
        ("rotations", field.rotations),
        ("colors", field.colors),
        ("scale_max", field.scale_max),
        ("color_grads", field.color_grads),
        ("filter_variance", field.filter_variance),
    )
    for label in labels.tolist():
        idx = torch.nonzero(group_ids == int(label), as_tuple=False).reshape(-1)
        if idx.numel() <= 1:
            continue
        for name, value in attributes:
            if value is None:
                continue
            reference = value.detach()[idx[0]]
            if not all(torch.equal(value.detach()[row], reference) for row in idx[1:]):
                raise ValueError(f"group {label} has non-identical {name}")
        if field.background_mask is not None:
            roles = field.background_mask.detach()[idx].to(dtype=torch.bool)
            if not bool((roles == roles[0]).all()):
                raise ValueError(f"group {label} mixes background and detail roles")


def group_responsibility_scores(
    field: GaussianField,
    group_ids: torch.Tensor,
    components: dict[str, torch.Tensor],
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate responsibility mass/error before applying the mass exponent."""
    _assert_exact_group_members(field, group_ids)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be finite in (0, 1]")
    mass = components["mass"]
    error = components["error"]
    if mass.ndim != 1 or error.shape != mass.shape or int(mass.numel()) != field.n:
        raise ValueError("responsibility components must contain one mass/error per row")
    labels, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    group_mass = mass.new_zeros(labels.numel())
    group_error = error.new_zeros(labels.numel())
    group_mass.index_add_(0, inverse, mass)
    group_error.index_add_(0, inverse, error)
    score = group_error / group_mass.clamp_min(_EPS).pow(float(alpha))
    return labels, score, group_mass, group_error


def _stable_topk(scores: torch.Tensor, k: int) -> list[int]:
    if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
        raise ValueError("stable top-k requires a finite vector")
    k = min(max(int(k), 0), int(scores.numel()))
    values = scores.detach().cpu().tolist()
    return sorted(range(len(values)), key=lambda idx: (-float(values[idx]), idx))[:k]


def _row_action(scores: torch.Tensor, group_ids: torch.Tensor, k: int) -> list[int]:
    return [int(group_ids[idx]) for idx in _stable_topk(scores, k)]


def _group_action(labels: torch.Tensor, scores: torch.Tensor, k: int) -> list[int]:
    return [int(labels[idx]) for idx in _stable_topk(scores, k)]


def _action_hash(action: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps(list(action), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _action_jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    return 1.0 if not union else len(set(left) & set(right)) / len(union)


def _multiset_jaccard(left: list[int], right: list[int]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = left_counts.keys() | right_counts.keys()
    denominator = sum(max(left_counts[label], right_counts[label]) for label in labels)
    if denominator == 0:
        return 1.0
    numerator = sum(min(left_counts[label], right_counts[label]) for label in labels)
    return numerator / denominator


def _ordered_topk_agreement(left: list[int], right: list[int]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return sum(a == b for a, b in zip(left, right, strict=False)) / denominator


def _target_suite(size: int = 48) -> dict[str, np.ndarray]:
    if size < 8:
        raise ValueError("procedural target size must be at least 8")
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    u = x / float(size - 1)
    v = y / float(size - 1)
    targets: dict[str, np.ndarray] = {}

    vertical = np.empty((size, size, 3), dtype=np.float32)
    vertical[:, : size // 2] = (0.08, 0.22, 0.88)
    vertical[:, size // 2 :] = (0.92, 0.78, 0.10)
    targets["vertical_step"] = vertical

    diagonal = np.where(
        (u + 0.85 * v > 0.92)[..., None],
        np.array([0.95, 0.18, 0.12], np.float32),
        np.array([0.05, 0.72, 0.88], np.float32),
    )
    targets["diagonal_step"] = diagonal.astype(np.float32)

    corner = np.zeros((size, size, 3), dtype=np.float32)
    corner[(u < 0.5) & (v < 0.5)] = (0.05, 0.10, 0.85)
    corner[(u >= 0.5) & (v < 0.5)] = (0.85, 0.10, 0.10)
    corner[(u < 0.5) & (v >= 0.5)] = (0.10, 0.80, 0.15)
    corner[(u >= 0.5) & (v >= 0.5)] = (0.90, 0.80, 0.10)
    targets["corner"] = corner

    radius = np.sqrt((u - 0.51) ** 2 + (v - 0.47) ** 2)
    disk_mask = radius <= 0.27
    disk = np.empty((size, size, 3), dtype=np.float32)
    disk[:] = (0.06, 0.14, 0.28)
    disk[disk_mask] = (0.92, 0.72, 0.16)
    targets["disk"] = disk

    distance = np.abs(v - (0.18 + 0.63 * u))
    line = np.empty((size, size, 3), dtype=np.float32)
    line[:] = (0.08, 0.08, 0.12)
    line[distance < 1.35 / size] = (0.96, 0.90, 0.20)
    targets["diagonal_line"] = line

    phase = 2.0 * np.pi * (5.0 * u + 1.25 * v)
    sinusoid = np.stack(
        [
            0.5 + 0.45 * np.sin(phase),
            0.5 + 0.45 * np.sin(phase + 2.0 * np.pi / 3.0),
            0.5 + 0.45 * np.sin(phase + 4.0 * np.pi / 3.0),
        ],
        axis=2,
    )
    targets["sinusoid"] = sinusoid.astype(np.float32)

    checker_mask = ((np.floor(u * 8) + np.floor(v * 8)).astype(np.int64) % 2) == 0
    checker = np.empty((size, size, 3), dtype=np.float32)
    checker[checker_mask] = (0.92, 0.18, 0.78)
    checker[~checker_mask] = (0.08, 0.82, 0.24)
    targets["checkerboard"] = checker

    opposing = np.stack([u, 1.0 - u, 0.15 + 0.7 * v], axis=2)
    targets["opposing_ramps"] = opposing.astype(np.float32)
    assert tuple(targets) == TARGET_NAMES
    return targets


def _counterexample_field() -> GaussianField:
    opacity = np.asarray([0.8, 0.4, 0.6], dtype=np.float32)
    logits = np.log(opacity / (1.0 - opacity))
    return GaussianField.from_numpy(
        means=np.asarray([[3.0, 3.0], [8.0, 3.0], [5.0, 8.0]], dtype=np.float32),
        scales=np.asarray([[2.0, 1.5], [1.7, 2.2], [2.1, 1.3]], dtype=np.float32),
        angles=np.asarray([0.2, 0.5, -0.2], dtype=np.float32),
        colors=np.asarray([[0.8, 0.1, 0.2], [0.1, 0.8, 0.3], [0.2, 0.3, 0.9]], dtype=np.float32),
        opacities=logits,
    )


def run_algebra_audit() -> dict[str, Any]:
    field = _counterexample_field()
    target = torch.rand(12, 12, 3, generator=torch.Generator().manual_seed(9))
    base_groups = torch.arange(field.n, dtype=torch.long)
    results: dict[str, Any] = {}
    for alpha_label, alpha in (("alpha0.7", 0.7), ("alpha1", 1.0)):
        cfg = FitConfig(renderer="normalized", responsibility_mass_alpha=alpha, render_chunk=1)
        base_render = _render(field, cfg, 12, 12)
        base_score, base_components = _responsibility_error_density_scores(
            field, target, base_render, cfg
        )
        split_group = int(_stable_topk(base_score, 1)[0])
        gauge, gauge_groups, fractions = exact_opacity_refinement(field, split_groups={split_group})
        gauge_render = _render(gauge, cfg, 12, 12)
        gauge_score, gauge_components = _responsibility_error_density_scores(
            gauge, target, gauge_render, cfg
        )
        labels, group_score, group_mass, group_error = group_responsibility_scores(
            gauge, gauge_groups, gauge_components, alpha=alpha
        )
        base_labels, base_group_score, base_mass, base_error = group_responsibility_scores(
            field, base_groups, base_components, alpha=alpha
        )
        expected = base_score[gauge_groups] * fractions.pow(1.0 - alpha)
        base_action = _row_action(base_score, base_groups, 2)
        gauge_action = _row_action(gauge_score, gauge_groups, 2)
        quotient_action = _group_action(labels, group_score, 2)
        results[alpha_label] = {
            "alpha": alpha,
            "split_group": split_group,
            "render_max_abs": float((base_render - gauge_render).abs().max()),
            "denominator_max_abs": float(
                (base_components["denominator"] - gauge_components["denominator"]).abs().max()
            ),
            "child_law_max_abs": float((gauge_score - expected).abs().max()),
            "group_score_max_abs": float((group_score - base_group_score).abs().max()),
            "group_mass_max_abs": float((group_mass - base_mass).abs().max()),
            "group_error_max_abs": float((group_error - base_error).abs().max()),
            "labels_equal": bool(torch.equal(labels, base_labels)),
            "base_action": base_action,
            "gauge_row_action": gauge_action,
            "quotient_action": quotient_action,
            "gauge_row_changed": gauge_action != base_action,
            "quotient_restored": quotient_action == base_action,
            "gauge_unique_groups": len(set(gauge_action)),
        }
    results["positive_control"] = {
        "alpha0.7_changed": results["alpha0.7"]["gauge_row_changed"],
        "alpha1_duplicate_ticket": results["alpha1"]["gauge_unique_groups"] < 2,
        "quotient_restored_both": all(
            results[key]["quotient_restored"] for key in ("alpha0.7", "alpha1")
        ),
    }
    return results


def _sequential_moment_action(
    base: GaussianField,
    action: list[int],
    *,
    target: torch.Tensor,
    render_img: torch.Tensor,
    cfg: FitConfig,
) -> GaussianField:
    field = base.detached()
    lineage = torch.arange(field.n, device=field.means.device, dtype=torch.long)
    H, W = target.shape[:2]
    for group in action:
        candidates = torch.nonzero(lineage == int(group), as_tuple=False).reshape(-1)
        if candidates.numel() == 0:
            raise ValueError(f"action references unknown canonical group {group}")
        opacity = field.opacity_values()
        if opacity is None:
            raise ValueError("FIT-019 actions require explicit opacity")
        candidate_rows = candidates.detach().cpu().tolist()
        selected = min(
            candidate_rows,
            key=lambda row: (-float(opacity[row].detach().cpu()), int(row)),
        )
        field, added = _moment_preserving_duplicate_indices(
            field,
            torch.as_tensor([selected], device=field.means.device, dtype=torch.long),
            cfg,
            H,
            W,
        )
        if added != 1:
            raise RuntimeError(f"expected one sequential split, got {added}")
        lineage = torch.cat(
            [
                lineage,
                torch.as_tensor([group], device=lineage.device, dtype=torch.long),
            ]
        )
    if hasattr(field, "_new_parent_idx"):
        delattr(field, "_new_parent_idx")
    return field


def _evaluate_action(
    base: GaussianField,
    target: torch.Tensor,
    before: torch.Tensor,
    action: list[int],
    *,
    seed: int,
    render_chunk: int,
) -> dict[str, Any]:
    action_cfg = FitConfig(
        renderer="normalized",
        render_chunk=render_chunk,
        refine_primitive="moment_preserving",
        split_scale=0.7,
    )
    start = time.perf_counter()
    grown = _sequential_moment_action(
        base, action, target=target, render_img=before, cfg=action_cfg
    )
    action_seconds = time.perf_counter() - start
    H, W = target.shape[:2]
    immediate = _render(grown, action_cfg, H, W)
    recovered: dict[int, dict[str, Any]] = {}
    for horizon in (20, 100):
        torch.manual_seed(seed)
        recover_cfg = FitConfig(
            iters=horizon,
            renderer="normalized",
            render_chunk=render_chunk,
            log_every=1,
        )
        result = fit(grown.detached(), target, recover_cfg, verbose=False)
        recovered[horizon] = {
            "psnr": float(result["psnr"]),
            "ms_ssim": float(result["ms_ssim"]),
            "auc": psnr_auc(result["history"], nominal_iters=horizon),
            "fit_seconds": float(result["fit_seconds"]),
            "count": int(result["field"].n),
            "finite": bool(torch.isfinite(result["render"]).all().detach().cpu()),
        }
    return {
        "immediate_psnr": float(M.psnr(immediate, target)),
        "immediate_ms_ssim": float(M.ms_ssim(immediate, target)),
        "post20_psnr": recovered[20]["psnr"],
        "post20_ms_ssim": recovered[20]["ms_ssim"],
        "post20_auc": recovered[20]["auc"],
        "post100_psnr": recovered[100]["psnr"],
        "post100_ms_ssim": recovered[100]["ms_ssim"],
        "post100_auc": recovered[100]["auc"],
        "action_seconds": action_seconds,
        "fit20_seconds": recovered[20]["fit_seconds"],
        "fit100_seconds": recovered[100]["fit_seconds"],
        "total100_seconds": action_seconds + recovered[100]["fit_seconds"],
        "final_count": int(grown.n),
        "post20_count": recovered[20]["count"],
        "post100_count": recovered[100]["count"],
        "renders_finite": bool(torch.isfinite(immediate).all().detach().cpu())
        and recovered[20]["finite"]
        and recovered[100]["finite"],
    }


def _arm_specs(
    base: GaussianField,
    gauge: GaussianField,
    base_groups: torch.Tensor,
    gauge_groups: torch.Tensor,
    target: torch.Tensor,
    base_render: torch.Tensor,
    gauge_render: torch.Tensor,
    *,
    k: int,
    render_chunk: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    def add_row_arm(name: str, view: str, alpha: float | None) -> None:
        field = base if view == "canonical" else gauge
        groups = base_groups if view == "canonical" else gauge_groups
        image = base_render if view == "canonical" else gauge_render
        cfg = FitConfig(
            renderer="normalized",
            render_chunk=render_chunk,
            responsibility_mass_alpha=1.0 if alpha is None else alpha,
        )
        start = time.perf_counter()
        if alpha is None:
            residual = (image - target).abs().mean(dim=2)
            score = _support_residual_scores(field, residual, cfg)
        else:
            score, _ = _responsibility_error_density_scores(field, target, image, cfg)
        seconds = time.perf_counter() - start
        specs.append(
            {
                "arm": name,
                "view": view,
                "selector": "support_row" if alpha is None else "responsibility_row",
                "alpha": alpha,
                "action": _row_action(score, groups, k),
                "score_seconds": seconds,
                "scores_finite": bool(torch.isfinite(score).all().detach().cpu()),
            }
        )

    add_row_arm("support_canonical", "canonical", None)
    add_row_arm("support_gauge_row", "gauge", None)
    add_row_arm("responsibility_alpha0.7_canonical", "canonical", 0.7)
    add_row_arm("responsibility_alpha0.7_gauge_row", "gauge", 0.7)
    add_row_arm("responsibility_alpha1_canonical", "canonical", 1.0)
    add_row_arm("responsibility_alpha1_gauge_row", "gauge", 1.0)

    group_data: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = {}
    for view, field, groups, image in (
        ("canonical", base, base_groups, base_render),
        ("gauge", gauge, gauge_groups, gauge_render),
    ):
        cfg = FitConfig(
            renderer="normalized",
            render_chunk=render_chunk,
            responsibility_mass_alpha=1.0,
        )
        start = time.perf_counter()
        row_score, components = _responsibility_error_density_scores(field, target, image, cfg)
        labels, group_score, _, _ = group_responsibility_scores(
            field, groups, components, alpha=1.0
        )
        seconds = time.perf_counter() - start
        group_data[view] = (labels, group_score, components)
        specs.append(
            {
                "arm": f"quotient_alpha1_{view}",
                "view": view,
                "selector": "quotient_responsibility",
                "alpha": 1.0,
                "action": _group_action(labels, group_score, k),
                "score_seconds": seconds,
                "scores_finite": bool(torch.isfinite(row_score).all().detach().cpu())
                and bool(torch.isfinite(group_score).all().detach().cpu()),
            }
        )

    base_labels, base_group_score, base_components = group_data["canonical"]
    gauge_labels, gauge_group_score, gauge_components = group_data["gauge"]
    if not torch.equal(base_labels, gauge_labels):
        raise RuntimeError("canonical/gauge group label mismatch")
    denom_delta = (base_components["denominator"] - gauge_components["denominator"]).abs()
    diagnostics["denominator_max_abs"] = float(denom_delta.max())
    diagnostics["quotient_alpha1_max_abs"] = float(
        (base_group_score - gauge_group_score).abs().max()
    )
    diagnostics["quotient_alpha1_max_rel"] = float(
        (
            (base_group_score - gauge_group_score).abs() / base_group_score.abs().clamp_min(_EPS)
        ).max()
    )

    for alpha in (0.7, 1.0):
        base_cfg = FitConfig(
            renderer="normalized",
            render_chunk=render_chunk,
            responsibility_mass_alpha=alpha,
        )
        gauge_cfg = FitConfig(
            renderer="normalized",
            render_chunk=render_chunk,
            responsibility_mass_alpha=alpha,
        )
        base_row, base_comp = _responsibility_error_density_scores(
            base, target, base_render, base_cfg
        )
        gauge_row, gauge_comp = _responsibility_error_density_scores(
            gauge, target, gauge_render, gauge_cfg
        )
        labels, group_score, _, _ = group_responsibility_scores(
            gauge, gauge_groups, gauge_comp, alpha=alpha
        )
        _, canonical_group_score, _, _ = group_responsibility_scores(
            base, base_groups, base_comp, alpha=alpha
        )
        diagnostics[f"quotient_alpha{alpha:g}_max_abs"] = float(
            (group_score - canonical_group_score).abs().max()
        )
        diagnostics[f"quotient_alpha{alpha:g}_max_rel"] = float(
            (
                (group_score - canonical_group_score).abs()
                / canonical_group_score.abs().clamp_min(_EPS)
            ).max()
        )
        diagnostics[f"quotient_alpha{alpha:g}_action_match"] = _group_action(
            labels, group_score, k
        ) == _group_action(base_labels, canonical_group_score, k)
        expected_fraction = torch.where(
            gauge_groups % 2 == 0,
            torch.full_like(gauge_row, 0.5),
            torch.ones_like(gauge_row),
        )
        expected = base_row[gauge_groups] * expected_fraction.pow(1.0 - alpha)
        diagnostics[f"alpha{alpha:g}_child_law_max_abs"] = float((gauge_row - expected).abs().max())
        if not torch.equal(labels, base_labels):
            raise RuntimeError("group labels changed during alpha diagnostic")

    reference_arm = {
        "support_canonical": "support_canonical",
        "support_gauge_row": "support_canonical",
        "responsibility_alpha0.7_canonical": "responsibility_alpha0.7_canonical",
        "responsibility_alpha0.7_gauge_row": "responsibility_alpha0.7_canonical",
        "responsibility_alpha1_canonical": "responsibility_alpha1_canonical",
        "responsibility_alpha1_gauge_row": "responsibility_alpha1_canonical",
        "quotient_alpha1_canonical": "quotient_alpha1_canonical",
        "quotient_alpha1_gauge": "quotient_alpha1_canonical",
    }
    action_by_arm = {spec["arm"]: list(spec["action"]) for spec in specs}
    for spec in specs:
        action = list(spec["action"])
        reference = action_by_arm[reference_arm[spec["arm"]]]
        spec["projected_topk_order_agreement"] = _ordered_topk_agreement(action, reference)
        spec["projected_topk_multiset_jaccard"] = _multiset_jaccard(action, reference)
        spec["projected_topk_unique_jaccard"] = _action_jaccard(action, reference)
        spec["projected_topk_exact"] = action == reference
    return specs, diagnostics


def _cluster_deltas(
    rows: list[dict[str, Any]], left_arm: str, right_arm: str, metric: str
) -> tuple[list[float], dict[str, float]]:
    by_key = {(row["target"], int(row["seed"]), row["arm"]): row for row in rows}
    per_target: dict[str, list[float]] = defaultdict(list)
    for target in TARGET_NAMES:
        for seed in (0, 1):
            left = by_key[(target, seed, left_arm)]
            right = by_key[(target, seed, right_arm)]
            per_target[target].append(float(left[metric]) - float(right[metric]))
    target_mean = {target: float(np.mean(values)) for target, values in per_target.items()}
    return list(target_mean.values()), target_mean


def aggregate_and_decide(rows: list[dict[str, Any]], algebra: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        (target, seed, arm) for target in TARGET_NAMES for seed in (0, 1) for arm in ARMS
    }
    row_keys = [(str(row["target"]), int(row["seed"]), str(row["arm"])) for row in rows]
    key_counts = Counter(row_keys)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count != 1)
    missing_keys = sorted(expected_keys - set(row_keys))
    unexpected_keys = sorted(set(row_keys) - expected_keys)
    if duplicate_keys or missing_keys or unexpected_keys:
        raise ValueError(
            "FIT-019 requires exactly one row per target/seed/arm; "
            f"duplicates={duplicate_keys[:3]}, missing={missing_keys[:3]}, "
            f"unexpected={unexpected_keys[:3]}"
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    arms: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        values = grouped[arm]
        arms[arm] = {
            "n": float(len(values)),
            "immediate_psnr": float(np.mean([row["immediate_psnr"] for row in values])),
            "post20_psnr": float(np.mean([row["post20_psnr"] for row in values])),
            "post100_psnr": float(np.mean([row["post100_psnr"] for row in values])),
            "post20_ms_ssim": float(np.mean([row["post20_ms_ssim"] for row in values])),
            "post100_ms_ssim": float(np.mean([row["post100_ms_ssim"] for row in values])),
            "post100_auc": float(np.mean([row["post100_auc"] for row in values])),
            "score_seconds": float(np.mean([row["score_seconds"] for row in values])),
            "total100_seconds": float(np.mean([row["total100_seconds"] for row in values])),
            "unique_action_groups": float(np.mean([row["unique_action_groups"] for row in values])),
        }

    q20, q20_by_target = _cluster_deltas(
        rows, "quotient_alpha1_gauge", "responsibility_alpha1_gauge_row", "post20_psnr"
    )
    q100, q100_by_target = _cluster_deltas(
        rows, "quotient_alpha1_gauge", "responsibility_alpha1_gauge_row", "post100_psnr"
    )
    support20, _ = _cluster_deltas(
        rows, "quotient_alpha1_gauge", "support_canonical", "post20_psnr"
    )
    support100, _ = _cluster_deltas(
        rows, "quotient_alpha1_gauge", "support_canonical", "post100_psnr"
    )
    q_rows = grouped["quotient_alpha1_gauge"]
    q_canonical = {
        (row["target"], int(row["seed"])): row for row in grouped["quotient_alpha1_canonical"]
    }
    quotient_action_match = all(
        row["action_hash"] == q_canonical[(row["target"], int(row["seed"]))]["action_hash"]
        for row in q_rows
    )
    raw_canonical = {
        (row["target"], int(row["seed"])): row for row in grouped["responsibility_alpha1_canonical"]
    }
    changed_targets = 0
    raw_set_jaccard_by_target: dict[str, float] = {}
    raw_multiset_jaccard_by_target: dict[str, float] = {}
    raw_changed_seeds_by_target: dict[str, int] = {}
    for target in TARGET_NAMES:
        set_agreements: list[float] = []
        multiset_agreements: list[float] = []
        changed_seeds = 0
        for seed in (0, 1):
            canonical_action = json.loads(raw_canonical[(target, seed)]["action_groups"])
            gauge_action = json.loads(
                next(
                    row["action_groups"]
                    for row in grouped["responsibility_alpha1_gauge_row"]
                    if row["target"] == target and int(row["seed"]) == seed
                )
            )
            set_agreements.append(_action_jaccard(canonical_action, gauge_action))
            multiset_agreements.append(_multiset_jaccard(canonical_action, gauge_action))
            changed_seeds += int(Counter(canonical_action) != Counter(gauge_action))
        raw_set_jaccard_by_target[target] = float(np.mean(set_agreements))
        raw_multiset_jaccard_by_target[target] = float(np.mean(multiset_agreements))
        raw_changed_seeds_by_target[target] = changed_seeds
        # Count a target only when both independent seeds change group multiplicity.
        changed_targets += int(changed_seeds == 2)

    parity_ok = all(float(row["render_max_abs"]) <= 2e-6 for row in rows)
    quotient_score_ok = all(
        float(row["quotient_alpha1_max_abs"]) <= 2e-5
        and float(row["quotient_alpha1_max_rel"]) <= 2e-5
        and float(row["quotient_alpha0.7_max_abs"]) <= 2e-5
        and float(row["quotient_alpha0.7_max_rel"]) <= 2e-5
        for row in rows
    )
    quotient_actions_both_alphas = all(
        bool(row["quotient_alpha1_action_match"]) and bool(row["quotient_alpha0.7_action_match"])
        for row in rows
    )
    child_law_ok = all(
        float(row["alpha1_child_law_max_abs"]) <= 2e-5
        and float(row["alpha0.7_child_law_max_abs"]) <= 2e-5
        for row in rows
    )
    positive_control = algebra["positive_control"]
    algebra_numeric_ok = all(
        float(algebra[key]["render_max_abs"]) <= 2e-6
        and float(algebra[key]["denominator_max_abs"]) <= 2e-6
        and float(algebra[key]["child_law_max_abs"]) <= 2e-5
        and float(algebra[key]["group_score_max_abs"]) <= 2e-5
        and float(algebra[key]["group_mass_max_abs"]) <= 2e-5
        and float(algebra[key]["group_error_max_abs"]) <= 2e-5
        and bool(algebra[key]["labels_equal"])
        for key in ("alpha0.7", "alpha1")
    )
    commutation_checks = {
        "render_parity_le_2e-6": parity_ok,
        "quotient_scores_both_alphas_within_2e-5": quotient_score_ok,
        "quotient_actions_both_alphas_match_16_of_16": quotient_actions_both_alphas,
        "child_laws_within_2e-5": child_law_ok,
        "algebra_numeric_invariants": algebra_numeric_ok,
        "algebra_positive_control": bool(positive_control["alpha0.7_changed"])
        and bool(positive_control["alpha1_duplicate_ticket"])
        and bool(positive_control["quotient_restored_both"]),
        "quotient_actions_match_16_of_16": quotient_action_match,
        "raw_alpha1_changes_ge_6_of_8_targets": changed_targets >= 6,
        "all_scores_and_renders_finite": bool(rows)
        and all(row["scores_finite"] and row["renders_finite"] for row in rows),
    }
    commutation_confirmed = all(commutation_checks.values())

    quotient_time = arms["quotient_alpha1_gauge"]["total100_seconds"]
    support_time = arms["support_canonical"]["total100_seconds"]
    overhead = quotient_time / support_time - 1.0 if support_time > 0.0 else float("inf")
    utility_checks = {
        "commutation_confirmed": commutation_confirmed,
        "quotient_actions_match_16_of_16": quotient_action_match,
        "post20_gain_ge_0.05_db": float(np.mean(q20)) >= 0.05,
        "post20_wins_ge_6_of_8_targets": sum(delta > 0.0 for delta in q20) >= 6,
        "post100_delta_ge_minus_0.02_db": float(np.mean(q100)) >= -0.02,
        "post20_vs_support_ge_minus_0.05_db": float(np.mean(support20)) >= -0.05,
        "post100_vs_support_ge_minus_0.05_db": float(np.mean(support100)) >= -0.05,
        "total100_overhead_le_0.15": overhead <= 0.15,
        "all_immediate_and_recovered_counts_eq_40": bool(rows)
        and all(
            int(row["final_count"]) == 40
            and int(row["post20_count"]) == 40
            and int(row["post100_count"]) == 40
            for row in rows
        ),
        "all_scores_and_renders_finite": bool(rows)
        and all(row["scores_finite"] and row["renders_finite"] for row in rows),
    }
    return {
        "arms": arms,
        "commutation": {
            "confirmed": commutation_confirmed,
            "checks": commutation_checks,
            "raw_alpha1_changed_target_count": changed_targets,
            "raw_alpha1_changed_seed_count_by_target": raw_changed_seeds_by_target,
            "raw_alpha1_multiset_jaccard_by_target": raw_multiset_jaccard_by_target,
            "raw_alpha1_set_jaccard_by_target": raw_set_jaccard_by_target,
        },
        "utility": {
            "survives": all(utility_checks.values()),
            "checks": utility_checks,
            "quotient_vs_raw_alpha1_post20_mean_db": float(np.mean(q20)),
            "quotient_vs_raw_alpha1_post100_mean_db": float(np.mean(q100)),
            "quotient_vs_raw_alpha1_post20_target_wins": int(sum(v > 0.0 for v in q20)),
            "quotient_vs_raw_alpha1_post20_by_target": q20_by_target,
            "quotient_vs_raw_alpha1_post100_by_target": q100_by_target,
            "quotient_vs_support_post20_mean_db": float(np.mean(support20)),
            "quotient_vs_support_post100_mean_db": float(np.mean(support100)),
            "total100_overhead_fraction_vs_support": overhead,
        },
    }


def _write_summary(outdir: Path, aggregate: dict[str, Any]) -> None:
    arms = aggregate["arms"]
    commutation = aggregate["commutation"]
    utility = aggregate["utility"]
    lines = [
        "# FIT-019 opacity-split gauge-equivalence audit",
        "",
        f"Commutation confirmed: **{commutation['confirmed']}**.  ",
        f"Recovery-utility guard survives: **{utility['survives']}**.",
        "",
        "| Arm | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Post-100 AUC | Unique groups | Total-100 s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = arms[arm]
        lines.append(
            f"| `{arm}` | {row['immediate_psnr']:.4f} | {row['post20_psnr']:.4f} | "
            f"{row['post100_psnr']:.4f} | {row['post100_auc']:.4f} | "
            f"{row['unique_action_groups']:.2f} | {row['total100_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Raw alpha-1 action changed on {commutation['raw_alpha1_changed_target_count']}/8 "
            "target clusters.",
            "",
            "Grouped alpha-1 versus gauge-row alpha-1: "
            f"post-20 {utility['quotient_vs_raw_alpha1_post20_mean_db']:+.4f} dB, "
            f"post-100 {utility['quotient_vs_raw_alpha1_post100_mean_db']:+.4f} dB, "
            f"post-20 wins {utility['quotient_vs_raw_alpha1_post20_target_wins']}/8 targets.",
            "",
            "Grouped alpha-1 versus canonical support: "
            f"post-20 {utility['quotient_vs_support_post20_mean_db']:+.4f} dB, "
            f"post-100 {utility['quotient_vs_support_post100_mean_db']:+.4f} dB; "
            f"total-100 overhead {utility['total100_overhead_fraction_vs_support']:+.1%}.",
            "",
            "This procedural benchmark is a mechanism guard. A commutation-only pass is a diagnostic,",
            "not authorization for production lineage metadata, natural-image claims, or defaults.",
            "",
        ]
    )
    outdir.joinpath("summary.md").write_text("\n".join(lines), encoding="utf-8")


def _reproduction_command(
    *,
    outdir: str,
    size: int,
    seeds: tuple[int, ...],
    start_count: int,
    add_count: int,
    pre_iters: int,
    device: str,
    render_chunk: int,
) -> str:
    command = [
        "python",
        "-m",
        "benchmarks.gauge_equivalence_audit",
        "--outdir",
        outdir,
        "--size",
        str(size),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--start-count",
        str(start_count),
        "--add-count",
        str(add_count),
        "--pre-iters",
        str(pre_iters),
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
    seeds: tuple[int, ...] = (0, 1),
    start_count: int = 32,
    add_count: int = 8,
    pre_iters: int = 40,
    device: str = "cpu",
    render_chunk: int = 512,
) -> list[dict[str, Any]]:
    if device != "cpu":
        raise ValueError("FIT-019's frozen guard requires device='cpu'")
    if tuple(seeds) != (0, 1):
        raise ValueError("FIT-019's frozen guard requires seeds=(0, 1)")
    if (size, start_count, add_count, pre_iters) != (48, 32, 8, 40):
        raise ValueError("FIT-019's frozen guard requires size/N/add/pre=(48,32,8,40)")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"FIT-019 requires a fresh output directory, found existing files in {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    targets = _target_suite(size)
    algebra = run_algebra_audit()
    output.joinpath("algebra.json").write_text(
        json.dumps(algebra, indent=2, allow_nan=False), encoding="utf-8"
    )
    config = run_config(
        {
            "protocol": "fit019_opacity_gauge_v2_audit_corrected",
            "v2_audit_correction": {
                "changed": [
                    "gate alpha-0.7 relative error and both-alpha quotient top-k equality",
                    "gate raw alpha-1 physical-group multisets on both seeds per target",
                    "gate Stage-A numerical invariants and recovered horizon counts",
                    "record projected top-k agreement and stronger source provenance",
                ],
                "unchanged": [
                    "targets",
                    "seeds",
                    "arms",
                    "thresholds",
                    "actions",
                    "recovery horizons",
                ],
            },
            "targets": list(targets),
            "target_sha256": {name: _tensor_sha256(value) for name, value in targets.items()},
            "size": size,
            "seeds": list(seeds),
            "start_count": start_count,
            "add_count": add_count,
            "final_count": start_count + add_count,
            "pre_iters": pre_iters,
            "recovery_horizons": [20, 100],
            "arms": list(ARMS),
            "gauge_rule": "split even canonical group IDs into exact 0.5/0.5 opacity copies",
            "selected_candidate": "quotient_alpha1",
            "renderer": "normalized",
            "strategy": "quadtree_wse",
            "refine_primitive": "sequential_moment_preserving_on_canonical_field",
            "recovery_optimizer_state": "fresh Adam restart per action and horizon",
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
                "pre_fit": asdict(
                    FitConfig(
                        iters=pre_iters,
                        renderer="normalized",
                        render_chunk=render_chunk,
                        log_every=pre_iters,
                    )
                ),
                "action": asdict(
                    FitConfig(
                        renderer="normalized",
                        render_chunk=render_chunk,
                        refine_primitive="moment_preserving",
                        split_scale=0.7,
                    )
                ),
                "recovery20": asdict(
                    FitConfig(
                        iters=20,
                        renderer="normalized",
                        render_chunk=render_chunk,
                        log_every=1,
                    )
                ),
                "recovery100": asdict(
                    FitConfig(
                        iters=100,
                        renderer="normalized",
                        render_chunk=render_chunk,
                        log_every=1,
                    )
                ),
            },
            "render_chunk": render_chunk,
            "torch_num_threads": torch.get_num_threads(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "claim_scope": "procedural mechanism guard; no production/default/natural-image claim",
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
    config["source"] = _source_provenance(root)
    config["original_invocation"] = _reproduction_command(
        outdir=outdir,
        size=size,
        seeds=seeds,
        start_count=start_count,
        add_count=add_count,
        pre_iters=pre_iters,
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
        device=device,
        render_chunk=render_chunk,
    )
    _write_source_snapshot(root, output, config["source"])
    write_config(str(output), config)

    rows: list[dict[str, Any]] = []
    scfg = StructureTensorConfig()
    for target_name, target_np in targets.items():
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
            pre_cfg = FitConfig(
                iters=pre_iters,
                renderer="normalized",
                render_chunk=render_chunk,
                log_every=pre_iters,
            )
            pre = fit(initial, target, pre_cfg, verbose=False)
            base = pre["field"].detached()
            base_render = _render(base, pre_cfg, H, W)
            base_groups = torch.arange(base.n, device=base.means.device, dtype=torch.long)
            gauge, gauge_groups, _ = exact_opacity_refinement(
                base, split_groups={idx for idx in range(base.n) if idx % 2 == 0}
            )
            canonical_field_sha256 = _field_sha256(base)
            gauge_field_sha256 = _field_sha256(gauge)
            gauge_render = _render(gauge, pre_cfg, H, W)
            render_max_abs = float((base_render - gauge_render).abs().max())
            specs, diagnostics = _arm_specs(
                base,
                gauge,
                base_groups,
                gauge_groups,
                target,
                base_render,
                gauge_render,
                k=add_count,
                render_chunk=render_chunk,
            )
            action_cache: dict[tuple[int, ...], dict[str, Any]] = {}
            for spec in specs:
                action = [int(value) for value in spec["action"]]
                key = tuple(action)
                if key not in action_cache:
                    action_cache[key] = _evaluate_action(
                        base,
                        target,
                        base_render,
                        action,
                        seed=seed,
                        render_chunk=render_chunk,
                    )
                evaluation = action_cache[key]
                row = {
                    "target": target_name,
                    "target_sha256": config["resolved"]["target_sha256"][target_name],
                    "seed": int(seed),
                    "arm": spec["arm"],
                    "view": spec["view"],
                    "selector": spec["selector"],
                    "alpha": spec["alpha"],
                    "canonical_count": int(base.n),
                    "gauge_count": int(gauge.n),
                    "canonical_field_sha256": canonical_field_sha256,
                    "gauge_field_sha256": gauge_field_sha256,
                    "pre_psnr": round(float(pre["psnr"]), 6),
                    "render_max_abs": render_max_abs,
                    "action_groups": json.dumps(action, separators=(",", ":")),
                    "action_hash": _action_hash(action),
                    "unique_action_groups": len(set(action)),
                    "repeated_action_count": len(action) - len(set(action)),
                    "projected_topk_order_agreement": float(spec["projected_topk_order_agreement"]),
                    "projected_topk_multiset_jaccard": float(
                        spec["projected_topk_multiset_jaccard"]
                    ),
                    "projected_topk_unique_jaccard": float(spec["projected_topk_unique_jaccard"]),
                    "projected_topk_exact": bool(spec["projected_topk_exact"]),
                    "score_seconds": float(spec["score_seconds"]),
                    "scores_finite": bool(spec["scores_finite"]),
                    **diagnostics,
                    **evaluation,
                }
                row["intervention_seconds"] = float(spec["score_seconds"]) + float(
                    evaluation["action_seconds"]
                )
                row["total100_seconds"] = row["intervention_seconds"] + float(
                    evaluation["fit100_seconds"]
                )
                rows.append(row)

    aggregate = aggregate_and_decide(rows, algebra)
    write_csv(output / "rows.csv", rows)
    output.joinpath("rows.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8"
    )
    output.joinpath("aggregate.json").write_text(
        json.dumps(aggregate, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_summary(output, aggregate)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results/fit019_opacity_gauge_guard")
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--start-count", type=int, default=32)
    parser.add_argument("--add-count", type=int, default=8)
    parser.add_argument("--pre-iters", type=int, default=40)
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
        device=args.device,
        render_chunk=args.render_chunk,
    )


if __name__ == "__main__":
    main()
