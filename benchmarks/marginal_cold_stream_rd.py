"""COMP-006 matched operational marginal rate--distortion audit.

This benchmark is deliberately narrower than a new codec or allocator.  It starts every action
from one persisted, cold-decoded SSPL1 parent, compares sixteen matched residual births and
birth-for-death replacements, and constructs an exhaustive global-precision envelope under exact
integer complete-stream byte caps.  The primary development cap is parent + 16 bytes.

The measured byte difference is a finite counterfactual between independently encoded complete
containers.  It is not an incremental patch cost, a per-row entropy price, or an additive budget.
"""
from __future__ import annotations

import argparse
import fcntl
from functools import lru_cache
import hashlib
import html
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from benchmarks._util import package_versions
from structsplat import codec
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import _add_from_residual, _render, fit
from structsplat.gaussians import GaussianField
from structsplat.init import build_field
from structsplat.render import gaussian_activity


SIZE = 64
START_COUNT = 64
PARENT_FIT_ITERS = 120
PARENT_QAT_ITERS = 40
RECOVERY_STEPS = (0, 20)
PRECISION_CHECKPOINTS = (20,)
ACTION_COUNT = 16
SEEDS = (0, 1)
DEVELOPMENT_VARIANTS = (0, 2, 4)
CONFIRMATION_VARIANTS = (1, 3, 5)
TARGET_FAMILIES = (
    "curved_step",
    "three_region_junction",
    "bezier_strokes",
    "hard_disks",
    "annular_sectors",
    "quadratic_patch",
)
BASE_BITS = (12, 6, 6, 6)
PRIMARY_CAP_INCREMENT = 16
SENSITIVITY_CAP_INCREMENTS = (0, 8, 16, 32)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260715
TARGET_MANIFEST_SHA256 = "a1ea1ea5be41e36c3e4a8557d01ce721167545d680a6945522c935b832f60f0e"


TARGET_PARAMETER_TABLE: dict[str, tuple[tuple[float, ...], ...]] = {
    "curved_step": (
        (-0.14, 0.35, 0.08, 0.10),
        (-0.07, -0.45, 0.18, 0.62),
        (0.02, 0.55, -0.22, 1.14),
        (0.10, -0.30, 0.28, 1.66),
        (-0.03, 0.72, 0.33, 2.18),
        (0.15, -0.62, -0.31, 2.70),
    ),
    "three_region_junction": (
        (0.31, 0.41, 7.0, 0.20),
        (0.57, 0.62, 38.0, 0.72),
        (0.43, 0.55, 69.0, 1.24),
        (0.66, 0.37, 101.0, 1.76),
        (0.35, 0.68, 133.0, 2.28),
        (0.59, 0.46, 164.0, 2.80),
    ),
    "bezier_strokes": (
        (0.08, 0.16, 0.50, 0.83, 0.91, 0.23, 0.020, 0.15),
        (0.11, 0.79, 0.45, 0.12, 0.88, 0.71, 0.027, 0.67),
        (0.06, 0.33, 0.61, 0.92, 0.94, 0.38, 0.034, 1.19),
        (0.14, 0.91, 0.48, 0.26, 0.85, 0.08, 0.023, 1.71),
        (0.05, 0.68, 0.57, 0.06, 0.93, 0.84, 0.030, 2.23),
        (0.09, 0.09, 0.40, 0.73, 0.89, 0.49, 0.037, 2.75),
    ),
    "hard_disks": (
        (0.27, 0.31, 0.105, 0.69, 0.66, 0.165, 0.05),
        (0.35, 0.72, 0.145, 0.73, 0.29, 0.085, 0.57),
        (0.22, 0.58, 0.075, 0.64, 0.43, 0.185, 1.09),
        (0.42, 0.24, 0.125, 0.78, 0.73, 0.115, 1.61),
        (0.30, 0.79, 0.175, 0.70, 0.48, 0.095, 2.13),
        (0.18, 0.39, 0.090, 0.61, 0.76, 0.155, 2.65),
    ),
    "annular_sectors": (
        (0.49, 0.48, 0.29, 0.055, -105.0, 65.0, 0.12),
        (0.54, 0.44, 0.35, 0.040, -40.0, 125.0, 0.64),
        (0.45, 0.56, 0.24, 0.070, 20.0, 178.0, 1.16),
        (0.58, 0.52, 0.31, 0.050, 82.0, 250.0, 1.68),
        (0.40, 0.46, 0.38, 0.045, 145.0, 315.0, 2.20),
        (0.52, 0.60, 0.27, 0.065, 205.0, 380.0, 2.72),
    ),
    "quadratic_patch": (
        (0.45, -0.30, 0.38, 0.22, -0.16, 0.08),
        (-0.35, 0.42, -0.28, 0.31, 0.12, 0.60),
        (0.28, 0.36, 0.44, -0.25, 0.18, 1.12),
        (-0.42, -0.24, 0.34, 0.29, -0.20, 1.64),
        (0.39, -0.41, -0.32, 0.27, 0.14, 2.16),
        (-0.31, 0.33, 0.41, -0.30, -0.11, 2.68),
    ),
}


SOURCE_PATHS = (
    "benchmarks/_util.py",
    "benchmarks/marginal_cold_stream_rd.py",
    "benchmarks/perturb_recover_spectroscopy.py",
    "tasks/COMP-006-marginal-cold-stream-rd.md",
    "tests/test_marginal_cold_stream_rd.py",
    "src/structsplat/codec.py",
    "src/structsplat/config.py",
    "src/structsplat/density.py",
    "src/structsplat/fit.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/init.py",
    "src/structsplat/metrics.py",
    "src/structsplat/render.py",
    "src/structsplat/sampling.py",
    "src/structsplat/structure_tensor.py",
    "pyproject.toml",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    array = np.ascontiguousarray(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    )
    prefix = _canonical_json({"dtype": str(array.dtype), "shape": list(array.shape)})
    return _sha256_bytes(prefix + b"\0" + array.tobytes(order="C"))


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
            digest.update(b"none\0")
        else:
            digest.update(_tensor_sha256(value).encode("ascii") + b"\0")
    return digest.hexdigest()


def _palette(hue: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = float(hue) + np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    a = 0.50 + 0.40 * np.cos(phases)
    b = 0.50 + 0.40 * np.cos(phases + 2.15)
    c = 0.50 + 0.40 * np.cos(phases - 2.15)
    return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)


def _aa_inside(signed_inside: np.ndarray, size: int, width_pixels: float = 1.0) -> np.ndarray:
    half_width = max(float(width_pixels) / float(size), 1e-6)
    t = np.clip(0.5 + 0.5 * signed_inside / half_width, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _angle_interval_mask(
    angle: np.ndarray, start_degrees: float, end_degrees: float, size: int
) -> np.ndarray:
    start = math.radians(float(start_degrees))
    span = math.radians(float(end_degrees) - float(start_degrees))
    rel = np.mod(angle - start, 2.0 * np.pi)
    span = float(np.mod(span, 2.0 * np.pi))
    angular_margin = np.minimum(rel, span - rel)
    radius = np.maximum(np.abs(np.cos(angle)), np.abs(np.sin(angle)))
    signed = angular_margin * np.maximum(radius, 0.25)
    return _aa_inside(signed, size, width_pixels=1.25)


def _bezier_distance(
    u: np.ndarray,
    v: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 192, dtype=np.float32)
    omt = 1.0 - t
    x = omt * omt * p0[0] + 2.0 * omt * t * p1[0] + t * t * p2[0]
    y = omt * omt * p0[1] + 2.0 * omt * t * p1[1] + t * t * p2[1]
    distance = np.full(u.shape, np.inf, dtype=np.float32)
    for start in range(0, t.size, 32):
        dx = u[..., None] - x[start : start + 32]
        dy = v[..., None] - y[start : start + 32]
        distance = np.minimum(distance, np.sqrt(np.min(dx * dx + dy * dy, axis=2)))
    return distance


def _make_target(
    family: str, params: tuple[float, ...], u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    size = int(u.shape[0])
    ca, cb, cc = _palette(params[-1])
    if family == "curved_step":
        offset, slope, curvature, _hue = params
        boundary = 0.5 + offset + slope * (u - 0.5) + curvature * (u - 0.5) ** 2
        mix = _aa_inside(v - boundary, size, width_pixels=1.25)
        image = (1.0 - mix[..., None]) * ca + mix[..., None] * cb
    elif family == "three_region_junction":
        cx, cy, angle_degrees, _hue = params
        angle = math.radians(angle_degrees)
        x = u - cx
        y = v - cy
        directions = angle + np.arange(3, dtype=np.float32) * (2.0 * np.pi / 3.0)
        logits = np.stack(
            [np.cos(theta) * x + np.sin(theta) * y for theta in directions], axis=2
        )
        temperature = 0.55 / float(size)
        logits = logits / temperature
        logits -= logits.max(axis=2, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=2, keepdims=True)
        image = weights[..., :1] * ca + weights[..., 1:2] * cb + weights[..., 2:] * cc
    elif family == "bezier_strokes":
        x0, y0, x1, y1, x2, y2, width, _hue = params
        distance = _bezier_distance(u, v, (x0, y0), (x1, y1), (x2, y2))
        mirror = _bezier_distance(
            u,
            v,
            (1.0 - x0, 1.0 - y0),
            (1.0 - x1, 1.0 - y1),
            (1.0 - x2, 1.0 - y2),
        )
        first = _aa_inside(width - distance, size, width_pixels=1.0)
        second = _aa_inside(0.72 * width - mirror, size, width_pixels=1.0)
        image = ca + first[..., None] * (cb - ca) + second[..., None] * (cc - ca)
    elif family == "hard_disks":
        x0, y0, r0, x1, y1, r1, _hue = params
        d0 = np.sqrt((u - x0) ** 2 + (v - y0) ** 2)
        d1 = np.sqrt((u - x1) ** 2 + (v - y1) ** 2)
        m0 = _aa_inside(r0 - d0, size, width_pixels=1.15)
        m1 = _aa_inside(r1 - d1, size, width_pixels=1.15)
        image = ca + m0[..., None] * (cb - ca)
        image = image * (1.0 - m1[..., None]) + cc * m1[..., None]
    elif family == "annular_sectors":
        cx, cy, radius, width, start, end, _hue = params
        x = u - cx
        y = v - cy
        radial = np.sqrt(x * x + y * y)
        ring = _aa_inside(width - np.abs(radial - radius), size, width_pixels=1.15)
        sector = _angle_interval_mask(np.arctan2(y, x), start, end, size)
        mask = ring * sector
        inner = _aa_inside(0.55 * radius - radial, size, width_pixels=1.2)
        image = ca + mask[..., None] * (cb - ca) + 0.45 * inner[..., None] * (cc - ca)
    elif family == "quadratic_patch":
        ax, ay, axy, radial, linear, _hue = params
        x = u - 0.5
        y = v - 0.5
        q = ax * x * x + ay * y * y + axy * x * y + radial * (x * x + y * y)
        linear_term = linear * (0.7 * x - 0.4 * y)
        t0 = np.clip(0.5 + 1.4 * q + linear_term, 0.0, 1.0)
        t1 = np.clip(0.5 - 1.1 * q + 0.6 * linear_term, 0.0, 1.0)
        image = (
            ca * (1.0 - t0[..., None])
            + cb * (t0 * (1.0 - t1))[..., None]
            + cc * (t0 * t1)[..., None]
        )
    else:  # pragma: no cover - guarded by the frozen manifest
        raise ValueError(f"unknown COMP-006 target family {family!r}")
    return np.clip(image, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=4)
def target_suite(size: int = SIZE) -> dict[str, dict[str, Any]]:
    if size < 16:
        raise ValueError("COMP-006 targets require size >= 16")
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    u = (x + 0.5) / float(size)
    v = (y + 0.5) / float(size)
    result: dict[str, dict[str, Any]] = {}
    for family in TARGET_FAMILIES:
        table = TARGET_PARAMETER_TABLE[family]
        if len(table) != 6:
            raise RuntimeError(f"frozen family {family} does not have six variants")
        for variant, params in enumerate(table):
            target = f"{family}_v{variant}"
            image = _make_target(family, tuple(map(float, params)), u, v)
            result[target] = {
                "family": family,
                "variant": variant,
                "split": (
                    "development" if variant in DEVELOPMENT_VARIANTS else "confirmation"
                ),
                "params": tuple(float(value) for value in params),
                "image": image,
                "pixel_sha256": _tensor_sha256(image),
            }
    return result


@lru_cache(maxsize=4)
def target_manifest(size: int = SIZE) -> dict[str, Any]:
    entries = []
    for target, item in sorted(target_suite(size).items()):
        entries.append(
            {
                "target": target,
                "family": item["family"],
                "variant": int(item["variant"]),
                "split": item["split"],
                "params": list(item["params"]),
                "pixel_sha256": item["pixel_sha256"],
            }
        )
    payload = {"size": int(size), "dtype": "float32", "targets": entries}
    payload["manifest_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


@lru_cache(maxsize=1)
def precision_bit_grid() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(itertools.product(range(10, 17), range(4, 9), range(4, 9), range(4, 9)))


def precision_grid_sha256() -> str:
    return _sha256_bytes(_canonical_json(precision_bit_grid()))


@lru_cache(maxsize=1)
def protocol_manifest() -> dict[str, Any]:
    targets = target_manifest()
    if TARGET_MANIFEST_SHA256 != "TO_BE_FROZEN" and (
        targets["manifest_sha256"] != TARGET_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "COMP-006 target manifest drift: "
            f"{targets['manifest_sha256']} != {TARGET_MANIFEST_SHA256}"
        )
    payload = {
        "task": "COMP-006",
        "protocol_version": 1,
        "size": SIZE,
        "start_count": START_COUNT,
        "parent_fit_iters": PARENT_FIT_ITERS,
        "parent_qat_iters": PARENT_QAT_ITERS,
        "recovery_steps": list(RECOVERY_STEPS),
        "precision_checkpoints": list(PRECISION_CHECKPOINTS),
        "action_count": ACTION_COUNT,
        "seeds": list(SEEDS),
        "development_variants": list(DEVELOPMENT_VARIANTS),
        "confirmation_variants": list(CONFIRMATION_VARIANTS),
        "base_bits": list(BASE_BITS),
        "precision_grid_sha256": precision_grid_sha256(),
        "precision_grid_count": len(precision_bit_grid()),
        "primary_cap_increment_bytes": PRIMARY_CAP_INCREMENT,
        "sensitivity_cap_increments_bytes": list(SENSITIVITY_CAP_INCREMENTS),
        "target_manifest": targets,
        "rate_definition": "counterfactual complete self-contained SSPL1 container bytes",
        "stream_codec": "zlib",
        "zlib_level": 9,
        "renderer": "normalized",
        "selection_metric": "cold decoded display-clamped MSE",
        "secondary_metric_policy": (
            "SSIM/MS-SSIM for all structural rows and matched bases; selected precision streams "
            "are rescored during analysis"
        ),
        "init_config": {
            "strategy": "quadtree_wse",
            "num_gaussians": START_COUNT,
            "opacity_mode": "none",
            "seed": "cell_seed",
        },
        "fit_config": {
            "renderer": "normalized",
            "pixel_loss": "l1",
            "ssim_weight": 0.3,
            "checkpoint_policy": "terminal",
            "color_basis": "constant",
            "covariance_filter_mode": "none",
            "support_fade": False,
            "aa_dilation": 0.0,
            "render_chunk": 512,
            "other_values": "FitConfig defaults frozen by source snapshot",
        },
        "qat": {
            "parent_steps": PARENT_QAT_ITERS,
            "branch_steps": list(RECOVERY_STEPS),
            "optimizer": "fresh Adam via codec.qat_finetune",
            "range_policy": (
                "parent uses pre-QAT fitted ranges; each birth/replace freezes its action-start "
                "ranges; all precision mixes share the matched no-edit checkpoint ranges"
            ),
        },
        "action_bank": {
            "constructor": "fit._add_from_residual called once for a 16-row cohort",
            "score": "legacy_abs",
            "split_scale": 0.7,
            "split_oversample": 6.0,
            "split_min_spacing": 0.75,
            "split_color_init": "target",
            "donor": "minimum gaussian_activity; torch.argmin gives lowest-index tie",
            "replacement": "delete fixed donor then append exact matched birth row",
        },
        "selection": {
            "feasible": "total_bytes <= matched_base_bytes + integer_cap_increment",
            "stable_order": ["mse", "bytes", "branch", "action_index_or_bits", "row_key"],
            "no_ratio_or_interpolation": True,
            "raw_bits_definition": "attribute-only nominal bits; excludes all container overhead",
        },
        "inference": {
            "seed_policy": "mean over feasible paired seeds; every target requires >=1",
            "independent_unit": "target",
            "bootstrap": "resample three target means within each of six families, then mean 18",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "quantiles": [0.025, 0.975],
        },
        "claims_excluded": [
            "incremental patch cost",
            "additive local entropy price",
            "deployable or sequential selector",
            "richer-atom expressiveness",
            "natural-image or external-codec superiority",
            "performance speedup",
        ],
    }
    payload["protocol_sha256"] = _sha256_bytes(_canonical_json(payload))
    return payload


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if number == len(lines) and not line.endswith(("\n", "\r")):
                break
            raise ValueError(f"invalid JSONL at {path}:{number}") from exc
    return rows


def _git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _source_provenance(root: Path) -> dict[str, Any]:
    missing = [path for path in SOURCE_PATHS if not root.joinpath(path).is_file()]
    if missing:
        raise FileNotFoundError(f"COMP-006 source provenance is missing: {missing}")
    files = {path: _sha256_file(root / path) for path in SOURCE_PATHS}
    joined = "\n".join(f"{name}:{digest}" for name, digest in files.items()).encode("utf-8")
    diff = _git_output(root, "diff", "--binary", "HEAD") or ""
    status = _git_output(root, "status", "--porcelain", "--untracked-files=normal") or ""
    return {
        "files": files,
        "combined_sha256": _sha256_bytes(joined),
        "commit": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "branch", "--show-current"),
        "dirty": bool(status.strip()),
        "tracked_diff_sha256": _sha256_bytes(diff.encode("utf-8")),
        "status_sha256": _sha256_bytes(status.encode("utf-8")),
        "snapshot_policy": "benchmark, frozen task/test, direct production dependencies, pyproject",
    }


def _write_source_snapshot(root: Path, outdir: Path, source: dict[str, Any]) -> None:
    for relative, expected in source["files"].items():
        payload = root.joinpath(relative).read_bytes()
        if _sha256_bytes(payload) != expected:
            raise RuntimeError(f"source changed while snapshotting {relative}")
        destination = outdir / "source_snapshot" / relative
        _atomic_bytes(destination, payload)


def _environment_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "numpy_config": {
            "blas_opt_info": str(np.__config__.get_info("blas_opt_info"))
            if hasattr(np.__config__, "get_info")
            else None,
        },
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "zlib_runtime": getattr(__import__("zlib"), "ZLIB_RUNTIME_VERSION", None),
        "zlib_compile": getattr(__import__("zlib"), "ZLIB_VERSION", None),
    }


def _environment_sha256(environment: dict[str, Any]) -> str:
    """Hash canonical environment provenance for resume/replay binding."""
    return _sha256_bytes(_canonical_json(environment))


def _base_codec_config(**updates: Any) -> codec.CodecConfig:
    values = {
        "bits_means": BASE_BITS[0],
        "bits_scales": BASE_BITS[1],
        "bits_rot": BASE_BITS[2],
        "bits_colors": BASE_BITS[3],
        "morton_reorder": True,
        "color_delta": True,
        "means_byte_planar": True,
        "rot_circular": True,
        "stream_codec": "zlib",
        "zlib_level": 9,
    }
    values.update(updates)
    return codec.CodecConfig(**values)


def _freeze_ranges(field: GaussianField, ccfg: codec.CodecConfig) -> codec.CodecConfig:
    color_lo, color_hi = codec.color_ranges(field)
    scale_lo, scale_hi = codec.scale_ranges(field)
    return replace(
        ccfg,
        color_lo=color_lo,
        color_hi=color_hi,
        scale_lo=scale_lo,
        scale_hi=scale_hi,
    )


def _fit_config(iters: int) -> FitConfig:
    return FitConfig(
        iters=int(iters),
        renderer="normalized",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=max(1, int(iters)),
        checkpoint_policy="terminal",
        color_basis="constant",
        covariance_filter_mode="none",
        support_fade=False,
        aa_dilation=0.0,
        render_chunk=512,
        qat_mode="off",
    )


def _raw_bits(n: int, bits: Sequence[int]) -> int:
    means, scales, rotation, colors = map(int, bits)
    return int(n) * (2 * means + 2 * scales + rotation + 3 * colors)


def _codec_config_payload(ccfg: codec.CodecConfig) -> dict[str, Any]:
    payload = asdict(ccfg)
    return {key: payload[key] for key in sorted(payload)}


def _header_semantics(header: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": int(header["n"]),
        "H": int(header["H"]),
        "W": int(header["W"]),
        "bits": [int(value) for value in header["bits"]],
        "has_opacity": bool(header.get("has_opacity", False)),
        "bits_opacity": int(header.get("bits_opacity", 0)),
        "morton": bool(header["morton"]),
        "color_delta": bool(header.get("color_delta", False)),
        "means_byte_planar": bool(header.get("means_byte_planar", False)),
        "rot_circular": bool(header.get("rot_circular", False)),
        "stream_codec": header.get("stream_codec", "zlib"),
        "renderer": header.get("renderer", "normalized"),
        "aa_dilation": float(header.get("aa_dilation", 0.0)),
        "sigma_cutoff": float(header.get("sigma_cutoff", 3.0)),
        "support_fade": bool(header.get("support_fade", False)),
        "render_chunk": int(header.get("render_chunk", 512)),
        "color_lo": [float(value) for value in header["color_lo"]],
        "color_hi": [float(value) for value in header["color_hi"]],
        "scale_lo": [float(value) for value in header["scale_lo"]],
        "scale_hi": [float(value) for value in header["scale_hi"]],
        "means_lo": [float(value) for value in header["means_lo"]],
        "means_hi": [float(value) for value in header["means_hi"]],
    }


def _component_payload(components: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_bytes": int(components["total_bytes"]),
        "header_bytes": int(components["header_bytes"]),
        "header_json_bytes": int(components["header_json_bytes"]),
        "streams": {
            name: {
                "payload_bytes": int(item["payload_bytes"]),
                "framed_bytes": int(item["framed_bytes"]),
                "raw_bytes": None if item["raw_bytes"] is None else int(item["raw_bytes"]),
            }
            for name, item in sorted(components["streams"].items())
        },
    }


def _validate_blob(
    blob: bytes,
    *,
    expected_n: int,
    expected_bits: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], GaussianField]:
    components = codec.blob_components(blob)
    header = codec.blob_header(blob)
    if (int(header["H"]), int(header["W"])) != (SIZE, SIZE):
        raise RuntimeError(f"cold stream dimensions drifted: {header['H']}x{header['W']}")
    if int(header["n"]) != int(expected_n):
        raise RuntimeError(f"cold stream count drifted: {header['n']} != {expected_n}")
    if tuple(map(int, header["bits"])) != tuple(map(int, expected_bits)):
        raise RuntimeError(f"cold stream bits drifted: {header['bits']} != {list(expected_bits)}")
    if bool(header.get("has_opacity", False)):
        raise RuntimeError("COMP-006 no-opacity schema unexpectedly acquired an opacity stream")
    if header.get("renderer") != "normalized" or header.get("stream_codec") != "zlib":
        raise RuntimeError("COMP-006 stream semantics drifted")
    if not (
        bool(header.get("morton"))
        and bool(header.get("color_delta"))
        and bool(header.get("means_byte_planar"))
        and bool(header.get("rot_circular"))
    ):
        raise RuntimeError("COMP-006 transform flags drifted")
    if set(components["streams"]) != {"means", "scales", "rotation", "colors"}:
        raise RuntimeError("COMP-006 stream set drifted")
    raw_expected = {
        "means": 2 * expected_n * np.dtype(
            np.uint8 if expected_bits[0] <= 8 else np.uint16 if expected_bits[0] <= 16 else np.uint32
        ).itemsize,
        "scales": 2 * expected_n * np.dtype(
            np.uint8 if expected_bits[1] <= 8 else np.uint16 if expected_bits[1] <= 16 else np.uint32
        ).itemsize,
        "rotation": expected_n * np.dtype(
            np.uint8 if expected_bits[2] <= 8 else np.uint16 if expected_bits[2] <= 16 else np.uint32
        ).itemsize,
        "colors": 3 * expected_n * np.dtype(
            np.uint8 if expected_bits[3] <= 8 else np.uint16 if expected_bits[3] <= 16 else np.uint32
        ).itemsize,
    }
    observed_raw = {
        name: int(item["raw_bytes"]) for name, item in components["streams"].items()
    }
    if observed_raw != raw_expected:
        raise RuntimeError(f"COMP-006 raw stream lengths drifted: {observed_raw} != {raw_expected}")
    partition = int(components["header_bytes"]) + sum(
        int(item["framed_bytes"]) for item in components["streams"].values()
    )
    if partition != len(blob) or int(components["total_bytes"]) != len(blob):
        raise RuntimeError("blob_components does not exactly partition the SSPL1 container")
    decoded = codec.decode(blob, device="cpu")
    for name in ("means", "log_scales", "rotations", "colors"):
        value = getattr(decoded, name)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"cold decoded field contains nonfinite {name}")
    return header, components, decoded


def _persist_and_score(
    field: GaussianField,
    target: torch.Tensor,
    ccfg: codec.CodecConfig,
    stream_path: Path,
    *,
    expected_n: int,
    secondary_metrics: bool = True,
) -> dict[str, Any]:
    fcfg = _fit_config(0)
    encode_start = time.perf_counter()
    blob = codec.encode(field, SIZE, SIZE, ccfg, fcfg)
    encode_seconds = time.perf_counter() - encode_start
    memory_components = codec.blob_components(blob)
    memory_decoded = codec.decode(blob, device="cpu")
    _atomic_bytes(stream_path, blob)
    persisted = stream_path.read_bytes()
    if persisted != blob:
        raise RuntimeError(f"persisted stream differs from encoded bytes: {stream_path}")
    header, components, decoded = _validate_blob(
        persisted,
        expected_n=expected_n,
        expected_bits=(ccfg.bits_means, ccfg.bits_scales, ccfg.bits_rot, ccfg.bits_colors),
    )
    if _component_payload(memory_components) != _component_payload(components):
        raise RuntimeError("in-memory and persisted component accounting differ")
    if _field_sha256(memory_decoded) != _field_sha256(decoded):
        raise RuntimeError("in-memory and persisted decoded fields differ")
    decode_start = time.perf_counter()
    prediction = codec.decode_and_render(persisted, device="cpu").clamp(0.0, 1.0)
    decode_render_seconds = time.perf_counter() - decode_start
    if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
        raise RuntimeError(f"invalid cold render for {stream_path}")
    mse = float(torch.mean((prediction - target) ** 2).detach().cpu())
    if not math.isfinite(mse):
        raise RuntimeError(f"nonfinite cold MSE for {stream_path}")
    return {
        "stream_path": stream_path.as_posix(),
        "stream_sha256": _sha256_bytes(persisted),
        "decoded_field_sha256": _field_sha256(decoded),
        "bytes": len(persisted),
        "bpp": 8.0 * len(persisted) / float(SIZE * SIZE),
        "raw_bits": _raw_bits(expected_n, header["bits"]),
        "mse": mse,
        "psnr": float(M.psnr(prediction, target)),
        "ssim": float(M.ssim(prediction, target)) if secondary_metrics else None,
        "ms_ssim": float(M.ms_ssim(prediction, target)) if secondary_metrics else None,
        "encode_seconds": float(encode_seconds),
        "decode_render_seconds": float(decode_render_seconds),
        "header": _header_semantics(header),
        "components": _component_payload(components),
        "codec_config": _codec_config_payload(ccfg),
    }


def _action_bank(
    parent: GaussianField, target: torch.Tensor
) -> tuple[list[GaussianField], int, dict[str, Any]]:
    if parent.n != START_COUNT or parent.opacities is not None:
        raise ValueError("COMP-006 action bank requires a no-opacity N=64 cold parent")
    fcfg = _fit_config(0)
    render = _render(parent, fcfg, SIZE, SIZE)
    grow_cfg = replace(
        fcfg,
        split_count=ACTION_COUNT,
        split_mode="residual_add_nms",
        refine_site=None,
        refine_primitive=None,
        refine_nms=None,
        split_scale=0.7,
        split_oversample=6.0,
        split_min_spacing=0.75,
        split_color_init="target",
        sampled_add_score="legacy_abs",
        max_gaussians=START_COUNT + ACTION_COUNT,
    )
    grown, added = _add_from_residual(parent.detached(), target, render, grow_cfg)
    if added != ACTION_COUNT or grown.n != START_COUNT + ACTION_COUNT:
        raise RuntimeError(f"action bank grew {added} rows, expected {ACTION_COUNT}")
    rows = [grown.subset(torch.tensor([START_COUNT + idx])) for idx in range(ACTION_COUNT)]
    row_hashes = [_field_sha256(row) for row in rows]
    if len(set(row_hashes)) != ACTION_COUNT:
        raise RuntimeError("COMP-006 action bank contains duplicate birth rows")

    activity = gaussian_activity(
        parent.means,
        parent.conics(fcfg.aa_dilation),
        parent.radii(fcfg.sigma_cutoff, fcfg.aa_dilation),
        SIZE,
        SIZE,
        fcfg.render_chunk,
        fcfg.support_fade,
        fcfg.sigma_cutoff,
    )
    donor_index = int(torch.argmin(activity).detach().cpu())
    metadata = {
        "candidate_row_hashes": row_hashes,
        "candidate_means": [
            [float(value) for value in row.means[0].detach().cpu().tolist()] for row in rows
        ],
        "candidate_action_sha256": _sha256_bytes(
            _canonical_json({"rows": row_hashes, "donor_index": donor_index})
        ),
        "donor_index": donor_index,
        "donor_activity": float(activity[donor_index].detach().cpu()),
        "parent_field_sha256": _field_sha256(parent),
    }
    return rows, donor_index, metadata


def _birth_field(parent: GaussianField, row: GaussianField) -> GaussianField:
    before = _field_sha256(parent)
    child = parent.detached().append(row.detached())
    if child.n != START_COUNT + 1 or _field_sha256(parent) != before:
        raise RuntimeError("birth action mutated its parent or produced the wrong count")
    return child


def _replace_field(parent: GaussianField, row: GaussianField, donor_index: int) -> GaussianField:
    before = _field_sha256(parent)
    keep = torch.ones(parent.n, dtype=torch.bool, device=parent.means.device)
    keep[int(donor_index)] = False
    child = parent.subset(keep).append(row.detached())
    if child.n != START_COUNT or _field_sha256(parent) != before:
        raise RuntimeError("replacement action mutated its parent or produced the wrong count")
    return child


def _qat_recover(
    start: GaussianField, target: torch.Tensor, steps: int
) -> tuple[GaussianField, codec.CodecConfig, float]:
    field = start.detached()
    ccfg = _freeze_ranges(field, _base_codec_config())
    if steps <= 0:
        return field, ccfg, 0.0
    started = time.perf_counter()
    ccfg = codec.qat_finetune(
        field,
        target,
        _fit_config(steps),
        ccfg,
        iters=int(steps),
        verbose=False,
    )
    seconds = time.perf_counter() - started
    if not all(
        bool(torch.isfinite(value).all())
        for value in (field.means, field.log_scales, field.rotations, field.colors)
    ):
        raise RuntimeError("QAT recovery produced a nonfinite field")
    return field, ccfg, float(seconds)


def _cell_key(split: str, target: str, seed: int) -> str:
    return _sha256_bytes(
        _canonical_json(
            [protocol_manifest()["protocol_sha256"], split, target, int(seed)]
        )
    )[:24]


def _row_key(
    cell_key: str,
    recovery_steps: int,
    branch: str,
    action_index: int | None,
    bits: Sequence[int],
) -> str:
    return _sha256_bytes(
        _canonical_json(
            [cell_key, int(recovery_steps), branch, action_index, list(map(int, bits))]
        )
    )[:24]


def _ccfg_from_header(header: dict[str, Any]) -> codec.CodecConfig:
    bits = tuple(map(int, header["bits"]))
    return _base_codec_config(
        bits_means=bits[0],
        bits_scales=bits[1],
        bits_rot=bits[2],
        bits_colors=bits[3],
        color_lo=[float(value) for value in header["color_lo"]],
        color_hi=[float(value) for value in header["color_hi"]],
        scale_lo=[float(value) for value in header["scale_lo"]],
        scale_hi=[float(value) for value in header["scale_hi"]],
    )


def _prepare_run_snapshot(outdir: Path, split: str, root: Path) -> dict[str, Any]:
    protocol = protocol_manifest()
    source = _source_provenance(root)
    environment = _environment_provenance()
    config_path = outdir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("protocol", {}).get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError(f"output directory has a different COMP-006 protocol: {outdir}")
        if config.get("split") != split:
            raise RuntimeError(f"output directory belongs to split {config.get('split')!r}")
        if config.get("source", {}).get("combined_sha256") != source["combined_sha256"]:
            raise RuntimeError(
                "source changed since this COMP-006 output directory was created; use a new "
                "outdir or the captured source snapshot"
            )
        if _environment_sha256(config.get("environment", {})) != _environment_sha256(
            environment
        ):
            raise RuntimeError(
                "environment changed since this COMP-006 output directory was created; use a "
                "new outdir"
            )
        return config
    outdir.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "COMP-006",
        "split": split,
        "protocol": protocol,
        "source": source,
        "environment": environment,
        "command": sys.argv,
    }
    _atomic_json(config_path, config)
    _atomic_json(outdir / "target_manifest.json", protocol["target_manifest"])
    _write_source_snapshot(root, outdir, source)
    return config


def _confirmation_authorized(development_summary: Path | None) -> bool:
    if development_summary is None or not development_summary.is_file():
        return False
    payload = json.loads(development_summary.read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    root = Path(__file__).resolve().parents[1]
    current_source = _source_provenance(root)["combined_sha256"]
    return bool(
        payload.get("schema") == "comp006-final-v1"
        and payload.get("split") == "development"
        and payload.get("decision") == "authorize_confirmation"
        and payload.get("protocol_sha256") == protocol_manifest()["protocol_sha256"]
        and payload.get("source_combined_sha256") == current_source
        and set(gates) == FINAL_GATE_KEYS
        and all(gates.values())
        and payload.get("checks", {}).get("streams") is True
        and payload.get("checks", {}).get("analysis") is True
    )


def _parent_for_cell(
    outdir: Path,
    split: str,
    target_name: str,
    target_item: dict[str, Any],
    seed: int,
    existing_cells: dict[str, dict[str, Any]],
    bindings: dict[str, str],
) -> tuple[GaussianField, codec.CodecConfig, dict[str, Any]]:
    key = _cell_key(split, target_name, seed)
    existing = existing_cells.get(key)
    target = torch.as_tensor(target_item["image"], dtype=torch.float32, device="cpu")
    if existing is not None:
        if existing.get("schema") != "comp006-cell-v1" or any(
            existing.get(name) != value for name, value in bindings.items()
        ):
            raise RuntimeError(f"cell binding drift for {key}")
        stream_path = outdir / existing["parent"]["stream_path"]
        blob = stream_path.read_bytes()
        if _sha256_bytes(blob) != existing["parent"]["stream_sha256"]:
            raise RuntimeError(f"parent stream hash mismatch for {key}")
        header, _components, parent = _validate_blob(
            blob, expected_n=START_COUNT, expected_bits=BASE_BITS
        )
        rows, donor, action = _action_bank(parent, target)
        del rows, donor
        if action != existing["action_bank"]:
            raise RuntimeError(f"action bank drift on resume for {key}")
        return parent, _ccfg_from_header(header), existing

    np.random.seed(seed)
    torch.manual_seed(seed)
    init_cfg = InitConfig(
        strategy="quadtree_wse",
        num_gaussians=START_COUNT,
        opacity_mode="none",
        seed=seed,
    )
    initial = build_field(
        target_item["image"], init_cfg, StructureTensorConfig(), device="cpu"
    )
    if initial.n != START_COUNT or initial.opacities is not None:
        raise RuntimeError("COMP-006 parent initializer violated its count/schema contract")
    fit_result = fit(initial, target, _fit_config(PARENT_FIT_ITERS), verbose=False)
    fitted = fit_result["field"]
    parent_ccfg = _freeze_ranges(fitted, _base_codec_config())
    qat_started = time.perf_counter()
    parent_ccfg = codec.qat_finetune(
        fitted,
        target,
        _fit_config(PARENT_QAT_ITERS),
        parent_ccfg,
        iters=PARENT_QAT_ITERS,
        verbose=False,
    )
    qat_seconds = time.perf_counter() - qat_started
    parent_path = outdir / "parents" / f"{key}.sspl"
    parent_score = _persist_and_score(
        fitted,
        target,
        parent_ccfg,
        parent_path,
        expected_n=START_COUNT,
    )
    parent_score["stream_path"] = parent_path.relative_to(outdir).as_posix()
    parent_blob = parent_path.read_bytes()
    header, _components, parent = _validate_blob(
        parent_blob, expected_n=START_COUNT, expected_bits=BASE_BITS
    )
    parent_cold_render = codec.decode_and_render(parent_blob, device="cpu").clamp(0.0, 1.0)
    direct_render = _render(parent, _fit_config(0), SIZE, SIZE).clamp(0.0, 1.0)
    cold_parity = float((parent_cold_render - direct_render).abs().max().detach().cpu())
    if cold_parity > 1e-6:
        raise RuntimeError(f"parent cold-render parity failed for {key}: {cold_parity}")
    _rows, _donor, action = _action_bank(parent, target)
    parent_score["timing"] = {
        "fit_seconds": float(fit_result["fit_seconds"]),
        "qat_seconds": float(qat_seconds),
        "encode_seconds": float(parent_score.pop("encode_seconds")),
        "decode_render_seconds": float(parent_score.pop("decode_render_seconds")),
    }
    parent_score["cold_render_max_abs"] = cold_parity
    record = {
        "schema": "comp006-cell-v1",
        **bindings,
        "cell_key": key,
        "split": split,
        "target": target_name,
        "family": target_item["family"],
        "variant": int(target_item["variant"]),
        "target_pixel_sha256": target_item["pixel_sha256"],
        "seed": int(seed),
        "parent_field_sha256": _field_sha256(parent),
        "parent": parent_score,
        "parent_codec_config": _codec_config_payload(_ccfg_from_header(header)),
        "action_bank": action,
    }
    _append_jsonl(outdir / "cells.jsonl", record)
    existing_cells[key] = record
    return parent, _ccfg_from_header(header), record


def _stream_row(
    *,
    outdir: Path,
    cell: dict[str, Any],
    target: torch.Tensor,
    recovered_field: GaussianField,
    ccfg: codec.CodecConfig,
    recovery_steps: int,
    recovery_seconds: float,
    action_seconds: float,
    branch: str,
    action_index: int | None,
    action_start_sha256: str,
    secondary_metrics: bool = True,
) -> dict[str, Any]:
    bits = (ccfg.bits_means, ccfg.bits_scales, ccfg.bits_rot, ccfg.bits_colors)
    key = _row_key(cell["cell_key"], recovery_steps, branch, action_index, bits)
    stream_path = (
        outdir
        / "streams"
        / cell["cell_key"]
        / f"step{int(recovery_steps)}"
        / f"{key}.sspl"
    )
    score = _persist_and_score(
        recovered_field,
        target,
        ccfg,
        stream_path,
        expected_n=int(recovered_field.n),
        secondary_metrics=secondary_metrics,
    )
    score["stream_path"] = stream_path.relative_to(outdir).as_posix()
    timing = {
        "action_seconds": float(action_seconds),
        "recovery_seconds": float(recovery_seconds),
        "encode_seconds": float(score.pop("encode_seconds")),
        "decode_render_seconds": float(score.pop("decode_render_seconds")),
    }
    return {
        "schema": "comp006-stream-v1",
        "protocol_sha256": cell["protocol_sha256"],
        "source_combined_sha256": cell["source_combined_sha256"],
        "row_key": key,
        "cell_key": cell["cell_key"],
        "split": cell["split"],
        "target": cell["target"],
        "family": cell["family"],
        "variant": int(cell["variant"]),
        "target_pixel_sha256": cell["target_pixel_sha256"],
        "seed": int(cell["seed"]),
        "parent_stream_sha256": cell["parent"]["stream_sha256"],
        "parent_field_sha256": cell["parent_field_sha256"],
        "action_bank_sha256": cell["action_bank"]["candidate_action_sha256"],
        "recovery_steps": int(recovery_steps),
        "branch": branch,
        "action_index": action_index,
        "action_start_field_sha256": action_start_sha256,
        "recovered_field_sha256": _field_sha256(recovered_field),
        "bits": list(map(int, bits)),
        "n_gaussians": int(recovered_field.n),
        "timing": timing,
        **score,
    }


def _validate_stream_cell_binding(row: dict[str, Any], cell: dict[str, Any]) -> None:
    expected = {
        "cell_key": cell["cell_key"],
        "split": cell["split"],
        "target": cell["target"],
        "family": cell["family"],
        "variant": int(cell["variant"]),
        "target_pixel_sha256": cell["target_pixel_sha256"],
        "seed": int(cell["seed"]),
        "parent_stream_sha256": cell["parent"]["stream_sha256"],
        "parent_field_sha256": cell["parent_field_sha256"],
        "action_bank_sha256": cell["action_bank"]["candidate_action_sha256"],
    }
    drift = {
        name: {"expected": value, "observed": row.get(name)}
        for name, value in expected.items()
        if row.get(name) != value
    }
    if drift:
        raise RuntimeError(f"stream row/cell binding drift for {row.get('row_key')}: {drift}")


def _validate_existing_stream_row(
    outdir: Path,
    row: dict[str, Any],
    bindings: dict[str, str],
    *,
    cell: dict[str, Any] | None = None,
) -> None:
    if row.get("schema") != "comp006-stream-v1" or any(
        row.get(name) != value for name, value in bindings.items()
    ):
        raise RuntimeError(f"stream row binding drift for {row.get('row_key')}")
    branch = row.get("branch")
    action_index = row.get("action_index")
    bits = tuple(map(int, row.get("bits", [])))
    n = int(row.get("n_gaussians", -1))
    if branch == "precision":
        valid = action_index is None and n == START_COUNT and bits in precision_bit_grid()
    elif branch == "birth":
        valid = action_index in range(ACTION_COUNT) and n == START_COUNT + 1 and bits == BASE_BITS
    elif branch == "replace":
        valid = action_index in range(ACTION_COUNT) and n == START_COUNT and bits == BASE_BITS
    else:
        valid = False
    recovery_steps = int(row.get("recovery_steps", -1))
    if not valid or recovery_steps not in RECOVERY_STEPS:
        raise RuntimeError(f"invalid COMP-006 stream row identity: {row.get('row_key')}")
    expected_row_key = _row_key(
        str(row.get("cell_key")), recovery_steps, str(branch), action_index, bits
    )
    if row.get("row_key") != expected_row_key:
        raise RuntimeError(
            f"stream row key drift: {row.get('row_key')} != {expected_row_key}"
        )
    if cell is not None:
        _validate_stream_cell_binding(row, cell)
    path = outdir / row["stream_path"]
    blob = path.read_bytes()
    if len(blob) != int(row["bytes"]) or _sha256_bytes(blob) != row["stream_sha256"]:
        raise RuntimeError(f"existing stream bytes/hash drift for {row['row_key']}")
    _header, components, decoded = _validate_blob(blob, expected_n=n, expected_bits=bits)
    if _component_payload(components) != row["components"]:
        raise RuntimeError(f"existing component accounting drift for {row['row_key']}")
    if _field_sha256(decoded) != row["decoded_field_sha256"]:
        raise RuntimeError(f"existing decoded field drift for {row['row_key']}")


def run_benchmark(
    outdir: str | Path,
    *,
    split: str = "development",
    development_summary: str | Path | None = None,
    max_cells: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    if split not in ("development", "confirmation"):
        raise ValueError("split must be development or confirmation")
    if split == "confirmation" and not _confirmation_authorized(
        None if development_summary is None else Path(development_summary)
    ):
        raise RuntimeError("confirmation is sealed until an unchanged development gate passes")
    if num_shards <= 0 or not (0 <= shard_index < num_shards):
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    outdir = Path(outdir)
    root = Path(__file__).resolve().parents[1]
    config = _prepare_run_snapshot(outdir, split, root)
    protocol_sha = config["protocol"]["protocol_sha256"]
    bindings = {
        "protocol_sha256": protocol_sha,
        "source_combined_sha256": config["source"]["combined_sha256"],
    }
    existing_cell_rows = read_jsonl(outdir / "cells.jsonl")
    existing_cells = {row["cell_key"]: row for row in existing_cell_rows}
    if len(existing_cells) != len(existing_cell_rows):
        raise RuntimeError("duplicate cell keys in COMP-006 journal")
    existing_stream_rows = read_jsonl(outdir / "streams.jsonl")
    existing_row_keys = {row["row_key"] for row in existing_stream_rows}
    if len(existing_row_keys) != len(existing_stream_rows):
        raise RuntimeError("duplicate stream row keys in COMP-006 journal")
    for row in existing_stream_rows:
        cell = existing_cells.get(row.get("cell_key"))
        if cell is None:
            raise RuntimeError(f"orphan stream row {row.get('row_key')}")
        _validate_existing_stream_row(outdir, row, bindings, cell=cell)

    suite = target_suite()
    targets = [name for name, item in sorted(suite.items()) if item["split"] == split]
    all_jobs = [(target_name, seed) for target_name in targets for seed in SEEDS]
    jobs = [job for index, job in enumerate(all_jobs) if index % num_shards == shard_index]
    completed_cells = 0
    for target_name, seed in jobs:
        item = suite[target_name]
        target = torch.as_tensor(item["image"], dtype=torch.float32, device="cpu")
        if max_cells is not None and completed_cells >= int(max_cells):
            print(f"COMP-006 stopped after max_cells={max_cells}; analysis will remain sealed")
            return
        cell_started = time.perf_counter()
        parent, parent_ccfg, cell = _parent_for_cell(
            outdir, split, target_name, item, seed, existing_cells, bindings
        )
        parent_hash = _field_sha256(parent)
        action_rows, donor_index, action = _action_bank(parent, target)
        if action != cell["action_bank"]:
            raise RuntimeError(f"action manifest drift for {cell['cell_key']}")

        for recovery_steps in RECOVERY_STEPS:
            no_edit, no_edit_ccfg, no_edit_seconds = _qat_recover(
                parent, target, recovery_steps
            )
            if recovery_steps == 0:
                no_edit_ccfg = parent_ccfg
            checkpoint_bits = (
                precision_bit_grid() if recovery_steps in PRECISION_CHECKPOINTS else (BASE_BITS,)
            )
            for bits in checkpoint_bits:
                ccfg = replace(
                    no_edit_ccfg,
                    bits_means=int(bits[0]),
                    bits_scales=int(bits[1]),
                    bits_rot=int(bits[2]),
                    bits_colors=int(bits[3]),
                )
                key = _row_key(
                    cell["cell_key"], recovery_steps, "precision", None, bits
                )
                if key in existing_row_keys:
                    continue
                row = _stream_row(
                    outdir=outdir,
                    cell=cell,
                    target=target,
                    recovered_field=no_edit,
                    ccfg=ccfg,
                    recovery_steps=recovery_steps,
                    recovery_seconds=no_edit_seconds,
                    action_seconds=0.0,
                    branch="precision",
                    action_index=None,
                    action_start_sha256=parent_hash,
                    secondary_metrics=tuple(bits) == BASE_BITS,
                )
                _append_jsonl(outdir / "streams.jsonl", row)
                existing_row_keys.add(row["row_key"])

            for action_index, candidate in enumerate(action_rows):
                for branch in ("birth", "replace"):
                    key = _row_key(
                        cell["cell_key"], recovery_steps, branch, action_index, BASE_BITS
                    )
                    if key in existing_row_keys:
                        continue
                    action_started = time.perf_counter()
                    start_field = (
                        _birth_field(parent, candidate)
                        if branch == "birth"
                        else _replace_field(parent, candidate, donor_index)
                    )
                    action_seconds = time.perf_counter() - action_started
                    start_hash = _field_sha256(start_field)
                    recovered, recovered_ccfg, recovery_seconds = _qat_recover(
                        start_field, target, recovery_steps
                    )
                    row = _stream_row(
                        outdir=outdir,
                        cell=cell,
                        target=target,
                        recovered_field=recovered,
                        ccfg=recovered_ccfg,
                        recovery_steps=recovery_steps,
                        recovery_seconds=recovery_seconds,
                        action_seconds=action_seconds,
                        branch=branch,
                        action_index=action_index,
                        action_start_sha256=start_hash,
                        secondary_metrics=True,
                    )
                    _append_jsonl(outdir / "streams.jsonl", row)
                    existing_row_keys.add(row["row_key"])

        if _field_sha256(parent) != parent_hash:
            raise RuntimeError(f"action branches mutated shared parent {cell['cell_key']}")
        completed_cells += 1
        elapsed = time.perf_counter() - cell_started
        print(
            f"COMP-006 {split} shard={shard_index}/{num_shards} "
            f"{completed_cells}/{len(jobs)} {target_name} seed={seed} {elapsed:.2f}s",
            flush=True,
        )

    marker = {
        "task": "COMP-006",
        "split": split,
        "protocol_sha256": protocol_sha,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "cells": len(jobs),
    }
    _atomic_json(outdir / "shards" / f"{shard_index:03d}-of-{num_shards:03d}.json", marker)
    if num_shards == 1:
        finalize_run(outdir, split=split, num_shards=1)


def finalize_run(outdir: str | Path, *, split: str, num_shards: int) -> dict[str, Any]:
    outdir = Path(outdir)
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    if config["split"] != split:
        raise RuntimeError("finalize split mismatch")
    markers = []
    for index in range(num_shards):
        path = outdir / "shards" / f"{index:03d}-of-{num_shards:03d}.json"
        marker = json.loads(path.read_text(encoding="utf-8"))
        if (
            marker["shard_index"] != index
            or marker["num_shards"] != num_shards
            or marker["split"] != split
            or marker["protocol_sha256"] != config["protocol"]["protocol_sha256"]
        ):
            raise RuntimeError(f"invalid shard marker {path}")
        markers.append(marker)
    expected_cells = len(_expected_cell_keys(split))
    expected_rows_per_cell = sum(
        (len(precision_bit_grid()) if step in PRECISION_CHECKPOINTS else 1)
        + 2 * ACTION_COUNT
        for step in RECOVERY_STEPS
    )
    cells = read_jsonl(outdir / "cells.jsonl")
    rows = read_jsonl(outdir / "streams.jsonl")
    if (
        sum(int(marker["cells"]) for marker in markers) != expected_cells
        or len(cells) != expected_cells
        or len({row["cell_key"] for row in cells}) != expected_cells
        or len(rows) != expected_cells * expected_rows_per_cell
        or len({row["row_key"] for row in rows}) != len(rows)
    ):
        raise RuntimeError("cannot finalize incomplete/duplicate COMP-006 shards")
    completion = {
        "task": "COMP-006",
        "split": split,
        "protocol_sha256": config["protocol"]["protocol_sha256"],
        "num_shards": num_shards,
        "cells": expected_cells,
        "rows": expected_cells * expected_rows_per_cell,
    }
    _atomic_json(outdir / "run_complete.json", completion)
    return completion


def _stable_candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    branch_rank = {"precision": 0, "replace": 1, "birth": 2}[row["branch"]]
    identity: Any = (
        tuple(map(int, row["bits"]))
        if row["branch"] == "precision"
        else int(row["action_index"])
    )
    return (
        float(row["mse"]),
        int(row["bytes"]),
        branch_rank,
        identity,
        row["row_key"],
    )


def _best(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = list(rows)
    return None if not candidates else min(candidates, key=_stable_candidate_key)


def _row_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_key": row["row_key"],
        "branch": row["branch"],
        "action_index": row["action_index"],
        "bits": row["bits"],
        "n_gaussians": row["n_gaussians"],
        "bytes": row["bytes"],
        "bpp": row["bpp"],
        "raw_bits": row["raw_bits"],
        "mse": row["mse"],
        "psnr": row["psnr"],
        "ssim": row["ssim"],
        "ms_ssim": row["ms_ssim"],
        "stream_sha256": row["stream_sha256"],
        "stream_path": row["stream_path"],
    }


def _component_delta(child: dict[str, Any], parent: dict[str, Any]) -> dict[str, int]:
    names = sorted(set(child["components"]["streams"]) | set(parent["components"]["streams"]))
    result = {
        "total_bytes": int(child["bytes"]) - int(parent["bytes"]),
        "header_bytes": int(child["components"]["header_bytes"])
        - int(parent["components"]["header_bytes"]),
    }
    for name in names:
        child_bytes = int(child["components"]["streams"].get(name, {}).get("framed_bytes", 0))
        parent_bytes = int(parent["components"]["streams"].get(name, {}).get("framed_bytes", 0))
        result[f"{name}_framed_bytes"] = child_bytes - parent_bytes
    return result


def _expected_cell_keys(split: str) -> set[str]:
    return {
        _cell_key(split, name, seed)
        for name, item in target_suite().items()
        if item["split"] == split
        for seed in SEEDS
    }


def _validate_complete_run(
    outdir: Path, split: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    if config.get("split") != split:
        raise RuntimeError(f"analysis split mismatch: {config.get('split')} != {split}")
    protocol = protocol_manifest()
    if config.get("protocol", {}).get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("analysis protocol hash drift")
    bindings = {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_combined_sha256": config["source"]["combined_sha256"],
    }
    completion = json.loads((outdir / "run_complete.json").read_text(encoding="utf-8"))
    expected_cells = _expected_cell_keys(split)
    expected_rows_per_cell = sum(
        (len(precision_bit_grid()) if step in PRECISION_CHECKPOINTS else 1)
        + 2 * ACTION_COUNT
        for step in RECOVERY_STEPS
    )
    if int(completion["cells"]) != len(expected_cells) or int(completion["rows"]) != (
        len(expected_cells) * expected_rows_per_cell
    ):
        raise RuntimeError("run_complete cardinality drift")

    cells = read_jsonl(outdir / "cells.jsonl")
    if len(cells) != len(expected_cells) or {row["cell_key"] for row in cells} != expected_cells:
        raise RuntimeError("COMP-006 cell journal is incomplete or contains extras")
    cell_by_key = {row["cell_key"]: row for row in cells}
    if len(cell_by_key) != len(cells):
        raise RuntimeError("duplicate COMP-006 cells")
    suite = target_suite()
    for cell in cells:
        if cell.get("schema") != "comp006-cell-v1" or any(
            cell.get(name) != value for name, value in bindings.items()
        ):
            raise RuntimeError(f"cell schema/binding drift for {cell.get('cell_key')}")
        item = suite[cell["target"]]
        if (
            cell["cell_key"] != _cell_key(split, cell["target"], int(cell["seed"]))
            or int(cell["seed"]) not in SEEDS
            or item["split"] != split
            or item["pixel_sha256"] != cell["target_pixel_sha256"]
            or item["family"] != cell["family"]
            or int(item["variant"]) != int(cell["variant"])
        ):
            raise RuntimeError(f"target binding drift for {cell['cell_key']}")
        parent_path = outdir / cell["parent"]["stream_path"]
        parent_blob = parent_path.read_bytes()
        if (
            len(parent_blob) != int(cell["parent"]["bytes"])
            or _sha256_bytes(parent_blob) != cell["parent"]["stream_sha256"]
        ):
            raise RuntimeError(f"parent stream drift for {cell['cell_key']}")
        _header, _components, decoded_parent = _validate_blob(
            parent_blob, expected_n=START_COUNT, expected_bits=BASE_BITS
        )
        decoded_parent_hash = _field_sha256(decoded_parent)
        if (
            decoded_parent_hash != cell["parent_field_sha256"]
            or decoded_parent_hash != cell["parent"]["decoded_field_sha256"]
            or cell["action_bank"]["parent_field_sha256"] != cell["parent_field_sha256"]
        ):
            raise RuntimeError(f"parent/action binding drift for {cell['cell_key']}")

    rows = read_jsonl(outdir / "streams.jsonl")
    if len(rows) != len(expected_cells) * expected_rows_per_cell:
        raise RuntimeError(
            f"COMP-006 stream journal has {len(rows)} rows; expected "
            f"{len(expected_cells) * expected_rows_per_cell}"
        )
    row_keys = [row["row_key"] for row in rows]
    if len(set(row_keys)) != len(row_keys):
        raise RuntimeError("duplicate COMP-006 stream row key")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["cell_key"] not in cell_by_key:
            raise RuntimeError(f"orphan stream row {row['row_key']}")
        _validate_existing_stream_row(
            outdir, row, bindings, cell=cell_by_key[row["cell_key"]]
        )
        grouped[(row["cell_key"], int(row["recovery_steps"]))].append(row)
    for cell_key in expected_cells:
        for checkpoint in RECOVERY_STEPS:
            group = grouped[(cell_key, checkpoint)]
            precision = [row for row in group if row["branch"] == "precision"]
            births = [row for row in group if row["branch"] == "birth"]
            replacements = [row for row in group if row["branch"] == "replace"]
            expected_precision = (
                set(precision_bit_grid()) if checkpoint in PRECISION_CHECKPOINTS else {BASE_BITS}
            )
            if (
                len(precision) != len(expected_precision)
                or {tuple(row["bits"]) for row in precision} != expected_precision
                or len(births) != ACTION_COUNT
                or {row["action_index"] for row in births} != set(range(ACTION_COUNT))
                or len(replacements) != ACTION_COUNT
                or {row["action_index"] for row in replacements} != set(range(ACTION_COUNT))
            ):
                raise RuntimeError(f"candidate grid drift for {cell_key} step {checkpoint}")
            if len({row["recovered_field_sha256"] for row in precision}) != 1:
                raise RuntimeError(f"precision mixes do not share one field for {cell_key}")
            range_keys = {
                _canonical_json(
                    {
                        name: row["header"][name]
                        for name in ("color_lo", "color_hi", "scale_lo", "scale_hi")
                    }
                )
                for row in precision
            }
            if len(range_keys) != 1:
                raise RuntimeError(f"precision mixes do not share frozen ranges for {cell_key}")
    return config, cells, rows


def _select_cell(
    rows: list[dict[str, Any]], *, cap_increment: int
) -> dict[str, Any]:
    base_matches = [
        row
        for row in rows
        if row["branch"] == "precision" and tuple(row["bits"]) == BASE_BITS
    ]
    if len(base_matches) != 1:
        raise RuntimeError("matched no-edit base row is missing or duplicated")
    base = base_matches[0]
    cap = int(base["bytes"]) + int(cap_increment)
    births = [row for row in rows if row["branch"] == "birth"]
    controls = [row for row in rows if row["branch"] in ("precision", "replace")]
    feasible_birth = [row for row in births if int(row["bytes"]) <= cap]
    feasible_control = [row for row in controls if int(row["bytes"]) <= cap]
    best_birth = _best(feasible_birth)
    best_control = _best(feasible_control)
    exact_winner = _best([*feasible_birth, *feasible_control])

    one_row_raw_bits = _raw_bits(1, BASE_BITS)
    raw_cap = int(base["raw_bits"]) + one_row_raw_bits
    raw_proxy = _best(row for row in rows if int(row["raw_bits"]) <= raw_cap)
    count_proxy_birth = _best(births)
    result: dict[str, Any] = {
        "cell_key": base["cell_key"],
        "split": base["split"],
        "target": base["target"],
        "family": base["family"],
        "variant": int(base["variant"]),
        "seed": int(base["seed"]),
        "recovery_steps": int(base["recovery_steps"]),
        "cap_increment_bytes": int(cap_increment),
        "base_bytes": int(base["bytes"]),
        "byte_cap": cap,
        "raw_proxy_cap_bits": raw_cap,
        "feasible_birth_count": len(feasible_birth),
        "feasible_control_count": len(feasible_control),
        "feasible": best_birth is not None and best_control is not None,
        "base": _row_identity(base),
        "best_birth": None if best_birth is None else _row_identity(best_birth),
        "best_control": None if best_control is None else _row_identity(best_control),
        "exact_winner": None if exact_winner is None else _row_identity(exact_winner),
        "raw_proxy_winner": None if raw_proxy is None else _row_identity(raw_proxy),
        "count_proxy_birth": (
            None if count_proxy_birth is None else _row_identity(count_proxy_birth)
        ),
        "raw_proxy_agrees": bool(
            raw_proxy is not None
            and exact_winner is not None
            and raw_proxy["row_key"] == exact_winner["row_key"]
        ),
        "count_proxy_birth_feasible": bool(
            count_proxy_birth is not None and int(count_proxy_birth["bytes"]) <= cap
        ),
    }
    if best_birth is not None and best_control is not None:
        result.update(
            {
                "birth_advantage_psnr": float(best_birth["psnr"] - best_control["psnr"]),
                "birth_advantage_mse": float(best_control["mse"] - best_birth["mse"]),
                "birth_delta_bytes": int(best_birth["bytes"] - base["bytes"]),
                "control_delta_bytes": int(best_control["bytes"] - base["bytes"]),
                "birth_component_delta": _component_delta(best_birth, base),
                "control_component_delta": _component_delta(best_control, base),
            }
        )
    else:
        result.update(
            {
                "birth_advantage_psnr": None,
                "birth_advantage_mse": None,
                "birth_delta_bytes": None,
                "control_delta_bytes": None,
                "birth_component_delta": None,
                "control_component_delta": None,
            }
        )
    return result


def _family_stratified_bootstrap(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, np.ndarray] = {}
    for family in TARGET_FAMILIES:
        values = np.asarray(
            [
                float(row["birth_advantage_psnr"])
                for row in target_rows
                if row["family"] == family
            ],
            dtype=np.float64,
        )
        if values.size != len(DEVELOPMENT_VARIANTS):
            raise RuntimeError(f"bootstrap requires three target means for family {family}")
        by_family[family] = values
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    distribution = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        draw = [
            values[rng.integers(0, values.size, size=values.size)]
            for values in by_family.values()
        ]
        distribution[replicate] = float(np.concatenate(draw).mean())
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "mean": float(np.mean([row["birth_advantage_psnr"] for row in target_rows])),
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
    }


def _write_overview_documents(outdir: Path, lines: Sequence[str]) -> None:
    markdown = "\n".join(lines) + "\n"
    _atomic_bytes(outdir / "summary.md", markdown.encode("utf-8"))
    escaped = html.escape("\n".join(lines))
    page = f"<!doctype html><meta charset='utf-8'><title>COMP-006</title><pre>{escaped}</pre>"
    _atomic_bytes(outdir / "index.html", page.encode("utf-8"))


def _write_analysis_overview(outdir: Path, summary: dict[str, Any]) -> None:
    primary = summary["primary"]
    lines = [
        "# COMP-006 marginal cold-stream RD",
        "",
        f"- Split: `{summary['split']}`",
        f"- Preliminary decision: **{summary['preliminary_decision']}**",
        f"- Mean birth minus strongest-control PSNR: "
        f"`{primary['mean_birth_advantage_psnr']:.4f} dB`",
        f"- Median advantage: `{primary['median_birth_advantage_psnr']:.4f} dB`",
        f"- Family-bootstrap 95% CI: "
        f"`[{primary['bootstrap']['ci95'][0]:.4f}, {primary['bootstrap']['ci95'][1]:.4f}] dB`",
        f"- Positive families: `{primary['positive_family_means']}/6`",
        f"- Feasible cells: `{primary['feasible_cells']}/{primary['total_cells']}`",
        "",
        "Replay is a separate integrity gate; this preliminary analysis cannot authorize the "
        "confirmation split.",
    ]
    _write_overview_documents(outdir, lines)


def _write_final_overview(outdir: Path, summary: dict[str, Any]) -> None:
    primary = summary["primary"]
    replay_passed = bool(summary["gates"]["replay_integrity"])
    confirmation = summary["decision"] == "authorize_confirmation"
    lines = [
        "# COMP-006 marginal cold-stream RD",
        "",
        f"- Split: `{summary['split']}`",
        f"- Final decision: **{summary['decision']}**",
        f"- Exact replay integrity: `{'pass' if replay_passed else 'fail'}`",
        f"- Mean birth minus strongest-control PSNR: "
        f"`{primary['mean_birth_advantage_psnr']:.4f} dB`",
        f"- Median advantage: `{primary['median_birth_advantage_psnr']:.4f} dB`",
        f"- Family-bootstrap 95% CI: "
        f"`[{primary['bootstrap']['ci95'][0]:.4f}, "
        f"{primary['bootstrap']['ci95'][1]:.4f}] dB`",
        f"- Positive families: `{primary['positive_family_means']}/6`",
        f"- Feasible cells: `{primary['feasible_cells']}/{primary['total_cells']}`",
        "",
        (
            "The frozen development gate and exact replay authorize the confirmation split."
            if confirmation
            else "The frozen development gate does not authorize the confirmation split."
        ),
    ]
    _write_overview_documents(outdir, lines)


def analyze(outdir: str | Path, *, split: str = "development") -> dict[str, Any]:
    outdir = Path(outdir)
    config, _cells, rows = _validate_complete_run(outdir, split)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    row_by_key = {row["row_key"]: row for row in rows}
    for row in rows:
        grouped[(row["cell_key"], int(row["recovery_steps"]))].append(row)
    selections: list[dict[str, Any]] = []
    for key in sorted(grouped):
        for cap in SENSITIVITY_CAP_INCREMENTS:
            selections.append(_select_cell(grouped[key], cap_increment=cap))
    secondary_cache: dict[str, tuple[float, float]] = {}
    suite = target_suite()
    for selection in selections:
        target = torch.as_tensor(
            suite[selection["target"]]["image"], dtype=torch.float32, device="cpu"
        )
        for name in ("base", "best_birth", "best_control", "exact_winner"):
            identity = selection[name]
            if identity is None or identity["ssim"] is not None:
                continue
            key = identity["row_key"]
            if key not in secondary_cache:
                raw = row_by_key[key]
                blob = (outdir / raw["stream_path"]).read_bytes()
                prediction = codec.decode_and_render(blob, device="cpu").clamp(0.0, 1.0)
                secondary_cache[key] = (
                    float(M.ssim(prediction, target)),
                    float(M.ms_ssim(prediction, target)),
                )
            identity["ssim"], identity["ms_ssim"] = secondary_cache[key]

    primary_cells = [
        row
        for row in selections
        if row["recovery_steps"] == 20
        and row["cap_increment_bytes"] == PRIMARY_CAP_INCREMENT
    ]
    feasible_cells = [row for row in primary_cells if row["feasible"]]
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feasible_cells:
        by_target[row["target"]].append(row)
    expected_targets = {
        name for name, item in target_suite().items() if item["split"] == split
    }
    if set(by_target) != expected_targets:
        missing = sorted(expected_targets - set(by_target))
        raise RuntimeError(f"at least one feasible seed is required for every target; missing {missing}")
    target_rows: list[dict[str, Any]] = []
    for target in sorted(expected_targets):
        seed_rows = by_target[target]
        target_rows.append(
            {
                "target": target,
                "family": seed_rows[0]["family"],
                "variant": seed_rows[0]["variant"],
                "feasible_seed_count": len(seed_rows),
                "birth_advantage_psnr": float(
                    np.mean([row["birth_advantage_psnr"] for row in seed_rows])
                ),
                "birth_advantage_mse": float(
                    np.mean([row["birth_advantage_mse"] for row in seed_rows])
                ),
                "birth_delta_bytes": float(
                    np.mean([row["birth_delta_bytes"] for row in seed_rows])
                ),
                "control_delta_bytes": float(
                    np.mean([row["control_delta_bytes"] for row in seed_rows])
                ),
                "raw_proxy_agreement": float(
                    np.mean([row["raw_proxy_agrees"] for row in seed_rows])
                ),
            }
        )
    values = np.asarray(
        [row["birth_advantage_psnr"] for row in target_rows], dtype=np.float64
    )
    family_means = {
        family: float(
            np.mean(
                [
                    row["birth_advantage_psnr"]
                    for row in target_rows
                    if row["family"] == family
                ]
            )
        )
        for family in TARGET_FAMILIES
    }
    bootstrap = _family_stratified_bootstrap(target_rows)
    primary = {
        "total_cells": len(primary_cells),
        "feasible_cells": len(feasible_cells),
        "feasible_fraction": len(feasible_cells) / len(primary_cells),
        "mean_birth_advantage_psnr": float(values.mean()),
        "median_birth_advantage_psnr": float(np.median(values)),
        "bootstrap": bootstrap,
        "family_means": family_means,
        "positive_family_means": int(sum(value > 0.0 for value in family_means.values())),
        "raw_proxy_top1_agreement": float(
            np.mean([row["raw_proxy_agrees"] for row in feasible_cells])
        ),
        "target_rows": target_rows,
    }
    gates = {
        "feasible_cells_at_least_90pct": primary["feasible_fraction"] >= 0.90,
        "mean_advantage_at_least_0_15_db": primary["mean_birth_advantage_psnr"] >= 0.15,
        "bootstrap_lower_bound_above_zero": bootstrap["ci95"][0] > 0.0,
        "median_advantage_at_least_0_10_db": primary["median_birth_advantage_psnr"] >= 0.10,
        "at_least_four_positive_families": primary["positive_family_means"] >= 4,
        "integrity": True,
    }
    preliminary = "await_replay" if all(gates.values()) else "stop_pending_replay"
    summary = {
        "schema": "comp006-analysis-v1",
        "task": "COMP-006",
        "split": split,
        "protocol_sha256": config["protocol"]["protocol_sha256"],
        "source_combined_sha256": config["source"]["combined_sha256"],
        "primary_cap_increment_bytes": PRIMARY_CAP_INCREMENT,
        "primary_recovery_steps": 20,
        "primary": primary,
        "gates": gates,
        "preliminary_decision": preliminary,
        "note": "replay is required before any final decision or confirmation authorization",
    }
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "selections.jsonl").open("w", encoding="utf-8") as handle:
        for row in selections:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _atomic_json(analysis_dir / "target_rows.json", target_rows)
    _atomic_json(analysis_dir / "summary.json", summary)
    _write_analysis_overview(outdir, summary)
    return summary


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"timing", "fit_seconds", "qat_seconds", "command"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _journal_hash(path: Path, key: str) -> str:
    rows = read_jsonl(path)
    normalized = [_without_timing(row) for row in sorted(rows, key=lambda row: row[key])]
    return _sha256_bytes(_canonical_json(normalized))


FINAL_GATE_KEYS = {
    "feasible_cells_at_least_90pct",
    "mean_advantage_at_least_0_15_db",
    "bootstrap_lower_bound_above_zero",
    "median_advantage_at_least_0_10_db",
    "at_least_four_positive_families",
    "integrity",
    "replay_integrity",
}


def verify_replay(primary: str | Path, replay: str | Path) -> dict[str, Any]:
    primary = Path(primary)
    replay = Path(replay)
    primary_config = json.loads((primary / "config.json").read_text(encoding="utf-8"))
    replay_config = json.loads((replay / "config.json").read_text(encoding="utf-8"))
    split = primary_config["split"]
    if replay_config["split"] != split:
        raise RuntimeError("replay split mismatch")
    primary_summary = analyze(primary, split=split)
    replay_summary = analyze(replay, split=split)
    checks = {
        "protocol_sha256": primary_config["protocol"]["protocol_sha256"]
        == replay_config["protocol"]["protocol_sha256"],
        "source_combined_sha256": primary_config["source"]["combined_sha256"]
        == replay_config["source"]["combined_sha256"],
        "environment_sha256": _environment_sha256(primary_config["environment"])
        == _environment_sha256(replay_config["environment"]),
        "cells": _journal_hash(primary / "cells.jsonl", "cell_key")
        == _journal_hash(replay / "cells.jsonl", "cell_key"),
        "streams": _journal_hash(primary / "streams.jsonl", "row_key")
        == _journal_hash(replay / "streams.jsonl", "row_key"),
        "analysis": _without_timing(primary_summary) == _without_timing(replay_summary),
    }
    replay_integrity = all(checks.values())
    final_gates = {**primary_summary["gates"], "replay_integrity": replay_integrity}
    if set(final_gates) != FINAL_GATE_KEYS:
        raise RuntimeError("final COMP-006 gate schema drift")
    decision = "authorize_confirmation" if all(final_gates.values()) else "stop"
    final = {
        "schema": "comp006-final-v1",
        "task": "COMP-006",
        "split": split,
        "protocol_sha256": primary_config["protocol"]["protocol_sha256"],
        "source_combined_sha256": primary_config["source"]["combined_sha256"],
        "environment_sha256": _environment_sha256(primary_config["environment"]),
        "checks": checks,
        "gates": final_gates,
        "decision": decision,
        "primary": primary_summary["primary"],
    }
    _atomic_json(primary / "replay_comparison.json", final)
    _atomic_json(primary / "final_summary.json", final)
    _write_final_overview(primary, final)
    if not replay_integrity:
        raise RuntimeError(f"COMP-006 replay mismatch: {checks}")
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="print the frozen protocol manifest")
    manifest.add_argument("--compact", action="store_true")

    run = subparsers.add_parser("run", help="run or resume one authorized split")
    run.add_argument("--outdir", required=True)
    run.add_argument("--split", choices=("development", "confirmation"), default="development")
    run.add_argument("--development-summary")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="debug/resume plumbing only; incomplete output cannot be analyzed",
    )

    analyze_parser = subparsers.add_parser("analyze", help="validate and analyze a complete run")
    analyze_parser.add_argument("--outdir", required=True)
    analyze_parser.add_argument(
        "--split", choices=("development", "confirmation"), default="development"
    )

    finalize = subparsers.add_parser("finalize-run", help="seal a complete set of run shards")
    finalize.add_argument("--outdir", required=True)
    finalize.add_argument(
        "--split", choices=("development", "confirmation"), default="development"
    )
    finalize.add_argument("--num-shards", type=int, required=True)

    replay = subparsers.add_parser(
        "verify-replay", help="compare two complete same-source runs and finalize the decision"
    )
    replay.add_argument("--primary", required=True)
    replay.add_argument("--replay", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        payload = protocol_manifest()
        print(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if args.compact
            else json.dumps(payload, indent=2, sort_keys=True)
        )
    elif args.command == "run":
        run_benchmark(
            args.outdir,
            split=args.split,
            development_summary=args.development_summary,
            max_cells=args.max_cells,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    elif args.command == "analyze":
        payload = analyze(args.outdir, split=args.split)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "finalize-run":
        payload = finalize_run(args.outdir, split=args.split, num_shards=args.num_shards)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "verify-replay":
        payload = verify_replay(args.primary, args.replay)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
