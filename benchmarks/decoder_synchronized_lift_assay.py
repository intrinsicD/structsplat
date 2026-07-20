"""Source-bound BENCH-015 decoder-synchronized affine-lift assay.

This benchmark is intentionally independent of the BENCH-014 runner.  It uses the audited
``GFCOV01`` field codec directly, wraps every field in the fixed-size ``DSLR015`` container, and
derives the robust affine lift from a cold-decoded ordinary field.  No target, coefficient,
residual, confidence, or other side information crosses the decoder boundary.

The runner writes its task/source/config binding before creating a canonical target or field,
uses append-only ledgers for every decision-bearing lane, builds an artifact manifest, and then
recomputes the evidence before writing ``completion.json`` last.  A post-creation exception seals
the directory as invalid; a non-empty directory is never resumed or repaired in place.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any, BinaryIO, Callable, Mapping, Sequence
import zipfile
import zlib

import numpy as np
import torch

from benchmarks import affine_carrier_core as render_core
from benchmarks import decoder_synchronized_lift_core as core
from benchmarks import gauge_free_covariance_codec as field_codec
from structsplat.config import InitConfig, StructureTensorConfig
from structsplat.gaussians import GaussianField
from structsplat.init import build_field
from structsplat.render import render as production_render


PROTOCOL = "bench015-decoder-synchronized-affine-lift-stage0-v1"
CONFIG_SCHEMA = "bench015-decoder-synchronized-affine-lift-config-v1"
HEIGHT = 83
WIDTH = 107
PROXY_COUNT = 78
PRIMARY_COUNT = 80
COUNTS = (PROXY_COUNT, PRIMARY_COUNT)
SEEDS = (401, 409, 419)
COHORTS = ("target_conditioned", "shared_constant")
ARMS = ("nw78", "dsl78", "nw80")

TARGETS = (
    "constant",
    "affine",
    "affine_chirp",
    "affine_soft_crease",
    "affine_ring",
    "zero_linear_quartic",
    "affine_occlusion",
    "continuous_crease",
    "isoluminant_step",
)
SMOOTH_TARGETS = ("affine_chirp", "affine_soft_crease", "affine_ring")
MIXED_HARD_TARGETS = (
    "affine_occlusion",
    "continuous_crease",
    "isoluminant_step",
)
TARGET_FORMULAS = {
    "constant": "(0.30,0.52,0.68)",
    "affine": ("(0.46+0.15*q_x-0.11*q_y,0.54-0.13*q_x+0.14*q_y,0.42+0.09*q_x+0.18*q_y)"),
    "affine_chirp": (
        "p=2*pi*(0.70*u+1.10*u^2+0.35*v);"
        "(0.46+0.13*q_x-0.08*q_y+0.045*sin(p),"
        "0.54-0.11*q_x+0.12*q_y+0.040*cos(p+0.5*pi*v),"
        "0.43+0.07*q_x+0.15*q_y+0.040*sin(p-0.7*pi*v))"
    ),
    "affine_soft_crease": (
        "t=(u-0.52)+0.45*(v-0.50);h=sqrt(t^2+0.035^2);"
        "(0.45+0.12*q_x-0.07*q_y+0.075*h,"
        "0.55-0.10*q_x+0.11*q_y-0.065*h,"
        "0.42+0.06*q_x+0.14*q_y+0.055*h)"
    ),
    "affine_ring": (
        "g=exp(-(((u-0.68)^2+(v-0.31)^2-0.18^2)^2)/(2*0.012^2));"
        "(0.45+0.12*q_x-0.08*q_y+0.11*g,"
        "0.55-0.10*q_x+0.11*q_y-0.09*g,"
        "0.42+0.07*q_x+0.14*q_y+0.10*g)"
    ),
    "zero_linear_quartic": (
        "r2=q_x^2+0.7*q_y^2;(0.46+0.045*(q_x^4-0.2),0.53+0.040*(q_y^4-0.2),0.44+0.035*(r2^2-0.35))"
    ),
    "affine_occlusion": (
        "rho=((u-0.66)/0.12)^2+((v-0.36)/0.16)^2;affine+where(rho<1,(0.16,-0.14,0.12),(0,0,0))"
    ),
    "continuous_crease": (
        "t=u+0.35*v-0.68;k=max(t,0);"
        "(0.38+0.18*u+0.12*v+0.20*k,"
        "0.58-0.14*u+0.10*v-0.16*k,"
        "0.36+0.10*u+0.16*v+0.18*k)"
    ),
    "isoluminant_step": ("(0.20,0.60,0.40) if u+0.60*v<0.82 else (0.75,0.45164988814317675,0.25)"),
}
REGION_FORMULAS = {
    "full": "all H*W pixels",
    "outer_ten": "x<10 or x>=97 or y<10 or y>=73",
    "affine_occlusion_edge": "abs(sqrt(rho)-1)<=0.18",
    "affine_occlusion_away": "sqrt(rho)>=1.35",
    "continuous_crease_edge": ("abs(u+0.35*v-0.68)/sqrt((1/106)^2+(0.35/82)^2)<=10"),
    "isoluminant_step_edge": ("abs(u+0.60*v-0.82)/sqrt((1/106)^2+(0.60/82)^2)<=10"),
}

MINIMUM_MASS = 1e-5
MINIMUM_ACTIVE = 1
PARTITION_MAX_ABS = 2e-5
CANDIDATE_REFERENCE_MAX_ABS = 2e-5
MINIMUM_EFFECTIVE_WEIGHT = -2e-7
RANGE_MARGIN = 2.0 / 255.0
PERMUTATION_REFERENCE_MAX_ABS = 1e-11
PERMUTATION_CANDIDATE_MAX_ABS = 2e-5

CONVERGENCE_STEPS = 60
CONVERGENCE_LR = 0.03
CONVERGENCE_BETAS = (0.9, 0.999)
CONVERGENCE_EPS = 1e-8
CONVERGENCE_CHECKPOINTS = (0, 20, 40, 60)
PERMUTATION_SEED = 20260716025
GRADIENT_SEED = 20260716026
GRADIENT_STEP = 2.0**-16
TIMING_SEED = 20260716027
TIMING_WARMUPS = 10
TIMING_REPETITIONS = 40

STREAM_MAGIC = b"DSLR015\0"
STREAM_VERSION = 1
STREAM_HEADER = struct.Struct("<8sBBHIHH")  # fixed 20 bytes
STREAM_CRC = struct.Struct("<I")
STREAM_ARRAYS = ("means", "log_scales", "rotations", "colors")

EXPECTED_FIELDS = len(TARGETS) * len(COUNTS) * len(SEEDS) + len(COUNTS) * len(SEEDS)
EXPECTED_CELLS = len(COHORTS) * len(TARGETS) * len(SEEDS)
EXPECTED_STATIC = EXPECTED_CELLS * len(ARMS)
EXPECTED_PERMUTATIONS = EXPECTED_STATIC * 3
EXPECTED_TRAJECTORIES = EXPECTED_CELLS * 2
EXPECTED_GRADIENTS = 6
EXPECTED_TIMING_UNITS = EXPECTED_CELLS

TASK_PATH = "tasks/BENCH-015-decoder-synchronized-affine-lift.md"
SOURCE_PATHS = (
    "benchmarks/__init__.py",
    "benchmarks/affine_carrier_core.py",
    "benchmarks/decoder_synchronized_lift_core.py",
    "benchmarks/decoder_synchronized_lift_assay.py",
    "benchmarks/gauge_free_covariance_codec.py",
    "benchmarks/gauge_free_covariance_core.py",
    "src/structsplat/__init__.py",
    "src/structsplat/config.py",
    "src/structsplat/density.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/init.py",
    "src/structsplat/render.py",
    "src/structsplat/sampling.py",
    "src/structsplat/structural_controls.py",
    "src/structsplat/structure_tensor.py",
    TASK_PATH,
    "pyproject.toml",
)
ARCHIVE_PATHS = SOURCE_PATHS + (
    "tests/test_decoder_synchronized_lift_core.py",
    "tests/test_decoder_synchronized_lift_assay.py",
)


class MethodUnavailableError(RuntimeError):
    """A valid decoded stream lies outside the frozen DSL method domain."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {"nbytes": len(payload), "sha256": _sha256_bytes(payload)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _array_record(value: np.ndarray | torch.Tensor) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(value)
    record: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "payload_sha256": _sha256_bytes(array.tobytes(order="C")),
    }
    record["record_sha256"] = _sha256_bytes(_canonical_json(record))
    return record


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    _atomic_bytes(path, payload)


def _append_jsonl(handle: BinaryIO, row: Mapping[str, Any]) -> None:
    handle.write(_canonical_json(_json_safe(dict(row))) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return buffer.getvalue()


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.ascontiguousarray(arrays[name])))
    _atomic_bytes(path, buffer.getvalue())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _persisted_npz_matches(
    path: Path,
    expected_arrays: Mapping[str, np.ndarray],
    expected_sha256: str,
    expected_records: Mapping[str, Any],
) -> bool:
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        return False
    try:
        observed = _load_npz(path)
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    expected_keys = set(expected_arrays)
    return bool(
        set(observed) == expected_keys
        and set(expected_records) == expected_keys
        and all(
            np.array_equal(observed[name], np.ascontiguousarray(expected_arrays[name]))
            and expected_records[name] == _array_record(observed[name])
            for name in expected_keys
        )
    )


def _normalized_coordinates(xy: np.ndarray, height: int, width: int) -> tuple[np.ndarray, ...]:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError("xy must be a finite array with trailing shape 2")
    x = points[..., 0]
    y = points[..., 1]
    u = x / float(width - 1)
    v = y / float(height - 1)
    return x, y, u, v, 2.0 * u - 1.0, 2.0 * v - 1.0


def target_values(
    target: str,
    xy: np.ndarray,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
) -> np.ndarray:
    """Evaluate one frozen BENCH-015 analytic target at continuous coordinates."""

    if target not in TARGETS:
        raise ValueError(f"unknown BENCH-015 target {target!r}")
    points = np.asarray(xy, dtype=np.float64)
    _, _, u, v, qx, qy = _normalized_coordinates(points, height, width)
    shape = points.shape[:-1] + (3,)
    if target == "constant":
        result = np.broadcast_to(np.array((0.30, 0.52, 0.68)), shape)
    elif target == "affine":
        result = np.stack(
            (
                0.46 + 0.15 * qx - 0.11 * qy,
                0.54 - 0.13 * qx + 0.14 * qy,
                0.42 + 0.09 * qx + 0.18 * qy,
            ),
            axis=-1,
        )
    elif target == "affine_chirp":
        phase = 2.0 * np.pi * (0.70 * u + 1.10 * u**2 + 0.35 * v)
        result = np.stack(
            (
                0.46 + 0.13 * qx - 0.08 * qy + 0.045 * np.sin(phase),
                0.54 - 0.11 * qx + 0.12 * qy + 0.040 * np.cos(phase + 0.5 * np.pi * v),
                0.43 + 0.07 * qx + 0.15 * qy + 0.040 * np.sin(phase - 0.7 * np.pi * v),
            ),
            axis=-1,
        )
    elif target == "affine_soft_crease":
        signed = (u - 0.52) + 0.45 * (v - 0.50)
        smooth = np.sqrt(signed**2 + 0.035**2)
        result = np.stack(
            (
                0.45 + 0.12 * qx - 0.07 * qy + 0.075 * smooth,
                0.55 - 0.10 * qx + 0.11 * qy - 0.065 * smooth,
                0.42 + 0.06 * qx + 0.14 * qy + 0.055 * smooth,
            ),
            axis=-1,
        )
    elif target == "affine_ring":
        ring = np.exp(-(((u - 0.68) ** 2 + (v - 0.31) ** 2 - 0.18**2) ** 2 / (2.0 * 0.012**2)))
        result = np.stack(
            (
                0.45 + 0.12 * qx - 0.08 * qy + 0.11 * ring,
                0.55 - 0.10 * qx + 0.11 * qy - 0.09 * ring,
                0.42 + 0.07 * qx + 0.14 * qy + 0.10 * ring,
            ),
            axis=-1,
        )
    elif target == "zero_linear_quartic":
        radius2 = qx**2 + 0.7 * qy**2
        result = np.stack(
            (
                0.46 + 0.045 * (qx**4 - 0.2),
                0.53 + 0.040 * (qy**4 - 0.2),
                0.44 + 0.035 * (radius2**2 - 0.35),
            ),
            axis=-1,
        )
    elif target == "affine_occlusion":
        background = np.stack(
            (
                0.46 + 0.15 * qx - 0.11 * qy,
                0.54 - 0.13 * qx + 0.14 * qy,
                0.42 + 0.09 * qx + 0.18 * qy,
            ),
            axis=-1,
        )
        rho = ((u - 0.66) / 0.12) ** 2 + ((v - 0.36) / 0.16) ** 2
        result = background + (rho < 1.0)[..., None] * np.array((0.16, -0.14, 0.12))
    elif target == "continuous_crease":
        kink = np.maximum(u + 0.35 * v - 0.68, 0.0)
        result = np.stack(
            (
                0.38 + 0.18 * u + 0.12 * v + 0.20 * kink,
                0.58 - 0.14 * u + 0.10 * v - 0.16 * kink,
                0.36 + 0.10 * u + 0.16 * v + 0.18 * kink,
            ),
            axis=-1,
        )
    else:
        low = np.array((0.20, 0.60, 0.40))
        high = np.array((0.75, 0.45164988814317675, 0.25))
        result = np.where((u + 0.60 * v < 0.82)[..., None], low, high)
    return np.ascontiguousarray(result, dtype=dtype)


def target_image(
    target: str,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return target_values(
        target,
        np.stack((xx, yy), axis=-1),
        height=height,
        width=width,
        dtype=dtype,
    )


def region_masks(target: str, *, height: int = HEIGHT, width: int = WIDTH) -> dict[str, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    u = xx / float(width - 1)
    v = yy / float(height - 1)
    result = {
        "all": np.ones((height, width), dtype=bool),
        "outer_ten": (xx < 10) | (xx >= width - 10) | (yy < 10) | (yy >= height - 10),
    }
    if target == "affine_occlusion":
        root_rho = np.sqrt(((u - 0.66) / 0.12) ** 2 + ((v - 0.36) / 0.16) ** 2)
        result["edge"] = np.abs(root_rho - 1.0) <= 0.18
        result["away"] = root_rho >= 1.35
    elif target == "continuous_crease":
        distance = np.abs(u + 0.35 * v - 0.68) / math.sqrt(
            (1.0 / float(width - 1)) ** 2 + (0.35 / float(height - 1)) ** 2
        )
        result["edge"] = distance <= 10.0
    elif target == "isoluminant_step":
        distance = np.abs(u + 0.60 * v - 0.82) / math.sqrt(
            (1.0 / float(width - 1)) ** 2 + (0.60 / float(height - 1)) ** 2
        )
        result["edge"] = distance <= 10.0
    return result


def _gradient_direction_arrays() -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(GRADIENT_SEED))
    fixtures = (
        (f"{_cell_id('target_conditioned', 'affine_chirp', 401)}__dsl78", PROXY_COUNT),
        (f"{_cell_id('target_conditioned', 'affine_occlusion', 401)}__dsl78", PROXY_COUNT),
        ("synthetic_mid_transition", 49),
    )
    arrays: dict[str, np.ndarray] = {}
    for fixture, count in fixtures:
        for direction_index in range(2):
            direction = rng.normal(size=(count, 3))
            direction /= np.linalg.norm(direction)
            arrays[f"{fixture}__d{direction_index}"] = np.ascontiguousarray(
                direction, dtype=np.float64
            )
    return arrays


def _synthetic_fixture_arrays() -> dict[str, np.ndarray]:
    ys = torch.linspace(0.0, HEIGHT - 1.0, 7, dtype=torch.float64)
    xs = torch.linspace(0.0, WIDTH - 1.0, 7, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    phase = 2.0 * math.pi * means[:, 0] / (WIDTH - 1)
    wave = 0.02 * torch.sin(phase) * torch.cos(math.pi * means[:, 1] / (HEIGHT - 1))
    colors = torch.stack((0.5 + wave, 0.5 + 0.8 * wave, 0.5 - 0.6 * wave), dim=1)
    pixels = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float64)
    distance = torch.cdist(pixels[:, 1:3], core.affine_design(means, HEIGHT, WIDTH)[:, 1:3])
    weights = torch.exp(-5.0 * distance**2)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    return {
        "synthetic_means": means.numpy(),
        "synthetic_colors": colors.numpy(),
        "synthetic_weights": weights.numpy(),
    }


def _registered_mask_arrays() -> dict[str, np.ndarray]:
    arrays = {
        "mask_full": region_masks("constant")["all"],
        "mask_outer_ten": region_masks("constant")["outer_ten"],
        "mask_affine_occlusion_edge": region_masks("affine_occlusion")["edge"],
        "mask_affine_occlusion_away": region_masks("affine_occlusion")["away"],
        "mask_continuous_crease_edge": region_masks("continuous_crease")["edge"],
        "mask_isoluminant_step_edge": region_masks("isoluminant_step")["edge"],
    }
    return {name: np.ascontiguousarray(value, dtype=bool) for name, value in arrays.items()}


def _prebinding_arrays() -> dict[str, np.ndarray]:
    return {
        **_gradient_direction_arrays(),
        **_synthetic_fixture_arrays(),
        **_registered_mask_arrays(),
    }


def _prebinding_records(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, value in arrays.items():
        record = _array_record(value)
        if name.startswith("mask_"):
            record["nonzero_pixels"] = int(np.count_nonzero(value))
            record["nonzero_pass"] = bool(np.count_nonzero(value) > 0)
        records[name] = record
    return records


def frozen_configs(count: int, seed: int) -> tuple[InitConfig, StructureTensorConfig]:
    init = InitConfig(
        strategy="quadtree_wse",
        num_gaussians=int(count),
        candidate_oversample=6.0,
        density_base=0.05,
        density_power=1.0,
        density_mode="structure",
        sampling_mode="wse",
        wse_progressive_order=False,
        max_axis_ratio=6.0,
        coherence_power=1.0,
        orientation_mode="tensor",
        scale_mode="spacing",
        init_scale_mult=1.0,
        scale_cap_mode="none",
        scale_cap_max=None,
        background_fraction=0.0,
        background_grid=0,
        flank_offset_frac=0.0,
        color_mode="bilinear",
        opacity_mode="none",
        seed=int(seed),
    )
    structure = StructureTensorConfig(
        grad_sigma=1.0,
        tensor_sigma=2.0,
        gradient_operator="central",
        color_space="luma",
        flat_frac=0.02,
        corner_frac=0.15,
    )
    return init, structure


def _field_id(cohort: str, source_target: str, count: int, seed: int) -> str:
    return f"{cohort}__{source_target}__n{count:04d}__s{seed}"


def _cell_id(cohort: str, target: str, seed: int) -> str:
    return f"{cohort}__{target}__s{seed}"


def field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for target in TARGETS:
        for count in COUNTS:
            for seed in SEEDS:
                specs.append(
                    {
                        "field_id": _field_id("target_conditioned", target, count, seed),
                        "cohort": "target_conditioned",
                        "source_target": target,
                        "count": count,
                        "seed": seed,
                    }
                )
    for count in COUNTS:
        for seed in SEEDS:
            specs.append(
                {
                    "field_id": _field_id("shared_constant", "constant", count, seed),
                    "cohort": "shared_constant",
                    "source_target": "constant",
                    "count": count,
                    "seed": seed,
                }
            )
    return tuple(sorted(specs, key=lambda row: str(row["field_id"])))


def cell_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for target in TARGETS:
            for seed in SEEDS:
                source = target if cohort == "target_conditioned" else "constant"
                specs.append(
                    {
                        "cell_id": _cell_id(cohort, target, seed),
                        "cohort": cohort,
                        "target": target,
                        "seed": seed,
                        "field78_id": _field_id(cohort, source, PROXY_COUNT, seed),
                        "field80_id": _field_id(cohort, source, PRIMARY_COUNT, seed),
                    }
                )
    return tuple(sorted(specs, key=lambda row: str(row["cell_id"])))


def encode_stream(
    arm: str,
    arrays: Mapping[str, np.ndarray],
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
) -> bytes:
    """Encode one no-tail DSLR015 wrapper around a fresh GFCOV01 field."""

    if arm not in ARMS:
        raise ValueError(f"unknown BENCH-015 arm {arm!r}")
    count = int(np.asarray(arrays["means"]).shape[0])
    expected_count = PRIMARY_COUNT if arm == "nw80" else PROXY_COUNT
    if count != expected_count:
        raise ValueError(f"{arm} requires exactly {expected_count} rows")
    mode = 1 if arm == "dsl78" else 0
    expected_shapes = {
        "means": (count, 2),
        "log_scales": (count, 2),
        "rotations": (count,),
        "colors": (count, 3),
    }
    state: dict[str, torch.Tensor] = {}
    for name in STREAM_ARRAYS:
        value = np.asarray(arrays[name])
        if value.shape != expected_shapes[name]:
            raise ValueError(f"stream {name} must have shape {expected_shapes[name]}")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"stream {name} must be finite floating point")
        state[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).clone()
    field = GaussianField(state["means"], state["log_scales"], state["rotations"], state["colors"])
    config = field_codec.CodecConfig(
        chart="current_rs",
        geometry_bits=(6, 6, 6),
        predictor="absolute",
        coder="zlib9",
        bits_means=12,
        bits_colors=8,
    )
    inner = field_codec.encode(field, int(height), int(width), config)
    header = STREAM_HEADER.pack(
        STREAM_MAGIC,
        STREAM_VERSION,
        mode,
        0,
        len(inner),
        int(height),
        int(width),
    )
    framed = header + inner
    return framed + STREAM_CRC.pack(zlib.crc32(framed) & 0xFFFFFFFF)


def _decode_stream_impl(
    blob: bytes,
    device: str | torch.device,
    *,
    audit_reencode: bool,
    collect_metadata: bool = True,
) -> dict[str, Any]:
    payload = bytes(blob)
    minimum = STREAM_HEADER.size + STREAM_CRC.size
    if len(payload) < minimum:
        raise ValueError("truncated DSLR015 stream")
    magic, version, mode, reserved, inner_length, height, width = STREAM_HEADER.unpack_from(payload)
    if magic != STREAM_MAGIC or version != STREAM_VERSION or reserved != 0:
        raise ValueError("invalid DSLR015 header")
    if mode not in (0, 1):
        raise ValueError("DSLR015 mode must be NW=0 or DSL=1")
    expected_length = STREAM_HEADER.size + inner_length + STREAM_CRC.size
    if expected_length != len(payload):
        raise ValueError("DSLR015 length metadata is inconsistent")
    stored_crc = STREAM_CRC.unpack_from(payload, len(payload) - STREAM_CRC.size)[0]
    if stored_crc != (zlib.crc32(payload[: -STREAM_CRC.size]) & 0xFFFFFFFF):
        raise ValueError("DSLR015 CRC mismatch")
    inner = bytes(payload[STREAM_HEADER.size : -STREAM_CRC.size])
    header = field_codec.blob_header(inner)
    if int(header["height"]) != int(height) or int(header["width"]) != int(width):
        raise ValueError("DSLR015 extent disagrees with inner field")
    count = int(header["n"])
    if count not in COUNTS or (mode == 1 and count != PROXY_COUNT):
        raise ValueError("DSLR015 mode/count combination is outside the frozen matrix")
    if (
        header["format"] != "GFCOV01"
        or header["chart"] != "current_rs"
        or header["predictor"] != "absolute"
        or header["coder"] != "zlib9"
        or int(header["bits_means"]) != 12
        or tuple(header["geometry_bits"]) != (6, 6, 6)
        or int(header["bits_colors"]) != 8
    ):
        raise ValueError("inner GFCOV01 stream violates the BENCH-015 profile")
    if audit_reencode:
        if field_codec.canonical_reencode(inner) != inner:
            raise ValueError("inner GFCOV01 stream is noncanonical")
        if field_codec.decoded_reencode(inner) != inner:
            raise ValueError("cold-decoded GFCOV01 stream does not re-encode exactly")
    field = field_codec.decode(inner, device=device)
    if field.means.device.type != "cpu":
        raise ValueError("BENCH-015 reference decoder requires CPU")
    arrays = {
        "means": field.means.detach().cpu().numpy().copy(),
        "log_scales": field.log_scales.detach().cpu().numpy().copy(),
        "rotations": field.rotations.detach().cpu().numpy().copy(),
        "colors": field.colors.detach().cpu().numpy().copy(),
    }
    lift_state = None
    method_error: dict[str, str] | None = None
    if mode == 1:
        try:
            lift_state = core.derive_lift(
                field.means.to(torch.float64),
                field.colors.to(torch.float64),
                int(height),
                int(width),
            )
        except (ValueError, RuntimeError) as error:
            method_error = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
    arm = "dsl78" if mode == 1 else f"nw{count}"
    return {
        "arm": arm,
        "mode": int(mode),
        "height": int(height),
        "width": int(width),
        "count": count,
        "arrays": arrays,
        "inner": inner,
        "inner_header": header,
        "inner_components": (field_codec.blob_components(inner) if collect_metadata else None),
        "lift_state": lift_state,
        "method_available": mode == 0 or lift_state is not None,
        "method_error": method_error,
    }


def decode_stream(blob: bytes, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Cold-decode from the complete blob and device only."""

    return _decode_stream_impl(blob, device, audit_reencode=True)


def _git_metadata(root: Path) -> dict[str, Any]:
    def invoke(*arguments: str, binary: bool = False) -> bytes | str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=not binary,
        )
        return result.stdout

    try:
        status = str(invoke("status", "--porcelain=v1", "--untracked-files=all")).strip()
        difference = bytes(invoke("diff", "--binary", "--no-ext-diff", binary=True))
        return {
            "commit": str(invoke("rev-parse", "HEAD")).strip(),
            "branch": str(invoke("branch", "--show-current")).strip(),
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
            "tracked_diff_sha256": _sha256_bytes(difference),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "branch": None,
            "dirty": None,
            "status_sha256": None,
            "tracked_diff_sha256": None,
        }


def _environment(root: Path) -> dict[str, Any]:
    try:
        zstandard_version = importlib.metadata.version("zstandard")
    except importlib.metadata.PackageNotFoundError:
        zstandard_version = None
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "cpu_model": cpu_model,
        "cpu_logical_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "torch_build_config": torch.__config__.show(),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "zstandard": zstandard_version,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "git": _git_metadata(root),
    }


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "python",
        "platform",
        "numpy",
        "torch",
        "device",
        "cpu_model",
        "cpu_logical_count",
        "cpu_affinity",
        "torch_build_config",
        "zlib_compile",
        "zlib_runtime",
        "zstandard",
        "torch_num_threads",
        "torch_num_interop_threads",
        "torch_deterministic_algorithms",
        "omp_num_threads",
        "mkl_num_threads",
    )
    return {key: environment.get(key) for key in keys}


def _source_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing BENCH-015 source dependency: {relative}")
        manifest[relative] = _sha256_file(path)
    return manifest


def _write_source_archive(root: Path, outdir: Path) -> dict[str, Any]:
    archive_path = outdir / "executed_sources.tar"
    members: dict[str, str] = {}
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative in sorted(ARCHIVE_PATHS):
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"missing BENCH-015 archive dependency: {relative}")
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            payload = path.read_bytes()
            archive.addfile(info, io.BytesIO(payload))
            members[relative] = _sha256_bytes(payload)
    return {
        "path": archive_path.name,
        "sha256": _sha256_file(archive_path),
        "members": members,
    }


def _verify_source_archive(root: Path, outdir: Path, record: Mapping[str, Any]) -> bool:
    path = outdir / str(record["path"])
    if not path.is_file() or _sha256_file(path) != record["sha256"]:
        return False
    expected_members = set(ARCHIVE_PATHS)
    recorded_members = record.get("members")
    if not isinstance(recorded_members, Mapping) or set(recorded_members) != expected_members:
        return False
    observed: dict[str, str] = {}
    with tarfile.open(path, mode="r") as archive:
        members = archive.getmembers()
        if (
            len(members) != len(expected_members)
            or any(not member.isfile() for member in members)
            or {member.name for member in members} != expected_members
        ):
            return False
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                return False
            observed[member.name] = _sha256_bytes(handle.read())
    return observed == recorded_members and all(
        relative in observed and observed[relative] == _sha256_file(root / relative)
        for relative in ARCHIVE_PATHS
    )


def _binding_sha256(science: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json({"science": dict(science), "environment": _stable_environment(environment)})
    )


def _target_manifest(outdir: Path) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for target in TARGETS:
        records[target] = {
            "formula": TARGET_FORMULAS[target],
            "formula_sha256": _sha256_bytes(TARGET_FORMULAS[target].encode("utf-8")),
        }
        for dtype in (np.float64, np.float32):
            image = target_image(target, dtype=dtype)
            name = np.dtype(dtype).name
            arrays[f"{target}__{name}"] = image
            records[target][name] = _array_record(image)
    _deterministic_npz(outdir / "targets.npz", arrays)
    return {
        "path": "targets.npz",
        "file_sha256": _sha256_file(outdir / "targets.npz"),
        "targets": records,
    }


def _build_field_arrays(spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    count = int(spec["count"])
    seed = int(spec["seed"])
    source_target = str(spec["source_target"])
    init, structure = frozen_configs(count, seed)
    field = build_field(
        target_image(source_target, dtype=np.float32),
        init,
        structure,
        device="cpu",
    )
    if int(field.n) != count:
        raise RuntimeError(f"initializer returned {field.n} rows for requested {count}")
    if (
        field.opacities is not None
        or field.filter_variance is not None
        or field.color_grads is not None
    ):
        raise RuntimeError("BENCH-015 requires ordinary no-opacity, no-gradient fields")
    means = field.means.detach().cpu().contiguous().numpy().copy()
    field.colors = torch.from_numpy(
        target_values(source_target, means, dtype=np.float64).astype(np.float32)
    ).clone()
    return {
        "means": field.means.detach().cpu().contiguous().numpy().copy(),
        "log_scales": field.log_scales.detach().cpu().contiguous().numpy().copy(),
        "rotations": field.rotations.detach().cpu().contiguous().numpy().copy(),
        "colors": field.colors.detach().cpu().contiguous().numpy().copy(),
    }


def _build_field(
    spec: Mapping[str, Any], outdir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = _build_field_arrays(spec)
    relative = Path("fields") / f"{spec['field_id']}.npz"
    _deterministic_npz(outdir / relative, arrays)
    init, structure = frozen_configs(int(spec["count"]), int(spec["seed"]))
    records = {name: _array_record(value) for name, value in arrays.items()}
    row = {
        "schema": "bench015-field-v1",
        **dict(spec),
        "status": "ok",
        "path": relative.as_posix(),
        "file_sha256": _sha256_file(outdir / relative),
        "arrays": records,
        "arrays_sha256": _sha256_bytes(_canonical_json(records)),
        "init_config": asdict(init),
        "structure_tensor_config": asdict(structure),
    }
    return row, arrays


def _cell_stream_arrays(
    cell: Mapping[str, Any],
    arrays_by_field: Mapping[str, Mapping[str, np.ndarray]],
    arm: str,
) -> dict[str, np.ndarray]:
    field_id = str(cell["field80_id"] if arm == "nw80" else cell["field78_id"])
    source = arrays_by_field[field_id]
    # Colors are sampled at the persisted original continuous means before mean quantization.
    colors = target_values(
        str(cell["target"]), np.asarray(source["means"]), dtype=np.float64
    ).astype(np.float32)
    return {
        "means": np.ascontiguousarray(source["means"], dtype=np.float32),
        "log_scales": np.ascontiguousarray(source["log_scales"], dtype=np.float32),
        "rotations": np.ascontiguousarray(source["rotations"], dtype=np.float32),
        "colors": np.ascontiguousarray(colors, dtype=np.float32),
    }


def _field_from_decoded(decoded: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    arrays = decoded["arrays"]
    state = {
        name: torch.from_numpy(np.ascontiguousarray(arrays[name])).clone() for name in STREAM_ARRAYS
    }
    field = GaussianField(state["means"], state["log_scales"], state["rotations"], state["colors"])
    state["conics"] = field.conics(dilation=0.0)
    radii = field.radii(render_core.SIGMA_CUTOFF, dilation=0.0)
    return state, radii


def _render_decoded(
    decoded: Mapping[str, Any],
) -> tuple[render_core.CandidateResult, render_core.ReferenceResult, torch.Tensor, torch.Tensor]:
    state32, radii = _field_from_decoded(decoded)
    height = int(decoded["height"])
    width = int(decoded["width"])
    lift_state = decoded.get("lift_state")
    if int(decoded["mode"]) == 1 and lift_state is None:
        raise MethodUnavailableError(str(decoded.get("method_error")))
    if lift_state is None:
        beta64 = torch.zeros((3, 3), dtype=torch.float64)
        residual64 = state32["colors"].to(torch.float64)
    else:
        beta64 = lift_state.beta
        residual64 = lift_state.residual
    beta32 = beta64.to(torch.float32)
    residual32 = residual64.to(torch.float32)
    candidate = render_core.candidate_render_residual(
        state32["means"],
        state32["conics"],
        residual32,
        beta32[1:3],
        radii,
        height,
        width,
    )
    candidate = render_core.CandidateResult(
        **{**candidate.__dict__, "output": candidate.output + beta32[0]}
    )
    means64 = state32["means"].to(torch.float64)
    reference = render_core.reference_render_residual(
        means64,
        render_core.conics_from_parameters(
            state32["log_scales"].to(torch.float64),
            state32["rotations"].to(torch.float64),
        ),
        residual64,
        beta64[1:3],
        radii,
        height,
        width,
        contributors=candidate.contributors,
    )
    reference = render_core.ReferenceResult(
        **{**reference.__dict__, "output": reference.output + beta64[0]}
    )
    return candidate, reference, beta64, residual64


def _array_hash(value: np.ndarray | torch.Tensor) -> str:
    return _array_record(value)["record_sha256"]


def _contributor_hash(contributors: render_core.FrozenContributors) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "gid": _array_record(contributors.gid),
                "px": _array_record(contributors.px),
                "py": _array_record(contributors.py),
            }
        )
    )


def _edge_hash(edge_pairs: torch.Tensor) -> str:
    pairs = np.ascontiguousarray(edge_pairs.detach().cpu().numpy())
    # Directed coordinate pairs are sorted so row permutations have a canonical identity.
    order = np.lexsort(tuple(pairs[:, column] for column in reversed(range(4))))
    return _array_hash(pairs[order])


def _metrics(output: np.ndarray, target: np.ndarray, target_name: str) -> dict[str, Any]:
    output64 = np.asarray(output, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    error = output64 - target64
    masks = region_masks(target_name, height=target64.shape[0], width=target64.shape[1])
    lower = target64.reshape(-1, 3).min(axis=0)
    upper = target64.reshape(-1, 3).max(axis=0)
    excursion = np.maximum(np.maximum(lower - output64, output64 - upper), 0.0)
    result: dict[str, Any] = {
        "mse": float(np.mean(error**2)),
        "max_abs": float(np.max(np.abs(error))),
        "outer_ten_mse": float(np.mean(error[masks["outer_ten"]] ** 2)),
        "range_excursion": float(np.max(excursion)),
        "output_min": output64.reshape(-1, 3).min(axis=0).tolist(),
        "output_max": output64.reshape(-1, 3).max(axis=0).tolist(),
    }
    for name in ("edge", "away"):
        if name in masks:
            result[f"{name}_mse"] = float(np.mean(error[masks[name]] ** 2))
    return result


def _stream_record(payload: bytes, decoded: Mapping[str, Any]) -> dict[str, Any]:
    components = {
        "wrapper_header_bytes": STREAM_HEADER.size,
        "inner_bytes": len(decoded["inner"]),
        "tail_bytes": 0,
        "wrapper_crc_bytes": STREAM_CRC.size,
    }
    rebuilt = encode_stream(
        str(decoded["arm"]),
        decoded["arrays"],
        height=int(decoded["height"]),
        width=int(decoded["width"]),
    )
    return {
        "complete_bytes": len(payload),
        "complete_sha256": _sha256_bytes(payload),
        "components": components,
        "component_sum": sum(components.values()),
        "component_accounting_pass": sum(components.values()) == len(payload),
        "mode": int(decoded["mode"]),
        "inner_sha256": _sha256_bytes(decoded["inner"]),
        "inner": _json_safe(decoded["inner_components"]),
        "deterministic_encode_pass": rebuilt == payload,
        "ordinary_decoded_reencode_pass": rebuilt == payload,
    }


def _lift_record(state: core.LiftState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return {
        "beta": state.beta.tolist(),
        "beta_sha256": _array_hash(state.beta),
        "unconstrained_beta": state.unconstrained_beta.tolist(),
        "residual_sha256": _array_hash(state.residual),
        "alpha": state.confidence.alpha.tolist(),
        "score": state.confidence.score.tolist(),
        "color_scale": state.confidence.color_scale.tolist(),
        "neighbor_indices_sha256": _array_hash(state.confidence.neighbor_indices),
        "edge_coordinate_pairs_sha256": _edge_hash(state.confidence.edge_coordinate_pairs),
        "design_rank": state.robust_plane.diagnostics.rank,
        "design_condition": state.robust_plane.diagnostics.condition_number,
        "weighted_condition_numbers": (state.robust_plane.weighted_condition_numbers.tolist()),
        "error_median_row_indices": (state.robust_plane.error_median_row_indices.tolist()),
        "absolute_deviation_median_row_indices": (
            state.robust_plane.absolute_deviation_median_row_indices.tolist()
        ),
        "robust_scales": state.robust_plane.scales.tolist(),
        "robust_weight_min": float(state.robust_plane.weights.min()),
        "robust_weight_max": float(state.robust_plane.weights.max()),
        "iterations": state.robust_plane.iterations,
    }


def _exact_active_counts(
    decoded: Mapping[str, Any],
    candidate: render_core.CandidateResult,
    reference: render_core.ReferenceResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    state32, _ = _field_from_decoded(decoded)
    weights32 = render_core._contribution_weights(  # noqa: SLF001
        state32["means"], state32["conics"], candidate.contributors
    )
    candidate_counts = torch.zeros(HEIGHT * WIDTH, dtype=torch.int64).index_add(
        0,
        candidate.contributors.flat,
        (weights32 > 0.0).to(torch.int64),
    )
    reference_counts = torch.sum(reference.effective_weights > 0.0, dim=1).to(torch.int64)
    return candidate_counts, reference_counts


def _evaluate_static(
    *,
    cell: Mapping[str, Any],
    arm: str,
    payload: bytes,
    decoded: Mapping[str, Any],
    target: np.ndarray,
    outdir: Path,
    binding_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    candidate, reference, _, _ = _render_decoded(decoded)
    candidate_output = candidate.output.detach().cpu().numpy()
    reference_output = reference.output.detach().cpu().numpy()
    candidate_active, reference_active = _exact_active_counts(decoded, candidate, reference)
    parity = candidate_output.astype(np.float64) - reference_output
    target_name = str(cell["target"])
    lift_state = decoded.get("lift_state")
    diagnostic: dict[str, Any] | None = None
    diagnostic_raw: dict[str, np.ndarray] = {}
    if arm == "dsl78":
        weights = reference.effective_weights.reshape(HEIGHT * WIDTH, PROXY_COUNT)
        means64 = torch.from_numpy(
            np.ascontiguousarray(decoded["arrays"]["means"], dtype=np.float64)
        )
        colors64 = torch.from_numpy(
            np.ascontiguousarray(decoded["arrays"]["colors"], dtype=np.float64)
        )
        ols_beta, _ = core.ordinary_plane_qr(means64, colors64, HEIGHT, WIDTH)
        robust = core.robust_plane_irls(means64, colors64, HEIGHT, WIDTH)
        ols_output = core.dense_render_with_beta(
            weights, means64, colors64, ols_beta, HEIGHT, WIDTH
        )
        robust_output = core.dense_render_with_beta(
            weights, means64, colors64, robust.beta, HEIGHT, WIDTH
        )
        operator = core.dense_effective_operator_diagnostics(weights, means64, HEIGHT, WIDTH)
        diagnostic = {
            "ols": _metrics(ols_output.detach().numpy(), target, target_name),
            "ungated_cauchy": _metrics(robust_output.detach().numpy(), target, target_name),
            "operator": {
                "rank_w": operator.base_rank,
                "rank_a": operator.operator_rank,
                "common_rank_threshold": operator.common_rank_threshold,
                "reproduction_max_abs": operator.reproduction_max_abs,
                "minimum_coefficient": operator.minimum_coefficient,
                "maximum_row_l1": operator.maximum_row_l1,
            },
        }
        diagnostic_raw = {
            "ols_output_float64": ols_output.detach().numpy(),
            "ungated_cauchy_output_float64": robust_output.detach().numpy(),
        }
    renderer_id = f"{cell['cell_id']}__{arm}"
    cold_relative = Path("cold_states") / f"{renderer_id}.npz"
    cold_arrays = {name: np.ascontiguousarray(decoded["arrays"][name]) for name in STREAM_ARRAYS}
    _deterministic_npz(outdir / cold_relative, cold_arrays)
    raw_relative = Path("static_raw") / f"{renderer_id}.npz"
    raw = {
        "candidate_output_float32": candidate_output,
        "reference_output_float64": reference_output,
        "candidate_error_float64": candidate_output.astype(np.float64) - target,
        "reference_error_float64": reference_output - target,
        **diagnostic_raw,
    }
    _deterministic_npz(outdir / raw_relative, raw)
    forward_gates = {
        "finite": bool(
            np.isfinite(candidate_output).all()
            and np.isfinite(reference_output).all()
            and bool(candidate.finite_pass.all())
            and bool(reference.finite_pass.all())
        ),
        "minimum_active": (
            int(candidate_active.min()) >= MINIMUM_ACTIVE
            and int(reference_active.min()) >= MINIMUM_ACTIVE
        ),
        "minimum_mass": (
            float(candidate.mass.min()) >= MINIMUM_MASS
            and float(reference.mass.min()) >= MINIMUM_MASS
        ),
        "partition": (
            float(torch.max(torch.abs(candidate.partition - 1.0))) <= PARTITION_MAX_ABS
            and float(torch.max(torch.abs(reference.partition - 1.0))) <= PARTITION_MAX_ABS
        ),
        "nonnegative_effective_weights": (
            float(candidate.minimum_effective_weight.min()) >= MINIMUM_EFFECTIVE_WEIGHT
            and float(reference.minimum_effective_weight.min()) >= MINIMUM_EFFECTIVE_WEIGHT
        ),
        "candidate_reference": float(np.max(np.abs(parity))) <= CANDIDATE_REFERENCE_MAX_ABS,
        "component_accounting": True,
    }
    stream = _stream_record(payload, decoded)
    forward_gates["component_accounting"] = bool(
        stream["component_accounting_pass"]
        and stream["deterministic_encode_pass"]
        and stream["ordinary_decoded_reencode_pass"]
    )
    row = {
        "schema": "bench015-static-v1",
        "binding_sha256": binding_sha256,
        "status": "ok",
        "renderer_id": renderer_id,
        "cell_id": cell["cell_id"],
        "cohort": cell["cohort"],
        "target": target_name,
        "seed": int(cell["seed"]),
        "arm": arm,
        "count": int(decoded["count"]),
        "stream_path": f"streams/{renderer_id}.dslr",
        "stream": stream,
        "candidate": _metrics(candidate_output, target, target_name),
        "reference": _metrics(reference_output, target, target_name),
        "candidate_reference_max_abs": float(np.max(np.abs(parity))),
        "candidate_reference_rmse": float(np.sqrt(np.mean(parity**2))),
        "mass_min": float(candidate.mass.min()),
        "candidate_active_min": int(candidate_active.min()),
        "reference_active_min": int(reference_active.min()),
        "candidate_mass_min": float(candidate.mass.min()),
        "reference_mass_min": float(reference.mass.min()),
        "candidate_partition_max_abs": float(torch.max(torch.abs(candidate.partition - 1.0))),
        "reference_partition_max_abs": float(torch.max(torch.abs(reference.partition - 1.0))),
        "candidate_minimum_effective_weight": float(candidate.minimum_effective_weight.min()),
        "reference_minimum_effective_weight": float(reference.minimum_effective_weight.min()),
        "candidate_contributor_sha256": _contributor_hash(candidate.contributors),
        "reference_contributor_sha256": _contributor_hash(reference.contributors),
        "decoded_geometry_sha256": _sha256_bytes(
            _canonical_json(
                {
                    name: _array_record(decoded["arrays"][name])
                    for name in ("means", "log_scales", "rotations")
                }
            )
        ),
        "decoded_colors_sha256": _array_hash(decoded["arrays"]["colors"]),
        "cold_state_path": cold_relative.as_posix(),
        "cold_state_sha256": _sha256_file(outdir / cold_relative),
        "cold_state_records": {name: _array_record(value) for name, value in cold_arrays.items()},
        "lift": _lift_record(lift_state),
        "diagnostic": diagnostic,
        "raw_path": raw_relative.as_posix(),
        "raw_sha256": _sha256_file(outdir / raw_relative),
        "raw_records": {name: _array_record(value) for name, value in raw.items()},
        "forward_gates": forward_gates,
        "forward_pass": bool(all(forward_gates.values())),
    }
    return row, raw


def _static_method_unavailable(
    *,
    cell: Mapping[str, Any],
    arm: str,
    payload: bytes,
    decoded: Mapping[str, Any],
    outdir: Path,
    binding_sha256: str,
) -> dict[str, Any]:
    renderer_id = f"{cell['cell_id']}__{arm}"
    cold_relative = Path("cold_states") / f"{renderer_id}.npz"
    cold_arrays = {name: np.ascontiguousarray(decoded["arrays"][name]) for name in STREAM_ARRAYS}
    _deterministic_npz(outdir / cold_relative, cold_arrays)
    return {
        "schema": "bench015-static-v1",
        "binding_sha256": binding_sha256,
        "status": "method_unavailable",
        "renderer_id": renderer_id,
        "cell_id": cell["cell_id"],
        "cohort": cell["cohort"],
        "target": cell["target"],
        "seed": int(cell["seed"]),
        "arm": arm,
        "count": int(decoded["count"]),
        "stream_path": f"streams/{renderer_id}.dslr",
        "stream": _stream_record(payload, decoded),
        "method_error": decoded.get("method_error"),
        "scientific_failure": True,
        "candidate": None,
        "reference": None,
        "cold_state_path": cold_relative.as_posix(),
        "cold_state_sha256": _sha256_file(outdir / cold_relative),
        "cold_state_records": {name: _array_record(value) for name, value in cold_arrays.items()},
        "decoded_geometry_sha256": _sha256_bytes(
            _canonical_json(
                {
                    name: _array_record(decoded["arrays"][name])
                    for name in ("means", "log_scales", "rotations")
                }
            )
        ),
        "decoded_colors_sha256": _array_hash(decoded["arrays"]["colors"]),
        "candidate_contributor_sha256": None,
        "reference_contributor_sha256": None,
        "forward_gates": {"method_available": False},
        "forward_pass": False,
    }


def _not_run_row(
    *,
    schema: str,
    key_name: str,
    key_value: str,
    binding_sha256: str,
    reason: Mapping[str, Any] | str | None,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "binding_sha256": binding_sha256,
        "status": "not_run_due_to_method_unavailable",
        key_name: key_value,
        **metadata,
        "method_error": reason,
        "scientific_failure": True,
    }


def _replace_outer_crc(payload_without_crc: bytes) -> bytes:
    return payload_without_crc + STREAM_CRC.pack(zlib.crc32(payload_without_crc) & 0xFFFFFFFF)


def _control_field_arrays(count: int = PROXY_COUNT) -> dict[str, np.ndarray]:
    rows = np.arange(count)
    x = ((rows * 37) % WIDTH).astype(np.float32)
    y = ((rows * 29) % HEIGHT).astype(np.float32)
    # The coprime walks are unique as coordinate pairs over the frozen count.
    return {
        "means": np.stack((x, y), axis=1),
        "log_scales": np.stack(
            (
                np.full(count, math.log(2.4), dtype=np.float32),
                np.full(count, math.log(2.0), dtype=np.float32),
            ),
            axis=1,
        ),
        "rotations": np.ascontiguousarray((rows % 17) / 17.0 * math.pi, dtype=np.float32),
        "colors": np.stack(
            (
                0.25 + 0.45 * rows / max(count - 1, 1),
                0.65 - 0.25 * rows / max(count - 1, 1),
                0.35 + 0.20 * ((rows * 11) % count) / max(count - 1, 1),
            ),
            axis=1,
        ).astype(np.float32),
    }


def _expect_rejection(payload: bytes) -> bool:
    try:
        decode_stream(payload, "cpu")
    except (ValueError, RuntimeError, struct.error, zlib.error):
        return True
    return False


def _noncanonical_inner_wrapper(valid_blob: bytes) -> bytes:
    _, _, mode, _, inner_length, height, width = STREAM_HEADER.unpack_from(valid_blob)
    inner = valid_blob[STREAM_HEADER.size : STREAM_HEADER.size + inner_length]
    frame = struct.Struct("<QQI")
    offset = field_codec.HEADER_BYTES
    rebuilt = bytearray(inner[:offset])
    for _ in field_codec.STREAM_NAMES:
        raw_bytes, payload_bytes, raw_crc = frame.unpack_from(inner, offset)
        offset += frame.size
        compressed = inner[offset : offset + payload_bytes]
        offset += payload_bytes
        raw = zlib.decompress(compressed)
        if len(raw) != raw_bytes or (zlib.crc32(raw) & 0xFFFFFFFF) != raw_crc:
            raise RuntimeError("control inner stream unexpectedly failed its own frame")
        alternate = zlib.compress(raw, level=1)
        rebuilt.extend(frame.pack(raw_bytes, len(alternate), raw_crc))
        rebuilt.extend(alternate)
    if offset != len(inner):
        raise RuntimeError("control inner frame parser did not consume the stream")
    noncanonical = bytes(rebuilt)
    if noncanonical == inner:
        raise RuntimeError("failed to construct an alternate valid zlib representation")
    header = STREAM_HEADER.pack(
        STREAM_MAGIC,
        STREAM_VERSION,
        mode,
        0,
        len(noncanonical),
        height,
        width,
    )
    return _replace_outer_crc(header + noncanonical)


def run_plumbing_controls() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run target-free positive/negative controls suitable for pre-binding preflight."""

    ys = torch.linspace(0.0, HEIGHT - 1.0, 15, dtype=torch.float64)
    xs = torch.linspace(0.0, WIDTH - 1.0, 15, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    means = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    design = core.affine_design(means, HEIGHT, WIDTH)
    exact_beta = torch.tensor(
        ((0.30, 0.52, 0.68), (0.12, -0.08, 0.05), (-0.07, 0.09, 0.11)),
        dtype=torch.float64,
    )
    affine_colors = design @ exact_beta
    affine_state = core.derive_lift(means, affine_colors, HEIGHT, WIDTH)
    pixel_design = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float64)
    weights = torch.full(
        (HEIGHT * WIDTH, means.shape[0]),
        1.0 / means.shape[0],
        dtype=torch.float64,
    )
    affine_output, _ = core.dense_weight_render(
        weights, affine_colors, HEIGHT, WIDTH, lift=True, means=means
    )
    affine_truth = (pixel_design @ exact_beta).reshape(HEIGHT, WIDTH, 3)
    constant_colors = torch.tensor((0.30, 0.52, 0.68), dtype=torch.float64).repeat(
        means.shape[0], 1
    )
    constant_output, _ = core.dense_weight_render(
        weights, constant_colors, HEIGHT, WIDTH, lift=True, means=means
    )
    phase = 2.0 * math.pi * means[:, 0] / (WIDTH - 1)
    smooth_wave = 0.02 * torch.sin(phase) * torch.cos(math.pi * means[:, 1] / (HEIGHT - 1))
    smooth_colors = torch.stack(
        (0.5 + smooth_wave, 0.5 + 0.8 * smooth_wave, 0.5 - 0.6 * smooth_wave),
        dim=1,
    )
    smooth_state = core.derive_lift(means, smooth_colors, HEIGHT, WIDTH)
    side = (means[:, 0] >= (WIDTH - 1) / 2.0).to(torch.float64)
    step_colors = torch.stack(
        (0.20 + 0.55 * side, 0.60 - 0.14835011185682325 * side, 0.40 - 0.15 * side),
        dim=1,
    )
    step_state = core.derive_lift(means, step_colors, HEIGHT, WIDTH)

    tie_means = torch.tensor(
        (
            (53.0, 41.0),
            (52.0, 41.0),
            (53.0, 41.0 - 82.0 / 106.0),
            (53.0, 41.0 + 82.0 / 106.0),
            (54.0, 41.0),
            (0.0, 0.0),
            (106.0, 0.0),
            (0.0, 82.0),
            (106.0, 82.0),
        ),
        dtype=torch.float64,
    )
    tie_design = core.affine_design(tie_means, HEIGHT, WIDTH)
    tie_colors = (
        tie_design @ exact_beta
        + 0.001 * torch.arange(tie_means.shape[0], dtype=torch.float64)[:, None]
    )
    tie_state = core.derive_lift(tie_means, tie_colors, HEIGHT, WIDTH)
    tie_break_pass = tuple(tie_state.confidence.neighbor_indices[0].tolist()) == (
        1,
        2,
        3,
        4,
    )

    permutation = torch.arange(means.shape[0] - 1, -1, -1)
    permuted = core.derive_lift(means[permutation], smooth_colors[permutation], HEIGHT, WIDTH)
    permuted_output, _ = core.dense_weight_render(
        weights[:, permutation],
        smooth_colors[permutation],
        HEIGHT,
        WIDTH,
        lift=True,
        means=means[permutation],
    )
    smooth_output, _ = core.dense_weight_render(
        weights, smooth_colors, HEIGHT, WIDTH, lift=True, means=means
    )

    rejection: dict[str, bool] = {}
    for name, bad_means, bad_colors in (
        ("rank_deficient", means.clone(), smooth_colors),
        ("duplicate_means", means.clone(), smooth_colors),
        ("nonfinite", means.clone(), smooth_colors.clone()),
    ):
        if name == "rank_deficient":
            bad_means[:, 1] = 0.0
        elif name == "duplicate_means":
            bad_means[1] = bad_means[0]
        else:
            bad_colors[0, 0] = float("nan")
        try:
            core.derive_lift(bad_means, bad_colors, HEIGHT, WIDTH)
        except (ValueError, RuntimeError):
            rejection[name] = True
        else:
            rejection[name] = False

    arrays = _control_field_arrays()
    valid_blob = encode_stream("dsl78", arrays)
    malformed: dict[str, bytes] = {}
    malformed["magic"] = b"BADMAGIC" + valid_blob[8:]
    mutated = bytearray(valid_blob)
    mutated[8] = STREAM_VERSION + 1
    malformed["version"] = bytes(mutated)
    mutated = bytearray(valid_blob)
    mutated[9] = 7
    malformed["mode"] = bytes(mutated)
    mutated = bytearray(valid_blob)
    mutated[10] = 1
    malformed["reserved"] = bytes(mutated)
    mutated = bytearray(valid_blob)
    struct.pack_into("<I", mutated, 12, len(valid_blob))
    malformed["length"] = _replace_outer_crc(bytes(mutated[: -STREAM_CRC.size]))
    mutated = bytearray(valid_blob)
    mutated[-1] ^= 0x80
    malformed["crc"] = bytes(mutated)
    malformed["trailing"] = valid_blob + b"x"
    malformed["truncation"] = valid_blob[:-5]
    # A changed inner byte with a repaired outer CRC is still rejected by GFCOV01.
    mutated = bytearray(valid_blob[: -STREAM_CRC.size])
    mutated[STREAM_HEADER.size + 5] ^= 1
    malformed["inner"] = _replace_outer_crc(bytes(mutated))
    malformed["noncanonical_inner"] = _noncanonical_inner_wrapper(valid_blob)
    invalid_mode_count = bytearray(encode_stream("nw80", _control_field_arrays(PRIMARY_COUNT)))
    invalid_mode_count[9] = 1
    malformed["invalid_mode_count"] = _replace_outer_crc(
        bytes(invalid_mode_count[: -STREAM_CRC.size])
    )
    malformed_rejected = {name: _expect_rejection(value) for name, value in malformed.items()}

    decoded = decode_stream(valid_blob, "cpu")
    signature = inspect.signature(decode_stream)
    controls = {
        "schema": "bench015-controls-v1",
        "constant_max_abs": float(
            torch.max(torch.abs(constant_output - constant_colors[0])).item()
        ),
        "affine_max_abs": float(torch.max(torch.abs(affine_output - affine_truth)).item()),
        "affine_beta_max_abs": float(torch.max(torch.abs(affine_state.beta - exact_beta)).item()),
        "smooth_alpha": smooth_state.confidence.alpha.tolist(),
        "step_alpha": step_state.confidence.alpha.tolist(),
        "permutation_output_max_abs": float(
            torch.max(torch.abs(permuted_output - smooth_output)).item()
        ),
        "permutation_beta_max_abs": float(
            torch.max(torch.abs(permuted.beta - smooth_state.beta)).item()
        ),
        "permutation_alpha_max_abs": float(
            torch.max(torch.abs(permuted.confidence.alpha - smooth_state.confidence.alpha)).item()
        ),
        "permutation_edges_identical": (
            _edge_hash(permuted.confidence.edge_coordinate_pairs)
            == _edge_hash(smooth_state.confidence.edge_coordinate_pairs)
        ),
        "input_rejections": rejection,
        "distance_tie_break_pass": tie_break_pass,
        "malformed_stream_rejections": malformed_rejected,
        "valid_stream_roundtrip": (
            decoded["arm"] == "dsl78"
            and decoded["mode"] == 1
            and encode_stream("dsl78", decoded["arrays"]) == valid_blob
        ),
        "decoder_parameters": list(signature.parameters),
        "decoder_blob_device_only": (
            list(signature.parameters) == ["blob", "device"]
            and all(
                forbidden not in inspect.getsource(decode_stream)
                for forbidden in ("target", "source", "beta", "confidence")
            )
        ),
    }
    controls["gates"] = {
        "constant_reproduction": controls["constant_max_abs"] <= 1e-12,
        "affine_reproduction": controls["affine_max_abs"] <= 1e-10,
        "affine_beta": controls["affine_beta_max_abs"] <= 1e-10,
        "smooth_confidence": bool(
            np.isfinite(controls["smooth_alpha"]).all() and np.max(controls["smooth_alpha"]) > 0.0
        ),
        "step_fallback": min(controls["step_alpha"]) <= 0.01,
        "permutation": bool(
            controls["permutation_output_max_abs"] <= 1e-11
            and controls["permutation_beta_max_abs"] <= 1e-11
            and controls["permutation_alpha_max_abs"] <= 1e-12
            and controls["permutation_edges_identical"]
        ),
        "input_rejections": all(rejection.values()),
        "distance_tie_break": controls["distance_tie_break_pass"],
        "malformed_stream_rejections": all(malformed_rejected.values()),
        "stream_roundtrip": controls["valid_stream_roundtrip"],
        "decoder_api": controls["decoder_blob_device_only"],
    }
    controls["valid"] = bool(all(controls["gates"].values()))
    raw = {
        "control_means": means.numpy(),
        "control_affine_colors": affine_colors.numpy(),
        "control_smooth_colors": smooth_colors.numpy(),
        "control_step_colors": step_colors.numpy(),
        "control_smooth_output": smooth_output.numpy(),
        "valid_stream": np.frombuffer(valid_blob, dtype=np.uint8).copy(),
    }
    return controls, raw


def _basic_forward_gates(
    candidate: render_core.CandidateResult,
    reference: render_core.ReferenceResult,
    decoded: Mapping[str, Any],
) -> dict[str, bool]:
    parity = candidate.output.to(torch.float64) - reference.output
    candidate_active, reference_active = _exact_active_counts(decoded, candidate, reference)
    return {
        "finite": bool(candidate.finite_pass.all() and reference.finite_pass.all()),
        "minimum_active": (
            int(candidate_active.min()) >= MINIMUM_ACTIVE
            and int(reference_active.min()) >= MINIMUM_ACTIVE
        ),
        "minimum_mass": (
            float(candidate.mass.min()) >= MINIMUM_MASS
            and float(reference.mass.min()) >= MINIMUM_MASS
        ),
        "partition": (
            float(torch.max(torch.abs(candidate.partition - 1.0))) <= PARTITION_MAX_ABS
            and float(torch.max(torch.abs(reference.partition - 1.0))) <= PARTITION_MAX_ABS
        ),
        "nonnegative": (
            float(candidate.minimum_effective_weight.min()) >= MINIMUM_EFFECTIVE_WEIGHT
            and float(reference.minimum_effective_weight.min()) >= MINIMUM_EFFECTIVE_WEIGHT
        ),
        "parity": float(torch.max(torch.abs(parity))) <= CANDIDATE_REFERENCE_MAX_ABS,
    }


def _decoded_with_permutation(
    decoded: Mapping[str, Any], permutation: np.ndarray
) -> dict[str, Any]:
    arrays = {
        name: np.ascontiguousarray(np.asarray(decoded["arrays"][name])[permutation])
        for name in STREAM_ARRAYS
    }
    lift_state = None
    if int(decoded["mode"]) == 1:
        lift_state = core.derive_lift(
            torch.from_numpy(np.ascontiguousarray(arrays["means"], dtype=np.float64)),
            torch.from_numpy(np.ascontiguousarray(arrays["colors"], dtype=np.float64)),
            int(decoded["height"]),
            int(decoded["width"]),
        )
    return {
        **{key: decoded[key] for key in ("arm", "mode", "height", "width", "count")},
        "arrays": arrays,
        "lift_state": lift_state,
    }


def _permutation_schedule(static_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED))
    schedule: dict[str, Any] = {}
    for row in sorted(static_rows, key=lambda item: str(item["renderer_id"])):
        count = int(row["count"])
        schedule[str(row["renderer_id"])] = {
            "identity": list(range(count)),
            "reverse": list(range(count - 1, -1, -1)),
            "pcg64": rng.permutation(count).tolist(),
        }
    return {
        "schema": "bench015-permutation-schedule-v1",
        "seed": PERMUTATION_SEED,
        "orders": schedule,
    }


def _run_permutations(
    *,
    static_rows: Sequence[Mapping[str, Any]],
    decoded_by_renderer: Mapping[str, Mapping[str, Any]],
    outdir: Path,
    binding_sha256: str,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    schedule = _permutation_schedule(static_rows)
    _atomic_json(outdir / "permutation_schedule.json", schedule)
    rows: list[dict[str, Any]] = []
    static_by_id = {str(row["renderer_id"]): row for row in static_rows}
    with (outdir / "permutations.jsonl").open("wb") as handle:
        for renderer_id in sorted(schedule["orders"]):
            base_decoded = decoded_by_renderer[renderer_id]
            static = static_by_id[renderer_id]
            if static["status"] != "ok":
                for order_name in schedule["orders"][renderer_id]:
                    row = _not_run_row(
                        schema="bench015-permutation-v1",
                        key_name="permutation_id",
                        key_value=f"{renderer_id}__{order_name}",
                        binding_sha256=binding_sha256,
                        reason=static.get("method_error"),
                        renderer_id=renderer_id,
                        arm=static["arm"],
                        order=order_name,
                        permutation_pass=False,
                    )
                    _append_jsonl(handle, row)
                    rows.append(row)
                continue
            raw = _load_npz((outdir if raw_root is None else raw_root) / str(static["raw_path"]))
            identity_candidate = raw["candidate_output_float32"]
            identity_reference = raw["reference_output_float64"]
            means64 = torch.from_numpy(
                np.ascontiguousarray(base_decoded["arrays"]["means"], dtype=np.float64)
            )
            colors64 = torch.from_numpy(
                np.ascontiguousarray(base_decoded["arrays"]["colors"], dtype=np.float64)
            )
            try:
                identity_state = core.derive_lift(means64, colors64, HEIGHT, WIDTH)
            except (ValueError, RuntimeError) as error:
                reason = {"error_type": type(error).__name__, "error": str(error)}
                for order_name in schedule["orders"][renderer_id]:
                    row = _not_run_row(
                        schema="bench015-permutation-v1",
                        key_name="permutation_id",
                        key_value=f"{renderer_id}__{order_name}",
                        binding_sha256=binding_sha256,
                        reason=reason,
                        renderer_id=renderer_id,
                        arm=static["arm"],
                        order=order_name,
                        permutation_pass=False,
                    )
                    _append_jsonl(handle, row)
                    rows.append(row)
                continue
            for order_name, values in schedule["orders"][renderer_id].items():
                permutation = np.asarray(values, dtype=np.int64)
                try:
                    permuted = _decoded_with_permutation(base_decoded, permutation)
                except (ValueError, RuntimeError) as error:
                    row = _not_run_row(
                        schema="bench015-permutation-v1",
                        key_name="permutation_id",
                        key_value=f"{renderer_id}__{order_name}",
                        binding_sha256=binding_sha256,
                        reason={
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        renderer_id=renderer_id,
                        arm=static["arm"],
                        order=order_name,
                        permutation_pass=False,
                    )
                    _append_jsonl(handle, row)
                    rows.append(row)
                    continue
                candidate, reference, _, _ = _render_decoded(permuted)
                try:
                    state = core.derive_lift(
                        torch.from_numpy(
                            np.ascontiguousarray(permuted["arrays"]["means"], dtype=np.float64)
                        ),
                        torch.from_numpy(
                            np.ascontiguousarray(permuted["arrays"]["colors"], dtype=np.float64)
                        ),
                        HEIGHT,
                        WIDTH,
                    )
                except (ValueError, RuntimeError) as error:
                    row = _not_run_row(
                        schema="bench015-permutation-v1",
                        key_name="permutation_id",
                        key_value=f"{renderer_id}__{order_name}",
                        binding_sha256=binding_sha256,
                        reason={
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        renderer_id=renderer_id,
                        arm=static["arm"],
                        order=order_name,
                        permutation_pass=False,
                    )
                    _append_jsonl(handle, row)
                    rows.append(row)
                    continue
                candidate_delta = float(
                    np.max(
                        np.abs(
                            candidate.output.detach().numpy().astype(np.float64)
                            - identity_candidate.astype(np.float64)
                        )
                    )
                )
                reference_delta = float(
                    np.max(np.abs(reference.output.detach().numpy() - identity_reference))
                )
                beta_delta = float(torch.max(torch.abs(state.beta - identity_state.beta)).item())
                alpha_delta = float(
                    torch.max(
                        torch.abs(state.confidence.alpha - identity_state.confidence.alpha)
                    ).item()
                )
                edge_identical = _edge_hash(state.confidence.edge_coordinate_pairs) == _edge_hash(
                    identity_state.confidence.edge_coordinate_pairs
                )
                gates = _basic_forward_gates(candidate, reference, permuted)
                decision_identical = bool(all(gates.values())) == bool(static["forward_pass"])
                passed = bool(
                    candidate_delta <= PERMUTATION_CANDIDATE_MAX_ABS
                    and reference_delta <= PERMUTATION_REFERENCE_MAX_ABS
                    and beta_delta <= 1e-11
                    and alpha_delta <= 1e-12
                    and edge_identical
                    and decision_identical
                )
                row = {
                    "schema": "bench015-permutation-v1",
                    "binding_sha256": binding_sha256,
                    "status": "ok",
                    "permutation_id": f"{renderer_id}__{order_name}",
                    "renderer_id": renderer_id,
                    "arm": static["arm"],
                    "order": order_name,
                    "permutation_sha256": _array_hash(permutation),
                    "candidate_max_abs": candidate_delta,
                    "reference_max_abs": reference_delta,
                    "beta_max_abs": beta_delta,
                    "alpha_max_abs": alpha_delta,
                    "edge_pairs_identical": edge_identical,
                    "decision_identical": decision_identical,
                    "permutation_pass": passed,
                }
                _append_jsonl(handle, row)
                rows.append(row)
    return rows


def _dense_opt_output(
    arm: str,
    colors: torch.Tensor,
    means64: torch.Tensor,
    weights32: torch.Tensor,
    pixels32: torch.Tensor | None = None,
) -> torch.Tensor:
    if arm == "nw78_opt":
        return (weights32 @ colors).reshape(HEIGHT, WIDTH, 3)
    if arm != "dsl78_opt":
        raise ValueError(f"unknown convergence arm {arm!r}")
    state = core.derive_lift(means64, colors.to(torch.float64), HEIGHT, WIDTH)
    beta32 = state.beta.to(torch.float32)
    residual32 = state.residual.to(torch.float32)
    if pixels32 is None:
        pixels32 = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float32)
    return (pixels32 @ beta32 + weights32 @ residual32).reshape(HEIGHT, WIDTH, 3)


def _convergence_trajectory(
    *,
    cell: Mapping[str, Any],
    arm: str,
    means64: torch.Tensor,
    weights32: torch.Tensor,
    pixels32: torch.Tensor,
    target32: torch.Tensor,
    outdir: Path,
    binding_sha256: str,
) -> dict[str, Any]:
    if means64.dtype != torch.float64:
        raise ValueError("convergence means must be the common float64 tensor")
    for name, value in (
        ("weights", weights32),
        ("pixel design", pixels32),
        ("target", target32),
    ):
        if value.dtype != torch.float32:
            raise ValueError(f"convergence {name} must be a once-cast float32 tensor")
    colors = torch.zeros((PROXY_COUNT, 3), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam(
        [colors],
        lr=CONVERGENCE_LR,
        betas=CONVERGENCE_BETAS,
        eps=CONVERGENCE_EPS,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        fused=False,
    )
    losses: list[float] = []
    output_hashes: list[str] = []
    color_fit_step_time_ns: list[int] = []
    checkpoints: dict[str, np.ndarray] = {}
    for step in range(CONVERGENCE_STEPS + 1):
        step_start_ns = time.perf_counter_ns()
        try:
            output = _dense_opt_output(arm, colors, means64, weights32, pixels32=pixels32)
        except (ValueError, RuntimeError) as error:
            if arm != "dsl78_opt":
                raise
            return _not_run_row(
                schema="bench015-convergence-v1",
                key_name="trajectory_id",
                key_value=f"{cell['cell_id']}__{arm}",
                binding_sha256=binding_sha256,
                reason={"error_type": type(error).__name__, "error": str(error)},
                cell_id=cell["cell_id"],
                cohort=cell["cohort"],
                target=cell["target"],
                seed=int(cell["seed"]),
                arm=arm,
                failure_step=step,
            )
        loss = torch.mean(torch.square(output - target32))
        losses.append(float(loss.detach()))
        output_hashes.append(_array_hash(output))
        if step in CONVERGENCE_CHECKPOINTS:
            checkpoints[f"colors_step_{step:03d}"] = colors.detach().numpy().copy()
            checkpoints[f"output_step_{step:03d}"] = output.detach().numpy().copy()
        if step == CONVERGENCE_STEPS:
            color_fit_step_time_ns.append(time.perf_counter_ns() - step_start_ns)
            break
        optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
            optimizer.step()
        except (ValueError, RuntimeError) as error:
            if arm != "dsl78_opt":
                raise
            return _not_run_row(
                schema="bench015-convergence-v1",
                key_name="trajectory_id",
                key_value=f"{cell['cell_id']}__{arm}",
                binding_sha256=binding_sha256,
                reason={"error_type": type(error).__name__, "error": str(error)},
                cell_id=cell["cell_id"],
                cohort=cell["cohort"],
                target=cell["target"],
                seed=int(cell["seed"]),
                arm=arm,
                failure_step=step,
                failure_phase="backward_or_optimizer_step",
            )
        color_fit_step_time_ns.append(time.perf_counter_ns() - step_start_ns)
    trajectory_id = f"{cell['cell_id']}__{arm}"
    relative = Path("convergence") / f"{trajectory_id}.npz"
    _deterministic_npz(outdir / relative, checkpoints)
    return {
        "schema": "bench015-convergence-v1",
        "binding_sha256": binding_sha256,
        "status": "ok",
        "trajectory_id": trajectory_id,
        "cell_id": cell["cell_id"],
        "cohort": cell["cohort"],
        "target": cell["target"],
        "seed": int(cell["seed"]),
        "arm": arm,
        "steps": list(range(CONVERGENCE_STEPS + 1)),
        "losses": losses,
        "output_sha256": output_hashes,
        "initial_output_zero": bool(
            output_hashes[0] == _array_hash(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
        ),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "normalized_auc": float(np.trapezoid(losses, dx=1.0) / CONVERGENCE_STEPS),
        "color_fit_step_time_ns": color_fit_step_time_ns,
        "color_fit_step_median_ns": float(np.median(color_fit_step_time_ns)),
        "checkpoint_path": relative.as_posix(),
        "checkpoint_sha256": _sha256_file(outdir / relative),
        "checkpoint_records": {name: _array_record(value) for name, value in checkpoints.items()},
    }


def _loss64(
    colors: torch.Tensor,
    means64: torch.Tensor,
    weights64: torch.Tensor,
    target64: torch.Tensor,
) -> torch.Tensor:
    output, _ = core.dense_weight_render(
        weights64,
        colors,
        HEIGHT,
        WIDTH,
        lift=True,
        means=means64,
    )
    return torch.mean(torch.square(output - target64))


def _gradient_record(
    *,
    gradient_id: str,
    fixture: str,
    direction_index: int,
    means64: torch.Tensor,
    colors64: torch.Tensor,
    weights64: torch.Tensor,
    target64: torch.Tensor,
    direction64: torch.Tensor,
    binding_sha256: str,
) -> dict[str, Any]:
    colors_ad64 = colors64.detach().clone().requires_grad_(True)
    loss_ad64 = _loss64(colors_ad64, means64, weights64, target64)
    gradient64 = torch.autograd.grad(loss_ad64, colors_ad64)[0]
    ad64 = float(torch.sum(gradient64 * direction64).item())
    with torch.no_grad():
        positive = _loss64(
            colors64 + GRADIENT_STEP * direction64,
            means64,
            weights64,
            target64,
        )
        negative = _loss64(
            colors64 - GRADIENT_STEP * direction64,
            means64,
            weights64,
            target64,
        )
        fd64 = float(((positive - negative) / (2.0 * GRADIENT_STEP)).item())

    colors32 = colors64.to(torch.float32).detach().clone().requires_grad_(True)
    output32 = _dense_opt_output("dsl78_opt", colors32, means64, weights64.to(torch.float32))
    loss32 = torch.mean(torch.square(output32 - target64.to(torch.float32)))
    gradient32 = torch.autograd.grad(loss32, colors32)[0]
    ad32 = float(torch.sum(gradient32 * direction64.to(torch.float32)).item())
    ad_fd_tolerance = 1e-8 + 2e-4 * max(abs(ad64), abs(fd64))
    ad32_tolerance = 2e-5 + 3e-3 * abs(ad64)
    state = core.derive_lift(means64, colors64, HEIGHT, WIDTH)
    colors_alpha = colors64.detach().clone().requires_grad_(True)
    alpha_state = core.derive_lift(means64, colors_alpha, HEIGHT, WIDTH)
    ad_alpha_values: list[float] = []
    for channel in range(3):
        alpha_gradient = torch.autograd.grad(
            alpha_state.confidence.alpha[channel],
            colors_alpha,
            retain_graph=channel < 2,
        )[0]
        ad_alpha_values.append(float(torch.sum(alpha_gradient * direction64).item()))
    with torch.no_grad():
        alpha_positive = core.derive_lift(
            means64,
            colors64 + GRADIENT_STEP * direction64,
            HEIGHT,
            WIDTH,
        ).confidence.alpha
        alpha_negative = core.derive_lift(
            means64,
            colors64 - GRADIENT_STEP * direction64,
            HEIGHT,
            WIDTH,
        ).confidence.alpha
        fd_alpha_tensor = (alpha_positive - alpha_negative) / (2.0 * GRADIENT_STEP)
    fd_alpha_values = fd_alpha_tensor.tolist()
    alpha_tolerances = [
        1e-8 + 2e-4 * max(abs(ad), abs(fd))
        for ad, fd in zip(ad_alpha_values, fd_alpha_values, strict=True)
    ]
    alpha_derivative_pass = all(
        abs(ad - fd) <= tolerance
        for ad, fd, tolerance in zip(
            ad_alpha_values, fd_alpha_values, alpha_tolerances, strict=True
        )
    )
    alpha_derivative_active = max(map(abs, ad_alpha_values)) >= 1e-4
    trace_base = core.robust_plane_order_statistic_trace(means64, colors64, HEIGHT, WIDTH)
    trace_positive = core.robust_plane_order_statistic_trace(
        means64, colors64 + GRADIENT_STEP * direction64, HEIGHT, WIDTH
    )
    trace_negative = core.robust_plane_order_statistic_trace(
        means64, colors64 - GRADIENT_STEP * direction64, HEIGHT, WIDTH
    )
    trace_arrays = {
        "base_error": trace_base.error_median_row_indices,
        "base_absolute_deviation": trace_base.absolute_deviation_median_row_indices,
        "positive_error": trace_positive.error_median_row_indices,
        "positive_absolute_deviation": (trace_positive.absolute_deviation_median_row_indices),
        "negative_error": trace_negative.error_median_row_indices,
        "negative_absolute_deviation": (trace_negative.absolute_deviation_median_row_indices),
    }
    changes = {
        "positive_error": int(
            torch.sum(trace_arrays["positive_error"] != trace_arrays["base_error"]).item()
        ),
        "positive_absolute_deviation": int(
            torch.sum(
                trace_arrays["positive_absolute_deviation"]
                != trace_arrays["base_absolute_deviation"]
            ).item()
        ),
        "negative_error": int(
            torch.sum(trace_arrays["negative_error"] != trace_arrays["base_error"]).item()
        ),
        "negative_absolute_deviation": int(
            torch.sum(
                trace_arrays["negative_absolute_deviation"]
                != trace_arrays["base_absolute_deviation"]
            ).item()
        ),
    }
    return {
        "schema": "bench015-gradient-v1",
        "binding_sha256": binding_sha256,
        "status": "ok",
        "gradient_id": gradient_id,
        "fixture": fixture,
        "direction_index": direction_index,
        "direction_sha256": _array_hash(direction64),
        "ad64": ad64,
        "fd64": fd64,
        "ad32": ad32,
        "ad64_fd64_abs": abs(ad64 - fd64),
        "ad32_ad64_abs": abs(ad32 - ad64),
        "ad64_fd64_tolerance": ad_fd_tolerance,
        "ad32_ad64_tolerance": ad32_tolerance,
        "alpha": state.confidence.alpha.tolist(),
        "mid_transition": bool(
            torch.any((state.confidence.alpha > 0.1) & (state.confidence.alpha < 0.9))
        ),
        "order_statistic_rows": {name: value.tolist() for name, value in trace_arrays.items()},
        "order_statistic_row_sha256": {
            name: _array_hash(value) for name, value in trace_arrays.items()
        },
        "order_statistic_identity_changes": changes,
        "any_order_statistic_identity_change": any(changes.values()),
        "alpha_directional_ad64": ad_alpha_values,
        "alpha_directional_fd64": fd_alpha_values,
        "alpha_directional_tolerances": alpha_tolerances,
        "alpha_directional_pass": alpha_derivative_pass,
        "alpha_directional_active": alpha_derivative_active,
        "gradient_pass": bool(
            math.isfinite(ad64)
            and math.isfinite(fd64)
            and math.isfinite(ad32)
            and abs(ad64 - fd64) <= ad_fd_tolerance
            and abs(ad32 - ad64) <= ad32_tolerance
        ),
    }


def _run_gradients(
    *,
    decoded_by_renderer: Mapping[str, Mapping[str, Any]],
    outdir: Path,
    binding_sha256: str,
) -> list[dict[str, Any]]:
    frozen_directions = _gradient_direction_arrays()
    fixtures: list[
        tuple[
            str,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            Mapping[str, Any] | None,
        ]
    ] = []
    for target in ("affine_chirp", "affine_occlusion"):
        renderer_id = f"{_cell_id('target_conditioned', target, 401)}__dsl78"
        decoded = decoded_by_renderer[renderer_id]
        if not decoded["method_available"]:
            fixtures.append((renderer_id, None, None, None, None, decoded.get("method_error")))
            continue
        _, reference, _, _ = _render_decoded(decoded)
        fixtures.append(
            (
                renderer_id,
                torch.from_numpy(
                    np.ascontiguousarray(decoded["arrays"]["means"], dtype=np.float64)
                ),
                torch.from_numpy(
                    np.ascontiguousarray(decoded["arrays"]["colors"], dtype=np.float64)
                ),
                reference.effective_weights.reshape(HEIGHT * WIDTH, PROXY_COUNT),
                torch.from_numpy(target_image(target, dtype=np.float64)),
                None,
            )
        )

    synthetic = _synthetic_fixture_arrays()
    means = torch.from_numpy(synthetic["synthetic_means"])
    colors = torch.from_numpy(synthetic["synthetic_colors"])
    weights = torch.from_numpy(synthetic["synthetic_weights"])
    fixtures.append(
        (
            "synthetic_mid_transition",
            means,
            colors,
            weights,
            torch.from_numpy(target_image("affine_chirp", dtype=np.float64)),
            None,
        )
    )

    direction_arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    with (outdir / "gradients.jsonl").open("wb") as handle:
        for fixture, means64, colors64, weights64, target64, unavailable in fixtures:
            for direction_index in range(2):
                gradient_id = f"{fixture}__d{direction_index}"
                direction64 = torch.from_numpy(frozen_directions[gradient_id])
                direction_arrays[gradient_id] = direction64.numpy()
                if unavailable is not None:
                    row = _not_run_row(
                        schema="bench015-gradient-v1",
                        key_name="gradient_id",
                        key_value=gradient_id,
                        binding_sha256=binding_sha256,
                        reason=unavailable,
                        fixture=fixture,
                        direction_index=direction_index,
                        direction_sha256=_array_hash(direction64),
                        gradient_pass=False,
                    )
                else:
                    assert (
                        means64 is not None
                        and colors64 is not None
                        and weights64 is not None
                        and target64 is not None
                    )
                    try:
                        row = _gradient_record(
                            gradient_id=gradient_id,
                            fixture=fixture,
                            direction_index=direction_index,
                            means64=means64,
                            colors64=colors64,
                            weights64=weights64,
                            target64=target64,
                            direction64=direction64,
                            binding_sha256=binding_sha256,
                        )
                    except (ValueError, RuntimeError) as error:
                        row = _not_run_row(
                            schema="bench015-gradient-v1",
                            key_name="gradient_id",
                            key_value=gradient_id,
                            binding_sha256=binding_sha256,
                            reason={
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                            fixture=fixture,
                            direction_index=direction_index,
                            direction_sha256=_array_hash(direction64),
                            gradient_pass=False,
                        )
                if fixture == "synthetic_mid_transition" and row["status"] == "ok":
                    row["gradient_pass"] = bool(
                        row["gradient_pass"]
                        and row["mid_transition"]
                        and row["alpha_directional_pass"]
                        and row["alpha_directional_active"]
                    )
                _append_jsonl(handle, row)
                rows.append(row)
    _deterministic_npz(outdir / "gradient_directions.npz", direction_arrays)
    _atomic_json(
        outdir / "gradient_direction_manifest.json",
        {
            "schema": "bench015-gradient-directions-v1",
            "seed": GRADIENT_SEED,
            "path": "gradient_directions.npz",
            "file_sha256": _sha256_file(outdir / "gradient_directions.npz"),
            "records": {name: _array_record(value) for name, value in direction_arrays.items()},
        },
    )
    return rows


def _prepare_timing_state(
    decoded: Mapping[str, Any],
    *,
    collect_operation_count: bool = True,
) -> dict[str, Any]:
    state32, radii = _field_from_decoded(decoded)
    contributor_evaluations = None
    if collect_operation_count:
        contributor_evaluations = int(
            render_core.enumerate_contributors(
                state32["means"],
                radii,
                int(decoded["height"]),
                int(decoded["width"]),
            ).gid.numel()
        )
    lift_state = decoded.get("lift_state")
    if int(decoded["mode"]) == 1 and lift_state is None:
        raise MethodUnavailableError(str(decoded.get("method_error")))
    if lift_state is None:
        return {
            "means": state32["means"],
            "conics": state32["conics"],
            "colors": state32["colors"],
            "radii": radii,
            "height": int(decoded["height"]),
            "width": int(decoded["width"]),
            "beta": None,
            "xn": None,
            "yn": None,
            "gaussian_pixel_weight_evaluations": contributor_evaluations,
        }
    height = int(decoded["height"])
    width = int(decoded["width"])
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return {
        "means": state32["means"],
        "conics": state32["conics"],
        "colors": lift_state.residual.to(torch.float32),
        "radii": radii,
        "height": height,
        "width": width,
        "beta": lift_state.beta.to(torch.float32),
        "xn": 2.0 * xx / float(width - 1) - 1.0,
        "yn": 2.0 * yy / float(height - 1) - 1.0,
        "gaussian_pixel_weight_evaluations": contributor_evaluations,
    }


def _candidate_output_prepared(prepared: Mapping[str, Any]) -> torch.Tensor:
    base = production_render(
        prepared["means"],
        prepared["conics"],
        prepared["colors"],
        prepared["radii"],
        int(prepared["height"]),
        int(prepared["width"]),
        support_fade=False,
        sigma_cutoff=render_core.SIGMA_CUTOFF,
    )
    beta32 = prepared["beta"]
    if beta32 is None:
        return base
    # Exactly six spatial multiplications and nine additions per pixel: two slope products and
    # two plane sums per RGB channel, followed by one add into the normalized residual.
    plane = (
        beta32[0][None, None, :]
        + prepared["xn"][:, :, None] * beta32[1][None, None, :]
        + prepared["yn"][:, :, None] * beta32[2][None, None, :]
    )
    return base + plane


def _candidate_output_only(decoded: Mapping[str, Any]) -> torch.Tensor:
    return _candidate_output_prepared(_prepare_timing_state(decoded, collect_operation_count=False))


def _time_call(callable_: Callable[[], Any]) -> int:
    with torch.no_grad():
        start = time.perf_counter_ns()
        result = callable_()
        # Force tensor work before stopping even if a future backend becomes asynchronous.
        if isinstance(result, torch.Tensor):
            _ = float(result.reshape(-1)[0])
        return time.perf_counter_ns() - start


def _timing_schedule(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(TIMING_SEED))
    schedules: dict[str, Any] = {}
    for cell in cells:
        schedules[str(cell["cell_id"])] = {
            phase: [rng.permutation(ARMS).tolist() for _ in range(rounds)]
            for phase, rounds in (
                ("warmup_render", TIMING_WARMUPS),
                ("record_render", TIMING_REPETITIONS),
                ("warmup_cold", TIMING_WARMUPS),
                ("record_cold", TIMING_REPETITIONS),
            )
        }
    return {
        "schema": "bench015-timing-schedule-v1",
        "seed": TIMING_SEED,
        "warmups": TIMING_WARMUPS,
        "repetitions": TIMING_REPETITIONS,
        "cells": schedules,
    }


def _run_timing(
    *,
    cells: Sequence[Mapping[str, Any]],
    streams_by_cell: Mapping[str, Mapping[str, bytes]],
    decoded_by_renderer: Mapping[str, Mapping[str, Any]],
    outdir: Path,
    binding_sha256: str,
) -> list[dict[str, Any]]:
    schedule = _timing_schedule(cells)
    _atomic_json(outdir / "timing_schedule.json", schedule)
    prepared_by_renderer: dict[str, dict[str, Any] | None] = {}
    method_errors: dict[str, Any] = {}
    for renderer_id, decoded in decoded_by_renderer.items():
        if int(decoded["mode"]) == 1 and not decoded["method_available"]:
            prepared_by_renderer[renderer_id] = None
            method_errors[renderer_id] = decoded.get("method_error")
        else:
            prepared_by_renderer[renderer_id] = _prepare_timing_state(decoded)
    runtime_unavailable = set(method_errors)
    samples: dict[tuple[str, str], dict[str, list[int]]] = {
        (str(cell["cell_id"]), arm): {"render": [], "cold": [], "derive": []}
        for cell in cells
        for arm in ARMS
    }
    for cell in cells:
        cell_id = str(cell["cell_id"])
        for phase in ("warmup_render", "record_render"):
            recorded = phase.startswith("record")
            for order in schedule["cells"][cell_id][phase]:
                for arm in order:
                    prepared = prepared_by_renderer[f"{cell_id}__{arm}"]
                    if prepared is None:
                        continue
                    elapsed = _time_call(
                        lambda prepared=prepared: _candidate_output_prepared(prepared)
                    )
                    if recorded:
                        samples[(cell_id, arm)]["render"].append(elapsed)
        for phase in ("warmup_cold", "record_cold"):
            recorded = phase.startswith("record")
            for order in schedule["cells"][cell_id][phase]:
                for arm in order:
                    blob = streams_by_cell[cell_id][arm]
                    renderer_id = f"{cell_id}__{arm}"
                    if renderer_id in runtime_unavailable:
                        continue

                    def cold(blob: bytes = blob) -> torch.Tensor:
                        decoded = _decode_stream_impl(
                            blob,
                            "cpu",
                            audit_reencode=False,
                            collect_metadata=False,
                        )
                        return _candidate_output_only(decoded)

                    try:
                        elapsed = _time_call(cold)
                    except MethodUnavailableError as error:
                        runtime_unavailable.add(renderer_id)
                        method_errors[renderer_id] = {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        samples[(cell_id, arm)] = {
                            "render": [],
                            "cold": [],
                            "derive": [],
                        }
                        continue
                    if recorded:
                        samples[(cell_id, arm)]["cold"].append(elapsed)
        # Persist decoder derivation separately; it is not part of render-only time.
        for arm in ARMS:
            decoded = decoded_by_renderer[f"{cell_id}__{arm}"]
            renderer_id = f"{cell_id}__{arm}"
            if renderer_id in runtime_unavailable:
                continue
            means64 = torch.from_numpy(
                np.ascontiguousarray(decoded["arrays"]["means"], dtype=np.float64)
            )
            colors64 = torch.from_numpy(
                np.ascontiguousarray(decoded["arrays"]["colors"], dtype=np.float64)
            )
            for _ in range(TIMING_REPETITIONS):
                if arm == "dsl78":
                    try:
                        elapsed = _time_call(
                            lambda: core.derive_lift(means64, colors64, HEIGHT, WIDTH)
                        )
                    except (ValueError, RuntimeError) as error:
                        runtime_unavailable.add(renderer_id)
                        method_errors[renderer_id] = {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        samples[(cell_id, arm)] = {
                            "render": [],
                            "cold": [],
                            "derive": [],
                        }
                        break
                else:
                    elapsed = 0
                samples[(cell_id, arm)]["derive"].append(elapsed)

    rows: list[dict[str, Any]] = []
    with (outdir / "timing.jsonl").open("wb") as handle:
        for cell in cells:
            cell_id = str(cell["cell_id"])
            for arm in ARMS:
                values = samples[(cell_id, arm)]
                renderer_id = f"{cell_id}__{arm}"
                if renderer_id in runtime_unavailable:
                    row = _not_run_row(
                        schema="bench015-timing-v1",
                        key_name="timing_id",
                        key_value=renderer_id,
                        binding_sha256=binding_sha256,
                        reason=method_errors[renderer_id],
                        cell_id=cell_id,
                        cohort=cell["cohort"],
                        target=cell["target"],
                        seed=int(cell["seed"]),
                        arm=arm,
                    )
                    _append_jsonl(handle, row)
                    rows.append(row)
                    continue
                row = {
                    "schema": "bench015-timing-v1",
                    "binding_sha256": binding_sha256,
                    "status": "ok",
                    "timing_id": f"{cell_id}__{arm}",
                    "cell_id": cell_id,
                    "cohort": cell["cohort"],
                    "target": cell["target"],
                    "seed": int(cell["seed"]),
                    "arm": arm,
                    "render_ns": values["render"],
                    "cold_decode_derive_render_ns": values["cold"],
                    "derive_ns": values["derive"],
                    "render_median_ns": float(np.median(values["render"])),
                    "cold_median_ns": float(np.median(values["cold"])),
                    "derive_median_ns": float(np.median(values["derive"])),
                    "operation_ledger": {
                        "gaussian_pixel_weight_evaluations": int(
                            prepared_by_renderer[renderer_id]["gaussian_pixel_weight_evaluations"]
                        ),
                        "accumulated_scalar_channels": 4,
                        "plane_multiplications_per_pixel": 6 if arm == "dsl78" else 0,
                        "plane_additions_per_pixel": 9 if arm == "dsl78" else 0,
                        "per_pixel_solve_or_factorization": False,
                    },
                }
                _append_jsonl(handle, row)
                rows.append(row)
    _atomic_json(
        outdir / "performance_environment.json",
        {
            "binding_sha256": binding_sha256,
            "torch_threads": torch.get_num_threads(),
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "timed_cells": len(cells),
            "all_frozen_cells_timed": (len(cells) == EXPECTED_CELLS and not runtime_unavailable),
            "method_unavailable_timing_rows": sorted(runtime_unavailable),
        },
    )
    return rows


def _linear_median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.5, method="linear"))


def analyze_results(
    *,
    static_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    _metric_key: str = "reference",
    _compute_shadow: bool = True,
) -> dict[str, Any]:
    """Apply only the frozen conjunctive gates to persisted reference evidence."""

    terminal_rows = [
        {
            "ledger": ledger,
            "id": row.get(key),
            "status": row.get("status"),
            "method_error": row.get("method_error"),
        }
        for ledger, rows, key in (
            ("static", static_rows, "renderer_id"),
            ("permutation", permutation_rows, "permutation_id"),
            ("convergence", convergence_rows, "trajectory_id"),
            ("gradient", gradient_rows, "gradient_id"),
            ("timing", timing_rows, "timing_id"),
        )
        for row in rows
        if row.get("status") != "ok"
    ]
    if terminal_rows:
        return {
            "schema": "bench015-analysis-v1",
            "complete": True,
            "decision": "kill",
            "classification": "scientific_kill_method_unavailable",
            "stage1_authorized": False,
            "method_unavailable": True,
            "terminal_rows": terminal_rows,
            "section_pass": {"method_availability": False},
            "f32_shadow_agreement": None,
            "decision_uses_unclipped_float64_reference": True,
        }

    static = {str(row["renderer_id"]): row for row in static_rows}
    convergence = {str(row["trajectory_id"]): row for row in convergence_rows}
    timing = {str(row["timing_id"]): row for row in timing_rows}
    cells = cell_specs()

    pair_records: dict[str, Any] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        nw = static[f"{cell_id}__nw78"]
        dsl = static[f"{cell_id}__dsl78"]
        nw80 = static[f"{cell_id}__nw80"]
        nw_mse = float(nw[_metric_key]["mse"])
        dsl_mse = float(dsl[_metric_key]["mse"])
        nw80_mse = float(nw80[_metric_key]["mse"])
        pair_records[cell_id] = {
            "cell_id": cell_id,
            "cohort": cell["cohort"],
            "target": cell["target"],
            "seed": int(cell["seed"]),
            "dsl_nw_mse_ratio": dsl_mse / max(nw_mse, 1e-20),
            "dsl_nw80_mse_ratio": dsl_mse / max(nw80_mse, 1e-20),
            "dsl_nw_outer_ratio": float(dsl[_metric_key]["outer_ten_mse"])
            / max(float(nw[_metric_key]["outer_ten_mse"]), 1e-20),
            "strict_win_nw": dsl_mse < nw_mse,
            "strict_win_nw80": dsl_mse < nw80_mse,
            "complete_bytes": {
                arm: int(static[f"{cell_id}__{arm}"]["stream"]["complete_bytes"]) for arm in ARMS
            },
            "pair_pass": bool(
                nw["forward_pass"]
                and dsl["forward_pass"]
                and nw80["forward_pass"]
                and int(nw["stream"]["complete_bytes"]) == int(dsl["stream"]["complete_bytes"])
                and nw["stream"]["inner_sha256"] == dsl["stream"]["inner_sha256"]
                and nw["decoded_geometry_sha256"] == dsl["decoded_geometry_sha256"]
                and nw["decoded_colors_sha256"] == dsl["decoded_colors_sha256"]
                and nw["candidate_contributor_sha256"] == dsl["candidate_contributor_sha256"]
                and nw["reference_contributor_sha256"] == dsl["reference_contributor_sha256"]
                and int(nw["stream"]["mode"]) == 0
                and int(dsl["stream"]["mode"]) == 1
                and int(nw80["stream"]["mode"]) == 0
            ),
        }

    quality: dict[str, Any] = {}
    constant_values = [
        float(static[f"{cell['cell_id']}__dsl78"][_metric_key]["max_abs"])
        for cell in cells
        if cell["target"] == "constant"
    ]
    affine_values = [
        float(static[f"{cell['cell_id']}__dsl78"][_metric_key]["max_abs"])
        for cell in cells
        if cell["target"] == "affine"
    ]
    quality["constant"] = {
        "values": constant_values,
        "pass": len(constant_values) == 6 and max(constant_values) <= 1e-6,
    }
    quality["affine"] = {
        "values": affine_values,
        "pass": len(affine_values) == 6 and max(affine_values) <= 1e-3,
    }
    for target in SMOOTH_TARGETS:
        records = [value for value in pair_records.values() if value["target"] == target]
        ratios = [float(value["dsl_nw_mse_ratio"]) for value in records]
        outer = [float(value["dsl_nw_outer_ratio"]) for value in records]
        control = [float(value["dsl_nw80_mse_ratio"]) for value in records]
        quality[target] = {
            "wins_nw": sum(bool(value["strict_win_nw"]) for value in records),
            "median_nw_ratio": _linear_median(ratios),
            "median_outer_ratio": _linear_median(outer),
            "wins_nw80": sum(bool(value["strict_win_nw80"]) for value in records),
            "median_nw80_ratio": _linear_median(control),
        }
        quality[target]["pass"] = bool(
            quality[target]["wins_nw"] >= 5
            and quality[target]["median_nw_ratio"] <= 0.90
            and quality[target]["median_outer_ratio"] <= 0.80
            and quality[target]["wins_nw80"] >= 4
            and quality[target]["median_nw80_ratio"] <= 0.98
        )

    quartic = [value for value in pair_records.values() if value["target"] == "zero_linear_quartic"]
    quality["zero_linear_quartic"] = {
        "ratios": [float(value["dsl_nw_mse_ratio"]) for value in quartic],
        "pass": len(quartic) == 6
        and all(float(value["dsl_nw_mse_ratio"]) <= 1.02 for value in quartic),
    }

    occlusion_records: list[dict[str, float]] = []
    hard_records: dict[str, list[dict[str, float]]] = {
        "continuous_crease": [],
        "isoluminant_step": [],
    }
    for cell in cells:
        target = str(cell["target"])
        if target not in ("affine_occlusion", *hard_records):
            continue
        cell_id = str(cell["cell_id"])
        nw = static[f"{cell_id}__nw78"][_metric_key]
        dsl = static[f"{cell_id}__dsl78"][_metric_key]
        record = {
            "full_ratio": float(dsl["mse"]) / max(float(nw["mse"]), 1e-20),
            "edge_ratio": float(dsl["edge_mse"]) / max(float(nw["edge_mse"]), 1e-20),
            "range_excursion": float(dsl["range_excursion"]),
        }
        if target == "affine_occlusion":
            record["away_ratio"] = float(dsl["away_mse"]) / max(float(nw["away_mse"]), 1e-20)
            occlusion_records.append(record)
        else:
            hard_records[target].append(record)
    quality["affine_occlusion"] = {
        "records": occlusion_records,
        "median_away_ratio": _linear_median([value["away_ratio"] for value in occlusion_records]),
    }
    quality["affine_occlusion"]["pass"] = bool(
        len(occlusion_records) == 6
        and all(value["away_ratio"] < 1.0 for value in occlusion_records)
        and quality["affine_occlusion"]["median_away_ratio"] <= 0.90
        and all(value["edge_ratio"] <= 1.01 for value in occlusion_records)
        and all(value["full_ratio"] <= 1.01 for value in occlusion_records)
        and all(value["range_excursion"] <= RANGE_MARGIN for value in occlusion_records)
    )
    for target, records in hard_records.items():
        quality[target] = {
            "records": records,
            "pass": len(records) == 6
            and all(value["full_ratio"] <= 1.01 for value in records)
            and all(value["edge_ratio"] <= 1.02 for value in records)
            and all(value["range_excursion"] <= RANGE_MARGIN for value in records),
        }

    dsl_rows = [row for row in static_rows if row["arm"] == "dsl78"]
    global_range_predicates = {
        str(row["renderer_id"]): bool(
            min(map(float, row[_metric_key]["output_min"])) >= -RANGE_MARGIN
            and max(map(float, row[_metric_key]["output_max"])) <= 1.0 + RANGE_MARGIN
        )
        for row in dsl_rows
    }
    global_range_pass = all(global_range_predicates.values())
    static_pair_pass = len(pair_records) == EXPECTED_CELLS and all(
        value["pair_pass"] for value in pair_records.values()
    )

    rate: dict[str, Any] = {}
    for target in TARGETS:
        records = [value for value in pair_records.values() if value["target"] == target]
        ratios = [
            value["complete_bytes"]["dsl78"] / value["complete_bytes"]["nw80"] for value in records
        ]
        rate[target] = {
            "wins": sum(
                value["complete_bytes"]["dsl78"] <= value["complete_bytes"]["nw80"]
                for value in records
            ),
            "median_ratio": _linear_median(ratios),
            "worst_ratio": max(ratios),
        }
        rate[target]["pass"] = bool(
            rate[target]["wins"] >= 5
            and rate[target]["median_ratio"] <= 1.00
            and rate[target]["worst_ratio"] <= 1.02
        )

    convergence_pairs: dict[str, Any] = {}
    for cell in cells:
        cell_id = str(cell["cell_id"])
        nw = convergence[f"{cell_id}__nw78_opt"]
        dsl = convergence[f"{cell_id}__dsl78_opt"]
        convergence_pairs[cell_id] = {
            "target": cell["target"],
            "auc_ratio": float(dsl["normalized_auc"]) / max(float(nw["normalized_auc"]), 1e-20),
            "final_ratio": float(dsl["final_loss"]) / max(float(nw["final_loss"]), 1e-20),
            "shared_zero_start": bool(
                nw["initial_output_zero"]
                and dsl["initial_output_zero"]
                and nw["output_sha256"][0] == dsl["output_sha256"][0]
                and nw["initial_loss"] == dsl["initial_loss"]
            ),
        }
    smooth_convergence = [
        value for value in convergence_pairs.values() if value["target"] in SMOOTH_TARGETS
    ]
    hard_convergence = [
        value for value in convergence_pairs.values() if value["target"] in MIXED_HARD_TARGETS
    ]
    convergence_gate = {
        "all_shared_zero_start": all(
            value["shared_zero_start"] for value in convergence_pairs.values()
        ),
        "smooth_median_auc_ratio": _linear_median(
            [value["auc_ratio"] for value in smooth_convergence]
        ),
        "smooth_median_final_ratio": _linear_median(
            [value["final_ratio"] for value in smooth_convergence]
        ),
        "smooth_worst_final_ratio": max(value["final_ratio"] for value in smooth_convergence),
        "hard_worst_final_ratio": max(value["final_ratio"] for value in hard_convergence),
    }
    convergence_gate["pass"] = bool(
        len(convergence_pairs) == EXPECTED_CELLS
        and convergence_gate["all_shared_zero_start"]
        and convergence_gate["smooth_median_auc_ratio"] <= 0.90
        and convergence_gate["smooth_median_final_ratio"] <= 0.90
        and convergence_gate["smooth_worst_final_ratio"] <= 1.05
        and convergence_gate["hard_worst_final_ratio"] <= 1.02
    )

    render_ratios: list[float] = []
    cold_ratios: list[float] = []
    timing_operation_pass = True
    for cell in cells:
        cell_id = str(cell["cell_id"])
        nw = timing[f"{cell_id}__nw78"]
        dsl = timing[f"{cell_id}__dsl78"]
        render_ratios.append(float(dsl["render_median_ns"]) / float(nw["render_median_ns"]))
        cold_ratios.append(float(dsl["cold_median_ns"]) / float(nw["cold_median_ns"]))
        nw_ledger = nw["operation_ledger"]
        ledger = dsl["operation_ledger"]
        timing_operation_pass = bool(
            timing_operation_pass
            and int(ledger["gaussian_pixel_weight_evaluations"]) > 0
            and ledger["gaussian_pixel_weight_evaluations"]
            == nw_ledger["gaussian_pixel_weight_evaluations"]
            and nw_ledger["accumulated_scalar_channels"] == 4
            and nw_ledger["plane_multiplications_per_pixel"] == 0
            and nw_ledger["plane_additions_per_pixel"] == 0
            and not nw_ledger["per_pixel_solve_or_factorization"]
            and ledger["accumulated_scalar_channels"] == 4
            and ledger["plane_multiplications_per_pixel"] == 6
            and ledger["plane_additions_per_pixel"] == 9
            and not ledger["per_pixel_solve_or_factorization"]
        )
    performance = {
        "timed_units": len(render_ratios),
        "median_render_ratio": _linear_median(render_ratios),
        "median_cold_ratio": _linear_median(cold_ratios),
        "operation_ledger_pass": timing_operation_pass,
    }
    performance["pass"] = bool(
        len(render_ratios) == EXPECTED_TIMING_UNITS
        and performance["median_render_ratio"] <= 1.15
        and performance["median_cold_ratio"] <= 1.50
        and performance["operation_ledger_pass"]
    )

    section_pass = {
        "static_numerical_and_pair": static_pair_pass,
        "constant_affine": bool(quality["constant"]["pass"] and quality["affine"]["pass"]),
        "smooth_quality": all(quality[target]["pass"] for target in SMOOTH_TARGETS),
        "no_harm": bool(
            quality["zero_linear_quartic"]["pass"]
            and quality["affine_occlusion"]["pass"]
            and quality["continuous_crease"]["pass"]
            and quality["isoluminant_step"]["pass"]
            and global_range_pass
        ),
        "rate": all(rate[target]["pass"] for target in TARGETS),
        "permutation": len(permutation_rows) == EXPECTED_PERMUTATIONS
        and all(row["permutation_pass"] for row in permutation_rows),
        "convergence": bool(convergence_gate["pass"]),
        "gradient": len(gradient_rows) == EXPECTED_GRADIENTS
        and all(row["gradient_pass"] for row in gradient_rows),
        "performance": bool(performance["pass"]),
    }
    passed = bool(all(section_pass.values()))
    result = {
        "schema": "bench015-analysis-v1",
        "complete": True,
        "decision": "pass" if passed else "kill",
        "classification": (
            "synthetic_stage0_full_pass" if passed else "scientific_kill_frozen_global_lift"
        ),
        "stage1_authorized": passed,
        "decision_uses_unclipped_float64_reference": True,
        "strict_wins": True,
        "linear_medians": True,
        "section_pass": section_pass,
        "quality": quality,
        "global_range_pass": global_range_pass,
        "global_range_predicates": global_range_predicates,
        "rate": rate,
        "convergence": convergence_gate,
        "performance": performance,
        "pairs": pair_records,
    }
    if not _compute_shadow:
        return result
    shadow = analyze_results(
        static_rows=static_rows,
        permutation_rows=permutation_rows,
        convergence_rows=convergence_rows,
        gradient_rows=gradient_rows,
        timing_rows=timing_rows,
        _metric_key="candidate",
        _compute_shadow=False,
    )
    primary_signature = _static_decision_signature(result)
    shadow_signature = _static_decision_signature(shadow)
    agreement = _nested_close(primary_signature, shadow_signature)
    result["static_decision_ledgers"] = {
        "float64_reference_primary": {
            "quality": result["quality"],
            "global_range_pass": result["global_range_pass"],
            "global_range_predicates": result["global_range_predicates"],
            "pairs": result["pairs"],
            "signature": primary_signature,
        },
        "float32_candidate_shadow": {
            "quality": shadow["quality"],
            "global_range_pass": shadow["global_range_pass"],
            "global_range_predicates": shadow["global_range_predicates"],
            "pairs": shadow["pairs"],
            "signature": shadow_signature,
        },
    }
    result["f32_shadow_agreement"] = agreement
    result["section_pass"]["f32_shadow_agreement"] = agreement
    if not agreement:
        result.update(
            {
                "complete": True,
                "decision": "kill",
                "classification": "scientific_kill_float32_shadow_decision_disagreement",
                "stage1_authorized": False,
            }
        )
    return result


def _static_decision_signature(analysis: Mapping[str, Any]) -> dict[str, Any]:
    quality = analysis["quality"]
    signature: dict[str, Any] = {
        "constant_predicates": [value <= 1e-6 for value in quality["constant"]["values"]],
        "affine_predicates": [value <= 1e-3 for value in quality["affine"]["values"]],
        "global_range_pass": analysis["global_range_pass"],
        "global_gamut_cells": analysis["global_range_predicates"],
        "static_sections": {
            key: analysis["section_pass"][key]
            for key in (
                "static_numerical_and_pair",
                "constant_affine",
                "smooth_quality",
                "no_harm",
                "rate",
            )
        },
        "strict_wins": {
            cell_id: {
                "nw": pair["strict_win_nw"],
                "nw80": pair["strict_win_nw80"],
            }
            for cell_id, pair in analysis["pairs"].items()
        },
    }
    signature["final_static_decision"] = all(signature["static_sections"].values())
    for target in SMOOTH_TARGETS:
        record = quality[target]
        signature[target] = {
            "wins_nw": record["wins_nw"],
            "median_nw": record["median_nw_ratio"] <= 0.90,
            "median_outer": record["median_outer_ratio"] <= 0.80,
            "wins_nw80": record["wins_nw80"],
            "median_nw80": record["median_nw80_ratio"] <= 0.98,
            "pass": record["pass"],
        }
    signature["zero_linear_quartic"] = {
        "cells": [value <= 1.02 for value in quality["zero_linear_quartic"]["ratios"]],
        "pass": quality["zero_linear_quartic"]["pass"],
    }
    signature["affine_occlusion"] = {
        "cells": [
            {
                "away_strict": value["away_ratio"] < 1.0,
                "edge": value["edge_ratio"] <= 1.01,
                "full": value["full_ratio"] <= 1.01,
                "range": value["range_excursion"] <= RANGE_MARGIN,
            }
            for value in quality["affine_occlusion"]["records"]
        ],
        "median_away": quality["affine_occlusion"]["median_away_ratio"] <= 0.90,
        "pass": quality["affine_occlusion"]["pass"],
    }
    for target in ("continuous_crease", "isoluminant_step"):
        signature[target] = {
            "cells": [
                {
                    "full": value["full_ratio"] <= 1.01,
                    "edge": value["edge_ratio"] <= 1.02,
                    "range": value["range_excursion"] <= RANGE_MARGIN,
                }
                for value in quality[target]["records"]
            ],
            "pass": quality[target]["pass"],
        }
    return signature


def _nested_close(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_nested_close(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _nested_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, bool) or isinstance(right, bool):
            return left == right
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return left == right
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-14)
    return left == right


def _json_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_json_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _ledger_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    expected: Sequence[str],
    binding_sha256: str,
    terminal_statuses: tuple[str, ...] = ("ok",),
) -> bool:
    identifiers = [str(row.get(key)) for row in rows]
    return bool(
        len(rows) == len(expected)
        and len(set(identifiers)) == len(identifiers)
        and sorted(identifiers) == sorted(expected)
        and all(row.get("binding_sha256") == binding_sha256 for row in rows)
        and all(row.get("status") in terminal_statuses for row in rows)
        and all(_json_finite(row) for row in rows)
    )


def _artifact_manifest(outdir: Path) -> dict[str, Any]:
    excluded = {
        "artifact_manifest.json",
        "replay.json",
        "analysis.json",
        "completion.json",
    }
    files = {
        path.relative_to(outdir).as_posix(): {
            "nbytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(outdir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    return {
        "schema": "bench015-artifact-manifest-v1",
        "files": files,
        "file_count": len(files),
        "payload_sha256": _sha256_bytes(_canonical_json(files)),
    }


def _validate_artifact_manifest(outdir: Path, manifest: Mapping[str, Any]) -> bool:
    return _nested_close(_artifact_manifest(outdir), manifest)


def _expected_keys() -> dict[str, list[str]]:
    cells = cell_specs()
    static = [f"{cell['cell_id']}__{arm}" for cell in cells for arm in ARMS]
    permutation = [
        f"{renderer_id}__{order}"
        for renderer_id in static
        for order in ("identity", "reverse", "pcg64")
    ]
    convergence = [
        f"{cell['cell_id']}__{arm}" for cell in cells for arm in ("nw78_opt", "dsl78_opt")
    ]
    gradient = [
        f"{fixture}__d{direction}"
        for fixture in (
            f"{_cell_id('target_conditioned', 'affine_chirp', 401)}__dsl78",
            f"{_cell_id('target_conditioned', 'affine_occlusion', 401)}__dsl78",
            "synthetic_mid_transition",
        )
        for direction in range(2)
    ]
    return {
        "fields": [str(spec["field_id"]) for spec in field_specs()],
        "static": static,
        "permutations": permutation,
        "convergence": convergence,
        "gradients": gradient,
        "timing": static,
    }


def _row_without(row: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in keys}


def _timing_ledger_valid(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    decoded_by_renderer: Mapping[str, Mapping[str, Any]],
) -> bool:
    expected_schedule = _timing_schedule(cells)
    expected_metadata = {
        f"{cell['cell_id']}__{arm}": {
            "cell_id": cell["cell_id"],
            "cohort": cell["cohort"],
            "target": cell["target"],
            "seed": int(cell["seed"]),
            "arm": arm,
        }
        for cell in cells
        for arm in ARMS
    }
    for row in rows:
        renderer_id = str(row.get("timing_id"))
        decoded = decoded_by_renderer.get(renderer_id)
        metadata = expected_metadata.get(renderer_id)
        if (
            decoded is None
            or metadata is None
            or any(row.get(name) != expected for name, expected in metadata.items())
        ):
            return False
        if row.get("status") == "not_run_due_to_method_unavailable":
            if metadata["arm"] != "dsl78" or decoded.get("method_available"):
                return False
            continue
        render_samples = row.get("render_ns", [])
        cold_samples = row.get("cold_decode_derive_render_ns", [])
        derive_samples = row.get("derive_ns", [])
        operation_ledger = row.get("operation_ledger", {})
        state32, radii = _field_from_decoded(decoded)
        expected_evaluations = int(
            render_core.enumerate_contributors(
                state32["means"],
                radii,
                int(decoded["height"]),
                int(decoded["width"]),
            ).gid.numel()
        )
        expected_operations = {
            "gaussian_pixel_weight_evaluations": expected_evaluations,
            "accumulated_scalar_channels": 4,
            "plane_multiplications_per_pixel": 6 if metadata["arm"] == "dsl78" else 0,
            "plane_additions_per_pixel": 9 if metadata["arm"] == "dsl78" else 0,
            "per_pixel_solve_or_factorization": False,
        }
        if not (
            len(render_samples) == TIMING_REPETITIONS
            and len(cold_samples) == TIMING_REPETITIONS
            and len(derive_samples) == TIMING_REPETITIONS
            and all(isinstance(value, int) and value > 0 for value in render_samples)
            and all(isinstance(value, int) and value > 0 for value in cold_samples)
            and all(isinstance(value, int) and value >= 0 for value in derive_samples)
            and float(row["render_median_ns"]) == float(np.median(render_samples))
            and float(row["cold_median_ns"]) == float(np.median(cold_samples))
            and float(row["derive_median_ns"]) == float(np.median(derive_samples))
            and operation_ledger == expected_operations
            and expected_evaluations > 0
        ):
            return False
    return len(expected_schedule["cells"]) == EXPECTED_CELLS


def replay_artifacts(outdir: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read-only independent replay of a BENCH-015 artifact."""

    _pin_runtime()
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((outdir / "environment.json").read_text(encoding="utf-8"))
    binding = str(config["binding_sha256"])
    science = dict(config)
    science.pop("binding_sha256")
    keys = _expected_keys()
    checks: dict[str, bool] = {}
    checks["binding"] = _binding_sha256(science, environment) == binding
    checks["source_manifest"] = bool(
        config.get("source_manifest") == _source_manifest(root)
        and config["task_sha256"] == _sha256_file(root / TASK_PATH)
    )
    archive_record = json.loads((outdir / "source_archive.json").read_text(encoding="utf-8"))
    checks["source_archive"] = _verify_source_archive(root, outdir, archive_record)
    manifest = json.loads((outdir / "artifact_manifest.json").read_text(encoding="utf-8"))
    checks["artifact_manifest"] = _validate_artifact_manifest(outdir, manifest)

    prebinding_manifest = json.loads(
        (outdir / "prebinding_arrays.json").read_text(encoding="utf-8")
    )
    observed_prebinding = _load_npz(outdir / str(prebinding_manifest["path"]))
    rebuilt_prebinding = _prebinding_arrays()
    rebuilt_prebinding_records = _prebinding_records(rebuilt_prebinding)
    checks["prebinding_arrays"] = bool(
        prebinding_manifest.get("binding_sha256") == binding
        and prebinding_manifest.get("persisted_before_controls_targets_and_fields")
        and prebinding_manifest.get("records") == config.get("prebinding_arrays")
        and _nested_close(prebinding_manifest.get("records"), rebuilt_prebinding_records)
        and _sha256_file(outdir / str(prebinding_manifest["path"]))
        == prebinding_manifest["file_sha256"]
        and set(observed_prebinding) == set(rebuilt_prebinding)
        and all(
            np.array_equal(observed_prebinding[name], rebuilt_prebinding[name])
            for name in rebuilt_prebinding
        )
    )

    controls_stored = json.loads((outdir / "controls.json").read_text(encoding="utf-8"))
    controls_rebuilt, control_arrays = run_plumbing_controls()
    controls_comparable = _row_without(
        controls_stored, "binding_sha256", "path", "file_sha256", "raw_records"
    )
    stored_control_raw = _load_npz(outdir / str(controls_stored["path"]))
    checks["controls"] = bool(
        controls_stored.get("binding_sha256") == binding
        and controls_stored.get("valid")
        and _nested_close(controls_comparable, controls_rebuilt)
        and _sha256_file(outdir / str(controls_stored["path"])) == controls_stored["file_sha256"]
        and set(stored_control_raw) == set(control_arrays)
        and set(controls_stored.get("raw_records", {})) == set(control_arrays)
        and all(
            np.array_equal(stored_control_raw[name], control_arrays[name])
            and controls_stored["raw_records"][name] == _array_record(control_arrays[name])
            for name in control_arrays
        )
    )

    target_manifest = json.loads((outdir / "target_manifest.json").read_text(encoding="utf-8"))
    target_arrays = _load_npz(outdir / str(target_manifest["path"]))
    expected_target_keys = {
        f"{target}__{np.dtype(dtype).name}"
        for target in TARGETS
        for dtype in (np.float64, np.float32)
    }
    target_check = bool(
        set(target_manifest) == {"path", "file_sha256", "targets"}
        and target_manifest["path"] == "targets.npz"
        and _sha256_file(outdir / str(target_manifest["path"])) == target_manifest["file_sha256"]
        and set(target_manifest["targets"]) == set(TARGETS)
        and set(target_arrays) == expected_target_keys
    )
    for target in TARGETS:
        target_check = bool(
            target_check
            and set(target_manifest["targets"][target])
            == {"formula", "formula_sha256", "float64", "float32"}
            and target_manifest["targets"][target]["formula"] == TARGET_FORMULAS[target]
            and target_manifest["targets"][target]["formula_sha256"]
            == _sha256_bytes(TARGET_FORMULAS[target].encode("utf-8"))
        )
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            key = f"{target}__{name}"
            rebuilt = target_image(target, dtype=dtype)
            target_check = bool(
                target_check
                and key in target_arrays
                and np.array_equal(target_arrays[key], rebuilt)
                and _nested_close(target_manifest["targets"][target][name], _array_record(rebuilt))
            )
    checks["targets"] = target_check

    field_rows = _read_jsonl(outdir / "fields.jsonl")
    checks["field_ledger"] = _ledger_contract(
        field_rows,
        key="field_id",
        expected=keys["fields"],
        binding_sha256=binding,
    )
    arrays_by_field: dict[str, dict[str, np.ndarray]] = {}
    fields_valid = checks["field_ledger"]
    field_by_id = {str(row["field_id"]): row for row in field_rows}
    with tempfile.TemporaryDirectory(prefix="bench015-field-replay-") as temporary:
        temp_root = Path(temporary)
        for spec in field_specs():
            field_id = str(spec["field_id"])
            observed = field_by_id[field_id]
            arrays = _load_npz(outdir / str(observed["path"]))
            arrays_by_field[field_id] = arrays
            expected, rebuilt_arrays = _build_field(spec, temp_root)
            expected["binding_sha256"] = binding
            fields_valid = bool(
                fields_valid
                and _sha256_file(outdir / str(observed["path"])) == observed["file_sha256"]
                and _nested_close(observed, expected)
                and set(arrays) == set(rebuilt_arrays)
                and all(np.array_equal(arrays[name], rebuilt_arrays[name]) for name in arrays)
            )
    checks["fields_rebuilt"] = fields_valid

    static_rows = _read_jsonl(outdir / "static.jsonl")
    stream_rows = _read_jsonl(outdir / "streams.jsonl")
    checks["static_ledger"] = _ledger_contract(
        static_rows,
        key="renderer_id",
        expected=keys["static"],
        binding_sha256=binding,
        terminal_statuses=("ok", "method_unavailable"),
    )
    checks["stream_ledger"] = _ledger_contract(
        stream_rows,
        key="renderer_id",
        expected=keys["static"],
        binding_sha256=binding,
    )
    stream_by_id = {str(row["renderer_id"]): row for row in stream_rows}
    static_by_id = {str(row["renderer_id"]): row for row in static_rows}
    decoded_by_renderer: dict[str, dict[str, Any]] = {}
    streams_by_cell: dict[str, dict[str, bytes]] = {}
    streams_valid = checks["static_ledger"]
    static_valid = checks["static_ledger"]
    static_artifacts_valid = checks["static_ledger"]
    with tempfile.TemporaryDirectory(prefix="bench015-static-replay-") as temporary:
        temp_root = Path(temporary)
        for cell in cell_specs():
            cell_id = str(cell["cell_id"])
            streams_by_cell[cell_id] = {}
            target = target_image(str(cell["target"]), dtype=np.float64)
            for arm in ARMS:
                renderer_id = f"{cell_id}__{arm}"
                observed = static_by_id[renderer_id]
                path = outdir / str(observed["stream_path"])
                payload = path.read_bytes()
                arrays = _cell_stream_arrays(cell, arrays_by_field, arm)
                rebuilt_payload = encode_stream(arm, arrays)
                decoded = decode_stream(payload, "cpu")
                decoded_by_renderer[renderer_id] = decoded
                streams_by_cell[cell_id][arm] = payload
                streams_valid = bool(
                    streams_valid
                    and payload == rebuilt_payload
                    and _sha256_bytes(payload) == observed["stream"]["complete_sha256"]
                    and decoded["arm"] == arm
                    and stream_by_id[renderer_id]["path"] == observed["stream_path"]
                    and _nested_close(stream_by_id[renderer_id]["record"], observed["stream"])
                )
                if arm == "dsl78" and not decoded["method_available"]:
                    expected = _static_method_unavailable(
                        cell=cell,
                        arm=arm,
                        payload=payload,
                        decoded=decoded,
                        outdir=temp_root,
                        binding_sha256=binding,
                    )
                    expected_raw = None
                else:
                    expected, expected_raw = _evaluate_static(
                        cell=cell,
                        arm=arm,
                        payload=payload,
                        decoded=decoded,
                        target=target,
                        outdir=temp_root,
                        binding_sha256=binding,
                    )
                expected_stream_row = {
                    "schema": "bench015-stream-v1",
                    "binding_sha256": binding,
                    "status": "ok",
                    "renderer_id": renderer_id,
                    "cell_id": cell_id,
                    "arm": arm,
                    "path": expected["stream_path"],
                    "record": expected["stream"],
                }
                streams_valid = bool(
                    streams_valid and _nested_close(stream_by_id[renderer_id], expected_stream_row)
                )
                cold_arrays = {
                    name: np.ascontiguousarray(decoded["arrays"][name]) for name in STREAM_ARRAYS
                }
                static_artifacts_valid = bool(
                    static_artifacts_valid
                    and _persisted_npz_matches(
                        outdir / str(observed["cold_state_path"]),
                        cold_arrays,
                        str(observed["cold_state_sha256"]),
                        observed["cold_state_records"],
                    )
                )
                if expected_raw is not None:
                    static_artifacts_valid = bool(
                        static_artifacts_valid
                        and _persisted_npz_matches(
                            outdir / str(observed["raw_path"]),
                            expected_raw,
                            str(observed["raw_sha256"]),
                            observed["raw_records"],
                        )
                    )
                if arm in ("nw78", "dsl78"):
                    pair_gates = observed.get("same_stream_pair_gates", {})
                    row_match = _nested_close(
                        _row_without(observed, "same_stream_pair_gates", "forward_pass"),
                        _row_without(expected, "forward_pass"),
                    ) and observed.get("forward_pass") == bool(
                        expected.get("forward_pass") and all(pair_gates.values())
                    )
                else:
                    row_match = _nested_close(observed, expected)
                static_valid = bool(static_valid and row_match)
    checks["streams_reencoded"] = streams_valid
    checks["static_recomputed"] = static_valid
    checks["static_artifacts"] = static_artifacts_valid
    pair_gates_valid = True
    for cell in cell_specs():
        cell_id = str(cell["cell_id"])
        nw = static_by_id[f"{cell_id}__nw78"]
        dsl = static_by_id[f"{cell_id}__dsl78"]
        expected_pair = {
            "dsl_method_available": dsl["status"] == "ok",
            "complete_lengths_equal": (
                nw["stream"]["complete_bytes"] == dsl["stream"]["complete_bytes"]
            ),
            "inner_stream_identical": (
                nw["stream"]["inner_sha256"] == dsl["stream"]["inner_sha256"]
            ),
            "decoded_geometry_identical": (
                nw["decoded_geometry_sha256"] == dsl["decoded_geometry_sha256"]
            ),
            "decoded_colors_identical": (
                nw["decoded_colors_sha256"] == dsl["decoded_colors_sha256"]
            ),
            "candidate_contributors_identical": bool(
                dsl["status"] == "ok"
                and nw["candidate_contributor_sha256"] == dsl["candidate_contributor_sha256"]
            ),
            "reference_contributors_identical": bool(
                dsl["status"] == "ok"
                and nw["reference_contributor_sha256"] == dsl["reference_contributor_sha256"]
            ),
        }
        pair_gates_valid = bool(
            pair_gates_valid
            and _nested_close(nw.get("same_stream_pair_gates"), expected_pair)
            and _nested_close(dsl.get("same_stream_pair_gates"), expected_pair)
        )
    checks["same_stream_pair_gates"] = pair_gates_valid
    checks["attribution_inventory"] = bool(
        all(
            (
                row.get("diagnostic") is not None
                and "ols_output_float64" in row["raw_records"]
                and "ungated_cauchy_output_float64" in row["raw_records"]
            )
            if row["arm"] == "dsl78" and row["status"] == "ok"
            else row.get("diagnostic") is None
            if row["arm"] == "dsl78"
            else row.get("diagnostic") is None
            for row in static_rows
        )
    )

    permutation_rows = _read_jsonl(outdir / "permutations.jsonl")
    checks["permutation_ledger"] = _ledger_contract(
        permutation_rows,
        key="permutation_id",
        expected=keys["permutations"],
        binding_sha256=binding,
        terminal_statuses=("ok", "not_run_due_to_method_unavailable"),
    )
    with tempfile.TemporaryDirectory(prefix="bench015-permutation-replay-") as temporary:
        rebuilt_permutations = _run_permutations(
            static_rows=static_rows,
            decoded_by_renderer=decoded_by_renderer,
            outdir=Path(temporary),
            binding_sha256=binding,
            raw_root=outdir,
        )
    stored_schedule = json.loads((outdir / "permutation_schedule.json").read_text(encoding="utf-8"))
    checks["permutations_recomputed"] = bool(
        _nested_close(stored_schedule, _permutation_schedule(static_rows))
        and _nested_close(permutation_rows, rebuilt_permutations)
    )

    convergence_rows = _read_jsonl(outdir / "convergence.jsonl")
    checks["convergence_ledger"] = _ledger_contract(
        convergence_rows,
        key="trajectory_id",
        expected=keys["convergence"],
        binding_sha256=binding,
        terminal_statuses=("ok", "not_run_due_to_method_unavailable"),
    )
    checks["convergence_inventory"] = bool(
        all(
            row["status"] == "not_run_due_to_method_unavailable"
            or (
                len(row.get("steps", [])) == CONVERGENCE_STEPS + 1
                and len(row.get("losses", [])) == CONVERGENCE_STEPS + 1
                and len(row.get("output_sha256", [])) == CONVERGENCE_STEPS + 1
                and len(row.get("checkpoint_records", {})) == 2 * len(CONVERGENCE_CHECKPOINTS)
                and len(row.get("color_fit_step_time_ns", [])) == CONVERGENCE_STEPS + 1
                and all(
                    isinstance(value, int) and value > 0
                    for value in row.get("color_fit_step_time_ns", [])
                )
                and float(row.get("color_fit_step_median_ns"))
                == float(np.median(row["color_fit_step_time_ns"]))
            )
            for row in convergence_rows
        )
    )
    rebuilt_convergence: list[dict[str, Any]] = []
    convergence_by_id = {str(row["trajectory_id"]): row for row in convergence_rows}
    convergence_artifacts_valid = checks["convergence_ledger"]
    with tempfile.TemporaryDirectory(prefix="bench015-convergence-replay-") as temporary:
        temp_root = Path(temporary)
        for cell in cell_specs():
            cell_id = str(cell["cell_id"])
            decoded = decoded_by_renderer[f"{cell_id}__nw78"]
            dsl_decoded = decoded_by_renderer[f"{cell_id}__dsl78"]
            _, reference, _, _ = _render_decoded(decoded)
            means64 = torch.from_numpy(
                np.ascontiguousarray(decoded["arrays"]["means"], dtype=np.float64)
            )
            weights64 = reference.effective_weights.reshape(HEIGHT * WIDTH, PROXY_COUNT)
            weights32 = weights64.to(torch.float32)
            pixels32 = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float64).to(torch.float32)
            target = target_image(str(cell["target"]), dtype=np.float64)
            target32 = torch.from_numpy(np.ascontiguousarray(target)).to(torch.float32)
            for arm in ("nw78_opt", "dsl78_opt"):
                if arm == "dsl78_opt" and not dsl_decoded["method_available"]:
                    rebuilt = _not_run_row(
                        schema="bench015-convergence-v1",
                        key_name="trajectory_id",
                        key_value=f"{cell_id}__dsl78_opt",
                        binding_sha256=binding,
                        reason=dsl_decoded.get("method_error"),
                        cell_id=cell_id,
                        cohort=cell["cohort"],
                        target=cell["target"],
                        seed=int(cell["seed"]),
                        arm="dsl78_opt",
                    )
                else:
                    rebuilt = _convergence_trajectory(
                        cell=cell,
                        arm=arm,
                        means64=means64,
                        weights32=weights32,
                        pixels32=pixels32,
                        target32=target32,
                        outdir=temp_root,
                        binding_sha256=binding,
                    )
                rebuilt_convergence.append(rebuilt)
                if rebuilt["status"] == "ok":
                    observed_row = convergence_by_id[str(rebuilt["trajectory_id"])]
                    rebuilt_arrays = _load_npz(temp_root / str(rebuilt["checkpoint_path"]))
                    convergence_artifacts_valid = bool(
                        convergence_artifacts_valid
                        and _persisted_npz_matches(
                            outdir / str(observed_row["checkpoint_path"]),
                            rebuilt_arrays,
                            str(observed_row["checkpoint_sha256"]),
                            observed_row["checkpoint_records"],
                        )
                    )
    checks["convergence_recomputed"] = _nested_close(
        [
            _row_without(row, "color_fit_step_time_ns", "color_fit_step_median_ns")
            for row in convergence_rows
        ],
        [
            _row_without(row, "color_fit_step_time_ns", "color_fit_step_median_ns")
            for row in rebuilt_convergence
        ],
    )
    checks["convergence_artifacts"] = convergence_artifacts_valid

    gradient_rows = _read_jsonl(outdir / "gradients.jsonl")
    checks["gradient_ledger"] = _ledger_contract(
        gradient_rows,
        key="gradient_id",
        expected=keys["gradients"],
        binding_sha256=binding,
        terminal_statuses=("ok", "not_run_due_to_method_unavailable"),
    )
    with tempfile.TemporaryDirectory(prefix="bench015-gradient-replay-") as temporary:
        rebuilt_gradients = _run_gradients(
            decoded_by_renderer=decoded_by_renderer,
            outdir=Path(temporary),
            binding_sha256=binding,
        )
    direction_manifest = json.loads(
        (outdir / "gradient_direction_manifest.json").read_text(encoding="utf-8")
    )
    observed_directions = _load_npz(outdir / str(direction_manifest["path"]))
    expected_directions = _gradient_direction_arrays()
    checks["gradients_recomputed"] = bool(
        _nested_close(gradient_rows, rebuilt_gradients)
        and direction_manifest["file_sha256"]
        == _sha256_file(outdir / str(direction_manifest["path"]))
        and set(observed_directions) == set(expected_directions)
        and set(direction_manifest.get("records", {})) == set(expected_directions)
        and all(
            np.array_equal(observed_directions[name], expected_directions[name])
            and np.array_equal(observed_directions[name], observed_prebinding[name])
            and direction_manifest["records"][name] == _array_record(expected_directions[name])
            for name in expected_directions
        )
    )

    timing_rows = _read_jsonl(outdir / "timing.jsonl")
    checks["timing_ledger"] = _ledger_contract(
        timing_rows,
        key="timing_id",
        expected=keys["timing"],
        binding_sha256=binding,
        terminal_statuses=("ok", "not_run_due_to_method_unavailable"),
    )
    stored_timing_schedule = json.loads(
        (outdir / "timing_schedule.json").read_text(encoding="utf-8")
    )
    checks["timing_schedule_and_arithmetic"] = bool(
        _nested_close(stored_timing_schedule, _timing_schedule(cell_specs()))
        and _timing_ledger_valid(timing_rows, cell_specs(), decoded_by_renderer)
    )
    decoder_source = inspect.getsource(decode_stream) + inspect.getsource(_decode_stream_impl)
    checks["decoder_target_blind_call_graph"] = bool(
        list(inspect.signature(decode_stream).parameters) == ["blob", "device"]
        and all(
            token not in decoder_source
            for token in ("target_image", "target_values", "source_target")
        )
    )

    validity_pass = bool(all(checks.values()))
    if validity_pass:
        aggregate = analyze_results(
            static_rows=static_rows,
            permutation_rows=permutation_rows,
            convergence_rows=convergence_rows,
            gradient_rows=gradient_rows,
            timing_rows=timing_rows,
        )
    else:
        aggregate = {
            "schema": "bench015-analysis-v1",
            "complete": False,
            "decision": "invalid",
            "classification": "unavailable_independent_replay_failure",
            "stage1_authorized": False,
        }
    replay = {
        "schema": "bench015-replay-v1",
        "binding_sha256": binding,
        "checks": checks,
        "check_count": len(checks),
        "replay_pass": validity_pass,
        "outcome_ledgers_interpreted": validity_pass,
        "f32_shadow_decision_agreement": (
            aggregate.get("f32_shadow_agreement") if validity_pass else None
        ),
    }
    return replay, aggregate


def _pin_runtime() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("BENCH-015 requires exactly one intra-op and one inter-op torch thread")
    torch.use_deterministic_algorithms(True)


def _science_config(
    source_manifest: Mapping[str, str], prebinding_records: Mapping[str, Any]
) -> dict[str, Any]:
    representative_init, representative_structure = frozen_configs(PROXY_COUNT, SEEDS[0])
    formula_records = {
        target: {
            "formula": TARGET_FORMULAS[target],
            "sha256": _sha256_bytes(TARGET_FORMULAS[target].encode("utf-8")),
        }
        for target in TARGETS
    }
    region_records = {
        name: {
            "formula": formula,
            "sha256": _sha256_bytes(formula.encode("utf-8")),
        }
        for name, formula in REGION_FORMULAS.items()
    }
    return {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "task_path": TASK_PATH,
        "task_sha256": source_manifest[TASK_PATH],
        "predecessor": {
            "bench014_task_sha256": (
                "e2b7661c8a856cb2b560bd36e34c0d89f4962f06cf850d94f73afeb98623dbc7"
            ),
            "bench014_binding_sha256": (
                "0a4febdbd9570bcb216a8f6ec1f3461af3577553401415900be4583e5f34825c"
            ),
        },
        "dimensions": {"height": HEIGHT, "width": WIDTH},
        "target_coordinates": "u=x/106;v=y/82;q_x=2*u-1;q_y=2*v-1",
        "counts": list(COUNTS),
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "cohorts": list(COHORTS),
        "arms": list(ARMS),
        "target_formulas": formula_records,
        "region_formulas": region_records,
        "initializer": {
            "fields": asdict(representative_init),
            "count_and_seed_supplied_by_matrix": True,
            "structure_tensor": asdict(representative_structure),
        },
        "codec": {
            "inner_format": "GFCOV01",
            "chart": "current_rs",
            "bits_means": 12,
            "geometry_bits": [6, 6, 6],
            "bits_colors": 8,
            "predictor": "absolute",
            "coder": "zlib9",
            "wrapper": {
                "format": "DSLR015",
                "header_struct": "<8sBBHIHH",
                "header_bytes": STREAM_HEADER.size,
                "crc_bytes": STREAM_CRC.size,
                "magic_hex": STREAM_MAGIC.hex(),
                "version": STREAM_VERSION,
                "nw_mode": 0,
                "dsl_mode": 1,
                "tail_bytes": 0,
                "mode_always_priced": True,
            },
        },
        "mechanism": {
            "irls_iterations": core.IRLS_ITERATIONS,
            "robust_scale_floor": core.ROBUST_SCALE_FLOOR,
            "robust_mad_factor": core.ROBUST_MAD_FACTOR,
            "median": (
                "explicit stable torch.sort; average central pair; persisted row IDs are "
                "the exact autograd support"
            ),
            "neighbor_count": core.NEIGHBOR_COUNT,
            "edge_abs_epsilon": core.EDGE_ABS_EPSILON,
            "color_scale_floor": core.COLOR_SCALE_FLOOR,
            "smooth_max_temperature": core.SMOOTH_MAX_TEMPERATURE,
            "confidence_low": core.CONFIDENCE_LOW,
            "confidence_high": core.CONFIDENCE_HIGH,
            "all_beta_rows_gated": True,
        },
        "gates": {
            "minimum_mass": MINIMUM_MASS,
            "minimum_active": MINIMUM_ACTIVE,
            "partition_max_abs": PARTITION_MAX_ABS,
            "candidate_reference_max_abs": CANDIDATE_REFERENCE_MAX_ABS,
            "minimum_effective_weight": MINIMUM_EFFECTIVE_WEIGHT,
            "range_margin": RANGE_MARGIN,
            "decision_path": "unclipped_float64_reference_vs_exact_float64_target",
            "float32_role": (
                "parity_performance_and_mandatory_shadow_decision; not_primary_numeric_path"
            ),
            "float32_shadow": (
                "numeric ledger persisted; atomic predicate/sign/count/aggregate/final-static "
                "decisions must agree or the mechanism is a scientific kill"
            ),
            "wins": "strict_for_quality; non-strict_for_bytes",
            "medians": "numpy.quantile(method=linear)",
            "quality_and_rate": TASK_PATH,
        },
        "convergence": {
            "updates": CONVERGENCE_STEPS,
            "lr": CONVERGENCE_LR,
            "betas": list(CONVERGENCE_BETAS),
            "eps": CONVERGENCE_EPS,
            "start": "exact_float32_zero",
            "checkpoints": list(CONVERGENCE_CHECKPOINTS),
            "adam_flags": {
                "amsgrad": False,
                "foreach": False,
                "maximize": False,
                "fused": False,
            },
        },
        "permutation": {
            "seed": PERMUTATION_SEED,
            "orders": ["identity", "reverse", "pcg64"],
        },
        "gradient": {
            "seed": GRADIENT_SEED,
            "directions_per_fixture": 2,
            "direction_draw": "rng.normal C-order then divide by Frobenius/L2 norm",
            "step": GRADIENT_STEP,
            "order_statistic_identity_trace": True,
            "synthetic_target_formula": TARGET_FORMULAS["affine_chirp"],
        },
        "prebinding_arrays": dict(prebinding_records),
        "timing": {
            "seed": TIMING_SEED,
            "warmups": TIMING_WARMUPS,
            "repetitions": TIMING_REPETITIONS,
            "threads": 1,
            "cells": "all_54",
            "arms": list(ARMS),
            "recorded_render_samples": EXPECTED_STATIC * TIMING_REPETITIONS,
            "recorded_cold_samples": EXPECTED_STATIC * TIMING_REPETITIONS,
        },
        "expected": {
            "fields": EXPECTED_FIELDS,
            "cells": EXPECTED_CELLS,
            "static": EXPECTED_STATIC,
            "permutations": EXPECTED_PERMUTATIONS,
            "trajectories": EXPECTED_TRAJECTORIES,
            "gradients": EXPECTED_GRADIENTS,
            "timing_units": EXPECTED_TIMING_UNITS,
            "timing_rows": EXPECTED_STATIC,
        },
        "expected_keys": _expected_keys(),
        "source_manifest": dict(source_manifest),
    }


def _run_impl(outdir: Path, root: Path) -> dict[str, Any]:
    _pin_runtime()
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError("BENCH-015 refuses a non-empty output directory")
    outdir.mkdir(parents=True, exist_ok=True)

    # Binding and executed-source archive precede controls, targets, fields, or streams.
    source_manifest = _source_manifest(root)
    environment = _environment(root)
    prebinding_arrays = _prebinding_arrays()
    prebinding_records = _prebinding_records(prebinding_arrays)
    if not all(record.get("nonzero_pass", True) for record in prebinding_records.values()):
        raise RuntimeError("a registered prebinding region mask is empty")
    science = _science_config(source_manifest, prebinding_records)
    binding = _binding_sha256(science, environment)
    _atomic_json(outdir / "config.json", {**science, "binding_sha256": binding})
    _atomic_json(outdir / "environment.json", environment)
    _atomic_json(outdir / "source_manifest.json", source_manifest)
    archive = _write_source_archive(root, outdir)
    _atomic_json(outdir / "source_archive.json", archive)
    _deterministic_npz(outdir / "prebinding_arrays.npz", prebinding_arrays)
    _atomic_json(
        outdir / "prebinding_arrays.json",
        {
            "schema": "bench015-prebinding-arrays-v1",
            "binding_sha256": binding,
            "path": "prebinding_arrays.npz",
            "file_sha256": _sha256_file(outdir / "prebinding_arrays.npz"),
            "records": prebinding_records,
            "persisted_before_controls_targets_and_fields": True,
        },
    )

    controls, control_arrays = run_plumbing_controls()
    _deterministic_npz(outdir / "controls_raw.npz", control_arrays)
    controls = {
        **controls,
        "binding_sha256": binding,
        "path": "controls_raw.npz",
        "file_sha256": _sha256_file(outdir / "controls_raw.npz"),
        "raw_records": {name: _array_record(value) for name, value in control_arrays.items()},
    }
    _atomic_json(outdir / "controls.json", controls)
    if not controls["valid"]:
        manifest = _artifact_manifest(outdir)
        _atomic_json(outdir / "artifact_manifest.json", manifest)
        replay = {
            "schema": "bench015-replay-v1",
            "binding_sha256": binding,
            "replay_pass": False,
            "controls_only": True,
        }
        analysis = {
            "schema": "bench015-analysis-v1",
            "binding_sha256": binding,
            "complete": False,
            "decision": "invalid",
            "classification": "unavailable_invalid_plumbing_controls",
            "stage1_authorized": False,
        }
        _atomic_json(outdir / "replay.json", replay)
        _atomic_json(outdir / "analysis.json", analysis)
        _atomic_json(
            outdir / "completion.json",
            {
                "schema": "bench015-completion-v1",
                "binding_sha256": binding,
                "complete": False,
                "decision": "invalid",
                "artifact_manifest_sha256": _sha256_file(outdir / "artifact_manifest.json"),
                "written_last_after_independent_replay": False,
            },
        )
        return analysis

    # Canonical targets are first materialized only after the persisted binding and controls.
    target_manifest = _target_manifest(outdir)
    _atomic_json(outdir / "target_manifest.json", target_manifest)

    arrays_by_field: dict[str, dict[str, np.ndarray]] = {}
    field_rows: list[dict[str, Any]] = []
    with (outdir / "fields.jsonl").open("wb") as handle:
        for spec in field_specs():
            row, arrays = _build_field(spec, outdir)
            row["binding_sha256"] = binding
            arrays_by_field[str(spec["field_id"])] = arrays
            field_rows.append(row)
            _append_jsonl(handle, row)

    static_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    decoded_by_renderer: dict[str, dict[str, Any]] = {}
    streams_by_cell: dict[str, dict[str, bytes]] = {}
    cells = cell_specs()
    with (
        (outdir / "static.jsonl").open("wb") as static_handle,
        (outdir / "streams.jsonl").open("wb") as stream_handle,
        (outdir / "convergence.jsonl").open("wb") as convergence_handle,
    ):
        for cell in cells:
            cell_id = str(cell["cell_id"])
            target = target_image(str(cell["target"]), dtype=np.float64)
            streams: dict[str, bytes] = {}
            decoded: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                arrays = _cell_stream_arrays(cell, arrays_by_field, arm)
                payload = encode_stream(arm, arrays)
                relative = Path("streams") / f"{cell_id}__{arm}.dslr"
                _atomic_bytes(outdir / relative, payload)
                streams[arm] = payload
                decoded[arm] = decode_stream(payload, "cpu")
            if not (
                len(streams["nw78"]) == len(streams["dsl78"])
                and decoded["nw78"]["inner"] == decoded["dsl78"]["inner"]
            ):
                raise RuntimeError("NW78/DSL78 common-stream invariant failed")
            streams_by_cell[cell_id] = streams
            cell_rows: list[dict[str, Any]] = []
            for arm in ARMS:
                renderer_id = f"{cell_id}__{arm}"
                decoded_by_renderer[renderer_id] = decoded[arm]
                if arm == "dsl78" and not decoded[arm]["method_available"]:
                    row = _static_method_unavailable(
                        cell=cell,
                        arm=arm,
                        payload=streams[arm],
                        decoded=decoded[arm],
                        outdir=outdir,
                        binding_sha256=binding,
                    )
                else:
                    row, _ = _evaluate_static(
                        cell=cell,
                        arm=arm,
                        payload=streams[arm],
                        decoded=decoded[arm],
                        target=target,
                        outdir=outdir,
                        binding_sha256=binding,
                    )
                cell_rows.append(row)
            by_arm = {str(row["arm"]): row for row in cell_rows}
            pair_gates = {
                "dsl_method_available": bool(decoded["dsl78"]["method_available"]),
                "complete_lengths_equal": (
                    by_arm["nw78"]["stream"]["complete_bytes"]
                    == by_arm["dsl78"]["stream"]["complete_bytes"]
                ),
                "inner_stream_identical": (
                    by_arm["nw78"]["stream"]["inner_sha256"]
                    == by_arm["dsl78"]["stream"]["inner_sha256"]
                ),
                "decoded_geometry_identical": (
                    by_arm["nw78"]["decoded_geometry_sha256"]
                    == by_arm["dsl78"]["decoded_geometry_sha256"]
                ),
                "decoded_colors_identical": (
                    by_arm["nw78"]["decoded_colors_sha256"]
                    == by_arm["dsl78"]["decoded_colors_sha256"]
                ),
                "candidate_contributors_identical": (
                    by_arm["nw78"]["candidate_contributor_sha256"]
                    == by_arm["dsl78"]["candidate_contributor_sha256"]
                ),
                "reference_contributors_identical": (
                    by_arm["nw78"]["reference_contributor_sha256"]
                    == by_arm["dsl78"]["reference_contributor_sha256"]
                ),
            }
            for row in cell_rows:
                if row["arm"] in ("nw78", "dsl78"):
                    row["same_stream_pair_gates"] = pair_gates
                    row["forward_pass"] = bool(row["forward_pass"] and all(pair_gates.values()))
                _append_jsonl(static_handle, row)
                static_rows.append(row)
                _append_jsonl(
                    stream_handle,
                    {
                        "schema": "bench015-stream-v1",
                        "binding_sha256": binding,
                        "status": "ok",
                        "renderer_id": row["renderer_id"],
                        "cell_id": cell_id,
                        "arm": row["arm"],
                        "path": row["stream_path"],
                        "record": row["stream"],
                    },
                )

            _, nw_reference, _, _ = _render_decoded(decoded["nw78"])
            means64 = torch.from_numpy(
                np.ascontiguousarray(decoded["nw78"]["arrays"]["means"], dtype=np.float64)
            )
            weights64 = nw_reference.effective_weights.reshape(HEIGHT * WIDTH, PROXY_COUNT)
            weights32 = weights64.to(torch.float32)
            pixels32 = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float64).to(torch.float32)
            target32 = torch.from_numpy(np.ascontiguousarray(target)).to(torch.float32)
            nw_trajectory = _convergence_trajectory(
                cell=cell,
                arm="nw78_opt",
                means64=means64,
                weights32=weights32,
                pixels32=pixels32,
                target32=target32,
                outdir=outdir,
                binding_sha256=binding,
            )
            if decoded["dsl78"]["method_available"]:
                dsl_trajectory = _convergence_trajectory(
                    cell=cell,
                    arm="dsl78_opt",
                    means64=means64,
                    weights32=weights32,
                    pixels32=pixels32,
                    target32=target32,
                    outdir=outdir,
                    binding_sha256=binding,
                )
                if dsl_trajectory["status"] == "ok" and not (
                    nw_trajectory["output_sha256"][0] == dsl_trajectory["output_sha256"][0]
                    and nw_trajectory["initial_output_zero"]
                    and dsl_trajectory["initial_output_zero"]
                ):
                    raise RuntimeError("zero-start convergence invariant failed")
            else:
                dsl_trajectory = _not_run_row(
                    schema="bench015-convergence-v1",
                    key_name="trajectory_id",
                    key_value=f"{cell_id}__dsl78_opt",
                    binding_sha256=binding,
                    reason=decoded["dsl78"].get("method_error"),
                    cell_id=cell_id,
                    cohort=cell["cohort"],
                    target=cell["target"],
                    seed=int(cell["seed"]),
                    arm="dsl78_opt",
                )
            trajectories = [nw_trajectory, dsl_trajectory]
            for row in trajectories:
                _append_jsonl(convergence_handle, row)
                convergence_rows.append(row)

    _run_permutations(
        static_rows=static_rows,
        decoded_by_renderer=decoded_by_renderer,
        outdir=outdir,
        binding_sha256=binding,
    )
    _run_gradients(
        decoded_by_renderer=decoded_by_renderer,
        outdir=outdir,
        binding_sha256=binding,
    )
    _run_timing(
        cells=cells,
        streams_by_cell=streams_by_cell,
        decoded_by_renderer=decoded_by_renderer,
        outdir=outdir,
        binding_sha256=binding,
    )

    manifest = _artifact_manifest(outdir)
    _atomic_json(outdir / "artifact_manifest.json", manifest)
    if not _validate_artifact_manifest(outdir, manifest):
        raise RuntimeError("BENCH-015 artifact manifest failed immediate validation")
    replay, aggregate = replay_artifacts(outdir, root)
    _atomic_json(outdir / "replay.json", replay)
    if replay["replay_pass"]:
        analysis = {**aggregate, "binding_sha256": binding, "replay_pass": True}
    else:
        analysis = {
            "schema": "bench015-analysis-v1",
            "binding_sha256": binding,
            "complete": False,
            "decision": "invalid",
            "classification": "unavailable_independent_replay_failure",
            "stage1_authorized": False,
            "replay_checks": replay["checks"],
        }
    _atomic_json(outdir / "analysis.json", analysis)
    manifest_valid = _validate_artifact_manifest(outdir, manifest)
    completion = {
        "schema": "bench015-completion-v1",
        "binding_sha256": binding,
        "complete": bool(
            replay["replay_pass"] and manifest_valid and analysis["decision"] in ("pass", "kill")
        ),
        "decision": analysis["decision"] if manifest_valid else "invalid",
        "analysis_sha256": _sha256_file(outdir / "analysis.json"),
        "replay_sha256": _sha256_file(outdir / "replay.json"),
        "artifact_manifest_sha256": _sha256_file(outdir / "artifact_manifest.json"),
        "artifact_manifest_validated": manifest_valid,
        "written_last_after_independent_replay": True,
    }
    _atomic_json(outdir / "completion.json", completion)
    return analysis


def run(outdir: Path, root: Path) -> dict[str, Any]:
    """Run BENCH-015 and seal any post-creation exception as an invalid attempt."""

    existed_before_call = outdir.exists()
    initially_nonempty = existed_before_call and any(outdir.iterdir())
    try:
        return _run_impl(outdir, root)
    except Exception as error:
        if (
            initially_nonempty
            or not outdir.exists()
            or (existed_before_call and not any(outdir.iterdir()))
        ):
            raise
        failure = {
            "schema": "bench015-failure-v1",
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(outdir / "failure.json", failure)
        binding = None
        if (outdir / "config.json").is_file():
            binding = json.loads((outdir / "config.json").read_text(encoding="utf-8")).get(
                "binding_sha256"
            )
        replay = {
            "schema": "bench015-replay-v1",
            "binding_sha256": binding,
            "replay_pass": False,
            "outcome_ledgers_interpreted": False,
            "failure": failure,
        }
        analysis = {
            "schema": "bench015-analysis-v1",
            "binding_sha256": binding,
            "complete": False,
            "decision": "invalid",
            "classification": "unavailable_execution_exception",
            "stage1_authorized": False,
        }
        manifest = _artifact_manifest(outdir)
        _atomic_json(outdir / "artifact_manifest.json", manifest)
        _atomic_json(outdir / "replay.json", replay)
        _atomic_json(outdir / "analysis.json", analysis)
        _atomic_json(
            outdir / "completion.json",
            {
                "schema": "bench015-completion-v1",
                "binding_sha256": binding,
                "complete": False,
                "decision": "invalid",
                "failure_sha256": _sha256_file(outdir / "failure.json"),
                "analysis_sha256": _sha256_file(outdir / "analysis.json"),
                "replay_sha256": _sha256_file(outdir / "replay.json"),
                "artifact_manifest_sha256": _sha256_file(outdir / "artifact_manifest.json"),
                "written_last_after_independent_replay": False,
            },
        )
        return analysis


def _replay_cli(outdir: Path, root: Path) -> int:
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    try:
        current_matches = _source_manifest(root) == config["source_manifest"]
    except RuntimeError:
        current_matches = False
    if current_matches:
        replay, aggregate = replay_artifacts(outdir, root)
        print(json.dumps(_json_safe({"replay": replay, "aggregate": aggregate}), indent=2))
        return 0 if replay["replay_pass"] else 1

    archive_record = json.loads((outdir / "source_archive.json").read_text(encoding="utf-8"))
    archive_path = outdir / str(archive_record["path"])
    if _sha256_file(archive_path) != archive_record["sha256"]:
        raise RuntimeError("executed-source archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="bench015-archived-replay-") as temporary:
        archived_root = Path(temporary)
        with tarfile.open(archive_path, mode="r") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError("unsafe member in executed-source archive")
            archive.extractall(archived_root, filter="data")
        environment = os.environ.copy()
        python_paths = [str(archived_root), str(archived_root / "src")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        result = subprocess.run(
            [
                sys.executable,
                str(archived_root / "benchmarks/decoder_synchronized_lift_assay.py"),
                str(outdir.resolve()),
                "--root",
                str(archived_root),
                "--replay",
            ],
            cwd=archived_root,
            env=environment,
            check=False,
        )
        return int(result.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path, nargs="?")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run target-free plumbing controls without creating an output directory",
    )
    arguments = parser.parse_args(argv)
    if arguments.preflight:
        _pin_runtime()
        controls, _ = run_plumbing_controls()
        print(json.dumps(_json_safe(controls), indent=2))
        return 0 if controls["valid"] else 1
    if arguments.outdir is None:
        parser.error("outdir is required unless --preflight is used")
    if arguments.replay:
        return _replay_cli(arguments.outdir, arguments.root)
    result = run(arguments.outdir, arguments.root)
    print(json.dumps(_json_safe(result), indent=2))
    return 0 if result["decision"] in ("pass", "kill") else 1


if __name__ == "__main__":
    raise SystemExit(main())
