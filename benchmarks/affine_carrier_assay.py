"""Source-bound BENCH-014 affine-carrier development assay.

The benchmark keeps StructSplat's positive normalized Gaussian compositor unchanged and
transmits a tiny, gauge-fixed linear carrier for the reproduction defect that the compositor
cannot represent near image boundaries.  This module is deliberately benchmark-only: it builds
production geometries, executes the complete frozen matrix, writes every science-bearing row
incrementally, and independently replays the resulting artifact before writing completion.

No canonical output directory is created by importing this module.  ``run`` refuses a non-empty
directory so a finished or failed attempt is never resumed or patched in place.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import importlib.metadata
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
from typing import Any, BinaryIO, Mapping, Sequence
import zipfile
import zlib

import numpy as np
import torch

from benchmarks import affine_carrier_core as core
from benchmarks import gauge_free_covariance_codec as field_codec
from structsplat.config import InitConfig, StructureTensorConfig
from structsplat.gaussians import GaussianField
from structsplat.init import build_field


PROTOCOL = "bench014-affine-carrier-stage0-v1"
CONFIG_SCHEMA = "bench014-affine-carrier-config-v1"
HEIGHT = 73
WIDTH = 89
PRIMARY_COUNT = 83
PROXY_COUNT = 81
COUNTS = (PRIMARY_COUNT, PROXY_COUNT)
SEEDS = (307, 311, 313)
COHORTS = ("target_conditioned", "shared_constant")
ARMS = ("nw81", "nw83", "ac81")
CODERS = ("gfcov01_zlib9",)

# The task file binds the final formulas.  Values here are evaluated in float64 and cast once for
# the production initializer, field colors, and float32 target path.
TARGETS = (
    "constant",
    "affine",
    "affine_sin",
    "affine_bump",
    "saddle",
    "zero_linear",
    "vertical_step",
    "checker9x7",
)
TARGET_FORMULAS = {
    "constant": "(0.25,0.50,0.75)",
    "affine": (
        "(0.45+0.18*q_x+0.12*q_y, 0.55-0.16*q_x-0.10*q_y, "
        "0.45+0.10*q_x+0.20*q_y)"
    ),
    "affine_sin": (
        "(0.45+0.16*q_x+0.10*q_y+0.07*sin(2*pi*u)*sin(pi*v), "
        "0.55-0.14*q_x-0.08*q_y+0.06*sin(2*pi*v), "
        "0.45+0.08*q_x+0.17*q_y+0.06*sin(2*pi*u)*sin(2*pi*v))"
    ),
    "affine_bump": (
        "g=exp(-((u-0.68)^2/(2*0.11^2)+(v-0.35)^2/(2*0.14^2))); "
        "(0.38+0.13*q_x+0.09*q_y+0.18*g, "
        "0.56-0.12*q_x-0.08*q_y-0.12*g, "
        "0.43+0.08*q_x+0.14*q_y+0.15*g)"
    ),
    "saddle": (
        "(0.48+0.10*q_x+0.07*q_y+0.10*q_x*q_y, "
        "0.52-0.08*q_x+0.06*q_y+0.09*(q_x^2-q_y^2), "
        "0.45+0.06*q_x+0.12*q_y-0.08*q_x*q_y)"
    ),
    "zero_linear": (
        "(0.45+0.12*cos(2*pi*u)*cos(2*pi*v), "
        "0.55+0.10*cos(2*pi*u)+0.06*cos(2*pi*v), "
        "0.45+0.10*sin(2*pi*u)*sin(2*pi*v))"
    ),
    "vertical_step": (
        "(0.1875,0.3125,0.75) for u<0.5, else (0.8125,0.6875,0.25)"
    ),
    "checker9x7": (
        "parity is (floor(9*u)+floor(7*v)) mod 2, using the two vertical_step colors "
        "in parity order zero/one"
    ),
}

SMOOTH_NONAFFINE_TARGETS = (
    "affine_sin",
    "affine_bump",
    "saddle",
)
DISCONTINUOUS_TARGETS = ("vertical_step", "checker9x7")

MINIMUM_MASS = 1e-5
MINIMUM_ACTIVE = 1
MAXIMUM_REPRODUCTION_CONDITION = 64.0
REFERENCE_MAX_ABS = 1e-10
CANDIDATE_REFERENCE_MAX_ABS = 2e-5
CANDIDATE_REFERENCE_RMSE = 2e-6
PARTITION_MAX_ABS = 2e-5
FIRST_MOMENT_DIAGNOSTIC_ONLY = True
PERMUTATION_REFERENCE_MAX_ABS = 1e-12
PERMUTATION_CANDIDATE_MAX_ABS = 2e-5
RANGE_EXCURSION_MAX = 2.0 / 255.0

CONVERGENCE_STEPS = 100
CONVERGENCE_LOG_EVERY = 1
CONVERGENCE_LR = 0.03
CONVERGENCE_BETAS = (0.9, 0.999)
CONVERGENCE_EPS = 1e-8
CONVERGENCE_WEIGHT_DECAY = 0.0
PERMUTATION_SEED = 20260716015
GRADIENT_SEED = 20260716016
TIMING_SEED = 20260716017
GRADIENT_STEP = 2.0**-12
GRADIENT_BLOCKS = (
    "means_normalized",
    "log_scales",
    "rotations_over_pi",
    "residual_rgb",
    "beta",
)

STREAM_MAGIC = b"AFCR014\0"
STREAM_VERSION = 1
STREAM_HEADER = struct.Struct("<8sBBHII")
STREAM_CRC = struct.Struct("<I")
STREAM_ARRAYS = ("means", "log_scales", "rotations", "colors")

EXPECTED_FIELDS = len(TARGETS) * len(COUNTS) * len(SEEDS) + len(COUNTS) * len(SEEDS)
EXPECTED_CELLS = len(COHORTS) * len(TARGETS) * len(SEEDS)
EXPECTED_FORWARD_ROWS = EXPECTED_CELLS * len(ARMS)
EXPECTED_STREAM_ROWS = EXPECTED_FORWARD_ROWS * len(CODERS)
EXPECTED_PERMUTATION_ROWS = EXPECTED_FORWARD_ROWS * 4
EXPECTED_CONVERGENCE_TRAJECTORIES = EXPECTED_CELLS * 2
EXPECTED_CONVERGENCE_ROWS = EXPECTED_CONVERGENCE_TRAJECTORIES * (
    CONVERGENCE_STEPS + 1
)

TASK_PATH = "tasks/BENCH-014-explicit-affine-carrier.md"
SOURCE_PATHS = (
    "benchmarks/__init__.py",
    "benchmarks/affine_carrier_core.py",
    "benchmarks/affine_carrier_assay.py",
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
    "tests/test_affine_carrier_core.py",
    "tests/test_affine_carrier_assay.py",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    payload = json.dumps(
        _json_safe(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return buffer.getvalue()


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(np.ascontiguousarray(arrays[name])))
    _atomic_bytes(path, buffer.getvalue())


def _append_jsonl(handle: BinaryIO, row: Mapping[str, Any]) -> None:
    handle.write(_canonical_json(_json_safe(dict(row))) + b"\n")
    handle.flush()
    os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalized_coordinates(xy: np.ndarray, height: int, width: int) -> tuple[np.ndarray, ...]:
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("xy must have trailing shape 2")
    x = points[..., 0]
    y = points[..., 1]
    xn = 2.0 * x / float(width - 1) - 1.0
    yn = 2.0 * y / float(height - 1) - 1.0
    return x, y, xn, yn


def target_values(
    target: str,
    xy: np.ndarray,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
) -> np.ndarray:
    """Evaluate a frozen BENCH-014 target at continuous pixel coordinates."""

    if target not in TARGETS:
        raise ValueError(f"unknown BENCH-014 target {target!r}")
    points = np.asarray(xy, dtype=np.float64)
    x, y, xn, yn = _normalized_coordinates(points, height, width)
    shape = points.shape[:-1] + (3,)
    if target == "constant":
        result = np.broadcast_to(np.array((0.25, 0.50, 0.75)), shape)
    else:
        u = x / float(width - 1)
        v = y / float(height - 1)
        if target == "affine":
            result = np.stack(
                (
                    0.45 + 0.18 * xn + 0.12 * yn,
                    0.55 - 0.16 * xn - 0.10 * yn,
                    0.45 + 0.10 * xn + 0.20 * yn,
                ),
                axis=-1,
            )
        elif target == "affine_sin":
            result = np.stack(
                (
                    0.45
                    + 0.16 * xn
                    + 0.10 * yn
                    + 0.07 * np.sin(2.0 * np.pi * u) * np.sin(np.pi * v),
                    0.55 - 0.14 * xn - 0.08 * yn + 0.06 * np.sin(2.0 * np.pi * v),
                    0.45
                    + 0.08 * xn
                    + 0.17 * yn
                    + 0.06 * np.sin(2.0 * np.pi * u) * np.sin(2.0 * np.pi * v),
                ),
                axis=-1,
            )
        elif target == "affine_bump":
            bump = np.exp(
                -(
                    (u - 0.68) ** 2 / (2.0 * 0.11**2)
                    + (v - 0.35) ** 2 / (2.0 * 0.14**2)
                )
            )
            result = np.stack(
                (
                    0.38 + 0.13 * xn + 0.09 * yn + 0.18 * bump,
                    0.56 - 0.12 * xn - 0.08 * yn - 0.12 * bump,
                    0.43 + 0.08 * xn + 0.14 * yn + 0.15 * bump,
                ),
                axis=-1,
            )
        elif target == "saddle":
            result = np.stack(
                (
                    0.48 + 0.10 * xn + 0.07 * yn + 0.10 * xn * yn,
                    0.52 - 0.08 * xn + 0.06 * yn + 0.09 * (xn**2 - yn**2),
                    0.45 + 0.06 * xn + 0.12 * yn - 0.08 * xn * yn,
                ),
                axis=-1,
            )
        elif target == "zero_linear":
            result = np.stack(
                (
                    0.45 + 0.12 * np.cos(2.0 * np.pi * u) * np.cos(2.0 * np.pi * v),
                    0.55
                    + 0.10 * np.cos(2.0 * np.pi * u)
                    + 0.06 * np.cos(2.0 * np.pi * v),
                    0.45 + 0.10 * np.sin(2.0 * np.pi * u) * np.sin(2.0 * np.pi * v),
                ),
                axis=-1,
            )
        else:
            low = np.array((0.1875, 0.3125, 0.75))
            high = np.array((0.8125, 0.6875, 0.25))
            if target == "vertical_step":
                result = np.where((u < 0.5)[..., None], low, high)
            else:
                parity = (np.floor(9.0 * u) + np.floor(7.0 * v)).astype(np.int64) % 2
                result = np.where((parity == 0)[..., None], low, high)
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
                        "evaluation_targets": [target],
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
                    "evaluation_targets": list(TARGETS),
                }
            )
    return tuple(sorted(specs, key=lambda item: str(item["field_id"])))


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
                        "baseline_field_id": _field_id(cohort, source, PROXY_COUNT, seed),
                        "challenger_field_id": _field_id(cohort, source, PRIMARY_COUNT, seed),
                    }
                )
    return tuple(sorted(specs, key=lambda item: str(item["cell_id"])))


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
        diff = bytes(invoke("diff", "--binary", "--no-ext-diff", binary=True))
        return {
            "commit": str(invoke("rev-parse", "HEAD")).strip(),
            "branch": str(invoke("branch", "--show-current")).strip(),
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
            "tracked_diff_sha256": _sha256_bytes(diff),
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
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "cpu_model": cpu_model,
        "platform_processor": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_available_count": None if affinity is None else len(affinity),
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


def _source_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing BENCH-014 source dependency: {relative}")
        manifest[relative] = _sha256_file(path)
    return manifest


def _write_source_archive(root: Path, outdir: Path) -> dict[str, Any]:
    archive_path = outdir / "executed_sources.tar"
    members: dict[str, str] = {}
    with tarfile.open(archive_path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative in sorted(ARCHIVE_PATHS):
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"missing BENCH-014 archive dependency: {relative}")
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            payload = path.read_bytes()
            archive.addfile(info, io.BytesIO(payload))
            members[relative] = _sha256_bytes(payload)
    return {
        "path": archive_path.name,
        "sha256": _sha256_file(archive_path),
        "members": members,
    }


def _stable_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "python",
        "platform",
        "numpy",
        "torch",
        "device",
        "cpu_model",
        "platform_processor",
        "cpu_logical_count",
        "cpu_affinity",
        "cpu_available_count",
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


def _binding_sha256(science: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {"science": dict(science), "environment": _stable_environment(environment)}
        )
    )


def _target_manifest(outdir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for target in TARGETS:
        target_record: dict[str, Any] = {
            "formula": TARGET_FORMULAS[target],
            "formula_sha256": _sha256_bytes(TARGET_FORMULAS[target].encode("utf-8")),
        }
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            image = target_image(target, dtype=dtype)
            arrays[f"{target}__{name}"] = image
            target_record[name] = _array_record(image)
        manifest[target] = target_record
    _deterministic_npz(outdir / "targets.npz", arrays)
    return {
        "path": "targets.npz",
        "file_sha256": _sha256_file(outdir / "targets.npz"),
        "targets": manifest,
    }


def _geometry_record(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    names = ("means", "log_scales", "rotations", "conics", "radii")
    records = {name: _array_record(arrays[name]) for name in names}
    return {"arrays": records, "sha256": _sha256_bytes(_canonical_json(records))}


def _build_field(
    spec: Mapping[str, Any], outdir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = int(spec["count"])
    seed = int(spec["seed"])
    source_target = str(spec["source_target"])
    init, structure = frozen_configs(count, seed)
    source_image = target_image(source_target, dtype=np.float32)
    field = build_field(source_image, init, structure, device="cpu")
    if int(field.n) != count:
        raise RuntimeError(f"initializer returned {field.n} rows for requested {count}")
    if (
        field.opacities is not None
        or field.filter_variance is not None
        or field.color_grads is not None
    ):
        raise RuntimeError(
            "BENCH-014 freezes opacity=None, color_grads=None, and zero covariance dilation"
        )
    means = field.means.detach().cpu().contiguous().numpy().copy()
    exact_source_colors = target_values(source_target, means, dtype=np.float64)
    field.colors = torch.from_numpy(exact_source_colors.astype(np.float32)).clone()
    arrays = {
        "means": field.means.detach().cpu().contiguous().numpy().copy(),
        "log_scales": field.log_scales.detach().cpu().contiguous().numpy().copy(),
        "rotations": field.rotations.detach().cpu().contiguous().numpy().copy(),
        "colors": field.colors.detach().cpu().contiguous().numpy().copy(),
        "conics": field.conics(dilation=0.0).detach().cpu().contiguous().numpy().copy(),
        "radii": field.radii(core.SIGMA_CUTOFF, dilation=0.0)
        .detach()
        .cpu()
        .contiguous()
        .numpy()
        .copy(),
    }
    relative = Path("fields") / f"{spec['field_id']}.npz"
    _deterministic_npz(outdir / relative, arrays)
    target_colors: dict[str, Any] = {}
    for target in spec["evaluation_targets"]:
        exact = target_values(str(target), arrays["means"], dtype=np.float64)
        stored = exact.astype(np.float32)
        target_colors[str(target)] = {
            "exact_float64": _array_record(exact),
            "stored_float32": _array_record(stored),
        }
    record = {
        "schema": "bench014-field-v1",
        **dict(spec),
        "path": relative.as_posix(),
        "file_sha256": _sha256_file(outdir / relative),
        "geometry": _geometry_record(arrays),
        "source_colors_float64": _array_record(exact_source_colors),
        "target_colors": target_colors,
        "init_config": asdict(init),
        "structure_tensor_config": asdict(structure),
    }
    return record, arrays


def encode_stream(
    arm: str,
    arrays: Mapping[str, np.ndarray],
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    beta_payload: bytes | None = None,
) -> bytes:
    """Encode the canonical AFCR014 wrapper around the audited GFCOV01 stream."""

    if arm not in ARMS:
        raise ValueError(f"unknown BENCH-014 arm {arm!r}")
    count = int(np.asarray(arrays["means"]).shape[0])
    expected_count = PRIMARY_COUNT if arm.endswith("83") else PROXY_COUNT
    if count != expected_count:
        raise ValueError(f"{arm} requires exactly {expected_count} Gaussian rows")
    has_beta = arm.startswith("ac")
    if has_beta != (beta_payload is not None):
        raise ValueError("only AC arms carry a beta payload")
    if beta_payload is not None and len(beta_payload) != core.BETA_PAYLOAD_BYTES:
        raise ValueError("carrier payload has the wrong byte length")
    expected_shapes: dict[str, tuple[int, ...]] = {
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
        if not np.issubdtype(value.dtype, np.floating) or not bool(np.isfinite(value).all()):
            raise ValueError(f"stream {name} must be finite floating point")
        state[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).clone()
    field = GaussianField(
        state["means"], state["log_scales"], state["rotations"], state["colors"]
    )
    config = field_codec.CodecConfig(
        chart="current_rs",
        geometry_bits=(6, 6, 6),
        predictor="absolute",
        coder="zlib9",
        bits_means=12,
        bits_colors=8,
    )
    inner = field_codec.encode(field, int(height), int(width), config)
    tail = b"" if beta_payload is None else bytes(beta_payload)
    header = STREAM_HEADER.pack(
        STREAM_MAGIC,
        STREAM_VERSION,
        int(has_beta),
        0,
        len(inner),
        len(tail),
    )
    checksummed = header + inner + tail
    return checksummed + STREAM_CRC.pack(zlib.crc32(checksummed) & 0xFFFFFFFF)


def _decode_stream_impl(
    payload: bytes, device: str | torch.device, *, audit_reencode: bool
) -> dict[str, Any]:
    """Cold-decode one canonical raw field wrapper without using encoder-side state."""

    minimum = STREAM_HEADER.size + STREAM_CRC.size
    if len(payload) < minimum:
        raise ValueError("truncated affine-carrier stream")
    magic, version, variant, reserved, inner_length, tail_length = STREAM_HEADER.unpack_from(
        payload
    )
    if magic != STREAM_MAGIC or version != STREAM_VERSION or reserved != 0:
        raise ValueError("invalid affine-carrier stream header")
    if variant not in (0, 1):
        raise ValueError("AFCR014 variant must be NW=0 or AC=1")
    if (variant == 0 and tail_length != 0) or (
        variant == 1 and tail_length != core.BETA_PAYLOAD_BYTES
    ):
        raise ValueError("AFCR014 variant/tail declaration is inconsistent")
    offset = STREAM_HEADER.size
    expected_size = offset + inner_length + tail_length + STREAM_CRC.size
    if expected_size != len(payload):
        raise ValueError("affine-carrier stream length metadata is inconsistent")
    stored_crc = STREAM_CRC.unpack_from(payload, len(payload) - STREAM_CRC.size)[0]
    crc_payload = payload[:-STREAM_CRC.size]
    if stored_crc != (zlib.crc32(crc_payload) & 0xFFFFFFFF):
        raise ValueError("affine-carrier wrapper CRC mismatch")
    inner = bytes(payload[offset : offset + inner_length])
    offset += inner_length
    tail = bytes(payload[offset : offset + tail_length])
    offset += tail_length
    if offset + STREAM_CRC.size != len(payload):
        raise AssertionError("internal AFCR014 decoder offset mismatch")
    header_record = field_codec.blob_header(inner)
    count = int(header_record["n"])
    height = int(header_record["height"])
    width = int(header_record["width"])
    if count not in COUNTS:
        raise ValueError("AFCR014 inner Gaussian count is outside the frozen matrix")
    arm = ("ac" if variant == 1 else "nw") + str(count)
    if arm not in ARMS:
        raise ValueError("AFCR014 inner count/variant does not name a frozen arm")
    if (
        header_record["format"] != "GFCOV01"
        or header_record["chart"] != "current_rs"
        or header_record["predictor"] != "absolute"
        or header_record["coder"] != "zlib9"
        or int(header_record["bits_means"]) != 12
        or tuple(header_record["geometry_bits"]) != (6, 6, 6)
        or int(header_record["bits_colors"]) != 8
    ):
        raise ValueError("inner GFCOV01 stream violates the frozen BENCH-014 profile")
    if audit_reencode:
        if field_codec.canonical_reencode(inner) != inner:
            raise ValueError("inner GFCOV01 stream is not canonically re-encodable")
        if field_codec.decoded_reencode(inner) != inner:
            raise ValueError("cold decoded GFCOV01 field does not ordinarily re-encode exactly")
    field = field_codec.decode(inner, device=device)
    if field.means.device.type != "cpu":
        raise ValueError("BENCH-014 reference decoder currently requires a CPU device")
    arrays = {
        "means": field.means.detach().cpu().numpy().copy(),
        "log_scales": field.log_scales.detach().cpu().numpy().copy(),
        "rotations": field.rotations.detach().cpu().numpy().copy(),
        "colors": field.colors.detach().cpu().numpy().copy(),
    }
    beta_payload: bytes | None = None
    beta: np.ndarray | None = None
    if variant == 1:
        if tail_length != core.BETA_PAYLOAD_BYTES:
            raise ValueError("AFCR014 carrier tail has the wrong size")
        beta_payload = tail
        beta = core.decode_beta_f16(beta_payload).numpy().copy()
    elif tail_length != 0:
        raise ValueError("NW wrapper must have an empty tail")
    return {
        "arm": arm,
        "height": int(height),
        "width": int(width),
        "count": int(count),
        "arrays": arrays,
        "inner": inner,
        "inner_header": header_record,
        "inner_components": field_codec.blob_components(inner),
        "beta_payload": beta_payload,
        "beta": beta,
    }


def decode_stream(
    payload: bytes, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Cold-decode one complete AFCR014 blob from blob and device alone."""

    return _decode_stream_impl(payload, device, audit_reencode=True)


def compress_stream(raw: bytes, coder: str) -> bytes:
    if coder != "gfcov01_zlib9":
        raise ValueError(f"unknown BENCH-014 coder {coder!r}")
    return raw


def decompress_stream(blob: bytes, coder: str) -> bytes:
    if coder != "gfcov01_zlib9":
        raise ValueError(f"unknown BENCH-014 coder {coder!r}")
    return blob


def _field_from_stream(decoded: Mapping[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    arrays = decoded["arrays"]
    state = {
        name: torch.from_numpy(np.ascontiguousarray(arrays[name])).clone()
        for name in STREAM_ARRAYS
    }
    field = GaussianField(
        state["means"], state["log_scales"], state["rotations"], state["colors"]
    )
    state["conics"] = field.conics(dilation=0.0)
    radii = field.radii(core.SIGMA_CUTOFF, dilation=0.0)
    return state, radii


def _mse(output: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((output.astype(np.float64) - target.astype(np.float64)) ** 2))


def _psnr_from_mse(mse: float) -> float:
    return float("inf") if mse == 0.0 else float(-10.0 * math.log10(mse))


def _outer_mask(height: int = HEIGHT, width: int = WIDTH, border: int = 9) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return (xx < border) | (xx >= width - border) | (yy < border) | (yy >= height - border)


def _output_metrics(output: np.ndarray, target: np.ndarray) -> dict[str, float]:
    output64 = np.asarray(output, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    error = output64 - target64
    mse = float(np.mean(error**2))
    outer = _outer_mask(target64.shape[0], target64.shape[1])
    lower = target64.reshape(-1, 3).min(axis=0)
    upper = target64.reshape(-1, 3).max(axis=0)
    excursion = np.maximum(np.maximum(lower - output64, output64 - upper), 0.0)
    return {
        "mse": mse,
        "psnr": _psnr_from_mse(mse),
        "mae": float(np.mean(np.abs(error))),
        "max_abs": float(np.max(np.abs(error))),
        "outer9_mse": float(np.mean(error[outer] ** 2)),
        "range_excursion": float(np.max(excursion)),
    }


def _contributor_hash(contributors: core.FrozenContributors) -> str:
    records = {
        "gid": _array_record(contributors.gid),
        "px": _array_record(contributors.px),
        "py": _array_record(contributors.py),
    }
    return _sha256_bytes(_canonical_json(records))


def _geometry_hash_from_decoded(decoded: Mapping[str, Any]) -> str:
    arrays = decoded["arrays"]
    records = {
        name: _array_record(np.asarray(arrays[name]))
        for name in ("means", "log_scales", "rotations")
    }
    return _sha256_bytes(_canonical_json(records))


def _render_decoded(
    decoded: Mapping[str, Any],
) -> tuple[core.CandidateResult, core.ReferenceResult]:
    state32, radii = _field_from_stream(decoded)
    beta_value = decoded.get("beta")
    beta32 = (
        torch.zeros(core.BETA_SHAPE, dtype=torch.float32)
        if beta_value is None
        else torch.from_numpy(np.ascontiguousarray(beta_value, dtype=np.float32)).clone()
    )
    candidate = core.candidate_render_residual(
        state32["means"],
        state32["conics"],
        state32["colors"],
        beta32,
        radii,
        int(decoded["height"]),
        int(decoded["width"]),
    )
    means64 = state32["means"].to(torch.float64)
    log_scales64 = state32["log_scales"].to(torch.float64)
    rotations64 = state32["rotations"].to(torch.float64)
    reference = core.reference_render_residual(
        means64,
        core.conics_from_parameters(log_scales64, rotations64),
        state32["colors"].to(torch.float64),
        beta32.to(torch.float64),
        radii,
        int(decoded["height"]),
        int(decoded["width"]),
    )
    return candidate, reference


def _stream_arrays_with_colors(
    decoded: Mapping[str, Any], colors: np.ndarray
) -> dict[str, np.ndarray]:
    arrays = decoded["arrays"]
    return {
        "means": np.asarray(arrays["means"], dtype=np.float32),
        "log_scales": np.asarray(arrays["log_scales"], dtype=np.float32),
        "rotations": np.asarray(arrays["rotations"], dtype=np.float32),
        "colors": np.ascontiguousarray(colors, dtype=np.float32),
    }


def _residual_for_beta(
    base_decoded: Mapping[str, Any], decoded_beta: torch.Tensor
) -> np.ndarray:
    means64 = torch.from_numpy(
        np.ascontiguousarray(base_decoded["arrays"]["means"], dtype=np.float64)
    )
    colors64 = torch.from_numpy(
        np.ascontiguousarray(base_decoded["arrays"]["colors"], dtype=np.float64)
    )
    beta64 = decoded_beta.to(torch.float64)
    residual = colors64 - core.carrier_design(
        means64, int(base_decoded["height"]), int(base_decoded["width"])
    ) @ beta64
    return np.ascontiguousarray(residual.numpy(), dtype=np.float32)


def _fit_defect(
    base_decoded: Mapping[str, Any], target: np.ndarray
) -> tuple[core.ReproductionDefectFit, core.ReferenceResult]:
    state32, radii = _field_from_stream(base_decoded)
    means64 = state32["means"].to(torch.float64)
    conics64 = core.conics_from_parameters(
        state32["log_scales"].to(torch.float64),
        state32["rotations"].to(torch.float64),
    )
    colors64 = state32["colors"].to(torch.float64)
    zero = torch.zeros(core.BETA_SHAPE, dtype=torch.float64)
    height = int(base_decoded["height"])
    width = int(base_decoded["width"])
    baseline = core.reference_render(
        means64,
        conics64,
        colors64,
        zero,
        radii,
        height,
        width,
    )
    fit = core.fit_reproduction_defect_qr(
        means64,
        conics64,
        colors64,
        radii,
        torch.from_numpy(np.ascontiguousarray(target, dtype=np.float64)),
        height,
        width,
        contributors=baseline.contributors,
    )
    return fit, baseline


def _stream_safe_carrier(
    base_decoded: Mapping[str, Any],
    target: np.ndarray,
    *,
    search_handle: BinaryIO | None = None,
    binding_sha256: str | None = None,
    cell_id: str | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Run the frozen channel-sequential, codec-in-the-loop carrier ray search."""

    search_start_ns = time.perf_counter_ns()
    target64 = np.ascontiguousarray(target, dtype=np.float64)
    height = int(base_decoded["height"])
    width = int(base_decoded["width"])
    if target64.shape != (height, width, 3):
        raise ValueError("target extent does not match the cold decoded stream")
    fit, baseline_reference = _fit_defect(base_decoded, target64)
    if not fit.diagnostics.accepted:
        raise RuntimeError("reproduction-defect design is rank deficient")
    if fit.diagnostics.condition_number > MAXIMUM_REPRODUCTION_CONDITION:
        raise RuntimeError("reproduction-defect design exceeds the frozen condition gate")
    pre_projection = core.project_beta_ray_on_finite_grid(
        fit.beta,
        fit.baseline,
        fit.reproduction_design,
        torch.from_numpy(target64.reshape(-1, 3)),
    )
    baseline_candidate, _ = _render_decoded(base_decoded)
    baseline_output = baseline_candidate.output.detach().numpy().astype(np.float64)
    baseline_sse = ((baseline_output - target64) ** 2).sum(axis=(0, 1))
    sse_limit = baseline_sse + np.maximum(1e-15, 1e-10 * baseline_sse)
    target_minimum = target64.reshape(-1, 3).min(axis=0)
    target_maximum = target64.reshape(-1, 3).max(axis=0)

    selected_unquantized = torch.zeros(core.BETA_SHAPE, dtype=torch.float64)
    selected_output = baseline_candidate.output.detach().numpy().copy()
    selected_indices = np.full(3, core.RAY_GRID_DENOMINATOR, dtype=np.int64)
    selected_alphas = np.zeros(3, dtype=np.float64)
    final_stream: bytes | None = None
    final_decoded: dict[str, Any] | None = None
    trial_count = 0
    codec_trial_count = 0
    binary16_overflow_rejections = 0
    for channel in range(3):
        accepted = False
        candidates = list(range(core.RAY_GRID_DENOMINATOR)) + [core.RAY_GRID_DENOMINATOR]
        for grid_index in candidates:
            alpha = (
                0.0
                if grid_index == core.RAY_GRID_DENOMINATOR
                else float(pre_projection.alpha_cap[channel])
                * (1.0 - grid_index / core.RAY_GRID_DENOMINATOR)
            )
            trial_unquantized = selected_unquantized.clone()
            trial_unquantized[:, channel] = alpha * fit.beta[:, channel]
            try:
                beta_payload, decoded_beta = core.quantize_beta_f16(trial_unquantized)
            except ValueError as exc:
                message = str(exc)
                if "binary16" not in message:
                    raise
                trial_count += 1
                binary16_overflow_rejections += 1
                if search_handle is not None:
                    _append_jsonl(
                        search_handle,
                        {
                            "schema": "bench014-beta-search-v1",
                            "binding_sha256": binding_sha256,
                            "cell_id": cell_id,
                            "trial_id": f"{cell_id}__rgb{channel}__k{grid_index:04d}",
                            "channel": channel,
                            "grid_index": grid_index,
                            "alpha_cap": float(pre_projection.alpha_cap[channel]),
                            "alpha": alpha,
                            "status": "rejected",
                            "reason": "binary16_range_overflow",
                            "accepted": False,
                        },
                    )
                continue
            codec_trial_count += 1
            residual = _residual_for_beta(base_decoded, decoded_beta)
            stream = encode_stream(
                "ac81",
                _stream_arrays_with_colors(base_decoded, residual),
                height=height,
                width=width,
                beta_payload=beta_payload,
            )
            decoded = _decode_stream_impl(stream, "cpu", audit_reencode=False)
            candidate, _ = _render_decoded(decoded)
            output = candidate.output.detach().numpy()
            previous_unchanged = bool(
                channel == 0
                or np.array_equal(output[..., :channel], selected_output[..., :channel])
            )
            channel_output = output[..., channel].astype(np.float64)
            sse = float(((channel_output - target64[..., channel]) ** 2).sum())
            minimum = float(channel_output.min())
            maximum = float(channel_output.max())
            excursion = max(
                float(target_minimum[channel]) - minimum,
                maximum - float(target_maximum[channel]),
                0.0,
            )
            range_pass = excursion <= core.RAY_GATE_MARGIN
            sse_pass = sse <= float(sse_limit[channel])
            passed = previous_unchanged and range_pass and sse_pass
            trial_count += 1
            if search_handle is not None:
                _append_jsonl(
                    search_handle,
                    {
                        "schema": "bench014-beta-search-v1",
                        "binding_sha256": binding_sha256,
                        "cell_id": cell_id,
                        "trial_id": f"{cell_id}__rgb{channel}__k{grid_index:04d}",
                        "channel": channel,
                        "grid_index": grid_index,
                        "alpha_cap": float(pre_projection.alpha_cap[channel]),
                        "alpha": alpha,
                        "status": "accepted" if passed else "rejected",
                        "reason": "accepted" if passed else "range_or_sse",
                        "beta_payload_sha256": _sha256_bytes(beta_payload),
                        "stream_sha256": _sha256_bytes(stream),
                        "output_minimum": minimum,
                        "output_maximum": maximum,
                        "excursion": excursion,
                        "sse": sse,
                        "sse_limit": float(sse_limit[channel]),
                        "previous_channels_unchanged": previous_unchanged,
                        "range_pass": range_pass,
                        "sse_pass": sse_pass,
                        "accepted": passed,
                    },
                )
            if passed:
                selected_unquantized = trial_unquantized
                selected_output = output.copy()
                selected_indices[channel] = grid_index
                selected_alphas[channel] = alpha
                final_stream = stream
                final_decoded = decoded
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"no feasible carrier ray candidate for RGB channel {channel}")
    if final_stream is None or final_decoded is None:
        raise AssertionError("carrier search did not produce a final stream")
    final_decoded = decode_stream(final_stream, "cpu")
    final_candidate, final_reference = _render_decoded(final_decoded)
    final_output = final_candidate.output.detach().numpy().astype(np.float64)
    final_sse = ((final_output - target64) ** 2).sum(axis=(0, 1))
    final_minimum = final_output.reshape(-1, 3).min(axis=0)
    final_maximum = final_output.reshape(-1, 3).max(axis=0)
    final_excursion = np.maximum(
        np.maximum(target_minimum - final_minimum, final_maximum - target_maximum), 0.0
    )
    projection_error = (
        fit.baseline
        + fit.reproduction_design @ fit.beta
        - torch.from_numpy(target64.reshape(-1, 3))
    )
    projection_sse = float(projection_error.square().sum().item())
    baseline_projection_sse = float(fit.target_residual.square().sum().item())
    target_flat = torch.from_numpy(target64.reshape(-1, 3))
    target_plane_design = core.pixel_design(height, width, dtype=torch.float64)
    target_plane = core.solve_reproduction_design_qr(target_plane_design, target_flat)
    target_plane_direct_sse = float(
        (target_plane_design @ target_plane.beta - target_flat).square().sum()
    )
    target_plane_defect_sse = float(
        (
            fit.baseline + fit.reproduction_design @ target_plane.beta - target_flat
        ).square().sum()
    )
    summary = {
        "schema": "bench014-beta-summary-v1",
        "beta_ls": _json_safe(fit.beta),
        "singular_values": _json_safe(fit.diagnostics.singular_values),
        "rank_threshold": fit.diagnostics.rank_threshold,
        "rank": fit.diagnostics.rank,
        "rank_pass": fit.diagnostics.rank == 2,
        "condition_number": fit.diagnostics.condition_number,
        "condition_pass": fit.diagnostics.condition_number <= MAXIMUM_REPRODUCTION_CONDITION,
        "alpha_cap": _json_safe(pre_projection.alpha_cap),
        "selected_grid_index": selected_indices.tolist(),
        "selected_alpha": selected_alphas.tolist(),
        "selected_beta_pre_f16": _json_safe(selected_unquantized),
        "decoded_beta": _json_safe(final_decoded["beta"]),
        "beta_payload_sha256": _sha256_bytes(final_decoded["beta_payload"]),
        "trial_count": trial_count,
        "encoder_work": {
            "carrier_qr_fits": 1,
            "target_plane_diagnostic_qr_fits": 1,
            "prestream_finite_grid_quantizations": int(
                (pre_projection.grid_index + 1).sum()
            ),
            "channel_trials": (selected_indices + 1).tolist(),
            "candidate_encodes": codec_trial_count,
            "candidate_cold_decodes": codec_trial_count,
            "candidate_renders": codec_trial_count,
            "binary16_overflow_rejections": binary16_overflow_rejections,
            "total_trials": trial_count,
            "elapsed_ns_diagnostic": time.perf_counter_ns() - search_start_ns,
        },
        "baseline_channel_sse": baseline_sse.tolist(),
        "final_channel_sse": final_sse.tolist(),
        "sse_nonregression_pass": bool(np.all(final_sse <= sse_limit)),
        "final_output_minimum": final_minimum.tolist(),
        "final_output_maximum": final_maximum.tolist(),
        "final_excursion": final_excursion.tolist(),
        "final_range_pass": bool(np.all(final_excursion <= core.RAY_GATE_MARGIN)),
        "unconstrained_projection_sse": projection_sse,
        "baseline_projection_sse": baseline_projection_sse,
        "unconstrained_projection_pass": projection_sse
        <= baseline_projection_sse + max(1e-15, 1e-12 * baseline_projection_sse),
        "prestream_projection": _json_safe(asdict(pre_projection)),
        "prestream_feasible": pre_projection.feasible,
        "target_plane_ols_diagnostic": {
            "beta": _json_safe(target_plane.beta),
            "direct_plane_sse": target_plane_direct_sse,
            "defect_application_sse": target_plane_defect_sse,
            "non_gating": True,
        },
        "reference_partition_max_abs": float(
            torch.max(torch.abs(baseline_reference.partition - 1.0))
        ),
        "final_stream_sha256": _sha256_bytes(final_stream),
        "final_candidate_sha256": _array_record(final_candidate.output)["record_sha256"],
        "final_reference_sha256": _array_record(final_reference.output)["record_sha256"],
    }
    return final_stream, final_decoded, summary


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _renderer_id(cell_id: str, arm: str) -> str:
    return f"{cell_id}__{arm}"


def _stream_component_record(payload: bytes, decoded: Mapping[str, Any]) -> dict[str, Any]:
    tail_bytes = 0 if decoded["beta_payload"] is None else len(decoded["beta_payload"])
    components = {
        "wrapper_header_bytes": STREAM_HEADER.size,
        "inner_bytes": len(decoded["inner"]),
        "tail_bytes": tail_bytes,
        "wrapper_crc_bytes": STREAM_CRC.size,
    }
    accounted = sum(components.values())
    reconstructed = encode_stream(
        str(decoded["arm"]),
        decoded["arrays"],
        height=int(decoded["height"]),
        width=int(decoded["width"]),
        beta_payload=decoded["beta_payload"],
    )
    return {
        "complete_bytes": len(payload),
        "complete_sha256": _sha256_bytes(payload),
        "components": components,
        "component_sum": accounted,
        "component_accounting_pass": accounted == len(payload),
        "common_wrapper_overhead_bytes": STREAM_HEADER.size + STREAM_CRC.size,
        "inner": _json_safe(decoded["inner_components"]),
        "inner_header": _json_safe(decoded["inner_header"]),
        "ordinary_decoded_reencode_pass": reconstructed == payload,
        "deterministic_encode_pass": reconstructed == payload,
    }


def _raw_render_arrays(
    candidate: core.CandidateResult,
    reference: core.ReferenceResult,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate_output = candidate.output.detach().numpy()
    reference_output = reference.output.detach().numpy()
    target64 = np.ascontiguousarray(target, dtype=np.float64)
    return {
        "target_float64": target64,
        "candidate_output_float32": candidate_output,
        "reference_output_float64": reference_output,
        "candidate_error_float64": candidate_output.astype(np.float64) - target64,
        "reference_error_float64": reference_output - target64,
        "candidate_mass": candidate.mass.detach().numpy(),
        "reference_mass": reference.mass.detach().numpy(),
        "candidate_partition": candidate.partition.detach().numpy(),
        "reference_partition": reference.partition.detach().numpy(),
        "candidate_minimum_effective_weight": (
            candidate.minimum_effective_weight.detach().numpy()
        ),
        "reference_minimum_effective_weight": (
            reference.minimum_effective_weight.detach().numpy()
        ),
        "candidate_first_moment": candidate.first_moment.detach().numpy(),
        "reference_first_moment": reference.first_moment.detach().numpy(),
        "candidate_a1": candidate.amplification_l1.detach().numpy(),
        "reference_a1": reference.amplification_l1.detach().numpy(),
        "candidate_active_counts": candidate.active_counts.detach().numpy(),
        "reference_active_counts": reference.active_counts.detach().numpy(),
        "candidate_finite_pass": candidate.finite_pass.detach().numpy(),
        "reference_finite_pass": reference.finite_pass.detach().numpy(),
        "reference_effective_weights": reference.effective_weights.detach().numpy(),
        "contributors_gid": candidate.contributors.gid.detach().numpy(),
        "contributors_px": candidate.contributors.px.detach().numpy(),
        "contributors_py": candidate.contributors.py.detach().numpy(),
    }


def _static_metrics_and_gates(
    candidate: core.CandidateResult,
    reference: core.ReferenceResult,
    target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, bool]]:
    candidate_output = candidate.output.detach().numpy()
    reference_output = reference.output.detach().numpy()
    candidate_metrics = _output_metrics(candidate_output, target)
    reference_metrics = _output_metrics(reference_output, target)
    parity = candidate_output.astype(np.float64) - reference_output
    contributor_match = _contributor_hash(candidate.contributors) == _contributor_hash(
        reference.contributors
    )
    metrics: dict[str, Any] = {
        "candidate": candidate_metrics,
        "reference": reference_metrics,
        "candidate_reference_max_abs": float(np.max(np.abs(parity))),
        "candidate_reference_rmse": float(np.sqrt(np.mean(parity**2))),
        "candidate_mass_min": float(candidate.mass.min()),
        "reference_mass_min": float(reference.mass.min()),
        "candidate_active_min": int(candidate.active_counts.min()),
        "reference_active_min": int(reference.active_counts.min()),
        "candidate_partition_max_abs": float((candidate.partition - 1.0).abs().max()),
        "reference_partition_max_abs": float((reference.partition - 1.0).abs().max()),
        "candidate_minimum_effective_weight": float(
            candidate.minimum_effective_weight.min()
        ),
        "reference_minimum_effective_weight": float(
            reference.minimum_effective_weight.min()
        ),
        "candidate_a1_max": float(candidate.amplification_l1.max()),
        "reference_a1_max": float(reference.amplification_l1.max()),
        "candidate_first_moment_max_abs_diagnostic": float(
            candidate.first_moment.abs().max()
        ),
        "reference_first_moment_max_abs_diagnostic": float(
            reference.first_moment.abs().max()
        ),
        "contributor_triplets": int(candidate.contributors.gid.numel()),
        "contributor_hash": _contributor_hash(candidate.contributors),
    }
    gates = {
        "finite": bool(candidate.finite_pass.all() and reference.finite_pass.all()),
        "mass": metrics["candidate_mass_min"] >= MINIMUM_MASS
        and metrics["reference_mass_min"] >= MINIMUM_MASS,
        "active": metrics["candidate_active_min"] >= MINIMUM_ACTIVE
        and metrics["reference_active_min"] >= MINIMUM_ACTIVE,
        "partition_float32": metrics["candidate_partition_max_abs"] <= PARTITION_MAX_ABS,
        "partition_float64": metrics["reference_partition_max_abs"] <= PARTITION_MAX_ABS,
        "nonnegative_float32": (
            metrics["candidate_minimum_effective_weight"] >= -2e-7
        ),
        "nonnegative_float64": metrics["reference_minimum_effective_weight"] >= 0.0,
        "a1_float32": metrics["candidate_a1_max"] <= 1.0 + PARTITION_MAX_ABS,
        "contributor_match": contributor_match,
    }
    return metrics, gates


def _evaluate_static_renderer(
    *,
    decoded: Mapping[str, Any],
    payload: bytes,
    cell: Mapping[str, Any],
    arm: str,
    target: np.ndarray,
    outdir: Path,
    binding_sha256: str,
    beta_summary: Mapping[str, Any] | None = None,
    augmented: core.AugmentedBasisDiagnostics | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    candidate, reference = _render_decoded(decoded)
    metrics, gates = _static_metrics_and_gates(candidate, reference, target)
    raw = _raw_render_arrays(candidate, reference, target)
    renderer_id = _renderer_id(str(cell["cell_id"]), arm)
    relative = Path("static") / f"{renderer_id}.npz"
    _deterministic_npz(outdir / relative, raw)
    stream_record = _stream_component_record(payload, decoded)
    stream_gates = {
        "component_accounting": bool(stream_record["component_accounting_pass"]),
        "ordinary_reencode": bool(stream_record["ordinary_decoded_reencode_pass"]),
        "deterministic_encode": bool(stream_record["deterministic_encode_pass"]),
        "frozen_count": int(decoded["count"])
        == (PRIMARY_COUNT if arm == "nw83" else PROXY_COUNT),
        "frozen_dimensions": (
            int(decoded["height"]) == HEIGHT and int(decoded["width"]) == WIDTH
        ),
        "tail_size": (
            (arm == "ac81" and stream_record["components"]["tail_bytes"] == 12)
            or (arm != "ac81" and stream_record["components"]["tail_bytes"] == 0)
        ),
    }
    row: dict[str, Any] = {
        "schema": "bench014-static-render-v1",
        "binding_sha256": binding_sha256,
        **dict(cell),
        "renderer_id": renderer_id,
        "arm": arm,
        "status": "ok",
        "path": relative.as_posix(),
        "file_sha256": _sha256_file(outdir / relative),
        "stream_path": (Path("streams") / f"{renderer_id}.afcr").as_posix(),
        "decoded_path": (Path("decoded") / f"{renderer_id}.npz").as_posix(),
        "decoded_geometry_sha256": _geometry_hash_from_decoded(decoded),
        "candidate_contributor_sha256": _contributor_hash(candidate.contributors),
        "reference_contributor_sha256": _contributor_hash(reference.contributors),
        "metrics": metrics,
        "gates": gates,
        "stream": stream_record,
        "stream_gates": stream_gates,
        "forward_pass": all(gates.values()) and all(stream_gates.values()),
    }
    if beta_summary is not None:
        row["beta_search"] = _json_safe(beta_summary)
        beta_gates = {
            "rank_two": bool(beta_summary["rank_pass"]),
            "condition": bool(beta_summary["condition_pass"]),
            "unconstrained_projection": bool(
                beta_summary["unconstrained_projection_pass"]
            ),
            "prestream_feasible": bool(beta_summary["prestream_feasible"]),
            "final_range": bool(beta_summary["final_range_pass"]),
            "channel_sse_nonregression": bool(
                beta_summary["sse_nonregression_pass"]
            ),
        }
        row["beta_gates"] = beta_gates
        row["forward_pass"] = bool(row["forward_pass"] and all(beta_gates.values()))
    if augmented is not None:
        row["augmented_basis"] = {
            "base_rank": augmented.base_rank,
            "augmented_rank": augmented.augmented_rank,
            "rank_gain": augmented.rank_gain,
            "common_rank_threshold": augmented.common_rank_threshold,
            "base_singular_values": _json_safe(augmented.base_singular_values),
            "augmented_singular_values": _json_safe(augmented.augmented_singular_values),
            "pass": augmented.full_tail_gain,
        }
        row["forward_pass"] = bool(row["forward_pass"] and augmented.full_tail_gain)
    return row, raw


def _ordinary_stream_arrays(
    source: Mapping[str, np.ndarray],
    target: str,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
) -> dict[str, np.ndarray]:
    """Bind an NW stream to target samples at the original, pre-codec means."""

    means = np.asarray(source["means"], dtype=np.float32)
    return {
        "means": means,
        "log_scales": np.asarray(source["log_scales"], dtype=np.float32),
        "rotations": np.asarray(source["rotations"], dtype=np.float32),
        "colors": target_values(
            target, means, height=height, width=width, dtype=np.float64
        ).astype(np.float32),
    }


def _prepare_cell_streams(
    cell: Mapping[str, Any],
    arrays_by_field: Mapping[str, Mapping[str, np.ndarray]],
    target: np.ndarray,
    *,
    outdir: Path,
    binding_sha256: str,
    search_handle: BinaryIO,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]], dict[str, Any]]:
    target_name = str(cell["target"])
    source81 = arrays_by_field[str(cell["baseline_field_id"])]
    source83 = arrays_by_field[str(cell["challenger_field_id"])]

    nw81 = encode_stream("nw81", _ordinary_stream_arrays(source81, target_name))
    nw83 = encode_stream("nw83", _ordinary_stream_arrays(source83, target_name))
    decoded81 = decode_stream(nw81, "cpu")
    decoded83 = decode_stream(nw83, "cpu")
    ac81, decoded_ac, beta_summary = _stream_safe_carrier(
        decoded81,
        target,
        search_handle=search_handle,
        binding_sha256=binding_sha256,
        cell_id=str(cell["cell_id"]),
    )
    streams = {"nw81": nw81, "ac81": ac81, "nw83": nw83}
    decoded = {"nw81": decoded81, "ac81": decoded_ac, "nw83": decoded83}
    for arm in ARMS:
        renderer_id = _renderer_id(str(cell["cell_id"]), arm)
        _atomic_bytes(outdir / "streams" / f"{renderer_id}.afcr", streams[arm])
        _deterministic_npz(
            outdir / "decoded" / f"{renderer_id}.npz",
            {
                **{
                    name: np.asarray(decoded[arm]["arrays"][name])
                    for name in STREAM_ARRAYS
                },
                "beta": (
                    np.zeros(core.BETA_SHAPE, dtype=np.float32)
                    if decoded[arm]["beta"] is None
                    else np.asarray(decoded[arm]["beta"], dtype=np.float32)
                ),
            },
        )
    if _geometry_hash_from_decoded(decoded81) != _geometry_hash_from_decoded(decoded_ac):
        raise RuntimeError("AC81 changed cold-decoded NW81 geometry")
    weights, design, _ = _convergence_context(decoded81)
    augmented = core.diagnose_augmented_basis(weights, design)
    return streams, decoded, {"beta": beta_summary, "augmented": augmented}


def _solve_with_frozen_qr(
    q_factor: torch.Tensor, r_factor: torch.Tensor, right_hand_side: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.solve_triangular(
        r_factor, q_factor.transpose(0, 1) @ right_hand_side, upper=True
    )


def _convergence_context(
    decoded81: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, core.ReferenceResult]:
    """Build W and D without fitting a coefficient or factoring D."""

    state32, radii = _field_from_stream(decoded81)
    means64 = state32["means"].to(torch.float64)
    reference = core.reference_render(
        means64,
        core.conics_from_parameters(
            state32["log_scales"].to(torch.float64),
            state32["rotations"].to(torch.float64),
        ),
        state32["colors"].to(torch.float64),
        torch.zeros(core.BETA_SHAPE, dtype=torch.float64),
        radii,
        HEIGHT,
        WIDTH,
    )
    weights = reference.effective_weights.detach().clone()
    design = core.reproduction_defect_design(means64, weights, HEIGHT, WIDTH)
    diagnostics = core.diagnose_reproduction_design(design)
    if not diagnostics.accepted or diagnostics.condition_number > MAXIMUM_REPRODUCTION_CONDITION:
        raise RuntimeError("convergence reproduction-defect design failed rank/condition")
    return weights, design, reference


def _convergence_trajectory(
    *,
    arm: str,
    cell: Mapping[str, Any],
    decoded81: Mapping[str, Any],
    target: np.ndarray,
    weights: torch.Tensor,
    contributor_sha256: str,
    design: torch.Tensor | None,
    frozen_qr: tuple[torch.Tensor, torch.Tensor] | None,
    outdir: Path,
    binding_sha256: str,
    checkpoint_handle: BinaryIO,
) -> dict[str, Any]:
    if arm not in ("nw81_opt", "ac81_varpro"):
        raise ValueError("unknown BENCH-014 convergence arm")
    target_flat = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float64)).reshape(-1, 3)
    if arm == "ac81_varpro":
        if design is None or frozen_qr is None:
            raise ValueError("AC81 variable projection requires a frozen defect design and QR")
        q_factor, r_factor = frozen_qr
        means64 = torch.from_numpy(
            np.ascontiguousarray(decoded81["arrays"]["means"], dtype=np.float64)
        )
        mean_design = core.carrier_design(means64, HEIGHT, WIDTH)
        pixel_design = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float64)
    else:
        if design is not None or frozen_qr is not None:
            raise ValueError("NW81 convergence must not receive a coefficient design or QR")
        q_factor = r_factor = mean_design = pixel_design = None
    initial = np.ascontiguousarray(decoded81["arrays"]["colors"], dtype=np.float32)
    appearance = torch.from_numpy(initial.copy()).requires_grad_(True)
    optimizer = torch.optim.Adam(
        [appearance],
        lr=CONVERGENCE_LR,
        betas=CONVERGENCE_BETAS,
        eps=CONVERGENCE_EPS,
        weight_decay=CONVERGENCE_WEIGHT_DECAY,
    )
    trajectory_id = f"{cell['cell_id']}__{arm}"
    losses = np.empty(CONVERGENCE_STEPS + 1, dtype=np.float64)
    betas = np.empty((CONVERGENCE_STEPS + 1, *core.BETA_SHAPE), dtype=np.float64)
    appearances = np.empty(
        (CONVERGENCE_STEPS + 1, initial.shape[0], initial.shape[1]), dtype=np.float32
    )
    materialization_errors = np.empty(CONVERGENCE_STEPS + 1, dtype=np.float64)
    outputs_sha256: list[str] = []

    def state(*, force_zero_beta: bool = False) -> tuple[torch.Tensor, torch.Tensor, float]:
        centered64 = appearance.to(torch.float64)
        if arm == "ac81_varpro" and not force_zero_beta:
            assert design is not None
            assert q_factor is not None and r_factor is not None
            assert mean_design is not None and pixel_design is not None
            right_hand_side = target_flat - weights @ centered64.detach()
            beta = _solve_with_frozen_qr(q_factor, r_factor, right_hand_side)
            direct = weights @ centered64 + design @ beta.detach()
            residual = centered64 - mean_design @ beta.detach()
            materialized = pixel_design @ beta.detach() + weights @ residual
            materialization_error = float(
                (materialized - direct).abs().max().detach()
            )
            output = direct
        else:
            beta = torch.zeros(core.BETA_SHAPE, dtype=torch.float64)
            output = weights @ centered64
            materialization_error = 0.0
        loss = (output - target_flat).square().mean()
        return loss, beta, materialization_error

    for step in range(CONVERGENCE_STEPS + 1):
        if step == 0:
            loss, beta, materialization_error = state(force_zero_beta=True)
        else:
            optimizer.zero_grad(set_to_none=True)
            training_loss, _, _ = state(force_zero_beta=False)
            training_loss.backward()
            optimizer.step()
            loss, beta, materialization_error = state(force_zero_beta=False)
        if materialization_error > 1e-10:
            raise RuntimeError("variable-projection residual materialization mismatch")
        centered64 = appearance.detach().to(torch.float64)
        output = (
            weights @ centered64
            if arm == "nw81_opt" or step == 0
            else weights @ centered64 + design @ beta
        )
        losses[step] = float(loss.detach())
        betas[step] = beta.detach().numpy()
        appearances[step] = appearance.detach().numpy()
        materialization_errors[step] = materialization_error
        output_record = _array_record(output)
        outputs_sha256.append(str(output_record["record_sha256"]))
        _append_jsonl(
            checkpoint_handle,
            {
                "schema": "bench014-convergence-checkpoint-v1",
                "binding_sha256": binding_sha256,
                "trajectory_id": trajectory_id,
                "checkpoint_id": f"{trajectory_id}__step{step:03d}",
                "cell_id": cell["cell_id"],
                "arm": arm,
                "step": step,
                "status": "ok",
                "loss": losses[step],
                "beta": betas[step].tolist(),
                "appearance": _array_record(appearances[step]),
                "output": output_record,
                "materialization_max_abs": materialization_error,
            },
        )
    auc = float(
        (0.5 * losses[0] + losses[1:-1].sum() + 0.5 * losses[-1])
        / CONVERGENCE_STEPS
    )
    relative = Path("convergence") / f"{trajectory_id}.npz"
    _deterministic_npz(
        outdir / relative,
        {
            "losses": losses,
            "betas": betas,
            "appearances": appearances,
            "materialization_max_abs": materialization_errors,
        },
    )
    return {
        "schema": "bench014-convergence-trajectory-v1",
        "binding_sha256": binding_sha256,
        "trajectory_id": trajectory_id,
        "cell_id": cell["cell_id"],
        "cohort": cell["cohort"],
        "target": cell["target"],
        "seed": cell["seed"],
        "arm": arm,
        "status": "ok",
        "path": relative.as_posix(),
        "file_sha256": _sha256_file(outdir / relative),
        "updates": CONVERGENCE_STEPS,
        "checkpoint_count": CONVERGENCE_STEPS + 1,
        "loss_step0": float(losses[0]),
        "loss_step100": float(losses[-1]),
        "auc": auc,
        "maximum_materialization_error": float(materialization_errors.max()),
        "step0_beta_zero": bool(np.array_equal(betas[0], np.zeros(core.BETA_SHAPE))),
        "step0_appearance": _array_record(appearances[0]),
        "weight_matrix": _array_record(weights),
        "contributor_sha256": contributor_sha256,
        "coefficient_solve_count": 0 if arm == "nw81_opt" else 2 * CONVERGENCE_STEPS,
        "qr_factorization_count": 0 if arm == "nw81_opt" else 1,
        "outputs_sha256": outputs_sha256,
    }


def _convergence_pair_gate(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], outdir: Path
) -> dict[str, Any]:
    if baseline["cell_id"] != candidate["cell_id"]:
        raise ValueError("convergence pair cell mismatch")
    baseline_losses = _load_npz(outdir / str(baseline["path"]))["losses"]
    candidate_losses = _load_npz(outdir / str(candidate["path"]))["losses"]
    step0_identical = bool(np.array_equal(baseline_losses[:1], candidate_losses[:1]))
    return {
        "cell_id": baseline["cell_id"],
        "target": baseline["target"],
        "auc_ratio": float(candidate["auc"] / max(float(baseline["auc"]), 1e-20)),
        "final_ratio": float(
            candidate["loss_step100"] / max(float(baseline["loss_step100"]), 1e-20)
        ),
        "step0_loss_bitwise_identical": step0_identical,
        "step0_output_hash_identical": (
            baseline["outputs_sha256"][0] == candidate["outputs_sha256"][0]
        ),
        "step0_appearance_hash_identical": (
            baseline["step0_appearance"] == candidate["step0_appearance"]
        ),
        "weight_hash_identical": baseline["weight_matrix"] == candidate["weight_matrix"],
        "contributor_hash_identical": (
            baseline["contributor_sha256"] == candidate["contributor_sha256"]
        ),
        "nw_coefficient_solve_count_zero": baseline["coefficient_solve_count"] == 0,
    }


def _permuted_decoded(
    decoded: Mapping[str, Any], permutation: np.ndarray
) -> dict[str, Any]:
    count = int(decoded["count"])
    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (count,) or not np.array_equal(np.sort(order), np.arange(count)):
        raise ValueError("invalid Gaussian row permutation")
    return {
        **{key: value for key, value in decoded.items() if key != "arrays"},
        "arrays": {
            name: np.ascontiguousarray(np.asarray(decoded["arrays"][name])[order])
            for name in STREAM_ARRAYS
        },
    }


def _canonical_support_hash(
    contributors: core.FrozenContributors, new_to_old: np.ndarray
) -> str:
    old_gid = np.asarray(new_to_old, dtype=np.int64)[contributors.gid.detach().numpy()]
    rows = np.stack(
        (
            old_gid,
            contributors.py.detach().numpy(),
            contributors.px.detach().numpy(),
        ),
        axis=1,
    )
    if rows.size:
        order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
        rows = rows[order]
    return _array_record(np.ascontiguousarray(rows))["record_sha256"]


def _run_permutations(
    *,
    static_rows: Sequence[Mapping[str, Any]],
    decoded_by_renderer: Mapping[str, Mapping[str, Any]],
    outdir: Path,
    binding_sha256: str,
) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED))
    rows: list[dict[str, Any]] = []
    with (outdir / "permutations.jsonl").open("wb") as handle:
        for static in sorted(static_rows, key=lambda item: str(item["renderer_id"])):
            renderer_id = str(static["renderer_id"])
            decoded = decoded_by_renderer[renderer_id]
            count = int(decoded["count"])
            orders = (
                ("identity", np.arange(count, dtype=np.int64)),
                ("reverse", np.arange(count - 1, -1, -1, dtype=np.int64)),
                ("random0", rng.permutation(count).astype(np.int64)),
                ("random1", rng.permutation(count).astype(np.int64)),
            )
            identity_candidate, identity_reference = _render_decoded(decoded)
            identity_target = target_image(str(static["target"]), dtype=np.float64)
            identity_metrics, identity_gates = _static_metrics_and_gates(
                identity_candidate, identity_reference, identity_target
            )
            identity_support = _canonical_support_hash(
                identity_candidate.contributors, np.arange(count, dtype=np.int64)
            )
            for order_name, permutation in orders:
                permuted = _permuted_decoded(decoded, permutation)
                candidate, reference = _render_decoded(permuted)
                metrics, gates = _static_metrics_and_gates(
                    candidate, reference, identity_target
                )
                candidate_difference = float(
                    (
                        candidate.output.to(torch.float64)
                        - identity_candidate.output.to(torch.float64)
                    )
                    .abs()
                    .max()
                )
                reference_difference = float(
                    (reference.output - identity_reference.output).abs().max()
                )
                support_hash = _canonical_support_hash(candidate.contributors, permutation)
                active_equal = bool(
                    torch.equal(candidate.active_counts, identity_candidate.active_counts)
                    and torch.equal(reference.active_counts, identity_reference.active_counts)
                )
                decisions_equal = gates == identity_gates
                range_decision = (
                    metrics["candidate"]["range_excursion"] <= RANGE_EXCURSION_MAX
                )
                identity_range_decision = (
                    identity_metrics["candidate"]["range_excursion"]
                    <= RANGE_EXCURSION_MAX
                )
                raw = _raw_render_arrays(candidate, reference, identity_target)
                raw["permutation"] = permutation
                permutation_id = f"{renderer_id}__{order_name}"
                relative = Path("permutations") / f"{permutation_id}.npz"
                _deterministic_npz(outdir / relative, raw)
                row = {
                    "schema": "bench014-permutation-v1",
                    "binding_sha256": binding_sha256,
                    "permutation_id": permutation_id,
                    "renderer_id": renderer_id,
                    "cell_id": static["cell_id"],
                    "target": static["target"],
                    "arm": static["arm"],
                    "order": order_name,
                    "status": "ok",
                    "seed": PERMUTATION_SEED,
                    "permutation": permutation.tolist(),
                    "permutation_sha256": _array_record(permutation)["record_sha256"],
                    "path": relative.as_posix(),
                    "file_sha256": _sha256_file(outdir / relative),
                    "candidate_identity_max_abs": candidate_difference,
                    "reference_identity_max_abs": reference_difference,
                    "support_sha256": support_hash,
                    "support_identical": support_hash == identity_support,
                    "active_counts_identical": active_equal,
                    "gate_decisions_identical": decisions_equal,
                    "range_decision_identical": range_decision == identity_range_decision,
                    "candidate_tolerance_pass": (
                        candidate_difference <= PERMUTATION_CANDIDATE_MAX_ABS
                    ),
                    "reference_tolerance_pass": (
                        reference_difference <= PERMUTATION_REFERENCE_MAX_ABS
                    ),
                }
                row["permutation_pass"] = all(
                    bool(row[name])
                    for name in (
                        "support_identical",
                        "active_counts_identical",
                        "gate_decisions_identical",
                        "range_decision_identical",
                        "candidate_tolerance_pass",
                        "reference_tolerance_pass",
                    )
                )
                _append_jsonl(handle, row)
                rows.append(row)
    return rows


def _dimensionless_gradients(
    raw: Sequence[torch.Tensor], *, width: int, height: int
) -> dict[str, torch.Tensor]:
    means_scale = torch.tensor((width - 1, height - 1), dtype=raw[0].dtype)
    return {
        "means_normalized": raw[0] * means_scale,
        "log_scales": raw[1],
        "rotations_over_pi": raw[2] * math.pi,
        "residual_rgb": raw[3],
        "beta": raw[4],
    }


def _gradient_scalar(
    parameters: Sequence[torch.Tensor],
    radii: torch.Tensor,
    contributors: core.FrozenContributors,
    cotangent: torch.Tensor,
    *,
    candidate: bool,
) -> torch.Tensor:
    means, log_scales, rotations, residuals, beta = parameters
    renderer = (
        core.candidate_render_residual_frozen
        if candidate
        else core.reference_render_residual_frozen
    )
    output = renderer(
        means,
        log_scales,
        rotations,
        residuals,
        beta,
        radii,
        contributors,
    )
    return torch.sum(output * cotangent)


def _run_gradient_audit(
    *,
    decoded: Mapping[str, Any],
    eligible: bool,
    outdir: Path,
    binding_sha256: str,
) -> dict[str, Any]:
    terminal_path = outdir / "gradient.jsonl"
    if not eligible:
        row = {
            "schema": "bench014-gradient-v1",
            "binding_sha256": binding_sha256,
            "cell": "target_conditioned/affine_sin/N81/seed307/AC81",
            "status": "not_reached_preregistered_base_failure",
            "gradient_pass": False,
        }
        with terminal_path.open("wb") as handle:
            _append_jsonl(handle, row)
        return row

    state32, radii = _field_from_stream(decoded)
    contributors = core.enumerate_contributors(state32["means"], radii, HEIGHT, WIDTH)
    beta32 = torch.from_numpy(
        np.ascontiguousarray(decoded["beta"], dtype=np.float32)
    )
    bases32 = (
        state32["means"],
        state32["log_scales"],
        state32["rotations"],
        state32["colors"],
        beta32,
    )
    rng = np.random.Generator(np.random.PCG64(GRADIENT_SEED))
    cotangent = rng.standard_normal((HEIGHT, WIDTH, 3))
    cotangent /= np.linalg.norm(cotangent.reshape(-1))
    cotangent = np.ascontiguousarray(cotangent, dtype=np.float64)
    cotangent32 = np.ascontiguousarray(cotangent.astype(np.float32))
    shapes = {
        block: tuple(value.shape)
        for block, value in zip(GRADIENT_BLOCKS, bases32, strict=True)
    }
    directions: dict[str, list[np.ndarray]] = {}
    for block in GRADIENT_BLOCKS:
        directions[block] = []
        for _ in range(2):
            direction = rng.standard_normal(shapes[block])
            direction /= np.linalg.norm(direction.reshape(-1))
            directions[block].append(np.ascontiguousarray(direction, dtype=np.float64))

    bases64 = tuple(
        value.detach().to(torch.float64).requires_grad_(True) for value in bases32
    )
    scalar64 = _gradient_scalar(
        bases64,
        radii,
        contributors,
        torch.from_numpy(cotangent),
        candidate=False,
    )
    raw64 = torch.autograd.grad(scalar64, bases64)
    gradients64 = _dimensionless_gradients(raw64, width=WIDTH, height=HEIGHT)
    variables32 = tuple(value.detach().requires_grad_(True) for value in bases32)
    scalar32 = _gradient_scalar(
        variables32,
        radii,
        contributors,
        torch.from_numpy(cotangent32),
        candidate=True,
    )
    raw32 = torch.autograd.grad(scalar32, variables32)
    gradients32 = _dimensionless_gradients(raw32, width=WIDTH, height=HEIGHT)

    raw_scales: dict[str, torch.Tensor | float] = {
        "means_normalized": torch.tensor((WIDTH - 1, HEIGHT - 1), dtype=torch.float64),
        "log_scales": 1.0,
        "rotations_over_pi": math.pi,
        "residual_rgb": 1.0,
        "beta": 1.0,
    }
    directional_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {
        "cotangent_float64": cotangent,
        "cotangent_float32": cotangent32,
        "contributors_gid": contributors.gid.numpy(),
        "contributors_px": contributors.px.numpy(),
        "contributors_py": contributors.py.numpy(),
        "radii": radii.numpy(),
    }
    for block_index, block in enumerate(GRADIENT_BLOCKS):
        gradient64 = gradients64[block]
        gradient32 = gradients32[block]
        finite = bool(torch.isfinite(gradient64).all() and torch.isfinite(gradient32).all())
        norm64 = float(torch.linalg.vector_norm(gradient64))
        absolute = float(torch.linalg.vector_norm(gradient32.to(torch.float64) - gradient64))
        tolerance = 2e-5 + 2e-3 * norm64
        block_rows.append(
            {
                "block": block,
                "gradient64_l2": norm64,
                "gradient32_l2": float(torch.linalg.vector_norm(gradient32)),
                "absolute_l2_error": absolute,
                "mixed_l2_tolerance": tolerance,
                "relative_l2_error": absolute / max(norm64, 1e-12),
                "finite": finite,
                "pass": finite and absolute <= tolerance,
            }
        )
        arrays[f"gradient64__{block}"] = gradient64.detach().numpy()
        arrays[f"gradient32__{block}"] = gradient32.detach().numpy()
        arrays[f"raw_gradient64__{block}"] = raw64[block_index].detach().numpy()
        arrays[f"raw_gradient32__{block}"] = raw32[block_index].detach().numpy()
        arrays[f"directions__{block}"] = np.stack(directions[block])
        for direction_index, direction in enumerate(directions[block]):
            direction64 = torch.from_numpy(direction)
            plus = [value.detach().clone() for value in bases64]
            minus = [value.detach().clone() for value in bases64]
            perturbation = GRADIENT_STEP * direction64 * raw_scales[block]
            plus[block_index] += perturbation
            minus[block_index] -= perturbation
            with torch.no_grad():
                plus_loss = _gradient_scalar(
                    plus,
                    radii,
                    contributors,
                    torch.from_numpy(cotangent),
                    candidate=False,
                )
                minus_loss = _gradient_scalar(
                    minus,
                    radii,
                    contributors,
                    torch.from_numpy(cotangent),
                    candidate=False,
                )
            finite_difference = float(
                (plus_loss - minus_loss) / (2.0 * GRADIENT_STEP)
            )
            ad64 = float(torch.sum(gradient64 * direction64))
            ad32 = float(torch.sum(gradient32.to(torch.float64) * direction64))
            fd_tolerance = 1e-8 + 1e-4 * max(abs(ad64), abs(finite_difference))
            cross_tolerance = 2e-5 + 2e-3 * abs(ad64)
            row_pass = (
                all(math.isfinite(value) for value in (finite_difference, ad64, ad32))
                and abs(ad64 - finite_difference) <= fd_tolerance
                and abs(ad32 - ad64) <= cross_tolerance
            )
            directional_rows.append(
                {
                    "block": block,
                    "direction": direction_index,
                    "ad64": ad64,
                    "fd64": finite_difference,
                    "ad32": ad32,
                    "ad64_fd64_abs_error": abs(ad64 - finite_difference),
                    "ad64_fd64_tolerance": fd_tolerance,
                    "ad32_ad64_abs_error": abs(ad32 - ad64),
                    "ad32_ad64_tolerance": cross_tolerance,
                    "pass": row_pass,
                }
            )
    relative = Path("gradient_raw.npz")
    _deterministic_npz(outdir / relative, arrays)
    summary = {
        "schema": "bench014-gradient-v1",
        "binding_sha256": binding_sha256,
        "cell": "target_conditioned/affine_sin/N81/seed307/AC81",
        "status": "ok",
        "seed": GRADIENT_SEED,
        "step": GRADIENT_STEP,
        "path": relative.as_posix(),
        "file_sha256": _sha256_file(outdir / relative),
        "directions": directional_rows,
        "blocks": block_rows,
        "directional_pass": all(bool(row["pass"]) for row in directional_rows),
        "block_pass": all(bool(row["pass"]) for row in block_rows),
    }
    summary["gradient_pass"] = bool(
        summary["directional_pass"] and summary["block_pass"]
    )
    with terminal_path.open("wb") as handle:
        _append_jsonl(handle, summary)
    return summary


def _timed_packed_output(
    state: Mapping[str, torch.Tensor],
    radii: torch.Tensor,
    beta: torch.Tensor | None,
    pixel_design: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Measured four-channel kernel; only a non-None beta executes the affine tail."""

    means = state["means"]
    conics = state["conics"]
    residuals = state["colors"]
    contributors = core.enumerate_contributors(means, radii, height, width)
    gid = contributors.gid
    dx = contributors.px.to(torch.float32) - means[gid, 0]
    dy = contributors.py.to(torch.float32) - means[gid, 1]
    quadratic = (
        conics[gid, 0] * dx.square()
        + 2.0 * conics[gid, 1] * dx * dy
        + conics[gid, 2] * dy.square()
    )
    weights = torch.exp(-0.5 * quadratic)
    contributions = torch.cat(
        (weights[:, None], weights[:, None] * residuals[gid]), dim=1
    )
    if contributions.shape[1] != core.CHANNEL_COUNT:
        raise RuntimeError("timed accumulator must have exactly four scalar channels")
    accumulators = torch.zeros((height * width, core.CHANNEL_COUNT), dtype=torch.float32)
    accumulators = accumulators.index_add(0, contributors.flat, contributions)
    normalized = accumulators[:, 1:4] / (accumulators[:, :1] + 1e-8)
    if beta is None:
        return normalized.reshape(height, width, 3)
    tail = pixel_design @ beta
    return (tail + normalized).reshape(height, width, 3)


def _prepared_timing_state(
    decoded: Mapping[str, Any]
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    state, radii = _field_from_stream(decoded)
    beta = (
        None
        if decoded["beta"] is None
        else torch.from_numpy(np.ascontiguousarray(decoded["beta"], dtype=np.float32))
    )
    return state, radii, beta


def _run_timing(
    *,
    cells: Sequence[Mapping[str, Any]],
    streams_by_cell: Mapping[str, Mapping[str, bytes]],
    decoded_by_renderer: Mapping[str, Mapping[str, Any]],
    static_by_renderer: Mapping[str, Mapping[str, Any]],
    outdir: Path,
    binding_sha256: str,
) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(TIMING_SEED))
    rows: list[dict[str, Any]] = []
    schedules: dict[str, np.ndarray] = {}
    sorted_cells = sorted(cells, key=lambda item: str(item["cell_id"]))
    for cell in sorted_cells:
        cell_id = str(cell["cell_id"])
        for kind in ("render", "decode_render"):
            schedules[f"{cell_id}__{kind}"] = np.stack(
                [rng.permutation(len(ARMS)) for _ in range(120)]
            ).astype(np.int64)
    _deterministic_npz(outdir / "timing_schedules.npz", schedules)
    pixel_design = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float32)
    prepared = {
        renderer_id: _prepared_timing_state(decoded)
        for renderer_id, decoded in decoded_by_renderer.items()
    }
    with torch.no_grad(), (outdir / "timing.jsonl").open("wb") as handle:
        for cell in sorted_cells:
            cell_id = str(cell["cell_id"])
            cell_samples: dict[tuple[str, str], list[int]] = {
                (kind, arm): []
                for kind in ("render", "decode_render")
                for arm in ARMS
            }
            for kind in ("render", "decode_render"):
                schedule = schedules[f"{cell_id}__{kind}"]
                for repetition in range(120):
                    for arm_index in schedule[repetition]:
                        arm = ARMS[int(arm_index)]
                        renderer_id = _renderer_id(cell_id, arm)
                        start = time.perf_counter_ns()
                        if kind == "render":
                            state, radii, beta = prepared[renderer_id]
                            _timed_packed_output(
                                state, radii, beta, pixel_design, HEIGHT, WIDTH
                            )
                        else:
                            decoded = _decode_stream_impl(
                                streams_by_cell[cell_id][arm],
                                "cpu",
                                audit_reencode=False,
                            )
                            state, radii, beta = _prepared_timing_state(decoded)
                            _timed_packed_output(
                                state, radii, beta, pixel_design, HEIGHT, WIDTH
                            )
                        elapsed = time.perf_counter_ns() - start
                        if repetition >= 20:
                            cell_samples[(kind, arm)].append(elapsed)
                schedules[f"{cell_id}__{kind}"] = schedule
            nw_static = static_by_renderer[_renderer_id(cell_id, "nw81")]
            ac_static = static_by_renderer[_renderer_id(cell_id, "ac81")]
            same_triplets = (
                nw_static["candidate_contributor_sha256"]
                == ac_static["candidate_contributor_sha256"]
            )
            same_evaluations = (
                nw_static["metrics"]["contributor_triplets"]
                == ac_static["metrics"]["contributor_triplets"]
            )
            for kind in ("render", "decode_render"):
                for arm in ARMS:
                    samples = np.asarray(cell_samples[(kind, arm)], dtype=np.int64)
                    renderer_id = _renderer_id(cell_id, arm)
                    row = {
                        "schema": "bench014-timing-v1",
                        "binding_sha256": binding_sha256,
                        "timing_id": f"{renderer_id}__{kind}",
                        "cell_id": cell_id,
                        "target": cell["target"],
                        "arm": arm,
                        "kind": kind,
                        "status": "ok",
                        "seed": TIMING_SEED,
                        "warmups": 20,
                        "repetitions": 100,
                        "samples_ns": samples.tolist(),
                        "median_ns": float(np.median(samples)),
                        "minimum_ns": int(samples.min()),
                        "maximum_ns": int(samples.max()),
                        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                        "accumulated_scalar_channels": 4,
                        "pixel_tail_multiplications": 6 if arm == "ac81" else 0,
                        "pixel_tail_additions": 6 if arm == "ac81" else 0,
                        "per_pixel_solve_or_factorization": False,
                        "weight_evaluations": static_by_renderer[renderer_id]["metrics"][
                            "contributor_triplets"
                        ],
                        "ac81_nw81_triplet_hash_identical": same_triplets,
                        "ac81_nw81_weight_evaluations_identical": same_evaluations,
                    }
                    row["operation_ledger_pass"] = bool(
                        row["accumulated_scalar_channels"] == 4
                        and row["per_pixel_solve_or_factorization"] is False
                        and same_triplets
                        and same_evaluations
                        and (
                            (arm == "ac81" and row["pixel_tail_multiplications"] == 6)
                            or (arm != "ac81" and row["pixel_tail_multiplications"] == 0)
                        )
                        and (
                            (arm == "ac81" and row["pixel_tail_additions"] == 6)
                            or (arm != "ac81" and row["pixel_tail_additions"] == 0)
                        )
                    )
                    _append_jsonl(handle, row)
                    rows.append(row)
    return rows


def _replace_outer_crc(payload_without_crc: bytes) -> bytes:
    return payload_without_crc + STREAM_CRC.pack(
        zlib.crc32(payload_without_crc) & 0xFFFFFFFF
    )


def _expect_decode_rejection(payload: bytes) -> bool:
    try:
        decode_stream(payload, "cpu")
    except (ValueError, RuntimeError, struct.error, zlib.error):
        return True
    return False


def run_plumbing_controls() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.Generator(np.random.PCG64(20260716014))
    symmetric_weights = np.asarray(
        [
            (0.5, 0.25, 0.125, 0.125),
            (0.125, 0.125, 0.25, 0.5),
            (0.25, 0.5, 0.125, 0.125),
            (0.125, 0.125, 0.5, 0.25),
            (0.125, 0.25, 0.5, 0.125),
            (0.125, 0.5, 0.25, 0.125),
            (0.5, 0.125, 0.125, 0.25),
            (0.25, 0.125, 0.125, 0.5),
        ],
        dtype=np.float64,
    )
    symmetric_q_mean = np.asarray(
        ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
        dtype=np.float64,
    )
    symmetric_q_pixel = np.asarray(
        (
            (-0.9, -0.8),
            (0.9, 0.8),
            (-0.6, 0.7),
            (0.6, -0.7),
            (-0.3, -0.9),
            (0.3, 0.9),
            (-0.8, 0.2),
            (0.8, -0.2),
        ),
        dtype=np.float64,
    )
    symmetric_defect = symmetric_q_pixel - symmetric_weights @ symmetric_q_mean
    symmetric_solution = core.solve_reproduction_design_qr(
        torch.from_numpy(symmetric_defect), torch.zeros((8, 3), dtype=torch.float64)
    )
    symmetric_beta_max = float(symmetric_solution.beta.abs().max())
    symmetric_weight_reflection = bool(
        np.array_equal(symmetric_weights[1::2], symmetric_weights[0::2, ::-1])
    )
    symmetric_pixel_reflection = bool(
        np.array_equal(symmetric_q_pixel[1::2], -symmetric_q_pixel[0::2])
    )
    symmetric_mean_reflection = bool(
        np.array_equal(symmetric_q_mean[::-1], -symmetric_q_mean)
    )
    raw_weights = rng.uniform(0.1, 1.0, size=(23, 7))
    weights = raw_weights / (raw_weights.sum(axis=1, keepdims=True) + 1e-8)
    q_pixel = rng.uniform(-1.0, 1.0, size=(23, 2))
    q_mean = rng.uniform(-1.0, 1.0, size=(7, 2))
    beta = rng.normal(scale=0.15, size=(2, 3))
    colors = rng.uniform(0.2, 0.8, size=(7, 3))
    baseline = weights @ colors
    defect = q_pixel - weights @ q_mean
    left = q_pixel @ beta + weights @ (colors - q_mean @ beta)
    right = baseline + defect @ beta
    algebra_error = float(np.max(np.abs(left - right)))

    control_height, control_width = 9, 11
    yy, xx = np.meshgrid(
        np.linspace(0.0, control_height - 1.0, 9),
        np.linspace(0.0, control_width - 1.0, 9),
        indexing="ij",
    )
    means = np.stack((xx, yy), axis=-1).reshape(-1, 2).astype(np.float32)
    logs = np.full((PROXY_COUNT, 2), np.log(18.0), dtype=np.float32)
    rotations = np.zeros(PROXY_COUNT, dtype=np.float32)
    beta_control = np.array(
        ((0.125, -0.0625, 0.03125), (0.0625, 0.125, -0.03125)),
        dtype=np.float32,
    )
    intercept = np.array((0.375, 0.5, 0.625), dtype=np.float64)
    q_mean_control = core.carrier_design(
        torch.from_numpy(means.astype(np.float64)), control_height, control_width
    ).numpy()
    ordinary_colors = (intercept + q_mean_control @ beta_control.astype(np.float64)).astype(
        np.float32
    )
    broad_arrays = {
        "means": means,
        "log_scales": logs,
        "rotations": rotations,
        "colors": ordinary_colors,
    }
    broad_nw = encode_stream(
        "nw81", broad_arrays, height=control_height, width=control_width
    )
    broad_decoded = decode_stream(broad_nw, "cpu")
    q_pixels = core.pixel_design(
        control_height, control_width, dtype=torch.float64
    ).numpy()
    broad_target = (intercept + q_pixels @ beta_control.astype(np.float64)).reshape(
        control_height, control_width, 3
    )
    broad_ac, broad_ac_decoded, broad_summary = _stream_safe_carrier(
        broad_decoded, broad_target
    )
    broad_output, _ = _render_decoded(broad_ac_decoded)
    broad_error = float(
        np.max(np.abs(broad_output.output.numpy().astype(np.float64) - broad_target))
    )

    constant_arrays = {**broad_arrays, "colors": np.full((PROXY_COUNT, 3), 0.5, np.float32)}
    constant_nw = encode_stream(
        "nw81", constant_arrays, height=control_height, width=control_width
    )
    constant_decoded = decode_stream(constant_nw, "cpu")
    constant_target = np.full((control_height, control_width, 3), 0.5, dtype=np.float64)
    constant_ac, constant_ac_decoded, constant_summary = _stream_safe_carrier(
        constant_decoded, constant_target
    )
    constant_nw_output, _ = _render_decoded(constant_decoded)
    constant_ac_output, _ = _render_decoded(constant_ac_decoded)
    constant_difference = float(
        (
            constant_nw_output.output.to(torch.float64)
            - constant_ac_output.output.to(torch.float64)
        )
        .abs()
        .max()
    )
    constant_nw_error = float(
        np.max(
            np.abs(
                constant_nw_output.output.numpy().astype(np.float64) - constant_target
            )
        )
    )
    constant_ac_error = float(
        np.max(
            np.abs(
                constant_ac_output.output.numpy().astype(np.float64) - constant_target
            )
        )
    )
    constant_nw_mse = _mse(constant_nw_output.output.numpy(), constant_target)
    constant_ac_mse = _mse(constant_ac_output.output.numpy(), constant_target)
    constant_ratio = constant_ac_mse / max(constant_nw_mse, 1e-20)

    zero_beta_payload = core.encode_beta_f16(
        torch.zeros(core.BETA_SHAPE, dtype=torch.float32)
    )
    zero_ac = encode_stream(
        "ac81",
        _stream_arrays_with_colors(constant_decoded, constant_decoded["arrays"]["colors"]),
        height=control_height,
        width=control_width,
        beta_payload=zero_beta_payload,
    )
    zero_decoded = decode_stream(zero_ac, "cpu")
    zero_output, _ = _render_decoded(zero_decoded)
    alpha_zero_difference = float(
        (
            zero_output.output.to(torch.float64)
            - constant_nw_output.output.to(torch.float64)
        )
        .abs()
        .max()
    )
    alpha_zero_sse = (
        (zero_output.output.numpy().astype(np.float64) - constant_target) ** 2
    ).sum(axis=(0, 1))
    alpha_zero_baseline_sse = (
        (constant_nw_output.output.numpy().astype(np.float64) - constant_target) ** 2
    ).sum(axis=(0, 1))
    alpha_zero_sse_limit = alpha_zero_baseline_sse + np.maximum(
        1e-15, 1e-10 * alpha_zero_baseline_sse
    )
    broad_stream_record = _stream_component_record(broad_ac, broad_ac_decoded)

    rank_x = torch.linspace(-1.0, 1.0, 17, dtype=torch.float64)
    rank_one = torch.stack((rank_x, 2.0 * rank_x), dim=1)
    rank_diagnostics = core.diagnose_reproduction_design(rank_one)
    rank_rejected = False
    try:
        core.solve_reproduction_design_qr(
            rank_one, torch.zeros((17, 3), dtype=torch.float64)
        )
    except ValueError:
        rank_rejected = True

    header = list(STREAM_HEADER.unpack_from(broad_ac))
    malformed: dict[str, bytes] = {}
    wrong_variant = header.copy()
    wrong_variant[2] = 2
    malformed["wrong_variant"] = _replace_outer_crc(
        STREAM_HEADER.pack(*wrong_variant) + broad_ac[STREAM_HEADER.size : -STREAM_CRC.size]
    )
    variant_tail = header.copy()
    variant_tail[2] = 0
    malformed["variant_tail_mismatch"] = _replace_outer_crc(
        STREAM_HEADER.pack(*variant_tail) + broad_ac[STREAM_HEADER.size : -STREAM_CRC.size]
    )
    wrong_version = header.copy()
    wrong_version[1] = 2
    malformed["wrong_version"] = _replace_outer_crc(
        STREAM_HEADER.pack(*wrong_version) + broad_ac[STREAM_HEADER.size : -STREAM_CRC.size]
    )
    nonzero_reserved = header.copy()
    nonzero_reserved[3] = 1
    malformed["nonzero_reserved"] = _replace_outer_crc(
        STREAM_HEADER.pack(*nonzero_reserved)
        + broad_ac[STREAM_HEADER.size : -STREAM_CRC.size]
    )
    wrong_tail = header.copy()
    wrong_tail[5] = 10
    malformed["wrong_tail_length"] = _replace_outer_crc(
        STREAM_HEADER.pack(*wrong_tail) + broad_ac[STREAM_HEADER.size : -STREAM_CRC.size]
    )
    corrupted_crc = bytearray(broad_ac)
    corrupted_crc[-1] ^= 1
    malformed["crc_corruption"] = bytes(corrupted_crc)
    malformed["trailing_byte"] = broad_ac + b"\0"
    nonfinite = bytearray(broad_ac[:-STREAM_CRC.size])
    tail_offset = STREAM_HEADER.size + int(header[4])
    nonfinite[tail_offset : tail_offset + 2] = b"\x00\x7c"
    malformed["nonfinite_beta"] = _replace_outer_crc(bytes(nonfinite))
    malformed_inner = bytearray(broad_ac[:-STREAM_CRC.size])
    malformed_inner[STREAM_HEADER.size] ^= 0xFF
    malformed["malformed_inner"] = _replace_outer_crc(bytes(malformed_inner))
    broad83_arrays = {
        "means": np.concatenate((means, means[:2]), axis=0),
        "log_scales": np.concatenate((logs, logs[:2]), axis=0),
        "rotations": np.concatenate((rotations, rotations[:2]), axis=0),
        "colors": np.concatenate((ordinary_colors, ordinary_colors[:2]), axis=0),
    }
    nw83_control = encode_stream(
        "nw83", broad83_arrays, height=control_height, width=control_width
    )
    nw83_header = list(STREAM_HEADER.unpack_from(nw83_control))
    nw83_inner = nw83_control[
        STREAM_HEADER.size : STREAM_HEADER.size + int(nw83_header[4])
    ]
    ac83_header = STREAM_HEADER.pack(
        STREAM_MAGIC, STREAM_VERSION, 1, 0, len(nw83_inner), 12
    )
    malformed["illegal_ac83"] = _replace_outer_crc(
        ac83_header + nw83_inner + bytes(12)
    )
    rejection = {name: _expect_decode_rejection(payload) for name, payload in malformed.items()}
    signature = inspect.signature(decode_stream)
    signature_names = tuple(signature.parameters)
    decoder_api_pass = signature_names == ("payload", "device")

    controls = {
        "schema": "bench014-plumbing-controls-v1",
        "algebra": {
            "max_abs": algebra_error,
            "minimum_weight": float(weights.min()),
            "defect_rank": int(np.linalg.matrix_rank(defect)),
            "row_sum_max_abs_from_one": float(
                np.max(np.abs(weights.sum(axis=1) - 1.0))
            ),
            "pass": algebra_error <= 1e-12
            and float(weights.min()) > 0.0
            and int(np.linalg.matrix_rank(defect)) == 2,
        },
        "symmetric_constant_gauge": {
            "row_sum_max_abs": float(
                np.max(np.abs(symmetric_weights.sum(axis=1) - 1.0))
            ),
            "defect_rank": symmetric_solution.diagnostics.rank,
            "beta_max_abs": symmetric_beta_max,
            "weight_reflection_pass": symmetric_weight_reflection,
            "pixel_reflection_pass": symmetric_pixel_reflection,
            "mean_reflection_pass": symmetric_mean_reflection,
            "pass": (
                np.array_equal(symmetric_weights.sum(axis=1), np.ones(8))
                and symmetric_solution.diagnostics.rank == 2
                and symmetric_beta_max <= 1e-12
                and symmetric_weight_reflection
                and symmetric_pixel_reflection
                and symmetric_mean_reflection
            ),
        },
        "broad_affine": {
            "decoded_beta": _json_safe(broad_ac_decoded["beta"]),
            "rank": broad_summary["rank"],
            "condition_number": broad_summary["condition_number"],
            "maximum_reproduction_error": broad_error,
            "stream_sha256": _sha256_bytes(broad_ac),
            "registered_beta": beta_control.tolist(),
            "decoded_beta_exact": bool(
                np.array_equal(
                    np.asarray(broad_ac_decoded["beta"], dtype=np.float32), beta_control
                )
            ),
            "stream_reencode_pass": broad_stream_record[
                "ordinary_decoded_reencode_pass"
            ],
            "stream_accounting_pass": broad_stream_record[
                "component_accounting_pass"
            ],
            "pass": (
                broad_summary["rank_pass"]
                and broad_summary["condition_pass"]
                and broad_summary["final_range_pass"]
                and broad_summary["sse_nonregression_pass"]
                and np.array_equal(
                    np.asarray(broad_ac_decoded["beta"], dtype=np.float32), beta_control
                )
                and broad_stream_record["ordinary_decoded_reencode_pass"]
                and broad_stream_record["component_accounting_pass"]
                and broad_error <= 1e-3
            ),
        },
        "constant_gauge": {
            "beta_ls_max_abs": float(
                np.max(np.abs(np.asarray(constant_summary["beta_ls"])))
            ),
            "nw_max_abs": constant_nw_error,
            "ac_max_abs": constant_ac_error,
            "ac_nw_mse_ratio": constant_ratio,
            "render_max_abs": constant_difference,
            "pass": (
                constant_nw_error <= 1e-6
                and constant_ac_error <= 1e-6
                and constant_ratio <= 1.001
            ),
        },
        "rank_one": {
            "rank": rank_diagnostics.rank,
            "accepted": rank_diagnostics.accepted,
            "solve_rejected": rank_rejected,
            "pass": rank_diagnostics.rank == 1
            and not rank_diagnostics.accepted
            and rank_rejected,
        },
        "alpha_zero": {
            "render_max_abs": alpha_zero_difference,
            "channel_sse": alpha_zero_sse.tolist(),
            "channel_sse_limit": alpha_zero_sse_limit.tolist(),
            "channel_sse_pass": bool(np.all(alpha_zero_sse <= alpha_zero_sse_limit)),
            "pass": alpha_zero_difference == 0.0
            and bool(np.all(alpha_zero_sse <= alpha_zero_sse_limit)),
        },
        "wrapper_rejections": {
            "cases": rejection,
            "pass": all(rejection.values()),
        },
        "decoder_api": {
            "parameters": list(signature_names),
            "pass": decoder_api_pass,
        },
    }
    controls["valid"] = all(bool(value["pass"]) for value in controls.values() if isinstance(value, Mapping) and "pass" in value)
    arrays = {
        "algebra_left": left,
        "algebra_right": right,
        "algebra_weights": weights,
        "symmetric_weights": symmetric_weights,
        "symmetric_q_pixel": symmetric_q_pixel,
        "symmetric_q_mean": symmetric_q_mean,
        "symmetric_defect": symmetric_defect,
        "broad_target": broad_target,
        "broad_output": broad_output.output.numpy(),
        "constant_nw_output": constant_nw_output.output.numpy(),
        "constant_ac_output": constant_ac_output.output.numpy(),
        "alpha_zero_output": zero_output.output.numpy(),
        "rank_one_design": rank_one.numpy(),
    }
    return controls, arrays


def _linear_median(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.5, method="linear"))


def analyze_results(
    *,
    static_rows: Sequence[Mapping[str, Any]],
    permutation_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    convergence_pairs: Sequence[Mapping[str, Any]],
    timing_rows: Sequence[Mapping[str, Any]],
    gradient: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    static_by_cell: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in static_rows:
        static_by_cell.setdefault(str(row["cell_id"]), {})[str(row["arm"])] = row
    comparisons: list[dict[str, Any]] = []
    for cell_id, arms in sorted(static_by_cell.items()):
        nw81, ac81, nw83 = arms["nw81"], arms["ac81"], arms["nw83"]
        nw81_mse = float(nw81["metrics"]["candidate"]["mse"])
        ac81_mse = float(ac81["metrics"]["candidate"]["mse"])
        nw83_mse = float(nw83["metrics"]["candidate"]["mse"])
        nw81_outer = float(nw81["metrics"]["candidate"]["outer9_mse"])
        ac81_outer = float(ac81["metrics"]["candidate"]["outer9_mse"])
        ac_bytes = int(ac81["stream"]["complete_bytes"])
        nw81_bytes = int(nw81["stream"]["complete_bytes"])
        nw83_bytes = int(nw83["stream"]["complete_bytes"])
        comparisons.append(
            {
                "cell_id": cell_id,
                "cohort": ac81["cohort"],
                "target": ac81["target"],
                "seed": ac81["seed"],
                "same_count_mse_ratio": ac81_mse / max(nw81_mse, 1e-20),
                "same_count_outer9_ratio": ac81_outer / max(nw81_outer, 1e-20),
                "same_count_strict_win": ac81_mse < nw81_mse,
                "rate_mse_ratio": ac81_mse / max(nw83_mse, 1e-20),
                "rate_byte_ratio": ac_bytes / nw83_bytes,
                "ac81_le_nw83_bytes": ac_bytes <= nw83_bytes,
                "ac81_minus_nw81_bytes": ac_bytes - nw81_bytes,
            }
        )

    constant_pass = all(
        float(row["metrics"]["candidate"]["max_abs"]) <= 1e-6
        for row in static_rows
        if row["target"] == "constant"
    )
    affine_pass = all(
        float(row["metrics"]["candidate"]["max_abs"]) <= 1e-3
        for row in static_rows
        if row["target"] == "affine" and row["arm"] == "ac81"
    )
    smooth_quality: dict[str, Any] = {}
    for target in SMOOTH_NONAFFINE_TARGETS:
        values = [row for row in comparisons if row["target"] == target]
        smooth_quality[target] = {
            "units": len(values),
            "strict_wins": sum(bool(row["same_count_strict_win"]) for row in values),
            "median_mse_ratio": _linear_median(
                [float(row["same_count_mse_ratio"]) for row in values]
            ),
            "median_outer9_ratio": _linear_median(
                [float(row["same_count_outer9_ratio"]) for row in values]
            ),
        }
        smooth_quality[target]["pass"] = bool(
            smooth_quality[target]["units"] == 6
            and smooth_quality[target]["strict_wins"] >= 5
            and smooth_quality[target]["median_mse_ratio"] <= 0.85
            and smooth_quality[target]["median_outer9_ratio"] <= 0.75
        )
    no_harm = [
        row
        for row in comparisons
        if row["target"] in ("constant", "zero_linear")
    ]
    no_harm_pass = all(float(row["same_count_mse_ratio"]) <= 1.001 for row in no_harm)
    discontinuity_rows = [
        row
        for row in static_rows
        if row["target"] in DISCONTINUOUS_TARGETS and row["arm"] == "ac81"
    ]
    discontinuity_pass = all(
        bool(row["beta_gates"]["final_range"]) for row in discontinuity_rows
    )
    sse_pass = all(
        bool(row["beta_gates"]["channel_sse_nonregression"])
        for row in static_rows
        if row["arm"] == "ac81"
    )
    quality = {
        "constant": constant_pass,
        "affine": affine_pass,
        "smooth": smooth_quality,
        "constant_zero_linear_no_harm": no_harm_pass,
        "discontinuity_range": discontinuity_pass,
        "channel_sse_nonregression": sse_pass,
    }
    quality["pass"] = bool(
        constant_pass
        and affine_pass
        and all(bool(value["pass"]) for value in smooth_quality.values())
        and no_harm_pass
        and discontinuity_pass
        and sse_pass
    )

    rate_targets: dict[str, Any] = {}
    for target in TARGETS:
        values = [row for row in comparisons if row["target"] == target]
        record: dict[str, Any] = {
            "units": len(values),
            "byte_wins": sum(bool(row["ac81_le_nw83_bytes"]) for row in values),
            "median_byte_ratio": _linear_median(
                [float(row["rate_byte_ratio"]) for row in values]
            ),
            "worst_byte_ratio": max(float(row["rate_byte_ratio"]) for row in values),
            "median_mse_ratio": _linear_median(
                [float(row["rate_mse_ratio"]) for row in values]
            ),
            "worst_mse_ratio": max(float(row["rate_mse_ratio"]) for row in values),
        }
        record["byte_pass"] = bool(
            record["units"] == 6
            and record["byte_wins"] >= 5
            and record["median_byte_ratio"] <= 1.0
            and record["worst_byte_ratio"] <= 1.02
        )
        record["quality_pass"] = bool(
            (target not in SMOOTH_NONAFFINE_TARGETS or record["median_mse_ratio"] <= 0.95)
            and (
                target not in ("zero_linear", "vertical_step", "checker9x7")
                or record["worst_mse_ratio"] <= 1.05
            )
        )
        record["pass"] = bool(record["byte_pass"] and record["quality_pass"])
        rate_targets[target] = record
    stream_pass = all(
        bool(row["stream_gates"][key])
        for row in static_rows
        for key in row["stream_gates"]
    )
    rate = {
        "targets": rate_targets,
        "stream_pass": stream_pass,
        "same_count_byte_delta": {
            "minimum": min(int(row["ac81_minus_nw81_bytes"]) for row in comparisons),
            "median": _linear_median(
                [float(row["ac81_minus_nw81_bytes"]) for row in comparisons]
            ),
            "maximum": max(int(row["ac81_minus_nw81_bytes"]) for row in comparisons),
        },
    }
    rate["pass"] = bool(
        stream_pass and all(bool(value["pass"]) for value in rate_targets.values())
    )

    static_pass = all(bool(row["forward_pass"]) for row in static_rows)
    augmented_pass = all(
        bool(row["augmented_basis"]["pass"])
        for row in static_rows
        if row["arm"] == "ac81"
    )
    augmented_count = sum(row["arm"] == "ac81" for row in static_rows)
    permutation_pass = (
        len(permutation_rows) == EXPECTED_PERMUTATION_ROWS
        and all(bool(row["permutation_pass"]) for row in permutation_rows)
    )
    convergence_smooth = [
        row for row in convergence_pairs if row["target"] in SMOOTH_NONAFFINE_TARGETS
    ]
    convergence_invariants = all(
        bool(row[key])
        for row in convergence_pairs
        for key in (
            "step0_loss_bitwise_identical",
            "step0_output_hash_identical",
            "step0_appearance_hash_identical",
            "weight_hash_identical",
            "contributor_hash_identical",
            "nw_coefficient_solve_count_zero",
        )
    )
    convergence = {
        "trajectories": len(convergence_rows),
        "checkpoints": sum(int(row["checkpoint_count"]) for row in convergence_rows),
        "smooth_units": len(convergence_smooth),
        "smooth_median_auc_ratio": _linear_median(
            [float(row["auc_ratio"]) for row in convergence_smooth]
        ),
        "smooth_worst_final_ratio": max(
            float(row["final_ratio"]) for row in convergence_smooth
        ),
        "shared_step0_and_design_pass": convergence_invariants,
    }
    convergence["pass"] = bool(
        convergence["trajectories"] == EXPECTED_CONVERGENCE_TRAJECTORIES
        and convergence["checkpoints"] == EXPECTED_CONVERGENCE_ROWS
        and convergence["smooth_units"] == 18
        and convergence["smooth_median_auc_ratio"] <= 0.90
        and convergence["smooth_worst_final_ratio"] <= 1.01
        and convergence_invariants
    )

    timing_by_id = {str(row["timing_id"]): row for row in timing_rows}
    render_ratios: list[float] = []
    for cell_id in sorted(static_by_cell):
        ac = timing_by_id[f"{_renderer_id(cell_id, 'ac81')}__render"]
        nw = timing_by_id[f"{_renderer_id(cell_id, 'nw81')}__render"]
        render_ratios.append(float(ac["median_ns"]) / max(float(nw["median_ns"]), 1.0))
    performance = {
        "rows": len(timing_rows),
        "median_render_ratio": _linear_median(render_ratios),
        "operation_ledger_pass": all(
            bool(row["operation_ledger_pass"]) for row in timing_rows
        ),
    }
    performance["pass"] = bool(
        performance["rows"] == EXPECTED_FORWARD_ROWS * 2
        and performance["median_render_ratio"] <= 1.15
        and performance["operation_ledger_pass"]
    )
    counts = {
        "static": len(static_rows),
        "permutations": len(permutation_rows),
        "trajectories": len(convergence_rows),
        "timing": len(timing_rows),
    }
    count_pass = counts == {
        "static": EXPECTED_FORWARD_ROWS,
        "permutations": EXPECTED_PERMUTATION_ROWS,
        "trajectories": EXPECTED_CONVERGENCE_TRAJECTORIES,
        "timing": EXPECTED_FORWARD_ROWS * 2,
    }
    sections = {
        "controls": bool(controls["valid"]),
        "counts": count_pass,
        "static_stability": static_pass,
        "quality": bool(quality["pass"]),
        "rate": bool(rate["pass"]),
        "expressiveness": augmented_pass and augmented_count == EXPECTED_CELLS,
        "permutation": permutation_pass,
        "convergence": bool(convergence["pass"]),
        "gradient": bool(gradient["gradient_pass"]),
        "performance": bool(performance["pass"]),
    }
    passed = all(sections.values())
    return {
        "schema": "bench014-analysis-v1",
        "complete": True,
        "counts": counts,
        "section_pass": sections,
        "quality": quality,
        "rate": rate,
        "convergence": convergence,
        "performance": performance,
        "comparisons": comparisons,
        "decision": "pass" if passed else "kill",
        "stage1_authorized": passed,
        "claim_boundary": {
            "static_quality": (
                "marginal affine-tail value over the frozen analytic-sampling initializer; "
                "NW node colors are not fitted"
            ),
            "convergence": (
                "iteration-indexed exact-variable-projection color loss only; not wall-clock "
                "or general optimizer speed"
            ),
            "dependence": (
                "designed analytic cells share target families and are correlated; the six "
                "cohort-by-seed rows are not independent datasets"
            ),
            "natural_images": False,
            "production_codec": False,
            "publication_novelty": False,
        },
    }


def _nested_close(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_close(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_close(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(left, right, rtol=1e-11, atol=1e-13, equal_nan=True))
    return left == right


def _json_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_json_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
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
    statuses: tuple[str, ...] = ("ok",),
) -> bool:
    keys = [str(row[key]) for row in rows]
    return bool(
        len(keys) == len(expected)
        and len(set(keys)) == len(keys)
        and set(keys) == set(expected)
        and all(row.get("binding_sha256") == binding_sha256 for row in rows)
        and all(str(row.get("status")) in statuses for row in rows)
        and all(_json_finite(row) for row in rows)
    )


def _expected_keys() -> dict[str, Any]:
    fields = [str(spec["field_id"]) for spec in field_specs()]
    cells = [str(spec["cell_id"]) for spec in cell_specs()]
    renderers = [_renderer_id(cell, arm) for cell in cells for arm in ARMS]
    trajectories = [
        f"{cell}__{arm}"
        for cell in cells
        for arm in ("nw81_opt", "ac81_varpro")
    ]
    return {
        "fields": fields,
        "cells": cells,
        "renderers": renderers,
        "permutations": [
            f"{renderer}__{order}"
            for renderer in renderers
            for order in ("identity", "reverse", "random0", "random1")
        ],
        "trajectories": trajectories,
        "checkpoints": [
            f"{trajectory}__step{step:03d}"
            for trajectory in trajectories
            for step in range(CONVERGENCE_STEPS + 1)
        ],
        "timing": [
            f"{renderer}__{kind}"
            for renderer in renderers
            for kind in ("render", "decode_render")
        ],
    }


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
        "schema": "bench014-artifact-manifest-v1",
        "files": files,
        "file_count": len(files),
        "payload_sha256": _sha256_bytes(_canonical_json(files)),
    }


def _validate_artifact_manifest(outdir: Path, manifest: Mapping[str, Any]) -> bool:
    rebuilt = _artifact_manifest(outdir)
    return _nested_close(rebuilt, manifest)


def _verify_source_archive(
    root: Path, outdir: Path, record: Mapping[str, Any]
) -> bool:
    archive_path = outdir / str(record["path"])
    if not archive_path.is_file() or _sha256_file(archive_path) != record["sha256"]:
        return False
    observed: dict[str, str] = {}
    with tarfile.open(archive_path, mode="r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            if handle is None:
                return False
            observed[member.name] = _sha256_bytes(handle.read())
    if observed != record["members"]:
        return False
    return all(
        relative in observed and observed[relative] == _sha256_file(root / relative)
        for relative in ARCHIVE_PATHS
    )


def replay_artifacts(outdir: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently replay a completed BENCH-014 directory from persisted artifacts."""

    _pin_runtime()
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((outdir / "environment.json").read_text(encoding="utf-8"))
    binding = str(config["binding_sha256"])
    science = dict(config)
    science.pop("binding_sha256")
    checks: dict[str, bool] = {}
    checks["binding"] = _binding_sha256(science, environment) == binding
    checks["task_hash"] = (
        _sha256_file(root / TASK_PATH) == science["task_sha256"]
    )
    checks["source_manifest"] = _source_manifest(root) == science["source_manifest"]
    archive_record = json.loads(
        (outdir / "source_archive.json").read_text(encoding="utf-8")
    )
    checks["source_archive"] = _verify_source_archive(root, outdir, archive_record)

    target_record = json.loads(
        (outdir / "target_manifest.json").read_text(encoding="utf-8")
    )
    target_arrays = _load_npz(outdir / str(target_record["path"]))
    target_ok = _sha256_file(outdir / str(target_record["path"])) == target_record[
        "file_sha256"
    ]
    for target in TARGETS:
        for dtype in (np.float64, np.float32):
            name = np.dtype(dtype).name
            key = f"{target}__{name}"
            reconstructed = target_image(target, dtype=dtype)
            target_ok &= key in target_arrays and np.array_equal(
                target_arrays[key], reconstructed
            )
            target_ok &= (
                target_record["targets"][target][name] == _array_record(reconstructed)
            )
        target_ok &= target_record["targets"][target]["formula"] == TARGET_FORMULAS[target]
    checks["targets"] = bool(target_ok)

    controls = json.loads((outdir / "controls.json").read_text(encoding="utf-8"))
    replayed_controls, replayed_control_arrays = run_plumbing_controls()
    persisted_control_arrays = _load_npz(outdir / str(controls["path"]))
    stripped_controls = {
        key: value
        for key, value in controls.items()
        if key not in ("binding_sha256", "path", "file_sha256")
    }
    checks["controls"] = bool(
        _nested_close(stripped_controls, replayed_controls)
        and set(persisted_control_arrays) == set(replayed_control_arrays)
        and all(
            np.array_equal(persisted_control_arrays[name], replayed_control_arrays[name])
            for name in replayed_control_arrays
        )
    )

    fields = _read_jsonl(outdir / "fields.jsonl")
    field_ok = len(fields) == EXPECTED_FIELDS
    fields_by_id = {str(row["field_id"]): row for row in fields}
    persisted_fields_by_id: dict[str, dict[str, np.ndarray]] = {}
    field_ok &= len(fields_by_id) == EXPECTED_FIELDS
    with tempfile.TemporaryDirectory(prefix="bench014-field-replay-") as temporary:
        temporary_root = Path(temporary)
        for spec in field_specs():
            row = fields_by_id[str(spec["field_id"])]
            path = outdir / str(row["path"])
            field_ok &= path.is_file() and _sha256_file(path) == row["file_sha256"]
            arrays = _load_npz(path)
            persisted_fields_by_id[str(row["field_id"])] = arrays
            field_ok &= row["geometry"] == _geometry_record(arrays)
            rebuilt_row, rebuilt_arrays = _build_field(spec, temporary_root)
            persisted_without_binding = {
                key: value
                for key, value in row.items()
                if key not in ("binding_sha256", "status")
            }
            field_ok &= _nested_close(rebuilt_row, persisted_without_binding)
            field_ok &= set(arrays) == set(rebuilt_arrays) and all(
                np.array_equal(arrays[name], rebuilt_arrays[name]) for name in arrays
            )
    field_ok &= {row["field_id"] for row in fields} == {
        spec["field_id"] for spec in field_specs()
    }
    checks["fields"] = bool(field_ok)

    static_rows = _read_jsonl(outdir / "static.jsonl")
    stream_rows = _read_jsonl(outdir / "streams.jsonl")
    stream_by_renderer = {str(row["renderer_id"]): row for row in stream_rows}
    static_by_renderer = {str(row["renderer_id"]): row for row in static_rows}
    decoded_by_renderer: dict[str, dict[str, Any]] = {}
    replay_static_state: dict[str, dict[str, Any]] = {}
    static_ok = (
        len(static_rows) == EXPECTED_FORWARD_ROWS
        and len(stream_rows) == EXPECTED_STREAM_ROWS
        and len(static_by_renderer) == EXPECTED_FORWARD_ROWS
        and len(stream_by_renderer) == EXPECTED_STREAM_ROWS
    )
    for renderer_id in sorted(static_by_renderer):
        row = static_by_renderer[renderer_id]
        stream_row = stream_by_renderer[renderer_id]
        static_ok &= row["binding_sha256"] == binding
        static_ok &= stream_row["binding_sha256"] == binding
        payload = (outdir / str(stream_row["path"])).read_bytes()
        static_ok &= _sha256_bytes(payload) == stream_row["complete_sha256"]
        decoded = decode_stream(payload, "cpu")
        decoded_by_renderer[renderer_id] = decoded
        replay_stream_record = _stream_component_record(payload, decoded)
        static_ok &= replay_stream_record == stream_row["record"] == row["stream"]
        replay_stream_gates = {
            "component_accounting": bool(
                replay_stream_record["component_accounting_pass"]
            ),
            "ordinary_reencode": bool(
                replay_stream_record["ordinary_decoded_reencode_pass"]
            ),
            "deterministic_encode": bool(
                replay_stream_record["deterministic_encode_pass"]
            ),
            "frozen_count": int(decoded["count"])
            == (PRIMARY_COUNT if row["arm"] == "nw83" else PROXY_COUNT),
            "frozen_dimensions": (
                int(decoded["height"]) == HEIGHT and int(decoded["width"]) == WIDTH
            ),
            "tail_size": (
                (
                    row["arm"] == "ac81"
                    and replay_stream_record["components"]["tail_bytes"] == 12
                )
                or (
                    row["arm"] != "ac81"
                    and replay_stream_record["components"]["tail_bytes"] == 0
                )
            ),
        }
        static_ok &= replay_stream_gates == row["stream_gates"]
        raw_path = outdir / str(row["path"])
        static_ok &= _sha256_file(raw_path) == row["file_sha256"]
        raw = _load_npz(raw_path)
        target = target_image(str(row["target"]), dtype=np.float64)
        candidate, reference = _render_decoded(decoded)
        static_ok &= row["decoded_geometry_sha256"] == _geometry_hash_from_decoded(
            decoded
        )
        static_ok &= row["candidate_contributor_sha256"] == _contributor_hash(
            candidate.contributors
        )
        static_ok &= row["reference_contributor_sha256"] == _contributor_hash(
            reference.contributors
        )
        replay_metrics, replay_gates = _static_metrics_and_gates(
            candidate, reference, target
        )
        replay_raw = _raw_render_arrays(candidate, reference, target)
        static_ok &= set(raw) == set(replay_raw) and all(
            np.array_equal(raw[name], replay_raw[name]) for name in replay_raw
        )
        static_ok &= _nested_close(replay_metrics, row["metrics"])
        static_ok &= replay_gates == row["gates"]
        decoded_path = outdir / str(row["decoded_path"])
        decoded_raw = _load_npz(decoded_path)
        expected_decoded_raw = {
            **{name: np.asarray(decoded["arrays"][name]) for name in STREAM_ARRAYS},
            "beta": (
                np.zeros(core.BETA_SHAPE, dtype=np.float32)
                if decoded["beta"] is None
                else np.asarray(decoded["beta"], dtype=np.float32)
            ),
        }
        static_ok &= set(decoded_raw) == set(expected_decoded_raw) and all(
            np.array_equal(decoded_raw[name], expected_decoded_raw[name])
            for name in expected_decoded_raw
        )
        replay_static_state[renderer_id] = {
            "metrics": replay_metrics,
            "gates": replay_gates,
            "stream_gates": replay_stream_gates,
            "candidate": candidate,
            "reference": reference,
        }

    nw_link_ok = True
    for cell in cell_specs():
        cell_id = str(cell["cell_id"])
        target_name = str(cell["target"])
        for arm, field_key in (
            ("nw81", "baseline_field_id"),
            ("nw83", "challenger_field_id"),
        ):
            source = persisted_fields_by_id[str(cell[field_key])]
            ordinary = _ordinary_stream_arrays(source, target_name)
            rebuilt_nw = encode_stream(arm, ordinary)
            persisted_nw = (
                outdir
                / str(stream_by_renderer[_renderer_id(cell_id, arm)]["path"])
            ).read_bytes()
            nw_link_ok &= rebuilt_nw == persisted_nw
    checks["field_to_nw_stream_link"] = bool(nw_link_ok)

    beta_ok = True
    rebuilt_summaries: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="bench014-beta-replay-") as temporary:
        trial_path = Path(temporary) / "beta_search.jsonl"
        with trial_path.open("wb") as trial_handle:
            for cell in cell_specs():
                cell_id = str(cell["cell_id"])
                nw = decoded_by_renderer[_renderer_id(cell_id, "nw81")]
                target = target_image(str(cell["target"]), dtype=np.float64)
                rebuilt, _, summary = _stream_safe_carrier(
                    nw,
                    target,
                    search_handle=trial_handle,
                    binding_sha256=binding,
                    cell_id=cell_id,
                )
                expected_payload = (
                    outdir
                    / str(stream_by_renderer[_renderer_id(cell_id, "ac81")]["path"])
                ).read_bytes()
                expected_summary = static_by_renderer[_renderer_id(cell_id, "ac81")][
                    "beta_search"
                ]
                replay_summary = _json_safe(summary)
                replay_summary["encoder_work"].pop("elapsed_ns_diagnostic")
                expected_summary_normalized = json.loads(json.dumps(expected_summary))
                expected_summary_normalized["encoder_work"].pop(
                    "elapsed_ns_diagnostic"
                )
                beta_ok &= rebuilt == expected_payload
                beta_ok &= _nested_close(replay_summary, expected_summary_normalized)
                rebuilt_summaries[cell_id] = summary
        replay_beta_trial_rows = _read_jsonl(trial_path)
        beta_trial_rows = _read_jsonl(outdir / "beta_search.jsonl")
        beta_ok &= replay_beta_trial_rows == beta_trial_rows
    checks["beta_search_rebuild"] = bool(beta_ok)

    for cell in cell_specs():
        cell_id = str(cell["cell_id"])
        nw_id = _renderer_id(cell_id, "nw81")
        ac_id = _renderer_id(cell_id, "ac81")
        nw_row = static_by_renderer[nw_id]
        ac_row = static_by_renderer[ac_id]
        target = target_image(str(cell["target"]), dtype=np.float64)
        weights, design, _ = _convergence_context(decoded_by_renderer[nw_id])
        augmented = core.diagnose_augmented_basis(weights, design)
        augmented_record = {
            "base_rank": augmented.base_rank,
            "augmented_rank": augmented.augmented_rank,
            "rank_gain": augmented.rank_gain,
            "common_rank_threshold": augmented.common_rank_threshold,
            "base_singular_values": _json_safe(augmented.base_singular_values),
            "augmented_singular_values": _json_safe(augmented.augmented_singular_values),
            "pass": augmented.full_tail_gain,
        }
        pair_gates = {
            "decoded_geometry_identical": (
                ac_row["decoded_geometry_sha256"]
                == nw_row["decoded_geometry_sha256"]
                == _geometry_hash_from_decoded(decoded_by_renderer[nw_id])
                == _geometry_hash_from_decoded(decoded_by_renderer[ac_id])
            ),
            "candidate_contributor_triplets_identical": (
                _contributor_hash(replay_static_state[ac_id]["candidate"].contributors)
                == _contributor_hash(replay_static_state[nw_id]["candidate"].contributors)
            ),
            "reference_contributor_triplets_identical": (
                _contributor_hash(replay_static_state[ac_id]["reference"].contributors)
                == _contributor_hash(replay_static_state[nw_id]["reference"].contributors)
            ),
            "weight_evaluation_count_identical": (
                replay_static_state[ac_id]["metrics"]["contributor_triplets"]
                == replay_static_state[nw_id]["metrics"]["contributor_triplets"]
            ),
        }
        summary = rebuilt_summaries[cell_id]
        beta_gates = {
            "rank_two": bool(summary["rank_pass"]),
            "condition": bool(summary["condition_pass"]),
            "unconstrained_projection": bool(summary["unconstrained_projection_pass"]),
            "prestream_feasible": bool(summary["prestream_feasible"]),
            "final_range": bool(summary["final_range_pass"]),
            "channel_sse_nonregression": bool(summary["sse_nonregression_pass"]),
        }
        for renderer_id in (nw_id, ac_id):
            row = static_by_renderer[renderer_id]
            static_ok &= row["augmented_basis"] == augmented_record
            static_ok &= row["same_geometry_pair_gates"] == pair_gates
            expected_forward = bool(
                all(replay_static_state[renderer_id]["gates"].values())
                and all(replay_static_state[renderer_id]["stream_gates"].values())
                and augmented.full_tail_gain
                and all(pair_gates.values())
            )
            if renderer_id == ac_id:
                static_ok &= row["beta_gates"] == beta_gates
                expected_forward &= all(beta_gates.values())
            static_ok &= bool(row["forward_pass"]) == expected_forward
        nw83_id = _renderer_id(cell_id, "nw83")
        nw83_expected = bool(
            all(replay_static_state[nw83_id]["gates"].values())
            and all(replay_static_state[nw83_id]["stream_gates"].values())
        )
        static_ok &= bool(static_by_renderer[nw83_id]["forward_pass"]) == nw83_expected
    checks["static_and_streams"] = bool(static_ok)

    permutation_rows = _read_jsonl(outdir / "permutations.jsonl")
    permutation_by_id = {
        str(row["permutation_id"]): row for row in permutation_rows
    }
    permutation_ok = (
        len(permutation_rows) == EXPECTED_PERMUTATION_ROWS
        and len(permutation_by_id) == EXPECTED_PERMUTATION_ROWS
    )
    expected_permutations: dict[str, np.ndarray] = {}
    replay_permutation_pass: dict[str, bool] = {}
    permutation_rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED))
    for renderer_id in sorted(static_by_renderer):
        count = int(decoded_by_renderer[renderer_id]["count"])
        for name, order in (
            ("identity", np.arange(count, dtype=np.int64)),
            ("reverse", np.arange(count - 1, -1, -1, dtype=np.int64)),
            ("random0", permutation_rng.permutation(count).astype(np.int64)),
            ("random1", permutation_rng.permutation(count).astype(np.int64)),
        ):
            expected_permutations[f"{renderer_id}__{name}"] = order
    for row in permutation_rows:
        renderer_id = str(row["renderer_id"])
        decoded = decoded_by_renderer[renderer_id]
        permutation = np.asarray(row["permutation"], dtype=np.int64)
        permutation_ok &= row["binding_sha256"] == binding and row["status"] == "ok"
        permutation_ok &= np.array_equal(
            permutation, expected_permutations[str(row["permutation_id"])]
        )
        candidate, reference = _render_decoded(_permuted_decoded(decoded, permutation))
        identity_candidate = replay_static_state[renderer_id]["candidate"]
        identity_reference = replay_static_state[renderer_id]["reference"]
        target = target_image(str(row["target"]), dtype=np.float64)
        metrics, gates = _static_metrics_and_gates(candidate, reference, target)
        identity_metrics = replay_static_state[renderer_id]["metrics"]
        identity_gates = replay_static_state[renderer_id]["gates"]
        candidate_difference = float(
            (candidate.output.to(torch.float64) - identity_candidate.output.to(torch.float64))
            .abs()
            .max()
        )
        reference_difference = float(
            (reference.output - identity_reference.output).abs().max()
        )
        support_hash = _canonical_support_hash(candidate.contributors, permutation)
        identity_support = _canonical_support_hash(
            identity_candidate.contributors,
            np.arange(int(decoded["count"]), dtype=np.int64),
        )
        active_equal = bool(
            torch.equal(candidate.active_counts, identity_candidate.active_counts)
            and torch.equal(reference.active_counts, identity_reference.active_counts)
        )
        range_decision_equal = (
            metrics["candidate"]["range_excursion"] <= RANGE_EXCURSION_MAX
        ) == (
            identity_metrics["candidate"]["range_excursion"] <= RANGE_EXCURSION_MAX
        )
        expected_decisions = {
            "candidate_identity_max_abs": candidate_difference,
            "reference_identity_max_abs": reference_difference,
            "support_sha256": support_hash,
            "support_identical": support_hash == identity_support,
            "active_counts_identical": active_equal,
            "gate_decisions_identical": gates == identity_gates,
            "range_decision_identical": range_decision_equal,
            "candidate_tolerance_pass": (
                candidate_difference <= PERMUTATION_CANDIDATE_MAX_ABS
            ),
            "reference_tolerance_pass": (
                reference_difference <= PERMUTATION_REFERENCE_MAX_ABS
            ),
        }
        expected_pass = all(
            bool(expected_decisions[name])
            for name in (
                "support_identical",
                "active_counts_identical",
                "gate_decisions_identical",
                "range_decision_identical",
                "candidate_tolerance_pass",
                "reference_tolerance_pass",
            )
        )
        replay_permutation_pass[str(row["permutation_id"])] = expected_pass
        permutation_ok &= all(
            _nested_close(row[name], value) for name, value in expected_decisions.items()
        )
        permutation_ok &= bool(row["permutation_pass"]) == expected_pass
        raw_path = outdir / str(row["path"])
        permutation_ok &= _sha256_file(raw_path) == row["file_sha256"]
        raw = _load_npz(raw_path)
        replay_raw = _raw_render_arrays(candidate, reference, target)
        replay_raw["permutation"] = permutation
        permutation_ok &= set(raw) == set(replay_raw) and all(
            np.array_equal(raw[name], replay_raw[name]) for name in replay_raw
        )
    checks["permutations"] = bool(permutation_ok)

    convergence_rows = _read_jsonl(outdir / "convergence.jsonl")
    checkpoints = _read_jsonl(outdir / "convergence_checkpoints.jsonl")
    convergence_by_id = {
        str(row["trajectory_id"]): row for row in convergence_rows
    }
    checkpoints_by_id = {
        str(row["checkpoint_id"]): row for row in checkpoints
    }
    convergence_ok = (
        len(convergence_rows) == EXPECTED_CONVERGENCE_TRAJECTORIES
        and len(checkpoints) == EXPECTED_CONVERGENCE_ROWS
        and len(convergence_by_id) == EXPECTED_CONVERGENCE_TRAJECTORIES
        and len(checkpoints_by_id) == EXPECTED_CONVERGENCE_ROWS
    )
    for row in convergence_rows:
        convergence_ok &= row["binding_sha256"] == binding and row["status"] == "ok"
        trajectory_path = outdir / str(row["path"])
        convergence_ok &= _sha256_file(trajectory_path) == row["file_sha256"]
        arrays = _load_npz(trajectory_path)
        losses = arrays["losses"]
        betas = arrays["betas"]
        appearances = arrays["appearances"]
        materialization_errors = arrays["materialization_max_abs"]
        cell_id = str(row["cell_id"])
        decoded = decoded_by_renderer[_renderer_id(cell_id, "nw81")]
        target = target_image(str(row["target"]), dtype=np.float64).reshape(-1, 3)
        weights, design, reference = _convergence_context(decoded)
        means64 = torch.from_numpy(
            np.ascontiguousarray(decoded["arrays"]["means"], dtype=np.float64)
        )
        mean_design = core.carrier_design(means64, HEIGHT, WIDTH)
        pixel_design = core.pixel_design(HEIGHT, WIDTH, dtype=torch.float64)
        target_tensor = torch.from_numpy(np.ascontiguousarray(target))
        auc = float(
            (0.5 * losses[0] + losses[1:-1].sum() + 0.5 * losses[-1])
            / CONVERGENCE_STEPS
        )
        convergence_ok &= losses.shape == (CONVERGENCE_STEPS + 1,)
        convergence_ok &= betas.shape == (
            CONVERGENCE_STEPS + 1,
            *core.BETA_SHAPE,
        )
        convergence_ok &= appearances.shape == (
            CONVERGENCE_STEPS + 1,
            int(decoded["count"]),
            3,
        )
        convergence_ok &= auc == float(row["auc"])
        convergence_ok &= float(losses[0]) == float(row["loss_step0"])
        convergence_ok &= float(losses[-1]) == float(row["loss_step100"])
        convergence_ok &= row["weight_matrix"] == _array_record(weights)
        convergence_ok &= row["contributor_sha256"] == _contributor_hash(
            reference.contributors
        )
        convergence_ok &= row["step0_appearance"] == _array_record(appearances[0])
        convergence_ok &= row["coefficient_solve_count"] == (
            0 if row["arm"] == "nw81_opt" else 2 * CONVERGENCE_STEPS
        )
        convergence_ok &= row["qr_factorization_count"] == (
            0 if row["arm"] == "nw81_opt" else 1
        )
        output_hashes: list[str] = []
        for step in range(CONVERGENCE_STEPS + 1):
            appearance = torch.from_numpy(
                np.ascontiguousarray(appearances[step], dtype=np.float32)
            ).to(torch.float64)
            beta = torch.from_numpy(np.ascontiguousarray(betas[step], dtype=np.float64))
            if row["arm"] == "nw81_opt" or step == 0:
                output = weights @ appearance
                expected_materialization = 0.0
                if step == 0:
                    convergence_ok &= np.array_equal(
                        betas[step], np.zeros(core.BETA_SHAPE)
                    )
            else:
                output = weights @ appearance + design @ beta
                residual = appearance - mean_design @ beta
                materialized = pixel_design @ beta + weights @ residual
                expected_materialization = float((materialized - output).abs().max())
            replay_loss = float((output - target_tensor).square().mean())
            convergence_ok &= replay_loss == float(losses[step])
            convergence_ok &= expected_materialization == float(
                materialization_errors[step]
            )
            output_record = _array_record(output)
            output_hashes.append(str(output_record["record_sha256"]))
            checkpoint_id = f"{row['trajectory_id']}__step{step:03d}"
            checkpoint = checkpoints_by_id[checkpoint_id]
            convergence_ok &= (
                checkpoint["binding_sha256"] == binding
                and checkpoint["status"] == "ok"
                and checkpoint["trajectory_id"] == row["trajectory_id"]
                and int(checkpoint["step"]) == step
                and float(checkpoint["loss"]) == replay_loss
                and checkpoint["appearance"] == _array_record(appearances[step])
                and checkpoint["output"] == output_record
                and _nested_close(checkpoint["beta"], betas[step].tolist())
                and _nested_close(
                    checkpoint["materialization_max_abs"], expected_materialization
                )
            )
        convergence_ok &= row["outputs_sha256"] == output_hashes
        convergence_ok &= float(row["maximum_materialization_error"]) == float(
            materialization_errors.max()
        )
    convergence_pairs = _read_jsonl(outdir / "convergence_pairs.jsonl")
    convergence_pair_by_cell = {
        str(row["cell_id"]): row for row in convergence_pairs
    }
    convergence_ok &= len(convergence_pair_by_cell) == EXPECTED_CELLS
    for cell in cell_specs():
        cell_id = str(cell["cell_id"])
        baseline = convergence_by_id[f"{cell_id}__nw81_opt"]
        candidate = convergence_by_id[f"{cell_id}__ac81_varpro"]
        replay_pair = {
            "schema": "bench014-convergence-pair-v1",
            "binding_sha256": binding,
            "status": "ok",
            **_convergence_pair_gate(baseline, candidate, outdir),
        }
        convergence_ok &= _nested_close(
            replay_pair, convergence_pair_by_cell[cell_id]
        )
    with tempfile.TemporaryDirectory(prefix="bench014-convergence-replay-") as temporary:
        convergence_root = Path(temporary)
        replay_checkpoint_path = convergence_root / "convergence_checkpoints.jsonl"
        with replay_checkpoint_path.open("wb") as replay_checkpoint_handle:
            for cell in cell_specs():
                cell_id = str(cell["cell_id"])
                decoded = decoded_by_renderer[_renderer_id(cell_id, "nw81")]
                target = target_image(str(cell["target"]), dtype=np.float64)
                weights, design, reference = _convergence_context(decoded)
                frozen_qr = torch.linalg.qr(design, mode="reduced")
                contributor_sha256 = _contributor_hash(reference.contributors)
                for arm, supplied_design, supplied_qr in (
                    ("nw81_opt", None, None),
                    ("ac81_varpro", design, frozen_qr),
                ):
                    replay_trajectory = _convergence_trajectory(
                        arm=arm,
                        cell=cell,
                        decoded81=decoded,
                        target=target,
                        weights=weights,
                        contributor_sha256=contributor_sha256,
                        design=supplied_design,
                        frozen_qr=supplied_qr,
                        outdir=convergence_root,
                        binding_sha256=binding,
                        checkpoint_handle=replay_checkpoint_handle,
                    )
                    persisted_trajectory = convergence_by_id[
                        str(replay_trajectory["trajectory_id"])
                    ]
                    convergence_ok &= _nested_close(
                        replay_trajectory, persisted_trajectory
                    )
                    replay_arrays = _load_npz(
                        convergence_root / str(replay_trajectory["path"])
                    )
                    persisted_arrays = _load_npz(
                        outdir / str(persisted_trajectory["path"])
                    )
                    convergence_ok &= set(replay_arrays) == set(
                        persisted_arrays
                    ) and all(
                        np.array_equal(replay_arrays[name], persisted_arrays[name])
                        for name in replay_arrays
                    )
        convergence_ok &= _read_jsonl(replay_checkpoint_path) == checkpoints
    checks["convergence"] = bool(convergence_ok)

    timing_rows = _read_jsonl(outdir / "timing.jsonl")
    schedules = _load_npz(outdir / "timing_schedules.npz")
    timing_ok = len(timing_rows) == EXPECTED_FORWARD_ROWS * 2
    timing_ok &= len(schedules) == EXPECTED_CELLS * 2
    expected_schedules: dict[str, np.ndarray] = {}
    timing_rng = np.random.Generator(np.random.PCG64(TIMING_SEED))
    for cell in sorted(cell_specs(), key=lambda item: str(item["cell_id"])):
        for kind in ("render", "decode_render"):
            expected_schedules[f"{cell['cell_id']}__{kind}"] = np.stack(
                [timing_rng.permutation(len(ARMS)) for _ in range(120)]
            ).astype(np.int64)
    timing_ok &= set(schedules) == set(expected_schedules) and all(
        np.array_equal(schedules[name], expected_schedules[name])
        for name in expected_schedules
    )
    timing_by_id = {str(row["timing_id"]): row for row in timing_rows}
    timing_ok &= len(timing_by_id) == EXPECTED_FORWARD_ROWS * 2
    for row in timing_rows:
        samples = np.asarray(row["samples_ns"], dtype=np.int64)
        renderer_id = _renderer_id(str(row["cell_id"]), str(row["arm"]))
        nw_id = _renderer_id(str(row["cell_id"]), "nw81")
        ac_id = _renderer_id(str(row["cell_id"]), "ac81")
        same_triplets = (
            static_by_renderer[nw_id]["candidate_contributor_sha256"]
            == static_by_renderer[ac_id]["candidate_contributor_sha256"]
        )
        same_evaluations = (
            replay_static_state[nw_id]["metrics"]["contributor_triplets"]
            == replay_static_state[ac_id]["metrics"]["contributor_triplets"]
        )
        expected_operation = bool(
            int(row["accumulated_scalar_channels"]) == 4
            and row["per_pixel_solve_or_factorization"] is False
            and same_triplets
            and same_evaluations
            and int(row["pixel_tail_multiplications"])
            == (6 if row["arm"] == "ac81" else 0)
            and int(row["pixel_tail_additions"])
            == (6 if row["arm"] == "ac81" else 0)
        )
        timing_ok &= (
            row["binding_sha256"] == binding
            and row["status"] == "ok"
            and len(samples) == 100
            and bool(np.isfinite(samples).all())
            and bool((samples > 0).all())
            and float(row["median_ns"]) == float(np.median(samples))
            and int(row["minimum_ns"]) == int(samples.min())
            and int(row["maximum_ns"]) == int(samples.max())
            and int(row["weight_evaluations"])
            == int(replay_static_state[renderer_id]["metrics"]["contributor_triplets"])
            and bool(row["operation_ledger_pass"]) == expected_operation
        )
    timing_ok &= (
        environment["torch_num_threads"] == 1
        and environment["torch_num_interop_threads"] == 1
        and science["timing"]["threads"] == 1
    )
    checks["timing"] = bool(timing_ok)

    gradient_rows = _read_jsonl(outdir / "gradient.jsonl")
    gradient = gradient_rows[0] if len(gradient_rows) == 1 else {"gradient_pass": False}
    gradient_renderer = _renderer_id(
        _cell_id("target_conditioned", "affine_sin", 307), "ac81"
    )
    expected_gradient_eligible = bool(
        static_by_renderer[gradient_renderer]["forward_pass"]
        and all(
            replay_permutation_pass[f"{gradient_renderer}__{order}"]
            for order in ("identity", "reverse", "random0", "random1")
        )
    )
    with tempfile.TemporaryDirectory(prefix="bench014-gradient-replay-") as temporary:
        gradient_root = Path(temporary)
        replay_gradient = _run_gradient_audit(
            decoded=decoded_by_renderer[gradient_renderer],
            eligible=expected_gradient_eligible,
            outdir=gradient_root,
            binding_sha256=binding,
        )
        gradient_ok = len(gradient_rows) == 1 and _nested_close(
            replay_gradient, gradient
        )
        if replay_gradient["status"] == "ok":
            replay_gradient_raw = _load_npz(
                gradient_root / str(replay_gradient["path"])
            )
            persisted_gradient_raw = _load_npz(
                outdir / str(gradient["path"])
            )
            gradient_ok &= (
                _sha256_file(outdir / str(gradient["path"]))
                == gradient["file_sha256"]
                and set(replay_gradient_raw) == set(persisted_gradient_raw)
                and all(
                    np.array_equal(
                        replay_gradient_raw[name], persisted_gradient_raw[name]
                    )
                    for name in replay_gradient_raw
                )
            )
    checks["gradient"] = bool(gradient_ok)
    decoder_source = "\n".join(
        (
            inspect.getsource(decode_stream),
            inspect.getsource(_decode_stream_impl),
            inspect.getsource(field_codec.decode),
            inspect.getsource(core.decode_beta_f16),
        )
    ).lower()
    checks["decoder_call_graph"] = bool(
        tuple(inspect.signature(decode_stream).parameters) == ("payload", "device")
        and all(token not in decoder_source for token in ("target", "lstsq", "solve", "qr"))
    )
    beta_trial_ids = [str(row["trial_id"]) for row in beta_trial_rows]
    beta_trial_contract = bool(
        len(beta_trial_ids) == len(set(beta_trial_ids))
        and all(row["binding_sha256"] == binding for row in beta_trial_rows)
        and all(row["status"] in ("accepted", "rejected") for row in beta_trial_rows)
        and all(_json_finite(row) for row in beta_trial_rows)
        and all(
            sum(
                row["cell_id"] == cell_id and bool(row["accepted"])
                for row in beta_trial_rows
            )
            == 3
            for cell_id in science["expected_keys"]["cells"]
        )
    )
    cell_metadata = {str(cell["cell_id"]): dict(cell) for cell in cell_specs()}
    renderer_metadata = {
        _renderer_id(cell_id, arm): {**metadata, "arm": arm}
        for cell_id, metadata in cell_metadata.items()
        for arm in ARMS
    }
    trajectory_metadata = {
        f"{cell_id}__{arm}": {**metadata, "arm": arm}
        for cell_id, metadata in cell_metadata.items()
        for arm in ("nw81_opt", "ac81_varpro")
    }
    metadata_contract = True
    field_metadata = {str(spec["field_id"]): spec for spec in field_specs()}
    for row in fields:
        expected = field_metadata[str(row["field_id"])]
        metadata_contract &= all(row[key] == value for key, value in expected.items())
    for row in static_rows:
        expected = renderer_metadata[str(row["renderer_id"])]
        metadata_contract &= all(row[key] == value for key, value in expected.items())
        metadata_contract &= decoded_by_renderer[str(row["renderer_id"])]["arm"] == row["arm"]
    for row in stream_rows:
        expected = renderer_metadata[str(row["renderer_id"])]
        metadata_contract &= (
            row["cell_id"] == expected["cell_id"] and row["arm"] == expected["arm"]
        )
    for row in permutation_rows:
        expected = renderer_metadata[str(row["renderer_id"])]
        expected_order = str(row["permutation_id"]).rsplit("__", 1)[1]
        metadata_contract &= (
            row["cell_id"] == expected["cell_id"]
            and row["target"] == expected["target"]
            and row["arm"] == expected["arm"]
            and row["order"] == expected_order
            and row["permutation_id"]
            == f"{row['renderer_id']}__{row['order']}"
        )
    for row in convergence_rows:
        expected = trajectory_metadata[str(row["trajectory_id"])]
        metadata_contract &= all(
            row[key] == expected[key]
            for key in ("cell_id", "cohort", "target", "seed", "arm")
        )
    for row in checkpoints:
        expected_trajectory = str(row["checkpoint_id"]).rsplit("__step", 1)[0]
        expected_step = int(str(row["checkpoint_id"]).rsplit("__step", 1)[1])
        metadata_contract &= (
            row["trajectory_id"] == expected_trajectory
            and int(row["step"]) == expected_step
            and expected_trajectory in trajectory_metadata
        )
    for row in convergence_pairs:
        expected = cell_metadata[str(row["cell_id"])]
        metadata_contract &= row["target"] == expected["target"]
    for row in timing_rows:
        renderer_id, expected_kind = str(row["timing_id"]).rsplit("__", 1)
        expected = renderer_metadata[renderer_id]
        metadata_contract &= (
            row["cell_id"] == expected["cell_id"]
            and row["target"] == expected["target"]
            and row["arm"] == expected["arm"]
            and row["kind"] == expected_kind
        )
    metadata_contract &= all(
        str(row["cell_id"]) in cell_metadata for row in beta_trial_rows
    )
    checks["metadata_mapping"] = bool(metadata_contract)
    checks["ledger_contracts"] = bool(
        _ledger_contract(
            fields,
            key="field_id",
            expected=science["expected_keys"]["fields"],
            binding_sha256=binding,
        )
        and _ledger_contract(
            static_rows,
            key="renderer_id",
            expected=science["expected_keys"]["renderers"],
            binding_sha256=binding,
        )
        and _ledger_contract(
            stream_rows,
            key="renderer_id",
            expected=science["expected_keys"]["renderers"],
            binding_sha256=binding,
        )
        and _ledger_contract(
            permutation_rows,
            key="permutation_id",
            expected=science["expected_keys"]["permutations"],
            binding_sha256=binding,
        )
        and _ledger_contract(
            convergence_rows,
            key="trajectory_id",
            expected=science["expected_keys"]["trajectories"],
            binding_sha256=binding,
        )
        and _ledger_contract(
            checkpoints,
            key="checkpoint_id",
            expected=science["expected_keys"]["checkpoints"],
            binding_sha256=binding,
        )
        and _ledger_contract(
            convergence_pairs,
            key="cell_id",
            expected=science["expected_keys"]["cells"],
            binding_sha256=binding,
        )
        and _ledger_contract(
            timing_rows,
            key="timing_id",
            expected=science["expected_keys"]["timing"],
            binding_sha256=binding,
        )
        and beta_trial_contract
        and len(gradient_rows) == 1
        and gradient_rows[0].get("binding_sha256") == binding
        and gradient_rows[0].get("status")
        in ("ok", "not_reached_preregistered_base_failure")
        and _json_finite(gradient_rows[0])
    )
    manifest_path = outdir / "artifact_manifest.json"
    if manifest_path.is_file():
        persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks["artifact_manifest"] = _validate_artifact_manifest(
            outdir, persisted_manifest
        )
    completion_path = outdir / "completion.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        checks["completion"] = bool(
            completion.get("binding_sha256") == binding
            and completion.get("written_last_after_independent_replay") is True
            and completion.get("artifact_manifest_validated") is True
            and completion.get("analysis_sha256")
            == _sha256_file(outdir / "analysis.json")
            and completion.get("replay_sha256")
            == _sha256_file(outdir / "replay.json")
            and completion.get("artifact_manifest_sha256")
            == _sha256_file(manifest_path)
        )
    replay_pass = all(checks.values())
    if replay_pass:
        aggregate = analyze_results(
            static_rows=static_rows,
            permutation_rows=permutation_rows,
            convergence_rows=convergence_rows,
            convergence_pairs=convergence_pairs,
            timing_rows=timing_rows,
            gradient=gradient,
            controls=controls,
        )
    else:
        aggregate = {
            "schema": "bench014-unavailable-aggregate-v1",
            "decision": "invalid",
            "complete": False,
            "stage1_authorized": False,
        }
    record = {
        "schema": "bench014-replay-v1",
        "binding_sha256": binding,
        "checks": checks,
        "replay_pass": replay_pass,
        "outcome_ledgers_interpreted": replay_pass,
        "aggregate_sha256": _sha256_bytes(_canonical_json(_json_safe(aggregate))),
    }
    return record, aggregate


def _pin_runtime() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("BENCH-014 timing requires one PyTorch intra/inter-op thread")


def _run_impl(outdir: Path, root: Path) -> dict[str, Any]:
    """Execute the complete source-bound assay in a fresh directory."""

    if outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {outdir}")
    _pin_runtime()
    outdir.mkdir(parents=True)
    environment = _environment(root)
    source_manifest = _source_manifest(root)
    formulas = {
        target: {
            "formula": TARGET_FORMULAS[target],
            "formula_sha256": _sha256_bytes(TARGET_FORMULAS[target].encode("utf-8")),
        }
        for target in TARGETS
    }
    representative_init, representative_structure = frozen_configs(
        PROXY_COUNT, SEEDS[0]
    )
    init_fields = asdict(representative_init)
    init_fields.pop("num_gaussians")
    init_fields.pop("seed")
    science: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "task_path": TASK_PATH,
        "task_sha256": source_manifest[TASK_PATH],
        "predecessor_bench013_task_sha256": (
            "b40b9075262c8c2dc07212490a1178b0168b6da9e433cb4ca23a10957bb1d0ad"
        ),
        "comp007_analysis_sha256": (
            "115c2e272a406b1d85313496a94c76e6a4f47c59e41b79e13f951f0ff464ea27"
        ),
        "comp007_artifact_audit_sha256": (
            "ad1ec6c889e818e4b1af4cc63a4f99959453534951f480554471bf14fc621aa5"
        ),
        "dimensions": {"height": HEIGHT, "width": WIDTH},
        "counts": list(COUNTS),
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "cohorts": list(COHORTS),
        "arms": list(ARMS),
        "coders": list(CODERS),
        "initializer": {
            "strategy": "quadtree_wse",
            "count_and_seed_supplied_by_matrix": True,
            "count_seed_independent_init_fields": init_fields,
            "structure_tensor_fields": asdict(representative_structure),
        },
        "codec": {
            "inner_format": "GFCOV01",
            "chart": "current_rs",
            "bits_means": 12,
            "geometry_bits": [6, 6, 6],
            "bits_colors": 8,
            "predictor": "absolute",
            "coder": "zlib9",
            "framing": {
                "header_struct": "<8sBBHII",
                "magic_hex": STREAM_MAGIC.hex(),
                "version": STREAM_VERSION,
                "nw_variant": 0,
                "ac_variant": 1,
                "nw_tail_bytes": 0,
                "ac_tail_bytes": core.BETA_PAYLOAD_BYTES,
                "crc": "little_endian_crc32_all_preceding_bytes",
                "common_overhead_bytes": STREAM_HEADER.size + STREAM_CRC.size,
            },
        },
        "gates": {
            "minimum_mass": MINIMUM_MASS,
            "minimum_active": MINIMUM_ACTIVE,
            "partition_max_abs_float32_float64": PARTITION_MAX_ABS,
            "minimum_effective_weight_float32": -2e-7,
            "a1_max_float32": 1.0 + PARTITION_MAX_ABS,
            "reproduction_rank": 2,
            "maximum_reproduction_condition": MAXIMUM_REPRODUCTION_CONDITION,
            "ray_design_margin": core.RAY_DESIGN_MARGIN,
            "ray_gate_margin": core.RAY_GATE_MARGIN,
            "ray_grid_denominator": core.RAY_GRID_DENOMINATOR,
            "outer_mask": "x<9 or x>=80 or y<9 or y>=64",
            "permutation_reference_max_abs": PERMUTATION_REFERENCE_MAX_ABS,
            "permutation_candidate_max_abs": PERMUTATION_CANDIDATE_MAX_ABS,
            "quality_and_rate": "tasks/BENCH-014-explicit-affine-carrier.md",
        },
        "discovery_exposure": {
            "dimensions": [71, 83],
            "counts": [79, 81],
            "seeds": [211, 223, 227],
            "target_min_max_exposure_disclosed_in_task": True,
            "ineligible_for_canonical_evidence": True,
        },
        "target_formulas": formulas,
        "permutation": {"seed": PERMUTATION_SEED, "orders": 4},
        "gradient": {
            "seed": GRADIENT_SEED,
            "step": GRADIENT_STEP,
            "blocks": list(GRADIENT_BLOCKS),
        },
        "timing": {
            "seed": TIMING_SEED,
            "warmups": 20,
            "repetitions": 100,
            "threads": 1,
        },
        "convergence": {
            "updates": CONVERGENCE_STEPS,
            "lr": CONVERGENCE_LR,
            "betas": list(CONVERGENCE_BETAS),
            "eps": CONVERGENCE_EPS,
            "weight_decay": CONVERGENCE_WEIGHT_DECAY,
        },
        "expected": {
            "fields": EXPECTED_FIELDS,
            "cells": EXPECTED_CELLS,
            "static": EXPECTED_FORWARD_ROWS,
            "streams": EXPECTED_STREAM_ROWS,
            "permutations": EXPECTED_PERMUTATION_ROWS,
            "trajectories": EXPECTED_CONVERGENCE_TRAJECTORIES,
            "checkpoints": EXPECTED_CONVERGENCE_ROWS,
            "timing": EXPECTED_FORWARD_ROWS * 2,
        },
        "expected_keys": _expected_keys(),
        "source_manifest": source_manifest,
    }
    binding = _binding_sha256(science, environment)
    _atomic_json(outdir / "config.json", {**science, "binding_sha256": binding})
    _atomic_json(outdir / "environment.json", environment)
    _atomic_json(outdir / "source_manifest.json", source_manifest)
    archive_record = _write_source_archive(root, outdir)
    _atomic_json(outdir / "source_archive.json", archive_record)

    target_manifest = _target_manifest(outdir)
    _atomic_json(outdir / "target_manifest.json", target_manifest)
    controls, control_arrays = run_plumbing_controls()
    _deterministic_npz(outdir / "controls_raw.npz", control_arrays)
    controls = {
        **controls,
        "binding_sha256": binding,
        "path": "controls_raw.npz",
        "file_sha256": _sha256_file(outdir / "controls_raw.npz"),
    }
    _atomic_json(outdir / "controls.json", controls)
    if not controls["valid"]:
        analysis = {
            "schema": "bench014-analysis-v1",
            "binding_sha256": binding,
            "complete": False,
            "decision": "invalid",
            "classification": "unavailable_invalid_plumbing_controls",
            "stage1_authorized": False,
        }
        manifest = _artifact_manifest(outdir)
        _atomic_json(outdir / "artifact_manifest.json", manifest)
        manifest_valid = _validate_artifact_manifest(outdir, manifest)
        _atomic_json(outdir / "replay.json", {"replay_pass": False, "controls_only": True})
        _atomic_json(outdir / "analysis.json", analysis)
        _atomic_json(
            outdir / "completion.json",
            {
                "complete": False,
                "decision": "invalid",
                "binding_sha256": binding,
                "artifact_manifest_sha256": _sha256_file(
                    outdir / "artifact_manifest.json"
                ),
                "artifact_manifest_validated": manifest_valid,
            },
        )
        return analysis

    arrays_by_field: dict[str, dict[str, np.ndarray]] = {}
    field_rows: list[dict[str, Any]] = []
    with (outdir / "fields.jsonl").open("wb") as handle:
        for spec in field_specs():
            row, arrays = _build_field(spec, outdir)
            row["binding_sha256"] = binding
            row["status"] = "ok"
            arrays_by_field[str(spec["field_id"])] = arrays
            field_rows.append(row)
            _append_jsonl(handle, row)

    static_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    convergence_pairs: list[dict[str, Any]] = []
    streams_by_cell: dict[str, dict[str, bytes]] = {}
    decoded_by_renderer: dict[str, dict[str, Any]] = {}
    static_by_renderer: dict[str, dict[str, Any]] = {}
    cells = cell_specs()
    with (outdir / "beta_search.jsonl").open("wb") as search_handle, (
        outdir / "static.jsonl"
    ).open("wb") as static_handle, (outdir / "streams.jsonl").open(
        "wb"
    ) as stream_handle, (outdir / "convergence.jsonl").open(
        "wb"
    ) as convergence_handle, (outdir / "convergence_checkpoints.jsonl").open(
        "wb"
    ) as checkpoint_handle, (outdir / "convergence_pairs.jsonl").open(
        "wb"
    ) as pair_handle:
        for cell in cells:
            target = target_image(str(cell["target"]), dtype=np.float64)
            streams, decoded, diagnostics = _prepare_cell_streams(
                cell,
                arrays_by_field,
                target,
                outdir=outdir,
                binding_sha256=binding,
                search_handle=search_handle,
            )
            cell_id = str(cell["cell_id"])
            streams_by_cell[cell_id] = streams
            cell_static_rows: list[dict[str, Any]] = []
            for arm in ARMS:
                row, _ = _evaluate_static_renderer(
                    decoded=decoded[arm],
                    payload=streams[arm],
                    cell=cell,
                    arm=arm,
                    target=target,
                    outdir=outdir,
                    binding_sha256=binding,
                    beta_summary=diagnostics["beta"] if arm == "ac81" else None,
                    augmented=(
                        diagnostics["augmented"] if arm in ("nw81", "ac81") else None
                    ),
                )
                cell_static_rows.append(row)
                stream_row = {
                    "schema": "bench014-stream-v1",
                    "binding_sha256": binding,
                    "renderer_id": row["renderer_id"],
                    "cell_id": cell_id,
                    "arm": arm,
                    "status": "ok",
                    "path": row["stream_path"],
                    "complete_sha256": row["stream"]["complete_sha256"],
                    "record": row["stream"],
                }
                _append_jsonl(stream_handle, stream_row)
            rows_by_arm = {str(row["arm"]): row for row in cell_static_rows}
            pair_gates = {
                "decoded_geometry_identical": (
                    rows_by_arm["ac81"]["decoded_geometry_sha256"]
                    == rows_by_arm["nw81"]["decoded_geometry_sha256"]
                ),
                "candidate_contributor_triplets_identical": (
                    rows_by_arm["ac81"]["candidate_contributor_sha256"]
                    == rows_by_arm["nw81"]["candidate_contributor_sha256"]
                ),
                "reference_contributor_triplets_identical": (
                    rows_by_arm["ac81"]["reference_contributor_sha256"]
                    == rows_by_arm["nw81"]["reference_contributor_sha256"]
                ),
                "weight_evaluation_count_identical": (
                    rows_by_arm["ac81"]["metrics"]["contributor_triplets"]
                    == rows_by_arm["nw81"]["metrics"]["contributor_triplets"]
                ),
            }
            for row in cell_static_rows:
                if row["arm"] in ("nw81", "ac81"):
                    row["same_geometry_pair_gates"] = pair_gates
                    row["forward_pass"] = bool(
                        row["forward_pass"] and all(pair_gates.values())
                    )
                _append_jsonl(static_handle, row)
                static_rows.append(row)
                static_by_renderer[str(row["renderer_id"])] = row
                decoded_by_renderer[str(row["renderer_id"])] = decoded[str(row["arm"])]

            weights, design, convergence_reference = _convergence_context(
                decoded["nw81"]
            )
            frozen_qr = torch.linalg.qr(design, mode="reduced")
            contributor_sha256 = _contributor_hash(convergence_reference.contributors)
            nw_trajectory = _convergence_trajectory(
                arm="nw81_opt",
                cell=cell,
                decoded81=decoded["nw81"],
                target=target,
                weights=weights,
                contributor_sha256=contributor_sha256,
                design=None,
                frozen_qr=None,
                outdir=outdir,
                binding_sha256=binding,
                checkpoint_handle=checkpoint_handle,
            )
            ac_trajectory = _convergence_trajectory(
                arm="ac81_varpro",
                cell=cell,
                decoded81=decoded["nw81"],
                target=target,
                weights=weights,
                contributor_sha256=contributor_sha256,
                design=design,
                frozen_qr=frozen_qr,
                outdir=outdir,
                binding_sha256=binding,
                checkpoint_handle=checkpoint_handle,
            )
            for trajectory in (nw_trajectory, ac_trajectory):
                _append_jsonl(convergence_handle, trajectory)
                convergence_rows.append(trajectory)
            pair = _convergence_pair_gate(nw_trajectory, ac_trajectory, outdir)
            pair = {
                "schema": "bench014-convergence-pair-v1",
                "binding_sha256": binding,
                "status": "ok",
                **pair,
            }
            _append_jsonl(pair_handle, pair)
            convergence_pairs.append(pair)

    permutation_rows = _run_permutations(
        static_rows=static_rows,
        decoded_by_renderer=decoded_by_renderer,
        outdir=outdir,
        binding_sha256=binding,
    )
    gradient_renderer = _renderer_id(
        _cell_id("target_conditioned", "affine_sin", 307), "ac81"
    )
    gradient_permutations = [
        row for row in permutation_rows if row["renderer_id"] == gradient_renderer
    ]
    gradient_eligible = bool(
        static_by_renderer[gradient_renderer]["forward_pass"]
        and len(gradient_permutations) == 4
        and all(row["permutation_pass"] for row in gradient_permutations)
    )
    _run_gradient_audit(
        decoded=decoded_by_renderer[gradient_renderer],
        eligible=gradient_eligible,
        outdir=outdir,
        binding_sha256=binding,
    )
    _run_timing(
        cells=cells,
        streams_by_cell=streams_by_cell,
        decoded_by_renderer=decoded_by_renderer,
        static_by_renderer=static_by_renderer,
        outdir=outdir,
        binding_sha256=binding,
    )

    manifest = _artifact_manifest(outdir)
    _atomic_json(outdir / "artifact_manifest.json", manifest)
    if not _validate_artifact_manifest(outdir, manifest):
        raise RuntimeError("BENCH-014 artifact manifest failed immediate validation")
    replay, aggregate = replay_artifacts(outdir, root)
    _atomic_json(outdir / "replay.json", replay)
    if not replay["replay_pass"]:
        analysis = {
            "schema": "bench014-analysis-v1",
            "binding_sha256": binding,
            "complete": False,
            "decision": "invalid",
            "classification": "unavailable_independent_replay_failure",
            "replay_checks": replay["checks"],
            "stage1_authorized": False,
        }
    else:
        analysis = {**aggregate, "binding_sha256": binding, "replay_pass": True}
    _atomic_json(outdir / "analysis.json", analysis)
    manifest_valid = _validate_artifact_manifest(outdir, manifest)
    completion = {
        "schema": "bench014-completion-v1",
        "binding_sha256": binding,
        "complete": bool(replay["replay_pass"] and manifest_valid),
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
    """Run BENCH-014 and seal any post-creation exception as an invalid attempt."""

    existed_before_call = outdir.exists()
    try:
        return _run_impl(outdir, root)
    except Exception as error:
        if existed_before_call or not outdir.exists():
            raise
        failure = {
            "schema": "bench014-failure-v1",
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(outdir / "failure.json", failure)
        binding = None
        config_path = outdir / "config.json"
        if config_path.is_file():
            binding = json.loads(config_path.read_text(encoding="utf-8")).get(
                "binding_sha256"
            )
        replay = {
            "schema": "bench014-replay-v1",
            "binding_sha256": binding,
            "replay_pass": False,
            "outcome_ledgers_interpreted": False,
            "failure": failure,
        }
        analysis = {
            "schema": "bench014-analysis-v1",
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
                "schema": "bench014-completion-v1",
                "binding_sha256": binding,
                "complete": False,
                "decision": "invalid",
                "failure_sha256": _sha256_file(outdir / "failure.json"),
                "analysis_sha256": _sha256_file(outdir / "analysis.json"),
                "replay_sha256": _sha256_file(outdir / "replay.json"),
                "artifact_manifest_sha256": _sha256_file(
                    outdir / "artifact_manifest.json"
                ),
                "written_last_after_independent_replay": False,
            },
        )
        return analysis


def _replay_cli(outdir: Path, root: Path) -> int:
    _pin_runtime()
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    current_matches = False
    try:
        current_matches = _source_manifest(root) == config["source_manifest"]
    except RuntimeError:
        pass
    if current_matches:
        replay, aggregate = replay_artifacts(outdir, root)
        print(json.dumps({"replay": replay, "aggregate": aggregate}, indent=2))
        return 0 if replay["replay_pass"] else 1

    archive_record = json.loads(
        (outdir / "source_archive.json").read_text(encoding="utf-8")
    )
    archive_path = outdir / str(archive_record["path"])
    if _sha256_file(archive_path) != archive_record["sha256"]:
        raise RuntimeError("executed-source archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="bench014-archived-replay-") as temporary:
        archived_root = Path(temporary)
        with tarfile.open(archive_path, mode="r") as archive:
            for member in archive.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError("unsafe member in executed-source archive")
            archive.extractall(archived_root, filter="data")
        archived_runner = archived_root / "benchmarks/affine_carrier_assay.py"
        environment = os.environ.copy()
        python_path = [str(archived_root), str(archived_root / "src")]
        if environment.get("PYTHONPATH"):
            python_path.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        result = subprocess.run(
            [
                sys.executable,
                str(archived_runner),
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
    parser.add_argument("outdir", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--replay", action="store_true", help="read-only independent artifact replay"
    )
    arguments = parser.parse_args(argv)
    if arguments.replay:
        return _replay_cli(arguments.outdir, arguments.root)
    result = run(arguments.outdir, arguments.root)
    print(json.dumps(_json_safe(result), indent=2))
    return 0 if result["decision"] in ("pass", "kill") else 1


if __name__ == "__main__":
    sys.exit(main())
