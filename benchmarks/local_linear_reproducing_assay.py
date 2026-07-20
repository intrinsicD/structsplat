"""Source-bound BENCH-013 Stage-0 early-screen runner.

This file implements only the preregistered plumbing controls and the complete 108-cell
mass/contributor/float64-rank/covariance-condition screen.  It never edits or substitutes the
production renderer.  A failed early gate is a scientific short circuit: downstream forward,
effective-weight, permutation, and gradient work is recorded explicitly as not reached.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from typing import Any, BinaryIO, Iterable, Mapping
import zipfile

import numpy as np
import torch

from benchmarks.local_linear_reproducing_core import (
    GaussianGeometry,
    build_local_linear_operator,
    geometry_diagnostics,
)
from structsplat.config import InitConfig, StructureTensorConfig
from structsplat.gaussians import GaussianField
from structsplat.init import build_field


PROTOCOL = "bench013-local-linear-reproducing-stage0-early-v1"
CONFIG_SCHEMA = "bench013-stage0-early-config-v1"
ROW_SCHEMA = "bench013-stage0-early-pixel-v1"
HEIGHT = 64
WIDTH = 64
COUNTS = (28, 56, 112)
SEEDS = (0, 1, 2)
TARGETS = (
    "constant",
    "affine_xy",
    "quadratic",
    "vertical_step",
    "diagonal_step",
    "checker8",
)
COHORTS = ("target_conditioned", "shared_constant")
MINIMUM_MASS = 1e-5
MINIMUM_CONTRIBUTORS = 3
MAXIMUM_CONDITION = 8192.0
SIGMA_CUTOFF = 3.0
AA_DILATION = 0.0
EXPECTED_FIELDS = 63
EXPECTED_DISTINCT_GEOMETRIES = 54
EXPECTED_CELLS = 108
EXPECTED_PIXELS = EXPECTED_CELLS * HEIGHT * WIDTH
SHORT_CIRCUIT = "not_reached_preregistered_short_circuit"
PENDING = "authorized_pending_sequential_implementation"

TARGET_FORMULAS = {
    "constant": "(0.25,0.50,0.75)",
    "affine_xy": "(0.10+0.35u+0.25v,0.80-0.30u-0.20v,0.20+0.20u+0.40v)",
    "quadratic": "(0.15+0.45u^2+0.15v,0.75-0.30v^2-0.15u,0.20+0.40uv)",
    "vertical_step": "u<0.5?(0.15,0.30,0.75):(0.85,0.70,0.20)",
    "diagonal_step": "u+v<1?(0.10,0.65,0.25):(0.90,0.20,0.75)",
    "checker8": "(floor(x/8)+floor(y/8))mod2 using vertical_step colors",
}

SOURCE_PATHS = (
    "benchmarks/__init__.py",
    "benchmarks/local_linear_reproducing_assay.py",
    "benchmarks/local_linear_reproducing_core.py",
    "src/structsplat/__init__.py",
    "src/structsplat/config.py",
    "src/structsplat/density.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/init.py",
    "src/structsplat/render.py",
    "src/structsplat/sampling.py",
    "src/structsplat/structural_controls.py",
    "src/structsplat/structure_tensor.py",
    "tasks/BENCH-013-local-linear-reproducing-compositor.md",
    "pyproject.toml",
)
ARCHIVE_PATHS = SOURCE_PATHS + (
    "tests/test_local_linear_reproducing_assay.py",
    "tests/test_local_linear_reproducing_core.py",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    payload = _sha256_bytes(array.tobytes(order="C"))
    record = {"dtype": str(array.dtype), "shape": list(array.shape), "payload_sha256": payload}
    record["record_sha256"] = _sha256_bytes(_canonical_json(record))
    return record


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))


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
    handle.write(_canonical_json(dict(row)) + b"\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def target_values(
    target: str,
    xy: np.ndarray,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
) -> np.ndarray:
    """Evaluate one frozen analytic target at arbitrary continuous ``(x,y)`` locations."""

    if target not in TARGETS:
        raise ValueError(f"unknown BENCH-013 target {target!r}")
    points = np.asarray(xy, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 2:
        raise ValueError("xy must have trailing shape 2")
    scale = float(max(int(height) - 1, int(width) - 1, 1))
    x = points[..., 0]
    y = points[..., 1]
    u = x / scale
    v = y / scale
    if target == "constant":
        result = np.broadcast_to(np.array((0.25, 0.50, 0.75)), points.shape[:-1] + (3,))
    elif target == "affine_xy":
        result = np.stack(
            (0.10 + 0.35 * u + 0.25 * v, 0.80 - 0.30 * u - 0.20 * v,
             0.20 + 0.20 * u + 0.40 * v),
            axis=-1,
        )
    elif target == "quadratic":
        result = np.stack(
            (0.15 + 0.45 * u**2 + 0.15 * v,
             0.75 - 0.30 * v**2 - 0.15 * u, 0.20 + 0.40 * u * v),
            axis=-1,
        )
    elif target == "vertical_step":
        low = np.array((0.15, 0.30, 0.75))
        high = np.array((0.85, 0.70, 0.20))
        result = np.where((u < 0.5)[..., None], low, high)
    elif target == "diagonal_step":
        low = np.array((0.10, 0.65, 0.25))
        high = np.array((0.90, 0.20, 0.75))
        result = np.where((u + v < 1.0)[..., None], low, high)
    else:
        low = np.array((0.15, 0.30, 0.75))
        high = np.array((0.85, 0.70, 0.20))
        parity = (np.floor(x / 8.0) + np.floor(y / 8.0)).astype(np.int64) % 2
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
        target, np.stack((xx, yy), axis=-1), height=height, width=width, dtype=dtype
    )


def _field_id(cohort: str, source_target: str, count: int, seed: int) -> str:
    return f"{cohort}__{source_target}__n{count:04d}__s{seed}"


def _cell_id(cohort: str, target: str, count: int, seed: int) -> str:
    return f"{cohort}__{target}__n{count:04d}__s{seed}"


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
    return tuple(sorted(specs, key=lambda item: item["field_id"]))


def cell_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for cohort in COHORTS:
        for target in TARGETS:
            for count in COUNTS:
                for seed in SEEDS:
                    source_target = target if cohort == "target_conditioned" else "constant"
                    specs.append(
                        {
                            "cell_id": _cell_id(cohort, target, count, seed),
                            "field_id": _field_id(cohort, source_target, count, seed),
                            "cohort": cohort,
                            "target": target,
                            "count": count,
                            "seed": seed,
                        }
                    )
    return tuple(sorted(specs, key=lambda item: item["cell_id"]))


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


def _source_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing BENCH-013 source dependency: {relative}")
        manifest[relative] = _sha256_file(path)
    return manifest


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None, "status_sha256": None}


def _environment(root: Path) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "git": _git_metadata(root),
    }


def _binding_sha256(science: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    stable_environment = {
        key: environment[key]
        for key in (
            "python",
            "platform",
            "numpy",
            "torch",
            "device",
            "torch_num_threads",
            "torch_num_interop_threads",
            "torch_deterministic_algorithms",
            "omp_num_threads",
            "mkl_num_threads",
        )
    }
    return _sha256_bytes(
        _canonical_json({"science": dict(science), "environment": stable_environment})
    )


def _target_manifest(outdir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in TARGETS:
        arrays: dict[str, Any] = {}
        for dtype in (np.float64, np.float32):
            image = target_image(target, dtype=dtype)
            suffix = np.dtype(dtype).name
            relative = Path("targets") / f"{target}_{suffix}.npy"
            _atomic_bytes(outdir / relative, _npy_bytes(image))
            arrays[suffix] = {
                **_array_record(image),
                "path": relative.as_posix(),
                "file_sha256": _sha256_file(outdir / relative),
            }
        formula = TARGET_FORMULAS[target]
        result[target] = {
            "formula": formula,
            "formula_sha256": _sha256_bytes(formula.encode("utf-8")),
            "arrays": arrays,
        }
    return result


def _geometry_hash(arrays: Mapping[str, np.ndarray]) -> str:
    keys = ("means", "log_scales", "rotations", "conics_float32", "radii")
    return _sha256_bytes(
        _canonical_json({name: _array_record(arrays[name]) for name in keys})
    )


def _field_content_hash(arrays: Mapping[str, np.ndarray]) -> str:
    return _sha256_bytes(
        _canonical_json({name: _array_record(value) for name, value in sorted(arrays.items())})
    )


def _compact_diagnostics(diagnostics: Any) -> dict[str, np.ndarray]:
    return {
        "mass": diagnostics.mass.detach().numpy(),
        "active_counts": diagnostics.active_counts.detach().numpy(),
        "singular_values": diagnostics.design_singular_values.detach().numpy(),
        "rank_thresholds": diagnostics.rank_thresholds.detach().numpy(),
        "ranks": diagnostics.ranks.detach().numpy(),
        "covariance_eigenvalues": diagnostics.covariance_eigenvalues.detach().numpy(),
        "condition_numbers": diagnostics.condition_numbers.detach().numpy(),
        "mass_pass": diagnostics.mass_pass.detach().numpy(),
        "contributor_pass": diagnostics.contributor_pass.detach().numpy(),
        "rank_pass": diagnostics.rank_pass.detach().numpy(),
        "condition_pass": diagnostics.condition_pass.detach().numpy(),
        "early_pass": diagnostics.early_pass.detach().numpy(),
    }


def _build_field_record(
    spec: Mapping[str, Any], outdir: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    count = int(spec["count"])
    seed = int(spec["seed"])
    source_target = str(spec["source_target"])
    init, structure = frozen_configs(count, seed)
    image = target_image(source_target, dtype=np.float32)
    field = build_field(image, init, structure, device="cpu")
    if int(field.n) != count:
        raise RuntimeError(f"initializer returned {field.n} rows for requested count {count}")
    if field.opacities is not None:
        raise RuntimeError("BENCH-013 freezes opacity=None")
    if field.filter_variance is not None:
        raise RuntimeError("BENCH-013 freezes aa_dilation=0 with no covariance filter")
    for name in ("means", "log_scales", "rotations", "colors"):
        value = getattr(field, name)
        if value.device.type != "cpu" or value.dtype != torch.float32:
            raise RuntimeError(f"production {name} was not CPU float32")

    means = field.means.detach().numpy().copy()
    base_colors = target_values(source_target, means, dtype=np.float64).astype(np.float32)
    field.colors = torch.from_numpy(np.ascontiguousarray(base_colors)).clone()
    conics_float32 = field.conics(dilation=AA_DILATION).detach().numpy().copy()
    radii = field.radii(SIGMA_CUTOFF, dilation=AA_DILATION).detach().numpy().copy()
    arrays = {
        "means": field.means.detach().numpy().copy(),
        "log_scales": field.log_scales.detach().numpy().copy(),
        "rotations": field.rotations.detach().numpy().copy(),
        "colors": field.colors.detach().numpy().copy(),
        "conics_float32": conics_float32,
        "radii": radii,
    }
    relative = Path("fields") / f"{spec['field_id']}.npz"
    _deterministic_npz(outdir / relative, arrays)

    means64 = torch.from_numpy(arrays["means"].astype(np.float64))
    log_scales64 = torch.from_numpy(arrays["log_scales"].astype(np.float64))
    rotations64 = torch.from_numpy(arrays["rotations"].astype(np.float64))
    colors64 = torch.zeros((count, 3), dtype=torch.float64)
    mathematical_field = GaussianField(means64, log_scales64, rotations64, colors64)
    conics64 = mathematical_field.conics(dilation=AA_DILATION)
    diagnostics = geometry_diagnostics(
        GaussianGeometry(means64, conics64, torch.from_numpy(radii)), HEIGHT, WIDTH,
        minimum_mass=MINIMUM_MASS, maximum_condition=MAXIMUM_CONDITION,
    )

    target_color_hashes: dict[str, Any] = {}
    for target in spec["evaluation_targets"]:
        colors = target_values(str(target), arrays["means"], dtype=np.float64).astype(np.float32)
        target_color_hashes[str(target)] = _array_record(colors)
    compact = _compact_diagnostics(diagnostics)
    diagnostic_relative = Path("diagnostics") / f"{spec['field_id']}.npz"
    _deterministic_npz(outdir / diagnostic_relative, compact)
    record = {
        "schema": "bench013-stage0-field-v1",
        **dict(spec),
        "path": relative.as_posix(),
        "file_sha256": _sha256_file(outdir / relative),
        "arrays": {name: _array_record(value) for name, value in sorted(arrays.items())},
        "field_content_sha256": _field_content_hash(arrays),
        "geometry_sha256": _geometry_hash(arrays),
        "mathematical_conics_float64": _array_record(conics64.detach().numpy()),
        "diagnostics_path": diagnostic_relative.as_posix(),
        "diagnostics_file_sha256": _sha256_file(outdir / diagnostic_relative),
        "diagnostic_arrays": {
            name: _array_record(value) for name, value in sorted(compact.items())
        },
        "target_color_hashes_float32": target_color_hashes,
        "init_config": asdict(init),
        "structure_tensor_config": asdict(structure),
    }
    return record, compact, arrays["means"]


def _control_summary(diagnostics: Any) -> dict[str, Any]:
    condition = diagnostics.condition_numbers
    finite = condition[torch.isfinite(condition)]
    return {
        "pixel_count": int(diagnostics.mass.numel()),
        "minimum_mass": float(diagnostics.mass.min()),
        "minimum_active_count": int(diagnostics.active_counts.min()),
        "minimum_rank": int(diagnostics.ranks.min()),
        "maximum_finite_condition": None if finite.numel() == 0 else float(finite.max()),
        "mass_failure_count": int((~diagnostics.mass_pass).sum()),
        "contributor_failure_count": int((~diagnostics.contributor_pass).sum()),
        "rank_failure_count": int((~diagnostics.rank_pass).sum()),
        "condition_failure_count": int((~diagnostics.condition_pass).sum()),
        "early_failure_count": int((~diagnostics.early_pass).sum()),
    }


def plumbing_controls() -> dict[str, Any]:
    dtype = torch.float64
    positive_means = torch.tensor(((0, 0), (8, 0), (0, 8), (8, 8)), dtype=dtype)
    collinear_means = torch.tensor(((1, 4), (4, 4), (7, 4)), dtype=dtype)
    one_means = torch.tensor(((4, 4),), dtype=dtype)

    def geometry(means: torch.Tensor) -> GaussianGeometry:
        count = int(means.shape[0])
        conics = torch.tensor((1.0 / 64.0, 0.0, 1.0 / 64.0), dtype=dtype).repeat(count, 1)
        radii = torch.full((count, 2), 8, dtype=torch.int64)
        return GaussianGeometry(means, conics, radii)

    positive_geometry = geometry(positive_means)
    positive = geometry_diagnostics(
        positive_geometry, 9, 9,
        minimum_mass=MINIMUM_MASS, maximum_condition=MAXIMUM_CONDITION,
    )
    operator = build_local_linear_operator(positive_geometry, 9, 9)
    sample_colors = torch.from_numpy(
        target_values("affine_xy", positive_means.numpy(), height=9, width=9)
    )
    expected = torch.from_numpy(target_image("affine_xy", height=9, width=9))
    error = float((operator.render(sample_colors) - expected).abs().max())

    collinear = geometry_diagnostics(
        geometry(collinear_means), 9, 9,
        minimum_mass=MINIMUM_MASS, maximum_condition=MAXIMUM_CONDITION,
    )
    one = geometry_diagnostics(
        geometry(one_means), 9, 9,
        minimum_mass=MINIMUM_MASS, maximum_condition=MAXIMUM_CONDITION,
    )
    positive_accepted = bool(positive.early_pass.all()) and error <= 1e-12
    collinear_rejected = bool((~collinear.rank_pass).any())
    one_rejected = bool((~one.contributor_pass).any()) and bool((~one.rank_pass).any())
    return {
        "schema": "bench013-stage0-controls-v1",
        "positive_affine": {
            **_control_summary(positive),
            "affine_max_abs_error": error,
            "accepted": positive_accepted,
        },
        "negative_collinear": {
            **_control_summary(collinear),
            "rejected_by_rank": collinear_rejected,
        },
        "negative_one_node": {
            **_control_summary(one),
            "rejected_by_contributor_and_rank": one_rejected,
        },
        "valid": positive_accepted and collinear_rejected and one_rejected,
    }


def _worst_coordinate(values: np.ndarray, *, maximum: bool) -> dict[str, Any]:
    index = int(np.argmax(values) if maximum else np.argmin(values))
    value = float(values[index])
    return {
        "x": index % WIDTH,
        "y": index // WIDTH,
        "value": value if math.isfinite(value) else None,
        "finite": math.isfinite(value),
    }


def _cell_summary(
    cell: Mapping[str, Any], diagnostics: Mapping[str, np.ndarray], means: np.ndarray
) -> dict[str, Any]:
    condition = diagnostics["condition_numbers"]
    margin = diagnostics["singular_values"][:, 2] - diagnostics["rank_thresholds"]
    target_colors64 = target_values(str(cell["target"]), means, dtype=np.float64)
    target_colors32 = target_colors64.astype(np.float32)
    return {
        "schema": "bench013-stage0-cell-v1",
        **dict(cell),
        "pixel_count": HEIGHT * WIDTH,
        "analytic_colors_float64": _array_record(target_colors64),
        "analytic_colors_float32": _array_record(target_colors32),
        "mass_failure_count": int((~diagnostics["mass_pass"]).sum()),
        "contributor_failure_count": int((~diagnostics["contributor_pass"]).sum()),
        "rank_failure_count": int((~diagnostics["rank_pass"]).sum()),
        "condition_failure_count": int((~diagnostics["condition_pass"]).sum()),
        "early_failure_count": int((~diagnostics["early_pass"]).sum()),
        "early_pass": bool(diagnostics["early_pass"].all()),
        "worst": {
            "minimum_mass": _worst_coordinate(diagnostics["mass"], maximum=False),
            "minimum_active_count": _worst_coordinate(
                diagnostics["active_counts"], maximum=False
            ),
            "minimum_rank_margin": _worst_coordinate(margin, maximum=False),
            "maximum_condition": _worst_coordinate(condition, maximum=True),
        },
    }


def _write_pixel_rows(
    handle: BinaryIO,
    binding: str,
    cell: Mapping[str, Any],
    diagnostics: Mapping[str, np.ndarray],
) -> None:
    for index in range(HEIGHT * WIDTH):
        condition = float(diagnostics["condition_numbers"][index])
        _append_jsonl(
            handle,
            {
                "schema": ROW_SCHEMA,
                "binding_sha256": binding,
                "cell_id": cell["cell_id"],
                "field_id": cell["field_id"],
                "cohort": cell["cohort"],
                "target": cell["target"],
                "count": cell["count"],
                "seed": cell["seed"],
                "pixel_index": index,
                "x": index % WIDTH,
                "y": index // WIDTH,
                "mass": float(diagnostics["mass"][index]),
                "active_count": int(diagnostics["active_counts"][index]),
                "singular_values": [
                    float(value) for value in diagnostics["singular_values"][index]
                ],
                "rank_threshold": float(diagnostics["rank_thresholds"][index]),
                "rank": int(diagnostics["ranks"][index]),
                "covariance_eigenvalues": [
                    float(value)
                    for value in diagnostics["covariance_eigenvalues"][index]
                ],
                "condition_number": condition if math.isfinite(condition) else None,
                "condition_finite": math.isfinite(condition),
                "mass_pass": bool(diagnostics["mass_pass"][index]),
                "contributor_pass": bool(diagnostics["contributor_pass"][index]),
                "rank_pass": bool(diagnostics["rank_pass"][index]),
                "condition_pass": bool(diagnostics["condition_pass"][index]),
                "early_pass": bool(diagnostics["early_pass"][index]),
            },
        )


def _write_downstream(
    outdir: Path, binding: str, cells: Iterable[Mapping[str, Any]], early_failed: bool
) -> dict[str, Any]:
    status = SHORT_CIRCUIT if early_failed else PENDING
    cell_list = list(cells)
    specifications: tuple[tuple[str, Iterable[Mapping[str, Any]]], ...] = (
        ("forward", cell_list),
        ("permutation", cell_list),
        (
            "gradient",
            ({"cell_id": _cell_id("target_conditioned", "affine_xy", 56, 0)},),
        ),
        (
            "effective_weights",
            ({"field_id": spec["field_id"]} for spec in field_specs()),
        ),
    )
    counts: dict[str, int] = {}
    for name, rows in specifications:
        path = outdir / f"{name}.jsonl"
        count = 0
        with path.open("wb") as handle:
            for row in rows:
                _append_jsonl(
                    handle,
                    {
                        "schema": f"bench013-stage0-{name}-v1",
                        "binding_sha256": binding,
                        **dict(row),
                        "status": status,
                    },
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        counts[name] = count
    return {"status": status, "row_counts": counts}


def derive_decision(
    *,
    controls_valid: bool,
    artifact_audit_passed: bool,
    complete: bool,
    early_failure_count: int,
) -> dict[str, Any]:
    if not controls_valid or not artifact_audit_passed or not complete:
        return {
            "decision": "unavailable",
            "classification": "invalid_or_incomplete_assay",
            "downstream_status": SHORT_CIRCUIT,
        }
    if early_failure_count > 0:
        return {
            "decision": "kill",
            "classification": "stage0_kill",
            "downstream_status": SHORT_CIRCUIT,
        }
    return {
        "decision": "continue",
        "classification": "early_screen_pass_downstream_required",
        "downstream_status": PENDING,
    }


def _write_source_archive(root: Path, outdir: Path) -> dict[str, Any]:
    path = outdir / "executed_sources.tar"
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative in ARCHIVE_PATHS:
            source = root / relative
            if not source.is_file():
                raise RuntimeError(f"missing BENCH-013 archive dependency: {relative}")
            info = archive.gettarinfo(str(source), arcname=relative)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "members": {relative: _sha256_file(root / relative) for relative in ARCHIVE_PATHS},
    }


def replay_early_artifacts(outdir: Path, root: Path | None = None) -> dict[str, Any]:
    """Independently reconstruct frozen hashes and all early gates from persisted artifacts."""

    checks: dict[str, bool] = {}
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    binding = str(config["binding_sha256"])
    environment = json.loads((outdir / "environment.json").read_text(encoding="utf-8"))
    science = dict(config)
    science.pop("binding_sha256")
    checks["config_binding"] = _binding_sha256(science, environment) == binding
    if root is not None:
        checks["current_source_hashes"] = (
            _source_manifest(root.resolve()) == config["source_manifest"]
        )
    targets = json.loads((outdir / "targets.json").read_text(encoding="utf-8"))
    target_ok = True
    for target in TARGETS:
        formula = TARGET_FORMULAS[target]
        target_ok &= targets[target]["formula_sha256"] == _sha256_bytes(
            formula.encode("utf-8")
        )
        for dtype in (np.float64, np.float32):
            suffix = np.dtype(dtype).name
            expected = target_image(target, dtype=dtype)
            stored = np.load(outdir / targets[target]["arrays"][suffix]["path"], allow_pickle=False)
            target_ok &= np.array_equal(expected, stored)
            target_ok &= _array_record(stored) == {
                key: targets[target]["arrays"][suffix][key]
                for key in ("dtype", "shape", "payload_sha256", "record_sha256")
            }
    checks["target_formula_and_array_hashes"] = bool(target_ok)

    field_rows = _read_jsonl(outdir / "fields.jsonl")
    field_ok = len(field_rows) == EXPECTED_FIELDS
    geometry_hashes: set[str] = set()
    field_ids: set[str] = set()
    means_by_field: dict[str, np.ndarray] = {}
    diagnostics_by_field: dict[str, dict[str, np.ndarray]] = {}
    expected_fields = {str(spec["field_id"]): spec for spec in field_specs()}
    for row in field_rows:
        field_id = str(row["field_id"])
        field_ids.add(field_id)
        expected_spec = expected_fields.get(field_id)
        field_ok &= expected_spec is not None
        field_ok &= row["binding_sha256"] == binding
        if expected_spec is not None:
            field_ok &= all(row[key] == value for key, value in expected_spec.items())
            field_ok &= set(row["target_color_hashes_float32"]) == set(
                expected_spec["evaluation_targets"]
            )
        with np.load(outdir / row["path"], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        with np.load(outdir / row["diagnostics_path"], allow_pickle=False) as archive:
            diagnostic_arrays = {name: archive[name] for name in archive.files}
        field_ok &= _sha256_file(outdir / row["path"]) == row["file_sha256"]
        field_ok &= (
            _sha256_file(outdir / row["diagnostics_path"])
            == row["diagnostics_file_sha256"]
        )
        field_ok &= _field_content_hash(arrays) == row["field_content_sha256"]
        field_ok &= _geometry_hash(arrays) == row["geometry_sha256"]
        field_ok &= {
            name: _array_record(value) for name, value in sorted(diagnostic_arrays.items())
        } == row["diagnostic_arrays"]
        geometry_hashes.add(str(row["geometry_sha256"]))
        means_by_field[field_id] = arrays["means"]
        diagnostics_by_field[field_id] = diagnostic_arrays
        for target, target_hash in row["target_color_hashes_float32"].items():
            colors = target_values(target, arrays["means"], dtype=np.float64).astype(np.float32)
            field_ok &= _array_record(colors) == target_hash
        if expected_spec is not None:
            init, structure = frozen_configs(int(row["count"]), int(row["seed"]))
            rebuilt = build_field(
                target_image(str(row["source_target"]), dtype=np.float32),
                init,
                structure,
                device="cpu",
            )
            rebuilt.colors = torch.from_numpy(
                target_values(
                    str(row["source_target"]),
                    rebuilt.means.detach().numpy(),
                    dtype=np.float64,
                ).astype(np.float32)
            ).clone()
            rebuilt_arrays = {
                "means": rebuilt.means.detach().numpy().copy(),
                "log_scales": rebuilt.log_scales.detach().numpy().copy(),
                "rotations": rebuilt.rotations.detach().numpy().copy(),
                "colors": rebuilt.colors.detach().numpy().copy(),
                "conics_float32": rebuilt.conics(dilation=AA_DILATION).detach().numpy().copy(),
                "radii": rebuilt.radii(
                    SIGMA_CUTOFF, dilation=AA_DILATION
                ).detach().numpy().copy(),
            }
            field_ok &= rebuilt_arrays.keys() == arrays.keys()
            field_ok &= all(
                np.array_equal(rebuilt_arrays[name], arrays[name]) for name in arrays
            )
            means64 = torch.from_numpy(rebuilt_arrays["means"].astype(np.float64))
            mathematical_field = GaussianField(
                means64,
                torch.from_numpy(rebuilt_arrays["log_scales"].astype(np.float64)),
                torch.from_numpy(rebuilt_arrays["rotations"].astype(np.float64)),
                torch.zeros((int(row["count"]), 3), dtype=torch.float64),
            )
            replayed_diagnostics = _compact_diagnostics(
                geometry_diagnostics(
                    GaussianGeometry(
                        means64,
                        mathematical_field.conics(dilation=AA_DILATION),
                        torch.from_numpy(rebuilt_arrays["radii"]),
                    ),
                    HEIGHT,
                    WIDTH,
                    minimum_mass=MINIMUM_MASS,
                    maximum_condition=MAXIMUM_CONDITION,
                )
            )
            field_ok &= replayed_diagnostics.keys() == diagnostic_arrays.keys()
            field_ok &= all(
                np.array_equal(replayed_diagnostics[name], diagnostic_arrays[name])
                for name in diagnostic_arrays
            )
    checks["field_hashes"] = bool(field_ok and len(field_ids) == EXPECTED_FIELDS)
    checks["field_id_set"] = field_ids == set(expected_fields)
    checks["distinct_geometry_hash_count"] = len(geometry_hashes) == EXPECTED_DISTINCT_GEOMETRIES

    aggregates: dict[str, dict[str, Any]] = {}
    raw_gate_ok = True
    raw_tensor_ok = True
    pixel_rows = 0
    expected_cells = {str(spec["cell_id"]): spec for spec in cell_specs()}
    with (outdir / "pixel_diagnostics.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pixel_rows += 1
            cell_id = str(row["cell_id"])
            expected_cell = expected_cells.get(cell_id)
            raw_gate_ok &= expected_cell is not None
            if expected_cell is not None:
                raw_gate_ok &= all(row[key] == value for key, value in expected_cell.items())
            aggregate = aggregates.setdefault(
                cell_id,
                {
                    "indices": set(),
                    "mass_failure_count": 0,
                    "contributor_failure_count": 0,
                    "rank_failure_count": 0,
                    "condition_failure_count": 0,
                    "early_failure_count": 0,
                },
            )
            aggregate["indices"].add(int(row["pixel_index"]))
            field_id = str(row["field_id"])
            persisted = diagnostics_by_field.get(field_id)
            index = int(row["pixel_index"])
            raw_tensor_ok &= persisted is not None and 0 <= index < HEIGHT * WIDTH
            if persisted is not None and 0 <= index < HEIGHT * WIDTH:
                persisted_condition = float(persisted["condition_numbers"][index])
                raw_tensor_ok &= float(row["mass"]) == float(persisted["mass"][index])
                raw_tensor_ok &= int(row["active_count"]) == int(
                    persisted["active_counts"][index]
                )
                raw_tensor_ok &= row["singular_values"] == [
                    float(value) for value in persisted["singular_values"][index]
                ]
                raw_tensor_ok &= float(row["rank_threshold"]) == float(
                    persisted["rank_thresholds"][index]
                )
                raw_tensor_ok &= int(row["rank"]) == int(persisted["ranks"][index])
                raw_tensor_ok &= row["covariance_eigenvalues"] == [
                    float(value) for value in persisted["covariance_eigenvalues"][index]
                ]
                raw_tensor_ok &= (
                    row["condition_number"] is None and not math.isfinite(persisted_condition)
                ) or row["condition_number"] == persisted_condition
            singular = [float(value) for value in row["singular_values"]]
            active = int(row["active_count"])
            threshold = max(active, 3) * np.finfo(np.float64).eps * singular[0]
            eigenvalues = [float(value) for value in row["covariance_eigenvalues"]]
            condition = (
                eigenvalues[1] / eigenvalues[0]
                if eigenvalues[0] > 0.0
                else float("inf")
            )
            gates = {
                "mass_pass": float(row["mass"]) >= MINIMUM_MASS,
                "contributor_pass": active >= MINIMUM_CONTRIBUTORS,
                "rank_pass": singular[2] > threshold,
                "condition_pass": math.isfinite(condition) and condition <= MAXIMUM_CONDITION,
            }
            gates["early_pass"] = all(gates.values())
            raw_gate_ok &= row["binding_sha256"] == binding
            raw_gate_ok &= int(row["x"]) == int(row["pixel_index"]) % WIDTH
            raw_gate_ok &= int(row["y"]) == int(row["pixel_index"]) // WIDTH
            raw_gate_ok &= math.isclose(
                float(row["rank_threshold"]), threshold, rel_tol=1e-15, abs_tol=0.0
            )
            raw_gate_ok &= int(row["rank"]) == sum(value > threshold for value in singular)
            raw_gate_ok &= bool(row["condition_finite"]) == math.isfinite(condition)
            stored_condition = row["condition_number"]
            raw_gate_ok &= (
                stored_condition is None and not math.isfinite(condition)
            ) or (
                stored_condition is not None
                and math.isclose(float(stored_condition), condition, rel_tol=1e-14, abs_tol=0.0)
            )
            for name, value in gates.items():
                raw_gate_ok &= bool(row[name]) == value
            for name in ("mass", "contributor", "rank", "condition", "early"):
                aggregate[f"{name}_failure_count"] += int(not gates[f"{name}_pass"])
    checks["pixel_row_count"] = pixel_rows == EXPECTED_PIXELS
    checks["pixel_coordinates_complete"] = (
        len(aggregates) == EXPECTED_CELLS
        and all(value["indices"] == set(range(HEIGHT * WIDTH)) for value in aggregates.values())
    )
    checks["raw_gate_reconstruction"] = bool(raw_gate_ok)
    checks["raw_rows_match_diagnostic_tensors"] = bool(raw_tensor_ok)

    cell_rows = _read_jsonl(outdir / "cells.jsonl")
    cell_ok = len(cell_rows) == EXPECTED_CELLS
    cell_ids: set[str] = set()
    for row in cell_rows:
        cell_id = str(row["cell_id"])
        cell_ids.add(cell_id)
        expected_cell = expected_cells.get(cell_id)
        cell_ok &= expected_cell is not None
        cell_ok &= row["binding_sha256"] == binding
        if expected_cell is not None:
            cell_ok &= all(row[key] == value for key, value in expected_cell.items())
        aggregate = aggregates.get(cell_id)
        if aggregate is None:
            cell_ok = False
            continue
        means = means_by_field.get(str(row["field_id"]))
        if means is None:
            cell_ok = False
            continue
        colors64 = target_values(str(row["target"]), means, dtype=np.float64)
        colors32 = colors64.astype(np.float32)
        cell_ok &= row["analytic_colors_float64"] == _array_record(colors64)
        cell_ok &= row["analytic_colors_float32"] == _array_record(colors32)
        for name in ("mass", "contributor", "rank", "condition", "early"):
            cell_ok &= int(row[f"{name}_failure_count"]) == int(
                aggregate[f"{name}_failure_count"]
            )
        cell_ok &= bool(row["early_pass"]) == (aggregate["early_failure_count"] == 0)
    checks["cell_ledger_reconstruction"] = bool(
        cell_ok and cell_ids == set(expected_cells)
    )
    early_failures = sum(value["early_failure_count"] for value in aggregates.values())

    expected_downstream_status = SHORT_CIRCUIT if early_failures else PENDING
    downstream_ok = True
    expected_downstream_counts = {
        "forward": EXPECTED_CELLS,
        "permutation": EXPECTED_CELLS,
        "gradient": 1,
        "effective_weights": EXPECTED_FIELDS,
    }
    for name, expected_count in expected_downstream_counts.items():
        rows = _read_jsonl(outdir / f"{name}.jsonl")
        downstream_ok &= len(rows) == expected_count
        downstream_ok &= all(row["binding_sha256"] == binding for row in rows)
        downstream_ok &= all(row["status"] == expected_downstream_status for row in rows)
    checks["downstream_ledger"] = bool(downstream_ok)

    stored_controls = json.loads((outdir / "controls.json").read_text(encoding="utf-8"))
    replayed_controls = plumbing_controls()
    controls_binding = stored_controls.pop("binding_sha256", None)
    checks["plumbing_controls"] = bool(
        controls_binding == binding and stored_controls == replayed_controls
    )

    source_record = json.loads((outdir / "source_archive.json").read_text(encoding="utf-8"))
    archive_path = outdir / source_record["path"]
    source_ok = _sha256_file(archive_path) == source_record["sha256"]
    archived_hashes: dict[str, str] = {}
    with tarfile.open(archive_path, mode="r") as archive:
        for member in archive.getmembers():
            extracted = archive.extractfile(member)
            if extracted is not None:
                archived_hashes[member.name] = _sha256_bytes(extracted.read())
    source_ok &= archived_hashes == source_record["members"]
    source_ok &= all(
        archived_hashes.get(relative) == digest
        for relative, digest in config["source_manifest"].items()
    )
    checks["executed_source_archive"] = bool(source_ok)
    result = {
        "schema": "bench013-stage0-replay-v1",
        "checks": checks,
        "artifact_audit_passed": all(checks.values()),
        "counts": {
            "logical_fields": len(field_rows),
            "distinct_geometry_hashes": len(geometry_hashes),
            "cells": len(cell_rows),
            "pixel_rows": pixel_rows,
            "early_failure_rows": int(early_failures),
        },
    }
    return result


def _artifact_manifest(outdir: Path) -> dict[str, Any]:
    files = {
        path.relative_to(outdir).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(outdir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    return {
        "schema": "bench013-stage0-artifact-manifest-v1",
        "file_count": len(files),
        "files": files,
    }


def run(outdir: Path, root: Path) -> dict[str, Any]:
    if outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {outdir}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    outdir.mkdir(parents=True)
    environment = _environment(root)
    targets = _target_manifest(outdir)
    source_manifest = _source_manifest(root)
    init_example, structure_example = frozen_configs(COUNTS[0], SEEDS[0])
    science = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "dimensions": {"height": HEIGHT, "width": WIDTH, "all_pixels_scored": True},
        "targets": targets,
        "counts": list(COUNTS),
        "seeds": list(SEEDS),
        "cohorts": list(COHORTS),
        "renderer": {
            "support": "production_clipped_aabb_via__tile_bounds_and__tile_coords",
            "rounded_float32_means": True,
            "sigma_cutoff": SIGMA_CUTOFF,
            "support_fade": False,
            "opacity": None,
            "aa_dilation": AA_DILATION,
            "ellipse_mask": False,
            "small_weight_drop": False,
        },
        "early_gates": {
            "minimum_mass": MINIMUM_MASS,
            "minimum_active_contributors": MINIMUM_CONTRIBUTORS,
            "rank": "sigma3>max(n_active,3)*eps64*sigma1",
            "maximum_covariance_condition": MAXIMUM_CONDITION,
        },
        "expected": {
            "logical_fields": EXPECTED_FIELDS,
            "distinct_geometry_hashes": EXPECTED_DISTINCT_GEOMETRIES,
            "cells": EXPECTED_CELLS,
            "pixel_rows": EXPECTED_PIXELS,
        },
        "init_config_template": asdict(init_example),
        "structure_tensor_config": asdict(structure_example),
        "source_manifest": source_manifest,
    }
    binding = _binding_sha256(science, environment)
    config = {**science, "binding_sha256": binding}
    _atomic_json(outdir / "config.json", config)
    _atomic_json(outdir / "environment.json", environment)
    _atomic_json(outdir / "targets.json", targets)

    controls = plumbing_controls()
    controls["binding_sha256"] = binding
    _atomic_json(outdir / "controls.json", controls)

    diagnostics_by_field: dict[str, dict[str, np.ndarray]] = {}
    means_by_field: dict[str, np.ndarray] = {}
    field_records: list[dict[str, Any]] = []
    with (outdir / "fields.jsonl").open("wb") as handle:
        for spec in field_specs():
            record, diagnostics, means = _build_field_record(spec, outdir)
            record["binding_sha256"] = binding
            _append_jsonl(handle, record)
            handle.flush()
            diagnostics_by_field[str(spec["field_id"])] = diagnostics
            means_by_field[str(spec["field_id"])] = means
            field_records.append(record)
        os.fsync(handle.fileno())

    cells = cell_specs()
    cell_records: list[dict[str, Any]] = []
    pixel_path = outdir / "pixel_diagnostics.jsonl"
    cell_path = outdir / "cells.jsonl"
    with pixel_path.open("wb") as pixel_handle, cell_path.open("wb") as cell_handle:
        for cell in cells:
            field_id = str(cell["field_id"])
            diagnostics = diagnostics_by_field[field_id]
            record = _cell_summary(cell, diagnostics, means_by_field[field_id])
            record["binding_sha256"] = binding
            _write_pixel_rows(pixel_handle, binding, cell, diagnostics)
            _append_jsonl(cell_handle, record)
            pixel_handle.flush()
            cell_handle.flush()
            os.fsync(pixel_handle.fileno())
            os.fsync(cell_handle.fileno())
            cell_records.append(record)

    early_failure_count = sum(int(row["early_failure_count"]) for row in cell_records)
    early_failed = early_failure_count > 0
    downstream = _write_downstream(outdir, binding, cells, early_failed)
    distinct_geometry_hashes = len({row["geometry_sha256"] for row in field_records})
    provisional_complete = (
        len(field_records) == EXPECTED_FIELDS
        and distinct_geometry_hashes == EXPECTED_DISTINCT_GEOMETRIES
        and len(cell_records) == EXPECTED_CELLS
        and sum(int(row["pixel_count"]) for row in cell_records) == EXPECTED_PIXELS
    )
    archive = _write_source_archive(root, outdir)
    _atomic_json(outdir / "source_archive.json", archive)
    replay = replay_early_artifacts(outdir, root)
    replay["binding_sha256"] = binding
    _atomic_json(outdir / "replay.json", replay)
    complete = bool(provisional_complete and replay["artifact_audit_passed"])
    completion = {
        "schema": "bench013-stage0-early-completion-v1",
        "binding_sha256": binding,
        "complete": complete,
        "written_after_independent_replay": True,
        "counts": replay["counts"],
        "downstream": downstream,
        "source_archive": archive,
    }
    if complete:
        _atomic_json(outdir / "completion.json", completion)
    decision = derive_decision(
        controls_valid=bool(controls["valid"]),
        artifact_audit_passed=bool(replay["artifact_audit_passed"]),
        complete=complete,
        early_failure_count=early_failure_count,
    )
    analysis = {
        "schema": "bench013-stage0-early-analysis-v1",
        "binding_sha256": binding,
        **decision,
        "controls_valid": bool(controls["valid"]),
        "artifact_audit_passed": bool(replay["artifact_audit_passed"]),
        "complete": complete,
        "cell_failure_count": sum(not bool(row["early_pass"]) for row in cell_records),
        "pixel_failure_count": early_failure_count,
        "gate_failure_counts": {
            name: sum(int(row[f"{name}_failure_count"]) for row in cell_records)
            for name in ("mass", "contributor", "rank", "condition", "early")
        },
        "downstream": downstream,
        "source_archive": archive,
        "claim_boundary": {
            "quality": False,
            "convergence": False,
            "performance": False,
            "compression": False,
            "expressiveness": False,
            "method_novelty": False,
        },
        "stage1_authorized": False,
    }
    _atomic_json(outdir / "analysis.json", analysis)
    _atomic_json(outdir / "artifact_manifest.json", _artifact_manifest(outdir))
    return analysis


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1],
        help="repository root whose executed sources are bound",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    analysis = run(arguments.outdir.resolve(), arguments.root.resolve())
    print(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False))
    if analysis["decision"] == "unavailable":
        return 3
    if analysis["decision"] == "kill":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
