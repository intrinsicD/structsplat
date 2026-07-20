"""Spent-data nested-subspace diagnostic following BENCH-009.

This executable does not select a new packet identity and does not consume new targets.  It
reconstructs only BENCH-009's near-plateau cross-fit units, keeps their selected affine/carrier
identities fixed, and tests the minimal nested-projector repair preregistered in BENCH-011.
Passing this diagnostic cannot promote a method; failing it closes the formulation without
retuning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from benchmarks import residual_tangent_auction_stage1 as manifest_module
from benchmarks.residual_tangent_auction_matrixfree import (
    MatrixFreeTangentFactorization,
    build_masked_jacobian_operators,
    factorize_masked_tangent,
    tensor_sha256,
)
from benchmarks.residual_tangent_auction_stage1_core import (
    affine_six_columns,
    carrier_six_columns,
    checkerboard_fold_masks,
    make_no_opacity_parameterization,
    realize_affine_six,
    realize_carrier_six,
    render_normalized,
    support_membership_change_summary,
)
from benchmarks.residual_tangent_auction_stage1_runner import RENDER_CHUNK, field_sha256
from structsplat.gaussians import GaussianField


PROTOCOL = "BENCH-011 nested residual-extension spent-data diagnostic v2"
ROW_SCHEMA = "structsplat.bench011.nested_extension_row.v2"
ANALYSIS_SCHEMA = "structsplat.bench011.nested_extension_analysis.v2"
PRIMARY_RCOND = 1e-5
PRIMARY_DAMPING = 1e-3
TRUST_RADII = (0.25, 0.75)
CANDIDATES = ("affine6", "carrier6")
EXPECTED_UNITS = 24
EXPECTED_ROWS = 96
SOURCE_PATHS = (
    "benchmarks/nested_residual_extension_diagnostic.py",
    "benchmarks/residual_tangent_auction_matrixfree.py",
    "benchmarks/residual_tangent_auction_stage1.py",
    "benchmarks/residual_tangent_auction_stage1_core.py",
    "benchmarks/residual_tangent_auction_stage1_runner.py",
    "benchmarks/residual_tangent_auction_stage1_science.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/render.py",
    "tasks/BENCH-011-nested-residual-extension-diagnostic.md",
    "tests/test_nested_residual_extension_diagnostic.py",
    "pyproject.toml",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(dict(value)))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(dict(value)).decode("utf-8") + "\n")
        handle.flush()


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(_canonical_json(dict(value)).decode("utf-8") + "\n")
        handle.flush()
    temporary.replace(path)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_reject_constant)
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"{path}:{line_number} lacks object key {key!r}")
            identifier = str(value[key])
            if identifier in records:
                raise ValueError(f"{path}:{line_number} duplicates {identifier}")
            records[identifier] = value
    return records


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_hashes() -> dict[str, str]:
    root = _repo_root()
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required BENCH-011 source is missing: {relative}")
        result[relative] = _sha256_file(path)
    return result


def _git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *arguments),
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _execution_environment() -> dict[str, Any]:
    porcelain = _git_value("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_porcelain_sha256": (
            None if porcelain is None else _sha256_bytes(porcelain.encode("utf-8"))
        ),
        "git_dirty": None if porcelain is None else bool(porcelain),
    }


def _numerical_tolerance(rows: int, parameter_count: int) -> float:
    return float(64.0 * np.finfo(np.float64).eps * max(rows, parameter_count, 6))


def _validate_float64_matrix(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if value.dtype != torch.float64 or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU float64 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _validate_float64_vector(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if value.dtype != torch.float64 or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU float64 tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def orthonormalize_fixed_base(base_basis: torch.Tensor) -> torch.Tensor:
    """Orthonormalize an immutable base span without consulting an extension or residual."""

    _validate_float64_matrix(base_basis, name="base_basis")
    if base_basis.shape[1] == 0:
        return base_basis.clone()
    left, singular, vh = torch.linalg.svd(base_basis, full_matrices=False)
    if singular.numel() != base_basis.shape[1] or float(singular[-1]) <= 0.0:
        raise ValueError("base_basis is rank deficient")
    # Polar orthonormalization preserves the supplied column span and column count.  It is
    # independent of A and r, so it cannot adapt the base comparator to a candidate.
    return left @ vh


def _project_out_twice(basis: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    result = value
    if basis.shape[1]:
        for _ in range(2):
            result = result - basis @ (basis.T @ result)
    return result


@dataclass(frozen=True)
class NestedExtensionBasis:
    base_basis: torch.Tensor
    extension_basis: torch.Tensor
    residual_perpendicular: torch.Tensor
    residualized_columns: torch.Tensor
    original_singular_values: torch.Tensor
    residualized_singular_values: torch.Tensor
    cutoff: float
    extension_rank: int
    nested_delta: float
    tolerance: float
    base_orthogonality_max_abs: float
    extension_orthogonality_max_abs: float
    cross_orthogonality_max_abs: float
    residual_base_component_max_abs: float


def nested_extension_basis(
    base_basis: torch.Tensor,
    columns: torch.Tensor,
    residual: torch.Tensor,
    *,
    parameter_count: int,
    rcond: float = PRIMARY_RCOND,
) -> NestedExtensionBasis:
    """Build the frozen nested extension basis and direct non-negative capacity statistic."""

    _validate_float64_matrix(base_basis, name="base_basis")
    _validate_float64_matrix(columns, name="columns")
    _validate_float64_vector(residual, name="residual")
    if columns.shape != (residual.numel(), 6):
        raise ValueError("columns must have shape (residual rows, 6)")
    if base_basis.shape[0] != residual.numel():
        raise ValueError("base_basis row count differs from residual")
    if not math.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("rcond must be finite and positive")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")

    q_j = orthonormalize_fixed_base(base_basis)
    r_perp = _project_out_twice(q_j, residual)
    residualized = _project_out_twice(q_j, columns)
    original_singular = torch.linalg.svdvals(columns)
    reference = float(original_singular[0]) if original_singular.numel() else 0.0
    cutoff = float(rcond) * reference
    if residualized.numel():
        left, residualized_singular, _vh = torch.linalg.svd(
            residualized, full_matrices=False
        )
    else:
        left = residualized.new_zeros((residualized.shape[0], 0))
        residualized_singular = residualized.new_zeros((0,))
    keep = residualized_singular > cutoff
    q_b = left[:, keep]
    if q_b.shape[1]:
        # SVD already yields an orthonormal Q_B.  A second fixed-base projection plus polar
        # correction only removes roundoff and does not change the threshold decision.
        q_b = _project_out_twice(q_j, q_b)
        q_b = orthonormalize_fixed_base(q_b)
    coordinates = q_b.T @ r_perp
    delta = float(coordinates.square().sum())
    tolerance = _numerical_tolerance(int(residual.numel()), int(parameter_count))
    identity_j = torch.eye(q_j.shape[1], dtype=torch.float64)
    identity_b = torch.eye(q_b.shape[1], dtype=torch.float64)
    base_error = (
        float((q_j.T @ q_j - identity_j).abs().max()) if q_j.shape[1] else 0.0
    )
    extension_error = (
        float((q_b.T @ q_b - identity_b).abs().max()) if q_b.shape[1] else 0.0
    )
    cross_error = float((q_j.T @ q_b).abs().max()) if q_j.shape[1] and q_b.shape[1] else 0.0
    residual_base = float((q_j.T @ r_perp).abs().max()) if q_j.shape[1] else 0.0
    if max(base_error, extension_error, cross_error) > tolerance:
        raise RuntimeError("nested basis failed frozen float64 orthogonality tolerance")
    residual_bound = tolerance * max(1.0, float(torch.linalg.vector_norm(residual)))
    if residual_base > residual_bound:
        raise RuntimeError("twice-residualized residual retained a base component")
    if delta < -tolerance or not math.isfinite(delta):
        raise RuntimeError("nested incremental energy is invalid")
    return NestedExtensionBasis(
        base_basis=q_j,
        extension_basis=q_b,
        residual_perpendicular=r_perp,
        residualized_columns=residualized,
        original_singular_values=original_singular,
        residualized_singular_values=residualized_singular,
        cutoff=cutoff,
        extension_rank=int(keep.sum()),
        nested_delta=max(0.0, delta),
        tolerance=tolerance,
        base_orthogonality_max_abs=base_error,
        extension_orthogonality_max_abs=extension_error,
        cross_orthogonality_max_abs=cross_error,
        residual_base_component_max_abs=residual_base,
    )


@dataclass(frozen=True)
class NestedPhysicalSolve:
    base_step: torch.Tensor
    extension_coefficients: torch.Tensor
    predicted_discovery_action: torch.Tensor
    singular_values: torch.Tensor
    numerical_rank: int
    condition: float | None
    damping: float
    compact_design: torch.Tensor


def solve_nested_physical_system(
    nested: NestedExtensionBasis,
    base_reduced_design: torch.Tensor,
    columns: torch.Tensor,
    residual: torch.Tensor,
    *,
    damping: float = PRIMARY_DAMPING,
) -> NestedPhysicalSolve:
    """Solve the nested compact system in raw base-plus-six physical coordinates."""

    _validate_float64_matrix(base_reduced_design, name="base_reduced_design")
    _validate_float64_matrix(columns, name="columns")
    _validate_float64_vector(residual, name="residual")
    q_j = nested.base_basis
    q_b = nested.extension_basis
    if base_reduced_design.shape[0] != q_j.shape[1]:
        raise ValueError("base_reduced_design must have one row per retained base direction")
    if columns.shape != (residual.numel(), 6):
        raise ValueError("columns must have shape (residual rows, 6)")
    if q_j.shape[0] != residual.numel() or q_b.shape[0] != residual.numel():
        raise ValueError("nested basis and residual rows differ")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    parameter_count = int(base_reduced_design.shape[1])
    top = torch.cat((base_reduced_design, q_j.T @ columns), dim=1)
    if q_b.shape[1]:
        bottom = torch.cat(
            (
                torch.zeros((q_b.shape[1], parameter_count), dtype=torch.float64),
                q_b.T @ columns,
            ),
            dim=1,
        )
        compact = torch.cat((top, bottom), dim=0)
    else:
        compact = top
    image_basis = torch.cat((q_j, q_b), dim=1)
    rhs = image_basis.T @ residual
    if compact.shape[0] == 0:
        coefficients = torch.zeros(parameter_count + 6, dtype=torch.float64)
        singular = torch.zeros(0, dtype=torch.float64)
        rank = 0
        condition = None
    else:
        left, all_singular, vh = torch.linalg.svd(compact, full_matrices=False)
        if all_singular.numel():
            cutoff = (
                64.0
                * torch.finfo(torch.float64).eps
                * max(compact.shape)
                * all_singular[0]
            )
            keep = all_singular > cutoff
        else:
            keep = torch.zeros(0, dtype=torch.bool)
        singular = all_singular[keep]
        if singular.numel():
            coordinates = singular / (singular.square() + float(damping) ** 2)
            coefficients = vh[keep].T @ (coordinates * (left[:, keep].T @ rhs))
            rank = int(keep.sum())
            condition = float(singular[0] / singular[-1])
        else:
            coefficients = torch.zeros(parameter_count + 6, dtype=torch.float64)
            rank = 0
            condition = None
    predicted = image_basis @ (compact @ coefficients)
    if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(predicted).all()):
        raise RuntimeError("nested physical solve produced non-finite values")
    return NestedPhysicalSolve(
        base_step=coefficients[:parameter_count],
        extension_coefficients=coefficients[parameter_count:],
        predicted_discovery_action=predicted,
        singular_values=singular,
        numerical_rank=rank,
        condition=condition,
        damping=float(damping),
        compact_design=compact,
    )


def uniform_joint_linf_scale(
    base_step: torch.Tensor,
    extension_coefficients: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    combined = torch.cat((base_step, extension_coefficients))
    maximum = float(combined.abs().max()) if combined.numel() else 0.0
    factor = min(1.0, float(radius) / max(maximum, 1e-300))
    return base_step * factor, extension_coefficients * factor, factor


def run_algebra_controls() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20260715)
    raw_q = torch.randn((29, 4), generator=generator, dtype=torch.float64)
    q_j, _r = torch.linalg.qr(raw_q, mode="reduced")
    raw_a = torch.randn((29, 6), generator=generator, dtype=torch.float64)
    residual = torch.randn(29, generator=generator, dtype=torch.float64)
    reference = nested_extension_basis(q_j, raw_a, residual, parameter_count=17)

    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    signs = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)
    transformed = raw_a[:, permutation] * signs
    invariant = nested_extension_basis(q_j, transformed, residual, parameter_count=17)
    invariant_error = abs(reference.nested_delta - invariant.nested_delta)

    coefficients = torch.randn((4, 6), generator=generator, dtype=torch.float64)
    inside = nested_extension_basis(q_j, q_j @ coefficients, residual, parameter_count=17)

    threshold_columns = torch.zeros((9, 6), dtype=torch.float64)
    threshold_columns[0, 0] = 1.0
    threshold_columns[1, 1] = 2.0e-5
    threshold_columns[2, 2] = 0.5e-5
    threshold = nested_extension_basis(
        torch.zeros((9, 0), dtype=torch.float64),
        threshold_columns,
        torch.arange(1, 10, dtype=torch.float64),
        parameter_count=6,
    )

    explicit = torch.cat((reference.base_basis, reference.extension_basis), dim=1)
    explicit_increment = float(
        (explicit.T @ residual).square().sum()
        - (reference.base_basis.T @ residual).square().sum()
    )
    base_design = reference.base_basis.T @ torch.randn(
        (29, 17), generator=generator, dtype=torch.float64
    )
    solve = solve_nested_physical_system(reference, base_design, raw_a, residual)
    scaled_base, scaled_extension, trust = uniform_joint_linf_scale(
        solve.base_step, solve.extension_coefficients, 0.25
    )
    scaled_maximum = float(torch.cat((scaled_base, scaled_extension)).abs().max())

    tolerance = reference.tolerance
    checks = {
        "sign_permutation_invariance": invariant_error <= 32.0 * tolerance,
        "inside_base_rank_zero": inside.extension_rank == 0,
        "inside_base_delta_zero": inside.nested_delta <= 32.0 * tolerance,
        "threshold_below_rejected_above_retained": threshold.extension_rank == 2,
        "explicit_nested_projector_agreement": (
            abs(reference.nested_delta - explicit_increment) <= 64.0 * tolerance
        ),
        "physical_solve_finite": bool(
            torch.isfinite(solve.base_step).all()
            and torch.isfinite(solve.extension_coefficients).all()
        ),
        "joint_trust_scale": scaled_maximum <= 0.25 + tolerance and 0.0 < trust <= 1.0,
    }
    return {
        "protocol": PROTOCOL,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "details": {
            "reference_delta": reference.nested_delta,
            "transformed_delta": invariant.nested_delta,
            "invariance_abs_error": invariant_error,
            "inside_base_rank": inside.extension_rank,
            "inside_base_delta": inside.nested_delta,
            "threshold_rank": threshold.extension_rank,
            "threshold_cutoff": threshold.cutoff,
            "explicit_increment": explicit_increment,
            "physical_rank": solve.numerical_rank,
            "trust_factor": trust,
            "scaled_linf": scaled_maximum,
        },
    }


def _binding(runner_dir: Path, science_dir: Path) -> dict[str, Any]:
    runner_config = runner_dir / "runner_config.json"
    science_config = science_dir / "science_config.json"
    required_inputs = {
        "runner_config.json": runner_config,
        "parents.jsonl": runner_dir / "parents.jsonl",
        "science_config.json": science_config,
        "immediate_cells.jsonl": science_dir / "immediate_cells.jsonl",
        "diagnostic_scores.jsonl": science_dir / "diagnostic_scores.jsonl",
        "completed_units.jsonl": science_dir / "completed_units.jsonl",
    }
    missing = [name for name, path in required_inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BENCH-011 inputs are missing: {missing}")
    runner = _load_json(runner_config)
    science = _load_json(science_config)
    if science.get("binding", {}).get("runner_binding_sha256") != runner.get("binding_sha256"):
        raise RuntimeError("BENCH-009 runner/science binding link does not match")
    binding = {
        "protocol": PROTOCOL,
        "scope": "spent_bench009_near_plateau_current_identities_only",
        "runner_binding_sha256": runner.get("binding_sha256"),
        "science_binding_sha256": science.get("binding_sha256"),
        "input_sha256": {name: _sha256_file(path) for name, path in required_inputs.items()},
        "source_sha256": _source_hashes(),
        "execution_environment": _execution_environment(),
        "frozen_axes": {
            "parent_horizon": "near_plateau",
            "parent_iterations": 160,
            "families": list(manifest_module.TARGET_FAMILIES),
            "seeds": list(manifest_module.SEEDS),
            "folds": list(manifest_module.FOLDS),
            "candidates": list(CANDIDATES),
            "rcond": PRIMARY_RCOND,
            "damping": PRIMARY_DAMPING,
            "trust_radii": list(TRUST_RADII),
            "expected_units": EXPECTED_UNITS,
            "expected_native_renders": EXPECTED_ROWS,
            "factorization_seed_source": (
                "exact BENCH-009 crossfit unit ID over target_id, seed, parent_horizon, "
                "selection_scope, heldout_fold"
            ),
        },
        "claim_boundary": {
            "spent_data_diagnostic_only": True,
            "new_data_claim": False,
            "method_promotion_authorized": False,
            "quality_claim_authorized": False,
            "convergence_claim_authorized": False,
            "performance_claim_authorized": False,
            "compression_claim_authorized": False,
        },
    }
    return {"binding": binding, "binding_sha256": _sha256_bytes(_canonical_json(binding))}


def _initialize_output(outdir: Path, binding: Mapping[str, Any], invocation: Mapping[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "diagnostic_config.json"
    wanted = {**binding, "invocation": dict(invocation)}
    if config_path.is_file():
        current = _load_json(config_path)
        if current.get("binding_sha256") != binding["binding_sha256"]:
            raise RuntimeError("output directory contains a different BENCH-011 binding")
    else:
        _write_json(config_path, wanted)
    controls = run_algebra_controls()
    _write_json(outdir / "algebra_controls.json", controls)
    if controls["status"] != "pass":
        raise RuntimeError("BENCH-011 algebra controls failed")


def _parents(runner_dir: Path, runner_binding: str) -> dict[tuple[str, int], dict[str, Any]]:
    records = _load_jsonl(runner_dir / "parents.jsonl", "parent_id")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records.values():
        if record.get("horizon_name") != "near_plateau":
            continue
        if record.get("binding_sha256") != runner_binding:
            raise RuntimeError("parent row has a stale runner binding")
        if int(record.get("iterations", -1)) != 160:
            raise RuntimeError("near-plateau parent is not the frozen step-160 checkpoint")
        path = Path(str(record["field_path"]))
        if not path.is_file() or _sha256_file(path) != record.get("field_file_sha256"):
            raise RuntimeError("parent field file is missing or changed")
        key = (str(record["target_id"]), int(record["seed"]))
        if key in result:
            raise RuntimeError(f"duplicate near-plateau parent {key}")
        result[key] = record
    if len(result) != 12:
        raise RuntimeError(f"expected 12 near-plateau parents, found {len(result)}")
    return result


def _identity_index(
    science_dir: Path,
    science_binding: str,
) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    rows = _load_jsonl(science_dir / "immediate_cells.jsonl", "cell_id")
    chosen: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    source_actions = {"affine6": "j_plus_affine6", "carrier6": "j_plus_carrier6"}
    for candidate, action in source_actions.items():
        for row in rows.values():
            if not (
                row.get("binding_sha256") == science_binding
                and row.get("status") == "ok"
                and row.get("selection_scope") == "crossfit"
                and row.get("parent_horizon") == "near_plateau"
                and row.get("action") == action
                and float(row.get("damping")) == PRIMARY_DAMPING
                and float(row.get("rcond")) == PRIMARY_RCOND
                and float(row.get("trust_radius")) in TRUST_RADII
            ):
                continue
            key = (
                str(row["target_id"]),
                int(row["seed"]),
                int(row["heldout_fold"]),
                candidate,
            )
            existing = chosen.get(key)
            projection = {
                "identity": row.get("identity"),
                "identity_sha256": row.get("identity_sha256"),
                "source_cell_ids": [str(row["cell_id"])],
                "source_realized_mse_gain": {str(row["trust_radius"]): row["realized_mse_gain"]},
            }
            if existing is None:
                chosen[key] = projection
            else:
                if (
                    existing["identity"] != projection["identity"]
                    or existing["identity_sha256"] != projection["identity_sha256"]
                ):
                    raise RuntimeError(f"identity changed across frozen trust radii for {key}")
                existing["source_cell_ids"].extend(projection["source_cell_ids"])
                existing["source_realized_mse_gain"].update(
                    projection["source_realized_mse_gain"]
                )
    if len(chosen) != 48:
        raise RuntimeError(f"expected 48 frozen candidate identities, found {len(chosen)}")
    if any(len(value["source_cell_ids"]) != 2 for value in chosen.values()):
        raise RuntimeError("a frozen identity does not link both trust-radius source rows")
    return chosen


def _unit_specs() -> list[dict[str, Any]]:
    targets = manifest_module.target_suite()
    result: list[dict[str, Any]] = []
    for family in manifest_module.TARGET_FAMILIES:
        target = targets[family]
        for seed in manifest_module.SEEDS:
            for fold in manifest_module.FOLDS:
                axes = {
                    "target_id": target["target_id"],
                    "family": family,
                    "seed": int(seed),
                    "parent_horizon": "near_plateau",
                    "heldout_fold": int(fold),
                }
                bench009_axes = {
                    "target_id": target["target_id"],
                    "seed": int(seed),
                    "parent_horizon": "near_plateau",
                    "selection_scope": "crossfit",
                    "heldout_fold": int(fold),
                }
                result.append(
                    {
                        **axes,
                        "unit_id": _stable_id(axes),
                        "bench009_unit_id": _stable_id(bench009_axes),
                    }
                )
    if len(result) != EXPECTED_UNITS:
        raise RuntimeError("BENCH-011 unit manifest drifted")
    return sorted(result, key=lambda value: str(value["unit_id"]))


def _validate_bench009_unit_ids(
    science_dir: Path,
    science_binding: str,
    specs: Sequence[Mapping[str, Any]],
) -> None:
    completed = _load_jsonl(science_dir / "completed_units.jsonl", "unit_id")
    for spec in specs:
        identifier = str(spec["bench009_unit_id"])
        record = completed.get(identifier)
        if record is None:
            raise RuntimeError(f"BENCH-009 completed-unit ledger lacks {identifier}")
        if record.get("binding_sha256") != science_binding or record.get("status") != "ok":
            raise RuntimeError(f"BENCH-009 completed unit {identifier} is not bound terminal ok")


def _base_reduced_design(
    factorization: MatrixFreeTangentFactorization,
    q_j: torch.Tensor,
) -> torch.Tensor:
    range_basis = factorization.range_result.basis.detach().cpu().to(torch.float64)
    reduced = factorization.reduced_matrix.detach().cpu().to(torch.float64)
    return q_j.T @ range_basis @ reduced


def _mask_mse(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    difference = render.reshape(-1)[mask] - target.reshape(-1)[mask]
    return float(difference.square().mean())


def _candidate_columns(
    field: GaussianField,
    parameterization: Any,
    identity: Mapping[str, Any],
    candidate: str,
) -> torch.Tensor:
    row = int(identity["parent_row"])
    if candidate == "affine6":
        return affine_six_columns(
            field,
            row,
            parameterization.height,
            parameterization.width,
            support=parameterization.frozen_support,
        )
    if candidate == "carrier6":
        frequency = tuple(float(value) for value in identity["frequency_xy"])
        return carrier_six_columns(
            field,
            row,
            frequency,
            parameterization.height,
            parameterization.width,
            support=parameterization.frozen_support,
        )
    raise ValueError(f"unknown candidate {candidate!r}")


def _native_render(
    candidate: str,
    parameterization: Any,
    base_step: torch.Tensor,
    extension: torch.Tensor,
    identity: Mapping[str, Any],
) -> tuple[GaussianField, torch.Tensor]:
    updated = parameterization.field_at(base_step)
    row = int(identity["parent_row"])
    if candidate == "affine6":
        render = realize_affine_six(
            updated,
            row,
            extension,
            parameterization.height,
            parameterization.width,
        )
    elif candidate == "carrier6":
        render = realize_carrier_six(
            updated,
            row,
            tuple(float(value) for value in identity["frequency_xy"]),
            extension,
            parameterization.height,
            parameterization.width,
        )
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    return updated, render


def _execute_unit(
    spec: Mapping[str, Any],
    parent_record: Mapping[str, Any],
    identities: Mapping[tuple[str, int, int, str], Mapping[str, Any]],
    binding_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    field = GaussianField.load(str(parent_record["field_path"]), device="cpu")
    if field_sha256(field) != parent_record.get("field_sha256"):
        raise RuntimeError("loaded parent tensor hash changed")
    target_record = manifest_module.target_suite()[str(spec["family"])]
    if target_record["target_id"] != spec["target_id"]:
        raise RuntimeError("target family/id map drifted")
    target = torch.as_tensor(target_record["image"], dtype=field.means.dtype)
    height, width = int(target.shape[0]), int(target.shape[1])
    parameterization = make_no_opacity_parameterization(
        field,
        height,
        width,
        sigma_cutoff=manifest_module.SIGMA_CUTOFF,
        render_chunk=RENDER_CHUNK,
    )
    flat_render = parameterization.flat_render_function()
    discovery_mask, heldout_mask = checkerboard_fold_masks(
        height, width, int(spec["heldout_fold"])
    )
    discovery_mask = discovery_mask.to(device=field.means.device)
    heldout_mask = heldout_mask.to(device=field.means.device)
    discovery_operator = build_masked_jacobian_operators(
        flat_render, parameterization.physical_vector, discovery_mask
    )
    heldout_operator = build_masked_jacobian_operators(
        flat_render, parameterization.physical_vector, heldout_mask
    )
    factors = factorize_masked_tangent(
        discovery_operator,
        rconds=(PRIMARY_RCOND,),
        seed=int(str(spec["bench009_unit_id"])[:16], 16),
        sample_cap=discovery_operator.parameter_count,
    )
    factor = factors[PRIMARY_RCOND]
    base = render_normalized(field, height, width)
    residual = (target - base).reshape(-1)
    discovery_residual64 = residual[discovery_mask].detach().cpu().to(torch.float64)
    score_residual = residual[heldout_mask]
    q_j_input = factor.left_basis.detach().cpu().to(torch.float64)
    rows: list[dict[str, Any]] = []
    candidate_diagnostics: dict[str, Any] = {}
    for candidate in CANDIDATES:
        identity_source = identities[
            (
                str(spec["target_id"]),
                int(spec["seed"]),
                int(spec["heldout_fold"]),
                candidate,
            )
        ]
        identity = identity_source["identity"]
        if not isinstance(identity, dict):
            raise RuntimeError("frozen candidate identity is not an object")
        columns = _candidate_columns(field, parameterization, identity, candidate)
        discovery_columns64 = columns[discovery_mask].detach().cpu().to(torch.float64)
        nested = nested_extension_basis(
            q_j_input,
            discovery_columns64,
            discovery_residual64,
            parameter_count=discovery_operator.parameter_count,
        )
        reduced = _base_reduced_design(factor, nested.base_basis)
        solved = solve_nested_physical_system(
            nested,
            reduced,
            discovery_columns64,
            discovery_residual64,
        )
        candidate_diagnostics[candidate] = {
            "identity_sha256": identity_source["identity_sha256"],
            "base_rank": int(nested.base_basis.shape[1]),
            "extension_rank": nested.extension_rank,
            "nested_delta": nested.nested_delta,
            "cutoff": nested.cutoff,
            "tolerance": nested.tolerance,
            "base_basis_sha256": tensor_sha256(nested.base_basis),
            "extension_basis_sha256": tensor_sha256(nested.extension_basis),
            "residualized_columns_sha256": tensor_sha256(nested.residualized_columns),
            "compact_design_sha256": tensor_sha256(solved.compact_design),
        }
        for radius in TRUST_RADII:
            base64, extension64, trust = uniform_joint_linf_scale(
                solved.base_step, solved.extension_coefficients, radius
            )
            base_step = base64.to(dtype=field.means.dtype)
            extension = extension64.to(dtype=field.means.dtype)
            predicted_score = heldout_operator.jvp(base_step) + columns[heldout_mask] @ extension
            predicted_mse = float((score_residual - predicted_score).square().mean())
            base_mse = _mask_mse(base, target, heldout_mask)
            updated, native = _native_render(
                candidate, parameterization, base_step, extension, identity
            )
            realized_mse = _mask_mse(native, target, heldout_mask)
            support = support_membership_change_summary(
                parameterization.frozen_support, updated
            ).as_record()
            axes = {
                "unit_id": spec["unit_id"],
                "candidate": candidate,
                "trust_radius": float(radius),
            }
            row = {
                "schema": ROW_SCHEMA,
                "row_id": _stable_id({"binding_sha256": binding_sha256, **axes}),
                "binding_sha256": binding_sha256,
                "status": "ok",
                **dict(spec),
                "candidate": candidate,
                "trust_radius": float(radius),
                "rcond": PRIMARY_RCOND,
                "damping": PRIMARY_DAMPING,
                "identity": identity,
                "identity_sha256": identity_source["identity_sha256"],
                "bench009_source_cell_ids": identity_source["source_cell_ids"],
                "base_rank": int(nested.base_basis.shape[1]),
                "extension_rank": nested.extension_rank,
                "physical_solve_rank": solved.numerical_rank,
                "physical_solve_condition": solved.condition,
                "nested_delta": nested.nested_delta,
                "nested_delta_per_discovery_row": (
                    nested.nested_delta / max(1, int(discovery_mask.sum()))
                ),
                "threshold_reference_sigma_max_a": (
                    float(nested.original_singular_values[0])
                    if nested.original_singular_values.numel()
                    else 0.0
                ),
                "residualized_singular_values": [
                    float(value) for value in nested.residualized_singular_values
                ],
                "nested_cutoff": nested.cutoff,
                "numerical_tolerance": nested.tolerance,
                "base_orthogonality_max_abs": nested.base_orthogonality_max_abs,
                "extension_orthogonality_max_abs": nested.extension_orthogonality_max_abs,
                "cross_orthogonality_max_abs": nested.cross_orthogonality_max_abs,
                "residual_base_component_max_abs": nested.residual_base_component_max_abs,
                "trust_scale": trust,
                "base_parameter_step": [float(value) for value in base64],
                "extension_coefficients": [float(value) for value in extension64],
                "base_parameter_step_sha256": tensor_sha256(base64),
                "extension_coefficients_sha256": tensor_sha256(extension64),
                "base_mse": base_mse,
                "predicted_mse": predicted_mse,
                "predicted_mse_gain": base_mse - predicted_mse,
                "realized_mse": realized_mse,
                "realized_mse_gain": base_mse - realized_mse,
                "native_render_sha256": tensor_sha256(native),
                "updated_field_sha256": field_sha256(updated),
                "support": support,
                "heldout_refit": False,
                "identity_reselected": False,
                "spent_data_diagnostic_only": True,
                "method_promotion_authorized": False,
                "quality_claim_authorized": False,
                "convergence_claim_authorized": False,
                "performance_claim_authorized": False,
                "compression_claim_authorized": False,
            }
            rows.append(row)
    marker = {
        "unit_id": spec["unit_id"],
        "binding_sha256": binding_sha256,
        "status": "ok",
        "row_count": len(rows),
        "candidate_diagnostics": candidate_diagnostics,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    return rows, marker


def run_shard(
    *,
    runner_dir: Path,
    science_dir: Path,
    outdir: Path,
    shard_index: int,
    shard_count: int,
    max_units: int | None,
) -> dict[str, Any]:
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    binding = _binding(runner_dir, science_dir)
    invocation = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "runner_dir": str(runner_dir.resolve()),
        "science_dir": str(science_dir.resolve()),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "max_units": max_units,
    }
    _initialize_output(outdir, binding, invocation)
    runner_binding = str(binding["binding"]["runner_binding_sha256"])
    science_binding = str(binding["binding"]["science_binding_sha256"])
    parents = _parents(runner_dir, runner_binding)
    identities = _identity_index(science_dir, science_binding)
    specs = _unit_specs()
    _validate_bench009_unit_ids(science_dir, science_binding, specs)
    chosen = [
        spec
        for spec in specs
        if int(str(spec["unit_id"]), 16) % shard_count == shard_index
    ]
    existing_rows = _load_jsonl(outdir / "rows.jsonl", "row_id") if (outdir / "rows.jsonl").is_file() else {}
    completed = (
        _load_jsonl(outdir / "completed_units.jsonl", "unit_id")
        if (outdir / "completed_units.jsonl").is_file()
        else {}
    )
    pending = [spec for spec in chosen if str(spec["unit_id"]) not in completed]
    if max_units is not None:
        pending = pending[:max_units]
    completed_this_invocation = 0
    for spec in pending:
        key = (str(spec["target_id"]), int(spec["seed"]))
        try:
            rows, marker = _execute_unit(
                spec,
                parents[key],
                identities,
                str(binding["binding_sha256"]),
            )
            for row in rows:
                identifier = str(row["row_id"])
                if identifier not in existing_rows:
                    _append_jsonl(outdir / "rows.jsonl", row)
                    existing_rows[identifier] = row
        except Exception as exc:
            marker = {
                "unit_id": spec["unit_id"],
                "binding_sha256": binding["binding_sha256"],
                "status": "error",
                "row_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        _append_jsonl(outdir / "completed_units.jsonl", marker)
        completed[str(spec["unit_id"])] = marker
        completed_this_invocation += 1
        _write_json(
            outdir / "run_state.json",
            {
                "binding_sha256": binding["binding_sha256"],
                "completed_units_this_invocation": completed_this_invocation,
                "last_completed_unit_id": spec["unit_id"],
                "selected_unit_count": len(chosen),
            },
        )
    errors = [value for value in completed.values() if value.get("status") != "ok"]
    return {
        "phase": "run",
        "binding_sha256": binding["binding_sha256"],
        "selected_units": len(chosen),
        "terminal_units": len(completed),
        "rows": len(existing_rows),
        "errors": len(errors),
        "completed_units_this_invocation": completed_this_invocation,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    x = _average_ranks(first)
    y = _average_ranks(second)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def _calibration_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    for row in rows:
        floor = max(1e-12, float(row["base_mse"]) * 1e-5)
        (eligible if float(row["predicted_mse_gain"]) > floor else excluded).append(row)
    predicted = np.asarray(
        [float(row["predicted_mse_gain"]) for row in eligible], dtype=np.float64
    )
    realized = np.asarray(
        [float(row["realized_mse_gain"]) for row in eligible], dtype=np.float64
    )
    enough = (
        len(eligible) >= 3
        and len(np.unique(predicted)) >= 3
        and len(np.unique(realized)) >= 3
    )
    result: dict[str, Any] = {
        "raw_row_count": len(rows),
        "eligible_row_count": len(eligible),
        "excluded_at_or_below_floor_count": len(excluded),
        "excluded_row_ids": [row["row_id"] for row in excluded],
        "negative_realized_gain_count": sum(
            float(row["realized_mse_gain"]) < 0.0 for row in rows
        ),
        "enough_non_tied_eligible_rows": enough,
    }
    if not enough:
        return {
            **result,
            "spearman": None,
            "median_realized_over_predicted": None,
            "gate_pass": False,
            "failure_reason": "fewer than three eligible non-tied rows",
        }
    spearman = _spearman(predicted, realized)
    ratio = float(np.median(realized / predicted))
    return {
        **result,
        "spearman": spearman,
        "median_realized_over_predicted": ratio,
        "spearman_minimum": 0.8,
        "gain_ratio_bounds": [0.5, 2.0],
        "gate_pass": math.isfinite(spearman) and spearman >= 0.8 and 0.5 <= ratio <= 2.0,
    }


def analyze_rows(rows: Sequence[Mapping[str, Any]], binding_sha256: str) -> dict[str, Any]:
    # Canonicalize reduction and reporting order so a replay from sorted canonical JSONL is
    # byte-stable with a replay from shard insertion order.
    rows = tuple(sorted(rows, key=lambda row: str(row.get("row_id"))))
    errors: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"row count is {len(rows)}, expected {EXPECTED_ROWS}")
    if any(row.get("binding_sha256") != binding_sha256 for row in rows):
        errors.append("one or more rows have a stale binding")
    if any(row.get("status") != "ok" for row in rows):
        errors.append("one or more rows are not terminal ok rows")
    if len({str(row.get("row_id")) for row in rows}) != len(rows):
        errors.append("row IDs are not unique")
    if len({str(row.get("unit_id")) for row in rows}) != EXPECTED_UNITS:
        errors.append("unit coverage is not exactly 24")
    expected_specs = {str(spec["unit_id"]): spec for spec in _unit_specs()}
    observed_by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        observed_by_unit.setdefault(str(row.get("unit_id")), []).append(row)
    if set(observed_by_unit) != set(expected_specs):
        errors.append("unit IDs do not equal the exact frozen 24-unit manifest")
    for unit_id, spec in expected_specs.items():
        unit_rows = observed_by_unit.get(unit_id, [])
        expected_cells = {(candidate, radius) for candidate in CANDIDATES for radius in TRUST_RADII}
        observed_cells = {
            (str(row.get("candidate")), float(row.get("trust_radius", -1.0)))
            for row in unit_rows
        }
        if len(unit_rows) != 4 or observed_cells != expected_cells:
            errors.append(f"unit {unit_id} does not contain the exact four-cell grid")
        frozen_axes = (
            "target_id",
            "family",
            "seed",
            "parent_horizon",
            "heldout_fold",
            "bench009_unit_id",
        )
        if any(any(row.get(axis) != spec[axis] for axis in frozen_axes) for row in unit_rows):
            errors.append(f"unit {unit_id} has drifted frozen axes")
    strata: dict[str, Any] = {}
    for candidate in CANDIDATES:
        for radius in TRUST_RADII:
            selected = [
                row
                for row in rows
                if row.get("candidate") == candidate
                and float(row.get("trust_radius", -1.0)) == radius
            ]
            key = f"candidate={candidate}|trust_radius={radius}"
            strata[key] = _calibration_cell(selected)
            if len(selected) != EXPECTED_UNITS:
                errors.append(f"{key} has {len(selected)} rows, expected {EXPECTED_UNITS}")
    algebra_errors: list[str] = []
    for row in rows:
        tolerance = float(row.get("numerical_tolerance", -1.0))
        if float(row.get("nested_delta", -math.inf)) < -max(0.0, tolerance):
            algebra_errors.append(f"negative nested delta at {row.get('row_id')}")
        rank = int(row.get("extension_rank", -1))
        if not 0 <= rank <= 6:
            algebra_errors.append(f"invalid extension rank at {row.get('row_id')}")
        if max(
            float(row.get("base_orthogonality_max_abs", math.inf)),
            float(row.get("extension_orthogonality_max_abs", math.inf)),
            float(row.get("cross_orthogonality_max_abs", math.inf)),
        ) > tolerance:
            algebra_errors.append(f"orthogonality tolerance failed at {row.get('row_id')}")
    errors.extend(algebra_errors)
    family_means: dict[str, Any] = {}
    for family in manifest_module.TARGET_FAMILIES:
        family_rows = [row for row in rows if row.get("family") == family]
        family_means[family] = {
            "row_count": len(family_rows),
            "mean_predicted_mse_gain": (
                float(np.mean([float(row["predicted_mse_gain"]) for row in family_rows]))
                if family_rows
                else None
            ),
            "mean_realized_mse_gain": (
                float(np.mean([float(row["realized_mse_gain"]) for row in family_rows]))
                if family_rows
                else None
            ),
            "mean_nested_delta": (
                float(np.mean([float(row["nested_delta"]) for row in family_rows]))
                if family_rows
                else None
            ),
        }
    gates_pass = not errors and len(strata) == 4 and all(
        value["gate_pass"] for value in strata.values()
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "protocol": PROTOCOL,
        "binding_sha256": binding_sha256,
        "status": "available" if not errors else "unavailable",
        "row_count": len(rows),
        "unit_count": len({str(row.get("unit_id")) for row in rows}),
        "algebra_valid": not algebra_errors,
        "calibration_strata": strata,
        "all_frozen_calibration_strata_pass": gates_pass,
        "classification": "diagnostic_pass" if gates_pass else "diagnostic_fail",
        "family_means_descriptive_only": family_means,
        "errors": errors,
        "no_rescue_rule": (
            "close_without_retuning" if not gates_pass else "may_preregister_disjoint_assay"
        ),
        "claim_boundary": {
            "spent_data_diagnostic_only": True,
            "method_promotion_authorized": False,
            "quality_claim_authorized": False,
            "convergence_claim_authorized": False,
            "performance_claim_authorized": False,
            "compression_claim_authorized": False,
        },
    }


def merge_shards(*, shard_dirs: Sequence[Path], outdir: Path) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    configs = [_load_json(directory / "diagnostic_config.json") for directory in shard_dirs]
    bindings = {str(config.get("binding_sha256")) for config in configs}
    if len(bindings) != 1:
        raise RuntimeError("shard bindings differ")
    binding_sha256 = next(iter(bindings))
    rows: dict[str, dict[str, Any]] = {}
    units: dict[str, dict[str, Any]] = {}
    for directory in shard_dirs:
        controls = _load_json(directory / "algebra_controls.json")
        if controls.get("status") != "pass":
            raise RuntimeError(f"algebra controls failed in {directory}")
        for identifier, row in _load_jsonl(directory / "rows.jsonl", "row_id").items():
            if identifier in rows and rows[identifier] != row:
                raise RuntimeError(f"conflicting duplicate row {identifier}")
            rows[identifier] = row
        for identifier, unit in _load_jsonl(
            directory / "completed_units.jsonl", "unit_id"
        ).items():
            if identifier in units and units[identifier] != unit:
                raise RuntimeError(f"conflicting duplicate unit {identifier}")
            units[identifier] = unit
    outdir.mkdir(parents=True, exist_ok=True)
    _write_json(
        outdir / "diagnostic_config.json",
        {
            "binding": configs[0]["binding"],
            "binding_sha256": binding_sha256,
            "invocation": {
                "merged_at": datetime.now(timezone.utc).isoformat(),
                "shard_dirs": [str(directory.resolve()) for directory in shard_dirs],
            },
        },
    )
    _write_json(outdir / "algebra_controls.json", run_algebra_controls())
    _write_jsonl(
        outdir / "rows.jsonl",
        sorted(rows.values(), key=lambda value: str(value["row_id"])),
    )
    _write_jsonl(
        outdir / "completed_units.jsonl",
        sorted(units.values(), key=lambda value: str(value["unit_id"])),
    )
    analysis = analyze_rows(list(rows.values()), binding_sha256)
    if len(units) != EXPECTED_UNITS:
        analysis["status"] = "unavailable"
        analysis["classification"] = "diagnostic_fail"
        analysis["all_frozen_calibration_strata_pass"] = False
        analysis["errors"].append(
            f"terminal unit count is {len(units)}, expected {EXPECTED_UNITS}"
        )
    unit_errors = [unit for unit in units.values() if unit.get("status") != "ok"]
    if unit_errors:
        analysis["status"] = "unavailable"
        analysis["classification"] = "diagnostic_fail"
        analysis["all_frozen_calibration_strata_pass"] = False
        analysis["errors"].append(f"{len(unit_errors)} terminal units contain errors")
    _write_json(outdir / "analysis.json", analysis)
    _write_json(
        outdir / "merge_state.json",
        {
            "binding_sha256": binding_sha256,
            "shard_count": len(shard_dirs),
            "unit_count": len(units),
            "row_count": len(rows),
            "unit_error_count": len(unit_errors),
            "analysis_status": analysis["status"],
            "classification": analysis["classification"],
        },
    )
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    controls = subparsers.add_parser("controls", help="run only deterministic algebra controls")
    controls.add_argument("--output", type=Path)
    run = subparsers.add_parser("run", help="run one resumable spent-data shard")
    run.add_argument("--runner-dir", type=Path, required=True)
    run.add_argument("--science-dir", type=Path, required=True)
    run.add_argument("--outdir", type=Path, required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--max-units", type=int)
    merge = subparsers.add_parser("merge", help="merge complete disjoint shards and analyze")
    merge.add_argument("--shard-dir", type=Path, action="append", required=True)
    merge.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "controls":
        result = run_algebra_controls()
        if args.output is not None:
            _write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "pass" else 2
    if args.command == "run":
        result = run_shard(
            runner_dir=args.runner_dir,
            science_dir=args.science_dir,
            outdir=args.outdir,
            shard_index=args.shard_index,
            shard_count=args.num_shards,
            max_units=args.max_units,
        )
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["errors"] == 0 else 2
    result = merge_shards(shard_dirs=args.shard_dir, outdir=args.outdir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["all_frozen_calibration_strata_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATES",
    "NestedExtensionBasis",
    "NestedPhysicalSolve",
    "TRUST_RADII",
    "analyze_rows",
    "main",
    "nested_extension_basis",
    "orthonormalize_fixed_base",
    "run_algebra_controls",
    "solve_nested_physical_system",
    "uniform_joint_linf_scale",
]
