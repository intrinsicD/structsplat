"""Fail-closed scientific execution layer for BENCH-009 Stage 1.

The executable has two deliberately different phases:

``toy-preflight``
    Exercises the complete numerical path on a synthetic field which is not part of the frozen
    development target manifest.  It is safe to run while the development screen is locked.

``immediate``
    Consumes the 24 source-bound parent checkpoints and emits the frozen 3,072 cross-fit and
    1,536 excluded all-pixel-oracle cells.  Work is resumable at a parent/scope unit, sharded by a
    stable unit ID, and every selected cell receives exactly one terminal ``ok`` or ``error`` row.

The scientific Jacobian is never materialized.  Current-grammar actions use matrix-free JVP/VJP
range discovery; only six-column packet designs and compact reduced factorizations are explicit.
This module intentionally does not analyze gates, authorize a method, claim actual rate, or run
the separately preregistered recovery component.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shlex
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from benchmarks import residual_tangent_auction_stage1 as manifest_module
from benchmarks.residual_tangent_auction_matrixfree import (
    DAMPING_DIMENSIONLESS_PARENT_CHART,
    MaskedJacobianOperators,
    MatrixFreeTangentFactorization,
    build_masked_jacobian_operators,
    factorize_joint_tangent_six,
    factorize_masked_tangent,
    literal_projector_energy,
    select_tangent_six_all_pixel_oracle,
    select_tangent_six_crossfit,
    solve_damped_normal,
    solve_joint_damped_normal,
    tensor_sha256,
    uniform_linf_scale_joint_coefficients,
)
from benchmarks.residual_tangent_auction_stage1_core import (
    BirthGeometry,
    FrozenPixelSupport,
    TwoBirthPacket,
    affine_six_columns,
    build_two_birth_packet,
    carrier_six_columns,
    centered_fd_jvp_validation,
    checkerboard_fold_masks,
    freeze_pixel_support,
    make_no_opacity_parameterization,
    packet_linear_realization,
    realize_affine_six,
    realize_carrier_six,
    realize_two_birth_six,
    render_normalized,
)
from benchmarks.residual_tangent_auction_stage1_runner import (
    CARRIER_BANKS,
    PRICE_SCOPE,
    RENDER_CHUNK,
    discovery_birth_geometry_bank,
    field_sha256,
)
from structsplat.gaussians import GaussianField


PROTOCOL = "BENCH-009 Stage-1 source-bound scientific immediate runner v1"
SCOPE = "frozen_immediate_cells_without_interpretation_or_recovery"
TOY_SCOPE = "synthetic_full_path_preflight_not_development_data"
EXPECTED_CROSSFIT_CELLS = 3_072
EXPECTED_ORACLE_CELLS = 1_536
EXPECTED_SCORE_ROWS = 1_152
EXPECTED_SMALL_CARRIER_SENSITIVITY_CELLS = 288
ACTION_COUNT = 8
CELLS_PER_SCOPE_UNIT = (
    ACTION_COUNT
    * len(manifest_module.DAMPINGS)
    * len(manifest_module.RCONDS)
    * len(manifest_module.TRUST_RADII)
)
WORK_SPECIFICATION = {
    "version": 1,
    "candidate_fit": "one scaled/truncated six-column solve",
    "linear_solve": "one damped reduced solve",
    "support_weight_evaluation": "one Gaussian-row/image compact support evaluation",
    "basis_sample": "one scalar basis value at one rendered pixel",
    "carrier_trig": "one paired sine/cosine evaluation",
    "rgb_fma": "one scalar multiply-add in one RGB channel",
    "normalized_division": "one output-channel normalized division",
    "rendered_pixel": "one RGB output location",
    "price_scope": PRICE_SCOPE,
}

SOURCE_PATHS = (
    "benchmarks/residual_tangent_auction_stage1_science.py",
    "benchmarks/residual_tangent_auction_stage1.py",
    "benchmarks/residual_tangent_auction_stage1_core.py",
    "benchmarks/residual_tangent_auction_stage1_runner.py",
    "benchmarks/residual_tangent_auction_matrixfree.py",
    "benchmarks/residual_tangent_auction_stage1_analysis.py",
    "benchmarks/residual_tangent_auction_stage1_recovery.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/render.py",
    "tasks/BENCH-009-residual-tangent-auction.md",
    "tests/test_residual_tangent_auction_stage1_science.py",
    "tests/test_residual_tangent_auction_stage1_analysis.py",
    "tests/test_residual_tangent_auction_stage1_recovery.py",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _live_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    missing = [relative for relative in SOURCE_PATHS if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"scientific source binding is incomplete: {missing}")
    return {relative: _sha256_file(root / relative) for relative in SOURCE_PATHS}


def _stable_id(payload: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload))


def _strict_load_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-strict JSON constant {value} in {path}")

    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object in {path}")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(
            line,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-strict JSON constant {item}")
            ),
        )
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"{path}:{line_number} lacks {key!r}")
        identifier = str(value[key])
        if identifier in result:
            raise ValueError(f"duplicate {key}={identifier} in {path}")
        result[identifier] = value
    return result


def _mask_mse(render: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    difference = (target - render).reshape(-1)[mask]
    return float(difference.square().mean().detach())


def _psnr(mse: float) -> float:
    return float(-10.0 * math.log10(max(float(mse), 1e-30)))


def _coordinate_ids(field: GaussianField) -> tuple[str, ...]:
    names: list[str] = []
    for group, width in (("mean", 2), ("log_scale", 2)):
        names.extend(
            f"row_{row:03d}.{group}_{axis}"
            for row in range(field.n)
            for axis in range(width)
        )
    names.extend(f"row_{row:03d}.rotation" for row in range(field.n))
    names.extend(
        f"row_{row:03d}.rgb_{channel}"
        for row in range(field.n)
        for channel in range(3)
    )
    return tuple(names)


def _direction(template: torch.Tensor, index: int) -> torch.Tensor:
    result = torch.zeros_like(template)
    result[index] = 1.0
    return result


def _solve_fixed_six(
    design: torch.Tensor,
    residual: torch.Tensor,
    *,
    damping: float,
    rcond: float,
) -> tuple[torch.Tensor, int, float | None]:
    """Solve in the physical six-coefficient chart after rank truncation."""

    if design.ndim != 2 or design.shape != (residual.numel(), 6):
        raise ValueError("fixed six design must have shape (residual rows, 6)")
    scales = torch.linalg.vector_norm(design, dim=0).clamp_min(1e-12)
    left, singular, vh = torch.linalg.svd(design / scales, full_matrices=False)
    keep = singular > float(rcond) * (singular[0] if singular.numel() else 0.0)
    retained = design
    # The retained image subspace defines rank, while damping is applied in the unscaled
    # dimensionless/RGB chart, matching the matrix-free parent solve.
    if not bool(keep.any()):
        return residual.new_zeros((6,)), 0, None
    basis = left[:, keep]
    reduced = basis.T @ retained
    rhs = basis.T @ residual
    raw_left, raw_singular, raw_vh = torch.linalg.svd(reduced, full_matrices=False)
    numerical = (
        32.0
        * torch.finfo(design.dtype).eps
        * max(reduced.shape)
        * raw_singular[0]
    )
    raw_keep = raw_singular > numerical
    retained_raw = raw_singular[raw_keep]
    inverse = retained_raw / (retained_raw.square() + float(damping) ** 2)
    coefficients = raw_vh[raw_keep].T @ (
        inverse * (raw_left[:, raw_keep].T @ rhs)
    )
    kept = singular[keep]
    return coefficients, int(keep.sum()), float(kept[0] / kept[-1])


def _uniform_scale(value: torch.Tensor, radius: float) -> tuple[torch.Tensor, float]:
    maximum = float(value.abs().max()) if value.numel() else 0.0
    factor = min(1.0, float(radius) / max(maximum, 1e-30))
    return value * factor, factor


def _support_record(parent: FrozenPixelSupport, updated: GaussianField) -> dict[str, Any]:
    native = freeze_pixel_support(
        updated,
        parent.height,
        parent.width,
        sigma_cutoff=parent.sigma_cutoff,
    )
    before = parent.membership.to(device=native.membership.device)
    after = native.membership
    added = after & ~before
    removed = before & ~after
    return {
        "parent_membership_sha256": tensor_sha256(before),
        "native_membership_sha256": tensor_sha256(after),
        "added_memberships": int(added.sum()),
        "removed_memberships": int(removed.sum()),
        "xor_memberships": int((added | removed).sum()),
        "parent_membership_count": int(before.sum()),
        "native_membership_count": int(after.sum()),
    }


@dataclass(frozen=True)
class SelectedIdentity:
    action: str
    identity: dict[str, Any]
    full_design: torch.Tensor
    offset: torch.Tensor
    packet: TwoBirthPacket | None = None
    candidate_count: int = 0

    @property
    def identity_sha256(self) -> str:
        return _stable_id(self.identity)


def _best_zero_offset_identity(
    candidates: Iterable[tuple[dict[str, Any], torch.Tensor]],
    residual: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    action: str,
    damping: float,
    rcond: float,
) -> SelectedIdentity:
    best: tuple[tuple[float, str], dict[str, Any], torch.Tensor] | None = None
    count = 0
    for identity, design in candidates:
        count += 1
        coefficients, _rank, _condition = _solve_fixed_six(
            design[fit_mask], residual[fit_mask], damping=damping, rcond=rcond
        )
        post = residual[fit_mask] - design[fit_mask] @ coefficients
        key = (float(post.square().mean()), _stable_id(identity))
        if best is None or key < best[0]:
            best = (key, identity, design)
    if best is None:
        raise RuntimeError(f"{action} candidate bank is empty")
    return SelectedIdentity(
        action=action,
        identity=best[1],
        full_design=best[2],
        offset=residual.new_zeros(residual.shape),
        candidate_count=count,
    )


def _select_affine(
    field: GaussianField,
    support: FrozenPixelSupport,
    residual: torch.Tensor,
    fit_mask: torch.Tensor,
) -> SelectedIdentity:
    candidates = (
        (
            {"parent_row": gid},
            affine_six_columns(
                field,
                gid,
                support.height,
                support.width,
                support=support,
            ),
        )
        for gid in range(field.n)
    )
    return _best_zero_offset_identity(
        candidates,
        residual,
        fit_mask,
        action="affine",
        damping=manifest_module.PRIMARY_DAMPING,
        rcond=manifest_module.PRIMARY_RCOND,
    )


def _select_carrier(
    field: GaussianField,
    support: FrozenPixelSupport,
    residual: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    bank_name: str,
) -> SelectedIdentity:
    bank = CARRIER_BANKS[bank_name]
    candidates = (
        (
            {
                "parent_row": gid,
                "bank_name": bank_name,
                "frequency_index": frequency.index,
                "frequency_xy": list(frequency.frequency_xy),
            },
            carrier_six_columns(
                field,
                gid,
                frequency.frequency_xy,
                support.height,
                support.width,
                support=support,
            ),
        )
        for gid in range(field.n)
        for frequency in bank
    )
    return _best_zero_offset_identity(
        candidates,
        residual,
        fit_mask,
        action=f"carrier_{bank_name}",
        damping=manifest_module.PRIMARY_DAMPING,
        rcond=manifest_module.PRIMARY_RCOND,
    )


def _select_birth(
    field: GaussianField,
    support: FrozenPixelSupport,
    residual: torch.Tensor,
    fit_mask: torch.Tensor,
) -> SelectedIdentity:
    geometries = discovery_birth_geometry_bank(
        residual,
        fit_mask,
        support.height,
        support.width,
        parent_count=field.n,
    )
    best: tuple[
        tuple[float, int, int], tuple[int, int], TwoBirthPacket, torch.Tensor
    ] | None = None
    pairs = tuple(itertools.combinations(range(len(geometries)), 2))
    for first, second in pairs:
        packet = build_two_birth_packet(
            field,
            (geometries[first], geometries[second]),
            support.height,
            support.width,
            frozen_parent_support=support,
        )
        adjusted = residual - packet.offset
        coefficients, _rank, _condition = _solve_fixed_six(
            packet.design[fit_mask],
            adjusted[fit_mask],
            damping=manifest_module.PRIMARY_DAMPING,
            rcond=manifest_module.PRIMARY_RCOND,
        )
        remaining = adjusted[fit_mask] - packet.design[fit_mask] @ coefficients
        key = (float(remaining.square().mean()), first, second)
        if best is None or key < best[0]:
            best = (key, (first, second), packet, coefficients)
    if best is None:
        raise RuntimeError("two-birth candidate bank is empty")
    pair = best[1]
    packet = best[2]
    identity = {
        "unordered_site_pair": list(pair),
        "geometry_bank_site_count": len(geometries),
        "geometries": [asdict(packet.geometries[0]), asdict(packet.geometries[1])],
        "geometry_source": "visible_scoring_rows_rgb_squared_residual_nms",
    }
    return SelectedIdentity(
        action="birth_pair",
        identity=identity,
        full_design=packet.design,
        offset=packet.offset,
        packet=packet,
        candidate_count=len(pairs),
    )


def _tangent_identity(
    field: GaussianField,
    full: MaskedJacobianOperators,
    discovery: MaskedJacobianOperators,
    heldout: MaskedJacobianOperators,
    discovery_residual: torch.Tensor,
    heldout_residual: torch.Tensor,
    *,
    oracle: bool,
) -> tuple[SelectedIdentity, Any]:
    kwargs = {
        "coordinate_ids": _coordinate_ids(field),
        "rcond": manifest_module.PRIMARY_RCOND,
        "damping": manifest_module.PRIMARY_DAMPING,
        "damping_coordinate_system": DAMPING_DIMENSIONLESS_PARENT_CHART,
    }
    if oracle:
        selected = select_tangent_six_all_pixel_oracle(
            discovery,
            heldout,
            discovery_residual,
            heldout_residual,
            **kwargs,
        )
    else:
        selected = select_tangent_six_crossfit(
            discovery,
            heldout,
            discovery_residual,
            heldout_residual,
            **kwargs,
        )
    full_design = torch.stack(
        [full.jvp(_direction(full.origin, index)) for index in selected.coordinate_indices],
        dim=1,
    )
    identity = {
        "coordinate_indices": list(selected.coordinate_indices),
        "coordinate_ids": list(selected.coordinate_ids),
        "selection_policy": selected.fit_policy,
    }
    return (
        SelectedIdentity(
            action="tangent6",
            identity=identity,
            full_design=full_design,
            offset=full_design.new_zeros((full_design.shape[0],)),
            candidate_count=discovery.parameter_count,
        ),
        selected,
    )


def _factorizations(
    operators: MaskedJacobianOperators,
    *,
    seed: int,
) -> dict[float, MatrixFreeTangentFactorization]:
    return factorize_masked_tangent(
        operators,
        rconds=manifest_module.RCONDS,
        seed=int(seed),
        sample_cap=operators.parameter_count,
    )


def _native_realization(
    action: str,
    parameterization: Any,
    base_step: torch.Tensor,
    extension: torch.Tensor,
    identity: SelectedIdentity | None,
) -> tuple[GaussianField, torch.Tensor, dict[str, Any]]:
    field = parameterization.field_at(base_step)
    height, width = parameterization.height, parameterization.width
    extra: dict[str, Any] = {}
    if action in ("full_j", "tangent6"):
        render = render_normalized(field, height, width)
    elif action in ("j_plus_affine6", "affine6"):
        if identity is None:
            raise RuntimeError("affine realization lacks a frozen identity")
        gid = int(identity.identity["parent_row"])
        render = realize_affine_six(field, gid, extension, height, width)
    elif action in ("j_plus_carrier6", "carrier6"):
        if identity is None:
            raise RuntimeError("carrier realization lacks a frozen identity")
        gid = int(identity.identity["parent_row"])
        frequency = tuple(float(v) for v in identity.identity["frequency_xy"])
        render = realize_carrier_six(field, gid, frequency, extension, height, width)
    elif action in ("j_plus_birth2_rgb6", "birth2_rgb6"):
        if identity is None or identity.packet is None:
            raise RuntimeError("birth realization lacks a frozen finite packet")
        packet = build_two_birth_packet(
            field,
            identity.packet.geometries,
            height,
            width,
        )
        render = realize_two_birth_six(field, packet, extension)
        extra["birth_membership_sha256"] = tensor_sha256(packet.birth_weights > 0)
        extra["birth_memberships"] = int((packet.birth_weights > 0).sum())
    else:
        raise ValueError(f"unknown action {action!r}")
    return field, render, extra


def _action_record(
    *,
    cell: dict[str, Any],
    binding_sha256: str,
    parameterization: Any,
    target: torch.Tensor,
    fit_mask: torch.Tensor,
    score_mask: torch.Tensor,
    fit_operator: MaskedJacobianOperators,
    score_operator: MaskedJacobianOperators,
    factorization: MatrixFreeTangentFactorization,
    residual: torch.Tensor,
    identity: SelectedIdentity | None,
    tangent_identity: SelectedIdentity,
) -> dict[str, Any]:
    started = time.perf_counter()
    action = str(cell["action"])
    damping = float(cell["damping"])
    rcond = float(cell["rcond"])
    radius = float(cell["trust_radius"])
    fit_residual = residual[fit_mask]
    score_residual = residual[score_mask]
    zero_base = fit_operator.origin.new_zeros((fit_operator.parameter_count,))
    zero_extension = fit_residual.new_zeros((6,))
    predicted_score: torch.Tensor
    trust_factor: float
    rank: int
    condition: float | None

    if action == "full_j":
        solved = solve_damped_normal(factorization, fit_residual, damping=damping)
        base_step, trust_factor = _uniform_scale(solved.parameter_step, radius)
        extension = zero_extension
        predicted_score = score_operator.jvp(base_step)
        rank, condition = solved.damping_rank, solved.damping_condition
    elif action == "tangent6":
        design_fit = tangent_identity.full_design[fit_mask]
        design_score = tangent_identity.full_design[score_mask]
        coefficients, rank, condition = _solve_fixed_six(
            design_fit, fit_residual, damping=damping, rcond=rcond
        )
        extension, trust_factor = _uniform_scale(coefficients, radius)
        base_step = zero_base.clone()
        indices = [int(v) for v in tangent_identity.identity["coordinate_indices"]]
        base_step[indices] = extension
        predicted_score = design_score @ extension
        extension = zero_extension
    elif action in ("affine6", "carrier6", "birth2_rgb6"):
        if identity is None:
            raise RuntimeError("matched packet action lacks an identity")
        adjusted = fit_residual - identity.offset[fit_mask]
        coefficients, rank, condition = _solve_fixed_six(
            identity.full_design[fit_mask],
            adjusted,
            damping=damping,
            rcond=rcond,
        )
        extension, trust_factor = _uniform_scale(coefficients, radius)
        base_step = zero_base
        predicted_score = identity.offset[score_mask] + (
            identity.full_design[score_mask] @ extension
        )
    else:
        if identity is None:
            raise RuntimeError("causal packet action lacks an identity")
        joint_factors = factorize_joint_tangent_six(
            factorization,
            identity.full_design[fit_mask],
            rconds=(rcond,),
        )
        finite_offset = (
            identity.offset[fit_mask] if "birth" in action else None
        )
        solved = solve_joint_damped_normal(
            joint_factors[rcond],
            fit_residual,
            damping=damping,
            finite_offset=finite_offset,
        )
        scaled = uniform_linf_scale_joint_coefficients(
            solved.base_parameter_step,
            solved.extension_coefficients,
            radius,
        )
        base_step = scaled.base_parameter_step
        extension = scaled.extension_coefficients
        trust_factor = scaled.factor
        predicted_score = score_operator.jvp(base_step) + (
            identity.full_design[score_mask] @ extension
        )
        if "birth" in action:
            predicted_score = predicted_score + identity.offset[score_mask]
        rank, condition = solved.damping_rank, solved.damping_condition

    updated, native, extra = _native_realization(
        action,
        parameterization,
        base_step,
        extension,
        identity,
    )
    base = render_normalized(
        parameterization.template,
        parameterization.height,
        parameterization.width,
    )
    base_mse = _mask_mse(base, target, score_mask)
    realized_mse = _mask_mse(native, target, score_mask)
    predicted_remaining = score_residual - predicted_score
    predicted_mse = float(predicted_remaining.square().mean())
    predicted_gain = base_mse - predicted_mse
    residual_energy_reduction = predicted_gain * int(score_mask.sum())
    support = _support_record(parameterization.frozen_support, updated)
    effective_identity = tangent_identity if action == "tangent6" else identity
    recovery_source_eligible = (
        cell["selection_scope"] == "crossfit"
        and damping == manifest_module.PRIMARY_DAMPING
        and rcond == manifest_module.PRIMARY_RCOND
    )
    source_hashes: dict[str, str] | None = None
    if recovery_source_eligible:
        # Persist the exact source-side contract consumed by the recovery engine.  Computing the
        # bundle here deliberately performs one independent native rerender; recovery later
        # performs the single required verification rerender and must reproduce every hash.
        from benchmarks.residual_tangent_auction_stage1_recovery import (
            capture_immediate_action_state,
            packet_for_action,
        )

        packet = packet_for_action(
            action,
            None if action in ("full_j", "tangent6") else extension,
            identity=(
                None if effective_identity is None else effective_identity.identity
            ),
        )
        captured = capture_immediate_action_state(
            source_cell_id=str(cell["cell_id"]),
            action=action,
            base_field=updated,
            packet=packet,
            target=target,
            discovery_mask=fit_mask,
            heldout_mask=score_mask,
            predicted_gain=predicted_gain,
            residual_energy_reduction=residual_energy_reduction,
            sigma_cutoff=parameterization.sigma_cutoff,
            render_chunk=parameterization.render_chunk,
        )
        if captured.expected_render_sha256 != tensor_sha256(native):
            raise RuntimeError("recovery source rerender differs from immediate native render")
        if field_sha256(captured.base_field) != field_sha256(updated):
            raise RuntimeError("recovery source state differs from immediate updated field")
        source_hashes = {
            "state_sha256": captured.expected_state_sha256,
            "packet_sha256": captured.expected_packet_sha256,
            "native_render_sha256": captured.expected_render_sha256,
            "native_support_sha256": captured.expected_support_sha256,
            "metric_sha256": captured.expected_metric_sha256,
        }
    packet_bytes = {
        "full_j": 0,
        "j_plus_affine6": 27,
        "j_plus_carrier6": 30,
        "j_plus_birth2_rgb6": 65,
        "tangent6": 37,
        "affine6": 27,
        "carrier6": 30,
        "birth2_rgb6": 65,
    }[action]
    pixels = parameterization.height * parameterization.width
    record = {
        **cell,
        "status": "ok",
        "binding_sha256": binding_sha256,
        "damping_coordinate_system": DAMPING_DIMENSIONLESS_PARENT_CHART,
        "identity": None if effective_identity is None else effective_identity.identity,
        "identity_sha256": (
            None if effective_identity is None else effective_identity.identity_sha256
        ),
        "identity_selected_at": {
            "damping": manifest_module.PRIMARY_DAMPING,
            "rcond": manifest_module.PRIMARY_RCOND,
            "frozen_across_numeric_grid": True,
        },
        "base_mse": base_mse,
        "base_psnr": _psnr(base_mse),
        "predicted_mse": predicted_mse,
        "predicted_psnr": _psnr(predicted_mse),
        "predicted_mse_gain": predicted_gain,
        "predicted_gain": predicted_gain,
        "residual_energy_reduction": residual_energy_reduction,
        "realized_mse": realized_mse,
        "realized_psnr": _psnr(realized_mse),
        "realized_mse_gain": base_mse - realized_mse,
        "realized_psnr_gain_db": _psnr(realized_mse) - _psnr(base_mse),
        "coefficient_linf_before_trust": (
            float(torch.cat((base_step, extension)).abs().max()) / max(trust_factor, 1e-30)
        ),
        "trust_scale": trust_factor,
        "effective_rank": int(rank),
        "condition_estimate": condition,
        "base_parameter_step_sha256": tensor_sha256(base_step),
        "extension_coefficients_sha256": tensor_sha256(extension),
        "base_parameter_step": [float(value) for value in base_step],
        "extension_coefficients": [float(value) for value in extension],
        "height": int(parameterization.height),
        "width": int(parameterization.width),
        "source_cell_id": str(cell["cell_id"]),
        "recovery_source_eligible": recovery_source_eligible,
        "source_hashes": source_hashes,
        "updated_field_sha256": field_sha256(updated),
        "native_render_sha256": tensor_sha256(native),
        "support": support,
        "heldout_refit": False if cell["selection_scope"] == "crossfit" else None,
        "oracle_excluded_from_gates": (
            cell["selection_scope"]
            == "all_pixel_identity_and_coefficient_oracle"
        ),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "work_specification_sha256": _stable_id(WORK_SPECIFICATION),
        "work_counters": {
            "counter_kind": "deterministic_analytical_protocol_count_not_instrumented",
            "selected_action_support_weight_evaluations": int(
                parameterization.template.n * pixels
            ),
            "selected_action_basis_samples": int(
                2 * pixels if "affine" in action or "carrier" in action else 0
            ),
            "selected_action_carrier_sin_cos_calls": int(
                pixels if "carrier" in action else 0
            ),
            "selected_action_rgb_fmas": int(6 * pixels),
            "selected_action_normalized_divisions": int(3 * pixels),
            "selected_action_rendered_pixels": int(pixels),
            "linear_solves": 1,
        },
        "provisional_packet_bytes": packet_bytes,
        "price_scope": PRICE_SCOPE,
        "scientific_interpretation_present": False,
        "recovery_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
        **extra,
    }
    _canonical_json(record)
    return record


@dataclass(frozen=True)
class ScopeExecution:
    action_records: tuple[dict[str, Any], ...]
    sensitivity_records: tuple[dict[str, Any], ...]
    selection_record: dict[str, Any]
    score_records: tuple[dict[str, Any], ...]


def _scope_execution(
    *,
    field: GaussianField,
    target: torch.Tensor,
    cells: Sequence[dict[str, Any]],
    sensitivity_cells: Sequence[dict[str, Any]] = (),
    binding_sha256: str,
    fold: int | None,
    seed: int,
) -> ScopeExecution:
    """Compute one parent/fold or parent/oracle unit and all of its 64 grid cells."""

    height, width = int(target.shape[0]), int(target.shape[1])
    parameterization = make_no_opacity_parameterization(
        field,
        height,
        width,
        sigma_cutoff=manifest_module.SIGMA_CUTOFF,
        render_chunk=RENDER_CHUNK,
    )
    flat_render = parameterization.flat_render_function()
    template = parameterization.physical_vector
    full_mask = torch.ones(target.numel(), dtype=torch.bool, device=target.device)
    full_operator = build_masked_jacobian_operators(flat_render, template, full_mask)
    pair_fold = 0 if fold is None else int(fold)
    discovery_mask, heldout_mask = checkerboard_fold_masks(height, width, pair_fold)
    discovery_mask = discovery_mask.to(device=target.device)
    heldout_mask = heldout_mask.to(device=target.device)
    discovery_operator = build_masked_jacobian_operators(
        flat_render, template, discovery_mask
    )
    heldout_operator = build_masked_jacobian_operators(
        flat_render, template, heldout_mask
    )
    base = render_normalized(field, height, width)
    residual = (target - base).reshape(-1)
    oracle = fold is None
    fit_mask = full_mask if oracle else discovery_mask
    score_mask = full_mask if oracle else heldout_mask
    fit_operator = full_operator if oracle else discovery_operator
    score_operator = full_operator if oracle else heldout_operator
    tangent, tangent_raw = _tangent_identity(
        field,
        full_operator,
        discovery_operator,
        heldout_operator,
        residual[discovery_mask],
        residual[heldout_mask],
        oracle=oracle,
    )
    affine = _select_affine(field, parameterization.frozen_support, residual, fit_mask)
    carrier = _select_carrier(
        field,
        parameterization.frozen_support,
        residual,
        fit_mask,
        bank_name="full",
    )
    carrier_small = _select_carrier(
        field,
        parameterization.frozen_support,
        residual,
        fit_mask,
        bank_name="small",
    )
    birth = _select_birth(field, parameterization.frozen_support, residual, fit_mask)
    identities = {
        "j_plus_affine6": affine,
        "affine6": affine,
        "j_plus_carrier6": carrier,
        "carrier6": carrier,
        "j_plus_birth2_rgb6": birth,
        "birth2_rgb6": birth,
    }
    factors = _factorizations(fit_operator, seed=seed)
    action_records = tuple(
        _action_record(
            cell=cell,
            binding_sha256=binding_sha256,
            parameterization=parameterization,
            target=target,
            fit_mask=fit_mask,
            score_mask=score_mask,
            fit_operator=fit_operator,
            score_operator=score_operator,
            factorization=factors[float(cell["rcond"])],
            residual=residual,
            identity=identities.get(str(cell["action"])),
            tangent_identity=tangent,
        )
        for cell in cells
    )
    sensitivity_records = tuple(
        _action_record(
            cell=cell,
            binding_sha256=binding_sha256,
            parameterization=parameterization,
            target=target,
            fit_mask=fit_mask,
            score_mask=score_mask,
            fit_operator=fit_operator,
            score_operator=score_operator,
            factorization=factors[manifest_module.PRIMARY_RCOND],
            residual=residual,
            identity=carrier_small,
            tangent_identity=tangent,
        )
        for cell in sensitivity_cells
    )

    selection_payload = {
        "tangent6": tangent.identity,
        "affine": affine.identity,
        "carrier_full": carrier.identity,
        "carrier_small_sensitivity": carrier_small.identity,
        "birth_pair": birth.identity,
    }
    selection_record = {
        "selection_record_id": _stable_id(
            {
                "binding_sha256": binding_sha256,
                "parent": cells[0]["target_id"],
                "seed": cells[0]["seed"],
                "horizon": cells[0]["parent_horizon"],
                "fold": fold,
            }
        ),
        "binding_sha256": binding_sha256,
        "selection_scope": cells[0]["selection_scope"],
        "heldout_fold": fold,
        "selected_at_primary_damping": manifest_module.PRIMARY_DAMPING,
        "selected_at_primary_rcond": manifest_module.PRIMARY_RCOND,
        "identities": selection_payload,
        "identity_sha256": _stable_id(selection_payload),
        "candidate_counts": {
            "tangent": tangent.candidate_count,
            "affine": affine.candidate_count,
            "carrier_full": carrier.candidate_count,
            "carrier_small": carrier_small.candidate_count,
            "birth": birth.candidate_count,
        },
        "tangent_selection_damping_coordinate_system": (
            tangent_raw.damping_coordinate_system
        ),
        "heldout_refit": False if not oracle else None,
        "oracle_independently_selected": oracle,
        "carrier_bank_sensitivity_is_separate": True,
        "price_scope": PRICE_SCOPE,
    }

    score_records: list[dict[str, Any]] = []
    for rcond in manifest_module.RCONDS:
        factor = factors[float(rcond)]
        fit_residual = residual[fit_mask]
        for ledger, action in (
            *(("causal", value) for value in manifest_module.CAUSAL_ACTIONS),
            *(("matched", value) for value in manifest_module.MATCHED_ACTIONS),
        ):
            selected = identities.get(action)
            statistic: str
            if action == "full_j":
                energy = float((factor.left_basis.T @ fit_residual).square().sum())
                statistic = "unregularized_current_tangent_projector_energy"
                incremental = energy
                joint = energy
                rank = factor.rank
            elif action == "tangent6":
                design = tangent.full_design[fit_mask]
                scales = torch.linalg.vector_norm(design, dim=0).clamp_min(1e-12)
                left, singular, _vh = torch.linalg.svd(
                    design / scales, full_matrices=False
                )
                keep = singular > float(rcond) * singular[0]
                rank = int(keep.sum())
                joint = float((left[:, keep].T @ fit_residual).square().sum())
                energy, incremental = joint, joint
                statistic = "unregularized_six_direction_projector_energy"
            elif selected is not None and "birth" not in action:
                projector = literal_projector_energy(
                    factor,
                    fit_residual,
                    selected.full_design[fit_mask],
                    rcond=float(rcond),
                )
                energy = projector.tangent_projector_energy
                incremental = projector.incremental_projector_energy
                joint = projector.joint_projector_energy
                rank = projector.joint_rank
                statistic = "unregularized_projector_energy"
            else:
                if selected is None:
                    raise RuntimeError("birth score lacks selected packet")
                adjusted = fit_residual - selected.offset[fit_mask]
                coefficients, rank, _condition = _solve_fixed_six(
                    selected.full_design[fit_mask],
                    adjusted,
                    damping=0.0,
                    rcond=float(rcond),
                )
                remaining = adjusted - selected.full_design[fit_mask] @ coefficients
                energy = 0.0
                joint = float(fit_residual.square().sum() - remaining.square().sum())
                incremental = joint
                statistic = "finite_birth_affine_action_gain_not_projector"
            payload = {
                "binding_sha256": binding_sha256,
                "selection_scope": cells[0]["selection_scope"],
                "target_id": cells[0]["target_id"],
                "family": cells[0]["family"],
                "seed": cells[0]["seed"],
                "parent_horizon": cells[0]["parent_horizon"],
                "heldout_fold": fold,
                "ledger": ledger,
                "action": action,
                "rcond": float(rcond),
                "statistic": statistic,
                "tangent_projector_energy": energy,
                "incremental_energy": incremental,
                "joint_or_action_energy": joint,
                "rank": rank,
            }
            payload["score_record_id"] = _stable_id(payload)
            score_records.append(payload)
    return ScopeExecution(
        action_records,
        sensitivity_records,
        selection_record,
        tuple(score_records),
    )


def _evidence_projection(record: dict[str, Any], scoring_mask: torch.Tensor) -> dict[str, Any]:
    from benchmarks import residual_tangent_auction_stage1_analysis as analysis

    component = (
        "crossfit_immediate"
        if record["selection_scope"] == "crossfit"
        else "all_pixel_oracle"
    )
    projection = {
        "schema": analysis.SCHEMA_ID,
        "record_kind": analysis.RECORD_KIND,
        "component": component,
        "selection_scope": record["selection_scope"],
        "ledger": analysis.MATCHED_LEDGER,
        "family": record["family"],
        "seed": int(record["seed"]),
        "parent_checkpoint": int(record["parent_iters"]),
        "fold": -1 if record["heldout_fold"] is None else int(record["heldout_fold"]),
        "action": record["action"],
        "damping": float(record["damping"]),
        "rcond": float(record["rcond"]),
        "trust_radius": float(record["trust_radius"]),
        "checkpoint": 0,
        "status": record["status"],
        "error": record.get("error"),
        "binding_sha256": record["binding_sha256"],
    }
    projection["cell_id"] = analysis.stable_cell_id(projection)
    if record["status"] == "ok":
        projection.update(
            {
                "scoring_mask_sha256": tensor_sha256(scoring_mask),
                "parent_mse": record["base_mse"],
                "predicted_gain": record["predicted_mse_gain"],
                "realized_gain": record["realized_mse_gain"],
                "residual_energy_reduction": (
                    record["predicted_mse_gain"] * int(scoring_mask.sum())
                ),
                "psnr_db": record["realized_psnr"],
                "metadata": {
                    "rich_cell_id": record["cell_id"],
                    "identity_sha256": record["identity_sha256"],
                    "support": record["support"],
                },
            }
        )
    return projection


def _science_binding(runner_dir: Path) -> dict[str, Any]:
    runner_path = runner_dir / "runner_config.json"
    runner = _strict_load_json(runner_path)
    root = Path(__file__).resolve().parents[1]
    runner_binding = runner.get("binding")
    if not isinstance(runner_binding, dict):
        raise RuntimeError("runner_config lacks its source binding")
    recorded_sources = runner_binding.get("source_sha256")
    if not isinstance(recorded_sources, dict):
        raise RuntimeError("runner_config lacks source hashes")
    stale = [
        relative
        for relative, recorded in recorded_sources.items()
        if not (root / relative).is_file()
        or _sha256_file(root / relative) != recorded
    ]
    if stale:
        raise RuntimeError(f"parent runner binding is stale: {stale}")
    return {
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "runner_config_sha256": _sha256_file(runner_path),
        "runner_binding_sha256": runner.get("binding_sha256"),
        "source_sha256": _live_source_hashes(),
        "work_specification": WORK_SPECIFICATION,
        "work_specification_sha256": _stable_id(WORK_SPECIFICATION),
        "expected": {
            "crossfit_immediate": EXPECTED_CROSSFIT_CELLS,
            "all_pixel_oracle": EXPECTED_ORACLE_CELLS,
            "diagnostic_score_rows": EXPECTED_SCORE_ROWS,
            "small_carrier_bank_sensitivity_cells": (
                EXPECTED_SMALL_CARRIER_SENSITIVITY_CELLS
            ),
            "recovery_trajectories": 816,
        },
        "implemented_components": [
            "toy_preflight",
            "immediate",
            "diagnostic_scores",
            "small_carrier_bank_sensitivity_cells",
            "scientific_recovery_816_separate_phase",
        ],
        "unimplemented_components": [],
        "recovery_results_present_at_science_config_creation": False,
        "recovery_execution_module_source_bound": True,
        "small_carrier_identity_diagnostic_only": False,
        "small_carrier_bank_sensitivity_component_valid": True,
        "scientific_interpretation_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }


def _initialize_output(outdir: Path, runner_dir: Path) -> dict[str, Any]:
    binding = _science_binding(runner_dir)
    record = {"binding": binding, "binding_sha256": _stable_id(binding)}
    path = outdir / "science_config.json"
    if path.exists():
        if _strict_load_json(path) != record:
            raise RuntimeError("scientific binding changed; use a fresh output directory")
    else:
        _write_json(path, record)
    return record


def _manifest_core_cells() -> tuple[dict[str, Any], ...]:
    manifest = manifest_module.build_manifest()
    cells = tuple(
        cell
        for cell in manifest["action_cells"]
        if cell["cell_kind"] == "core_immediate"
    )
    crossfit = sum(cell["selection_scope"] == "crossfit" for cell in cells)
    oracle = len(cells) - crossfit
    if (crossfit, oracle) != (EXPECTED_CROSSFIT_CELLS, EXPECTED_ORACLE_CELLS):
        raise RuntimeError("frozen immediate manifest counts drifted")
    return cells


def _manifest_sensitivity_cells() -> tuple[dict[str, Any], ...]:
    cells = tuple(
        cell
        for cell in manifest_module.build_manifest()["action_cells"]
        if cell["cell_kind"] == "carrier_bank_sensitivity"
        and cell["carrier_bank"] == "small"
    )
    if len(cells) != EXPECTED_SMALL_CARRIER_SENSITIVITY_CELLS:
        raise RuntimeError("small-carrier sensitivity manifest count drifted")
    return cells


def _load_parents(runner_dir: Path, runner_binding_sha256: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    records = _load_jsonl(runner_dir / "parents.jsonl", "parent_id")
    if len(records) != 24:
        raise RuntimeError(f"immediate phase requires exactly 24 parents, found {len(records)}")
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records.values():
        if record.get("binding_sha256") != runner_binding_sha256:
            raise RuntimeError("parent record has a stale runner binding")
        if record.get("independent_horizon_run") is True:
            raise RuntimeError("independent horizon parent is forbidden by BENCH-009")
        path = Path(str(record["field_path"]))
        if not path.is_file() or _sha256_file(path) != record["field_file_sha256"]:
            raise RuntimeError("parent field file is missing or changed")
        key = (
            record["target_id"],
            int(record["seed"]),
            str(record["horizon_name"]),
        )
        if key in result:
            raise RuntimeError(f"duplicate parent axes {key}")
        result[key] = record
    return result


def _unit_groups(cells: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        key = {
            "target_id": cell["target_id"],
            "seed": cell["seed"],
            "parent_horizon": cell["parent_horizon"],
            "selection_scope": cell["selection_scope"],
            "heldout_fold": cell["heldout_fold"],
        }
        unit_id = _stable_id(key)
        groups.setdefault(unit_id, []).append(cell)
    if len(groups) != 72 or any(len(value) != CELLS_PER_SCOPE_UNIT for value in groups.values()):
        raise RuntimeError("immediate scope-unit partition drifted")
    return groups


def _unchecked_unit_groups(
    cells: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        key = {
            "target_id": cell["target_id"],
            "seed": cell["seed"],
            "parent_horizon": cell["parent_horizon"],
            "selection_scope": cell["selection_scope"],
            "heldout_fold": cell["heldout_fold"],
        }
        groups.setdefault(_stable_id(key), []).append(cell)
    return groups


def _error_record(cell: dict[str, Any], binding_sha256: str, error: Exception) -> dict[str, Any]:
    return {
        **cell,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "binding_sha256": binding_sha256,
        "scientific_interpretation_present": False,
        "recovery_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }


def run_immediate(
    *,
    outdir: Path,
    runner_dir: Path,
    binding: dict[str, Any],
    shard_index: int,
    shard_count: int,
    max_units: int | None,
) -> dict[str, Any]:
    binding_sha256 = str(binding["binding_sha256"])
    parent_binding = str(binding["binding"]["runner_binding_sha256"])
    parents = _load_parents(runner_dir, parent_binding)
    targets = manifest_module.target_suite()
    targets_by_id = {value["target_id"]: value for value in targets.values()}
    groups = _unit_groups(_manifest_core_cells())
    sensitivity_groups = _unchecked_unit_groups(_manifest_sensitivity_cells())
    if set(sensitivity_groups) != set(groups) or any(
        len(values) != 4 for values in sensitivity_groups.values()
    ):
        raise RuntimeError("small-carrier sensitivity unit partition drifted")
    chosen = [
        (unit_id, cells)
        for unit_id, cells in sorted(groups.items())
        if int(unit_id, 16) % shard_count == shard_index
    ]
    action_path = outdir / "immediate_cells.jsonl"
    existing = _load_jsonl(action_path, "cell_id")
    sensitivity_path = outdir / "small_carrier_sensitivity.jsonl"
    existing_sensitivity = _load_jsonl(sensitivity_path, "cell_id")
    existing_selections = _load_jsonl(outdir / "selections.jsonl", "selection_record_id")
    existing_scores = _load_jsonl(outdir / "diagnostic_scores.jsonl", "score_record_id")
    existing_evidence = _load_jsonl(outdir / "matched_evidence.jsonl", "cell_id")
    completed_units = _load_jsonl(outdir / "completed_units.jsonl", "unit_id")
    for label, records in (
        ("immediate", existing),
        ("sensitivity", existing_sensitivity),
        ("selection", existing_selections),
        ("score", existing_scores),
        ("evidence", existing_evidence),
        ("completed-unit", completed_units),
    ):
        stale_ids = [
            identifier
            for identifier, record in records.items()
            if record.get("binding_sha256") != binding_sha256
        ]
        if stale_ids:
            raise RuntimeError(f"{label} rows contain stale bindings: {stale_ids[:3]}")
    pending = [
        (unit_id, cells)
        for unit_id, cells in chosen
        if unit_id not in completed_units
    ]
    if max_units is not None:
        pending = pending[:max_units]
    completed = 0
    for unit_id, cells in pending:
        first = cells[0]
        key = (first["target_id"], int(first["seed"]), first["parent_horizon"])
        parent_record = parents[key]
        field = GaussianField.load(str(parent_record["field_path"]), device="cpu")
        if field_sha256(field) != parent_record["field_sha256"]:
            raise RuntimeError("parent tensor hash changed after load")
        target_record = targets_by_id[first["target_id"]]
        target = torch.as_tensor(target_record["image"], dtype=field.means.dtype)
        fold = first["heldout_fold"]
        try:
            execution = _scope_execution(
                field=field,
                target=target,
                cells=cells,
                sensitivity_cells=sensitivity_groups[unit_id],
                binding_sha256=binding_sha256,
                fold=None if fold is None else int(fold),
                seed=int(unit_id[:16], 16),
            )
            for record in execution.action_records:
                rich_id = str(record["cell_id"])
                if rich_id not in existing:
                    _append_jsonl(action_path, record)
                    existing[rich_id] = record
                if record["ledger"] == "matched":
                    scoring = torch.ones(target.numel(), dtype=torch.bool)
                    if fold is not None:
                        _discovery, scoring = checkerboard_fold_masks(
                            target.shape[0], target.shape[1], int(fold)
                        )
                    projection = _evidence_projection(record, scoring)
                    evidence_id = str(projection["cell_id"])
                    if evidence_id not in existing_evidence:
                        _append_jsonl(outdir / "matched_evidence.jsonl", projection)
                        existing_evidence[evidence_id] = projection
            for record in execution.sensitivity_records:
                sensitivity_id = str(record["cell_id"])
                if sensitivity_id not in existing_sensitivity:
                    _append_jsonl(sensitivity_path, record)
                    existing_sensitivity[sensitivity_id] = record
            selection_id = str(execution.selection_record["selection_record_id"])
            if selection_id not in existing_selections:
                _append_jsonl(outdir / "selections.jsonl", execution.selection_record)
                existing_selections[selection_id] = execution.selection_record
            for record in execution.score_records:
                score_id = str(record["score_record_id"])
                if score_id not in existing_scores:
                    _append_jsonl(outdir / "diagnostic_scores.jsonl", record)
                    existing_scores[score_id] = record
            marker = {
                "unit_id": unit_id,
                "binding_sha256": binding_sha256,
                "core_cell_count": len(execution.action_records),
                "sensitivity_cell_count": len(execution.sensitivity_records),
                "score_record_count": len(execution.score_records),
                "status": "ok",
            }
            _append_jsonl(outdir / "completed_units.jsonl", marker)
            completed_units[unit_id] = marker
        except Exception as error:  # terminal fail-closed rows are part of the protocol
            for cell in cells:
                if str(cell["cell_id"]) not in existing:
                    _append_jsonl(action_path, _error_record(cell, binding_sha256, error))
        completed += 1
        _write_json(
            outdir / "run_state.json",
            {
                "last_completed_unit_id": unit_id,
                "completed_units_this_invocation": completed,
                "recovery_present": False,
            },
        )
    terminal = _load_jsonl(action_path, "cell_id")
    stale = [
        key for key, value in terminal.items() if value.get("binding_sha256") != binding_sha256
    ]
    if stale:
        raise RuntimeError("terminal immediate rows contain stale bindings")
    expected_ids = {str(cell["cell_id"]) for cell in _manifest_core_cells()}
    expected_sensitivity_ids = {
        str(cell["cell_id"]) for cell in _manifest_sensitivity_cells()
    }
    complete = set(terminal) == expected_ids
    errors = sum(value.get("status") == "error" for value in terminal.values())
    return {
        "phase": "immediate",
        "completed_units_this_invocation": completed,
        "terminal_cells": len(terminal),
        "expected_terminal_cells": len(expected_ids),
        "component_complete": complete,
        "component_valid": complete and errors == 0,
        "terminal_error_cells": errors,
        "small_carrier_sensitivity_cells": len(existing_sensitivity),
        "small_carrier_sensitivity_expected": len(expected_sensitivity_ids),
        "small_carrier_sensitivity_component_valid": (
            set(existing_sensitivity) == expected_sensitivity_ids
            and all(
                value.get("status") == "ok"
                for value in existing_sensitivity.values()
            )
        ),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "recovery_present": False,
        "scientific_interpretation_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }


def _toy_field() -> GaussianField:
    return GaussianField.from_numpy(
        means=np.asarray(
            [[5.49, 5.2], [17.1, 6.8], [7.6, 17.2], [17.0, 17.1]],
            dtype=np.float64,
        ),
        scales=np.asarray(
            [[3.1, 2.0], [2.4, 3.0], [2.8, 2.2], [2.6, 2.9]],
            dtype=np.float64,
        ),
        angles=np.asarray([0.21, -0.47, 0.72, -0.16], dtype=np.float64),
        colors=np.asarray(
            [
                [0.22, 0.51, 0.74],
                [0.78, 0.31, 0.42],
                [0.29, 0.72, 0.36],
                [0.68, 0.61, 0.25],
            ],
            dtype=np.float64,
        ),
        dtype=torch.float64,
    )


def _toy_cells(fold: int | None) -> tuple[dict[str, Any], ...]:
    scope = "crossfit" if fold is not None else "all_pixel_identity_and_coefficient_oracle"
    values: list[dict[str, Any]] = []
    for ledger, actions in manifest_module.ACTION_LEDGERS.items():
        for action in actions:
            payload = {
                "target_id": "synthetic-toy-not-development",
                "target_pixel_sha256": "0" * 64,
                "family": "synthetic_toy",
                "seed": 7,
                "parent_horizon": "toy",
                "parent_iters": 0,
                "parent_count": 4,
                "cell_kind": "toy_immediate",
                "selection_scope": scope,
                "heldout_fold": fold,
                "ledger": ledger,
                "action": action,
                "carrier_bank": "full" if "carrier" in action else None,
                "damping": manifest_module.PRIMARY_DAMPING,
                "rcond": manifest_module.PRIMARY_RCOND,
                "trust_radius": manifest_module.TRUST_RADII[0],
                "trust_norm": "linf",
            }
            payload["cell_id"] = _stable_id(payload)
            values.append(payload)
    return tuple(values)


def _toy_recovery_step(
    field: GaussianField,
    target: torch.Tensor,
    discovery: torch.Tensor,
) -> dict[str, Any]:
    means = torch.nn.Parameter(field.means.detach().clone())
    log_scales = torch.nn.Parameter(field.log_scales.detach().clone())
    rotations = torch.nn.Parameter(field.rotations.detach().clone())
    colors = torch.nn.Parameter(field.colors.detach().clone())
    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 5e-2},
            {"params": [log_scales], "lr": 3e-2},
            {"params": [rotations], "lr": 1e-2},
            {"params": [colors], "lr": 3e-2},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    def current() -> GaussianField:
        return GaussianField(means, log_scales, rotations, colors)

    before = render_normalized(current(), target.shape[0], target.shape[1])
    optimizer.zero_grad(set_to_none=True)
    loss = (target - before).reshape(-1)[discovery].square().mean()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        log_scales.clamp_(math.log(0.35), math.log(64.0))
    after_field = current()
    after = render_normalized(after_field, target.shape[0], target.shape[1])
    return {
        "optimizer": "fresh_adam",
        "steps": 1,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "discovery_mse_before": _mask_mse(before, target, discovery),
        "discovery_mse_after": _mask_mse(after, target, discovery),
        "state_sha256": field_sha256(after_field),
        "native_render_sha256": tensor_sha256(after),
        "support": _support_record(
            freeze_pixel_support(field, target.shape[0], target.shape[1]),
            after_field,
        ),
        "native_dynamic_support": True,
    }


def run_toy_preflight() -> dict[str, Any]:
    """Exercise the full path without importing a frozen development target."""

    field = _toy_field()
    height = width = 24
    parameterization = make_no_opacity_parameterization(field, height, width)
    base = render_normalized(field, height, width)
    affine = affine_six_columns(
        field,
        1,
        height,
        width,
        support=parameterization.frozen_support,
    )
    carrier = carrier_six_columns(
        field,
        2,
        CARRIER_BANKS["small"][3].frequency_xy,
        height,
        width,
        support=parameterization.frozen_support,
    )
    coefficients = torch.tensor(
        [0.07, -0.05, 0.03, -0.04, 0.06, 0.02], dtype=torch.float64
    )
    target = base + (affine @ coefficients).reshape_as(base)
    target = target + 0.35 * (carrier @ coefficients.flip(0)).reshape_as(base)
    toy_binding = {
        "protocol": PROTOCOL,
        "scope": TOY_SCOPE,
        "source_sha256": _live_source_hashes(),
        "work_specification_sha256": _stable_id(WORK_SPECIFICATION),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
    }
    binding_sha256 = _stable_id(toy_binding)
    executions = [
        _scope_execution(
            field=field,
            target=target,
            cells=_toy_cells(fold),
            binding_sha256=binding_sha256,
            fold=fold,
            seed=700 + (2 if fold is None else fold),
        )
        for fold in (0, 1, None)
    ]
    jvp = centered_fd_jvp_validation(parameterization, seed=19, epsilon=1e-4)

    geometries = (
        BirthGeometry((4.0, 19.0), (2.1, 1.7), 0.2),
        BirthGeometry((19.0, 12.0), (1.8, 2.3), -0.3),
    )
    packet = build_two_birth_packet(field, geometries, height, width)
    birth_coefficients = torch.tensor(
        [0.4, 0.2, 0.3, 0.1, 0.5, 0.25], dtype=torch.float64
    )
    linear_birth = packet_linear_realization(
        packet.base_render,
        packet.design,
        birth_coefficients,
        offset=packet.offset,
    )
    direct_birth = realize_two_birth_six(field, packet, birth_coefficients)
    birth_parity = float((linear_birth - direct_birth).abs().max())

    discovery, heldout = checkerboard_fold_masks(height, width, 0)
    discovery = discovery.to(dtype=torch.bool)
    heldout = heldout.to(dtype=torch.bool)
    flat_render = parameterization.flat_render_function()
    discovery_op = build_masked_jacobian_operators(
        flat_render, parameterization.physical_vector, discovery
    )
    heldout_op = build_masked_jacobian_operators(
        flat_render, parameterization.physical_vector, heldout
    )
    residual = (target - base).reshape(-1)
    original = select_tangent_six_crossfit(
        discovery_op,
        heldout_op,
        residual[discovery],
        residual[heldout],
        coordinate_ids=_coordinate_ids(field),
        rcond=manifest_module.PRIMARY_RCOND,
        damping=manifest_module.PRIMARY_DAMPING,
    )
    perturbed_heldout = residual[heldout] + 0.37 * torch.sin(
        torch.arange(int(heldout.sum()), dtype=residual.dtype)
    )
    repeated = select_tangent_six_crossfit(
        discovery_op,
        heldout_op,
        residual[discovery],
        perturbed_heldout,
        coordinate_ids=_coordinate_ids(field),
        rcond=manifest_module.PRIMARY_RCOND,
        damping=manifest_module.PRIMARY_DAMPING,
    )
    leak_safe = (
        original.coordinate_indices == repeated.coordinate_indices
        and torch.equal(original.coefficients, repeated.coefficients)
    )

    crossing_step = torch.zeros_like(parameterization.physical_vector)
    crossing_step[0] = 0.03
    support_crossing = _support_record(
        parameterization.frozen_support,
        parameterization.field_at(crossing_step),
    )
    recovery = _toy_recovery_step(field, target, discovery)
    all_records = [record for execution in executions for record in execution.action_records]
    checks = {
        "two_checkerboard_folds_exercised": {
            record["heldout_fold"] for record in all_records if record["heldout_fold"] is not None
        }
        == {0, 1},
        "all_pixel_oracle_independently_selected": executions[2].selection_record[
            "oracle_independently_selected"
        ]
        is True,
        "full_j_and_tangent6_exercised": {record["action"] for record in all_records}
        >= {"full_j", "tangent6"},
        "affine_carrier_birth_exercised": {record["action"] for record in all_records}
        >= {"affine6", "carrier6", "birth2_rgb6"},
        "causal_joint_actions_exercised": {record["action"] for record in all_records}
        >= {"j_plus_affine6", "j_plus_carrier6", "j_plus_birth2_rgb6"},
        "uniform_concatenated_linf_trust": all(
            record["trust_scale"] <= 1.0 for record in all_records
        ),
        "native_support_hashes_and_counts": support_crossing["xor_memberships"] > 0,
        "heldout_no_refit": leak_safe,
        "centered_fd_jvp": jvp["relative_l2"] <= 1e-3,
        "exact_birth_direct_parity": birth_parity <= 1e-10,
        "one_step_fresh_adam_recovery": recovery["steps"] == 1,
        "small_and_full_carrier_banks_selected": all(
            "carrier_small_sensitivity" in execution.selection_record["identities"]
            for execution in executions
        ),
    }
    return {
        "protocol": PROTOCOL,
        "scope": TOY_SCOPE,
        "binding_sha256": binding_sha256,
        "binding": toy_binding,
        "target_source": "synthetic_local_fixture_not_development_manifest",
        "checks": checks,
        "preflight_pass": all(checks.values()),
        "jvp": jvp,
        "birth_direct_parity_max_abs": birth_parity,
        "support_crossing": support_crossing,
        "recovery_one_step": recovery,
        "scope_action_record_count": len(all_records),
        "scope_selection_records": [item.selection_record for item in executions],
        "development_targets_accessed": False,
        "scientific_cells_present": False,
        "recovery_component_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }


def validate_toy_preflight(path: str | Path) -> dict[str, Any]:
    """Strictly validate a persisted toy artifact and its live source binding."""

    artifact = _strict_load_json(Path(path))
    checks = artifact.get("checks")
    if not isinstance(checks, dict) or len(checks) != 12 or not all(
        value is True for value in checks.values()
    ):
        raise RuntimeError("toy preflight does not contain twelve passing checks")
    required = {
        "preflight_pass": True,
        "development_targets_accessed": False,
        "scientific_cells_present": False,
        "recovery_component_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }
    drift = [key for key, value in required.items() if artifact.get(key) is not value]
    if drift:
        raise RuntimeError(f"toy preflight safety flags drifted: {drift}")
    binding = artifact.get("binding")
    if not isinstance(binding, dict):
        raise RuntimeError("toy preflight lacks its source binding")
    if artifact.get("binding_sha256") != _stable_id(binding):
        raise RuntimeError("toy preflight binding hash mismatch")
    live = _live_source_hashes()
    if binding.get("source_sha256") != live:
        raise RuntimeError("toy preflight source hashes are stale")
    if binding.get("work_specification_sha256") != _stable_id(WORK_SPECIFICATION):
        raise RuntimeError("toy preflight work specification is stale")
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("toy-preflight", "immediate"), required=True)
    parser.add_argument(
        "--outdir", default="results/bench009_tangent_auction_stage1_science"
    )
    parser.add_argument("--runner-dir", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-units", type=int, default=None)
    args = parser.parse_args(argv)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard axes require 0 <= shard-index < positive shard-count")
    if args.max_units is not None and args.max_units <= 0:
        parser.error("--max-units must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    outdir = Path(args.outdir)
    if args.phase == "toy-preflight":
        result = run_toy_preflight()
        _write_json(outdir / "toy_preflight.json", result)
    else:
        if args.runner_dir is None:
            parser.error("immediate phase requires --runner-dir")
        runner_dir = Path(args.runner_dir)
        binding = _initialize_output(outdir, runner_dir)
        _append_jsonl(
            outdir / "invocations.jsonl",
            {
                "invocation_id": _stable_id(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "argv": argv or sys.argv[1:],
                    }
                ),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join([sys.executable, "-m", __name__, *(argv or sys.argv[1:])]),
                "binding_sha256": binding["binding_sha256"],
            },
        )
        result = run_immediate(
            outdir=outdir,
            runner_dir=runner_dir,
            binding=binding,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_units=args.max_units,
        )
        _write_json(outdir / "run_state.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("preflight_pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
