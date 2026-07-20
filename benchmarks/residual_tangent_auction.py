"""Stage-0 residual tangent-space auction for StructSplat.

This benchmark asks a diagnostic question before implementing another fitter or primitive:
does a frozen reconstruction residual lie in the render-identifiable tangent space of the
current field, or does a small appearance/structure packet add a genuinely new image-space
direction?

The mechanisms are not method-novel. Matrix-free JVP/Gauss--Newton and LM methods (including
Pehlivan et al.), Golub--Pereyra variable projection, and OMP/matching pursuit are direct prior
art for the numerical operations. WIPES, SVGS, Textured Gaussians, and related work are direct
prior-art threats for frequency-bearing or spatially varying Gaussian appearance. The only claim
this file may support is that a controlled *evidence program* can distinguish optimization-space
from representation-space residual on this repository's normalized renderer.

All prices emitted here are diagnostic degrees of freedom and packet fields. They are not bytes,
entropy, BPP, or a compression result. In particular, the affine and carrier packets have no
SSPL1 syntax. A positive Stage-0 result can authorize a complete-stream experiment; it cannot be
reported as rate--distortion evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable

import numpy as np
import torch

from structsplat.gaussians import GaussianField
from structsplat.render import render_field


DIAGNOSTIC_PRICE_SCOPE = "dof_packet_only_not_bytes_or_compression"
FAMILIES = ("affine", "carrier", "finite_birth")
TARGET_KINDS = ("null", "current_tangent", "affine", "carrier", "finite_birth")


@dataclass(frozen=True)
class SVDSystem:
    """Scaled, truncated SVD of an image-space design matrix.

    Columns are normalized before factorization. This makes the numerical rank invariant to the
    units used for position, log-scale, angle, color, and opacity coordinates. The retained left
    singular vectors define the render-identifiable quotient of a potentially gauge-singular
    parameterization.
    """

    design: torch.Tensor
    scales: torch.Tensor
    u: torch.Tensor
    singular_values: torch.Tensor
    vh: torch.Tensor
    rank: int
    raw_columns: int
    rcond: float


@dataclass
class Packet:
    name: str
    columns: torch.Tensor
    offset: torch.Tensor
    optimized_dofs: int
    packet_dofs: int
    fixed_dofs: int
    selection_candidates: int
    description: str
    realize: Callable[[torch.Tensor], torch.Tensor]


@dataclass
class AuctionReference:
    field: GaussianField
    height: int
    width: int
    sigma_cutoff: float
    parameter_vector: torch.Tensor
    parameter_scales: torch.Tensor
    frozen_radii: torch.Tensor
    base_render: torch.Tensor
    tangent: torch.Tensor
    base_system: SVDSystem
    selected_gid: int
    packets: dict[str, Packet]
    render_normalized_step: Callable[[torch.Tensor], torch.Tensor]


def _sha256_tensor(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _safe_psnr(mse: float) -> float | None:
    if mse <= 0.0:
        # JSON has no representation for infinity. Exact reconstruction is carried by MSE=0;
        # null PSNR keeps every emitted artifact RFC-8259 compliant.
        return None
    return -10.0 * math.log10(mse)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_worktree_entry(path: Path) -> dict[str, Any]:
    """Hash raw worktree state without consulting the Git index."""
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
        return {
            "kind": "symlink_target",
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
    payload = path.read_bytes()
    return {
        "kind": "file_bytes",
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _collect_provenance() -> dict[str, Any]:
    """Bind results to exact sources plus tracked and untracked worktree state."""
    repo_root = Path(__file__).resolve().parents[1]
    critical_paths = (
        "benchmarks/residual_tangent_auction.py",
        "src/structsplat/gaussians.py",
        "src/structsplat/render.py",
    )
    critical_sources = {
        relative: _hash_worktree_entry(repo_root / relative)
        for relative in critical_paths
    }

    commit = _git_bytes(repo_root, "rev-parse", "HEAD").decode("ascii").strip()
    branch = _git_bytes(repo_root, "branch", "--show-current").decode("utf-8").strip()
    status = _git_bytes(
        repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    tracked_diff = _git_bytes(repo_root, "diff", "--binary", "HEAD", "--", ".")
    untracked_raw = _git_bytes(
        repo_root, "ls-files", "--others", "--exclude-standard", "-z"
    )
    untracked_paths = sorted(
        path.decode("utf-8", errors="surrogateescape")
        for path in untracked_raw.split(b"\0")
        if path
    )
    untracked_files = {
        relative: _hash_worktree_entry(repo_root / relative)
        for relative in untracked_paths
    }
    manifest_payload = json.dumps(
        untracked_files, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "git_commit": commit,
        "git_branch": branch or None,
        "git_dirty": bool(status),
        "git_status_porcelain_sha256": _sha256_bytes(status),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "tracked_diff_size_bytes": len(tracked_diff),
        "critical_source_hash_mode": "raw_worktree_bytes_independent_of_git_index",
        "critical_sources": critical_sources,
        "untracked_file_count": len(untracked_files),
        "untracked_manifest_sha256": _sha256_bytes(manifest_payload),
        "untracked_files": untracked_files,
    }


def _mse(residual: torch.Tensor) -> float:
    return float(residual.square().mean())


def _logit(probability: float, *, dtype: torch.dtype) -> torch.Tensor:
    value = torch.tensor(probability, dtype=dtype)
    return torch.log(value) - torch.log1p(-value)


def _reference_field(dtype: torch.dtype = torch.float64) -> GaussianField:
    """A small non-symmetric field whose image-space tangent has useful rank."""
    probabilities = np.asarray([0.82, 0.67, 0.74], dtype=np.float64)
    logits = np.log(probabilities / (1.0 - probabilities))
    return GaussianField.from_numpy(
        means=np.asarray([[3.17, 3.31], [8.23, 5.19], [5.41, 9.27]], dtype=np.float64),
        scales=np.asarray([[2.35, 1.45], [1.65, 2.55], [2.15, 1.75]], dtype=np.float64),
        angles=np.asarray([0.19, -0.51, 0.83], dtype=np.float64),
        colors=np.asarray(
            [[0.31, 0.54, 0.72], [0.76, 0.29, 0.43], [0.27, 0.71, 0.38]],
            dtype=np.float64,
        ),
        opacities=logits,
        device="cpu",
        dtype=dtype,
    )


def _pack_field(field: GaussianField) -> tuple[torch.Tensor, torch.Tensor]:
    if field.opacities is None:
        raise ValueError("Stage-0 reference requires explicit finite opacity logits")
    pieces = (
        field.means.reshape(-1),
        field.log_scales.reshape(-1),
        field.rotations.reshape(-1),
        field.colors.reshape(-1),
        field.opacities.reshape(-1),
    )
    vector = torch.cat(pieces).detach().clone()
    # A unit normalized coordinate is deliberately a small, locally valid physical update.
    scales = torch.cat(
        (
            torch.full_like(pieces[0], 0.08),       # pixels
            torch.full_like(pieces[1], 0.04),       # log sigma
            torch.full_like(pieces[2], 0.04),       # radians
            torch.full_like(pieces[3], 0.04),       # RGB
            torch.full_like(pieces[4], 0.05),       # opacity logit
        )
    )
    return vector, scales


def _unpack_field(vector: torch.Tensor, n: int) -> GaussianField:
    cursor = 0

    def take(count: int) -> torch.Tensor:
        nonlocal cursor
        value = vector[cursor:cursor + count]
        cursor += count
        return value

    means = take(2 * n).reshape(n, 2)
    log_scales = take(2 * n).reshape(n, 2)
    rotations = take(n)
    colors = take(3 * n).reshape(n, 3)
    opacities = take(n)
    if cursor != int(vector.numel()):
        raise ValueError("parameter vector has an unexpected length")
    return GaussianField(means, log_scales, rotations, colors, opacities)


def _render_field_reference(
    field: GaussianField,
    height: int,
    width: int,
    *,
    sigma_cutoff: float,
    frozen_radii: torch.Tensor | None = None,
    color_grads: torch.Tensor | None = None,
) -> torch.Tensor:
    radii = field.radii(sigma_cutoff) if frozen_radii is None else frozen_radii
    return render_field(
        field.means,
        field.conics(),
        field.colors,
        radii,
        height,
        width,
        chunk=1,
        mode="normalized",
        opacities=field.opacity_values(),
        scales=field.scales(),
        rotations=field.rotations,
        color_grads=color_grads,
        sigma_cutoff=sigma_cutoff,
    )


def _scaled_svd(
    design: torch.Tensor,
    *,
    rcond: float = 1e-7,
    scale_floor: float = 1e-12,
) -> SVDSystem:
    if design.ndim != 2:
        raise ValueError("design must be a matrix")
    if not math.isfinite(rcond) or rcond < 0.0:
        raise ValueError("rcond must be finite and non-negative")
    if design.shape[1] == 0:
        empty = design.new_zeros((design.shape[0], 0))
        return SVDSystem(
            design=design,
            scales=design.new_zeros((0,)),
            u=empty,
            singular_values=design.new_zeros((0,)),
            vh=design.new_zeros((0, 0)),
            rank=0,
            raw_columns=0,
            rcond=float(rcond),
        )
    if not bool(torch.isfinite(design).all()):
        raise ValueError("design contains non-finite values")
    scales = torch.linalg.vector_norm(design, dim=0).clamp_min(float(scale_floor))
    scaled = design / scales
    u, singular_values, vh = torch.linalg.svd(scaled, full_matrices=False)
    if singular_values.numel() == 0:
        rank = 0
    else:
        cutoff = float(rcond) * float(singular_values[0])
        rank = int((singular_values > cutoff).sum())
    return SVDSystem(
        design=design,
        scales=scales,
        u=u[:, :rank],
        singular_values=singular_values[:rank],
        vh=vh[:rank],
        rank=rank,
        raw_columns=int(design.shape[1]),
        rcond=float(rcond),
    )


def _solve_svd(
    system: SVDSystem,
    rhs: torch.Tensor,
    *,
    ridge: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return coefficients in original units and the fitted image-space vector."""
    if rhs.ndim != 1 or rhs.shape[0] != system.design.shape[0]:
        raise ValueError("rhs shape does not match design")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    if system.rank == 0:
        coefficients = rhs.new_zeros((system.raw_columns,))
        return coefficients, rhs.new_zeros(rhs.shape)
    projected = system.u.T @ rhs
    singular = system.singular_values
    if ridge == 0.0:
        inverse = singular.reciprocal()
    else:
        inverse = singular / (singular.square() + float(ridge) ** 2)
    scaled_coefficients = system.vh.T @ (inverse * projected)
    coefficients = scaled_coefficients / system.scales
    return coefficients, system.design @ coefficients


def _residualize(system: SVDSystem, value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] != system.design.shape[0]:
        raise ValueError("value shape does not match SVD system")
    if system.rank == 0:
        return value.clone()
    return value - system.u @ (system.u.T @ value)


def _jvp_matrix(
    function: Callable[[torch.Tensor], torch.Tensor],
    origin: torch.Tensor,
) -> torch.Tensor:
    """Materialize a small Stage-0 Jacobian using matrix-free JVP calls."""
    columns: list[torch.Tensor] = []
    for column in range(int(origin.numel())):
        direction = torch.zeros_like(origin)
        direction[column] = 1.0
        _, tangent = torch.autograd.functional.jvp(
            function,
            origin,
            direction,
            create_graph=False,
            strict=True,
        )
        columns.append(tangent.detach())
    return torch.stack(columns, dim=1)


def _limit_step(coefficients: torch.Tensor, limit: float) -> tuple[torch.Tensor, float]:
    if limit <= 0.0 or not math.isfinite(limit):
        raise ValueError("coefficient limit must be finite and positive")
    if coefficients.numel() == 0:
        return coefficients, 1.0
    largest = float(coefficients.abs().max())
    fraction = min(1.0, float(limit) / max(largest, 1e-30))
    return coefficients * fraction, fraction


def _affine_packet(
    field: GaussianField,
    base_render: torch.Tensor,
    height: int,
    width: int,
    *,
    gid: int,
    sigma_cutoff: float,
    frozen_radii: torch.Tensor,
) -> Packet:
    columns: list[torch.Tensor] = []
    for axis in range(2):
        for channel in range(3):
            gradients = torch.zeros(field.n, 2, 3, dtype=field.colors.dtype)
            gradients[gid, axis, channel] = 1.0
            rendered = _render_field_reference(
                field,
                height,
                width,
                sigma_cutoff=sigma_cutoff,
                frozen_radii=frozen_radii,
                color_grads=gradients,
            )
            columns.append((rendered - base_render).reshape(-1))
    design = torch.stack(columns, dim=1)

    def realize(coefficients: torch.Tensor) -> torch.Tensor:
        gradients = torch.zeros(field.n, 2, 3, dtype=field.colors.dtype)
        gradients[gid] = coefficients.reshape(2, 3)
        return _render_field_reference(
            field,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            frozen_radii=frozen_radii,
            color_grads=gradients,
        )

    return Packet(
        name="affine",
        columns=design,
        offset=torch.zeros_like(base_render).reshape(-1),
        optimized_dofs=6,
        packet_dofs=6,
        fixed_dofs=0,
        selection_candidates=field.n,
        description="Inherited-support local x/y RGB jet; parent selection is disclosed, unpriced.",
        realize=realize,
    )


def _responsibility_image(
    field: GaussianField,
    height: int,
    width: int,
    *,
    gid: int,
    sigma_cutoff: float,
    frozen_radii: torch.Tensor,
) -> torch.Tensor:
    colors = torch.zeros_like(field.colors)
    colors[gid] = 1.0
    probe = GaussianField(
        field.means,
        field.log_scales,
        field.rotations,
        colors,
        field.opacities,
    )
    return _render_field_reference(
        probe,
        height,
        width,
        sigma_cutoff=sigma_cutoff,
        frozen_radii=frozen_radii,
    )[..., 0]


def _carrier_packet(
    field: GaussianField,
    base_render: torch.Tensor,
    height: int,
    width: int,
    *,
    gid: int,
    frequency: tuple[float, float],
    sigma_cutoff: float,
    frozen_radii: torch.Tensor,
) -> Packet:
    responsibility = _responsibility_image(
        field,
        height,
        width,
        gid=gid,
        sigma_cutoff=sigma_cutoff,
        frozen_radii=frozen_radii,
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=field.means.dtype),
        torch.arange(width, dtype=field.means.dtype),
        indexing="ij",
    )
    dx = xx - field.means[gid, 0]
    dy = yy - field.means[gid, 1]
    angle = field.rotations[gid]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    scales = field.scales()[gid]
    local_x = (cosine * dx + sine * dy) / scales[0].clamp_min(1e-6)
    local_y = (-sine * dx + cosine * dy) / scales[1].clamp_min(1e-6)
    phase = 2.0 * math.pi * (float(frequency[0]) * local_x + float(frequency[1]) * local_y)
    carriers = (torch.sin(phase), torch.cos(phase))
    columns: list[torch.Tensor] = []
    for carrier in carriers:
        for channel in range(3):
            image = torch.zeros_like(base_render)
            image[..., channel] = responsibility * carrier
            columns.append(image.reshape(-1))
    design = torch.stack(columns, dim=1)

    def realize(coefficients: torch.Tensor) -> torch.Tensor:
        return base_render + (design @ coefficients).reshape_as(base_render)

    return Packet(
        name="carrier",
        columns=design,
        offset=torch.zeros_like(base_render).reshape(-1),
        optimized_dofs=6,
        packet_dofs=8,
        fixed_dofs=2,
        selection_candidates=field.n,
        description=(
            "Inherited Gaussian envelope with sin/cos RGB amplitudes and a fixed 2-D frequency; "
            "parent/frequency search bits are not priced."
        ),
        realize=realize,
    )


def _append_birth(
    field: GaussianField,
    *,
    mean: tuple[float, float],
    scale: tuple[float, float],
    angle: float,
    color: torch.Tensor,
    opacity_probability: float,
) -> GaussianField:
    if field.opacities is None:
        raise ValueError("finite birth requires explicit parent opacity logits")
    dtype = field.means.dtype
    return GaussianField(
        torch.cat((field.means, torch.tensor([mean], dtype=dtype)), dim=0),
        torch.cat((field.log_scales, torch.log(torch.tensor([scale], dtype=dtype))), dim=0),
        torch.cat((field.rotations, torch.tensor([angle], dtype=dtype)), dim=0),
        torch.cat((field.colors, color.reshape(1, 3).to(dtype=dtype)), dim=0),
        torch.cat((field.opacities, _logit(opacity_probability, dtype=dtype).reshape(1)), dim=0),
    )


def _finite_birth_packet(
    field: GaussianField,
    base_render: torch.Tensor,
    height: int,
    width: int,
    *,
    sigma_cutoff: float,
    mean: tuple[float, float],
    scale: tuple[float, float],
    angle: float,
    opacity_probability: float,
) -> Packet:
    zero = torch.zeros(3, dtype=field.colors.dtype)
    zero_field = _append_birth(
        field,
        mean=mean,
        scale=scale,
        angle=angle,
        color=zero,
        opacity_probability=opacity_probability,
    )
    zero_render = _render_field_reference(
        zero_field, height, width, sigma_cutoff=sigma_cutoff
    )
    offset = (zero_render - base_render).reshape(-1)
    columns: list[torch.Tensor] = []
    for channel in range(3):
        color = torch.zeros(3, dtype=field.colors.dtype)
        color[channel] = 1.0
        unit_field = _append_birth(
            field,
            mean=mean,
            scale=scale,
            angle=angle,
            color=color,
            opacity_probability=opacity_probability,
        )
        unit_render = _render_field_reference(
            unit_field, height, width, sigma_cutoff=sigma_cutoff
        )
        columns.append((unit_render - zero_render).reshape(-1))
    design = torch.stack(columns, dim=1)

    def realize(coefficients: torch.Tensor) -> torch.Tensor:
        birth = _append_birth(
            field,
            mean=mean,
            scale=scale,
            angle=angle,
            color=coefficients,
            opacity_probability=opacity_probability,
        )
        return _render_field_reference(birth, height, width, sigma_cutoff=sigma_cutoff)

    if float(offset.abs().max()) <= 1e-10:
        raise RuntimeError("finite birth denominator offset vanished; this is not a finite test")
    return Packet(
        name="finite_birth",
        columns=design,
        offset=offset,
        optimized_dofs=3,
        packet_dofs=9,
        fixed_dofs=6,
        selection_candidates=1,
        description=(
            "Exact finite-opacity normalized birth: geometry and opacity are frozen for the "
            "linear color solve but all 9 row scalars are charged diagnostically."
        ),
        realize=realize,
    )


def build_reference(
    *,
    height: int = 13,
    width: int = 13,
    sigma_cutoff: float = 3.0,
    rcond: float = 1e-5,
    dtype: torch.dtype = torch.float64,
) -> AuctionReference:
    if height < 8 or width < 8:
        raise ValueError("Stage-0 reference requires height and width >= 8")
    field = _reference_field(dtype)
    parameter_vector, parameter_scales = _pack_field(field)
    frozen_radii = field.radii(sigma_cutoff).detach()

    def render_normalized_step(step: torch.Tensor) -> torch.Tensor:
        if step.shape != parameter_vector.shape:
            raise ValueError("normalized parameter step has the wrong shape")
        updated = _unpack_field(parameter_vector + parameter_scales * step, field.n)
        return _render_field_reference(
            updated,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            frozen_radii=frozen_radii,
        )

    origin = torch.zeros_like(parameter_vector)
    base_render = render_normalized_step(origin).detach()
    tangent = _jvp_matrix(lambda value: render_normalized_step(value).reshape(-1), origin)
    base_system = _scaled_svd(tangent, rcond=rcond)
    selected_gid = 1
    packets = {
        "affine": _affine_packet(
            field,
            base_render,
            height,
            width,
            gid=selected_gid,
            sigma_cutoff=sigma_cutoff,
            frozen_radii=frozen_radii,
        ),
        "carrier": _carrier_packet(
            field,
            base_render,
            height,
            width,
            gid=selected_gid,
            frequency=(0.41, 0.17),
            sigma_cutoff=sigma_cutoff,
            frozen_radii=frozen_radii,
        ),
        "finite_birth": _finite_birth_packet(
            field,
            base_render,
            height,
            width,
            sigma_cutoff=sigma_cutoff,
            mean=(10.15, 9.85),
            scale=(1.25, 1.65),
            angle=-0.31,
            opacity_probability=0.58,
        ),
    }
    return AuctionReference(
        field=field,
        height=height,
        width=width,
        sigma_cutoff=sigma_cutoff,
        parameter_vector=parameter_vector,
        parameter_scales=parameter_scales,
        frozen_radii=frozen_radii,
        base_render=base_render,
        tangent=tangent,
        base_system=base_system,
        selected_gid=selected_gid,
        packets=packets,
        render_normalized_step=render_normalized_step,
    )


def finite_difference_jvp_error(
    reference: AuctionReference,
    *,
    seed: int = 0,
    epsilon: float = 1e-4,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(
        reference.parameter_vector.shape,
        generator=generator,
        dtype=reference.parameter_vector.dtype,
    )
    direction = direction / direction.norm().clamp_min(1e-12)
    positive = reference.render_normalized_step(epsilon * direction).reshape(-1)
    negative = reference.render_normalized_step(-epsilon * direction).reshape(-1)
    finite_difference = (positive - negative) / (2.0 * epsilon)
    jvp = reference.tangent @ direction
    absolute = float((finite_difference - jvp).abs().max())
    relative = float(
        torch.linalg.vector_norm(finite_difference - jvp)
        / torch.linalg.vector_norm(finite_difference).clamp_min(1e-12)
    )
    return {"max_abs": absolute, "relative_l2": relative}


def _fit_packet(
    residual: torch.Tensor,
    reference: AuctionReference,
    packet: Packet,
    *,
    ridge: float,
    rcond: float,
    base_trust_radius: float,
    packet_coefficient_limit: float,
) -> dict[str, Any]:
    shifted = residual - packet.offset
    residualized_rhs = _residualize(reference.base_system, shifted)
    residualized_columns = _residualize(reference.base_system, packet.columns)
    packet_system = _scaled_svd(residualized_columns, rcond=rcond)
    packet_coefficients, _ = _solve_svd(packet_system, residualized_rhs, ridge=ridge)
    packet_coefficients, packet_fraction = _limit_step(
        packet_coefficients, packet_coefficient_limit
    )

    base_rhs = shifted - packet.columns @ packet_coefficients
    base_coefficients, _ = _solve_svd(reference.base_system, base_rhs, ridge=ridge)
    base_coefficients, base_fraction = _limit_step(base_coefficients, base_trust_radius)
    predicted_delta = (
        packet.offset
        + packet.columns @ packet_coefficients
        + reference.tangent @ base_coefficients
    )
    remaining = residual - predicted_delta

    packet_render = packet.realize(packet_coefficients)
    packet_linear_render = (
        reference.base_render.reshape(-1)
        + packet.offset
        + packet.columns @ packet_coefficients
    ).reshape_as(reference.base_render)
    realization_max_abs = float((packet_render - packet_linear_render).abs().max())

    return {
        "name": packet.name,
        "predicted_mse": _mse(remaining),
        "predicted_psnr": _safe_psnr(_mse(remaining)),
        "packet_coefficients": [float(v) for v in packet_coefficients],
        "base_coefficients": [float(v) for v in base_coefficients],
        "packet_step_fraction": packet_fraction,
        "base_step_fraction": base_fraction,
        "residualized_rank": packet_system.rank,
        "residualized_raw_columns": packet_system.raw_columns,
        "residualized_column_norm": float(torch.linalg.matrix_norm(residualized_columns)),
        "residualized_orthogonality_max_abs": float(
            (reference.base_system.u.T @ residualized_columns).abs().max()
            if reference.base_system.rank and residualized_columns.numel()
            else 0.0
        ),
        "packet_realization_max_abs": realization_max_abs,
        "optimized_dofs": packet.optimized_dofs,
        "packet_dofs": packet.packet_dofs,
        "fixed_dofs": packet.fixed_dofs,
        "selection_candidates": packet.selection_candidates,
        "price_scope": DIAGNOSTIC_PRICE_SCOPE,
        "description": packet.description,
    }


def auction_target(
    reference: AuctionReference,
    target: torch.Tensor,
    *,
    ridge: float = 1e-3,
    rcond: float = 1e-5,
    base_trust_radius: float = 0.75,
    packet_coefficient_limit: float = 1.0,
) -> dict[str, Any]:
    if target.shape != reference.base_render.shape:
        raise ValueError("target shape does not match reference render")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains non-finite values")
    residual = (target - reference.base_render).reshape(-1)
    initial_mse = _mse(residual)
    base_coefficients, _ = _solve_svd(reference.base_system, residual, ridge=ridge)
    base_coefficients, base_fraction = _limit_step(base_coefficients, base_trust_radius)
    base_predicted_delta = reference.tangent @ base_coefficients
    base_remaining = residual - base_predicted_delta
    base_predicted_mse = _mse(base_remaining)

    base_realized = reference.render_normalized_step(base_coefficients)
    base_realized_delta = (base_realized - reference.base_render).reshape(-1)
    base_linearization_max_abs = float((base_realized_delta - base_predicted_delta).abs().max())
    base_realized_mse = _mse((target - base_realized).reshape(-1))

    families = {
        name: _fit_packet(
            residual,
            reference,
            reference.packets[name],
            ridge=ridge,
            rcond=rcond,
            base_trust_radius=base_trust_radius,
            packet_coefficient_limit=packet_coefficient_limit,
        )
        for name in FAMILIES
    }
    for family in families.values():
        family["incremental_mse_reduction_vs_base"] = (
            base_predicted_mse - float(family["predicted_mse"])
        )
        family["incremental_gain_per_packet_dof"] = (
            family["incremental_mse_reduction_vs_base"] / max(int(family["packet_dofs"]), 1)
        )
    extension_winner = max(
        FAMILIES,
        key=lambda name: (
            float(families[name]["incremental_mse_reduction_vs_base"]),
            -int(families[name]["packet_dofs"]),
            name,
        ),
    )
    best_incremental_gain = float(
        families[extension_winner]["incremental_mse_reduction_vs_base"]
    )
    if best_incremental_gain <= max(1e-14, initial_mse * 1e-4):
        extension_winner = "none"
    return {
        "initial_mse": initial_mse,
        "initial_psnr": _safe_psnr(initial_mse),
        "base_predicted_mse": base_predicted_mse,
        "base_predicted_psnr": _safe_psnr(base_predicted_mse),
        "base_realized_mse": base_realized_mse,
        "base_realized_psnr": _safe_psnr(base_realized_mse),
        "base_step_fraction": base_fraction,
        "base_linearization_max_abs": base_linearization_max_abs,
        "base_coefficients": [float(v) for v in base_coefficients],
        "base_tangent_rank": reference.base_system.rank,
        "base_tangent_raw_columns": reference.base_system.raw_columns,
        "base_tangent_nullity": (
            reference.base_system.raw_columns - reference.base_system.rank
        ),
        "extension_winner": extension_winner,
        "families": families,
        "price_scope": DIAGNOSTIC_PRICE_SCOPE,
        "compression_claim_authorized": False,
    }


def _bounded_packet_target(
    base: torch.Tensor,
    packet: Packet,
    coefficients: torch.Tensor,
    *,
    margin: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale a zero-offset packet so its target remains inside display range."""
    if float(packet.offset.abs().max()) > 1e-12:
        raise ValueError("bounded packet helper only accepts zero-offset packets")
    delta = (packet.columns @ coefficients).reshape_as(base)
    fraction = 1.0
    for _ in range(40):
        candidate = base + fraction * delta
        if float(candidate.min()) >= margin and float(candidate.max()) <= 1.0 - margin:
            return candidate, coefficients * fraction
        fraction *= 0.8
    raise RuntimeError("unable to construct an in-range packet target")


def identifiability_targets(reference: AuctionReference) -> dict[str, torch.Tensor]:
    dtype = reference.base_render.dtype
    targets: dict[str, torch.Tensor] = {"null": reference.base_render.clone()}

    current_step = torch.zeros_like(reference.parameter_vector)
    # Deliberately combine geometry, appearance, and opacity directions.
    current_step[0] = 0.22
    current_step[7] = -0.18
    current_step[17] = 0.16
    current_step[-1] = -0.14
    targets["current_tangent"] = reference.render_normalized_step(current_step).detach()

    affine_coefficients = torch.tensor(
        [0.085, -0.045, 0.065, -0.055, 0.075, 0.035], dtype=dtype
    )
    affine_target, _ = _bounded_packet_target(
        reference.base_render, reference.packets["affine"], affine_coefficients
    )
    targets["affine"] = affine_target

    carrier_coefficients = torch.tensor(
        [0.11, -0.075, 0.06, -0.055, 0.09, 0.08], dtype=dtype
    )
    carrier_target, _ = _bounded_packet_target(
        reference.base_render, reference.packets["carrier"], carrier_coefficients
    )
    targets["carrier"] = carrier_target

    birth_color = torch.tensor([0.88, 0.18, 0.67], dtype=dtype)
    birth_target = reference.packets["finite_birth"].realize(birth_color)
    if float(birth_target.min()) < -1e-9 or float(birth_target.max()) > 1.0 + 1e-9:
        raise RuntimeError("finite birth identifiability target is outside [0,1]")
    targets["finite_birth"] = birth_target
    return targets


def _stage0_gates(
    by_target: dict[str, Any],
    jvp_error: dict[str, float],
    *,
    tangent_nullity: int,
) -> dict[str, Any]:
    expected_extensions = {
        "null": "none",
        "current_tangent": "none",
        "affine": "affine",
        "carrier": "carrier",
        "finite_birth": "finite_birth",
    }
    winner_match = all(
        by_target[name]["extension_winner"] == expected
        for name, expected in expected_extensions.items()
    )
    current = by_target["current_tangent"]
    current_initial = max(float(current["initial_mse"]), 1e-30)
    tangent_residual_ratio = max(
        float(current["base_predicted_mse"]) / current_initial,
        float(current["base_realized_mse"]) / current_initial,
    )
    winner_residual_ratios = []
    packet_errors = []
    orthogonality_errors = []
    for target_name, target in by_target.items():
        for family in FAMILIES:
            values = target["families"][family]
            packet_errors.append(float(values["packet_realization_max_abs"]))
            orthogonality_errors.append(
                float(values["residualized_orthogonality_max_abs"])
            )
        if target_name in FAMILIES:
            winner_residual_ratios.append(
                float(target["families"][target_name]["predicted_mse"])
                / max(float(target["initial_mse"]), 1e-30)
            )

    base_gain_relative_errors = []
    for target in by_target.values():
        initial = float(target["initial_mse"])
        predicted_gain = initial - float(target["base_predicted_mse"])
        realized_gain = initial - float(target["base_realized_mse"])
        if abs(predicted_gain) > 1e-20:
            base_gain_relative_errors.append(
                abs(predicted_gain - realized_gain) / abs(predicted_gain)
            )

    checks = {
        "jvp_relative_l2": {
            "value": float(jvp_error["relative_l2"]),
            "threshold": "<=1e-6",
            "pass": float(jvp_error["relative_l2"]) <= 1e-6,
        },
        "jvp_max_abs": {
            "value": float(jvp_error["max_abs"]),
            "threshold": "<=1e-8",
            "pass": float(jvp_error["max_abs"]) <= 1e-8,
        },
        "image_quotient_nullity": {
            "value": int(tangent_nullity),
            "threshold": ">=1",
            "pass": int(tangent_nullity) >= 1,
        },
        "residualized_orthogonality_max_abs": {
            "value": max(orthogonality_errors, default=0.0),
            "threshold": "<=1e-10",
            "pass": max(orthogonality_errors, default=0.0) <= 1e-10,
        },
        "packet_realization_max_abs": {
            "value": max(packet_errors, default=0.0),
            "threshold": "<=1e-10",
            "pass": max(packet_errors, default=0.0) <= 1e-10,
        },
        "current_tangent_residual_ratio": {
            "value": tangent_residual_ratio,
            "threshold": "<=1e-3",
            "pass": tangent_residual_ratio <= 1e-3,
        },
        "base_predicted_realized_gain_relative_error": {
            "value": max(base_gain_relative_errors, default=0.0),
            "threshold": "<=0.02",
            "pass": max(base_gain_relative_errors, default=0.0) <= 0.02,
        },
        "extension_winner_identifiability": {
            "value": winner_match,
            "threshold": "all 5 synthetic controls match",
            "pass": winner_match,
        },
        "extension_winner_residual_ratio": {
            "value": max(winner_residual_ratios, default=0.0),
            "threshold": "<=1e-6",
            "pass": max(winner_residual_ratios, default=0.0) <= 1e-6,
        },
    }
    return {
        "pass": all(bool(check["pass"]) for check in checks.values()),
        "meaning": (
            "self-consistency gate only; a pass authorizes Stage-1 instrument implementation "
            "and full-path preflight, not development measurement or any method, renderer, "
            "rate, or compression claim"
        ),
        "checks": checks,
    }


def run_identifiability_suite(
    *,
    height: int = 13,
    width: int = 13,
    sigma_cutoff: float = 3.0,
    ridge: float = 1e-3,
    rcond: float = 1e-5,
    base_trust_radius: float = 0.75,
    packet_coefficient_limit: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    reference = build_reference(
        height=height,
        width=width,
        sigma_cutoff=sigma_cutoff,
        rcond=rcond,
    )
    targets = identifiability_targets(reference)
    rows: list[dict[str, Any]] = []
    by_target: dict[str, Any] = {}
    for target_kind in TARGET_KINDS:
        result = auction_target(
            reference,
            targets[target_kind],
            ridge=ridge,
            rcond=rcond,
            base_trust_radius=base_trust_radius,
            packet_coefficient_limit=packet_coefficient_limit,
        )
        by_target[target_kind] = result
        for family in FAMILIES:
            values = result["families"][family]
            rows.append(
                {
                    "target_kind": target_kind,
                    "family": family,
                    "initial_mse": result["initial_mse"],
                    "base_predicted_mse": result["base_predicted_mse"],
                    "family_predicted_mse": values["predicted_mse"],
                    "incremental_mse_reduction_vs_base": values[
                        "incremental_mse_reduction_vs_base"
                    ],
                    "optimized_dofs": values["optimized_dofs"],
                    "packet_dofs": values["packet_dofs"],
                    "packet_realization_max_abs": values["packet_realization_max_abs"],
                    "price_scope": DIAGNOSTIC_PRICE_SCOPE,
                }
            )
    jvp_error = finite_difference_jvp_error(reference, seed=seed)
    tangent_nullity = reference.base_system.raw_columns - reference.base_system.rank
    stage0_gate = _stage0_gates(
        by_target,
        jvp_error,
        tangent_nullity=tangent_nullity,
    )
    return {
        "protocol": "BENCH-009 residual tangent auction Stage 0",
        "scope": "deterministic_small_cpu_identifiability_only",
        "seed": seed,
        "height": height,
        "width": width,
        "sigma_cutoff": sigma_cutoff,
        "ridge": ridge,
        "rcond": rcond,
        "base_trust_radius": base_trust_radius,
        "packet_coefficient_limit": packet_coefficient_limit,
        "price_scope": DIAGNOSTIC_PRICE_SCOPE,
        "compression_claim_authorized": False,
        "provenance": _collect_provenance(),
        "base_field_sha256": _sha256_tensor(reference.parameter_vector),
        "target_sha256": {name: _sha256_tensor(value) for name, value in targets.items()},
        "jvp_finite_difference": jvp_error,
        "stage0_gate": stage0_gate,
        "stage1_instrument_implementation_authorized": bool(stage0_gate["pass"]),
        "stage1_development_run_authorized": False,
        "method_promotion_authorized": False,
        "base_tangent": {
            "raw_columns": reference.base_system.raw_columns,
            "rank": reference.base_system.rank,
            "nullity": tangent_nullity,
            "retained_singular_values": [
                float(value) for value in reference.base_system.singular_values
            ],
            "scaling": "per-column L2 normalization before truncated SVD",
            "gauge_treatment": (
                "retain only image-space left singular vectors above rcond*s_max; candidate "
                "columns and residual are projected off that identifiable span"
            ),
        },
        "by_target": by_target,
        "rows": rows,
        "limitations": [
            "Stage 0 uses one tiny synthetic field, one selected parent, and one candidate birth.",
            (
                "Extension controls are self-generated from the same benchmark packet definitions; "
                "they validate identifiability and realization plumbing, not relevance to natural residuals."
            ),
            "DOF/packet counts are not actual stream bytes and omit selection signaling.",
            "Carrier rendering is benchmark-local and has no production/CUDA/codec path.",
            "The current tangent freezes discrete support radii and is valid only locally.",
            "Finite birth geometry/opacity are searched nowhere and frozen before its color solve.",
            (
                "Provenance binds source/worktree bytes but does not make this dirty-tree run "
                "equivalent to an immutable commit."
            ),
            "No natural image, recovery trajectory, wall-clock comparison, or compression claim is made.",
        ],
    }


def _write_outputs(result: dict[str, Any], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    config = {
        key: result[key]
        for key in (
            "protocol",
            "scope",
            "seed",
            "height",
            "width",
            "sigma_cutoff",
            "ridge",
            "rcond",
            "base_trust_radius",
            "packet_coefficient_limit",
            "price_scope",
            "compression_claim_authorized",
        )
    }
    config["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "dtype": "float64",
    }
    config["base_field_sha256"] = result["base_field_sha256"]
    config["target_sha256"] = result["target_sha256"]
    config["provenance"] = result["provenance"]
    (outdir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (outdir / "identifiability.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    row_fields = list(result["rows"][0]) if result["rows"] else []
    with (outdir / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields)
        writer.writeheader()
        writer.writerows(result["rows"])

    lines = [
        "# BENCH-009 residual tangent auction — Stage 0",
        "",
        "**Scope:** deterministic small-CPU identifiability control only.",
        "",
        "**Rate warning:** DOF/packet accounting is diagnostic only; this is not bytes, BPP, "
        "or compression evidence.",
        "",
        "| Target | Base MSE | Best extension | Affine incremental | Carrier incremental | "
        "Finite-birth incremental |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for target_kind in TARGET_KINDS:
        target = result["by_target"][target_kind]
        family = target["families"]
        lines.append(
            f"| {target_kind} | {target['base_predicted_mse']:.8g} | "
            f"{target['extension_winner']} | "
            f"{family['affine']['incremental_mse_reduction_vs_base']:.8g} | "
            f"{family['carrier']['incremental_mse_reduction_vs_base']:.8g} | "
            f"{family['finite_birth']['incremental_mse_reduction_vs_base']:.8g} |"
        )
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(("", "## Stage-0 instrument gate", ""))
    lines.append(
        f"**Decision:** {'pass' if result['stage0_gate']['pass'] else 'kill'} — "
        f"{result['stage0_gate']['meaning']}."
    )
    lines.extend(("", "| Check | Value | Threshold | Pass |", "|---|---:|---|:---:|"))
    for name, check in result["stage0_gate"]["checks"].items():
        lines.append(
            f"| {name} | {check['value']} | {check['threshold']} | "
            f"{'yes' if check['pass'] else 'no'} |"
        )
    lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="results/bench009_tangent_auction_stage0",
        help="output directory",
    )
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--width", type=int, default=13)
    parser.add_argument("--sigma-cutoff", type=float, default=3.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--rcond", type=float, default=1e-5)
    parser.add_argument("--base-trust-radius", type=float, default=0.75)
    parser.add_argument("--packet-coefficient-limit", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    result = run_identifiability_suite(
        height=args.height,
        width=args.width,
        sigma_cutoff=args.sigma_cutoff,
        ridge=args.ridge,
        rcond=args.rcond,
        base_trust_radius=args.base_trust_radius,
        packet_coefficient_limit=args.packet_coefficient_limit,
        seed=args.seed,
    )
    _write_outputs(result, Path(args.outdir))
    print(json.dumps({
        "outdir": str(args.outdir),
        "jvp_finite_difference": result["jvp_finite_difference"],
        "extension_winners": {
            name: result["by_target"][name]["extension_winner"] for name in TARGET_KINDS
        },
        "stage0_gate_pass": result["stage0_gate"]["pass"],
        "compression_claim_authorized": False,
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
