"""Standalone matrix-free tangent utilities for BENCH-009.

This module builds a masked Jacobian operator at the zero step of a differentiable flat-render
closure.  Its normal path does not materialize the Jacobian: range discovery uses JVPs, the
reduced matrix ``B = Q.T @ J`` uses VJPs, and all subsequent solves operate on the small
factorization.  Only after a full-parameter randomized attempt misses its fixed-probe tolerance,
a fail-safe path materializes coordinate JVP columns for robust float64 rank discovery.

The utility deliberately contains no parent fitting, action search, runner, recovery, result
writer, quality gate, or codec logic.  A damped predicted action is also kept separate from the
literal unregularized projector statistic used for zero-offset six-column extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Callable, Sequence

import torch


DEFAULT_BLOCK_SIZE = 32
DEFAULT_PROBE_COUNT = 8
DEFAULT_PROBE_TOLERANCE = 1e-5
FROZEN_RCONDS = (1e-5, 1e-4)
SCALE_FLOOR = 1e-12
DAMPING_DIMENSIONLESS_PARENT_CHART = "dimensionless_parent_chart"
DAMPING_COLUMN_NORMALIZED = "column_normalized_image_sensitivity"
DAMPING_COORDINATE_SYSTEMS = (
    DAMPING_DIMENSIONLESS_PARENT_CHART,
    DAMPING_COLUMN_NORMALIZED,
)

FlatRender = Callable[[torch.Tensor], torch.Tensor]


def _tensor_payload(value: torch.Tensor) -> bytes:
    detached = value.detach().contiguous().cpu()
    metadata = f"dtype={detached.dtype};shape={tuple(detached.shape)};".encode()
    raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    return metadata + raw


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact bytes without assuming a CPU input."""

    return hashlib.sha256(_tensor_payload(value)).hexdigest()


def _strings_sha256(values: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        hasher.update(len(payload).to_bytes(8, byteorder="little", signed=False))
        hasher.update(payload)
    return hasher.hexdigest()


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _validate_damping_coordinate_system(value: str) -> str:
    if value not in DAMPING_COORDINATE_SYSTEMS:
        raise ValueError(
            "damping_coordinate_system must be 'dimensionless_parent_chart' or "
            "'column_normalized_image_sensitivity'"
        )
    return value


def _validate_like_vector(
    value: torch.Tensor,
    template: torch.Tensor,
    length: int,
    *,
    name: str,
) -> None:
    if value.ndim != 1 or int(value.numel()) != int(length):
        raise ValueError(f"{name} must be a flat vector of length {length}")
    if value.device != template.device or value.dtype != template.dtype:
        raise ValueError(f"{name} must match operator dtype and device")
    if not _finite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class MaskedJacobianOperators:
    """JVP/VJP access to a boolean-row-masked Jacobian at a zero parameter step."""

    flat_render: FlatRender = field(repr=False)
    origin: torch.Tensor
    mask: torch.Tensor
    output_at_zero: torch.Tensor
    strict: bool = True

    @property
    def parameter_count(self) -> int:
        return int(self.origin.numel())

    @property
    def full_output_count(self) -> int:
        return int(self.output_at_zero.numel())

    @property
    def output_count(self) -> int:
        return int(self.mask.sum())

    @property
    def masked_output_at_zero(self) -> torch.Tensor:
        return self.output_at_zero[self.mask]

    @property
    def mask_sha256(self) -> str:
        return tensor_sha256(self.mask)

    @property
    def output_at_zero_sha256(self) -> str:
        return tensor_sha256(self.output_at_zero)

    def _masked_render(self, step: torch.Tensor) -> torch.Tensor:
        output = self.flat_render(step)
        if not isinstance(output, torch.Tensor) or output.ndim != 1:
            raise ValueError("flat render closure must return one flat tensor")
        if int(output.numel()) != self.full_output_count:
            raise ValueError("flat render closure changed output length")
        if output.device != self.origin.device or output.dtype != self.origin.dtype:
            raise ValueError("flat render output must match parameter dtype and device")
        return output[self.mask]

    def jvp(self, direction: torch.Tensor) -> torch.Tensor:
        """Apply the masked Jacobian to one parameter-space direction."""

        _validate_like_vector(
            direction,
            self.origin,
            self.parameter_count,
            name="JVP direction",
        )
        _primal, tangent = torch.autograd.functional.jvp(
            self._masked_render,
            self.origin,
            direction,
            create_graph=False,
            strict=self.strict,
        )
        tangent = tangent.detach()
        if not _finite(tangent):
            raise RuntimeError("masked JVP produced non-finite values")
        return tangent

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        """Apply the transpose masked Jacobian to one image-space cotangent."""

        _validate_like_vector(
            cotangent,
            self.masked_output_at_zero,
            self.output_count,
            name="VJP cotangent",
        )
        _primal, transpose_product = torch.autograd.functional.vjp(
            self._masked_render,
            self.origin,
            v=cotangent,
            create_graph=False,
            strict=self.strict,
        )
        transpose_product = transpose_product.detach()
        if not _finite(transpose_product):
            raise RuntimeError("masked VJP produced non-finite values")
        return transpose_product

    def select_output(self, value: torch.Tensor) -> torch.Tensor:
        """Select masked rows from a full output vector/matrix, or validate an already masked one."""

        if value.ndim not in (1, 2):
            raise ValueError("output value must be a vector or row-major matrix")
        if value.device != self.origin.device or value.dtype != self.origin.dtype:
            raise ValueError("output value must match operator dtype and device")
        rows = int(value.shape[0])
        if rows == self.full_output_count:
            selected = value[self.mask]
        elif rows == self.output_count:
            selected = value
        else:
            raise ValueError("output value row count matches neither full nor masked output")
        if not _finite(selected):
            raise ValueError("output value must be finite")
        return selected


def build_masked_jacobian_operators(
    flat_render: FlatRender,
    parameter_template: torch.Tensor,
    mask: torch.Tensor | Sequence[bool] | None = None,
    *,
    strict: bool = True,
) -> MaskedJacobianOperators:
    """Construct masked JVP/VJP operators at an internally created all-zero step.

    ``parameter_template`` supplies only the flat parameter shape, dtype, and device.  The render
    closure is always linearized at ``zeros_like(parameter_template)``.
    """

    if not isinstance(parameter_template, torch.Tensor):
        raise TypeError("parameter_template must be a tensor")
    if parameter_template.ndim != 1 or parameter_template.numel() == 0:
        raise ValueError("parameter_template must be a non-empty flat tensor")
    if not torch.is_floating_point(parameter_template):
        raise ValueError("parameter_template must have a floating dtype")
    if not _finite(parameter_template):
        raise ValueError("parameter_template must be finite")
    origin = torch.zeros_like(parameter_template).detach()
    output = flat_render(origin)
    if not isinstance(output, torch.Tensor) or output.ndim != 1 or output.numel() == 0:
        raise ValueError("flat render closure must return one non-empty flat tensor")
    if output.device != origin.device or output.dtype != origin.dtype:
        raise ValueError("flat render output must match parameter dtype and device")
    if not _finite(output):
        raise ValueError("flat render at zero must be finite")
    if mask is None:
        selected = torch.ones(output.numel(), dtype=torch.bool, device=output.device)
    else:
        selected = torch.as_tensor(mask, device=output.device)
        if selected.dtype != torch.bool:
            raise ValueError("output mask must be boolean")
        selected = selected.reshape(-1)
        if selected.numel() != output.numel():
            raise ValueError("output mask length does not match flat render")
        if not bool(selected.any()):
            raise ValueError("output mask selects no rows")
    return MaskedJacobianOperators(
        flat_render=flat_render,
        origin=origin,
        mask=selected.detach().clone(),
        output_at_zero=output.detach().clone(),
        strict=bool(strict),
    )


@dataclass(frozen=True)
class RandomizedRangeResult:
    """Diagnostics and basis from the fixed-probe blocked randomized range finder."""

    basis: torch.Tensor
    probe_directions: torch.Tensor
    probe_images: torch.Tensor
    converged: bool
    reconstruction_error: float
    reconstruction_error_history: tuple[float, ...]
    basis_dimension_history: tuple[int, ...]
    block_size: int
    sample_cap: int
    sample_count: int
    probe_count: int
    jvp_calls: int
    seed: int
    probe_tolerance: float
    basis_sha256: str
    probe_directions_sha256: str
    probe_images_sha256: str
    sample_directions_sha256: str
    basis_construction: str
    randomized_reconstruction_error: float
    exact_fallback_attempted: bool
    exact_fallback_converged: bool
    exact_fallback_jvp_calls: int
    exact_fallback_rank: int | None
    exact_fallback_reconstruction_error: float | None
    exact_fallback_coordinate_images_sha256: str | None
    exact_fallback_basis_float64_sha256: str | None

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])


class MatrixFreeConvergenceError(RuntimeError):
    """Raised when a factorization is requested from an unconverged range estimate."""

    def __init__(self, result: RandomizedRangeResult):
        self.result = result
        fallback = (
            "; exact coordinate-JVP fallback also failed"
            if result.exact_fallback_attempted
            else ""
        )
        super().__init__(
            "matrix-free range finder did not converge: "
            f"fixed-probe relative error {result.reconstruction_error:.6g} exceeds "
            f"{result.probe_tolerance:.6g} after {result.sample_count}/{result.sample_cap} "
            f"sample directions{fallback}"
        )


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _orthonormalize_columns(
    values: torch.Tensor,
    against: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("range candidates must be a matrix")
    candidate = values
    if against is not None and against.numel():
        for _ in range(2):
            candidate = candidate - against @ (against.T @ candidate)
    if candidate.numel() == 0:
        return candidate.new_zeros((candidate.shape[0], 0))
    # A small rank-revealing SVD is used only on the current random block (or six extension
    # columns), never on J.  Unlike unpivoted QR it remains valid when leading columns vanish
    # after residualization or later columns duplicate earlier ones.
    q, singular, _vh = torch.linalg.svd(candidate, full_matrices=False)
    if singular.numel() == 0:
        return q[:, :0]
    maximum = singular[0]
    if float(maximum) == 0.0:
        return q[:, :0]
    epsilon = torch.finfo(candidate.dtype).eps
    cutoff = 16.0 * epsilon * max(candidate.shape) * maximum
    keep = singular > cutoff
    q = q[:, keep]
    if against is not None and against.numel() and q.numel():
        for _ in range(2):
            q = q - against @ (against.T @ q)
        q, singular, _vh = torch.linalg.svd(q, full_matrices=False)
        keep = singular > 16.0 * epsilon * max(q.shape) * singular[0]
        q = q[:, keep]
    return q


def _relative_probe_error(probe_images: torch.Tensor, basis: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(probe_images)
    if float(denominator) == 0.0:
        return 0.0
    if basis.numel():
        remainder = probe_images - basis @ (basis.T @ probe_images)
    else:
        remainder = probe_images
    return float(torch.linalg.vector_norm(remainder) / denominator)


def _canonicalize_svd_column_signs(basis: torch.Tensor) -> torch.Tensor:
    """Choose a deterministic sign for each SVD column without changing its projector."""

    if basis.numel() == 0:
        return basis
    pivot_rows = basis.abs().argmax(dim=0)
    pivot_values = basis[pivot_rows, torch.arange(basis.shape[1], device=basis.device)]
    signs = torch.where(pivot_values < 0.0, -torch.ones_like(pivot_values), torch.ones_like(pivot_values))
    return basis * signs


@dataclass(frozen=True)
class _ExactCoordinateRangeFallback:
    basis: torch.Tensor
    basis_float64: torch.Tensor
    coordinate_images: torch.Tensor
    reconstruction_error: float
    converged: bool
    jvp_calls: int


def _exact_coordinate_jvp_range(
    operators: MaskedJacobianOperators,
    probe_images: torch.Tensor,
    *,
    probe_tolerance: float,
) -> _ExactCoordinateRangeFallback:
    """Build the complete Jacobian range from coordinate JVPs and a CPU float64 SVD.

    This is deliberately an exhaustion-only fallback: it costs exactly ``P`` additional JVPs.
    Rank discovery happens in float64, including for a float32 renderer, so a random image block
    cannot erase weak but probe-relevant directions through its lower-precision block cutoff.
    The resulting basis is cast back to the operator dtype/device and checked against the original
    fixed probe images.  Thus the fallback does not weaken or reinterpret the frozen tolerance.
    """

    coordinate_images = torch.stack(
        [
            operators.jvp(_coordinate_direction(operators.origin, index))
            for index in range(operators.parameter_count)
        ],
        dim=1,
    )
    exact_matrix = coordinate_images.detach().to(device="cpu", dtype=torch.float64)
    if exact_matrix.numel() == 0:
        basis_float64 = exact_matrix.new_zeros((operators.output_count, 0))
    else:
        left, singular, _vh = torch.linalg.svd(exact_matrix, full_matrices=False)
        if singular.numel() == 0 or float(singular[0]) == 0.0:
            basis_float64 = left[:, :0]
        else:
            cutoff = (
                16.0
                * torch.finfo(torch.float64).eps
                * max(exact_matrix.shape)
                * singular[0]
            )
            basis_float64 = left[:, singular > cutoff]
            basis_float64 = _canonicalize_svd_column_signs(basis_float64)
    basis = basis_float64.to(
        device=operators.origin.device,
        dtype=operators.origin.dtype,
    )
    error = _relative_probe_error(probe_images, basis)
    return _ExactCoordinateRangeFallback(
        basis=basis,
        basis_float64=basis_float64,
        coordinate_images=coordinate_images,
        reconstruction_error=error,
        converged=bool(error <= probe_tolerance),
        jvp_calls=operators.parameter_count,
    )


def randomized_range_finder(
    operators: MaskedJacobianOperators,
    *,
    seed: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sample_cap: int | None = None,
    probe_count: int = DEFAULT_PROBE_COUNT,
    probe_tolerance: float = DEFAULT_PROBE_TOLERANCE,
) -> RandomizedRangeResult:
    """Find the masked Jacobian range with deterministic random blocks and fixed probes.

    Parameter-space sample blocks are orthonormalized before JVP application.  The default sample
    cap is exactly ``P`` (the parameter count).  If that full-cap attempt misses tolerance, an
    exact coordinate-JVP fallback performs rank discovery by CPU float64 SVD and checks the
    cast-back basis against the unchanged probes.  A partial cap never invokes this fallback.
    Probe directions and their JVP images are generated once and remain fixed for every check.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    cap = operators.parameter_count if sample_cap is None else int(sample_cap)
    if cap <= 0 or cap > operators.parameter_count:
        raise ValueError("sample_cap must be in [1, P]")
    if probe_count <= 0:
        raise ValueError("probe_count must be positive")
    if not math.isfinite(probe_tolerance) or probe_tolerance <= 0.0:
        raise ValueError("probe_tolerance must be finite and positive")
    canonical_seed = int(seed) % (2**63 - 1)
    probe_columns = min(int(probe_count), operators.parameter_count)
    probe_generator = _make_generator(
        operators.origin.device,
        (canonical_seed + 1_000_003) % (2**63 - 1),
    )
    raw_probes = torch.randn(
        operators.parameter_count,
        probe_columns,
        generator=probe_generator,
        device=operators.origin.device,
        dtype=operators.origin.dtype,
    )
    probe_directions = _orthonormalize_columns(raw_probes)
    probe_columns = int(probe_directions.shape[1])
    if probe_columns == 0:
        raise RuntimeError("deterministic probe block was numerically rank zero")
    probe_images = torch.stack(
        [operators.jvp(probe_directions[:, index]) for index in range(probe_columns)],
        dim=1,
    )

    basis = operators.masked_output_at_zero.new_zeros((operators.output_count, 0))
    parameter_basis = operators.origin.new_zeros((operators.parameter_count, 0))
    error = _relative_probe_error(probe_images, basis)
    errors = [error]
    dimensions = [0]
    sampled = 0
    sample_hasher = hashlib.sha256()
    sample_generator = _make_generator(operators.origin.device, canonical_seed)

    while error > probe_tolerance and sampled < cap:
        requested = min(int(block_size), cap - sampled)
        raw_block = torch.randn(
            operators.parameter_count,
            requested,
            generator=sample_generator,
            device=operators.origin.device,
            dtype=operators.origin.dtype,
        )
        parameter_block = _orthonormalize_columns(raw_block, parameter_basis)
        if parameter_block.shape[1] == 0:
            break
        if parameter_block.shape[1] > requested:
            parameter_block = parameter_block[:, :requested]
        parameter_basis = torch.cat((parameter_basis, parameter_block), dim=1)
        sample_hasher.update(_tensor_payload(parameter_block))
        sampled += int(parameter_block.shape[1])

        image_block = torch.stack(
            [
                operators.jvp(parameter_block[:, index])
                for index in range(int(parameter_block.shape[1]))
            ],
            dim=1,
        )
        new_basis = _orthonormalize_columns(image_block, basis)
        if new_basis.numel():
            basis = torch.cat((basis, new_basis), dim=1)
        error = _relative_probe_error(probe_images, basis)
        errors.append(error)
        dimensions.append(int(basis.shape[1]))

    randomized_error = error
    exact_fallback: _ExactCoordinateRangeFallback | None = None
    # A full-cap randomized attempt was allowed to span the complete parameter space.  If it
    # still misses the fixed probes, retry from the actual coordinate JVP columns.  Partial caps
    # remain intentionally fail-closed because an exact fallback would violate their work budget.
    if (
        error > probe_tolerance
        and cap == operators.parameter_count
        and sampled == cap
    ):
        exact_fallback = _exact_coordinate_jvp_range(
            operators,
            probe_images,
            probe_tolerance=probe_tolerance,
        )
        basis = exact_fallback.basis
        error = exact_fallback.reconstruction_error
        errors.append(error)
        dimensions.append(int(basis.shape[1]))
    converged = bool(error <= probe_tolerance)
    fallback_jvp_calls = 0 if exact_fallback is None else exact_fallback.jvp_calls
    return RandomizedRangeResult(
        basis=basis,
        probe_directions=probe_directions,
        probe_images=probe_images,
        converged=converged,
        reconstruction_error=error,
        reconstruction_error_history=tuple(errors),
        basis_dimension_history=tuple(dimensions),
        block_size=int(block_size),
        sample_cap=cap,
        sample_count=sampled,
        probe_count=probe_columns,
        jvp_calls=probe_columns + sampled + fallback_jvp_calls,
        seed=canonical_seed,
        probe_tolerance=float(probe_tolerance),
        basis_sha256=tensor_sha256(basis),
        probe_directions_sha256=tensor_sha256(probe_directions),
        probe_images_sha256=tensor_sha256(probe_images),
        sample_directions_sha256=sample_hasher.hexdigest(),
        basis_construction=(
            "exact_coordinate_jvp_float64_svd"
            if exact_fallback is not None
            else "blocked_randomized_jvp"
        ),
        randomized_reconstruction_error=randomized_error,
        exact_fallback_attempted=exact_fallback is not None,
        exact_fallback_converged=(
            False if exact_fallback is None else exact_fallback.converged
        ),
        exact_fallback_jvp_calls=fallback_jvp_calls,
        exact_fallback_rank=(
            None if exact_fallback is None else int(exact_fallback.basis.shape[1])
        ),
        exact_fallback_reconstruction_error=(
            None if exact_fallback is None else exact_fallback.reconstruction_error
        ),
        exact_fallback_coordinate_images_sha256=(
            None
            if exact_fallback is None
            else tensor_sha256(exact_fallback.coordinate_images)
        ),
        exact_fallback_basis_float64_sha256=(
            None
            if exact_fallback is None
            else tensor_sha256(exact_fallback.basis_float64)
        ),
    )


def form_reduced_matrix(
    operators: MaskedJacobianOperators,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Form ``B = Q.T @ J`` one row at a time through VJPs."""

    if basis.ndim != 2 or basis.shape[0] != operators.output_count:
        raise ValueError("range basis shape does not match masked output")
    if basis.device != operators.origin.device or basis.dtype != operators.origin.dtype:
        raise ValueError("range basis must match operator dtype and device")
    if not _finite(basis):
        raise ValueError("range basis must be finite")
    if basis.shape[1] == 0:
        return operators.origin.new_zeros((0, operators.parameter_count))
    return torch.stack(
        [operators.vjp(basis[:, index]) for index in range(int(basis.shape[1]))],
        dim=0,
    )


@dataclass(frozen=True)
class MatrixFreeTangentFactorization:
    """Column-scaled truncated SVD of a converged matrix-free tangent approximation."""

    operators: MaskedJacobianOperators = field(repr=False)
    range_result: RandomizedRangeResult
    reduced_matrix: torch.Tensor
    column_scales: torch.Tensor
    scaled_reduced_matrix: torch.Tensor
    rcond: float
    cutoff: float
    rank: int
    left_basis: torch.Tensor
    singular_values: torch.Tensor
    right_basis: torch.Tensor
    all_singular_values: torch.Tensor
    vjp_calls: int
    left_basis_sha256: str
    singular_values_sha256: str

    @property
    def parameter_count(self) -> int:
        return self.operators.parameter_count

    @property
    def output_count(self) -> int:
        return self.operators.output_count


def factorize_masked_tangent(
    operators: MaskedJacobianOperators,
    *,
    rconds: Sequence[float] = FROZEN_RCONDS,
    seed: int = 0,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sample_cap: int | None = None,
    probe_count: int = DEFAULT_PROBE_COUNT,
    probe_tolerance: float = DEFAULT_PROBE_TOLERANCE,
) -> dict[float, MatrixFreeTangentFactorization]:
    """Build one range/reduced SVD and expose truncations for each requested ``rcond``.

    This function fails closed: no factorization is returned when the fixed-probe range check has
    not converged.  Call :func:`randomized_range_finder` directly when diagnostics from a failed
    attempt are needed.
    """

    thresholds = tuple(float(value) for value in rconds)
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("rconds must be a non-empty sequence of unique values")
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("every rcond must be finite and positive")
    range_result = randomized_range_finder(
        operators,
        seed=seed,
        block_size=block_size,
        sample_cap=sample_cap,
        probe_count=probe_count,
        probe_tolerance=probe_tolerance,
    )
    if not range_result.converged:
        raise MatrixFreeConvergenceError(range_result)
    reduced = form_reduced_matrix(operators, range_result.basis)
    column_scales = torch.linalg.vector_norm(reduced, dim=0).clamp_min(SCALE_FLOOR)
    scaled_reduced = reduced / column_scales
    if reduced.shape[0] == 0:
        reduced_left = reduced.new_zeros((0, 0))
        all_singular = reduced.new_zeros((0,))
        reduced_vh = reduced.new_zeros((0, operators.parameter_count))
    else:
        reduced_left, all_singular, reduced_vh = torch.linalg.svd(
            scaled_reduced,
            full_matrices=False,
        )
    full_left = range_result.basis @ reduced_left
    largest = float(all_singular[0]) if all_singular.numel() else 0.0
    factorizations: dict[float, MatrixFreeTangentFactorization] = {}
    for rcond in thresholds:
        cutoff = rcond * largest
        keep = all_singular > cutoff
        left = full_left[:, keep]
        singular = all_singular[keep]
        right = reduced_vh[keep].T
        factorizations[rcond] = MatrixFreeTangentFactorization(
            operators=operators,
            range_result=range_result,
            reduced_matrix=reduced,
            column_scales=column_scales,
            scaled_reduced_matrix=scaled_reduced,
            rcond=rcond,
            cutoff=cutoff,
            rank=int(keep.sum()),
            left_basis=left,
            singular_values=singular,
            right_basis=right,
            all_singular_values=all_singular,
            vjp_calls=int(range_result.basis.shape[1]),
            left_basis_sha256=tensor_sha256(left),
            singular_values_sha256=tensor_sha256(singular),
        )
    return factorizations


@dataclass(frozen=True)
class _DampedChartSolve:
    coefficients: torch.Tensor
    predicted_action: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    condition: float | None


def _solve_in_dimensionless_chart(
    retained_left_basis: torch.Tensor,
    retained_design: torch.Tensor,
    rhs: torch.Tensor,
    *,
    damping: float,
) -> _DampedChartSolve:
    """Damp raw frozen-chart coefficients inside an already rank-truncated image subspace."""

    coefficient_count = int(retained_design.shape[1])
    if retained_left_basis.shape[1] == 0:
        return _DampedChartSolve(
            coefficients=rhs.new_zeros((coefficient_count,)),
            predicted_action=rhs.new_zeros(rhs.shape),
            singular_values=rhs.new_zeros((0,)),
            rank=0,
            condition=None,
        )
    local_left, all_singular, vh = torch.linalg.svd(
        retained_design,
        full_matrices=False,
    )
    largest = all_singular[0]
    numerical_cutoff = (
        16.0
        * torch.finfo(retained_design.dtype).eps
        * max(retained_design.shape)
        * largest
    )
    keep = all_singular > numerical_cutoff
    singular = all_singular[keep]
    if singular.numel() == 0:
        return _DampedChartSolve(
            coefficients=rhs.new_zeros((coefficient_count,)),
            predicted_action=rhs.new_zeros(rhs.shape),
            singular_values=singular,
            rank=0,
            condition=None,
        )
    effective_left = retained_left_basis @ local_left[:, keep]
    coordinates = singular / (
        singular.square() + float(damping) ** 2
    ) * (effective_left.T @ rhs)
    coefficients = vh[keep].T @ coordinates
    predicted = effective_left @ (singular * coordinates)
    return _DampedChartSolve(
        coefficients=coefficients,
        predicted_action=predicted,
        singular_values=singular,
        rank=int(keep.sum()),
        condition=float(singular[0] / singular[-1]),
    )


@dataclass(frozen=True)
class DampedNormalAction:
    """Truncated ``(J.T J + damping^2 I)`` solve and its linear predicted action."""

    damping: float
    damping_coordinate_system: str
    damping_rank: int
    damping_condition: float | None
    damping_singular_values: torch.Tensor
    parameter_step: torch.Tensor
    predicted_action: torch.Tensor
    residual_energy: float
    predicted_post_energy: float
    predicted_energy_reduction: float


def solve_damped_normal(
    factorization: MatrixFreeTangentFactorization,
    residual: torch.Tensor,
    *,
    damping: float,
    damping_coordinate_system: str = DAMPING_DIMENSIONLESS_PARENT_CHART,
) -> DampedNormalAction:
    """Solve the truncated damped normal system and return ``J @ step`` prediction.

    The Stage-1 default damps the raw frozen dimensionless parent chart.  The separately named
    column-normalized option reproduces the Stage-0 sensitivity-normalized ridge convention; its
    damping values are not dimensionless-chart damping values and must not be reported as such.
    """

    _validate_like_vector(
        residual,
        factorization.left_basis.new_empty((factorization.output_count,)),
        factorization.output_count,
        name="residual",
    )
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    coordinate_system = _validate_damping_coordinate_system(damping_coordinate_system)
    if factorization.rank == 0:
        step = factorization.operators.origin.new_zeros((factorization.parameter_count,))
        predicted = residual.new_zeros(residual.shape)
        damping_singular = factorization.singular_values
        damping_rank = 0
        damping_condition = None
    elif coordinate_system == DAMPING_COLUMN_NORMALIZED:
        projected = factorization.left_basis.T @ residual
        singular = factorization.singular_values
        denominator = singular.square() + float(damping) ** 2
        coordinates = singular / denominator * projected
        step = (factorization.right_basis @ coordinates) / factorization.column_scales
        step = torch.where(
            factorization.column_scales > SCALE_FLOOR,
            step,
            torch.zeros_like(step),
        )
        predicted = factorization.range_result.basis @ (
            factorization.reduced_matrix @ step
        )
        damping_singular = singular
        damping_rank = factorization.rank
        damping_condition = float(singular[0] / singular[-1])
    else:
        retained_design = (
            factorization.left_basis.T
            @ factorization.range_result.basis
            @ factorization.reduced_matrix
        )
        chart_solve = _solve_in_dimensionless_chart(
            factorization.left_basis,
            retained_design,
            residual,
            damping=damping,
        )
        step = chart_solve.coefficients
        predicted = chart_solve.predicted_action
        damping_singular = chart_solve.singular_values
        damping_rank = chart_solve.rank
        damping_condition = chart_solve.condition
    residual_energy = float(residual.square().sum())
    post_energy = float((residual - predicted).square().sum())
    return DampedNormalAction(
        damping=float(damping),
        damping_coordinate_system=coordinate_system,
        damping_rank=damping_rank,
        damping_condition=damping_condition,
        damping_singular_values=damping_singular,
        parameter_step=step,
        predicted_action=predicted,
        residual_energy=residual_energy,
        predicted_post_energy=post_energy,
        predicted_energy_reduction=residual_energy - post_energy,
    )


def _validate_six_columns(
    factorization: MatrixFreeTangentFactorization,
    columns: torch.Tensor,
) -> None:
    if columns.ndim != 2 or columns.shape != (factorization.output_count, 6):
        raise ValueError("extension columns must have masked shape (output_count, 6)")
    if (
        columns.device != factorization.left_basis.device
        or columns.dtype != factorization.left_basis.dtype
    ):
        raise ValueError("extension columns must match factorization dtype and device")
    if not _finite(columns):
        raise ValueError("extension columns must be finite")


@dataclass(frozen=True)
class JointTangentSixFactorization:
    """Compact jointly scaled/truncated factorization of ``[J, A6]``.

    ``J`` is represented by the already computed matrix-free range basis and reduced matrix.
    The six extension columns are represented exactly in that range plus their orthogonal
    complement, so this construction performs no additional JVP or VJP calls.
    """

    base_factorization: MatrixFreeTangentFactorization = field(repr=False)
    extension_columns: torch.Tensor
    compact_image_basis: torch.Tensor
    compact_reduced_matrix: torch.Tensor
    column_scales: torch.Tensor
    scaled_compact_reduced_matrix: torch.Tensor
    rcond: float
    cutoff: float
    rank: int
    left_basis: torch.Tensor
    singular_values: torch.Tensor
    right_basis: torch.Tensor
    all_singular_values: torch.Tensor
    base_range_rank: int
    extension_complement_rank: int
    reused_range_jvp_calls: int
    reused_reduced_vjp_calls: int
    additional_jvp_calls: int
    additional_vjp_calls: int
    extension_columns_sha256: str
    compact_reduced_matrix_sha256: str
    left_basis_sha256: str
    singular_values_sha256: str

    @property
    def base_parameter_count(self) -> int:
        return self.base_factorization.parameter_count

    @property
    def coefficient_count(self) -> int:
        return self.base_parameter_count + 6

    @property
    def output_count(self) -> int:
        return self.base_factorization.output_count


def factorize_joint_tangent_six(
    base_factorization: MatrixFreeTangentFactorization,
    extension_columns: torch.Tensor,
    *,
    rconds: Sequence[float] | None = None,
) -> dict[float, JointTangentSixFactorization]:
    """Factor ``[J, A6]`` exactly in a compact image basis, reusing matrix-free ``J`` factors."""

    _validate_six_columns(base_factorization, extension_columns)
    thresholds = (
        (float(base_factorization.rcond),)
        if rconds is None
        else tuple(float(value) for value in rconds)
    )
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("rconds must be a non-empty sequence of unique values")
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("every rcond must be finite and positive")
    if not base_factorization.range_result.converged:
        raise RuntimeError("joint factorization requires a converged base range")

    range_basis = base_factorization.range_result.basis
    complement = _orthonormalize_columns(extension_columns, range_basis)
    compact_basis = torch.cat((range_basis, complement), dim=1)
    top = torch.cat(
        (
            base_factorization.reduced_matrix,
            range_basis.T @ extension_columns,
        ),
        dim=1,
    )
    if complement.shape[1]:
        bottom = torch.cat(
            (
                extension_columns.new_zeros(
                    (complement.shape[1], base_factorization.parameter_count)
                ),
                complement.T @ extension_columns,
            ),
            dim=1,
        )
        compact_reduced = torch.cat((top, bottom), dim=0)
    else:
        compact_reduced = top
    column_scales = torch.linalg.vector_norm(compact_reduced, dim=0).clamp_min(
        SCALE_FLOOR
    )
    scaled_reduced = compact_reduced / column_scales
    if compact_reduced.shape[0] == 0:
        reduced_left = compact_reduced.new_zeros((0, 0))
        all_singular = compact_reduced.new_zeros((0,))
        reduced_vh = compact_reduced.new_zeros(
            (0, base_factorization.parameter_count + 6)
        )
    else:
        reduced_left, all_singular, reduced_vh = torch.linalg.svd(
            scaled_reduced,
            full_matrices=False,
        )
    full_left = compact_basis @ reduced_left
    largest = float(all_singular[0]) if all_singular.numel() else 0.0
    common = {
        "base_factorization": base_factorization,
        "extension_columns": extension_columns,
        "compact_image_basis": compact_basis,
        "compact_reduced_matrix": compact_reduced,
        "column_scales": column_scales,
        "scaled_compact_reduced_matrix": scaled_reduced,
        "all_singular_values": all_singular,
        "base_range_rank": int(range_basis.shape[1]),
        "extension_complement_rank": int(complement.shape[1]),
        "reused_range_jvp_calls": base_factorization.range_result.jvp_calls,
        "reused_reduced_vjp_calls": base_factorization.vjp_calls,
        "additional_jvp_calls": 0,
        "additional_vjp_calls": 0,
        "extension_columns_sha256": tensor_sha256(extension_columns),
        "compact_reduced_matrix_sha256": tensor_sha256(compact_reduced),
    }
    result: dict[float, JointTangentSixFactorization] = {}
    for rcond in thresholds:
        cutoff = rcond * largest
        keep = all_singular > cutoff
        left = full_left[:, keep]
        singular = all_singular[keep]
        right = reduced_vh[keep].T
        result[rcond] = JointTangentSixFactorization(
            **common,
            rcond=rcond,
            cutoff=cutoff,
            rank=int(keep.sum()),
            left_basis=left,
            singular_values=singular,
            right_basis=right,
            left_basis_sha256=tensor_sha256(left),
            singular_values_sha256=tensor_sha256(singular),
        )
    return result


@dataclass(frozen=True)
class JointDampedAction:
    """Damped ``[J, A6]`` action with an explicitly named optional finite offset."""

    damping: float
    damping_coordinate_system: str
    damping_rank: int
    damping_condition: float | None
    damping_singular_values: torch.Tensor
    finite_offset_present: bool
    finite_offset: torch.Tensor
    base_parameter_step: torch.Tensor
    extension_coefficients: torch.Tensor
    concatenated_coefficients: torch.Tensor
    predicted_linear_action: torch.Tensor
    predicted_action: torch.Tensor
    residual_energy: float
    adjusted_rhs_energy: float
    predicted_post_energy: float
    predicted_energy_reduction: float
    finite_offset_sha256: str
    coefficients_sha256: str
    predicted_action_sha256: str


def solve_joint_damped_normal(
    factorization: JointTangentSixFactorization,
    residual: torch.Tensor,
    *,
    damping: float,
    finite_offset: torch.Tensor | None = None,
    damping_coordinate_system: str = DAMPING_DIMENSIONLESS_PARENT_CHART,
) -> JointDampedAction:
    """Solve the jointly truncated damped normal system for ``[J, A6]``.

    ``finite_offset`` is subtracted before fitting and added back to the predicted action.  It is
    intended for a renderer-derived finite affine term such as a normalized two-birth denominator
    offset; it is never interpreted as a literal projector contribution or trust-scaled.  The
    default damping metric is the frozen dimensionless base-plus-RGB coefficient chart.
    """

    template = factorization.left_basis.new_empty((factorization.output_count,))
    _validate_like_vector(
        residual,
        template,
        factorization.output_count,
        name="residual",
    )
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")
    coordinate_system = _validate_damping_coordinate_system(damping_coordinate_system)
    if finite_offset is None:
        offset = residual.new_zeros(residual.shape)
        offset_present = False
    else:
        _validate_like_vector(
            finite_offset,
            template,
            factorization.output_count,
            name="finite offset",
        )
        offset = finite_offset
        offset_present = True
    adjusted = residual - offset
    if factorization.rank == 0:
        coefficients = factorization.column_scales.new_zeros(
            (factorization.coefficient_count,)
        )
        predicted_linear = residual.new_zeros(residual.shape)
        damping_singular = factorization.singular_values
        damping_rank = 0
        damping_condition = None
    elif coordinate_system == DAMPING_COLUMN_NORMALIZED:
        projected = factorization.left_basis.T @ adjusted
        singular = factorization.singular_values
        coordinates = singular / (
            singular.square() + float(damping) ** 2
        ) * projected
        coefficients = (
            factorization.right_basis @ coordinates
        ) / factorization.column_scales
        active_columns = factorization.column_scales > SCALE_FLOOR
        coefficients = torch.where(
            active_columns,
            coefficients,
            torch.zeros_like(coefficients),
        )
        predicted_linear = factorization.compact_image_basis @ (
            factorization.compact_reduced_matrix @ coefficients
        )
        damping_singular = singular
        damping_rank = factorization.rank
        damping_condition = float(singular[0] / singular[-1])
    else:
        retained_design = (
            factorization.left_basis.T
            @ factorization.compact_image_basis
            @ factorization.compact_reduced_matrix
        )
        chart_solve = _solve_in_dimensionless_chart(
            factorization.left_basis,
            retained_design,
            adjusted,
            damping=damping,
        )
        coefficients = chart_solve.coefficients
        predicted_linear = chart_solve.predicted_action
        damping_singular = chart_solve.singular_values
        damping_rank = chart_solve.rank
        damping_condition = chart_solve.condition
    predicted = offset + predicted_linear
    post_energy = float((residual - predicted).square().sum())
    residual_energy = float(residual.square().sum())
    base_count = factorization.base_parameter_count
    return JointDampedAction(
        damping=float(damping),
        damping_coordinate_system=coordinate_system,
        damping_rank=damping_rank,
        damping_condition=damping_condition,
        damping_singular_values=damping_singular,
        finite_offset_present=offset_present,
        finite_offset=offset.clone(),
        base_parameter_step=coefficients[:base_count],
        extension_coefficients=coefficients[base_count:],
        concatenated_coefficients=coefficients,
        predicted_linear_action=predicted_linear,
        predicted_action=predicted,
        residual_energy=residual_energy,
        adjusted_rhs_energy=float(adjusted.square().sum()),
        predicted_post_energy=post_energy,
        predicted_energy_reduction=residual_energy - post_energy,
        finite_offset_sha256=tensor_sha256(offset),
        coefficients_sha256=tensor_sha256(coefficients),
        predicted_action_sha256=tensor_sha256(predicted),
    )


@dataclass(frozen=True)
class UniformLinfJointScale:
    """One uniform trust factor applied to a concatenated base-plus-extension vector."""

    radius: float
    factor: float
    unscaled_max_abs: float
    scaled_max_abs: float
    base_parameter_step: torch.Tensor
    extension_coefficients: torch.Tensor
    concatenated_coefficients: torch.Tensor
    coefficients_sha256: str


def uniform_linf_scale_joint_coefficients(
    base_parameter_step: torch.Tensor,
    extension_coefficients: torch.Tensor,
    radius: float,
) -> UniformLinfJointScale:
    """Apply exactly one ``Linf`` trust factor to all base and six extension coordinates."""

    if base_parameter_step.ndim != 1 or base_parameter_step.numel() == 0:
        raise ValueError("base parameter step must be a non-empty flat vector")
    if extension_coefficients.ndim != 1 or extension_coefficients.numel() != 6:
        raise ValueError("extension coefficients must be a flat six-vector")
    if (
        extension_coefficients.device != base_parameter_step.device
        or extension_coefficients.dtype != base_parameter_step.dtype
    ):
        raise ValueError("base and extension coefficients must match dtype and device")
    if not _finite(base_parameter_step) or not _finite(extension_coefficients):
        raise ValueError("base and extension coefficients must be finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("trust radius must be finite and positive")
    concatenated = torch.cat((base_parameter_step, extension_coefficients))
    maximum = float(concatenated.abs().max())
    factor = min(1.0, float(radius) / max(maximum, 1e-30))
    scaled = concatenated * factor
    base_count = int(base_parameter_step.numel())
    return UniformLinfJointScale(
        radius=float(radius),
        factor=factor,
        unscaled_max_abs=maximum,
        scaled_max_abs=float(scaled.abs().max()),
        base_parameter_step=scaled[:base_count],
        extension_coefficients=scaled[base_count:],
        concatenated_coefficients=scaled,
        coefficients_sha256=tensor_sha256(scaled),
    )


def _validate_paired_operators(
    discovery: MaskedJacobianOperators,
    heldout: MaskedJacobianOperators,
) -> None:
    if discovery.parameter_count != heldout.parameter_count:
        raise ValueError("paired operators must have the same parameter count")
    if discovery.origin.device != heldout.origin.device or discovery.origin.dtype != heldout.origin.dtype:
        raise ValueError("paired operators must match dtype and device")
    if discovery.full_output_count != heldout.full_output_count:
        raise ValueError("paired operators must share one full output layout")
    if bool((discovery.mask & heldout.mask).any()):
        raise ValueError("discovery and held-out operator masks overlap")
    if not bool((discovery.mask | heldout.mask).all()):
        raise ValueError("discovery and held-out operator masks must cover all output rows")


def _canonical_coordinate_ids(
    parameter_count: int,
    coordinate_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    if coordinate_ids is None:
        return tuple(f"parameter_{index:06d}" for index in range(parameter_count))
    values = tuple(coordinate_ids)
    if len(values) != parameter_count:
        raise ValueError("coordinate ID count must match the parameter count")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("coordinate IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("coordinate IDs must be unique")
    return values


def _coordinate_direction(template: torch.Tensor, index: int) -> torch.Tensor:
    direction = torch.zeros_like(template)
    direction[index] = 1.0
    return direction


def _append_modified_gram_schmidt(
    basis: torch.Tensor,
    column: torch.Tensor,
) -> torch.Tensor:
    """Append one numerically independent column to an incremental QR image basis."""

    candidate = column
    if basis.numel():
        for _ in range(2):
            candidate = candidate - basis @ (basis.T @ candidate)
    raw_norm = torch.linalg.vector_norm(column)
    candidate_norm = torch.linalg.vector_norm(candidate)
    epsilon = torch.finfo(column.dtype).eps
    threshold = (
        32.0
        * epsilon
        * max(int(column.numel()), int(basis.shape[1]) + 1)
        * raw_norm
    )
    if float(raw_norm) == 0.0 or bool(candidate_norm <= threshold):
        return basis
    unit = candidate / candidate_norm
    if basis.numel():
        for _ in range(2):
            unit = unit - basis @ (basis.T @ unit)
        unit_norm = torch.linalg.vector_norm(unit)
        if bool(unit_norm <= 32.0 * epsilon * max(int(column.numel()), 1)):
            return basis
        unit = unit / unit_norm
    return torch.cat((basis, unit[:, None]), dim=1)


@dataclass(frozen=True)
class _OmpSixSelection:
    coordinate_ids: tuple[str, ...]
    coordinate_indices: tuple[int, ...]
    design: torch.Tensor
    selected_correlations: tuple[float, ...]
    residual_energy_history: tuple[float, ...]
    qr_rank_history: tuple[int, ...]
    vjp_calls: int
    jvp_calls: int


def _select_six_by_vjp_omp(
    residual: torch.Tensor,
    parameter_template: torch.Tensor,
    coordinate_ids: tuple[str, ...],
    *,
    vjp: Callable[[torch.Tensor], torch.Tensor],
    jvp: Callable[[torch.Tensor], torch.Tensor],
) -> _OmpSixSelection:
    """Select six unique coordinates by VJP correlation and selected-column QR updates."""

    parameter_count = int(parameter_template.numel())
    if parameter_count < 6:
        raise ValueError("tangent6 requires at least six parameter coordinates")
    selected_indices: list[int] = []
    selected_ids: list[str] = []
    selected_correlations: list[float] = []
    columns: list[torch.Tensor] = []
    image_basis = residual.new_zeros((residual.numel(), 0))
    remaining = residual
    energy_history = [float(remaining.square().sum())]
    rank_history = [0]
    for _ in range(6):
        correlations = vjp(remaining)
        available = [
            index for index in range(parameter_count) if index not in selected_indices
        ]
        chosen = min(
            available,
            key=lambda index: (
                -float(correlations[index].abs()),
                coordinate_ids[index],
                index,
            ),
        )
        selected_indices.append(chosen)
        selected_ids.append(coordinate_ids[chosen])
        selected_correlations.append(float(correlations[chosen]))
        column = jvp(_coordinate_direction(parameter_template, chosen))
        columns.append(column)
        image_basis = _append_modified_gram_schmidt(image_basis, column)
        if image_basis.numel():
            remaining = residual - image_basis @ (image_basis.T @ residual)
        else:
            remaining = residual
        energy_history.append(float(remaining.square().sum()))
        rank_history.append(int(image_basis.shape[1]))
    return _OmpSixSelection(
        coordinate_ids=tuple(selected_ids),
        coordinate_indices=tuple(selected_indices),
        design=torch.stack(columns, dim=1),
        selected_correlations=tuple(selected_correlations),
        residual_energy_history=tuple(energy_history),
        qr_rank_history=tuple(rank_history),
        vjp_calls=6,
        jvp_calls=6,
    )


@dataclass(frozen=True)
class _SmallSixSolve:
    coefficients: torch.Tensor
    predicted_action: torch.Tensor
    column_scales: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    condition: float | None
    damping_coordinate_system: str
    damping_singular_values: torch.Tensor
    damping_rank: int
    damping_condition: float | None


def _solve_selected_six(
    design: torch.Tensor,
    residual: torch.Tensor,
    *,
    rcond: float,
    damping: float,
    damping_coordinate_system: str,
) -> _SmallSixSolve:
    if design.ndim != 2 or design.shape != (residual.numel(), 6):
        raise ValueError("selected tangent design must have shape (output_count, 6)")
    if design.device != residual.device or design.dtype != residual.dtype:
        raise ValueError("selected tangent design must match residual dtype and device")
    if not _finite(design) or not _finite(residual):
        raise ValueError("selected tangent design and residual must be finite")
    if not math.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("tangent6 rcond must be finite and positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("tangent6 damping must be finite and non-negative")
    coordinate_system = _validate_damping_coordinate_system(damping_coordinate_system)
    scales = torch.linalg.vector_norm(design, dim=0).clamp_min(SCALE_FLOOR)
    left, all_singular, vh = torch.linalg.svd(design / scales, full_matrices=False)
    largest = float(all_singular[0]) if all_singular.numel() else 0.0
    keep = all_singular > float(rcond) * largest
    singular = all_singular[keep]
    rank = int(keep.sum())
    if rank == 0:
        coefficients = residual.new_zeros((6,))
        predicted = residual.new_zeros(residual.shape)
        condition = None
        damping_singular = singular
        damping_rank = 0
        damping_condition = None
    elif coordinate_system == DAMPING_COLUMN_NORMALIZED:
        coordinates = singular / (
            singular.square() + float(damping) ** 2
        ) * (left[:, keep].T @ residual)
        coefficients = (vh[keep].T @ coordinates) / scales
        coefficients = torch.where(
            scales > SCALE_FLOOR,
            coefficients,
            torch.zeros_like(coefficients),
        )
        predicted = design @ coefficients
        condition = float(singular[0] / singular[-1])
        damping_singular = singular
        damping_rank = rank
        damping_condition = condition
    else:
        retained_left = left[:, keep]
        chart_solve = _solve_in_dimensionless_chart(
            retained_left,
            retained_left.T @ design,
            residual,
            damping=damping,
        )
        coefficients = chart_solve.coefficients
        predicted = chart_solve.predicted_action
        condition = float(singular[0] / singular[-1])
        damping_singular = chart_solve.singular_values
        damping_rank = chart_solve.rank
        damping_condition = chart_solve.condition
    return _SmallSixSolve(
        coefficients=coefficients,
        predicted_action=predicted,
        column_scales=scales,
        singular_values=singular,
        rank=rank,
        condition=condition,
        damping_coordinate_system=coordinate_system,
        damping_singular_values=damping_singular,
        damping_rank=damping_rank,
        damping_condition=damping_condition,
    )


def _selected_jvp_columns(
    operators: MaskedJacobianOperators,
    coordinate_indices: Sequence[int],
) -> torch.Tensor:
    return torch.stack(
        [
            operators.jvp(_coordinate_direction(operators.origin, index))
            for index in coordinate_indices
        ],
        dim=1,
    )


@dataclass(frozen=True)
class TangentSixCrossfitResult:
    """Discovery-selected tangent6 fit transported unchanged to paired held-out rows."""

    coordinate_ids: tuple[str, ...]
    coordinate_indices: tuple[int, ...]
    coefficients: torch.Tensor
    sparse_parameter_step: torch.Tensor
    discovery_design: torch.Tensor
    heldout_design: torch.Tensor
    column_scales: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    condition: float | None
    rcond: float
    damping: float
    damping_coordinate_system: str
    damping_rank: int
    damping_condition: float | None
    discovery_predicted_action: torch.Tensor
    heldout_predicted_action: torch.Tensor
    discovery_residual_energy: float
    discovery_post_energy: float
    discovery_energy_reduction: float
    heldout_residual_energy: float
    heldout_post_energy: float
    heldout_energy_reduction: float
    selected_correlations: tuple[float, ...]
    omp_residual_energy_history: tuple[float, ...]
    omp_qr_rank_history: tuple[int, ...]
    discovery_vjp_calls: int
    discovery_jvp_calls: int
    heldout_vjp_calls: int
    heldout_jvp_calls: int
    coordinate_identity_sha256: str
    discovery_design_sha256: str
    heldout_design_sha256: str
    coefficients_sha256: str
    heldout_prediction_sha256: str
    fit_policy: str = "discovery_only_identity_and_coefficients"
    heldout_refit: bool = False


def select_tangent_six_crossfit(
    discovery_operators: MaskedJacobianOperators,
    heldout_operators: MaskedJacobianOperators,
    discovery_residual: torch.Tensor,
    heldout_residual: torch.Tensor,
    *,
    coordinate_ids: Sequence[str] | None = None,
    rcond: float = 1e-5,
    damping: float = 0.0,
    damping_coordinate_system: str = DAMPING_DIMENSIONLESS_PARENT_CHART,
) -> TangentSixCrossfitResult:
    """Fit tangent6 identity/coefficients on discovery rows and only score held-out rows."""

    _validate_paired_operators(discovery_operators, heldout_operators)
    _validate_like_vector(
        discovery_residual,
        discovery_operators.masked_output_at_zero,
        discovery_operators.output_count,
        name="discovery residual",
    )
    _validate_like_vector(
        heldout_residual,
        heldout_operators.masked_output_at_zero,
        heldout_operators.output_count,
        name="held-out residual",
    )
    ids = _canonical_coordinate_ids(discovery_operators.parameter_count, coordinate_ids)
    selection = _select_six_by_vjp_omp(
        discovery_residual,
        discovery_operators.origin,
        ids,
        vjp=discovery_operators.vjp,
        jvp=discovery_operators.jvp,
    )
    solve = _solve_selected_six(
        selection.design,
        discovery_residual,
        rcond=rcond,
        damping=damping,
        damping_coordinate_system=damping_coordinate_system,
    )
    heldout_design = _selected_jvp_columns(
        heldout_operators,
        selection.coordinate_indices,
    )
    heldout_prediction = heldout_design @ solve.coefficients
    sparse_step = discovery_operators.origin.new_zeros(
        (discovery_operators.parameter_count,)
    )
    sparse_step[list(selection.coordinate_indices)] = solve.coefficients
    discovery_energy = float(discovery_residual.square().sum())
    discovery_post = float(
        (discovery_residual - solve.predicted_action).square().sum()
    )
    heldout_energy = float(heldout_residual.square().sum())
    heldout_post = float((heldout_residual - heldout_prediction).square().sum())
    return TangentSixCrossfitResult(
        coordinate_ids=selection.coordinate_ids,
        coordinate_indices=selection.coordinate_indices,
        coefficients=solve.coefficients,
        sparse_parameter_step=sparse_step,
        discovery_design=selection.design,
        heldout_design=heldout_design,
        column_scales=solve.column_scales,
        singular_values=solve.singular_values,
        rank=solve.rank,
        condition=solve.condition,
        rcond=float(rcond),
        damping=float(damping),
        damping_coordinate_system=solve.damping_coordinate_system,
        damping_rank=solve.damping_rank,
        damping_condition=solve.damping_condition,
        discovery_predicted_action=solve.predicted_action,
        heldout_predicted_action=heldout_prediction,
        discovery_residual_energy=discovery_energy,
        discovery_post_energy=discovery_post,
        discovery_energy_reduction=discovery_energy - discovery_post,
        heldout_residual_energy=heldout_energy,
        heldout_post_energy=heldout_post,
        heldout_energy_reduction=heldout_energy - heldout_post,
        selected_correlations=selection.selected_correlations,
        omp_residual_energy_history=selection.residual_energy_history,
        omp_qr_rank_history=selection.qr_rank_history,
        discovery_vjp_calls=selection.vjp_calls,
        discovery_jvp_calls=selection.jvp_calls,
        heldout_vjp_calls=0,
        heldout_jvp_calls=6,
        coordinate_identity_sha256=_strings_sha256(selection.coordinate_ids),
        discovery_design_sha256=tensor_sha256(selection.design),
        heldout_design_sha256=tensor_sha256(heldout_design),
        coefficients_sha256=tensor_sha256(solve.coefficients),
        heldout_prediction_sha256=tensor_sha256(heldout_prediction),
    )


@dataclass(frozen=True)
class TangentSixAllPixelOracle:
    """Separately labeled identity-and-coefficient oracle over discovery plus held-out rows."""

    coordinate_ids: tuple[str, ...]
    coordinate_indices: tuple[int, ...]
    coefficients: torch.Tensor
    sparse_parameter_step: torch.Tensor
    discovery_design: torch.Tensor
    heldout_design: torch.Tensor
    column_scales: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    condition: float | None
    rcond: float
    damping: float
    damping_coordinate_system: str
    damping_rank: int
    damping_condition: float | None
    discovery_predicted_action: torch.Tensor
    heldout_predicted_action: torch.Tensor
    all_pixel_residual_energy: float
    all_pixel_post_energy: float
    all_pixel_energy_reduction: float
    selected_correlations: tuple[float, ...]
    omp_residual_energy_history: tuple[float, ...]
    omp_qr_rank_history: tuple[int, ...]
    discovery_vjp_calls: int
    discovery_jvp_calls: int
    heldout_vjp_calls: int
    heldout_jvp_calls: int
    coordinate_identity_sha256: str
    combined_design_sha256: str
    coefficients_sha256: str
    fit_policy: str = "all_pixel_oracle_identity_and_coefficients"


def select_tangent_six_all_pixel_oracle(
    discovery_operators: MaskedJacobianOperators,
    heldout_operators: MaskedJacobianOperators,
    discovery_residual: torch.Tensor,
    heldout_residual: torch.Tensor,
    *,
    coordinate_ids: Sequence[str] | None = None,
    rcond: float = 1e-5,
    damping: float = 0.0,
    damping_coordinate_system: str = DAMPING_DIMENSIONLESS_PARENT_CHART,
) -> TangentSixAllPixelOracle:
    """Independently select and fit tangent6 on all rows as an excluded oracle."""

    _validate_paired_operators(discovery_operators, heldout_operators)
    _validate_like_vector(
        discovery_residual,
        discovery_operators.masked_output_at_zero,
        discovery_operators.output_count,
        name="discovery residual",
    )
    _validate_like_vector(
        heldout_residual,
        heldout_operators.masked_output_at_zero,
        heldout_operators.output_count,
        name="held-out residual",
    )
    ids = _canonical_coordinate_ids(discovery_operators.parameter_count, coordinate_ids)
    split = discovery_operators.output_count
    combined_residual = torch.cat((discovery_residual, heldout_residual))

    def combined_vjp(cotangent: torch.Tensor) -> torch.Tensor:
        return discovery_operators.vjp(cotangent[:split]) + heldout_operators.vjp(
            cotangent[split:]
        )

    def combined_jvp(direction: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                discovery_operators.jvp(direction),
                heldout_operators.jvp(direction),
            )
        )

    selection = _select_six_by_vjp_omp(
        combined_residual,
        discovery_operators.origin,
        ids,
        vjp=combined_vjp,
        jvp=combined_jvp,
    )
    solve = _solve_selected_six(
        selection.design,
        combined_residual,
        rcond=rcond,
        damping=damping,
        damping_coordinate_system=damping_coordinate_system,
    )
    discovery_design = selection.design[:split]
    heldout_design = selection.design[split:]
    sparse_step = discovery_operators.origin.new_zeros(
        (discovery_operators.parameter_count,)
    )
    sparse_step[list(selection.coordinate_indices)] = solve.coefficients
    total_energy = float(combined_residual.square().sum())
    post_energy = float((combined_residual - solve.predicted_action).square().sum())
    return TangentSixAllPixelOracle(
        coordinate_ids=selection.coordinate_ids,
        coordinate_indices=selection.coordinate_indices,
        coefficients=solve.coefficients,
        sparse_parameter_step=sparse_step,
        discovery_design=discovery_design,
        heldout_design=heldout_design,
        column_scales=solve.column_scales,
        singular_values=solve.singular_values,
        rank=solve.rank,
        condition=solve.condition,
        rcond=float(rcond),
        damping=float(damping),
        damping_coordinate_system=solve.damping_coordinate_system,
        damping_rank=solve.damping_rank,
        damping_condition=solve.damping_condition,
        discovery_predicted_action=solve.predicted_action[:split],
        heldout_predicted_action=solve.predicted_action[split:],
        all_pixel_residual_energy=total_energy,
        all_pixel_post_energy=post_energy,
        all_pixel_energy_reduction=total_energy - post_energy,
        selected_correlations=selection.selected_correlations,
        omp_residual_energy_history=selection.residual_energy_history,
        omp_qr_rank_history=selection.qr_rank_history,
        discovery_vjp_calls=selection.vjp_calls,
        discovery_jvp_calls=selection.jvp_calls,
        heldout_vjp_calls=selection.vjp_calls,
        heldout_jvp_calls=selection.jvp_calls,
        coordinate_identity_sha256=_strings_sha256(selection.coordinate_ids),
        combined_design_sha256=tensor_sha256(selection.design),
        coefficients_sha256=tensor_sha256(solve.coefficients),
    )


def residualize_six_columns(
    factorization: MatrixFreeTangentFactorization,
    columns: torch.Tensor,
) -> torch.Tensor:
    """Remove the truncated current-tangent projector from exactly six masked columns."""

    _validate_six_columns(factorization, columns)
    if factorization.rank == 0:
        return columns.clone()
    residualized = columns
    for _ in range(2):
        residualized = residualized - factorization.left_basis @ (
            factorization.left_basis.T @ residualized
        )
    return residualized


@dataclass(frozen=True)
class LiteralProjectorEnergy:
    """Literal unregularized projector-energy decomposition for six zero-offset columns."""

    rcond: float
    base_rank: int
    joint_rank: int
    residualized_columns: torch.Tensor
    joint_left_basis: torch.Tensor
    joint_singular_values: torch.Tensor
    residual_energy: float
    tangent_projector_energy: float
    incremental_projector_energy: float
    joint_projector_energy: float
    unexplained_energy: float
    tangent_orthogonality_max_abs: float


def literal_projector_energy(
    factorization: MatrixFreeTangentFactorization,
    residual: torch.Tensor,
    columns: torch.Tensor,
    *,
    rcond: float | None = None,
) -> LiteralProjectorEnergy:
    """Compute ``||P_[J,A] r||^2 - ||P_J r||^2`` without damping or trust clipping."""

    _validate_like_vector(
        residual,
        factorization.left_basis.new_empty((factorization.output_count,)),
        factorization.output_count,
        name="residual",
    )
    threshold = factorization.rcond if rcond is None else float(rcond)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("projector rcond must be finite and positive")
    residualized = residualize_six_columns(factorization, columns)

    # The literal statistic and the finite joint solve share the exact same compact, jointly
    # scaled/truncated [J, A6] factorization.  Only this function interprets its left basis as an
    # unregularized projector.
    joint_factorization = factorize_joint_tangent_six(
        factorization,
        columns,
        rconds=(threshold,),
    )[threshold]
    joint_left = joint_factorization.left_basis
    kept_singular = joint_factorization.singular_values
    tangent_coordinates = factorization.left_basis.T @ residual
    joint_coordinates = joint_left.T @ residual
    tangent_energy = float(tangent_coordinates.square().sum())
    joint = float(joint_coordinates.square().sum())
    incremental = joint - tangent_energy
    total = float(residual.square().sum())
    if factorization.rank:
        orthogonality = float(
            (factorization.left_basis.T @ residualized).abs().max()
        )
    else:
        orthogonality = 0.0
    return LiteralProjectorEnergy(
        rcond=threshold,
        base_rank=factorization.rank,
        joint_rank=joint_factorization.rank,
        residualized_columns=residualized,
        joint_left_basis=joint_left,
        joint_singular_values=kept_singular,
        residual_energy=total,
        tangent_projector_energy=tangent_energy,
        incremental_projector_energy=incremental,
        joint_projector_energy=joint,
        unexplained_energy=total - joint,
        tangent_orthogonality_max_abs=orthogonality,
    )


__all__ = [
    "DAMPING_COLUMN_NORMALIZED",
    "DAMPING_COORDINATE_SYSTEMS",
    "DAMPING_DIMENSIONLESS_PARENT_CHART",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_PROBE_COUNT",
    "DEFAULT_PROBE_TOLERANCE",
    "FROZEN_RCONDS",
    "DampedNormalAction",
    "JointDampedAction",
    "JointTangentSixFactorization",
    "LiteralProjectorEnergy",
    "MaskedJacobianOperators",
    "MatrixFreeConvergenceError",
    "MatrixFreeTangentFactorization",
    "RandomizedRangeResult",
    "TangentSixAllPixelOracle",
    "TangentSixCrossfitResult",
    "UniformLinfJointScale",
    "build_masked_jacobian_operators",
    "factorize_masked_tangent",
    "factorize_joint_tangent_six",
    "form_reduced_matrix",
    "literal_projector_energy",
    "randomized_range_finder",
    "residualize_six_columns",
    "select_tangent_six_all_pixel_oracle",
    "select_tangent_six_crossfit",
    "solve_damped_normal",
    "solve_joint_damped_normal",
    "tensor_sha256",
    "uniform_linf_scale_joint_coefficients",
]
