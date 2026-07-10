"""Unambiguous storage accounting for fixed-budget Gaussian-image benchmarks.

The canonical fair-comparison lane counts only the frozen, constant-RGB 2D RS payload:
two means, two log-scales, one rotation, and three color values.  At float32 precision that is
eight scalars, or 32 bytes per Gaussian.  Image dimensions, renderer settings, Python/Tensor
objects, optimizer state, and training-only auxiliaries such as ``scale_max`` are benchmark
metadata and are deliberately outside this analytical payload.

This is *not* an actual codec or checkpoint size.  A generation-density covariance filter must be
materialized into the two stored scales before applying this accounting.  Packed codec sizes,
NPZ/checkpoint sizes, and source/target/reconstruction file sizes must be reported separately.
The explicit field names returned here are designed to prevent those quantities from being
silently compared as though they had the same semantics.
"""

from __future__ import annotations

import os
from numbers import Integral
from pathlib import Path
from typing import Any


STORAGE_ACCOUNTING_NAME = "float32_constant_rgb_rs_payload_v1"
KIBIBYTE_BYTES = 1024
FIXED_STORAGE_BUDGET_KIB = 168
FIXED_STORAGE_BUDGET_BYTES = FIXED_STORAGE_BUDGET_KIB * KIBIBYTE_BYTES

FLOAT32_BYTES = 4
CONSTANT_RGB_RS_SCALARS_PER_GAUSSIAN = 8
CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN = FLOAT32_BYTES * CONSTANT_RGB_RS_SCALARS_PER_GAUSSIAN
CONSTANT_RGB_RS_FIXED_OVERHEAD_BYTES = 0


def _integer(value: int, name: str, *, minimum: int) -> int:
    """Return ``value`` as an int while rejecting booleans and lossy coercions."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {resolved}")
    return resolved


def gaussian_cap_for_budget(
    budget_bytes: int,
    *,
    bytes_per_gaussian: int = CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN,
    fixed_overhead_bytes: int = CONSTANT_RGB_RS_FIXED_OVERHEAD_BYTES,
) -> int:
    """Return the largest whole Gaussian payload that fits within ``budget_bytes``.

    The result uses floor division.  Use :func:`exact_gaussian_cap_for_budget` when the budget must
    be exhausted exactly rather than merely respected.
    """
    budget = _integer(budget_bytes, "budget_bytes", minimum=1)
    per_gaussian = _integer(bytes_per_gaussian, "bytes_per_gaussian", minimum=1)
    overhead = _integer(fixed_overhead_bytes, "fixed_overhead_bytes", minimum=0)
    usable = budget - overhead
    if usable < per_gaussian:
        raise ValueError(
            "budget_bytes must fit at least one Gaussian after fixed overhead: "
            f"budget={budget}, overhead={overhead}, bytes_per_gaussian={per_gaussian}"
        )
    return usable // per_gaussian


def representation_payload_bytes(
    n_gaussians: int,
    *,
    bytes_per_gaussian: int = CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN,
    fixed_overhead_bytes: int = CONSTANT_RGB_RS_FIXED_OVERHEAD_BYTES,
) -> int:
    """Return analytical frozen-representation bytes for ``n_gaussians``.

    A zero count is accepted so callers can account for an explicitly empty or failed allocation,
    although successful fixed-budget benchmark rows should contain a positive count.
    """
    count = _integer(n_gaussians, "n_gaussians", minimum=0)
    per_gaussian = _integer(bytes_per_gaussian, "bytes_per_gaussian", minimum=1)
    overhead = _integer(fixed_overhead_bytes, "fixed_overhead_bytes", minimum=0)
    return overhead + count * per_gaussian


def exact_gaussian_cap_for_budget(
    budget_bytes: int,
    *,
    bytes_per_gaussian: int = CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN,
    fixed_overhead_bytes: int = CONSTANT_RGB_RS_FIXED_OVERHEAD_BYTES,
) -> int:
    """Return the Gaussian cap, failing if whole Gaussians do not exhaust the budget exactly."""
    cap = gaussian_cap_for_budget(
        budget_bytes,
        bytes_per_gaussian=bytes_per_gaussian,
        fixed_overhead_bytes=fixed_overhead_bytes,
    )
    used = representation_payload_bytes(
        cap,
        bytes_per_gaussian=bytes_per_gaussian,
        fixed_overhead_bytes=fixed_overhead_bytes,
    )
    if used != int(budget_bytes):
        raise ValueError(
            "budget is not exactly divisible by the Gaussian layout after fixed overhead: "
            f"budget={int(budget_bytes)}, used={used}, remainder={int(budget_bytes) - used}"
        )
    return cap


FIXED_STORAGE_GAUSSIAN_CAP = exact_gaussian_cap_for_budget(FIXED_STORAGE_BUDGET_BYTES)


def _file_size(path: str | os.PathLike[str] | None, name: str) -> int | None:
    """Return an exact serialized-file size, with missing requested artifacts failing closed."""
    if path is None:
        return None
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"{name} does not exist: {artifact}")
    if not artifact.is_file():
        raise ValueError(f"{name} must be a regular file: {artifact}")
    return int(artifact.stat().st_size)


def row_storage_accounting(
    n_gaussians: int,
    *,
    source_path: str | os.PathLike[str] | None = None,
    target_path: str | os.PathLike[str] | None = None,
    reconstruction_path: str | os.PathLike[str] | None = None,
    budget_bytes: int = FIXED_STORAGE_BUDGET_BYTES,
    bytes_per_gaussian: int = CONSTANT_RGB_RS_BYTES_PER_GAUSSIAN,
    fixed_overhead_bytes: int = CONSTANT_RGB_RS_FIXED_OVERHEAD_BYTES,
) -> dict[str, Any]:
    """Build non-ambiguous storage fields for one completed benchmark row.

    ``source_file_bytes``, ``target_file_bytes``, and ``reconstruction_file_bytes`` are on-disk
    container sizes.  ``representation_payload_bytes`` is the analytical float32 Gaussian payload.
    The function intentionally emits no generic ``bytes`` or ``image_size_bytes`` key.

    ``storage_budget_status`` is ``underfilled``, ``exact``, or ``overfilled`` based on the actual
    Gaussian count.  This makes naturally underfilled and adaptive over-budget methods visible
    instead of silently treating the requested cap as the achieved representation size.
    """
    budget = _integer(budget_bytes, "budget_bytes", minimum=1)
    per_gaussian = _integer(bytes_per_gaussian, "bytes_per_gaussian", minimum=1)
    overhead = _integer(fixed_overhead_bytes, "fixed_overhead_bytes", minimum=0)
    cap = gaussian_cap_for_budget(
        budget,
        bytes_per_gaussian=per_gaussian,
        fixed_overhead_bytes=overhead,
    )
    actual_count = _integer(n_gaussians, "n_gaussians", minimum=0)
    payload = representation_payload_bytes(
        actual_count,
        bytes_per_gaussian=per_gaussian,
        fixed_overhead_bytes=overhead,
    )
    if payload < budget:
        status = "underfilled"
    elif payload == budget:
        status = "exact"
    else:
        status = "overfilled"

    return {
        "storage_accounting": STORAGE_ACCOUNTING_NAME,
        "storage_budget_bytes": budget,
        "storage_bytes_per_gaussian": per_gaussian,
        "storage_fixed_overhead_bytes": overhead,
        "storage_gaussian_cap": cap,
        "storage_actual_gaussians": actual_count,
        "representation_payload_bytes": payload,
        "storage_budget_status": status,
        "storage_exact_budget": status == "exact",
        "storage_unused_budget_bytes": max(0, budget - payload),
        "storage_over_budget_bytes": max(0, payload - budget),
        "source_file_bytes": _file_size(source_path, "source_path"),
        "target_file_bytes": _file_size(target_path, "target_path"),
        "reconstruction_file_bytes": _file_size(reconstruction_path, "reconstruction_path"),
    }
