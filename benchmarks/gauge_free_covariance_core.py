"""Pure helpers for the COMP-007 gauge-free covariance codec assay.

The production SSPL1 codec stores two log standard deviations and an angle.  That chart is
redundant: swapping the axes and rotating by pi/2 produces the same covariance, and the angle is
undefined at isotropy.  This module exposes the exact two-dimensional log-Euclidean chart used by
the benchmark without changing the production codec.
"""
from __future__ import annotations

import math

import numpy as np


def _rs_arrays(
    log_scales: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scales = np.asarray(log_scales, dtype=np.float64)
    angles = np.asarray(rotations, dtype=np.float64).reshape(-1)
    if scales.ndim != 2 or scales.shape[1] != 2:
        raise ValueError(f"log_scales must have shape (N,2), got {scales.shape}")
    if angles.shape[0] != scales.shape[0]:
        raise ValueError("rotations and log_scales must have the same row count")
    if not np.isfinite(scales).all() or not np.isfinite(angles).all():
        raise ValueError("covariance coordinates must be finite")
    return scales, angles


def canonicalize_rs(
    log_scales: np.ndarray, rotations: np.ndarray, *, isotropy_tol: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the major/minor-axis representative of an RS covariance.

    The result has ``u_major >= u_minor`` and an angle in ``[0, pi)``.  Exact or tolerance-level
    isotropy receives angle zero because no orientation is identifiable there.
    """
    scales, angles = _rs_arrays(log_scales, rotations)
    if isotropy_tol < 0.0:
        raise ValueError("isotropy_tol must be non-negative")
    swap = scales[:, 1] > scales[:, 0]
    ordered = np.ascontiguousarray(np.sort(scales, axis=1)[:, ::-1])
    theta = np.mod(angles + swap.astype(np.float64) * (0.5 * math.pi), math.pi)
    theta[np.abs(ordered[:, 0] - ordered[:, 1]) <= isotropy_tol] = 0.0
    return ordered, theta


def rs_to_log_spd(log_scales: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Map RS parameters to a gauge-free Euclidean chart of the log covariance.

    For log standard deviations ``u, v`` and rotation ``theta``, the coordinates are

    ``ell=u+v, a=(u-v)*cos(2*theta), b=(u-v)*sin(2*theta)``.

    These are the three unique entries of ``log(Sigma)`` arranged as
    ``[[ell+a, b], [b, ell-a]]``.

    They are unchanged by ``(u,v,theta) -> (v,u,theta+pi/2)`` and by angle-period wrapping.
    """
    scales, theta = _rs_arrays(log_scales, rotations)
    trace_half = scales[:, 0] + scales[:, 1]
    difference = scales[:, 0] - scales[:, 1]
    return np.stack(
        [
            trace_half,
            difference * np.cos(2.0 * theta),
            difference * np.sin(2.0 * theta),
        ],
        axis=1,
    )


def log_spd_to_rs(
    coordinates: np.ndarray, *, isotropy_tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`rs_to_log_spd` to the canonical major/minor RS representative."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (N,3), got {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("log-SPD coordinates must be finite")
    if isotropy_tol < 0.0:
        raise ValueError("isotropy_tol must be non-negative")
    trace_half, axis_x, axis_y = coords.T
    difference = np.hypot(axis_x, axis_y)
    theta = np.mod(0.5 * np.arctan2(axis_y, axis_x), math.pi)
    theta[difference <= isotropy_tol] = 0.0
    scales = 0.5 * np.stack([trace_half + difference, trace_half - difference], axis=1)
    return scales, theta


def covariance_entries(log_scales: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Return ``(Sigma_xx, Sigma_xy, Sigma_yy)`` for RS log standard deviations."""
    scales, theta = _rs_arrays(log_scales, rotations)
    first = np.exp(2.0 * scales[:, 0])
    second = np.exp(2.0 * scales[:, 1])
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return np.stack(
        [
            first * cosine**2 + second * sine**2,
            (first - second) * cosine * sine,
            first * sine**2 + second * cosine**2,
        ],
        axis=1,
    )


def log_covariance_entries(coordinates: np.ndarray) -> np.ndarray:
    """Return ``(log(Sigma)_xx, log(Sigma)_xy, log(Sigma)_yy)`` from ``(m,a,b)``."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (N,3), got {coords.shape}")
    trace_half, axis_x, axis_y = coords.T
    return np.stack([trace_half + axis_x, axis_y, trace_half - axis_x], axis=1)


def quantize_linear(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-coordinate min/max uniform quantization with an exact constant-coordinate path."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("values must be a non-empty rank-two array")
    if not 1 <= bits <= 24:
        raise ValueError("bits must be in [1,24]")
    if not np.isfinite(x).all():
        raise ValueError("values must be finite")
    # The container stores range endpoints as float32.  Quantize against those exact
    # representable endpoints rather than float64 extrema which the decoder can never recover.
    # This makes a decoded value land on the same grid used by the encoder.
    lo = x.min(axis=0).astype(np.float32).astype(np.float64)
    hi = x.max(axis=0).astype(np.float32).astype(np.float64)
    # Canonicalize signed zero because the binary header distinguishes -0.0 from +0.0 while the
    # reconstructed numerical field does not.
    lo[lo == 0.0] = 0.0
    hi[hi == 0.0] = 0.0
    span = hi - lo
    levels = (1 << bits) - 1
    normalized = np.divide(x - lo, span, out=np.zeros_like(x), where=span > 0.0)
    quantized = np.rint(np.clip(normalized, 0.0, 1.0) * levels).astype(np.uint32)
    return quantized, lo, hi


def dequantize_linear(
    quantized: np.ndarray, lo: np.ndarray, hi: np.ndarray, bits: int
) -> np.ndarray:
    q = np.asarray(quantized, dtype=np.uint32)
    lower = np.asarray(lo, dtype=np.float64)
    upper = np.asarray(hi, dtype=np.float64)
    if q.ndim != 2 or lower.shape != (q.shape[1],) or upper.shape != (q.shape[1],):
        raise ValueError("quantized shape and range vectors disagree")
    levels = (1 << bits) - 1
    if levels <= 0 or np.any(q > levels):
        raise ValueError("quantized symbols exceed the requested bit depth")
    return lower + q.astype(np.float64) * ((upper - lower) / levels)


def quantize_circular(values: np.ndarray, bits: int, *, period: float = math.pi) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if not 1 <= bits <= 24 or not math.isfinite(period) or period <= 0.0:
        raise ValueError("invalid circular quantizer parameters")
    levels = 1 << bits
    return np.mod(np.rint(np.mod(x, period) / period * levels).astype(np.int64), levels).astype(
        np.uint32
    )


def dequantize_circular(
    quantized: np.ndarray, bits: int, *, period: float = math.pi
) -> np.ndarray:
    q = np.asarray(quantized, dtype=np.uint32)
    levels = 1 << bits
    if np.any(q >= levels):
        raise ValueError("circular symbols exceed the requested bit depth")
    return q.astype(np.float64) * (period / levels)


def modular_delta(quantized: np.ndarray, bits: int) -> np.ndarray:
    q = np.asarray(quantized, dtype=np.uint32)
    if q.ndim != 2:
        raise ValueError("quantized must be rank two")
    mask = np.uint32((1 << bits) - 1)
    previous = np.concatenate([np.zeros((1, q.shape[1]), dtype=np.uint32), q[:-1]], axis=0)
    return (q - previous) & mask


def modular_undelta(deltas: np.ndarray, bits: int) -> np.ndarray:
    d = np.asarray(deltas, dtype=np.uint32)
    if d.ndim != 2:
        raise ValueError("deltas must be rank two")
    mask = np.uint64((1 << bits) - 1)
    return (np.cumsum(d, axis=0, dtype=np.uint64) & mask).astype(np.uint32)


def bitpack(quantized: np.ndarray, bits: int) -> bytes:
    """Pack unsigned symbols densely, least-significant bit first, without dtype padding."""
    q = np.asarray(quantized, dtype=np.uint32).reshape(-1)
    if not 1 <= bits <= 24 or np.any(q >= (1 << bits)):
        raise ValueError("symbol does not fit requested bit depth")
    bit_positions = np.arange(bits, dtype=np.uint32)
    bit_rows = ((q[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8)
    return np.packbits(bit_rows.reshape(-1), bitorder="little").tobytes()


def bitunpack(payload: bytes, bits: int, count: int) -> np.ndarray:
    if not 1 <= bits <= 24 or count < 0:
        raise ValueError("invalid unpack parameters")
    required_bits = bits * count
    if len(payload) * 8 < required_bits:
        raise ValueError("packed payload is truncated")
    unpacked = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    rows = unpacked[:required_bits].reshape(count, bits).astype(np.uint32)
    weights = np.left_shift(np.uint32(1), np.arange(bits, dtype=np.uint32))
    return np.sum(rows * weights[None, :], axis=1, dtype=np.uint32)
