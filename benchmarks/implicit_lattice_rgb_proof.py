"""Target-free implicit RGB-lattice feasibility proof.

This diagnostic compares the production COMP-011 explicit flat/RVQ RGB
representations with a deliberately simple no-codebook control.  The control
uses either RGB or reversible YCoCg-R coordinates, scalar lattice steps, one
empirical arithmetic stream per coordinate, and an exact hypothetical 26-byte
descriptor.  It reports only symbol-domain SSE and representation-specific
bytes on deterministic synthetic fixtures.  It is not target-quality evidence
and it is not part of the frozen COMP-011 method menu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import tempfile
import time
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2v_container as container
from benchmarks import ssp2v_lloyd as lloyd


ROW_COUNT = 8192
TOTAL_FREQUENCY = 32768
FIXTURE_DOMAIN = b"structsplat-implicit-lattice-proof-v1\0"
LATTICE_DESCRIPTOR = struct.Struct("<4sBBH3H3h3H")
LATTICE_DIRECTORY_BYTES = 5 * 20


class LatticeProofError(RuntimeError):
    """The synthetic feasibility proof violated an exact invariant."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _digest_rows(kind: str) -> np.ndarray:
    if kind not in {"correlated", "palette", "uniform"}:
        raise LatticeProofError(f"unknown fixture {kind!r}")
    domain = FIXTURE_DOMAIN + kind.encode("ascii") + b"\0"
    payload = b"".join(
        hashlib.sha256(domain + struct.pack("<I", index)).digest() for index in range(ROW_COUNT)
    )
    return np.frombuffer(payload, dtype=np.uint8).reshape(ROW_COUNT, 32)


def inverse_ycocg(y: np.ndarray, co: np.ndarray, cg: np.ndarray) -> np.ndarray:
    """Apply the exact integer inverse used by the implicit-lattice control."""

    y32 = np.asarray(y, dtype=np.int32)
    co32 = np.asarray(co, dtype=np.int32)
    cg32 = np.asarray(cg, dtype=np.int32)
    if y32.shape != co32.shape or y32.shape != cg32.shape:
        raise LatticeProofError("YCoCg coordinate shapes differ")
    temporary = y32 - np.floor_divide(cg32, 2)
    green = cg32 + temporary
    blue = temporary - np.floor_divide(co32, 2)
    red = blue + co32
    return np.clip(np.stack((red, green, blue), axis=-1), 0, 255).astype(np.uint8)


def ycocg(rgb: np.ndarray) -> np.ndarray:
    """Apply the exact reversible integer YCoCg-R forward transform."""

    values = np.asarray(rgb)
    if values.shape != (ROW_COUNT, 3) or values.dtype != np.uint8:
        raise LatticeProofError("YCoCg input must be uint8[8192,3]")
    values32 = values.astype(np.int32)
    red, green, blue = (values32[:, index] for index in range(3))
    co = red - blue
    temporary = blue + np.floor_divide(co, 2)
    cg = green - temporary
    y = temporary + np.floor_divide(cg, 2)
    result = np.ascontiguousarray(np.stack((y, co, cg), axis=1))
    if not np.array_equal(inverse_ycocg(result[:, 0], result[:, 1], result[:, 2]), values):
        raise LatticeProofError("integer YCoCg transform is not reversible")
    return result


def synthetic_fixture(kind: str) -> np.ndarray:
    """Construct one deterministic target-free RGB fixture."""

    raw = _digest_rows(kind)
    if kind == "uniform":
        result = raw[:, :3].copy()
    elif kind == "correlated":
        y = raw[:, 0].astype(np.int32)
        co = raw[:, 1].astype(np.int32) % 41 - 20
        cg = raw[:, 2].astype(np.int32) % 41 - 20
        result = inverse_ycocg(y, co, cg)
    else:
        palette = raw[:64, :3].astype(np.int32)
        choices = raw[:, 3].astype(np.int32) % 64
        jitter = raw[:, 4:7].astype(np.int32) % 9 - 4
        result = np.clip(palette[choices] + jitter, 0, 255).astype(np.uint8)
    if result.shape != (ROW_COUNT, 3) or result.dtype != np.uint8:
        raise AssertionError("synthetic fixture construction drift")
    result.setflags(write=False)
    return result


def empirical_frequencies(values: np.ndarray, alphabet: int) -> np.ndarray:
    """Generalize the frozen positive-frequency rule to a lattice alphabet."""

    indices = np.asarray(values)
    if (
        indices.shape != (ROW_COUNT,)
        or indices.dtype.kind not in "iu"
        or alphabet < 2
        or alphabet >= TOTAL_FREQUENCY
        or (indices.dtype.kind == "i" and np.any(indices < 0))
        or int(indices.max(initial=0)) >= alphabet
    ):
        raise LatticeProofError("invalid implicit-lattice index stream")
    counts = np.bincount(indices.astype(np.int64), minlength=alphabet)
    residual = TOTAL_FREQUENCY - alphabet
    products = residual * counts
    allocation = products // ROW_COUNT
    remainders = products % ROW_COUNT
    remaining = residual - int(allocation.sum())
    order = sorted(range(alphabet), key=lambda index: (-int(remainders[index]), index))
    allocation[np.asarray(order[:remaining], dtype=np.int64)] += 1
    frequencies = np.ascontiguousarray(allocation + 1, dtype=np.uint32)
    if np.any(frequencies == 0) or int(frequencies.sum()) != TOTAL_FREQUENCY:
        raise LatticeProofError("implicit-lattice frequency normalization drift")
    return frequencies


def _cdf_rows(frequencies: np.ndarray) -> np.ndarray:
    cdf = np.concatenate((np.zeros(1, dtype=np.uint32), np.cumsum(frequencies, dtype=np.uint32)))
    return np.repeat(cdf[None, :], ROW_COUNT, axis=0)


def _round_nearest_tie_smaller(values: np.ndarray, step: int) -> np.ndarray:
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise LatticeProofError("lattice step must be a positive integer")
    quotient = np.floor_divide(values, step)
    remainder = values - quotient * step
    return quotient + (2 * remainder > step)


def _error_record(source: np.ndarray, reconstruction: np.ndarray) -> dict[str, Any]:
    difference = source.astype(np.int32) - reconstruction.astype(np.int32)
    squared = difference.astype(np.int64) ** 2
    mse = float(np.mean(squared, dtype=np.float64))
    return {
        "sse": int(np.sum(squared, dtype=np.int64)),
        "max_abs": int(np.abs(difference).max()),
        "psnr_rgb": 10.0 * math.log10(255.0**2 / mse) if mse else "infinity",
    }


def lattice_record(
    coder: arithmetic.NativeArithmeticCoder,
    rgb: np.ndarray,
    *,
    transform: str,
    steps: Sequence[int],
) -> dict[str, Any]:
    """Encode and score one exact implicit-lattice control point."""

    if transform not in {"rgb", "ycocg"} or len(steps) != 3:
        raise LatticeProofError("invalid lattice transform or step tuple")
    started = time.perf_counter_ns()
    normalized_steps = tuple(int(value) for value in steps)
    coefficients = ycocg(rgb) if transform == "ycocg" else rgb.astype(np.int32)
    quantized = np.stack(
        [
            _round_nearest_tie_smaller(coefficients[:, index], normalized_steps[index])
            for index in range(3)
        ],
        axis=1,
    )
    reconstructed_coefficients = np.stack(
        [quantized[:, index] * normalized_steps[index] for index in range(3)],
        axis=1,
    )
    reconstruction = (
        inverse_ycocg(
            reconstructed_coefficients[:, 0],
            reconstructed_coefficients[:, 1],
            reconstructed_coefficients[:, 2],
        )
        if transform == "ycocg"
        else np.clip(reconstructed_coefficients, 0, 255).astype(np.uint8)
    )
    offsets: list[int] = []
    alphabets: list[int] = []
    frequencies: list[np.ndarray] = []
    arithmetic_payloads: list[bytes] = []
    for index in range(3):
        offset = int(quantized[:, index].min())
        shifted = np.ascontiguousarray(quantized[:, index] - offset, dtype=np.uint32)
        # Keep the arithmetic grammar canonical even for a constant coordinate:
        # a two-symbol alphabet with one unused positive-frequency symbol.
        alphabet = max(2, int(shifted.max()) + 1)
        stage_frequencies = empirical_frequencies(shifted, alphabet)
        payload = coder.encode(shifted, _cdf_rows(stage_frequencies))
        if not np.array_equal(
            coder.decode(payload, _cdf_rows(stage_frequencies), ROW_COUNT), shifted
        ):
            raise LatticeProofError("implicit-lattice arithmetic round trip drift")
        offsets.append(offset)
        alphabets.append(alphabet)
        frequencies.append(stage_frequencies)
        arithmetic_payloads.append(payload)
    descriptor = LATTICE_DESCRIPTOR.pack(
        b"ILQ1",
        1,
        0 if transform == "rgb" else 1,
        0,
        *normalized_steps,
        *offsets,
        *alphabets,
    )
    model_raw = b"".join(
        value.astype("<u2", copy=False).tobytes(order="C") for value in frequencies
    )
    model = zlib.compress(model_raw, level=9)
    descriptor_bytes = len(descriptor)
    model_bytes = len(model)
    arithmetic_bytes = sum(len(value) for value in arithmetic_payloads)
    return {
        "method": f"implicit_{transform}",
        "steps": list(normalized_steps),
        "alphabets": alphabets,
        "offsets": offsets,
        "representation_bytes": LATTICE_DIRECTORY_BYTES
        + descriptor_bytes
        + model_bytes
        + arithmetic_bytes,
        "directory_bytes": LATTICE_DIRECTORY_BYTES,
        "descriptor_bytes": descriptor_bytes,
        "codebook_bytes": 0,
        "model_bytes": model_bytes,
        "arithmetic_bytes": arithmetic_bytes,
        "encode_wall_ns": time.perf_counter_ns() - started,
        **_error_record(rgb, reconstruction),
    }


def vq_records(
    coder: arithmetic.NativeArithmeticCoder,
    fitter: lloyd.NativeLloydFitter,
    rgb: np.ndarray,
) -> list[dict[str, Any]]:
    """Fit and score the frozen eight-arm COMP-011 RGB representation menu."""

    records: list[dict[str, Any]] = []
    for family in ("flat", "rvq"):
        for matched_stages in (2, 3):
            for base_k in (64, 128):
                started = time.perf_counter_ns()
                fit = (
                    lloyd.fit_flat(rgb, matched_stages, base_k, fitter=fitter)
                    if family == "flat"
                    else lloyd.fit_rvq(rgb, matched_stages, base_k, fitter=fitter)
                )
                actual_stages = len(fit.stages)
                entries = (
                    (2 * matched_stages - 1) * base_k
                    if family == "flat"
                    else matched_stages * base_k
                )
                descriptor = container.SSP2VDescriptor(
                    family_id=0 if family == "flat" else 1,
                    actual_stage_count=actual_stages,
                    matched_rvq_stage_count=matched_stages,
                    base_k=base_k,
                    codebook_entries=entries,
                )
                codebooks = tuple(stage.codebook for stage in fit.stages)
                codebook = container.serialize_codebooks(codebooks, descriptor)
                model = container.serialize_index_model(fit.stage_indices, descriptor)
                arithmetic_payloads: list[bytes] = []
                for indices, alphabet in zip(
                    fit.stage_indices, descriptor.stage_alphabets, strict=True
                ):
                    frequencies = container.empirical_frequencies(indices, alphabet)
                    rows = container.cdf_rows(frequencies)
                    payload = coder.encode(indices, rows)
                    if not np.array_equal(coder.decode(payload, rows, ROW_COUNT), indices):
                        raise LatticeProofError("VQ arithmetic round trip drift")
                    arithmetic_payloads.append(payload)
                directory_bytes = 20 * (3 + actual_stages)
                descriptor_bytes = len(descriptor.to_bytes())
                arithmetic_bytes = sum(len(value) for value in arithmetic_payloads)
                records.append(
                    {
                        "method": family,
                        "matched_stages": matched_stages,
                        "base_k": base_k,
                        "representation_bytes": directory_bytes
                        + descriptor_bytes
                        + len(codebook)
                        + len(model)
                        + arithmetic_bytes,
                        "directory_bytes": directory_bytes,
                        "descriptor_bytes": descriptor_bytes,
                        "codebook_bytes": len(codebook),
                        "model_bytes": len(model),
                        "arithmetic_bytes": arithmetic_bytes,
                        "fit_wall_ns": time.perf_counter_ns() - started,
                        **_error_record(rgb, fit.reconstructed_rgb),
                    }
                )
    return records


def _pareto_frontier(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    best_sse: int | None = None
    for value in sorted(
        records,
        key=lambda record: (
            int(record["representation_bytes"]),
            int(record["sse"]),
            str(record["method"]),
        ),
    ):
        sse = int(value["sse"])
        if best_sse is None or sse < best_sse:
            frontier.append(dict(value))
            best_sse = sse
    return frontier


def _deterministic_projection(value: Any) -> Any:
    """Remove explicitly nongating wall-clock diagnostics before evidence sealing."""

    if isinstance(value, Mapping):
        return {
            str(key): _deterministic_projection(item)
            for key, item in value.items()
            if not (isinstance(key, str) and (key == "wall_ns" or key.endswith("_wall_ns")))
        }
    if isinstance(value, list):
        return [_deterministic_projection(item) for item in value]
    return value


def run_proof(fixtures: Sequence[str]) -> dict[str, Any]:
    """Build native authorities and run the requested deterministic fixtures."""

    with tempfile.TemporaryDirectory(prefix="structsplat-lattice-proof-") as temporary:
        root = Path(temporary)
        arithmetic_path = arithmetic.build_native(root / "libssp2e_range_v1.so")
        lloyd_path = lloyd.build_native(root / "libssp2v_lloyd_v1.so")
        coder = arithmetic.NativeArithmeticCoder(arithmetic_path)
        fitter = lloyd.NativeLloydFitter(lloyd_path)
        results: dict[str, Any] = {}
        for kind in fixtures:
            rgb = synthetic_fixture(kind)
            started = time.perf_counter_ns()
            explicit = vq_records(coder, fitter, rgb)
            implicit = []
            for step in (1, 2, 4, 8, 16, 32):
                implicit.append(
                    lattice_record(coder, rgb, transform="rgb", steps=(step, step, step))
                )
                implicit.append(
                    lattice_record(
                        coder,
                        rgb,
                        transform="ycocg",
                        steps=(step, 2 * step, 2 * step),
                    )
                )
            results[kind] = {
                "rgb_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
                "unique_rgb": int(len(np.unique(rgb, axis=0))),
                "explicit_vq": explicit,
                "implicit_lattice": implicit,
                "joint_frontier": _pareto_frontier([*explicit, *implicit]),
                "wall_ns": time.perf_counter_ns() - started,
            }
    record = {
        "schema": "structsplat.implicit-lattice-rgb-proof.v1",
        "claim_scope": "target-free synthetic symbol-rate mechanics only",
        "row_count": ROW_COUNT,
        "fixtures": results,
    }
    record["deterministic_evidence_sha256"] = hashlib.sha256(
        _canonical_json(_deterministic_projection(record))
    ).hexdigest()
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        action="append",
        choices=("correlated", "palette", "uniform"),
        dest="fixtures",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixtures = tuple(args.fixtures or ("correlated", "palette"))
    print(_canonical_json(run_proof(fixtures)).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LatticeProofError",
    "empirical_frequencies",
    "inverse_ycocg",
    "lattice_record",
    "run_proof",
    "synthetic_fixture",
    "vq_records",
    "ycocg",
]
