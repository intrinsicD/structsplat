"""Source-bound normalized-renderer operator for COMP-012 RGB coordinate RDO.

The older :mod:`benchmarks.renderer_gram_structsplat_bridge` is deliberately
synthetic-only.  This module upgrades that mechanics proof by binding its sparse
operator to one exact SSPL1 byte string, the strict absolute-symbol parse, the
independently reconstructed float32 codec boundary, and the production
``GaussianField`` geometry calls.

No target enters operator construction.  Target or source objectives are made
only after the immutable operator and its source binding have been sealed.
Production use requires an 8192-row source and a fresh deterministic CPU worker;
small inputs are accepted only behind the explicit ``synthetic=True`` marker.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import torch

from benchmarks import renderer_gram_structsplat_bridge as gram_bridge
from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_execute as execute
from benchmarks import ssp2f_rdo_objective as objective
from structsplat.gaussians import GaussianField
from structsplat.render import render_field


OPERATOR_BINDING_SCHEMA = "structsplat.comp012.ssp2f-rdo-operator-binding.v1"
SYNTHETIC_GATE_SCHEMA = "structsplat.comp012.ssp2f-rdo-operator-synthetic-gate.v1"
CONSTRUCTION_SCHEMA = "sspl1-f32-gaussianfield-normalized-sparse-columns-v1"
TASK_SHA256 = "335366dc0e925de2b18f2a131e3ad902b8f2e3a13fbb4e23d68cc5c4327bd5f7"
PRODUCTION_ROW_COUNT = 8192
RENDER_EPS = 1e-8
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CPU_COLUMN_DELTA_ATOL = 2.0 * gram_bridge.CPU_FORWARD_ATOL
CPU_COLUMN_DELTA_RTOL = 2.0 * gram_bridge.CPU_FORWARD_RTOL
SYNTHETIC_OBJECTIVE_DELTA_ATOL = 2.5e-5
SYNTHETIC_OBJECTIVE_DELTA_RTOL = 2.0e-5

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]


class SSP2FRDOOperatorError(ValueError):
    """The source, deterministic runtime, or operator binding is invalid."""


class SSP2FRDOOperatorParityError(RuntimeError):
    """A synthetic production-path parity gate failed."""

    def __init__(self, report: SyntheticOperatorParity) -> None:
        super().__init__(
            "source-bound sparse operator failed synthetic CPU parity: "
            f"forward={report.forward_max_abs_error:.9g}, "
            f"column_delta={report.column_delta_max_abs_error:.9g}, "
            f"objective_delta={report.objective_delta_abs_error:.9g}"
        )
        self.report = report


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SSP2FRDOOperatorError(f"binding record is not finite canonical JSON: {exc}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    payload = array.tobytes(order="C")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def _symbol_subset_sha256(
    symbols: Mapping[str, np.ndarray],
    names: tuple[str, ...],
) -> str:
    digest = hashlib.sha256(b"structsplat.comp012.symbol-subset.v1\x00")
    for name in names:
        if name not in symbols:
            raise SSP2FRDOOperatorError(f"decoded symbols omit {name}")
        value = np.ascontiguousarray(symbols[name], dtype="<u4")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(int(value.size).to_bytes(8, "little", signed=False))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _support_sha256(operator_value: objective.FrozenNormalizedOperator) -> str:
    digest = hashlib.sha256(b"structsplat.comp012.normalized-support.v1\x00")
    digest.update(operator_value.pixel_count.to_bytes(8, "little", signed=False))
    digest.update(operator_value.column_count.to_bytes(8, "little", signed=False))
    for column in range(operator_value.column_count):
        indices = operator_value.footprint(column).pixel_indices.astype("<i8", copy=False)
        digest.update(column.to_bytes(8, "little", signed=False))
        digest.update(indices.size.to_bytes(8, "little", signed=False))
        digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


def _require_deterministic_cpu() -> dict[str, Any]:
    environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if any(value != "1" for value in environment.values()):
        raise SSP2FRDOOperatorError(
            "operator worker requires "
            "OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS="
            "NUMEXPR_NUM_THREADS=1"
        )

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise SSP2FRDOOperatorError(
                "operator construction requires a fresh worker whose Torch inter-op "
                "thread count can be set to one"
            ) from exc
    torch.use_deterministic_algorithms(True)
    if (
        torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or not torch.are_deterministic_algorithms_enabled()
    ):
        raise SSP2FRDOOperatorError("deterministic Torch CPU state did not take effect")
    return {
        "environment": environment,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "deterministic_algorithms": True,
        "device": "cpu",
    }


def _validate_supplied_parts(
    supplied: container.SSPL1Parts,
    exact: container.SSPL1Parts,
) -> None:
    if not isinstance(supplied, container.SSPL1Parts):
        raise SSP2FRDOOperatorError("cold_parts must be an SSPL1Parts instance")
    if (
        supplied.header != exact.header
        or supplied.original_header != exact.original_header
        or supplied.blob_sha256 != exact.blob_sha256
        or supplied.symbol_sha256 != exact.symbol_sha256
        or supplied.total_bytes != exact.total_bytes
        or supplied.payloads != exact.payloads
        or set(supplied.symbols) != set(exact.symbols)
    ):
        raise SSP2FRDOOperatorError("supplied cold parts differ from the exact SSPL1 bytes")
    for name in container.SYMBOL_CHANNELS:
        supplied_value = np.asarray(supplied.symbols[name])
        exact_value = exact.symbols[name]
        if (
            supplied_value.dtype != exact_value.dtype
            or supplied_value.shape != exact_value.shape
            or not np.array_equal(supplied_value, exact_value)
        ):
            raise SSP2FRDOOperatorError(
                f"supplied cold parts differ from exact decoded symbol {name}"
            )


def _boundary_from_parts(parts: container.SSPL1Parts) -> dict[str, np.ndarray]:
    try:
        boundary = execute.boundary_arrays(parts.header, parts.symbols)
    except (execute.ExecutionError, TypeError, ValueError) as exc:
        raise SSP2FRDOOperatorError(f"exact float32 boundary reconstruction failed: {exc}") from exc
    expected_shapes = {
        "means": (parts.header.n, 2),
        "log_scales": (parts.header.n, 2),
        "rotations": (parts.header.n,),
        "colors": (parts.header.n, 3),
    }
    if set(boundary) != set(expected_shapes):
        raise SSP2FRDOOperatorError("decoded boundary component set is not canonical")
    normalized: dict[str, np.ndarray] = {}
    for name, shape in expected_shapes.items():
        value = np.asarray(boundary[name])
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise SSP2FRDOOperatorError(
                f"decoded boundary {name} must be finite float32 with shape {shape}"
            )
        normalized[name] = _readonly(value, np.float32)
    return normalized


def _source_rgb(parts: container.SSPL1Parts) -> UInt8Array:
    if parts.header.bits[3] != 8:
        raise SSP2FRDOOperatorError("COMP-012 RGB state requires exactly eight color bits")
    values = np.column_stack(
        (parts.symbols["red"], parts.symbols["green"], parts.symbols["blue"])
    )
    if values.shape != (parts.header.n, 3) or np.any(values > 255):
        raise SSP2FRDOOperatorError("decoded source RGB symbols are outside uint8")
    return _readonly(values, np.uint8)


def _field_from_boundary(
    boundary: Mapping[str, np.ndarray],
    colors: np.ndarray | None = None,
) -> GaussianField:
    selected_colors = boundary["colors"] if colors is None else colors

    def tensor(value: np.ndarray) -> torch.Tensor:
        writable = np.array(value, dtype=np.float32, order="C", copy=True)
        return torch.from_numpy(writable).clone()

    values = {
        "means": tensor(boundary["means"]),
        "log_scales": tensor(boundary["log_scales"]),
        "rotations": tensor(boundary["rotations"]),
        "colors": tensor(selected_colors),
    }
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise SSP2FRDOOperatorError("GaussianField boundary contains a non-finite tensor")
    return GaussianField(
        values["means"],
        values["log_scales"],
        values["rotations"],
        values["colors"],
        opacities=None,
    )


def _parts_and_boundary(
    sspl1_blob: bytes,
    cold_parts: container.SSPL1Parts | None,
) -> tuple[container.SSPL1Parts, dict[str, np.ndarray]]:
    if not isinstance(sspl1_blob, bytes):
        raise SSP2FRDOOperatorError("SSPL1 source must be immutable bytes")
    try:
        exact = container.extract_sspl1_parts(sspl1_blob)
    except container.ContainerError as exc:
        raise SSP2FRDOOperatorError(f"strict SSPL1 extraction failed: {exc}") from exc
    if cold_parts is not None:
        _validate_supplied_parts(cold_parts, exact)
    return exact, _boundary_from_parts(exact)


@dataclass(frozen=True)
class BoundSSP2FRDOOperator:
    """Immutable objective operator plus detached source/header binding."""

    operator: objective.FrozenNormalizedOperator
    header: container.ContainerHeader
    _source_rgb: UInt8Array
    _boundary: tuple[tuple[str, np.ndarray], ...]
    _conics: Float32Array
    _radii: Int64Array
    _denominator: Float32Array
    _source_reference: Float64Array
    _binding_json: bytes

    @property
    def source_rgb_symbols(self) -> UInt8Array:
        return self._source_rgb.copy()

    @property
    def boundary_arrays(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._boundary}

    @property
    def conics(self) -> Float32Array:
        return self._conics.copy()

    @property
    def radii(self) -> Int64Array:
        return self._radii.copy()

    @property
    def denominator(self) -> Float32Array:
        return self._denominator.copy()

    @property
    def source_reference(self) -> Float64Array:
        return self._source_reference.copy()

    @property
    def binding_record(self) -> dict[str, Any]:
        result = json.loads(self._binding_json)
        if not isinstance(result, dict):  # pragma: no cover - builder seals an object
            raise SSP2FRDOOperatorError("internal binding JSON is not an object")
        verify_binding_record(result)
        return result

    @property
    def binding_sha256(self) -> str:
        return str(self.binding_record["binding_sha256"])

    @property
    def synthetic(self) -> bool:
        return bool(self.binding_record["synthetic"])

    def source_objective(
        self,
        rgb_symbols: np.ndarray | None = None,
    ) -> objective.SSP2FRGBObjective:
        """Build ``D_source`` without introducing a target input."""

        _require_deterministic_cpu()
        initial = self._source_rgb if rgb_symbols is None else rgb_symbols
        return objective.SSP2FRGBObjective(
            self.operator,
            np.asarray(initial),
            self._source_reference,
            reference_kind="source",
            color_lo=self.header.color_lo,
            color_hi=self.header.color_hi,
            bits_color=self.header.bits[3],
        )

    def target_objective(
        self,
        target_rgb: np.ndarray,
        rgb_symbols: np.ndarray | None = None,
    ) -> objective.SSP2FRGBObjective:
        """Attach a target only after the target-independent operator is sealed."""

        _require_deterministic_cpu()
        if (
            not isinstance(target_rgb, np.ndarray)
            or target_rgb.dtype != np.uint8
            or target_rgb.shape != (self.header.height, self.header.width, 3)
        ):
            raise SSP2FRDOOperatorError(
                "target must be uint8 with the exact header shape [H,W,3]"
            )
        initial = self._source_rgb if rgb_symbols is None else rgb_symbols
        return objective.SSP2FRGBObjective(
            self.operator,
            np.asarray(initial),
            target_rgb,
            reference_kind="target",
            color_lo=self.header.color_lo,
            color_hi=self.header.color_hi,
            bits_color=self.header.bits[3],
        )


def verify_binding_record(record: Mapping[str, Any]) -> None:
    """Fail closed unless a detached binding record has its exact canonical seal."""

    if not isinstance(record, Mapping):
        raise SSP2FRDOOperatorError("operator binding record must be a mapping")
    observed = record.get("binding_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise SSP2FRDOOperatorError("operator binding record lacks binding_sha256")
    core = dict(record)
    core.pop("binding_sha256", None)
    if core.get("schema") != OPERATOR_BINDING_SCHEMA:
        raise SSP2FRDOOperatorError("operator binding record schema differs")
    if core.get("task_sha256") != TASK_SHA256:
        raise SSP2FRDOOperatorError("operator binding record task SHA-256 differs")
    expected = _sha256(_canonical_json(core))
    if observed != expected:
        raise SSP2FRDOOperatorError("operator binding record SHA-256 mismatch")


def build_source_bound_operator(
    sspl1_blob: bytes,
    *,
    cold_parts: container.SSPL1Parts | None = None,
    synthetic: bool = False,
) -> BoundSSP2FRDOOperator:
    """Construct and seal target-independent ``A`` from one exact SSPL1 source."""

    if not isinstance(synthetic, bool):
        raise SSP2FRDOOperatorError("synthetic must be exactly boolean")
    if not synthetic and cold_parts is None:
        raise SSP2FRDOOperatorError(
            "production operator requires caller-supplied exact cold-decoded SSPL1 parts"
        )
    thread_state = _require_deterministic_cpu()
    parts, boundary = _parts_and_boundary(sspl1_blob, cold_parts)
    header = parts.header.validated()
    if not synthetic and header.n != PRODUCTION_ROW_COUNT:
        raise SSP2FRDOOperatorError(
            f"production operator requires N={PRODUCTION_ROW_COUNT}, found {header.n}"
        )
    if header.support_fade != 0:
        raise SSP2FRDOOperatorError("COMP-012 requires support_fade=false")
    if not math.isfinite(header.aa_dilation) or header.aa_dilation < 0.0:
        raise SSP2FRDOOperatorError("aa_dilation must be finite and nonnegative")
    if not math.isfinite(header.sigma_cutoff) or header.sigma_cutoff <= 0.0:
        raise SSP2FRDOOperatorError("sigma_cutoff must be finite and positive")
    if any(
        upper <= lower
        for lower, upper in zip(header.scale_lo, header.scale_hi, strict=True)
    ):
        raise SSP2FRDOOperatorError("every scale_hi must be greater than scale_lo")
    if any(
        upper <= lower
        for lower, upper in zip(header.color_lo, header.color_hi, strict=True)
    ):
        raise SSP2FRDOOperatorError("every color_hi must be greater than color_lo")

    source_rgb = _source_rgb(parts)
    decoded_table = objective.affine_color_table(
        header.color_lo,
        header.color_hi,
        bits_color=header.bits[3],
    )
    decoded_source_colors = decoded_table[source_rgb, np.arange(3)[None, :]].astype(np.float32)
    if not np.array_equal(decoded_source_colors, boundary["colors"]):
        raise SSP2FRDOOperatorError(
            "source RGB symbols do not reproduce the exact decoded color boundary"
        )

    field = _field_from_boundary(boundary)
    if (
        field.opacities is not None
        or field.color_grads is not None
        or field.filter_variance is not None
    ):
        raise SSP2FRDOOperatorError("COMP-012 GaussianField must have no opacity or extensions")
    with torch.inference_mode():
        conics_tensor = field.conics(float(header.aa_dilation))
        radii_tensor = field.radii(
            float(header.sigma_cutoff),
            float(header.aa_dilation),
        )
    conics = np.ascontiguousarray(conics_tensor.cpu().numpy(), dtype=np.float32)
    radii = np.ascontiguousarray(radii_tensor.cpu().numpy(), dtype=np.int64)
    if (
        conics.shape != (header.n, 3)
        or not np.isfinite(conics).all()
        or radii.shape != (header.n, 2)
        or np.any(radii < 1)
    ):
        raise SSP2FRDOOperatorError("production conics/radii construction is malformed")

    try:
        bridge = gram_bridge.build_normalized_renderer_bridge(
            field.means,
            conics_tensor,
            radii_tensor,
            None,
            header.height,
            header.width,
            eps=RENDER_EPS,
            support_fade=False,
            sigma_cutoff=float(header.sigma_cutoff),
        )
        frozen = objective.FrozenNormalizedOperator.from_operator(bridge.linear_map)
    except (ValueError, objective.SSP2FRDOObjectiveError) as exc:
        raise SSP2FRDOOperatorError(f"normalized sparse operator construction failed: {exc}") from exc
    if (
        bridge.eps != RENDER_EPS
        or bridge.support_fade
        or bridge.opacities is not None
        or bridge.height != header.height
        or bridge.width != header.width
        or frozen.pixel_count != header.height * header.width
        or frozen.column_count != header.n
    ):
        raise SSP2FRDOOperatorError("sparse bridge differs from the frozen COMP-012 parameters")
    for column in range(frozen.column_count):
        footprint = frozen.footprint(column)
        if footprint.pixel_indices.size and np.any(
            footprint.pixel_indices[1:] <= footprint.pixel_indices[:-1]
        ):
            raise SSP2FRDOOperatorError("generated per-column pixel order was not preserved")

    denominator = np.ascontiguousarray(bridge.denominator, dtype=np.float32)
    if (
        denominator.shape != (header.height * header.width,)
        or not np.isfinite(denominator).all()
        or np.any(denominator < 0.0)
    ):
        raise SSP2FRDOOperatorError("float32 normalized-render denominator is malformed")
    source_colors64 = np.ascontiguousarray(
        decoded_table[source_rgb, np.arange(3)[None, :]],
        dtype=np.float64,
    )
    source_unclamped = frozen.apply(source_colors64).reshape(
        header.height,
        header.width,
        3,
    )
    source_render = np.clip(source_unclamped, 0.0, 1.0)
    boundary_state_sha256 = execute._array_state_digest(boundary)  # noqa: SLF001
    boundary_records = {name: _array_record(boundary[name]) for name in sorted(boundary)}
    geometry_records = {
        name: boundary_records[name] for name in ("log_scales", "means", "rotations")
    }
    geometry_state_sha256 = _sha256(_canonical_json(geometry_records))
    record: dict[str, Any] = {
        "schema": OPERATOR_BINDING_SCHEMA,
        "task_sha256": TASK_SHA256,
        "construction_schema": CONSTRUCTION_SCHEMA,
        "synthetic": synthetic,
        "authority_scope": (
            "synthetic-mechanics-only" if synthetic else "source-bound-production-operator"
        ),
        "target_input_surface": "absent",
        "target_accessed": False,
        "source": {
            "magic": "SSPL1",
            "bytes": len(sspl1_blob),
            "blob_sha256": parts.blob_sha256,
            "symbol_sha256": parts.symbol_sha256,
            "common_symbol_sha256": _symbol_subset_sha256(
                parts.symbols,
                ("mean_x", "mean_y", "scale_x", "scale_y", "rotation"),
            ),
            "rgb_symbol_record": _array_record(source_rgb),
            "cold_parts_cross_checked": cold_parts is not None,
            "strict_complete_parse": True,
        },
        "header": {
            "bytes": container.HEADER.size,
            "sha256": _sha256(header.to_bytes()),
            "n": header.n,
            "height": header.height,
            "width": header.width,
            "bits": list(header.bits),
            "aa_dilation_hex": float(header.aa_dilation).hex(),
            "sigma_cutoff_hex": float(header.sigma_cutoff).hex(),
            "render_chunk": header.render_chunk,
            "renderer_id": header.renderer_id,
            "support_fade": False,
        },
        "boundary": {
            "state_sha256": boundary_state_sha256,
            "geometry_state_sha256": geometry_state_sha256,
            "components": boundary_records,
        },
        "construction": {
            "field": "GaussianField(means,decoded_log_scales,rotations,colors,opacities=None)",
            "conics_call": "field.conics(header.aa_dilation)",
            "radii_call": "field.radii(header.sigma_cutoff,header.aa_dilation)",
            "conics": _array_record(conics),
            "radii": _array_record(radii),
            "denominator_float32": _array_record(denominator),
            "support_fade": False,
            "opacities": "absent",
            "eps_hex": float(RENDER_EPS).hex(),
            "normalized_zero_drop": "exact-float32-zero-only",
            "retained_weight_storage": "single-float32-to-float64-promotion",
            "column_order": "decoded-gaussian-row-order",
            "pixel_order": "generated-row-major-support-order",
            "thread_state": thread_state,
        },
        "operator": {
            "sha256": frozen.sha256,
            "support_sha256": _support_sha256(frozen),
            "pixel_count": frozen.pixel_count,
            "column_count": frozen.column_count,
            "raw_support_elements": bridge.raw_support_elements,
            "nonzero_elements": bridge.nonzero_elements,
            "apply_order": "float64-zero-then-columns-ascending",
            "source_unclamped_render": _array_record(source_unclamped),
            "source_clamped_render": _array_record(source_render),
        },
        "objective": {
            "ready": True,
            "source_rgb_aligned": True,
            "source_reference": _array_record(source_render),
            "color_lo_hex": [float(value).hex() for value in header.color_lo],
            "color_hi_hex": [float(value).hex() for value in header.color_hi],
            "bits_color": header.bits[3],
            "arithmetic_schema": objective.OBJECTIVE_ARITHMETIC_SCHEMA,
            "block_size": objective.OBJECTIVE_BLOCK_SIZE,
        },
    }
    record["binding_sha256"] = _sha256(_canonical_json(record))
    verify_binding_record(record)
    binding_json = _canonical_json(record)
    return BoundSSP2FRDOOperator(
        operator=frozen,
        header=header,
        _source_rgb=_readonly(source_rgb, np.uint8),
        _boundary=tuple(
            (name, _readonly(boundary[name], np.float32)) for name in sorted(boundary)
        ),
        _conics=_readonly(conics, np.float32),
        _radii=_readonly(radii, np.int64),
        _denominator=_readonly(denominator, np.float32),
        _source_reference=_readonly(source_render, np.float64),
        _binding_json=binding_json,
    )


def _render_cpu(bound: BoundSSP2FRDOOperator, rgb_symbols: UInt8Array) -> Float32Array:
    colors64 = objective.affine_color_table(
        bound.header.color_lo,
        bound.header.color_hi,
        bits_color=bound.header.bits[3],
    )[rgb_symbols, np.arange(3)[None, :]]
    colors = np.ascontiguousarray(colors64, dtype=np.float32)
    field = _field_from_boundary(bound.boundary_arrays, colors)
    with torch.inference_mode():
        rendered = render_field(
            field.means,
            field.conics(float(bound.header.aa_dilation)),
            field.colors,
            field.radii(
                float(bound.header.sigma_cutoff),
                float(bound.header.aa_dilation),
            ),
            bound.header.height,
            bound.header.width,
            bound.header.render_chunk,
            "normalized",
            opacities=None,
            scales=field.effective_scales(float(bound.header.aa_dilation)),
            rotations=field.rotations,
            support_fade=False,
            sigma_cutoff=float(bound.header.sigma_cutoff),
        )
    value = rendered.detach().cpu().numpy()
    if (
        value.dtype != np.float32
        or value.shape != (bound.header.height, bound.header.width, 3)
        or not np.isfinite(value).all()
    ):
        raise SSP2FRDOOperatorError("repository CPU renderer returned a malformed image")
    return np.ascontiguousarray(value)


def _block_sse(residual: Float64Array) -> float:
    squared = np.ascontiguousarray(residual * residual, dtype=np.float64).reshape(-1)
    if squared.size == 0 or not np.isfinite(squared).all() or np.any(squared < 0.0):
        raise SSP2FRDOOperatorError("synthetic parity residual is invalid")
    blocks = np.empty(
        (squared.size + objective.OBJECTIVE_BLOCK_SIZE - 1)
        // objective.OBJECTIVE_BLOCK_SIZE,
        dtype=np.float64,
    )
    for index in range(blocks.size):
        start = index * objective.OBJECTIVE_BLOCK_SIZE
        stop = min(start + objective.OBJECTIVE_BLOCK_SIZE, squared.size)
        blocks[index] = np.sum(squared[start:stop], dtype=np.float64)
    return float(np.sum(blocks, dtype=np.float64))


@dataclass(frozen=True)
class SyntheticOperatorParity:
    """Mechanics-only CPU forward, scalar-column, and clamped-SSE gate."""

    schema: str
    binding_sha256: str
    row: int
    channel: int
    old_symbol: int
    new_symbol: int
    forward_max_abs_error: float
    forward_atol: float
    forward_rtol: float
    forward_passed: bool
    column_delta_max_abs_error: float
    column_delta_atol: float
    column_delta_rtol: float
    column_delta_passed: bool
    objective_operator_delta: float
    objective_renderer_delta: float
    objective_delta_abs_error: float
    objective_delta_tolerance: float
    objective_delta_passed: bool
    passed: bool


def validate_synthetic_cpu_parity(
    bound: BoundSSP2FRDOOperator,
    *,
    row: int,
    channel: int,
    new_symbol: int,
) -> SyntheticOperatorParity:
    """Gate one synthetic source and legal scalar edit against the CPU renderer."""

    if not isinstance(bound, BoundSSP2FRDOOperator) or not bound.synthetic:
        raise SSP2FRDOOperatorError("synthetic CPU parity requires a synthetic bound operator")
    _require_deterministic_cpu()
    if (
        isinstance(row, (bool, np.bool_))
        or not isinstance(row, (int, np.integer))
        or not 0 <= int(row) < bound.header.n
    ):
        raise SSP2FRDOOperatorError("synthetic gate row is out of range")
    if (
        isinstance(channel, (bool, np.bool_))
        or not isinstance(channel, (int, np.integer))
        or not 0 <= int(channel) < 3
    ):
        raise SSP2FRDOOperatorError("synthetic gate channel is out of range")
    if (
        isinstance(new_symbol, (bool, np.bool_))
        or not isinstance(new_symbol, (int, np.integer))
        or not 0 <= int(new_symbol) <= 255
    ):
        raise SSP2FRDOOperatorError("synthetic gate symbol is outside uint8")

    row_index = int(row)
    channel_index = int(channel)
    replacement = int(new_symbol)
    source_rgb = bound.source_rgb_symbols
    old_symbol = int(source_rgb[row_index, channel_index])
    if replacement == old_symbol:
        raise SSP2FRDOOperatorError("synthetic gate scalar edit must change the source symbol")
    candidate_rgb = source_rgb.copy()
    candidate_rgb[row_index, channel_index] = replacement
    table = objective.affine_color_table(
        bound.header.color_lo,
        bound.header.color_hi,
        bits_color=bound.header.bits[3],
    )
    source_colors = np.ascontiguousarray(
        table[source_rgb, np.arange(3)[None, :]],
        dtype=np.float64,
    )
    candidate_colors = np.ascontiguousarray(
        table[candidate_rgb, np.arange(3)[None, :]],
        dtype=np.float64,
    )
    operator_source = bound.operator.apply(source_colors)
    renderer_source = _render_cpu(bound, source_rgb).reshape(-1, 3).astype(np.float64)
    forward_difference = np.abs(operator_source - renderer_source)
    forward_passed = bool(
        np.allclose(
            operator_source,
            renderer_source,
            atol=gram_bridge.CPU_FORWARD_ATOL,
            rtol=gram_bridge.CPU_FORWARD_RTOL,
        )
    )

    color_delta = float(
        candidate_colors[row_index, channel_index]
        - source_colors[row_index, channel_index]
    )
    expected_column_delta = np.zeros_like(operator_source)
    bound.operator.add_scaled_scalar_column(
        expected_column_delta,
        row_index,
        channel_index,
        color_delta,
    )
    operator_candidate = bound.operator.apply(candidate_colors)
    internal_column_error = float(
        np.max(
            np.abs(
                (operator_candidate - operator_source)
                - expected_column_delta
            ),
            initial=0.0,
        )
    )
    if internal_column_error > 4.0 * np.finfo(np.float64).eps:
        raise SSP2FRDOOperatorError(
            "frozen operator scalar-column action disagrees with full apply"
        )
    renderer_candidate = _render_cpu(bound, candidate_rgb).reshape(-1, 3).astype(np.float64)
    renderer_delta = renderer_candidate - renderer_source
    column_difference = np.abs(expected_column_delta - renderer_delta)
    column_passed = bool(
        np.allclose(
            expected_column_delta,
            renderer_delta,
            atol=CPU_COLUMN_DELTA_ATOL,
            rtol=CPU_COLUMN_DELTA_RTOL,
        )
    )

    operator_objective_delta = _block_sse(
        np.clip(operator_candidate, 0.0, 1.0)
        - np.clip(operator_source, 0.0, 1.0)
    )
    renderer_objective_delta = _block_sse(
        np.clip(renderer_candidate, 0.0, 1.0)
        - np.clip(renderer_source, 0.0, 1.0)
    )
    objective_error = abs(operator_objective_delta - renderer_objective_delta)
    objective_tolerance = float(
        SYNTHETIC_OBJECTIVE_DELTA_ATOL
        + SYNTHETIC_OBJECTIVE_DELTA_RTOL
        * max(abs(operator_objective_delta), abs(renderer_objective_delta))
    )
    objective_passed = objective_error <= objective_tolerance
    passed = forward_passed and column_passed and objective_passed
    report = SyntheticOperatorParity(
        schema=SYNTHETIC_GATE_SCHEMA,
        binding_sha256=bound.binding_sha256,
        row=row_index,
        channel=channel_index,
        old_symbol=old_symbol,
        new_symbol=replacement,
        forward_max_abs_error=float(np.max(forward_difference, initial=0.0)),
        forward_atol=gram_bridge.CPU_FORWARD_ATOL,
        forward_rtol=gram_bridge.CPU_FORWARD_RTOL,
        forward_passed=forward_passed,
        column_delta_max_abs_error=float(np.max(column_difference, initial=0.0)),
        column_delta_atol=CPU_COLUMN_DELTA_ATOL,
        column_delta_rtol=CPU_COLUMN_DELTA_RTOL,
        column_delta_passed=column_passed,
        objective_operator_delta=operator_objective_delta,
        objective_renderer_delta=renderer_objective_delta,
        objective_delta_abs_error=objective_error,
        objective_delta_tolerance=objective_tolerance,
        objective_delta_passed=objective_passed,
        passed=passed,
    )
    if not passed:
        raise SSP2FRDOOperatorParityError(report)
    return report


__all__ = [
    "BoundSSP2FRDOOperator",
    "CONSTRUCTION_SCHEMA",
    "OPERATOR_BINDING_SCHEMA",
    "SSP2FRDOOperatorError",
    "SSP2FRDOOperatorParityError",
    "SYNTHETIC_GATE_SCHEMA",
    "TASK_SHA256",
    "SyntheticOperatorParity",
    "build_source_bound_operator",
    "validate_synthetic_cpu_parity",
    "verify_binding_record",
]
