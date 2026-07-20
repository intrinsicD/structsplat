"""Pure in-memory execution of one frozen COMP-011 SSP2V development cell.

The lifecycle owns authority lookup, filesystem access, persistence, target import, and process
isolation.  This module accepts one already-authorized canonical SSPL1 byte string and explicit
in-memory adapters.  It reproduces the four COMP-009 streams only through a callback whose output
is checked against sealed expected identities, builds SSP2Z, fits the frozen eight-arm menu, and
constructs/cold-decodes every SSP2V stream through both the Python and injected arithmetic cores.

No target or target-derived value is accepted by this API.  A required Lloyd cycle/cap is returned
as :class:`LloydMethodFailure`: it is a valid method outcome, not an invalid protocol exception.
Every other contract, parity, canonicality, or accounting failure raises :class:`ExecutionError`.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as comp009_assets
from benchmarks import ssp2e_container as comp009_container
from benchmarks import ssp2v_container as container
from benchmarks import ssp2v_lloyd as lloyd


TASK_SHA256 = "e715e979b4453e45a8352fbc8fc06614adbeb47193142e0c6b8b02f5523676d0"
COMP009_CAPTURED_SOURCE_SHA256 = (
    "fac4ca0978891b3cd16d477ffc04d12dc120168478192293d39af635fca7eb50"
)
EXECUTION_SCHEMA = "structsplat.comp011.ssp2v-cell-execution.v1"
METHOD_FAILURE_SCHEMA = "structsplat.comp011.ssp2v-lloyd-method-failure.v1"

COMP009_NAMES = ("ssp2e", "ssp2s", "ssp2l", "ssp2f")
COMP009_MAGICS = {
    "ssp2e": b"SSP2E",
    "ssp2s": b"SSP2S",
    "ssp2l": b"SSP2L",
    "ssp2f": b"SSP2F",
}
EXACT_FORMAT_ORDER = ("sspl1", "ssp2z", *COMP009_NAMES)
NON_RGB_CHANNELS = ("mean_x", "mean_y", "scale_x", "scale_y", "rotation")
BOUNDARY_NAMES = ("means", "log_scales", "rotations", "colors")
_EXECUTION_INTEGRITY_KEYS = frozenset(
    {
        "canonical_sspl1_cold_parse",
        "ssp2z_exact_symbols_and_boundary",
        "all_four_comp009_authorities_reproduced",
        "all_four_comp009_independently_dual_decoded",
        "comp009_captured_source_and_execution_bound",
        "all_eight_variants_fit_before_encoding",
        "all_required_lloyd_starts_fixed",
        "all_frozen_lloyd_initial_states_verified",
        "all_completed_rvq_stages_validated_before_next",
        "all_python_injected_arithmetic_payloads_equal",
        "all_python_injected_cold_decodes_equal",
        "all_variant_blobs_canonical",
        "all_non_rgb_symbols_exact",
        "all_non_rgb_boundaries_exact",
        "all_fit_decode_rgb_and_accounting_exact",
        "all_matched_codebook_byte_charges_equal",
        "selected_q_recomputed_from_all_six_exact_formats",
        "candidate_seal_binds_full_normalized_evidence",
        "complete_candidate_set_present",
    }
)


class ExecutionError(RuntimeError):
    """A frozen cell execution contract was violated."""


class _RequiredStartFailure(RuntimeError):
    def __init__(
        self,
        starts: tuple[lloyd.StartResult, ...],
        stage_input: np.ndarray,
        spec: lloyd.LloydSpec,
        completed_stages: tuple[dict[str, Any], ...] = (),
    ) -> None:
        super().__init__("a required prescribed Lloyd start did not converge")
        self.starts = starts
        self.stage_input = _readonly(stage_input, dtype=np.int32)
        self.spec = spec
        self.completed_stages = completed_stages


class ArithmeticAdapter(Protocol):
    """Injected native arithmetic interface used beside the Python proof coder."""

    def encode(self, symbols: np.ndarray, cdf_rows: np.ndarray) -> bytes: ...

    def decode(self, payload: bytes, cdf_rows: np.ndarray, n: int) -> np.ndarray: ...


@dataclass(frozen=True)
class VariantSpec:
    """One descriptor in the frozen, target-blind eight-arm order."""

    label: str
    family_id: int
    matched_stage_count: int
    base_k: int

    @property
    def descriptor(self) -> container.SSP2VDescriptor:
        actual_stages = 1 if self.family_id == 0 else self.matched_stage_count
        entries = (
            (2 * self.matched_stage_count - 1) * self.base_k
            if self.family_id == 0
            else self.matched_stage_count * self.base_k
        )
        descriptor = container.SSP2VDescriptor(
            family_id=self.family_id,
            actual_stage_count=actual_stages,
            matched_rvq_stage_count=self.matched_stage_count,
            base_k=self.base_k,
            codebook_entries=entries,
        ).validated()
        if descriptor.label != self.label:
            raise ExecutionError("frozen variant label disagrees with its descriptor")
        return descriptor


VARIANT_SPECS = (
    VariantSpec("flat_s2_k64_l192", 0, 2, 64),
    VariantSpec("flat_s2_k128_l384", 0, 2, 128),
    VariantSpec("flat_s3_k64_l320", 0, 3, 64),
    VariantSpec("flat_s3_k128_l640", 0, 3, 128),
    VariantSpec("rvq_s2_k64", 1, 2, 64),
    VariantSpec("rvq_s2_k128", 1, 2, 128),
    VariantSpec("rvq_s3_k64", 1, 3, 64),
    VariantSpec("rvq_s3_k128", 1, 3, 128),
)
VARIANT_LABELS = tuple(spec.label for spec in VARIANT_SPECS)


@dataclass(frozen=True)
class Comp009Authority:
    """Sealed expected identity for one frozen COMP-009 stream reproduction."""

    complete_bytes: int
    blob_sha256: str
    model_state_sha256: str
    symbol_sha256: str
    boundary_state_sha256: str
    captured_source_sha256: str
    execution_sha256: str
    verification_sha256: str


@dataclass(frozen=True)
class ReproducedComp009Stream:
    """In-memory result and cold-decode attestations returned by the injected callback."""

    blob: bytes
    model_state_sha256: str
    symbol_sha256: str
    boundary_state_sha256: str
    canonical_cold_decode: bool
    python_native_decode_parity: bool
    source_blob_sha256: str
    symbols: Mapping[str, np.ndarray]
    boundary_arrays: Mapping[str, np.ndarray]
    components: Mapping[str, Mapping[str, Any]]
    captured_source_sha256: str
    execution_sha256: str
    verification_sha256: str


Comp009Reproducer = Callable[[bytes], Mapping[str, ReproducedComp009Stream | Mapping[str, Any]]]


@dataclass(frozen=True)
class CellExecution:
    """Complete, deterministic, target-free products for one successful cell."""

    record: dict[str, Any]
    baseline_blobs: dict[str, bytes]
    variant_blobs: dict[str, bytes]
    boundary_arrays: dict[str, dict[str, np.ndarray]]
    fits: dict[str, lloyd.VariantFitResult]


@dataclass(frozen=True)
class LloydMethodFailure:
    """Typed valid method failure caused by a required prescribed start cycle/cap."""

    record: dict[str, Any]
    baseline_blobs: dict[str, bytes]
    failed_starts: tuple[lloyd.StartResult, ...]


CellResult = CellExecution | LloydMethodFailure


_AUTHORITY_KEYS = frozenset(
    {
        "complete_bytes",
        "blob_sha256",
        "model_state_sha256",
        "symbol_sha256",
        "boundary_state_sha256",
        "captured_source_sha256",
        "execution_sha256",
        "verification_sha256",
    }
)
_REPRODUCED_KEYS = frozenset(
    {
        "blob",
        "model_state_sha256",
        "symbol_sha256",
        "boundary_state_sha256",
        "canonical_cold_decode",
        "python_native_decode_parity",
        "source_blob_sha256",
        "symbols",
        "boundary_arrays",
        "components",
        "captured_source_sha256",
        "execution_sha256",
        "verification_sha256",
    }
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _seal(record: dict[str, Any]) -> dict[str, Any]:
    if "execution_sha256" in record:
        raise ExecutionError("cannot seal a record that already contains execution_sha256")
    result = dict(record)
    result["execution_sha256"] = _sha256(_canonical_json(record))
    return result


def verify_record(record: Mapping[str, Any]) -> None:
    """Verify a returned execution or method-failure JSON seal."""

    if not isinstance(record, Mapping):
        raise ExecutionError("record must be a mapping")
    observed = record.get("execution_sha256")
    if not isinstance(observed, str):
        raise ExecutionError("record lacks execution_sha256")
    core = dict(record)
    del core["execution_sha256"]
    if observed != _sha256(_canonical_json(core)):
        raise ExecutionError("execution_sha256 mismatch")
    schema = record.get("schema")
    if schema == EXECUTION_SCHEMA:
        try:
            if record.get("task_sha256") != TASK_SHA256:
                raise ExecutionError("sealed execution task identity drifted")
            if record.get("status") != "valid_complete_candidate_set":
                raise ExecutionError("sealed complete execution status drifted")
            if (
                record.get("target_input_surface") != "absent"
                or record.get("target_accessed") is not False
                or record.get("candidate_set_seals_full_normalized_evidence") is not True
                or record.get("claim_scope") != "target-free-development-candidate-set"
            ):
                raise ExecutionError("sealed complete execution lifecycle flags drifted")
            integrity = record.get("integrity")
            if (
                not isinstance(integrity, Mapping)
                or frozenset(integrity) != _EXECUTION_INTEGRITY_KEYS
                or any(value is not True for value in integrity.values())
            ):
                raise ExecutionError("sealed complete execution integrity predicates drifted")
            baseline_records = record["exact_formats"]
            variant_records = record["variants"]
            if not isinstance(baseline_records, Mapping) or set(baseline_records) != set(
                EXACT_FORMAT_ORDER
            ):
                raise ExecutionError("sealed exact-format set drifted")
            if record.get("source") != baseline_records.get("sspl1"):
                raise ExecutionError("sealed source differs from exact SSPL1 evidence")
            if record.get("exact_format_order") != list(EXACT_FORMAT_ORDER):
                raise ExecutionError("sealed exact-format order drifted")
            if (
                not isinstance(variant_records, Mapping)
                or set(variant_records) != set(VARIANT_LABELS)
                or record.get("variant_order") != list(VARIANT_LABELS)
            ):
                raise ExecutionError("sealed variant set/order drifted")
            expected_q = _select_exact_q(baseline_records)
            if record.get("selected_q") != expected_q:
                raise ExecutionError("sealed selected Q differs from the exact six-format minimum")
            matched = record["matched_codebook_bytes"]
            if record.get("candidate_evidence_schema") != (
                "structsplat.comp011.candidate-evidence.v1"
            ):
                raise ExecutionError("candidate evidence schema drifted")
            evidence = {
                "schema": "structsplat.comp011.candidate-evidence.v1",
                "task_sha256": TASK_SHA256,
                "exact_format_order": list(EXACT_FORMAT_ORDER),
                "exact_formats": baseline_records,
                "selected_q": expected_q,
                "variant_order": list(VARIANT_LABELS),
                "variants": variant_records,
                "matched_codebook_bytes": matched,
            }
            expected_candidate_seal = _sha256(_canonical_json(evidence))
            if record.get("candidate_set_sha256") != expected_candidate_seal:
                raise ExecutionError("candidate_set_sha256 mismatch")
            expected_inventory = _candidate_inventory(baseline_records, variant_records)
            if record.get("candidate_inventory") != expected_inventory:
                raise ExecutionError("candidate inventory differs from its full evidence")
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"malformed complete execution evidence: {exc}") from exc
    elif schema == METHOD_FAILURE_SCHEMA:
        if (
            record.get("task_sha256") != TASK_SHA256
            or record.get("status") != "valid_method_failure"
            or record.get("method_state") != "required_lloyd_nonconvergence"
            or record.get("target_input_surface") != "absent"
            or record.get("target_accessed") is not False
            or record.get("target_import_authorized") is not False
            or record.get("partial_ssp2v_candidate_set_emitted") is not False
            or record.get("claim_scope") != "valid-preregistered-method-failure"
        ):
            raise ExecutionError("malformed Lloyd method-failure seal")
        exact_formats = record.get("exact_formats")
        if (
            not isinstance(exact_formats, Mapping)
            or set(exact_formats) != set(EXACT_FORMAT_ORDER)
            or record.get("source") != exact_formats.get("sspl1")
        ):
            raise ExecutionError("malformed Lloyd method-failure exact-format evidence")
        failure = record.get("failure")
        if not isinstance(failure, Mapping) or failure.get("variant_label") not in VARIANT_LABELS:
            raise ExecutionError("malformed Lloyd method-failure detail")
        spec = next(item for item in VARIANT_SPECS if item.label == failure["variant_label"])
        completed_variant_fits = record.get("completed_variant_fits")
        expected_completed_labels = set(
            VARIANT_LABELS[: VARIANT_LABELS.index(failure["variant_label"])]
        )
        if (
            not isinstance(completed_variant_fits, Mapping)
            or set(completed_variant_fits) != expected_completed_labels
        ):
            raise ExecutionError("malformed Lloyd method-failure completed-variant history")
        if (
            failure.get("family_id") != spec.family_id
            or failure.get("matched_stage_count") != spec.matched_stage_count
            or failure.get("base_k") != spec.base_k
        ):
            raise ExecutionError("malformed Lloyd method-failure variant descriptor")
        stage_index = _exact_nonnegative_int(
            failure.get("actual_stage_index"),
            "failure.actual_stage_index",
        )
        if stage_index >= spec.descriptor.actual_stage_count:
            raise ExecutionError("malformed Lloyd method-failure stage index")
        completed_stages = failure.get("completed_stages")
        if not isinstance(completed_stages, list) or len(completed_stages) != stage_index:
            raise ExecutionError("malformed Lloyd method-failure completed-stage history")
        for expected_stage, stage in enumerate(completed_stages):
            if (
                not isinstance(stage, Mapping)
                or stage.get("all_four_starts_fixed") is not True
                or not isinstance(stage.get("spec"), Mapping)
                or stage["spec"].get("actual_stage_index") != expected_stage
            ):
                raise ExecutionError("malformed Lloyd method-failure completed-stage evidence")
        starts = failure.get("starts")
        failed_ids = failure.get("failed_start_ids")
        if (
            not isinstance(starts, list)
            or len(starts) != len(lloyd.START_IDS)
            or [item.get("start_id") for item in starts if isinstance(item, Mapping)]
            != list(lloyd.START_IDS)
            or not isinstance(failed_ids, list)
            or not failed_ids
            or any(
                _exact_nonnegative_int(value, "failure.failed_start_id") not in lloyd.START_IDS
                for value in failed_ids
            )
        ):
            raise ExecutionError("malformed Lloyd method-failure start evidence")
    else:
        raise ExecutionError("unknown execution record schema")


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExecutionError(f"{name} must be a positive integer")
    return value


def _exact_nonnegative_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ExecutionError(f"{name} must be an exact nonnegative integer")
    integer = int(value)
    if integer < 0 or (maximum is not None and integer > maximum):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise ExecutionError(f"{name} must be an exact nonnegative integer{suffix}")
    return integer


def _exact_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ExecutionError(f"{name} must be exactly boolean")
    return value


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | str | None = None) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype).copy()
    result.flags.writeable = False
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


def _array_state_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def boundary_arrays(
    header: container.SSP2VHeader,
    symbols: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Independently construct the exact float32 decoded field boundary."""

    header.validated()
    if set(symbols) != set(container.SYMBOL_CHANNELS):
        raise ExecutionError("boundary reconstruction requires all eight symbol arrays")
    limits = {
        "mean_x": 1 << header.bits[0],
        "mean_y": 1 << header.bits[0],
        "scale_x": 1 << header.bits[1],
        "scale_y": 1 << header.bits[1],
        "rotation": 1 << header.bits[2],
        "red": 1 << header.bits[3],
        "green": 1 << header.bits[3],
        "blue": 1 << header.bits[3],
    }
    normalized_symbols: dict[str, np.ndarray] = {}
    for name in container.SYMBOL_CHANNELS:
        value = np.asarray(symbols[name])
        if value.shape != (container.ROW_COUNT,) or value.dtype.kind not in "iu":
            raise ExecutionError(f"boundary symbol {name} must be an integer N-vector")
        if value.dtype.kind == "i" and np.any(value < 0):
            raise ExecutionError(f"boundary symbol {name} is negative")
        if value.size and int(value.max()) >= limits[name]:
            raise ExecutionError(f"boundary symbol {name} exceeds its declared bit depth")
        normalized_symbols[name] = np.ascontiguousarray(value, dtype=np.uint32)

    bm, bs, br, bc = header.bits
    mean_symbols = np.column_stack(
        (normalized_symbols["mean_x"], normalized_symbols["mean_y"])
    )
    scale_symbols = np.column_stack(
        (normalized_symbols["scale_x"], normalized_symbols["scale_y"])
    )
    color_symbols = np.column_stack(
        (
            normalized_symbols["red"],
            normalized_symbols["green"],
            normalized_symbols["blue"],
        )
    )

    def dequantize(
        values: np.ndarray,
        lower: Sequence[int | float],
        upper: Sequence[int | float],
        bits: int,
    ) -> np.ndarray:
        levels = (1 << bits) - 1
        result = (
            np.asarray(lower, dtype=np.float64)
            + np.asarray(values, dtype=np.float64)
            / levels
            * (np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64))
        ).astype(np.float32)
        return _readonly(result)

    result = {
        "means": dequantize(mean_symbols, header.means_lo, header.means_hi, bm),
        "log_scales": dequantize(
            scale_symbols, header.scale_lo, header.scale_hi, bs
        ),
        "rotations": _readonly(
            (
                np.asarray(normalized_symbols["rotation"], dtype=np.float64)
                / (1 << br)
                * math.pi
            ).astype(np.float32)
        ),
        "colors": dequantize(color_symbols, header.color_lo, header.color_hi, bc),
    }
    if any(value.dtype != np.float32 for value in result.values()):
        raise ExecutionError("boundary construction did not produce float32 arrays")
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ExecutionError("boundary construction produced a non-finite value")
    if any(
        np.shares_memory(boundary, symbol)
        for boundary in result.values()
        for symbol in normalized_symbols.values()
    ):
        raise ExecutionError("boundary construction aliased a decoded symbol array")
    return result


def _boundary_record(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(arrays) != set(BOUNDARY_NAMES):
        raise ExecutionError("boundary array set drifted")
    return {
        "state_sha256": _array_state_sha256(arrays),
        "components": {name: _array_record(arrays[name]) for name in BOUNDARY_NAMES},
    }


def _source_rgb(parts: container.SSPL1Parts) -> np.ndarray:
    rgb = np.column_stack(
        (parts.symbols["red"], parts.symbols["green"], parts.symbols["blue"])
    )
    if rgb.shape != (container.ROW_COUNT, 3) or np.any(rgb > 255):
        raise ExecutionError("COMP-011 source RGB symbols must be uint8-representable N-by-3")
    return _readonly(rgb, dtype=np.uint8)


def _normalize_symbol_mapping(value: Any, name: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != set(container.SYMBOL_CHANNELS):
        raise ExecutionError(f"{name} must contain exactly all eight decoded symbol arrays")
    result: dict[str, np.ndarray] = {}
    for channel in container.SYMBOL_CHANNELS:
        array = np.asarray(value[channel])
        if array.shape != (container.ROW_COUNT,) or array.dtype.kind not in "iu":
            raise ExecutionError(f"{name}.{channel} must be an integer N-vector")
        if array.dtype.kind == "i" and np.any(array < 0):
            raise ExecutionError(f"{name}.{channel} contains a negative symbol")
        if array.size and int(array.max()) >= 1 << 32:
            raise ExecutionError(f"{name}.{channel} exceeds uint32")
        result[channel] = _readonly(array, dtype=np.uint32)
    return result


def _normalize_boundary_mapping(value: Any, name: str) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != set(BOUNDARY_NAMES):
        raise ExecutionError(f"{name} must contain the exact decoded boundary array set")
    expected_shapes = {
        "means": (container.ROW_COUNT, 2),
        "log_scales": (container.ROW_COUNT, 2),
        "rotations": (container.ROW_COUNT,),
        "colors": (container.ROW_COUNT, 3),
    }
    result: dict[str, np.ndarray] = {}
    for component in BOUNDARY_NAMES:
        array = np.asarray(value[component])
        if array.shape != expected_shapes[component] or array.dtype != np.float32:
            raise ExecutionError(f"{name}.{component} must be exact float32 boundary state")
        if not np.isfinite(array).all():
            raise ExecutionError(f"{name}.{component} contains a non-finite value")
        result[component] = _readonly(array)
    return result


def _normalize_component_mapping(
    value: Any,
    name: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise ExecutionError(f"{name} must be a nonempty component mapping")
    result: dict[str, dict[str, Any]] = {}
    for component, raw_record in value.items():
        if not isinstance(component, str) or not component:
            raise ExecutionError(f"{name} component names must be nonempty strings")
        if not isinstance(raw_record, Mapping) or set(raw_record) != {"bytes", "sha256"}:
            raise ExecutionError(f"{name}.{component} must contain bytes and sha256")
        result[component] = {
            "bytes": _positive_int(raw_record["bytes"], f"{name}.{component}.bytes"),
            "sha256": _digest(raw_record["sha256"], f"{name}.{component}.sha256"),
        }
    if "total" not in result:
        raise ExecutionError(f"{name} must contain a total component")
    return result


def _normalize_authority(value: Any, name: str) -> Comp009Authority:
    if isinstance(value, Comp009Authority):
        record = asdict(value)
    elif isinstance(value, Mapping):
        if frozenset(value) != _AUTHORITY_KEYS:
            raise ExecutionError(f"{name} authority fields differ from the frozen contract")
        record = dict(value)
    else:
        raise ExecutionError(f"{name} authority must be Comp009Authority or a mapping")
    return Comp009Authority(
        complete_bytes=_positive_int(record["complete_bytes"], f"{name}.complete_bytes"),
        blob_sha256=_digest(record["blob_sha256"], f"{name}.blob_sha256"),
        model_state_sha256=_digest(
            record["model_state_sha256"], f"{name}.model_state_sha256"
        ),
        symbol_sha256=_digest(record["symbol_sha256"], f"{name}.symbol_sha256"),
        boundary_state_sha256=_digest(
            record["boundary_state_sha256"], f"{name}.boundary_state_sha256"
        ),
        captured_source_sha256=_digest(
            record["captured_source_sha256"], f"{name}.captured_source_sha256"
        ),
        execution_sha256=_digest(record["execution_sha256"], f"{name}.execution_sha256"),
        verification_sha256=_digest(
            record["verification_sha256"], f"{name}.verification_sha256"
        ),
    )


def _normalize_reproduction(value: Any, name: str) -> ReproducedComp009Stream:
    if isinstance(value, ReproducedComp009Stream):
        record = {
            field: getattr(value, field)
            for field in _REPRODUCED_KEYS
        }
    elif isinstance(value, Mapping):
        if frozenset(value) != _REPRODUCED_KEYS:
            raise ExecutionError(f"{name} reproduction fields differ from the frozen contract")
        record = dict(value)
    else:
        raise ExecutionError(
            f"{name} reproduction must be ReproducedComp009Stream or a mapping"
        )
    blob = record["blob"]
    if not isinstance(blob, bytes) or not blob:
        raise ExecutionError(f"{name}.blob must be nonempty immutable bytes")
    return ReproducedComp009Stream(
        blob=blob,
        model_state_sha256=_digest(
            record["model_state_sha256"], f"{name}.model_state_sha256"
        ),
        symbol_sha256=_digest(record["symbol_sha256"], f"{name}.symbol_sha256"),
        boundary_state_sha256=_digest(
            record["boundary_state_sha256"], f"{name}.boundary_state_sha256"
        ),
        canonical_cold_decode=_exact_bool(
            record["canonical_cold_decode"], f"{name}.canonical_cold_decode"
        ),
        python_native_decode_parity=_exact_bool(
            record["python_native_decode_parity"],
            f"{name}.python_native_decode_parity",
        ),
        source_blob_sha256=_digest(
            record["source_blob_sha256"], f"{name}.source_blob_sha256"
        ),
        symbols=_normalize_symbol_mapping(record["symbols"], f"{name}.symbols"),
        boundary_arrays=_normalize_boundary_mapping(
            record["boundary_arrays"], f"{name}.boundary_arrays"
        ),
        components=_normalize_component_mapping(
            record["components"], f"{name}.components"
        ),
        captured_source_sha256=_digest(
            record["captured_source_sha256"], f"{name}.captured_source_sha256"
        ),
        execution_sha256=_digest(record["execution_sha256"], f"{name}.execution_sha256"),
        verification_sha256=_digest(
            record["verification_sha256"], f"{name}.verification_sha256"
        ),
    )


def comp009_verification_sha256(
    *,
    name: str,
    source_blob_sha256: str,
    complete_bytes: int,
    blob_sha256: str,
    model_state_sha256: str,
    symbol_sha256: str,
    boundary_state_sha256: str,
    components: Mapping[str, Mapping[str, Any]],
    captured_source_sha256: str,
    execution_sha256: str,
    canonical_cold_decode: bool,
    python_native_decode_parity: bool,
) -> str:
    """Seal the normalized, source-bound evidence returned by a COMP-009 callback."""

    if name not in COMP009_NAMES:
        raise ExecutionError("COMP-009 verification name is outside SSP2E/S/L/F")
    normalized_components = _normalize_component_mapping(
        components, f"comp009_verification.{name}.components"
    )
    record = {
        "schema": "structsplat.comp011.comp009-reproduction-proof.v1",
        "task_sha256": TASK_SHA256,
        "name": name,
        "source_blob_sha256": _digest(source_blob_sha256, "source_blob_sha256"),
        "complete_bytes": _positive_int(complete_bytes, "complete_bytes"),
        "blob_sha256": _digest(blob_sha256, "blob_sha256"),
        "model_state_sha256": _digest(model_state_sha256, "model_state_sha256"),
        "symbol_sha256": _digest(symbol_sha256, "symbol_sha256"),
        "boundary_state_sha256": _digest(
            boundary_state_sha256, "boundary_state_sha256"
        ),
        "components": normalized_components,
        "captured_source_sha256": _digest(
            captured_source_sha256, "captured_source_sha256"
        ),
        "execution_sha256": _digest(execution_sha256, "execution_sha256"),
        "canonical_cold_decode": _exact_bool(
            canonical_cold_decode, "canonical_cold_decode"
        ),
        "python_native_decode_parity": _exact_bool(
            python_native_decode_parity, "python_native_decode_parity"
        ),
    }
    return _sha256(_canonical_json(record))


def _comp009_component_records(
    parsed: comp009_container.ParsedContainer,
    blob: bytes,
) -> dict[str, dict[str, Any]]:
    directory_begin = comp009_container.OUTER.size + comp009_container.HEADER.size
    payloads: dict[str, bytes] = {
        "outer": blob[: comp009_container.OUTER.size],
        "header": blob[comp009_container.OUTER.size : directory_begin],
        "directory": blob[directory_begin : parsed.prefix_bytes],
        **parsed.payloads,
    }
    records = {name: _arrayless_bytes_record(payload) for name, payload in payloads.items()}
    records["total"] = _arrayless_bytes_record(blob)
    if sum(record["bytes"] for name, record in records.items() if name != "total") != len(
        blob
    ):
        raise ExecutionError("COMP-009 component records do not partition the complete blob")
    return records


def _comp009_frequency_provider() -> Callable[[int, int, int], Sequence[int]]:
    tables = dict(comp009_assets.load_tables())
    if set(tables) != set(comp009_assets.ALPHABETS):
        raise ExecutionError("frozen COMP-009 frequency tables are incomplete")

    def provide(alphabet: int, location: int, scale: int) -> Sequence[int]:
        try:
            return tables[alphabet].frequency_row(location, scale)
        except Exception as exc:
            raise ExecutionError(f"frozen COMP-009 frequency lookup failed: {exc}") from exc

    return provide


def _reproduce_comp009(
    sspl1_blob: bytes,
    reproduce: Comp009Reproducer,
    expected: Mapping[str, Comp009Authority | Mapping[str, Any]],
    *,
    source_parts: container.SSPL1Parts,
    source_boundary: Mapping[str, np.ndarray],
    source_symbol_sha256: str,
    source_boundary_sha256: str,
    native_coder: ArithmeticAdapter,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if not callable(reproduce):
        raise ExecutionError("reproduce_comp009 must be callable")
    if not isinstance(expected, Mapping) or set(expected) != set(COMP009_NAMES):
        raise ExecutionError("expected_comp009 must contain exactly SSP2E/S/L/F")
    authorities = {
        name: _normalize_authority(expected[name], f"expected_comp009.{name}")
        for name in COMP009_NAMES
    }
    if {authority.captured_source_sha256 for authority in authorities.values()} != {
        COMP009_CAPTURED_SOURCE_SHA256
    }:
        raise ExecutionError("COMP-009 authorities do not bind the frozen captured source archive")
    if len({authority.execution_sha256 for authority in authorities.values()}) != 1:
        raise ExecutionError("COMP-009 authorities do not share one captured execution seal")
    frequency_provider = _comp009_frequency_provider()
    try:
        raw_reproduced = reproduce(sspl1_blob)
    except Exception as exc:
        raise ExecutionError(f"COMP-009 reproduction callback failed: {exc}") from exc
    if not isinstance(raw_reproduced, Mapping) or set(raw_reproduced) != set(COMP009_NAMES):
        raise ExecutionError("COMP-009 callback must return exactly SSP2E/S/L/F")

    blobs: dict[str, bytes] = {}
    records: dict[str, dict[str, Any]] = {}
    for name in COMP009_NAMES:
        value = _normalize_reproduction(raw_reproduced[name], f"reproduced_comp009.{name}")
        authority = authorities[name]
        observed_hash = _sha256(value.blob)
        try:
            parsed = comp009_container.parse_container(
                value.blob, reference_sspl1=sspl1_blob
            )
        except Exception as exc:
            raise ExecutionError(f"reproduced {name} strict parse failed: {exc}") from exc
        if parsed.magic != COMP009_MAGICS[name]:
            raise ExecutionError(f"reproduced {name} magic mismatch")
        if len(value.blob) != authority.complete_bytes or observed_hash != authority.blob_sha256:
            raise ExecutionError(f"reproduced {name} complete-byte authority mismatch")
        model_payload = (
            parsed.payloads["grid"] + parsed.payloads["head"]
            if name in ("ssp2e", "ssp2s")
            else parsed.payloads["model"]
        )
        if (
            value.model_state_sha256 != authority.model_state_sha256
            or value.model_state_sha256 != _sha256(model_payload)
        ):
            raise ExecutionError(f"reproduced {name} model/state authority mismatch")
        if (
            value.symbol_sha256 != authority.symbol_sha256
            or value.symbol_sha256 != source_symbol_sha256
        ):
            raise ExecutionError(f"reproduced {name} symbol authority mismatch")
        if (
            value.boundary_state_sha256 != authority.boundary_state_sha256
            or value.boundary_state_sha256 != source_boundary_sha256
        ):
            raise ExecutionError(f"reproduced {name} boundary authority mismatch")
        if not value.canonical_cold_decode or not value.python_native_decode_parity:
            raise ExecutionError(f"reproduced {name} lacks canonical dual cold-decode proof")
        if (
            value.captured_source_sha256 != authority.captured_source_sha256
            or value.captured_source_sha256 != COMP009_CAPTURED_SOURCE_SHA256
        ):
            raise ExecutionError(f"reproduced {name} captured-source authority mismatch")
        if value.execution_sha256 != authority.execution_sha256:
            raise ExecutionError(f"reproduced {name} captured execution authority mismatch")
        source_hash = _sha256(sspl1_blob)
        if value.source_blob_sha256 != source_hash:
            raise ExecutionError(f"reproduced {name} proof is not bound to this SSPL1 source")
        computed_components = _comp009_component_records(parsed, value.blob)
        if value.components != computed_components:
            raise ExecutionError(f"reproduced {name} component records differ from its blob")
        provider = None if name == "ssp2f" else frequency_provider
        try:
            python_decoded = comp009_container.decode_container(
                value.blob,
                arithmetic_decode=arithmetic.decode,
                arithmetic_encode=arithmetic.encode,
                frequency_provider=provider,
                reference_sspl1=sspl1_blob,
            )
            native_decoded = comp009_container.decode_container(
                value.blob,
                arithmetic_decode=native_coder.decode,
                arithmetic_encode=native_coder.encode,
                frequency_provider=provider,
                reference_sspl1=sspl1_blob,
            )
        except Exception as exc:
            raise ExecutionError(f"reproduced {name} dual strict cold-decode failed: {exc}") from exc
        _assert_symbol_arrays_equal(
            python_decoded.symbols,
            native_decoded.symbols,
            container.SYMBOL_CHANNELS,
            f"reproduced {name} Python/native",
        )
        _assert_symbol_arrays_equal(
            python_decoded.symbols,
            source_parts.symbols,
            container.SYMBOL_CHANNELS,
            f"reproduced {name} independently decoded",
        )
        _assert_symbol_arrays_equal(
            value.symbols,
            python_decoded.symbols,
            container.SYMBOL_CHANNELS,
            f"reproduced {name}",
        )
        reconstructed_boundary = boundary_arrays(
            source_parts.header, python_decoded.symbols
        )
        for component in BOUNDARY_NAMES:
            if (
                not np.array_equal(value.boundary_arrays[component], reconstructed_boundary[component])
                or not np.array_equal(value.boundary_arrays[component], source_boundary[component])
            ):
                raise ExecutionError(f"reproduced {name} boundary differs in {component}")
        if _array_state_sha256(value.boundary_arrays) != value.boundary_state_sha256:
            raise ExecutionError(f"reproduced {name} boundary digest disagrees with its arrays")
        verification = comp009_verification_sha256(
            name=name,
            source_blob_sha256=value.source_blob_sha256,
            complete_bytes=len(value.blob),
            blob_sha256=observed_hash,
            model_state_sha256=value.model_state_sha256,
            symbol_sha256=value.symbol_sha256,
            boundary_state_sha256=value.boundary_state_sha256,
            components=computed_components,
            captured_source_sha256=value.captured_source_sha256,
            execution_sha256=value.execution_sha256,
            canonical_cold_decode=value.canonical_cold_decode,
            python_native_decode_parity=value.python_native_decode_parity,
        )
        if verification != value.verification_sha256 or verification != authority.verification_sha256:
            raise ExecutionError(f"reproduced {name} source-bound verification seal mismatch")
        blobs[name] = value.blob
        records[name] = {
            "magic": COMP009_MAGICS[name].decode("ascii"),
            "complete_bytes": len(value.blob),
            "blob_sha256": observed_hash,
            "model_state_sha256": value.model_state_sha256,
            "symbol_sha256": value.symbol_sha256,
            "boundary_state_sha256": value.boundary_state_sha256,
            "source_blob_sha256": value.source_blob_sha256,
            "components": computed_components,
            "captured_source_sha256": value.captured_source_sha256,
            "execution_sha256": value.execution_sha256,
            "verification_sha256": verification,
            "authority_hashes_verified": True,
            "canonical_cold_decode": True,
            "python_native_decode_parity": True,
            "independent_dual_cold_decode": True,
            "reproduction_mechanism": "explicit_in_memory_callback",
        }
    return blobs, records


def _container_component_record(
    parsed: container.ParsedContainer,
    blob: bytes,
) -> dict[str, Any]:
    prefix = parsed.prefix_bytes
    directory_begin = container.OUTER.size + container.HEADER.size
    component_payloads: dict[str, bytes] = {
        "outer": blob[: container.OUTER.size],
        "header": blob[container.OUTER.size : directory_begin],
        "directory": blob[directory_begin:prefix],
    }
    component_payloads.update(parsed.payloads)
    records = {name: _arrayless_bytes_record(value) for name, value in component_payloads.items()}
    records["total"] = _arrayless_bytes_record(blob)
    byte_counts = parsed.component_bytes()
    for name, count in byte_counts.items():
        if records[name]["bytes"] != count:
            raise ExecutionError(f"container component byte mismatch for {name}")
    if sum(item["bytes"] for name, item in records.items() if name != "total") != len(blob):
        raise ExecutionError("container component records do not partition the complete blob")
    return {"bytes": byte_counts, "records": records}


def _arrayless_bytes_record(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": _sha256(payload)}


def _sspl1_component_records(
    blob: bytes,
    parts: container.SSPL1Parts,
) -> dict[str, dict[str, Any]]:
    if blob[:5] != b"SSPL1" or len(blob) < 9:
        raise ExecutionError("SSPL1 component parser received invalid framing")
    header_length = struct.unpack_from("<I", blob, 5)[0]
    header_begin = 9
    header_end = header_begin + header_length
    if header_end > len(blob):
        raise ExecutionError("SSPL1 component parser found a truncated header")
    records = {
        "magic": _arrayless_bytes_record(blob[:5]),
        "header_length": _arrayless_bytes_record(blob[5:9]),
        "header_json": _arrayless_bytes_record(blob[header_begin:header_end]),
    }
    offset = header_end
    for name in ("means", "scales", "rotation", "colors"):
        if offset + 4 > len(blob):
            raise ExecutionError(f"SSPL1 component parser found truncated {name} length")
        payload_length = struct.unpack_from("<I", blob, offset)[0]
        prefix = blob[offset : offset + 4]
        offset += 4
        end = offset + payload_length
        if end > len(blob):
            raise ExecutionError(f"SSPL1 component parser found truncated {name} payload")
        payload = blob[offset:end]
        if payload != parts.payloads[name]:
            raise ExecutionError(f"SSPL1 component {name} differs from its strict parse")
        records[f"{name}_length"] = _arrayless_bytes_record(prefix)
        records[f"{name}_zlib"] = _arrayless_bytes_record(payload)
        offset = end
    if offset != len(blob):
        raise ExecutionError("SSPL1 component parser found an unclaimed suffix")
    records["total"] = _arrayless_bytes_record(blob)
    if sum(record["bytes"] for name, record in records.items() if name != "total") != len(
        blob
    ):
        raise ExecutionError("SSPL1 component records do not partition the complete blob")
    return records


def _nearest_indices_and_sse(
    rows: np.ndarray,
    codebook: np.ndarray,
) -> tuple[np.ndarray, int]:
    source = np.ascontiguousarray(rows, dtype=np.int32)
    centroids = np.ascontiguousarray(codebook, dtype=np.int32)
    if (
        source.ndim != 2
        or source.shape[1:] != (3,)
        or len(source) == 0
        or centroids.ndim != 2
        or centroids.shape[1] != 3
        or len(centroids) == 0
    ):
        raise ExecutionError("Lloyd terminal arrays have invalid shapes")
    best_indices = np.zeros(len(source), dtype=np.uint32)
    best_distances = np.full(len(source), lloyd.INT64_MAX, dtype=np.int64)
    source64 = source.astype(np.int64, copy=False)
    for begin in range(0, len(centroids), 128):
        block = centroids[begin : begin + 128].astype(np.int64, copy=False)
        difference = source64[:, None, :] - block[None, :, :]
        distances = np.sum(difference * difference, axis=2, dtype=np.int64)
        local = np.argmin(distances, axis=1)
        local_distances = distances[np.arange(len(source)), local]
        better = local_distances < best_distances
        best_distances[better] = local_distances[better]
        best_indices[better] = (local[better] + begin).astype(np.uint32)
    objective = sum(int(value) for value in best_distances)
    if objective < 0 or objective > lloyd.UINT64_MAX:
        raise ExecutionError("Lloyd terminal objective is outside checked uint64")
    return best_indices, objective


def _exact_lloyd_update(
    weighted: lloyd.WeightedVectors,
    codebook: np.ndarray,
    spec: lloyd.LloydSpec,
) -> np.ndarray:
    assignments, _ = _nearest_indices_and_sse(weighted.vectors, codebook)
    counts = [0] * len(codebook)
    sums = [[0, 0, 0] for _ in range(len(codebook))]
    for vector, weight, raw_index in zip(
        weighted.vectors, weighted.counts, assignments, strict=True
    ):
        index = int(raw_index)
        counts[index] += weight
        for coordinate in range(3):
            sums[index][coordinate] += int(vector[coordinate]) * weight
    updated: list[tuple[int, int, int]] = []
    for index, old in enumerate(np.asarray(codebook, dtype=np.int32)):
        if counts[index] == 0:
            updated.append(tuple(int(value) for value in old))
        else:
            updated.append(
                tuple(
                    lloyd.round_weighted_mean(sums[index][coordinate], counts[index])
                    for coordinate in range(3)
                )
            )
    result = np.ascontiguousarray(sorted(updated), dtype=np.int32)
    lloyd.serialize_codebook(result, spec)
    return result


def _trajectory_records(
    start: lloyd.StartResult,
    *,
    expected_initial_sha256: str,
    expected_initial_sse: int,
) -> list[dict[str, Any]]:
    result = []
    update_sweeps = _exact_nonnegative_int(
        start.update_sweeps,
        "Lloyd update_sweeps",
        maximum=lloyd.MAX_UPDATE_SWEEPS,
    )
    if update_sweeps == 0:
        raise ExecutionError("a Lloyd trajectory must contain at least one exact update")
    terminal_sse = _exact_nonnegative_int(
        start.terminal_sse,
        "Lloyd terminal_sse",
        maximum=lloyd.UINT64_MAX,
    )
    if len(start.trajectory) != update_sweeps + 1 or not start.trajectory:
        raise ExecutionError("Lloyd trajectory length disagrees with update_sweeps")
    for index, item in enumerate(start.trajectory):
        state_index = _exact_nonnegative_int(item.state_index, "trajectory.state_index")
        update_sweep = _exact_nonnegative_int(item.update_sweep, "trajectory.update_sweep")
        if state_index != index or update_sweep != index:
            raise ExecutionError("Lloyd trajectory state/sweep indices are not canonical")
        result.append(
            {
                "state_index": state_index,
                "update_sweep": update_sweep,
                "codebook_sha256": _digest(
                    item.codebook_sha256, "trajectory.codebook_sha256"
                ),
                "objective_sse": _exact_nonnegative_int(
                    item.objective_sse,
                    "trajectory.objective_sse",
                    maximum=lloyd.UINT64_MAX,
                ),
            }
        )
    if any(
        later["objective_sse"] > earlier["objective_sse"]
        for earlier, later in zip(result, result[1:], strict=False)
    ):
        raise ExecutionError("Lloyd trajectory objective increased")
    if result[-1]["objective_sse"] != terminal_sse:
        raise ExecutionError("Lloyd terminal SSE disagrees with its trajectory")
    if result[-1]["codebook_sha256"] != lloyd.codebook_state_sha256(start.codebook):
        raise ExecutionError("Lloyd terminal codebook hash disagrees with its trajectory")
    hashes = [item["codebook_sha256"] for item in result]
    if (
        result[0]["codebook_sha256"] != expected_initial_sha256
        or result[0]["objective_sse"] != expected_initial_sse
    ):
        raise ExecutionError("Lloyd trajectory does not begin at its frozen initialized state")
    objective_by_hash: dict[str, int] = {}
    for item in result:
        previous = objective_by_hash.setdefault(
            item["codebook_sha256"], item["objective_sse"]
        )
        if previous != item["objective_sse"]:
            raise ExecutionError("one Lloyd codebook state has inconsistent trajectory objectives")
    if start.status == "fixed":
        if len(hashes) < 2 or hashes[-1] != hashes[-2]:
            raise ExecutionError("fixed Lloyd status lacks a repeated terminal state")
    elif start.status == "cycle":
        if len(hashes) < 3 or hashes[-1] == hashes[-2] or hashes[-1] not in hashes[:-2]:
            raise ExecutionError("cycle Lloyd status lacks a repeated non-fixed state")
    elif start.status == "cap":
        if start.update_sweeps != lloyd.MAX_UPDATE_SWEEPS or len(set(hashes)) != len(hashes):
            raise ExecutionError("capped Lloyd status lacks 100 distinct updates")
    return result


def _start_record(
    start: lloyd.StartResult,
    stage_input: np.ndarray,
    weighted: lloyd.WeightedVectors,
    *,
    selected: bool,
    cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> dict[str, Any]:
    start_id = _exact_nonnegative_int(start.start_id, "Lloyd start_id")
    if start_id not in lloyd.START_IDS or start.spec.codebook_size != len(start.codebook):
        raise ExecutionError("Lloyd start metadata is malformed")
    if start.status not in ("fixed", "cycle", "cap"):
        raise ExecutionError("Lloyd start returned an unknown convergence status")
    serialized = start.serialized_codebook
    indices = np.asarray(start.indices)
    if indices.shape != (container.ROW_COUNT,) or indices.dtype.kind not in "iu":
        raise ExecutionError("Lloyd materialized indices must be an integer N-vector")
    if indices.dtype.kind == "i" and np.any(indices < 0):
        raise ExecutionError("Lloyd materialized index is negative")
    if int(indices.max()) >= start.spec.codebook_size:
        raise ExecutionError("Lloyd materialized index exceeds its codebook")
    key = lloyd.codebook_state_sha256(start.codebook)
    if cache is not None and key in cache:
        nearest, objective = cache[key]
    else:
        nearest, objective = _nearest_indices_and_sse(stage_input, start.codebook)
        if cache is not None:
            cache[key] = (nearest, objective)
    if not np.array_equal(indices.astype(np.uint32), nearest):
        raise ExecutionError("Lloyd terminal indices are not exact nearest-label assignments")
    terminal_sse = _exact_nonnegative_int(
        start.terminal_sse, "Lloyd terminal_sse", maximum=lloyd.UINT64_MAX
    )
    if terminal_sse != objective:
        raise ExecutionError("Lloyd terminal SSE disagrees with exact row accounting")
    initial = lloyd.initialize_codebook(weighted, start.spec, start_id)
    _, initial_sse = _nearest_indices_and_sse(stage_input, initial)
    initial_sha256 = lloyd.codebook_state_sha256(initial)
    trajectories = _trajectory_records(
        start,
        expected_initial_sha256=initial_sha256,
        expected_initial_sse=initial_sse,
    )
    return {
        "start_id": start_id,
        "status": start.status,
        "converged": start.converged,
        "selected": selected,
        "update_sweeps": _exact_nonnegative_int(start.update_sweeps, "Lloyd update_sweeps"),
        "terminal_sse": terminal_sse,
        "codebook_state_sha256": lloyd.codebook_state_sha256(start.codebook),
        "serialized_codebook_bytes": len(serialized),
        "serialized_codebook_sha256": _sha256(serialized),
        "indices": _array_record(np.ascontiguousarray(indices, dtype="<u4")),
        "trajectory": trajectories,
        "frozen_initial_state_verified": True,
    }


def _stage_record(
    stage: lloyd.StageFitResult,
    stage_input: np.ndarray,
) -> dict[str, Any]:
    if len(stage.starts) != len(lloyd.START_IDS):
        raise ExecutionError("a Lloyd stage must contain exactly four starts")
    if tuple(start.start_id for start in stage.starts) != lloyd.START_IDS:
        raise ExecutionError("Lloyd starts are not in frozen ID order")
    if any(start.spec != stage.spec for start in stage.starts):
        raise ExecutionError("Lloyd start spec differs from its stage")
    weighted = lloyd.collapse_weighted(stage_input)
    cache: dict[str, tuple[np.ndarray, int]] = {}
    records = [
        _start_record(
            start,
            stage_input,
            weighted,
            selected=start.start_id == stage.selected_start_id,
            cache=cache,
        )
        for start in stage.starts
    ]
    if any(not start.converged for start in stage.starts):
        raise ExecutionError("a completed variant contains a nonconvergent required start")
    update_cache: dict[str, np.ndarray] = {}
    for start in stage.starts:
        state_hash = lloyd.codebook_state_sha256(start.codebook)
        if state_hash not in update_cache:
            update_cache[state_hash] = _exact_lloyd_update(
                weighted, start.codebook, stage.spec
            )
        updated = update_cache[state_hash]
        if not np.array_equal(updated, start.codebook):
            raise ExecutionError("claimed fixed Lloyd start changes under one exact update")
    winner = min(
        stage.starts,
        key=lambda value: (
            value.terminal_sse,
            value.serialized_codebook,
            tuple(int(index) for index in value.indices),
            value.start_id,
        ),
    )
    if stage.selected_start_id != winner.start_id:
        raise ExecutionError("Lloyd stage winner differs from the frozen exact tie order")
    if (
        stage.terminal_sse != winner.terminal_sse
        or not np.array_equal(stage.codebook, winner.codebook)
        or not np.array_equal(stage.indices, winner.indices)
    ):
        raise ExecutionError("selected Lloyd stage state differs from its winning start")
    return {
        "spec": {
            "family_id": stage.spec.family_id,
            "matched_stage_count": stage.spec.matched_stage_count,
            "base_k": stage.spec.base_k,
            "actual_stage_index": stage.spec.actual_stage_index,
            "codebook_size": stage.spec.codebook_size,
        },
        "stage_input": _array_record(np.ascontiguousarray(stage_input, dtype="<i4")),
        "selected_start_id": stage.selected_start_id,
        "terminal_sse": stage.terminal_sse,
        "starts": records,
        "all_four_starts_fixed": True,
    }


def _fit_record(
    spec: VariantSpec,
    fit: lloyd.VariantFitResult,
    source_rgb: np.ndarray,
) -> tuple[dict[str, Any], container.RGBAccounting]:
    if (
        fit.family_id != spec.family_id
        or fit.matched_stage_count != spec.matched_stage_count
        or fit.base_k != spec.base_k
    ):
        raise ExecutionError(f"fit metadata drifted for {spec.label}")
    descriptor = spec.descriptor
    if len(fit.stages) != descriptor.actual_stage_count or len(fit.stage_indices) != len(
        fit.stages
    ):
        raise ExecutionError(f"fit stage count drifted for {spec.label}")
    source_i32 = np.ascontiguousarray(source_rgb, dtype=np.int32)
    accumulator = np.zeros_like(source_i32)
    stage_records = []
    for stage_index, stage in enumerate(fit.stages):
        stage_input = source_i32 if spec.family_id == 0 else source_i32 - accumulator
        stage_records.append(_stage_record(stage, stage_input))
        contribution = stage.codebook[stage.indices].astype(np.int64, copy=False)
        if spec.family_id == 0:
            updated = contribution
        else:
            updated = accumulator.astype(np.int64) + contribution
        if np.any(updated < lloyd.INT32_MIN) or np.any(updated > lloyd.INT32_MAX):
            raise ExecutionError(f"fit accumulator overflowed signed int32 for {spec.label}")
        accumulator = np.ascontiguousarray(updated, dtype=np.int32)
        if not np.array_equal(fit.stage_indices[stage_index], stage.indices):
            raise ExecutionError(f"fit stage-index alias drifted for {spec.label}")
    if not np.array_equal(accumulator, fit.raw_accumulator):
        raise ExecutionError(f"fit raw accumulator disagrees for {spec.label}")
    reconstructed = np.ascontiguousarray(np.clip(accumulator, 0, 255), dtype=np.uint8)
    if not np.array_equal(reconstructed, fit.reconstructed_rgb):
        raise ExecutionError(f"fit clipped reconstruction disagrees for {spec.label}")
    accounting = container.rgb_accounting(source_rgb, accumulator)
    if spec.family_id == 0 and accounting.clip_components:
        raise ExecutionError("a flat VQ fit unexpectedly requires clipping")
    return (
        {
            "label": spec.label,
            "descriptor": {
                "family_id": descriptor.family_id,
                "actual_stage_count": descriptor.actual_stage_count,
                "matched_rvq_stage_count": descriptor.matched_rvq_stage_count,
                "base_k": descriptor.base_k,
                "codebook_entries": descriptor.codebook_entries,
                "descriptor_bytes": len(descriptor.to_bytes()),
                "descriptor_sha256": _sha256(descriptor.to_bytes()),
            },
            "stages": stage_records,
            "all_lloyd_starts_fixed": True,
            "raw_accumulator": _array_record(
                np.ascontiguousarray(fit.raw_accumulator, dtype="<i4")
            ),
            "reconstructed_rgb": _array_record(
                np.ascontiguousarray(fit.reconstructed_rgb, dtype=np.uint8)
            ),
            "rgb_accounting": accounting.to_dict(),
        },
        accounting,
    )


def _failure_start_records(
    starts: tuple[lloyd.StartResult, ...],
    stage_input: np.ndarray,
    expected_spec: lloyd.LloydSpec,
) -> list[dict[str, Any]]:
    if len(starts) != len(lloyd.START_IDS):
        raise ExecutionError("nonconvergence result does not contain all four starts")
    if tuple(start.start_id for start in starts) != lloyd.START_IDS:
        raise ExecutionError("nonconvergence starts are not in frozen ID order")
    if any(start.spec != expected_spec for start in starts):
        raise ExecutionError("nonconvergence start spec differs from the failed stage")
    if all(start.converged for start in starts):
        raise ExecutionError("nonconvergence exception contains no failed start")
    weighted = lloyd.collapse_weighted(stage_input)
    cache: dict[str, tuple[np.ndarray, int]] = {}
    records = [
        _start_record(
            start,
            stage_input,
            weighted,
            selected=False,
            cache=cache,
        )
        for start in starts
    ]
    for start, record in zip(starts, records, strict=True):
        updated = _exact_lloyd_update(weighted, start.codebook, expected_spec)
        updated_hash = lloyd.codebook_state_sha256(updated)
        if start.status == "fixed":
            if not np.array_equal(updated, start.codebook):
                raise ExecutionError("converged sibling start is not an exact fixed point")
        elif start.status == "cycle":
            hashes = [item["codebook_sha256"] for item in record["trajectory"]]
            first = hashes.index(hashes[-1])
            if first + 1 >= len(hashes) - 1 or updated_hash != hashes[first + 1]:
                raise ExecutionError("cycle terminal update does not follow its repeated state")
        else:
            if np.array_equal(updated, start.codebook):
                raise ExecutionError(
                    "capped Lloyd terminal is an exact fixed point after sweep 100"
                )
            record["post_cap_update_sha256"] = updated_hash
    return records


def _fit_variant(
    source_rgb: np.ndarray,
    spec: VariantSpec,
    fitter: lloyd.StartFitter | None,
) -> lloyd.VariantFitResult:
    if spec.family_id == 0:
        stage_spec = lloyd.LloydSpec(
            0,
            spec.matched_stage_count,
            spec.base_k,
            0,
            (2 * spec.matched_stage_count - 1) * spec.base_k,
        )
        try:
            stage = lloyd.fit_stage(source_rgb, stage_spec, fitter=fitter)
        except lloyd.LloydConvergenceError as exc:
            raise _RequiredStartFailure(exc.starts, source_rgb, stage_spec) from exc
        accumulator = stage.codebook[stage.indices].astype(np.int32, copy=False)
        return lloyd.VariantFitResult(
            0,
            spec.matched_stage_count,
            spec.base_k,
            (stage,),
            _readonly(accumulator, dtype=np.int32),
            _readonly(np.clip(accumulator, 0, 255), dtype=np.uint8),
            (_readonly(stage.indices, dtype=np.uint32),),
        )

    source = np.ascontiguousarray(source_rgb, dtype=np.int32)
    accumulator = np.zeros_like(source)
    stages: list[lloyd.StageFitResult] = []
    indices: list[np.ndarray] = []
    completed_stage_records: list[dict[str, Any]] = []
    for stage_index in range(spec.matched_stage_count):
        residual64 = source.astype(np.int64) - accumulator.astype(np.int64)
        if np.any(residual64 < lloyd.INT32_MIN) or np.any(residual64 > lloyd.INT32_MAX):
            raise ExecutionError("sequential RVQ residual is outside signed int32")
        stage_input = np.ascontiguousarray(residual64, dtype=np.int32)
        stage_spec = lloyd.LloydSpec(
            1,
            spec.matched_stage_count,
            spec.base_k,
            stage_index,
            spec.base_k,
        )
        try:
            stage = lloyd.fit_stage(stage_input, stage_spec, fitter=fitter)
        except lloyd.LloydConvergenceError as exc:
            raise _RequiredStartFailure(
                exc.starts,
                stage_input,
                stage_spec,
                tuple(completed_stage_records),
            ) from exc
        if stage_index + 1 < spec.matched_stage_count:
            completed_stage_records.append(_stage_record(stage, stage_input))
        contribution = stage.codebook[stage.indices].astype(np.int64, copy=False)
        updated = accumulator.astype(np.int64) + contribution
        if np.any(updated < lloyd.INT32_MIN) or np.any(updated > lloyd.INT32_MAX):
            raise ExecutionError("sequential RVQ accumulator is outside signed int32")
        accumulator = np.ascontiguousarray(updated, dtype=np.int32)
        stages.append(stage)
        indices.append(_readonly(stage.indices, dtype=np.uint32))
    return lloyd.VariantFitResult(
        1,
        spec.matched_stage_count,
        spec.base_k,
        tuple(stages),
        _readonly(accumulator, dtype=np.int32),
        _readonly(np.clip(accumulator, 0, 255), dtype=np.uint8),
        tuple(indices),
    )


def _assert_symbol_arrays_equal(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    names: Sequence[str],
    context: str,
) -> None:
    for name in names:
        if not np.array_equal(left[name], right[name]):
            raise ExecutionError(f"{context} differs in decoded {name}")


def _assert_decoded_parity(
    python_value: container.DecodedContainer,
    injected_value: container.DecodedContainer,
    label: str,
) -> None:
    if (
        python_value.canonical_blob_sha256 != injected_value.canonical_blob_sha256
        or python_value.symbol_sha256 != injected_value.symbol_sha256
        or python_value.parsed.descriptor != injected_value.parsed.descriptor
        or python_value.rgb_accounting != injected_value.rgb_accounting
        or python_value.used_entries_per_stage != injected_value.used_entries_per_stage
        or python_value.unique_index_tuples != injected_value.unique_index_tuples
        or python_value.coder_certificates != injected_value.coder_certificates
    ):
        raise ExecutionError(f"Python/injected cold-decode scalar parity failed for {label}")
    _assert_symbol_arrays_equal(
        python_value.symbols,
        injected_value.symbols,
        container.SYMBOL_CHANNELS,
        f"Python/injected {label}",
    )
    for stage, (left, right) in enumerate(
        zip(python_value.codebooks, injected_value.codebooks, strict=True)
    ):
        if not np.array_equal(left, right):
            raise ExecutionError(f"Python/injected {label} codebook {stage} differs")
    for stage, (left, right) in enumerate(
        zip(python_value.indices, injected_value.indices, strict=True)
    ):
        if not np.array_equal(left, right):
            raise ExecutionError(f"Python/injected {label} index {stage} differs")
    if python_value.raw_rgb is None or injected_value.raw_rgb is None or not np.array_equal(
        python_value.raw_rgb, injected_value.raw_rgb
    ):
        raise ExecutionError(f"Python/injected {label} raw RGB differs")


def _encode_and_verify_variant(
    sspl1_blob: bytes,
    source_parts: container.SSPL1Parts,
    source_rgb: np.ndarray,
    source_boundary: Mapping[str, np.ndarray],
    spec: VariantSpec,
    fit: lloyd.VariantFitResult,
    expected_accounting: container.RGBAccounting,
    native_coder: ArithmeticAdapter,
) -> tuple[bytes, dict[str, Any], dict[str, np.ndarray]]:
    descriptor = spec.descriptor
    codebooks = tuple(stage.codebook for stage in fit.stages)
    indices = fit.stage_indices
    try:
        python_blob = container.encode_ssp2v(
            sspl1_blob,
            descriptor,
            codebooks,
            indices,
            arithmetic_encode=arithmetic.encode,
        )
        injected_blob = container.encode_ssp2v(
            sspl1_blob,
            descriptor,
            codebooks,
            indices,
            arithmetic_encode=native_coder.encode,
        )
    except Exception as exc:
        raise ExecutionError(f"SSP2V encode failed for {spec.label}: {exc}") from exc
    if python_blob != injected_blob:
        raise ExecutionError(f"Python/injected complete-blob parity failed for {spec.label}")
    try:
        python_decoded = container.decode_ssp2v(
            python_blob,
            arithmetic_decode=arithmetic.decode,
            arithmetic_encode=arithmetic.encode,
            reference_sspl1=sspl1_blob,
        )
        injected_decoded = container.decode_ssp2v(
            python_blob,
            arithmetic_decode=native_coder.decode,
            arithmetic_encode=native_coder.encode,
            reference_sspl1=sspl1_blob,
        )
    except Exception as exc:
        raise ExecutionError(f"dual SSP2V cold-decode failed for {spec.label}: {exc}") from exc
    _assert_decoded_parity(python_decoded, injected_decoded, spec.label)
    if python_decoded.canonical_blob_sha256 != _sha256(python_blob):
        raise ExecutionError(f"canonical complete-blob hash mismatch for {spec.label}")
    _assert_symbol_arrays_equal(
        python_decoded.symbols,
        source_parts.symbols,
        NON_RGB_CHANNELS,
        spec.label,
    )
    if python_decoded.rgb_accounting != expected_accounting:
        raise ExecutionError(f"decoded RGB accounting differs from fit for {spec.label}")
    if python_decoded.raw_rgb is None or not np.array_equal(
        python_decoded.raw_rgb, fit.raw_accumulator
    ):
        raise ExecutionError(f"decoded raw RGB differs from fit for {spec.label}")
    decoded_rgb = np.column_stack(
        (
            python_decoded.symbols["red"],
            python_decoded.symbols["green"],
            python_decoded.symbols["blue"],
        )
    ).astype(np.uint8)
    if not np.array_equal(decoded_rgb, fit.reconstructed_rgb):
        raise ExecutionError(f"decoded clipped RGB differs from fit for {spec.label}")
    if any(not certificate.get("inequality_holds") for certificate in python_decoded.coder_certificates):
        raise ExecutionError(f"arithmetic certificate failed for {spec.label}")

    candidate_boundary = boundary_arrays(python_decoded.parsed.header, python_decoded.symbols)
    native_boundary = boundary_arrays(injected_decoded.parsed.header, injected_decoded.symbols)
    for name in BOUNDARY_NAMES:
        if not np.array_equal(candidate_boundary[name], native_boundary[name]):
            raise ExecutionError(f"Python/injected boundary differs for {spec.label}/{name}")
    for name in ("means", "log_scales", "rotations"):
        if not np.array_equal(candidate_boundary[name], source_boundary[name]):
            raise ExecutionError(f"decoded non-RGB boundary differs for {spec.label}/{name}")

    parsed = python_decoded.parsed
    components = _container_component_record(parsed, python_blob)
    if components["bytes"]["codebook_raw"] != descriptor.codebook_bytes:
        raise ExecutionError(f"wire codebook byte count drifted for {spec.label}")
    stream_record = {
        "magic": "SSP2V",
        "complete_bytes": len(python_blob),
        "blob_sha256": _sha256(python_blob),
        "symbol_sha256": python_decoded.symbol_sha256,
        "boundary": _boundary_record(candidate_boundary),
        "components": components,
        "used_entries_per_stage": list(python_decoded.used_entries_per_stage),
        "unique_index_tuples": python_decoded.unique_index_tuples,
        "coder_certificates": list(python_decoded.coder_certificates),
        "python_injected_encode_parity": True,
        "python_injected_cold_decode_parity": True,
        "canonical_decode_to_reencode": True,
        "non_rgb_symbols_exact": True,
        "non_rgb_boundary_exact": True,
        "fit_decode_rgb_exact": True,
        "rgb_accounting": python_decoded.rgb_accounting.to_dict()
        if python_decoded.rgb_accounting is not None
        else None,
        "source_rgb": _array_record(source_rgb),
    }
    return python_blob, stream_record, candidate_boundary


def _encode_and_verify_ssp2z(
    sspl1_blob: bytes,
    source_parts: container.SSPL1Parts,
    source_boundary: Mapping[str, np.ndarray],
) -> tuple[bytes, dict[str, Any], dict[str, np.ndarray]]:
    try:
        blob = container.encode_ssp2z(sspl1_blob)
        decoded = container.decode_ssp2z(blob, reference_sspl1=sspl1_blob)
    except Exception as exc:
        raise ExecutionError(f"SSP2Z construction/cold-decode failed: {exc}") from exc
    if decoded.canonical_blob_sha256 != _sha256(blob):
        raise ExecutionError("SSP2Z canonical complete-blob hash mismatch")
    _assert_symbol_arrays_equal(
        decoded.symbols,
        source_parts.symbols,
        container.SYMBOL_CHANNELS,
        "SSP2Z",
    )
    arrays = boundary_arrays(decoded.parsed.header, decoded.symbols)
    if any(not np.array_equal(arrays[name], source_boundary[name]) for name in BOUNDARY_NAMES):
        raise ExecutionError("SSP2Z decoded boundary differs from SSPL1")
    record = {
        "magic": "SSP2Z",
        "complete_bytes": len(blob),
        "blob_sha256": _sha256(blob),
        "symbol_sha256": decoded.symbol_sha256,
        "boundary": _boundary_record(arrays),
        "components": _container_component_record(decoded.parsed, blob),
        "canonical_decode_to_reencode": True,
        "all_symbols_exact": True,
        "boundary_exact": True,
    }
    return blob, record, arrays


def _matched_codebook_records(variant_records: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for stages in (2, 3):
        for base_k in (64, 128):
            flat_label = f"flat_s{stages}_k{base_k}_l{(2 * stages - 1) * base_k}"
            rvq_label = f"rvq_s{stages}_k{base_k}"
            flat_bytes = variant_records[flat_label]["stream"]["components"]["bytes"][
                "codebook_raw"
            ]
            rvq_bytes = variant_records[rvq_label]["stream"]["components"]["bytes"][
                "codebook_raw"
            ]
            expected = 3 * (2 * stages - 1) * base_k
            if flat_bytes != rvq_bytes or flat_bytes != expected:
                raise ExecutionError("matched flat/RVQ codebook byte equality failed")
            result.append(
                {
                    "matched_stage_count": stages,
                    "base_k": base_k,
                    "flat_label": flat_label,
                    "rvq_label": rvq_label,
                    "flat_codebook_bytes": flat_bytes,
                    "rvq_codebook_bytes": rvq_bytes,
                    "expected_codebook_bytes": expected,
                    "equal": True,
                }
            )
    return result


def _candidate_inventory(
    baseline_records: Mapping[str, Mapping[str, Any]],
    variant_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = [
        {
            "name": name,
            "complete_bytes": baseline_records[name]["complete_bytes"],
            "blob_sha256": baseline_records[name]["blob_sha256"],
        }
        for name in EXACT_FORMAT_ORDER
    ]
    result.extend(
        {
            "name": label,
            "complete_bytes": variant_records[label]["stream"]["complete_bytes"],
            "blob_sha256": variant_records[label]["stream"]["blob_sha256"],
        }
        for label in VARIANT_LABELS
    )
    return result


def _select_exact_q(
    baseline_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = min(
        enumerate(EXACT_FORMAT_ORDER),
        key=lambda item: (baseline_records[item[1]]["complete_bytes"], item[0]),
    )[1]
    record = baseline_records[selected]
    boundary_hash = (
        record["boundary"]["state_sha256"]
        if selected in ("sspl1", "ssp2z")
        else record["boundary_state_sha256"]
    )
    return {
        "name": selected,
        "tie_rank": EXACT_FORMAT_ORDER.index(selected),
        "complete_bytes": record["complete_bytes"],
        "blob_sha256": record["blob_sha256"],
        "symbol_sha256": record["symbol_sha256"],
        "boundary_state_sha256": boundary_hash,
        "selection_rule": "minimum_complete_bytes_then_frozen_six_format_order",
    }


def execute_cell(
    sspl1_blob: bytes,
    native_coder: ArithmeticAdapter,
    *,
    reproduce_comp009: Comp009Reproducer,
    expected_comp009: Mapping[str, Comp009Authority | Mapping[str, Any]],
    fitter: lloyd.StartFitter | None = None,
) -> CellResult:
    """Execute one complete target-free COMP-011 cell entirely from supplied bytes.

    ``reproduce_comp009`` is the sole route for SSP2E/S/L/F.  It receives the immutable source
    bytes and must return complete streams plus already-performed canonical dual-cold-decode
    attestations.  The executor checks every attestation and byte identity against
    ``expected_comp009``.  It never discovers or opens an upstream artifact.
    """

    if not isinstance(sspl1_blob, bytes):
        raise ExecutionError("execute_cell requires immutable SSPL1 bytes")
    if not callable(getattr(native_coder, "encode", None)) or not callable(
        getattr(native_coder, "decode", None)
    ):
        raise ExecutionError("native_coder must expose callable encode and decode methods")
    if fitter is not None and not callable(getattr(fitter, "fit_start", None)):
        raise ExecutionError("fitter must expose a callable fit_start method")
    try:
        parts = container.extract_sspl1_parts(sspl1_blob)
    except Exception as exc:
        raise ExecutionError(f"invalid canonical SSPL1 source: {exc}") from exc
    if parts.header.n != container.ROW_COUNT:
        raise ExecutionError(f"COMP-011 execution requires N={container.ROW_COUNT}")

    source_rgb = _source_rgb(parts)
    source_boundary = boundary_arrays(parts.header, parts.symbols)
    source_boundary_record = _boundary_record(source_boundary)
    source_record = {
        "magic": "SSPL1",
        "complete_bytes": len(sspl1_blob),
        "blob_sha256": parts.blob_sha256,
        "symbol_sha256": parts.symbol_sha256,
        "boundary": source_boundary_record,
        "bit_tuple": list(parts.header.bits),
        "n": parts.header.n,
        "rgb": _array_record(source_rgb),
        "components": _sspl1_component_records(sspl1_blob, parts),
        "canonical_cold_parse": True,
    }
    z_blob, z_record, z_boundary = _encode_and_verify_ssp2z(
        sspl1_blob, parts, source_boundary
    )
    comp009_blobs, comp009_records = _reproduce_comp009(
        sspl1_blob,
        reproduce_comp009,
        expected_comp009,
        source_parts=parts,
        source_boundary=source_boundary,
        source_symbol_sha256=parts.symbol_sha256,
        source_boundary_sha256=source_boundary_record["state_sha256"],
        native_coder=native_coder,
    )
    baseline_blobs = {
        "sspl1": sspl1_blob,
        "ssp2z": z_blob,
        **comp009_blobs,
    }
    baseline_records = {
        "sspl1": source_record,
        "ssp2z": z_record,
        **comp009_records,
    }

    fits: dict[str, lloyd.VariantFitResult] = {}
    fit_records: dict[str, dict[str, Any]] = {}
    accountings: dict[str, container.RGBAccounting] = {}
    for spec in VARIANT_SPECS:
        try:
            fit = _fit_variant(source_rgb, spec, fitter)
        except _RequiredStartFailure as exc:
            failed_start_records = _failure_start_records(
                exc.starts, exc.stage_input, exc.spec
            )
            failure_record = _seal(
                {
                    "schema": METHOD_FAILURE_SCHEMA,
                    "task_sha256": TASK_SHA256,
                    "status": "valid_method_failure",
                    "method_state": "required_lloyd_nonconvergence",
                    "source": source_record,
                    "exact_formats": baseline_records,
                    "completed_variant_fits": fit_records,
                    "failure": {
                        "variant_label": spec.label,
                        "family_id": spec.family_id,
                        "matched_stage_count": spec.matched_stage_count,
                        "base_k": spec.base_k,
                        "actual_stage_index": exc.spec.actual_stage_index,
                        "completed_stages": list(exc.completed_stages),
                        "failed_stage_input": _array_record(exc.stage_input),
                        "starts": failed_start_records,
                        "failed_start_ids": [
                            start.start_id for start in exc.starts if not start.converged
                        ],
                    },
                    "target_input_surface": "absent",
                    "target_accessed": False,
                    "target_import_authorized": False,
                    "partial_ssp2v_candidate_set_emitted": False,
                    "claim_scope": "valid-preregistered-method-failure",
                }
            )
            return LloydMethodFailure(failure_record, baseline_blobs, exc.starts)
        except Exception as exc:
            raise ExecutionError(f"Lloyd fit failed invalidly for {spec.label}: {exc}") from exc
        try:
            fit_record, accounting = _fit_record(spec, fit, source_rgb)
        except Exception as exc:
            if isinstance(exc, ExecutionError):
                raise
            raise ExecutionError(f"fit validation failed for {spec.label}: {exc}") from exc
        fits[spec.label] = fit
        fit_records[spec.label] = fit_record
        accountings[spec.label] = accounting

    variant_blobs: dict[str, bytes] = {}
    variant_boundaries: dict[str, dict[str, np.ndarray]] = {}
    variant_records: dict[str, dict[str, Any]] = {}
    for spec in VARIANT_SPECS:
        blob, stream_record, arrays = _encode_and_verify_variant(
            sspl1_blob,
            parts,
            source_rgb,
            source_boundary,
            spec,
            fits[spec.label],
            accountings[spec.label],
            native_coder,
        )
        variant_blobs[spec.label] = blob
        variant_boundaries[spec.label] = arrays
        variant_records[spec.label] = {
            **fit_records[spec.label],
            "stream": stream_record,
        }

    matched_records = _matched_codebook_records(variant_records)
    inventory = _candidate_inventory(baseline_records, variant_records)
    selected_q = _select_exact_q(baseline_records)
    candidate_evidence = {
        "schema": "structsplat.comp011.candidate-evidence.v1",
        "task_sha256": TASK_SHA256,
        "exact_format_order": list(EXACT_FORMAT_ORDER),
        "exact_formats": baseline_records,
        "selected_q": selected_q,
        "variant_order": list(VARIANT_LABELS),
        "variants": variant_records,
        "matched_codebook_bytes": matched_records,
    }
    candidate_set_sha256 = _sha256(_canonical_json(candidate_evidence))
    record = _seal(
        {
            "schema": EXECUTION_SCHEMA,
            "task_sha256": TASK_SHA256,
            "status": "valid_complete_candidate_set",
            "source": source_record,
            "exact_format_order": list(EXACT_FORMAT_ORDER),
            "exact_formats": baseline_records,
            "variant_order": list(VARIANT_LABELS),
            "variants": variant_records,
            "matched_codebook_bytes": matched_records,
            "selected_q": selected_q,
            "candidate_inventory": inventory,
            "candidate_evidence_schema": "structsplat.comp011.candidate-evidence.v1",
            "candidate_set_sha256": candidate_set_sha256,
            "candidate_set_seals_full_normalized_evidence": True,
            "target_input_surface": "absent",
            "target_accessed": False,
            "integrity": {
                "canonical_sspl1_cold_parse": True,
                "ssp2z_exact_symbols_and_boundary": True,
                "all_four_comp009_authorities_reproduced": True,
                "all_four_comp009_independently_dual_decoded": True,
                "comp009_captured_source_and_execution_bound": True,
                "all_eight_variants_fit_before_encoding": True,
                "all_required_lloyd_starts_fixed": True,
                "all_frozen_lloyd_initial_states_verified": True,
                "all_completed_rvq_stages_validated_before_next": True,
                "all_python_injected_arithmetic_payloads_equal": True,
                "all_python_injected_cold_decodes_equal": True,
                "all_variant_blobs_canonical": True,
                "all_non_rgb_symbols_exact": True,
                "all_non_rgb_boundaries_exact": True,
                "all_fit_decode_rgb_and_accounting_exact": True,
                "all_matched_codebook_byte_charges_equal": True,
                "selected_q_recomputed_from_all_six_exact_formats": True,
                "candidate_seal_binds_full_normalized_evidence": True,
                "complete_candidate_set_present": True,
            },
            "claim_scope": "target-free-development-candidate-set",
        }
    )
    if set(record["integrity"].values()) != {True}:
        raise ExecutionError("one or more cell execution integrity predicates failed")
    boundaries = {"sspl1": source_boundary, "ssp2z": z_boundary, **variant_boundaries}
    return CellExecution(record, baseline_blobs, variant_blobs, boundaries, fits)


__all__ = [
    "ArithmeticAdapter",
    "CellExecution",
    "CellResult",
    "Comp009Authority",
    "Comp009Reproducer",
    "COMP009_CAPTURED_SOURCE_SHA256",
    "COMP009_MAGICS",
    "COMP009_NAMES",
    "EXACT_FORMAT_ORDER",
    "EXECUTION_SCHEMA",
    "ExecutionError",
    "LloydMethodFailure",
    "METHOD_FAILURE_SCHEMA",
    "ReproducedComp009Stream",
    "TASK_SHA256",
    "VARIANT_LABELS",
    "VARIANT_SPECS",
    "VariantSpec",
    "boundary_arrays",
    "comp009_verification_sha256",
    "execute_cell",
    "verify_record",
]
