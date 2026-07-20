"""End-to-end execution boundary for one frozen COMP-009 codec cell.

This module is deliberately free of lifecycle and filesystem policy.  It accepts one canonical
SSPL1 byte string plus an already-built native arithmetic adapter, executes the two contextual
fits and both factorized controls, and returns deterministic records and byte strings for the
lifecycle runner to persist.  In particular, model selection is made from *actual* arithmetic
payload lengths, not from the Q24 fitting objective.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from benchmarks import sgi_entropy_oracle as oracle
from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_model as model


EXECUTION_SCHEMA = "structsplat.comp009.ssp2e-execution.v1"
STREAM_NAMES = ("ssp2e", "ssp2s", "ssp2l", "ssp2f")


class ExecutionError(RuntimeError):
    """An end-to-end COMP-009 execution invariant failed."""


class ArithmeticAdapter(Protocol):
    """The Python/native arithmetic surface used by the container implementation."""

    def encode(self, symbols: np.ndarray, cdf_rows: np.ndarray) -> bytes: ...

    def decode(self, payload: bytes, cdf_rows: np.ndarray, n: int) -> np.ndarray: ...


@dataclass(frozen=True)
class CellExecution:
    """Deterministic in-memory products of :func:`execute_cell`.

    ``record`` contains only JSON-compatible values.  The four complete codec ``blobs`` and the
    independently reconstructed float32 ``boundary_arrays`` stay binary so a lifecycle layer can
    persist them without a lossy JSON conversion.  The two selections retain typed model objects
    for replay code that wants to inspect them directly.
    """

    record: dict[str, Any]
    blobs: dict[str, bytes]
    boundary_arrays: dict[str, np.ndarray]
    true_selection: model.ModelSelection
    shuffled_selection: model.ModelSelection


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _alphabet_by_channel(header: container.ContainerHeader) -> dict[str, int]:
    return {
        "scale_x": 1 << header.bits[1],
        "scale_y": 1 << header.bits[1],
        "red": 1 << header.bits[3],
        "green": 1 << header.bits[3],
        "blue": 1 << header.bits[3],
    }


def _context_sha256(cells: np.ndarray) -> str:
    values = np.ascontiguousarray(cells, dtype="<u2")
    return _sha256(values.tobytes(order="C"))


def _array_state_digest(arrays: dict[str, np.ndarray]) -> str:
    """Match the COMP-008 decoded-boundary state digest exactly."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _component_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
        "sha256": _sha256(array.tobytes(order="C")),
    }


def boundary_arrays(
    header: container.ContainerHeader,
    symbols: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Independently reconstruct the exact float32 field boundary from absolute symbols."""
    if set(symbols) != set(container.SYMBOL_CHANNELS):
        raise ExecutionError("boundary reconstruction requires all eight absolute symbol arrays")
    bm, bs, br, bc = header.bits
    mean_symbols = np.column_stack((symbols["mean_x"], symbols["mean_y"]))
    scale_symbols = np.column_stack((symbols["scale_x"], symbols["scale_y"]))
    color_symbols = np.column_stack((symbols["red"], symbols["green"], symbols["blue"]))

    def dequantize(
        quantized: np.ndarray,
        lower: tuple[int, ...] | tuple[float, ...],
        upper: tuple[int, ...] | tuple[float, ...],
        bits: int,
    ) -> np.ndarray:
        levels = (1 << bits) - 1
        return (
            np.asarray(lower, dtype=np.float64)
            + np.asarray(quantized, dtype=np.float64)
            / levels
            * (np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64))
        ).astype(np.float32)

    arrays = {
        "means": dequantize(mean_symbols, header.means_lo, header.means_hi, bm),
        "log_scales": dequantize(scale_symbols, header.scale_lo, header.scale_hi, bs),
        "rotations": (np.asarray(symbols["rotation"], dtype=np.float64) / (1 << br) * np.pi).astype(
            np.float32
        ),
        "colors": dequantize(color_symbols, header.color_lo, header.color_hi, bc),
    }
    normalized = {name: np.ascontiguousarray(value) for name, value in arrays.items()}
    if any(value.dtype != np.float32 for value in normalized.values()):
        raise ExecutionError("decoded boundary reconstruction did not produce float32 arrays")
    if any(not np.isfinite(value).all() for value in normalized.values()):
        raise ExecutionError("decoded boundary reconstruction produced a non-finite value")
    return normalized


def _static_state() -> tuple[
    dict[int, assets.StaticCDFTable],
    tuple[int, ...],
    dict[str, Any],
]:
    loaded_tables = dict(assets.load_tables())
    loaded_costs = assets.load_costs()
    if set(loaded_tables) != set(assets.ALPHABETS):
        raise ExecutionError("static CDF asset is missing a frozen alphabet")
    identity = {
        "cdf_table_id": container.CDF_TABLE_ID,
        "cdf_sha256": assets.CDF_ASSET_SHA256,
        "cdf_bytes": assets.CDF_ASSET_SIZE,
        "cost_sha256": assets.COST_ASSET_SHA256,
        "cost_bytes": assets.COST_ASSET_SIZE,
        "total_frequency": assets.TOTAL_FREQUENCY,
        "cost_q": assets.COST_Q,
        "alphabets": list(assets.ALPHABETS),
    }
    return loaded_tables, loaded_costs, identity


def _model_frequency_provider(
    tables: dict[int, assets.StaticCDFTable],
    alphabets: dict[str, int],
):
    def provide(channel: str, location: int, scale: int) -> tuple[int, ...]:
        return tables[alphabets[channel]].frequency_row(location, scale)

    return provide


def _container_frequency_provider(tables: dict[int, assets.StaticCDFTable]):
    def provide(alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        return tables[alphabet].frequency_row(location, scale)

    return provide


def _contextual_cdf_rows(
    fitted: model.SSP2EModel,
    contexts: np.ndarray,
    alphabets: dict[str, int],
    frequency_provider,
) -> dict[str, np.ndarray]:
    inferred_model = model.infer_row_parameters(fitted, contexts)
    inferred_container = container.infer_head_rows(
        fitted.grid_raw, fitted.head_raw, np.asarray(contexts, dtype=np.uint16)
    )
    result: dict[str, np.ndarray] = {}
    for name in model.CHANNELS:
        for axis in range(2):
            if not np.array_equal(inferred_model[name][axis], inferred_container[name][axis]):
                raise ExecutionError(f"model/container inference parity failed for {name}")
        locations, scales = inferred_model[name]
        result[name] = container.cdf_rows_from_models(
            alphabets[name], locations, scales, frequency_provider
        )
    return result


class _ActualLengthSelector:
    """Encode each ordered fitted start with both coders and retain its byte diagnostics."""

    def __init__(
        self,
        *,
        contexts: np.ndarray,
        symbols: dict[str, np.ndarray],
        alphabets: dict[str, int],
        frequency_provider,
        native: ArithmeticAdapter,
    ) -> None:
        self._contexts = contexts
        self._symbols = symbols
        self._alphabets = alphabets
        self._frequency_provider = frequency_provider
        self._native = native
        self.records: list[dict[str, Any]] = []

    def __call__(self, fitted: model.SSP2EModel) -> dict[str, int]:
        start_id = len(self.records)
        if start_id >= 8:
            raise ExecutionError("model selector encoded more than eight starts")
        rows = _contextual_cdf_rows(
            fitted,
            self._contexts,
            self._alphabets,
            self._frequency_provider,
        )
        lengths: dict[str, int] = {}
        payload_hashes: dict[str, str] = {}
        for name in model.CHANNELS:
            values = np.ascontiguousarray(self._symbols[name], dtype=np.uint32)
            python_payload = arithmetic.encode(values, rows[name])
            native_payload = self._native.encode(values, rows[name])
            if python_payload != native_payload:
                raise ExecutionError(
                    f"Python/native arithmetic byte parity failed for start {start_id}/{name}"
                )
            lengths[name] = len(python_payload)
            payload_hashes[name] = _sha256(python_payload)
        self.records.append(
            {
                "start_id": start_id,
                "payload_lengths": lengths,
                "total_payload_bytes": sum(lengths.values()),
                "payload_sha256": payload_hashes,
                "python_native_byte_parity": True,
            }
        )
        return lengths


def _trajectory_record(fit: model.FitResult) -> list[dict[str, int]]:
    return [
        {
            "sweep": item.sweep,
            "objective_q24": item.objective_q24,
            "strict_updates": item.strict_updates,
            "head_updates": item.head_updates,
            "grid_updates": item.grid_updates,
        }
        for item in fit.trajectory
    ]


def _selection_record(
    objective: model.ModelObjective,
    selection: model.ModelSelection,
    encoding_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(encoding_records) != 8:
        raise ExecutionError("model selection did not encode exactly eight starts")
    starts = []
    for encoded, payload_record in zip(selection.encoded_starts, encoding_records, strict=True):
        fit = encoded.fit
        if fit.start_id != payload_record["start_id"]:
            raise ExecutionError("start encoding order drifted from fitted start order")
        fixed = model.verify_fixed_point(objective, fit.model)
        if not fixed:
            raise ExecutionError(f"fitted start {fit.start_id} is not an exact fixed point")
        lengths = tuple(payload_record["payload_lengths"][name] for name in model.CHANNELS)
        if lengths != encoded.payload_lengths:
            raise ExecutionError("persisted payload lengths differ from selection inputs")
        starts.append(
            {
                **payload_record,
                "objective_q24": fit.objective_q24,
                "sweeps": fit.sweeps,
                "trajectory": _trajectory_record(fit),
                "grid_sha256": _sha256(fit.model.grid_raw),
                "head_sha256": _sha256(fit.model.head_raw),
                "model_sha256": _sha256(fit.model.lexicographic_bytes),
                "fixed_point_verified": True,
            }
        )
    return {
        "start_count": len(starts),
        "starts": starts,
        "restart_spread": {
            "objective_q24_min": min(item["objective_q24"] for item in starts),
            "objective_q24_max": max(item["objective_q24"] for item in starts),
            "objective_q24_range": max(item["objective_q24"] for item in starts)
            - min(item["objective_q24"] for item in starts),
            "payload_bytes_min": min(item["total_payload_bytes"] for item in starts),
            "payload_bytes_max": max(item["total_payload_bytes"] for item in starts),
            "payload_bytes_range": max(item["total_payload_bytes"] for item in starts)
            - min(item["total_payload_bytes"] for item in starts),
            "sweeps_min": min(item["sweeps"] for item in starts),
            "sweeps_max": max(item["sweeps"] for item in starts),
            "sweeps_range": max(item["sweeps"] for item in starts)
            - min(item["sweeps"] for item in starts),
        },
        "selection_reason": selection.selection_reason,
        "selected_grid_sha256": _sha256(selection.selected.fit.model.grid_raw),
        "selected_head_sha256": _sha256(selection.selected.fit.model.head_raw),
        "selected_model_sha256": _sha256(selection.selected.fit.model.lexicographic_bytes),
    }


def _fit_selection(
    *,
    contexts: np.ndarray,
    parts: container.SSPL1Parts,
    alphabets: dict[str, int],
    model_frequency_provider,
    container_frequency_provider,
    costs: tuple[int, ...],
    native: ArithmeticAdapter,
) -> tuple[model.ModelObjective, model.ModelSelection, dict[str, Any]]:
    modeled_symbols = {name: parts.symbols[name] for name in model.CHANNELS}
    histograms = model.aggregate_context_histograms(contexts, modeled_symbols, alphabets)
    objective = model.build_model_objective(histograms, model_frequency_provider, costs)
    fits = model.fit_eight_starts(objective)
    selector = _ActualLengthSelector(
        contexts=contexts,
        symbols=modeled_symbols,
        alphabets=alphabets,
        frequency_provider=container_frequency_provider,
        native=native,
    )
    selection = model.select_fitted_model(fits, selector)
    return objective, selection, _selection_record(objective, selection, selector.records)


def _assert_symbol_identity(
    decoded: container.DecodedContainer,
    parts: container.SSPL1Parts,
    stream_name: str,
) -> None:
    if decoded.symbol_sha256 != parts.symbol_sha256:
        raise ExecutionError(f"{stream_name} cold-decode symbol hash differs from SSPL1")
    for name in container.SYMBOL_CHANNELS:
        if not np.array_equal(decoded.symbols[name], parts.symbols[name]):
            raise ExecutionError(f"{stream_name} cold-decode differs in {name}")


def _encode_complete_blobs(
    source: bytes,
    true_model: model.SSP2EModel,
    shuffled_model: model.SSP2EModel,
    global_models: dict[str, tuple[int, int]],
    frequency_provider,
    native: ArithmeticAdapter,
) -> tuple[dict[str, bytes], dict[str, container.DecodedContainer]]:
    def encode_all(encoder) -> dict[str, bytes]:
        return {
            "ssp2e": container.encode_contextual(
                source,
                true_model.grid_raw,
                true_model.head_raw,
                arithmetic_encode=encoder,
                frequency_provider=frequency_provider,
                magic=b"SSP2E",
            ),
            "ssp2s": container.encode_contextual(
                source,
                shuffled_model.grid_raw,
                shuffled_model.head_raw,
                arithmetic_encode=encoder,
                frequency_provider=frequency_provider,
                magic=b"SSP2S",
            ),
            "ssp2l": container.encode_ssp2l(
                source,
                global_models,
                arithmetic_encode=encoder,
                frequency_provider=frequency_provider,
            ),
            "ssp2f": container.encode_ssp2f(source, arithmetic_encode=encoder),
        }

    python_blobs = encode_all(arithmetic.encode)
    native_blobs = encode_all(native.encode)
    for name in STREAM_NAMES:
        if python_blobs[name] != native_blobs[name]:
            raise ExecutionError(
                "Python-framed container parity between Python/native arithmetic cores "
                f"failed for {name}"
            )
        container.parse_container(python_blobs[name], reference_sspl1=source)

    python_decoded: dict[str, container.DecodedContainer] = {}
    for name in STREAM_NAMES:
        provider = frequency_provider if name != "ssp2f" else None
        python_value = container.decode_container(
            python_blobs[name],
            arithmetic_decode=arithmetic.decode,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=provider,
        )
        native_value = container.decode_container(
            python_blobs[name],
            arithmetic_decode=native.decode,
            arithmetic_encode=native.encode,
            frequency_provider=provider,
        )
        if python_value.symbol_sha256 != native_value.symbol_sha256:
            raise ExecutionError(f"Python/native cold-decode hash parity failed for {name}")
        for channel in container.SYMBOL_CHANNELS:
            if not np.array_equal(python_value.symbols[channel], native_value.symbols[channel]):
                raise ExecutionError(
                    f"Python/native arithmetic cold-decode differs for {name}/{channel}"
                )
        python_decoded[name] = python_value
    return python_blobs, python_decoded


def _cdf_rows_for_stream(
    parsed: container.ParsedContainer,
    parts: container.SSPL1Parts,
    frequency_provider,
) -> dict[str, np.ndarray]:
    header = parsed.header
    alphabets = _alphabet_by_channel(header)
    if parsed.magic in container.CONTEXTUAL_MAGICS:
        cells = model.position_cells(
            parts.symbols["mean_x"], parts.symbols["mean_y"], header.bits[0]
        )
        if parsed.magic == b"SSP2S":
            cells = cells[container.shuffled_permutation(header.n)]
        fitted = model.SSP2EModel.from_raw(parsed.payloads["grid"], parsed.payloads["head"])
        return _contextual_cdf_rows(fitted, cells, alphabets, frequency_provider)
    if parsed.magic == b"SSP2L":
        globals_by_channel = container.parse_global_model(parsed.payloads["model"])
        return {
            name: container.cdf_rows_from_models(
                alphabets[name],
                np.full(header.n, globals_by_channel[name][0], dtype=np.uint8),
                np.full(header.n, globals_by_channel[name][1], dtype=np.uint8),
                frequency_provider,
            )
            for name in model.CHANNELS
        }
    frequencies = container.parse_empirical_model(parsed.payloads["model"], header)
    result = {}
    for name in model.CHANNELS:
        cdf = np.concatenate(
            (
                np.zeros(1, dtype=np.uint32),
                np.cumsum(frequencies[name], dtype=np.uint32),
            )
        )
        result[name] = np.repeat(cdf[None, :], header.n, axis=0)
    return result


def _stream_record(
    name: str,
    blob: bytes,
    parts: container.SSPL1Parts,
    frequency_provider,
) -> dict[str, Any]:
    parsed = container.parse_container(blob, reference_sspl1=None)
    rows = _cdf_rows_for_stream(parsed, parts, frequency_provider)
    payload_lengths = {channel: len(parsed.payloads[channel]) for channel in model.CHANNELS}
    certificates = {
        channel: arithmetic.coder_certificate(
            parts.symbols[channel], rows[channel], parsed.payloads[channel]
        )
        for channel in model.CHANNELS
    }
    arithmetic_bytes = sum(payload_lengths.values())
    if parsed.magic in container.CONTEXTUAL_MAGICS:
        model_bytes = len(parsed.payloads["grid"]) + len(parsed.payloads["head"])
        modeled_bytes = container.CONTEXTUAL_FIXED_BYTES + arithmetic_bytes
    else:
        model_bytes = len(parsed.payloads["model"])
        modeled_bytes = container.FACTORIZED_PREFIX_BYTES + model_bytes + arithmetic_bytes
    passthrough_bytes = len(parsed.payloads["means"]) + len(parsed.payloads["rotation"])
    if modeled_bytes + passthrough_bytes != len(blob):
        raise ExecutionError(f"complete/modeled byte identity failed for {name}")
    return {
        "magic": parsed.magic.decode("ascii"),
        "complete_bytes": len(blob),
        "blob_sha256": _sha256(blob),
        "modeled_bytes": modeled_bytes,
        "shared_passthrough_bytes": passthrough_bytes,
        "model_bytes": model_bytes,
        "arithmetic_payload_bytes": payload_lengths,
        "arithmetic_payload_sha256": {
            channel: _sha256(parsed.payloads[channel]) for channel in model.CHANNELS
        },
        "arithmetic_bytes": arithmetic_bytes,
        "certificates": certificates,
        "complete_bytes_equal_modeled_plus_passthrough": True,
    }


def execute_cell(
    sspl1_blob: bytes,
    native_coder: ArithmeticAdapter,
    *,
    synthetic: bool = False,
    expected_symbol_sha256: str | None = None,
    expected_boundary_state_sha256: str | None = None,
) -> CellExecution:
    """Execute one synthetic or production COMP-009 cell without filesystem access.

    The caller owns native-binary provenance and all artifact persistence.  This function fails
    before returning unless all eight starts for each contextual variant converge, all start and
    final payloads agree byte-for-byte between Python and native coders, all four complete streams
    cold-decode through both coders, every arithmetic certificate holds, and all decoded boundary
    arrays are exact.
    """
    execution_started = time.perf_counter()
    if not isinstance(sspl1_blob, bytes):
        raise ExecutionError("execute_cell requires one immutable SSPL1 byte string")
    if not isinstance(synthetic, bool):
        raise ExecutionError("synthetic must be exactly boolean")
    if not callable(getattr(native_coder, "encode", None)) or not callable(
        getattr(native_coder, "decode", None)
    ):
        raise ExecutionError("native_coder must expose callable encode and decode methods")

    parts = container.extract_sspl1_parts(sspl1_blob)
    expected_values = {
        "expected_symbol_sha256": expected_symbol_sha256,
        "expected_boundary_state_sha256": expected_boundary_state_sha256,
    }
    if not synthetic:
        if parts.header.n != model.GAUSSIAN_COUNT:
            raise ExecutionError(
                f"production execution requires N={model.GAUSSIAN_COUNT}, found {parts.header.n}"
            )
        if any(value is None for value in expected_values.values()):
            raise ExecutionError(
                "production execution requires the sealed COMP-008 symbol and boundary hashes"
            )
    for name, value in expected_values.items():
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ExecutionError(f"{name} must be a lowercase SHA-256 hex digest")
    if expected_symbol_sha256 is not None and parts.symbol_sha256 != expected_symbol_sha256:
        raise ExecutionError("SSPL1 symbol hash differs from the sealed expected identity")
    source_boundary = boundary_arrays(parts.header, parts.symbols)
    boundary_hash = _array_state_digest(source_boundary)
    if (
        expected_boundary_state_sha256 is not None
        and boundary_hash != expected_boundary_state_sha256
    ):
        raise ExecutionError("SSPL1 boundary hash differs from the sealed expected identity")

    tables, costs, asset_record = _static_state()
    alphabets = _alphabet_by_channel(parts.header)
    model_provider = _model_frequency_provider(tables, alphabets)
    container_provider = _container_frequency_provider(tables)

    true_contexts = model.position_cells(
        parts.symbols["mean_x"], parts.symbols["mean_y"], parts.header.bits[0]
    )
    oracle_contexts = oracle.position_cells(
        parts.symbols["mean_x"], parts.symbols["mean_y"], parts.header.bits[0]
    )
    if not np.array_equal(true_contexts, oracle_contexts):
        raise ExecutionError("model/container position-context derivations disagree")
    if parts.header.n == model.GAUSSIAN_COUNT:
        shuffled_contexts = model.shuffled_position_contexts(true_contexts)
        expected_shuffle = true_contexts[container.shuffled_permutation(parts.header.n)]
        if not np.array_equal(shuffled_contexts, expected_shuffle):
            raise ExecutionError("model/container shuffled-context parity failed")
    else:
        # Small canonical SSPL1 inputs are allowed only for synthetic proof tests.  The production
        # path above additionally verifies the frozen 8192-row permutation hash in the model layer.
        shuffled_contexts = true_contexts[container.shuffled_permutation(parts.header.n)]

    true_fit_started = time.perf_counter()
    true_objective, true_selection, true_diagnostics = _fit_selection(
        contexts=true_contexts,
        parts=parts,
        alphabets=alphabets,
        model_frequency_provider=model_provider,
        container_frequency_provider=container_provider,
        costs=costs,
        native=native_coder,
    )
    true_fit_seconds = time.perf_counter() - true_fit_started
    shuffled_fit_started = time.perf_counter()
    shuffled_objective, shuffled_selection, shuffled_diagnostics = _fit_selection(
        contexts=shuffled_contexts,
        parts=parts,
        alphabets=alphabets,
        model_frequency_provider=model_provider,
        container_frequency_provider=container_provider,
        costs=costs,
        native=native_coder,
    )
    shuffled_fit_seconds = time.perf_counter() - shuffled_fit_started

    control_model_started = time.perf_counter()
    true_pairs = model.best_global_pairs(true_objective)
    shuffled_pairs = model.best_global_pairs(shuffled_objective)
    if true_pairs != shuffled_pairs:
        raise ExecutionError("global SSP2L model changed under a context-only permutation")
    global_models = dict(zip(model.CHANNELS, true_pairs, strict=True))
    control_model_seconds = time.perf_counter() - control_model_started

    encode_started = time.perf_counter()
    blobs, decoded = _encode_complete_blobs(
        sspl1_blob,
        true_selection.selected.fit.model,
        shuffled_selection.selected.fit.model,
        global_models,
        container_provider,
        native_coder,
    )
    complete_encode_and_cold_decode_seconds = time.perf_counter() - encode_started
    for name, value in decoded.items():
        _assert_symbol_identity(value, parts, name)

    decoded_hashes = {}
    for name, value in decoded.items():
        candidate_boundary = boundary_arrays(value.parsed.header, value.symbols)
        candidate_hash = _array_state_digest(candidate_boundary)
        if candidate_hash != boundary_hash or any(
            not np.array_equal(candidate_boundary[key], source_boundary[key])
            for key in source_boundary
        ):
            raise ExecutionError(f"{name} decoded boundary differs from the SSPL1 boundary")
        decoded_hashes[name] = candidate_hash

    streams = {
        name: _stream_record(name, blobs[name], parts, container_provider) for name in STREAM_NAMES
    }
    selected_true_lengths = tuple(
        streams["ssp2e"]["arithmetic_payload_bytes"][name] for name in model.CHANNELS
    )
    selected_shuffle_lengths = tuple(
        streams["ssp2s"]["arithmetic_payload_bytes"][name] for name in model.CHANNELS
    )
    if selected_true_lengths != true_selection.selected.payload_lengths:
        raise ExecutionError("final SSP2E payload lengths differ from the selection lengths")
    if selected_shuffle_lengths != shuffled_selection.selected.payload_lengths:
        raise ExecutionError("final SSP2S payload lengths differ from the selection lengths")

    record: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "source": {
            "magic": "SSPL1",
            "bytes": len(sspl1_blob),
            "sha256": parts.blob_sha256,
            "symbol_sha256": parts.symbol_sha256,
            "n": parts.header.n,
            "bit_tuple": list(parts.header.bits),
            "execution_mode": "synthetic-proof" if synthetic else "production-development",
            "expected_symbol_sha256": expected_symbol_sha256,
            "expected_boundary_state_sha256": expected_boundary_state_sha256,
        },
        "assets": asset_record,
        "native_scope": {
            "native_component": "cxx17_arithmetic_core",
            "python_components": [
                "container_parsing_and_framing",
                "head_and_table_inference",
                "decoded_boundary_tensor_construction",
            ],
            "claim": "native_arithmetic_core_inside_strict_python_bytes_to_boundary_path",
        },
        "contexts": {
            "true_sha256": _context_sha256(true_contexts),
            "shuffled_sha256": _context_sha256(shuffled_contexts),
            "row_count": int(true_contexts.size),
            "production_shuffle_permutation_sha256": (
                model.SHUFFLE_PERMUTATION_SHA256 if parts.header.n == model.GAUSSIAN_COUNT else None
            ),
        },
        "models": {
            "ssp2e": true_diagnostics,
            "ssp2s": shuffled_diagnostics,
            "ssp2l": {
                "global_pairs": {name: list(global_models[name]) for name in model.CHANNELS},
                "context_permutation_invariant": True,
                "model_raw_sha256": _sha256(container.serialize_global_model(global_models)),
            },
        },
        "timings": {
            "clock": "time.perf_counter",
            "ssp2e_eight_start_fit_and_selection_seconds": true_fit_seconds,
            "ssp2s_eight_start_fit_and_selection_seconds": shuffled_fit_seconds,
            "factorized_control_model_selection_seconds": control_model_seconds,
            "complete_encode_and_dual_cold_decode_seconds": (
                complete_encode_and_cold_decode_seconds
            ),
            "execution_total_seconds": time.perf_counter() - execution_started,
            "decision_use": False,
        },
        "streams": streams,
        "boundary": {
            "state_sha256": boundary_hash,
            "components": {
                name: _component_record(value) for name, value in source_boundary.items()
            },
            "decoded_state_sha256": decoded_hashes,
            "all_streams_exact": True,
        },
        "integrity": {
            "all_eight_true_starts_encoded": len(true_diagnostics["starts"]) == 8,
            "all_eight_shuffled_starts_encoded": len(shuffled_diagnostics["starts"]) == 8,
            "all_fits_fixed_points": True,
            "all_start_python_native_payloads_equal": True,
            "all_python_framed_blobs_with_python_vs_native_arithmetic_equal": True,
            "all_python_container_decodes_with_python_vs_native_arithmetic_equal": True,
            "all_symbols_exact": True,
            "all_boundary_arrays_exact": True,
            "all_arithmetic_certificates_hold": all(
                certificate["inequality_holds"] is True
                for stream in streams.values()
                for certificate in stream["certificates"].values()
            ),
            "complete_byte_accounting_exact": True,
        },
    }
    if set(record["integrity"].values()) != {True}:
        raise ExecutionError("one or more execution integrity predicates failed")
    record["execution_sha256"] = _sha256(_canonical_json(record))
    return CellExecution(
        record=record,
        blobs=blobs,
        boundary_arrays=source_boundary,
        true_selection=true_selection,
        shuffled_selection=shuffled_selection,
    )


__all__ = [
    "ArithmeticAdapter",
    "CellExecution",
    "EXECUTION_SCHEMA",
    "ExecutionError",
    "STREAM_NAMES",
    "boundary_arrays",
    "execute_cell",
]
