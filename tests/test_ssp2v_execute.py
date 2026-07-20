from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import zlib
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as comp009_container
from benchmarks import ssp2e_model as comp009_model
from benchmarks import ssp2v_container as container
from benchmarks import ssp2v_execute as execute
from benchmarks import ssp2v_lloyd as lloyd


def _pack(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    return np.asarray(values, dtype=dtype).tobytes(order="C")


def _planar(values: np.ndarray, bits: int) -> bytes:
    dtype = np.dtype("u1" if bits <= 8 else "<u2" if bits <= 16 else "<u4")
    raw = np.asarray(values, dtype=dtype).tobytes(order="C")
    if dtype.itemsize == 1:
        return raw
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, dtype.itemsize).T.ravel().tobytes()


def _deltas(values: np.ndarray, bits: int) -> np.ndarray:
    preceding = np.vstack((np.zeros((1, values.shape[1]), dtype=np.uint32), values[:-1]))
    return (values - preceding) & ((1 << bits) - 1)


def _synthetic_sspl1(
    bits: tuple[int, int, int, int] = (12, 6, 6, 8),
) -> bytes:
    n = container.ROW_COUNT
    bm, bs, br, bc = bits
    row = np.arange(n, dtype=np.uint32)
    means = np.column_stack(
        ((29 * row) % (1 << bm), (47 * row[::-1]) % (1 << bm))
    ).astype(np.uint32)
    scales = np.column_stack((row % (1 << bs), (3 * row + 2) % (1 << bs))).astype(
        np.uint32
    )
    rotation = ((5 * row + 1) % (1 << br)).astype(np.uint32)
    colors = np.column_stack(
        (
            (7 * row) % (1 << bc),
            (11 * row + 1) % (1 << bc),
            (13 * row + 2) % (1 << bc),
        )
    ).astype(np.uint32)
    raw_streams = (
        _planar(_deltas(means, bm).T.ravel(), bm),
        _pack(scales.T.ravel(), bs),
        _pack(rotation, br),
        _pack(_deltas(colors, bc).T.ravel(), bc),
    )
    header = {
        "n": n,
        "H": 64,
        "W": 80,
        "bits": list(bits),
        "morton": True,
        "color_delta": True,
        "means_byte_planar": True,
        "rot_circular": True,
        "stream_codec": "zlib",
        "stream_raw_lengths": [len(raw) for raw in raw_streams],
        "color_lo": [-1.0, -0.5, 0.0],
        "color_hi": [1.0, 1.5, 2.0],
        "scale_lo": [-3.0, -2.0],
        "scale_hi": [2.0, 3.0],
        "means_lo": [0.0, 0.0],
        "means_hi": [79.0, 63.0],
        "has_opacity": False,
        "bits_opacity": 8,
        "renderer": "cuda",
        "aa_dilation": 0.0,
        "sigma_cutoff": 3.0,
        "support_fade": False,
        "render_chunk": 512,
    }
    header_raw = json.dumps(header).encode("utf-8")
    blob = b"SSPL1" + struct.pack("<I", len(header_raw)) + header_raw
    for raw in raw_streams:
        payload = zlib.compress(raw, level=9)
        blob += struct.pack("<I", len(payload)) + payload
    return blob


def _state_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _reseal_record(record: dict) -> dict:
    result = copy.deepcopy(record)
    result.pop("execution_sha256", None)
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    result["execution_sha256"] = hashlib.sha256(payload).hexdigest()
    return result


def _comp009_components(
    parsed: comp009_container.ParsedContainer,
    blob: bytes,
) -> dict[str, dict[str, int | str]]:
    directory_begin = comp009_container.OUTER.size + comp009_container.HEADER.size
    payloads = {
        "outer": blob[: comp009_container.OUTER.size],
        "header": blob[comp009_container.OUTER.size : directory_begin],
        "directory": blob[directory_begin : parsed.prefix_bytes],
        **parsed.payloads,
    }
    records = {
        name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }
    records["total"] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    return records


@lru_cache(maxsize=1)
def _comp009_blobs(source: bytes) -> dict[str, bytes]:
    tables = dict(assets.load_tables())

    def provider(alphabet: int, location: int, scale: int):
        return tables[alphabet].frequency_row(location, scale)

    zeros = (0,) * comp009_model.FEATURE_COUNT
    heads = tuple(
        comp009_model.ChannelHead(zeros, 0, zeros, 0)
        for _ in comp009_model.CHANNELS
    )
    contextual = comp009_model.SSP2EModel(comp009_model.initial_grid(0), heads)
    global_models = {name: (0, 0) for name in comp009_model.CHANNELS}
    blobs = {
        "ssp2e": comp009_container.encode_contextual(
            source,
            contextual.grid_raw,
            contextual.head_raw,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=provider,
            magic=b"SSP2E",
        ),
        "ssp2s": comp009_container.encode_contextual(
            source,
            contextual.grid_raw,
            contextual.head_raw,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=provider,
            magic=b"SSP2S",
        ),
        "ssp2l": comp009_container.encode_ssp2l(
            source,
            global_models,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=provider,
        ),
        "ssp2f": comp009_container.encode_ssp2f(
            source, arithmetic_encode=arithmetic.encode
        ),
    }
    for name, blob in blobs.items():
        decoded = comp009_container.decode_container(
            blob,
            arithmetic_decode=arithmetic.decode,
            arithmetic_encode=arithmetic.encode,
            frequency_provider=provider if name != "ssp2f" else None,
            reference_sspl1=source,
        )
        if decoded.symbol_sha256 != comp009_container.extract_sspl1_parts(source).symbol_sha256:
            raise AssertionError("synthetic COMP-009 reproduction did not cold-decode exactly")
    return blobs


class Reproducer:
    def __init__(self, source: bytes) -> None:
        parts = container.extract_sspl1_parts(source)
        arrays = execute.boundary_arrays(parts.header, parts.symbols)
        boundary_hash = _state_sha256(arrays)
        source_hash = hashlib.sha256(source).hexdigest()
        self.calls = 0
        self.seen_source: bytes | None = None
        self.values: dict[str, execute.ReproducedComp009Stream] = {}
        self.authorities: dict[str, execute.Comp009Authority] = {}
        blobs = _comp009_blobs(source)
        execution_payload = (
            json.dumps(
                {
                    "schema": "structsplat.comp009.synthetic-captured-execution.v1",
                    "source_blob_sha256": source_hash,
                    "streams": {
                        name: hashlib.sha256(blob).hexdigest()
                        for name, blob in sorted(blobs.items())
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        execution_sha256 = hashlib.sha256(execution_payload).hexdigest()
        for name, blob in blobs.items():
            parsed = comp009_container.parse_container(blob, reference_sspl1=source)
            model_payload = (
                parsed.payloads["grid"] + parsed.payloads["head"]
                if name in ("ssp2e", "ssp2s")
                else parsed.payloads["model"]
            )
            state_hash = hashlib.sha256(model_payload).hexdigest()
            components = _comp009_components(parsed, blob)
            verification = execute.comp009_verification_sha256(
                name=name,
                source_blob_sha256=source_hash,
                complete_bytes=len(blob),
                blob_sha256=hashlib.sha256(blob).hexdigest(),
                model_state_sha256=state_hash,
                symbol_sha256=parts.symbol_sha256,
                boundary_state_sha256=boundary_hash,
                components=components,
                captured_source_sha256=execute.COMP009_CAPTURED_SOURCE_SHA256,
                execution_sha256=execution_sha256,
                canonical_cold_decode=True,
                python_native_decode_parity=True,
            )
            self.values[name] = execute.ReproducedComp009Stream(
                blob=blob,
                model_state_sha256=state_hash,
                symbol_sha256=parts.symbol_sha256,
                boundary_state_sha256=boundary_hash,
                canonical_cold_decode=True,
                python_native_decode_parity=True,
                source_blob_sha256=source_hash,
                symbols=parts.symbols,
                boundary_arrays=arrays,
                components=components,
                captured_source_sha256=execute.COMP009_CAPTURED_SOURCE_SHA256,
                execution_sha256=execution_sha256,
                verification_sha256=verification,
            )
            self.authorities[name] = execute.Comp009Authority(
                complete_bytes=len(blob),
                blob_sha256=hashlib.sha256(blob).hexdigest(),
                model_state_sha256=state_hash,
                symbol_sha256=parts.symbol_sha256,
                boundary_state_sha256=boundary_hash,
                captured_source_sha256=execute.COMP009_CAPTURED_SOURCE_SHA256,
                execution_sha256=execution_sha256,
                verification_sha256=verification,
            )

    def __call__(self, source: bytes):
        self.calls += 1
        self.seen_source = source
        return dict(self.values)


class ConstantFitter:
    """Fast deterministic fitter double with exact terminal indices/SSE records."""

    def __init__(self) -> None:
        self.calls = 0

    def fit_start(
        self,
        rows,
        spec: lloyd.LloydSpec,
        start_id: int,
        *,
        max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
    ) -> lloyd.StartResult:
        del max_update_sweeps
        self.calls += 1
        source = np.ascontiguousarray(rows, dtype=np.int32)
        centroid = np.asarray(
            [
                lloyd.round_weighted_mean(
                    int(source[:, coordinate].astype(object).sum()), len(source)
                )
                for coordinate in range(3)
            ],
            dtype=np.int32,
        )
        codebook = np.repeat(centroid[None, :], spec.codebook_size, axis=0)
        indices = np.zeros(len(source), dtype=np.uint32)
        difference = source.astype(np.int64) - centroid.astype(np.int64)
        objective = int(np.square(difference, dtype=np.int64).sum(dtype=np.int64))
        digest = lloyd.codebook_state_sha256(codebook)
        trajectory = (
            lloyd.TrajectoryRecord(0, 0, digest, objective),
            lloyd.TrajectoryRecord(1, 1, digest, objective),
        )
        codebook.flags.writeable = False
        indices.flags.writeable = False
        return lloyd.StartResult(
            spec=spec,
            start_id=start_id,
            status="fixed",
            update_sweeps=1,
            terminal_sse=objective,
            codebook=codebook,
            indices=indices,
            trajectory=trajectory,
        )


def _synthetic_cap_result(
    result: lloyd.StartResult,
    rows,
    *,
    fixed_terminal: bool,
) -> lloyd.StartResult:
    prepared = np.ascontiguousarray(rows, dtype=np.int32)
    weighted = lloyd.collapse_weighted(prepared)
    initial = lloyd.initialize_codebook(weighted, result.spec, result.start_id)
    initial_objective, _ = lloyd._objective(weighted, initial)
    if fixed_terminal:
        codebook = np.ascontiguousarray(result.codebook, dtype=np.int32)
    else:
        updated = lloyd._updated_codebook(weighted, initial, result.spec)
        codebook = None
        for replaced_rows in range(1, len(initial) + 1):
            candidate = initial.copy()
            candidate[:replaced_rows] = updated[:replaced_rows]
            candidate = np.asarray(sorted(map(tuple, candidate.tolist())), dtype=np.int32)
            candidate_objective, _ = lloyd._objective(weighted, candidate)
            candidate_next = lloyd._updated_codebook(weighted, candidate, result.spec)
            if (
                candidate_objective <= initial_objective
                and not np.array_equal(candidate, initial)
                and not np.array_equal(candidate_next, candidate)
            ):
                codebook = candidate
                break
        if codebook is None:
            raise AssertionError("could not construct a nonfixed cap contract fixture")
    objective, _ = lloyd._objective(weighted, codebook)
    indices = lloyd._materialize_indices(prepared, codebook)
    initial_hash = lloyd.codebook_state_sha256(initial)
    terminal_hash = lloyd.codebook_state_sha256(codebook)
    trajectory = [lloyd.TrajectoryRecord(0, 0, initial_hash, initial_objective)]
    trajectory.extend(
        lloyd.TrajectoryRecord(
            index,
            index,
            hashlib.sha256(
                f"synthetic-cap-intermediate-{result.start_id}-{index}".encode()
            ).hexdigest(),
            objective,
        )
        for index in range(1, lloyd.MAX_UPDATE_SWEEPS)
    )
    trajectory.append(
        lloyd.TrajectoryRecord(
            lloyd.MAX_UPDATE_SWEEPS,
            lloyd.MAX_UPDATE_SWEEPS,
            terminal_hash,
            objective,
        )
    )
    return replace(
        result,
        status="cap",
        update_sweeps=lloyd.MAX_UPDATE_SWEEPS,
        terminal_sse=objective,
        codebook=codebook,
        indices=indices,
        trajectory=tuple(trajectory),
    )


class NonfixedCapFitter:
    def __init__(self, delegate) -> None:
        self.delegate = delegate

    def fit_start(
        self,
        rows,
        spec: lloyd.LloydSpec,
        start_id: int,
        *,
        max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
    ) -> lloyd.StartResult:
        result = self.delegate.fit_start(
            rows,
            spec,
            start_id,
            max_update_sweeps=max_update_sweeps,
        )
        if spec.family_id == 0 and spec.matched_stage_count == 2 and start_id == 0:
            return _synthetic_cap_result(result, rows, fixed_terminal=False)
        return result


class FixedPointCapFitter(NonfixedCapFitter):
    def fit_start(
        self,
        rows,
        spec: lloyd.LloydSpec,
        start_id: int,
        *,
        max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
    ) -> lloyd.StartResult:
        result = self.delegate.fit_start(
            rows,
            spec,
            start_id,
            max_update_sweeps=max_update_sweeps,
        )
        if spec.family_id == 0 and spec.matched_stage_count == 2 and start_id == 0:
            return _synthetic_cap_result(result, rows, fixed_terminal=True)
        return result


@pytest.fixture(scope="module")
def source() -> bytes:
    return _synthetic_sspl1()


@pytest.fixture(scope="module")
def native_coder(tmp_path_factory: pytest.TempPathFactory):
    path = Path(tmp_path_factory.mktemp("ssp2v-execute-native")) / "ssp2e_range_ext.so"
    arithmetic.build_native(path)
    return arithmetic.NativeArithmeticCoder(path)


@pytest.fixture(scope="module")
def native_fitter(tmp_path_factory: pytest.TempPathFactory):
    path = Path(tmp_path_factory.mktemp("ssp2v-execute-lloyd")) / "ssp2v_lloyd_ext.so"
    lloyd.build_native(path)
    return lloyd.NativeLloydFitter(path)


@pytest.fixture(scope="module")
def completed(source: bytes, native_coder, native_fitter):
    reproducer = Reproducer(source)
    result = execute.execute_cell(
        source,
        native_coder,
        reproduce_comp009=reproducer,
        expected_comp009=reproducer.authorities,
        fitter=native_fitter,
    )
    assert isinstance(result, execute.CellExecution)
    return result, reproducer


def test_boundary_arrays_are_exact_independent_float32_state(source: bytes) -> None:
    parts = container.extract_sspl1_parts(source)
    arrays = execute.boundary_arrays(parts.header, parts.symbols)
    assert set(arrays) == {"means", "log_scales", "rotations", "colors"}
    assert arrays["means"].shape == (container.ROW_COUNT, 2)
    assert arrays["colors"].shape == (container.ROW_COUNT, 3)
    assert arrays["rotations"].shape == (container.ROW_COUNT,)
    assert all(value.dtype == np.float32 for value in arrays.values())
    assert all(not value.flags.writeable for value in arrays.values())
    assert not any(
        np.shares_memory(value, symbol)
        for value in arrays.values()
        for symbol in parts.symbols.values()
    )

    incomplete = dict(parts.symbols)
    incomplete.pop("blue")
    with pytest.raises(execute.ExecutionError, match="all eight"):
        execute.boundary_arrays(parts.header, incomplete)

    oversized = dict(parts.symbols)
    oversized["red"] = np.full(container.ROW_COUNT, 1 << 40, dtype=np.uint64)
    with pytest.raises(execute.ExecutionError, match="bit depth"):
        execute.boundary_arrays(parts.header, oversized)


def test_complete_cell_has_frozen_menu_dual_parity_and_exact_components(
    source: bytes,
    completed,
) -> None:
    result, reproducer = completed
    record = result.record
    execute.verify_record(record)
    assert record["schema"] == execute.EXECUTION_SCHEMA
    assert record["task_sha256"] == execute.TASK_SHA256
    assert record["status"] == "valid_complete_candidate_set"
    assert record["target_input_surface"] == "absent"
    assert record["target_accessed"] is False
    assert set(record["integrity"].values()) == {True}
    assert record["variant_order"] == list(execute.VARIANT_LABELS)
    assert list(record["variants"]) == list(execute.VARIANT_LABELS)
    assert record["exact_format_order"] == list(execute.EXACT_FORMAT_ORDER)
    assert set(result.baseline_blobs) == set(execute.EXACT_FORMAT_ORDER)
    assert set(result.variant_blobs) == set(execute.VARIANT_LABELS)
    assert set(result.fits) == set(execute.VARIANT_LABELS)
    assert reproducer.calls == 1
    assert reproducer.seen_source is source

    inventory = record["candidate_inventory"]
    assert len(inventory) == len(execute.EXACT_FORMAT_ORDER) + len(execute.VARIANT_LABELS)
    candidate_evidence = {
        "schema": record["candidate_evidence_schema"],
        "task_sha256": execute.TASK_SHA256,
        "exact_format_order": record["exact_format_order"],
        "exact_formats": record["exact_formats"],
        "selected_q": record["selected_q"],
        "variant_order": record["variant_order"],
        "variants": record["variants"],
        "matched_codebook_bytes": record["matched_codebook_bytes"],
    }
    expected_evidence_hash = hashlib.sha256(
        (
            json.dumps(
                candidate_evidence,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    assert record["candidate_set_sha256"] == expected_evidence_hash
    assert record["candidate_set_seals_full_normalized_evidence"] is True
    expected_q = min(
        enumerate(execute.EXACT_FORMAT_ORDER),
        key=lambda item: (record["exact_formats"][item[1]]["complete_bytes"], item[0]),
    )[1]
    assert record["selected_q"]["name"] == expected_q
    assert record["selected_q"]["tie_rank"] == execute.EXACT_FORMAT_ORDER.index(expected_q)
    source_components = record["exact_formats"]["sspl1"]["components"]
    assert sum(
        item["bytes"] for name, item in source_components.items() if name != "total"
    ) == len(source)

    for name in execute.COMP009_NAMES:
        baseline = record["exact_formats"][name]
        assert baseline["authority_hashes_verified"] is True
        assert baseline["canonical_cold_decode"] is True
        assert baseline["python_native_decode_parity"] is True
        assert baseline["blob_sha256"] == hashlib.sha256(result.baseline_blobs[name]).hexdigest()
        assert baseline["verification_sha256"] == reproducer.authorities[
            name
        ].verification_sha256
        assert sum(
            item["bytes"]
            for component, item in baseline["components"].items()
            if component != "total"
        ) == len(result.baseline_blobs[name])

    for spec in execute.VARIANT_SPECS:
        value = record["variants"][spec.label]
        assert value["descriptor"]["family_id"] == spec.family_id
        assert value["descriptor"]["matched_rvq_stage_count"] == spec.matched_stage_count
        assert value["descriptor"]["base_k"] == spec.base_k
        assert value["all_lloyd_starts_fixed"] is True
        assert len(value["stages"]) == spec.descriptor.actual_stage_count
        for stage in value["stages"]:
            assert [item["start_id"] for item in stage["starts"]] == [0, 1, 2, 3]
            selected = [item["selected"] for item in stage["starts"]]
            assert selected.count(True) == 1
            assert selected[stage["selected_start_id"]] is True
            assert all(item["status"] == "fixed" for item in stage["starts"])
        stream = value["stream"]
        assert stream["python_injected_encode_parity"] is True
        assert stream["python_injected_cold_decode_parity"] is True
        assert stream["canonical_decode_to_reencode"] is True
        assert stream["non_rgb_symbols_exact"] is True
        assert stream["non_rgb_boundary_exact"] is True
        assert stream["fit_decode_rgb_exact"] is True
        assert stream["complete_bytes"] == len(result.variant_blobs[spec.label])
        components = stream["components"]["records"]
        assert sum(item["bytes"] for name, item in components.items() if name != "total") == len(
            result.variant_blobs[spec.label]
        )
        assert len(stream["coder_certificates"]) == spec.descriptor.actual_stage_count
        assert all(item["inequality_holds"] for item in stream["coder_certificates"])
        assert len(stream["used_entries_per_stage"]) == spec.descriptor.actual_stage_count
        assert all(
            1 <= used <= alphabet
            for used, alphabet in zip(
                stream["used_entries_per_stage"],
                spec.descriptor.stage_alphabets,
                strict=True,
            )
        )
        assert 1 <= stream["unique_index_tuples"] <= container.ROW_COUNT

    assert all(item["equal"] for item in record["matched_codebook_bytes"])
    assert all(item["flat_codebook_bytes"] == item["rvq_codebook_bytes"] for item in record[
        "matched_codebook_bytes"
    ])


def test_collision_and_boundary_accounting_survive_cold_decode(completed) -> None:
    result, _ = completed
    variants = result.record["variants"]
    for label in execute.VARIANT_LABELS:
        accounting = variants[label]["stream"]["rgb_accounting"]
        assert accounting["clip_components"] == 0
        assert accounting["collision_pairs"] >= 0
        assert accounting == variants[label]["rgb_accounting"]

    source_boundary = result.boundary_arrays["sspl1"]
    assert set(result.boundary_arrays) == {
        "sspl1",
        "ssp2z",
        *execute.VARIANT_LABELS,
    }
    for name, arrays in result.boundary_arrays.items():
        if name == "sspl1":
            continue
        for component in ("means", "log_scales", "rotations"):
            assert np.array_equal(arrays[component], source_boundary[component])
        assert all(not value.flags.writeable for value in arrays.values())


def test_exact_clipping_accounting_covers_low_high_edges_and_distinct_collisions() -> None:
    source = np.asarray(((0, 1, 2), (255, 254, 253), (0, 1, 2)), dtype=np.uint8)
    raw = np.asarray(((-1, 1, 2), (256, 254, 253), (0, 1, 2)), dtype=np.int32)
    accounting = container.rgb_accounting(source, raw)
    assert accounting.clip_low_components == 1
    assert accounting.clip_high_components == 1
    assert accounting.clip_components == 2
    assert accounting.clip_rows == 2
    assert accounting.rgb_sse == 0
    assert accounting.collision_pairs == 0


def test_injected_arithmetic_byte_drift_fails_closed(source: bytes, native_fitter) -> None:
    reproducer = Reproducer(source)

    class DriftingCoder:
        @staticmethod
        def encode(symbols: np.ndarray, cdf_rows: np.ndarray) -> bytes:
            payload = arithmetic.encode(symbols, cdf_rows)
            return payload + b"\0" if cdf_rows.shape[1] - 1 == 192 else payload

        decode = staticmethod(arithmetic.decode)

    with pytest.raises(execute.ExecutionError, match="encode failed|parity|dual strict"):
        execute.execute_cell(
            source,
            DriftingCoder(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=native_fitter,
        )

    class DriftingDecoder:
        encode = staticmethod(arithmetic.encode)

        @staticmethod
        def decode(payload: bytes, cdf_rows: np.ndarray, n: int) -> np.ndarray:
            values = arithmetic.decode(payload, cdf_rows, n).copy()
            values[0] = (int(values[0]) + 1) % (cdf_rows.shape[1] - 1)
            return values

    with pytest.raises(execute.ExecutionError, match="cold-decode"):
        execute.execute_cell(
            source,
            DriftingDecoder(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=native_fitter,
        )


def test_comp009_reproduction_hash_mismatch_precedes_every_fit(source: bytes) -> None:
    reproducer = Reproducer(source)
    authorities = dict(reproducer.authorities)
    authorities["ssp2e"] = replace(authorities["ssp2e"], blob_sha256="0" * 64)
    fitter = ConstantFitter()
    with pytest.raises(execute.ExecutionError, match="complete-byte authority"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=reproducer,
            expected_comp009=authorities,
            fitter=fitter,
        )
    assert fitter.calls == 0
    assert reproducer.calls == 1


def test_comp009_source_bound_seal_and_component_mutations_fail_before_fit(
    source: bytes,
) -> None:
    source_drift = Reproducer(source)
    source_drift.values["ssp2e"] = replace(
        source_drift.values["ssp2e"], source_blob_sha256="0" * 64
    )
    fitter = ConstantFitter()
    with pytest.raises(execute.ExecutionError, match="not bound"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=source_drift,
            expected_comp009=source_drift.authorities,
            fitter=fitter,
        )
    assert fitter.calls == 0

    captured_source_drift = Reproducer(source)
    captured_source_drift.values["ssp2e"] = replace(
        captured_source_drift.values["ssp2e"], captured_source_sha256="0" * 64
    )
    fitter = ConstantFitter()
    with pytest.raises(execute.ExecutionError, match="captured-source authority"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=captured_source_drift,
            expected_comp009=captured_source_drift.authorities,
            fitter=fitter,
        )
    assert fitter.calls == 0

    execution_drift = Reproducer(source)
    execution_drift.values["ssp2e"] = replace(
        execution_drift.values["ssp2e"], execution_sha256="0" * 64
    )
    fitter = ConstantFitter()
    with pytest.raises(execute.ExecutionError, match="captured execution authority"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=execution_drift,
            expected_comp009=execution_drift.authorities,
            fitter=fitter,
        )
    assert fitter.calls == 0

    component_drift = Reproducer(source)
    components = copy.deepcopy(component_drift.values["ssp2e"].components)
    components["outer"]["sha256"] = "0" * 64
    component_drift.values["ssp2e"] = replace(
        component_drift.values["ssp2e"], components=components
    )
    fitter = ConstantFitter()
    with pytest.raises(execute.ExecutionError, match="component records"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=component_drift,
            expected_comp009=component_drift.authorities,
            fitter=fitter,
        )
    assert fitter.calls == 0


def test_comp009_callback_booleans_cannot_replace_real_dual_cold_decode(
    source: bytes,
) -> None:
    reproducer = Reproducer(source)
    fitter = ConstantFitter()

    class DriftingNativeDecoder:
        encode = staticmethod(arithmetic.encode)

        @staticmethod
        def decode(payload: bytes, cdf_rows: np.ndarray, n: int) -> np.ndarray:
            values = arithmetic.decode(payload, cdf_rows, n).copy()
            values[0] = (int(values[0]) + 1) % (cdf_rows.shape[1] - 1)
            return values

    with pytest.raises(execute.ExecutionError, match="dual strict cold-decode"):
        execute.execute_cell(
            source,
            DriftingNativeDecoder(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=fitter,
        )
    assert fitter.calls == 0


def test_internal_candidate_seal_q_and_component_mutations_survive_outer_reseal_rejection(
    completed,
) -> None:
    result, _ = completed

    candidate_drift = copy.deepcopy(result.record)
    candidate_drift["candidate_set_sha256"] = "0" * 64
    with pytest.raises(execute.ExecutionError, match="candidate_set_sha256"):
        execute.verify_record(_reseal_record(candidate_drift))

    q_drift = copy.deepcopy(result.record)
    original = q_drift["selected_q"]["name"]
    q_drift["selected_q"]["name"] = next(
        name for name in execute.EXACT_FORMAT_ORDER if name != original
    )
    with pytest.raises(execute.ExecutionError, match="selected Q"):
        execute.verify_record(_reseal_record(q_drift))

    component_drift = copy.deepcopy(result.record)
    component_drift["exact_formats"]["sspl1"]["components"]["magic"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(execute.ExecutionError, match="candidate_set_sha256"):
        execute.verify_record(_reseal_record(component_drift))

    task_drift = copy.deepcopy(result.record)
    task_drift["task_sha256"] = "0" * 64
    with pytest.raises(execute.ExecutionError, match="task identity"):
        execute.verify_record(_reseal_record(task_drift))

    lifecycle_drift = copy.deepcopy(result.record)
    lifecycle_drift["target_accessed"] = True
    with pytest.raises(execute.ExecutionError, match="lifecycle flags"):
        execute.verify_record(_reseal_record(lifecycle_drift))

    integrity_drift = copy.deepcopy(result.record)
    integrity_drift["integrity"]["all_required_lloyd_starts_fixed"] = False
    with pytest.raises(execute.ExecutionError, match="integrity predicates"):
        execute.verify_record(_reseal_record(integrity_drift))


def object_with_python_coder():
    class PythonCoder:
        encode = staticmethod(arithmetic.encode)
        decode = staticmethod(arithmetic.decode)

    return PythonCoder()


def test_lloyd_nonconvergence_is_typed_valid_failure_without_partial_candidates(
    source: bytes,
    native_fitter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reproducer = Reproducer(source)

    def forbidden_encode(*args, **kwargs):
        del args, kwargs
        raise AssertionError("SSP2V encoding must not begin after a required start fails")

    monkeypatch.setattr(container, "encode_ssp2v", forbidden_encode)
    result = execute.execute_cell(
        source,
        object_with_python_coder(),
        reproduce_comp009=reproducer,
        expected_comp009=reproducer.authorities,
        fitter=NonfixedCapFitter(native_fitter),
    )
    assert isinstance(result, execute.LloydMethodFailure)
    execute.verify_record(result.record)
    assert result.record["schema"] == execute.METHOD_FAILURE_SCHEMA
    assert result.record["status"] == "valid_method_failure"
    assert result.record["method_state"] == "required_lloyd_nonconvergence"
    assert result.record["failure"]["variant_label"] == execute.VARIANT_LABELS[0]
    assert result.record["failure"]["failed_start_ids"] == [0]
    assert result.record["target_input_surface"] == "absent"
    assert result.record["target_accessed"] is False
    assert result.record["target_import_authorized"] is False
    assert result.record["partial_ssp2v_candidate_set_emitted"] is False
    assert result.record["completed_variant_fits"] == {}
    assert len(result.failed_starts) == 4
    assert set(result.baseline_blobs) == set(execute.EXACT_FORMAT_ORDER)

    task_drift = copy.deepcopy(result.record)
    task_drift["task_sha256"] = "0" * 64
    with pytest.raises(execute.ExecutionError, match="method-failure seal"):
        execute.verify_record(_reseal_record(task_drift))
    flag_drift = copy.deepcopy(result.record)
    flag_drift["target_import_authorized"] = True
    with pytest.raises(execute.ExecutionError, match="method-failure seal"):
        execute.verify_record(_reseal_record(flag_drift))


def test_later_rvq_failure_seals_validated_completed_stage_history(
    source: bytes, native_fitter
) -> None:
    reproducer = Reproducer(source)

    class LaterStageCap:
        def fit_start(
            self,
            rows,
            spec: lloyd.LloydSpec,
            start_id: int,
            *,
            max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
        ) -> lloyd.StartResult:
            result = native_fitter.fit_start(
                rows,
                spec,
                start_id,
                max_update_sweeps=max_update_sweeps,
            )
            if (
                spec.family_id == 1
                and spec.matched_stage_count == 2
                and spec.base_k == 64
                and spec.actual_stage_index == 1
                and start_id == 0
            ):
                return _synthetic_cap_result(result, rows, fixed_terminal=False)
            return result

    result = execute.execute_cell(
        source,
        object_with_python_coder(),
        reproduce_comp009=reproducer,
        expected_comp009=reproducer.authorities,
        fitter=LaterStageCap(),
    )
    assert isinstance(result, execute.LloydMethodFailure)
    execute.verify_record(result.record)
    failure = result.record["failure"]
    assert failure["variant_label"] == "rvq_s2_k64"
    assert failure["actual_stage_index"] == 1
    assert len(failure["completed_stages"]) == 1
    assert failure["completed_stages"][0]["spec"]["actual_stage_index"] == 0
    assert failure["completed_stages"][0]["all_four_starts_fixed"] is True


def test_malformed_prior_rvq_stage_cannot_hide_behind_later_failure(
    source: bytes, native_fitter
) -> None:
    reproducer = Reproducer(source)

    class MalformedPriorStage:
        def __init__(self) -> None:
            self.later_stage_calls = 0

        def fit_start(
            self,
            rows,
            spec: lloyd.LloydSpec,
            start_id: int,
            *,
            max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
        ) -> lloyd.StartResult:
            if spec.family_id == 1 and spec.actual_stage_index == 1:
                self.later_stage_calls += 1
            result = native_fitter.fit_start(
                rows,
                spec,
                start_id,
                max_update_sweeps=max_update_sweeps,
            )
            if (
                spec.family_id == 1
                and spec.matched_stage_count == 2
                and spec.base_k == 64
                and spec.actual_stage_index == 0
                and start_id == 0
            ):
                trajectory = list(result.trajectory)
                trajectory[0] = replace(trajectory[0], codebook_sha256="0" * 64)
                return replace(result, trajectory=tuple(trajectory))
            return result

    fitter = MalformedPriorStage()
    with pytest.raises(execute.ExecutionError, match="frozen initialized state"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=fitter,
        )
    assert fitter.later_stage_calls == 0


def test_claimed_cap_terminal_must_not_already_be_an_exact_fixed_point(
    source: bytes,
    native_fitter,
) -> None:
    reproducer = Reproducer(source)
    with pytest.raises(execute.ExecutionError, match="exact fixed point after sweep 100"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=FixedPointCapFitter(native_fitter),
        )


def test_claimed_fixed_state_must_survive_one_independent_exact_update(
    source: bytes, native_fitter
) -> None:
    reproducer = Reproducer(source)

    class NonFixedFitter:
        def fit_start(
            self,
            rows,
            spec: lloyd.LloydSpec,
            start_id: int,
            *,
            max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
        ) -> lloyd.StartResult:
            result = native_fitter.fit_start(
                rows,
                spec,
                start_id,
                max_update_sweeps=max_update_sweeps,
            )
            if spec.family_id != 0 or spec.matched_stage_count != 2 or start_id != 0:
                return result
            cap = _synthetic_cap_result(result, rows, fixed_terminal=False)
            initial = cap.trajectory[0]
            terminal = cap.trajectory[-1]
            trajectory = (
                initial,
                lloyd.TrajectoryRecord(
                    1, 1, terminal.codebook_sha256, terminal.objective_sse
                ),
                lloyd.TrajectoryRecord(
                    2, 2, terminal.codebook_sha256, terminal.objective_sse
                ),
            )
            return replace(
                result,
                status="fixed",
                update_sweeps=2,
                terminal_sse=cap.terminal_sse,
                codebook=cap.codebook,
                indices=cap.indices,
                trajectory=trajectory,
            )

    with pytest.raises(execute.ExecutionError, match="changes under one exact update"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=NonFixedFitter(),
        )


def test_lloyd_start_and_trajectory_metadata_are_independently_bound(
    source: bytes, native_fitter
) -> None:
    reproducer = Reproducer(source)

    class MutatingFitter:
        def __init__(self, mutation: str) -> None:
            self.mutation = mutation

        def fit_start(
            self,
            rows,
            spec: lloyd.LloydSpec,
            start_id: int,
            *,
            max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
        ) -> lloyd.StartResult:
            result = native_fitter.fit_start(
                rows,
                spec,
                start_id,
                max_update_sweeps=max_update_sweeps,
            )
            if spec.family_id != 0 or spec.matched_stage_count != 2 or start_id != 0:
                return result
            if self.mutation == "initial":
                trajectory = list(result.trajectory)
                trajectory[0] = replace(trajectory[0], codebook_sha256="0" * 64)
                return replace(result, trajectory=tuple(trajectory))
            if self.mutation == "late":
                initial = result.trajectory[0]
                terminal = result.trajectory[-1]
                trajectory = [initial]
                trajectory.extend(
                    lloyd.TrajectoryRecord(
                        index,
                        index,
                        terminal.codebook_sha256,
                        terminal.objective_sse,
                    )
                    for index in range(1, lloyd.MAX_UPDATE_SWEEPS + 2)
                )
                return replace(
                    result,
                    update_sweeps=lloyd.MAX_UPDATE_SWEEPS + 1,
                    trajectory=tuple(trajectory),
                )
            if self.mutation == "objective":
                initial = result.trajectory[0]
                terminal = result.trajectory[-1]
                trajectory = (
                    initial,
                    lloyd.TrajectoryRecord(
                        1,
                        1,
                        terminal.codebook_sha256,
                        terminal.objective_sse + 1,
                    ),
                    lloyd.TrajectoryRecord(
                        2,
                        2,
                        terminal.codebook_sha256,
                        terminal.objective_sse,
                    ),
                )
                return replace(result, update_sweeps=2, trajectory=trajectory)
            if self.mutation == "bool_id":
                return replace(result, start_id=False)
            raise AssertionError("unknown mutation")

    expectations = (
        ("initial", "frozen initialized state"),
        ("late", "no greater than 100"),
        ("objective", "inconsistent trajectory objectives"),
        ("bool_id", "exact nonnegative integer"),
    )
    for mutation, message in expectations:
        with pytest.raises(execute.ExecutionError, match=message):
            execute.execute_cell(
                source,
                object_with_python_coder(),
                reproduce_comp009=reproducer,
                expected_comp009=reproducer.authorities,
                fitter=MutatingFitter(mutation),
            )


def test_malformed_one_sweep_cap_is_not_a_valid_method_failure(
    source: bytes, native_fitter
) -> None:
    reproducer = Reproducer(source)

    class OneSweepCap(NonfixedCapFitter):
        def fit_start(
            self,
            rows,
            spec: lloyd.LloydSpec,
            start_id: int,
            *,
            max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
        ) -> lloyd.StartResult:
            result = super().fit_start(
                rows,
                spec,
                start_id,
                max_update_sweeps=max_update_sweeps,
            )
            if result.status != "cap":
                return result
            terminal = result.trajectory[-1]
            initial = result.trajectory[0]
            return replace(
                result,
                update_sweeps=1,
                trajectory=(
                    initial,
                    lloyd.TrajectoryRecord(
                        1, 1, terminal.codebook_sha256, terminal.objective_sse
                    ),
                ),
            )

    with pytest.raises(execute.ExecutionError, match="100 distinct updates"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=OneSweepCap(native_fitter),
        )


def test_adapter_and_callback_contracts_are_checked_before_payload_execution(source: bytes) -> None:
    reproducer = Reproducer(source)
    with pytest.raises(execute.ExecutionError, match="encode and decode"):
        execute.execute_cell(
            source,
            object(),
            reproduce_comp009=reproducer,
            expected_comp009=reproducer.authorities,
            fitter=ConstantFitter(),
        )
    assert reproducer.calls == 0

    with pytest.raises(execute.ExecutionError, match="exactly SSP2E/S/L/F"):
        execute.execute_cell(
            source,
            object_with_python_coder(),
            reproduce_comp009=reproducer,
            expected_comp009={"ssp2e": reproducer.authorities["ssp2e"]},
            fitter=ConstantFitter(),
        )


def test_rotation_boundary_uses_float64_pi_then_float32(source: bytes) -> None:
    parts = container.extract_sspl1_parts(source)
    arrays = execute.boundary_arrays(parts.header, parts.symbols)
    expected = (
        parts.symbols["rotation"].astype(np.float64)
        / (1 << parts.header.bits[2])
        * math.pi
    ).astype(np.float32)
    assert np.array_equal(arrays["rotations"], expected)
