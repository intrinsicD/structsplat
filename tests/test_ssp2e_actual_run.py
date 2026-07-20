from __future__ import annotations

import copy
import hashlib
import io
import json
import struct
import tarfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import ssp2e_actual_coder as actual
from benchmarks import ssp2e_actual_run as run
from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_decode_worker as decode_worker
from benchmarks import ssp2e_execute as execute


def test_source_manifest_is_stable_and_contains_assets_and_native_source():
    manifest = run.source_manifest()
    paths = {record["path"] for record in manifest["files"]}
    assert "benchmarks/assets/ssp2e_cdf_v1.bin" in paths
    assert "benchmarks/assets/ssp2e_cost_q24_v1.bin" in paths
    assert "benchmarks/ssp2e_range_ext.cpp" in paths
    assert "tasks/COMP-009-ssp2e-actual-coder.md" in paths
    assert any(path.startswith("src/structsplat/") for path in paths)
    assert len(manifest["source_manifest_sha256"]) == 64


def test_exclusive_write_refuses_replacement(tmp_path: Path):
    path = tmp_path / "sealed.bin"
    run._exclusive_write(path, b"first")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(b"first").hexdigest()
    with pytest.raises(run.LifecycleError, match="refusing to replace"):
        run._exclusive_write(path, b"second")


def test_preflight_rejects_nonempty_output_before_any_input_check(tmp_path: Path):
    outdir = tmp_path / "occupied"
    outdir.mkdir()
    (outdir / "foreign").write_bytes(b"x")
    with pytest.raises(run.LifecycleError, match="empty output"):
        run.preflight(outdir, tmp_path / "not-used")


def test_environment_record_is_finite_and_explicit():
    record = run.environment_record()
    assert record["byteorder"] in {"little", "big"}
    assert record["python_executable"]
    assert isinstance(record["cpu_affinity"], list)


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


def _synthetic_sspl1(n: int = 19) -> bytes:
    bits = (12, 6, 6, 8)
    bm, bs, br, bc = bits
    row = np.arange(n, dtype=np.uint32)
    means = np.column_stack(((29 * row) % (1 << bm), (47 * row[::-1]) % (1 << bm))).astype(
        np.uint32
    )
    scales = np.column_stack((row % (1 << bs), (3 * row + 2) % (1 << bs))).astype(np.uint32)
    rotation = ((5 * row + 1) % (1 << br)).astype(np.uint32)
    colors = np.column_stack(
        ((7 * row) % (1 << bc), (11 * row + 1) % (1 << bc), (13 * row + 2) % (1 << bc))
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


@pytest.fixture(scope="module")
def synthetic_execution(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("comp009-run-cell-source")
    native_path = arithmetic.build_native(root / "libssp2e_range_v1.so")
    native = arithmetic.NativeArithmeticCoder(native_path)
    source = _synthetic_sspl1()
    parts = container.extract_sspl1_parts(source)
    boundary_sha256 = execute._array_state_digest(  # noqa: SLF001
        execute.boundary_arrays(parts.header, parts.symbols)
    )
    result = execute.execute_cell(
        source,
        native,
        synthetic=True,
        expected_symbol_sha256=parts.symbol_sha256,
        expected_boundary_state_sha256=boundary_sha256,
    )
    return {
        "source": source,
        "parts": parts,
        "boundary_sha256": boundary_sha256,
        "result": result,
        "native_path": native_path,
    }


def _materialize_run_cell(
    outdir: Path, synthetic_execution: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    source = synthetic_execution["source"]
    parts = synthetic_execution["parts"]
    image_id = run.oracle.FROZEN_IMAGE_IDS[0]
    bits = run.oracle.FROZEN_BIT_TUPLES[0]
    label = run._tuple_label(bits)
    input_path = outdir / "inputs" / image_id / f"{label}.sspl1"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(source)
    metadata = {
        "image_id": image_id,
        "bit_tuple": list(bits),
        "imported_path": input_path.relative_to(outdir).as_posix(),
        "bytes": len(source),
        "sha256": hashlib.sha256(source).hexdigest(),
        "symbol_sha256": parts.symbol_sha256,
        "decoded_state_sha256": synthetic_execution["boundary_sha256"],
        "cell_record_sha256": "c" * 64,
    }
    manifest = actual.seal_record(
        {
            "schema": run.INPUT_SCHEMA,
            "streams": [metadata],
        },
        "input_manifest_sha256",
    )
    (outdir / "input_manifest.json").write_bytes(run._canonical_json(manifest))
    cell = run._persist_execution_cell(
        outdir,
        metadata,
        synthetic_execution["result"],
        1.25,
    )
    return cell, metadata


def _reseal_execution(record: dict[str, object]) -> None:
    record.pop("execution_sha256", None)
    record["execution_sha256"] = hashlib.sha256(run._canonical_json(record)).hexdigest()


def _reseal_cell(record: dict[str, object]) -> dict[str, object]:
    return actual.seal_record(record, "cell_record_sha256")


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact_bytes",
        "input_sha256",
        "execution_source",
        "selected_model",
        "boundary_component",
        "integrity",
        "confirmation",
    ],
)
def test_run_cell_resealed_single_field_mutations_fail(
    tmp_path: Path,
    synthetic_execution: dict[str, object],
    mutation: str,
) -> None:
    valid, _ = _materialize_run_cell(tmp_path, synthetic_execution)
    assert run._verify_run_cell(tmp_path, valid) == valid
    changed = copy.deepcopy(valid)
    if mutation == "artifact_bytes":
        changed["artifacts"][0]["bytes"] += 1
    elif mutation == "input_sha256":
        changed["input"]["sha256"] = "0" * 64
    elif mutation == "execution_source":
        changed["execution"]["source"]["sha256"] = "1" * 64
        _reseal_execution(changed["execution"])
    elif mutation == "selected_model":
        changed["execution"]["models"]["ssp2e"]["selected_model_sha256"] = "2" * 64
        _reseal_execution(changed["execution"])
    elif mutation == "boundary_component":
        changed["execution"]["boundary"]["components"]["means"]["sha256"] = "3" * 64
        _reseal_execution(changed["execution"])
    elif mutation == "integrity":
        changed["integrity"]["all_symbols_exact"] = False
    elif mutation == "confirmation":
        changed["confirmation_payloads_accessed"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)
    changed = _reseal_cell(changed)
    with pytest.raises((run.LifecycleError, actual.ActualCoderError)):
        run._verify_run_cell(tmp_path, changed)


@pytest.mark.parametrize("artifact_kind", ["symbol", "boundary", "model"])
def test_run_cell_rejects_rehashed_artifact_content_drift(
    tmp_path: Path,
    synthetic_execution: dict[str, object],
    artifact_kind: str,
) -> None:
    valid, _ = _materialize_run_cell(tmp_path, synthetic_execution)
    changed = copy.deepcopy(valid)
    base = run._cell_path(tmp_path, valid["image_id"], valid["bit_tuple"])
    if artifact_kind == "symbol":
        relative = "symbols/red.npy"
        value = np.load(base / relative, allow_pickle=False)
        value[0] = (int(value[0]) + 1) % 256
        payload = run._npy_payload(value)
    elif artifact_kind == "boundary":
        relative = "boundary/colors.npy"
        value = np.load(base / relative, allow_pickle=False)
        value[0, 0] += np.float32(0.25)
        payload = run._npy_payload(value)
    else:
        relative = "models/ssp2e/start_0.grid"
        payload = bytearray((base / relative).read_bytes())
        payload[0] ^= 1
        payload = bytes(payload)
    path = base / relative
    path.chmod(0o644)
    path.write_bytes(payload)
    artifact_path = (base.relative_to(tmp_path) / relative).as_posix()
    artifact = next(item for item in changed["artifacts"] if item["path"] == artifact_path)
    artifact["bytes"] = len(payload)
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    changed = _reseal_cell(changed)
    with pytest.raises(run.LifecycleError):
        run._verify_run_cell(tmp_path, changed)


def _worker_sample_record(
    *,
    mode: str,
    path: Path,
    native_path: Path,
    bits: tuple[int, ...],
    symbol_sha256: str,
    boundary_sha256: str,
    elapsed_ns: int,
    peak_rss_bytes: int,
    pid: int,
    cpu: int,
) -> dict[str, object]:
    native_scope = {
        "library_bytes": native_path.stat().st_size,
        "library_sha256": run._sha256_file(native_path),
        "library_loaded_before_timing": True,
        "arithmetic_core_used_in_timed_region": mode != "sspl1",
        "canonical_validation": (
            "native_decode_internal_reencode_and_byte_compare"
            if mode != "sspl1"
            else "not_applicable_sspl1"
        ),
        "redundant_second_native_encode_eliminated": mode != "sspl1",
        "proof_adapter": (
            "one_shot_exact_payload_cdf_identity_content_and_symbol_content"
            if mode != "sspl1"
            else "not_applicable_sspl1"
        ),
    }
    return {
        "schema": decode_worker.WORKER_SCHEMA,
        "mode": mode,
        "execution_mode": "production-resource",
        "blob_sha256": run._sha256_file(path),
        "blob_bytes": path.stat().st_size,
        "n": 19,
        "bit_tuple": list(bits),
        "elapsed_ns": elapsed_ns,
        "peak_rss_bytes": peak_rss_bytes,
        "symbol_sha256": symbol_sha256,
        "boundary_state_sha256": boundary_sha256,
        "exactness": {
            "expected_symbol_sha256": symbol_sha256,
            "expected_boundary_state_sha256": boundary_sha256,
            "symbol_hash_exact": True,
            "boundary_hash_exact": True,
            "all_exact": True,
        },
        "native": native_scope,
        "assets": {
            "cdf_sha256": assets.CDF_ASSET_SHA256,
            "cdf_bytes": assets.CDF_ASSET_SIZE,
            "loaded_before_timing": True,
        },
        "timing": {
            "scope": list(decode_worker.TIMING_SCOPE),
            "renderer_included": False,
        },
        "process": {
            "pid": pid,
            "platform": "linux",
            "pinned_cpu": cpu,
            "cpu_affinity": [cpu],
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss_linux_kib_times_1024",
        },
    }


def _synthetic_benchmark_record(
    outdir: Path,
    cell: dict[str, object],
    native_path: Path,
) -> dict[str, object]:
    image_id = cell["image_id"]
    bits = tuple(cell["bit_tuple"])
    base = run._cell_path(outdir, image_id, bits)
    primary_name, primary_bytes = run._primary_selection(cell)
    candidate_path = base / "streams/ssp2e.bin"
    primary_path = (
        outdir / cell["input"]["path"]
        if primary_name == "sspl1"
        else base / f"streams/{primary_name}.bin"
    )
    symbol_sha256 = cell["input"]["symbol_sha256"]
    boundary_sha256 = cell["input"]["decoded_state_sha256"]
    launches = []
    measured = {"candidate": [], "primary": []}

    def append(phase: str, repetition: int | None, arm: str) -> None:
        mode = "ssp2e" if arm == "candidate" else primary_name
        path = candidate_path if arm == "candidate" else primary_path
        offset = 0 if arm == "candidate" else 100
        repetition_value = 0 if repetition is None else repetition
        sample = _worker_sample_record(
            mode=mode,
            path=path,
            native_path=native_path,
            bits=bits,
            symbol_sha256=symbol_sha256,
            boundary_sha256=boundary_sha256,
            elapsed_ns=1_000 + offset + repetition_value * 10 + len(launches),
            peak_rss_bytes=10_000 + offset + repetition_value,
            pid=20_000 + len(launches),
            cpu=3,
        )
        launches.append(
            {
                "launch_index": len(launches),
                "phase": phase,
                "repetition": repetition,
                "arm": arm,
                "mode": mode,
                "sample": sample,
            }
        )
        if phase == "measured":
            measured[arm].append(sample)

    append("warmup", None, "primary")
    append("warmup", None, "candidate")
    image_index = run.oracle.FROZEN_IMAGE_IDS.index(image_id)
    tuple_index = run.oracle.FROZEN_BIT_TUPLES.index(bits)
    for repetition in range(9):
        order = (
            ("candidate", "primary")
            if (image_index + tuple_index + repetition) % 2 == 0
            else ("primary", "candidate")
        )
        for arm in order:
            append("measured", repetition, arm)
    summaries = {}
    for arm in ("candidate", "primary"):
        elapsed = sorted(sample["elapsed_ns"] for sample in measured[arm])
        rss = [sample["peak_rss_bytes"] for sample in measured[arm]]
        summaries[arm] = {
            "measured_count": 9,
            "fifth_sorted_elapsed_ns": elapsed[4],
            "maximum_peak_rss_bytes": max(rss),
            "sorted_elapsed_ns": elapsed,
            "peak_rss_bytes": rss,
        }
    render = {
        "order": ["primary", "candidate"],
        "primary": {
            "blob_path": str(primary_path),
            "blob_sha256": run._sha256_file(primary_path),
            "symbol_sha256": symbol_sha256,
            "boundary_state_sha256": boundary_sha256,
            "all_finite": True,
            "image_shape": [64, 80, 3],
            "elapsed_ns": 901,
        },
        "candidate": {
            "blob_path": str(candidate_path),
            "blob_sha256": run._sha256_file(candidate_path),
            "symbol_sha256": symbol_sha256,
            "boundary_state_sha256": boundary_sha256,
            "all_finite": True,
            "image_shape": [64, 80, 3],
            "elapsed_ns": 902,
        },
        "parity": {
            "rtol": 5e-4,
            "atol": 5e-4,
            "max_abs": 0.25,
            "within_tolerance": False,
            "bit_exact_diagnostic": False,
        },
        "decision_use": False,
    }
    streams = cell["execution"]["streams"]
    row = actual.seal_record(
        {
            "schema": actual.ROW_SCHEMA,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "sspl1_bytes": cell["execution"]["source"]["bytes"],
            "ssp2e_bytes": streams["ssp2e"]["complete_bytes"],
            "ssp2s_bytes": streams["ssp2s"]["complete_bytes"],
            "ssp2l_bytes": streams["ssp2l"]["complete_bytes"],
            "ssp2f_bytes": streams["ssp2f"]["complete_bytes"],
            "ssp2e_modeled_bytes": streams["ssp2e"]["modeled_bytes"],
            "ssp2s_modeled_bytes": streams["ssp2s"]["modeled_bytes"],
            "ssp2e_decode_ns": summaries["candidate"]["fifth_sorted_elapsed_ns"],
            "primary_decode_ns": summaries["primary"]["fifth_sorted_elapsed_ns"],
            "ssp2e_peak_rss_bytes": summaries["candidate"]["maximum_peak_rss_bytes"],
            "primary_peak_rss_bytes": summaries["primary"]["maximum_peak_rss_bytes"],
            "primary_bytes": primary_bytes,
            "primary_name": primary_name,
            "integrity": cell["integrity"],
        }
    )
    return actual.seal_record(
        {
            "schema": "structsplat.comp009.benchmark-cell.v1",
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "image_id": image_id,
            "bit_tuple": list(bits),
            "run_cell_sha256": cell["cell_record_sha256"],
            "primary_selection": {
                "name": primary_name,
                "bytes": primary_bytes,
                "tie_order": ["sspl1", "ssp2l", "ssp2f"],
            },
            "schedule_rule": "warmup-primary-candidate; measured-candidate-first-iff-(i+t+r)%2==0",
            "launches": launches,
            "summaries": summaries,
            "render_diagnostic": render,
            "actual_row": row,
            "confirmation_payloads_accessed": False,
        },
        "benchmark_cell_sha256",
    )


@pytest.fixture
def synthetic_benchmark(
    tmp_path: Path,
    synthetic_execution: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object], dict[str, object], Path]:
    cell, _ = _materialize_run_cell(tmp_path, synthetic_execution)
    native_path = synthetic_execution["native_path"]
    (tmp_path / "preflight.json").write_bytes(
        run._canonical_json({"environment": {"cpu_affinity": [3, 7]}})
    )
    monkeypatch.setattr(run.oracle, "FROZEN_GAUSSIAN_COUNT", 19)
    record = _synthetic_benchmark_record(tmp_path, cell, native_path)
    return tmp_path, cell, record, native_path


def test_benchmark_cell_recomputes_schedule_summaries_row_and_nongating_render(
    synthetic_benchmark: tuple[Path, dict[str, object], dict[str, object], Path],
) -> None:
    outdir, cell, record, native_path = synthetic_benchmark
    checked = run._verify_benchmark_cell(outdir, record, cell, native_path)
    assert checked == record
    assert checked["render_diagnostic"]["parity"]["within_tolerance"] is False
    assert checked["render_diagnostic"]["decision_use"] is False
    candidate = checked["summaries"]["candidate"]
    assert (
        candidate["fifth_sorted_elapsed_ns"]
        == sorted(
            launch["sample"]["elapsed_ns"]
            for launch in checked["launches"]
            if launch["phase"] == "measured" and launch["arm"] == "candidate"
        )[4]
    )
    assert candidate["maximum_peak_rss_bytes"] == max(candidate["peak_rss_bytes"])


def _mutate_benchmark_record(record: dict[str, object], mutation: str) -> None:
    launches = record["launches"]
    if mutation == "schedule_arm":
        launches[2]["arm"] = "primary"
    elif mutation == "schedule_repetition":
        launches[2]["repetition"] = 8
    elif mutation == "duplicate_pid":
        launches[2]["sample"]["process"]["pid"] = launches[1]["sample"]["process"]["pid"]
    elif mutation == "thread_map":
        launches[2]["sample"]["process"]["thread_environment"]["OMP_NUM_THREADS"] = "2"
    elif mutation == "cpu_affinity":
        launches[2]["sample"]["process"]["cpu_affinity"] = [7]
    elif mutation == "execution_mode":
        launches[2]["sample"]["execution_mode"] = "synthetic-test"
    elif mutation == "blob_hash":
        launches[2]["sample"]["blob_sha256"] = "0" * 64
    elif mutation == "bit_tuple":
        launches[2]["sample"]["bit_tuple"] = [16, 8, 8, 8]
    elif mutation == "native_hash":
        launches[2]["sample"]["native"]["library_sha256"] = "1" * 64
    elif mutation == "summary_median":
        record["summaries"]["candidate"]["fifth_sorted_elapsed_ns"] += 1
    elif mutation == "summary_rss":
        record["summaries"]["candidate"]["maximum_peak_rss_bytes"] += 1
    elif mutation == "primary_name":
        record["primary_selection"]["name"] = "ssp2f"
    elif mutation == "tie_order":
        record["primary_selection"]["tie_order"] = ["ssp2l", "sspl1", "ssp2f"]
    elif mutation == "actual_row":
        row = record["actual_row"]
        row["ssp2e_decode_ns"] += 1
        record["actual_row"] = actual.seal_record(row)
    elif mutation == "render_omission":
        record["render_diagnostic"]["candidate"].pop("blob_path")
    elif mutation == "render_decision_use":
        record["render_diagnostic"]["decision_use"] = True
    elif mutation == "render_hash":
        record["render_diagnostic"]["candidate"]["blob_sha256"] = "2" * 64
    elif mutation == "render_tolerance":
        record["render_diagnostic"]["parity"]["rtol"] = 1e-3
    elif mutation == "confirmation":
        record["confirmation_payloads_accessed"] = True
    elif mutation == "schedule_rule":
        record["schedule_rule"] = "nine candidate launches, then nine primary"
    else:  # pragma: no cover
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "schedule_arm",
        "schedule_repetition",
        "duplicate_pid",
        "thread_map",
        "cpu_affinity",
        "execution_mode",
        "blob_hash",
        "bit_tuple",
        "native_hash",
        "summary_median",
        "summary_rss",
        "primary_name",
        "tie_order",
        "actual_row",
        "render_omission",
        "render_decision_use",
        "render_hash",
        "render_tolerance",
        "confirmation",
        "schedule_rule",
    ],
)
def test_benchmark_cell_resealed_single_field_mutations_fail(
    synthetic_benchmark: tuple[Path, dict[str, object], dict[str, object], Path],
    mutation: str,
) -> None:
    outdir, cell, valid, native_path = synthetic_benchmark
    changed = copy.deepcopy(valid)
    _mutate_benchmark_record(changed, mutation)
    changed = actual.seal_record(changed, "benchmark_cell_sha256")
    with pytest.raises((run.LifecycleError, actual.ActualCoderError)):
        run._verify_benchmark_cell(outdir, changed, cell, native_path)


def test_primary_selection_uses_frozen_exact_tie_order() -> None:
    cell = {
        "execution": {
            "source": {"bytes": 123},
            "streams": {
                "ssp2l": {"complete_bytes": 123},
                "ssp2f": {"complete_bytes": 123},
            },
        }
    }
    assert run._primary_selection(cell) == ("sspl1", 123)
    cell["execution"]["source"]["bytes"] = 124
    assert run._primary_selection(cell) == ("ssp2l", 123)


def _resume_cell(metadata: dict[str, object], index: int) -> dict[str, object]:
    return {
        "image_id": metadata["image_id"],
        "bit_tuple": metadata["bit_tuple"],
        "cell_record_sha256": f"{index + 1:064x}",
    }


def test_run_development_resumes_journal_then_cell_artifact_at_cell_granularity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = []
    sources: dict[tuple[str, tuple[int, ...]], bytes] = {}
    for index, (image_id, bits) in enumerate(
        (image_id, bits)
        for image_id in run.oracle.FROZEN_IMAGE_IDS
        for bits in run.oracle.FROZEN_BIT_TUPLES
    ):
        relative = Path("inputs") / image_id / f"{run._tuple_label(bits)}.sspl1"
        source = f"source-{index}".encode()
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
        metadata = {
            "image_id": image_id,
            "bit_tuple": list(bits),
            "imported_path": relative.as_posix(),
            "symbol_sha256": f"{index + 100:064x}",
            "decoded_state_sha256": f"{index + 200:064x}",
        }
        streams.append(metadata)
        sources[(image_id, bits)] = source
    input_manifest = {
        "streams": streams,
        "input_manifest_sha256": "a" * 64,
    }
    first = _resume_cell(streams[0], 0)
    recovered = _resume_cell(streams[1], 1)
    run._append_jsonl(tmp_path / "run_cells.jsonl", first)
    recovered_path = (
        run._cell_path(tmp_path, recovered["image_id"], recovered["bit_tuple"]) / "cell.json"
    )
    recovered_path.parent.mkdir(parents=True, exist_ok=True)
    recovered_path.write_text("{}", encoding="utf-8")

    loaded: list[tuple[str, tuple[int, ...]]] = []
    executed: list[bytes] = []
    persisted: list[tuple[str, tuple[int, ...]]] = []
    monkeypatch.setattr(run, "_verify_input_manifest", lambda _outdir: input_manifest)
    monkeypatch.setattr(run, "_verify_run_cell", lambda _outdir, row: dict(row))

    def fake_load(_outdir: Path, image_id: str, bits: tuple[int, ...]) -> dict[str, object]:
        loaded.append((image_id, tuple(bits)))
        return recovered

    monkeypatch.setattr(run, "_load_run_cell", fake_load)
    monkeypatch.setattr(
        run,
        "verify_preflight",
        lambda _outdir: {
            "native": {"path": "native/lib.so"},
            "preflight_binding_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(arithmetic, "NativeArithmeticCoder", lambda _path: object())

    def fake_execute(source: bytes, _native: object, **_kwargs: object) -> object:
        executed.append(source)
        return SimpleNamespace()

    def fake_persist(
        _outdir: Path,
        metadata: dict[str, object],
        _result: object,
        _wall_seconds: float,
    ) -> dict[str, object]:
        identity = (str(metadata["image_id"]), tuple(metadata["bit_tuple"]))
        persisted.append(identity)
        return _resume_cell(metadata, streams.index(metadata))

    monkeypatch.setattr(execute, "execute_cell", fake_execute)
    monkeypatch.setattr(run, "_persist_execution_cell", fake_persist)
    manifest = run.run_development(tmp_path)

    expected_new = [
        (str(metadata["image_id"]), tuple(metadata["bit_tuple"])) for metadata in streams[2:]
    ]
    assert loaded == [(str(streams[1]["image_id"]), tuple(streams[1]["bit_tuple"]))]
    assert persisted == expected_new
    assert executed == [sources[identity] for identity in expected_new]
    journal = run._read_jsonl(tmp_path / "run_cells.jsonl")
    assert len(journal) == 16
    assert [row["cell_record_sha256"] for row in journal] == [
        f"{index + 1:064x}" for index in range(16)
    ]
    assert manifest["cell_count"] == 16


def test_resumable_journal_preserves_and_truncates_crash_tail(tmp_path: Path) -> None:
    journal = tmp_path / "run_cells.jsonl"
    row = {"cell": 0}
    prefix = run._canonical_json(row)
    tail = b'{"cell":1,"unterminated"'
    journal.write_bytes(prefix + tail)
    assert run._read_resumable_journal(journal, "run-dev") == [row]
    assert journal.read_bytes() == prefix
    recovery = (
        tmp_path / "recovery" / f"run-dev-partial-tail-{hashlib.sha256(tail).hexdigest()}.bin"
    )
    assert recovery.read_bytes() == tail
    assert run._read_resumable_journal(journal, "run-dev") == [row]


def _analysis_run_cell(index: int) -> dict[str, object]:
    return {
        "image_id": f"image-{index}",
        "bit_tuple": [12, 6, 6, 8],
        "execution": {
            "timings": {"fit_seconds": 1.0 + index},
            "models": {
                "ssp2e": {
                    "restart_spread": index,
                    "starts": [{"sweeps": start + 1} for start in range(8)],
                },
                "ssp2s": {
                    "restart_spread": index + 1,
                    "starts": [{"sweeps": start + 2} for start in range(8)],
                },
            },
        },
    }


def test_analysis_is_recomputed_from_verified_raw_rows_and_binds_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        actual.seal_record({"schema": actual.ROW_SCHEMA, "index": index}) for index in range(16)
    ]
    benchmark_cells = [{"benchmark_cell_sha256": f"{index + 1:064x}"} for index in range(16)]
    run_cells = [_analysis_run_cell(index) for index in range(16)]
    benchmark_manifest = {"benchmark_manifest_sha256": "b" * 64}
    observed: list[list[dict[str, object]]] = []

    def fake_analyze(received: list[dict[str, object]]) -> dict[str, object]:
        observed.append(received)
        return actual.seal_record(
            {
                "schema": actual.ANALYSIS_SCHEMA,
                "decision": "SYNTHETIC",
                "row_count": len(received),
            },
            "analysis_sha256",
        )

    monkeypatch.setattr(
        run,
        "verify_benchmark_manifest",
        lambda _outdir: (benchmark_manifest, benchmark_cells, rows),
    )
    monkeypatch.setattr(
        run,
        "verify_run_manifest",
        lambda _outdir: ({"run_manifest_sha256": "c" * 64}, run_cells),
    )
    monkeypatch.setattr(actual, "analyze_rows", fake_analyze)
    record = run.analyze_results(tmp_path)

    assert observed == [rows]
    assert record["raw_blobs_and_worker_samples_reparsed"] is True
    assert record["actual_row_sha256"] == [row["record_sha256"] for row in rows]
    assert record["benchmark_cell_sha256"] == [
        cell["benchmark_cell_sha256"] for cell in benchmark_cells
    ]
    assert len(record["execution_diagnostics"]) == 16
    assert all(
        diagnostic["decision_use"] is False for diagnostic in record["execution_diagnostics"]
    )


def test_analysis_rejects_stale_file_against_raw_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [actual.seal_record({"index": index}) for index in range(16)]
    monkeypatch.setattr(
        run,
        "verify_benchmark_manifest",
        lambda _outdir: (
            {"benchmark_manifest_sha256": "b" * 64},
            [{"benchmark_cell_sha256": f"{index + 1:064x}"} for index in range(16)],
            rows,
        ),
    )
    monkeypatch.setattr(
        run,
        "verify_run_manifest",
        lambda _outdir: (
            {"run_manifest_sha256": "c" * 64},
            [_analysis_run_cell(index) for index in range(16)],
        ),
    )
    recomputed = actual.seal_record(
        {"schema": actual.ANALYSIS_SCHEMA, "decision": "RECOMPUTED"},
        "analysis_sha256",
    )
    monkeypatch.setattr(actual, "analyze_rows", lambda _rows: recomputed)
    (tmp_path / "analysis.json").write_bytes(
        run._canonical_json(
            actual.seal_record(
                {"schema": actual.ANALYSIS_SCHEMA, "decision": "STALE"},
                "analysis_sha256",
            )
        )
    )
    with pytest.raises(run.LifecycleError, match="differs from raw-artifact"):
        run.analyze_results(tmp_path)


def _write_tar(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_captured_source_extraction_rejects_archive_hash_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_tar(tmp_path / "source_snapshot.tar", {"safe.py": b"pass\n"})
    monkeypatch.setattr(
        run,
        "verify_preflight",
        lambda _outdir: {"source_archive_sha256": "0" * 64},
    )
    destination = tmp_path / "extracted"
    destination.mkdir()
    with pytest.raises(run.LifecycleError, match="archive drift"):
        run._extract_captured_source(tmp_path, destination)


def test_captured_source_extraction_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "source_snapshot.tar"
    _write_tar(archive_path, {"../escape.py": b"pass\n"})
    monkeypatch.setattr(
        run,
        "verify_preflight",
        lambda _outdir: {"source_archive_sha256": run._sha256_file(archive_path)},
    )
    destination = tmp_path / "extracted"
    destination.mkdir()
    with pytest.raises(run.LifecycleError, match="unsafe path"):
        run._extract_captured_source(tmp_path, destination)
    assert not (tmp_path / "escape.py").exists()


def test_captured_source_extraction_rejects_manifested_file_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"print('captured')\n"
    manifest = actual.seal_record(
        {
            "schema": "structsplat.comp009.source-manifest.v1",
            "files": [
                {
                    "path": "captured.py",
                    "bytes": len(payload),
                    "sha256": "0" * 64,
                }
            ],
        },
        "source_manifest_sha256",
    )
    manifest_payload = run._canonical_json(manifest)
    archive_path = tmp_path / "source_snapshot.tar"
    _write_tar(
        archive_path,
        {"SOURCE_MANIFEST.json": manifest_payload, "captured.py": payload},
    )
    (tmp_path / "source_manifest.json").write_bytes(manifest_payload)
    monkeypatch.setattr(
        run,
        "verify_preflight",
        lambda _outdir: {"source_archive_sha256": run._sha256_file(archive_path)},
    )
    destination = tmp_path / "extracted"
    destination.mkdir()
    with pytest.raises(run.LifecycleError, match="extraction drift"):
        run._extract_captured_source(tmp_path, destination)


def test_execution_without_timings_is_deep_nonmutating_and_exact() -> None:
    first = {
        "schema": execute.EXECUTION_SCHEMA,
        "timings": {"fit_seconds": 1.0, "encode_seconds": 2.0},
        "nested": {"timings": {"semantic": "retained"}, "value": [1, 2, 3]},
        "execution_sha256": "a" * 64,
    }
    second = copy.deepcopy(first)
    second["timings"] = {"fit_seconds": 9.0, "encode_seconds": 10.0}
    second["execution_sha256"] = "b" * 64
    original = copy.deepcopy(first)
    normalized = run._execution_without_timings(first)
    assert normalized == run._execution_without_timings(second)
    assert normalized == {
        "schema": execute.EXECUTION_SCHEMA,
        "nested": {"timings": {"semantic": "retained"}, "value": [1, 2, 3]},
    }
    assert first == original


@pytest.mark.parametrize(
    ("command", "function_name", "seal_field"),
    [
        ("preflight", "preflight", "preflight_binding_sha256"),
        ("import-dev", "import_development", "input_manifest_sha256"),
        ("run-dev", "run_development", "run_manifest_sha256"),
        ("benchmark-dev", "benchmark_development", "benchmark_manifest_sha256"),
        ("analyze", "analyze_results", "analysis_record_sha256"),
        ("replay", "replay_results", "replay_sha256"),
    ],
)
def test_cli_routes_each_public_stage_and_prints_its_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    function_name: str,
    seal_field: str,
) -> None:
    observed: list[tuple[Path, ...]] = []
    seal = hashlib.sha256(command.encode()).hexdigest()

    def fake_stage(*paths: Path) -> dict[str, str]:
        observed.append(paths)
        return {seal_field: seal}

    monkeypatch.setattr(run, function_name, fake_stage)
    arguments = [command, "--outdir", str(tmp_path)]
    if command in {"preflight", "import-dev"}:
        arguments.extend(("--input-dir", str(tmp_path / "input")))
    assert run.main(arguments) == 0
    assert capsys.readouterr().out == f"{seal}\n"
    assert observed
    assert observed[0][0] == tmp_path.resolve()
    if command in {"preflight", "import-dev"}:
        assert observed[0][1] == (tmp_path / "input").resolve()


def test_parser_wires_captured_replay_worker() -> None:
    args = run.build_parser().parse_args(["replay-worker", "--outdir", "/tmp/comp009-output"])
    assert args.command == "replay-worker"
    assert args.outdir == Path("/tmp/comp009-output")


def test_captured_replay_worker_requires_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COMP009_CAPTURED_SOURCE", raising=False)
    monkeypatch.delenv("COMP009_CAPTURED_ROOT", raising=False)
    with pytest.raises(run.LifecycleError, match="restricted"):
        run.captured_replay_worker(tmp_path)


def test_captured_replay_worker_rejects_wrong_module_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMP009_CAPTURED_SOURCE", "1")
    monkeypatch.setenv("COMP009_CAPTURED_ROOT", str(tmp_path / "different-source"))
    with pytest.raises(run.LifecycleError, match="did not load from"):
        run.captured_replay_worker(tmp_path)


def test_module_record_captures_resolved_origin_size_and_hash() -> None:
    record = run._module_record("benchmarks.ssp2e_execute")
    origin = Path(record["origin"])
    assert origin == Path(execute.__file__).resolve()
    assert record["bytes"] == origin.stat().st_size
    assert record["sha256"] == hashlib.sha256(origin.read_bytes()).hexdigest()
