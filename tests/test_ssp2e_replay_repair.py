from __future__ import annotations

import copy
import io
import os
import platform
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import ssp2e_replay_repair as repair


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/comp009_ssp2e_actual_dev_v1_2026-07-16"


def _records() -> dict[str, dict]:
    return {
        name: repair.read_object(ARTIFACT / spec[0])
        for name, spec in repair._JSON_SPECS.items()
    }


def _valid_worker() -> dict:
    return repair.seal_record(
        {
            "schema": "structsplat.comp009.captured-replay-worker.v1",
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "source_manifest_sha256": repair.FROZEN_BINDINGS["source_manifest_sha256"],
            "runner_relative_path": "benchmarks/ssp2e_actual_run.py",
            "runner_sha256": repair.EXPECTED_CAPTURED_RUNNER_SHA256,
            "run_manifest_sha256": repair.FROZEN_BINDINGS["run_manifest_sha256"],
            "analysis_record_sha256": repair.FROZEN_BINDINGS["analysis_record_sha256"],
            "analysis_sha256": repair.FROZEN_BINDINGS["analysis_sha256"],
            "decision": repair.FROZEN_BINDINGS["decision"],
            "cells": repair.expected_worker_cells(ARTIFACT),
            "cell_count": 16,
            "actual_timing_samples_reused_not_rerun": True,
            "underlying_field_or_qat_refit": False,
            "target_pixels_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "captured_replay_worker_sha256",
    )


def _reseal_worker(worker: dict) -> dict:
    return repair.seal_record(worker, "captured_replay_worker_sha256")


def test_frozen_artifact_exact_metadata_stream_model_and_timing_bindings() -> None:
    result = repair.verify_frozen_artifact(ARTIFACT)
    assert result["comp009_seals"] == repair.FROZEN_BINDINGS
    assert result["scientific_pixels_accessed"] is False
    assert result["confirmation_payloads_accessed"] is False


@pytest.mark.parametrize(
    ("name", "seal_field"),
    [
        (name, spec[1])
        for name, spec in repair._JSON_SPECS.items()
    ],
)
def test_every_frozen_record_seal_mutation_is_rejected(name: str, seal_field: str) -> None:
    record = repair.read_object(ARTIFACT / repair._JSON_SPECS[name][0])
    record[seal_field] = "0" * 64
    with pytest.raises(repair.RepairError, match=seal_field):
        repair.verify_record(record, seal_field)


def test_coherently_resealed_but_substantively_mutated_frozen_record_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = repair.read_object

    def changed(path: Path, *, require_canonical: bool = True):
        record = original(path, require_canonical=require_canonical)
        if path.name == "analysis.json":
            record["decision"] = "KEEP_MUTATED_CODEC"
            return repair.seal_record(record, "analysis_sha256")
        return record

    monkeypatch.setattr(repair, "read_object", changed)
    with pytest.raises(repair.RepairError, match="frozen analysis seal/schema drift"):
        repair.verify_frozen_artifact(ARTIFACT)


@pytest.mark.parametrize("count", [15, 17])
def test_worker_rejects_wrong_cell_count_even_when_coherently_resealed(count: int) -> None:
    worker = _valid_worker()
    cells = worker["cells"]
    worker["cells"] = cells[:count] if count == 15 else [*cells, copy.deepcopy(cells[-1])]
    worker["cell_count"] = count
    with pytest.raises(repair.RepairError, match="identity/scope/decision drift"):
        repair.validate_worker_record(_reseal_worker(worker), ARTIFACT)


@pytest.mark.parametrize("mutation", ["reorder", "duplicate"])
def test_worker_rejects_reordered_or_duplicate_cells(mutation: str) -> None:
    worker = _valid_worker()
    if mutation == "reorder":
        worker["cells"][0], worker["cells"][1] = worker["cells"][1], worker["cells"][0]
    else:
        worker["cells"][1] = copy.deepcopy(worker["cells"][0])
    with pytest.raises(repair.RepairError, match="identity/scope/decision drift"):
        repair.validate_worker_record(_reseal_worker(worker), ARTIFACT)


@pytest.mark.parametrize("stream", repair.STREAM_NAMES)
def test_worker_rejects_each_complete_stream_hash_mutation(stream: str) -> None:
    worker = _valid_worker()
    worker["cells"][0]["complete_stream_sha256"][stream] = "0" * 64
    with pytest.raises(repair.RepairError, match="identity/scope/decision drift"):
        repair.validate_worker_record(_reseal_worker(worker), ARTIFACT)


@pytest.mark.parametrize(
    "field",
    ["normalized_execution_sha256", "input_sha256"],
)
def test_worker_rejects_execution_model_or_input_identity_mutation(field: str) -> None:
    worker = _valid_worker()
    worker["cells"][0][field] = "0" * 64
    with pytest.raises(repair.RepairError, match="identity/scope/decision drift"):
        repair.validate_worker_record(_reseal_worker(worker), ARTIFACT)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runner_sha256", "0" * 64),
        ("analysis_sha256", "0" * 64),
        ("decision", "KEEP"),
        ("actual_timing_samples_reused_not_rerun", False),
        ("underlying_field_or_qat_refit", True),
        ("target_pixels_opened", True),
        ("confirmation_payloads_accessed", True),
    ],
)
def test_worker_rejects_runner_timing_analysis_or_scope_drift(field: str, value: object) -> None:
    worker = _valid_worker()
    worker[field] = value
    with pytest.raises(repair.RepairError, match="identity/scope/decision drift"):
        repair.validate_worker_record(_reseal_worker(worker), ARTIFACT)


def test_worker_rejects_coherently_resealed_extra_field() -> None:
    worker = _valid_worker()
    worker["ignored_success"] = True
    with pytest.raises(repair.RepairError, match="identity/scope/decision drift"):
        repair.validate_worker_record(_reseal_worker(worker), ARTIFACT)


@pytest.mark.parametrize(
    "selection_field",
    ["selected_grid_sha256", "selected_head_sha256", "selected_model_sha256"],
)
def test_raw_binding_rejects_model_grid_head_selection_mutation(
    selection_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _records()
    original = repair.read_jsonl

    def mutated(path: Path):
        rows = original(path)
        if path.name == "run_cells.jsonl":
            rows = copy.deepcopy(rows)
            execution = rows[0]["execution"]
            execution["models"]["ssp2e"][selection_field] = "0" * 64
            rows[0]["execution"] = repair.seal_record(execution, "execution_sha256")
            rows[0] = repair.seal_record(rows[0], "cell_record_sha256")
        return rows

    monkeypatch.setattr(repair, "read_jsonl", mutated)
    with pytest.raises(repair.RepairError, match="run-manifest cell-seal binding drift"):
        repair._verify_manifest_file_bindings(ARTIFACT, records)


def test_raw_binding_rejects_persisted_timing_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = repair.sha256_file

    def wrong_timing_hash(path: Path) -> str:
        if path.name == "benchmark_cells.jsonl":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(repair, "sha256_file", wrong_timing_hash)
    with pytest.raises(repair.RepairError, match="frozen raw artifact drift"):
        repair._verify_manifest_file_bindings(ARTIFACT, _records())


def _asset_fixture(tmp_path: Path):
    directory = tmp_path / "captured/benchmarks/assets"
    directory.mkdir(parents=True)
    (directory / "ssp2e_cdf_v1.bin").write_bytes(b"cdf")
    (directory / "ssp2e_cost_q24_v1.bin").write_bytes(b"cost")
    expected = {
        "cdf_path": "/frozen/cdf.bin",
        "cdf_sha256": repair.EXPECTED_CDF_SHA256,
        "cdf_size": 655388,
        "cost_path": "/frozen/cost.bin",
        "cost_sha256": repair.EXPECTED_COST_SHA256,
        "cost_size": 131088,
        "alphabets": [64, 256],
        "cost_count": 32768,
    }
    observed = {
        **expected,
        "cdf_path": str(directory / "ssp2e_cdf_v1.bin"),
        "cost_path": str(directory / "ssp2e_cost_q24_v1.bin"),
    }
    calls = []

    def verify_assets(path=directory):
        calls.append(Path(path).resolve())
        return copy.deepcopy(observed)

    assets = SimpleNamespace(verify_assets=verify_assets)
    runner = SimpleNamespace(assets=assets)
    return directory.parents[1], runner, expected, observed, calls


def test_asset_wrapper_actively_parses_extracted_assets_and_only_substitutes_paths(
    tmp_path: Path,
) -> None:
    extracted, runner, expected, _, calls = _asset_fixture(tmp_path)
    original, state = repair.install_asset_path_normalizer(runner, expected, extracted)
    for _ in range(5):
        assert runner.assets.verify_assets() == expected
    assert calls == [(extracted / "benchmarks/assets").resolve()] * 5
    assert state["calls"] == 5
    runner.assets.verify_assets = original


@pytest.mark.parametrize(
    "field",
    ["cdf_sha256", "cdf_size", "cost_sha256", "cost_size", "alphabets", "cost_count"],
)
def test_asset_wrapper_rejects_every_nonpath_identity_mutation(
    field: str, tmp_path: Path
) -> None:
    extracted, runner, expected, observed, _ = _asset_fixture(tmp_path)
    observed[field] = "mutated" if not isinstance(observed[field], list) else [256, 64]
    repair.install_asset_path_normalizer(runner, expected, extracted)
    with pytest.raises(repair.RepairError, match="asset identity drift"):
        runner.assets.verify_assets()


@pytest.mark.parametrize("target", ["cdf_path", "cost_path"])
def test_asset_wrapper_rejects_wrong_extracted_target(target: str, tmp_path: Path) -> None:
    extracted, runner, expected, observed, _ = _asset_fixture(tmp_path)
    observed[target] = str(extracted / "benchmarks/assets/wrong.bin")
    repair.install_asset_path_normalizer(runner, expected, extracted)
    with pytest.raises(repair.RepairError, match="wrong extracted target"):
        runner.assets.verify_assets()


def test_asset_wrapper_rejects_extra_normalized_field(tmp_path: Path) -> None:
    extracted, runner, expected, observed, _ = _asset_fixture(tmp_path)
    observed["extra"] = True
    repair.install_asset_path_normalizer(runner, expected, extracted)
    with pytest.raises(repair.RepairError, match="unexpected identity surface"):
        runner.assets.verify_assets()


def _renderer_record() -> dict:
    return repair.read_object(ARTIFACT / "preflight.json")["renderer"]


def test_renderer_reuse_returns_independent_exact_copies_and_leaves_verifier_unchanged() -> None:
    renderer = _renderer_record()

    def original_proof():
        raise AssertionError("must not execute")

    def unchanged_verifier(expected):
        assert runner.renderer_runtime_proof() == expected

    runner = SimpleNamespace(
        renderer_runtime_proof=original_proof,
        verify_renderer_runtime=unchanged_verifier,
    )
    verifier = runner.verify_renderer_runtime
    original, state = repair.install_renderer_proof_reuse(runner, renderer)
    first = runner.renderer_runtime_proof()
    first["scope"] = "mutated-copy"
    second = runner.renderer_runtime_proof()
    assert second == renderer
    assert first != second
    assert runner.verify_renderer_runtime is verifier
    assert state["calls"] == 2
    runner.renderer_runtime_proof = original


@pytest.mark.parametrize("mutation", ["scope", "toy", "abi", "binary"])
def test_renderer_reuse_rejects_scope_toy_abi_or_binary_mutation(mutation: str) -> None:
    renderer = copy.deepcopy(_renderer_record())
    if mutation == "scope":
        renderer["scope"] = "gating"
    elif mutation == "toy":
        renderer["toy_output_values"][0] = 1.0
    elif mutation == "abi":
        renderer["extension"]["abi"]["ldd"]["output_sha256"] = "0" * 64
    else:
        renderer["extension"]["sha256"] = "0" * 64
    with pytest.raises(repair.RepairError, match="renderer proof drift"):
        repair.verify_frozen_renderer_record(renderer)


def test_renderer_source_attestation_and_each_source_manifest_mutation(tmp_path: Path) -> None:
    extracted = tmp_path / "source"
    manifest = repair.extract_captured_source(ARTIFACT, extracted)
    assert repair.attest_renderer_sources(manifest, extracted) == [
        {"path": path, "bytes": size, "sha256": digest}
        for path, (size, digest) in repair.EXPECTED_RENDERER_SOURCES.items()
    ]
    for relative in repair.EXPECTED_RENDERER_SOURCES:
        changed = copy.deepcopy(manifest)
        entry = next(item for item in changed["files"] if item["path"] == relative)
        entry["sha256"] = "0" * 64
        with pytest.raises(repair.RepairError, match="source-manifest drift"):
            repair.attest_renderer_sources(changed, extracted)


def _make_archive_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_name: str | None = None,
    link_name: str | None = None,
    omit_source: bool = False,
    corrupt_source: bool = False,
):
    root = tmp_path / "artifact"
    root.mkdir()
    source_name = "benchmarks/captured.py"
    source_payload = b"captured\n"
    manifest = repair.seal_record(
        {
            "schema": "synthetic",
            "files": [
                {
                    "path": source_name,
                    "bytes": len(source_payload),
                    "sha256": repair.sha256_bytes(source_payload),
                }
            ],
        },
        "source_manifest_sha256",
    )
    (root / "source_manifest.json").write_bytes(repair.canonical_json(manifest))
    archive_path = root / "source_snapshot.tar"
    with tarfile.open(archive_path, "w") as archive:
        manifest_payload = repair.canonical_json(manifest)
        info = tarfile.TarInfo("SOURCE_MANIFEST.json")
        info.size = len(manifest_payload)
        archive.addfile(info, io.BytesIO(manifest_payload))
        if not omit_source:
            payload = b"wrong\n" if corrupt_source else source_payload
            info = tarfile.TarInfo(source_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if extra_name is not None:
            payload = b"extra"
            info = tarfile.TarInfo(extra_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if link_name is not None:
            info = tarfile.TarInfo(link_name)
            info.type = tarfile.SYMTYPE
            info.linkname = source_name
            archive.addfile(info)
    monkeypatch.setitem(
        repair.FROZEN_BINDINGS, "source_archive_sha256", repair.sha256_file(archive_path)
    )
    monkeypatch.setitem(
        repair.FROZEN_BINDINGS,
        "source_manifest_sha256",
        manifest["source_manifest_sha256"],
    )
    return root


def test_safe_archive_extraction_accepts_exact_regular_manifested_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_archive_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "destination"
    repair.extract_captured_source(root, destination)
    assert (destination / "benchmarks/captured.py").read_bytes() == b"captured\n"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "bad\\path"])
def test_safe_archive_rejects_traversal_absolute_and_backslash_paths(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_archive_fixture(tmp_path, monkeypatch, extra_name=name)
    with pytest.raises(repair.RepairError, match="unsafe source archive member path"):
        repair.extract_captured_source(root, tmp_path / "destination")


def test_safe_archive_rejects_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_archive_fixture(tmp_path, monkeypatch, link_name="linked")
    with pytest.raises(repair.RepairError, match="link or non-regular"):
        repair.extract_captured_source(root, tmp_path / "destination")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extra_name": "extra.py"}, "missing or unmanifested"),
        ({"omit_source": True}, "missing or unmanifested"),
        ({"corrupt_source": True}, "member drift"),
    ],
)
def test_safe_archive_rejects_extra_missing_or_drifted_files(
    kwargs: dict, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_archive_fixture(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(repair.RepairError, match=message):
        repair.extract_captured_source(root, tmp_path / "destination")


def test_native_build_and_renderer_import_guards_fail_closed() -> None:
    runner = SimpleNamespace(arithmetic=SimpleNamespace(build_native=lambda: "built"))
    original, state = repair.install_native_build_guard(runner)
    with pytest.raises(repair.RepairError, match="native arithmetic rebuild"):
        runner.arithmetic.build_native()
    assert state["attempts"] == 1
    runner.arithmetic.build_native = original

    guard = repair._RendererImportGuard()
    with pytest.raises(repair.RepairError, match="renderer JIT/load"):
        guard.find_spec("structsplat.cuda_render", None)
    assert guard.attempts == ["structsplat.cuda_render"]


def test_comp009_write_guard_blocks_writes_and_restores_every_patch(tmp_path: Path) -> None:
    root = tmp_path / "comp009"
    root.mkdir()
    path = root / "data.bin"
    path.write_bytes(b"frozen")
    guard = repair._Comp009WriteGuard(root)
    guard.install()
    originals = dict(guard._original_mutators)
    try:
        assert path.read_bytes() == b"frozen"
        with pytest.raises(repair.RepairError, match="write into frozen"):
            path.write_bytes(b"mutated")
        with pytest.raises(repair.RepairError, match="write into frozen"):
            os.unlink(path)
    finally:
        guard.restore()
    assert all(getattr(os, name) is original for name, original in originals.items())
    path.write_bytes(b"restored")
    assert path.read_bytes() == b"restored"


def test_module_origin_guard_rejects_live_leakage(tmp_path: Path) -> None:
    extracted = tmp_path / "extracted"
    benchmark = extracted / "benchmarks/module.py"
    package = extracted / "src/structsplat/__init__.py"
    benchmark.parent.mkdir(parents=True)
    package.parent.mkdir(parents=True)
    benchmark.write_text("# benchmark\n", encoding="utf-8")
    package.write_text("# package\n", encoding="utf-8")
    modules = {
        name: SimpleNamespace(__file__=str(benchmark)) for name in repair._CAPTURED_MODULES
    }
    modules["structsplat"] = SimpleNamespace(__file__=str(package))
    assert repair.assert_module_origins(extracted, modules)
    outside = tmp_path / "live.py"
    outside.write_text("# live\n", encoding="utf-8")
    modules[repair._CAPTURED_MODULES[0]] = SimpleNamespace(__file__=str(outside))
    with pytest.raises(repair.RepairError, match="live-module leakage"):
        repair.assert_module_origins(extracted, modules)


def test_module_origin_guard_canonicalizes_python_m_main(
    tmp_path: Path,
) -> None:
    extracted = tmp_path / "extracted"
    repair.extract_captured_source(ARTIFACT, extracted)
    child = extracted / "benchmarks/ssp2e_replay_repair.py"
    child.write_bytes((ROOT / repair.REPAIR_SOURCE_PATHS[1]).read_bytes())
    modules = {
        "benchmarks": SimpleNamespace(__file__=str(extracted / "benchmarks/__init__.py")),
        "__main__": SimpleNamespace(__file__=str(child)),
        **{
            name: SimpleNamespace(__file__=str(extracted / f"{name.replace('.', '/')}.py"))
            for name in repair._CAPTURED_MODULES
        },
    }
    preflight = _child_preflight()
    expected_before, expected_after = repair.expected_module_origin_records(
        ARTIFACT, preflight
    )

    before = repair.assert_module_origins(extracted, modules)
    assert before == expected_before
    assert [record["name"] for record in before].count(
        "benchmarks.ssp2e_replay_repair"
    ) == 1
    assert all(record["name"] != "__main__" for record in before)
    assert "benchmarks.ssp2e_replay_repair" not in modules

    modules["structsplat"] = SimpleNamespace(
        __file__=str(extracted / "src/structsplat/__init__.py")
    )
    assert repair.assert_module_origins(extracted, modules) == expected_after


def _synthetic_runtime_inputs(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("COMP010_SYNTHETIC_THREAD", "1")
    environment = {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {"COMP010_SYNTHETIC_THREAD": "1"},
        "accelerator": {"modules": []},
    }
    loader = {
        "loader_environment": {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
        },
        "resolved_libraries": [],
    }
    return {"environment": environment, "loader": loader}


def test_runtime_state_rejects_environment_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _synthetic_runtime_inputs(monkeypatch)
    assert repair.runtime_state(frozen)["thread_environment"] == {
        "COMP010_SYNTHETIC_THREAD": "1"
    }
    monkeypatch.setenv("COMP010_SYNTHETIC_THREAD", "2")
    with pytest.raises(repair.RepairError, match="environment differs"):
        repair.runtime_state(frozen)


def _strict_preflight_record(comp009: Path) -> dict:
    return repair.seal_record(
        {
            "schema": repair.PREFLIGHT_SCHEMA,
            "protocol": repair.PROTOCOL,
            "stage": "preflight",
            "comp009_directory": str(comp009.resolve()),
            "frozen_inputs": {},
            "comp009_inventory": {},
            "repair_sources": {},
            "runtime": {},
            "lifecycle_exception_allowlist": [
                "assets.cdf_path",
                "assets.cost_path",
                "sealed_non_gating_renderer_proof_reuse",
            ],
            "codec_streams_will_be_regenerated_ephemerally": True,
            "persisted_codec_streams_will_be_created_or_replaced": False,
            "ephemeral_execution_diagnostic_timings_will_be_generated_and_discarded": True,
            "persisted_benchmark_resource_timing_samples_will_be_reused_not_generated_or_replaced": True,
            "scientific_pixels_accessed": False,
            "confirmation_payloads_accessed": False,
        },
        "repair_preflight_sha256",
    )


def test_repair_preflight_rejects_coherently_resealed_extra_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    record = _strict_preflight_record(tmp_path)
    record["ignored_success"] = True
    record = repair.seal_record(record, "repair_preflight_sha256")
    (outdir / "preflight.json").write_bytes(repair.canonical_json(record))
    monkeypatch.setattr(repair, "verify_frozen_artifact", lambda _path: {})
    monkeypatch.setattr(repair, "artifact_inventory", lambda _path: {})
    monkeypatch.setattr(repair, "repair_source_manifest", lambda: {})
    monkeypatch.setattr(repair, "runtime_state", lambda _frozen: {})
    with pytest.raises(repair.RepairError, match="scope/schema drift"):
        repair.verify_repair_preflight(outdir, tmp_path)


def _child_preflight() -> dict:
    source = ROOT / repair.REPAIR_SOURCE_PATHS[1]
    return {
        "repair_preflight_sha256": "p" * 64,
        "repair_sources": {
            "files": [
                {
                    "path": repair.REPAIR_SOURCE_PATHS[1],
                    "bytes": source.stat().st_size,
                    "sha256": repair.sha256_file(source),
                }
            ]
        },
    }


def _valid_child() -> tuple[dict, dict]:
    preflight = _child_preflight()
    before, after = repair.expected_module_origin_records(ARTIFACT, preflight)
    child = repair.seal_record(
        {
            "schema": repair.CHILD_SCHEMA,
            "protocol": repair.PROTOCOL,
            "repair_preflight_sha256": preflight["repair_preflight_sha256"],
            "source_manifest_sha256": repair.FROZEN_BINDINGS["source_manifest_sha256"],
            "source_archive_sha256": repair.FROZEN_BINDINGS["source_archive_sha256"],
            "child_entry_relative_path": "benchmarks/ssp2e_replay_repair.py",
            "child_entry_sha256": preflight["repair_sources"]["files"][0]["sha256"],
            "module_origins_before_worker": before,
            "module_origins_after_worker": after,
            "asset_path_normalization": {
                "calls": 5,
                "changed_fields": ["cdf_path", "cost_path"],
                "original_function_module": "benchmarks.ssp2e_assets",
                "original_function_name": "verify_assets",
                "extracted_cdf_sha256": repair.EXPECTED_CDF_SHA256,
                "extracted_cost_sha256": repair.EXPECTED_COST_SHA256,
                "all_non_path_fields_equal": True,
                "original_called_on_extracted_asset_directory": True,
            },
            "renderer_source_attestation": [
                {"path": path, "bytes": size, "sha256": digest}
                for path, (size, digest) in repair.EXPECTED_RENDERER_SOURCES.items()
            ],
            "renderer_proof_reuse": {
                "calls": 5,
                "renderer_runtime_reexecuted": False,
                "exact_renderer_binary_replay_claimed": False,
                "sealed_renderer_record_sha256": repair.EXPECTED_RENDERER_RECORD_SHA256,
                "captured_verify_renderer_runtime_left_unchanged": True,
                "isolated_copy_returned_each_call": True,
            },
            "renderer_jit_load_guard": {
                "blocked_modules": sorted(repair._RendererImportGuard.BLOCKED),
                "attempts": [],
                "residency_before": {"modules": [], "mapped_paths": []},
                "residency_after": {"modules": [], "mapped_paths": []},
            },
            "native_rebuild_guard": {
                "attempts": 0,
                "persisted_native_loading_allowed": True,
            },
            "comp009_python_write_guard": {
                "blocked_operations": [],
                "python_open_and_os_mutators_guarded": True,
                "guard_restored": True,
                "transient_write_guard_limit": repair.WRITE_GUARD_LIMIT,
            },
            "persisted_native_sha256": repair.EXPECTED_NATIVE_SHA256,
            "persisted_native_loaded_and_frozen_proof_reverified": True,
            "captured_worker": _valid_worker(),
            "exactly_two_lifecycle_exceptions_exercised": True,
            "all_wrappers_and_guards_restored": True,
            "ephemeral_execution_diagnostic_timings_generated_and_discarded": True,
            "persisted_benchmark_resource_timing_samples_reused_not_rerun": True,
            "codec_streams_regenerated_ephemerally": True,
            "persisted_codec_streams_created_or_replaced": False,
            "underlying_field_or_qat_refit": False,
            "renderer_runtime_reexecuted": False,
            "exact_renderer_binary_replay_claimed": False,
            "scientific_pixels_accessed": False,
            "confirmation_payloads_accessed": False,
            "decision": repair.FROZEN_BINDINGS["decision"],
        },
        "repair_child_sha256",
    )
    return child, preflight


def test_strict_child_record_accepts_only_exact_nested_hash_bound_surface() -> None:
    child, preflight = _valid_child()
    repair.validate_child_record(child, ARTIFACT, preflight)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("asset_path_normalization", "calls"), 4),
        (("asset_path_normalization", "extracted_cdf_sha256"), "0" * 64),
        (("asset_path_normalization", "original_function_name"), "fake"),
        (("renderer_proof_reuse", "calls"), 6),
        (("renderer_proof_reuse", "sealed_renderer_record_sha256"), "0" * 64),
        (("renderer_jit_load_guard", "attempts"), ["structsplat.cuda_render"]),
        (("renderer_jit_load_guard", "residency_after"), {"modules": ["x"], "mapped_paths": []}),
        (("native_rebuild_guard", "attempts"), 1),
        (("comp009_python_write_guard", "blocked_operations"), ["open:w"]),
        (("module_origins_before_worker", 0, "sha256"), "0" * 64),
        (("module_origins_after_worker", 0, "name"), "live.module"),
        (("persisted_native_sha256",), "0" * 64),
        (("persisted_codec_streams_created_or_replaced",), True),
        (("renderer_runtime_reexecuted",), True),
        (("scientific_pixels_accessed",), True),
        (("confirmation_payloads_accessed",), True),
    ],
)
def test_strict_child_rejects_nested_exception_origin_guard_native_or_scope_drift(
    path: tuple, value: object
) -> None:
    child, preflight = _valid_child()
    target = child
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    child = repair.seal_record(child, "repair_child_sha256")
    with pytest.raises(repair.RepairError, match="child binding/scope drift"):
        repair.validate_child_record(child, ARTIFACT, preflight)


def test_strict_child_rejects_coherently_resealed_extra_nested_field() -> None:
    child, preflight = _valid_child()
    child["renderer_proof_reuse"]["ignored_success"] = True
    child = repair.seal_record(child, "repair_child_sha256")
    with pytest.raises(repair.RepairError, match="child binding/scope drift"):
        repair.validate_child_record(child, ARTIFACT, preflight)


def test_two_randomized_launches_require_identical_children_and_unchanged_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    inventory = {"inventory_sha256": "i" * 64}
    preflight = {
        "repair_preflight_sha256": "p" * 64,
        "comp009_inventory": inventory,
        "frozen_inputs": {},
    }
    monkeypatch.setattr(repair, "verify_repair_preflight", lambda *_args: preflight)
    monkeypatch.setattr(repair, "artifact_inventory", lambda _path: inventory)
    child = {"repair_child_sha256": "c" * 64, "scientific": "exact"}
    calls = []

    def launch(index, *_args):
        calls.append(index)
        return (
            {"source_manifest_sha256": repair.FROZEN_BINDINGS["source_manifest_sha256"]},
            copy.deepcopy(child),
            {"random_extraction_root": f"/random/{index}"},
        )

    monkeypatch.setattr(repair, "_launch_captured_child", launch)
    result = repair.run_repair(outdir, tmp_path)
    assert calls == [0, 1]
    assert result["captured_children"] == [child, child]
    assert result["captured_children_scientifically_byte_identical"] is True
    assert result["comp009_inventory_exactly_unchanged"] is True


def test_two_randomized_launches_reject_scientific_child_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    inventory = {"inventory_sha256": "i" * 64}
    preflight = {
        "repair_preflight_sha256": "p" * 64,
        "comp009_inventory": inventory,
        "frozen_inputs": {},
    }
    monkeypatch.setattr(repair, "verify_repair_preflight", lambda *_args: preflight)
    monkeypatch.setattr(repair, "artifact_inventory", lambda _path: inventory)

    def launch(index, *_args):
        return (
            {"source_manifest_sha256": repair.FROZEN_BINDINGS["source_manifest_sha256"]},
            {"repair_child_sha256": str(index), "scientific": index},
            {"random_extraction_root": f"/random/{index}"},
        )

    monkeypatch.setattr(repair, "_launch_captured_child", launch)
    with pytest.raises(repair.RepairError, match="differ scientifically"):
        repair.run_repair(outdir, tmp_path)
