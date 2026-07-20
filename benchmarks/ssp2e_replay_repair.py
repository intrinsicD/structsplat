"""Source-bound relocation repair for the frozen COMP-009 captured replay.

This harness is intentionally separate from COMP-009.  Its two explicit lifecycle exceptions
are a checked ``ssp2e_assets.verify_assets`` wrapper which replaces two relocation-sensitive path
strings and reuse of the exact sealed non-gating renderer proof through the unchanged captured
renderer verifier.  Codec streams and execution diagnostics are regenerated only ephemerally;
persisted benchmark timing samples and every persisted scientific artifact remain unchanged.
"""

from __future__ import annotations

import argparse
import builtins
import copy
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import io
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMP009 = ROOT / "results/comp009_ssp2e_actual_dev_v1_2026-07-16"
PROTOCOL = "COMP-010-ssp2e-captured-replay-repair-v2"
PREFLIGHT_SCHEMA = "structsplat.comp010.ssp2e-replay-repair-preflight.v2"
CHILD_SCHEMA = "structsplat.comp010.ssp2e-replay-repair-child.v2"
RESULT_SCHEMA = "structsplat.comp010.ssp2e-replay-repair.v2"

REPAIR_SOURCE_PATHS = (
    "tasks/COMP-010-ssp2e-captured-replay-repair.md",
    "benchmarks/ssp2e_replay_repair.py",
    "tests/test_ssp2e_replay_repair.py",
)

FROZEN_BINDINGS = {
    "preflight_binding_sha256": "b4d843ce22356839fa3fa39082044e61cc7ee82cc71176780a192d811d63fadc",
    "source_manifest_sha256": "d75ee551ee650f0d465a01fc8f49b2ee0c4eee15579cf362ee3dfda2b96ad7d6",
    "source_archive_sha256": "fac4ca0978891b3cd16d477ffc04d12dc120168478192293d39af635fca7eb50",
    "input_manifest_sha256": "8f12a64d484f4239a121277ff0467409ed374856ddbf134dfaf49c737fea5f1a",
    "run_manifest_sha256": "cc21fc84aab5cd55ecfffc358f67782013d7a7cf56ffc9b8ef34f8ea98de0a24",
    "benchmark_manifest_sha256": "88a706da64836544d997b177b3b4dda2610e6ede3eceb457e98a4aa7899fe162",
    "analysis_record_sha256": "435f011fe598263cd304b1cbe0754ca82e5fe1e7a49536db4099bdfff9166201",
    "analysis_sha256": "f10c4a1906e4bd240c10253508f38c4053cfe332dec13b220262fee7ae990b30",
    "decision": "ABANDON_FIXED_SSP2E_V1",
}

FROZEN_IMAGES = (
    "nomao-saeki-33553",
    "martyn-seddon-220",
    "zugr-108",
    "jason-briscoe-149782",
    "martin-wessely-211",
    "stefan-kunze-26931",
    "vita-vilcina-3055",
    "philippe-wuyts-45997",
)
FROZEN_TUPLES = ((12, 6, 6, 8), (16, 8, 8, 8))
EXPECTED_IDENTITIES = tuple((image, bits) for image in FROZEN_IMAGES for bits in FROZEN_TUPLES)
STREAM_NAMES = ("ssp2e", "ssp2s", "ssp2l", "ssp2f")
EXPECTED_NATIVE_SHA256 = "5e7ae9813b5d56b51bdba2475268cb9158814f214cb34d6f390f8242f1de336a"
EXPECTED_CDF_SHA256 = "89534aa31405e5129fc57727076dfc32829deb8090362e43869110db2a45c493"
EXPECTED_COST_SHA256 = "b67d48caf7c35a79a78663fba83ce63d1df5d892efaf43c921d056d8fdf65c7e"
EXPECTED_CAPTURED_RUNNER_SHA256 = (
    "aeb1d76f040315454e3b456f2a5f03efdbfaee2b1c89125386203a4638d91987"
)
EXPECTED_RENDERER_RECORD_SHA256 = (
    "ccbf9f0bb5cfcdbc976e6bbbae3e687b5fc102a82fdbc6a3a7c207537e74482a"
)
EXPECTED_RENDERER_EXTENSION_SHA256 = (
    "8ad8a640ad9af29fce5c2d4841fc85495ebe6fd1f5c22ae13950fd454b6c6075"
)
EXPECTED_RENDERER_TOY_SHA256 = (
    "f53ab7de7f05942b410c106dcb7a24cdf02d4c082196e765491773e6f70725e4"
)
EXPECTED_RENDERER_SOURCES = {
    "src/structsplat/cuda_render.py": (
        9747,
        "1a3be3a92520511c703c0d5d9d7f5a60dbf230d5131fc23899c716a835783a62",
    ),
    "src/structsplat/render.py": (
        20144,
        "af26dd0f3d1eb6389c50e4baca8fa6e2ddf754c248c856f87e9c0d49863063f9",
    ),
    "src/structsplat/gaussians.py": (
        15154,
        "26d8ddf03740c8ed44ae9c97d2bb75438dd5a992327fe70cbd3c91f2e5e6fefe",
    ),
    "src/structsplat/cuda/render_ext.cpp": (
        8648,
        "eec0e011189a34d3489c6f719d15528a3511758b79b5cfad145f91323ef5e356",
    ),
    "src/structsplat/cuda/render_ext.cu": (
        27145,
        "5493130d56cdf24dd67d54111eb14a701f5e2b143173cdee27a90a7fa988e150",
    ),
}
WRITE_GUARD_LIMIT = (
    "full before/after inventory detects net changes; direct native, pre-opened-descriptor, or "
    "dir_fd-relative writes perfectly reverted before inventory are outside this Python path guard"
)

_JSON_SPECS = {
    "preflight": (
        "preflight.json",
        "preflight_binding_sha256",
        "structsplat.comp009.preflight.v1",
    ),
    "source_manifest": (
        "source_manifest.json",
        "source_manifest_sha256",
        "structsplat.comp009.source-manifest.v1",
    ),
    "input_manifest": (
        "input_manifest.json",
        "input_manifest_sha256",
        "structsplat.comp009.input-manifest.v1",
    ),
    "run_manifest": (
        "run_manifest.json",
        "run_manifest_sha256",
        "structsplat.comp009.run-manifest.v1",
    ),
    "benchmark_manifest": (
        "benchmark_manifest.json",
        "benchmark_manifest_sha256",
        "structsplat.comp009.benchmark-manifest.v1",
    ),
    "analysis_record": (
        "analysis_record.json",
        "analysis_record_sha256",
        "structsplat.comp009.analysis-record.v1",
    ),
    "analysis": (
        "analysis.json",
        "analysis_sha256",
        "structsplat.comp009.ssp2e-actual-analysis.v1",
    ),
}

_CAPTURED_MODULES = (
    "benchmarks.clic2020_subset_acquire",
    "benchmarks.sgi_entropy_oracle",
    "benchmarks.sgi_entropy_oracle_run",
    "benchmarks.ssp2e_actual_coder",
    "benchmarks.ssp2e_actual_run",
    "benchmarks.ssp2e_arithmetic",
    "benchmarks.ssp2e_assets",
    "benchmarks.ssp2e_container",
    "benchmarks.ssp2e_decode_worker",
    "benchmarks.ssp2e_execute",
    "benchmarks.ssp2e_model",
)


class RepairError(RuntimeError):
    """A COMP-010 binding, isolation, or replay condition failed."""


def canonical_json(value: Any) -> bytes:
    """Encode the finite canonical JSON representation used by COMP-009."""
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RepairError("value is not finite canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RepairError(f"cannot hash required file {path}") from exc
    return digest.hexdigest()


def seal_record(record: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(record)
    result.pop(field, None)
    result[field] = sha256_bytes(canonical_json(result))
    return result


def verify_record(record: Mapping[str, Any], field: str) -> None:
    observed = record.get(field)
    core = dict(record)
    core.pop(field, None)
    if not isinstance(observed, str) or observed != sha256_bytes(canonical_json(core)):
        raise RepairError(f"invalid {field}")


def read_object(path: Path, *, require_canonical: bool = True) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise RepairError(f"{path} is not a JSON object")
    if require_canonical and payload != canonical_json(value):
        raise RepairError(f"{path} is not canonical JSON")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RepairError(f"cannot read JSONL {path}") from exc
    if not payload or not payload.endswith(b"\n"):
        raise RepairError(f"{path} is empty or has an incomplete tail")
    rows = []
    for index, line in enumerate(payload.splitlines(keepends=True), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RepairError(f"invalid JSONL row {index} in {path}") from exc
        if not isinstance(row, dict) or line != canonical_json(row):
            raise RepairError(f"noncanonical JSONL row {index} in {path}")
        rows.append(row)
    return rows


def exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise RepairError(f"refusing to replace {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RepairError("invalid relative artifact path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise RepairError("unsafe relative artifact path")
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RepairError("artifact path escapes its root") from exc
    return path


def _verify_referenced_file(root: Path, relative: Any, size: Any, digest: Any) -> None:
    path = _inside(root, relative)
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or not path.is_file()
        or path.stat().st_size != size
        or sha256_file(path) != digest
    ):
        raise RepairError(f"referenced artifact drift: {relative}")


def _verify_manifest_file_bindings(root: Path, records: Mapping[str, dict[str, Any]]) -> None:
    run_manifest = records["run_manifest"]
    benchmark = records["benchmark_manifest"]
    analysis_record = records["analysis_record"]
    bound_files = (
        ("run_cells.jsonl", run_manifest.get("cells_file_sha256")),
        ("benchmark_cells.jsonl", benchmark.get("benchmark_cells_file_sha256")),
        ("actual_rows.jsonl", benchmark.get("actual_rows_file_sha256")),
    )
    for relative, expected in bound_files:
        path = root / relative
        if not isinstance(expected, str) or sha256_file(path) != expected:
            raise RepairError(f"frozen raw artifact drift: {relative}")

    input_manifest = records["input_manifest"]
    streams = input_manifest.get("streams")
    if not isinstance(streams, list) or len(streams) != 16:
        raise RepairError("input manifest does not contain 16 frozen streams")
    identities = []
    for stream in streams:
        if not isinstance(stream, dict):
            raise RepairError("invalid frozen input stream record")
        identities.append((stream.get("image_id"), tuple(stream.get("bit_tuple", ()))))
        _verify_referenced_file(
            root, stream.get("imported_path"), stream.get("bytes"), stream.get("sha256")
        )
        if stream.get("imported_sha256") != stream.get("sha256"):
            raise RepairError("imported stream identity drift")
    if tuple(identities) != EXPECTED_IDENTITIES:
        raise RepairError("frozen input identity/order drift")

    run_cells = read_jsonl(root / "run_cells.jsonl")
    if len(run_cells) != 16:
        raise RepairError("run journal does not contain exactly 16 cells")
    run_identities = []
    run_seals = []
    for cell in run_cells:
        verify_record(cell, "cell_record_sha256")
        run_seals.append(cell["cell_record_sha256"])
        identity = (cell.get("image_id"), tuple(cell.get("bit_tuple", ())))
        run_identities.append(identity)
        execution = cell.get("execution")
        if not isinstance(execution, dict):
            raise RepairError("run cell lacks its execution record")
        verify_record(execution, "execution_sha256")
        artifacts = cell.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RepairError("run cell has no artifact inventory")
        artifact_paths = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != {"path", "bytes", "sha256"}:
                raise RepairError("run-cell artifact inventory has an unexpected field")
            relative = artifact.get("path")
            if relative in artifact_paths:
                raise RepairError("run-cell artifact inventory contains a duplicate path")
            artifact_paths.add(relative)
            _verify_referenced_file(
                root, relative, artifact.get("bytes"), artifact.get("sha256")
            )
        bits = identity[1]
        if len(bits) != 4:
            raise RepairError("run cell has a malformed bit tuple")
        label = f"m{bits[0]}_s{bits[1]}_r{bits[2]}_c{bits[3]}"
        required_suffixes = {
            f"cells/{identity[0]}/{label}/streams/{name}.bin" for name in STREAM_NAMES
        }
        required_suffixes.update(
            f"cells/{identity[0]}/{label}/models/{variant}/start_{start}.{kind}"
            for variant in ("ssp2e", "ssp2s")
            for start in range(8)
            for kind in ("grid", "head")
        )
        if not required_suffixes.issubset(artifact_paths):
            raise RepairError("run cell lacks a stream/model/grid/head artifact")
        if (
            cell.get("confirmation_payloads_accessed") is not False
            or cell.get("integrity", {}).get("input_sspl1_hash_exact") is not True
            or cell.get("integrity", {}).get("persisted_artifacts_reverified") is not True
        ):
            raise RepairError("run-cell scope/integrity drift")
    if tuple(run_identities) != EXPECTED_IDENTITIES:
        raise RepairError("run-cell identity/order/uniqueness drift")
    if run_seals != run_manifest.get("cell_record_sha256"):
        raise RepairError("run-manifest cell-seal binding drift")

    benchmark_cells = read_jsonl(root / "benchmark_cells.jsonl")
    actual_rows = read_jsonl(root / "actual_rows.jsonl")
    if len(benchmark_cells) != 16 or len(actual_rows) != 16:
        raise RepairError("benchmark journals do not contain exactly 16 cells")
    benchmark_seals = []
    row_seals = []
    benchmark_identities = []
    for cell, row in zip(benchmark_cells, actual_rows, strict=True):
        verify_record(cell, "benchmark_cell_sha256")
        verify_record(row, "record_sha256")
        benchmark_seals.append(cell["benchmark_cell_sha256"])
        row_seals.append(row["record_sha256"])
        identity = (cell.get("image_id"), tuple(cell.get("bit_tuple", ())))
        benchmark_identities.append(identity)
        if (
            cell.get("actual_row") != row
            or row.get("image_id") != identity[0]
            or tuple(row.get("bit_tuple", ())) != identity[1]
            or cell.get("confirmation_payloads_accessed") is not False
        ):
            raise RepairError("benchmark cell/actual-row identity or scope drift")
    if tuple(benchmark_identities) != EXPECTED_IDENTITIES:
        raise RepairError("benchmark-cell identity/order/uniqueness drift")
    if (
        benchmark_seals != benchmark.get("benchmark_cell_sha256")
        or row_seals != benchmark.get("actual_row_sha256")
        or benchmark_seals != analysis_record.get("benchmark_cell_sha256")
        or row_seals != analysis_record.get("actual_row_sha256")
    ):
        raise RepairError("benchmark/analysis-record raw identity binding drift")


def verify_frozen_artifact(comp009_dir: Path) -> dict[str, Any]:
    """Verify all fixed COMP-009 records without importing live COMP-009 code."""
    root = comp009_dir.resolve()
    if not root.is_dir():
        raise RepairError("COMP-009 artifact directory is missing")
    if (root / "replay.json").exists():
        raise RepairError("COMP-009 already has a replay record; repair is not one-shot")

    records: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}
    for name, (filename, seal_field, schema) in _JSON_SPECS.items():
        path = root / filename
        record = read_object(path)
        verify_record(record, seal_field)
        if record.get("schema") != schema or record.get(seal_field) != FROZEN_BINDINGS[seal_field]:
            raise RepairError(f"frozen {name} seal/schema drift")
        records[name] = record
        files[name] = {
            "path": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "record_seal_field": seal_field,
            "record_seal": record[seal_field],
        }

    preflight = records["preflight"]
    input_manifest = records["input_manifest"]
    run_manifest = records["run_manifest"]
    benchmark = records["benchmark_manifest"]
    analysis_record = records["analysis_record"]
    analysis = records["analysis"]
    native = preflight.get("native")
    renderer = preflight.get("renderer")
    if (
        preflight.get("protocol") != "COMP-009-ssp2e-actual-coder-v1"
        or preflight.get("stage") != "preflight"
        or preflight.get("source_manifest_sha256")
        != FROZEN_BINDINGS["source_manifest_sha256"]
        or preflight.get("source_archive_sha256")
        != FROZEN_BINDINGS["source_archive_sha256"]
        or preflight.get("confirmation_payloads_accessed") is not False
        or input_manifest.get("preflight_binding_sha256")
        != FROZEN_BINDINGS["preflight_binding_sha256"]
        or input_manifest.get("confirmation_payloads_accessed") is not False
        or input_manifest.get("target_pixels_opened") is not False
        or run_manifest.get("preflight_binding_sha256")
        != FROZEN_BINDINGS["preflight_binding_sha256"]
        or run_manifest.get("input_manifest_sha256")
        != FROZEN_BINDINGS["input_manifest_sha256"]
        or run_manifest.get("target_pixels_opened") is not False
        or run_manifest.get("confirmation_payloads_accessed") is not False
        or benchmark.get("preflight_binding_sha256")
        != FROZEN_BINDINGS["preflight_binding_sha256"]
        or benchmark.get("run_manifest_sha256") != FROZEN_BINDINGS["run_manifest_sha256"]
        or benchmark.get("confirmation_payloads_accessed") is not False
        or analysis_record.get("benchmark_manifest_sha256")
        != FROZEN_BINDINGS["benchmark_manifest_sha256"]
        or analysis_record.get("analysis_sha256") != FROZEN_BINDINGS["analysis_sha256"]
        or analysis_record.get("decision") != FROZEN_BINDINGS["decision"]
        or analysis_record.get("target_pixels_opened") is not False
        or analysis_record.get("confirmation_payloads_accessed") is not False
        or analysis.get("decision") != FROZEN_BINDINGS["decision"]
        or analysis.get("confirmation_access_authorized_in_this_task") is not False
        or not isinstance(native, dict)
        or native.get("path") != "native/libssp2e_range_v1.so"
        or native.get("bytes") != 31400
        or native.get("sha256") != EXPECTED_NATIVE_SHA256
        or not isinstance(renderer, dict)
        or sha256_bytes(canonical_json(renderer)) != EXPECTED_RENDERER_RECORD_SHA256
    ):
        raise RepairError("frozen COMP-009 cross-binding/scope drift")

    _verify_referenced_file(
        root,
        native["path"],
        native["bytes"],
        native["sha256"],
    )

    archive = root / "source_snapshot.tar"
    if sha256_file(archive) != FROZEN_BINDINGS["source_archive_sha256"]:
        raise RepairError("frozen source archive drift")
    files["source_archive"] = {
        "path": "source_snapshot.tar",
        "bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
    }
    _verify_manifest_file_bindings(root, records)
    return {
        "comp009_seals": dict(FROZEN_BINDINGS),
        "files": files,
        "assets": preflight.get("assets"),
        "environment": preflight.get("environment"),
        "loader": preflight.get("native", {}).get("abi"),
        "scientific_pixels_accessed": False,
        "confirmation_payloads_accessed": False,
    }


def repair_source_manifest() -> dict[str, Any]:
    files = []
    for relative in REPAIR_SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise RepairError(f"repair source is missing: {relative}")
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return seal_record(
        {"schema": "structsplat.comp010.source-manifest.v2", "files": files},
        "source_manifest_sha256",
    )


def _tree_inventory(root: Path, schema: str) -> dict[str, Any]:
    """Seal every file and directory below ``root`` without following links."""
    base = root.resolve()
    if not base.is_dir():
        raise RepairError("cannot inventory a missing tree")
    files = []
    directories = []
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        if path.is_symlink():
            raise RepairError(f"COMP-009 inventory contains a link: {relative}")
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            raise RepairError(f"COMP-009 inventory contains a special file: {relative}")
    return seal_record(
        {
            "schema": schema,
            "directories": directories,
            "files": files,
        },
        "inventory_sha256",
    )


def artifact_inventory(root: Path) -> dict[str, Any]:
    """Seal the complete COMP-009 artifact tree."""
    return _tree_inventory(root, "structsplat.comp010.comp009-inventory.v2")


def extraction_inventory(root: Path) -> dict[str, Any]:
    """Seal one captured-source extraction including the explicit child addition."""
    return _tree_inventory(root, "structsplat.comp010.extraction-inventory.v2")


def runtime_state(frozen_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return and validate the Python/thread/loader state needed by captured verification."""
    environment = frozen_inputs.get("environment")
    loader = frozen_inputs.get("loader")
    if not isinstance(environment, dict) or not isinstance(loader, dict):
        raise RepairError("frozen runtime state is missing")
    frozen_thread = environment.get("thread_environment")
    if not isinstance(frozen_thread, dict):
        raise RepairError("frozen thread environment is missing")
    observed_thread = {name: os.environ.get(name) for name in frozen_thread}
    if observed_thread != frozen_thread:
        raise RepairError("thread/loader environment differs from frozen COMP-009")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    selected = {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "cpu_affinity": affinity,
        "thread_environment": observed_thread,
    }
    expected_selected = {name: environment.get(name) for name in selected}
    if selected != expected_selected:
        raise RepairError("Python/platform state differs from frozen COMP-009")

    module_records = []
    for expected in environment.get("accelerator", {}).get("modules", []):
        if not isinstance(expected, dict) or not isinstance(expected.get("name"), str):
            raise RepairError("frozen module loader record is invalid")
        spec = importlib.util.find_spec(expected["name"])
        if spec is None or spec.origin is None:
            raise RepairError(f"cannot resolve frozen module {expected['name']}")
        origin = Path(spec.origin).resolve()
        observed = {
            "name": expected["name"],
            "origin": str(origin),
            "bytes": origin.stat().st_size,
            "sha256": sha256_file(origin),
        }
        if observed != expected:
            raise RepairError(f"module loader drift for {expected['name']}")
        module_records.append(observed)

    libraries = []
    for expected in loader.get("resolved_libraries", []):
        if not isinstance(expected, dict) or not isinstance(expected.get("path"), str):
            raise RepairError("frozen loader library record is invalid")
        path = Path(expected["path"])
        observed = {
            "path": str(path),
            "bytes": path.stat().st_size if path.is_file() else -1,
            "sha256": sha256_file(path),
        }
        if observed != expected:
            raise RepairError(f"dynamic loader library drift for {path}")
        libraries.append(observed)
    if loader.get("loader_environment") != {
        name: os.environ.get(name) for name in ("LD_LIBRARY_PATH", "LD_PRELOAD")
    }:
        raise RepairError("dynamic loader environment drift")
    return {
        **selected,
        "modules": module_records,
        "resolved_libraries": libraries,
        "frozen_environment_sha256": sha256_bytes(canonical_json(environment)),
        "frozen_loader_sha256": sha256_bytes(canonical_json(loader)),
    }


def preflight(outdir: Path, comp009_dir: Path) -> dict[str, Any]:
    if outdir.exists() and any(outdir.iterdir()):
        raise RepairError("repair preflight requires an empty output directory")
    outdir.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_artifact(comp009_dir)
    sources = repair_source_manifest()
    record = seal_record(
        {
            "schema": PREFLIGHT_SCHEMA,
            "protocol": PROTOCOL,
            "stage": "preflight",
            "comp009_directory": str(comp009_dir.resolve()),
            "frozen_inputs": frozen,
            "comp009_inventory": artifact_inventory(comp009_dir),
            "repair_sources": sources,
            "runtime": runtime_state(frozen),
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
    exclusive_write(outdir / "preflight.json", canonical_json(record))
    return record


def verify_repair_preflight(outdir: Path, comp009_dir: Path) -> dict[str, Any]:
    record = read_object(outdir / "preflight.json")
    verify_record(record, "repair_preflight_sha256")
    expected_keys = {
        "schema",
        "protocol",
        "stage",
        "comp009_directory",
        "frozen_inputs",
        "comp009_inventory",
        "repair_sources",
        "runtime",
        "lifecycle_exception_allowlist",
        "codec_streams_will_be_regenerated_ephemerally",
        "persisted_codec_streams_will_be_created_or_replaced",
        "ephemeral_execution_diagnostic_timings_will_be_generated_and_discarded",
        "persisted_benchmark_resource_timing_samples_will_be_reused_not_generated_or_replaced",
        "scientific_pixels_accessed",
        "confirmation_payloads_accessed",
        "repair_preflight_sha256",
    }
    if (
        set(record) != expected_keys
        or record.get("schema") != PREFLIGHT_SCHEMA
        or record.get("protocol") != PROTOCOL
        or record.get("stage") != "preflight"
        or record.get("comp009_directory") != str(comp009_dir.resolve())
        or record.get("lifecycle_exception_allowlist")
        != [
            "assets.cdf_path",
            "assets.cost_path",
            "sealed_non_gating_renderer_proof_reuse",
        ]
        or record.get("codec_streams_will_be_regenerated_ephemerally") is not True
        or record.get("persisted_codec_streams_will_be_created_or_replaced") is not False
        or record.get("ephemeral_execution_diagnostic_timings_will_be_generated_and_discarded")
        is not True
        or record.get(
            "persisted_benchmark_resource_timing_samples_will_be_reused_not_generated_or_replaced"
        )
        is not True
        or record.get("scientific_pixels_accessed") is not False
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise RepairError("repair preflight scope/schema drift")
    frozen = verify_frozen_artifact(comp009_dir)
    if frozen != record.get("frozen_inputs"):
        raise RepairError("frozen COMP-009 inputs drifted after repair preflight")
    if artifact_inventory(comp009_dir) != record.get("comp009_inventory"):
        raise RepairError("COMP-009 inventory drifted after repair preflight")
    sources = repair_source_manifest()
    if sources != record.get("repair_sources"):
        raise RepairError("repair source drift after preflight")
    if runtime_state(frozen) != record.get("runtime"):
        raise RepairError("repair runtime drift after preflight")
    return record


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise RepairError("unsafe source archive member path")
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise RepairError("unsafe source archive member path")
    return pure.as_posix()


def extract_captured_source(comp009_dir: Path, destination: Path) -> dict[str, Any]:
    """Extract and byte-verify only regular, manifested files from the fixed archive."""
    archive_path = comp009_dir.resolve() / "source_snapshot.tar"
    if sha256_file(archive_path) != FROZEN_BINDINGS["source_archive_sha256"]:
        raise RepairError("source archive hash drift before extraction")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        names = []
        for member in members:
            name = _safe_member_name(member.name)
            if not member.isfile() or member.issym() or member.islnk():
                raise RepairError("source archive contains a link or non-regular member")
            if name in names:
                raise RepairError("source archive contains a duplicate member")
            names.append(name)
        if "SOURCE_MANIFEST.json" not in names:
            raise RepairError("source archive lacks SOURCE_MANIFEST.json")
        manifest_member = members[names.index("SOURCE_MANIFEST.json")]
        handle = archive.extractfile(manifest_member)
        if handle is None:
            raise RepairError("cannot read archived source manifest")
        manifest_payload = handle.read()
        try:
            manifest = json.loads(manifest_payload)
        except json.JSONDecodeError as exc:
            raise RepairError("archived source manifest is invalid JSON") from exc
        if not isinstance(manifest, dict) or manifest_payload != canonical_json(manifest):
            raise RepairError("archived source manifest is not canonical")
        verify_record(manifest, "source_manifest_sha256")
        disk_manifest = read_object(comp009_dir.resolve() / "source_manifest.json")
        if (
            manifest != disk_manifest
            or manifest.get("source_manifest_sha256")
            != FROZEN_BINDINGS["source_manifest_sha256"]
        ):
            raise RepairError("archived source manifest differs from frozen manifest")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise RepairError("archived source manifest has no file list")
        by_name: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise RepairError("invalid archived source entry")
            name = _safe_member_name(entry.get("path"))
            if name == "SOURCE_MANIFEST.json" or name in by_name:
                raise RepairError("duplicate/reserved archived source entry")
            if set(entry) != {"path", "bytes", "sha256"}:
                raise RepairError("archived source entry has an unexpected field")
            by_name[name] = entry
        if set(names) != {"SOURCE_MANIFEST.json", *by_name}:
            raise RepairError("archive has missing or unmanifested files")

        payloads: dict[str, bytes] = {"SOURCE_MANIFEST.json": manifest_payload}
        for member, name in zip(members, names, strict=True):
            if name == "SOURCE_MANIFEST.json":
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RepairError(f"cannot read source archive member {name}")
            payload = handle.read()
            entry = by_name[name]
            if (
                isinstance(entry.get("bytes"), bool)
                or not isinstance(entry.get("bytes"), int)
                or len(payload) != entry["bytes"]
                or sha256_bytes(payload) != entry.get("sha256")
            ):
                raise RepairError(f"source archive member drift: {name}")
            payloads[name] = payload

    for name, payload in payloads.items():
        path = _inside(destination, name)
        exclusive_write(path, payload)
    observed = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if observed != set(payloads):
        raise RepairError("captured extraction produced an unexpected file set")
    return manifest


def attest_renderer_sources(
    captured_manifest: Mapping[str, Any], extracted_root: Path
) -> list[dict[str, Any]]:
    """Attest every frozen renderer source needed to justify proof reuse."""
    entries = captured_manifest.get("files")
    if not isinstance(entries, list):
        raise RepairError("captured manifest has no renderer source inventory")
    by_path = {
        entry.get("path"): entry for entry in entries if isinstance(entry, dict)
    }
    result = []
    for relative, (expected_bytes, expected_hash) in EXPECTED_RENDERER_SOURCES.items():
        expected = {
            "path": relative,
            "bytes": expected_bytes,
            "sha256": expected_hash,
        }
        if by_path.get(relative) != expected:
            raise RepairError(f"captured renderer source-manifest drift: {relative}")
        _verify_referenced_file(extracted_root, relative, expected_bytes, expected_hash)
        result.append(expected)
    return result


def verify_frozen_renderer_record(renderer: Mapping[str, Any]) -> None:
    """Reject any renderer scope, device, source-independent proof, ABI, or toy drift."""
    extension = renderer.get("extension")
    if (
        sha256_bytes(canonical_json(renderer)) != EXPECTED_RENDERER_RECORD_SHA256
        or renderer.get("scope") != "render-only-nongating-parity-and-timing-diagnostic"
        or renderer.get("device_index") != 0
        or renderer.get("toy_output_shape") != [8, 8, 3]
        or renderer.get("toy_output_all_finite") is not True
        or renderer.get("toy_output_sha256") != EXPECTED_RENDERER_TOY_SHA256
        or not isinstance(renderer.get("toy_output_values"), list)
        or len(renderer["toy_output_values"]) != 192
        or not isinstance(extension, dict)
        or extension.get("bytes") != 397376
        or extension.get("sha256") != EXPECTED_RENDERER_EXTENSION_SHA256
        or not isinstance(extension.get("abi"), dict)
    ):
        raise RepairError("sealed non-gating renderer proof drift")


def install_renderer_proof_reuse(
    captured_runner: ModuleType, renderer: Mapping[str, Any]
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Replace only renderer_runtime_proof with isolated sealed-record reuse."""
    verify_frozen_renderer_record(renderer)
    original = captured_runner.renderer_runtime_proof
    state: dict[str, Any] = {
        "calls": 0,
        "renderer_runtime_reexecuted": False,
        "exact_renderer_binary_replay_claimed": False,
        "sealed_renderer_record_sha256": EXPECTED_RENDERER_RECORD_SHA256,
        "captured_verify_renderer_runtime_left_unchanged": True,
        "isolated_copy_returned_each_call": True,
    }

    def reused_renderer_runtime_proof() -> dict[str, Any]:
        state["calls"] += 1
        return copy.deepcopy(dict(renderer))

    captured_runner.renderer_runtime_proof = reused_renderer_runtime_proof
    return original, state


class _RendererImportGuard(importlib.abc.MetaPathFinder):
    """Fail closed if replay tries to import renderer JIT/build machinery."""

    BLOCKED = frozenset({"structsplat.cuda_render", "torch.utils.cpp_extension"})

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname in self.BLOCKED:
            self.attempts.append(fullname)
            raise RepairError(f"renderer JIT/load attempt is forbidden: {fullname}")
        return None


def renderer_extension_residency() -> dict[str, list[str]]:
    """Return fail-closed module/map evidence for the forbidden renderer extension."""
    modules = sorted(name for name in sys.modules if name.startswith("structsplat_render_ext"))
    mapped_paths: set[str] = set()
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RepairError("/proc/self/maps is unavailable for renderer residency proof")
    try:
        lines = maps.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RepairError("cannot read /proc/self/maps for renderer residency proof") from exc
    for line in lines:
        if "structsplat_render_ext" in line:
            fields = line.split()
            mapped_paths.add(fields[-1] if fields else line)
    return {"modules": modules, "mapped_paths": sorted(mapped_paths)}


class _Comp009WriteGuard:
    """Block Python-level writes into the frozen artifact while allowing exact reads."""

    _MUTATORS = (
        "remove",
        "unlink",
        "rename",
        "replace",
        "mkdir",
        "makedirs",
        "rmdir",
        "removedirs",
        "chmod",
        "truncate",
        "utime",
        "link",
        "symlink",
        "mknod",
        "mkfifo",
        "chown",
        "lchown",
    )

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.blocked_operations: list[str] = []
        self._original_builtin_open: Callable[..., Any] | None = None
        self._original_io_open: Callable[..., Any] | None = None
        self._original_os_open: Callable[..., Any] | None = None
        self._original_mutators: dict[str, Callable[..., Any]] = {}

    def _under_root(self, value: Any) -> bool:
        if isinstance(value, int):
            return False
        try:
            path = Path(os.fsdecode(value)).resolve()
            path.relative_to(self.root)
        except (TypeError, ValueError):
            return False
        return True

    def _reject(self, operation: str) -> None:
        self.blocked_operations.append(operation)
        raise RepairError(f"write into frozen COMP-009 is forbidden: {operation}")

    def install(self) -> None:
        if self._original_builtin_open is not None:
            raise RepairError("COMP-009 write guard was installed twice")
        self._original_builtin_open = builtins.open
        self._original_io_open = io.open
        self._original_os_open = os.open

        def guarded_open(
            file: Any, mode: str = "r", *args: Any, **kwargs: Any
        ) -> Any:
            if self._under_root(file) and any(token in mode for token in "wax+"):
                self._reject(f"open:{mode}")
            assert self._original_builtin_open is not None
            return self._original_builtin_open(file, mode, *args, **kwargs)

        def guarded_io_open(
            file: Any, mode: str = "r", *args: Any, **kwargs: Any
        ) -> Any:
            if self._under_root(file) and any(token in mode for token in "wax+"):
                self._reject(f"io.open:{mode}")
            assert self._original_io_open is not None
            return self._original_io_open(file, mode, *args, **kwargs)

        def guarded_os_open(file: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if self._under_root(file) and flags & write_flags:
                self._reject(f"os.open:{flags}")
            assert self._original_os_open is not None
            return self._original_os_open(file, flags, *args, **kwargs)

        try:
            builtins.open = guarded_open
            io.open = guarded_io_open
            os.open = guarded_os_open
            for name in self._MUTATORS:
                original = getattr(os, name)
                self._original_mutators[name] = original

                def guarded_mutator(
                    *args: Any,
                    __name: str = name,
                    __original: Callable[..., Any] = original,
                    **kwargs: Any,
                ) -> Any:
                    if any(self._under_root(value) for value in args[:2]):
                        self._reject(f"os.{__name}")
                    return __original(*args, **kwargs)

                setattr(os, name, guarded_mutator)
        except Exception:
            builtins.open = self._original_builtin_open
            io.open = self._original_io_open
            os.open = self._original_os_open
            for name, original in self._original_mutators.items():
                setattr(os, name, original)
            raise

    def restore(self) -> None:
        if (
            self._original_builtin_open is None
            or self._original_io_open is None
            or self._original_os_open is None
        ):
            raise RepairError("COMP-009 write guard was not installed")
        builtins.open = self._original_builtin_open
        io.open = self._original_io_open
        os.open = self._original_os_open
        for name, original in self._original_mutators.items():
            setattr(os, name, original)


def install_native_build_guard(
    captured_runner: ModuleType,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Guard the build entry point while leaving exact persisted-native loading active."""
    original = captured_runner.arithmetic.build_native
    state: dict[str, Any] = {"attempts": 0, "persisted_native_loading_allowed": True}

    def forbidden_native_build(*_args: Any, **_kwargs: Any) -> None:
        state["attempts"] += 1
        raise RepairError("native arithmetic rebuild is forbidden during repair")

    captured_runner.arithmetic.build_native = forbidden_native_build
    return original, state


def install_asset_path_normalizer(
    captured_runner: ModuleType,
    expected_assets: Mapping[str, Any],
    extracted_root: Path,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Install the one permitted wrapper and return the original function plus audit state."""
    asset_module = captured_runner.assets
    original = asset_module.verify_assets
    asset_dir = (extracted_root / "benchmarks/assets").resolve()
    cdf_target = (asset_dir / "ssp2e_cdf_v1.bin").resolve()
    cost_target = (asset_dir / "ssp2e_cost_q24_v1.bin").resolve()
    expected_keys = {
        "cdf_path",
        "cdf_sha256",
        "cdf_size",
        "cost_path",
        "cost_sha256",
        "cost_size",
        "alphabets",
        "cost_count",
    }
    if set(expected_assets) != expected_keys:
        raise RepairError("frozen asset record has an unexpected identity surface")
    state: dict[str, Any] = {
        "calls": 0,
        "changed_fields": ["cdf_path", "cost_path"],
        "original_function_module": getattr(original, "__module__", None),
        "original_function_name": getattr(original, "__name__", None),
        "extracted_cdf_sha256": sha256_file(cdf_target),
        "extracted_cost_sha256": sha256_file(cost_target),
        "all_non_path_fields_equal": True,
        "original_called_on_extracted_asset_directory": True,
    }

    def normalized_verify_assets(directory: Path = asset_dir) -> dict[str, Any]:
        supplied = Path(directory).resolve()
        if supplied != asset_dir:
            raise RepairError("asset verifier was called on a non-captured directory")
        observed = original(asset_dir)
        if not isinstance(observed, dict) or set(observed) != expected_keys:
            raise RepairError("captured asset verifier returned an unexpected identity surface")
        for field in expected_keys - {"cdf_path", "cost_path"}:
            if observed[field] != expected_assets[field]:
                raise RepairError(f"captured asset identity drift: {field}")
        try:
            observed_cdf = Path(observed["cdf_path"]).resolve()
            observed_cost = Path(observed["cost_path"]).resolve()
        except TypeError as exc:
            raise RepairError("captured asset verifier returned a non-path target") from exc
        if observed_cdf != cdf_target or observed_cost != cost_target:
            raise RepairError("captured asset verifier returned the wrong extracted target")
        state["calls"] += 1
        normalized = dict(observed)
        normalized["cdf_path"] = expected_assets["cdf_path"]
        normalized["cost_path"] = expected_assets["cost_path"]
        if normalized != dict(expected_assets):
            raise RepairError("asset path wrapper changed more than two path strings")
        return normalized

    asset_module.verify_assets = normalized_verify_assets
    return original, state


def assert_module_origins(
    extracted_root: Path, modules: Mapping[str, ModuleType] | None = None
) -> list[dict[str, str]]:
    """Reject captured benchmark or StructSplat imports that escape the extraction."""
    table = sys.modules if modules is None else modules
    root = extracted_root.resolve()
    package_root = (root / "src/structsplat").resolve()
    for required in _CAPTURED_MODULES:
        if required not in table:
            raise RepairError(f"captured worker did not import required module {required}")
    logical_modules = dict(table)
    child_name = "benchmarks.ssp2e_replay_repair"
    if child_name not in logical_modules:
        main_module = logical_modules.get("__main__")
        main_origin = getattr(main_module, "__file__", None)
        if main_origin is not None and Path(main_origin).resolve() == (
            root / "benchmarks/ssp2e_replay_repair.py"
        ).resolve():
            logical_modules[child_name] = main_module
    records = []
    for name, module in sorted(logical_modules.items()):
        if not (name == "benchmarks" or name.startswith("benchmarks.")) and not (
            name == "structsplat" or name.startswith("structsplat.")
        ):
            continue
        origin_text = getattr(module, "__file__", None)
        if origin_text is None:
            raise RepairError(f"captured module {name} has no file origin")
        origin = Path(origin_text).resolve()
        required_root = package_root if name == "structsplat" or name.startswith("structsplat.") else root
        try:
            origin.relative_to(required_root)
        except ValueError as exc:
            raise RepairError(f"live-module leakage for {name}: {origin}") from exc
        records.append(
            {
                "name": name,
                "origin_relative_to_extraction": origin.relative_to(root).as_posix(),
                "sha256": sha256_file(origin),
            }
        )
    return records


def expected_module_origin_records(
    comp009_dir: Path, preflight_record: Mapping[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Reconstruct the exact relative module-origin records for the fixed child call graph."""
    manifest = read_object(comp009_dir.resolve() / "source_manifest.json")
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    repair_entry = next(
        entry
        for entry in preflight_record["repair_sources"]["files"]
        if entry["path"] == REPAIR_SOURCE_PATHS[1]
    )
    module_paths = {
        "benchmarks": "benchmarks/__init__.py",
        "benchmarks.ssp2e_replay_repair": "benchmarks/ssp2e_replay_repair.py",
        **{
            name: f"{name.replace('.', '/')}.py"
            for name in _CAPTURED_MODULES
        },
    }
    before = []
    for name, relative in module_paths.items():
        entry = repair_entry if name == "benchmarks.ssp2e_replay_repair" else by_path.get(relative)
        if entry is None:
            raise RepairError(f"frozen module source entry is missing: {name}")
        before.append(
            {
                "name": name,
                "origin_relative_to_extraction": relative,
                "sha256": entry["sha256"],
            }
        )
    before.sort(key=lambda record: record["name"])
    package = by_path.get("src/structsplat/__init__.py")
    if package is None:
        raise RepairError("frozen StructSplat package source entry is missing")
    after = [*before]
    after.append(
        {
            "name": "structsplat",
            "origin_relative_to_extraction": "src/structsplat/__init__.py",
            "sha256": package["sha256"],
        }
    )
    after.sort(key=lambda record: record["name"])
    return before, after


def _execution_without_timings(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json(record))
    normalized.pop("execution_sha256", None)
    normalized.pop("timings", None)
    return normalized


def expected_worker_cells(comp009_dir: Path) -> list[dict[str, Any]]:
    root = comp009_dir.resolve()
    cells = read_jsonl(root / "run_cells.jsonl")
    if len(cells) != 16:
        raise RepairError("frozen run journal does not contain 16 cells")
    result = []
    identities = []
    for cell in cells:
        image = cell.get("image_id")
        bits = tuple(cell.get("bit_tuple", ()))
        identities.append((image, bits))
        input_record = cell.get("input")
        execution = cell.get("execution")
        if not isinstance(input_record, dict) or not isinstance(execution, dict):
            raise RepairError("frozen run cell lacks input/execution identity")
        input_path = _inside(root, input_record.get("path"))
        if sha256_file(input_path) != input_record.get("sha256"):
            raise RepairError("frozen replay input drift")
        label = f"m{bits[0]}_s{bits[1]}_r{bits[2]}_c{bits[3]}"
        stream_hashes = {}
        for name in STREAM_NAMES:
            path = root / "cells" / str(image) / label / "streams" / f"{name}.bin"
            stream_hashes[name] = sha256_file(path)
        normalized = _execution_without_timings(execution)
        result.append(
            {
                "image_id": image,
                "bit_tuple": list(bits),
                "input_sha256": input_record["sha256"],
                "normalized_execution_sha256": sha256_bytes(canonical_json(normalized)),
                "complete_stream_sha256": stream_hashes,
                "all_non_timing_values_exact": True,
            }
        )
    if tuple(identities) != EXPECTED_IDENTITIES:
        raise RepairError("frozen run-cell identity/order drift")
    return result


def validate_worker_record(worker: Mapping[str, Any], comp009_dir: Path) -> None:
    verify_record(worker, "captured_replay_worker_sha256")
    expected_cells = expected_worker_cells(comp009_dir)
    expected_keys = {
        "schema",
        "protocol",
        "source_manifest_sha256",
        "runner_relative_path",
        "runner_sha256",
        "run_manifest_sha256",
        "analysis_record_sha256",
        "analysis_sha256",
        "decision",
        "cells",
        "cell_count",
        "actual_timing_samples_reused_not_rerun",
        "underlying_field_or_qat_refit",
        "target_pixels_opened",
        "confirmation_payloads_accessed",
        "captured_replay_worker_sha256",
    }
    if (
        set(worker) != expected_keys
        or worker.get("schema") != "structsplat.comp009.captured-replay-worker.v1"
        or worker.get("protocol") != "COMP-009-ssp2e-actual-coder-v1"
        or worker.get("source_manifest_sha256")
        != FROZEN_BINDINGS["source_manifest_sha256"]
        or worker.get("runner_relative_path") != "benchmarks/ssp2e_actual_run.py"
        or worker.get("runner_sha256") != EXPECTED_CAPTURED_RUNNER_SHA256
        or worker.get("run_manifest_sha256") != FROZEN_BINDINGS["run_manifest_sha256"]
        or worker.get("analysis_record_sha256") != FROZEN_BINDINGS["analysis_record_sha256"]
        or worker.get("analysis_sha256") != FROZEN_BINDINGS["analysis_sha256"]
        or worker.get("decision") != FROZEN_BINDINGS["decision"]
        or worker.get("cell_count") != 16
        or worker.get("cells") != expected_cells
        or worker.get("actual_timing_samples_reused_not_rerun") is not True
        or worker.get("underlying_field_or_qat_refit") is not False
        or worker.get("target_pixels_opened") is not False
        or worker.get("confirmation_payloads_accessed") is not False
    ):
        raise RepairError("captured worker identity/scope/decision drift")


def child_replay(comp009_dir: Path, repair_preflight_path: Path) -> dict[str, Any]:
    extracted_text = os.environ.get("COMP010_CAPTURED_ROOT")
    if os.environ.get("COMP010_REPAIR_CHILD") != "1" or not extracted_text:
        raise RepairError("child entry point is restricted to the captured subprocess")
    extracted = Path(extracted_text).resolve()
    if ROOT.resolve() != extracted:
        raise RepairError("repair child did not load from the captured extraction")
    repair_preflight = read_object(repair_preflight_path.resolve())
    verify_record(repair_preflight, "repair_preflight_sha256")
    source_entries = repair_preflight.get("repair_sources", {}).get("files", [])
    own = next(
        (entry for entry in source_entries if entry.get("path") == REPAIR_SOURCE_PATHS[1]), None
    )
    if (
        own is None
        or Path(__file__).stat().st_size != own.get("bytes")
        or sha256_file(Path(__file__)) != own.get("sha256")
    ):
        raise RepairError("copied child entry point differs from repair preflight")
    frozen = verify_frozen_artifact(comp009_dir)
    if frozen != repair_preflight.get("frozen_inputs"):
        raise RepairError("child observed different frozen inputs")

    captured_manifest = read_object(extracted / "SOURCE_MANIFEST.json")
    if (
        captured_manifest.get("source_manifest_sha256")
        != FROZEN_BINDINGS["source_manifest_sha256"]
    ):
        raise RepairError("child source manifest differs from the frozen extraction")
    renderer_sources = attest_renderer_sources(captured_manifest, extracted)
    captured = importlib.import_module("benchmarks.ssp2e_actual_run")
    modules_before = assert_module_origins(extracted)
    frozen_preflight = read_object(comp009_dir.resolve() / "preflight.json")
    expected_assets = frozen_preflight["assets"]
    renderer = frozen_preflight["renderer"]
    native = frozen_preflight["native"]
    if (
        native.get("sha256") != EXPECTED_NATIVE_SHA256
        or sha256_file(comp009_dir.resolve() / native["path"]) != EXPECTED_NATIVE_SHA256
    ):
        raise RepairError("persisted native arithmetic identity drift")
    if "structsplat.cuda_render" in sys.modules or "torch.utils.cpp_extension" in sys.modules:
        raise RepairError("renderer JIT/load module was imported before guards were installed")
    renderer_residency_before = renderer_extension_residency()
    if renderer_residency_before != {"modules": [], "mapped_paths": []}:
        raise RepairError("renderer extension was resident before guards were installed")

    original_asset = captured.assets.verify_assets
    original_renderer = captured.renderer_runtime_proof
    original_native_build = captured.arithmetic.build_native
    original_verify_renderer = captured.verify_renderer_runtime
    import_guard = _RendererImportGuard()
    write_guard = _Comp009WriteGuard(comp009_dir)
    write_guard_installed = False
    normalization: dict[str, Any] | None = None
    renderer_reuse: dict[str, Any] | None = None
    native_guard: dict[str, Any] | None = None
    captured_markers_before = {
        name: os.environ.get(name)
        for name in ("COMP009_CAPTURED_SOURCE", "COMP009_CAPTURED_ROOT")
    }
    try:
        _, normalization = install_asset_path_normalizer(captured, expected_assets, extracted)
        _, renderer_reuse = install_renderer_proof_reuse(captured, renderer)
        _, native_guard = install_native_build_guard(captured)
        sys.meta_path.insert(0, import_guard)
        write_guard.install()
        write_guard_installed = True
        os.environ["COMP009_CAPTURED_SOURCE"] = "1"
        os.environ["COMP009_CAPTURED_ROOT"] = str(extracted)
        worker = captured.captured_replay_worker(comp009_dir.resolve())
    finally:
        captured.assets.verify_assets = original_asset
        captured.renderer_runtime_proof = original_renderer
        captured.arithmetic.build_native = original_native_build
        if import_guard in sys.meta_path:
            sys.meta_path.remove(import_guard)
        if write_guard_installed:
            write_guard.restore()
        for name, value in captured_markers_before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if normalization is None or renderer_reuse is None or native_guard is None:
        raise RepairError("captured lifecycle exceptions were not fully installed")
    if (
        captured.assets.verify_assets is not original_asset
        or captured.renderer_runtime_proof is not original_renderer
        or captured.arithmetic.build_native is not original_native_build
        or captured.verify_renderer_runtime is not original_verify_renderer
        or builtins.open is not write_guard._original_builtin_open
        or io.open is not write_guard._original_io_open
        or os.open is not write_guard._original_os_open
        or any(
            getattr(os, name) is not original
            for name, original in write_guard._original_mutators.items()
        )
    ):
        raise RepairError("captured wrapper/guard restoration failed")
    renderer_residency_after = renderer_extension_residency()
    if renderer_residency_after != {"modules": [], "mapped_paths": []}:
        raise RepairError("renderer extension was loaded during captured replay")
    if normalization["calls"] != 5:
        raise RepairError("captured replay did not exercise asset normalization exactly five times")
    if renderer_reuse["calls"] != 5:
        raise RepairError("captured replay did not reuse the renderer proof exactly five times")
    if import_guard.attempts or native_guard["attempts"]:
        raise RepairError("renderer JIT/load or native rebuild was attempted")
    modules_after = assert_module_origins(extracted)
    validate_worker_record(worker, comp009_dir)
    return seal_record(
        {
            "schema": CHILD_SCHEMA,
            "protocol": PROTOCOL,
            "repair_preflight_sha256": repair_preflight["repair_preflight_sha256"],
            "source_manifest_sha256": FROZEN_BINDINGS["source_manifest_sha256"],
            "source_archive_sha256": FROZEN_BINDINGS["source_archive_sha256"],
            "child_entry_relative_path": "benchmarks/ssp2e_replay_repair.py",
            "child_entry_sha256": own["sha256"],
            "module_origins_before_worker": modules_before,
            "module_origins_after_worker": modules_after,
            "asset_path_normalization": normalization,
            "renderer_source_attestation": renderer_sources,
            "renderer_proof_reuse": renderer_reuse,
            "renderer_jit_load_guard": {
                "blocked_modules": sorted(import_guard.BLOCKED),
                "attempts": import_guard.attempts,
                "residency_before": renderer_residency_before,
                "residency_after": renderer_residency_after,
            },
            "native_rebuild_guard": native_guard,
            "comp009_python_write_guard": {
                "blocked_operations": write_guard.blocked_operations,
                "python_open_and_os_mutators_guarded": True,
                "guard_restored": True,
                "transient_write_guard_limit": WRITE_GUARD_LIMIT,
            },
            "persisted_native_sha256": EXPECTED_NATIVE_SHA256,
            "persisted_native_loaded_and_frozen_proof_reverified": True,
            "captured_worker": worker,
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
            "decision": FROZEN_BINDINGS["decision"],
        },
        "repair_child_sha256",
    )


def _child_environment(extracted: Path, frozen: Mapping[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(extracted), str(extracted / "src"))),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "COMP010_REPAIR_CHILD": "1",
            "COMP010_CAPTURED_ROOT": str(extracted),
        }
    )
    thread = frozen.get("environment", {}).get("thread_environment")
    if not isinstance(thread, dict):
        raise RepairError("frozen thread environment is missing")
    for name, value in thread.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def validate_child_record(
    child: Mapping[str, Any], comp009_dir: Path, preflight_record: Mapping[str, Any]
) -> None:
    verify_record(child, "repair_child_sha256")
    normalization = child.get("asset_path_normalization")
    renderer_reuse = child.get("renderer_proof_reuse")
    renderer_guard = child.get("renderer_jit_load_guard")
    native_guard = child.get("native_rebuild_guard")
    write_guard = child.get("comp009_python_write_guard")
    worker = child.get("captured_worker")
    own = next(
        entry
        for entry in preflight_record["repair_sources"]["files"]
        if entry["path"] == REPAIR_SOURCE_PATHS[1]
    )
    expected_before, expected_after = expected_module_origin_records(comp009_dir, preflight_record)
    expected_normalization = {
        "calls": 5,
        "changed_fields": ["cdf_path", "cost_path"],
        "original_function_module": "benchmarks.ssp2e_assets",
        "original_function_name": "verify_assets",
        "extracted_cdf_sha256": EXPECTED_CDF_SHA256,
        "extracted_cost_sha256": EXPECTED_COST_SHA256,
        "all_non_path_fields_equal": True,
        "original_called_on_extracted_asset_directory": True,
    }
    expected_renderer_reuse = {
        "calls": 5,
        "renderer_runtime_reexecuted": False,
        "exact_renderer_binary_replay_claimed": False,
        "sealed_renderer_record_sha256": EXPECTED_RENDERER_RECORD_SHA256,
        "captured_verify_renderer_runtime_left_unchanged": True,
        "isolated_copy_returned_each_call": True,
    }
    expected_renderer_guard = {
        "blocked_modules": sorted(_RendererImportGuard.BLOCKED),
        "attempts": [],
        "residency_before": {"modules": [], "mapped_paths": []},
        "residency_after": {"modules": [], "mapped_paths": []},
    }
    expected_native_guard = {"attempts": 0, "persisted_native_loading_allowed": True}
    expected_write_guard = {
        "blocked_operations": [],
        "python_open_and_os_mutators_guarded": True,
        "guard_restored": True,
        "transient_write_guard_limit": WRITE_GUARD_LIMIT,
    }
    expected_child_keys = {
        "schema",
        "protocol",
        "repair_preflight_sha256",
        "source_manifest_sha256",
        "source_archive_sha256",
        "child_entry_relative_path",
        "child_entry_sha256",
        "module_origins_before_worker",
        "module_origins_after_worker",
        "asset_path_normalization",
        "renderer_source_attestation",
        "renderer_proof_reuse",
        "renderer_jit_load_guard",
        "native_rebuild_guard",
        "comp009_python_write_guard",
        "persisted_native_sha256",
        "persisted_native_loaded_and_frozen_proof_reverified",
        "captured_worker",
        "exactly_two_lifecycle_exceptions_exercised",
        "all_wrappers_and_guards_restored",
        "ephemeral_execution_diagnostic_timings_generated_and_discarded",
        "persisted_benchmark_resource_timing_samples_reused_not_rerun",
        "codec_streams_regenerated_ephemerally",
        "persisted_codec_streams_created_or_replaced",
        "underlying_field_or_qat_refit",
        "renderer_runtime_reexecuted",
        "exact_renderer_binary_replay_claimed",
        "scientific_pixels_accessed",
        "confirmation_payloads_accessed",
        "decision",
        "repair_child_sha256",
    }
    if (
        set(child) != expected_child_keys
        or child.get("schema") != CHILD_SCHEMA
        or child.get("protocol") != PROTOCOL
        or child.get("repair_preflight_sha256")
        != preflight_record.get("repair_preflight_sha256")
        or child.get("source_manifest_sha256") != FROZEN_BINDINGS["source_manifest_sha256"]
        or child.get("source_archive_sha256") != FROZEN_BINDINGS["source_archive_sha256"]
        or child.get("child_entry_relative_path") != "benchmarks/ssp2e_replay_repair.py"
        or child.get("child_entry_sha256") != own["sha256"]
        or normalization != expected_normalization
        or renderer_reuse != expected_renderer_reuse
        or child.get("renderer_source_attestation")
        != [
            {"path": path, "bytes": size, "sha256": digest}
            for path, (size, digest) in EXPECTED_RENDERER_SOURCES.items()
        ]
        or renderer_guard != expected_renderer_guard
        or native_guard != expected_native_guard
        or write_guard != expected_write_guard
        or child.get("module_origins_before_worker") != expected_before
        or child.get("module_origins_after_worker") != expected_after
        or child.get("persisted_native_sha256") != EXPECTED_NATIVE_SHA256
        or child.get("persisted_native_loaded_and_frozen_proof_reverified") is not True
        or child.get("exactly_two_lifecycle_exceptions_exercised") is not True
        or child.get("all_wrappers_and_guards_restored") is not True
        or child.get("ephemeral_execution_diagnostic_timings_generated_and_discarded")
        is not True
        or child.get("persisted_benchmark_resource_timing_samples_reused_not_rerun")
        is not True
        or child.get("codec_streams_regenerated_ephemerally") is not True
        or child.get("persisted_codec_streams_created_or_replaced") is not False
        or child.get("underlying_field_or_qat_refit") is not False
        or child.get("renderer_runtime_reexecuted") is not False
        or child.get("exact_renderer_binary_replay_claimed") is not False
        or child.get("scientific_pixels_accessed") is not False
        or child.get("confirmation_payloads_accessed") is not False
        or child.get("decision") != FROZEN_BINDINGS["decision"]
        or not isinstance(worker, dict)
    ):
        raise RepairError("repair child binding/scope drift")
    validate_worker_record(worker, comp009_dir)


def _launch_captured_child(
    launch_index: int,
    outdir: Path,
    comp009_dir: Path,
    preflight_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one captured child from a newly randomized extraction root."""
    with tempfile.TemporaryDirectory(prefix="comp010-captured-repair-") as temporary:
        extracted = Path(temporary) / "source"
        manifest = extract_captured_source(comp009_dir, extracted)
        source_entry = next(
            entry
            for entry in preflight_record["repair_sources"]["files"]
            if entry["path"] == REPAIR_SOURCE_PATHS[1]
        )
        source_path = ROOT / REPAIR_SOURCE_PATHS[1]
        child_path = extracted / "benchmarks/ssp2e_replay_repair.py"
        exclusive_write(child_path, source_path.read_bytes())
        if (
            child_path.stat().st_size != source_entry["bytes"]
            or sha256_file(child_path) != source_entry["sha256"]
        ):
            raise RepairError("copied child entry point drift")
        child_addition = {
            "path": "benchmarks/ssp2e_replay_repair.py",
            "bytes": child_path.stat().st_size,
            "sha256": sha256_file(child_path),
            "present_in_frozen_source_manifest": False,
            "purpose": "source-bound-child-entry-point-only",
        }
        extraction_before_child = extraction_inventory(extracted)
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2e_replay_repair",
            "child",
            "--comp009-dir",
            str(comp009_dir.resolve()),
            "--repair-preflight",
            str((outdir / "preflight.json").resolve()),
        ]
        completed = subprocess.run(
            command,
            cwd=extracted,
            env=_child_environment(extracted, preflight_record["frozen_inputs"]),
            check=False,
            capture_output=True,
            timeout=3600,
        )
        extraction_after_child = extraction_inventory(extracted)
        if extraction_after_child != extraction_before_child:
            raise RepairError("captured child changed its exact source-plus-entry extraction")
        if completed.returncode != 0 or completed.stderr:
            raise RepairError(
                "captured repair child failed: "
                f"returncode={completed.returncode}, "
                f"stderr={completed.stderr.decode(errors='replace')}"
            )
        try:
            child = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RepairError("captured repair child emitted invalid JSON") from exc
        if not isinstance(child, dict) or completed.stdout != canonical_json(child):
            raise RepairError("captured repair child output is not canonical JSON")
        validate_child_record(child, comp009_dir, preflight_record)
        envelope = {
            "launch_index": launch_index,
            "random_extraction_root": str(extracted),
            "random_child_entry_path": str(child_path),
            "child_entry_addition": child_addition,
            "child_record_sha256": child["repair_child_sha256"],
            "extraction_inventory_before_child_sha256": extraction_before_child[
                "inventory_sha256"
            ],
            "extraction_inventory_after_child_sha256": extraction_after_child[
                "inventory_sha256"
            ],
            "extraction_inventory_exactly_unchanged": True,
            "random_paths_are_lifecycle_only": True,
        }
        return manifest, child, envelope


def run_repair(outdir: Path, comp009_dir: Path) -> dict[str, Any]:
    preflight_record = verify_repair_preflight(outdir, comp009_dir)
    if (outdir / "repair.json").exists():
        raise RepairError("repair is one-shot and already has a result")
    inventory_before = artifact_inventory(comp009_dir)
    if inventory_before != preflight_record.get("comp009_inventory"):
        raise RepairError("COMP-009 inventory differs immediately before repair")
    launches = []
    children = []
    manifest: dict[str, Any] | None = None
    for launch_index in range(2):
        observed_manifest, child, envelope = _launch_captured_child(
            launch_index, outdir, comp009_dir, preflight_record
        )
        if manifest is None:
            manifest = observed_manifest
        elif observed_manifest != manifest:
            raise RepairError("randomized extractions produced different source manifests")
        inventory_after_launch = artifact_inventory(comp009_dir)
        if inventory_after_launch != inventory_before:
            raise RepairError("COMP-009 inventory changed during captured replay")
        envelope["comp009_inventory_before_sha256"] = inventory_before["inventory_sha256"]
        envelope["comp009_inventory_after_sha256"] = inventory_after_launch["inventory_sha256"]
        envelope["comp009_inventory_exactly_unchanged"] = True
        launches.append(envelope)
        children.append(child)
    if manifest is None or len(children) != 2:
        raise RepairError("repair requires exactly two randomized captured launches")
    if launches[0]["random_extraction_root"] == launches[1]["random_extraction_root"]:
        raise RepairError("captured launches did not use independent randomized roots")
    if canonical_json(children[0]) != canonical_json(children[1]):
        raise RepairError("randomized captured children differ scientifically")
    inventory_after = artifact_inventory(comp009_dir)
    if inventory_after != inventory_before:
        raise RepairError("COMP-009 inventory changed across repair")

    result = seal_record(
        {
            "schema": RESULT_SCHEMA,
            "protocol": PROTOCOL,
            "stage": "repair",
            "repair_preflight_sha256": preflight_record["repair_preflight_sha256"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "source_archive_sha256": FROZEN_BINDINGS["source_archive_sha256"],
            "randomized_launch_envelopes": launches,
            "captured_children": children,
            "captured_children_scientifically_byte_identical": True,
            "randomized_extraction_path_fields_excluded_from_scientific_equality": [
                "random_extraction_root",
                "random_child_entry_path",
            ],
            "comp009_inventory_before": inventory_before,
            "comp009_inventory_after": inventory_after,
            "comp009_inventory_exactly_unchanged": True,
            "decision": FROZEN_BINDINGS["decision"],
            "lifecycle_repair_passed": True,
            "compression_claim_strengthened": False,
            "codec_streams_regenerated_ephemerally": True,
            "persisted_codec_streams_created_or_replaced": False,
            "ephemeral_execution_diagnostic_timings_generated_and_discarded": True,
            "persisted_benchmark_resource_timing_samples_reused_not_generated_or_replaced": True,
            "underlying_field_or_qat_refit": False,
            "renderer_runtime_reexecuted": False,
            "exact_renderer_binary_replay_claimed": False,
            "scientific_pixels_accessed": False,
            "confirmation_payloads_accessed": False,
        },
        "repair_sha256",
    )
    exclusive_write(outdir / "repair.json", canonical_json(result))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--outdir", type=Path, required=True)
    preflight_parser.add_argument("--comp009-dir", type=Path, default=DEFAULT_COMP009)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--outdir", type=Path, required=True)
    run_parser.add_argument("--comp009-dir", type=Path, default=DEFAULT_COMP009)
    child_parser = subparsers.add_parser("child")
    child_parser.add_argument("--comp009-dir", type=Path, required=True)
    child_parser.add_argument("--repair-preflight", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.outdir.resolve(), args.comp009_dir.resolve())
        print(result["repair_preflight_sha256"])
    elif args.command == "run":
        result = run_repair(args.outdir.resolve(), args.comp009_dir.resolve())
        print(result["repair_sha256"])
    else:
        result = child_replay(args.comp009_dir.resolve(), args.repair_preflight.resolve())
        sys.stdout.buffer.write(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
