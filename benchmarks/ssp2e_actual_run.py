"""Lifecycle runner for the source-bound COMP-009 SSP2E actual-coder test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import sgi_entropy_oracle as oracle
from benchmarks import sgi_entropy_oracle_run as oracle_run
from benchmarks import ssp2e_actual_coder as actual
from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as assets
from benchmarks import ssp2e_container as container
from benchmarks import ssp2e_decode_worker as decode_worker
from benchmarks import ssp2e_execute as execute
from benchmarks import ssp2e_model as model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/comp008_sgi_entropy_oracle_dev_v3_2026-07-16"
EXPECTED_INPUT_BINDING = "9c5dca682d490567a56ce4fc4043112a5bacf72b2fee908e95e86e691a1ea0f2"
EXPECTED_INPUT_ANALYSIS = "3368a3f96926b0eaf2e18119ae587d4bb3822c450c0f6f5a33e883c4e87e6792"
EXPECTED_INPUT_REPLAY = "95829d0c563842adcc0a7788d7ff02a8e6bd74596f80a3a3966d78c2da9b1397"
EXPECTED_INPUT_CELLS_FILE = "9f14774f091afb0b8c2e9c471f9e9bd21f2d8800f44da6e413a076acf33ccb8b"
EXPECTED_INPUT_ANALYSIS_RECORD_FILE = (
    "c465bff62384d1219098076b11a9f571fb84352030b1b29307c1525d1352464a"
)
EXPECTED_INPUT_ANALYSIS_RECORD = "1081a77e7d5d58c04ef8695e0b45d0508b6e0835e5db597094fc1d61597f0c56"

PREFLIGHT_SCHEMA = "structsplat.comp009.preflight.v1"
INPUT_SCHEMA = "structsplat.comp009.input-manifest.v1"
SOURCE_SCHEMA = "structsplat.comp009.source-manifest.v1"
RUN_SCHEMA = "structsplat.comp009.run-manifest.v1"
BENCHMARK_SCHEMA = "structsplat.comp009.benchmark-manifest.v1"
REPLAY_SCHEMA = "structsplat.comp009.replay.v1"
REQUIRED_RENDERER_PRELOAD = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
FROZEN_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_EXPLICIT_SOURCES = (
    "pyproject.toml",
    "benchmarks/__init__.py",
    "tasks/COMP-008-mean-conditioned-entropy-oracle.md",
    "tasks/COMP-009-ssp2e-actual-coder.md",
    "benchmarks/sgi_entropy_oracle.py",
    "benchmarks/sgi_entropy_oracle_run.py",
    "benchmarks/clic2020_subset_acquire.py",
    "benchmarks/ssp2e_assets.py",
    "benchmarks/assets/ssp2e_cdf_v1.bin",
    "benchmarks/assets/ssp2e_cost_q24_v1.bin",
    "benchmarks/ssp2e_arithmetic.py",
    "benchmarks/ssp2e_range_ext.cpp",
    "benchmarks/ssp2e_model.py",
    "benchmarks/ssp2e_container.py",
    "benchmarks/ssp2e_execute.py",
    "benchmarks/ssp2e_decode_worker.py",
    "benchmarks/ssp2e_actual_coder.py",
    "benchmarks/ssp2e_actual_run.py",
    "tests/test_sgi_entropy_oracle.py",
    "tests/test_sgi_entropy_oracle_run.py",
    "tests/test_clic2020_subset_acquire.py",
    "tests/test_ssp2e_arithmetic.py",
    "tests/test_ssp2e_model.py",
    "tests/test_ssp2e_container.py",
    "tests/test_ssp2e_execute.py",
    "tests/test_ssp2e_decode_worker.py",
    "tests/test_ssp2e_actual_coder.py",
    "tests/test_ssp2e_actual_run.py",
)

_FOCUSED_TESTS = (
    "tests/test_ssp2e_arithmetic.py",
    "tests/test_ssp2e_model.py",
    "tests/test_ssp2e_container.py",
    "tests/test_ssp2e_execute.py",
    "tests/test_ssp2e_decode_worker.py",
    "tests/test_ssp2e_actual_coder.py",
    "tests/test_ssp2e_actual_run.py",
    "tests/test_sgi_entropy_oracle.py",
)


class LifecycleError(RuntimeError):
    """A COMP-009 lifecycle or provenance condition failed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return actual.canonical_json(value)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise LifecycleError(f"cannot read JSONL {path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line:
            raise LifecycleError(f"blank JSONL line {index} in {path}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifecycleError(f"invalid JSONL line {index} in {path}") from exc
        if not isinstance(value, dict):
            raise LifecycleError(f"JSONL line {index} in {path} is not an object")
        rows.append(value)
    return rows


def _exclusive_write(path: Path, payload: bytes, *, readonly: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444 if readonly else 0o644
        )
    except FileExistsError as exc:
        raise LifecycleError(f"refusing to replace existing artifact {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def source_paths() -> tuple[Path, ...]:
    """Return the frozen task/harness closure plus all production StructSplat source files."""
    relative = set(_EXPLICIT_SOURCES)
    for path in (ROOT / "src/structsplat").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            relative.add(path.relative_to(ROOT).as_posix())
    paths = tuple(ROOT / name for name in sorted(relative))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise LifecycleError(f"source closure is incomplete: {[str(path) for path in missing]}")
    return paths


def source_manifest(paths: Iterable[Path] | None = None) -> dict[str, Any]:
    records = []
    for path in paths or source_paths():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise LifecycleError(f"source lies outside repository: {resolved}") from exc
        records.append(
            {
                "path": relative,
                "bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        )
    manifest = {
        "schema": SOURCE_SCHEMA,
        "files": sorted(records, key=lambda record: record["path"]),
    }
    return actual.seal_record(manifest, "source_manifest_sha256")


def _write_source_archive(path: Path, manifest: Mapping[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="comp009-source-") as temporary:
        staging = Path(temporary) / "source_snapshot.tar"
        with tarfile.open(staging, "w", format=tarfile.PAX_FORMAT) as archive:
            manifest_bytes = _canonical_json(manifest)
            info = tarfile.TarInfo("SOURCE_MANIFEST.json")
            info.size = len(manifest_bytes)
            info.mtime = 0
            info.mode = 0o444
            with tempfile.SpooledTemporaryFile() as handle:
                handle.write(manifest_bytes)
                handle.seek(0)
                archive.addfile(info, handle)
            for record in manifest["files"]:
                source = ROOT / record["path"]
                info = archive.gettarinfo(str(source), arcname=record["path"])
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
        _exclusive_write(path, staging.read_bytes())


def environment_record() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "cpu_affinity": affinity,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_CACHE_PATH",
            )
        },
        "cpu": cpu_identity_record(),
        "accelerator": accelerator_record(),
    }


def _module_record(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise LifecycleError(f"cannot resolve required module {name!r}")
    origin = Path(spec.origin).resolve()
    return {
        "name": name,
        "origin": str(origin),
        "bytes": origin.stat().st_size,
        "sha256": _sha256_file(origin),
    }


def cpu_identity_record() -> dict[str, Any]:
    """Return stable CPU identity fields without volatile frequency telemetry."""
    fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        first = cpuinfo.read_text(encoding="utf-8").split("\n\n", 1)[0]
        allowed = {
            "vendor_id",
            "cpu family",
            "model",
            "model name",
            "stepping",
            "microcode",
            "cache size",
            "physical id",
            "siblings",
            "cpu cores",
            "flags",
            "bugs",
        }
        for line in first.splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in allowed:
                fields[key] = value
    return {
        "uname": list(platform.uname()),
        "logical_cpu_count": os.cpu_count(),
        "first_processor_stable_fields": fields,
    }


def accelerator_record() -> dict[str, Any]:
    """Bind Python/CUDA/GPU state used by the mandatory render-only diagnostic."""
    import torch

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "major": properties.major,
                "minor": properties.minor,
                "total_memory": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
                "uuid": str(properties.uuid),
                "pci_bus_id": properties.pci_bus_id,
                "pci_device_id": properties.pci_device_id,
                "pci_domain_id": properties.pci_domain_id,
                "l2_cache_size": properties.L2_cache_size,
            }
        )
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "devices": devices,
        "modules": [_module_record("numpy"), _module_record("torch")],
    }


def renderer_runtime_proof() -> dict[str, Any]:
    """Build/load and bind the exact CUDA renderer used only for the parity diagnostic."""
    import torch

    from structsplat import cuda_render
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    if not torch.cuda.is_available():
        raise LifecycleError("the frozen render diagnostic requires one available CUDA device")
    device = torch.device("cuda:0")
    field = GaussianField(
        torch.tensor([[3.5, 3.5]], device=device, dtype=torch.float32),
        torch.zeros((1, 2), device=device, dtype=torch.float32),
        torch.zeros(1, device=device, dtype=torch.float32),
        torch.tensor([[0.25, 0.5, 0.75]], device=device, dtype=torch.float32),
    )
    torch.cuda.synchronize(device)
    image = render_field(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        8,
        8,
        64,
        "cuda",
        scales=field.scales(),
        rotations=field.rotations,
        sigma_cutoff=3.0,
    )
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(image).all()):
        raise LifecycleError("CUDA renderer preflight produced a non-finite tensor")
    extension = getattr(cuda_render, "_EXT", None)
    extension_path_raw = getattr(extension, "__file__", None)
    if not isinstance(extension_path_raw, str):
        raise LifecycleError("CUDA renderer extension has no bindable binary path")
    extension_path = Path(extension_path_raw).resolve()
    image_array = image.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    image_bytes = image_array.tobytes()
    return {
        "scope": "render-only-nongating-parity-and-timing-diagnostic",
        "device_index": 0,
        "toy_output_sha256": _sha256_bytes(image_bytes),
        "toy_output_shape": list(image.shape),
        "toy_output_values": image_array.reshape(-1).tolist(),
        "toy_output_all_finite": True,
        "extension": {
            "path": str(extension_path),
            "bytes": extension_path.stat().st_size,
            "sha256": _sha256_file(extension_path),
            "abi": dynamic_library_record(extension_path),
        },
    }


def verify_renderer_runtime(expected: Mapping[str, Any]) -> None:
    """Verify exact binary provenance and numerically stable toy-render behavior."""
    current = renderer_runtime_proof()
    for name in ("scope", "device_index", "toy_output_shape", "extension"):
        if current.get(name) != expected.get(name):
            raise LifecycleError(f"CUDA renderer runtime drift in {name}")
    reference = np.asarray(expected.get("toy_output_values"), dtype=np.float32)
    observed = np.asarray(current.get("toy_output_values"), dtype=np.float32)
    if (
        expected.get("toy_output_all_finite") is not True
        or current.get("toy_output_all_finite") is not True
        or reference.shape != observed.shape
        or not np.allclose(reference, observed, rtol=1e-6, atol=1e-7)
    ):
        raise LifecycleError("CUDA toy-render numerical proof drift")


def dynamic_library_record(library_path: Path) -> dict[str, Any]:
    """Bind the loader resolution and ELF ABI surfaces for the exact persisted `.so`."""
    commands = {
        "ldd": ["ldd", str(library_path)],
        "dynamic": ["readelf", "--dynamic", str(library_path)],
        "version_info": ["readelf", "--version-info", str(library_path)],
    }
    outputs: dict[str, Any] = {}
    resolved_paths: set[Path] = set()
    for name, command in commands.items():
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        output = completed.stdout + completed.stderr
        if name == "ldd":
            output = re.sub(r"\(0x[0-9a-fA-F]+\)", "(ASLR)", output)
        outputs[name] = {
            "command": command,
            "output": output,
            "output_sha256": _sha256_bytes(output.encode("utf-8")),
        }
        if name == "ldd":
            for token in output.replace("=>", " ").split():
                candidate = Path(token)
                if candidate.is_absolute() and candidate.is_file():
                    resolved_paths.add(candidate.resolve())
    outputs["resolved_libraries"] = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(resolved_paths)
    ]
    outputs["loader_environment"] = {
        name: os.environ.get(name) for name in ("LD_LIBRARY_PATH", "LD_PRELOAD")
    }
    return outputs


def compiler_record() -> dict[str, Any]:
    """Bind the exact compiler executable and flags used for the persisted native core."""
    requested = os.environ.get("CXX") or "c++"
    resolved_raw = shutil.which(requested)
    if resolved_raw is None:
        raise LifecycleError(f"cannot resolve C++ compiler {requested!r}")
    resolved = Path(resolved_raw).resolve()
    completed = subprocess.run(
        [str(resolved), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "requested": requested,
        "resolved": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "version_output": completed.stdout + completed.stderr,
        "build_flags": ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared"],
        "libc": list(platform.libc_ver()),
    }


def repository_record() -> dict[str, Any]:
    """Capture dirty-worktree provenance while leaving the source snapshot authoritative."""

    def git(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True, timeout=60
        )
        return completed.stdout

    head = git("rev-parse", "HEAD").decode("ascii").strip()
    branch = git("branch", "--show-current").decode("utf-8").strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    tracked_diff = git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    return {
        "head": head,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain_z_sha256": _sha256_bytes(status),
        "status_entries": [entry for entry in status.decode("utf-8").split("\0") if entry],
        "tracked_diff_bytes": len(tracked_diff),
        "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        "source_snapshot_is_execution_authority": True,
    }


def static_constant_record() -> dict[str, Any]:
    permutation = np.ascontiguousarray(model.shuffle_permutation(), dtype="<u2")
    bootstrap = oracle.bootstrap_index_matrix()
    return {
        "cdf_asset_sha256": assets.CDF_ASSET_SHA256,
        "cost_asset_sha256": assets.COST_ASSET_SHA256,
        "shuffle_permutation_shape": list(permutation.shape),
        "shuffle_permutation_sha256": _sha256_bytes(permutation.tobytes(order="C")),
        "expected_shuffle_permutation_sha256": model.SHUFFLE_PERMUTATION_SHA256,
        "bootstrap_shape": list(bootstrap.shape),
        "bootstrap_sha256": _sha256_bytes(bootstrap.tobytes(order="C")),
        "expected_bootstrap_sha256": oracle.BOOTSTRAP_MATRIX_SHA256,
    }


def persisted_native_proof(library_path: Path) -> dict[str, Any]:
    """Exercise the exact persisted native library against the Python proof implementation."""
    native = arithmetic.NativeArithmeticCoder(library_path)
    tables = assets.load_tables()
    cases = []
    for alphabet, location, scale, count in ((64, 17, 5, 127), (256, 43, 11, 193)):
        row = np.asarray(tables[alphabet].row(location, scale), dtype=np.uint32)
        cdfs = np.repeat(row[None, :], count, axis=0)
        symbols = np.asarray(
            [(index * 37 + location + scale) % alphabet for index in range(count)],
            dtype=np.uint32,
        )
        python_payload = arithmetic.encode(symbols, cdfs)
        native_payload = native.encode(symbols, cdfs)
        if native_payload != python_payload:
            raise LifecycleError("persisted native encoder differs from Python proof coder")
        if tuple(native.decode(native_payload, cdfs, count)) != tuple(int(v) for v in symbols):
            raise LifecycleError("persisted native decoder differs from source symbols")
        if tuple(arithmetic.decode(native_payload, cdfs, count)) != tuple(int(v) for v in symbols):
            raise LifecycleError("Python proof decoder differs from persisted native payload")
        cases.append(
            {
                "alphabet": alphabet,
                "location": location,
                "scale": scale,
                "symbols": count,
                "payload_bytes": len(native_payload),
                "payload_sha256": _sha256_bytes(native_payload),
            }
        )
    return {
        "scope": "native-cpp17-arithmetic-core-with-python-container-and-boundary-path",
        "python_native_payload_and_decode_parity": True,
        "cases": cases,
    }


def focused_proof_suite() -> dict[str, Any]:
    """Run the source-bound, pixel-free COMP-009 proof suite in a fresh process."""
    command = [sys.executable, "-m", "pytest", "-q", *_FOCUSED_TESTS]
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise LifecycleError(f"focused pre-data proof suite failed:\n{output}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "output": output,
        "output_sha256": _sha256_bytes(output.encode("utf-8")),
        "development_streams_opened": False,
    }


def _verify_input_metadata(input_dir: Path) -> dict[str, Any]:
    preflight = _read_object(input_dir / "preflight.json")
    oracle_run._verify_seal(preflight, "binding_sha256")
    if (
        preflight.get("binding_sha256") != EXPECTED_INPUT_BINDING
        or preflight.get("schema") != "structsplat.comp008.preflight.v1"
        or preflight.get("protocol") != "COMP-008-mean-conditioned-entropy-oracle-v1"
        or preflight.get("stage") != "preflight"
        or preflight.get("scientific_pixels_accessed") is not False
    ):
        raise LifecycleError("COMP-008 input binding differs from the audited v3 binding")
    analysis = _read_object(input_dir / "analysis.json")
    oracle.verify_record(analysis, "analysis_sha256")
    if analysis.get("analysis_sha256") != EXPECTED_INPUT_ANALYSIS:
        raise LifecycleError("COMP-008 analysis seal differs from the audited result")
    if analysis.get("decision") != "ORACLE_INCONCLUSIVE_IMPLEMENT_CODER":
        raise LifecycleError("COMP-008 did not authorize the actual-coder task")
    replay = _read_object(input_dir / "replay.json")
    oracle.verify_record(replay, "lifecycle_replay_sha256")
    if (
        replay.get("lifecycle_replay_sha256") != EXPECTED_INPUT_REPLAY
        or replay.get("analysis_record_sha256") != EXPECTED_INPUT_ANALYSIS_RECORD
        or replay.get("schema") != "structsplat.comp008.lifecycle-replay.v1"
        or replay.get("protocol") != "COMP-008-mean-conditioned-entropy-oracle-v1"
        or replay.get("stage") != "replay"
        or replay.get("ok") is not True
        or replay.get("preflight_binding_sha256") != EXPECTED_INPUT_BINDING
        or replay.get("confirmation_payloads_accessed") is not False
        or replay.get("pixel_payloads_decoded") is not False
    ):
        raise LifecycleError("COMP-008 replay seal differs from the audited result")
    rows = _read_jsonl(input_dir / "cells.jsonl")
    if _sha256_file(input_dir / "cells.jsonl") != EXPECTED_INPUT_CELLS_FILE:
        raise LifecycleError("COMP-008 cell journal differs from the independently audited file")
    if len(rows) != 16:
        raise LifecycleError("COMP-008 input metadata does not contain 16 cells")
    identities: set[tuple[str, tuple[int, ...]]] = set()
    stream_records = []
    for row in rows:
        oracle_run._verify_seal(row, "cell_record_sha256")
        identity = (str(row.get("image_id")), tuple(row.get("bit_tuple", ())))
        identities.add(identity)
        stream = row.get("sspl1")
        if not isinstance(stream, dict):
            raise LifecycleError("COMP-008 cell lacks SSPL1 metadata")
        absolute_symbols = row.get("absolute_symbols")
        integrity = row.get("integrity")
        oracle_row = row.get("oracle_row")
        if (
            not isinstance(absolute_symbols, dict)
            or not isinstance(integrity, dict)
            or not isinstance(oracle_row, dict)
        ):
            raise LifecycleError("COMP-008 cell lacks symbol/state integrity metadata")
        if (
            integrity.get("boundary_state_hash_exact") is not True
            or integrity.get("production_vs_independent_max_abs_lte_1e_6") is not True
            or oracle_row.get("image_id") != identity[0]
            or tuple(oracle_row.get("bit_tuple", ())) != identity[1]
            or oracle_row.get("sspl1_sha256") != stream.get("sha256")
            or oracle_row.get("symbol_sha256") != absolute_symbols.get("symbol_sha256")
        ):
            raise LifecycleError("COMP-008 cell completion/integrity identity is not exact")
        if (
            not isinstance(stream.get("path"), str)
            or isinstance(stream.get("bytes"), bool)
            or not isinstance(stream.get("bytes"), int)
            or int(stream["bytes"]) <= 0
            or not isinstance(stream.get("sha256"), str)
            or len(stream["sha256"]) != 64
            or not isinstance(absolute_symbols.get("symbol_sha256"), str)
            or len(absolute_symbols["symbol_sha256"]) != 64
            or not isinstance(integrity.get("decoded_state_sha256"), str)
            or len(integrity["decoded_state_sha256"]) != 64
        ):
            raise LifecycleError("COMP-008 cell stream metadata is malformed")
        stream_records.append(
            {
                "image_id": identity[0],
                "bit_tuple": list(identity[1]),
                "cell_record_sha256": row["cell_record_sha256"],
                "relative_path": stream.get("path"),
                "bytes": stream.get("bytes"),
                "sha256": stream.get("sha256"),
                "symbol_sha256": absolute_symbols.get("symbol_sha256"),
                "decoded_state_sha256": integrity.get("decoded_state_sha256"),
            }
        )
    expected = {
        (image, bits) for image in oracle.FROZEN_IMAGE_IDS for bits in oracle.FROZEN_BIT_TUPLES
    }
    if identities != expected:
        raise LifecycleError("COMP-008 input metadata has an unexpected image/tuple grid")
    analysis_record = _read_object(input_dir / "analysis_record.json")
    oracle_run._verify_seal(analysis_record, "analysis_record_sha256")
    if (
        _sha256_file(input_dir / "analysis_record.json") != EXPECTED_INPUT_ANALYSIS_RECORD_FILE
        or analysis_record.get("analysis_record_sha256") != EXPECTED_INPUT_ANALYSIS_RECORD
        or analysis_record.get("analysis_sha256") != EXPECTED_INPUT_ANALYSIS
        or analysis_record.get("preflight_binding_sha256") != EXPECTED_INPUT_BINDING
        or analysis_record.get("cell_record_sha256") != [row["cell_record_sha256"] for row in rows]
    ):
        raise LifecycleError(
            "COMP-008 analysis record/cell binding differs from the audited result"
        )
    return {
        "directory": str(input_dir.resolve()),
        "preflight_file_sha256": _sha256_file(input_dir / "preflight.json"),
        "preflight_binding_sha256": preflight["binding_sha256"],
        "analysis_file_sha256": _sha256_file(input_dir / "analysis.json"),
        "analysis_sha256": analysis["analysis_sha256"],
        "replay_file_sha256": _sha256_file(input_dir / "replay.json"),
        "replay_sha256": replay["lifecycle_replay_sha256"],
        "cells_file_sha256": _sha256_file(input_dir / "cells.jsonl"),
        "analysis_record_file_sha256": _sha256_file(input_dir / "analysis_record.json"),
        "analysis_record_sha256": analysis_record["analysis_record_sha256"],
        "stream_metadata": sorted(
            stream_records,
            key=lambda row: (
                oracle.FROZEN_IMAGE_IDS.index(row["image_id"]),
                oracle.FROZEN_BIT_TUPLES.index(tuple(row["bit_tuple"])),
            ),
        ),
        "development_streams_opened": False,
        "confirmation_payloads_accessed": False,
    }


def preflight(outdir: Path, input_dir: Path) -> dict[str, Any]:
    if outdir.exists() and any(outdir.iterdir()):
        raise LifecycleError("preflight requires an empty output directory")
    outdir.mkdir(parents=True, exist_ok=True)
    preload_entries = [entry for entry in os.environ.get("LD_PRELOAD", "").split(":") if entry]
    if REQUIRED_RENDERER_PRELOAD not in preload_entries:
        raise LifecycleError(
            "preflight requires LD_PRELOAD="
            f"{REQUIRED_RENDERER_PRELOAD} for the frozen CUDA renderer ABI"
        )
    if any(os.environ.get(name) != value for name, value in FROZEN_THREAD_ENVIRONMENT.items()):
        raise LifecycleError(
            "preflight requires OMP_NUM_THREADS=OPENBLAS_NUM_THREADS="
            "MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1"
        )
    started = time.perf_counter()
    started_utc = _utc_now()
    manifest_before_proofs = source_manifest()
    asset_record = assets.verify_assets()
    input_record = _verify_input_metadata(input_dir.resolve())
    constants = static_constant_record()
    if (
        constants["shuffle_permutation_sha256"] != constants["expected_shuffle_permutation_sha256"]
        or constants["bootstrap_sha256"] != constants["expected_bootstrap_sha256"]
    ):
        raise LifecycleError("frozen shuffle/bootstrap constant drift")
    compiler = compiler_record()
    native_path = arithmetic.build_native(
        outdir / "native/libssp2e_range_v1.so", compiler=compiler["resolved"]
    )
    native_record = {
        "path": native_path.relative_to(outdir).as_posix(),
        "bytes": native_path.stat().st_size,
        "sha256": _sha256_file(native_path),
        "abi": dynamic_library_record(native_path),
        "proof": persisted_native_proof(native_path),
    }
    renderer_record = renderer_runtime_proof()
    proof_record = focused_proof_suite()
    manifest = source_manifest()
    if manifest != manifest_before_proofs:
        raise LifecycleError("execution source closure changed during preflight proofs/build")
    _exclusive_write(outdir / "source_manifest.json", _canonical_json(manifest))
    _write_source_archive(outdir / "source_snapshot.tar", manifest)
    record = {
        "schema": PREFLIGHT_SCHEMA,
        "protocol": "COMP-009-ssp2e-actual-coder-v1",
        "stage": "preflight",
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "source_archive_sha256": _sha256_file(outdir / "source_snapshot.tar"),
        "repository": repository_record(),
        "assets": asset_record,
        "static_constants": constants,
        "compiler": compiler,
        "native": native_record,
        "renderer": renderer_record,
        "environment": environment_record(),
        "input_metadata": input_record,
        "proofs": proof_record,
        "development_streams_opened": False,
        "confirmation_payloads_accessed": False,
        "proof_status": "passed",
    }
    sealed = actual.seal_record(record, "preflight_binding_sha256")
    _exclusive_write(outdir / "preflight.json", _canonical_json(sealed))
    return sealed


def verify_preflight(outdir: Path) -> dict[str, Any]:
    record = _read_object(outdir / "preflight.json")
    actual.verify_record(record, "preflight_binding_sha256")
    if (
        record.get("schema") != PREFLIGHT_SCHEMA
        or record.get("protocol") != "COMP-009-ssp2e-actual-coder-v1"
        or record.get("stage") != "preflight"
        or record.get("proof_status") != "passed"
        or record.get("development_streams_opened") is not False
        or record.get("confirmation_payloads_accessed") is not False
        or record.get("proofs", {}).get("development_streams_opened") is not False
        or record.get("input_metadata", {}).get("development_streams_opened") is not False
        or record.get("input_metadata", {}).get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("unexpected COMP-009 preflight schema")
    current = source_manifest()
    if current["source_manifest_sha256"] != record.get("source_manifest_sha256"):
        raise LifecycleError("COMP-009 source drift after preflight")
    persisted_manifest = _read_object(outdir / "source_manifest.json")
    actual.verify_record(persisted_manifest, "source_manifest_sha256")
    if persisted_manifest != current:
        raise LifecycleError("persisted COMP-009 source manifest drift")
    if _sha256_file(outdir / "source_snapshot.tar") != record.get("source_archive_sha256"):
        raise LifecycleError("COMP-009 source archive drift")
    native = record.get("native")
    if not isinstance(native, dict):
        raise LifecycleError("preflight native record is missing")
    native_path = outdir / str(native.get("path"))
    if native_path.stat().st_size != native.get("bytes") or _sha256_file(native_path) != native.get(
        "sha256"
    ):
        raise LifecycleError("native decoder binary drift")
    if assets.verify_assets() != record.get("assets"):
        raise LifecycleError("static asset drift")
    if static_constant_record() != record.get("static_constants"):
        raise LifecycleError("frozen shuffle/bootstrap constant drift")
    if compiler_record() != record.get("compiler"):
        raise LifecycleError("compiler or ABI binding drift")
    if persisted_native_proof(native_path) != native.get("proof"):
        raise LifecycleError("persisted native proof drift")
    if dynamic_library_record(native_path) != native.get("abi"):
        raise LifecycleError("persisted native library ABI drift")
    renderer = record.get("renderer")
    if not isinstance(renderer, dict):
        raise LifecycleError("CUDA renderer preflight record is missing")
    verify_renderer_runtime(renderer)
    if environment_record() != record.get("environment"):
        raise LifecycleError("runtime environment drift after preflight")
    return record


def import_development(outdir: Path, input_dir: Path) -> dict[str, Any]:
    preflight_record = verify_preflight(outdir)
    current_input = _verify_input_metadata(input_dir.resolve())
    if current_input != preflight_record.get("input_metadata"):
        raise LifecycleError("COMP-008 input metadata drift after COMP-009 preflight")
    imported = []
    for metadata in current_input["stream_metadata"]:
        relative = metadata.get("relative_path")
        if not isinstance(relative, str):
            raise LifecycleError("input stream path is invalid")
        source = (input_dir / relative).resolve()
        try:
            source.relative_to(input_dir.resolve())
        except ValueError as exc:
            raise LifecycleError("input stream path escapes the COMP-008 artifact") from exc
        blob = source.read_bytes()  # first legal opening of one development SSPL1 payload
        if len(blob) != metadata["bytes"] or _sha256_bytes(blob) != metadata["sha256"]:
            raise LifecycleError("COMP-008 development stream byte drift")
        parsed = oracle.parse_sspl1(blob)
        if parsed.header["n"] != oracle.FROZEN_GAUSSIAN_COUNT:
            raise LifecycleError("COMP-009 production import requires exactly N=8192")
        if tuple(parsed.header["bits"]) != tuple(metadata["bit_tuple"]):
            raise LifecycleError("COMP-009 parsed tuple differs from its sealed cell identity")
        if max(int(parsed.header["H"]), int(parsed.header["W"])) != 768:
            raise LifecycleError("COMP-009 input is not the frozen max-side-768 field")
        if parsed.symbol_sha256 != metadata["symbol_sha256"]:
            raise LifecycleError("COMP-008 development stream symbol drift")
        label = f"m{metadata['bit_tuple'][0]}_s{metadata['bit_tuple'][1]}_r{metadata['bit_tuple'][2]}_c{metadata['bit_tuple'][3]}"
        destination = outdir / "inputs" / metadata["image_id"] / f"{label}.sspl1"
        _exclusive_write(destination, blob)
        imported.append(
            {
                **metadata,
                "imported_path": destination.relative_to(outdir).as_posix(),
                "imported_sha256": _sha256_file(destination),
                "parsed_symbol_sha256": parsed.symbol_sha256,
            }
        )
    manifest = actual.seal_record(
        {
            "schema": INPUT_SCHEMA,
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "preflight_binding_sha256": preflight_record["preflight_binding_sha256"],
            "source_input_directory": str(input_dir.resolve()),
            "streams": imported,
            "stream_count": len(imported),
            "development_streams_opened": True,
            "target_pixels_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "input_manifest_sha256",
    )
    _exclusive_write(outdir / "input_manifest.json", _canonical_json(manifest))
    return manifest


def _tuple_label(bits: Sequence[int]) -> str:
    if tuple(bits) not in oracle.FROZEN_BIT_TUPLES:
        raise LifecycleError(f"unexpected COMP-009 bit tuple {tuple(bits)}")
    return f"m{bits[0]}_s{bits[1]}_r{bits[2]}_c{bits[3]}"


def _verify_input_manifest(outdir: Path) -> dict[str, Any]:
    preflight_record = verify_preflight(outdir)
    manifest = _read_object(outdir / "input_manifest.json")
    actual.verify_record(manifest, "input_manifest_sha256")
    if manifest.get("schema") != INPUT_SCHEMA:
        raise LifecycleError("unexpected COMP-009 input-manifest schema")
    if manifest.get("preflight_binding_sha256") != preflight_record.get("preflight_binding_sha256"):
        raise LifecycleError("input manifest is not bound to this preflight")
    streams = manifest.get("streams")
    if not isinstance(streams, list) or len(streams) != 16:
        raise LifecycleError("input manifest must contain exactly 16 streams")
    frozen_input = preflight_record.get("input_metadata")
    if not isinstance(frozen_input, dict):
        raise LifecycleError("preflight lacks its frozen COMP-008 input metadata")
    if (
        manifest.get("source_input_directory") != frozen_input.get("directory")
        or manifest.get("stream_count") != 16
        or manifest.get("development_streams_opened") is not True
        or manifest.get("target_pixels_opened") is not False
        or manifest.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("input-manifest scope/count differs from the frozen import contract")
    identities = []
    frozen_streams = frozen_input.get("stream_metadata")
    if not isinstance(frozen_streams, list) or len(frozen_streams) != 16:
        raise LifecycleError("preflight input metadata lacks sixteen frozen stream records")
    for metadata, frozen_metadata in zip(streams, frozen_streams, strict=True):
        if not isinstance(metadata, dict):
            raise LifecycleError("input-manifest stream record is not an object")
        if any(metadata.get(name) != value for name, value in frozen_metadata.items()):
            raise LifecycleError("imported stream metadata detached from frozen COMP-008 identity")
        identity = (str(metadata.get("image_id")), tuple(metadata.get("bit_tuple", ())))
        identities.append(identity)
        relative = metadata.get("imported_path")
        if not isinstance(relative, str):
            raise LifecycleError("imported stream path is invalid")
        path = (outdir / relative).resolve()
        try:
            path.relative_to(outdir.resolve())
        except ValueError as exc:
            raise LifecycleError("imported stream path escapes the output directory") from exc
        if (
            not path.is_file()
            or path.stat().st_size != metadata.get("bytes")
            or _sha256_file(path) != metadata.get("sha256")
            or metadata.get("imported_sha256") != metadata.get("sha256")
        ):
            raise LifecycleError("imported development stream drift")
        parsed = oracle.parse_sspl1(path.read_bytes())
        if (
            tuple(parsed.header["bits"]) != identity[1]
            or parsed.symbol_sha256 != metadata.get("symbol_sha256")
            or parsed.header["n"] != oracle.FROZEN_GAUSSIAN_COUNT
        ):
            raise LifecycleError("imported development stream semantic drift")
    expected = [
        (image, bits) for image in oracle.FROZEN_IMAGE_IDS for bits in oracle.FROZEN_BIT_TUPLES
    ]
    if identities != expected:
        raise LifecycleError("input-manifest stream order/grid differs from the frozen order")
    return manifest


def _verify_execution_record(record: Mapping[str, Any]) -> None:
    observed = record.get("execution_sha256")
    core = dict(record)
    core.pop("execution_sha256", None)
    expected = _sha256_bytes(_canonical_json(core))
    if record.get("schema") != execute.EXECUTION_SCHEMA or observed != expected:
        raise LifecycleError("persisted execution record seal/schema mismatch")


def _npy_payload(value: np.ndarray) -> bytes:
    with io.BytesIO() as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        return handle.getvalue()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(_canonical_json(record))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _read_resumable_journal(path: Path, stage: str) -> list[dict[str, Any]]:
    """Preserve and truncate only a non-newline crash tail, then parse the canonical prefix."""
    if not path.exists():
        return []
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        prefix_end = payload.rfind(b"\n") + 1
        prefix = payload[:prefix_end]
        tail = payload[prefix_end:]
        recovery = path.parent / "recovery" / (f"{stage}-partial-tail-{_sha256_bytes(tail)}.bin")
        if recovery.exists():
            if recovery.read_bytes() != tail:
                raise LifecycleError("journal recovery artifact hash collision/drift")
        else:
            _exclusive_write(recovery, tail)
        descriptor = os.open(path, os.O_WRONLY)
        try:
            os.ftruncate(descriptor, len(prefix))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return _read_jsonl(path)


def _recovery_artifacts(outdir: Path, stage: str) -> list[dict[str, Any]]:
    directory = outdir / "recovery"
    if not directory.exists():
        return []
    return [
        {
            "path": path.relative_to(outdir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(directory.glob(f"{stage}-partial-tail-*.bin"))
    ]


def _cell_path(outdir: Path, image_id: str, bits: Sequence[int]) -> Path:
    return outdir / "cells" / image_id / _tuple_label(bits)


def _verify_run_cell(outdir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    actual.verify_record(record, "cell_record_sha256")
    if (
        record.get("schema") != "structsplat.comp009.run-cell.v1"
        or record.get("protocol") != "COMP-009-ssp2e-actual-coder-v1"
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("unexpected COMP-009 run-cell schema")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise LifecycleError("run-cell artifact manifest is absent")
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise LifecycleError("run-cell artifact entry is malformed")
        relative = artifact["path"]
        if relative in seen:
            raise LifecycleError("duplicate run-cell artifact path")
        seen.add(relative)
        path = (outdir / relative).resolve()
        try:
            path.relative_to(outdir.resolve())
        except ValueError as exc:
            raise LifecycleError("run-cell artifact path escapes output directory") from exc
        if (
            not path.is_file()
            or path.stat().st_size != artifact.get("bytes")
            or _sha256_file(path) != artifact.get("sha256")
        ):
            raise LifecycleError(f"persisted run-cell artifact drift: {relative}")
    execution_record = record.get("execution")
    if not isinstance(execution_record, dict):
        raise LifecycleError("run cell lacks its sealed execution record")
    _verify_execution_record(execution_record)
    integrity = record.get("integrity")
    if (
        not isinstance(integrity, dict)
        or set(integrity) != actual.REQUIRED_INTEGRITY_KEYS
        or any(value is not True for value in integrity.values())
    ):
        raise LifecycleError("run-cell integrity contract is not exact")
    image_id = str(record.get("image_id"))
    bits = tuple(record.get("bit_tuple", ()))
    base = _cell_path(outdir, image_id, bits)
    input_record = record.get("input")
    if not isinstance(input_record, dict) or not isinstance(input_record.get("path"), str):
        raise LifecycleError("run cell lacks its exact input identity")
    input_manifest = _read_object(outdir / "input_manifest.json")
    actual.verify_record(input_manifest, "input_manifest_sha256")
    matching_inputs = [
        item
        for item in input_manifest.get("streams", [])
        if item.get("image_id") == image_id and tuple(item.get("bit_tuple", ())) == bits
    ]
    if len(matching_inputs) != 1:
        raise LifecycleError("run cell does not identify exactly one imported input")
    expected_input = matching_inputs[0]
    expected_input_record = {
        "path": expected_input["imported_path"],
        "bytes": expected_input["bytes"],
        "sha256": expected_input["sha256"],
        "symbol_sha256": expected_input["symbol_sha256"],
        "decoded_state_sha256": expected_input["decoded_state_sha256"],
        "comp008_cell_record_sha256": expected_input["cell_record_sha256"],
    }
    if input_record != expected_input_record:
        raise LifecycleError("run-cell input record differs from its imported manifest entry")
    source_path = (outdir / input_record["path"]).resolve()
    try:
        source_path.relative_to(outdir.resolve())
    except ValueError as exc:
        raise LifecycleError("run-cell input path escapes output directory") from exc
    source = source_path.read_bytes()
    parts = container.extract_sspl1_parts(source)
    if (
        len(source) != input_record.get("bytes")
        or _sha256_bytes(source) != input_record.get("sha256")
        or parts.symbol_sha256 != input_record.get("symbol_sha256")
        or tuple(parts.header.bits) != bits
        or execution_record["source"]["sha256"] != input_record.get("sha256")
        or execution_record["source"]["symbol_sha256"] != input_record.get("symbol_sha256")
        or tuple(execution_record["source"]["bit_tuple"]) != bits
    ):
        raise LifecycleError("run-cell input/execution semantic identity drift")
    if (base / "execution.json").read_bytes() != _canonical_json(execution_record):
        raise LifecycleError("persisted execution.json differs from the embedded execution record")

    expected_relative = {"execution.json"}
    provider = _static_frequency_provider()
    for name, magic in zip(
        execute.STREAM_NAMES, (b"SSP2E", b"SSP2S", b"SSP2L", b"SSP2F"), strict=True
    ):
        relative = f"streams/{name}.bin"
        expected_relative.add(relative)
        blob = (base / relative).read_bytes()
        parsed = container.parse_container(blob, reference_sspl1=source)
        if parsed.magic != magic:
            raise LifecycleError(f"persisted {name} stream has the wrong magic")
        recomputed_stream = execute._stream_record(name, blob, parts, provider)  # noqa: SLF001
        if recomputed_stream != execution_record["streams"][name]:
            raise LifecycleError(f"persisted {name} stream accounting/certificate drift")

    symbol_arrays: dict[str, np.ndarray] = {}
    for name in container.SYMBOL_CHANNELS:
        relative = f"symbols/{name}.npy"
        expected_relative.add(relative)
        value = np.load(io.BytesIO((base / relative).read_bytes()), allow_pickle=False)
        symbol_arrays[name] = np.ascontiguousarray(value)
        if (
            symbol_arrays[name].dtype != np.uint32
            or symbol_arrays[name].shape != (parts.header.n,)
            or not np.array_equal(symbol_arrays[name], parts.symbols[name])
        ):
            raise LifecycleError(f"persisted absolute symbol array drift in {name}")
    if execute._array_state_digest(  # noqa: SLF001
        execute.boundary_arrays(parts.header, symbol_arrays)
    ) != input_record.get("decoded_state_sha256"):
        raise LifecycleError("persisted symbol arrays do not reproduce the frozen boundary hash")

    expected_boundary = execute.boundary_arrays(parts.header, parts.symbols)
    persisted_boundary: dict[str, np.ndarray] = {}
    for name in expected_boundary:
        relative = f"boundary/{name}.npy"
        expected_relative.add(relative)
        value = np.load(io.BytesIO((base / relative).read_bytes()), allow_pickle=False)
        persisted_boundary[name] = np.ascontiguousarray(value)
        if (
            persisted_boundary[name].dtype != np.float32
            or persisted_boundary[name].shape != expected_boundary[name].shape
            or not np.array_equal(persisted_boundary[name], expected_boundary[name])
            or execute._component_record(persisted_boundary[name])  # noqa: SLF001
            != execution_record["boundary"]["components"][name]
        ):
            raise LifecycleError(f"persisted decoded boundary array drift in {name}")
    if (
        execute._array_state_digest(persisted_boundary)
        != execution_record["boundary"][  # noqa: SLF001
            "state_sha256"
        ]
    ):
        raise LifecycleError("persisted boundary arrays differ from execution boundary identity")

    for variant in ("ssp2e", "ssp2s"):
        starts = execution_record["models"][variant]["starts"]
        if len(starts) != 8 or [start.get("start_id") for start in starts] != list(range(8)):
            raise LifecycleError("persisted execution does not describe eight starts")
        for start in starts:
            start_id = int(start["start_id"])
            grid_relative = f"models/{variant}/start_{start_id}.grid"
            head_relative = f"models/{variant}/start_{start_id}.head"
            expected_relative.update((grid_relative, head_relative))
            grid_raw = (base / grid_relative).read_bytes()
            head_raw = (base / head_relative).read_bytes()
            if (
                _sha256_bytes(grid_raw) != start["grid_sha256"]
                or _sha256_bytes(head_raw) != start["head_sha256"]
                or _sha256_bytes(grid_raw + head_raw) != start["model_sha256"]
            ):
                raise LifecycleError(f"persisted {variant} start {start_id} model drift")
        model_hashes = {start["model_sha256"] for start in starts}
        if execution_record["models"][variant]["selected_model_sha256"] not in model_hashes:
            raise LifecycleError(f"selected {variant} model is not one of the persisted starts")
        parsed_selected = container.parse_container((base / f"streams/{variant}.bin").read_bytes())
        if (
            _sha256_bytes(parsed_selected.payloads["grid"])
            != execution_record["models"][variant]["selected_grid_sha256"]
            or _sha256_bytes(parsed_selected.payloads["head"])
            != execution_record["models"][variant]["selected_head_sha256"]
            or _sha256_bytes(parsed_selected.payloads["grid"] + parsed_selected.payloads["head"])
            != execution_record["models"][variant]["selected_model_sha256"]
        ):
            raise LifecycleError(f"selected {variant} blob/model identity drift")

    artifact_relative = {
        str(Path(artifact["path"]).relative_to(base.relative_to(outdir))) for artifact in artifacts
    }
    if artifact_relative != expected_relative:
        raise LifecycleError("run-cell artifact set differs from the exact persistence contract")
    return dict(record)


def _persist_execution_cell(
    outdir: Path,
    metadata: Mapping[str, Any],
    result: execute.CellExecution,
    wall_seconds: float,
) -> dict[str, Any]:
    image_id = str(metadata["image_id"])
    bits = tuple(int(value) for value in metadata["bit_tuple"])
    final = _cell_path(outdir, image_id, bits)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging_root = outdir / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{image_id}-{_tuple_label(bits)}-", dir=staging_root))
    try:
        payloads: dict[str, bytes] = {
            "execution.json": _canonical_json(result.record),
            **{f"streams/{name}.bin": blob for name, blob in result.blobs.items()},
            **{
                f"boundary/{name}.npy": _npy_payload(value)
                for name, value in result.boundary_arrays.items()
            },
        }
        source_parts = container.extract_sspl1_parts(
            (outdir / str(metadata["imported_path"])).read_bytes()
        )
        payloads.update(
            {
                f"symbols/{name}.npy": _npy_payload(value)
                for name, value in source_parts.symbols.items()
            }
        )
        for variant, selection in (
            ("ssp2e", result.true_selection),
            ("ssp2s", result.shuffled_selection),
        ):
            for encoded in selection.encoded_starts:
                fitted = encoded.fit
                payloads[f"models/{variant}/start_{fitted.start_id}.grid"] = fitted.model.grid_raw
                payloads[f"models/{variant}/start_{fitted.start_id}.head"] = fitted.model.head_raw
        for relative, payload in sorted(payloads.items()):
            _exclusive_write(staging / relative, payload)
        final_relative = final.relative_to(outdir)
        artifacts = [
            {
                "path": (final_relative / relative).as_posix(),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
            for relative, payload in sorted(payloads.items())
        ]
        integrity = {
            **result.record["integrity"],
            "input_sspl1_hash_exact": result.record["source"]["sha256"] == metadata["sha256"],
            "boundary_hash_matches_comp008": result.record["boundary"]["state_sha256"]
            == metadata["decoded_state_sha256"],
            "persisted_artifacts_reverified": all(
                (staging / Path(artifact["path"]).relative_to(final_relative)).stat().st_size
                == artifact["bytes"]
                and _sha256_file(staging / Path(artifact["path"]).relative_to(final_relative))
                == artifact["sha256"]
                for artifact in artifacts
            ),
        }
        record = actual.seal_record(
            {
                "schema": "structsplat.comp009.run-cell.v1",
                "protocol": "COMP-009-ssp2e-actual-coder-v1",
                "image_id": image_id,
                "bit_tuple": list(bits),
                "input": {
                    "path": metadata["imported_path"],
                    "bytes": metadata["bytes"],
                    "sha256": metadata["sha256"],
                    "symbol_sha256": metadata["symbol_sha256"],
                    "decoded_state_sha256": metadata["decoded_state_sha256"],
                    "comp008_cell_record_sha256": metadata["cell_record_sha256"],
                },
                "execution_wall_seconds": wall_seconds,
                "execution": result.record,
                "integrity": integrity,
                "artifacts": artifacts,
                "confirmation_payloads_accessed": False,
            },
            "cell_record_sha256",
        )
        _exclusive_write(staging / "cell.json", _canonical_json(record))
        if final.exists():
            raise LifecycleError(f"refusing to replace existing run cell {final}")
        os.rename(staging, final)
        return _verify_run_cell(outdir, record)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_run_cell(outdir: Path, image_id: str, bits: Sequence[int]) -> dict[str, Any]:
    path = _cell_path(outdir, image_id, bits) / "cell.json"
    return _verify_run_cell(outdir, _read_object(path))


def run_development(outdir: Path) -> dict[str, Any]:
    input_manifest = _verify_input_manifest(outdir)
    if (outdir / "benchmark_manifest.json").exists() or (outdir / "analysis.json").exists():
        raise LifecycleError("run-dev cannot follow benchmark-dev or analysis")
    journal = outdir / "run_cells.jsonl"
    rows = _read_resumable_journal(journal, "run-dev")
    existing: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    if len(rows) > len(input_manifest["streams"]):
        raise LifecycleError("run-dev journal exceeds the frozen input grid")
    for index, row in enumerate(rows):
        checked = _verify_run_cell(outdir, row)
        identity = (str(checked["image_id"]), tuple(checked["bit_tuple"]))
        frozen = input_manifest["streams"][index]
        expected_identity = (str(frozen["image_id"]), tuple(frozen["bit_tuple"]))
        if identity != expected_identity:
            raise LifecycleError("run-dev journal is not an exact frozen-order prefix")
        if identity in existing:
            raise LifecycleError("duplicate run-dev journal identity")
        existing[identity] = checked
    preflight_record = verify_preflight(outdir)
    native_path = outdir / str(preflight_record["native"]["path"])
    native = arithmetic.NativeArithmeticCoder(native_path)
    ordered_seals = []
    for metadata in input_manifest["streams"]:
        identity = (str(metadata["image_id"]), tuple(metadata["bit_tuple"]))
        if identity in existing:
            cell = existing[identity]
        elif (_cell_path(outdir, *identity) / "cell.json").exists():
            cell = _load_run_cell(outdir, *identity)
            _append_jsonl(journal, cell)
            existing[identity] = cell
        else:
            source = (outdir / str(metadata["imported_path"])).read_bytes()
            started = time.perf_counter()
            result = execute.execute_cell(
                source,
                native,
                expected_symbol_sha256=str(metadata["symbol_sha256"]),
                expected_boundary_state_sha256=str(metadata["decoded_state_sha256"]),
            )
            cell = _persist_execution_cell(outdir, metadata, result, time.perf_counter() - started)
            _append_jsonl(journal, cell)
            existing[identity] = cell
        ordered_seals.append(cell["cell_record_sha256"])
    if len(existing) != 16:
        raise LifecycleError("run-dev did not produce exactly 16 cells")
    manifest = actual.seal_record(
        {
            "schema": RUN_SCHEMA,
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "preflight_binding_sha256": preflight_record["preflight_binding_sha256"],
            "input_manifest_sha256": input_manifest["input_manifest_sha256"],
            "cells_file_sha256": _sha256_file(journal),
            "cell_record_sha256": ordered_seals,
            "cell_count": len(ordered_seals),
            "journal_recovery_artifacts": _recovery_artifacts(outdir, "run-dev"),
            "development_streams_opened": True,
            "target_pixels_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "run_manifest_sha256",
    )
    manifest_path = outdir / "run_manifest.json"
    if manifest_path.exists():
        if _read_object(manifest_path) != manifest:
            raise LifecycleError("existing run manifest differs from replayed run state")
    else:
        _exclusive_write(manifest_path, _canonical_json(manifest))
        os.chmod(journal, 0o444)
    return manifest


def verify_run_manifest(outdir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_manifest = _verify_input_manifest(outdir)
    manifest = _read_object(outdir / "run_manifest.json")
    actual.verify_record(manifest, "run_manifest_sha256")
    preflight = _read_object(outdir / "preflight.json")
    if (
        manifest.get("schema") != RUN_SCHEMA
        or manifest.get("protocol") != "COMP-009-ssp2e-actual-coder-v1"
        or manifest.get("preflight_binding_sha256") != preflight.get("preflight_binding_sha256")
        or manifest.get("cell_count") != 16
        or manifest.get("development_streams_opened") is not True
        or manifest.get("target_pixels_opened") is not False
        or manifest.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("unexpected run-manifest schema")
    rows = _read_jsonl(outdir / "run_cells.jsonl")
    checked = []
    if len(rows) != len(input_manifest["streams"]):
        raise LifecycleError("run journal/input-manifest row count mismatch")
    for row, frozen in zip(rows, input_manifest["streams"], strict=True):
        cell = _verify_run_cell(outdir, row)
        if (
            cell.get("image_id") != frozen.get("image_id")
            or tuple(cell.get("bit_tuple", ())) != tuple(frozen.get("bit_tuple", ()))
            or cell.get("input")
            != {
                "path": frozen["imported_path"],
                "bytes": frozen["bytes"],
                "sha256": frozen["sha256"],
                "symbol_sha256": frozen["symbol_sha256"],
                "decoded_state_sha256": frozen["decoded_state_sha256"],
                "comp008_cell_record_sha256": frozen["cell_record_sha256"],
            }
        ):
            raise LifecycleError("run cell detached from its ordered imported input")
        checked.append(cell)
    if (
        len(checked) != 16
        or _sha256_file(outdir / "run_cells.jsonl") != manifest.get("cells_file_sha256")
        or [row["cell_record_sha256"] for row in checked] != manifest.get("cell_record_sha256")
        or manifest.get("input_manifest_sha256") != input_manifest.get("input_manifest_sha256")
        or manifest.get("journal_recovery_artifacts") != _recovery_artifacts(outdir, "run-dev")
    ):
        raise LifecycleError("run-manifest/journal binding mismatch")
    return manifest, checked


def _static_frequency_provider():
    tables = assets.load_tables()

    def provide(alphabet: int, location: int, scale: int) -> tuple[int, ...]:
        return tables[alphabet].frequency_row(location, scale)

    return provide


def _decode_state_for_render(
    blob: bytes, native: arithmetic.NativeArithmeticCoder
) -> tuple[container.ContainerHeader, dict[str, np.ndarray], str]:
    if blob[:5] == b"SSPL1":
        parts = container.extract_sspl1_parts(blob)
        return (
            parts.header,
            execute.boundary_arrays(parts.header, parts.symbols),
            parts.symbol_sha256,
        )
    decoded = container.decode_container(
        blob,
        arithmetic_decode=native.decode,
        arithmetic_encode=native.encode,
        frequency_provider=None if blob[:5] == b"SSP2F" else _static_frequency_provider(),
    )
    return (
        decoded.parsed.header,
        execute.boundary_arrays(decoded.parsed.header, decoded.symbols),
        decoded.symbol_sha256,
    )


def _render_once(
    blob_path: Path,
    native: arithmetic.NativeArithmeticCoder,
    expected_symbol_sha256: str,
    expected_boundary_sha256: str,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    blob = blob_path.read_bytes()
    header, arrays, symbol_sha256 = _decode_state_for_render(blob, native)
    boundary_sha256 = execute._array_state_digest(arrays)  # noqa: SLF001
    if symbol_sha256 != expected_symbol_sha256 or boundary_sha256 != expected_boundary_sha256:
        raise LifecycleError("render diagnostic decoded a nonexact field state")
    device = torch.device("cuda:0")
    field = GaussianField(
        torch.as_tensor(arrays["means"], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["log_scales"], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["rotations"], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["colors"], device=device, dtype=torch.float32),
    )
    torch.cuda.synchronize(device)
    started_ns = time.perf_counter_ns()
    with torch.no_grad():
        image = render_field(
            field.means,
            field.conics(header.aa_dilation),
            field.colors,
            field.radii(header.sigma_cutoff, header.aa_dilation),
            header.height,
            header.width,
            header.render_chunk,
            "cuda",
            scales=field.effective_scales(header.aa_dilation),
            rotations=field.rotations,
            support_fade=bool(header.support_fade),
            sigma_cutoff=header.sigma_cutoff,
        )
    torch.cuda.synchronize(device)
    elapsed_ns = time.perf_counter_ns() - started_ns
    image_array = image.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    if not np.isfinite(image_array).all():
        raise LifecycleError("render diagnostic produced a non-finite image")
    return (
        {
            "blob_path": str(blob_path),
            "blob_sha256": _sha256_bytes(blob),
            "symbol_sha256": symbol_sha256,
            "boundary_state_sha256": boundary_sha256,
            "elapsed_ns": elapsed_ns,
            "image_shape": list(image_array.shape),
            "image_sha256_diagnostic": _sha256_bytes(image_array.tobytes()),
            "all_finite": True,
            "decision_use": False,
        },
        image_array,
    )


def _render_pair_diagnostic(
    primary_path: Path,
    candidate_path: Path,
    native: arithmetic.NativeArithmeticCoder,
    expected_symbol_sha256: str,
    expected_boundary_sha256: str,
) -> dict[str, Any]:
    primary, primary_image = _render_once(
        primary_path, native, expected_symbol_sha256, expected_boundary_sha256
    )
    candidate, candidate_image = _render_once(
        candidate_path, native, expected_symbol_sha256, expected_boundary_sha256
    )
    difference = np.abs(candidate_image.astype(np.float64) - primary_image.astype(np.float64))
    maximum = float(difference.max(initial=0.0))
    tolerance_pass = bool(
        np.allclose(candidate_image, primary_image, rtol=5e-4, atol=5e-4, equal_nan=False)
    )
    return {
        "order": ["primary", "candidate"],
        "primary": primary,
        "candidate": candidate,
        "parity": {
            "rtol": 5e-4,
            "atol": 5e-4,
            "max_abs": maximum,
            "within_tolerance": tolerance_pass,
            "bit_exact_diagnostic": candidate["image_sha256_diagnostic"]
            == primary["image_sha256_diagnostic"],
        },
        "decision_use": False,
    }


def _worker_sample(
    blob_path: Path,
    native_path: Path,
    expected_symbol_sha256: str,
    expected_boundary_sha256: str,
    expected_mode: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2e_decode_worker",
        "--blob",
        str(blob_path),
        "--native-library",
        str(native_path),
        "--expected-symbol-sha256",
        expected_symbol_sha256,
        "--expected-boundary-state-sha256",
        expected_boundary_sha256,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        timeout=300,
    )
    if completed.returncode != 0 or completed.stderr:
        raise LifecycleError(
            "fresh decode worker failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr.decode(errors='replace')}"
        )
    try:
        record = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LifecycleError("fresh decode worker emitted invalid JSON") from exc
    if (
        not isinstance(record, dict)
        or completed.stdout != decode_worker.canonical_json(record)
        or record.get("schema") != decode_worker.WORKER_SCHEMA
        or record.get("mode") != expected_mode
        or record.get("exactness", {}).get("all_exact") is not True
        or record.get("blob_sha256") != _sha256_file(blob_path)
        or record.get("native", {}).get("library_sha256") != _sha256_file(native_path)
        or record.get("n") != oracle.FROZEN_GAUSSIAN_COUNT
    ):
        raise LifecycleError("fresh decode worker record failed exact validation")
    return record


def _primary_selection(cell: Mapping[str, Any]) -> tuple[str, int]:
    execution_record = cell["execution"]
    candidates = (
        ("sspl1", int(execution_record["source"]["bytes"])),
        ("ssp2l", int(execution_record["streams"]["ssp2l"]["complete_bytes"])),
        ("ssp2f", int(execution_record["streams"]["ssp2f"]["complete_bytes"])),
    )
    return min(candidates, key=lambda item: (item[1], ("sspl1", "ssp2l", "ssp2f").index(item[0])))


def _verify_benchmark_cell(
    outdir: Path,
    record: Mapping[str, Any],
    run_cell: Mapping[str, Any],
    native_path: Path,
) -> dict[str, Any]:
    actual.verify_record(record, "benchmark_cell_sha256")
    if record.get("schema") != "structsplat.comp009.benchmark-cell.v1":
        raise LifecycleError("unexpected benchmark-cell schema")
    if (
        record.get("protocol") != "COMP-009-ssp2e-actual-coder-v1"
        or record.get("schedule_rule")
        != "warmup-primary-candidate; measured-candidate-first-iff-(i+t+r)%2==0"
    ):
        raise LifecycleError("benchmark-cell protocol/schedule declaration drift")
    image_id = str(run_cell["image_id"])
    bits = tuple(run_cell["bit_tuple"])
    if (
        record.get("image_id") != image_id
        or tuple(record.get("bit_tuple", ())) != bits
        or record.get("run_cell_sha256") != run_cell.get("cell_record_sha256")
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("benchmark/run-cell identity binding mismatch")
    execution_record = run_cell["execution"]
    expected_primary_name, expected_primary_bytes = _primary_selection(run_cell)
    if record.get("primary_selection") != {
        "name": expected_primary_name,
        "bytes": expected_primary_bytes,
        "tie_order": ["sspl1", "ssp2l", "ssp2f"],
    }:
        raise LifecycleError("benchmark primary selection/tie order drift")
    source_path = outdir / str(run_cell["input"]["path"])
    candidate_path = _cell_path(outdir, image_id, bits) / "streams/ssp2e.bin"
    primary_path = (
        source_path
        if expected_primary_name == "sspl1"
        else _cell_path(outdir, image_id, bits) / f"streams/{expected_primary_name}.bin"
    )
    source = source_path.read_bytes()
    native = arithmetic.NativeArithmeticCoder(native_path)
    expected_symbol = str(run_cell["input"]["symbol_sha256"])
    expected_boundary = str(run_cell["input"]["decoded_state_sha256"])
    for name in execute.STREAM_NAMES:
        path = _cell_path(outdir, image_id, bits) / f"streams/{name}.bin"
        blob = path.read_bytes()
        decoded = container.decode_container(
            blob,
            arithmetic_decode=native.decode,
            arithmetic_encode=native.encode,
            frequency_provider=None if name == "ssp2f" else _static_frequency_provider(),
            reference_sspl1=source,
        )
        boundary = execute.boundary_arrays(decoded.parsed.header, decoded.symbols)
        if (
            len(blob) != execution_record["streams"][name]["complete_bytes"]
            or _sha256_bytes(blob) != execution_record["streams"][name]["blob_sha256"]
            or decoded.symbol_sha256 != expected_symbol
            or execute._array_state_digest(boundary) != expected_boundary  # noqa: SLF001
        ):
            raise LifecycleError(f"benchmark source stream drift in {name}")

    launches = record.get("launches")
    if not isinstance(launches, list) or len(launches) != 20:
        raise LifecycleError("benchmark cell must contain two warm-ups and 18 measured launches")
    image_index = oracle.FROZEN_IMAGE_IDS.index(image_id)
    tuple_index = oracle.FROZEN_BIT_TUPLES.index(bits)
    expected_schedule: list[tuple[str, int | None, str]] = [
        ("warmup", None, "primary"),
        ("warmup", None, "candidate"),
    ]
    for repetition in range(9):
        order = (
            ("candidate", "primary")
            if (image_index + tuple_index + repetition) % 2 == 0
            else ("primary", "candidate")
        )
        expected_schedule.extend(("measured", repetition, arm) for arm in order)
    measured: dict[str, list[dict[str, Any]]] = {"candidate": [], "primary": []}
    preflight = _read_object(outdir / "preflight.json")
    frozen_affinity = preflight.get("environment", {}).get("cpu_affinity")
    if not isinstance(frozen_affinity, list) or not frozen_affinity:
        raise LifecycleError("preflight lacks a nonempty frozen CPU affinity set")
    expected_cpu = min(int(value) for value in frozen_affinity)
    observed_pids: list[int] = []
    for index, (launch, expected_launch) in enumerate(
        zip(launches, expected_schedule, strict=True)
    ):
        phase, repetition, arm = expected_launch
        mode = "ssp2e" if arm == "candidate" else expected_primary_name
        path = candidate_path if arm == "candidate" else primary_path
        sample = launch.get("sample") if isinstance(launch, dict) else None
        if (
            not isinstance(sample, dict)
            or launch.get("launch_index") != index
            or launch.get("phase") != phase
            or launch.get("repetition") != repetition
            or launch.get("arm") != arm
            or launch.get("mode") != mode
            or sample.get("schema") != decode_worker.WORKER_SCHEMA
            or sample.get("mode") != mode
            or sample.get("execution_mode") != "production-resource"
            or sample.get("blob_sha256") != _sha256_file(path)
            or sample.get("blob_bytes") != path.stat().st_size
            or sample.get("n") != oracle.FROZEN_GAUSSIAN_COUNT
            or tuple(sample.get("bit_tuple", ())) != bits
            or sample.get("symbol_sha256") != expected_symbol
            or sample.get("boundary_state_sha256") != expected_boundary
            or sample.get("exactness", {}).get("all_exact") is not True
            or sample.get("exactness")
            != {
                "expected_symbol_sha256": expected_symbol,
                "expected_boundary_state_sha256": expected_boundary,
                "symbol_hash_exact": True,
                "boundary_hash_exact": True,
                "all_exact": True,
            }
            or sample.get("native", {}).get("library_sha256") != _sha256_file(native_path)
            or sample.get("native", {}).get("library_bytes") != native_path.stat().st_size
            or sample.get("assets")
            != {
                "cdf_sha256": assets.CDF_ASSET_SHA256,
                "cdf_bytes": assets.CDF_ASSET_SIZE,
                "loaded_before_timing": True,
            }
            or sample.get("timing", {}).get("scope") != list(decode_worker.TIMING_SCOPE)
            or sample.get("timing", {}).get("renderer_included") is not False
            or not isinstance(sample.get("elapsed_ns"), int)
            or int(sample["elapsed_ns"]) <= 0
            or not isinstance(sample.get("peak_rss_bytes"), int)
            or int(sample["peak_rss_bytes"]) <= 0
            or sample.get("process", {}).get("cpu_affinity") != [expected_cpu]
            or sample.get("process", {}).get("pinned_cpu") != expected_cpu
            or sample.get("process", {}).get("platform") != "linux"
            or sample.get("process", {}).get("thread_environment")
            != {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            or sample.get("process", {}).get("rss_source")
            != "resource.getrusage(RUSAGE_SELF).ru_maxrss_linux_kib_times_1024"
            or not isinstance(sample.get("process", {}).get("pid"), int)
            or int(sample["process"]["pid"]) <= 0
        ):
            raise LifecycleError(f"benchmark worker sample {index} violates the frozen contract")
        observed_pids.append(int(sample["process"]["pid"]))
        if mode != "sspl1" and (
            sample["native"].get("canonical_validation")
            != "native_decode_internal_reencode_and_byte_compare"
            or sample["native"].get("redundant_second_native_encode_eliminated") is not True
            or sample["native"].get("proof_adapter")
            != "one_shot_exact_payload_cdf_identity_content_and_symbol_content"
        ):
            raise LifecycleError("native canonical-proof adapter scope drift")
        if phase == "measured":
            measured[arm].append(sample)
    if len(set(observed_pids)) != 20:
        raise LifecycleError("benchmark launches were not twenty distinct fresh processes")

    recomputed_summaries = {}
    for arm in ("candidate", "primary"):
        elapsed = sorted(int(sample["elapsed_ns"]) for sample in measured[arm])
        rss = [int(sample["peak_rss_bytes"]) for sample in measured[arm]]
        recomputed_summaries[arm] = {
            "measured_count": 9,
            "fifth_sorted_elapsed_ns": elapsed[4],
            "maximum_peak_rss_bytes": max(rss),
            "sorted_elapsed_ns": elapsed,
            "peak_rss_bytes": rss,
        }
    if record.get("summaries") != recomputed_summaries:
        raise LifecycleError("benchmark summary does not recompute from raw worker samples")

    render = record.get("render_diagnostic")
    primary_render = render.get("primary", {}) if isinstance(render, dict) else {}
    candidate_render = render.get("candidate", {}) if isinstance(render, dict) else {}
    parity = render.get("parity", {}) if isinstance(render, dict) else {}
    header = container.extract_sspl1_parts(source).header
    if (
        not isinstance(render, dict)
        or render.get("order") != ["primary", "candidate"]
        or render.get("decision_use") is not False
        or primary_render.get("blob_path") != str(primary_path)
        or candidate_render.get("blob_path") != str(candidate_path)
        or primary_render.get("blob_sha256") != _sha256_file(primary_path)
        or candidate_render.get("blob_sha256") != _sha256_file(candidate_path)
        or primary_render.get("symbol_sha256") != expected_symbol
        or candidate_render.get("symbol_sha256") != expected_symbol
        or primary_render.get("boundary_state_sha256") != expected_boundary
        or candidate_render.get("boundary_state_sha256") != expected_boundary
        or primary_render.get("all_finite") is not True
        or candidate_render.get("all_finite") is not True
        or primary_render.get("image_shape") != [header.height, header.width, 3]
        or candidate_render.get("image_shape") != [header.height, header.width, 3]
        or not isinstance(primary_render.get("elapsed_ns"), int)
        or int(primary_render["elapsed_ns"]) <= 0
        or not isinstance(candidate_render.get("elapsed_ns"), int)
        or int(candidate_render["elapsed_ns"]) <= 0
        or parity.get("rtol") != 5e-4
        or parity.get("atol") != 5e-4
        or not isinstance(parity.get("max_abs"), float)
        or not math.isfinite(parity["max_abs"])
        or not isinstance(parity.get("within_tolerance"), bool)
        or not isinstance(parity.get("bit_exact_diagnostic"), bool)
    ):
        raise LifecycleError("render-only benchmark diagnostic binding drift")

    streams = execution_record["streams"]
    recomputed_row = actual.seal_record(
        {
            "schema": actual.ROW_SCHEMA,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "sspl1_bytes": len(source),
            "ssp2e_bytes": int(streams["ssp2e"]["complete_bytes"]),
            "ssp2s_bytes": int(streams["ssp2s"]["complete_bytes"]),
            "ssp2l_bytes": int(streams["ssp2l"]["complete_bytes"]),
            "ssp2f_bytes": int(streams["ssp2f"]["complete_bytes"]),
            "ssp2e_modeled_bytes": int(streams["ssp2e"]["modeled_bytes"]),
            "ssp2s_modeled_bytes": int(streams["ssp2s"]["modeled_bytes"]),
            "ssp2e_decode_ns": recomputed_summaries["candidate"]["fifth_sorted_elapsed_ns"],
            "primary_decode_ns": recomputed_summaries["primary"]["fifth_sorted_elapsed_ns"],
            "ssp2e_peak_rss_bytes": recomputed_summaries["candidate"]["maximum_peak_rss_bytes"],
            "primary_peak_rss_bytes": recomputed_summaries["primary"]["maximum_peak_rss_bytes"],
            "primary_bytes": expected_primary_bytes,
            "primary_name": expected_primary_name,
            "integrity": run_cell["integrity"],
        }
    )
    if record.get("actual_row") != recomputed_row:
        raise LifecycleError("benchmark actual row does not recompute from raw artifacts")
    return dict(record)


def _benchmark_one(
    outdir: Path,
    cell: Mapping[str, Any],
    native_path: Path,
    image_index: int,
    tuple_index: int,
) -> dict[str, Any]:
    image_id = str(cell["image_id"])
    bits = tuple(cell["bit_tuple"])
    execution_record = cell["execution"]
    candidate_path = _cell_path(outdir, image_id, bits) / "streams/ssp2e.bin"
    primary_name, primary_bytes = _primary_selection(cell)
    primary_path = (
        outdir / str(cell["input"]["path"])
        if primary_name == "sspl1"
        else _cell_path(outdir, image_id, bits) / f"streams/{primary_name}.bin"
    )
    expected_symbol = str(cell["input"]["symbol_sha256"])
    expected_boundary = str(cell["input"]["decoded_state_sha256"])
    launches = []
    samples: dict[str, list[dict[str, Any]]] = {"candidate": [], "primary": []}

    def launch(phase: str, repetition: int | None, arm: str) -> None:
        path = candidate_path if arm == "candidate" else primary_path
        mode = "ssp2e" if arm == "candidate" else primary_name
        sample = _worker_sample(path, native_path, expected_symbol, expected_boundary, mode)
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
            samples[arm].append(sample)

    launch("warmup", None, "primary")
    launch("warmup", None, "candidate")
    for repetition in range(9):
        order = (
            ("candidate", "primary")
            if (image_index + tuple_index + repetition) % 2 == 0
            else ("primary", "candidate")
        )
        for arm in order:
            launch("measured", repetition, arm)
    summaries = {}
    for arm in ("candidate", "primary"):
        elapsed = sorted(int(sample["elapsed_ns"]) for sample in samples[arm])
        rss = [int(sample["peak_rss_bytes"]) for sample in samples[arm]]
        summaries[arm] = {
            "measured_count": len(elapsed),
            "fifth_sorted_elapsed_ns": elapsed[4],
            "maximum_peak_rss_bytes": max(rss),
            "sorted_elapsed_ns": elapsed,
            "peak_rss_bytes": rss,
        }
    render = _render_pair_diagnostic(
        primary_path,
        candidate_path,
        arithmetic.NativeArithmeticCoder(native_path),
        expected_symbol,
        expected_boundary,
    )
    streams = execution_record["streams"]
    row = actual.seal_record(
        {
            "schema": actual.ROW_SCHEMA,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "sspl1_bytes": int(execution_record["source"]["bytes"]),
            "ssp2e_bytes": int(streams["ssp2e"]["complete_bytes"]),
            "ssp2s_bytes": int(streams["ssp2s"]["complete_bytes"]),
            "ssp2l_bytes": int(streams["ssp2l"]["complete_bytes"]),
            "ssp2f_bytes": int(streams["ssp2f"]["complete_bytes"]),
            "ssp2e_modeled_bytes": int(streams["ssp2e"]["modeled_bytes"]),
            "ssp2s_modeled_bytes": int(streams["ssp2s"]["modeled_bytes"]),
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


def benchmark_development(outdir: Path) -> dict[str, Any]:
    run_manifest, cells = verify_run_manifest(outdir)
    if (outdir / "analysis.json").exists():
        raise LifecycleError("benchmark-dev cannot follow analysis")
    preflight_record = verify_preflight(outdir)
    native_path = outdir / str(preflight_record["native"]["path"])
    journal = outdir / "benchmark_cells.jsonl"
    existing_rows = _read_resumable_journal(journal, "benchmark-dev")
    if len(existing_rows) > 16:
        raise LifecycleError("benchmark journal has more than sixteen cells")
    existing: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for index, row in enumerate(existing_rows):
        expected_cell = cells[index]
        checked = _verify_benchmark_cell(outdir, row, expected_cell, native_path)
        identity = (str(checked["image_id"]), tuple(checked["bit_tuple"]))
        expected_identity = (
            str(expected_cell["image_id"]),
            tuple(expected_cell["bit_tuple"]),
        )
        if identity != expected_identity or identity in existing:
            raise LifecycleError("benchmark journal is not an exact frozen-order prefix")
        existing[identity] = checked
    benchmark_cells: list[dict[str, Any]] = []
    actual_rows: list[dict[str, Any]] = []
    for cell in cells:
        identity = (str(cell["image_id"]), tuple(cell["bit_tuple"]))
        if identity in existing:
            checked = existing[identity]
        else:
            image_index = oracle.FROZEN_IMAGE_IDS.index(str(cell["image_id"]))
            tuple_index = oracle.FROZEN_BIT_TUPLES.index(tuple(cell["bit_tuple"]))
            result = _benchmark_one(outdir, cell, native_path, image_index, tuple_index)
            checked = _verify_benchmark_cell(outdir, result, cell, native_path)
            _append_jsonl(journal, checked)
            existing[identity] = checked
        benchmark_cells.append(checked)
        actual_rows.append(checked["actual_row"])
    actual_rows_path = outdir / "actual_rows.jsonl"
    actual_rows_payload = b"".join(_canonical_json(row) for row in actual_rows)
    if actual_rows_path.exists():
        if actual_rows_path.read_bytes() != actual_rows_payload:
            raise LifecycleError("stale actual_rows.jsonl differs from recomputed benchmark rows")
    else:
        _exclusive_write(actual_rows_path, actual_rows_payload)
    manifest = actual.seal_record(
        {
            "schema": BENCHMARK_SCHEMA,
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "preflight_binding_sha256": preflight_record["preflight_binding_sha256"],
            "run_manifest_sha256": run_manifest["run_manifest_sha256"],
            "benchmark_cells_file_sha256": _sha256_file(journal),
            "actual_rows_file_sha256": _sha256_file(actual_rows_path),
            "benchmark_cell_sha256": [cell["benchmark_cell_sha256"] for cell in benchmark_cells],
            "actual_row_sha256": [row["record_sha256"] for row in actual_rows],
            "cell_count": len(benchmark_cells),
            "fresh_measured_launches": 18 * len(benchmark_cells),
            "fresh_warmup_launches": 2 * len(benchmark_cells),
            "renderer_diagnostic_is_nongating": True,
            "journal_recovery_artifacts": _recovery_artifacts(outdir, "benchmark-dev"),
            "confirmation_payloads_accessed": False,
        },
        "benchmark_manifest_sha256",
    )
    manifest_path = outdir / "benchmark_manifest.json"
    if manifest_path.exists():
        if _read_object(manifest_path) != manifest:
            raise LifecycleError("existing benchmark manifest differs from recomputed state")
    else:
        _exclusive_write(manifest_path, _canonical_json(manifest))
    os.chmod(journal, 0o444)
    return manifest


def verify_benchmark_manifest(
    outdir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_manifest, run_cells = verify_run_manifest(outdir)
    preflight_record = verify_preflight(outdir)
    native_path = outdir / str(preflight_record["native"]["path"])
    manifest = _read_object(outdir / "benchmark_manifest.json")
    actual.verify_record(manifest, "benchmark_manifest_sha256")
    if (
        manifest.get("schema") != BENCHMARK_SCHEMA
        or manifest.get("protocol") != "COMP-009-ssp2e-actual-coder-v1"
        or manifest.get("cell_count") != 16
        or manifest.get("run_manifest_sha256") != run_manifest.get("run_manifest_sha256")
        or manifest.get("preflight_binding_sha256")
        != preflight_record.get("preflight_binding_sha256")
        or manifest.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("benchmark manifest provenance/scope drift")
    journal = outdir / "benchmark_cells.jsonl"
    records = _read_jsonl(journal)
    if len(records) != 16 or len(run_cells) != 16:
        raise LifecycleError("benchmark verification requires sixteen exact cells")
    checked = [
        _verify_benchmark_cell(outdir, record, run_cell, native_path)
        for record, run_cell in zip(records, run_cells, strict=True)
    ]
    actual_rows = _read_jsonl(outdir / "actual_rows.jsonl")
    expected_rows = [record["actual_row"] for record in checked]
    if (
        actual_rows != expected_rows
        or _sha256_file(journal) != manifest.get("benchmark_cells_file_sha256")
        or _sha256_file(outdir / "actual_rows.jsonl") != manifest.get("actual_rows_file_sha256")
        or [record["benchmark_cell_sha256"] for record in checked]
        != manifest.get("benchmark_cell_sha256")
        or [row["record_sha256"] for row in actual_rows] != manifest.get("actual_row_sha256")
        or manifest.get("fresh_measured_launches") != 288
        or manifest.get("fresh_warmup_launches") != 32
        or manifest.get("renderer_diagnostic_is_nongating") is not True
        or manifest.get("journal_recovery_artifacts")
        != _recovery_artifacts(outdir, "benchmark-dev")
    ):
        raise LifecycleError("benchmark manifest/raw artifact binding mismatch")
    return manifest, checked, actual_rows


def analyze_results(outdir: Path) -> dict[str, Any]:
    benchmark_manifest, benchmark_cells, rows = verify_benchmark_manifest(outdir)
    _, run_cells = verify_run_manifest(outdir)
    analysis = actual.analyze_rows(rows)
    analysis_path = outdir / "analysis.json"
    if analysis_path.exists():
        if _read_object(analysis_path) != analysis:
            raise LifecycleError("existing analysis differs from raw-artifact recomputation")
    else:
        _exclusive_write(analysis_path, _canonical_json(analysis))
    record = actual.seal_record(
        {
            "schema": "structsplat.comp009.analysis-record.v1",
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "stage": "analyze",
            "benchmark_manifest_sha256": benchmark_manifest["benchmark_manifest_sha256"],
            "benchmark_cell_sha256": [cell["benchmark_cell_sha256"] for cell in benchmark_cells],
            "actual_row_sha256": [row["record_sha256"] for row in rows],
            "analysis_file_sha256": _sha256_file(analysis_path),
            "analysis_sha256": analysis["analysis_sha256"],
            "decision": analysis["decision"],
            "row_count": len(rows),
            "execution_diagnostics": [
                {
                    "image_id": cell["image_id"],
                    "bit_tuple": cell["bit_tuple"],
                    "combined_timings": cell["execution"]["timings"],
                    "ssp2e_restart_spread": cell["execution"]["models"]["ssp2e"]["restart_spread"],
                    "ssp2e_sweeps": [
                        start["sweeps"] for start in cell["execution"]["models"]["ssp2e"]["starts"]
                    ],
                    "ssp2s_restart_spread": cell["execution"]["models"]["ssp2s"]["restart_spread"],
                    "ssp2s_sweeps": [
                        start["sweeps"] for start in cell["execution"]["models"]["ssp2s"]["starts"]
                    ],
                    "decision_use": False,
                }
                for cell in run_cells
            ],
            "raw_blobs_and_worker_samples_reparsed": True,
            "target_pixels_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "analysis_record_sha256",
    )
    record_path = outdir / "analysis_record.json"
    if record_path.exists():
        if _read_object(record_path) != record:
            raise LifecycleError("existing analysis record differs from recomputed bindings")
    else:
        _exclusive_write(record_path, _canonical_json(record))
    return record


def verify_analysis(outdir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark_manifest, benchmark_cells, rows = verify_benchmark_manifest(outdir)
    _, run_cells = verify_run_manifest(outdir)
    analysis = _read_object(outdir / "analysis.json")
    actual.verify_record(analysis, "analysis_sha256")
    recomputed = actual.analyze_rows(rows)
    if analysis != recomputed:
        raise LifecycleError("analysis does not recompute from verified actual rows")
    record = _read_object(outdir / "analysis_record.json")
    actual.verify_record(record, "analysis_record_sha256")
    expected = actual.seal_record(
        {
            "schema": "structsplat.comp009.analysis-record.v1",
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "stage": "analyze",
            "benchmark_manifest_sha256": benchmark_manifest["benchmark_manifest_sha256"],
            "benchmark_cell_sha256": [cell["benchmark_cell_sha256"] for cell in benchmark_cells],
            "actual_row_sha256": [row["record_sha256"] for row in rows],
            "analysis_file_sha256": _sha256_file(outdir / "analysis.json"),
            "analysis_sha256": analysis["analysis_sha256"],
            "decision": analysis["decision"],
            "row_count": len(rows),
            "execution_diagnostics": [
                {
                    "image_id": cell["image_id"],
                    "bit_tuple": cell["bit_tuple"],
                    "combined_timings": cell["execution"]["timings"],
                    "ssp2e_restart_spread": cell["execution"]["models"]["ssp2e"]["restart_spread"],
                    "ssp2e_sweeps": [
                        start["sweeps"] for start in cell["execution"]["models"]["ssp2e"]["starts"]
                    ],
                    "ssp2s_restart_spread": cell["execution"]["models"]["ssp2s"]["restart_spread"],
                    "ssp2s_sweeps": [
                        start["sweeps"] for start in cell["execution"]["models"]["ssp2s"]["starts"]
                    ],
                    "decision_use": False,
                }
                for cell in run_cells
            ],
            "raw_blobs_and_worker_samples_reparsed": True,
            "target_pixels_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "analysis_record_sha256",
    )
    if record != expected:
        raise LifecycleError("analysis record binding drift")
    return record, analysis


def _execution_without_timings(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical_json(record))
    normalized.pop("execution_sha256", None)
    normalized.pop("timings", None)
    return normalized


def captured_replay_worker(outdir: Path) -> dict[str, Any]:
    """Rerun deterministic coding from the extracted preflight source closure."""
    expected_root = os.environ.get("COMP009_CAPTURED_ROOT")
    if os.environ.get("COMP009_CAPTURED_SOURCE") != "1" or not expected_root:
        raise LifecycleError("replay-worker is restricted to the captured-source subprocess")
    if ROOT.resolve() != Path(expected_root).resolve():
        raise LifecycleError("replay-worker module did not load from the extracted source root")
    import structsplat

    expected_package_root = (ROOT / "src/structsplat").resolve()
    module_paths = {
        "benchmarks.ssp2e_actual_run": Path(__file__).resolve(),
        "benchmarks.ssp2e_execute": Path(execute.__file__).resolve(),
        "benchmarks.sgi_entropy_oracle": Path(oracle.__file__).resolve(),
        "structsplat": Path(structsplat.__file__).resolve(),
    }
    for name, path in module_paths.items():
        required_root = ROOT.resolve() if name.startswith("benchmarks.") else expected_package_root
        try:
            path.relative_to(required_root)
        except ValueError as exc:
            raise LifecycleError(
                f"captured replay imported {name} outside extracted source"
            ) from exc
    analysis_record, analysis = verify_analysis(outdir)
    run_manifest, cells = verify_run_manifest(outdir)
    preflight_record = verify_preflight(outdir)
    native_path = outdir / str(preflight_record["native"]["path"])
    native = arithmetic.NativeArithmeticCoder(native_path)
    replay_cells = []
    for cell in cells:
        source = (outdir / str(cell["input"]["path"])).read_bytes()
        regenerated = execute.execute_cell(
            source,
            native,
            expected_symbol_sha256=str(cell["input"]["symbol_sha256"]),
            expected_boundary_state_sha256=str(cell["input"]["decoded_state_sha256"]),
        )
        if _execution_without_timings(regenerated.record) != _execution_without_timings(
            cell["execution"]
        ):
            raise LifecycleError("captured-source execution replay drift")
        blob_hashes = {}
        for name, blob in regenerated.blobs.items():
            persisted = (
                _cell_path(outdir, str(cell["image_id"]), cell["bit_tuple"]) / f"streams/{name}.bin"
            ).read_bytes()
            if blob != persisted:
                raise LifecycleError(f"captured-source complete-stream replay drift in {name}")
            blob_hashes[name] = _sha256_bytes(blob)
        normalized = _execution_without_timings(regenerated.record)
        replay_cells.append(
            {
                "image_id": cell["image_id"],
                "bit_tuple": cell["bit_tuple"],
                "input_sha256": _sha256_bytes(source),
                "normalized_execution_sha256": _sha256_bytes(_canonical_json(normalized)),
                "complete_stream_sha256": blob_hashes,
                "all_non_timing_values_exact": True,
            }
        )
    replayed_analysis = actual.analyze_rows(_read_jsonl(outdir / "actual_rows.jsonl"))
    if replayed_analysis != analysis:
        raise LifecycleError("captured-source aggregate gate/decision replay drift")
    manifest = source_manifest()
    return actual.seal_record(
        {
            "schema": "structsplat.comp009.captured-replay-worker.v1",
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "runner_relative_path": Path(__file__).resolve().relative_to(ROOT.resolve()).as_posix(),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "run_manifest_sha256": run_manifest["run_manifest_sha256"],
            "analysis_record_sha256": analysis_record["analysis_record_sha256"],
            "analysis_sha256": analysis["analysis_sha256"],
            "decision": analysis["decision"],
            "cells": replay_cells,
            "cell_count": len(replay_cells),
            "actual_timing_samples_reused_not_rerun": True,
            "underlying_field_or_qat_refit": False,
            "target_pixels_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "captured_replay_worker_sha256",
    )


def _extract_captured_source(outdir: Path, destination: Path) -> dict[str, Any]:
    preflight_record = verify_preflight(outdir)
    archive_path = outdir / "source_snapshot.tar"
    if _sha256_file(archive_path) != preflight_record.get("source_archive_sha256"):
        raise LifecycleError("source snapshot archive drift before captured replay")
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise LifecycleError("unsafe path/link in captured source archive")
        archive.extractall(destination, members=members, filter="data")
    archived_manifest = _read_object(destination / "SOURCE_MANIFEST.json")
    actual.verify_record(archived_manifest, "source_manifest_sha256")
    disk_manifest = _read_object(outdir / "source_manifest.json")
    if archived_manifest != disk_manifest:
        raise LifecycleError("archived source manifest differs from preflight manifest")
    expected_paths = {"SOURCE_MANIFEST.json"}
    for entry in archived_manifest["files"]:
        path = destination / entry["path"]
        expected_paths.add(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _sha256_file(path) != entry["sha256"]
        ):
            raise LifecycleError(f"captured source extraction drift: {entry['path']}")
    observed_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise LifecycleError("captured source archive contains an unmanifested file")
    return archived_manifest


def replay_results(outdir: Path) -> dict[str, Any]:
    analysis_record, analysis = verify_analysis(outdir)
    preflight_record = verify_preflight(outdir)
    with tempfile.TemporaryDirectory(prefix="comp009-captured-replay-") as temporary:
        extracted = Path(temporary) / "source"
        extracted.mkdir()
        source_record = _extract_captured_source(outdir, extracted)
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join((str(extracted), str(extracted / "src"))),
                "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0",
                **FROZEN_THREAD_ENVIRONMENT,
                "COMP009_CAPTURED_SOURCE": "1",
                "COMP009_CAPTURED_ROOT": str(extracted),
            }
        )
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2e_actual_run",
            "replay-worker",
            "--outdir",
            str(outdir.resolve()),
        ]
        completed = subprocess.run(
            command,
            cwd=extracted,
            env=environment,
            check=False,
            capture_output=True,
            timeout=3600,
        )
        if completed.returncode != 0 or completed.stderr:
            raise LifecycleError(
                "captured-source replay subprocess failed: "
                f"returncode={completed.returncode}, stderr={completed.stderr.decode(errors='replace')}"
            )
        try:
            worker_record = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise LifecycleError("captured-source replay emitted invalid JSON") from exc
        if completed.stdout != _canonical_json(worker_record):
            raise LifecycleError("captured-source replay output is not canonical JSON")
        actual.verify_record(worker_record, "captured_replay_worker_sha256")
        runner_entry = next(
            (
                entry
                for entry in source_record["files"]
                if entry["path"] == "benchmarks/ssp2e_actual_run.py"
            ),
            None,
        )
        run_manifest = _read_object(outdir / "run_manifest.json")
        expected_identities = [
            (image, list(bits))
            for image in oracle.FROZEN_IMAGE_IDS
            for bits in oracle.FROZEN_BIT_TUPLES
        ]
        worker_cells = worker_record.get("cells")
        if (
            worker_record.get("schema") != "structsplat.comp009.captured-replay-worker.v1"
            or worker_record.get("source_manifest_sha256")
            != source_record.get("source_manifest_sha256")
            or worker_record.get("analysis_record_sha256")
            != analysis_record.get("analysis_record_sha256")
            or worker_record.get("analysis_sha256") != analysis.get("analysis_sha256")
            or worker_record.get("decision") != analysis.get("decision")
            or worker_record.get("cell_count") != 16
            or runner_entry is None
            or worker_record.get("runner_relative_path") != "benchmarks/ssp2e_actual_run.py"
            or worker_record.get("runner_sha256") != runner_entry.get("sha256")
            or worker_record.get("run_manifest_sha256") != run_manifest.get("run_manifest_sha256")
            or not isinstance(worker_cells, list)
            or [(cell.get("image_id"), cell.get("bit_tuple")) for cell in worker_cells]
            != expected_identities
            or any(cell.get("all_non_timing_values_exact") is not True for cell in worker_cells)
            or worker_record.get("actual_timing_samples_reused_not_rerun") is not True
            or worker_record.get("underlying_field_or_qat_refit") is not False
            or worker_record.get("target_pixels_opened") is not False
            or worker_record.get("confirmation_payloads_accessed") is not False
        ):
            raise LifecycleError("captured-source replay binding/decision drift")
    replay = actual.seal_record(
        {
            "schema": REPLAY_SCHEMA,
            "protocol": "COMP-009-ssp2e-actual-coder-v1",
            "stage": "replay",
            "preflight_binding_sha256": preflight_record["preflight_binding_sha256"],
            "source_archive_sha256": preflight_record["source_archive_sha256"],
            "analysis_record_sha256": analysis_record["analysis_record_sha256"],
            "analysis_sha256": analysis["analysis_sha256"],
            "decision": analysis["decision"],
            "captured_worker": worker_record,
            "captured_source_subprocess": True,
            "all_non_timing_values_exact": True,
            "actual_timing_samples_reused_not_rerun": True,
            "target_pixels_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "replay_sha256",
    )
    path = outdir / "replay.json"
    if path.exists():
        if _read_object(path) != replay:
            raise LifecycleError("existing replay record differs from captured-source replay")
    else:
        _exclusive_write(path, _canonical_json(replay))
    return replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "import-dev"):
        child = subparsers.add_parser(name)
        child.add_argument("--outdir", type=Path, required=True)
        child.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    for name in ("run-dev", "benchmark-dev", "analyze", "replay", "replay-worker"):
        child = subparsers.add_parser(name)
        child.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.outdir.resolve(), args.input_dir.resolve())
        print(result["preflight_binding_sha256"])
    elif args.command == "import-dev":
        result = import_development(args.outdir.resolve(), args.input_dir.resolve())
        print(result["input_manifest_sha256"])
    elif args.command == "run-dev":
        result = run_development(args.outdir.resolve())
        print(result["run_manifest_sha256"])
    elif args.command == "benchmark-dev":
        result = benchmark_development(args.outdir.resolve())
        print(result["benchmark_manifest_sha256"])
    elif args.command == "analyze":
        result = analyze_results(args.outdir.resolve())
        print(result["analysis_record_sha256"])
    elif args.command == "replay":
        result = replay_results(args.outdir.resolve())
        print(result["replay_sha256"])
    elif args.command == "replay-worker":
        result = captured_replay_worker(args.outdir.resolve())
        sys.stdout.buffer.write(_canonical_json(result))
        sys.stdout.buffer.flush()
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
