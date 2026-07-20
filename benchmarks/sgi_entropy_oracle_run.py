"""Frozen COMP-008 lifecycle for the mean-conditioned entropy oracle.

This module is intentionally benchmark-only.  It prepares the eight bound CLIC development
images, fits one shared StructSplat field per image, creates two independent QAT copies, and hands
the resulting ordinary SSPL1 streams to :mod:`benchmarks.sgi_entropy_oracle`.  It cannot name or
open a confirmation payload and it contains no implementation of the counterfactual SSP2E coder.

The lifecycle is fail-closed:

``preflight`` -> external acquisition -> ``prepare-dev`` -> ``run-dev`` -> ``analyze`` ->
``replay``.

Preflight must happen in an empty output directory.  It freezes the task, benchmark sources,
tests, all repository StructSplat Python/CUDA sources, the resolved scientific configuration, the
environment, the CUDA extension binary, and the exact bootstrap matrix before ``prepare-dev`` is
allowed to invoke Pillow.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import random
import shutil
import struct
import sys
import tarfile
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from benchmarks import clic2020_subset_acquire as acquisition
from benchmarks import sgi_entropy_oracle as oracle


PROTOCOL = "COMP-008-mean-conditioned-entropy-oracle-v1"
PREFLIGHT_SCHEMA = "structsplat.comp008.preflight.v1"
TARGETS_SCHEMA = "structsplat.comp008.targets.v1"
BASE_SCHEMA = "structsplat.comp008.base-field.v1"
CELL_SCHEMA = "structsplat.comp008.cell.v1"
ANALYSIS_RECORD_SCHEMA = "structsplat.comp008.analysis-record.v1"
REPLAY_RECORD_SCHEMA = "structsplat.comp008.lifecycle-replay.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = REPO_ROOT / "tasks" / "COMP-008-mean-conditioned-entropy-oracle.md"
ACQUISITION_PATH = REPO_ROOT / "benchmarks" / "clic2020_subset_acquire.py"
CORE_PATH = REPO_ROOT / "benchmarks" / "sgi_entropy_oracle.py"
RUNNER_PATH = Path(__file__).resolve()
ACQUISITION_TEST_PATH = REPO_ROOT / "tests" / "test_clic2020_subset_acquire.py"
CORE_TEST_PATH = REPO_ROOT / "tests" / "test_sgi_entropy_oracle.py"
RUNNER_TEST_PATH = REPO_ROOT / "tests" / "test_sgi_entropy_oracle_run.py"

PREFLIGHT_NAME = "preflight.json"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SOURCE_ARCHIVE_NAME = "source_snapshot.tar"
BOOTSTRAP_NAME = "bootstrap_indices_u8.bin"
TARGETS_NAME = "targets_manifest.json"
JOURNAL_NAME = "cells.jsonl"
ANALYSIS_NAME = "analysis.json"
ANALYSIS_RECORD_NAME = "analysis_record.json"
REPLAY_NAME = "replay.json"
RUN_INVOCATIONS_NAME = "run_invocations.jsonl"

MAX_SIDE = 768
GAUSSIAN_COUNT = 8192
FIT_ITERATIONS = 4000
QAT_ITERATIONS = 150
SEED = 0
DECODED_FIELD_ATOL = 1e-6
RENDER_DIAGNOSTIC_ATOL = 5e-4
RENDER_DIAGNOSTIC_RTOL = 5e-4
REQUIRED_LD_PRELOAD = Path("/lib/x86_64-linux-gnu/libstdc++.so.6")

DEVELOPMENT_NAMES = acquisition.DEVELOPMENT_NAMES
BIT_TUPLES = oracle.FROZEN_BIT_TUPLES


class LifecycleError(RuntimeError):
    """A frozen lifecycle, provenance, or integrity condition was violated."""


def canonical_json(value: Any) -> bytes:
    return oracle.canonical_json(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(record: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(record)
    result.pop(field, None)
    result[field] = sha256_bytes(canonical_json(result))
    return result


def _verify_seal(record: Mapping[str, Any], field: str) -> None:
    observed = record.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise LifecycleError(f"missing or invalid {field}")
    core = dict(record)
    core.pop(field, None)
    expected = sha256_bytes(canonical_json(core))
    if observed != expected:
        raise LifecycleError(f"{field} mismatch: {observed} != {expected}")


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read canonical JSON object {path}") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"{path} must contain one JSON object")
    if canonical_json(value) != path.read_bytes():
        raise LifecycleError(f"{path} is not canonical JSON")
    return value


def _exclusive_write(path: Path, payload: bytes, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise LifecycleError(f"refusing to replace existing artifact: {path}")
        os.replace(temporary, path)
        if readonly:
            path.chmod(0o444)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(record))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise LifecycleError(f"truncated append-only journal at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifecycleError(f"invalid journal JSON at line {line_number}") from exc
        if not isinstance(value, dict) or canonical_json(value) != line:
            raise LifecycleError(f"non-canonical journal record at line {line_number}")
        records.append(value)
    return records


def _repair_truncated_jsonl_tail(path: Path) -> dict[str, Any] | None:
    """Discard only an uncommitted final JSONL fragment left by a killed process."""
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise LifecycleError(f"unsafe append-only journal: {path}")
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return None
    boundary = payload.rfind(b"\n") + 1
    committed = payload[:boundary]
    for line_number, line in enumerate(committed.splitlines(keepends=True), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifecycleError(f"invalid committed journal JSON at line {line_number}") from exc
        if not isinstance(value, dict) or canonical_json(value) != line:
            raise LifecycleError(f"non-canonical committed journal record at line {line_number}")
    fragment = payload[boundary:]
    with path.open("r+b") as handle:
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "journal": path.name,
        "committed_bytes": boundary,
        "discarded_fragment_bytes": len(fragment),
        "discarded_fragment_sha256": sha256_bytes(fragment),
        "repair_utc": _utc_now(),
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _process_stage_record(started_utc: str, started_perf: float) -> dict[str, Any]:
    return {
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started_perf,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "working_directory": str(Path.cwd().resolve()),
        "python_executable": str(Path(sys.executable).resolve()),
        "invocation_argv": [str(value) for value in sys.argv],
    }


def _verify_process_stage_record(value: Any, *, stage: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "started_utc",
        "ended_utc",
        "wall_seconds",
        "pid",
        "parent_pid",
        "working_directory",
        "python_executable",
        "invocation_argv",
    }:
        raise LifecycleError(f"{stage} process provenance has an invalid schema")
    parsed_times = []
    for key in ("started_utc", "ended_utc"):
        try:
            parsed_times.append(time.strptime(value[key], "%Y-%m-%dT%H:%M:%SZ"))
        except (TypeError, ValueError) as exc:
            raise LifecycleError(f"{stage} process provenance has an invalid {key}") from exc
    if parsed_times[1] < parsed_times[0]:
        raise LifecycleError(f"{stage} process provenance ends before it starts")
    wall = value["wall_seconds"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not np.isfinite(wall)
        or wall < 0
    ):
        raise LifecycleError(f"{stage} process provenance has an invalid wall time")
    for key in ("pid", "parent_pid"):
        observed = value[key]
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise LifecycleError(f"{stage} process provenance has an invalid {key}")
    if value["pid"] == 0:
        raise LifecycleError(f"{stage} process provenance has an invalid pid")
    for key in ("working_directory", "python_executable"):
        observed = value[key]
        if not isinstance(observed, str) or not Path(observed).is_absolute():
            raise LifecycleError(f"{stage} process provenance has a non-absolute {key}")
    argv = value["invocation_argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
        raise LifecycleError(f"{stage} process provenance has an invalid argv")
    return value


def _assert_regular_files(directory: Path, allowed: set[str], *, exact: bool) -> None:
    observed = {path.name for path in directory.iterdir()}
    if (exact and observed != allowed) or (not exact and not observed <= allowed):
        raise LifecycleError(
            f"unexpected artifact inventory in {directory}: observed={sorted(observed)}"
        )
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            raise LifecycleError(f"unsafe or non-file artifact in {directory}: {path.name}")


def _assert_known_artifact_layout(outdir: Path, *, allow_staging: bool) -> None:
    """Reject every path outside the stage-aware COMP-008 artifact grammar."""
    if not outdir.is_dir() or outdir.is_symlink():
        raise LifecycleError(f"unsafe COMP-008 artifact root: {outdir}")
    top_files = {
        PREFLIGHT_NAME,
        SOURCE_MANIFEST_NAME,
        SOURCE_ARCHIVE_NAME,
        BOOTSTRAP_NAME,
        TARGETS_NAME,
        JOURNAL_NAME,
        RUN_INVOCATIONS_NAME,
        ANALYSIS_NAME,
        ANALYSIS_RECORD_NAME,
        REPLAY_NAME,
    }
    top_directories = {"targets", "fields", "cells"}
    for path in outdir.iterdir():
        if path.name in top_files:
            if not path.is_file() or path.is_symlink():
                raise LifecycleError(f"unsafe top-level COMP-008 artifact: {path}")
        elif path.name in top_directories:
            if not path.is_dir() or path.is_symlink():
                raise LifecycleError(f"unsafe top-level COMP-008 directory: {path}")
        else:
            raise LifecycleError(f"unexpected top-level COMP-008 artifact: {path.name}")

    targets = outdir / "targets"
    if targets.exists():
        _assert_regular_files(targets, {f"{name}.png" for name in DEVELOPMENT_NAMES}, exact=True)

    base_files = {"initial.npz", "base.npz", "fit_history.json", "base_record.json"}
    base_staging_files = base_files | {f"{name}.part" for name in base_files}
    fields = outdir / "fields"
    if fields.exists():
        allowed = set(DEVELOPMENT_NAMES)
        if allow_staging:
            allowed |= {f"{name}.part" for name in DEVELOPMENT_NAMES}
        for path in fields.iterdir():
            if path.name not in allowed or not path.is_dir() or path.is_symlink():
                raise LifecycleError(f"unexpected base-field artifact: {path}")
            staging = path.name.endswith(".part")
            _assert_regular_files(
                path,
                base_staging_files if staging else base_files,
                exact=not staging,
            )

    cell_files = {
        "qat_start.npz",
        "qat_terminal.npz",
        "stream.sspl1",
        "absolute_symbols.npz",
        "cell_record.json",
    }
    cell_staging_files = cell_files | {f"{name}.part" for name in cell_files}
    cells = outdir / "cells"
    if cells.exists():
        labels = {_bit_label(bits) for bits in BIT_TUPLES}
        allowed_labels = labels | (
            {f"{label}.part" for label in labels} if allow_staging else set()
        )
        for image_directory in cells.iterdir():
            if (
                image_directory.name not in DEVELOPMENT_NAMES
                or not image_directory.is_dir()
                or image_directory.is_symlink()
            ):
                raise LifecycleError(f"unexpected cell-image artifact: {image_directory}")
            for path in image_directory.iterdir():
                if path.name not in allowed_labels or not path.is_dir() or path.is_symlink():
                    raise LifecycleError(f"unexpected cell artifact: {path}")
                staging = path.name.endswith(".part")
                _assert_regular_files(
                    path,
                    cell_staging_files if staging else cell_files,
                    exact=not staging,
                )


def _remove_unsealed_staging_directories(outdir: Path) -> list[dict[str, Any]]:
    removed: list[dict[str, Any]] = []
    candidates = [
        *(outdir / "fields" / f"{name}.part" for name in DEVELOPMENT_NAMES),
        *(
            outdir / "cells" / name / f"{_bit_label(bits)}.part"
            for name in DEVELOPMENT_NAMES
            for bits in BIT_TUPLES
        ),
    ]
    for path in candidates:
        if not path.exists():
            continue
        if not path.is_dir() or path.is_symlink():
            raise LifecycleError(f"unsafe unsealed staging artifact: {path}")
        entry_count = sum(1 for _ in path.iterdir())
        shutil.rmtree(path)
        removed.append(
            {
                "path": path.relative_to(outdir).as_posix(),
                "discarded_entries": entry_count,
                "repair_utc": _utc_now(),
            }
        )
    return removed


def _relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise LifecycleError(f"source path escapes repository: {path}") from exc
    if path.is_symlink():
        raise LifecycleError(f"source path may not be a symlink: {path}")
    return relative.as_posix()


def source_paths() -> tuple[Path, ...]:
    """Return a conservative closure containing every repository production source.

    The complete ``src/structsplat`` tree is deliberately bound instead of attempting a fragile
    dynamic-import trace.  This is an over-approximation of the transitive Python/CUDA closure and
    therefore cannot omit a lazily imported renderer, metric, initializer, or fitter dependency.
    """
    required = [
        TASK_PATH,
        ACQUISITION_PATH,
        CORE_PATH,
        RUNNER_PATH,
        ACQUISITION_TEST_PATH,
        CORE_TEST_PATH,
        RUNNER_TEST_PATH,
        REPO_ROOT / "benchmarks" / "__init__.py",
        REPO_ROOT / "pyproject.toml",
    ]
    production = [
        path
        for path in (REPO_ROOT / "src" / "structsplat").rglob("*")
        if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".h", ".cuh"}
    ]
    paths = sorted({path.resolve(strict=True) for path in required + production}, key=_relative)
    return tuple(paths)


def _source_manifest(paths: Iterable[Path]) -> dict[str, Any]:
    records = []
    for path in paths:
        records.append(
            {
                "path": _relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return _seal(
        {
            "schema": "structsplat.comp008.source-manifest.v1",
            "closure_policy": "all-repository-structsplat-python-and-cuda-plus-protocol",
            "files": records,
        },
        "manifest_sha256",
    )


def _write_source_archive(path: Path, manifest: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".part")
    try:
        with tarfile.open(temporary, "x", format=tarfile.USTAR_FORMAT) as archive:
            entries: list[tuple[str, bytes]] = [
                (SOURCE_MANIFEST_NAME, canonical_json(dict(manifest)))
            ]
            for record in manifest["files"]:
                source = REPO_ROOT / str(record["path"])
                entries.append((f"sources/{record['path']}", source.read_bytes()))
            for name, payload in entries:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o444
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
        if path.exists():
            raise LifecycleError(f"source archive already exists: {path}")
        os.replace(temporary, path)
        path.chmod(0o444)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _set_runtime_controls() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this only before inter-op work begins.  The observed value is bound and
        # subsequently checked, so a process that initialized it differently still fails closed.
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _seed_everything() -> None:
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            result[str(name).lower().replace("_", "-")] = str(distribution.version)
    return dict(sorted(result.items()))


def environment_record() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise LifecycleError("COMP-008 requires a CUDA device")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "packages": _installed_versions(),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "cuda_device": {
            "index": device,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "total_memory": int(properties.total_memory),
            "major": int(properties.major),
            "minor": int(properties.minor),
            "multi_processor_count": int(properties.multi_processor_count),
        },
        "environment_variables": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
                "TORCH_EXTENSIONS_DIR",
            )
        },
    }


def runtime_abi_record() -> dict[str, Any]:
    """Bind byte order and libraries that define packed symbols and canonical zlib bytes."""
    if sys.byteorder != "little" or not np.dtype("<u2").isnative:
        raise LifecycleError("COMP-008 SSPL1 execution requires a little-endian host")
    required = REQUIRED_LD_PRELOAD.resolve(strict=True)
    raw_preload = os.environ.get("LD_PRELOAD")
    try:
        observed_preloads = [
            Path(value).resolve(strict=True) for value in (raw_preload or "").split(":") if value
        ]
    except OSError as exc:
        raise LifecycleError("COMP-008 LD_PRELOAD contains a missing library") from exc
    if observed_preloads != [required]:
        raise LifecycleError(
            f"COMP-008 requires the exact sole LD_PRELOAD {REQUIRED_LD_PRELOAD}, "
            f"got {raw_preload!r}"
        )
    return {
        "sys_byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "numpy_little_endian_u16_is_native": bool(np.dtype("<u2").isnative),
        "zlib_compile_version": zlib.ZLIB_VERSION,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        "libstdcxx_preload": {
            "raw": raw_preload,
            "resolved_path": str(required),
            "bytes": required.stat().st_size,
            "sha256": sha256_file(required),
        },
    }


def module_origin_record() -> dict[str, str]:
    """Prove that execution imports the source tree captured by the preflight snapshot."""
    expected = {
        "structsplat": REPO_ROOT / "src" / "structsplat" / "__init__.py",
        "structsplat.codec": REPO_ROOT / "src" / "structsplat" / "codec.py",
        "structsplat.config": REPO_ROOT / "src" / "structsplat" / "config.py",
        "structsplat.cuda_render": REPO_ROOT / "src" / "structsplat" / "cuda_render.py",
        "structsplat.fit": REPO_ROOT / "src" / "structsplat" / "fit.py",
        "structsplat.gaussians": REPO_ROOT / "src" / "structsplat" / "gaussians.py",
        "structsplat.init": REPO_ROOT / "src" / "structsplat" / "init.py",
        "structsplat.metrics": REPO_ROOT / "src" / "structsplat" / "metrics.py",
        "structsplat.render": REPO_ROOT / "src" / "structsplat" / "render.py",
    }
    observed: dict[str, str] = {}
    for name, expected_path in expected.items():
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            raise LifecycleError(f"bound module has no filesystem origin: {name}")
        resolved = Path(origin).resolve(strict=True)
        if resolved != expected_path.resolve(strict=True):
            raise LifecycleError(
                f"bound module origin escaped the captured repository source: {name}: {resolved}"
            )
        observed[name] = str(resolved)
    return observed


def _prebuild_cuda() -> dict[str, Any]:
    """Build and execute the owned exact CUDA forward/backward on synthetic tensors only."""
    import torch

    from structsplat.cuda_render import _load_extension
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    _set_runtime_controls()
    _seed_everything()
    extension = _load_extension()
    binary = Path(str(extension.__file__)).resolve(strict=True)
    field = GaussianField(
        means=torch.tensor([[1.25, 1.75], [2.5, 2.25]], device="cuda"),
        log_scales=torch.log(torch.tensor([[0.9, 1.1], [1.2, 0.8]], device="cuda")),
        rotations=torch.tensor([0.2, 0.7], device="cuda"),
        colors=torch.tensor([[0.2, 0.4, 0.7], [0.8, 0.3, 0.1]], device="cuda"),
    ).trainable()
    image = render_field(
        field.means,
        field.conics(0.0),
        field.colors,
        field.radii(3.0, 0.0),
        4,
        5,
        512,
        "cuda",
        field.opacity_values(),
        scales=field.scales(),
        rotations=field.rotations,
        support_fade=False,
        sigma_cutoff=3.0,
    )
    image.square().mean().backward()
    torch.cuda.synchronize()
    digest = hashlib.sha256()
    digest.update(image.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    for tensor in (
        field.means.grad,
        field.log_scales.grad,
        field.rotations.grad,
        field.colors.grad,
    ):
        if tensor is None:
            raise LifecycleError("synthetic CUDA prebuild omitted a required gradient")
        digest.update(tensor.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    return {
        "extension_binary": str(binary),
        "extension_binary_bytes": binary.stat().st_size,
        "extension_binary_sha256": sha256_file(binary),
        "synthetic_forward_backward_sha256": digest.hexdigest(),
        "renderer": "cuda",
        "scientific_pixels_accessed": False,
    }


def synthetic_self_checks() -> dict[str, Any]:
    """Exercise the real codec/parser/decode boundary on N=8192 synthetic fields only."""
    from structsplat import codec
    from structsplat.config import FitConfig
    from structsplat.gaussians import GaussianField

    n = GAUSSIAN_COUNT
    index = np.arange(n, dtype=np.float32)
    means = np.column_stack([np.remainder(index * 7.0, 15.0), np.remainder(index * 11.0, 767.0)])
    scales = np.column_stack(
        [0.6 + np.remainder(index, 13.0) / 20.0, 0.7 + np.remainder(index, 17.0) / 25.0]
    )
    rotations = np.remainder(index * 0.17, np.pi)
    colors = np.column_stack(
        [
            np.remainder(index, 251.0) / 250.0,
            np.remainder(index * 3.0, 241.0) / 240.0,
            np.remainder(index * 5.0, 239.0) / 238.0,
        ]
    )
    fit_config = FitConfig(renderer="cuda", support_fade=False)
    checks = []
    for tuple_index, bits in enumerate(BIT_TUPLES):
        field = GaussianField.from_numpy(means, scales, rotations, colors)
        config = codec.CodecConfig(
            bits_means=bits[0],
            bits_scales=bits[1],
            bits_rot=bits[2],
            bits_colors=bits[3],
            morton_reorder=True,
            color_delta=True,
            means_byte_planar=True,
            rot_circular=True,
            stream_codec="zlib",
            zlib_level=9,
        )
        blob = codec.encode(field, 768, 16, config, fit_config)
        image_id = DEVELOPMENT_NAMES[tuple_index]
        parsed = oracle.parse_sspl1(blob)
        row = oracle.make_oracle_row(blob, image_id)
        integrity = _decoded_integrity(blob)
        if (
            parsed.total_bytes != len(blob)
            or tuple(int(value) for value in parsed.header["bits"]) != bits
            or not oracle.replay_row(blob, row)["row_exact"]
            or not integrity["boundary_state_hash_exact"]
        ):
            raise LifecycleError("synthetic N8192 codec/parser/decode self-check failed")
        checks.append(
            {
                "bit_tuple": list(bits),
                "n": n,
                "height": 768,
                "width": 16,
                "sspl1_bytes": len(blob),
                "sspl1_sha256": parsed.blob_sha256,
                "symbol_sha256": parsed.symbol_sha256,
                "oracle_row_sha256": row["record_sha256"],
                "decoded_state_sha256": integrity["decoded_state_sha256"],
                "boundary_state_hash_exact": integrity["boundary_state_hash_exact"],
            }
        )
    return {
        "schema": "structsplat.comp008.synthetic-self-check.v1",
        "scientific_pixels_accessed": False,
        "field_source": "closed-form deterministic synthetic tensors",
        "checks": checks,
        "ok": True,
    }


def resolved_configs() -> dict[str, Any]:
    from structsplat.codec import CodecConfig
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig

    init_config = InitConfig(
        strategy="quadtree_wse",
        num_gaussians=GAUSSIAN_COUNT,
        seed=SEED,
        opacity_mode="none",
        color_mode="bilinear",
        scale_cap_mode="none",
        background_fraction=0.0,
        background_grid=0,
    )
    fit_config = FitConfig(
        iters=FIT_ITERATIONS,
        renderer="cuda",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=100,
        checkpoint_policy="terminal",
        compute_lpips=False,
        color_basis="constant",
        qat_mode="off",
        adaptive_count=False,
        covariance_filter_mode="none",
        support_fade=False,
        early_stop_patience=None,
        prune_every=None,
        split_every=None,
        split_count=0,
        relocate_every=None,
        relocate_at_split=False,
        relocate_count=0,
    )
    codec_configs = []
    for bits in BIT_TUPLES:
        codec_configs.append(
            asdict(
                CodecConfig(
                    bits_means=bits[0],
                    bits_scales=bits[1],
                    bits_rot=bits[2],
                    bits_colors=bits[3],
                    morton_reorder=True,
                    color_delta=True,
                    means_byte_planar=True,
                    rot_circular=True,
                    stream_codec="zlib",
                    zlib_level=9,
                )
            )
        )
    return {
        "development_names": list(DEVELOPMENT_NAMES),
        "confirmation_names_metadata_only": list(acquisition.CONFIRMATION_NAMES),
        "max_side": MAX_SIDE,
        "gaussian_count": GAUSSIAN_COUNT,
        "fit_iterations": FIT_ITERATIONS,
        "qat_iterations": QAT_ITERATIONS,
        "seed": SEED,
        "renderer": "cuda",
        "cpu_reference_renderer": "normalized",
        "decoded_field_max_abs_atol": DECODED_FIELD_ATOL,
        "cpu_reference_render_is_diagnostic_only": True,
        "render_diagnostic_atol": RENDER_DIAGNOSTIC_ATOL,
        "render_diagnostic_rtol": RENDER_DIAGNOSTIC_RTOL,
        "init": asdict(init_config),
        "structure_tensor": asdict(StructureTensorConfig()),
        "fit": asdict(fit_config),
        "codec_arms": codec_configs,
    }


def preflight(
    outdir: str | Path,
    *,
    prebuild_fn: Callable[[], Mapping[str, Any]] = _prebuild_cuda,
    environment_fn: Callable[[], Mapping[str, Any]] = environment_record,
) -> dict[str, Any]:
    started_utc = _utc_now()
    started_perf = time.perf_counter()
    destination = Path(outdir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise LifecycleError(f"preflight output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _set_runtime_controls()

    paths = source_paths()
    source_manifest = _source_manifest(paths)
    _exclusive_write(
        destination / SOURCE_MANIFEST_NAME,
        canonical_json(source_manifest),
        readonly=True,
    )
    _write_source_archive(destination / SOURCE_ARCHIVE_NAME, source_manifest)

    matrix = oracle.bootstrap_index_matrix()
    matrix_payload = matrix.tobytes(order="C")
    if sha256_bytes(matrix_payload) != oracle.BOOTSTRAP_MATRIX_SHA256:
        raise LifecycleError("frozen bootstrap matrix hash drift")
    _exclusive_write(destination / BOOTSTRAP_NAME, matrix_payload, readonly=True)

    environment = dict(environment_fn())
    runtime_abi = runtime_abi_record()
    cuda = dict(prebuild_fn())
    self_checks = synthetic_self_checks()
    record = _seal(
        {
            "schema": PREFLIGHT_SCHEMA,
            "protocol": PROTOCOL,
            "stage": "preflight",
            "created_utc": _utc_now(),
            "scientific_pixels_accessed": False,
            "source_manifest_path": SOURCE_MANIFEST_NAME,
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "source_archive_path": SOURCE_ARCHIVE_NAME,
            "source_archive_sha256": sha256_file(destination / SOURCE_ARCHIVE_NAME),
            "bootstrap": {
                "path": BOOTSTRAP_NAME,
                "dtype": "uint8",
                "shape": list(matrix.shape),
                "order": "C",
                "bytes": len(matrix_payload),
                "sha256": oracle.BOOTSTRAP_MATRIX_SHA256,
            },
            "remote_object": {
                "url": acquisition.FROZEN_URL,
                "content_length": acquisition.EXPECTED_CONTENT_LENGTH,
                "gcs_generation": acquisition.EXPECTED_GCS_GENERATION,
                "remotezip_version": acquisition.REMOTEZIP_VERSION,
            },
            "resolved_configs": resolved_configs(),
            "environment": environment,
            "runtime_abi": runtime_abi,
            "module_origins": module_origin_record(),
            "cuda_prebuild": cuda,
            "synthetic_self_checks": self_checks,
            "stage_process": _process_stage_record(started_utc, started_perf),
            "commands": {
                "preflight": (
                    "LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 "
                    "python -m benchmarks.sgi_entropy_oracle_run preflight --outdir OUT"
                ),
                "acquire": (
                    "LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 "
                    "python -m benchmarks.clic2020_subset_acquire --outdir ACQUISITION"
                ),
                "prepare_dev": (
                    "LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 "
                    "python -m benchmarks.sgi_entropy_oracle_run prepare-dev --outdir OUT "
                    "--acquisition-dir ACQUISITION"
                ),
                "run_dev": (
                    "LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 "
                    "python -m benchmarks.sgi_entropy_oracle_run run-dev --outdir OUT"
                ),
                "analyze": (
                    "LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 "
                    "python -m benchmarks.sgi_entropy_oracle_run analyze --outdir OUT"
                ),
                "replay": (
                    "LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 "
                    "python -m benchmarks.sgi_entropy_oracle_run replay --outdir OUT"
                ),
            },
        },
        "binding_sha256",
    )
    _exclusive_write(destination / PREFLIGHT_NAME, canonical_json(record), readonly=True)
    return record


def _verify_source_archive(outdir: Path, preflight_record: Mapping[str, Any]) -> None:
    manifest_path = outdir / str(preflight_record["source_manifest_path"])
    manifest = _read_object(manifest_path)
    _verify_seal(manifest, "manifest_sha256")
    if manifest["manifest_sha256"] != preflight_record["source_manifest_sha256"]:
        raise LifecycleError("source manifest binding drift")
    for item in manifest.get("files", []):
        path = REPO_ROOT / str(item["path"])
        if not path.is_file() or path.is_symlink():
            raise LifecycleError(f"bound source is missing or unsafe: {path}")
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise LifecycleError(f"bound source changed after preflight: {item['path']}")
    archive = outdir / str(preflight_record["source_archive_path"])
    if sha256_file(archive) != preflight_record["source_archive_sha256"]:
        raise LifecycleError("source snapshot archive drift")


def verify_preflight(
    outdir: str | Path,
    *,
    environment_fn: Callable[[], Mapping[str, Any]] = environment_record,
) -> dict[str, Any]:
    root = Path(outdir)
    record = _read_object(root / PREFLIGHT_NAME)
    _verify_seal(record, "binding_sha256")
    if record.get("schema") != PREFLIGHT_SCHEMA or record.get("protocol") != PROTOCOL:
        raise LifecycleError("unexpected COMP-008 preflight protocol")
    _verify_source_archive(root, record)
    bootstrap = record["bootstrap"]
    payload = (root / str(bootstrap["path"])).read_bytes()
    if (
        len(payload) != bootstrap["bytes"]
        or sha256_bytes(payload) != bootstrap["sha256"]
        or bootstrap["sha256"] != oracle.BOOTSTRAP_MATRIX_SHA256
    ):
        raise LifecycleError("persisted bootstrap matrix drift")
    if record["resolved_configs"] != resolved_configs():
        raise LifecycleError("resolved scientific configuration drift")
    if record.get("runtime_abi") != runtime_abi_record():
        raise LifecycleError("little-endian/NumPy/zlib runtime ABI drift")
    if record.get("module_origins") != module_origin_record():
        raise LifecycleError("captured repository module-origin binding drift")
    _set_runtime_controls()
    if record["environment"] != dict(environment_fn()):
        raise LifecycleError("bound process/device environment drift")
    cuda = record["cuda_prebuild"]
    binary = Path(str(cuda["extension_binary"]))
    if (
        not binary.is_file()
        or binary.stat().st_size != cuda["extension_binary_bytes"]
        or sha256_file(binary) != cuda["extension_binary_sha256"]
    ):
        raise LifecycleError("prebuilt exact-CUDA extension binary drift")
    _assert_known_artifact_layout(root, allow_staging=True)
    return record


def verify_acquisition(acquisition_dir: str | Path) -> dict[str, Any]:
    root = Path(acquisition_dir)
    if not root.is_dir() or root.is_symlink():
        raise LifecycleError(f"unsafe acquisition directory: {root}")
    expected_files = {
        *(f"{name}.png" for name in DEVELOPMENT_NAMES),
        acquisition.MANIFEST_NAME,
        acquisition.MANIFEST_HASH_NAME,
    }
    observed_files = {path.name for path in root.iterdir()}
    if observed_files != expected_files:
        raise LifecycleError(
            "acquisition directory must contain only the eight development PNGs and manifest; "
            f"observed {sorted(observed_files)}"
        )
    for path in root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise LifecycleError(f"unsafe acquisition artifact: {path}")

    manifest_path = root / acquisition.MANIFEST_NAME
    manifest = _read_object(manifest_path)
    payload_hash = manifest.get("manifest_payload_sha256")
    core = dict(manifest)
    core.pop("manifest_payload_sha256", None)
    if payload_hash != sha256_bytes(canonical_json(core)):
        raise LifecycleError("acquisition manifest payload self-hash mismatch")
    sidecar = (root / acquisition.MANIFEST_HASH_NAME).read_text(encoding="ascii")
    expected_sidecar = f"{sha256_file(manifest_path)}  {acquisition.MANIFEST_NAME}\n"
    if sidecar != expected_sidecar:
        raise LifecycleError("acquisition manifest file-hash sidecar mismatch")
    if manifest.get("schema") != acquisition.MANIFEST_SCHEMA:
        raise LifecycleError("unexpected acquisition manifest schema")
    _verify_process_stage_record(manifest.get("stage_process"), stage="acquisition")
    if tuple(manifest.get("development_names", ())) != DEVELOPMENT_NAMES:
        raise LifecycleError("development-name binding drift")
    if tuple(manifest.get("confirmation_names", ())) != acquisition.CONFIRMATION_NAMES:
        raise LifecycleError("confirmation-name metadata binding drift")
    if manifest.get("remotezip_version") != acquisition.REMOTEZIP_VERSION:
        raise LifecycleError("remotezip version drift")
    before = manifest.get("archive")
    after = manifest.get("post_acquisition_head")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise LifecycleError("acquisition remote-object binding is missing")
    acquisition.assert_same_remote_object(before, after)
    if (
        before.get("requested_url") != acquisition.FROZEN_URL
        or before.get("content_length") != acquisition.EXPECTED_CONTENT_LENGTH
        or before.get("gcs_generation") != acquisition.EXPECTED_GCS_GENERATION
    ):
        raise LifecycleError("acquisition remote-object identity drift")
    archive_index = manifest.get("archive_index")
    if not isinstance(archive_index, list) or manifest.get(
        "archive_index_payload_sha256"
    ) != sha256_bytes(canonical_json(archive_index)):
        raise LifecycleError("archive central-directory binding drift")
    if not isinstance(manifest.get("range_requests"), list) or not manifest["range_requests"]:
        raise LifecycleError("acquisition contains no audited HTTP Range requests")

    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(acquisition.FROZEN_NAMES):
        raise LifecycleError("acquisition member set is incomplete")
    by_name: dict[str, dict[str, Any]] = {}
    for raw in members:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise LifecycleError("invalid acquisition member record")
        if raw["name"] in by_name:
            raise LifecycleError(f"duplicate acquisition member: {raw['name']}")
        by_name[raw["name"]] = raw
    if tuple(by_name) != acquisition.FROZEN_NAMES:
        raise LifecycleError("acquisition members are not in frozen order")
    for name in DEVELOPMENT_NAMES:
        record = by_name[name]
        if (
            record.get("partition") != "development"
            or record.get("payload_extracted") is not True
            or record.get("local_path") != f"{name}.png"
        ):
            raise LifecycleError(f"invalid development acquisition record for {name}")
        path = root / f"{name}.png"
        if path.stat().st_size != record.get("byte_size") or sha256_file(path) != record.get(
            "sha256"
        ):
            raise LifecycleError(f"encoded development PNG drift: {name}")
    for name in acquisition.CONFIRMATION_NAMES:
        record = by_name[name]
        if (
            record.get("partition") != "confirmation"
            or record.get("payload_extracted") is not False
            or record.get("local_path") is not None
            or any(key in record for key in ("byte_size", "sha256", "download_crc32"))
            or (root / f"{name}.png").exists()
        ):
            raise LifecycleError(f"confirmation payload is not sealed: {name}")
    return manifest


def _pixel_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise LifecycleError("pixel hash requires contiguous HxWx3 RGB uint8")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def prepare_rgb_png(source: Path, destination: Path, *, max_side: int = MAX_SIDE) -> dict[str, Any]:
    """Decode one already-authorized development PNG and persist the exact RGB derivative."""
    from PIL import Image

    if max_side <= 0:
        raise LifecycleError("max_side must be positive")
    with Image.open(source) as image:
        if image.format != "PNG" or bool(getattr(image, "is_animated", False)):
            raise LifecycleError(f"development source is not one static PNG: {source}")
        image.load()
        source_mode = image.mode
        rgb_image = image.convert("RGB")
        native = np.asarray(rgb_image, dtype=np.uint8)
    if native.ndim != 3 or native.shape[2] != 3:
        raise LifecycleError(f"strict RGB conversion failed for {source}")
    height, width = (int(native.shape[0]), int(native.shape[1]))
    if height <= 0 or width <= 0:
        raise LifecycleError(f"empty development PNG: {source}")
    scale = max_side / max(height, width)
    derived_height = max(1, round(height * scale))
    derived_width = max(1, round(width * scale))
    with Image.fromarray(native, mode="RGB") as rgb_image:
        resized = rgb_image.resize(
            (derived_width, derived_height),
            resample=Image.Resampling.LANCZOS,
        )
        derived = np.asarray(resized, dtype=np.uint8).copy()
    if max(derived.shape[:2]) != max_side:
        raise LifecycleError("frozen resize rule did not produce the exact max side")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.fromarray(derived, mode="RGB") as output:
        output.save(destination, format="PNG", optimize=False, compress_level=9)
    with Image.open(destination) as check:
        check.load()
        persisted = np.asarray(check.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(persisted, derived):
        raise LifecycleError("derived PNG pixel round trip drift")
    return {
        "source_mode": source_mode,
        "source_encoded_bytes": source.stat().st_size,
        "source_encoded_sha256": sha256_file(source),
        "native_rgb_dimensions_hw": [height, width],
        "native_rgb_pixel_sha256": _pixel_sha256(native),
        "resize_scale": scale,
        "derived_path": destination.as_posix(),
        "derived_encoded_bytes": destination.stat().st_size,
        "derived_encoded_sha256": sha256_file(destination),
        "derived_rgb_dimensions_hw": [derived_height, derived_width],
        "derived_rgb_pixel_sha256": _pixel_sha256(derived),
    }


def prepare_development(
    outdir: str | Path,
    acquisition_dir: str | Path,
    *,
    environment_fn: Callable[[], Mapping[str, Any]] = environment_record,
) -> dict[str, Any]:
    started_utc = _utc_now()
    started_perf = time.perf_counter()
    root = Path(outdir)
    preflight_record = verify_preflight(root, environment_fn=environment_fn)
    _assert_known_artifact_layout(root, allow_staging=False)
    acquired = verify_acquisition(acquisition_dir)
    destination = root / "targets"
    if destination.exists() or (root / TARGETS_NAME).exists():
        raise LifecycleError("prepare-dev artifacts already exist")
    temporary = root / "targets.part"
    if temporary.exists():
        raise LifecycleError("stale prepare-dev staging directory exists")
    temporary.mkdir()
    records = []
    try:
        by_name = {record["name"]: record for record in acquired["members"]}
        for name in DEVELOPMENT_NAMES:
            # Name authorization occurs before constructing or opening a local payload path.
            if name not in DEVELOPMENT_NAMES:
                raise AssertionError("unreachable non-development name")
            source = Path(acquisition_dir) / str(by_name[name]["local_path"])
            derived = temporary / f"{name}.png"
            record = prepare_rgb_png(source, derived)
            record["name"] = name
            record["derived_path"] = f"targets/{name}.png"
            if record["source_encoded_sha256"] != by_name[name]["sha256"]:
                raise LifecycleError(f"development source changed during decode: {name}")
            records.append(record)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    for path in destination.iterdir():
        path.chmod(0o444)
    manifest = _seal(
        {
            "schema": TARGETS_SCHEMA,
            "protocol": PROTOCOL,
            "stage": "prepare-dev",
            "preflight_binding_sha256": preflight_record["binding_sha256"],
            "acquisition_directory": str(Path(acquisition_dir).resolve()),
            "acquisition_manifest_file_sha256": sha256_file(
                Path(acquisition_dir) / acquisition.MANIFEST_NAME
            ),
            "acquisition_manifest_payload_sha256": acquired["manifest_payload_sha256"],
            "remote_object": preflight_record["remote_object"],
            "resize": {
                "mode": "RGB",
                "resampling": "Pillow.Image.Resampling.LANCZOS",
                "max_side": MAX_SIDE,
                "rule": "scale=768/max(H,W); round(H*scale), round(W*scale)",
                "pixel_hash": "sha256(int64_le_shape || contiguous_rgb8_bytes)",
            },
            "development_names": list(DEVELOPMENT_NAMES),
            "confirmation_payloads_accessed": False,
            "targets": records,
            "stage_process": _process_stage_record(started_utc, started_perf),
        },
        "targets_sha256",
    )
    _exclusive_write(root / TARGETS_NAME, canonical_json(manifest), readonly=True)
    return manifest


def _load_target_manifest(outdir: Path, *, decode_pixels: bool = True) -> dict[str, Any]:
    manifest = _read_object(outdir / TARGETS_NAME)
    _verify_seal(manifest, "targets_sha256")
    expected_manifest_keys = {
        "schema",
        "protocol",
        "stage",
        "preflight_binding_sha256",
        "acquisition_directory",
        "acquisition_manifest_file_sha256",
        "acquisition_manifest_payload_sha256",
        "remote_object",
        "resize",
        "development_names",
        "confirmation_payloads_accessed",
        "targets",
        "stage_process",
        "targets_sha256",
    }
    expected_remote = {
        "url": acquisition.FROZEN_URL,
        "content_length": acquisition.EXPECTED_CONTENT_LENGTH,
        "gcs_generation": acquisition.EXPECTED_GCS_GENERATION,
        "remotezip_version": acquisition.REMOTEZIP_VERSION,
    }
    expected_resize = {
        "mode": "RGB",
        "resampling": "Pillow.Image.Resampling.LANCZOS",
        "max_side": MAX_SIDE,
        "rule": "scale=768/max(H,W); round(H*scale), round(W*scale)",
        "pixel_hash": "sha256(int64_le_shape || contiguous_rgb8_bytes)",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema") != TARGETS_SCHEMA
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("stage") != "prepare-dev"
        or tuple(manifest.get("development_names", ())) != DEVELOPMENT_NAMES
        or manifest.get("confirmation_payloads_accessed") is not False
        or manifest.get("remote_object") != expected_remote
        or manifest.get("resize") != expected_resize
        or not isinstance(manifest.get("acquisition_directory"), str)
        or not Path(manifest["acquisition_directory"]).is_absolute()
        or any(
            not isinstance(manifest.get(key), str) or len(manifest[key]) != 64
            for key in (
                "preflight_binding_sha256",
                "acquisition_manifest_file_sha256",
                "acquisition_manifest_payload_sha256",
            )
        )
    ):
        raise LifecycleError("target manifest binding drift")
    _verify_process_stage_record(manifest.get("stage_process"), stage="prepare-dev")
    records = manifest.get("targets")
    if not isinstance(records, list) or [row.get("name") for row in records] != list(
        DEVELOPMENT_NAMES
    ):
        raise LifecycleError("target manifest is not the exact frozen development set")
    expected = {f"{name}.png" for name in DEVELOPMENT_NAMES}
    target_dir = outdir / "targets"
    if not target_dir.is_dir() or {path.name for path in target_dir.iterdir()} != expected:
        raise LifecycleError("derived target directory contains a foreign or missing artifact")
    expected_record_keys = {
        "source_mode",
        "source_encoded_bytes",
        "source_encoded_sha256",
        "native_rgb_dimensions_hw",
        "native_rgb_pixel_sha256",
        "resize_scale",
        "derived_path",
        "derived_encoded_bytes",
        "derived_encoded_sha256",
        "derived_rgb_dimensions_hw",
        "derived_rgb_pixel_sha256",
        "name",
    }
    for record in records:
        native_dimensions = record.get("native_rgb_dimensions_hw")
        derived_dimensions = record.get("derived_rgb_dimensions_hw")
        if (
            set(record) != expected_record_keys
            or record.get("derived_path") != f"targets/{record['name']}.png"
            or not isinstance(record.get("source_mode"), str)
            or not isinstance(native_dimensions, list)
            or len(native_dimensions) != 2
            or not isinstance(derived_dimensions, list)
            or len(derived_dimensions) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (*native_dimensions, *derived_dimensions)
            )
            or max(derived_dimensions) != MAX_SIDE
            or record.get("resize_scale") != MAX_SIDE / max(native_dimensions)
            or any(
                not isinstance(record.get(key), str) or len(record[key]) != 64
                for key in (
                    "source_encoded_sha256",
                    "native_rgb_pixel_sha256",
                    "derived_encoded_sha256",
                    "derived_rgb_pixel_sha256",
                )
            )
        ):
            raise LifecycleError(f"prepared target record schema drift: {record.get('name')}")
        path = outdir / str(record["derived_path"])
        if not path.is_file() or path.is_symlink():
            raise LifecycleError(f"unsafe derived target: {path}")
        if (
            path.stat().st_size != record["derived_encoded_bytes"]
            or sha256_file(path) != record["derived_encoded_sha256"]
        ):
            raise LifecycleError(f"derived target byte drift: {record['name']}")
        if decode_pixels:
            from PIL import Image

            with Image.open(path) as image:
                image.load()
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if (
                list(pixels.shape[:2]) != record["derived_rgb_dimensions_hw"]
                or _pixel_sha256(pixels) != record["derived_rgb_pixel_sha256"]
            ):
                raise LifecycleError(f"derived target pixel drift: {record['name']}")
        base_path = _base_record_path(outdir, str(record["name"]))
        if base_path.exists():
            base = _load_base_record(outdir, str(record["name"]))
            if (
                base["target_pixel_sha256"] != record["derived_rgb_pixel_sha256"]
                or base["dimensions_hw"] != record["derived_rgb_dimensions_hw"]
            ):
                raise LifecycleError(
                    f"base field targets a different prepared image: {record['name']}"
                )
    return manifest


def _array_state_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _field_arrays(field: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "opacities",
        "scale_max",
        "color_grads",
        "background_mask",
        "filter_variance",
    ):
        tensor = getattr(field, name)
        if tensor is not None:
            arrays[name] = tensor.detach().cpu().contiguous().numpy()
    return arrays


def field_state(field: Any) -> dict[str, Any]:
    arrays = _field_arrays(field)
    if not arrays or any(
        not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"
    ):
        raise LifecycleError("field state is empty or non-finite")
    return {
        "n": int(field.n),
        "present_arrays": sorted(arrays),
        "state_sha256": _array_state_digest(arrays),
    }


def _persist_field(field: Any, path: Path) -> dict[str, Any]:
    if path.exists():
        raise LifecycleError(f"field artifact already exists: {path}")
    field.save(str(path))
    path.chmod(0o444)
    from structsplat.gaussians import GaussianField

    loaded = GaussianField.load(str(path), device="cpu")
    state = field_state(loaded)
    if state != field_state(field):
        raise LifecycleError("persisted field state is not an exact round trip")
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **state,
    }


def _verify_field_artifact(outdir: Path, record: Mapping[str, Any], *, device: str = "cpu") -> Any:
    from structsplat.gaussians import GaussianField

    path = outdir / str(record["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record["bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise LifecycleError(f"field artifact drift: {path}")
    field = GaussianField.load(str(path), device=device)
    state = field_state(field)
    if state["state_sha256"] != record["state_sha256"] or state["n"] != record["n"]:
        raise LifecycleError(f"field state drift: {path}")
    return field


def _target_pixels(outdir: Path, record: Mapping[str, Any]) -> np.ndarray:
    from PIL import Image

    path = outdir / str(record["derived_path"])
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise LifecycleError(f"prepared target is not strict RGB PNG: {path}")
        image.load()
        pixels = np.asarray(image, dtype=np.uint8)
    if _pixel_sha256(pixels) != record["derived_rgb_pixel_sha256"]:
        raise LifecycleError(f"prepared target pixels drifted: {record['name']}")
    return pixels


def _make_configs() -> tuple[Any, Any, Any]:
    from structsplat.config import FitConfig, InitConfig, StructureTensorConfig

    frozen = resolved_configs()
    return (
        InitConfig(**frozen["init"]),
        StructureTensorConfig(**frozen["structure_tensor"]),
        FitConfig(**frozen["fit"]),
    )


def _base_directory(outdir: Path, image_id: str) -> Path:
    if image_id not in DEVELOPMENT_NAMES:
        raise LifecycleError(f"non-development image ID rejected: {image_id!r}")
    return outdir / "fields" / image_id


def _base_record_path(outdir: Path, image_id: str) -> Path:
    return _base_directory(outdir, image_id) / "base_record.json"


def _fit_base(outdir: Path, image_record: Mapping[str, Any], *, verbose: bool) -> dict[str, Any]:
    import torch

    from structsplat.fit import fit
    from structsplat.init import build_field

    stage_started_utc = _utc_now()
    stage_started_perf = time.perf_counter()
    image_id = str(image_record["name"])
    final_dir = _base_directory(outdir, image_id)
    if final_dir.exists():
        raise LifecycleError(
            f"base field directory already exists without valid record: {final_dir}"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.with_name(final_dir.name + ".part")
    staging_restarted = staging.exists()
    if staging_restarted:
        if not staging.is_dir() or staging.is_symlink():
            raise LifecycleError(f"unsafe base-field staging artifact: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()
    pixels = _target_pixels(outdir, image_record)
    image = pixels.astype(np.float32) / 255.0
    target = torch.as_tensor(image, dtype=torch.float32, device="cuda")
    init_config, tensor_config, fit_config = _make_configs()
    try:
        _seed_everything()
        started = time.perf_counter()
        initial = build_field(
            image,
            init_config,
            tensor_config,
            device="cuda",
        )
        init_seconds = time.perf_counter() - started
        if initial.n != GAUSSIAN_COUNT or initial.opacities is not None:
            raise LifecycleError("frozen initializer violated N/opacity contract")
        initial_artifact = _persist_field(initial, staging / "initial.npz")
        initial_artifact["path"] = f"fields/{image_id}/initial.npz"

        started = time.perf_counter()
        result = fit(initial, target, fit_config, verbose=verbose)
        torch.cuda.synchronize()
        fit_wall_seconds = time.perf_counter() - started
        if (
            result["iterations_run"] != FIT_ITERATIONS
            or result["stopped_early"]
            or result["selected_iter"] != FIT_ITERATIONS
            or result["selected_from_checkpoint"]
            or result["field"].n != GAUSSIAN_COUNT
        ):
            raise LifecycleError("ordinary fit did not return the frozen terminal N8192 state")
        base = result["field"]
        if (
            base.opacities is not None
            or base.color_grads is not None
            or base.filter_variance is not None
        ):
            raise LifecycleError("ordinary fit introduced a forbidden field component")
        base_artifact = _persist_field(base, staging / "base.npz")
        base_artifact["path"] = f"fields/{image_id}/base.npz"
        history = result["history"]
        history_record = {
            "iterations_run": result["iterations_run"],
            "selected_iter": result["selected_iter"],
            "selected_psnr": result["selected_psnr"],
            "fit_seconds_internal": result["fit_seconds"],
            "history": history,
        }
        _exclusive_write(
            staging / "fit_history.json", canonical_json(history_record), readonly=True
        )
        record = _seal(
            {
                "schema": BASE_SCHEMA,
                "protocol": PROTOCOL,
                "image_id": image_id,
                "target_pixel_sha256": image_record["derived_rgb_pixel_sha256"],
                "dimensions_hw": image_record["derived_rgb_dimensions_hw"],
                "init_config": asdict(init_config),
                "structure_tensor_config": asdict(tensor_config),
                "fit_config": asdict(fit_config),
                "initial_field": initial_artifact,
                "base_field": base_artifact,
                "fit_history": {
                    "path": f"fields/{image_id}/fit_history.json",
                    "bytes": (staging / "fit_history.json").stat().st_size,
                    "sha256": sha256_file(staging / "fit_history.json"),
                },
                "timings": {
                    "initialization_wall_seconds": init_seconds,
                    "ordinary_fit_wall_seconds": fit_wall_seconds,
                },
                "unsealed_staging_restarted_after_interruption": staging_restarted,
                "stage_process": _process_stage_record(stage_started_utc, stage_started_perf),
            },
            "base_record_sha256",
        )
        _exclusive_write(staging / "base_record.json", canonical_json(record), readonly=True)
        os.replace(staging, final_dir)
        return record
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_base_record(outdir: Path, image_id: str) -> dict[str, Any]:
    record = _read_object(_base_record_path(outdir, image_id))
    _verify_seal(record, "base_record_sha256")
    expected_keys = {
        "schema",
        "protocol",
        "image_id",
        "target_pixel_sha256",
        "dimensions_hw",
        "init_config",
        "structure_tensor_config",
        "fit_config",
        "initial_field",
        "base_field",
        "fit_history",
        "timings",
        "unsealed_staging_restarted_after_interruption",
        "stage_process",
        "base_record_sha256",
    }
    frozen = resolved_configs()
    dimensions = record.get("dimensions_hw")
    if (
        set(record) != expected_keys
        or record.get("schema") != BASE_SCHEMA
        or record.get("protocol") != PROTOCOL
        or record.get("image_id") != image_id
        or not isinstance(record.get("target_pixel_sha256"), str)
        or len(record["target_pixel_sha256"]) != 64
        or not isinstance(dimensions, list)
        or len(dimensions) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        )
        or max(dimensions) != MAX_SIDE
        or record.get("init_config") != frozen["init"]
        or record.get("structure_tensor_config") != frozen["structure_tensor"]
        or record.get("fit_config") != frozen["fit"]
        or not isinstance(record.get("unsealed_staging_restarted_after_interruption"), bool)
    ):
        raise LifecycleError(f"base-field record drift: {image_id}")
    if (
        record["initial_field"].get("path") != f"fields/{image_id}/initial.npz"
        or record["base_field"].get("path") != f"fields/{image_id}/base.npz"
        or record["fit_history"].get("path") != f"fields/{image_id}/fit_history.json"
    ):
        raise LifecycleError(f"base-field artifact path drift: {image_id}")
    initial = _verify_field_artifact(outdir, record["initial_field"])
    base = _verify_field_artifact(outdir, record["base_field"])
    if initial.n != GAUSSIAN_COUNT or base.n != GAUSSIAN_COUNT:
        raise LifecycleError(f"base-field Gaussian count drift: {image_id}")
    history = outdir / str(record["fit_history"]["path"])
    if (
        history.stat().st_size != record["fit_history"]["bytes"]
        or sha256_file(history) != record["fit_history"]["sha256"]
    ):
        raise LifecycleError(f"fit history drift: {image_id}")
    history_record = _read_object(history)
    trajectory = history_record.get("history")
    if (
        history_record.get("iterations_run") != FIT_ITERATIONS
        or history_record.get("selected_iter") != FIT_ITERATIONS
        or not isinstance(trajectory, dict)
        or not isinstance(trajectory.get("iter"), list)
        or not trajectory["iter"]
        or trajectory["iter"][-1] != FIT_ITERATIONS - 1
        or not isinstance(trajectory.get("n_gaussians"), list)
        or len(trajectory["n_gaussians"]) != len(trajectory["iter"])
        or any(value != GAUSSIAN_COUNT for value in trajectory["n_gaussians"])
    ):
        raise LifecycleError(f"fit history semantics drift: {image_id}")
    _verify_process_stage_record(record.get("stage_process"), stage="base-fit")
    return record


def _bit_label(bits: Sequence[int]) -> str:
    bit_tuple = tuple(int(value) for value in bits)
    if bit_tuple not in BIT_TUPLES:
        raise LifecycleError(f"unknown COMP-008 bit tuple: {bit_tuple}")
    return "m%d_s%d_r%d_c%d" % bit_tuple


def _cell_directory(outdir: Path, image_id: str, bits: Sequence[int]) -> Path:
    if image_id not in DEVELOPMENT_NAMES:
        raise LifecycleError(f"non-development image ID rejected: {image_id!r}")
    return outdir / "cells" / image_id / _bit_label(bits)


def _symbols_to_field_arrays(parsed: Any) -> dict[str, np.ndarray]:
    header = parsed.header
    bm, bs, br, bc = (int(value) for value in header["bits"])
    q_means = np.column_stack([parsed.symbols["mean_x"], parsed.symbols["mean_y"]])
    q_scales = np.column_stack([parsed.symbols["scale_x"], parsed.symbols["scale_y"]])
    q_colors = np.column_stack(
        [parsed.symbols["red"], parsed.symbols["green"], parsed.symbols["blue"]]
    )

    def dequant(q: np.ndarray, lo: Any, hi: Any, bits: int) -> np.ndarray:
        levels = (1 << bits) - 1
        return (
            np.asarray(lo, dtype=np.float64)
            + q.astype(np.float64)
            / levels
            * (np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64))
        ).astype(np.float32)

    means = dequant(q_means, header["means_lo"], header["means_hi"], bm)
    log_scales = dequant(q_scales, header["scale_lo"], header["scale_hi"], bs)
    if header.get("rot_circular", False):
        rotations = (parsed.symbols["rotation"].astype(np.float64) / (1 << br) * np.pi).astype(
            np.float32
        )
    else:
        rotations = dequant(parsed.symbols["rotation"], 0.0, np.pi, br)
    colors = dequant(q_colors, header["color_lo"], header["color_hi"], bc)
    return {
        "means": np.ascontiguousarray(means),
        "log_scales": np.ascontiguousarray(log_scales),
        "rotations": np.ascontiguousarray(rotations),
        "colors": np.ascontiguousarray(colors),
    }


def _mean_range_diagnostic(field: Any, height: int, width: int) -> dict[str, Any]:
    means = field.means.detach().cpu().numpy().astype(np.float64)
    off_image = (
        (means[:, 0] < 0.0)
        | (means[:, 0] > width - 1.0)
        | (means[:, 1] < 0.0)
        | (means[:, 1] > height - 1.0)
    )
    endpoints_lo = [
        float(min(0.0, np.floor(means[:, 0].min()))),
        float(min(0.0, np.floor(means[:, 1].min()))),
    ]
    endpoints_hi = [
        float(max(width - 1.0, np.ceil(means[:, 0].max()))),
        float(max(height - 1.0, np.ceil(means[:, 1].max()))),
    ]
    return {
        "qat_mean_fake_quantizer_domain": "image_box",
        "encoder_mean_domain_rule": "integer_floor_ceil_field_extent_unioned_with_image_box",
        "off_image_mean_count": int(np.count_nonzero(off_image)),
        "mean_min_xy": [float(means[:, 0].min()), float(means[:, 1].min())],
        "mean_max_xy": [float(means[:, 0].max()), float(means[:, 1].max())],
        "encoded_mean_range_lo_xy": endpoints_lo,
        "encoded_mean_range_hi_xy": endpoints_hi,
        "encoded_mean_range_differs_from_image_box": bool(
            endpoints_lo != [0.0, 0.0] or endpoints_hi != [float(width - 1), float(height - 1)]
        ),
        "claim_limitation": "shipped QAT mean lattice can differ from final SSPL1 mean lattice",
    }


def _decoded_integrity(blob: bytes) -> dict[str, Any]:
    from structsplat import codec

    parsed = oracle.parse_sspl1(blob)
    direct = codec.decode(blob, device="cpu")
    expected = _symbols_to_field_arrays(parsed)
    direct_arrays = _field_arrays(direct)
    if set(direct_arrays) != set(expected):
        raise LifecycleError("production decode returned unexpected field arrays")
    per_tensor_max_abs: dict[str, float] = {}
    for name in expected:
        maximum = float(
            np.max(
                np.abs(direct_arrays[name].astype(np.float64) - expected[name].astype(np.float64))
            )
        )
        per_tensor_max_abs[name] = maximum
        if maximum > DECODED_FIELD_ATOL:
            raise LifecycleError(
                f"production decode disagrees with independent symbols beyond 1e-6: {name}"
            )

    decoded_hash = _array_state_digest(direct_arrays)
    independent_hash = _array_state_digest(expected)
    if decoded_hash != independent_hash:
        raise LifecycleError("decoded field boundary state hash differs from independent symbols")
    return {
        "decoded_state_sha256": decoded_hash,
        "independent_symbol_field_state_sha256": independent_hash,
        "boundary_state_hash_exact": True,
        "production_vs_independent_per_tensor_max_abs": per_tensor_max_abs,
        "production_vs_independent_max_abs_lte_1e_6": all(
            value <= DECODED_FIELD_ATOL for value in per_tensor_max_abs.values()
        ),
        "decoded_field_atol": DECODED_FIELD_ATOL,
        "cpu_reference_render_diagnostic": {
            "validity_gate": False,
            "status": "not-run-optional-diagnostic",
            "atol": RENDER_DIAGNOSTIC_ATOL,
            "rtol": RENDER_DIAGNOSTIC_RTOL,
        },
    }


def _execute_cell(
    outdir: Path,
    image_record: Mapping[str, Any],
    base_record: Mapping[str, Any],
    bits: tuple[int, int, int, int],
    *,
    verbose: bool,
) -> dict[str, Any]:
    import torch

    from structsplat import codec
    from structsplat.config import FitConfig

    stage_started_utc = _utc_now()
    stage_started_perf = time.perf_counter()
    image_id = str(image_record["name"])
    final_dir = _cell_directory(outdir, image_id, bits)
    if final_dir.exists():
        raise LifecycleError(f"cell directory already exists without valid record: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = final_dir.with_name(final_dir.name + ".part")
    staging_restarted = staging.exists()
    if staging_restarted:
        if not staging.is_dir() or staging.is_symlink():
            raise LifecycleError(f"unsafe cell staging artifact: {staging}")
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        base_source = outdir / str(base_record["base_field"]["path"])
        qat_start = staging / "qat_start.npz"
        shutil.copyfile(base_source, qat_start)
        qat_start.chmod(0o444)
        if sha256_file(qat_start) != base_record["base_field"]["sha256"]:
            raise LifecycleError("QAT arm did not receive a byte-identical base field")
        field = _verify_field_artifact(
            outdir,
            base_record["base_field"],
            device="cuda",
        )
        if field_state(field)["state_sha256"] != base_record["base_field"]["state_sha256"]:
            raise LifecycleError("QAT arm did not receive a state-identical base field")
        pixels = _target_pixels(outdir, image_record)
        target = torch.as_tensor(pixels.astype(np.float32) / 255.0, device="cuda")
        fit_config = FitConfig(**resolved_configs()["fit"])
        codec_config = codec.CodecConfig(
            bits_means=bits[0],
            bits_scales=bits[1],
            bits_rot=bits[2],
            bits_colors=bits[3],
            morton_reorder=True,
            color_delta=True,
            means_byte_planar=True,
            rot_circular=True,
            stream_codec="zlib",
            zlib_level=9,
        )
        _seed_everything()
        started = time.perf_counter()
        frozen_codec = codec.qat_finetune(
            field,
            target,
            fit_config,
            codec_config,
            iters=QAT_ITERATIONS,
            verbose=verbose,
        )
        torch.cuda.synchronize()
        qat_seconds = time.perf_counter() - started
        if frozen_codec.color_lo is None or frozen_codec.scale_lo is None:
            raise LifecycleError("QAT failed to return frozen codec ranges")
        terminal = _persist_field(field, staging / "qat_terminal.npz")
        terminal["path"] = f"cells/{image_id}/{_bit_label(bits)}/qat_terminal.npz"

        height, width = (int(value) for value in image_record["derived_rgb_dimensions_hw"])
        mean_range_diagnostic = _mean_range_diagnostic(field, height, width)
        started = time.perf_counter()
        blob = codec.encode(field, height, width, frozen_codec, fit_config)
        encode_seconds = time.perf_counter() - started
        blob_path = staging / "stream.sspl1"
        _exclusive_write(blob_path, blob, readonly=True)
        components = codec.blob_components(blob)
        if components["total_bytes"] != len(blob) or (
            components["header_bytes"]
            + sum(int(item["framed_bytes"]) for item in components["streams"].values())
            != len(blob)
        ):
            raise LifecycleError("production SSPL1 component accounting mismatch")

        parsed_direct = oracle.parse_sspl1(blob)
        started = time.perf_counter()
        cold_blob = blob_path.read_bytes()
        cold_read_seconds = time.perf_counter() - started
        started = time.perf_counter()
        parsed_cold = oracle.parse_sspl1(cold_blob)
        cold_parse_seconds = time.perf_counter() - started
        if [float(value) for value in parsed_cold.header["means_lo"]] != mean_range_diagnostic[
            "encoded_mean_range_lo_xy"
        ] or [float(value) for value in parsed_cold.header["means_hi"]] != mean_range_diagnostic[
            "encoded_mean_range_hi_xy"
        ]:
            raise LifecycleError("persisted QAT mean-range diagnostic disagrees with SSPL1")
        started = time.perf_counter()
        decode_integrity = _decoded_integrity(cold_blob)
        cold_decode_integrity_seconds = time.perf_counter() - started
        if (
            parsed_direct.blob_sha256 != parsed_cold.blob_sha256
            or parsed_direct.symbol_sha256 != parsed_cold.symbol_sha256
            or any(
                not np.array_equal(parsed_direct.symbols[name], parsed_cold.symbols[name])
                for name in parsed_direct.symbols
            )
        ):
            raise LifecycleError("in-memory and fresh persisted SSPL1 parses disagree")
        if tuple(int(value) for value in parsed_cold.header["bits"]) != bits:
            raise LifecycleError("persisted SSPL1 bit tuple drift")
        symbols_path = staging / "absolute_symbols.npz"
        np.savez(symbols_path, **parsed_cold.symbols)
        symbols_path.chmod(0o444)
        with np.load(symbols_path, allow_pickle=False) as archive:
            if any(
                not np.array_equal(archive[name], parsed_cold.symbols[name])
                for name in parsed_cold.symbols
            ):
                raise LifecycleError("persisted absolute-symbol arrays drift")
        row = oracle.make_oracle_row(cold_blob, image_id)
        oracle.verify_record(row)
        relative = f"cells/{image_id}/{_bit_label(bits)}"
        start_artifact = {
            "path": f"{relative}/qat_start.npz",
            "bytes": qat_start.stat().st_size,
            "sha256": sha256_file(qat_start),
            "n": base_record["base_field"]["n"],
            "state_sha256": base_record["base_field"]["state_sha256"],
        }
        record = _seal(
            {
                "schema": CELL_SCHEMA,
                "protocol": PROTOCOL,
                "image_id": image_id,
                "bit_tuple": list(bits),
                "base_record_sha256": base_record["base_record_sha256"],
                "qat_start": start_artifact,
                "qat_terminal": terminal,
                "frozen_codec_config": asdict(frozen_codec),
                "qat_mean_range_diagnostic": mean_range_diagnostic,
                "fit_config": asdict(fit_config),
                "sspl1": {
                    "path": f"{relative}/stream.sspl1",
                    "bytes": len(blob),
                    "sha256": sha256_bytes(blob),
                    "components": components,
                },
                "absolute_symbols": {
                    "path": f"{relative}/absolute_symbols.npz",
                    "bytes": symbols_path.stat().st_size,
                    "sha256": sha256_file(symbols_path),
                    "symbol_sha256": parsed_cold.symbol_sha256,
                },
                "integrity": decode_integrity,
                "oracle_row": row,
                "timings": {
                    "qat_wall_seconds": qat_seconds,
                    "encode_wall_seconds": encode_seconds,
                    "cold_file_read_wall_seconds": cold_read_seconds,
                    "cold_parse_wall_seconds": cold_parse_seconds,
                    "cold_decode_integrity_wall_seconds": cold_decode_integrity_seconds,
                },
                "unsealed_staging_restarted_after_interruption": staging_restarted,
                "stage_process": _process_stage_record(stage_started_utc, stage_started_perf),
            },
            "cell_record_sha256",
        )
        _exclusive_write(staging / "cell_record.json", canonical_json(record), readonly=True)
        os.replace(staging, final_dir)
        return record
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_cell_record(outdir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    _verify_seal(value, "cell_record_sha256")
    image_id = value.get("image_id")
    raw_bits = value.get("bit_tuple")
    bits = tuple(raw_bits) if isinstance(raw_bits, list) else ()
    expected_keys = {
        "schema",
        "protocol",
        "image_id",
        "bit_tuple",
        "base_record_sha256",
        "qat_start",
        "qat_terminal",
        "frozen_codec_config",
        "qat_mean_range_diagnostic",
        "fit_config",
        "sspl1",
        "absolute_symbols",
        "integrity",
        "oracle_row",
        "timings",
        "unsealed_staging_restarted_after_interruption",
        "stage_process",
        "cell_record_sha256",
    }
    if (
        set(value) != expected_keys
        or value.get("schema") != CELL_SCHEMA
        or value.get("protocol") != PROTOCOL
        or image_id not in DEVELOPMENT_NAMES
        or bits not in BIT_TUPLES
        or any(isinstance(item, bool) or not isinstance(item, int) for item in bits)
        or not isinstance(value.get("unsealed_staging_restarted_after_interruption"), bool)
    ):
        raise LifecycleError("invalid COMP-008 cell identity")
    directory = _cell_directory(outdir, str(image_id), bits)
    persisted = _read_object(directory / "cell_record.json")
    if canonical_json(persisted) != canonical_json(value):
        raise LifecycleError("journal and persisted cell record disagree")
    base = _load_base_record(outdir, str(image_id))
    if value["base_record_sha256"] != base["base_record_sha256"]:
        raise LifecycleError("cell references a different shared base field")
    relative = f"cells/{image_id}/{_bit_label(bits)}"
    if (
        value["qat_start"].get("path") != f"{relative}/qat_start.npz"
        or value["qat_terminal"].get("path") != f"{relative}/qat_terminal.npz"
        or value["sspl1"].get("path") != f"{relative}/stream.sspl1"
        or value["absolute_symbols"].get("path") != f"{relative}/absolute_symbols.npz"
    ):
        raise LifecycleError("cell artifact path drift")
    start_field = _verify_field_artifact(outdir, value["qat_start"])
    terminal_field = _verify_field_artifact(outdir, value["qat_terminal"])
    if (
        value["qat_start"]["sha256"] != base["base_field"]["sha256"]
        or value["qat_start"]["state_sha256"] != base["base_field"]["state_sha256"]
        or field_state(start_field)["state_sha256"] != base["base_field"]["state_sha256"]
    ):
        raise LifecycleError("QAT cell does not start at the byte/state-identical shared base")
    blob_path = outdir / str(value["sspl1"]["path"])
    blob = blob_path.read_bytes()
    if len(blob) != value["sspl1"]["bytes"] or sha256_bytes(blob) != value["sspl1"]["sha256"]:
        raise LifecycleError("persisted SSPL1 stream drift")
    components = value["sspl1"]["components"]
    from structsplat.codec import blob_components

    if blob_components(blob) != components:
        raise LifecycleError("SSPL1 component accounting drift")
    symbols = value["absolute_symbols"]
    symbols_path = outdir / str(symbols["path"])
    if (
        symbols_path.stat().st_size != symbols["bytes"]
        or sha256_file(symbols_path) != symbols["sha256"]
    ):
        raise LifecycleError("persisted symbol archive drift")
    parsed = oracle.parse_sspl1(blob)
    if value.get("fit_config") != resolved_configs()["fit"]:
        raise LifecycleError("cell fit configuration drift")
    frozen_codec = value.get("frozen_codec_config")
    if not isinstance(frozen_codec, dict) or (
        [
            frozen_codec.get(name)
            for name in ("bits_means", "bits_scales", "bits_rot", "bits_colors")
        ]
        != list(parsed.header["bits"])
        or frozen_codec.get("morton_reorder") is not parsed.header["morton"]
        or frozen_codec.get("color_delta") is not parsed.header["color_delta"]
        or frozen_codec.get("means_byte_planar") is not parsed.header["means_byte_planar"]
        or frozen_codec.get("rot_circular") is not parsed.header["rot_circular"]
        or frozen_codec.get("stream_codec") != parsed.header["stream_codec"]
        or frozen_codec.get("zlib_level") != 9
        or frozen_codec.get("color_lo") != parsed.header["color_lo"]
        or frozen_codec.get("color_hi") != parsed.header["color_hi"]
        or frozen_codec.get("scale_lo") != parsed.header["scale_lo"]
        or frozen_codec.get("scale_hi") != parsed.header["scale_hi"]
    ):
        raise LifecycleError("frozen codec configuration disagrees with persisted SSPL1")
    mean_diagnostic = _mean_range_diagnostic(
        terminal_field,
        int(parsed.header["H"]),
        int(parsed.header["W"]),
    )
    if canonical_json(mean_diagnostic) != canonical_json(value.get("qat_mean_range_diagnostic")):
        raise LifecycleError("persisted QAT mean-range diagnostic replay drift")
    decoded_integrity = _decoded_integrity(blob)
    if canonical_json(decoded_integrity) != canonical_json(value.get("integrity")):
        raise LifecycleError("persisted decoded-field integrity replay drift")
    if parsed.symbol_sha256 != symbols["symbol_sha256"]:
        raise LifecycleError("persisted SSPL1 symbols drift")
    with np.load(symbols_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(parsed.symbols) or any(
            not np.array_equal(arrays[name], parsed.symbols[name]) for name in parsed.symbols
        ):
            raise LifecycleError("symbol NPZ disagrees with cold SSPL1 parse")
    if not oracle.replay_row(blob, value["oracle_row"])["row_exact"]:
        raise LifecycleError("persisted oracle row replay drift")
    _verify_process_stage_record(value.get("stage_process"), stage="QAT cell")
    return value


def _journal_cells(outdir: Path) -> dict[tuple[str, tuple[int, ...]], dict[str, Any]]:
    result: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for record in _load_jsonl(outdir / JOURNAL_NAME):
        verified = _verify_cell_record(outdir, record)
        key = (str(verified["image_id"]), tuple(int(v) for v in verified["bit_tuple"]))
        if key in result:
            raise LifecycleError(f"duplicate append-only journal cell: {key}")
        result[key] = verified
    return result


def _load_run_invocations(
    outdir: Path,
    *,
    preflight_binding_sha256: str,
    targets_sha256: str,
) -> list[dict[str, Any]]:
    records = _load_jsonl(outdir / RUN_INVOCATIONS_NAME)
    seen: set[str] = set()
    expected_keys = {
        "schema",
        "protocol",
        "stage",
        "preflight_binding_sha256",
        "targets_sha256",
        "requested_image_id",
        "validated_or_completed_cells",
        "unsealed_staging_repairs",
        "truncated_tail_repairs",
        "stage_process",
        "run_invocation_sha256",
    }
    for record in records:
        _verify_seal(record, "run_invocation_sha256")
        digest = record["run_invocation_sha256"]
        requested = record.get("requested_image_id")
        cell_hashes = record.get("validated_or_completed_cells")
        if (
            set(record) != expected_keys
            or record.get("schema") != "structsplat.comp008.run-dev-invocation.v1"
            or record.get("protocol") != PROTOCOL
            or record.get("stage") != "run-dev"
            or record.get("preflight_binding_sha256") != preflight_binding_sha256
            or record.get("targets_sha256") != targets_sha256
            or (requested is not None and requested not in DEVELOPMENT_NAMES)
            or not isinstance(cell_hashes, list)
            or not cell_hashes
            or any(not isinstance(value, str) or len(value) != 64 for value in cell_hashes)
            or digest in seen
        ):
            raise LifecycleError("run-dev invocation schema or binding drift")
        _verify_process_stage_record(record.get("stage_process"), stage="run-dev")
        if not isinstance(record.get("unsealed_staging_repairs"), list) or not isinstance(
            record.get("truncated_tail_repairs"), list
        ):
            raise LifecycleError("run-dev repair provenance drift")
        seen.add(digest)
    return records


def _verify_run_invocation_coverage(
    invocations: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> None:
    recorded = {
        value for invocation in invocations for value in invocation["validated_or_completed_cells"]
    }
    expected = {str(cell["cell_record_sha256"]) for cell in cells}
    if recorded != expected:
        raise LifecycleError(
            "run-dev invocation coverage differs from the complete persisted cell grid"
        )


def _recover_unjournaled_cells(
    outdir: Path, known: dict[tuple[str, tuple[int, ...]], dict[str, Any]]
) -> None:
    for image_id in DEVELOPMENT_NAMES:
        for bits in BIT_TUPLES:
            key = (image_id, bits)
            path = _cell_directory(outdir, image_id, bits) / "cell_record.json"
            if path.exists() and key not in known:
                record = _verify_cell_record(outdir, _read_object(path))
                _append_jsonl(outdir / JOURNAL_NAME, record)
                known[key] = record


def run_development(
    outdir: str | Path,
    *,
    image_id: str | None = None,
    verbose: bool = False,
    environment_fn: Callable[[], Mapping[str, Any]] = environment_record,
) -> list[dict[str, Any]]:
    stage_started_utc = _utc_now()
    stage_started_perf = time.perf_counter()
    root = Path(outdir)
    preflight_record = verify_preflight(root, environment_fn=environment_fn)
    targets = _load_target_manifest(root)
    if targets["preflight_binding_sha256"] != preflight_record["binding_sha256"]:
        raise LifecycleError("target/preflight binding mismatch")
    if (root / ANALYSIS_NAME).exists() or (root / REPLAY_NAME).exists():
        raise LifecycleError("run-dev cannot mutate an analyzed artifact")
    selected = list(DEVELOPMENT_NAMES)
    if image_id is not None:
        if image_id not in DEVELOPMENT_NAMES:
            raise LifecycleError(f"run-dev rejects non-development image ID: {image_id!r}")
        selected = [image_id]
    by_name = {record["name"]: record for record in targets["targets"]}
    staging_repairs = _remove_unsealed_staging_directories(root)
    journal_repairs = [
        repair
        for name in (JOURNAL_NAME, RUN_INVOCATIONS_NAME)
        if (repair := _repair_truncated_jsonl_tail(root / name)) is not None
    ]
    _assert_known_artifact_layout(root, allow_staging=False)
    _load_run_invocations(
        root,
        preflight_binding_sha256=preflight_record["binding_sha256"],
        targets_sha256=targets["targets_sha256"],
    )
    known = _journal_cells(root)
    _recover_unjournaled_cells(root, known)
    completed: list[dict[str, Any]] = []
    for name in selected:
        if _base_record_path(root, name).exists():
            base = _load_base_record(root, name)
        else:
            base = _fit_base(root, by_name[name], verbose=verbose)
        for bits in BIT_TUPLES:
            key = (name, bits)
            if key in known:
                completed.append(known[key])
                continue
            record = _execute_cell(root, by_name[name], base, bits, verbose=verbose)
            _append_jsonl(root / JOURNAL_NAME, record)
            known[key] = record
            completed.append(record)
        first, second = (known[(name, bits)] for bits in BIT_TUPLES)
        if (
            first["qat_start"]["sha256"] != second["qat_start"]["sha256"]
            or first["qat_start"]["state_sha256"] != second["qat_start"]["state_sha256"]
        ):
            raise LifecycleError(f"QAT arms for {name} do not share the exact base")
    invocation = _seal(
        {
            "schema": "structsplat.comp008.run-dev-invocation.v1",
            "protocol": PROTOCOL,
            "stage": "run-dev",
            "preflight_binding_sha256": preflight_record["binding_sha256"],
            "targets_sha256": targets["targets_sha256"],
            "requested_image_id": image_id,
            "validated_or_completed_cells": [record["cell_record_sha256"] for record in completed],
            "unsealed_staging_repairs": staging_repairs,
            "truncated_tail_repairs": journal_repairs,
            "stage_process": _process_stage_record(stage_started_utc, stage_started_perf),
        },
        "run_invocation_sha256",
    )
    _append_jsonl(root / RUN_INVOCATIONS_NAME, invocation)
    _load_run_invocations(
        root,
        preflight_binding_sha256=preflight_record["binding_sha256"],
        targets_sha256=targets["targets_sha256"],
    )
    return completed


def _complete_cells(outdir: Path) -> list[dict[str, Any]]:
    known = _journal_cells(outdir)
    _recover_unjournaled_cells(outdir, known)
    expected = {(name, bits) for name in DEVELOPMENT_NAMES for bits in BIT_TUPLES}
    if set(known) != expected:
        missing = sorted(expected - set(known))
        extra = sorted(set(known) - expected)
        raise LifecycleError(
            f"analysis requires exact 16-cell grid; missing={missing}, extra={extra}"
        )
    return [known[(name, bits)] for name in DEVELOPMENT_NAMES for bits in BIT_TUPLES]


def analyze(
    outdir: str | Path,
    *,
    environment_fn: Callable[[], Mapping[str, Any]] = environment_record,
) -> dict[str, Any]:
    stage_started_utc = _utc_now()
    stage_started_perf = time.perf_counter()
    root = Path(outdir)
    preflight_record = verify_preflight(root, environment_fn=environment_fn)
    _assert_known_artifact_layout(root, allow_staging=False)
    targets = _load_target_manifest(root)
    if targets["preflight_binding_sha256"] != preflight_record["binding_sha256"]:
        raise LifecycleError("target/preflight binding mismatch")
    if (root / ANALYSIS_NAME).exists() or (root / ANALYSIS_RECORD_NAME).exists():
        raise LifecycleError("analysis artifacts already exist")
    cells = _complete_cells(root)
    invocations = _load_run_invocations(
        root,
        preflight_binding_sha256=preflight_record["binding_sha256"],
        targets_sha256=targets["targets_sha256"],
    )
    _verify_run_invocation_coverage(invocations, cells)
    # Scientific aggregation is blob-backed, never journal-backed.  Re-read every immutable
    # stream, reconstruct its canonical row, and give the core the complete cell->blob mapping.
    blobs: dict[tuple[str, tuple[int, int, int, int]], bytes] = {}
    rows = []
    for cell in cells:
        image_id = str(cell["image_id"])
        bits = tuple(int(value) for value in cell["bit_tuple"])
        blob = (root / str(cell["sspl1"]["path"])).read_bytes()
        recomputed = oracle.make_oracle_row(blob, image_id)
        if canonical_json(recomputed) != canonical_json(cell["oracle_row"]):
            raise LifecycleError("cell journal row differs from its persisted SSPL1")
        blobs[(image_id, bits)] = blob
        rows.append(recomputed)
    analysis_result = oracle.analyze_rows(rows, blobs=blobs)
    _exclusive_write(root / ANALYSIS_NAME, canonical_json(analysis_result), readonly=True)
    record = _seal(
        {
            "schema": ANALYSIS_RECORD_SCHEMA,
            "protocol": PROTOCOL,
            "stage": "analyze",
            "preflight_binding_sha256": preflight_record["binding_sha256"],
            "targets_sha256": targets["targets_sha256"],
            "cell_record_sha256": [cell["cell_record_sha256"] for cell in cells],
            "analysis_path": ANALYSIS_NAME,
            "analysis_file_sha256": sha256_file(root / ANALYSIS_NAME),
            "analysis_sha256": analysis_result["analysis_sha256"],
            "decision": analysis_result["decision"],
            "coder_authorized": analysis_result["coder_authorized"],
            "confirmation_payloads_accessed": False,
            "stage_process": _process_stage_record(stage_started_utc, stage_started_perf),
        },
        "analysis_record_sha256",
    )
    _exclusive_write(root / ANALYSIS_RECORD_NAME, canonical_json(record), readonly=True)
    return analysis_result


def replay(
    outdir: str | Path,
    *,
    environment_fn: Callable[[], Mapping[str, Any]] = environment_record,
) -> dict[str, Any]:
    stage_started_utc = _utc_now()
    stage_started_perf = time.perf_counter()
    root = Path(outdir)
    preflight_record = verify_preflight(root, environment_fn=environment_fn)
    _assert_known_artifact_layout(root, allow_staging=False)
    targets = _load_target_manifest(root, decode_pixels=False)
    if targets["preflight_binding_sha256"] != preflight_record["binding_sha256"]:
        raise LifecycleError("target/preflight binding mismatch")
    if (root / REPLAY_NAME).exists():
        raise LifecycleError("replay artifact already exists")
    analysis_record = _read_object(root / ANALYSIS_RECORD_NAME)
    _verify_seal(analysis_record, "analysis_record_sha256")
    recorded_analysis = _read_object(root / ANALYSIS_NAME)
    oracle.verify_record(recorded_analysis, "analysis_sha256")
    expected_analysis_record_keys = {
        "schema",
        "protocol",
        "stage",
        "preflight_binding_sha256",
        "targets_sha256",
        "cell_record_sha256",
        "analysis_path",
        "analysis_file_sha256",
        "analysis_sha256",
        "decision",
        "coder_authorized",
        "confirmation_payloads_accessed",
        "stage_process",
        "analysis_record_sha256",
    }
    if (
        set(analysis_record) != expected_analysis_record_keys
        or analysis_record.get("schema") != ANALYSIS_RECORD_SCHEMA
        or analysis_record.get("protocol") != PROTOCOL
        or analysis_record.get("stage") != "analyze"
        or analysis_record.get("preflight_binding_sha256") != preflight_record["binding_sha256"]
        or analysis_record.get("targets_sha256") != targets["targets_sha256"]
        or analysis_record.get("analysis_path") != ANALYSIS_NAME
        or analysis_record.get("analysis_file_sha256") != sha256_file(root / ANALYSIS_NAME)
        or analysis_record.get("analysis_sha256") != recorded_analysis["analysis_sha256"]
        or analysis_record.get("decision") != recorded_analysis["decision"]
        or analysis_record.get("coder_authorized") is not recorded_analysis["coder_authorized"]
        or analysis_record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("analysis provenance drift")
    _verify_process_stage_record(analysis_record.get("stage_process"), stage="analyze")

    cells = _complete_cells(root)
    if analysis_record.get("cell_record_sha256") != [cell["cell_record_sha256"] for cell in cells]:
        raise LifecycleError("analysis cell provenance drift")
    invocations = _load_run_invocations(
        root,
        preflight_binding_sha256=preflight_record["binding_sha256"],
        targets_sha256=targets["targets_sha256"],
    )
    _verify_run_invocation_coverage(invocations, cells)
    replayed_rows = []
    blobs: dict[tuple[str, tuple[int, int, int, int]], bytes] = {}
    row_checks = []
    for cell in cells:
        blob = (root / str(cell["sspl1"]["path"])).read_bytes()
        recomputed = oracle.make_oracle_row(blob, str(cell["image_id"]))
        exact = canonical_json(recomputed) == canonical_json(cell["oracle_row"])
        row_checks.append(
            {
                "image_id": cell["image_id"],
                "bit_tuple": cell["bit_tuple"],
                "row_exact": exact,
                "recorded_row_sha256": cell["oracle_row"]["record_sha256"],
                "recomputed_row_sha256": recomputed["record_sha256"],
            }
        )
        if not exact:
            raise LifecycleError("cold persisted-stream oracle row replay drift")
        blobs[(str(cell["image_id"]), tuple(int(v) for v in cell["bit_tuple"]))] = blob
        replayed_rows.append(recomputed)
    core_replay = oracle.replay_analysis(replayed_rows, recorded_analysis, blobs=blobs)
    if not core_replay["ok"]:
        raise LifecycleError("aggregate oracle analysis replay drift")
    result = _seal(
        {
            "schema": REPLAY_RECORD_SCHEMA,
            "protocol": PROTOCOL,
            "stage": "replay",
            "preflight_binding_sha256": preflight_record["binding_sha256"],
            "targets_sha256": targets["targets_sha256"],
            "analysis_record_sha256": analysis_record["analysis_record_sha256"],
            "replay_scope": (
                "cold SSPL1 parse, absolute symbols, exact entropy lower bounds, rows, "
                "aggregates, gates, and decision; no refit and no pixel decode"
            ),
            "row_checks": row_checks,
            "core_replay": core_replay,
            "oracle_rows_aggregates_gates_and_decision_exact": True,
            "pixel_payloads_decoded": False,
            "confirmation_payloads_accessed": False,
            "ok": True,
            "stage_process": _process_stage_record(stage_started_utc, stage_started_perf),
        },
        "lifecycle_replay_sha256",
    )
    _exclusive_write(root / REPLAY_NAME, canonical_json(result), readonly=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--outdir", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare-dev")
    prepare_parser.add_argument("--outdir", type=Path, required=True)
    prepare_parser.add_argument("--acquisition-dir", type=Path, required=True)

    run_parser = subparsers.add_parser("run-dev")
    run_parser.add_argument("--outdir", type=Path, required=True)
    run_parser.add_argument("--image-id", choices=DEVELOPMENT_NAMES)
    run_parser.add_argument("--verbose", action="store_true")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--outdir", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--outdir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "preflight":
        result = preflight(args.outdir)
        print(result["binding_sha256"])
    elif args.stage == "prepare-dev":
        result = prepare_development(args.outdir, args.acquisition_dir)
        print(result["targets_sha256"])
    elif args.stage == "run-dev":
        result = run_development(
            args.outdir,
            image_id=args.image_id,
            verbose=args.verbose,
        )
        print(f"validated/completed {len(result)} cells")
    elif args.stage == "analyze":
        result = analyze(args.outdir)
        print(result["decision"])
    elif args.stage == "replay":
        result = replay(args.outdir)
        print(result["lifecycle_replay_sha256"])
    else:  # pragma: no cover
        raise AssertionError(args.stage)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
