"""Source-bound lifecycle runner for the frozen COMP-011 SSP2V experiment.

The public commands implement the complete linear, append-only experiment lifecycle.  Preflight
binds metadata and executable authorities without opening scientific payloads; every later stage
can read only the path classes enabled by its exact predecessor seal.  Cell products are committed
as immutable directories and every stage manifest binds an immutable journal prefix so interrupted
work can resume only from a byte-identical frozen-order prefix.
"""

from __future__ import annotations

import argparse
import ast
import csv
import errno
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import resource
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings
import zlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_assets as comp009_assets
from benchmarks import ssp2e_container as comp009_container
from benchmarks import ssp2e_execute as comp009_execute
from benchmarks import ssp2v_actual_coder as actual
from benchmarks import ssp2v_container as container
from benchmarks import ssp2v_decode_worker as decode_worker
from benchmarks import ssp2v_execute as execute
from benchmarks import ssp2v_landlock as sandbox
from benchmarks import ssp2v_lloyd as lloyd
from benchmarks import ssp2v_quality as quality


ROOT = Path(__file__).resolve().parents[1]
TASK_SHA256 = "e73b69321de353c572292e17688019e6c1a62b55ebe1d1e3e66bac4120bd9ea5"
PROTOCOL = "COMP-011-complete-stream-rgb-vq-v2"
PREFLIGHT_SCHEMA = "structsplat.comp011.preflight.v1"
SOURCE_SCHEMA = "structsplat.comp011.source-manifest.v1"
JOURNAL_SCHEMA = "structsplat.comp011.access-event.v1"
RENDERER_BUILD_SCHEMA = "structsplat.comp011.renderer-build-worker.v1"
RENDERER_PROOF_SCHEMA = "structsplat.comp011.renderer-proof-worker.v1"
LLOYD_PROOF_SCHEMA = "structsplat.comp011.lloyd-complexity-worker.v1"
IMPORT_STREAMS_SCHEMA = "structsplat.comp011.import-streams.v1"
CANDIDATE_STAGE_SCHEMA = "structsplat.comp011.candidate-stage.v1"
TARGET_COPY_SCHEMA = "structsplat.comp011.target-copy.v1"
QUALITY_STAGE_SCHEMA = "structsplat.comp011.quality-stage.v1"
BENCHMARK_STAGE_SCHEMA = "structsplat.comp011.benchmark-stage.v1"
ANALYSIS_RECORD_SCHEMA = "structsplat.comp011.analysis-record.v1"
REPLAY_SCHEMA = "structsplat.comp011.replay.v2"
REPLAY_PHASE_A_SCHEMA = "structsplat.comp011.replay-phase-a.v1"
REPLAY_PHASE_B_SCHEMA = "structsplat.comp011.replay-phase-b.v2"
REPLAY_QUALITY_PLAN_SCHEMA = "structsplat.comp011.replay-quality-plan.v1"
REPLAY_QUALITY_RESULTS_SCHEMA = "structsplat.comp011.replay-quality-results.v1"
REPLAY_QUALITY_AUTHORIZATION_SCHEMA = (
    "structsplat.comp011.replay-quality-authorization.v1"
)
REPLAY_RESOURCE_RESULTS_SCHEMA = "structsplat.comp011.replay-resource-results.v1"
PRODUCER_POLICY_SCHEMA = "structsplat.comp011.producing-cell-launch-policy.v1"
WORKER_POLICY_SCHEMA = "structsplat.comp011.worker-launch-policy.v1"
FOCUSED_DEPENDENCY_SCHEMA = "structsplat.comp011.focused-dependencies.v1"
RUNTIME_SOURCE_SCHEMA = "structsplat.comp011.runtime-source.v1"
CPUINFO_IDENTITY_SCHEMA = "structsplat.comp011.cpuinfo-identity.v1"
LSCPU_IDENTITY_SCHEMA = "structsplat.comp011.lscpu-identity.v1"
NVIDIA_SMI_IDENTITY_SCHEMA = "structsplat.comp011.nvidia-smi-identity.v1"
_ALEXNET_CACHE_RELATIVE = Path("hub/checkpoints/alexnet-owt-7be5be79.pth")
_RUNTIME_SOURCE_RELATIVE = Path("runtime/source")
_CAPTURED_REPLAY_SOURCE_ENV = "COMP011_CAPTURED_REPLAY_SOURCE_ROOT"
_REPLAY_PREFLIGHT_BINDING_ENV = "COMP011_REPLAY_PREFLIGHT_BINDING_SHA256"
_REPLAY_SYSTEM_READ_ONLY_ENV = "COMP011_REPLAY_SYSTEM_READ_ONLY_JSON"
_REPLAY_GPU_READ_WRITE_ENV = "COMP011_REPLAY_GPU_READ_WRITE_JSON"
_CPUINFO_VOLATILE_FIELDS = frozenset({"cpu MHz"})
_LSCPU_VOLATILE_FIELDS = frozenset({"CPU(s) scaling MHz:"})
_NVIDIA_SMI_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "driver_version",
    "memory.total",
    "compute_cap",
)
_EXPECTED_METRIC_DEPENDENCY_WARNINGS = (
    (
        "UserWarning",
        "The parameter 'pretrained' is deprecated since 0.13 and may be removed in the "
        "future, please use 'weights' instead.",
    ),
    (
        "UserWarning",
        "Arguments other than a weight enum or `None` for 'weights' are deprecated since "
        "0.13 and may be removed in the future. The current behavior is equivalent to "
        "passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use "
        "`weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.",
    ),
)

_FOCUSED_BASE_FIELDS = (
    "command",
    "returncode",
    "stdout_bytes",
    "stdout_sha256",
    "stdout_hex",
    "stderr_bytes",
    "stderr_sha256",
    "stderr_hex",
    "wall_ns",
    "captured_source_manifest_sha256",
    "captured_source_file_count",
    "captured_source_reverified_after_proof",
    "captured_runtime_dependencies",
    "captured_dependencies_reverified_after_proof",
)

_STAGE_SEAL_FIELDS = {
    "preflight": "preflight_binding_sha256",
    "import-streams": "import_streams_sha256",
    "fit-encode-cold-decode-dev": "fit_stage_sha256",
    "import-targets": "target_copy_sha256",
    "quality-select-dev": "quality_stage_sha256",
    "benchmark-dev": "benchmark_stage_sha256",
    "analyze": "analysis_record_sha256",
    "replay": "replay_sha256",
}

_STAGE_MANIFEST_PATHS = {
    "preflight": "preflight.json",
    "import-streams": "import_streams.json",
    "fit-encode-cold-decode-dev": "candidate_stage.json",
    "import-targets": "target_copy.json",
    "quality-select-dev": "quality_stage.json",
    "benchmark-dev": "benchmark_stage.json",
    "analyze": "analysis_record.json",
    "replay": "replay.json",
}

_STAGE_SCHEMAS = {
    "preflight": PREFLIGHT_SCHEMA,
    "import-streams": IMPORT_STREAMS_SCHEMA,
    "fit-encode-cold-decode-dev": CANDIDATE_STAGE_SCHEMA,
    "import-targets": TARGET_COPY_SCHEMA,
    "quality-select-dev": QUALITY_STAGE_SCHEMA,
    "benchmark-dev": BENCHMARK_STAGE_SCHEMA,
    "analyze": ANALYSIS_RECORD_SCHEMA,
    "replay": REPLAY_SCHEMA,
}

DEFAULT_COMP008 = ROOT / "results/comp008_sgi_entropy_oracle_dev_v3_2026-07-16"
DEFAULT_COMP009 = ROOT / "results/comp009_ssp2e_actual_dev_v1_2026-07-16"
DEFAULT_COMP010 = ROOT / "results/comp010_ssp2e_replay_repair_v2r2_2026-07-16"

EXPECTED_COMP008 = {
    "binding_sha256": "9c5dca682d490567a56ce4fc4043112a5bacf72b2fee908e95e86e691a1ea0f2",
    "analysis_sha256": "3368a3f96926b0eaf2e18119ae587d4bb3822c450c0f6f5a33e883c4e87e6792",
    "lifecycle_replay_sha256": "95829d0c563842adcc0a7788d7ff02a8e6bd74596f80a3a3966d78c2da9b1397",
    "cells_file_sha256": "9f14774f091afb0b8c2e9c471f9e9bd21f2d8800f44da6e413a076acf33ccb8b",
    "analysis_record_sha256": "1081a77e7d5d58c04ef8695e0b45d0508b6e0835e5db597094fc1d61597f0c56",
    "targets_sha256": "0b08414bd2d72ff3b3e889807e9754b9454cf6f8e015f7f57df3a23f2fda57ab",
}
EXPECTED_COMP009 = {
    "preflight_binding_sha256": "b4d843ce22356839fa3fa39082044e61cc7ee82cc71176780a192d811d63fadc",
    "source_manifest_sha256": "d75ee551ee650f0d465a01fc8f49b2ee0c4eee15579cf362ee3dfda2b96ad7d6",
    "source_archive_sha256": "fac4ca0978891b3cd16d477ffc04d12dc120168478192293d39af635fca7eb50",
    "input_manifest_sha256": "8f12a64d484f4239a121277ff0467409ed374856ddbf134dfaf49c737fea5f1a",
    "run_manifest_sha256": "cc21fc84aab5cd55ecfffc358f67782013d7a7cf56ffc9b8ef34f8ea98de0a24",
    "benchmark_manifest_sha256": "88a706da64836544d997b177b3b4dda2610e6ede3eceb457e98a4aa7899fe162",
    "analysis_record_sha256": "435f011fe598263cd304b1cbe0754ca82e5fe1e7a49536db4099bdfff9166201",
    "analysis_sha256": "f10c4a1906e4bd240c10253508f38c4053cfe332dec13b220262fee7ae990b30",
}
EXPECTED_COMP010 = {
    "repair_preflight_sha256": "a00a7a553468dbe1f2e00e1d6e5ecb38fb3227356e090c73bbf564b39ec1bc00",
    "repair_sha256": "adbf2c48e1721b6d4b74960211e08ec3c770bebde1d068ddfc88e5a83f78f3a8",
    "repair_child_sha256": "b482a866742d472d02d7c80db1c86640e5e17a85fdf025a56096173c08fd7ac4",
    "captured_replay_worker_sha256": "0d7d74a8e6f6a552eeb4f1d3df4a0aeb8f6e6977496400d7c65845c4fd494309",
}

STAGES = (
    "preflight",
    "import-streams",
    "fit-encode-cold-decode-dev",
    "import-targets",
    "quality-select-dev",
    "benchmark-dev",
    "analyze",
    "replay",
)

_STAGE_ACCESS: dict[str, frozenset[str]] = {
    "repository_source": frozenset(STAGES),
    "upstream_metadata": frozenset({"preflight"}),
    "runtime_dependency": frozenset(STAGES),
    "artifact_runtime": frozenset(STAGES),
    "development_stream": frozenset({"import-streams"}),
    "artifact_stream": frozenset(STAGES[1:]),
    "upstream_target": frozenset({"import-targets"}),
    "artifact_target": frozenset({"import-targets", "quality-select-dev", "replay"}),
    "confirmation": frozenset(),
}

_OPERATION_PATH_CLASSES: dict[str, frozenset[str]] = {
    "open-read-complete": frozenset(set(_STAGE_ACCESS) - {"confirmation"}),
    "authorize-create-immutable": frozenset(
        {"artifact_runtime", "artifact_stream", "artifact_target"}
    ),
    "create-immutable-complete": frozenset(
        {"artifact_runtime", "artifact_stream", "artifact_target"}
    ),
    "resume-verify-immutable-complete": frozenset(
        {"artifact_runtime", "artifact_stream", "artifact_target"}
    ),
    "commit-immutable-cell-directory": frozenset({"artifact_stream"}),
}

_EXPLICIT_SOURCES = (
    "pyproject.toml",
    "benchmarks/__init__.py",
    "tasks/COMP-008-mean-conditioned-entropy-oracle.md",
    "tasks/COMP-009-ssp2e-actual-coder.md",
    "tasks/COMP-010-ssp2e-captured-replay-repair.md",
    "tasks/COMP-011-complete-stream-rgb-vq.md",
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
    "benchmarks/ssp2e_replay_repair.py",
    "benchmarks/ssp2v_actual_coder.py",
    "benchmarks/ssp2v_container.py",
    "benchmarks/ssp2v_decode_worker.py",
    "benchmarks/ssp2v_execute.py",
    "benchmarks/ssp2v_landlock.py",
    "benchmarks/ssp2v_lloyd.py",
    "benchmarks/ssp2v_lloyd_ext.cpp",
    "benchmarks/ssp2v_quality.py",
    "benchmarks/ssp2v_actual_run.py",
    "tests/test_ssp2e_arithmetic.py",
    "tests/test_ssp2e_container.py",
    "tests/test_ssp2e_execute.py",
    "tests/test_ssp2e_decode_worker.py",
    "tests/test_ssp2e_actual_coder.py",
    "tests/test_ssp2e_actual_run.py",
    "tests/test_ssp2e_replay_repair.py",
    "tests/test_ssp2v_actual_coder.py",
    "tests/test_ssp2v_container.py",
    "tests/test_ssp2v_decode_worker.py",
    "tests/test_ssp2v_execute.py",
    "tests/test_ssp2v_landlock.py",
    "tests/test_ssp2v_lloyd.py",
    "tests/test_ssp2v_quality.py",
    "tests/test_ssp2v_actual_run.py",
    "tests/test_ssp2v_actual_run_producer.py",
    "tests/test_ssp2v_actual_run_preflight_policy.py",
    "tests/test_ssp2v_actual_run_recovery.py",
    "tests/test_ssp2v_actual_run_replay.py",
    "tests/test_ssp2v_actual_run_stage_resume.py",
    "tests/test_ssp2v_actual_run_fresh_broker.py",
)

_FOCUSED_TESTS = tuple(
    f"tests/test_ssp2v_{name}.py"
    for name in (
        "lloyd",
        "container",
        "quality",
        "decode_worker",
        "actual_coder",
        "execute",
        "landlock",
        "actual_run",
        "actual_run_producer",
        "actual_run_preflight_policy",
        "actual_run_recovery",
        "actual_run_replay",
        "actual_run_stage_resume",
        "actual_run_fresh_broker",
    )
)

# This probe intentionally observes the unrestricted host environment.  lscpu
# enumerates the CPU sysfs tree and /proc rather than reading only its final
# leaf files, so running it inside the focused proof would either measure a
# different restricted environment or require excessive read authority.  It
# remains an ordinary mandatory test in the independent outer regression;
# only this nested focused invocation deselects it.
_FOCUSED_HOST_PROBE_DESELECTS = (
    "test_ssp2v_actual_run.py::test_live_environment_binding_is_immediately_repeatable",
    "test_ssp2v_actual_run.py::test_bound_hardware_environment_replays_without_gpu_device_authority",
)


class LifecycleError(RuntimeError):
    """A lifecycle, provenance, or fail-closed access condition failed."""


def _canonical_json(value: Any) -> bytes:
    return actual.canonical_json(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_regular_no_follow(path))


def _read_regular_no_follow(path: Path) -> bytes:
    lexical = _lexical(path)
    _reject_symlink_components(lexical)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(lexical, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise LifecycleError("regular-file identity changed during no-follow read")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LifecycleError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _safe_relative_path(value: Any, name: str = "artifact-relative path") -> Path:
    """Return one canonical POSIX relative path with no traversal aliases."""

    if not isinstance(value, (str, os.PathLike)):
        raise LifecycleError(f"{name} must be a relative path string")
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise LifecycleError(f"{name} is empty or noncanonical")
    relative = Path(raw)
    if (
        relative.is_absolute()
        or not relative.parts
        or raw != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise LifecycleError(f"{name} is absolute, traversing, or noncanonical")
    return relative


def _artifact_path(outdir: Path, value: Any, name: str = "artifact-relative path") -> Path:
    root = _lexical(outdir)
    relative = _safe_relative_path(value, name)
    candidate = _lexical(root / relative)
    try:
        if candidate.relative_to(root) != relative:
            raise LifecycleError(f"{name} escapes the artifact root")
    except ValueError as exc:
        raise LifecycleError(f"{name} escapes the artifact root") from exc
    return candidate


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        _lexical(directory),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_parents_no_symlink(directory: Path) -> None:
    directory = _lexical(directory)
    current = Path(directory.anchor)
    for part in directory.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o755)
                _fsync_directory(current.parent)
            except FileExistsError:
                pass
            info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise LifecycleError(f"directory path contains a symlink or non-directory: {current}")


def _pending_publication_path(path: Path, payload: bytes) -> Path:
    return path.parent / f".pending-{path.name}-{_sha256_bytes(payload)}"


def _recover_pending_directory(directory: Path) -> set[Path]:
    """Finish or discard only cryptographically named interrupted publications."""

    directory = _lexical(directory)
    if not directory.exists():
        return set()
    _reject_symlink_components(directory, require_regular=False)
    recovered: set[Path] = set()
    pending_paths = sorted(
        (path for path in directory.iterdir() if path.name.startswith(".pending-")),
        key=lambda value: value.name,
    )
    for pending in pending_paths:
        match = re.fullmatch(r"\.pending-(.+)-([0-9a-f]{64})", pending.name)
        if match is None or "/" in match.group(1) or match.group(1).startswith(".pending-"):
            raise LifecycleError("artifact directory contains a malformed pending publication")
        target = directory / match.group(1)
        info = os.lstat(pending)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise LifecycleError("pending publication is not a regular no-link file")
        payload = _read_regular_no_follow(pending)
        digest_matches = _sha256_bytes(payload) == match.group(2)
        if info.st_mode & 0o222:
            # The only writable state is the pre-publication write window.  It
            # is never an authority and is safe to restart from its bound name.
            if info.st_nlink != 1:
                raise LifecycleError("writable pending publication has an external hard link")
            pending.unlink()
            _fsync_directory(directory)
            continue
        if not digest_matches:
            raise LifecycleError("readonly pending publication digest drift")
        if target.exists():
            target_info = _reject_symlink_components(target)
            if (
                target_info.st_mode & 0o222
                or _read_regular_no_follow(target) != payload
            ):
                raise LifecycleError("pending publication conflicts with final artifact")
            same_inode = (info.st_dev, info.st_ino) == (
                target_info.st_dev,
                target_info.st_ino,
            )
            if (same_inode and (info.st_nlink != 2 or target_info.st_nlink != 2)) or (
                not same_inode and (info.st_nlink != 1 or target_info.st_nlink != 1)
            ):
                raise LifecycleError("pending publication has an external hard link")
        else:
            if info.st_nlink != 1:
                raise LifecycleError("pending publication has an external hard link")
            try:
                os.link(pending, target, follow_symlinks=False)
            except FileExistsError:
                if _read_regular_no_follow(target) != payload:
                    raise LifecycleError("concurrent immutable publication drift") from None
            _fsync_directory(directory)
        recovered.add(target)
        pending.unlink()
        _fsync_directory(directory)
    return recovered


def _exclusive_write(path: Path, payload: bytes, *, readonly: bool = True) -> None:
    path = _lexical(path)
    _mkdir_parents_no_symlink(path.parent)
    _reject_symlink_components(path.parent, require_regular=False)
    recovered = _recover_pending_directory(path.parent)
    if path in recovered:
        info = _reject_symlink_components(path)
        if info.st_mode & 0o222 or _read_regular_no_follow(path) != payload:
            raise LifecycleError("recovered immutable publication differs from requested bytes")
        return
    pending = _pending_publication_path(path, payload)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(pending, flags, 0o600)
    except FileExistsError as exc:
        raise LifecycleError(f"pending immutable publication collision for {path}") from exc
    linked = False
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise LifecycleError("pending artifact destination is not a new regular file")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if readonly:
            os.chmod(pending, 0o444)
        _fsync_directory(path.parent)
        try:
            os.link(pending, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise LifecycleError(f"refusing to replace existing artifact {path}") from exc
        linked = True
        _fsync_directory(path.parent)
        pending.unlink()
        _fsync_directory(path.parent)
    except Exception:
        if not linked:
            pending.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        raise
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        or named.st_nlink != 1
    ):
        raise LifecycleError("exclusive artifact identity changed during creation")
    _fsync_directory(path.parent)


def _reject_symlink_components(path: Path, *, require_regular: bool = True) -> os.stat_result:
    candidate = _lexical(path)
    current = Path(candidate.anchor)
    info = os.lstat(current)
    for part in candidate.parts[1:]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError as exc:
            raise LifecycleError(f"registered path component does not exist: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise LifecycleError(f"symlink access is forbidden: {current}")
        if current != candidate and not stat.S_ISDIR(info.st_mode):
            raise LifecycleError(f"non-directory path component: {current}")
    if require_regular and not stat.S_ISREG(info.st_mode):
        raise LifecycleError(f"authorized read is not a regular file: {candidate}")
    return info


class EventJournal:
    """O_EXCL one-file-per-event hash chain with immutable prefix semantics."""

    def __init__(self, root: Path, *, resume: bool = False) -> None:
        self.root = _lexical(root)
        if os.path.lexists(self.root):
            _reject_symlink_components(self.root, require_regular=False)
        if self.root.exists():
            if not resume:
                raise LifecycleError("journal already exists; explicit resume is required")
        else:
            _mkdir_parents_no_symlink(self.root)
        _reject_symlink_components(self.root, require_regular=False)
        events = self.verify(self.root)
        self._next = len(events)
        self._previous = events[-1]["event_sha256"] if events else "0" * 64
        self._stage_seals: dict[str, str] = {}

    def bind_stage_seal(self, stage: str, digest: str) -> None:
        if stage not in STAGES:
            raise LifecycleError("cannot bind journal seal for an unknown stage")
        normalized = _require_sha256(digest, f"{stage} journal seal")
        existing = self._stage_seals.get(stage)
        if existing is not None and existing != normalized:
            raise LifecycleError("a journal stage seal cannot be rebound")
        self._stage_seals[stage] = normalized

    @staticmethod
    def verify(root: Path) -> list[dict[str, Any]]:
        root = _lexical(root)
        if not root.exists():
            return []
        _reject_symlink_components(root, require_regular=False)
        _recover_pending_directory(root)
        names = sorted(path.name for path in root.iterdir())
        expected_names = [f"{index:08d}.json" for index in range(len(names))]
        if names != expected_names:
            raise LifecycleError("journal inventory is non-contiguous or contains an extra file")
        previous = "0" * 64
        records: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            path = root / name
            info = _reject_symlink_components(path)
            if info.st_mode & 0o222:
                raise LifecycleError("journal event is not immutable")
            try:
                payload = _read_regular_no_follow(path)
                record = json.loads(payload)
            except (OSError, json.JSONDecodeError) as exc:
                raise LifecycleError("journal event is not canonical JSON") from exc
            if not isinstance(record, dict) or payload != _canonical_json(record):
                raise LifecycleError("journal event is not a canonical JSON object")
            digest = record.get("event_sha256")
            unsigned = dict(record)
            unsigned.pop("event_sha256", None)
            if (
                record.get("schema") != JOURNAL_SCHEMA
                or record.get("sequence") != index
                or record.get("previous_event_sha256") != previous
                or digest != _sha256_bytes(_canonical_json(unsigned))
            ):
                raise LifecycleError("journal hash chain is invalid")
            stage = record.get("stage")
            path_class = record.get("path_class")
            operation = record.get("operation")
            prior = record.get("prior_stage_seal")
            if (
                stage not in STAGES
                or path_class not in _STAGE_ACCESS
                or stage not in _STAGE_ACCESS[path_class]
                or path_class == "confirmation"
                or operation not in _OPERATION_PATH_CLASSES
                or path_class not in _OPERATION_PATH_CLASSES[operation]
                or (stage == "preflight" and prior is not None)
            ):
                raise LifecycleError("journal event semantics are invalid")
            if stage != "preflight":
                _require_sha256(prior, "journal predecessor seal")
            receipt_fields = {"artifact_sha256", "artifact_bytes"} & set(record)
            if receipt_fields and receipt_fields != {"artifact_sha256", "artifact_bytes"}:
                raise LifecycleError("journal artifact receipt is incomplete")
            if receipt_fields:
                _require_sha256(record["artifact_sha256"], "journal artifact SHA-256")
                if (
                    isinstance(record["artifact_bytes"], bool)
                    or not isinstance(record["artifact_bytes"], int)
                    or record["artifact_bytes"] <= 0
                ):
                    raise LifecycleError("journal artifact receipt byte count is invalid")
            previous = str(digest)
            records.append(record)
        return records

    @property
    def head(self) -> str:
        return self._previous

    @property
    def length(self) -> int:
        return self._next

    def append(
        self,
        *,
        stage: str,
        operation: str,
        path_class: str,
        purpose: str,
        path: str,
        prior_stage_seal: str | None = None,
        artifact_sha256: str | None = None,
        artifact_bytes: int | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGES or path_class not in _STAGE_ACCESS:
            raise LifecycleError("unknown journal stage or path class")
        if stage not in _STAGE_ACCESS[path_class] or path_class == "confirmation":
            raise LifecycleError("journal path class is forbidden for this stage")
        if operation not in _OPERATION_PATH_CLASSES:
            raise LifecycleError("unknown journal operation")
        if path_class not in _OPERATION_PATH_CLASSES[operation]:
            raise LifecycleError("journal operation is illegal for its path class")
        if stage == "preflight":
            if prior_stage_seal is not None:
                raise LifecycleError("preflight journal event cannot claim a predecessor")
        else:
            previous = STAGES[STAGES.index(stage) - 1]
            expected = self._stage_seals.get(previous)
            if expected is None or prior_stage_seal != expected:
                raise LifecycleError(
                    f"{stage} journal event requires exact bound {previous} seal"
                )
        if (artifact_sha256 is None) != (artifact_bytes is None):
            raise LifecycleError("journal artifact receipt hash/bytes must be supplied together")
        if artifact_sha256 is not None:
            _require_sha256(artifact_sha256, "journal artifact SHA-256")
            if (
                isinstance(artifact_bytes, bool)
                or not isinstance(artifact_bytes, int)
                or artifact_bytes <= 0
            ):
                raise LifecycleError("journal artifact receipt bytes must be positive")
        record: dict[str, Any] = {
            "schema": JOURNAL_SCHEMA,
            "sequence": self._next,
            "stage": stage,
            "operation": operation,
            "path_class": path_class,
            "purpose": purpose,
            "path": path,
            "prior_stage_seal": prior_stage_seal,
            "previous_event_sha256": self._previous,
        }
        if artifact_sha256 is not None:
            record["artifact_sha256"] = artifact_sha256
            record["artifact_bytes"] = artifact_bytes
        record["event_sha256"] = _sha256_bytes(_canonical_json(record))
        _exclusive_write(self.root / f"{self._next:08d}.json", _canonical_json(record))
        self._next += 1
        self._previous = record["event_sha256"]
        return record


class AccessGuard:
    """Exact registered-path authorization; path classes are never guessed."""

    def __init__(self, journal: EventJournal, *, artifact_root: Path | None = None) -> None:
        self.journal = journal
        self.artifact_root = _lexical(artifact_root) if artifact_root is not None else None
        self._registered: dict[str, str] = {}
        self._stage_seals: dict[str, str] = {}
        self._candidate_set_seal: str | None = None
        self._target_copy_seal: str | None = None

    def bind_stage_seal(self, stage: str, digest: str) -> None:
        if stage not in STAGES:
            raise LifecycleError("cannot bind a seal for an unknown stage")
        normalized = _require_sha256(digest, f"{stage} seal")
        existing = self._stage_seals.get(stage)
        if existing is not None and existing != normalized:
            raise LifecycleError("a verified stage seal cannot be rebound")
        self._stage_seals[stage] = normalized
        self.journal.bind_stage_seal(stage, normalized)

    def bind_candidate_set_seal(self, digest: str) -> None:
        normalized = _require_sha256(digest, "candidate-set seal")
        if self._candidate_set_seal is not None and self._candidate_set_seal != normalized:
            raise LifecycleError("candidate-set seal cannot be rebound")
        self._candidate_set_seal = normalized

    def bind_target_copy_seal(self, digest: str) -> None:
        normalized = _require_sha256(digest, "target-copy seal")
        if self._target_copy_seal is not None and self._target_copy_seal != normalized:
            raise LifecycleError("target-copy seal cannot be rebound")
        self._target_copy_seal = normalized

    def register(self, path: Path, path_class: str) -> None:
        if path_class not in _STAGE_ACCESS:
            raise LifecycleError(f"unknown path class {path_class!r}")
        key = os.fspath(_lexical(path))
        existing = self._registered.get(key)
        if existing is not None and existing != path_class:
            raise LifecycleError("one lexical path cannot have two access classes")
        self._registered[key] = path_class

    def _label(self, path: Path) -> str:
        lexical = _lexical(path)
        for label, root in (("repository", ROOT), ("artifact", self.artifact_root)):
            if root is None:
                continue
            try:
                relative = lexical.relative_to(_lexical(root)).as_posix()
            except ValueError:
                continue
            return f"{label}:{relative}"
        return f"external-sha256:{_sha256_bytes(os.fsencode(lexical))}"

    def authorize(
        self,
        path: Path,
        *,
        stage: str,
        purpose: str,
        candidate_seal: str | None = None,
        target_seal: str | None = None,
        prior_stage_seal: str | None = None,
    ) -> str:
        lexical = _lexical(path)
        path_class = self._registered.get(os.fspath(lexical))
        if path_class is None:
            raise LifecycleError(f"path was not explicitly registered: {self._label(lexical)}")
        if stage not in STAGES or stage not in _STAGE_ACCESS[path_class]:
            raise LifecycleError(f"{path_class} access is forbidden during {stage}")
        if path_class == "confirmation":
            raise LifecycleError("confirmation access is forbidden in COMP-011")
        if stage != "preflight":
            previous_stage = STAGES[STAGES.index(stage) - 1]
            expected_prior = self._stage_seals.get(previous_stage)
            if expected_prior is None or prior_stage_seal != expected_prior:
                raise LifecycleError(
                    f"{stage} access requires the exact verified {previous_stage} seal"
                )
        elif prior_stage_seal is not None:
            raise LifecycleError("preflight events cannot claim a prior-stage seal")
        if path_class in {"upstream_target", "artifact_target"} and (
            self._candidate_set_seal is None or candidate_seal != self._candidate_set_seal
        ):
            raise LifecycleError("target access requires the exact verified candidate-set seal")
        if path_class == "artifact_target" and stage != "import-targets" and (
            self._target_copy_seal is None or target_seal != self._target_copy_seal
        ):
            raise LifecycleError("artifact target access requires the exact target-copy seal")
        _reject_symlink_components(lexical)
        return path_class

    def read_bytes(self, path: Path, **authorization: Any) -> bytes:
        path_class = self.authorize(path, **authorization)
        lexical = _lexical(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lexical, flags)
        try:
            opened = os.fstat(descriptor)
            named = os.stat(lexical, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise LifecycleError("authorized path identity changed during open")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                payload = handle.read()
        finally:
            os.close(descriptor)
        self.journal.append(
            stage=str(authorization["stage"]),
            operation="open-read-complete",
            path_class=path_class,
            purpose=str(authorization["purpose"]),
            path=self._label(lexical),
            prior_stage_seal=authorization.get("prior_stage_seal"),
        )
        return payload

    def read_json(self, path: Path, **authorization: Any) -> dict[str, Any]:
        try:
            value = json.loads(self.read_bytes(path, **authorization))
        except json.JSONDecodeError as exc:
            raise LifecycleError(f"authorized metadata is not JSON: {self._label(path)}") from exc
        if not isinstance(value, dict):
            raise LifecycleError("authorized metadata must be a JSON object")
        return value


def _resolve_local_module(module: str) -> Path | None:
    parts = module.split(".")
    if parts[0] == "benchmarks":
        base = ROOT.joinpath(*parts)
    elif parts[0] == "structsplat":
        base = ROOT / "src" / Path(*parts)
    else:
        return None
    py = base.with_suffix(".py")
    package = base / "__init__.py"
    if py.is_file():
        return py
    if package.is_file():
        return package
    return None


def _source_package_parts(path: Path) -> list[str]:
    relative = path.relative_to(ROOT)
    parts = list(relative.parts)
    if parts[:1] == ["src"]:
        parts = parts[1:]
    if parts[-1] == "__init__.py":
        return parts[:-1]
    return parts[:-1]


def _transitive_python_sources(
    seed: Iterable[str], *, guard: AccessGuard | None = None
) -> set[str]:
    pending = [ROOT / value for value in seed if value.endswith(".py")]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            if guard is None:
                source_text = _read_regular_no_follow(path).decode("utf-8")
            else:
                guard.register(path, "repository_source")
                source_text = guard.read_bytes(
                    path,
                    stage="preflight",
                    purpose="discover transitive local source imports",
                ).decode("utf-8")
            tree = ast.parse(source_text, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise LifecycleError(f"cannot parse source dependency {path}") from exc
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    base = node.module or ""
                else:
                    package = _source_package_parts(path)
                    ascend = node.level - 1
                    if ascend > len(package):
                        raise LifecycleError(
                            f"relative import escapes local package in source dependency {path}"
                        )
                    prefix = package[: len(package) - ascend]
                    suffix = node.module.split(".") if node.module else []
                    base = ".".join([*prefix, *suffix])
                if base:
                    modules.append(base)
                    modules.extend(f"{base}.{alias.name}" for alias in node.names)
            for module in modules:
                dependency = _resolve_local_module(module)
                if dependency is not None and dependency not in seen:
                    pending.append(dependency)
    return {path.relative_to(ROOT).as_posix() for path in seen}


def source_paths(*, guard: AccessGuard | None = None) -> tuple[Path, ...]:
    """Return explicit roots, discovered local imports, and the production tree."""
    relative = set(_EXPLICIT_SOURCES)
    relative.update(_transitive_python_sources(relative, guard=guard))
    production_root = ROOT / "src/structsplat"
    for path in production_root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and not path.is_symlink():
            relative.add(path.relative_to(ROOT).as_posix())
    paths = tuple(ROOT / name for name in sorted(relative))
    missing = [path for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise LifecycleError(f"source closure is incomplete or unsafe: {missing}")
    task = ROOT / "tasks/COMP-011-complete-stream-rgb-vq.md"
    if guard is None:
        task_payload = _read_regular_no_follow(task)
    else:
        guard.register(task, "repository_source")
        task_payload = guard.read_bytes(
            task, stage="preflight", purpose="verify frozen COMP-011 task digest"
        )
    if _sha256_bytes(task_payload) != TASK_SHA256:
        raise LifecycleError("COMP-011 task source differs from the frozen task SHA-256")
    return paths


def source_manifest(
    paths: Iterable[Path] | None = None, *, guard: AccessGuard | None = None
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    explicit = set(_EXPLICIT_SOURCES)
    for path in paths or source_paths(guard=guard):
        lexical = _lexical(path)
        try:
            relative = lexical.relative_to(_lexical(ROOT)).as_posix()
        except ValueError as exc:
            raise LifecycleError("source closure contains a path outside the repository") from exc
        if guard is not None:
            guard.register(lexical, "repository_source")
            payload = guard.read_bytes(
                lexical, stage="preflight", purpose="hash source closure"
            )
        else:
            _reject_symlink_components(lexical)
            payload = _read_regular_no_follow(lexical)
        records.append(
            {
                "path": relative,
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "closure": "explicit" if relative in explicit else "transitive-or-production",
            }
        )
    record = {
        "schema": SOURCE_SCHEMA,
        "task_sha256": TASK_SHA256,
        "files": sorted(records, key=lambda item: item["path"]),
    }
    return actual.seal_record(record, "source_manifest_sha256")


def _source_payload(record: Mapping[str, Any], guard: AccessGuard | None) -> bytes:
    path = ROOT / str(record["path"])
    if guard is None:
        return _read_regular_no_follow(path)
    return guard.read_bytes(path, stage="preflight", purpose="archive source closure")


def write_source_archive(
    path: Path, manifest: Mapping[str, Any], *, guard: AccessGuard | None = None
) -> None:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
        entries = [("SOURCE_MANIFEST.json", _canonical_json(manifest), 0o444)]
        entries.extend(
            (str(item["path"]), _source_payload(item, guard), 0o444)
            for item in manifest["files"]
        )
        for name, payload, mode in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = mode
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    _exclusive_write(path, output.getvalue())


def safe_extract_source_archive(
    archive_path: Path,
    destination: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    guard: AccessGuard | None = None,
    stage: str = "preflight",
) -> dict[str, Any]:
    """Extract only exact regular manifest members; never use ``extractall``."""
    if guard is None:
        archive_payload = _read_regular_no_follow(archive_path)
    else:
        guard.register(archive_path, "artifact_runtime")
        archive_payload = guard.read_bytes(
            archive_path, stage=stage, purpose="verify and relocate source archive"
        )
    with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise LifecycleError("source archive contains a duplicate member")
        for member in members:
            name = Path(member.name)
            if (
                name.is_absolute()
                or any(part in ("", ".", "..") for part in name.parts)
                or not member.isfile()
                or member.size < 0
            ):
                raise LifecycleError("source archive contains an unsafe member")
        manifest_member = next(
            (member for member in members if member.name == "SOURCE_MANIFEST.json"), None
        )
        if manifest_member is None:
            raise LifecycleError("source archive has no embedded manifest")
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise LifecycleError("cannot read embedded source manifest")
        try:
            manifest = json.loads(manifest_file.read())
        except json.JSONDecodeError as exc:
            raise LifecycleError("embedded source manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise LifecycleError("embedded source manifest is not an object")
        actual.verify_record(manifest, "source_manifest_sha256")
        if expected_manifest is not None and manifest != dict(expected_manifest):
            raise LifecycleError("archive manifest differs from expected source manifest")
        records = {str(item["path"]): item for item in manifest.get("files", [])}
        if set(names) != {"SOURCE_MANIFEST.json", *records}:
            raise LifecycleError("source archive inventory differs from its manifest")
        destination = _lexical(destination)
        if destination.exists() and any(destination.iterdir()):
            raise LifecycleError("source extraction destination must be empty")
        _mkdir_parents_no_symlink(destination)
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise LifecycleError("cannot read regular source member")
            payload = extracted.read()
            if member.name != "SOURCE_MANIFEST.json":
                expected = records[member.name]
                if len(payload) != expected["bytes"] or _sha256_bytes(payload) != expected["sha256"]:
                    raise LifecycleError("source archive member hash/size differs from manifest")
            output_path = destination.joinpath(*Path(member.name).parts)
            try:
                output_path.relative_to(destination)
            except ValueError as exc:  # pragma: no cover - lexical checks above establish this
                raise LifecycleError("source member escapes destination") from exc
            _exclusive_write(output_path, payload)
    _freeze_runtime_source_tree(destination)
    return manifest


def _freeze_runtime_source_tree(source_root: Path) -> None:
    paths = [source_root, *(path for path in source_root.rglob("*") if path.is_dir())]
    for path in sorted(paths, key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise LifecycleError("runtime source tree contains a symlink directory")
        os.chmod(path, 0o555)


def _verify_runtime_source_tree(
    source_root: Path, *, expected_manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    source_root = _lexical(source_root)
    root_info = _reject_symlink_components(source_root, require_regular=False)
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_IMODE(root_info.st_mode) != 0o555:
        raise LifecycleError("runtime source root is not an immutable directory")
    manifest_path = source_root / "SOURCE_MANIFEST.json"
    manifest_payload = _read_regular_no_follow(manifest_path)
    try:
        manifest = json.loads(manifest_payload)
    except json.JSONDecodeError as exc:
        raise LifecycleError("runtime source embedded manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest_payload != _canonical_json(manifest)
    ):
        raise LifecycleError("runtime source embedded manifest is not canonical")
    actual.verify_record(manifest, "source_manifest_sha256")
    if expected_manifest is not None and manifest != dict(expected_manifest):
        raise LifecycleError("runtime source manifest differs from its authority")
    records = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    if len(records) != len(manifest.get("files", [])):
        raise LifecycleError("runtime source manifest file inventory is malformed")
    expected_files = {"SOURCE_MANIFEST.json", *records}
    expected_directories = {
        parent.as_posix()
        for name in expected_files
        for parent in Path(name).parents
        if parent != Path(".")
    }
    observed_files: dict[str, Path] = {}
    observed_directories: set[str] = set()
    for path in source_root.rglob("*"):
        relative = path.relative_to(source_root).as_posix()
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise LifecycleError("runtime source tree contains a symlink")
        if stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o444:
                raise LifecycleError("runtime source file mode/link identity drift")
            observed_files[relative] = path
        elif stat.S_ISDIR(info.st_mode):
            if stat.S_IMODE(info.st_mode) != 0o555:
                raise LifecycleError("runtime source directory became writable")
            observed_directories.add(relative)
        else:
            raise LifecycleError("runtime source tree contains a non-file entry")
    if set(observed_files) != expected_files or observed_directories != expected_directories:
        raise LifecycleError("runtime source tree inventory drift")
    for relative, record in records.items():
        payload = _read_regular_no_follow(observed_files[relative])
        if len(payload) != record.get("bytes") or _sha256_bytes(payload) != record.get("sha256"):
            raise LifecycleError("runtime source file differs from its manifest")
    return manifest


def _runtime_source_record(
    source_root: Path,
    *,
    outdir: Path,
    manifest: Mapping[str, Any],
    archive_sha256: str,
) -> dict[str, Any]:
    verified = _verify_runtime_source_tree(source_root, expected_manifest=manifest)
    by_path = {str(item["path"]): item for item in verified["files"]}
    runner = by_path["benchmarks/ssp2v_actual_run.py"]
    launcher = by_path["benchmarks/ssp2v_landlock.py"]
    return actual.seal_record(
        {
            "schema": RUNTIME_SOURCE_SCHEMA,
            "relative_root": source_root.relative_to(outdir).as_posix(),
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "source_archive_sha256": archive_sha256,
            "source_file_count": len(manifest["files"]),
            "embedded_manifest_bytes": len(_canonical_json(manifest)),
            "embedded_manifest_sha256": _sha256_bytes(_canonical_json(manifest)),
            "runner": {
                "path": "benchmarks/ssp2v_actual_run.py",
                "bytes": runner["bytes"],
                "sha256": runner["sha256"],
            },
            "launcher": {
                "path": "benchmarks/ssp2v_landlock.py",
                "bytes": launcher["bytes"],
                "sha256": launcher["sha256"],
            },
            "file_mode": 0o444,
            "directory_mode": 0o555,
            "all_files_single_link": True,
            "inventory_reverified": True,
        },
        "runtime_source_sha256",
    )


def _verify_runtime_source_record(
    outdir: Path,
    record: Mapping[str, Any],
    *,
    expected_manifest: Mapping[str, Any],
    expected_archive_sha256: str,
) -> Path:
    actual.verify_record(record, "runtime_source_sha256")
    if set(record) != {
        "schema",
        "relative_root",
        "source_manifest_sha256",
        "source_archive_sha256",
        "source_file_count",
        "embedded_manifest_bytes",
        "embedded_manifest_sha256",
        "runner",
        "launcher",
        "file_mode",
        "directory_mode",
        "all_files_single_link",
        "inventory_reverified",
        "runtime_source_sha256",
    } or record.get("schema") != RUNTIME_SOURCE_SCHEMA:
        raise LifecycleError("runtime source record schema drift")
    source_root = _artifact_path(
        outdir, record.get("relative_root"), "runtime source root"
    )
    if source_root != outdir / _RUNTIME_SOURCE_RELATIVE:
        raise LifecycleError("runtime source root path drift")
    manifest = _verify_runtime_source_tree(
        source_root, expected_manifest=expected_manifest
    )
    by_path = {str(item["path"]): item for item in manifest["files"]}
    expected_runner = by_path["benchmarks/ssp2v_actual_run.py"]
    expected_launcher = by_path["benchmarks/ssp2v_landlock.py"]
    if (
        record.get("source_manifest_sha256")
        != expected_manifest.get("source_manifest_sha256")
        or record.get("source_archive_sha256") != expected_archive_sha256
        or record.get("source_file_count") != len(expected_manifest["files"])
        or record.get("embedded_manifest_bytes") != len(_canonical_json(expected_manifest))
        or record.get("embedded_manifest_sha256")
        != _sha256_bytes(_canonical_json(expected_manifest))
        or record.get("runner")
        != {
            "path": "benchmarks/ssp2v_actual_run.py",
            "bytes": expected_runner["bytes"],
            "sha256": expected_runner["sha256"],
        }
        or record.get("launcher")
        != {
            "path": "benchmarks/ssp2v_landlock.py",
            "bytes": expected_launcher["bytes"],
            "sha256": expected_launcher["sha256"],
        }
        or record.get("file_mode") != 0o444
        or record.get("directory_mode") != 0o555
        or record.get("all_files_single_link") is not True
        or record.get("inventory_reverified") is not True
    ):
        raise LifecycleError("runtime source authority drift")
    return source_root


def _verify_runtime_source_worker_policy(
    bundle: Mapping[str, Any],
    *,
    outdir: Path,
    runtime_source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    archive_sha256: str,
) -> None:
    source_root = _verify_runtime_source_record(
        outdir,
        runtime_source,
        expected_manifest=manifest,
        expected_archive_sha256=archive_sha256,
    )
    launch = bundle.get("launch")
    policy = launch.get("worker_policy") if isinstance(launch, Mapping) else None
    if not isinstance(policy, Mapping):
        raise LifecycleError("runtime-source worker policy is absent")
    environment = policy.get("environment")
    read_only = policy.get("read_only_request")
    expected_launcher = source_root / "benchmarks/ssp2v_landlock.py"
    forbidden_live_paths = {
        os.fspath(_lexical(ROOT)),
        os.fspath(_lexical(ROOT / "benchmarks")),
        os.fspath(_lexical(ROOT / "src")),
    }
    if (
        not isinstance(environment, Mapping)
        or not isinstance(read_only, list)
        or policy.get("cwd") != os.fspath(source_root / "benchmarks")
        or policy.get("launcher_path") != os.fspath(expected_launcher)
        or policy.get("launcher_sha256") != _sha256_file(expected_launcher)
        or _worker_policy_path(source_root) not in read_only
        or forbidden_live_paths & set(read_only)
        or environment.get("PYTHONPATH")
        != _worker_environment(outdir, source_root=source_root).get("PYTHONPATH")
        or environment.get("STRUCTSPLAT_SOURCE_MANIFEST_SHA256")
        != manifest.get("source_manifest_sha256")
        or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise LifecycleError("worker did not use the exact immutable runtime source")


def _preflight_worker_scratch(
    policy: Mapping[str, Any],
    *,
    prefix: str,
    forbidden_roots: Sequence[Path],
) -> Path:
    """Recover one deleted preflight scratch root under the frozen /tmp grammar."""

    environment = policy.get("environment")
    value = environment.get("TMPDIR") if isinstance(environment, Mapping) else None
    if not isinstance(value, str):
        raise LifecycleError("preflight worker scratch environment is absent")
    scratch = Path(value)
    if (
        not scratch.is_absolute()
        or _lexical(scratch) != scratch
        or scratch.parent != Path("/tmp")
        or re.fullmatch(rf"{re.escape(prefix)}[a-z0-9_]{{8}}", scratch.name) is None
        or os.path.lexists(scratch)
    ):
        raise LifecycleError("preflight worker scratch path grammar drift")
    for raw_root in forbidden_roots:
        root = _lexical(raw_root)
        if (
            scratch == root
            or scratch.is_relative_to(root)
            or root.is_relative_to(scratch)
        ):
            raise LifecycleError("preflight worker scratch overlaps protected authority")
    return scratch


def _expected_preflight_worker_policy(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    read_only: Sequence[Path],
    read_write: Sequence[Path],
    ephemeral_paths: Sequence[Path],
    launcher_path: Path,
    timeout: float,
) -> dict[str, Any]:
    ephemeral = {_lexical(path) for path in ephemeral_paths}
    launcher = launcher_path.resolve(strict=True)
    return actual.seal_record(
        {
            "schema": WORKER_POLICY_SCHEMA,
            "command": list(command),
            "cwd": cwd.resolve(strict=True).as_posix(),
            "environment": dict(environment),
            "read_only_request": sorted(
                {
                    _producer_policy_request_path(path)
                    for path in read_only
                }
            ),
            "read_write_request": sorted(
                {
                    _producer_policy_request_path(
                        path, allow_missing=_lexical(path) in ephemeral
                    )
                    for path in read_write
                }
            ),
            "denied_read_probe_paths": [],
            "launcher_path": launcher.as_posix(),
            "launcher_sha256": _sha256_file(launcher),
            "timeout_seconds": float(timeout),
        },
        "worker_policy_sha256",
    )


def _verify_exact_preflight_worker_policy(
    name: str,
    bundle: Mapping[str, Any],
    worker: Mapping[str, Any],
    *,
    outdir: Path,
    runtime_source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    archive_sha256: str,
    lloyd_path: Path,
    renderer_record_path: Path,
    forbidden_roots: Sequence[Path],
) -> None:
    """Recompute every authority-bearing field of a phase-1 worker launch."""

    _verify_runtime_source_worker_policy(
        bundle,
        outdir=outdir,
        runtime_source=runtime_source,
        manifest=manifest,
        archive_sha256=archive_sha256,
    )
    launch = bundle.get("launch")
    policy = launch.get("worker_policy") if isinstance(launch, Mapping) else None
    if not isinstance(policy, Mapping):
        raise LifecycleError("preflight worker policy is absent")

    source_root = outdir / _RUNTIME_SOURCE_RELATIVE
    launcher = source_root / "benchmarks/ssp2v_landlock.py"
    environment = _worker_environment(outdir, source_root=source_root)
    read_only = [*_sandbox_system_read_only(), *_worker_source_read_only(source_root)]
    read_write: list[Path] = []
    ephemeral: list[Path] = []
    timeout: float

    if name == "lloyd_proof":
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_lloyd-proof",
            "--native",
            os.fspath(lloyd_path),
        ]
        read_only.append(lloyd_path)
        timeout = 360.0
    elif name == "renderer_build":
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_build-renderer",
        ]
        scratch = _preflight_worker_scratch(
            policy,
            prefix="comp011-renderer-build-",
            forbidden_roots=forbidden_roots,
        )
        environment["TMPDIR"] = os.fspath(scratch)
        environment["TORCH_EXTENSIONS_DIR"] = os.fspath(scratch / "extensions")
        extension_value = worker.get("extension_path")
        extension = Path(extension_value) if isinstance(extension_value, str) else Path()
        if (
            not extension.is_absolute()
            or _lexical(extension) != extension
            or not extension.is_relative_to(scratch / "extensions")
        ):
            raise LifecycleError("renderer build output escaped its exact scratch authority")
        read_write = [scratch, *_sandbox_gpu_read_write()]
        ephemeral = [scratch]
        timeout = 1800.0
    elif name == "renderer_proof":
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_renderer-proof",
            "--artifact-root",
            os.fspath(outdir),
            "--renderer-record",
            os.fspath(renderer_record_path),
        ]
        scratch = _preflight_worker_scratch(
            policy,
            prefix="comp011-renderer-proof-",
            forbidden_roots=forbidden_roots,
        )
        environment["TMPDIR"] = os.fspath(scratch)
        read_only.append(outdir / "runtime")
        read_write = [scratch, *_sandbox_gpu_read_write()]
        ephemeral = [scratch]
        timeout = 1800.0
    else:
        raise LifecycleError("unknown preflight worker policy")

    expected = _expected_preflight_worker_policy(
        command=command,
        cwd=source_root / "benchmarks",
        environment=environment,
        read_only=read_only,
        read_write=read_write,
        ephemeral_paths=ephemeral,
        launcher_path=launcher,
        timeout=timeout,
    )
    if policy != expected:
        raise LifecycleError(f"{name} exact launch authority drift")


def _command_record(command: Sequence[str], *, timeout: float = 60.0) -> dict[str, Any]:
    completed = subprocess.run(
        list(command), check=False, capture_output=True, text=True, timeout=timeout
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_sha256": _sha256_bytes((completed.stdout + completed.stderr).encode()),
    }


def _nvidia_smi_executable_record() -> dict[str, Any]:
    discovered = shutil.which("nvidia-smi")
    if not discovered:
        raise LifecycleError("nvidia-smi executable is absent")
    try:
        executable = Path(discovered).resolve(strict=True)
        payload = _read_regular_no_follow(executable)
    except OSError as exc:
        raise LifecycleError("nvidia-smi executable identity cannot be read") from exc
    return {
        "path": os.fspath(executable),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _nvidia_smi_rows(stdout: str) -> list[dict[str, str]]:
    try:
        parsed = [
            [field.strip() for field in row]
            for row in csv.reader(io.StringIO(stdout, newline=""))
            if row
        ]
    except csv.Error as exc:
        raise LifecycleError("nvidia-smi inventory CSV is malformed") from exc
    if (
        not parsed
        or any(
            len(row) != len(_NVIDIA_SMI_QUERY_FIELDS)
            or any(not field for field in row)
            for row in parsed
        )
    ):
        raise LifecycleError("nvidia-smi inventory CSV is malformed")
    rows = [
        dict(zip(_NVIDIA_SMI_QUERY_FIELDS, row, strict=True))
        for row in parsed
    ]
    indices = [row["index"] for row in rows]
    uuids = [row["uuid"] for row in rows]
    if (
        any(re.fullmatch(r"0|[1-9][0-9]*", index) is None for index in indices)
        or len(indices) != len(set(indices))
        or len(uuids) != len(set(uuids))
    ):
        raise LifecycleError("nvidia-smi inventory identities are malformed")
    return rows


def _nvidia_smi_identity_record(
    command_record: Mapping[str, Any],
    *,
    executable: Mapping[str, Any],
) -> dict[str, Any]:
    command = [
        executable.get("path"),
        f"--query-gpu={','.join(_NVIDIA_SMI_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    stdout = command_record.get("stdout")
    stderr = command_record.get("stderr")
    if (
        command_record.get("command") != command
        or command_record.get("returncode") != 0
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or command_record.get("output_sha256")
        != _sha256_bytes((stdout + stderr).encode())
    ):
        raise LifecycleError("nvidia-smi hardware binding failed")
    identity = {
        "schema": NVIDIA_SMI_IDENTITY_SCHEMA,
        "command": command,
        "returncode": 0,
        "executable": dict(executable),
        "query_fields": list(_NVIDIA_SMI_QUERY_FIELDS),
        "rows": _nvidia_smi_rows(stdout),
        "stdout": stdout,
        "stderr": stderr,
        "output_sha256": command_record["output_sha256"],
    }
    identity["identity_sha256"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _bound_nvidia_smi_identity_record(
    record: Mapping[str, Any],
    *,
    executable: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise LifecycleError("bound nvidia-smi identity is absent")
    identity = dict(record)
    observed_sha256 = identity.pop("identity_sha256", None)
    if (
        not isinstance(observed_sha256, str)
        or len(observed_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in observed_sha256
        )
        or _sha256_bytes(_canonical_json(identity)) != observed_sha256
    ):
        raise LifecycleError("bound nvidia-smi identity seal mismatch")
    if identity.get("executable") != dict(executable):
        raise LifecycleError(
            "bound nvidia-smi identity differs from its live executable"
        )
    expected = _nvidia_smi_identity_record(
        {
            "command": identity.get("command"),
            "returncode": identity.get("returncode"),
            "stdout": identity.get("stdout"),
            "stderr": identity.get("stderr"),
            "output_sha256": identity.get("output_sha256"),
        },
        executable=executable,
    )
    if dict(record) != expected:
        raise LifecycleError(
            "bound nvidia-smi identity differs from its sealed inventory"
        )
    return expected


def compiler_record() -> dict[str, Any]:
    compiler = os.environ.get("CXX") or shutil.which("c++")
    nvcc = shutil.which("nvcc")
    if not compiler or not nvcc:
        raise LifecycleError("COMP-011 preflight requires C++ and nvcc compilers")
    return {
        "cxx": _lexical(Path(compiler)).as_posix(),
        "cxx_version": _command_record((compiler, "--version")),
        "nvcc": _lexical(Path(nvcc)).as_posix(),
        "nvcc_version": _command_record((nvcc, "--version")),
        "arithmetic_source_sha256": _sha256_file(arithmetic.NATIVE_SOURCE_PATH),
        "lloyd_source_sha256": _sha256_file(lloyd.NATIVE_SOURCE_PATH),
        "flags": ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared"],
    }


def _cpuinfo_identity_record(payload: bytes) -> dict[str, Any]:
    """Canonicalize stable ``/proc/cpuinfo`` identity without live clock telemetry."""

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LifecycleError("/proc/cpuinfo is not valid UTF-8") from exc
    blocks = [block for block in re.split(r"\n[ \t]*\n", text.strip()) if block.strip()]
    if not blocks:
        raise LifecycleError("/proc/cpuinfo contains no logical processors")
    logical_processors: list[tuple[int, dict[str, str]]] = []
    seen_processor_ids: set[int] = set()
    for block in blocks:
        fields: dict[str, str] = {}
        seen_fields: set[str] = set()
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            normalized_key = key.strip()
            if not separator or not normalized_key:
                raise LifecycleError("/proc/cpuinfo contains a malformed field")
            if normalized_key in seen_fields:
                raise LifecycleError(
                    f"/proc/cpuinfo contains duplicate field {normalized_key!r}"
                )
            seen_fields.add(normalized_key)
            if normalized_key not in _CPUINFO_VOLATILE_FIELDS:
                fields[normalized_key] = value.strip()
        if not _CPUINFO_VOLATILE_FIELDS <= seen_fields:
            raise LifecycleError(
                "/proc/cpuinfo required volatile telemetry field is absent"
            )
        processor = fields.get("processor")
        if processor is None or re.fullmatch(r"0|[1-9][0-9]*", processor) is None:
            raise LifecycleError("/proc/cpuinfo logical processor id is absent or malformed")
        processor_id = int(processor)
        if processor_id in seen_processor_ids:
            raise LifecycleError("/proc/cpuinfo contains a duplicate logical processor id")
        seen_processor_ids.add(processor_id)
        logical_processors.append((processor_id, fields))
    logical_processors.sort(key=lambda item: item[0])
    identity: dict[str, Any] = {
        "schema": CPUINFO_IDENTITY_SCHEMA,
        "excluded_volatile_fields": sorted(_CPUINFO_VOLATILE_FIELDS),
        "logical_processors": [fields for _, fields in logical_processors],
    }
    identity["identity_sha256"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _lscpu_executable_record() -> dict[str, Any]:
    discovered = shutil.which("lscpu")
    if not discovered:
        raise LifecycleError("lscpu executable is absent")
    try:
        executable = Path(discovered).resolve(strict=True)
        payload = _read_regular_no_follow(executable)
    except OSError as exc:
        raise LifecycleError("lscpu executable identity cannot be read") from exc
    version = _command_record((os.fspath(executable), "--version"))
    if version["returncode"] != 0:
        raise LifecycleError("lscpu executable version binding failed")
    return {
        "path": os.fspath(executable),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "version": version,
    }


def _lscpu_identity_record(
    command_record: Mapping[str, Any], *, executable: Mapping[str, Any]
) -> dict[str, Any]:
    """Canonicalize stable ``lscpu`` fields while excluding scaling telemetry."""

    if (
        command_record.get("command") != [executable.get("path"), "--json"]
        or command_record.get("returncode") != 0
        or not isinstance(command_record.get("stdout"), str)
        or not isinstance(command_record.get("stderr"), str)
    ):
        raise LifecycleError("lscpu hardware binding failed")
    try:
        payload = json.loads(str(command_record["stdout"]))
    except (TypeError, ValueError) as exc:
        raise LifecycleError("lscpu JSON is malformed") from exc
    if not isinstance(payload, dict) or set(payload) != {"lscpu"}:
        raise LifecycleError("lscpu JSON root is malformed")
    entries = payload["lscpu"]
    if not isinstance(entries, list) or not entries:
        raise LifecycleError("lscpu JSON contains no hardware fields")
    fields: dict[str, str | None] = {}
    seen_fields: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"field", "data"}
            or not isinstance(entry.get("field"), str)
            or not entry["field"]
            or (entry.get("data") is not None and not isinstance(entry.get("data"), str))
        ):
            raise LifecycleError("lscpu JSON contains a malformed field")
        field = str(entry["field"])
        if field in seen_fields:
            raise LifecycleError(f"lscpu JSON contains duplicate field {field!r}")
        seen_fields.add(field)
        if field not in _LSCPU_VOLATILE_FIELDS:
            fields[field] = entry.get("data")
    if not _LSCPU_VOLATILE_FIELDS <= seen_fields:
        raise LifecycleError("lscpu required volatile telemetry field is absent")
    if not fields:
        raise LifecycleError("lscpu JSON contains no stable hardware fields")
    identity = {
        "schema": LSCPU_IDENTITY_SCHEMA,
        "command": [executable["path"], "--json"],
        "returncode": 0,
        "executable": dict(executable),
        "excluded_volatile_fields": sorted(_LSCPU_VOLATILE_FIELDS),
        "fields": dict(sorted(fields.items())),
        "stderr": command_record["stderr"],
    }
    identity["identity_sha256"] = _sha256_bytes(_canonical_json(identity))
    return identity


def _bound_lscpu_identity_record(
    record: Mapping[str, Any], *, executable: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a sealed outer ``lscpu`` observation without re-enumerating sysfs."""

    if not isinstance(record, Mapping):
        raise LifecycleError("bound lscpu identity is absent")
    identity = dict(record)
    observed_sha256 = identity.pop("identity_sha256", None)
    if (
        not isinstance(observed_sha256, str)
        or len(observed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in observed_sha256)
        or _sha256_bytes(_canonical_json(identity)) != observed_sha256
    ):
        raise LifecycleError("bound lscpu identity seal mismatch")
    if set(identity) != {
        "schema",
        "command",
        "returncode",
        "executable",
        "excluded_volatile_fields",
        "fields",
        "stderr",
    }:
        raise LifecycleError("bound lscpu identity key set drift")
    fields = identity.get("fields")
    if (
        identity.get("schema") != LSCPU_IDENTITY_SCHEMA
        or identity.get("command") != [executable.get("path"), "--json"]
        or identity.get("returncode") != 0
        or identity.get("executable") != dict(executable)
        or identity.get("excluded_volatile_fields")
        != sorted(_LSCPU_VOLATILE_FIELDS)
        or not isinstance(fields, Mapping)
        or not fields
        or not all(
            isinstance(key, str)
            and key
            and (value is None or isinstance(value, str))
            for key, value in fields.items()
        )
        or not isinstance(identity.get("stderr"), str)
    ):
        raise LifecycleError("bound lscpu identity differs from its live executable")
    return {**identity, "identity_sha256": observed_sha256}


def environment_record(
    *,
    bound_lscpu: Mapping[str, Any] | None = None,
    bound_nvidia_smi: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (bound_lscpu is None) != (bound_nvidia_smi is None):
        raise LifecycleError(
            "bound hardware inventory must provide both lscpu and nvidia-smi"
        )
    packages: dict[str, str] = {}
    for distribution in ("numpy", "torch", "torchvision", "Pillow", "pytorch-msssim", "lpips"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise LifecycleError(f"required package is absent: {distribution}") from exc
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    nvidia_executable = _nvidia_smi_executable_record()
    nvidia = (
        _nvidia_smi_identity_record(
            _command_record(
                (
                    nvidia_executable["path"],
                    f"--query-gpu={','.join(_NVIDIA_SMI_QUERY_FIELDS)}",
                    "--format=csv,noheader,nounits",
                )
            ),
            executable=nvidia_executable,
        )
        if bound_nvidia_smi is None
        else _bound_nvidia_smi_identity_record(
            bound_nvidia_smi,
            executable=nvidia_executable,
        )
    )
    cpuinfo_payload = _read_regular_no_follow(Path("/proc/cpuinfo"))
    cpu_fields: dict[str, list[str]] = {"model name": [], "microcode": []}
    for line in cpuinfo_payload.decode("utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        normalized = key.strip()
        if separator and normalized in cpu_fields:
            cpu_fields[normalized].append(value.strip())
    cpuinfo_identity = _cpuinfo_identity_record(cpuinfo_payload)
    lscpu_executable = _lscpu_executable_record()
    lscpu = (
        _lscpu_identity_record(
            _command_record((lscpu_executable["path"], "--json")),
            executable=lscpu_executable,
        )
        if bound_lscpu is None
        else _bound_lscpu_identity_record(
            bound_lscpu,
            executable=lscpu_executable,
        )
    )
    return {
        "python": sys.version,
        "python_executable": os.fspath(_lexical(Path(sys.executable))),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "cpu_affinity": affinity,
        "cpu_count": os.cpu_count(),
        "cpu_model_names": sorted(set(cpu_fields["model name"])),
        "cpu_microcodes": sorted(set(cpu_fields["microcode"])),
        "proc_cpuinfo": cpuinfo_identity,
        "lscpu": lscpu,
        "zlib_compile_version": zlib.ZLIB_VERSION,
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
        "packages": packages,
        "thread_environment": quality.frozen_runtime_environment(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": nvidia,
    }


def repository_provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the dirty checkout and show which executed sources are untracked."""

    head = _command_record(("git", "rev-parse", "HEAD"))
    branch = _command_record(("git", "branch", "--show-current"))
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=60,
    )
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=60,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        timeout=60,
    )
    if any(
        item.returncode != 0
        for item in (status, tracked_diff, untracked)
    ) or head["returncode"] != 0 or branch["returncode"] != 0:
        raise LifecycleError("git repository provenance capture failed")
    untracked_paths = sorted(
        value.decode("utf-8", errors="strict")
        for value in untracked.stdout.split(b"\0")
        if value
    )
    source_by_path = {str(item["path"]): item for item in source["files"]}
    untracked_sources = [
        {
            "path": path,
            "bytes": source_by_path[path]["bytes"],
            "sha256": source_by_path[path]["sha256"],
            "preserved_in_source_snapshot": True,
        }
        for path in untracked_paths
        if path in source_by_path
    ]
    return {
        "head": head["stdout"].strip(),
        "branch": branch["stdout"].strip(),
        "dirty": bool(status.stdout),
        "status_porcelain_v1_z_bytes": len(status.stdout),
        "status_porcelain_v1_z_sha256": _sha256_bytes(status.stdout),
        "status_porcelain_v1_z_hex": status.stdout.hex(),
        "tracked_diff_binary_bytes": len(tracked_diff.stdout),
        "tracked_diff_binary_sha256": _sha256_bytes(tracked_diff.stdout),
        "tracked_diff_binary_hex": tracked_diff.stdout.hex(),
        "untracked_inventory": untracked_paths,
        "untracked_inventory_sha256": _sha256_bytes(untracked.stdout),
        "untracked_executed_sources": untracked_sources,
        "source_snapshot_sha256": source["source_manifest_sha256"],
    }


def _metadata_file(
    guard: AccessGuard, directory: Path, filename: str, *, purpose: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = directory / filename
    guard.register(path, "upstream_metadata")
    payload = guard.read_bytes(path, stage="preflight", purpose=purpose)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LifecycleError(f"invalid upstream metadata {filename}") from exc
    if not isinstance(value, dict):
        raise LifecycleError(f"upstream metadata {filename} is not an object")
    if payload != _canonical_json(value):
        raise LifecycleError(f"upstream metadata {filename} is not canonical JSON")
    return value, {"file": filename, "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _metadata_jsonl(
    guard: AccessGuard, directory: Path, filename: str, *, purpose: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = directory / filename
    guard.register(path, "upstream_metadata")
    payload = guard.read_bytes(path, stage="preflight", purpose=purpose)
    if not payload.endswith(b"\n"):
        raise LifecycleError(f"upstream JSONL {filename} lacks its terminal newline")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines()):
        if not line:
            raise LifecycleError(f"upstream JSONL {filename} contains a blank line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifecycleError(f"invalid {filename} JSONL row {index}") from exc
        if not isinstance(value, dict) or _canonical_json(value).removesuffix(b"\n") != line:
            raise LifecycleError(f"noncanonical {filename} JSONL row {index}")
        records.append(value)
    return records, {
        "file": filename,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "row_count": len(records),
    }


def _identity(record: Mapping[str, Any], name: str) -> tuple[str, tuple[int, int, int, int]]:
    image_id = record.get("image_id")
    bits = record.get("bit_tuple")
    if (
        not isinstance(image_id, str)
        or not image_id
        or not isinstance(bits, list)
        or len(bits) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bits)
    ):
        raise LifecycleError(f"{name} has an invalid cell identity")
    return image_id, tuple(bits)


def _normalized_comp008_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for row in rows:
        actual.verify_record(row, "cell_record_sha256")
        identity = _identity(row, "COMP-008 cell")
        if identity in seen:
            raise LifecycleError("duplicate COMP-008 cell identity")
        seen.add(identity)
        sspl1 = row.get("sspl1")
        symbols = row.get("absolute_symbols")
        integrity = row.get("integrity")
        oracle_row = row.get("oracle_row")
        if not all(isinstance(value, Mapping) for value in (sspl1, symbols, integrity, oracle_row)):
            raise LifecycleError("COMP-008 cell authority record is incomplete")
        if (
            oracle_row.get("n") != container.ROW_COUNT
            or sspl1.get("sha256") != oracle_row.get("sspl1_sha256")
            or symbols.get("symbol_sha256") != oracle_row.get("symbol_sha256")
            or integrity.get("decoded_state_sha256") is None
        ):
            raise LifecycleError("COMP-008 cell authority hashes disagree")
        normalized.append(
            {
                "image_id": identity[0],
                "bit_tuple": list(identity[1]),
                "cell_record_sha256": row["cell_record_sha256"],
                "sspl1": {
                    key: sspl1[key] for key in ("path", "bytes", "sha256", "components")
                },
                "absolute_symbols": {
                    key: symbols[key]
                    for key in ("path", "bytes", "sha256", "symbol_sha256")
                },
                "decoded_boundary_state_sha256": integrity["decoded_state_sha256"],
                "n": oracle_row["n"],
                "height": oracle_row["height"],
                "width": oracle_row["width"],
            }
        )
    if len(normalized) != 16:
        raise LifecycleError("COMP-008 authority inventory must contain exactly 16 cells")
    return normalized


def _normalized_targets(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    actual.verify_record(manifest, "targets_sha256")
    targets = manifest.get("targets")
    if not isinstance(targets, list) or len(targets) != 8:
        raise LifecycleError("COMP-008 target authority must contain exactly eight images")
    normalized = []
    names: set[str] = set()
    for target in targets:
        if not isinstance(target, Mapping):
            raise LifecycleError("COMP-008 target entry is not an object")
        name = target.get("name")
        dimensions = target.get("derived_rgb_dimensions_hw")
        if (
            not isinstance(name, str)
            or name in names
            or not isinstance(dimensions, list)
            or len(dimensions) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions)
        ):
            raise LifecycleError("COMP-008 target identity/dimensions are invalid")
        names.add(name)
        normalized.append(
            {
                "image_id": name,
                "relative_path": target["derived_path"],
                "encoded_bytes": target["derived_encoded_bytes"],
                "encoded_sha256": target["derived_encoded_sha256"],
                "rgb_dimensions_hw": dimensions,
                "rgb_pixel_sha256": target["derived_rgb_pixel_sha256"],
            }
        )
    if names != set(manifest.get("development_names", [])):
        raise LifecycleError("COMP-008 target names differ from the development inventory")
    return normalized


def _normalized_comp009_cells(
    rows: Sequence[Mapping[str, Any]],
    input_streams: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    inputs = {_identity(row, "COMP-009 input"): row for row in input_streams}
    if len(inputs) != 16:
        raise LifecycleError("COMP-009 input authority must contain 16 unique cells")
    expected_cell_seals = run_manifest.get("cell_record_sha256")
    if not isinstance(expected_cell_seals, list) or len(expected_cell_seals) != 16:
        raise LifecycleError("COMP-009 run manifest has an invalid cell-seal vector")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for row in rows:
        actual.verify_record(row, "cell_record_sha256")
        identity = _identity(row, "COMP-009 run cell")
        if identity in seen or identity not in inputs:
            raise LifecycleError("duplicate or unexpected COMP-009 run cell")
        seen.add(identity)
        source_input = inputs[identity]
        execution = row.get("execution")
        cell_input = row.get("input")
        artifacts = row.get("artifacts")
        if (
            not isinstance(execution, Mapping)
            or not isinstance(cell_input, Mapping)
            or not isinstance(artifacts, list)
        ):
            raise LifecycleError("COMP-009 run-cell authority is incomplete")
        actual.verify_record(execution, "execution_sha256")
        execution_core = dict(execution)
        execution_core.pop("execution_sha256")
        execution_core.pop("timings")
        streams = execution.get("streams")
        models = execution.get("models")
        boundary = execution.get("boundary")
        source = execution.get("source")
        if (
            not isinstance(streams, Mapping)
            or set(streams) != {"ssp2e", "ssp2s", "ssp2l", "ssp2f"}
            or not isinstance(models, Mapping)
            or not isinstance(boundary, Mapping)
            or not isinstance(source, Mapping)
        ):
            raise LifecycleError("COMP-009 E/S/L/F authority is incomplete")
        if (
            cell_input.get("sha256") != source_input.get("sha256")
            or cell_input.get("bytes") != source_input.get("bytes")
            or cell_input.get("symbol_sha256") != source_input.get("symbol_sha256")
            or cell_input.get("decoded_state_sha256") != source_input.get("decoded_state_sha256")
            or source.get("n") != container.ROW_COUNT
            or boundary.get("state_sha256") != source_input.get("decoded_state_sha256")
            or any(
                not isinstance(streams[name], Mapping)
                or streams[name].get("magic") != name.upper()
                or streams[name].get("blob_sha256") is None
                or streams[name].get("complete_bytes") is None
                for name in ("ssp2e", "ssp2s", "ssp2l", "ssp2f")
            )
        ):
            raise LifecycleError("COMP-009 stream/source/boundary authorities disagree")
        normalized.append(
            {
                "image_id": identity[0],
                "bit_tuple": list(identity[1]),
                "cell_record_sha256": row["cell_record_sha256"],
                "execution_sha256": execution["execution_sha256"],
                "execution_core_without_timings": execution_core,
                "input": dict(cell_input),
                "source": dict(source),
                "streams": dict(streams),
                "models": dict(models),
                "boundary": dict(boundary),
                "artifacts": artifacts,
                "integrity": row.get("integrity"),
                "confirmation_payloads_accessed": row.get("confirmation_payloads_accessed"),
            }
        )
    if (
        len(normalized) != 16
        or seen != set(inputs)
        or [row["cell_record_sha256"] for row in normalized] != expected_cell_seals
    ):
        raise LifecycleError("COMP-009 run-cell inventory/order differs from its manifest")
    if any(row["confirmation_payloads_accessed"] is not False for row in normalized):
        raise LifecycleError("COMP-009 authority claims confirmation access")
    return normalized


def bind_upstream_metadata(
    guard: AccessGuard,
    comp008: Path,
    comp009: Path,
    comp010: Path,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    p8, r8p = _metadata_file(guard, comp008, "preflight.json", purpose="bind COMP-008 seal")
    a8, r8a = _metadata_file(guard, comp008, "analysis.json", purpose="bind COMP-008 analysis")
    ar8, r8ar = _metadata_file(
        guard, comp008, "analysis_record.json", purpose="bind COMP-008 analysis/target inventory"
    )
    x8, r8r = _metadata_file(guard, comp008, "replay.json", purpose="bind COMP-008 replay")
    targets8, r8t = _metadata_file(
        guard, comp008, "targets_manifest.json", purpose="bind COMP-008 target authority metadata"
    )
    cells8, r8c = _metadata_jsonl(
        guard, comp008, "cells.jsonl", purpose="bind COMP-008 per-cell authority metadata"
    )
    actual.verify_record(p8, "binding_sha256")
    actual.verify_record(a8, "analysis_sha256")
    actual.verify_record(ar8, "analysis_record_sha256")
    actual.verify_record(x8, "lifecycle_replay_sha256")
    if (
        p8.get("binding_sha256") != EXPECTED_COMP008["binding_sha256"]
        or a8.get("analysis_sha256") != EXPECTED_COMP008["analysis_sha256"]
        or ar8.get("analysis_record_sha256")
        != EXPECTED_COMP008["analysis_record_sha256"]
        or ar8.get("analysis_sha256") != EXPECTED_COMP008["analysis_sha256"]
        or ar8.get("targets_sha256") != EXPECTED_COMP008["targets_sha256"]
        or ar8.get("cell_record_sha256")
        != [row.get("cell_record_sha256") for row in cells8]
        or x8.get("lifecycle_replay_sha256") != EXPECTED_COMP008["lifecycle_replay_sha256"]
        or r8c["sha256"] != EXPECTED_COMP008["cells_file_sha256"]
        or p8.get("scientific_pixels_accessed") is not False
        or x8.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("COMP-008 metadata seal drift")
    normalized8 = _normalized_comp008_cells(cells8)
    normalized_targets = _normalized_targets(targets8)
    if targets8.get("targets_sha256") != EXPECTED_COMP008["targets_sha256"]:
        raise LifecycleError("COMP-008 target manifest seal is not the audited authority")
    if {row["image_id"] for row in normalized8} != {
        row["image_id"] for row in normalized_targets
    }:
        raise LifecycleError("COMP-008 cell and target image inventories disagree")

    mapping9 = {
        "preflight.json": (
            ("preflight_binding_sha256", "source_manifest_sha256", "source_archive_sha256"),
            "bind COMP-009 preflight/source seals",
        ),
        "input_manifest.json": (("input_manifest_sha256",), "bind COMP-009 input seal"),
        "run_manifest.json": (("run_manifest_sha256",), "bind COMP-009 run seal"),
        "benchmark_manifest.json": (
            ("benchmark_manifest_sha256",),
            "bind COMP-009 benchmark seal",
        ),
        "analysis_record.json": (("analysis_record_sha256",), "bind COMP-009 analysis record"),
        "analysis.json": (("analysis_sha256",), "bind COMP-009 analysis"),
        "source_manifest.json": (
            ("source_manifest_sha256",),
            "bind COMP-009 captured source manifest",
        ),
    }
    files9: list[dict[str, Any]] = []
    values9: dict[str, dict[str, Any]] = {}
    seal_fields9 = {
        "preflight.json": "preflight_binding_sha256",
        "input_manifest.json": "input_manifest_sha256",
        "run_manifest.json": "run_manifest_sha256",
        "benchmark_manifest.json": "benchmark_manifest_sha256",
        "analysis_record.json": "analysis_record_sha256",
        "analysis.json": "analysis_sha256",
        "source_manifest.json": "source_manifest_sha256",
    }
    for filename, (fields, purpose) in mapping9.items():
        value, record = _metadata_file(guard, comp009, filename, purpose=purpose)
        actual.verify_record(value, seal_fields9[filename])
        for field in fields:
            if value.get(field) != EXPECTED_COMP009[field]:
                raise LifecycleError(f"COMP-009 {field} drift")
        files9.append(record)
        values9[filename] = value
    input9 = values9["input_manifest.json"]
    run9 = values9["run_manifest.json"]
    streams9 = input9.get("streams")
    if (
        input9.get("stream_count") != 16
        or not isinstance(streams9, list)
        or len(streams9) != 16
        or input9.get("target_pixels_opened") is not False
        or input9.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("COMP-009 input manifest is not target-free or complete")
    run_cells9, run_cells_file = _metadata_jsonl(
        guard, comp009, "run_cells.jsonl", purpose="bind COMP-009 E/S/L/F cell authorities"
    )
    if (
        run9.get("cell_count") != 16
        or run_cells_file["sha256"] != run9.get("cells_file_sha256")
        or run9.get("target_pixels_opened") is not False
        or run9.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("COMP-009 run-cell metadata hash/count/access drift")
    normalized9 = _normalized_comp009_cells(run_cells9, streams9, run9)
    by8 = {
        (row["image_id"], tuple(row["bit_tuple"])): row for row in normalized8
    }
    for stream in streams9:
        identity = _identity(stream, "COMP-009 input stream")
        authority8 = by8.get(identity)
        if authority8 is None or (
            stream.get("cell_record_sha256") != authority8["cell_record_sha256"]
            or stream.get("sha256") != authority8["sspl1"]["sha256"]
            or stream.get("bytes") != authority8["sspl1"]["bytes"]
            or stream.get("symbol_sha256")
            != authority8["absolute_symbols"]["symbol_sha256"]
            or stream.get("decoded_state_sha256")
            != authority8["decoded_boundary_state_sha256"]
        ):
            raise LifecycleError("COMP-008/009 per-cell authorities disagree")

    source_archive = comp009 / "source_snapshot.tar"
    guard.register(source_archive, "runtime_dependency")
    source_archive_payload = guard.read_bytes(
        source_archive,
        stage="preflight",
        purpose="copy exact COMP-009 captured source archive",
    )
    if _sha256_bytes(source_archive_payload) != EXPECTED_COMP009["source_archive_sha256"]:
        raise LifecycleError("actual COMP-009 source archive hash drift")
    artifact_root = _lexical(artifact_root)
    copied_source_archive = artifact_root / "upstream/comp009_source_snapshot.tar"
    _planned_write(
        guard.journal,
        artifact_root,
        copied_source_archive,
        purpose="persist exact COMP-009 captured source archive",
    )
    _exclusive_write(copied_source_archive, source_archive_payload)
    with tempfile.TemporaryDirectory(prefix="comp011-upstream-source-proof-") as temporary:
        extracted9 = safe_extract_source_archive(
            copied_source_archive,
            Path(temporary) / "source",
            expected_manifest=values9["source_manifest.json"],
            guard=guard,
        )
    if extracted9 != values9["source_manifest.json"]:
        raise LifecycleError("COMP-009 captured source relocation proof drift")

    p10, r10p = _metadata_file(guard, comp010, "preflight.json", purpose="bind COMP-010 preflight")
    a10, r10a = _metadata_file(guard, comp010, "repair.json", purpose="bind COMP-010 repair")
    actual.verify_record(p10, "repair_preflight_sha256")
    actual.verify_record(a10, "repair_sha256")
    children = a10.get("captured_children")
    if (
        p10.get("repair_preflight_sha256") != EXPECTED_COMP010["repair_preflight_sha256"]
        or a10.get("repair_sha256") != EXPECTED_COMP010["repair_sha256"]
        or not isinstance(children, list)
        or len(children) != 2
        or children[0] != children[1]
        or a10.get("lifecycle_repair_passed") is not True
        or a10.get("comp009_inventory_exactly_unchanged") is not True
        or a10.get("captured_children_scientifically_byte_identical") is not True
        or a10.get("codec_streams_regenerated_ephemerally") is not True
        or a10.get("persisted_codec_streams_created_or_replaced") is not False
        or a10.get("persisted_benchmark_resource_timing_samples_reused_not_generated_or_replaced")
        is not True
        or a10.get("renderer_runtime_reexecuted") is not False
        or a10.get("exact_renderer_binary_replay_claimed") is not False
        or a10.get("underlying_field_or_qat_refit") is not False
        or a10.get("scientific_pixels_accessed") is not False
        or a10.get("confirmation_payloads_accessed") is not False
        or any(
            child.get("repair_child_sha256") != EXPECTED_COMP010["repair_child_sha256"]
            or not isinstance(child.get("captured_worker"), dict)
            or child["captured_worker"].get("captured_replay_worker_sha256")
            != EXPECTED_COMP010["captured_replay_worker_sha256"]
            for child in children
        )
    ):
        raise LifecycleError("COMP-010 replay-repair metadata drift")
    for child in children:
        actual.verify_record(child, "repair_child_sha256")
        actual.verify_record(child["captured_worker"], "captured_replay_worker_sha256")
        if (
            child.get("schema")
            != "structsplat.comp010.ssp2e-replay-repair-child.v2"
            or child.get("protocol") != "COMP-010-ssp2e-captured-replay-repair-v2"
            or child.get("scientific_pixels_accessed") is not False
            or child.get("confirmation_payloads_accessed") is not False
            or child.get("underlying_field_or_qat_refit") is not False
            or child.get("renderer_runtime_reexecuted") is not False
            or child.get("exact_renderer_binary_replay_claimed") is not False
            or child.get("persisted_codec_streams_created_or_replaced") is not False
            or child["captured_worker"].get("schema")
            != "structsplat.comp009.captured-replay-worker.v1"
            or child["captured_worker"].get("protocol")
            != "COMP-009-ssp2e-actual-coder-v1"
            or child["captured_worker"].get("cell_count") != 16
            or child["captured_worker"].get("target_pixels_opened") is not False
            or child["captured_worker"].get("confirmation_payloads_accessed") is not False
            or child["captured_worker"].get("underlying_field_or_qat_refit") is not False
        ):
            raise LifecycleError("COMP-010 captured child opened forbidden scientific data")
    authority = {
        "comp008": {
            **EXPECTED_COMP008,
            "files": [r8p, r8a, r8ar, r8r, r8t, r8c],
            "cells": normalized8,
            "targets_manifest_sha256": targets8["targets_sha256"],
            "targets": normalized_targets,
        },
        "comp009": {
            **EXPECTED_COMP009,
            "files": [*files9, run_cells_file],
            "input_streams": streams9,
            "cells": normalized9,
            "captured_source_manifest": values9["source_manifest.json"],
            "captured_source_archive": {
                "relative_path": copied_source_archive.relative_to(artifact_root).as_posix(),
                "bytes": len(source_archive_payload),
                "sha256": _sha256_bytes(source_archive_payload),
                "safe_extraction_verified": True,
            },
        },
        "comp010": {
            **EXPECTED_COMP010,
            "files": [r10p, r10a],
            "captured_child_count": len(children),
            "captured_children": children,
        },
        "development_streams_opened": False,
        "target_or_pixel_payloads_opened": False,
        "confirmation_payloads_accessed": False,
    }
    return actual.seal_record(authority, "authority_manifest_sha256")


def _library_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "mode": stat.S_IMODE(path.stat().st_mode),
    }


def arithmetic_native_proof(path: Path) -> dict[str, Any]:
    native = arithmetic.NativeArithmeticCoder(path)
    records: list[dict[str, Any]] = []
    for alphabet in sorted(container.LEGAL_INDEX_ALPHABETS):
        symbols = np.arange(container.ROW_COUNT, dtype=np.uint32) % alphabet
        frequencies = container.empirical_frequencies(symbols, alphabet)
        rows = container.cdf_rows(frequencies)
        reference = arithmetic.encode(symbols, rows)
        payload = native.encode(symbols, rows)
        decoded = native.decode(payload, rows, container.ROW_COUNT)
        if payload != reference or not np.array_equal(decoded, symbols):
            raise LifecycleError(f"persisted arithmetic parity failed for alphabet {alphabet}")
        records.append(
            {
                "alphabet": alphabet,
                "payload_bytes": len(payload),
                "payload_sha256": _sha256_bytes(payload),
                "frequencies_sha256": _sha256_bytes(frequencies.astype("<u4").tobytes()),
            }
        )
    return {"alphabets": records, "all_python_native_reencode_exact": True}


def _start_record(start: lloyd.StartResult) -> dict[str, Any]:
    return {
        "start_id": start.start_id,
        "status": start.status,
        "update_sweeps": start.update_sweeps,
        "terminal_sse": start.terminal_sse,
        "codebook_sha256": lloyd.codebook_state_sha256(start.codebook),
        "indices_sha256": _sha256_bytes(start.indices.astype("<u4").tobytes()),
        "trajectory_sha256": _sha256_bytes(
            _canonical_json(
                [
                    [row.state_index, row.update_sweep, row.codebook_sha256, row.objective_sse]
                    for row in start.trajectory
                ]
            )
        ),
    }


def _fit_record(label: str, result: lloyd.VariantFitResult) -> dict[str, Any]:
    return {
        "label": label,
        "family_id": result.family_id,
        "matched_stage_count": result.matched_stage_count,
        "base_k": result.base_k,
        "stages": [
            {
                "actual_stage_index": stage.spec.actual_stage_index,
                "codebook_size": stage.spec.codebook_size,
                "selected_start_id": stage.selected_start_id,
                "starts": [_start_record(start) for start in stage.starts],
            }
            for stage in result.stages
        ],
        "reconstructed_rgb_sha256": _sha256_bytes(result.reconstructed_rgb.tobytes()),
    }


def lloyd_complexity_worker(native_path: Path) -> dict[str, Any]:
    started = time.perf_counter_ns()
    fixture = lloyd.complexity_fixture()
    fitter = lloyd.NativeLloydFitter(native_path)
    arms: list[dict[str, Any]] = []
    worst_before: list[dict[str, Any]] | None = None
    for family in ("flat", "rvq"):
        for stages in (2, 3):
            for base_k in (64, 128):
                label = f"{family}_s{stages}_k{base_k}"
                result = (
                    lloyd.fit_flat(fixture, stages, base_k, fitter=fitter)
                    if family == "flat"
                    else lloyd.fit_rvq(fixture, stages, base_k, fitter=fitter)
                )
                record = _fit_record(label, result)
                arms.append(record)
                if label == "flat_s3_k128":
                    worst_before = record["stages"][0]["starts"]
    wall_ns = time.perf_counter_ns() - started
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if any(start["status"] != "fixed" for arm in arms for stage in arm["stages"] for start in stage["starts"]):
        raise LifecycleError("synthetic full-menu Lloyd start did not converge")
    if wall_ns > 300_000_000_000 or peak_rss_bytes > 4 * 2**30:
        raise LifecycleError("synthetic full-menu Lloyd complexity gate failed")

    worst_spec = lloyd.LloydSpec(0, 3, 128, 0, 640)
    forced_started = time.perf_counter_ns()
    forced = [fitter.force_sweeps(fixture, worst_spec, start, forced_sweeps=10) for start in range(4)]
    forced_ns = time.perf_counter_ns() - forced_started
    projected_ns = 10 * forced_ns
    if projected_ns > 300_000_000_000:
        raise LifecycleError("forced-ten-sweep Lloyd projection gate failed")
    worst_after = [_start_record(fitter.fit_start(fixture, worst_spec, start)) for start in range(4)]
    if worst_before != worst_after:
        raise LifecycleError("timing-only forced sweeps changed the scientific fitter result")
    record = {
        "schema": LLOYD_PROOF_SCHEMA,
        "fixture_sha256": _sha256_bytes(fixture.tobytes()),
        "rows": len(fixture),
        "arms": arms,
        "full_menu_wall_ns": wall_ns,
        "full_menu_peak_rss_bytes": peak_rss_bytes,
        "full_menu_wall_pass": True,
        "full_menu_rss_pass": True,
        "forced_ten_sweep_ns": forced_ns,
        "forced_projection_100_sweeps_ns": projected_ns,
        "forced_projection_pass": True,
        "forced_outputs": [
            {"start_id": value.start_id, "final_codebook_sha256": value.final_codebook_sha256}
            for value in forced
        ],
        "scientific_result_unchanged_by_forced_mode": True,
    }
    return actual.seal_record(record, "lloyd_proof_sha256")


def validate_lloyd_proof(record: Mapping[str, Any]) -> None:
    actual.verify_record(record, "lloyd_proof_sha256")
    expected_labels = [
        f"{family}_s{stages}_k{base_k}"
        for family in ("flat", "rvq")
        for stages in (2, 3)
        for base_k in (64, 128)
    ]
    arms = record.get("arms")
    starts = (
        [start for arm in arms for stage in arm["stages"] for start in stage["starts"]]
        if isinstance(arms, list)
        else []
    )
    if (
        record.get("schema") != LLOYD_PROOF_SCHEMA
        or record.get("rows") != container.ROW_COUNT
        or record.get("fixture_sha256")
        != _sha256_bytes(lloyd.complexity_fixture().tobytes())
        or [arm.get("label") for arm in arms or []] != expected_labels
        or len(starts) != 56
        or any(start.get("status") != "fixed" for start in starts)
        or record.get("full_menu_wall_ns", 300_000_000_001) > 300_000_000_000
        or record.get("full_menu_peak_rss_bytes", 4 * 2**30 + 1) > 4 * 2**30
        or record.get("forced_projection_100_sweeps_ns", 300_000_000_001)
        > 300_000_000_000
        or record.get("full_menu_wall_pass") is not True
        or record.get("full_menu_rss_pass") is not True
        or record.get("forced_projection_pass") is not True
        or record.get("scientific_result_unchanged_by_forced_mode") is not True
    ):
        raise LifecycleError("Lloyd complexity worker schema/gate validation failed")


def validate_renderer_build_worker(record: Mapping[str, Any]) -> None:
    actual.verify_record(record, "renderer_build_worker_sha256")
    sources = record.get("source")
    fixture = record.get("fixture")
    if (
        record.get("schema") != RENDERER_BUILD_SCHEMA
        or not isinstance(record.get("extension_path"), str)
        or not isinstance(record.get("extension_bytes"), int)
        or record.get("extension_bytes", 0) <= 0
        or not isinstance(sources, list)
        or sources != _renderer_source_record()
        or not isinstance(fixture, Mapping)
        or set(fixture)
        != {
            "means",
            "physical_scales",
            "rotations",
            "colors",
            "field_log_scales",
            "field_scales",
            "field_effective_scales",
            "field_conics",
            "field_radii",
            "target_u8",
            "target_tensor",
        }
    ):
        raise LifecycleError("renderer build-worker schema/provenance validation failed")
    _require_sha256(record.get("extension_sha256"), "renderer build extension")
    _require_sha256(record.get("toy_output_sha256"), "renderer build toy output")


def validate_renderer_proof_worker(
    record: Mapping[str, Any], renderer_record: Mapping[str, Any]
) -> None:
    actual.verify_record(record, "renderer_proof_worker_sha256")
    metrics = record.get("metrics")
    hashes = record.get("render_tensor_sha256")
    spreads = record.get("spreads")
    repeat = record.get("same_worker_metric_repeat")
    parity = record.get("comp009_call_semantics")
    if not isinstance(metrics, list) or len(metrics) != 9:
        raise LifecycleError("renderer proof must contain exactly nine metric triples")
    recomputed = {
        name: max(float(row[name]) for row in metrics) - min(float(row[name]) for row in metrics)
        for name in quality.METRIC_NAMES
    }
    if (
        record.get("schema") != RENDERER_PROOF_SCHEMA
        or record.get("renderer_relative_path") != renderer_record.get("relative_path")
        or record.get("renderer_sha256") != renderer_record.get("sha256")
        or not isinstance(hashes, list)
        or len(hashes) != 9
        or any(_require_sha256(value, "render tensor") != value for value in hashes)
        or not isinstance(spreads, Mapping)
        or any(float(spreads[name]) != recomputed[name] for name in quality.METRIC_NAMES)
        or float(spreads["psnr"]) > quality.PSNR_SPREAD_CAP
        or float(spreads["ms_ssim"]) > quality.MS_SSIM_SPREAD_CAP
        or float(spreads["lpips"]) > quality.LPIPS_SPREAD_CAP
        or record.get("all_spreads_pass") is not True
        or not isinstance(repeat, Mapping)
        or repeat.get("metric_ieee754_bytes_identical") is not True
        or repeat.get("lpips_state_identical") is not True
        or repeat.get("first") != repeat.get("second")
        or not isinstance(parity, Mapping)
        or parity.get("passed") is not True
        or parity.get("call_records_identical") is not True
        or parity.get("boundary_call") != parity.get("comp009_call")
        or float(parity.get("output_max_abs_difference", float("inf"))) > 1e-5
        or record.get("loaded_only_artifact_relative_binary") is not True
    ):
        raise LifecycleError("persisted renderer proof worker schema/gate validation failed")
    _require_sha256(repeat.get("lpips_state_sha256"), "LPIPS state")


def _renderer_source_record() -> list[dict[str, Any]]:
    paths = (
        ROOT / "src/structsplat/cuda_render.py",
        ROOT / "src/structsplat/render.py",
        ROOT / "src/structsplat/gaussians.py",
        ROOT / "src/structsplat/cuda/render_ext.cpp",
        ROOT / "src/structsplat/cuda/render_ext.cu",
    )
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def renderer_abi_record(
    renderer_path: Path, guard: AccessGuard, artifact_root: Path
) -> dict[str, Any]:
    """Bind ELF identity plus the bytes of every resolved loader dependency."""
    identity = quality.renderer_elf_identity(renderer_path)
    resolved: dict[Path, set[str]] = {}
    for match in re.finditer(r"(?:=>\s+)?(/[^\s]+)", identity["ldd_output"]):
        candidate = Path(match.group(1))
        if candidate.is_file():
            canonical = candidate.resolve(strict=True)
            resolved.setdefault(canonical, set()).add(os.fspath(candidate))
    dependencies: list[dict[str, Any]] = []
    for dependency in sorted(resolved):
        guard.register(dependency, "runtime_dependency")
        payload = guard.read_bytes(
            dependency,
            stage="preflight",
            purpose="bind renderer loader dependency bytes",
        )
        dependencies.append(
            {
                "path": os.fspath(dependency),
                "loader_reported_paths": sorted(resolved[dependency]),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    preload_requested = Path(quality.REQUIRED_RENDERER_PRELOAD)
    preload_info = os.lstat(preload_requested)
    preload_link_target = (
        os.readlink(preload_requested) if stat.S_ISLNK(preload_info.st_mode) else None
    )
    preload = preload_requested.resolve(strict=True)
    guard.register(preload, "runtime_dependency")
    preload_payload = guard.read_bytes(
        preload, stage="preflight", purpose="bind required system libstdc++ ABI"
    )
    version_info = _command_record(("readelf", "--version-info", os.fspath(preload)))
    if version_info["returncode"] != 0:
        raise LifecycleError("cannot bind required libstdc++ version information")
    renderer_relative = renderer_path.relative_to(artifact_root).as_posix()
    return {
        "renderer_relative_path": renderer_relative,
        "elf_identity": identity,
        "resolved_libraries": dependencies,
        "required_preload": {
            "path": os.fspath(preload),
            "requested_path": os.fspath(preload_requested),
            "requested_symlink_target": preload_link_target,
            "bytes": len(preload_payload),
            "sha256": _sha256_bytes(preload_payload),
            "version_info": version_info,
        },
        "loader_environment": {
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
    }


def renderer_build_worker() -> dict[str, Any]:
    """Build the production renderer once in a dedicated disposable child."""
    quality.frozen_runtime_environment()
    import torch

    from structsplat import cuda_render
    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    if not torch.cuda.is_available():
        raise LifecycleError("CUDA is unavailable for the renderer build proof")
    device = torch.device("cuda:0")
    field = GaussianField.from_numpy(
        means=np.asarray([[3.5, 3.5]], dtype=np.float32),
        scales=np.asarray([[1.0, 1.0]], dtype=np.float32),
        angles=np.asarray([0.0], dtype=np.float32),
        colors=np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32),
        opacities=None,
        scale_max=None,
        device=device,
        dtype=torch.float32,
        color_grads=None,
        background_mask=None,
        filter_variance=None,
    )
    torch.cuda.synchronize(device)
    image = render_field(
        field.means,
        field.conics(0.0),
        field.colors,
        field.radii(3.0, 0.0),
        8,
        8,
        64,
        "cuda",
        scales=field.effective_scales(0.0),
        rotations=field.rotations,
        support_fade=False,
        sigma_cutoff=3.0,
    )
    torch.cuda.synchronize(device)
    extension = getattr(cuda_render, "_EXT", None)
    raw_path = getattr(extension, "__file__", None)
    if not isinstance(raw_path, str) or not bool(torch.isfinite(image).all()):
        raise LifecycleError("production renderer did not build/load a finite extension")
    path = Path(raw_path).resolve(strict=True)
    build_ninja = path.parent / "build.ninja"
    _, _, fixture_record = quality.build_repeat_fixture(device)
    record = {
        "schema": RENDERER_BUILD_SCHEMA,
        "extension_path": os.fspath(path),
        "extension_bytes": path.stat().st_size,
        "extension_sha256": _sha256_file(path),
        "source": _renderer_source_record(),
        "build_ninja": (
            {
                "bytes": build_ninja.stat().st_size,
                "sha256": _sha256_file(build_ninja),
                "content": build_ninja.read_text(encoding="utf-8"),
            }
            if build_ninja.is_file()
            else None
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),  # noqa: SLF001
        "device": {
            "name": torch.cuda.get_device_properties(device).name,
            "uuid": str(torch.cuda.get_device_properties(device).uuid),
            "compute_capability": [
                torch.cuda.get_device_properties(device).major,
                torch.cuda.get_device_properties(device).minor,
            ],
        },
        "fixture": fixture_record,
        "toy_output_sha256": _sha256_bytes(image.detach().cpu().numpy().astype("<f4").tobytes()),
    }
    return actual.seal_record(record, "renderer_build_worker_sha256")


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _module_provenance(
    module_name: str, distribution: str, guard: AccessGuard
) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise LifecycleError(f"cannot resolve metric module {module_name!r}")
    origin = Path(spec.origin)
    guard.register(origin, "runtime_dependency")
    payload = guard.read_bytes(
        origin, stage="preflight", purpose=f"bind metric source {module_name}"
    )
    return {
        "module": module_name,
        "distribution": distribution,
        "version": importlib.metadata.version(distribution),
        "origin_name": origin.name,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _alexnet_cache_path() -> Path:
    cache_root = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
    return cache_root / _ALEXNET_CACHE_RELATIVE


def copy_metric_dependencies(outdir: Path, guard: AccessGuard) -> dict[str, Any]:
    lpips_spec = importlib.util.find_spec("lpips")
    if lpips_spec is None or lpips_spec.origin is None:
        raise LifecycleError("cannot resolve installed LPIPS source")
    source_root = Path(lpips_spec.origin).resolve().parent
    destination = outdir / "runtime/metrics/site/lpips"
    if destination.exists():
        raise LifecycleError("artifact-local LPIPS package already exists")
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise LifecycleError("installed LPIPS package contains a symlink")
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        guard.register(source, "runtime_dependency")
        payload = guard.read_bytes(
            source, stage="preflight", purpose="copy artifact-local LPIPS dependency"
        )
        target = destination / source.relative_to(source_root)
        _exclusive_write(target, payload)
    alexnet = _alexnet_cache_path()
    guard.register(alexnet, "runtime_dependency")
    alexnet_payload = guard.read_bytes(
        alexnet, stage="preflight", purpose="copy artifact-local AlexNet weights"
    )
    alexnet_target = outdir / "runtime/metrics/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
    _exclusive_write(alexnet_target, alexnet_payload)
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    provenance = [
        _module_provenance(module, distribution, guard)
        for module, distribution in (
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("PIL", "Pillow"),
            ("pytorch_msssim", "pytorch-msssim"),
            ("lpips", "lpips"),
        )
    ]
    return {
        "lpips_relative_root": "runtime/metrics/site/lpips",
        "lpips_files": _tree_manifest(destination),
        "alexnet_relative_path": alexnet_target.relative_to(outdir).as_posix(),
        "alexnet_bytes": len(alexnet_payload),
        "alexnet_sha256": _sha256_bytes(alexnet_payload),
        "module_provenance": provenance,
        "ms_ssim_call": {
            "data_range": 1.0,
            "size_average": True,
            "win_size": 11,
            "win_sigma": 1.5,
            "win": None,
            "weights": weights,
            "weights_sha256": _sha256_bytes(_canonical_json(weights)),
            "K": [0.01, 0.03],
        },
        "network_download_permitted": False,
    }


def _header() -> SimpleNamespace:
    return SimpleNamespace(
        n=8192,
        renderer_id=1,
        support_fade=0,
        aa_dilation=0.0,
        sigma_cutoff=3.0,
        height=192,
        width=192,
        render_chunk=512,
    )


def _fixture_boundary(field: Any) -> dict[str, np.ndarray]:
    return {
        "means": field.means.detach().cpu().contiguous().numpy().astype(np.float32, copy=True),
        "log_scales": field.log_scales.detach().cpu().contiguous().numpy().astype(np.float32, copy=True),
        "rotations": field.rotations.detach().cpu().contiguous().numpy().astype(np.float32, copy=True),
        "colors": field.colors.detach().cpu().contiguous().numpy().astype(np.float32, copy=True),
    }


def _frozen_comp009_render(header: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    import torch

    from structsplat.gaussians import GaussianField
    from structsplat.render import render_field

    device = torch.device("cuda:0")
    field = GaussianField(
        torch.as_tensor(arrays["means"], device=device).clone(),
        torch.as_tensor(arrays["log_scales"], device=device).clone(),
        torch.as_tensor(arrays["rotations"], device=device).clone(),
        torch.as_tensor(arrays["colors"], device=device).clone(),
        opacities=None,
    )
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        rendered = render_field(
            field.means,
            field.conics(float(header.aa_dilation)),
            field.colors,
            field.radii(float(header.sigma_cutoff), float(header.aa_dilation)),
            int(header.height),
            int(header.width),
            int(header.render_chunk),
            "cuda",
            scales=field.effective_scales(float(header.aa_dilation)),
            rotations=field.rotations,
            support_fade=bool(header.support_fade),
            sigma_cutoff=float(header.sigma_cutoff),
        )
    torch.cuda.synchronize(device)
    return quality.strict_render_tensor(rendered)


def _capture_render_call(callback: Any) -> tuple[Any, dict[str, Any]]:
    from structsplat import render as render_module

    original = render_module.render_field
    captured: dict[str, Any] | None = None

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured
        if captured is not None or len(args) != 8:
            raise LifecycleError("synthetic renderer call count/signature drift")
        tensor_names = ("means", "conics", "colors", "radii")
        tensor_records = {
            name: quality.tensor_record(value.detach().to(device="cpu").contiguous())
            for name, value in zip(tensor_names, args[:4], strict=True)
        }
        for name in ("scales", "rotations"):
            value = kwargs.get(name)
            if value is None:
                raise LifecycleError(f"synthetic renderer call omitted {name}")
            tensor_records[name] = quality.tensor_record(
                value.detach().to(device="cpu").contiguous()
            )
        captured = {
            "tensors": tensor_records,
            "height": args[4],
            "width": args[5],
            "render_chunk": args[6],
            "mode": args[7],
            "support_fade": kwargs.get("support_fade"),
            "sigma_cutoff": kwargs.get("sigma_cutoff"),
            "keyword_names": sorted(kwargs),
            "opacity_argument_absent": "opacities" not in kwargs,
        }
        return original(*args, **kwargs)

    render_module.render_field = wrapper
    try:
        output = callback()
    finally:
        render_module.render_field = original
    if captured is None:
        raise LifecycleError("synthetic renderer callback made no production call")
    return output, captured


def renderer_proof_worker(artifact_root: Path, renderer_record: Mapping[str, Any]) -> dict[str, Any]:
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"The parameter 'pretrained' is deprecated.*",
        category=UserWarning,
        module=r"torchvision\.models\._utils",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
        category=UserWarning,
        module=r"torchvision\.models\._utils",
    )
    quality.frozen_runtime_environment()
    quality.load_persisted_renderer(
        artifact_root,
        str(renderer_record["relative_path"]),
        str(renderer_record["sha256"]),
        int(renderer_record["bytes"]),
        renderer_record["elf_identity"],
    )
    field, target, fixture_record = quality.build_repeat_fixture("cuda:0")
    arrays = _fixture_boundary(field)
    header = _header()
    first, boundary_call = _capture_render_call(
        lambda: quality.render_boundary_once(header, arrays)
    )
    comp009, comp009_call = _capture_render_call(
        lambda: _frozen_comp009_render(header, arrays)
    )
    if boundary_call != comp009_call:
        raise LifecycleError("synthetic COMP-009 renderer call parity failed")
    max_abs = float((first - comp009).abs().max().item())
    if max_abs > 1e-5:
        raise LifecycleError("synthetic COMP-009 render output parity failed")
    repeat = quality.metric_repeat_proof(first, target)
    if not repeat["metric_ieee754_bytes_identical"] or not repeat["lpips_state_identical"]:
        raise LifecycleError("same-worker metric repeat proof failed")
    metrics: list[quality.MetricTriple] = []
    render_hashes: list[str] = []
    for index in range(9):
        rendered = first if index == 0 else quality.render_boundary_once(header, arrays)
        render_hashes.append(quality.tensor_record(rendered)["sha256"])
        triple, state_sha = quality.exact_metric_values(rendered, target)
        if state_sha != repeat["lpips_state_sha256"]:
            raise LifecycleError("LPIPS state drift across renderer repetitions")
        metrics.append(triple)
    spread = {
        name: max(float(getattr(value, name)) for value in metrics)
        - min(float(getattr(value, name)) for value in metrics)
        for name in quality.METRIC_NAMES
    }
    if (
        spread["psnr"] > quality.PSNR_SPREAD_CAP
        or spread["ms_ssim"] > quality.MS_SSIM_SPREAD_CAP
        or spread["lpips"] > quality.LPIPS_SPREAD_CAP
    ):
        raise LifecycleError("nine-render synthetic metric spread cap failed")
    record = {
        "schema": RENDERER_PROOF_SCHEMA,
        "renderer_relative_path": renderer_record["relative_path"],
        "renderer_sha256": renderer_record["sha256"],
        "fixture": fixture_record,
        "comp009_call_semantics": {
            "gaussian_field_log_scale_constructor": True,
            "aa_dilation": 0.0,
            "sigma_cutoff": 3.0,
            "support_fade": False,
            "render_chunk": 512,
            "output_max_abs_difference": max_abs,
            "boundary_call": boundary_call,
            "comp009_call": comp009_call,
            "call_records_identical": True,
            "passed": True,
        },
        "same_worker_metric_repeat": repeat,
        "render_tensor_sha256": render_hashes,
        "metrics": [value.as_dict() for value in metrics],
        "spreads": spread,
        "spread_caps": {
            "psnr": quality.PSNR_SPREAD_CAP,
            "ms_ssim": quality.MS_SSIM_SPREAD_CAP,
            "lpips": quality.LPIPS_SPREAD_CAP,
        },
        "all_spreads_pass": True,
        "loaded_only_artifact_relative_binary": True,
    }
    return actual.seal_record(record, "renderer_proof_worker_sha256")


def _captured_replay_source_root() -> Path | None:
    captured = os.environ.get(_CAPTURED_REPLAY_SOURCE_ENV)
    if captured is None:
        return None
    captured_root = _lexical(Path(captured))
    if (
        captured_root != _lexical(ROOT)
        or not (captured_root / "SOURCE_MANIFEST.json").is_file()
    ):
        raise LifecycleError("captured replay source-root marker differs from module origin")
    return captured_root


def _worker_source_root(outdir: Path | None) -> Path:
    captured_root = _captured_replay_source_root()
    if captured_root is not None:
        _verify_runtime_source_tree(captured_root)
        return captured_root
    if outdir is not None:
        candidate = _lexical(outdir) / _RUNTIME_SOURCE_RELATIVE
        if (candidate / "SOURCE_MANIFEST.json").is_file():
            return candidate
    return _lexical(ROOT)


def _worker_source_read_only(source_root: Path) -> list[Path]:
    source_root = _lexical(source_root)
    if (source_root / "SOURCE_MANIFEST.json").is_file():
        return [source_root]
    return [source_root / "benchmarks", source_root / "src"]


def _worker_environment(
    outdir: Path | None = None, *, source_root: Path | None = None
) -> dict[str, str]:
    inherited = {
        name: os.environ[name]
        for name in (
            "PATH",
            "LANG",
            "LC_ALL",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_HOME",
            "LD_LIBRARY_PATH",
            "TORCH_CUDA_ARCH_LIST",
            "CXX",
            "CC",
        )
        if name in os.environ
    }
    environment = inherited
    environment.update(quality.FROZEN_THREAD_ENVIRONMENT)
    environment["LD_PRELOAD"] = quality.REQUIRED_RENDERER_PRELOAD
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["NO_PROXY"] = "*"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HOME"] = "/nonexistent-comp011-home"
    source_root = _worker_source_root(outdir) if source_root is None else _lexical(source_root)
    roots = [os.fspath(source_root), os.fspath(source_root / "src")]
    if (source_root / "SOURCE_MANIFEST.json").is_file():
        source = _verify_runtime_source_tree(source_root)
        environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
            source["source_manifest_sha256"]
        )
    if outdir is not None:
        roots.insert(0, os.fspath(outdir / "runtime/metrics/site"))
        environment["TORCH_HOME"] = os.fspath(outdir / "runtime/metrics/torch")
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def _sandbox_system_read_only() -> list[Path]:
    frozen = _captured_replay_path_inventory(
        _REPLAY_SYSTEM_READ_ONLY_ENV,
        kind="system-read-only",
    )
    if frozen is not None:
        return frozen
    candidates = [
        Path(sys.prefix),
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc/ld.so.cache"),
        Path("/etc/localtime"),
        # Ninja redirects each edge's stdin from /dev/null via a posix_spawn
        # file action.  Without this exact read grant the file action returns
        # EACCES before the compiler is executed.
        Path("/dev/null"),
        Path("/dev/urandom"),
        # Debian's nvcc.profile under /usr is a symlink to this exact file.
        # nvcc needs it to add its private compiler tools (for example cicc)
        # to PATH; granting /etc wholesale would be unnecessary authority.
        Path("/etc/nvcc.profile"),
        Path("/proc/self/maps"),
        Path("/proc/self/status"),
        Path("/proc/self/stat"),
        Path("/proc/self/statm"),
        Path("/proc/self/cmdline"),
        Path("/proc/cpuinfo"),
        Path("/proc/meminfo"),
        Path("/proc/devices"),
        Path("/proc/sys/vm/mmap_min_addr"),
        Path("/proc/driver/nvidia"),
        Path("/sys/devices/system/cpu/online"),
        Path("/sys/devices/system/cpu/possible"),
        Path("/sys/devices/system/node"),
        Path("/sys/devices/system/memory/block_size_bytes"),
    ]
    candidates.extend(sorted(Path("/sys/devices/system/cpu").glob("cpu*/online")))
    candidates.extend(sorted(Path("/sys/module").glob("nvidia*")))
    candidates.extend(sorted(Path("/sys/bus/pci/devices").glob("*/numa_node")))
    return [path for path in candidates if os.path.exists(path)]


def _sandbox_gpu_read_write() -> list[Path]:
    frozen = _captured_replay_path_inventory(
        _REPLAY_GPU_READ_WRITE_ENV,
        kind="gpu-read-write",
    )
    if frozen is not None:
        return frozen
    candidates = [
        Path("/proc/self/task"),
        Path("/dev/nvidia0"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
    ]
    return [
        path if path.is_relative_to(Path("/proc/self")) else path.resolve(strict=True)
        for path in candidates
        if os.path.exists(path)
    ]


def _sandbox_process_read_write() -> list[Path]:
    path = Path("/proc/self/task")
    return [path] if path.exists() else []


def _producer_command(
    *,
    outdir: Path,
    source_path: Path,
    captured_comp009_root: Path,
    native_path: Path,
    lloyd_path: Path,
    config_path: Path,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_produce-cell",
        "--artifact-root",
        os.fspath(_lexical(outdir)),
        "--source",
        os.fspath(_lexical(source_path)),
        "--captured-comp009-root",
        os.fspath(_lexical(captured_comp009_root)),
        "--native",
        os.fspath(_lexical(native_path)),
        "--lloyd",
        os.fspath(_lexical(lloyd_path)),
        "--config",
        os.fspath(_lexical(config_path)),
        "--output-root",
        os.fspath(_lexical(output_root)),
    ]


def _producer_policy_request_path(path: Path, *, allow_missing: bool = False) -> str:
    candidate = _lexical(path)
    proc_self = Path("/proc/self")
    if candidate.is_relative_to(proc_self):
        if not candidate.exists():
            raise LifecycleError(f"producer sandbox request path is absent: {candidate}")
        return candidate.as_posix()
    if allow_missing:
        # A captured-source or private-workspace directory is deleted after the
        # launch.  Reject symlink aliases while it exists, then retain its
        # lexical identity so persisted verification never needs to stat it.
        if candidate.exists() and candidate.resolve(strict=True) != candidate:
            raise LifecycleError("producer sandbox ephemeral path is a symlink alias")
        return candidate.as_posix()
    try:
        return candidate.resolve(strict=True).as_posix()
    except OSError as exc:
        raise LifecycleError(f"producer sandbox request path is absent: {candidate}") from exc


def _producer_launch_policy(
    *,
    outdir: Path,
    source_path: Path,
    captured_comp009_root: Path,
    native_path: Path,
    lloyd_path: Path,
    config_path: Path,
    output_root: Path,
    denied_read_probes: Sequence[Path],
) -> dict[str, Any]:
    """Seal the exact parent request before the producing subprocess exists."""

    outdir = _lexical(outdir)
    workspace = _lexical(config_path).parent
    if (
        _lexical(config_path) != workspace / "config.json"
        or _lexical(output_root) != workspace / "output"
        or not workspace.is_dir()
    ):
        raise LifecycleError("producing-cell private workspace grammar drift")
    if len(denied_read_probes) != 1:
        raise LifecycleError("producing-cell requires exactly one denied target probe")
    source_root = _worker_source_root(outdir)
    environment = _worker_environment(outdir)
    environment["TMPDIR"] = os.fspath(workspace)
    read_only_request = sorted(
        {
            _producer_policy_request_path(
                path,
                allow_missing=path == _lexical(captured_comp009_root),
            )
            for path in (
                *_sandbox_system_read_only(),
                *_worker_source_read_only(source_root),
                outdir / "runtime",
                _lexical(source_path),
                _lexical(captured_comp009_root),
            )
        }
    )
    read_write_request = sorted(
        {
            _producer_policy_request_path(
                path,
                allow_missing=path == workspace,
            )
            for path in (workspace, *_sandbox_process_read_write())
        }
    )
    launcher = source_root / "benchmarks/ssp2v_landlock.py"
    return actual.seal_record(
        {
            "schema": PRODUCER_POLICY_SCHEMA,
            "command": _producer_command(
                outdir=outdir,
                source_path=source_path,
                captured_comp009_root=captured_comp009_root,
                native_path=native_path,
                lloyd_path=lloyd_path,
                config_path=config_path,
                output_root=output_root,
            ),
            "cwd": (source_root / "benchmarks").resolve(strict=True).as_posix(),
            "environment": environment,
            "read_only_request": read_only_request,
            "read_write_request": read_write_request,
            "denied_read_probe_paths": sorted(
                os.fspath(_lexical(path)) for path in denied_read_probes
            ),
            "launcher_path": launcher.as_posix(),
            "launcher_sha256": _sha256_bytes(launcher.read_bytes()),
            "include_repository_source": True,
        },
        "producer_policy_sha256",
    )


def _materialize_producer_request_paths(
    paths: Sequence[str], *, child_pid: int
) -> list[str]:
    proc_self = Path("/proc/self")
    materialized: set[str] = set()
    for value in paths:
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise LifecycleError("producer sandbox request path inventory is malformed")
        path = Path(value)
        if path.is_relative_to(proc_self):
            relative = path.relative_to(proc_self)
            materialized.add(
                os.fspath(Path("/proc") / str(child_pid) / relative)
            )
        else:
            materialized.add(os.fspath(path))
    return sorted(materialized)


def _verify_producer_launch_policy(
    *,
    outdir: Path,
    image_id: str,
    bits: Sequence[int],
    config: Mapping[str, Any],
    worker: Mapping[str, Any],
    attestation: Mapping[str, Any],
    launch: Mapping[str, Any],
    denied_read_probe_paths: Sequence[str],
    expected_source_root: Path | None = None,
    expected_native_record: Mapping[str, Any] | None = None,
    expected_lloyd_record: Mapping[str, Any] | None = None,
    expected_comp009_authority: Mapping[str, Any] | None = None,
    expected_captured_source_manifest_sha256: str | None = None,
    expected_denied_read_probe_paths: Sequence[Path] | None = None,
    preflight_record: Mapping[str, Any] | None = None,
    replay_source_authority: Mapping[str, Any] | None = None,
    expected_parent_sandbox_attestation_sha256: str | None = None,
) -> None:
    """Recompute one persisted producing launch from artifact identities."""

    actual.verify_record(config, "producing_config_sha256")
    policy = config.get("producer_policy")
    if not isinstance(policy, Mapping):
        raise LifecycleError("producing-cell launch policy is absent")
    actual.verify_record(policy, "producer_policy_sha256")
    if set(policy) != {
        "schema",
        "command",
        "cwd",
        "environment",
        "read_only_request",
        "read_write_request",
        "denied_read_probe_paths",
        "launcher_path",
        "launcher_sha256",
        "include_repository_source",
        "producer_policy_sha256",
    }:
        raise LifecycleError("producing-cell launch policy key set drift")
    if set(config) != {
        "schema",
        "protocol",
        "task_sha256",
        "image_id",
        "bit_tuple",
        "source_bytes",
        "source_sha256",
        "native_record",
        "lloyd_record",
        "comp009_authority",
        "captured_source_manifest_sha256",
        "producer_policy",
        "producing_config_sha256",
    }:
        raise LifecycleError("producing-cell config key set drift")
    if (
        policy.get("schema") != PRODUCER_POLICY_SCHEMA
        or policy.get("include_repository_source") is not True
        or config.get("schema") != "structsplat.comp011.producing-cell-config.v1"
        or config.get("protocol") != PROTOCOL
        or config.get("task_sha256") != TASK_SHA256
        or _identity(config, "producing-cell config") != (image_id, tuple(bits))
        or worker.get("producing_config_sha256")
        != config.get("producing_config_sha256")
        or worker.get("producer_policy_sha256")
        != policy.get("producer_policy_sha256")
    ):
        raise LifecycleError("producing-cell config/policy identity drift")
    policy_cwd = policy.get("cwd")
    if not isinstance(policy_cwd, str):
        raise LifecycleError("producing-cell policy cwd is malformed")
    policy_benchmarks = _lexical(Path(policy_cwd))
    policy_root = policy_benchmarks.parent
    if policy_benchmarks != policy_root / "benchmarks" or (
        expected_source_root is not None
        and policy_root != _lexical(expected_source_root)
    ):
        raise LifecycleError("producing-cell policy source root grammar drift")
    replay_verification = (
        preflight_record is not None or replay_source_authority is not None
    )
    if replay_verification and not (
        isinstance(preflight_record, Mapping)
        and isinstance(replay_source_authority, Mapping)
    ):
        raise LifecycleError("producing-cell replay source authority is incomplete")
    replay_authority_root: Path | None = None
    if replay_verification:
        replay_authority_root = _verify_replay_captured_source_authority(
            replay_source_authority,
            preflight_record=preflight_record,
            expected_source_root=policy_root,
        )

    command = policy.get("command")
    if not isinstance(command, list) or len(command) != 18:
        raise LifecycleError("producing-cell command grammar drift")
    command_paths = {
        name: _lexical(Path(command[index]))
        for name, index in (
            ("artifact_root", 5),
            ("source", 7),
            ("captured", 9),
            ("native", 11),
            ("lloyd", 13),
            ("config", 15),
            ("output", 17),
        )
    }
    outdir = _lexical(outdir)
    expected_source = (
        outdir / "streams" / image_id / _tuple_label(bits) / "sspl1.bin"
    )
    artifact_identity = worker.get("artifact_root_identity")
    if not isinstance(artifact_identity, Mapping):
        raise LifecycleError("producing-cell native artifact identities are absent")
    if not all(
        isinstance(value, Mapping)
        for value in (
            expected_native_record,
            expected_lloyd_record,
            expected_comp009_authority,
        )
    ) or expected_captured_source_manifest_sha256 is None:
        raise LifecycleError("producing-cell expected preflight authorities are absent")
    expected_native = outdir / "runtime/native" / str(expected_native_record.get("path"))
    expected_lloyd = outdir / "runtime/native" / str(expected_lloyd_record.get("path"))
    if artifact_identity != {
        "native": expected_native.relative_to(outdir).as_posix(),
        "lloyd": expected_lloyd.relative_to(outdir).as_posix(),
    }:
        raise LifecycleError("producing-cell native artifact identities differ from preflight")
    workspace = command_paths["config"].parent
    expected_command = _producer_command(
        outdir=outdir,
        source_path=expected_source,
        captured_comp009_root=command_paths["captured"],
        native_path=expected_native,
        lloyd_path=expected_lloyd,
        config_path=workspace / "config.json",
        output_root=workspace / "output",
    )
    if command != expected_command or command_paths != {
        "artifact_root": outdir,
        "source": expected_source,
        "captured": command_paths["captured"],
        "native": expected_native,
        "lloyd": expected_lloyd,
        "config": workspace / "config.json",
        "output": workspace / "output",
    }:
        raise LifecycleError("producing-cell command/path binding drift")
    source_blob = _read_regular_no_follow(expected_source)
    if (
        config.get("source_bytes") != len(source_blob)
        or config.get("source_sha256") != _sha256_bytes(source_blob)
    ):
        raise LifecycleError("producing-cell config source identity drift")
    for name, path in (("native_record", expected_native), ("lloyd_record", expected_lloyd)):
        authority = config.get(name)
        if not isinstance(authority, Mapping):
            raise LifecycleError(f"producing-cell {name} is absent")
        _verify_regular_record(path, authority)
    if (
        config.get("native_record") != expected_native_record
        or config.get("lloyd_record") != expected_lloyd_record
        or config.get("comp009_authority") != expected_comp009_authority
        or config.get("captured_source_manifest_sha256")
        != expected_captured_source_manifest_sha256
    ):
        raise LifecycleError("producing-cell config differs from preflight/upstream authorities")

    source_read_only: list[Path]
    if replay_verification:
        policy_environment = policy.get("environment")
        if not isinstance(policy_environment, Mapping):
            raise LifecycleError("producing-cell replay environment is absent")
        _captured_worker_environment_semantics(
            policy_environment,
            outdir=outdir,
            captured_source_root=policy_root,
            tmpdir=workspace,
            preflight_record=preflight_record,
        )
        if (
            policy_environment.get("STRUCTSPLAT_SOURCE_MANIFEST_SHA256")
            != replay_source_authority.get("source_manifest_sha256")
        ):
            raise LifecycleError(
                "producing-cell replay source manifest environment drift"
            )
        expected_environment = dict(policy_environment)
        source_read_only = [replay_authority_root]
        expected_launcher_sha256 = str(
            replay_source_authority["launcher"]["sha256"]
        )
    else:
        expected_environment = _worker_environment(outdir)
        if policy_root != _worker_source_root(outdir):
            expected_environment["PYTHONPATH"] = os.pathsep.join(
                (
                    os.fspath(outdir / "runtime/metrics/site"),
                    os.fspath(policy_root),
                    os.fspath(policy_root / "src"),
                )
            )
            captured_manifest = _verify_runtime_source_tree(policy_root)
            expected_environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
                captured_manifest["source_manifest_sha256"]
            )
        expected_environment["TMPDIR"] = os.fspath(workspace)
        source_read_only = _worker_source_read_only(policy_root)
        live_launcher = Path(sandbox.__file__).resolve(strict=True)
        expected_launcher_sha256 = _sha256_bytes(live_launcher.read_bytes())
    expected_read_only = sorted(
        {
            _producer_policy_request_path(
                path,
                allow_missing=path
                in {
                    command_paths["captured"],
                    *source_read_only,
                },
            )
            for path in (
                *_sandbox_system_read_only(),
                *source_read_only,
                outdir / "runtime",
                expected_source,
                command_paths["captured"],
            )
        }
    )
    expected_read_write = sorted(
        {
            _producer_policy_request_path(
                path,
                allow_missing=path == workspace,
            )
            for path in (workspace, *_sandbox_process_read_write())
        }
    )
    launcher = policy_root / "benchmarks/ssp2v_landlock.py"
    denied_paths = sorted(str(value) for value in denied_read_probe_paths)
    expected_denied = (
        sorted(os.fspath(_lexical(path)) for path in expected_denied_read_probe_paths)
        if expected_denied_read_probe_paths is not None
        else None
    )
    if (
        policy.get("cwd") != policy_benchmarks.as_posix()
        or policy.get("environment") != expected_environment
        or policy.get("read_only_request") != expected_read_only
        or policy.get("read_write_request") != expected_read_write
        or policy.get("denied_read_probe_paths") != denied_paths
        or len(denied_paths) != 1
        or denied_paths != expected_denied
        or policy.get("launcher_path") != launcher.as_posix()
        or policy.get("launcher_sha256") != expected_launcher_sha256
    ):
        raise LifecycleError("producing-cell persisted launch request drift")

    _verify_launch_bundle(
        {"attestation": attestation, "launch": launch},
        worker,
        "producing_worker_sha256",
        expected_parent_sandbox_attestation_sha256=(
            expected_parent_sandbox_attestation_sha256
        ),
    )
    nonce = str(launch.get("nonce"))
    attested_environment = dict(expected_environment)
    attested_environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
    child_pid = attestation.get("pid")
    if isinstance(child_pid, bool) or not isinstance(child_pid, int):
        raise LifecycleError("producing-cell attested PID is malformed")
    denied_records = attestation.get("denied_read_probes")
    if (
        attestation.get("command") != expected_command
        or attestation.get("cwd") != policy.get("cwd")
        or attestation.get("environment_keys") != sorted(attested_environment)
        or attestation.get("environment_sha256_before_attestation_marker")
        != _sha256_bytes(_canonical_json(attested_environment))
        or attestation.get("read_only")
        != _materialize_producer_request_paths(expected_read_only, child_pid=child_pid)
        or attestation.get("read_write")
        != _materialize_producer_request_paths(expected_read_write, child_pid=child_pid)
        or attestation.get("launcher_sha256") != policy.get("launcher_sha256")
        or not isinstance(denied_records, list)
        or [item.get("path") for item in denied_records] != denied_paths
        or any(
            item.get("errno") not in (errno.EACCES, errno.EPERM)
            for item in denied_records
        )
    ):
        raise LifecycleError("producing-cell materialized sandbox policy drift")


def _worker_policy_path(path: Path) -> str:
    path = _lexical(path)
    if path.is_relative_to(Path("/proc/self")):
        return path.as_posix()
    return path.resolve(strict=True).as_posix()


def _required_fixed_system_read_only_policy_paths(
    *, include_devnull: bool
) -> set[Path]:
    candidates = [
        Path(sys.prefix),
        Path("/usr"),
        Path("/lib"),
        Path("/etc/ld.so.cache"),
        Path("/dev/urandom"),
        Path("/proc/self/maps"),
        Path("/proc/self/status"),
        Path("/proc/self/stat"),
        Path("/proc/self/statm"),
        Path("/proc/self/cmdline"),
        Path("/proc/cpuinfo"),
        Path("/proc/meminfo"),
        Path("/proc/devices"),
        Path("/proc/sys/vm/mmap_min_addr"),
        Path("/sys/devices/system/cpu/online"),
        Path("/sys/devices/system/cpu/possible"),
    ]
    if include_devnull:
        candidates.append(Path(os.devnull))
    return {
        _lexical(Path(_worker_policy_path(path)))
        for path in candidates
    }


def _encode_replay_path_inventory(paths: Sequence[Path]) -> str:
    values = sorted({_worker_policy_path(path) for path in paths})
    return _canonical_json(values).decode("utf-8").removesuffix("\n")


def _captured_replay_path_inventory(
    environment_name: str,
    *,
    kind: str,
) -> list[Path] | None:
    raw = os.environ.get(environment_name)
    if raw is None:
        return None
    counterpart = (
        _REPLAY_GPU_READ_WRITE_ENV
        if environment_name == _REPLAY_SYSTEM_READ_ONLY_ENV
        else _REPLAY_SYSTEM_READ_ONLY_ENV
    )
    if (
        _captured_replay_source_root() is None
        or os.environ.get(counterpart) is None
        or os.environ.get(_REPLAY_PREFLIGHT_BINDING_ENV) is None
        or os.environ.get(sandbox.SANDBOX_ATTESTATION_ENV) is None
    ):
        raise LifecycleError(
            "captured replay path inventory lacks its source, preflight, or sandbox binding"
        )
    _require_sha256(
        os.environ.get(_REPLAY_PREFLIGHT_BINDING_ENV),
        "captured replay path-inventory preflight binding",
    )
    _require_sha256(
        os.environ.get(sandbox.SANDBOX_ATTESTATION_ENV),
        "captured replay path-inventory sandbox binding",
    )
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LifecycleError("captured replay path inventory is invalid JSON") from exc
    if (
        not isinstance(values, list)
        or any(
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or _lexical(Path(value)).as_posix() != value
            for value in values
        )
    ):
        raise LifecycleError("captured replay path inventory is malformed")
    if (
        values != sorted(set(values))
        or raw != _canonical_json(values).decode("utf-8").removesuffix("\n")
    ):
        raise LifecycleError("captured replay path inventory is malformed")
    paths = [_lexical(Path(value)) for value in values]
    if kind == "system-read-only":
        if (
            not _required_fixed_system_read_only_policy_paths(
                include_devnull=True
            ).issubset(set(paths))
            or any(not _safe_system_read_only_policy_path(path) for path in paths)
        ):
            raise LifecycleError(
                "captured replay system read-only inventory exceeds frozen grammar"
            )
    elif kind == "gpu-read-write":
        allowed = {
            Path("/proc/self/task"),
            Path("/dev/nvidia0"),
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia-uvm"),
        }
        if Path("/proc/self/task") not in paths or any(
            path not in allowed for path in paths
        ):
            raise LifecycleError(
                "captured replay GPU/process inventory exceeds frozen grammar"
            )
    else:  # pragma: no cover - private caller invariant
        raise AssertionError(f"unknown captured replay path inventory kind {kind}")
    return paths


def _replay_path_inventory_environment(
    *,
    system_read_only: Sequence[Path] | None = None,
    gpu_read_write: Sequence[Path] | None = None,
) -> dict[str, str]:
    system_paths = list(
        _sandbox_system_read_only()
        if system_read_only is None
        else system_read_only
    )
    gpu_paths = list(
        _sandbox_gpu_read_write()
        if gpu_read_write is None
        else gpu_read_write
    )
    return {
        _REPLAY_SYSTEM_READ_ONLY_ENV: _encode_replay_path_inventory(system_paths),
        _REPLAY_GPU_READ_WRITE_ENV: _encode_replay_path_inventory(gpu_paths),
    }


def _worker_launch_policy(
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    read_only: Sequence[Path],
    read_write: Sequence[Path],
    denied_read_probes: Sequence[Path],
    launcher_path: Path,
    timeout: float,
) -> dict[str, Any]:
    if timeout <= 0 or not np.isfinite(timeout):
        raise LifecycleError("worker launch timeout is invalid")
    launcher = launcher_path.resolve(strict=True)
    return actual.seal_record(
        {
            "schema": WORKER_POLICY_SCHEMA,
            "command": list(command),
            "cwd": cwd.resolve(strict=True).as_posix(),
            "environment": dict(environment),
            "read_only_request": sorted({_worker_policy_path(path) for path in read_only}),
            "read_write_request": sorted({_worker_policy_path(path) for path in read_write}),
            "denied_read_probe_paths": sorted(
                os.fspath(_lexical(path)) for path in denied_read_probes
            ),
            "launcher_path": launcher.as_posix(),
            "launcher_sha256": _sha256_file(launcher),
            "timeout_seconds": float(timeout),
        },
        "worker_policy_sha256",
    )


def _verify_worker_launch_policy(
    policy: Mapping[str, Any],
    *,
    attestation: Mapping[str, Any],
    launch_nonce: str,
) -> None:
    actual.verify_record(policy, "worker_policy_sha256")
    if set(policy) != {
        "schema",
        "command",
        "cwd",
        "environment",
        "read_only_request",
        "read_write_request",
        "denied_read_probe_paths",
        "launcher_path",
        "launcher_sha256",
        "timeout_seconds",
        "worker_policy_sha256",
    } or policy.get("schema") != WORKER_POLICY_SCHEMA:
        raise LifecycleError("worker launch policy schema drift")
    environment = policy.get("environment")
    readonly = policy.get("read_only_request")
    writable = policy.get("read_write_request")
    denied = policy.get("denied_read_probe_paths")
    child_pid = attestation.get("pid")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(readonly, list)
        or not isinstance(writable, list)
        or not isinstance(denied, list)
        or isinstance(child_pid, bool)
        or not isinstance(child_pid, int)
    ):
        raise LifecycleError("worker launch policy inventories are malformed")
    expected_environment = dict(environment)
    expected_environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = launch_nonce
    denied_records = attestation.get("denied_read_probes")
    launcher_path = policy.get("launcher_path")
    if (
        not isinstance(launcher_path, str)
        or not Path(launcher_path).is_absolute()
        or attestation.get("command") != policy.get("command")
        or attestation.get("cwd") != policy.get("cwd")
        or attestation.get("environment_keys") != sorted(expected_environment)
        or attestation.get("environment_sha256_before_attestation_marker")
        != _sha256_bytes(_canonical_json(expected_environment))
        or attestation.get("read_only")
        != _materialize_producer_request_paths(readonly, child_pid=child_pid)
        or attestation.get("read_write")
        != _materialize_producer_request_paths(writable, child_pid=child_pid)
        or attestation.get("launcher_sha256") != policy.get("launcher_sha256")
        or not isinstance(denied_records, list)
        or [item.get("path") for item in denied_records] != denied
        or any(
            item.get("errno") not in (errno.EACCES, errno.EPERM)
            for item in denied_records
        )
        or any(
            path in set(attestation.get(name, []))
            for path in denied
            for name in ("read_only", "read_write")
        )
    ):
        raise LifecycleError("worker launch attestation differs from persisted request")


def _launch_receipt(
    *,
    nonce: str,
    command: Sequence[str],
    worker: Mapping[str, Any],
    worker_seal_field: str,
    attestation: Mapping[str, Any],
    worker_policy: Mapping[str, Any],
) -> dict[str, Any]:
    worker_seal = (
        _sha256_bytes(_canonical_json(worker))
        if worker_seal_field == "canonical_worker_sha256"
        else _require_sha256(worker.get(worker_seal_field), worker_seal_field)
    )
    return actual.seal_record(
        {
            "schema": "structsplat.comp011.parent-launch-receipt.v1",
            "nonce": nonce,
            "command_sha256": _sha256_bytes(_canonical_json(list(command))),
            "worker_seal_field": worker_seal_field,
            "worker_seal_sha256": worker_seal,
            "child_pid": attestation["pid"],
            "sandbox_attestation_sha256": attestation["attestation_sha256"],
            "worker_policy": dict(worker_policy),
            "fresh_subprocess_launch": True,
        },
        "launch_receipt_sha256",
    )


def _verify_launch_bundle(
    bundle: Mapping[str, Any],
    worker: Mapping[str, Any],
    worker_seal_field: str,
    *,
    expected_parent_sandbox_attestation_sha256: str | None = None,
) -> None:
    if set(bundle) != {"attestation", "launch"}:
        raise LifecycleError("sandbox/launch bundle key set drift")
    attestation = bundle["attestation"]
    launch = bundle["launch"]
    if not isinstance(attestation, Mapping) or not isinstance(launch, Mapping):
        raise LifecycleError("sandbox/launch bundle is malformed")
    sandbox.verify_attestation(attestation)
    actual.verify_record(launch, "launch_receipt_sha256")
    policy = launch.get("worker_policy")
    if (
        launch.get("schema") != "structsplat.comp011.parent-launch-receipt.v1"
        or not isinstance(policy, Mapping)
        or launch.get("worker_seal_field") != worker_seal_field
        or launch.get("worker_seal_sha256")
        != (
            _sha256_bytes(_canonical_json(worker))
            if worker_seal_field == "canonical_worker_sha256"
            else worker.get(worker_seal_field)
        )
        or launch.get("child_pid") != attestation.get("pid")
        or launch.get("sandbox_attestation_sha256")
        != attestation.get("attestation_sha256")
        or launch.get("command_sha256") != attestation.get("command_sha256")
        or launch.get("fresh_subprocess_launch") is not True
    ):
        raise LifecycleError("sandbox/launch bundle is detached from worker")
    _require_sha256(launch.get("command_sha256"), "launch command")
    nonce = launch.get("nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 64
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise LifecycleError("launch nonce is malformed")
    _verify_worker_launch_policy(
        policy,
        attestation=attestation,
        launch_nonce=nonce,
    )
    _verify_sandbox_ancestry(
        attestation,
        expected_parent_sandbox_attestation_sha256=(
            expected_parent_sandbox_attestation_sha256
        ),
    )


def _verify_sandbox_ancestry(
    attestation: Mapping[str, Any],
    *,
    expected_parent_sandbox_attestation_sha256: str | None,
) -> None:
    """Require one exact fresh-root or inherited-nested sandbox lineage edge."""

    sandbox.verify_attestation(attestation)
    mode = attestation.get("restriction_mode")
    parent = attestation.get("parent_sandbox_attestation_sha256")
    if expected_parent_sandbox_attestation_sha256 is None:
        if mode != sandbox.RESTRICTION_MODE_FRESH or parent is not None:
            raise LifecycleError(
                "sandbox ancestry is not a fresh root with null parent"
            )
        return
    expected_parent = _require_sha256(
        expected_parent_sandbox_attestation_sha256,
        "parent sandbox attestation",
    )
    if mode != sandbox.RESTRICTION_MODE_INHERITED or parent != expected_parent:
        raise LifecycleError(
            "nested sandbox ancestry differs from its immediate parent"
        )


def _run_json_worker(
    command: Sequence[str],
    *,
    outdir: Path | None,
    timeout: float,
    read_only: Sequence[Path] = (),
    read_write: Sequence[Path] = (),
    denied_read_probes: Sequence[Path] = (),
    worker_seal_field: str | None = None,
    launcher_path: Path | None = None,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
    include_repository_source: bool = True,
    system_read_only: Sequence[Path] | None = None,
) -> Any:
    nonce = os.urandom(32).hex()
    source_root = _worker_source_root(outdir)
    captured_source = (source_root / "SOURCE_MANIFEST.json").is_file()
    if include_repository_source and captured_source:
        _verify_runtime_source_tree(source_root)
    policy_environment = dict(environment or _worker_environment(outdir))
    worker_environment = dict(policy_environment)
    worker_environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
    system_paths = (
        list(_sandbox_system_read_only())
        if system_read_only is None
        else list(system_read_only)
    )
    readonly = [
        *system_paths,
        *(_worker_source_read_only(source_root) if include_repository_source else []),
        *read_only,
    ]
    writable = list(read_write)
    worker_cwd = cwd or source_root / "benchmarks"
    worker_launcher = launcher_path or source_root / "benchmarks/ssp2v_landlock.py"
    resolved_readonly = {path.resolve(strict=True) for path in readonly}
    resolved_writable = {path.resolve(strict=True) for path in writable}
    if any(
        read == write or read.is_relative_to(write) or write.is_relative_to(read)
        for read in resolved_readonly
        for write in resolved_writable
    ):
        raise LifecycleError("sandbox read-only/read-write allowlists overlap by ancestry")
    policy = _worker_launch_policy(
        command=command,
        cwd=worker_cwd,
        environment=policy_environment,
        read_only=readonly,
        read_write=writable,
        denied_read_probes=denied_read_probes,
        launcher_path=worker_launcher,
        timeout=timeout,
    )
    try:
        completed, attestation = sandbox.run_sandboxed(
            list(command),
            read_only=readonly,
            read_write=writable,
            denied_read_probes=list(denied_read_probes),
            cwd=worker_cwd,
            env=worker_environment,
            timeout=timeout,
            launcher_path=worker_launcher,
        )
    finally:
        if include_repository_source and captured_source:
            _verify_runtime_source_tree(source_root)
    if completed.returncode != 0 or completed.stderr:
        raise LifecycleError(
            "preflight proof worker failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr.decode(errors='replace')}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LifecycleError("preflight proof worker emitted invalid JSON") from exc
    if not isinstance(value, dict) or completed.stdout != _canonical_json(value):
        raise LifecycleError("preflight proof worker output is not canonical JSON")
    sandbox.verify_attestation(attestation)
    _verify_sandbox_ancestry(
        attestation,
        expected_parent_sandbox_attestation_sha256=os.environ.get(
            sandbox.SANDBOX_ATTESTATION_ENV
        ),
    )
    expected_readonly = sorted(
        (
            f"/proc/{attestation['pid']}/{path.relative_to('/proc/self').as_posix()}"
            if path.is_relative_to(Path("/proc/self"))
            else os.fspath(path.resolve(strict=True))
        )
        for path in readonly
    )
    expected_writable = sorted(
        (
            f"/proc/{attestation['pid']}/{path.relative_to('/proc/self').as_posix()}"
            if path.is_relative_to(Path("/proc/self"))
            else os.fspath(path.resolve(strict=True))
        )
        for path in writable
    )
    if (
        attestation.get("read_only") != sorted(set(expected_readonly))
        or attestation.get("read_write") != sorted(set(expected_writable))
    ):
        raise LifecycleError("sandbox attestation allowlists differ from the exact request")
    if worker_seal_field is None:
        return value
    receipt = _launch_receipt(
        nonce=nonce,
        command=command,
        worker=value,
        worker_seal_field=worker_seal_field,
        attestation=attestation,
        worker_policy=policy,
    )
    return value, attestation, receipt


def _focused_proof_command(base_temp: Path) -> list[str]:
    selected_tests = [Path(relative).name for relative in _FOCUSED_TESTS]
    return [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--noconftest",
        "--confcutdir",
        ".",
        "-c",
        os.devnull,
        "--rootdir",
        ".",
        "--basetemp",
        os.fspath(base_temp),
        "--deselect=test_ssp2v_actual_run.py"
        "::test_upstream_binding_opens_only_sealed_metadata_and_captured_source",
        *(f"--deselect={node}" for node in _FOCUSED_HOST_PROBE_DESELECTS),
        *selected_tests,
    ]


def _focused_proof_read_only(
    source_root: Path, dependencies_root: Path | None = None
) -> list[Path]:
    dependency_root = dependencies_root or source_root.parent / "dependencies"
    read_only = [
        *(
            path
            for path in _sandbox_system_read_only()
            if path.resolve(strict=True) != Path(os.devnull).resolve(strict=True)
        ),
        source_root,
        dependency_root,
    ]
    forbidden_host_probe_ancestors = {
        Path("/proc"),
        Path("/sys/devices/system/cpu"),
    }
    if forbidden_host_probe_ancestors & {_lexical(path) for path in read_only}:
        raise LifecycleError("focused proof grants excessive host-probe read authority")
    return read_only


def _focused_proof_read_write(workspace: Path) -> list[Path]:
    return [workspace, Path(os.devnull), *_sandbox_gpu_read_write()]


def _focused_proof_environment(
    source_root: Path,
    dependencies_root: Path,
    scratch: Path,
    *,
    source_manifest_sha256: str,
) -> dict[str, str]:
    environment = _worker_environment(source_root=source_root)
    environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = _require_sha256(
        source_manifest_sha256, "focused proof source manifest"
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (os.fspath(source_root), os.fspath(source_root / "src"))
    )
    environment["TMPDIR"] = os.fspath(scratch)
    environment["TORCH_HOME"] = os.fspath(dependencies_root / "torch")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def _verify_focused_proof_path_policy(
    policy: Mapping[str, Any],
    *,
    source_root: Path,
    dependencies_root: Path,
    scratch: Path,
    expected_probe_paths: Sequence[Path],
    require_live_inventory: bool,
) -> None:
    """Verify focused-proof capabilities, optionally without dynamic tree scans.

    Captured replay cannot enumerate the parent directories of dynamically
    granted CPU, PCI, and NVIDIA leaves under Landlock.  It still requires the
    fixed runtime baseline below; only those dynamic leaves are accepted by
    grammar under a prevalidated outer replay launch.
    """

    read_only = policy.get("read_only_request")
    read_write = policy.get("read_write_request")
    denied = policy.get("denied_read_probe_paths")
    if (
        not isinstance(read_only, list)
        or not isinstance(read_write, list)
        or not isinstance(denied, list)
        or read_only != sorted(set(read_only))
        or read_write != sorted(set(read_write))
        or denied != sorted(set(denied))
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in (*read_only, *read_write, *denied)
        )
        or any(
            _lexical(Path(value)).as_posix() != value
            for value in (*read_only, *read_write, *denied)
        )
    ):
        raise LifecycleError("focused proof path policy inventory is malformed")
    ro_paths = {_lexical(Path(value)) for value in read_only}
    rw_paths = {_lexical(Path(value)) for value in read_write}
    workspace = scratch.parent
    exact_workspace_ro = {_lexical(source_root), _lexical(dependencies_root)}
    exact_workspace_rw = {_lexical(scratch)}
    writable_devnull = _lexical(Path(_worker_policy_path(Path(os.devnull))))
    required_fixed_ro = _required_fixed_system_read_only_policy_paths(
        include_devnull=False
    )
    allowed_gpu_rw = {
        Path("/proc/self/task"),
        Path("/dev/nvidia0"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
    }
    expected_denied = sorted(
        os.fspath(_lexical(path)) for path in expected_probe_paths
    )
    live_system_ro: set[Path] = set()
    live_rw: set[Path] = set()
    if require_live_inventory:
        live_system_ro = {
            _lexical(Path(_worker_policy_path(path)))
            for path in _sandbox_system_read_only()
            if _lexical(Path(_worker_policy_path(path))) != writable_devnull
        }
        live_rw = {
            _lexical(Path(_worker_policy_path(path)))
            for path in _sandbox_gpu_read_write()
        }
    if (
        not exact_workspace_ro.issubset(ro_paths)
        or not required_fixed_ro.issubset(ro_paths)
        or not exact_workspace_rw.issubset(rw_paths)
        or writable_devnull not in rw_paths
        or (
            require_live_inventory
            and ro_paths != exact_workspace_ro | live_system_ro
        )
        or (
            require_live_inventory
            and rw_paths
            != exact_workspace_rw | {writable_devnull} | live_rw
        )
        or {path for path in ro_paths if path.is_relative_to(workspace)}
        != exact_workspace_ro
        or {path for path in rw_paths if path.is_relative_to(workspace)}
        != exact_workspace_rw
        or any(
            path not in exact_workspace_ro
            and not _safe_system_read_only_policy_path(path)
            for path in ro_paths
        )
        or any(
            path not in exact_workspace_rw
            and path != writable_devnull
            and path not in allowed_gpu_rw
            for path in rw_paths
        )
        or ro_paths & rw_paths
        or denied != expected_denied
        or any(path in ro_paths or path in rw_paths for path in map(Path, denied))
    ):
        raise LifecycleError("focused proof path capabilities differ from frozen policy")


def _materialize_focused_source_copy(
    source_root: Path,
    manifest: Mapping[str, Any],
    *,
    guard: AccessGuard | None,
) -> None:
    actual.verify_record(manifest, "source_manifest_sha256")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise LifecycleError("focused proof source manifest is empty")
    source_root.mkdir(mode=0o755, parents=False, exist_ok=False)
    expected: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise LifecycleError("focused proof source record is malformed")
        relative = _safe_relative_path(item.get("path"), "focused proof source path")
        payload = _source_payload(item, guard)
        if len(payload) != item.get("bytes") or _sha256_bytes(payload) != item.get("sha256"):
            raise LifecycleError("focused proof source copy differs from its manifest")
        _exclusive_write(source_root / relative, payload)
        expected.add(relative.as_posix())
    observed = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise LifecycleError("focused proof source copy inventory drift")


def _verify_focused_source_copy(source_root: Path, manifest: Mapping[str, Any]) -> None:
    expected = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in manifest["files"]
    }
    observed = {
        path.relative_to(source_root).as_posix(): path
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if set(observed) != set(expected):
        raise LifecycleError("focused proof source copy changed during execution")
    for relative, path in observed.items():
        payload = _read_regular_no_follow(path)
        if (len(payload), _sha256_bytes(payload)) != expected[relative]:
            raise LifecycleError("focused proof source bytes changed during execution")


def _validate_focused_dependencies(record: Mapping[str, Any]) -> dict[str, Any]:
    actual.verify_record(record, "focused_dependency_manifest_sha256")
    files = record.get("files")
    expected_path = (Path("torch") / _ALEXNET_CACHE_RELATIVE).as_posix()
    if (
        set(record)
        != {
            "schema",
            "files",
            "network_download_permitted",
            "focused_dependency_manifest_sha256",
        }
        or record.get("schema") != FOCUSED_DEPENDENCY_SCHEMA
        or record.get("network_download_permitted") is not False
        or not isinstance(files, list)
        or len(files) != 1
        or not isinstance(files[0], Mapping)
        or set(files[0]) != {"path", "bytes", "sha256"}
        or files[0].get("path") != expected_path
        or isinstance(files[0].get("bytes"), bool)
        or not isinstance(files[0].get("bytes"), int)
        or int(files[0]["bytes"]) <= 0
    ):
        raise LifecycleError("focused proof dependency manifest is malformed")
    _require_sha256(files[0].get("sha256"), "focused AlexNet dependency")
    return dict(files[0])


def _materialize_focused_dependencies(
    dependencies_root: Path, *, guard: AccessGuard | None
) -> dict[str, Any]:
    source = _alexnet_cache_path()
    if guard is None:
        payload = _read_regular_no_follow(source)
    else:
        guard.register(source, "runtime_dependency")
        payload = guard.read_bytes(
            source,
            stage="preflight",
            purpose="copy focused-proof AlexNet dependency",
        )
    relative = Path("torch") / _ALEXNET_CACHE_RELATIVE
    _exclusive_write(dependencies_root / relative, payload)
    record = actual.seal_record(
        {
            "schema": FOCUSED_DEPENDENCY_SCHEMA,
            "files": [
                {
                    "path": relative.as_posix(),
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            ],
            "network_download_permitted": False,
        },
        "focused_dependency_manifest_sha256",
    )
    _validate_focused_dependencies(record)
    return record


def _verify_focused_dependencies(
    dependencies_root: Path, record: Mapping[str, Any]
) -> None:
    expected = _validate_focused_dependencies(record)
    observed = {
        path.relative_to(dependencies_root).as_posix(): path
        for path in dependencies_root.rglob("*")
        if path.is_file()
    }
    if set(observed) != {expected["path"]}:
        raise LifecycleError("focused proof dependency inventory changed during execution")
    payload = _read_regular_no_follow(observed[str(expected["path"])])
    if len(payload) != expected["bytes"] or _sha256_bytes(payload) != expected["sha256"]:
        raise LifecycleError("focused proof dependency bytes changed during execution")


def _verify_focused_metric_dependency_binding(
    focused: Mapping[str, Any], metrics: Mapping[str, Any]
) -> None:
    dependencies = focused.get("captured_runtime_dependencies")
    if not isinstance(dependencies, Mapping):
        raise LifecycleError("focused proof dependency manifest is absent")
    item = _validate_focused_dependencies(dependencies)
    expected_metric_path = (Path("runtime/metrics") / str(item["path"])).as_posix()
    if (
        metrics.get("alexnet_relative_path") != expected_metric_path
        or metrics.get("alexnet_bytes") != item["bytes"]
        or metrics.get("alexnet_sha256") != item["sha256"]
        or metrics.get("network_download_permitted") is not False
    ):
        raise LifecycleError(
            "focused proof and persisted metric dependency identities differ"
        )


def focused_proof_suite(
    *,
    denied_read_probes: Sequence[Path],
    source_manifest_record: Mapping[str, Any] | None = None,
    guard: AccessGuard | None = None,
) -> dict[str, Any]:
    if (
        len(_FOCUSED_TESTS) != len(set(_FOCUSED_TESTS))
        or not set(_FOCUSED_TESTS).issubset(_EXPLICIT_SOURCES)
        or any(
            not (ROOT / relative).is_file() or (ROOT / relative).is_symlink()
            for relative in _FOCUSED_TESTS
        )
    ):
        raise LifecycleError("focused proof tests are absent from the source closure")
    manifest = dict(source_manifest_record or source_manifest(guard=guard))
    actual.verify_record(manifest, "source_manifest_sha256")
    with tempfile.TemporaryDirectory(prefix="comp011-focused-suite-") as temporary:
        workspace = Path(temporary)
        source_root = workspace / "source"
        dependencies_root = workspace / "dependencies"
        scratch = workspace / "scratch"
        _materialize_focused_source_copy(source_root, manifest, guard=guard)
        dependency_record = _materialize_focused_dependencies(
            dependencies_root, guard=guard
        )
        scratch.mkdir(mode=0o755, parents=False, exist_ok=False)
        base_temp = scratch / "pytest"
        base_temp.mkdir()
        command = _focused_proof_command(base_temp)
        environment = _focused_proof_environment(
            source_root,
            dependencies_root,
            scratch,
            source_manifest_sha256=str(manifest["source_manifest_sha256"]),
        )
        focused_readonly = _focused_proof_read_only(source_root, dependencies_root)
        focused_writable = _focused_proof_read_write(scratch)
        launcher = source_root / "benchmarks/ssp2v_landlock.py"
        policy = _worker_launch_policy(
            command=command,
            cwd=source_root / "tests",
            environment=environment,
            read_only=focused_readonly,
            read_write=focused_writable,
            denied_read_probes=denied_read_probes,
            launcher_path=launcher,
            timeout=1800,
        )
        nonce = os.urandom(32).hex()
        environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
        started = time.perf_counter_ns()
        completed, attestation = sandbox.run_sandboxed(
            command,
            read_only=focused_readonly,
            read_write=focused_writable,
            denied_read_probes=list(denied_read_probes),
            cwd=source_root / "tests",
            env=environment,
            timeout=1800,
            launcher_path=launcher,
        )
        _verify_sandbox_ancestry(
            attestation,
            expected_parent_sandbox_attestation_sha256=os.environ.get(
                sandbox.SANDBOX_ATTESTATION_ENV
            ),
        )
        _verify_focused_source_copy(source_root, manifest)
        _verify_focused_dependencies(dependencies_root, dependency_record)
        base_record = {
            "command": command,
            "returncode": completed.returncode,
            "stdout_bytes": len(completed.stdout),
            "stdout_sha256": _sha256_bytes(completed.stdout),
            "stdout_hex": completed.stdout.hex(),
            "stderr_bytes": len(completed.stderr),
            "stderr_sha256": _sha256_bytes(completed.stderr),
            "stderr_hex": completed.stderr.hex(),
            "wall_ns": time.perf_counter_ns() - started,
            "captured_source_manifest_sha256": manifest["source_manifest_sha256"],
            "captured_source_file_count": len(manifest["files"]),
            "captured_source_reverified_after_proof": True,
            "captured_runtime_dependencies": dependency_record,
            "captured_dependencies_reverified_after_proof": True,
        }
        launch = _launch_receipt(
            nonce=nonce,
            command=command,
            worker=base_record,
            worker_seal_field="canonical_worker_sha256",
            attestation=attestation,
            worker_policy=policy,
        )
        record = {
            **base_record,
            "sandbox_attestation": attestation,
            "launch": launch,
        }
    if completed.returncode != 0:
        raise LifecycleError(
            "focused COMP-011 proof suite failed: "
            + completed.stdout.decode(errors="replace")
            + completed.stderr.decode(errors="replace")
        )
    return record


def _verify_focused_proof_policy(
    focused: Mapping[str, Any],
    *,
    expected_probe_paths: Sequence[Path],
    forbidden_roots: Sequence[Path] = (),
    expected_source_manifest_sha256: str | None = None,
    expected_source_file_count: int | None = None,
    require_live_inventory: bool = True,
) -> None:
    """Verify the target-free pytest sandbox request and attestation.

    The default mode recomputes exact live membership.  Captured replay uses a
    receipt-bound semantic inventory because Landlock intentionally prevents
    re-enumerating the parents of granted dynamic system leaves.
    """

    if set(focused) != {*_FOCUSED_BASE_FIELDS, "sandbox_attestation", "launch"}:
        raise LifecycleError("focused proof record key set drift")
    command = focused.get("command")
    attestation = focused.get("sandbox_attestation")
    launch = focused.get("launch")
    if (
        not isinstance(command, list)
        or "--basetemp" not in command
        or not isinstance(attestation, Mapping)
        or not isinstance(launch, Mapping)
    ):
        raise LifecycleError("focused proof launch evidence is malformed")
    index = command.index("--basetemp")
    if index + 1 >= len(command):
        raise LifecycleError("focused proof basetemp argument is absent")
    base_temp = _lexical(Path(str(command[index + 1])))
    scratch = base_temp.parent
    workspace = scratch.parent
    dependencies_root = workspace / "dependencies"
    policy = launch.get("worker_policy")
    policy_cwd = policy.get("cwd") if isinstance(policy, Mapping) else None
    source_root = (
        _lexical(Path(policy_cwd)).parent
        if isinstance(policy_cwd, str)
        else Path()
    )
    expected_source_sha256 = _require_sha256(
        (
            expected_source_manifest_sha256
            if expected_source_manifest_sha256 is not None
            else focused.get("captured_source_manifest_sha256")
        ),
        "focused proof source manifest",
    )
    expected_file_count = (
        expected_source_file_count
        if expected_source_file_count is not None
        else focused.get("captured_source_file_count")
    )
    if (
        base_temp != scratch / "pytest"
        or scratch != workspace / "scratch"
        or source_root != workspace / "source"
        or policy_cwd != os.fspath(source_root / "tests")
        or not workspace.name.startswith("comp011-focused-suite-")
        or command != _focused_proof_command(base_temp)
        or focused.get("captured_source_manifest_sha256")
        != expected_source_sha256
        or isinstance(focused.get("captured_source_file_count"), bool)
        or not isinstance(focused.get("captured_source_file_count"), int)
        or focused.get("captured_source_file_count") <= 0
        or focused.get("captured_source_file_count") != expected_file_count
        or focused.get("captured_source_reverified_after_proof") is not True
        or focused.get("captured_dependencies_reverified_after_proof") is not True
        or any(
            workspace == _lexical(root)
            or workspace.is_relative_to(_lexical(root))
            or _lexical(root).is_relative_to(workspace)
            for root in forbidden_roots
        )
    ):
        raise LifecycleError("focused proof command/private-workspace drift")
    dependencies = focused.get("captured_runtime_dependencies")
    if not isinstance(dependencies, Mapping):
        raise LifecycleError("focused proof dependency manifest is absent")
    _validate_focused_dependencies(dependencies)
    focused_base = {key: focused[key] for key in _FOCUSED_BASE_FIELDS}
    _verify_launch_bundle(
        {"attestation": attestation, "launch": launch},
        focused_base,
        "canonical_worker_sha256",
    )
    if not isinstance(policy, Mapping):
        raise LifecycleError("focused proof worker policy is absent")
    _verify_focused_proof_path_policy(
        policy,
        source_root=source_root,
        dependencies_root=dependencies_root,
        scratch=scratch,
        expected_probe_paths=expected_probe_paths,
        require_live_inventory=require_live_inventory,
    )
    nonce = str(launch.get("nonce"))
    environment = _focused_proof_environment(
        source_root,
        dependencies_root,
        scratch,
        source_manifest_sha256=expected_source_sha256,
    )
    environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
    child_pid = attestation.get("pid")
    if isinstance(child_pid, bool) or not isinstance(child_pid, int):
        raise LifecycleError("focused proof child PID is malformed")
    denied_paths = sorted(os.fspath(_lexical(path)) for path in expected_probe_paths)
    denied_records = attestation.get("denied_read_probes")
    launcher = source_root / "benchmarks/ssp2v_landlock.py"
    live_launcher = Path(sandbox.__file__).resolve(strict=True)
    if (
        attestation.get("command") != command
        or attestation.get("cwd") != os.fspath(source_root / "tests")
        or attestation.get("environment_keys") != sorted(environment)
        or attestation.get("environment_sha256_before_attestation_marker")
        != _sha256_bytes(_canonical_json(environment))
        or policy.get("command") != command
        or policy.get("cwd") != os.fspath(source_root / "tests")
        or policy.get("environment")
        != {
            key: value
            for key, value in environment.items()
            if key != "STRUCTSPLAT_PARENT_LAUNCH_NONCE"
        }
        or policy.get("timeout_seconds") != 1800.0
        or policy.get("launcher_path") != os.fspath(launcher)
        or policy.get("launcher_sha256") != _sha256_file(live_launcher)
        or attestation.get("launcher_sha256") != _sha256_file(live_launcher)
        or not isinstance(denied_records, list)
        or [item.get("path") for item in denied_records] != denied_paths
        or any(
            item.get("errno") not in (errno.EACCES, errno.EPERM)
            for item in denied_records
        )
        or any(
            path in set(attestation.get(name, []))
            for path in denied_paths
            for name in ("read_only", "read_write")
        )
    ):
        raise LifecycleError("focused proof launch differs from its exact frozen policy")


def _planned_write(
    journal: EventJournal, outdir: Path, path: Path, *, purpose: str, path_class: str = "artifact_runtime"
) -> None:
    journal.append(
        stage="preflight",
        operation="authorize-create-immutable",
        path_class=path_class,
        purpose=purpose,
        path=f"artifact:{path.relative_to(outdir).as_posix()}",
    )


def _verify_preflight_environment_unchanged(captured: Mapping[str, Any]) -> None:
    if environment_record() != captured:
        raise LifecycleError(
            "runtime/hardware/package environment changed during preflight"
        )


def preflight(
    outdir: Path,
    comp008: Path = DEFAULT_COMP008,
    comp009: Path = DEFAULT_COMP009,
    comp010: Path = DEFAULT_COMP010,
    *,
    run_focused_tests: bool = True,
) -> dict[str, Any]:
    """Run target-free phase-1 proof in a fresh, truly empty artifact directory."""
    outdir = _lexical(outdir)
    if outdir.exists():
        _reject_symlink_components(outdir, require_regular=False)
        if not outdir.is_dir() or any(outdir.iterdir()):
            raise LifecycleError("preflight requires a truly empty output directory")
    _mkdir_parents_no_symlink(outdir)
    _reject_symlink_components(outdir, require_regular=False)
    if not run_focused_tests:
        raise LifecycleError("production preflight cannot seal with focused tests skipped")
    quality.frozen_runtime_environment()
    started_utc = _utc_now()
    started = time.perf_counter_ns()
    journal = EventJournal(outdir / "journal/events")
    guard = AccessGuard(journal, artifact_root=outdir)

    before = source_manifest(guard=guard)
    repository = repository_provenance(before)
    source_manifest_path = outdir / "source_manifest.json"
    _planned_write(journal, outdir, source_manifest_path, purpose="persist source manifest")
    _exclusive_write(source_manifest_path, _canonical_json(before))
    archive_path = outdir / "source_snapshot.tar"
    _planned_write(journal, outdir, archive_path, purpose="persist safe deterministic source archive")
    write_source_archive(archive_path, before, guard=guard)
    source_archive_sha256 = _sha256_file(archive_path)
    runtime_source_root = outdir / _RUNTIME_SOURCE_RELATIVE
    _planned_write(
        journal,
        outdir,
        runtime_source_root / "SOURCE_MANIFEST.json",
        purpose="materialize immutable runtime source authority",
    )
    runtime_manifest = safe_extract_source_archive(
        archive_path,
        runtime_source_root,
        expected_manifest=before,
        guard=guard,
    )
    if runtime_manifest != before:
        raise LifecycleError("runtime source extraction differs from frozen manifest")
    runtime_source = _runtime_source_record(
        runtime_source_root,
        outdir=outdir,
        manifest=before,
        archive_sha256=source_archive_sha256,
    )
    upstream = bind_upstream_metadata(
        guard,
        _lexical(comp008),
        _lexical(comp009),
        _lexical(comp010),
        artifact_root=outdir,
    )
    upstream_authority_path = outdir / "upstream/authority_manifest.json"
    _planned_write(
        journal,
        outdir,
        upstream_authority_path,
        purpose="persist normalized upstream authority inventory",
    )
    _exclusive_write(upstream_authority_path, _canonical_json(upstream))
    compiler = compiler_record()
    environment = environment_record()
    denied_payload_probes = [
        _lexical(comp008)
        / _safe_relative_path(
            upstream["comp008"]["cells"][0]["sspl1"]["path"],
            "development stream probe path",
        ),
        _lexical(comp008)
        / _safe_relative_path(
            upstream["comp008"]["targets"][0]["relative_path"],
            "development target probe path",
        ),
    ]
    focused = (
        focused_proof_suite(
            denied_read_probes=denied_payload_probes,
            source_manifest_record=before,
            guard=guard,
        )
        if run_focused_tests
        else {"skipped_for_unit_test": True}
    )

    native_dir = outdir / "runtime/native"
    arithmetic_path = native_dir / "libssp2e_range_v1.so"
    lloyd_path = native_dir / "libssp2v_lloyd_v1.so"
    _planned_write(journal, outdir, arithmetic_path, purpose="build persisted arithmetic native")
    arithmetic.build_native(arithmetic_path, compiler=compiler["cxx"])
    os.chmod(arithmetic_path, 0o444)
    _planned_write(journal, outdir, lloyd_path, purpose="build persisted Lloyd native")
    lloyd.build_native(lloyd_path, compiler=compiler["cxx"])
    os.chmod(lloyd_path, 0o444)
    native_record = {
        "arithmetic": {
            **_library_record(arithmetic_path),
            "proof": arithmetic_native_proof(arithmetic_path),
        },
        "lloyd": _library_record(lloyd_path),
    }
    lloyd_proof, lloyd_sandbox, lloyd_launch = _run_json_worker(
        [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_lloyd-proof",
            "--native",
            os.fspath(lloyd_path),
        ],
        outdir=outdir,
        timeout=360,
        read_only=[lloyd_path],
        worker_seal_field="lloyd_proof_sha256",
    )
    validate_lloyd_proof(lloyd_proof)

    metrics = copy_metric_dependencies(outdir, guard)
    _verify_focused_metric_dependency_binding(focused, metrics)
    build_scratch = Path(
        tempfile.mkdtemp(prefix="comp011-renderer-build-", dir="/tmp")
    )
    build_environment = _worker_environment(outdir)
    build_environment["TORCH_EXTENSIONS_DIR"] = os.fspath(build_scratch / "extensions")
    build_environment["TMPDIR"] = os.fspath(build_scratch)
    build_record, build_sandbox, build_launch = _run_json_worker(
        [sys.executable, "-m", "benchmarks.ssp2v_actual_run", "_build-renderer"],
        outdir=outdir,
        timeout=1800,
        read_write=[build_scratch, *_sandbox_gpu_read_write()],
        worker_seal_field="renderer_build_worker_sha256",
        environment=build_environment,
    )
    validate_renderer_build_worker(build_record)
    source_renderer = Path(str(build_record["extension_path"]))
    guard.register(source_renderer, "runtime_dependency")
    renderer_payload = guard.read_bytes(
        source_renderer, stage="preflight", purpose="copy exact loaded renderer binary"
    )
    if (
        len(renderer_payload) != build_record["extension_bytes"]
        or _sha256_bytes(renderer_payload) != build_record["extension_sha256"]
    ):
        raise LifecycleError("loaded renderer changed before artifact copy")
    renderer_path = outdir / quality.RENDERER_RELATIVE_PATH
    _planned_write(journal, outdir, renderer_path, purpose="persist exact loaded renderer")
    _exclusive_write(renderer_path, renderer_payload)
    renderer_abi = renderer_abi_record(renderer_path, guard, outdir)
    renderer_record = {
        "relative_path": quality.RENDERER_RELATIVE_PATH,
        "bytes": len(renderer_payload),
        "sha256": _sha256_bytes(renderer_payload),
        "elf_identity": renderer_abi["elf_identity"],
        "abi": renderer_abi,
        "build_worker": build_record,
    }
    renderer_record_path = outdir / "runtime/renderer/record.json"
    _planned_write(journal, outdir, renderer_record_path, purpose="persist renderer identity record")
    _exclusive_write(renderer_record_path, _canonical_json(renderer_record))
    shutil.rmtree(build_scratch)
    with tempfile.TemporaryDirectory(
        prefix="comp011-renderer-proof-", dir="/tmp"
    ) as proof_temporary:
        proof_environment = _worker_environment(outdir)
        proof_environment["TMPDIR"] = proof_temporary
        renderer_proof, renderer_sandbox, renderer_launch = _run_json_worker(
            [
                sys.executable,
                "-m",
                "benchmarks.ssp2v_actual_run",
                "_renderer-proof",
                "--artifact-root",
                os.fspath(outdir),
                "--renderer-record",
                os.fspath(renderer_record_path),
            ],
            outdir=outdir,
            timeout=1800,
            read_only=[outdir / "runtime"],
            read_write=[Path(proof_temporary), *_sandbox_gpu_read_write()],
            worker_seal_field="renderer_proof_worker_sha256",
            environment=proof_environment,
        )
    validate_renderer_proof_worker(renderer_proof, renderer_record)
    if renderer_proof.get("fixture") != build_record.get("fixture"):
        raise LifecycleError("independent copied-renderer fixture reconstruction drift")
    _verify_preflight_environment_unchanged(environment)

    after = source_manifest(guard=guard)
    if after != before:
        raise LifecycleError("source closure changed during preflight")
    if repository_provenance(after) != repository:
        raise LifecycleError("repository checkout changed during preflight")
    if _read_canonical_object(source_manifest_path) != after:
        raise LifecycleError("persisted early source manifest drift")
    if _sha256_file(archive_path) != source_archive_sha256:
        raise LifecycleError("persisted early source archive drift")
    if (
        _runtime_source_record(
            runtime_source_root,
            outdir=outdir,
            manifest=after,
            archive_sha256=source_archive_sha256,
        )
        != runtime_source
    ):
        raise LifecycleError("runtime source authority changed during preflight")
    with tempfile.TemporaryDirectory(prefix="comp011-source-proof-") as temporary:
        extracted = safe_extract_source_archive(
            archive_path,
            Path(temporary) / "source",
            expected_manifest=after,
            guard=guard,
        )
    if extracted != after:
        raise LifecycleError("source archive relocation proof drift")

    preflight_path = outdir / "preflight.json"
    _planned_write(journal, outdir, preflight_path, purpose="seal phase-1 preflight record")
    record = {
        "schema": PREFLIGHT_SCHEMA,
        "protocol": PROTOCOL,
        "stage": "preflight",
        "task_sha256": TASK_SHA256,
        "started_utc": started_utc,
        "ended_utc": _utc_now(),
        "wall_ns": time.perf_counter_ns() - started,
        "source_manifest_sha256": after["source_manifest_sha256"],
        "source_archive_sha256": source_archive_sha256,
        "runtime_source": runtime_source,
        "repository": repository,
        "journal": {"length": journal.length, "head_sha256": journal.head},
        "upstream_authority": {
            "relative_path": upstream_authority_path.relative_to(outdir).as_posix(),
            "bytes": upstream_authority_path.stat().st_size,
            "sha256": _sha256_file(upstream_authority_path),
            "authority_manifest_sha256": upstream["authority_manifest_sha256"],
        },
        "operational_upstream_roots": {
            "comp008": os.fspath(_lexical(comp008)),
            "comp009": os.fspath(_lexical(comp009)),
            "comp010": os.fspath(_lexical(comp010)),
        },
        "compiler": compiler,
        "environment": environment,
        "focused_tests": focused,
        "native": native_record,
        "lloyd_complexity": lloyd_proof,
        "metric_dependencies": metrics,
        "renderer": renderer_record,
        "renderer_proof": renderer_proof,
        "sandboxed_workers": {
            "lloyd_proof": {"attestation": lloyd_sandbox, "launch": lloyd_launch},
            "renderer_build": {"attestation": build_sandbox, "launch": build_launch},
            "renderer_proof": {
                "attestation": renderer_sandbox,
                "launch": renderer_launch,
            },
        },
        "development_streams_opened": False,
        "target_or_pixel_payloads_opened": False,
        "confirmation_payloads_accessed": False,
        "proof_status": "passed",
    }
    sealed = actual.seal_record(record, "preflight_binding_sha256")
    preflight_payload = _canonical_json(sealed)
    _exclusive_write(preflight_path, preflight_payload)
    journal.append(
        stage="preflight",
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose="completed phase-1 preflight record",
        path="artifact:preflight.json",
        artifact_sha256=_sha256_bytes(preflight_payload),
        artifact_bytes=len(preflight_payload),
    )
    return sealed


def _read_canonical_object(path: Path) -> dict[str, Any]:
    # Captured replay is a read-only verifier of an already sealed artifact.
    # Its phase-A policy deliberately grants named top-level manifests without
    # granting directory enumeration of the artifact root, both to minimize
    # authority and to keep target paths hidden.  Recovery is a writer/resume
    # operation and is therefore neither possible nor appropriate there.
    if _captured_replay_source_root() is None:
        _recover_pending_directory(_lexical(path).parent)
    payload = _read_regular_no_follow(path)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LifecycleError(f"artifact JSON is invalid: {path.name}") from exc
    if not isinstance(value, dict) or payload != _canonical_json(value):
        raise LifecycleError(f"artifact JSON is not canonical: {path.name}")
    return value


def _verify_regular_record(path: Path, record: Mapping[str, Any], *, readonly: bool = True) -> None:
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise LifecycleError(f"sealed artifact byte count is malformed: {path.name}")
    _require_sha256(expected_sha256, f"sealed artifact SHA-256 for {path.name}")
    payload = _read_regular_no_follow(path)
    if len(payload) != expected_bytes or _sha256_bytes(payload) != expected_sha256:
        raise LifecycleError(f"sealed artifact file drift: {path.name}")
    if readonly and os.stat(path, follow_symlinks=False).st_mode & 0o222:
        raise LifecycleError(f"sealed artifact became writable: {path.name}")


def _verify_upstream_authority_copy(outdir: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    relative = reference.get("relative_path")
    path = _artifact_path(outdir, relative, "upstream authority path")
    _verify_regular_record(path, reference)
    authority = _read_canonical_object(path)
    actual.verify_record(authority, "authority_manifest_sha256")
    if authority.get("authority_manifest_sha256") != reference.get("authority_manifest_sha256"):
        raise LifecycleError("upstream authority manifest seal drift")
    comp008 = authority.get("comp008")
    comp009 = authority.get("comp009")
    comp010 = authority.get("comp010")
    if (
        not isinstance(comp008, Mapping)
        or not isinstance(comp009, Mapping)
        or not isinstance(comp010, Mapping)
        or len(comp008.get("cells", [])) != 16
        or len(comp008.get("targets", [])) != 8
        or len(comp009.get("input_streams", [])) != 16
        or len(comp009.get("cells", [])) != 16
        or comp010.get("captured_child_count") != 2
        or len(comp010.get("captured_children", [])) != 2
        or authority.get("development_streams_opened") is not False
        or authority.get("target_or_pixel_payloads_opened") is not False
        or authority.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("upstream normalized authority inventory drift")
    for name, expected in (
        ("comp008", EXPECTED_COMP008),
        ("comp009", EXPECTED_COMP009),
        ("comp010", EXPECTED_COMP010),
    ):
        value = authority[name]
        if any(value.get(field) != digest for field, digest in expected.items()):
            raise LifecycleError(f"sealed {name} authority digest drift")
    children = comp010["captured_children"]
    for child in children:
        actual.verify_record(child, "repair_child_sha256")
        worker = child.get("captured_worker")
        if not isinstance(worker, Mapping):
            raise LifecycleError("sealed COMP-010 child worker is absent")
        actual.verify_record(worker, "captured_replay_worker_sha256")
    source = comp009.get("captured_source_archive")
    manifest = comp009.get("captured_source_manifest")
    if not isinstance(source, Mapping) or not isinstance(manifest, Mapping):
        raise LifecycleError("captured COMP-009 source authority is absent")
    source_path = _artifact_path(outdir, source.get("relative_path"), "authority-copy path")
    _verify_regular_record(source_path, source)
    with tempfile.TemporaryDirectory(prefix="comp011-verify-upstream-source-") as temporary:
        extracted = safe_extract_source_archive(
            source_path, Path(temporary) / "source", expected_manifest=manifest
        )
    if extracted != dict(manifest):
        raise LifecycleError("captured COMP-009 source verification drift")
    return authority


def _verify_metric_dependencies(outdir: Path, record: Mapping[str, Any]) -> None:
    lpips_root = _artifact_path(outdir, record.get("lpips_relative_root"), "LPIPS root")
    expected_files = record.get("lpips_files")
    if not isinstance(expected_files, list):
        raise LifecycleError("artifact-local LPIPS file manifest is absent")
    expected_paths = {str(item.get("path")) for item in expected_files}
    observed_paths = {
        path.relative_to(lpips_root).as_posix()
        for path in lpips_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if expected_paths != observed_paths:
        raise LifecycleError("artifact-local LPIPS inventory drift")
    for item in expected_files:
        relative = Path(str(item.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise LifecycleError("unsafe LPIPS manifest path")
        _verify_regular_record(lpips_root / relative, item)
    alexnet = _artifact_path(outdir, record.get("alexnet_relative_path"), "AlexNet path")
    _verify_regular_record(
        alexnet,
        {"bytes": record.get("alexnet_bytes"), "sha256": record.get("alexnet_sha256")},
    )
    if record.get("network_download_permitted") is not False:
        raise LifecycleError("metric dependency record permits a network download")
    provenance = record.get("module_provenance")
    if not isinstance(provenance, list) or len(provenance) != 5:
        raise LifecycleError("metric module provenance is incomplete")
    for item in provenance:
        module = str(item.get("module"))
        distribution = str(item.get("distribution"))
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise LifecycleError(f"metric module disappeared: {module}")
        payload = _read_regular_no_follow(Path(spec.origin))
        if (
            importlib.metadata.version(distribution) != item.get("version")
            or Path(spec.origin).name != item.get("origin_name")
            or len(payload) != item.get("bytes")
            or _sha256_bytes(payload) != item.get("sha256")
        ):
            raise LifecycleError(f"metric module provenance drift: {module}")
    ms_ssim = record.get("ms_ssim_call")
    if (
        not isinstance(ms_ssim, Mapping)
        or ms_ssim.get("weights") != [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        or ms_ssim.get("weights_sha256")
        != _sha256_bytes(_canonical_json(ms_ssim.get("weights")))
    ):
        raise LifecycleError("MS-SSIM call/weight provenance drift")


def _verify_renderer_abi(record: Mapping[str, Any]) -> None:
    abi = record.get("abi")
    if not isinstance(abi, Mapping) or abi.get("elf_identity") != record.get("elf_identity"):
        raise LifecycleError("renderer ABI record is absent or inconsistent")
    dependencies = abi.get("resolved_libraries")
    if not isinstance(dependencies, list) or not dependencies:
        raise LifecycleError("renderer resolved-library inventory is empty")
    for dependency in dependencies:
        _verify_regular_record(Path(str(dependency.get("path"))), dependency, readonly=False)
    preload = abi.get("required_preload")
    if not isinstance(preload, Mapping):
        raise LifecycleError("renderer preload ABI record is absent")
    _verify_regular_record(Path(str(preload.get("path"))), preload, readonly=False)
    current_version = _command_record(
        ("readelf", "--version-info", str(preload.get("path")))
    )
    if current_version != preload.get("version_info"):
        raise LifecycleError("system libstdc++ version information drift")
    if abi.get("loader_environment") != {
        "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    }:
        raise LifecycleError("renderer loader environment drift")


def verify_journal_prefix(
    events: Sequence[Mapping[str, Any]], binding: Mapping[str, Any]
) -> None:
    length = binding.get("length")
    head = binding.get("head_sha256")
    if (
        isinstance(length, bool)
        or not isinstance(length, int)
        or length <= 0
        or length > len(events)
        or events[length - 1].get("event_sha256") != head
    ):
        raise LifecycleError("sealed journal prefix binding is invalid")


def _verify_stage_journal_segment(
    events: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    *,
    stage: str,
    manifest_path: str,
    manifest_payload: bytes,
    prior_stage_seal: str | None,
    predecessor_binding: Mapping[str, Any] | None = None,
) -> None:
    """Require a legal stage-only suffix and an immediate manifest completion receipt."""

    verify_journal_prefix(events, binding)
    length = int(binding["length"])
    start = 0
    if predecessor_binding is not None:
        verify_journal_prefix(events, predecessor_binding)
        start = int(predecessor_binding["length"]) + 1
    if start >= length or len(events) <= length:
        raise LifecycleError(f"{stage} journal segment/completion receipt is absent")
    for event in events[start:length]:
        if (
            event.get("stage") != stage
            or event.get("prior_stage_seal") != prior_stage_seal
        ):
            raise LifecycleError(f"{stage} journal suffix contains a foreign predecessor/event")
    authorization = events[length - 1]
    expected_path = f"artifact:{manifest_path}"
    if (
        authorization.get("operation") != "authorize-create-immutable"
        or authorization.get("path_class") != "artifact_runtime"
        or authorization.get("path") != expected_path
    ):
        raise LifecycleError(f"{stage} manifest authorization is not the sealed-prefix tail")
    completion = events[length]
    if (
        completion.get("stage") != stage
        or completion.get("operation") != "create-immutable-complete"
        or completion.get("path_class") != "artifact_runtime"
        or completion.get("path") != expected_path
        or completion.get("prior_stage_seal") != prior_stage_seal
        or completion.get("artifact_bytes") != len(manifest_payload)
        or completion.get("artifact_sha256") != _sha256_bytes(manifest_payload)
    ):
        raise LifecycleError(f"{stage} immediate manifest completion receipt is invalid")


def _verify_preflight_base(outdir: Path) -> dict[str, Any]:
    """Verify only the sealed preflight manifest and its journal completion.

    Later lifecycle stages use this form so checking their predecessor does not
    silently reopen repository, upstream, renderer, or scientific payloads.
    """

    outdir = _lexical(outdir)
    try:
        record = _read_canonical_object(outdir / "preflight.json")
    except (OSError, LifecycleError) as exc:
        raise LifecycleError("cannot read sealed COMP-011 preflight") from exc
    actual.verify_record(record, "preflight_binding_sha256")
    events = EventJournal.verify(outdir / "journal/events")
    journal_binding = record.get("journal")
    if not isinstance(journal_binding, Mapping):
        raise LifecycleError("preflight journal prefix binding is absent")
    _verify_stage_journal_segment(
        events,
        journal_binding,
        stage="preflight",
        manifest_path="preflight.json",
        manifest_payload=_canonical_json(record),
        prior_stage_seal=None,
    )
    if (
        record.get("schema") != PREFLIGHT_SCHEMA
        or record.get("protocol") != PROTOCOL
        or record.get("stage") != "preflight"
        or record.get("task_sha256") != TASK_SHA256
        or record.get("proof_status") != "passed"
        or record.get("development_streams_opened") is not False
        or record.get("target_or_pixel_payloads_opened") is not False
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("sealed COMP-011 preflight invariant failed")
    return record


def verify_preflight(outdir: Path, *, captured_replay: bool = False) -> dict[str, Any]:
    outdir = _lexical(outdir)
    record = _verify_preflight_base(outdir)
    manifest = source_manifest()
    if manifest["source_manifest_sha256"] != record.get("source_manifest_sha256"):
        raise LifecycleError("COMP-011 source drift after preflight")
    persisted_manifest = _read_canonical_object(outdir / "source_manifest.json")
    actual.verify_record(persisted_manifest, "source_manifest_sha256")
    if persisted_manifest != manifest:
        raise LifecycleError("persisted source manifest drift")
    if not captured_replay and record.get("repository") != repository_provenance(manifest):
        raise LifecycleError("bound repository checkout provenance drift")
    archive = outdir / "source_snapshot.tar"
    if _sha256_file(archive) != record.get("source_archive_sha256"):
        raise LifecycleError("source archive drift")
    with tempfile.TemporaryDirectory(prefix="comp011-verify-source-") as temporary:
        extracted = safe_extract_source_archive(
            archive, Path(temporary) / "source", expected_manifest=manifest
        )
    if extracted != manifest:
        raise LifecycleError("source archive relocation verification drift")
    runtime_source = record.get("runtime_source")
    if not isinstance(runtime_source, Mapping):
        raise LifecycleError("runtime source authority is absent")
    _verify_runtime_source_record(
        outdir,
        runtime_source,
        expected_manifest=manifest,
        expected_archive_sha256=str(record["source_archive_sha256"]),
    )

    upstream_reference = record.get("upstream_authority")
    if not isinstance(upstream_reference, Mapping):
        raise LifecycleError("upstream authority reference is absent")
    authority_copy = _verify_upstream_authority_copy(outdir, upstream_reference)
    if compiler_record() != record.get("compiler"):
        raise LifecycleError("compiler/source provenance drift")
    expected_environment = record.get("environment")
    if not isinstance(expected_environment, Mapping):
        raise LifecycleError("bound runtime/hardware/package environment is absent")
    if captured_replay:
        if _captured_replay_source_root() is None:
            raise LifecycleError("captured replay environment check lacks its source marker")
        if os.environ.get(_REPLAY_PREFLIGHT_BINDING_ENV) != record.get(
            "preflight_binding_sha256"
        ):
            raise LifecycleError(
                "captured replay environment check lacks its outer preflight authority"
            )
        bound_lscpu = expected_environment.get("lscpu")
        if not isinstance(bound_lscpu, Mapping):
            raise LifecycleError("captured replay lscpu binding is absent")
        bound_nvidia_smi = expected_environment.get("nvidia_smi")
        if not isinstance(bound_nvidia_smi, Mapping):
            raise LifecycleError(
                "captured replay nvidia-smi binding is absent"
            )
        observed_environment = environment_record(
            bound_lscpu=bound_lscpu,
            bound_nvidia_smi=bound_nvidia_smi,
        )
    else:
        observed_environment = environment_record()
    if observed_environment != expected_environment:
        raise LifecycleError("bound runtime/hardware/package environment drift")
    focused = record.get("focused_tests")
    try:
        focused_stdout = bytes.fromhex(str(focused.get("stdout_hex", ""))) if isinstance(focused, Mapping) else b""
        focused_stderr = bytes.fromhex(str(focused.get("stderr_hex", ""))) if isinstance(focused, Mapping) else b""
    except ValueError as exc:
        raise LifecycleError("focused preflight stdout/stderr evidence is malformed") from exc
    if (
        not isinstance(focused, Mapping)
        or focused.get("returncode") != 0
        or focused.get("skipped_for_unit_test") is not None
        or focused.get("stdout_bytes")
        != len(focused_stdout)
        or focused.get("stdout_sha256")
        != _sha256_bytes(focused_stdout)
        or focused.get("stderr_bytes")
        != len(focused_stderr)
        or focused.get("stderr_sha256")
        != _sha256_bytes(focused_stderr)
        or any(key not in focused for key in _FOCUSED_BASE_FIELDS)
    ):
        raise LifecycleError("focused preflight tests were absent, skipped, or failed")
    focused_base = {key: focused[key] for key in _FOCUSED_BASE_FIELDS}
    _verify_launch_bundle(
        {
            "attestation": focused.get("sandbox_attestation"),
            "launch": focused.get("launch"),
        },
        focused_base,
        "canonical_worker_sha256",
    )
    roots_for_probe = record.get("operational_upstream_roots")
    if not isinstance(roots_for_probe, Mapping):
        raise LifecycleError("focused preflight payload-probe roots are absent")
    expected_probe_paths = [
        _lexical(Path(str(roots_for_probe["comp008"])))
        / _safe_relative_path(
            authority_copy["comp008"]["cells"][0]["sspl1"]["path"],
            "development stream probe path",
        ),
        _lexical(Path(str(roots_for_probe["comp008"])))
        / _safe_relative_path(
            authority_copy["comp008"]["targets"][0]["relative_path"],
            "development target probe path",
        ),
    ]
    _verify_focused_proof_policy(
        focused,
        expected_probe_paths=expected_probe_paths,
        expected_source_manifest_sha256=str(record["source_manifest_sha256"]),
        expected_source_file_count=len(manifest["files"]),
        forbidden_roots=[
            ROOT,
            outdir,
            *(Path(str(value)) for value in roots_for_probe.values()),
            *expected_probe_paths,
        ],
        require_live_inventory=True,
    )
    focused_attestation = focused["sandbox_attestation"]
    denied_probe_records = focused_attestation.get("denied_read_probes")
    if (
        not isinstance(denied_probe_records, list)
        or [record.get("path") for record in denied_probe_records]
        != [os.fspath(path) for path in sorted(expected_probe_paths)]
        or any(record.get("errno") not in (errno.EACCES, errno.EPERM) for record in denied_probe_records)
    ):
        raise LifecycleError("focused preflight denied-payload probes differ from frozen paths")

    natives = record.get("native")
    if not isinstance(natives, Mapping):
        raise LifecycleError("persisted native records are absent")
    arithmetic_record = natives.get("arithmetic")
    lloyd_record = natives.get("lloyd")
    if not isinstance(arithmetic_record, Mapping) or not isinstance(lloyd_record, Mapping):
        raise LifecycleError("persisted native record is incomplete")
    arithmetic_path = outdir / "runtime/native" / str(arithmetic_record.get("path"))
    lloyd_path = outdir / "runtime/native" / str(lloyd_record.get("path"))
    _verify_regular_record(arithmetic_path, arithmetic_record)
    _verify_regular_record(lloyd_path, lloyd_record)
    if stat.S_IMODE(arithmetic_path.stat().st_mode) != arithmetic_record.get("mode") or stat.S_IMODE(
        lloyd_path.stat().st_mode
    ) != lloyd_record.get("mode"):
        raise LifecycleError("persisted native mode drift")
    if arithmetic_native_proof(arithmetic_path) != arithmetic_record.get("proof"):
        raise LifecycleError("persisted arithmetic native proof drift")
    lloyd_proof = record.get("lloyd_complexity")
    if not isinstance(lloyd_proof, Mapping):
        raise LifecycleError("Lloyd complexity proof is absent")
    validate_lloyd_proof(lloyd_proof)

    metric_dependencies = record.get("metric_dependencies")
    if not isinstance(metric_dependencies, Mapping):
        raise LifecycleError("metric dependency record is absent")
    _verify_metric_dependencies(outdir, metric_dependencies)
    _verify_focused_metric_dependency_binding(focused, metric_dependencies)
    renderer = record.get("renderer")
    if not isinstance(renderer, dict):
        raise LifecycleError("renderer record is absent")
    quality.verify_persisted_binary(
        outdir,
        str(renderer["relative_path"]),
        str(renderer["sha256"]),
        int(renderer["bytes"]),
        renderer["elf_identity"],
    )
    _verify_renderer_abi(renderer)
    build_worker = renderer.get("build_worker")
    renderer_proof = record.get("renderer_proof")
    if not isinstance(build_worker, Mapping) or not isinstance(renderer_proof, Mapping):
        raise LifecycleError("renderer build/load proof is incomplete")
    validate_renderer_build_worker(build_worker)
    validate_renderer_proof_worker(renderer_proof, renderer)
    if renderer_proof.get("fixture") != build_worker.get("fixture"):
        raise LifecycleError("renderer fixture proof drift")
    sandboxed_workers = record.get("sandboxed_workers")
    if not isinstance(sandboxed_workers, Mapping) or set(sandboxed_workers) != {
        "lloyd_proof",
        "renderer_build",
        "renderer_proof",
    }:
        raise LifecycleError("preflight sandboxed-worker inventory is absent")
    _verify_launch_bundle(
        sandboxed_workers["lloyd_proof"], lloyd_proof, "lloyd_proof_sha256"
    )
    _verify_launch_bundle(
        sandboxed_workers["renderer_build"], build_worker, "renderer_build_worker_sha256"
    )
    _verify_launch_bundle(
        sandboxed_workers["renderer_proof"], renderer_proof, "renderer_proof_worker_sha256"
    )
    forbidden_worker_scratch_roots = [
        ROOT,
        outdir,
        *(Path(str(value)) for value in roots_for_probe.values()),
        *expected_probe_paths,
    ]
    for name, worker in (
        ("lloyd_proof", lloyd_proof),
        ("renderer_build", build_worker),
        ("renderer_proof", renderer_proof),
    ):
        _verify_exact_preflight_worker_policy(
            name,
            sandboxed_workers[name],
            worker,
            outdir=outdir,
            runtime_source=runtime_source,
            manifest=manifest,
            archive_sha256=str(record["source_archive_sha256"]),
            lloyd_path=lloyd_path,
            renderer_record_path=outdir / "runtime/renderer/record.json",
            forbidden_roots=forbidden_worker_scratch_roots,
        )
    roots = record.get("operational_upstream_roots")
    if not isinstance(roots, Mapping) or set(roots) != {"comp008", "comp009", "comp010"}:
        raise LifecycleError("preflight operational upstream roots are absent")
    for name, value in roots.items():
        if not isinstance(value, str) or _lexical(Path(value)) != Path(value):
            raise LifecycleError(f"preflight operational {name} root is not lexical absolute")
    return record


def _tuple_label(bits: Sequence[int]) -> str:
    normalized = tuple(int(value) for value in bits)
    if normalized not in actual.FROZEN_TUPLES:
        raise LifecycleError(f"unexpected COMP-011 tuple {normalized}")
    return f"m{normalized[0]}_s{normalized[1]}_r{normalized[2]}_c{normalized[3]}"


def _expected_identities() -> list[tuple[str, tuple[int, int, int, int]]]:
    return [
        (image_id, tuple(bits))
        for image_id in actual.FROZEN_IMAGES
        for bits in actual.FROZEN_TUPLES
    ]


def _authority_copy(outdir: Path, preflight_record: Mapping[str, Any]) -> dict[str, Any]:
    reference = preflight_record.get("upstream_authority")
    if not isinstance(reference, Mapping):
        raise LifecycleError("preflight upstream-authority reference is absent")
    return _verify_upstream_authority_copy(outdir, reference)


def _artifact_label(outdir: Path, path: Path) -> str:
    try:
        relative = _lexical(path).relative_to(_lexical(outdir)).as_posix()
    except ValueError as exc:
        raise LifecycleError("artifact path escapes its output root") from exc
    return f"artifact:{relative}"


def _journaled_immutable(
    journal: EventJournal,
    outdir: Path,
    path: Path,
    payload: bytes,
    *,
    stage: str,
    path_class: str,
    purpose: str,
    prior_stage_seal: str,
) -> dict[str, Any]:
    """Create one immutable artifact, or verify an identical interrupted-stage product."""

    path = _lexical(path)
    _recover_pending_directory(path.parent)
    if path.exists():
        observed = _read_regular_no_follow(path)
        if observed != payload or os.stat(path, follow_symlinks=False).st_mode & 0o222:
            raise LifecycleError(f"resumed immutable artifact drift: {path.name}")
        operation = "resume-verify-immutable-complete"
    else:
        _exclusive_write(path, payload)
        operation = "create-immutable-complete"
    journal.append(
        stage=stage,
        operation=operation,
        path_class=path_class,
        purpose=purpose,
        path=_artifact_label(outdir, path),
        prior_stage_seal=prior_stage_seal,
        artifact_sha256=_sha256_bytes(payload),
        artifact_bytes=len(payload),
    )
    return {
        "relative_path": path.relative_to(_lexical(outdir)).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _verify_stage_cell_receipts(
    outdir: Path,
    events: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    prior_stage_seal: str,
    expected_paths: Sequence[Path],
    purpose: str,
) -> dict[str, dict[str, Any]]:
    """Verify one stochastic stage's durable cell receipts and frozen prefix."""

    outdir = _lexical(outdir)
    paths = [_lexical(path) for path in expected_paths]
    labels = [_artifact_label(outdir, path) for path in paths]
    if len(labels) != len(set(labels)):
        raise LifecycleError(f"{stage} frozen cell paths are not unique")
    expected_by_label = dict(zip(labels, paths, strict=True))
    receipts: dict[str, dict[str, Any]] = {}
    receipt_order: list[str] = []
    for event in events:
        if event.get("stage") != stage:
            continue
        label = event.get("path")
        targets_cell = isinstance(label, str) and label in expected_by_label
        claims_cell_receipt = event.get("purpose") == purpose
        if not targets_cell and not claims_cell_receipt:
            continue
        if (
            event.get("operation")
            not in {"create-immutable-complete", "resume-verify-immutable-complete"}
            or event.get("path_class") != "artifact_runtime"
            or event.get("purpose") != purpose
            or event.get("prior_stage_seal") != prior_stage_seal
            or not isinstance(label, str)
            or label not in expected_by_label
        ):
            raise LifecycleError(f"{stage} cell receipt metadata is invalid")
        if label in receipts:
            raise LifecycleError(f"{stage} cell has duplicate journal receipts")
        path = expected_by_label[label]
        _recover_pending_directory(path.parent)
        if not os.path.lexists(path):
            raise LifecycleError(f"{stage} committed cell is missing")
        payload = _read_regular_no_follow(path)
        record = _read_canonical_object(path)
        if (
            event.get("artifact_sha256") != _sha256_bytes(payload)
            or event.get("artifact_bytes") != len(payload)
        ):
            raise LifecycleError(f"{stage} cell receipt hash/bytes differ from artifact")
        receipts[label] = {"event": event, "payload": payload, "record": record}
        receipt_order.append(label)
    if receipt_order != labels[: len(receipt_order)]:
        raise LifecycleError(f"{stage} cell receipts are not a frozen-order prefix")

    for path in paths:
        _recover_pending_directory(path.parent)
    observed = [
        label
        for label, path in zip(labels, paths, strict=True)
        if os.path.lexists(path)
    ]
    if observed != labels[: len(observed)]:
        raise LifecycleError(f"{stage} cell artifacts are not a frozen-order prefix")
    if len(observed) > len(receipts) + 1:
        raise LifecycleError(f"{stage} contains an uncommitted cell suffix")
    for label in observed[len(receipts) :]:
        _read_canonical_object(expected_by_label[label])
    return receipts


def _verify_stage_base(
    outdir: Path,
    *,
    stage: str,
    schema: str,
    prior_record: Mapping[str, Any],
) -> dict[str, Any]:
    path = outdir / _STAGE_MANIFEST_PATHS[stage]
    record = _read_canonical_object(path)
    seal_field = _STAGE_SEAL_FIELDS[stage]
    actual.verify_record(record, seal_field)
    previous = STAGES[STAGES.index(stage) - 1]
    if (
        record.get("schema") != schema
        or record.get("protocol") != PROTOCOL
        or record.get("task_sha256") != TASK_SHA256
        or record.get("stage") != stage
        or record.get("prior_stage") != previous
        or record.get("prior_stage_seal") != prior_record.get(_STAGE_SEAL_FIELDS[previous])
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError(f"{stage} stage provenance/schema drift")
    events = EventJournal.verify(outdir / "journal/events")
    binding = record.get("journal")
    if not isinstance(binding, Mapping):
        raise LifecycleError(f"{stage} journal-prefix binding is absent")
    predecessor_binding = prior_record.get("journal")
    if not isinstance(predecessor_binding, Mapping):
        raise LifecycleError(f"{stage} predecessor journal-prefix binding is absent")
    _verify_stage_journal_segment(
        events,
        binding,
        stage=stage,
        manifest_path=_STAGE_MANIFEST_PATHS[stage],
        manifest_payload=_canonical_json(record),
        prior_stage_seal=str(prior_record[_STAGE_SEAL_FIELDS[previous]]),
        predecessor_binding=predecessor_binding,
    )
    if stage == "replay" and len(events) != int(binding["length"]) + 1:
        raise LifecycleError("replay terminal manifest has a journal suffix")
    return record


def _finalize_stage(
    outdir: Path,
    journal: EventJournal,
    *,
    stage: str,
    schema: str,
    prior_stage_seal: str,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    path = outdir / _STAGE_MANIFEST_PATHS[stage]
    if path.exists():
        raise LifecycleError(f"refusing to replace existing {stage} manifest")
    journal.append(
        stage=stage,
        operation="authorize-create-immutable",
        path_class="artifact_runtime",
        purpose=f"authorize final {stage} manifest",
        path=_artifact_label(outdir, path),
        prior_stage_seal=prior_stage_seal,
    )
    record = {
        "schema": schema,
        "protocol": PROTOCOL,
        "task_sha256": TASK_SHA256,
        "stage": stage,
        "prior_stage": STAGES[STAGES.index(stage) - 1],
        "prior_stage_seal": prior_stage_seal,
        **dict(body),
        "journal": {"length": journal.length, "head_sha256": journal.head},
        "confirmation_payloads_accessed": False,
    }
    seal_field = _STAGE_SEAL_FIELDS[stage]
    sealed = actual.seal_record(record, seal_field)
    payload = _canonical_json(sealed)
    _exclusive_write(path, payload)
    journal.append(
        stage=stage,
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose=f"completed final {stage} manifest",
        path=_artifact_label(outdir, path),
        prior_stage_seal=prior_stage_seal,
        artifact_sha256=_sha256_bytes(payload),
        artifact_bytes=len(payload),
    )
    return sealed


def _recover_interrupted_stage_completion(outdir: Path, stage: str) -> None:
    """Append only a missing receipt for an already exact immutable manifest."""

    if stage == "preflight":
        return
    outdir = _lexical(outdir)
    path = outdir / _STAGE_MANIFEST_PATHS[stage]
    _recover_pending_directory(path.parent)
    if not path.exists():
        return
    record = _read_canonical_object(path)
    actual.verify_record(record, _STAGE_SEAL_FIELDS[stage])
    if (
        record.get("schema") != _STAGE_SCHEMAS[stage]
        or record.get("stage") != stage
        or record.get("protocol") != PROTOCOL
        or record.get("task_sha256") != TASK_SHA256
    ):
        raise LifecycleError("interrupted stage manifest provenance drift")
    binding = record.get("journal")
    if not isinstance(binding, Mapping):
        raise LifecycleError("interrupted stage manifest journal binding is absent")
    events = EventJournal.verify(outdir / "journal/events")
    verify_journal_prefix(events, binding)
    length = int(binding["length"])
    if len(events) > length:
        return
    if len(events) != length or length <= 0:
        raise LifecycleError("interrupted stage journal position is not recoverable")
    authorization = events[-1]
    prior_seal = record.get("prior_stage_seal")
    manifest_label = f"artifact:{_STAGE_MANIFEST_PATHS[stage]}"
    if (
        authorization.get("stage") != stage
        or authorization.get("operation") != "authorize-create-immutable"
        or authorization.get("path_class") != "artifact_runtime"
        or authorization.get("path") != manifest_label
        or authorization.get("prior_stage_seal") != prior_seal
        or os.stat(path, follow_symlinks=False).st_mode & 0o222
    ):
        raise LifecycleError("interrupted stage lacks its exact final authorization")
    payload = _read_regular_no_follow(path)
    journal = EventJournal(outdir / "journal/events", resume=True)
    previous = STAGES[STAGES.index(stage) - 1]
    journal.bind_stage_seal(previous, str(prior_seal))
    journal.append(
        stage=stage,
        operation="create-immutable-complete",
        path_class="artifact_runtime",
        purpose=f"recovered completed final {stage} manifest",
        path=manifest_label,
        prior_stage_seal=str(prior_seal),
        artifact_sha256=_sha256_bytes(payload),
        artifact_bytes=len(payload),
    )


def _verified_stage_records(outdir: Path, stage: str) -> dict[str, dict[str, Any]]:
    """Verify the sealed manifest chain without reopening prior-stage payloads."""

    records: dict[str, dict[str, Any]] = {"preflight": verify_preflight(outdir)}
    for name in STAGES[1 : STAGES.index(stage)]:
        previous = STAGES[STAGES.index(name) - 1]
        records[name] = _verify_stage_base(
            outdir,
            stage=name,
            schema=_STAGE_SCHEMAS[name],
            prior_record=records[previous],
        )
    return records


def _stage_guard(
    outdir: Path, stage: str
) -> tuple[EventJournal, AccessGuard, dict[str, dict[str, Any]]]:
    outdir = _lexical(outdir)
    records = _verified_stage_records(outdir, stage)
    journal = EventJournal(outdir / "journal/events", resume=True)
    guard = AccessGuard(journal, artifact_root=outdir)
    for name, record in records.items():
        digest = str(record[_STAGE_SEAL_FIELDS[name]])
        journal.bind_stage_seal(name, digest)
        guard.bind_stage_seal(name, digest)
    candidate = records.get("fit-encode-cold-decode-dev")
    if candidate is not None and candidate.get("state") == "complete_candidate_set":
        guard.bind_candidate_set_seal(str(candidate["candidate_set_sha256"]))
    target = records.get("import-targets")
    if target is not None:
        guard.bind_target_copy_seal(str(target["target_copy_sha256"]))
    return journal, guard, records


def _parse_imported_sspl1(blob: bytes, authority: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parts = container.extract_sspl1_parts(blob)
        arrays = execute.boundary_arrays(parts.header, parts.symbols)
    except Exception as exc:
        raise LifecycleError(f"strict SSPL1 import failed: {exc}") from exc
    if (
        parts.header.n != container.ROW_COUNT
        or list(parts.header.bits) != authority.get("bit_tuple")
        or parts.blob_sha256 != authority.get("sspl1", {}).get("sha256")
        or parts.symbol_sha256 != authority.get("absolute_symbols", {}).get("symbol_sha256")
        or execute._array_state_sha256(arrays)  # noqa: SLF001 - shared frozen digest contract.
        != authority.get("decoded_boundary_state_sha256")
    ):
        raise LifecycleError("SSPL1 import differs from its normalized authority")
    return {
        "n": parts.header.n,
        "bit_tuple": list(parts.header.bits),
        "symbol_sha256": parts.symbol_sha256,
        "boundary_state_sha256": execute._array_state_sha256(arrays),  # noqa: SLF001
        "canonical_cold_parse": True,
    }


def import_stream_worker(source_path: Path, config_path: Path) -> dict[str, Any]:
    config = _read_canonical_object(config_path)
    actual.verify_record(config, "import_worker_config_sha256")
    if (
        config.get("schema") != "structsplat.comp011.import-worker-config.v1"
        or config.get("protocol") != PROTOCOL
        or config.get("task_sha256") != TASK_SHA256
        or not isinstance(config.get("authority"), Mapping)
    ):
        raise LifecycleError("import worker config drift")
    blob = _read_regular_no_follow(source_path)
    authority = config["authority"]
    if (
        len(blob) != authority["sspl1"]["bytes"]
        or _sha256_bytes(blob) != authority["sspl1"]["sha256"]
    ):
        raise LifecycleError("import worker complete-stream authority drift")
    return actual.seal_record(
        {
            "schema": "structsplat.comp011.import-stream-worker.v1",
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "image_id": authority["image_id"],
            "bit_tuple": authority["bit_tuple"],
            "import_worker_config_sha256": config["import_worker_config_sha256"],
            "blob_bytes": len(blob),
            "blob_sha256": _sha256_bytes(blob),
            "parsed": _parse_imported_sspl1(blob, authority),
            "target_or_pixel_payloads_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "import_worker_sha256",
    )


def _import_worker_config(authority: Mapping[str, Any]) -> dict[str, Any]:
    return actual.seal_record(
        {
            "schema": "structsplat.comp011.import-worker-config.v1",
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "authority": dict(authority),
        },
        "import_worker_config_sha256",
    )


def _run_import_stream_worker(
    *,
    outdir: Path,
    source_path: Path,
    authority: Mapping[str, Any],
    denied_read_probes: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="comp011-import-worker-") as temporary:
        workspace = Path(temporary)
        config_path = workspace / "config.json"
        config = _import_worker_config(authority)
        _exclusive_write(config_path, _canonical_json(config))
        command = [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_import-stream",
            "--source",
            os.fspath(source_path),
            "--config",
            os.fspath(config_path),
        ]
        worker, attestation, launch = _run_json_worker(
            command,
            outdir=outdir,
            timeout=600,
            read_only=[source_path],
            read_write=[workspace, *_sandbox_process_read_write()],
            denied_read_probes=denied_read_probes,
            worker_seal_field="import_worker_sha256",
        )
    actual.verify_record(worker, "import_worker_sha256")
    if (
        worker.get("schema") != "structsplat.comp011.import-stream-worker.v1"
        or _identity(worker, "import stream worker")
        != _identity(authority, "import stream authority")
        or worker.get("import_worker_config_sha256")
        != config["import_worker_config_sha256"]
        or worker.get("blob_bytes") != authority["sspl1"]["bytes"]
        or worker.get("blob_sha256") != authority["sspl1"]["sha256"]
        or worker.get("target_or_pixel_payloads_opened") is not False
        or worker.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("import stream worker authority/access drift")
    return dict(worker["parsed"]), {
        "worker": worker,
        "sandbox_attestation": attestation,
        "launch": launch,
        "denied_read_probe_paths": [os.fspath(path) for path in denied_read_probes],
    }


def _verify_import_stream_launch_policy(
    *,
    outdir: Path,
    preflight_record: Mapping[str, Any],
    normalized_authority: Mapping[str, Any],
    expected_cell: Mapping[str, Any],
    artifact_source: Path,
    worker: Mapping[str, Any],
    attestation: Mapping[str, Any],
    launch: Mapping[str, Any],
    denied_read_probe_paths: Sequence[str],
) -> None:
    """Recompute the one exact target-free import-worker capability request."""

    actual.verify_record(worker, "import_worker_sha256")
    _verify_launch_bundle(
        {"attestation": attestation, "launch": launch},
        worker,
        "import_worker_sha256",
    )
    roots = preflight_record.get("operational_upstream_roots")
    comp008 = normalized_authority.get("comp008")
    targets = comp008.get("targets") if isinstance(comp008, Mapping) else None
    policy = launch.get("worker_policy")
    command = policy.get("command") if isinstance(policy, Mapping) else None
    if (
        not isinstance(roots, Mapping)
        or set(roots) != {"comp008", "comp009", "comp010"}
        or not isinstance(roots.get("comp008"), str)
        or not isinstance(targets, list)
        or not targets
        or not isinstance(policy, Mapping)
        or not isinstance(command, list)
        or len(command) != 8
        or command[:5]
        != [
            sys.executable,
            "-m",
            "benchmarks.ssp2v_actual_run",
            "_import-stream",
            "--source",
        ]
        or command[6] != "--config"
        or not isinstance(command[5], str)
        or not isinstance(command[7], str)
    ):
        raise LifecycleError("import-worker persisted launch policy is malformed")

    upstream_root = _lexical(Path(roots["comp008"]))
    source_relative = _safe_relative_path(
        expected_cell.get("sspl1", {}).get("path"),
        "normalized import-worker source path",
    )
    upstream_source = upstream_root / source_relative
    artifact_source = _lexical(artifact_source)
    source_path = Path(command[5])
    config_path = Path(command[7])
    if (
        _lexical(source_path) != source_path
        or _lexical(config_path) != config_path
        or source_path not in {upstream_source, artifact_source}
    ):
        raise LifecycleError("import-worker command source/config binding drift")
    workspace = config_path.parent
    prefix = "comp011-import-worker-"
    if config_path != workspace / "config.json" or not (
        workspace.name.startswith(prefix) and len(workspace.name) > len(prefix)
    ):
        raise LifecycleError("import-worker private config workspace drift")

    target_relative = _safe_relative_path(
        targets[0].get("relative_path") if isinstance(targets[0], Mapping) else None,
        "import-worker denied target probe",
    )
    expected_denied = [os.fspath(upstream_root / target_relative)]
    operational_roots = [
        _lexical(Path(value))
        for value in roots.values()
        if isinstance(value, str)
    ]
    forbidden_roots = [
        ROOT,
        _lexical(outdir),
        *operational_roots,
        upstream_source,
        artifact_source,
        *(Path(path) for path in expected_denied),
    ]
    if any(
        workspace == root
        or workspace.is_relative_to(root)
        or root.is_relative_to(workspace)
        for root in forbidden_roots
    ):
        raise LifecycleError("import-worker private config workspace overlaps protected data")

    expected_config = _import_worker_config(expected_cell)
    # This verifies the historical import launch, not a worker that is about to
    # run from the currently active source tree.  During captured replay the
    # active module root is the transient relocated snapshot, whereas the
    # persisted import receipt was created from ``outdir/runtime/source``.
    # Rebuild that original authority explicitly so relocation cannot rewrite
    # an otherwise exact historical launch policy.
    worker_source_root = _ordinary_worker_source_root(outdir)
    launcher = worker_source_root / "benchmarks/ssp2v_landlock.py"
    expected_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_import-stream",
        "--source",
        os.fspath(source_path),
        "--config",
        os.fspath(config_path),
    ]
    expected_read_only = sorted(
        {
            _worker_policy_path(path)
            for path in (
                *_sandbox_system_read_only(),
                *_worker_source_read_only(worker_source_root),
                source_path,
            )
        }
    )
    expected_read_write = sorted(
        {
            os.fspath(workspace),
            *(_worker_policy_path(path) for path in _sandbox_process_read_write()),
        }
    )
    expected_policy = actual.seal_record(
        {
            "schema": WORKER_POLICY_SCHEMA,
            "command": expected_command,
            "cwd": (worker_source_root / "benchmarks").resolve(strict=True).as_posix(),
            "environment": _ordinary_worker_environment(outdir, worker_source_root),
            "read_only_request": expected_read_only,
            "read_write_request": expected_read_write,
            "denied_read_probe_paths": expected_denied,
            "launcher_path": launcher.as_posix(),
            "launcher_sha256": _sha256_file(launcher),
            "timeout_seconds": 600.0,
        },
        "worker_policy_sha256",
    )
    if (
        policy != expected_policy
        or list(denied_read_probe_paths) != expected_denied
        or worker.get("protocol") != PROTOCOL
        or worker.get("task_sha256") != TASK_SHA256
        or worker.get("import_worker_config_sha256")
        != expected_config["import_worker_config_sha256"]
    ):
        raise LifecycleError("import-worker launch differs from its exact sealed authority policy")


def import_streams(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    _recover_interrupted_stage_completion(outdir, "import-streams")
    existing = outdir / _STAGE_MANIFEST_PATHS["import-streams"]
    if existing.exists():
        return verify_import_streams(outdir)
    journal, guard, records = _stage_guard(outdir, "import-streams")
    preflight_record = records["preflight"]
    preflight_seal = str(preflight_record["preflight_binding_sha256"])
    authority = _authority_copy(outdir, preflight_record)
    cells = authority["comp008"]["cells"]
    by_identity = {
        (str(cell["image_id"]), tuple(cell["bit_tuple"])): cell for cell in cells
    }
    if list(by_identity) != _expected_identities():
        raise LifecycleError("normalized SSPL1 authorities are not in frozen order")
    root = Path(str(preflight_record["operational_upstream_roots"]["comp008"]))
    target_probe = root / _safe_relative_path(
        authority["comp008"]["targets"][0]["relative_path"],
        "import-worker denied target probe",
    )
    imported: list[dict[str, Any]] = []
    for image_id, bits in _expected_identities():
        cell = by_identity[(image_id, bits)]
        relative = Path(str(cell["sspl1"]["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise LifecycleError("unsafe normalized development-stream path")
        source = root / relative
        destination = outdir / "streams" / image_id / _tuple_label(bits) / "sspl1.bin"
        guard.register(source, "development_stream")
        guard.register(destination, "artifact_stream")
        if destination.exists():
            blob = guard.read_bytes(
                destination,
                stage="import-streams",
                purpose="resume-verify immutable SSPL1 copy",
                prior_stage_seal=preflight_seal,
            )
        else:
            blob = guard.read_bytes(
                source,
                stage="import-streams",
                purpose="first legal open of sealed development SSPL1",
                prior_stage_seal=preflight_seal,
            )
        if len(blob) != cell["sspl1"]["bytes"] or _sha256_bytes(blob) != cell["sspl1"]["sha256"]:
            raise LifecycleError("development SSPL1 complete-byte authority drift")
        parsed, parse_worker = _run_import_stream_worker(
            outdir=outdir,
            source_path=source if not destination.exists() else destination,
            authority=cell,
            denied_read_probes=[target_probe],
        )
        artifact = _journaled_immutable(
            journal,
            outdir,
            destination,
            blob,
            stage="import-streams",
            path_class="artifact_stream",
            purpose="persist immutable verified SSPL1 copy",
            prior_stage_seal=preflight_seal,
        )
        imported.append(
            {
                "image_id": image_id,
                "bit_tuple": list(bits),
                **artifact,
                **parsed,
                "comp008_cell_record_sha256": cell["cell_record_sha256"],
                "parse_worker": parse_worker,
            }
        )
    return _finalize_stage(
        outdir,
        journal,
        stage="import-streams",
        schema=IMPORT_STREAMS_SCHEMA,
        prior_stage_seal=preflight_seal,
        body={
            "authority_manifest_sha256": authority["authority_manifest_sha256"],
            "streams": imported,
            "stream_count": len(imported),
            "development_streams_opened": True,
            "target_or_pixel_payloads_opened": False,
            "candidate_or_model_products_created": False,
        },
    )


def verify_import_streams(outdir: Path, *, captured_replay: bool = False) -> dict[str, Any]:
    outdir = _lexical(outdir)
    preflight_record = verify_preflight(outdir, captured_replay=captured_replay)
    record = _verify_stage_base(
        outdir,
        stage="import-streams",
        schema=IMPORT_STREAMS_SCHEMA,
        prior_record=preflight_record,
    )
    streams = record.get("streams")
    if (
        not isinstance(streams, list)
        or len(streams) != 16
        or record.get("stream_count") != 16
        or record.get("development_streams_opened") is not True
        or record.get("target_or_pixel_payloads_opened") is not False
        or record.get("candidate_or_model_products_created") is not False
    ):
        raise LifecycleError("import-streams inventory/access flags drift")
    authority = _authority_copy(outdir, preflight_record)
    if record.get("authority_manifest_sha256") != authority.get("authority_manifest_sha256"):
        raise LifecycleError("import-streams detached from normalized upstream authority")
    authority_by_identity = {
        _identity(cell, "normalized SSPL1 authority"): cell
        for cell in authority["comp008"]["cells"]
    }
    if list(authority_by_identity) != _expected_identities():
        raise LifecycleError("normalized SSPL1 authorities are not in frozen order")
    identities: list[tuple[str, tuple[int, int, int, int]]] = []
    for stream in streams:
        if not isinstance(stream, Mapping):
            raise LifecycleError("imported stream record is not an object")
        identity = _identity(stream, "imported stream")
        identities.append(identity)
        expected = authority_by_identity.get(identity)
        if expected is None:
            raise LifecycleError("imported SSPL1 has no normalized per-cell authority")
        expected_fields = {
            "image_id": expected["image_id"],
            "bit_tuple": expected["bit_tuple"],
            "bytes": expected["sspl1"]["bytes"],
            "sha256": expected["sspl1"]["sha256"],
            "symbol_sha256": expected["absolute_symbols"]["symbol_sha256"],
            "boundary_state_sha256": expected["decoded_boundary_state_sha256"],
            "comp008_cell_record_sha256": expected["cell_record_sha256"],
        }
        if any(stream.get(key) != value for key, value in expected_fields.items()):
            raise LifecycleError("imported SSPL1 record differs from normalized per-cell authority")
        path = _artifact_path(outdir, stream.get("relative_path"), "imported-stream path")
        _verify_regular_record(path, stream)
        blob = _read_regular_no_follow(path)
        parse_bundle = stream.get("parse_worker")
        if not isinstance(parse_bundle, Mapping):
            raise LifecycleError("imported SSPL1 sandboxed parse evidence is absent")
        parse_worker = parse_bundle.get("worker")
        parse_attestation = parse_bundle.get("sandbox_attestation")
        parse_launch = parse_bundle.get("launch")
        denied_paths = parse_bundle.get("denied_read_probe_paths")
        if not all(
            isinstance(value, Mapping)
            for value in (parse_worker, parse_attestation, parse_launch)
        ) or not isinstance(denied_paths, list):
            raise LifecycleError("imported SSPL1 sandboxed parse evidence is malformed")
        _verify_import_stream_launch_policy(
            outdir=outdir,
            preflight_record=preflight_record,
            normalized_authority=authority,
            expected_cell=expected,
            artifact_source=path,
            worker=parse_worker,
            attestation=parse_attestation,
            launch=parse_launch,
            denied_read_probe_paths=denied_paths,
        )
        denied_records = parse_attestation.get("denied_read_probes")
        if (
            parse_worker.get("schema")
            != "structsplat.comp011.import-stream-worker.v1"
            or _identity(parse_worker, "import stream worker") != identity
            or parse_worker.get("blob_bytes") != stream.get("bytes")
            or parse_worker.get("blob_sha256") != stream.get("sha256")
            or parse_worker.get("target_or_pixel_payloads_opened") is not False
            or parse_worker.get("confirmation_payloads_accessed") is not False
            or not isinstance(denied_records, list)
            or [item.get("path") for item in denied_records] != denied_paths
            or any(
                item.get("errno") not in (errno.EACCES, errno.EPERM)
                for item in denied_records
            )
            or parse_attestation.get("command", [None, None, None, None])[2:4]
            != ["benchmarks.ssp2v_actual_run", "_import-stream"]
        ):
            raise LifecycleError("imported SSPL1 sandboxed parse authority/access drift")
        parsed = parse_worker.get("parsed")
        if not isinstance(parsed, Mapping):
            raise LifecycleError("imported SSPL1 sandboxed parse result is absent")
        if any(parsed.get(key) != stream.get(key) for key in parsed):
            raise LifecycleError("imported SSPL1 semantic record drift")
        if captured_replay and _parse_imported_sspl1(blob, expected) != parsed:
            raise LifecycleError("captured replay SSPL1 reparse differs from import worker")
    if identities != _expected_identities():
        raise LifecycleError("imported stream inventory differs from the frozen grid")
    return record


def _native_arithmetic_record(outdir: Path, preflight_record: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    native = preflight_record.get("native")
    record = native.get("arithmetic") if isinstance(native, Mapping) else None
    if not isinstance(record, dict):
        raise LifecycleError("preflight arithmetic native authority is absent")
    path = outdir / "runtime/native" / str(record.get("path"))
    _verify_regular_record(path, record)
    return path, record


_CAPTURED_COMP009_SCRIPT = r"""
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
sys.path[:0] = [str(root), str(root / "src")]
from benchmarks import ssp2e_arithmetic as arithmetic
from benchmarks import ssp2e_execute as execute

for module in (arithmetic, execute):
    origin = pathlib.Path(module.__file__).resolve(strict=True)
    origin.relative_to(root)
source = pathlib.Path(sys.argv[2]).read_bytes()
native = arithmetic.NativeArithmeticCoder(pathlib.Path(sys.argv[3]))
output = pathlib.Path(sys.argv[4])
result = execute.execute_cell(
    source,
    native,
    expected_symbol_sha256=sys.argv[5],
    expected_boundary_state_sha256=sys.argv[6],
)
output.mkdir(parents=True, exist_ok=False)
for name, blob in result.blobs.items():
    (output / (name + ".bin")).write_bytes(blob)
payload = (json.dumps(result.record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
(output / "execution.json").write_bytes(payload)
(output / "worker.json").write_bytes((json.dumps({
    "isolated_python": True,
    "captured_root": str(root),
    "module_origins": {
        "benchmarks.ssp2e_arithmetic": str(pathlib.Path(arithmetic.__file__).resolve(strict=True).relative_to(root)),
        "benchmarks.ssp2e_execute": str(pathlib.Path(execute.__file__).resolve(strict=True).relative_to(root)),
    },
}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
"""


def _captured_comp009_system_read_only(
    parent_read_only_request: Sequence[Any],
) -> list[str]:
    """Carry only the sealed parent's system-read authority into COMP-009."""

    if isinstance(parent_read_only_request, (str, bytes)):
        raise LifecycleError("captured COMP-009 parent read policy is malformed")
    values = list(parent_read_only_request)
    if (
        any(not isinstance(value, str) for value in values)
        or values != sorted(set(values))
        or any(
            not Path(value).is_absolute()
            or _lexical(Path(value)).as_posix() != value
            for value in values
        )
    ):
        raise LifecycleError("captured COMP-009 parent read policy is malformed")
    return [
        value
        for value in values
        if _safe_system_read_only_policy_path(Path(value))
    ]


def _captured_comp009_reproduction(
    *,
    extracted_source: Path,
    source_blob: bytes,
    authority: Mapping[str, Any],
    native_path: Path,
    captured_source_manifest_sha256: str,
    parent_read_only_request: Sequence[Any],
) -> tuple[
    dict[str, execute.ReproducedComp009Stream],
    dict[str, execute.Comp009Authority],
    dict[str, Any],
]:
    """Regenerate E/S/L/F through only the safely extracted frozen COMP-009 source."""

    expected_symbol = str(authority["input"]["symbol_sha256"])
    expected_boundary = str(authority["input"]["decoded_state_sha256"])
    with tempfile.TemporaryDirectory(prefix="comp011-comp009-cell-") as temporary:
        workspace = Path(temporary)
        source_path = workspace / "source.sspl1"
        _exclusive_write(source_path, source_blob)
        source_copy_blob = _read_regular_no_follow(source_path)
        if source_copy_blob != source_blob:
            raise LifecycleError("captured COMP-009 private source copy drift")
        output_parent = workspace / "output"
        output_parent.mkdir()
        output = output_parent / "result"
        environment = _worker_environment()
        environment["PYTHONPATH"] = os.pathsep.join(
            (os.fspath(extracted_source), os.fspath(extracted_source / "src"))
        )
        environment["COMP009_CAPTURED_SOURCE"] = "1"
        environment["COMP009_CAPTURED_ROOT"] = os.fspath(extracted_source)
        policy_environment = dict(environment)
        nonce = os.urandom(32).hex()
        environment["STRUCTSPLAT_PARENT_LAUNCH_NONCE"] = nonce
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _CAPTURED_COMP009_SCRIPT,
            os.fspath(extracted_source),
            os.fspath(source_path),
            os.fspath(native_path),
            os.fspath(output),
            expected_symbol,
            expected_boundary,
        ]
        captured_readonly = [
            *(
                Path(value)
                for value in _captured_comp009_system_read_only(
                    parent_read_only_request
                )
            ),
            extracted_source,
            source_path,
            native_path,
        ]
        captured_writable = [output_parent]
        captured_launcher = Path(sandbox.__file__).resolve(strict=True)
        policy = _worker_launch_policy(
            command=command,
            cwd=extracted_source,
            environment=policy_environment,
            read_only=captured_readonly,
            read_write=captured_writable,
            denied_read_probes=[],
            launcher_path=captured_launcher,
            timeout=3600,
        )
        completed, attestation = sandbox.run_sandboxed(
            command,
            read_only=captured_readonly,
            read_write=captured_writable,
            denied_read_probes=[],
            cwd=extracted_source,
            env=environment,
            timeout=3600,
            # COMP-009 predates the COMP-011 launcher.  Use the launcher from
            # the executing COMP-011 source closure (the captured COMP-011
            # copy during replay), never a guessed path in the older archive.
            launcher_path=captured_launcher,
        )
        _verify_sandbox_ancestry(
            attestation,
            expected_parent_sandbox_attestation_sha256=os.environ.get(
                sandbox.SANDBOX_ATTESTATION_ENV
            ),
        )
        if completed.returncode != 0 or completed.stdout:
            raise LifecycleError(
                "captured COMP-009 reproduction failed: "
                f"returncode={completed.returncode}, stdout={completed.stdout!r}, "
                f"stderr={completed.stderr.decode(errors='replace')}"
            )
        regenerated = _read_canonical_object(output / "execution.json")
        worker = _read_canonical_object(output / "worker.json")
        launch = _launch_receipt(
            nonce=nonce,
            command=command,
            worker=worker,
            worker_seal_field="canonical_worker_sha256",
            attestation=attestation,
            worker_policy=policy,
        )
        if (
            worker.get("isolated_python") is not True
            or set(worker.get("module_origins", {}))
            != {"benchmarks.ssp2e_arithmetic", "benchmarks.ssp2e_execute"}
            or any(
                Path(str(relative)).is_absolute() or ".." in Path(str(relative)).parts
                for relative in worker["module_origins"].values()
            )
        ):
            raise LifecycleError("captured COMP-009 worker module-origin attestation drift")
        observed_seal = regenerated.get("execution_sha256")
        unsigned = dict(regenerated)
        unsigned.pop("execution_sha256", None)
        if observed_seal != _sha256_bytes(_canonical_json(unsigned)):
            raise LifecycleError("captured COMP-009 regenerated execution seal is invalid")
        regenerated_core = dict(unsigned)
        regenerated_core.pop("timings", None)
        if regenerated_core != authority.get("execution_core_without_timings"):
            raise LifecycleError("captured COMP-009 deterministic execution record drift")

        source_parts = container.extract_sspl1_parts(source_blob)
        frequency_provider = execute._comp009_frequency_provider()  # noqa: SLF001
        reproduced: dict[str, execute.ReproducedComp009Stream] = {}
        expected: dict[str, execute.Comp009Authority] = {}
        for name in execute.COMP009_NAMES:
            blob = _read_regular_no_follow(output / f"{name}.bin")
            stream_authority = authority["streams"][name]
            if (
                len(blob) != stream_authority["complete_bytes"]
                or _sha256_bytes(blob) != stream_authority["blob_sha256"]
            ):
                raise LifecycleError(f"captured COMP-009 {name} blob authority drift")
            parsed = comp009_container.parse_container(blob, reference_sspl1=source_blob)
            provider = None if name == "ssp2f" else frequency_provider
            decoded = comp009_container.decode_container(
                blob,
                arithmetic_decode=arithmetic.decode,
                arithmetic_encode=arithmetic.encode,
                frequency_provider=provider,
                reference_sspl1=source_blob,
            )
            model_payload = (
                parsed.payloads["grid"] + parsed.payloads["head"]
                if name in ("ssp2e", "ssp2s")
                else parsed.payloads["model"]
            )
            model_state_sha256 = _sha256_bytes(model_payload)
            boundary_arrays = execute.boundary_arrays(source_parts.header, decoded.symbols)
            boundary_state_sha256 = execute._array_state_sha256(boundary_arrays)  # noqa: SLF001
            components = execute._comp009_component_records(parsed, blob)  # noqa: SLF001
            verification = execute.comp009_verification_sha256(
                name=name,
                source_blob_sha256=_sha256_bytes(source_blob),
                complete_bytes=len(blob),
                blob_sha256=_sha256_bytes(blob),
                model_state_sha256=model_state_sha256,
                symbol_sha256=decoded.symbol_sha256,
                boundary_state_sha256=boundary_state_sha256,
                components=components,
                captured_source_sha256=execute.COMP009_CAPTURED_SOURCE_SHA256,
                execution_sha256=str(authority["execution_sha256"]),
                canonical_cold_decode=True,
                python_native_decode_parity=True,
            )
            reproduced[name] = execute.ReproducedComp009Stream(
                blob=blob,
                model_state_sha256=model_state_sha256,
                symbol_sha256=decoded.symbol_sha256,
                boundary_state_sha256=boundary_state_sha256,
                canonical_cold_decode=True,
                python_native_decode_parity=True,
                source_blob_sha256=_sha256_bytes(source_blob),
                symbols=decoded.symbols,
                boundary_arrays=boundary_arrays,
                components=components,
                captured_source_sha256=execute.COMP009_CAPTURED_SOURCE_SHA256,
                execution_sha256=str(authority["execution_sha256"]),
                verification_sha256=verification,
            )
            expected[name] = execute.Comp009Authority(
                complete_bytes=len(blob),
                blob_sha256=_sha256_bytes(blob),
                model_state_sha256=model_state_sha256,
                symbol_sha256=decoded.symbol_sha256,
                boundary_state_sha256=boundary_state_sha256,
                captured_source_sha256=execute.COMP009_CAPTURED_SOURCE_SHA256,
                execution_sha256=str(authority["execution_sha256"]),
                verification_sha256=verification,
            )
        provenance = {
            "isolated_python_flags": ["-I", "-B"],
            "script_sha256": _sha256_bytes(_CAPTURED_COMP009_SCRIPT.encode("utf-8")),
            "command": [Path(command[0]).name, *command[1:4], "<captured-root>", "<source>", "<native>", "<output>", "<symbol-sha256>", "<boundary-sha256>"],
            "source_copy": {
                "bytes": len(source_copy_blob),
                "sha256": _sha256_bytes(source_copy_blob),
                "private_copy_path": os.fspath(source_path),
                "exclusive_write_before_launch": True,
                "identity_verified_from_same_opened_copy_before_launch": True,
            },
            "returncode": completed.returncode,
            "stdout_bytes": len(completed.stdout),
            "stdout_sha256": _sha256_bytes(completed.stdout),
            "stdout_hex": completed.stdout.hex(),
            "stderr_bytes": len(completed.stderr),
            "stderr_sha256": _sha256_bytes(completed.stderr),
            "stderr_hex": completed.stderr.hex(),
            "worker": worker,
            "sandbox_attestation": attestation,
            "launch": launch,
            "captured_source_manifest_sha256": captured_source_manifest_sha256,
            "captured_source_archive_sha256": execute.COMP009_CAPTURED_SOURCE_SHA256,
            "deterministic_execution_core_equal": True,
            "decision_use": False,
        }
        return reproduced, expected, provenance


def _verify_captured_comp009_reproduction_policy(
    captured: Mapping[str, Any],
    *,
    producing_config: Mapping[str, Any],
    producing_policy: Mapping[str, Any],
    expected_authority: Mapping[str, Any],
    producer_sandbox_attestation_sha256: str,
) -> None:
    """Bind the nested COMP-009 launch to its already-verified producer authority."""

    actual.verify_record(producing_config, "producing_config_sha256")
    actual.verify_record(producing_policy, "producer_policy_sha256")
    if producing_config.get("producer_policy") != producing_policy:
        raise LifecycleError("captured COMP-009 producer policy binding drift")
    outer_command = producing_policy.get("command")
    outer_environment = producing_policy.get("environment")
    outer_readonly = producing_policy.get("read_only_request")
    authority_input = expected_authority.get("input")
    if (
        not isinstance(outer_command, list)
        or len(outer_command) != 18
        or not isinstance(outer_environment, Mapping)
        or not isinstance(outer_readonly, list)
        or not isinstance(authority_input, Mapping)
        or producing_config.get("comp009_authority") != expected_authority
    ):
        raise LifecycleError("captured COMP-009 producer authority is malformed")

    worker = captured.get("worker")
    attestation = captured.get("sandbox_attestation")
    launch = captured.get("launch")
    if not all(isinstance(value, Mapping) for value in (worker, attestation, launch)):
        raise LifecycleError("captured COMP-009 sandbox/worker launch evidence is malformed")
    _verify_launch_bundle(
        {"attestation": attestation, "launch": launch},
        worker,
        "canonical_worker_sha256",
        expected_parent_sandbox_attestation_sha256=(
            producer_sandbox_attestation_sha256
        ),
    )
    policy = launch.get("worker_policy")
    if not isinstance(policy, Mapping):
        raise LifecycleError("captured COMP-009 nested launch policy is absent")
    command = policy.get("command")
    environment = policy.get("environment")
    read_only = policy.get("read_only_request")
    read_write = policy.get("read_write_request")
    denied = policy.get("denied_read_probe_paths")
    if (
        not isinstance(command, list)
        or len(command) != 11
        or not isinstance(environment, Mapping)
        or not isinstance(read_only, list)
        or read_only != sorted(set(read_only))
        or not isinstance(read_write, list)
        or read_write != sorted(set(read_write))
        or denied != []
    ):
        raise LifecycleError("captured COMP-009 nested launch policy is malformed")

    producer_source_root = _lexical(Path(str(producing_policy.get("cwd")))).parent
    captured_root = Path(str(outer_command[9]))
    native_path = Path(str(outer_command[11]))
    producer_config_path = Path(str(outer_command[15]))
    source_path = Path(str(command[6]))
    output_path = Path(str(command[8]))
    absolute_paths = (
        captured_root,
        native_path,
        producer_config_path,
        source_path,
        output_path,
    )
    if any(not path.is_absolute() or _lexical(path) != path for path in absolute_paths):
        raise LifecycleError("captured COMP-009 nested command path is noncanonical")
    producer_workspace = producer_config_path.parent
    nested_workspace = source_path.parent
    if (
        producer_config_path != producer_workspace / "config.json"
        or nested_workspace.parent != producer_workspace
        or re.fullmatch(r"comp011-comp009-cell-[a-z0-9_]{8}", nested_workspace.name)
        is None
        or source_path != nested_workspace / "source.sspl1"
        or output_path != nested_workspace / "output/result"
    ):
        raise LifecycleError("captured COMP-009 private workspace grammar drift")

    expected_symbol = _require_sha256(
        authority_input.get("symbol_sha256"), "captured COMP-009 authority symbols"
    )
    expected_boundary = _require_sha256(
        authority_input.get("decoded_state_sha256"),
        "captured COMP-009 authority boundary",
    )
    expected_command = [
        outer_command[0],
        "-I",
        "-B",
        "-c",
        _CAPTURED_COMP009_SCRIPT,
        os.fspath(captured_root),
        os.fspath(source_path),
        os.fspath(native_path),
        os.fspath(output_path),
        expected_symbol,
        expected_boundary,
    ]
    if command != expected_command:
        raise LifecycleError("captured COMP-009 nested command authority drift")

    inherited_names = (
        "PATH",
        "LANG",
        "LC_ALL",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST",
        "CXX",
        "CC",
    )
    expected_environment = {
        name: outer_environment[name]
        for name in inherited_names
        if name in outer_environment
    }
    expected_environment.update(quality.FROZEN_THREAD_ENVIRONMENT)
    expected_environment.update(
        {
            "LD_PRELOAD": quality.REQUIRED_RENDERER_PRELOAD,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "NO_PROXY": "*",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HOME": "/nonexistent-comp011-home",
            "PYTHONPATH": os.pathsep.join(
                (os.fspath(captured_root), os.fspath(captured_root / "src"))
            ),
            "COMP009_CAPTURED_SOURCE": "1",
            "COMP009_CAPTURED_ROOT": os.fspath(captured_root),
        }
    )
    if "STRUCTSPLAT_SOURCE_MANIFEST_SHA256" in outer_environment:
        expected_environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
            outer_environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"]
        )
    system_readonly = _captured_comp009_system_read_only(outer_readonly)
    expected_readonly = sorted(
        {
            *system_readonly,
            os.fspath(captured_root),
            os.fspath(source_path),
            os.fspath(native_path),
        }
    )
    expected_readwrite = [os.fspath(output_path.parent)]
    expected_launcher = producer_source_root / "benchmarks/ssp2v_landlock.py"
    if (
        policy.get("cwd") != os.fspath(captured_root)
        or environment != expected_environment
        or read_only != expected_readonly
        or read_write != expected_readwrite
        or policy.get("launcher_path") != os.fspath(expected_launcher)
        or policy.get("launcher_sha256") != producing_policy.get("launcher_sha256")
        or policy.get("timeout_seconds") != 3600.0
    ):
        raise LifecycleError("captured COMP-009 exact nested sandbox policy drift")

    expected_module_origins = {
        "benchmarks.ssp2e_arithmetic": "benchmarks/ssp2e_arithmetic.py",
        "benchmarks.ssp2e_execute": "benchmarks/ssp2e_execute.py",
    }
    if (
        set(worker) != {"isolated_python", "captured_root", "module_origins"}
        or worker.get("isolated_python") is not True
        or worker.get("captured_root") != os.fspath(captured_root)
        or worker.get("module_origins") != expected_module_origins
    ):
        raise LifecycleError("captured COMP-009 nested worker source identity drift")

    source_copy = captured.get("source_copy")
    expected_source_bytes = producing_config.get("source_bytes")
    expected_source_sha256 = producing_config.get("source_sha256")
    if (
        authority_input.get("bytes") != expected_source_bytes
        or authority_input.get("sha256") != expected_source_sha256
        or source_copy
        != {
            "bytes": expected_source_bytes,
            "sha256": expected_source_sha256,
            "private_copy_path": os.fspath(source_path),
            "exclusive_write_before_launch": True,
            "identity_verified_from_same_opened_copy_before_launch": True,
        }
    ):
        raise LifecycleError("captured COMP-009 private source-copy receipt drift")

    expected_keys = {
        "isolated_python_flags",
        "script_sha256",
        "command",
        "source_copy",
        "returncode",
        "stdout_bytes",
        "stdout_sha256",
        "stdout_hex",
        "stderr_bytes",
        "stderr_sha256",
        "stderr_hex",
        "worker",
        "sandbox_attestation",
        "launch",
        "captured_source_manifest_sha256",
        "captured_source_archive_sha256",
        "deterministic_execution_core_equal",
        "decision_use",
    }
    try:
        stdout = bytes.fromhex(str(captured.get("stdout_hex")))
        stderr = bytes.fromhex(str(captured.get("stderr_hex")))
    except ValueError as exc:
        raise LifecycleError("captured COMP-009 stdout/stderr evidence is malformed") from exc
    if (
        set(captured) != expected_keys
        or captured.get("isolated_python_flags") != ["-I", "-B"]
        or captured.get("script_sha256")
        != _sha256_bytes(_CAPTURED_COMP009_SCRIPT.encode("utf-8"))
        or captured.get("command")
        != [
            Path(str(outer_command[0])).name,
            "-I",
            "-B",
            "-c",
            "<captured-root>",
            "<source>",
            "<native>",
            "<output>",
            "<symbol-sha256>",
            "<boundary-sha256>",
        ]
        or captured.get("returncode") != 0
        or captured.get("stdout_bytes") != len(stdout)
        or captured.get("stdout_sha256") != _sha256_bytes(stdout)
        or captured.get("stderr_bytes") != len(stderr)
        or captured.get("stderr_sha256") != _sha256_bytes(stderr)
        or captured.get("captured_source_manifest_sha256")
        != producing_config.get("captured_source_manifest_sha256")
        or captured.get("captured_source_archive_sha256")
        != execute.COMP009_CAPTURED_SOURCE_SHA256
        or captured.get("deterministic_execution_core_equal") is not True
        or captured.get("decision_use") is not False
    ):
        raise LifecycleError("captured COMP-009 provenance/output binding drift")


def _cold_decode_expectations(
    execution_record: Mapping[str, Any], name: str
) -> tuple[str, str]:
    if name in execute.EXACT_FORMAT_ORDER:
        stream = execution_record["exact_formats"][name]
        symbol = str(stream["symbol_sha256"])
        boundary = (
            stream["boundary"]["state_sha256"]
            if name in ("sspl1", "ssp2z")
            else stream["boundary_state_sha256"]
        )
        return symbol, str(boundary)
    stream = execution_record["variants"][name]["stream"]
    return str(stream["symbol_sha256"]), str(stream["boundary"]["state_sha256"])


def _candidate_cold_worker_command(
    *,
    artifact_root: Path,
    blob_relative_path: str,
    native_relative_path: str,
    blob_sha256: str,
    blob_bytes: int,
    native_record: Mapping[str, Any],
    worker_source: Mapping[str, Any],
    symbol_sha256: str,
    boundary_state_sha256: str,
    stream_components_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_decode_worker",
        "candidate-cold-seal",
        "--artifact-root",
        os.fspath(_lexical(artifact_root)),
        "--blob",
        _safe_relative_path(blob_relative_path, "cold worker blob path").as_posix(),
        "--native-library",
        _safe_relative_path(native_relative_path, "cold worker native path").as_posix(),
        "--expected-blob-sha256",
        _require_sha256(blob_sha256, "cold worker blob"),
        "--expected-blob-bytes",
        str(blob_bytes),
        "--expected-native-library-sha256",
        str(native_record["sha256"]),
        "--expected-native-library-bytes",
        str(native_record["bytes"]),
        "--expected-worker-module-sha256",
        str(worker_source["sha256"]),
        "--expected-worker-module-bytes",
        str(worker_source["bytes"]),
        "--expected-symbol-sha256",
        _require_sha256(symbol_sha256, "cold worker symbols"),
        "--expected-boundary-state-sha256",
        _require_sha256(boundary_state_sha256, "cold worker boundary"),
        "--expected-stream-components-sha256",
        _require_sha256(stream_components_sha256, "cold worker stream components"),
    ]


def _is_worker_system_readonly_path(value: str) -> bool:
    """Recognize only the fixed system-path grammar used by `_sandbox_system_read_only`."""

    if not isinstance(value, str) or not Path(value).is_absolute():
        return False
    path = Path(value)
    proc_self = Path("/proc/self")
    exact = {
        Path(sys.prefix).resolve(strict=False),
        Path("/usr").resolve(strict=False),
        Path("/lib").resolve(strict=False),
        Path("/lib64").resolve(strict=False),
        Path("/etc/ld.so.cache").resolve(strict=False),
        Path("/etc/localtime").resolve(strict=False),
        Path(os.devnull).resolve(strict=False),
        Path("/dev/urandom").resolve(strict=False),
        Path("/etc/nvcc.profile").resolve(strict=False),
        proc_self / "maps",
        proc_self / "status",
        proc_self / "stat",
        proc_self / "statm",
        proc_self / "cmdline",
        Path("/proc/cpuinfo"),
        Path("/proc/meminfo"),
        Path("/proc/devices"),
        Path("/proc/sys/vm/mmap_min_addr"),
        Path("/proc/driver/nvidia"),
        Path("/sys/devices/system/cpu/online"),
        Path("/sys/devices/system/cpu/possible"),
        Path("/sys/devices/system/node"),
        Path("/sys/devices/system/memory/block_size_bytes"),
    }
    if path in exact:
        return True
    parts = path.parts
    pci_parts = parts[3:-1]
    return (
        len(parts) == 7
        and parts[:5] == ("/", "sys", "devices", "system", "cpu")
        and parts[5].startswith("cpu")
        and parts[5][3:].isdigit()
        and parts[6] == "online"
    ) or (
        len(parts) == 4
        and parts[:3] == ("/", "sys", "module")
        and parts[3].startswith("nvidia")
    ) or (
        len(parts) >= 6
        and parts[:3] == ("/", "sys", "devices")
        and parts[-1] == "numa_node"
        and re.fullmatch(r"pci[0-9a-f]{4}:[0-9a-f]{2}", pci_parts[0])
        is not None
        and all(
            re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", part)
            is not None
            for part in pci_parts[1:]
        )
    )


def _verify_captured_decode_worker_policy(
    bundle: Mapping[str, Any],
    *,
    expected_command: Sequence[str],
    artifact_root: Path,
    blob_relative_path: str,
    native_relative_path: str,
    expected_source_root: Path | None,
    preflight_record: Mapping[str, Any],
    replay_source_authority: Mapping[str, Any],
) -> Path:
    """Bind a persisted cold/resource request to captured source and scratch artifacts."""

    launch = bundle.get("launch")
    policy = launch.get("worker_policy") if isinstance(launch, Mapping) else None
    if not isinstance(policy, Mapping):
        raise LifecycleError("captured decode-worker launch policy is absent")
    command = policy.get("command")
    cwd_value = policy.get("cwd")
    environment = policy.get("environment")
    readonly = policy.get("read_only_request")
    writable = policy.get("read_write_request")
    denied = policy.get("denied_read_probe_paths")
    launcher_value = policy.get("launcher_path")
    if (
        command != list(expected_command)
        or not isinstance(cwd_value, str)
        or not Path(cwd_value).is_absolute()
        or not isinstance(environment, Mapping)
        or not isinstance(readonly, list)
        or any(not isinstance(value, str) for value in readonly)
        or readonly != sorted(set(readonly))
        or not isinstance(writable, list)
        or writable != []
        or not isinstance(denied, list)
        or denied != []
        or policy.get("timeout_seconds") != 600.0
        or not isinstance(launcher_value, str)
    ):
        raise LifecycleError("captured decode-worker persisted request drift")

    cwd = _lexical(Path(cwd_value))
    source_root = cwd.parent
    authority_root = _verify_replay_captured_source_authority(
        replay_source_authority,
        preflight_record=preflight_record,
        expected_source_root=(
            expected_source_root if expected_source_root is not None else source_root
        ),
    )
    launcher_authority = replay_source_authority["launcher"]
    if (
        cwd != source_root / "benchmarks"
        or source_root != authority_root
        or Path(launcher_value) != cwd / "ssp2v_landlock.py"
        or policy.get("launcher_sha256") != launcher_authority.get("sha256")
    ):
        raise LifecycleError("captured decode-worker source-root/launcher drift")

    root = _lexical(artifact_root)
    blob_relative = _safe_relative_path(
        blob_relative_path, "captured decode-worker blob path"
    )
    native_relative = _safe_relative_path(
        native_relative_path, "captured decode-worker native path"
    )
    _captured_worker_environment_semantics(
        environment,
        outdir=root,
        captured_source_root=source_root,
        tmpdir=None,
        preflight_record=preflight_record,
    )

    required_readonly = {
        os.fspath(authority_root),
        os.fspath(root / "runtime"),
        os.fspath(root / blob_relative),
    }
    if (
        environment.get("STRUCTSPLAT_SOURCE_MANIFEST_SHA256")
        != replay_source_authority.get("source_manifest_sha256")
    ):
        raise LifecycleError("captured decode-worker source manifest environment drift")
    if not required_readonly.issubset(readonly) or any(
        value not in required_readonly and not _is_worker_system_readonly_path(value)
        for value in readonly
    ):
        raise LifecycleError("captured decode-worker read-only policy drift")
    if not native_relative.is_relative_to(Path("runtime")):
        raise LifecycleError("captured decode-worker native path is outside runtime")
    return source_root


def _run_cold_decode_sample(
    *,
    outdir: Path,
    blob_path: Path,
    blob_sha256: str,
    symbol_sha256: str,
    boundary_state_sha256: str,
    native_path: Path,
    native_record: Mapping[str, Any],
    stream_components_sha256: str,
) -> dict[str, Any]:
    """Narrow adapter for one fresh, nongating strict cold-seal worker launch."""

    worker_source = _decode_worker_source_record()
    command = _candidate_cold_worker_command(
        artifact_root=outdir,
        blob_relative_path=blob_path.relative_to(outdir).as_posix(),
        native_relative_path=native_path.relative_to(outdir).as_posix(),
        blob_sha256=blob_sha256,
        blob_bytes=blob_path.stat().st_size,
        native_record=native_record,
        worker_source=worker_source,
        symbol_sha256=symbol_sha256,
        boundary_state_sha256=boundary_state_sha256,
        stream_components_sha256=stream_components_sha256,
    )
    record, attestation, launch = _run_json_worker(
        command,
        outdir=outdir,
        timeout=600,
        read_only=[outdir / "runtime", blob_path],
        worker_seal_field="cold_seal_sha256",
    )
    decode_worker.verify_candidate_cold_seal(
        record,
        expected_blob_sha256=blob_sha256,
        expected_blob_bytes=blob_path.stat().st_size,
        expected_symbol_sha256=symbol_sha256,
        expected_boundary_state_sha256=boundary_state_sha256,
        expected_stream_components_sha256=stream_components_sha256,
    )
    return {"cold_seal": record, "sandbox_attestation": attestation, "launch": launch}


def _ordinary_worker_source_root(outdir: Path) -> Path:
    """Return the only source authority permitted for ordinary stage workers."""

    outdir = _lexical(outdir)
    runtime_source = outdir / _RUNTIME_SOURCE_RELATIVE
    if (runtime_source / "SOURCE_MANIFEST.json").is_file():
        _verify_runtime_source_tree(runtime_source)
        return runtime_source
    source_root = _worker_source_root(outdir)
    if (source_root / "SOURCE_MANIFEST.json").is_file():
        _verify_runtime_source_tree(source_root)
    return source_root


def _ordinary_worker_environment(outdir: Path, source_root: Path) -> dict[str, str]:
    """Rebuild the original-worker import authority without ambient replay routing."""

    outdir = _lexical(outdir)
    source_root = _lexical(source_root)
    return _worker_environment(outdir, source_root=source_root)


def _policy_path_for_exact_verification(path: Path, *, allow_missing: bool = False) -> str:
    """Mirror a launch-time policy path, permitting only explicit ephemeral paths."""

    lexical = _lexical(path)
    if lexical.is_relative_to(Path("/proc/self")):
        return lexical.as_posix()
    if allow_missing:
        if lexical.exists() and lexical.resolve(strict=True) != lexical:
            raise LifecycleError("ordinary worker ephemeral path is a symlink alias")
        return lexical.as_posix()
    return _worker_policy_path(lexical)


def _verify_exact_ordinary_worker_policy(
    bundle: Mapping[str, Any],
    *,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    read_only_request: Sequence[str],
    read_write_request: Sequence[str],
    denied_read_probe_paths: Sequence[str],
    launcher_path: Path,
    timeout: float,
) -> None:
    """Reject a coherently resealed ordinary-worker capability substitution."""

    launch = bundle.get("launch")
    policy = launch.get("worker_policy") if isinstance(launch, Mapping) else None
    if not isinstance(policy, Mapping):
        raise LifecycleError("ordinary worker persisted launch policy is absent")
    launcher = launcher_path.resolve(strict=True)
    expected = actual.seal_record(
        {
            "schema": WORKER_POLICY_SCHEMA,
            "command": list(command),
            "cwd": cwd.resolve(strict=True).as_posix(),
            "environment": dict(environment),
            "read_only_request": sorted(set(read_only_request)),
            "read_write_request": sorted(set(read_write_request)),
            "denied_read_probe_paths": sorted(set(denied_read_probe_paths)),
            "launcher_path": launcher.as_posix(),
            "launcher_sha256": _sha256_file(launcher),
            "timeout_seconds": float(timeout),
        },
        "worker_policy_sha256",
    )
    if policy != expected:
        raise LifecycleError("ordinary worker launch differs from its exact frozen policy")


def _verify_ordinary_decode_worker_policy(
    bundle: Mapping[str, Any],
    *,
    outdir: Path,
    expected_command: Sequence[str],
    blob_path: Path,
) -> None:
    """Bind one ordinary cold/resource worker to runtime source and one stream."""

    outdir = _lexical(outdir)
    source_root = _ordinary_worker_source_root(outdir)
    read_only = [
        *(_policy_path_for_exact_verification(path) for path in _sandbox_system_read_only()),
        *(
            _policy_path_for_exact_verification(path)
            for path in _worker_source_read_only(source_root)
        ),
        _policy_path_for_exact_verification(outdir / "runtime"),
        _policy_path_for_exact_verification(_lexical(blob_path)),
    ]
    launcher = source_root / "benchmarks/ssp2v_landlock.py"
    _verify_exact_ordinary_worker_policy(
        bundle,
        command=expected_command,
        cwd=source_root / "benchmarks",
        environment=_ordinary_worker_environment(outdir, source_root),
        read_only_request=read_only,
        read_write_request=[],
        denied_read_probe_paths=[],
        launcher_path=launcher,
        timeout=600,
    )


def _candidate_cell_relative(image_id: str, bits: Sequence[int]) -> Path:
    return Path("candidates") / image_id / _tuple_label(bits)


def _decode_worker_source_record() -> dict[str, Any]:
    path = Path(decode_worker.__file__).resolve()
    payload = _read_regular_no_follow(path)
    return {
        "path": decode_worker.WORKER_MODULE_LABEL,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


class _TimedNativeLloydFitter:
    """Observe the actual producing native Lloyd calls without changing their results."""

    def __init__(self, delegate: lloyd.NativeLloydFitter) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    def fit_start(
        self,
        rows: Any,
        spec: lloyd.LloydSpec,
        start_id: int,
        *,
        max_update_sweeps: int = lloyd.MAX_UPDATE_SWEEPS,
    ) -> lloyd.StartResult:
        started = time.perf_counter_ns()
        result = self.delegate.fit_start(
            rows, spec, start_id, max_update_sweeps=max_update_sweeps
        )
        self.calls.append(
            {
                "family_id": spec.family_id,
                "matched_stage_count": spec.matched_stage_count,
                "base_k": spec.base_k,
                "actual_stage_index": spec.actual_stage_index,
                "codebook_size": spec.codebook_size,
                "start_id": start_id,
                "wall_ns": time.perf_counter_ns() - started,
                "status": result.status,
                "update_sweeps": result.update_sweeps,
                "decision_use": False,
                "measurement_scope": "actual_producing_native_fit_start_call",
            }
        )
        return result

    def diagnostics(self) -> dict[str, Any]:
        by_variant: dict[str, list[dict[str, Any]]] = {
            label: [] for label in execute.VARIANT_LABELS
        }
        for call in self.calls:
            family = "flat" if call["family_id"] == 0 else "rvq"
            stages = call["matched_stage_count"]
            base_k = call["base_k"]
            label = (
                f"flat_s{stages}_k{base_k}_l{(2 * stages - 1) * base_k}"
                if family == "flat"
                else f"rvq_s{stages}_k{base_k}"
            )
            by_variant[label].append(call)
        return {
            "clock": "time.perf_counter_ns",
            "scope": "actual producing fitter calls, including every prescribed start/stage",
            "decision_use": False,
            "variants": {
                label: {
                    "calls": calls,
                    "wall_ns_sum": sum(int(call["wall_ns"]) for call in calls),
                }
                for label, calls in by_variant.items()
            },
        }


def _timing_only_reencode_diagnostics(
    source_blob: bytes, result: execute.CellExecution
) -> dict[str, Any]:
    records: dict[str, Any] = {}
    specs = {spec.label: spec for spec in execute.VARIANT_SPECS}
    for label in execute.VARIANT_LABELS:
        fit = result.fits[label]
        spec = specs[label]
        started = time.perf_counter_ns()
        blob = container.encode_ssp2v(
            source_blob,
            spec.descriptor,
            tuple(stage.codebook for stage in fit.stages),
            fit.stage_indices,
            arithmetic_encode=arithmetic.encode,
        )
        elapsed = time.perf_counter_ns() - started
        if blob != result.variant_blobs[label]:
            raise LifecycleError("timing-only canonical re-encode changed produced SSP2V bytes")
        records[label] = {
            "wall_ns": elapsed,
            "blob_sha256": _sha256_bytes(blob),
            "complete_bytes": len(blob),
            "measurement_scope": "separately_labeled_timing_only_canonical_python_reencode",
            "producing_encode_duration_claimed": False,
            "decision_use": False,
        }
    return {
        "clock": "time.perf_counter_ns",
        "variants": records,
        "matched_pairs_descriptive_only": True,
        "decision_use": False,
    }


def produce_candidate_cell_worker(
    *,
    artifact_root: Path,
    source_path: Path,
    captured_comp009_root: Path,
    native_path: Path,
    lloyd_path: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Target-confined producing worker for one complete deterministic cell."""

    config = _read_canonical_object(config_path)
    actual.verify_record(config, "producing_config_sha256")
    if (
        config.get("schema") != "structsplat.comp011.producing-cell-config.v1"
        or config.get("protocol") != PROTOCOL
        or config.get("task_sha256") != TASK_SHA256
    ):
        raise LifecycleError("producing-cell worker config drift")
    producer_policy = config.get("producer_policy")
    if not isinstance(producer_policy, Mapping):
        raise LifecycleError("producing-cell worker launch policy is absent")
    actual.verify_record(producer_policy, "producer_policy_sha256")
    expected_worker_command = _producer_command(
        outdir=artifact_root,
        source_path=source_path,
        captured_comp009_root=captured_comp009_root,
        native_path=native_path,
        lloyd_path=lloyd_path,
        config_path=config_path,
        output_root=output_root,
    )
    policy_environment = producer_policy.get("environment")
    if (
        producer_policy.get("schema") != PRODUCER_POLICY_SCHEMA
        or producer_policy.get("command") != expected_worker_command
        or producer_policy.get("cwd") != Path.cwd().resolve(strict=True).as_posix()
        or not isinstance(policy_environment, Mapping)
        or any(os.environ.get(str(name)) != value for name, value in policy_environment.items())
        or set(os.environ)
        != {
            *policy_environment,
            "STRUCTSPLAT_PARENT_LAUNCH_NONCE",
            "STRUCTSPLAT_SANDBOX_ATTESTATION_SHA256",
        }
    ):
        raise LifecycleError("producing-cell worker launch request drift")
    source_blob = _read_regular_no_follow(source_path)
    if (
        len(source_blob) != config.get("source_bytes")
        or _sha256_bytes(source_blob) != config.get("source_sha256")
    ):
        raise LifecycleError("producing-cell worker source identity drift")
    native_record = config.get("native_record")
    lloyd_record = config.get("lloyd_record")
    authority = config.get("comp009_authority")
    if not all(
        isinstance(value, Mapping)
        for value in (native_record, lloyd_record, authority)
    ):
        raise LifecycleError("producing-cell worker authority config is incomplete")
    _verify_regular_record(native_path, native_record)
    _verify_regular_record(lloyd_path, lloyd_record)
    reproduced, expected, captured_provenance = _captured_comp009_reproduction(
        extracted_source=captured_comp009_root,
        source_blob=source_blob,
        authority=authority,
        native_path=native_path,
        captured_source_manifest_sha256=str(
            config["captured_source_manifest_sha256"]
        ),
        parent_read_only_request=producer_policy.get("read_only_request", ()),
    )

    def reproduce(value: bytes) -> Mapping[str, execute.ReproducedComp009Stream]:
        if value != source_blob:
            raise LifecycleError("producing-cell COMP-009 callback source drift")
        return reproduced

    native = arithmetic.NativeArithmeticCoder(native_path)
    timed_fitter = _TimedNativeLloydFitter(lloyd.NativeLloydFitter(lloyd_path))
    producing_started = time.perf_counter_ns()
    result = execute.execute_cell(
        source_blob,
        native,
        reproduce_comp009=reproduce,
        expected_comp009=expected,
        fitter=timed_fitter,
    )
    producing_wall_ns = time.perf_counter_ns() - producing_started
    execute.verify_record(result.record)
    diagnostics: dict[str, Any] = {
        "producing_total": {
            "clock": "time.perf_counter_ns",
            "wall_ns": producing_wall_ns,
            "scope": (
                "complete sandboxed producing execute_cell call including COMP009 "
                "reproduction callback, fits, encodes, and in-process exact checks"
            ),
            "decision_use": False,
        },
        "producing_fit": timed_fitter.diagnostics(),
        "timing_only_reencode": (
            _timing_only_reencode_diagnostics(source_blob, result)
            if isinstance(result, execute.CellExecution)
            else None
        ),
        "captured_comp009_reproduction": captured_provenance,
        "decision_use": False,
    }
    files = _candidate_blob_files(result)
    output_root.mkdir(parents=False, exist_ok=False)
    for relative, payload in files.items():
        _exclusive_write(output_root / _safe_relative_path(relative), payload)
    record = actual.seal_record(
        {
            "schema": "structsplat.comp011.producing-cell-worker.v1",
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "image_id": config["image_id"],
            "bit_tuple": config["bit_tuple"],
            "producing_config_sha256": config["producing_config_sha256"],
            "producer_policy_sha256": producer_policy["producer_policy_sha256"],
            "execution": result.record,
            "artifacts": _artifact_records_for_files(files),
            "diagnostics": diagnostics,
            "artifact_root_identity": {
                "native": native_path.relative_to(artifact_root).as_posix(),
                "lloyd": lloyd_path.relative_to(artifact_root).as_posix(),
            },
            "target_or_pixel_payloads_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "producing_worker_sha256",
    )
    return record


def _run_producing_cell_worker(
    *,
    outdir: Path,
    image_id: str,
    bits: Sequence[int],
    source_path: Path,
    source_record: Mapping[str, Any],
    captured_comp009_root: Path,
    comp009_authority: Mapping[str, Any],
    captured_source_manifest_sha256: str,
    native_path: Path,
    native_record: Mapping[str, Any],
    lloyd_path: Path,
    lloyd_record: Mapping[str, Any],
    denied_read_probes: Sequence[Path],
) -> tuple[Any, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="comp011-produce-cell-") as temporary:
        workspace = Path(temporary)
        config_path = workspace / "config.json"
        output_root = workspace / "output"
        producer_policy = _producer_launch_policy(
            outdir=outdir,
            source_path=source_path,
            captured_comp009_root=captured_comp009_root,
            native_path=native_path,
            lloyd_path=lloyd_path,
            config_path=config_path,
            output_root=output_root,
            denied_read_probes=denied_read_probes,
        )
        config = actual.seal_record(
            {
                "schema": "structsplat.comp011.producing-cell-config.v1",
                "protocol": PROTOCOL,
                "task_sha256": TASK_SHA256,
                "image_id": image_id,
                "bit_tuple": list(bits),
                "source_bytes": source_record["bytes"],
                "source_sha256": source_record["sha256"],
                "native_record": dict(native_record),
                "lloyd_record": dict(lloyd_record),
                "comp009_authority": dict(comp009_authority),
                "captured_source_manifest_sha256": captured_source_manifest_sha256,
                "producer_policy": producer_policy,
            },
            "producing_config_sha256",
        )
        _exclusive_write(config_path, _canonical_json(config))
        command = list(producer_policy["command"])
        environment = dict(producer_policy["environment"])
        worker, attestation, launch = _run_json_worker(
            command,
            outdir=outdir,
            timeout=24 * 3600,
            read_only=[
                outdir / "runtime",
                source_path,
                captured_comp009_root,
            ],
            read_write=[workspace, *_sandbox_process_read_write()],
            denied_read_probes=denied_read_probes,
            worker_seal_field="producing_worker_sha256",
            environment=environment,
        )
        actual.verify_record(worker, "producing_worker_sha256")
        _verify_producer_launch_policy(
            outdir=outdir,
            image_id=image_id,
            bits=bits,
            config=config,
            worker=worker,
            attestation=attestation,
            launch=launch,
            denied_read_probe_paths=[os.fspath(path) for path in denied_read_probes],
            expected_source_root=_worker_source_root(outdir),
            expected_native_record=native_record,
            expected_lloyd_record=lloyd_record,
            expected_comp009_authority=comp009_authority,
            expected_captured_source_manifest_sha256=captured_source_manifest_sha256,
            expected_denied_read_probe_paths=denied_read_probes,
            expected_parent_sandbox_attestation_sha256=os.environ.get(
                sandbox.SANDBOX_ATTESTATION_ENV
            ),
        )
        worker_diagnostics = worker.get("diagnostics")
        captured_provenance = (
            worker_diagnostics.get("captured_comp009_reproduction")
            if isinstance(worker_diagnostics, Mapping)
            else None
        )
        if not isinstance(captured_provenance, Mapping):
            raise LifecycleError("producing-cell captured COMP-009 evidence is absent")
        _verify_captured_comp009_reproduction_policy(
            captured_provenance,
            producing_config=config,
            producing_policy=producer_policy,
            expected_authority=comp009_authority,
            producer_sandbox_attestation_sha256=str(
                attestation["attestation_sha256"]
            ),
        )
        if (
            worker.get("schema") != "structsplat.comp011.producing-cell-worker.v1"
            or _identity(worker, "producing-cell worker") != (image_id, tuple(bits))
            or worker.get("producing_config_sha256")
            != config["producing_config_sha256"]
            or worker.get("producer_policy_sha256")
            != producer_policy["producer_policy_sha256"]
            or worker.get("target_or_pixel_payloads_opened") is not False
            or worker.get("confirmation_payloads_accessed") is not False
        ):
            raise LifecycleError("producing-cell worker identity/access drift")
        execution_record = worker.get("execution")
        if not isinstance(execution_record, Mapping):
            raise LifecycleError("producing-cell worker execution is absent")
        execute.verify_record(execution_record)
        artifacts = worker.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise LifecycleError("producing-cell worker artifact inventory is absent")
        blobs: dict[str, bytes] = {}
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise LifecycleError("producing-cell worker artifact is malformed")
            relative = _safe_relative_path(
                artifact.get("path"), "producing-cell artifact path"
            )
            path = output_root / relative
            _verify_regular_record(path, artifact)
            blobs[relative.as_posix()] = _read_regular_no_follow(path)
        exact = {
            Path(path).stem: payload
            for path, payload in blobs.items()
            if path.startswith("exact/")
        }
        variants = {
            Path(path).stem: payload
            for path, payload in blobs.items()
            if path.startswith("variants/")
        }
        result = SimpleNamespace(
            record=dict(execution_record),
            baseline_blobs=exact,
            variant_blobs=variants,
        )
        diagnostics = dict(worker["diagnostics"])
        diagnostics["producing_worker"] = {
            "config": config,
            "worker": worker,
            "sandbox_attestation": attestation,
            "launch": launch,
            "denied_read_probe_paths": [
                os.fspath(path) for path in denied_read_probes
            ],
        }
        return result, diagnostics


def _artifact_records_for_files(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {"path": name, "bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for name, payload in sorted(files.items())
    ]


def _candidate_blob_files(result: Any) -> dict[str, bytes]:
    complete = result.record.get("status") == "valid_complete_candidate_set"
    return {
        **{f"exact/{name}.bin": blob for name, blob in result.baseline_blobs.items()},
        **(
            {f"variants/{name}.ssp2v": blob for name, blob in result.variant_blobs.items()}
            if complete
            else {}
        ),
    }


def _prepare_candidate_blob_directory(
    outdir: Path,
    *,
    image_id: str,
    bits: Sequence[int],
    result: Any,
) -> tuple[Path, dict[str, bytes]]:
    final = outdir / _candidate_cell_relative(image_id, bits)
    if final.exists():
        _reject_symlink_components(final, require_regular=False)
        if not final.is_dir():
            raise LifecycleError("candidate-cell destination is not a directory")
    else:
        _mkdir_parents_no_symlink(final.parent)
        try:
            os.mkdir(final, 0o755)
        except FileExistsError:
            _reject_symlink_components(final, require_regular=False)
    for directory in (final, final / "exact", final / "variants"):
        _recover_pending_directory(directory)
    files = _candidate_blob_files(result)
    expected_paths = set(files)
    observed_paths = {
        path.relative_to(final).as_posix()
        for path in final.rglob("*")
        if path.is_file() and path.name != "cell.json"
    }
    if observed_paths - expected_paths:
        raise LifecycleError("candidate directory contains an unexpected immutable file")
    for name, payload in files.items():
        path = final / name
        if path.exists():
            if (
                _read_regular_no_follow(path) != payload
                or os.stat(path, follow_symlinks=False).st_mode & 0o222
            ):
                raise LifecycleError("resumed candidate blob differs from producing bytes")
        else:
            _exclusive_write(path, payload)
    return final, files


def _commit_candidate_cell(
    outdir: Path,
    journal: EventJournal,
    *,
    image_id: str,
    bits: Sequence[int],
    execution_result: Any,
    cold_decodes: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    prior_stage_seal: str,
) -> dict[str, Any]:
    relative = _candidate_cell_relative(image_id, bits)
    cell_label = _artifact_label(outdir, outdir / relative)
    events = EventJournal.verify(journal.root)
    prior_commits = _verify_candidate_commit_receipts(
        outdir, events, prior_stage_seal=prior_stage_seal
    )
    if any(
        committed["record"].get("state") == "terminal_lloyd_failure"
        and label != cell_label
        for label, committed in prior_commits.items()
    ):
        raise LifecycleError("terminal Lloyd candidate forbids any later cell suffix")
    final, files = _prepare_candidate_blob_directory(
        outdir, image_id=image_id, bits=bits, result=execution_result
    )
    cell_path = final / "cell.json"
    state = (
        "complete_candidate_cell"
        if execution_result.record.get("status") == "valid_complete_candidate_set"
        else "terminal_lloyd_failure"
    )
    record = actual.seal_record(
        {
            "schema": "structsplat.comp011.candidate-cell.v1",
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "state": state,
            "relative_path": relative.as_posix(),
            "execution": execution_result.record,
            "artifacts": _artifact_records_for_files(files),
            "cold_decodes": dict(cold_decodes),
            "diagnostics": dict(diagnostics),
            "target_accessed": False,
            "confirmation_payloads_accessed": False,
        },
        "cell_record_sha256",
    )
    payload = _canonical_json(record)
    if cell_path.exists():
        if (
            _read_canonical_object(cell_path) != record
            or os.stat(cell_path, follow_symlinks=False).st_mode & 0o222
        ):
            raise LifecycleError("resumed candidate cell differs from reproduced evidence")
    else:
        _exclusive_write(cell_path, payload)
    committed = prior_commits.get(cell_label)
    if committed is not None:
        if committed["payload"] != payload:
            raise LifecycleError("candidate cell commit receipt differs from immutable cell")
    else:
        if events and events[-1].get("stage") not in {
            "import-streams",
            "fit-encode-cold-decode-dev",
        }:
            raise LifecycleError("candidate cell commit recovery crossed a later stage")
        journal.append(
            stage="fit-encode-cold-decode-dev",
            operation="commit-immutable-cell-directory",
            path_class="artifact_stream",
            purpose="commit target-free fit/encode/cold-decode cell",
            path=cell_label,
            prior_stage_seal=prior_stage_seal,
            artifact_sha256=_sha256_bytes(payload),
            artifact_bytes=len(payload),
        )
        committed = _verify_candidate_commit_receipts(
            outdir,
            EventJournal.verify(journal.root),
            prior_stage_seal=prior_stage_seal,
        ).get(cell_label)
        if committed is None or committed["payload"] != payload:
            raise LifecycleError("candidate cell commit receipt was not durably recorded")
    return _verify_candidate_cell(outdir, record)


def _verify_candidate_commit_receipts(
    outdir: Path,
    events: Sequence[Mapping[str, Any]],
    *,
    prior_stage_seal: str,
) -> dict[str, dict[str, Any]]:
    """Bind every candidate-directory commit to its exact canonical cell record."""

    receipts: dict[str, dict[str, Any]] = {}
    for event in events:
        if not (
            event.get("stage") == "fit-encode-cold-decode-dev"
            and event.get("operation") == "commit-immutable-cell-directory"
        ):
            continue
        label = event.get("path")
        if not isinstance(label, str) or not label.startswith("artifact:"):
            raise LifecycleError("candidate commit receipt path is malformed")
        if label in receipts:
            raise LifecycleError("candidate cell has duplicate journal commits")
        directory = _artifact_path(
            outdir, label.removeprefix("artifact:"), "candidate commit path"
        )
        payload = _read_regular_no_follow(directory / "cell.json")
        record = _read_canonical_object(directory / "cell.json")
        if (
            event.get("path_class") != "artifact_stream"
            or event.get("purpose")
            != "commit target-free fit/encode/cold-decode cell"
            or event.get("prior_stage_seal") != prior_stage_seal
            or event.get("artifact_sha256") != _sha256_bytes(payload)
            or event.get("artifact_bytes") != len(payload)
        ):
            raise LifecycleError("candidate cell commit receipt is invalid")
        receipts[label] = {"event": event, "payload": payload, "record": record}
    return receipts


def _verify_candidate_cell(outdir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    actual.verify_record(record, "cell_record_sha256")
    if (
        record.get("schema") != "structsplat.comp011.candidate-cell.v1"
        or record.get("protocol") != PROTOCOL
        or record.get("task_sha256") != TASK_SHA256
        or record.get("state") not in ("complete_candidate_cell", "terminal_lloyd_failure")
        or record.get("target_accessed") is not False
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("candidate-cell schema/state/access drift")
    execution_record = record.get("execution")
    if not isinstance(execution_record, Mapping):
        raise LifecycleError("candidate-cell execution record is absent")
    execute.verify_record(execution_record)
    expected_state = (
        "complete_candidate_cell"
        if execution_record.get("status") == "valid_complete_candidate_set"
        else "terminal_lloyd_failure"
    )
    if record.get("state") != expected_state:
        raise LifecycleError("candidate-cell state differs from execution state")
    base = _artifact_path(outdir, record.get("relative_path"), "candidate-cell path")
    if base != outdir / _candidate_cell_relative(str(record.get("image_id")), record.get("bit_tuple", [])):
        raise LifecycleError("candidate-cell path differs from frozen identity")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise LifecycleError("candidate-cell artifact inventory is empty")
    expected_paths = {
        f"exact/{name}.bin" for name in execute.EXACT_FORMAT_ORDER
    }
    if expected_state == "complete_candidate_cell":
        expected_paths |= {f"variants/{name}.ssp2v" for name in execute.VARIANT_LABELS}
    artifact_paths = [
        item.get("path") for item in artifacts if isinstance(item, Mapping)
    ]
    if (
        len(artifact_paths) != len(artifacts)
        or artifact_paths != sorted(expected_paths)
    ):
        raise LifecycleError("candidate-cell blob inventory differs from its execution branch")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise LifecycleError("candidate-cell artifact record is malformed")
        path = base / str(artifact.get("path"))
        _verify_regular_record(path, artifact)
        name = Path(str(artifact["path"])).stem
        blob_record = (
            execution_record["exact_formats"][name]
            if str(artifact["path"]).startswith("exact/")
            else execution_record["variants"][name]["stream"]
        )
        if (
            artifact.get("bytes") != blob_record.get("complete_bytes")
            or artifact.get("sha256") != blob_record.get("blob_sha256")
        ):
            raise LifecycleError("candidate blob detached from execution record")
    cold = record.get("cold_decodes")
    if not isinstance(cold, Mapping) or set(cold) != expected_paths:
        raise LifecycleError("candidate-cell fresh cold-decode inventory is incomplete")
    worker_source = _decode_worker_source_record()
    preflight = _read_canonical_object(outdir / "preflight.json")
    actual.verify_record(preflight, "preflight_binding_sha256")
    native_authority = preflight["native"]["arithmetic"]
    expected_source_bindings = {
        "worker_module": {
            "source_label": decode_worker.WORKER_MODULE_LABEL,
            "bytes": worker_source["bytes"],
            "sha256": worker_source["sha256"],
            "identity_verified_before_operation": True,
        },
        "native_library": {
            "artifact_relative_path": f"runtime/native/{native_authority['path']}",
            "bytes": native_authority["bytes"],
            "sha256": native_authority["sha256"],
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
        "cdf_asset": {
            "source_label": "benchmarks/assets/ssp2e_cdf_v1.bin",
            "bytes": comp009_assets.CDF_ASSET_SIZE,
            "sha256": comp009_assets.CDF_ASSET_SHA256,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
    }
    for artifact in artifacts:
        relative = str(artifact["path"])
        blob = _read_regular_no_follow(base / relative)
        name = Path(relative).stem
        expected_symbol, expected_boundary = _cold_decode_expectations(execution_record, name)
        components_sha256 = decode_worker.component_records_sha256(
            decode_worker.stream_component_records(blob)
        )
        cold_bundle = cold[relative]
        if not isinstance(cold_bundle, Mapping) or set(cold_bundle) != {
            "cold_seal",
            "sandbox_attestation",
            "launch",
        }:
            raise LifecycleError("candidate cold-decode launch bundle is malformed")
        cold_record = cold_bundle["cold_seal"]
        if not isinstance(cold_record, Mapping):
            raise LifecycleError("candidate cold-decode seal is absent")
        decode_worker.verify_candidate_cold_seal(
            cold_record,
            expected_blob_sha256=str(artifact["sha256"]),
            expected_blob_bytes=int(artifact["bytes"]),
            expected_symbol_sha256=expected_symbol,
            expected_boundary_state_sha256=expected_boundary,
            expected_stream_components_sha256=components_sha256,
        )
        if (
            cold_record["blob"]["artifact_relative_path"]
            != (base / relative).relative_to(outdir).as_posix()
            or cold_record.get("source_bindings") != expected_source_bindings
        ):
            raise LifecycleError("candidate cold seal source/path binding drift")
        _verify_launch_bundle(
            {
                "attestation": cold_bundle["sandbox_attestation"],
                "launch": cold_bundle["launch"],
            },
            cold_record,
            "cold_seal_sha256",
        )
        native_path = outdir / "runtime/native" / str(native_authority["path"])
        expected_command = _candidate_cold_worker_command(
            artifact_root=outdir,
            blob_relative_path=(base / relative).relative_to(outdir).as_posix(),
            native_relative_path=native_path.relative_to(outdir).as_posix(),
            blob_sha256=str(artifact["sha256"]),
            blob_bytes=int(artifact["bytes"]),
            native_record=native_authority,
            worker_source=worker_source,
            symbol_sha256=expected_symbol,
            boundary_state_sha256=expected_boundary,
            stream_components_sha256=components_sha256,
        )
        _verify_ordinary_decode_worker_policy(
            cold_bundle,
            outdir=outdir,
            expected_command=expected_command,
            blob_path=base / relative,
        )
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("decision_use") is not False:
        raise LifecycleError("candidate-cell nongating diagnostic scope is absent")
    producing = diagnostics.get("producing_worker")
    if not isinstance(producing, Mapping):
        raise LifecycleError("sandboxed producing-cell evidence is absent")
    producing_config = producing.get("config")
    producing_worker = producing.get("worker")
    producing_attestation = producing.get("sandbox_attestation")
    producing_launch = producing.get("launch")
    denied_paths = producing.get("denied_read_probe_paths")
    if not all(
        isinstance(value, Mapping)
        for value in (
            producing_config,
            producing_worker,
            producing_attestation,
            producing_launch,
        )
    ) or not isinstance(denied_paths, list):
        raise LifecycleError("sandboxed producing-cell evidence is malformed")
    authority_copy = _authority_copy(outdir, preflight)
    comp009_by_identity = {
        (str(cell["image_id"]), tuple(cell["bit_tuple"])): cell
        for cell in authority_copy["comp009"]["cells"]
    }
    expected_comp009_authority = comp009_by_identity[
        _identity(record, "candidate cell")
    ]
    expected_upstream_probe = _lexical(
        Path(str(preflight["operational_upstream_roots"]["comp008"]))
    ) / _safe_relative_path(
        authority_copy["comp008"]["targets"][0]["relative_path"],
        "producing-worker denied target probe",
    )
    _verify_producer_launch_policy(
        outdir=outdir,
        image_id=str(record["image_id"]),
        bits=record["bit_tuple"],
        config=producing_config,
        worker=producing_worker,
        attestation=producing_attestation,
        launch=producing_launch,
        denied_read_probe_paths=denied_paths,
        # The persisted producing-worker receipt belongs to the original
        # ordinary stage launch.  Captured replay routes newly launched
        # workers through a relocated source root, but that ambient routing
        # must not rewrite this historical receipt.
        expected_source_root=_ordinary_worker_source_root(outdir),
        expected_native_record=preflight["native"]["arithmetic"],
        expected_lloyd_record=preflight["native"]["lloyd"],
        expected_comp009_authority=expected_comp009_authority,
        expected_captured_source_manifest_sha256=str(
            authority_copy["comp009"]["source_manifest_sha256"]
        ),
        expected_denied_read_probe_paths=[expected_upstream_probe],
    )
    denied_records = producing_attestation.get("denied_read_probes")
    if (
        producing_worker.get("schema")
        != "structsplat.comp011.producing-cell-worker.v1"
        or _identity(producing_worker, "producing-cell worker")
        != _identity(record, "candidate cell")
        or producing_worker.get("execution") != execution_record
        or producing_worker.get("artifacts") != artifacts
        or producing_worker.get("target_or_pixel_payloads_opened") is not False
        or producing_worker.get("confirmation_payloads_accessed") is not False
        or not isinstance(denied_records, list)
        or [item.get("path") for item in denied_records] != denied_paths
        or any(
            item.get("errno") not in (errno.EACCES, errno.EPERM)
            for item in denied_records
        )
        or any(
            path in set(producing_attestation.get(name, []))
            for path in denied_paths
            for name in ("read_only", "read_write")
        )
        or producing_attestation.get("command", [None, None, None, None])[2:4]
        != ["benchmarks.ssp2v_actual_run", "_produce-cell"]
    ):
        raise LifecycleError("sandboxed producing-cell authority/access drift")
    captured = diagnostics.get("captured_comp009_reproduction")
    if not isinstance(captured, Mapping):
        raise LifecycleError("captured COMP-009 reproduction provenance is absent")
    _verify_captured_comp009_reproduction_policy(
        captured,
        producing_config=producing_config,
        producing_policy=producing_config["producer_policy"],
        expected_authority=expected_comp009_authority,
        producer_sandbox_attestation_sha256=str(
            producing_attestation["attestation_sha256"]
        ),
    )
    return dict(record)


def _candidate_global_evidence(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "structsplat.comp011.global-candidate-evidence.v1",
        "task_sha256": TASK_SHA256,
        "cell_execution_sha256": [cell["execution"]["execution_sha256"] for cell in cells],
        "cell_candidate_set_sha256": [cell["execution"]["candidate_set_sha256"] for cell in cells],
        "cell_artifacts": [cell["artifacts"] for cell in cells],
        "cell_cold_decodes": [
            [
                {
                    "path": path,
                    "cold_seal_sha256": cell["cold_decodes"][path]["cold_seal"][
                        "cold_seal_sha256"
                    ],
                    "cold_decode_state_sha256": cell["cold_decodes"][path]["cold_seal"][
                        "cold_decode_state_sha256"
                    ],
                }
                for path in sorted(cell["cold_decodes"])
            ]
            for cell in cells
        ],
        "all_six_exact_formats_per_cell": True,
        "all_eight_variants_per_cell": True,
        "all_fresh_cold_decodes_complete": True,
        "target_accessed": False,
    }


def fit_encode_cold_decode(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    _recover_interrupted_stage_completion(outdir, "fit-encode-cold-decode-dev")
    existing_manifest = outdir / _STAGE_MANIFEST_PATHS["fit-encode-cold-decode-dev"]
    if existing_manifest.exists():
        return verify_candidate_stage(outdir)
    journal, guard, records = _stage_guard(outdir, "fit-encode-cold-decode-dev")
    preflight_record = records["preflight"]
    imported = verify_import_streams(outdir)
    prior_seal = str(imported["import_streams_sha256"])
    events = EventJournal.verify(journal.root)
    committed = _verify_candidate_commit_receipts(
        outdir, events, prior_stage_seal=prior_seal
    )
    identities = _expected_identities()
    expected_labels = [
        _artifact_label(outdir, outdir / _candidate_cell_relative(image_id, bits))
        for image_id, bits in identities
    ]
    if list(committed) != expected_labels[: len(committed)]:
        raise LifecycleError("candidate commit journal is not a frozen-order prefix")
    committed_cells = [
        _verify_candidate_cell(outdir, item["record"])
        for item in committed.values()
    ]
    observed_labels = [
        label
        for label, (image_id, bits) in zip(expected_labels, identities, strict=True)
        if os.path.lexists(
            outdir / _candidate_cell_relative(image_id, bits) / "cell.json"
        )
    ]
    if observed_labels != expected_labels[: len(observed_labels)]:
        raise LifecycleError("candidate cell artifacts are not a frozen-order prefix")
    if len(observed_labels) > len(committed_cells) + 1:
        raise LifecycleError("candidate stage contains an uncommitted cell suffix")
    terminal_cells = [
        cell
        for cell in committed_cells
        if cell.get("state") == "terminal_lloyd_failure"
    ]
    if terminal_cells:
        terminal_event = list(committed.values())[-1]["event"]
        terminal_sequence = int(terminal_event["sequence"])
        suffix = events[terminal_sequence + 1 :]
        manifest_label = f"artifact:{_STAGE_MANIFEST_PATHS['fit-encode-cold-decode-dev']}"
        if (
            len(terminal_cells) != 1
            or committed_cells[-1] != terminal_cells[0]
            or len(observed_labels) != len(committed_cells)
            or len(suffix) > 1
            or any(
                event.get("stage") != "fit-encode-cold-decode-dev"
                or event.get("operation") != "authorize-create-immutable"
                or event.get("path_class") != "artifact_runtime"
                or event.get("purpose")
                != "authorize final fit-encode-cold-decode-dev manifest"
                or event.get("path") != manifest_label
                or event.get("prior_stage_seal") != prior_seal
                for event in suffix
            )
        ):
            raise LifecycleError("terminal Lloyd commit is not the exact terminal prefix")
        return _finalize_stage(
            outdir,
            journal,
            stage="fit-encode-cold-decode-dev",
            schema=CANDIDATE_STAGE_SCHEMA,
            prior_stage_seal=prior_seal,
            body={
                "state": "valid_terminal_lloyd_failure",
                "cells": committed_cells,
                "cell_count": len(committed_cells),
                "candidate_set_sha256": None,
                "target_import_authorized": False,
                "target_or_pixel_payloads_opened": False,
            },
        )
    authority = _authority_copy(outdir, preflight_record)
    authority_by_identity = {
        (str(cell["image_id"]), tuple(cell["bit_tuple"])): cell
        for cell in authority["comp009"]["cells"]
    }
    imported_by_identity = {
        (str(cell["image_id"]), tuple(cell["bit_tuple"])): cell
        for cell in imported["streams"]
    }
    native_path, native_record = _native_arithmetic_record(outdir, preflight_record)
    lloyd_record = preflight_record["native"]["lloyd"]
    lloyd_path = outdir / "runtime/native" / str(lloyd_record["path"])
    _verify_regular_record(lloyd_path, lloyd_record)
    archive = _artifact_path(
        outdir,
        authority["comp009"]["captured_source_archive"]["relative_path"],
        "captured COMP-009 archive path",
    )
    with tempfile.TemporaryDirectory(prefix="comp011-comp009-source-") as temporary:
        extracted = Path(temporary) / "captured"
        safe_extract_source_archive(
            archive,
            extracted,
            expected_manifest=authority["comp009"]["captured_source_manifest"],
        )
        completed_cells: list[dict[str, Any]] = []
        for image_id, bits in _expected_identities():
            final = outdir / _candidate_cell_relative(image_id, bits) / "cell.json"
            if final.exists():
                persisted_cell = _verify_candidate_cell(
                    outdir, _read_canonical_object(final)
                )
                base = _artifact_path(
                    outdir,
                    persisted_cell["relative_path"],
                    "resumed candidate-cell path",
                )
                resumed_blobs = {
                    artifact["path"]: _read_regular_no_follow(
                        base
                        / _safe_relative_path(
                            artifact["path"], "resumed candidate artifact path"
                        )
                    )
                    for artifact in persisted_cell["artifacts"]
                }
                resumed_result = SimpleNamespace(
                    record=persisted_cell["execution"],
                    baseline_blobs={
                        Path(path).stem: blob
                        for path, blob in resumed_blobs.items()
                        if path.startswith("exact/")
                    },
                    variant_blobs={
                        Path(path).stem: blob
                        for path, blob in resumed_blobs.items()
                        if path.startswith("variants/")
                    },
                )
                cell = _commit_candidate_cell(
                    outdir,
                    journal,
                    image_id=image_id,
                    bits=bits,
                    execution_result=resumed_result,
                    cold_decodes=persisted_cell["cold_decodes"],
                    diagnostics=persisted_cell["diagnostics"],
                    prior_stage_seal=prior_seal,
                )
            else:
                stream_record = imported_by_identity[(image_id, bits)]
                stream_path = _artifact_path(
                    outdir, stream_record["relative_path"], "imported-stream path"
                )
                guard.register(stream_path, "artifact_stream")
                _source_blob = guard.read_bytes(
                    stream_path,
                    stage="fit-encode-cold-decode-dev",
                    purpose="fit only from immutable imported SSPL1 copy",
                    prior_stage_seal=prior_seal,
                )
                comp009_authority = authority_by_identity[(image_id, bits)]
                upstream_target_probe = _lexical(
                    Path(
                        str(
                            preflight_record["operational_upstream_roots"]["comp008"]
                        )
                    )
                ) / _safe_relative_path(
                    authority["comp008"]["targets"][0]["relative_path"],
                    "producing-worker denied target probe",
                )
                result, diagnostics = _run_producing_cell_worker(
                    outdir=outdir,
                    image_id=image_id,
                    bits=bits,
                    source_path=stream_path,
                    source_record=stream_record,
                    captured_comp009_root=extracted,
                    comp009_authority=comp009_authority,
                    captured_source_manifest_sha256=str(
                        authority["comp009"]["source_manifest_sha256"]
                    ),
                    native_path=native_path,
                    native_record=native_record,
                    lloyd_path=lloyd_path,
                    lloyd_record=lloyd_record,
                    denied_read_probes=[upstream_target_probe],
                )
                execute.verify_record(result.record)
                candidate_root, files = _prepare_candidate_blob_directory(
                    outdir, image_id=image_id, bits=bits, result=result
                )
                cold_decodes: dict[str, dict[str, Any]] = {}
                for relative_name, blob in files.items():
                    name = Path(relative_name).stem
                    blob_path = candidate_root / relative_name
                    symbol_sha256, boundary_sha256 = _cold_decode_expectations(
                        result.record, name
                    )
                    component_records = decode_worker.stream_component_records(blob)
                    components_sha256 = decode_worker.component_records_sha256(component_records)
                    cold_decodes[relative_name] = _run_cold_decode_sample(
                        outdir=outdir,
                        blob_path=blob_path,
                        blob_sha256=_sha256_bytes(blob),
                        symbol_sha256=symbol_sha256,
                        boundary_state_sha256=boundary_sha256,
                        native_path=native_path,
                        native_record=native_record,
                        stream_components_sha256=components_sha256,
                    )
                cell = _commit_candidate_cell(
                    outdir,
                    journal,
                    image_id=image_id,
                    bits=bits,
                    execution_result=result,
                    cold_decodes=cold_decodes,
                    diagnostics=diagnostics,
                    prior_stage_seal=prior_seal,
                )
            completed_cells.append(cell)
            if cell["state"] == "terminal_lloyd_failure":
                if len(completed_cells) != 1 and any(
                    item["state"] == "terminal_lloyd_failure" for item in completed_cells[:-1]
                ):
                    raise LifecycleError("candidate stage continued after terminal Lloyd failure")
                return _finalize_stage(
                    outdir,
                    journal,
                    stage="fit-encode-cold-decode-dev",
                    schema=CANDIDATE_STAGE_SCHEMA,
                    prior_stage_seal=prior_seal,
                    body={
                        "state": "valid_terminal_lloyd_failure",
                        "cells": completed_cells,
                        "cell_count": len(completed_cells),
                        "candidate_set_sha256": None,
                        "target_import_authorized": False,
                        "target_or_pixel_payloads_opened": False,
                    },
                )
    if len(completed_cells) != 16 or any(
        cell["state"] != "complete_candidate_cell" for cell in completed_cells
    ):
        raise LifecycleError("candidate stage did not complete the exact sixteen-cell grid")
    evidence = _candidate_global_evidence(completed_cells)
    candidate_set_sha256 = _sha256_bytes(_canonical_json(evidence))
    return _finalize_stage(
        outdir,
        journal,
        stage="fit-encode-cold-decode-dev",
        schema=CANDIDATE_STAGE_SCHEMA,
        prior_stage_seal=prior_seal,
        body={
            "state": "complete_candidate_set",
            "cells": completed_cells,
            "cell_count": 16,
            "candidate_evidence": evidence,
            "candidate_set_sha256": candidate_set_sha256,
            "target_import_authorized": True,
            "target_or_pixel_payloads_opened": False,
        },
    )


def verify_candidate_stage(outdir: Path, *, captured_replay: bool = False) -> dict[str, Any]:
    outdir = _lexical(outdir)
    imported = verify_import_streams(outdir, captured_replay=captured_replay)
    record = _verify_stage_base(
        outdir,
        stage="fit-encode-cold-decode-dev",
        schema=CANDIDATE_STAGE_SCHEMA,
        prior_record=imported,
    )
    cells = record.get("cells")
    if not isinstance(cells, list) or not cells:
        raise LifecycleError("candidate-stage cell inventory is absent")
    checked = [_verify_candidate_cell(outdir, cell) for cell in cells]
    receipts = _verify_candidate_commit_receipts(
        outdir,
        EventJournal.verify(outdir / "journal/events"),
        prior_stage_seal=str(record["prior_stage_seal"]),
    )
    expected_receipts = [
        _artifact_label(
            outdir,
            _artifact_path(outdir, cell["relative_path"], "candidate-cell path"),
        )
        for cell in checked
    ]
    if list(receipts) != expected_receipts or any(
        receipts[label]["record"] != cell
        for label, cell in zip(expected_receipts, checked, strict=True)
    ):
        raise LifecycleError("candidate commit journal differs from frozen cell prefix")
    launch_nonces = [
        bundle["launch"]["nonce"]
        for cell in checked
        for bundle in cell["cold_decodes"].values()
    ]
    if len(launch_nonces) != len(set(launch_nonces)):
        raise LifecycleError("candidate cold-decode launch nonce was reused")
    identities = [(cell["image_id"], tuple(cell["bit_tuple"])) for cell in checked]
    if identities != _expected_identities()[: len(checked)]:
        raise LifecycleError("candidate-stage cells are not a frozen-order prefix")
    state = record.get("state")
    if state == "valid_terminal_lloyd_failure":
        if (
            len(checked) != record.get("cell_count")
            or checked[-1]["state"] != "terminal_lloyd_failure"
            or any(cell["state"] != "complete_candidate_cell" for cell in checked[:-1])
            or record.get("candidate_set_sha256") is not None
            or record.get("target_import_authorized") is not False
        ):
            raise LifecycleError("terminal Lloyd branch manifest drift")
    elif state == "complete_candidate_set":
        evidence = _candidate_global_evidence(checked)
        if (
            len(checked) != 16
            or record.get("cell_count") != 16
            or any(cell["state"] != "complete_candidate_cell" for cell in checked)
            or record.get("candidate_evidence") != evidence
            or record.get("candidate_set_sha256") != _sha256_bytes(_canonical_json(evidence))
            or record.get("target_import_authorized") is not True
        ):
            raise LifecycleError("complete global candidate-set seal drift")
    else:
        raise LifecycleError("unknown candidate-stage terminal state")
    if record.get("target_or_pixel_payloads_opened") is not False:
        raise LifecycleError("candidate stage claims early target access")
    return record


def _comp008_rgb_pixel_sha256(pixels: np.ndarray) -> str:
    value = np.asarray(pixels)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise LifecycleError("COMP-008 pixel hash requires HxWx3 RGB uint8")
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _strict_png_samples(payload: bytes, authority: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB":
                raise LifecycleError("prepared target must already be strict RGB PNG")
            forbidden_metadata = {
                "icc_profile",
                "gamma",
                "transparency",
                "srgb",
                "chromaticity",
            } & set(image.info)
            if forbidden_metadata:
                raise LifecycleError(
                    "prepared target exposes forbidden color/alpha metadata: "
                    f"{sorted(forbidden_metadata)}"
                )
            width, height = image.size
            pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    except LifecycleError:
        raise
    except Exception as exc:
        raise LifecycleError(f"strict prepared-target decode failed: {exc}") from exc
    dimensions = [height, width]
    pixel_sha256 = _comp008_rgb_pixel_sha256(pixels)
    if (
        dimensions != authority.get("rgb_dimensions_hw")
        or pixels.shape != (height, width, 3)
        or pixels.dtype != np.uint8
        or not pixels.flags.c_contiguous
        or pixel_sha256 != authority.get("rgb_pixel_sha256")
    ):
        raise LifecycleError("prepared target samples differ from their strict-RGB authority")
    target_tensor = quality.strict_target_tensor(pixels)
    return pixels, {
        "rgb_dimensions_hw": dimensions,
        "rgb_pixel_sha256": pixel_sha256,
        "strict_target_tensor": quality.tensor_record(target_tensor),
        "decode": {
            "container": "PNG",
            "stored_mode": "RGB",
            "conversion_applied": False,
            "icc_gamma_alpha_resize_crop_color_management_applied": False,
        },
    }


def _candidate_transitive_snapshot(
    outdir: Path,
    candidate: Mapping[str, Any],
    *,
    captured_replay: bool = False,
) -> dict[str, Any]:
    imported = verify_import_streams(outdir, captured_replay=captured_replay)
    expected: set[Path] = {
        Path(_STAGE_MANIFEST_PATHS["import-streams"]),
        Path(_STAGE_MANIFEST_PATHS["fit-encode-cold-decode-dev"]),
    }
    expected.update(
        _safe_relative_path(stream["relative_path"], "imported-stream path")
        for stream in imported["streams"]
    )
    for cell in candidate["cells"]:
        base = _safe_relative_path(cell["relative_path"], "candidate-cell path")
        expected.add(base / "cell.json")
        expected.update(
            base / _safe_relative_path(artifact["path"], "candidate artifact path")
            for artifact in cell["artifacts"]
        )
    observed: set[Path] = set()
    for relative_root in (Path("streams"), Path("candidates")):
        root = outdir / relative_root
        if root.exists():
            _reject_symlink_components(root, require_regular=False)
            for path in root.rglob("*"):
                relative = path.relative_to(outdir)
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode):
                    raise LifecycleError("candidate transitive tree contains a symlink")
                if stat.S_ISREG(info.st_mode):
                    observed.add(relative)
                elif not stat.S_ISDIR(info.st_mode):
                    raise LifecycleError("candidate transitive tree contains a special file")
    expected_tree = {path for path in expected if path.parts[0] in {"streams", "candidates"}}
    if observed != expected_tree:
        raise LifecycleError("candidate transitive tree has missing or unexpected artifacts")
    entries: list[dict[str, Any]] = []
    for relative in sorted(expected, key=lambda value: value.as_posix()):
        path = _artifact_path(outdir, relative, "candidate transitive artifact path")
        payload = _read_regular_no_follow(path)
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            }
        )
    entries.sort(key=lambda item: item["path"])
    return {
        "entries": entries,
        "inventory_sha256": _sha256_bytes(_canonical_json(entries)),
        "candidate_set_sha256": candidate["candidate_set_sha256"],
        "import_streams_sha256": imported["import_streams_sha256"],
    }


def import_targets(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    _recover_interrupted_stage_completion(outdir, "import-targets")
    existing = outdir / _STAGE_MANIFEST_PATHS["import-targets"]
    if existing.exists():
        return verify_target_copy(outdir)
    journal, guard, records = _stage_guard(outdir, "import-targets")
    preflight_record = records["preflight"]
    candidate = verify_candidate_stage(outdir)
    prior_seal = str(candidate["fit_stage_sha256"])
    if candidate.get("state") == "valid_terminal_lloyd_failure":
        return _finalize_stage(
            outdir,
            journal,
            stage="import-targets",
            schema=TARGET_COPY_SCHEMA,
            prior_stage_seal=prior_seal,
            body={
                "state": "inapplicable_terminal_lloyd_failure",
                "candidate_set_sha256": None,
                "upstream_targets_sha256": None,
                "targets": [],
                "target_count": 0,
                "candidate_inventory_before": None,
                "candidate_inventory_after": None,
                "target_or_pixel_payloads_opened": False,
                "stream_or_candidate_products_modified": False,
            },
        )
    if candidate.get("state") != "complete_candidate_set":
        raise LifecycleError("unknown candidate terminal state before target import")
    candidate_seal = str(candidate["candidate_set_sha256"])
    authority = _authority_copy(outdir, preflight_record)
    targets = authority["comp008"]["targets"]
    by_image = {str(target["image_id"]): target for target in targets}
    if list(by_image) != list(actual.FROZEN_IMAGES):
        raise LifecycleError("normalized target authorities are not in frozen image order")
    root = Path(str(preflight_record["operational_upstream_roots"]["comp008"]))
    candidate_inventory_before = _candidate_transitive_snapshot(outdir, candidate)
    copies: list[dict[str, Any]] = []
    for image_id in actual.FROZEN_IMAGES:
        target = by_image[image_id]
        relative = Path(str(target["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise LifecycleError("unsafe normalized upstream target path")
        source = root / relative
        png_path = outdir / "targets" / image_id / "target.png"
        rgb_path = outdir / "targets" / image_id / "target.rgb"
        guard.register(source, "upstream_target")
        payload = guard.read_bytes(
            source,
            stage="import-targets",
            purpose="first legal open of sealed prepared target PNG",
            candidate_seal=candidate_seal,
            prior_stage_seal=prior_seal,
        )
        if (
            len(payload) != target["encoded_bytes"]
            or _sha256_bytes(payload) != target["encoded_sha256"]
        ):
            raise LifecycleError("prepared target encoded-byte authority drift")
        pixels, sample_record = _strict_png_samples(payload, target)
        raw = pixels.tobytes(order="C")
        png_artifact = _journaled_immutable(
            journal,
            outdir,
            png_path,
            payload,
            stage="import-targets",
            path_class="artifact_target",
            purpose="persist verbatim prepared target PNG copy",
            prior_stage_seal=prior_seal,
        )
        rgb_artifact = _journaled_immutable(
            journal,
            outdir,
            rgb_path,
            raw,
            stage="import-targets",
            path_class="artifact_target",
            purpose="persist canonical strict-RGB sample bytes",
            prior_stage_seal=prior_seal,
        )
        copies.append(
            {
                "image_id": image_id,
                "authority": dict(target),
                "encoded_copy": png_artifact,
                "rgb_copy": rgb_artifact,
                **sample_record,
            }
        )
    candidate_inventory_after = _candidate_transitive_snapshot(outdir, candidate)
    if candidate_inventory_after != candidate_inventory_before:
        raise LifecycleError("candidate transitive inventory changed during target import")
    return _finalize_stage(
        outdir,
        journal,
        stage="import-targets",
        schema=TARGET_COPY_SCHEMA,
        prior_stage_seal=prior_seal,
        body={
            "state": "complete_target_copy",
            "candidate_set_sha256": candidate_seal,
            "upstream_targets_sha256": authority["comp008"]["targets_sha256"],
            "targets": copies,
            "target_count": len(copies),
            "candidate_inventory_before": candidate_inventory_before,
            "candidate_inventory_after": candidate_inventory_after,
            "target_or_pixel_payloads_opened": True,
            "stream_or_candidate_products_modified": False,
        },
    )


def verify_target_copy(
    outdir: Path,
    *,
    captured_replay: bool = False,
    verify_payloads: bool = True,
) -> dict[str, Any]:
    outdir = _lexical(outdir)
    candidate = verify_candidate_stage(outdir, captured_replay=captured_replay)
    record = _verify_stage_base(
        outdir,
        stage="import-targets",
        schema=TARGET_COPY_SCHEMA,
        prior_record=candidate,
    )
    targets = record.get("targets")
    if candidate.get("state") == "valid_terminal_lloyd_failure":
        if record != actual.seal_record(
            {
                key: value
                for key, value in record.items()
                if key != "target_copy_sha256"
            },
            "target_copy_sha256",
        ):
            raise LifecycleError("terminal target-skip seal does not verify")
        if (
            record.get("state") != "inapplicable_terminal_lloyd_failure"
            or targets != []
            or record.get("target_count") != 0
            or record.get("candidate_set_sha256") is not None
            or record.get("target_or_pixel_payloads_opened") is not False
            or record.get("stream_or_candidate_products_modified") is not False
        ):
            raise LifecycleError("terminal Lloyd target-skip manifest drift")
        return record
    if (
        not isinstance(targets, list)
        or len(targets) != 8
        or record.get("target_count") != 8
        or record.get("state") != "complete_target_copy"
        or record.get("candidate_set_sha256") != candidate.get("candidate_set_sha256")
        or record.get("target_or_pixel_payloads_opened") is not True
        or record.get("stream_or_candidate_products_modified") is not False
        or record.get("candidate_inventory_before")
        != _candidate_transitive_snapshot(
            outdir, candidate, captured_replay=captured_replay
        )
        or record.get("candidate_inventory_after") != record.get("candidate_inventory_before")
    ):
        raise LifecycleError("target-copy inventory/provenance/access drift")
    if [target.get("image_id") for target in targets if isinstance(target, Mapping)] != list(
        actual.FROZEN_IMAGES
    ):
        raise LifecycleError("target-copy inventory differs from frozen image order")
    for target in targets:
        if not isinstance(target, Mapping):
            raise LifecycleError("target-copy record is malformed")
        encoded = target.get("encoded_copy")
        rgb = target.get("rgb_copy")
        authority = target.get("authority")
        if not all(isinstance(value, Mapping) for value in (encoded, rgb, authority)):
            raise LifecycleError("target-copy authority/artifact record is incomplete")
        _artifact_path(outdir, encoded["relative_path"], "target PNG path")
        _artifact_path(outdir, rgb["relative_path"], "target RGB path")
        if verify_payloads:
            encoded_path = _artifact_path(
                outdir, encoded["relative_path"], "target PNG path"
            )
            rgb_path = _artifact_path(outdir, rgb["relative_path"], "target RGB path")
            _verify_regular_record(encoded_path, encoded)
            _verify_regular_record(rgb_path, rgb)
            pixels, expected = _strict_png_samples(
                _read_regular_no_follow(encoded_path), authority
            )
            raw = pixels.tobytes(order="C")
            if (
                _read_regular_no_follow(rgb_path) != raw
                or rgb.get("bytes") != len(raw)
                or rgb.get("sha256") != _sha256_bytes(raw)
                or any(target.get(key) != value for key, value in expected.items())
            ):
                raise LifecycleError("target strict-RGB copy/tensor record drift")
    preflight_record = verify_preflight(outdir, captured_replay=captured_replay)
    authority = _authority_copy(outdir, preflight_record)
    if record.get("upstream_targets_sha256") != authority["comp008"]["targets_sha256"]:
        raise LifecycleError("target copies detached from upstream target authority")
    normalized_by_image = {
        target["image_id"]: target for target in authority["comp008"]["targets"]
    }
    if any(
        target.get("authority") != normalized_by_image.get(target.get("image_id"))
        for target in targets
    ):
        raise LifecycleError("embedded target authority differs from normalized preflight authority")
    return record


def render_arm_worker(
    artifact_root: Path,
    renderer_record: Mapping[str, Any],
    blob_path: Path,
    native_path: Path,
    *,
    expected_blob_sha256: str,
    expected_native_sha256: str,
    expected_native_bytes: int,
    expected_symbol_sha256: str,
    expected_boundary_state_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Fresh CUDA worker: strict-decode one complete stream and render one CPU tensor."""

    blob = _read_regular_no_follow(blob_path)
    if _sha256_bytes(blob) != expected_blob_sha256:
        raise LifecycleError("render worker blob hash drift before decode")
    runtime = decode_worker.load_runtime(
        artifact_root,
        native_path.relative_to(artifact_root).as_posix(),
        expected_native_library_sha256=expected_native_sha256,
        expected_native_library_bytes=expected_native_bytes,
        expected_worker_module_sha256=_decode_worker_source_record()["sha256"],
        expected_worker_module_bytes=_decode_worker_source_record()["bytes"],
    )
    mode, _version = decode_worker._dispatch(blob)  # noqa: SLF001
    header, symbols, symbol_sha256 = decode_worker._strict_decode(mode, blob, runtime)  # noqa: SLF001
    arrays = comp009_execute.boundary_arrays(header, symbols)
    boundary_sha256 = comp009_execute._array_state_digest(arrays)  # noqa: SLF001
    if symbol_sha256 != expected_symbol_sha256 or boundary_sha256 != expected_boundary_state_sha256:
        raise LifecycleError("render worker strict-decode identity drift")
    quality.load_persisted_renderer(
        artifact_root,
        str(renderer_record["relative_path"]),
        str(renderer_record["sha256"]),
        int(renderer_record["bytes"]),
        renderer_record["elf_identity"],
    )
    rendered = quality.render_boundary_once(header, arrays)
    array = rendered.detach().to(device="cpu").contiguous().numpy()
    with io.BytesIO() as handle:
        np.save(handle, array, allow_pickle=False)
        npy_payload = handle.getvalue()
    _exclusive_write(output_path, npy_payload)
    return actual.seal_record(
        {
            "schema": "structsplat.comp011.render-arm-worker.v1",
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "pid": os.getpid(),
            "blob_sha256": expected_blob_sha256,
            "symbol_sha256": symbol_sha256,
            "boundary_state_sha256": boundary_sha256,
            "renderer_relative_path": renderer_record["relative_path"],
            "renderer_sha256": renderer_record["sha256"],
            "native_library": {
                "relative_path": native_path.relative_to(artifact_root).as_posix(),
                "bytes": expected_native_bytes,
                "sha256": expected_native_sha256,
            },
            "decode_worker": _decode_worker_source_record(),
            "render_tensor": quality.tensor_record(rendered),
            "render_npy": {
                "bytes": len(npy_payload),
                "sha256": _sha256_bytes(npy_payload),
            },
            "cuda_synchronized_before_and_after": True,
            "strict_complete_stream_decode": True,
        },
        "render_worker_sha256",
    )


def metric_worker(
    render_path: Path,
    target_path: Path,
    *,
    height: int,
    width: int,
    expected_render_npy_sha256: str,
    expected_render_tensor_sha256: str,
    expected_target_rgb_sha256: str,
    expected_lpips_state_sha256: str,
) -> dict[str, Any]:
    """Fresh single-thread CPU worker for exactly one render/target metric triple."""

    import torch

    render_payload = _read_regular_no_follow(render_path)
    target_payload = _read_regular_no_follow(target_path)
    if (
        _sha256_bytes(render_payload) != expected_render_npy_sha256
        or _sha256_bytes(target_payload) != expected_target_rgb_sha256
    ):
        raise LifecycleError("metric worker input artifact hash drift")
    try:
        render_array = np.load(io.BytesIO(render_payload), allow_pickle=False)
    except Exception as exc:
        raise LifecycleError(f"metric render NPY parse failed: {exc}") from exc
    if (
        render_array.dtype != np.float32
        or render_array.shape != (1, 3, height, width)
        or not render_array.flags.c_contiguous
    ):
        raise LifecycleError("metric render tensor storage is not canonical NCHW float32")
    render = torch.from_numpy(render_array.copy(order="C"))
    if quality.tensor_record(render)["sha256"] != expected_render_tensor_sha256:
        raise LifecycleError("metric render tensor hash drift")
    expected_target_bytes = height * width * 3
    if len(target_payload) != expected_target_bytes:
        raise LifecycleError("metric target raw-byte length drift")
    target_u8 = np.frombuffer(target_payload, dtype=np.uint8).reshape(height, width, 3).copy()
    target = quality.strict_target_tensor(np.ascontiguousarray(target_u8))
    triple, lpips_state = _metric_values_with_warning_audit(render, target)
    if lpips_state != expected_lpips_state_sha256:
        raise LifecycleError("metric worker LPIPS state authority drift")
    return actual.seal_record(
        {
            "schema": "structsplat.comp011.metric-worker.v1",
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "pid": os.getpid(),
            "thread_state": quality.configure_metric_worker(),
            "render_tensor": quality.tensor_record(render),
            "target_tensor": quality.tensor_record(target),
            "lpips_state_sha256": lpips_state,
            "call_order": ["psnr", "ms_ssim", "lpips"],
            "metrics": triple.as_dict(),
            "fresh_single_thread_cpu_worker": True,
        },
        "metric_worker_sha256",
    )


def _metric_values_with_warning_audit(
    render: Any, target: Any
) -> tuple[Any, str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        triple, lpips_state = quality.exact_metric_values(render, target)
    observed = tuple(
        (warning.category.__name__, str(warning.message)) for warning in caught
    )
    if observed != _EXPECTED_METRIC_DEPENDENCY_WARNINGS:
        raise LifecycleError(
            "metric dependency warning surface differs from the frozen fresh-worker authority"
        )
    return triple, lpips_state


def _run_render_metric_task(
    *,
    outdir: Path,
    renderer_record_path: Path,
    renderer_record: Mapping[str, Any],
    blob_path: Path,
    blob_sha256: str,
    symbol_sha256: str,
    boundary_state_sha256: str,
    native_path: Path,
    native_record: Mapping[str, Any],
    target_record: Mapping[str, Any],
    expected_lpips_state_sha256: str,
    worker_source_root: Path | None = None,
    worker_system_read_only: Sequence[Path] | None = None,
    worker_gpu_read_write: Sequence[Path] | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    explicit_source_root = (
        None if worker_source_root is None else _lexical(worker_source_root)
    )
    task_scratch_root = (
        Path(os.environ.get("COMP011_WORKER_SCRATCH", os.fspath(outdir)))
        if scratch_root is None
        else _lexical(scratch_root)
    )
    if explicit_source_root is None:
        source_read_only: list[Path] = []
        worker_launcher = None
        worker_cwd = None
        include_repository_source = True
    else:
        _verify_runtime_source_tree(explicit_source_root)
        source_read_only = _worker_source_read_only(explicit_source_root)
        worker_launcher = explicit_source_root / "benchmarks/ssp2v_landlock.py"
        worker_cwd = explicit_source_root / "benchmarks"
        include_repository_source = False
    captured_launch_options: dict[str, Any] = {}
    if explicit_source_root is not None:
        captured_launch_options = {
            "launcher_path": worker_launcher,
            "cwd": worker_cwd,
            "include_repository_source": include_repository_source,
            "system_read_only": worker_system_read_only,
        }
    gpu_read_write = (
        _sandbox_gpu_read_write()
        if worker_gpu_read_write is None
        else list(worker_gpu_read_write)
    )
    with tempfile.TemporaryDirectory(
        prefix="comp011-render-metric-", dir=task_scratch_root
    ) as temporary:
        render_path = Path(temporary) / "render.npy"
        render_environment = (
            _worker_environment(outdir)
            if explicit_source_root is None
            else _worker_environment(
                outdir, source_root=explicit_source_root
            )
        )
        render_environment["TMPDIR"] = temporary
        render_record, render_sandbox, render_launch = _run_json_worker(
            [
                sys.executable,
                "-m",
                "benchmarks.ssp2v_actual_run",
                "_render-arm",
                "--artifact-root",
                os.fspath(outdir),
                "--renderer-record",
                os.fspath(renderer_record_path),
                "--blob",
                os.fspath(blob_path),
                "--native",
                os.fspath(native_path),
                "--expected-blob-sha256",
                blob_sha256,
                "--expected-native-sha256",
                str(native_record["sha256"]),
                "--expected-native-bytes",
                str(native_record["bytes"]),
                "--expected-symbol-sha256",
                symbol_sha256,
                "--expected-boundary-state-sha256",
                boundary_state_sha256,
                "--output",
                os.fspath(render_path),
            ],
            outdir=outdir,
            timeout=600,
            read_only=[*source_read_only, outdir / "runtime", blob_path],
            read_write=[Path(temporary), *gpu_read_write],
            worker_seal_field="render_worker_sha256",
            environment=render_environment,
            **captured_launch_options,
        )
        actual.verify_record(render_record, "render_worker_sha256")
        if render_record.get("schema") != "structsplat.comp011.render-arm-worker.v1":
            raise LifecycleError("render-arm worker schema drift")
        target_path = _artifact_path(
            outdir, target_record["rgb_copy"]["relative_path"], "target RGB path"
        )
        metric_scratch = Path(temporary) / "metric-scratch"
        metric_scratch.mkdir()
        metric_environment = (
            _worker_environment(outdir)
            if explicit_source_root is None
            else _worker_environment(
                outdir, source_root=explicit_source_root
            )
        )
        metric_environment["TMPDIR"] = os.fspath(metric_scratch)
        metric_record, metric_sandbox, metric_launch = _run_json_worker(
            [
                sys.executable,
                "-m",
                "benchmarks.ssp2v_actual_run",
                "_metric-worker",
                "--render",
                os.fspath(render_path),
                "--target-rgb",
                os.fspath(target_path),
                "--height",
                str(target_record["rgb_dimensions_hw"][0]),
                "--width",
                str(target_record["rgb_dimensions_hw"][1]),
                "--expected-render-npy-sha256",
                str(render_record["render_npy"]["sha256"]),
                "--expected-render-tensor-sha256",
                str(render_record["render_tensor"]["sha256"]),
                "--expected-target-rgb-sha256",
                str(target_record["rgb_copy"]["sha256"]),
                "--expected-lpips-state-sha256",
                expected_lpips_state_sha256,
            ],
            outdir=outdir,
            timeout=600,
            read_only=[
                *source_read_only,
                outdir / "runtime",
                render_path,
                target_path,
            ],
            read_write=[metric_scratch],
            worker_seal_field="metric_worker_sha256",
            environment=metric_environment,
            **captured_launch_options,
        )
        actual.verify_record(metric_record, "metric_worker_sha256")
        if (
            metric_record.get("schema") != "structsplat.comp011.metric-worker.v1"
            or metric_record.get("render_tensor") != render_record.get("render_tensor")
            or metric_record.get("fresh_single_thread_cpu_worker") is not True
        ):
            raise LifecycleError("fresh metric-worker/render binding drift")
        return {
            "render": render_record,
            "metric": metric_record,
            "render_sandbox": render_sandbox,
            "render_launch": render_launch,
            "metric_sandbox": metric_sandbox,
            "metric_launch": metric_launch,
        }


def _verify_ordinary_render_metric_policy(
    task: Mapping[str, Any],
    *,
    outdir: Path,
    blob_path: Path,
    expected_stream: Mapping[str, Any],
    render: Mapping[str, Any],
    target_record: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    expected_lpips_state_sha256: str,
) -> None:
    """Recompute the exact ordinary render and metric capability requests."""

    outdir = _lexical(outdir)
    source_root = _ordinary_worker_source_root(outdir)
    render_launch = task.get("render_launch")
    metric_launch = task.get("metric_launch")
    if not isinstance(render_launch, Mapping) or not isinstance(metric_launch, Mapping):
        raise LifecycleError("ordinary quality launch receipts are absent")
    render_policy = render_launch.get("worker_policy")
    metric_policy = metric_launch.get("worker_policy")
    if not isinstance(render_policy, Mapping) or not isinstance(metric_policy, Mapping):
        raise LifecycleError("ordinary quality persisted worker policies are absent")

    renderer_path = outdir / "runtime/renderer/record.json"
    native = preflight_record["native"]["arithmetic"]
    native_path = outdir / "runtime/native" / str(native["path"])
    render_command = render_policy.get("command")
    if not isinstance(render_command, list) or len(render_command) != 24:
        raise LifecycleError("ordinary render command vector is malformed")
    render_path = Path(str(render_command[-1]))
    if not render_path.is_absolute() or _lexical(render_path) != render_path:
        raise LifecycleError("ordinary render output path is noncanonical")
    render_temp = render_path.parent
    prefix = "comp011-render-metric-"
    if (
        render_path.name != "render.npy"
        or render_temp.parent != outdir
        or not render_temp.name.startswith(prefix)
        or len(render_temp.name) <= len(prefix)
    ):
        raise LifecycleError("ordinary render private workspace grammar drift")
    expected_render_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_render-arm",
        "--artifact-root",
        os.fspath(outdir),
        "--renderer-record",
        os.fspath(renderer_path),
        "--blob",
        os.fspath(_lexical(blob_path)),
        "--native",
        os.fspath(native_path),
        "--expected-blob-sha256",
        str(expected_stream["blob_sha256"]),
        "--expected-native-sha256",
        str(native["sha256"]),
        "--expected-native-bytes",
        str(native["bytes"]),
        "--expected-symbol-sha256",
        str(expected_stream["symbol_sha256"]),
        "--expected-boundary-state-sha256",
        str(expected_stream["boundary_state_sha256"]),
        "--output",
        os.fspath(render_path),
    ]
    render_environment = _ordinary_worker_environment(outdir, source_root)
    render_environment["TMPDIR"] = os.fspath(render_temp)
    common_read_only = [
        *(_policy_path_for_exact_verification(path) for path in _sandbox_system_read_only()),
        *(
            _policy_path_for_exact_verification(path)
            for path in _worker_source_read_only(source_root)
        ),
        _policy_path_for_exact_verification(outdir / "runtime"),
    ]
    render_read_only = [
        *common_read_only,
        _policy_path_for_exact_verification(_lexical(blob_path)),
    ]
    render_read_write = [
        _policy_path_for_exact_verification(render_temp, allow_missing=True),
        *(
            _policy_path_for_exact_verification(path)
            for path in _sandbox_gpu_read_write()
        ),
    ]
    launcher = source_root / "benchmarks/ssp2v_landlock.py"
    _verify_exact_ordinary_worker_policy(
        {"launch": render_launch},
        command=expected_render_command,
        cwd=source_root / "benchmarks",
        environment=render_environment,
        read_only_request=render_read_only,
        read_write_request=render_read_write,
        denied_read_probe_paths=[],
        launcher_path=launcher,
        timeout=600,
    )

    target_path = _artifact_path(
        outdir, target_record["rgb_copy"]["relative_path"], "target RGB path"
    )
    expected_metric_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_metric-worker",
        "--render",
        os.fspath(render_path),
        "--target-rgb",
        os.fspath(target_path),
        "--height",
        str(target_record["rgb_dimensions_hw"][0]),
        "--width",
        str(target_record["rgb_dimensions_hw"][1]),
        "--expected-render-npy-sha256",
        str(render["render_npy"]["sha256"]),
        "--expected-render-tensor-sha256",
        str(render["render_tensor"]["sha256"]),
        "--expected-target-rgb-sha256",
        str(target_record["rgb_copy"]["sha256"]),
        "--expected-lpips-state-sha256",
        expected_lpips_state_sha256,
    ]
    metric_read_only = [
        *common_read_only,
        _policy_path_for_exact_verification(render_path, allow_missing=True),
        _policy_path_for_exact_verification(target_path),
    ]
    metric_scratch = render_temp / "metric-scratch"
    metric_environment = _ordinary_worker_environment(outdir, source_root)
    metric_environment["TMPDIR"] = os.fspath(metric_scratch)
    _verify_exact_ordinary_worker_policy(
        {"launch": metric_launch},
        command=expected_metric_command,
        cwd=source_root / "benchmarks",
        environment=metric_environment,
        read_only_request=metric_read_only,
        read_write_request=[
            _policy_path_for_exact_verification(metric_scratch, allow_missing=True)
        ],
        denied_read_probe_paths=[],
        launcher_path=launcher,
        timeout=600,
    )


def _quality_cell_relative(image_id: str, bits: Sequence[int]) -> Path:
    return Path("quality") / image_id / f"{_tuple_label(bits)}.json"


def _quality_scalar_inputs(
    candidate_cell: Mapping[str, Any], metrics: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    execution_record = candidate_cell["execution"]
    formats = {
        name: {
            "complete_bytes": execution_record["exact_formats"][name]["complete_bytes"],
            "exact": True,
        }
        for name in actual.PRIMARY_ORDER
    }
    variants = []
    for label in actual.VARIANT_ORDER:
        family, stages, base_k = actual.VARIANT_SPECS[label]
        variants.append(
            {
                "label": label,
                "family_rank": family,
                "matched_stage_count": stages,
                "base_k": base_k,
                "complete_bytes": execution_record["variants"][label]["stream"]["complete_bytes"],
                "exact": True,
                "all_lloyd_starts_fixed": True,
                "metrics": [dict(item) for item in metrics[label]],
            }
        )
    return formats, variants


def _quality_metric_envelopes(
    metrics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        arm: {
            metric: {
                "min": min(float(value[metric]) for value in values),
                "max": max(float(value[metric]) for value in values),
            }
            for metric in quality.METRIC_NAMES
        }
        for arm, values in metrics.items()
    }


def quality_select(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    _recover_interrupted_stage_completion(outdir, "quality-select-dev")
    existing = outdir / _STAGE_MANIFEST_PATHS["quality-select-dev"]
    if existing.exists():
        return verify_quality_stage(outdir)
    journal, guard, records = _stage_guard(outdir, "quality-select-dev")
    preflight_record = records["preflight"]
    candidate = verify_candidate_stage(outdir)
    targets = verify_target_copy(outdir)
    prior_seal = str(targets["target_copy_sha256"])
    if targets.get("state") == "inapplicable_terminal_lloyd_failure":
        return _finalize_stage(
            outdir,
            journal,
            stage="quality-select-dev",
            schema=QUALITY_STAGE_SCHEMA,
            prior_stage_seal=prior_seal,
            body={
                "state": "inapplicable_terminal_lloyd_failure",
                "candidate_set_sha256": None,
                "target_copy_sha256": prior_seal,
                "cells": [],
                "cell_count": 0,
                "all_arms_scored_without_early_stopping": False,
                "selected_cell_count": 0,
                "target_or_pixel_payloads_opened": False,
            },
        )
    quality_paths = [
        outdir / _quality_cell_relative(image_id, bits)
        for image_id, bits in _expected_identities()
    ]
    quality_receipts = _verify_stage_cell_receipts(
        outdir,
        EventJournal.verify(journal.root),
        stage="quality-select-dev",
        prior_stage_seal=prior_seal,
        expected_paths=quality_paths,
        purpose="persist complete nine-arm quality cell",
    )
    candidate_seal = str(candidate["candidate_set_sha256"])
    target_seal = str(targets["target_copy_sha256"])
    native_path, native_record = _native_arithmetic_record(outdir, preflight_record)
    renderer_record = preflight_record["renderer"]
    renderer_record_path = outdir / "runtime/renderer/record.json"
    target_by_image = {target["image_id"]: target for target in targets["targets"]}
    candidate_by_identity = {
        (cell["image_id"], tuple(cell["bit_tuple"])): cell for cell in candidate["cells"]
    }
    expected_lpips = str(
        preflight_record["renderer_proof"]["same_worker_metric_repeat"][
            "lpips_state_sha256"
        ]
    )
    quality_cells: list[dict[str, Any]] = []
    for image_id, bits in _expected_identities():
        destination = outdir / _quality_cell_relative(image_id, bits)
        destination_label = _artifact_label(outdir, destination)
        if destination.exists():
            record = _verify_quality_cell(
                outdir,
                _read_canonical_object(destination),
                candidate_by_identity[(image_id, bits)],
                preflight_record=preflight_record,
                target_record=target_by_image[image_id],
            )
            if destination_label not in quality_receipts:
                _journaled_immutable(
                    journal,
                    outdir,
                    destination,
                    _canonical_json(record),
                    stage="quality-select-dev",
                    path_class="artifact_runtime",
                    purpose="persist complete nine-arm quality cell",
                    prior_stage_seal=prior_seal,
                )
            quality_cells.append(record)
            continue
        cell = candidate_by_identity[(image_id, bits)]
        execution_record = cell["execution"]
        q_name = str(execution_record["selected_q"]["name"])
        target = target_by_image[image_id]
        target_path = _artifact_path(
            outdir, target["rgb_copy"]["relative_path"], "target RGB path"
        )
        guard.register(target_path, "artifact_target")
        guard.read_bytes(
            target_path,
            stage="quality-select-dev",
            purpose="verify immutable target copy before quality schedule",
            candidate_seal=candidate_seal,
            target_seal=target_seal,
            prior_stage_seal=prior_seal,
        )
        repetitions: dict[str, list[dict[str, float]]] = {
            "primary": [],
            **{label: [] for label in execute.VARIANT_LABELS},
        }
        task_records: list[dict[str, Any]] = []
        for repetition, arm_order in enumerate(quality.RENDER_SCHEDULE):
            for order_index, arm in enumerate(arm_order):
                name = q_name if arm == "primary" else arm
                relative_blob = (
                    f"exact/{name}.bin"
                    if arm == "primary"
                    else f"variants/{name}.ssp2v"
                )
                artifact = next(item for item in cell["artifacts"] if item["path"] == relative_blob)
                blob_path = _artifact_path(
                    outdir,
                    _safe_relative_path(cell["relative_path"], "candidate-cell path")
                    / _safe_relative_path(relative_blob, "candidate stream path"),
                    "candidate stream path",
                )
                guard.register(blob_path, "artifact_stream")
                guard.read_bytes(
                    blob_path,
                    stage="quality-select-dev",
                    purpose="independently verify selected-Q/variant blob before rendering",
                    prior_stage_seal=prior_seal,
                )
                symbol_sha256, boundary_sha256 = _cold_decode_expectations(execution_record, name)
                task = _run_render_metric_task(
                    outdir=outdir,
                    renderer_record_path=renderer_record_path,
                    renderer_record=renderer_record,
                    blob_path=blob_path,
                    blob_sha256=str(artifact["sha256"]),
                    symbol_sha256=symbol_sha256,
                    boundary_state_sha256=boundary_sha256,
                    native_path=native_path,
                    native_record=native_record,
                    target_record=target,
                    expected_lpips_state_sha256=expected_lpips,
                )
                metrics = dict(task["metric"]["metrics"])
                repetitions[arm].append(metrics)
                task_records.append(
                    {
                        "repetition": repetition,
                        "order_index": order_index,
                        "arm": arm,
                        "stream_name": name,
                        **task,
                    }
                )
        if any(len(values) != 3 for values in repetitions.values()):
            raise LifecycleError("quality schedule did not score every arm exactly three times")
        formats, variants = _quality_scalar_inputs(cell, repetitions)
        selection = actual.select_variant(repetitions["primary"], variants)
        record = actual.seal_record(
            {
                "schema": "structsplat.comp011.quality-cell.v1",
                "protocol": PROTOCOL,
                "task_sha256": TASK_SHA256,
                "image_id": image_id,
                "bit_tuple": list(bits),
                "candidate_cell_sha256": cell["cell_record_sha256"],
                "q_name": q_name,
                "formats": formats,
                "primary_metrics": repetitions["primary"],
                "variants": variants,
                "metric_envelopes": _quality_metric_envelopes(repetitions),
                "selection": selection,
                "render_schedule": [list(order) for order in quality.RENDER_SCHEDULE],
                "tasks": task_records,
                "all_eight_arms_scored": True,
                "fresh_metric_worker_count": len(task_records),
                "target_copy_sha256": target_seal,
                "confirmation_payloads_accessed": False,
            },
            "quality_cell_sha256",
        )
        _journaled_immutable(
            journal,
            outdir,
            destination,
            _canonical_json(record),
            stage="quality-select-dev",
            path_class="artifact_runtime",
            purpose="persist complete nine-arm quality cell",
            prior_stage_seal=prior_seal,
        )
        quality_cells.append(
            _verify_quality_cell(
                outdir,
                record,
                cell,
                preflight_record=preflight_record,
                target_record=target,
            )
        )
    completed_receipts = _verify_stage_cell_receipts(
        outdir,
        EventJournal.verify(journal.root),
        stage="quality-select-dev",
        prior_stage_seal=prior_seal,
        expected_paths=quality_paths,
        purpose="persist complete nine-arm quality cell",
    )
    if len(completed_receipts) != len(quality_paths):
        raise LifecycleError("quality-select-dev cell receipt grid is incomplete")
    return _finalize_stage(
        outdir,
        journal,
        stage="quality-select-dev",
        schema=QUALITY_STAGE_SCHEMA,
        prior_stage_seal=prior_seal,
        body={
            "state": "complete_quality_grid",
            "candidate_set_sha256": candidate_seal,
            "target_copy_sha256": target_seal,
            "cells": quality_cells,
            "cell_count": len(quality_cells),
            "all_arms_scored_without_early_stopping": True,
            "selected_cell_count": sum(
                cell["selection"]["state"] == "selected" for cell in quality_cells
            ),
            "target_or_pixel_payloads_opened": True,
        },
    )


def _verify_quality_cell(
    outdir: Path,
    record: Mapping[str, Any],
    candidate_cell: Mapping[str, Any],
    *,
    preflight_record: Mapping[str, Any],
    target_record: Mapping[str, Any],
) -> dict[str, Any]:
    actual.verify_record(record, "quality_cell_sha256")
    record_identity = _identity(record, "quality cell")
    candidate_identity = _identity(candidate_cell, "candidate cell")
    if (
        record_identity != candidate_identity
        or target_record.get("image_id") != record_identity[0]
    ):
        raise LifecycleError("quality-cell identity is detached from candidate/target identity")
    if (
        record.get("schema") != "structsplat.comp011.quality-cell.v1"
        or record.get("protocol") != PROTOCOL
        or record.get("task_sha256") != TASK_SHA256
        or record.get("candidate_cell_sha256") != candidate_cell.get("cell_record_sha256")
        or record.get("all_eight_arms_scored") is not True
        or record.get("fresh_metric_worker_count") != 27
        or record.get("render_schedule") != [list(order) for order in quality.RENDER_SCHEDULE]
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("quality-cell schema/schedule/provenance drift")
    tasks = record.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 27:
        raise LifecycleError("quality-cell worker task inventory is incomplete")
    expected_tasks = [
        (repetition, order_index, arm)
        for repetition, order in enumerate(quality.RENDER_SCHEDULE)
        for order_index, arm in enumerate(order)
    ]
    derived_metrics: dict[str, list[dict[str, float]]] = {
        "primary": [],
        **{label: [] for label in execute.VARIANT_LABELS},
    }
    metric_pids: list[int] = []
    render_pids: list[int] = []
    launch_nonces: list[str] = []
    execution_record = candidate_cell["execution"]
    q_name = str(execution_record["selected_q"]["name"])
    renderer = preflight_record["renderer"]
    native = preflight_record["native"]["arithmetic"]
    worker_source = _decode_worker_source_record()
    candidate_base = _artifact_path(
        outdir, candidate_cell["relative_path"], "quality candidate-cell path"
    )
    lpips_state = preflight_record["renderer_proof"]["same_worker_metric_repeat"][
        "lpips_state_sha256"
    ]
    for task, (repetition, order_index, arm) in zip(tasks, expected_tasks, strict=True):
        if not isinstance(task, Mapping):
            raise LifecycleError("quality-cell task record is malformed")
        stream_name = q_name if arm == "primary" else arm
        if (
            set(task)
            != {
                "repetition",
                "order_index",
                "arm",
                "stream_name",
                "render",
                "metric",
                "render_sandbox",
                "render_launch",
                "metric_sandbox",
                "metric_launch",
            }
            or task.get("repetition") != repetition
            or task.get("order_index") != order_index
            or task.get("arm") != arm
            or task.get("stream_name") != stream_name
        ):
            raise LifecycleError("quality-cell task vector differs from frozen schedule")
        render = task.get("render")
        metric = task.get("metric")
        if not isinstance(render, Mapping) or not isinstance(metric, Mapping):
            raise LifecycleError("quality-cell render/metric evidence is absent")
        actual.verify_record(render, "render_worker_sha256")
        actual.verify_record(metric, "metric_worker_sha256")
        _verify_launch_bundle(
            {"attestation": task["render_sandbox"], "launch": task["render_launch"]},
            render,
            "render_worker_sha256",
        )
        _verify_launch_bundle(
            {"attestation": task["metric_sandbox"], "launch": task["metric_launch"]},
            metric,
            "metric_worker_sha256",
        )
        expected_stream = _decoded_stream_record(execution_record, stream_name)
        relative_stream = (
            Path("exact") / f"{stream_name}.bin"
            if arm == "primary"
            else Path("variants") / f"{stream_name}.ssp2v"
        )
        _verify_ordinary_render_metric_policy(
            task,
            outdir=outdir,
            blob_path=candidate_base / relative_stream,
            expected_stream=expected_stream,
            render=render,
            target_record=target_record,
            preflight_record=preflight_record,
            expected_lpips_state_sha256=str(lpips_state),
        )
        render_pid = render.get("pid")
        metric_pid = metric.get("pid")
        target_height, target_width = target_record["rgb_dimensions_hw"]
        render_tensor = render.get("render_tensor")
        if (
            render.get("schema") != "structsplat.comp011.render-arm-worker.v1"
            or render.get("protocol") != PROTOCOL
            or render.get("task_sha256") != TASK_SHA256
            or render.get("blob_sha256") != expected_stream["blob_sha256"]
            or render.get("symbol_sha256") != expected_stream["symbol_sha256"]
            or render.get("boundary_state_sha256")
            != expected_stream["boundary_state_sha256"]
            or render.get("renderer_relative_path") != renderer["relative_path"]
            or render.get("renderer_sha256") != renderer["sha256"]
            or render.get("native_library")
            != {
                "relative_path": f"runtime/native/{native['path']}",
                "bytes": native["bytes"],
                "sha256": native["sha256"],
            }
            or render.get("decode_worker") != worker_source
            or render.get("cuda_synchronized_before_and_after") is not True
            or render.get("strict_complete_stream_decode") is not True
            or metric.get("schema") != "structsplat.comp011.metric-worker.v1"
            or metric.get("protocol") != PROTOCOL
            or metric.get("task_sha256") != TASK_SHA256
            or metric.get("render_tensor") != render.get("render_tensor")
            or metric.get("target_tensor") != target_record["strict_target_tensor"]
            or metric.get("lpips_state_sha256") != lpips_state
            or metric.get("call_order") != ["psnr", "ms_ssim", "lpips"]
            or metric.get("fresh_single_thread_cpu_worker") is not True
            or isinstance(render_pid, bool)
            or not isinstance(render_pid, int)
            or render_pid <= 0
            or isinstance(metric_pid, bool)
            or not isinstance(metric_pid, int)
            or metric_pid <= 0
            or not isinstance(render_tensor, Mapping)
            or render_tensor.get("dtype") != "torch.float32"
            or render_tensor.get("shape") != [1, 3, target_height, target_width]
            or render_tensor.get("bytes") != 4 * 3 * target_height * target_width
        ):
            raise LifecycleError("quality worker receipt is detached from frozen authorities")
        thread_state = metric.get("thread_state")
        if (
            not isinstance(thread_state, Mapping)
            or set(thread_state)
            != {
                "torch_num_threads",
                "torch_num_interop_threads",
                "deterministic_algorithms",
                "manual_seed",
                "thread_environment",
            }
            or thread_state.get("torch_num_threads") != 1
            or thread_state.get("torch_num_interop_threads") != 1
            or thread_state.get("deterministic_algorithms") is not True
            or thread_state.get("manual_seed") != 0
            or thread_state.get("thread_environment") != quality.FROZEN_THREAD_ENVIRONMENT
        ):
            raise LifecycleError("quality metric-worker deterministic thread state drift")
        values = metric.get("metrics")
        if not isinstance(values, Mapping) or set(values) != set(quality.METRIC_NAMES) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
            for value in values.values()
        ):
            raise LifecycleError("quality metric-worker emitted malformed/nonfinite metrics")
        derived_metrics[arm].append({name: float(values[name]) for name in quality.METRIC_NAMES})
        metric_pids.append(metric_pid)
        render_pids.append(render_pid)
        launch_nonces.extend(
            (str(task["render_launch"]["nonce"]), str(task["metric_launch"]["nonce"]))
        )
    if len(set(metric_pids)) != 27 or len(set(render_pids)) != 27:
        raise LifecycleError("quality arms/metrics did not run in one fresh process per task")
    if len(launch_nonces) != 54 or len(set(launch_nonces)) != 54:
        raise LifecycleError("quality worker launch nonce was reused")
    formats, variants = _quality_scalar_inputs(candidate_cell, derived_metrics)
    if (
        record.get("q_name") != q_name
        or record.get("formats") != formats
        or record.get("primary_metrics") != derived_metrics["primary"]
        or record.get("variants") != variants
        or record.get("metric_envelopes") != _quality_metric_envelopes(derived_metrics)
    ):
        raise LifecycleError("quality scalar metrics are detached from exact worker receipts")
    selection = actual.select_variant(derived_metrics["primary"], variants)
    if record.get("selection") != selection:
        raise LifecycleError("quality-cell selection does not recompute from worker receipts")
    return dict(record)


def verify_quality_stage(
    outdir: Path,
    *,
    captured_replay: bool = False,
    verify_target_payloads: bool = True,
) -> dict[str, Any]:
    outdir = _lexical(outdir)
    targets = verify_target_copy(
        outdir,
        captured_replay=captured_replay,
        verify_payloads=verify_target_payloads,
    )
    candidate = verify_candidate_stage(outdir, captured_replay=captured_replay)
    preflight_record = verify_preflight(outdir, captured_replay=captured_replay)
    target_by_image = {target["image_id"]: target for target in targets.get("targets", [])}
    record = _verify_stage_base(
        outdir,
        stage="quality-select-dev",
        schema=QUALITY_STAGE_SCHEMA,
        prior_record=targets,
    )
    cells = record.get("cells")
    if targets.get("state") == "inapplicable_terminal_lloyd_failure":
        if (
            record.get("state") != "inapplicable_terminal_lloyd_failure"
            or cells != []
            or record.get("cell_count") != 0
            or record.get("candidate_set_sha256") is not None
            or record.get("target_copy_sha256") != targets.get("target_copy_sha256")
            or record.get("all_arms_scored_without_early_stopping") is not False
            or record.get("selected_cell_count") != 0
            or record.get("target_or_pixel_payloads_opened") is not False
        ):
            raise LifecycleError("terminal Lloyd quality-skip manifest drift")
        return record
    if (
        not isinstance(cells, list)
        or len(cells) != 16
        or record.get("state") != "complete_quality_grid"
        or record.get("cell_count") != 16
        or record.get("candidate_set_sha256") != candidate.get("candidate_set_sha256")
        or record.get("target_copy_sha256") != targets.get("target_copy_sha256")
        or record.get("all_arms_scored_without_early_stopping") is not True
        or record.get("target_or_pixel_payloads_opened") is not True
    ):
        raise LifecycleError("quality-stage inventory/provenance/access drift")
    cell_identities = [_identity(cell, "quality cell") for cell in cells]
    candidate_identities = [
        _identity(candidate_cell, "candidate cell") for candidate_cell in candidate["cells"]
    ]
    if cell_identities != _expected_identities() or cell_identities != candidate_identities:
        raise LifecycleError("quality-stage identities differ from the frozen candidate grid")
    checked = []
    for cell, candidate_cell in zip(cells, candidate["cells"], strict=True):
        image_id, _bits = _identity(cell, "quality cell")
        target_record = target_by_image.get(image_id)
        if target_record is None:
            raise LifecycleError("quality cell has no frozen target copy")
        checked.append(
            _verify_quality_cell(
                outdir,
                cell,
                candidate_cell,
                preflight_record=preflight_record,
                target_record=target_record,
            )
        )
    for cell in checked:
        path = outdir / _quality_cell_relative(cell["image_id"], cell["bit_tuple"])
        if _read_canonical_object(path) != cell:
            raise LifecycleError("quality-stage cell differs from immutable cell file")
    quality_paths = [
        outdir / _quality_cell_relative(cell["image_id"], cell["bit_tuple"])
        for cell in checked
    ]
    receipts = _verify_stage_cell_receipts(
        outdir,
        EventJournal.verify(outdir / "journal/events"),
        stage="quality-select-dev",
        prior_stage_seal=str(targets["target_copy_sha256"]),
        expected_paths=quality_paths,
        purpose="persist complete nine-arm quality cell",
    )
    if len(receipts) != len(quality_paths):
        raise LifecycleError("quality-stage cell receipt grid is incomplete")
    if record.get("selected_cell_count") != sum(
        cell["selection"]["state"] == "selected" for cell in checked
    ):
        raise LifecycleError("quality-stage selection count drift")
    return record


def _resource_worker_command(
    *,
    artifact_root: Path,
    blob_relative_path: str,
    native_relative_path: str,
    stream_record: Mapping[str, Any],
    native_record: Mapping[str, Any],
    worker_source: Mapping[str, Any],
    measurement_session_sha256: str,
    launch: decode_worker.LaunchSpec,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_decode_worker",
        "resource",
        "--artifact-root",
        os.fspath(_lexical(artifact_root)),
        "--blob",
        _safe_relative_path(blob_relative_path, "resource worker blob path").as_posix(),
        "--native-library",
        _safe_relative_path(native_relative_path, "resource worker native path").as_posix(),
        "--expected-blob-sha256",
        str(stream_record["blob_sha256"]),
        "--expected-blob-bytes",
        str(stream_record["blob_bytes"]),
        "--expected-native-library-sha256",
        str(native_record["sha256"]),
        "--expected-native-library-bytes",
        str(native_record["bytes"]),
        "--expected-worker-module-sha256",
        str(worker_source["sha256"]),
        "--expected-worker-module-bytes",
        str(worker_source["bytes"]),
        "--expected-symbol-sha256",
        str(stream_record["symbol_sha256"]),
        "--expected-boundary-state-sha256",
        str(stream_record["boundary_state_sha256"]),
        "--measurement-session-sha256",
        measurement_session_sha256,
        "--schedule-kind",
        launch.schedule_kind,
        "--launch-index",
        str(launch.launch_index),
        "--phase",
        launch.phase,
        "--arm",
        launch.arm,
    ]
    if launch.repetition is not None:
        command.extend(("--repetition", str(launch.repetition)))
    return command


def _run_decode_schedule_samples(
    *,
    outdir: Path,
    schedule: Sequence[decode_worker.LaunchSpec],
    arm_paths: Mapping[str, Path],
    arm_records: Mapping[str, Mapping[str, Any]],
    native_path: Path,
    native_record: Mapping[str, Any],
    session_sha256: str,
    worker_source_root: Path | None = None,
    worker_system_read_only: Sequence[Path] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if {launch.arm for launch in schedule} != set(arm_paths) or set(arm_paths) != set(arm_records):
        raise LifecycleError("decode schedule arm bindings are incomplete")
    samples: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    worker_source = _decode_worker_source_record()
    explicit_source_root = (
        None if worker_source_root is None else _lexical(worker_source_root)
    )
    if explicit_source_root is None:
        source_read_only: list[Path] = []
        worker_launcher = None
        worker_cwd = None
        include_repository_source = True
    else:
        _verify_runtime_source_tree(explicit_source_root)
        source_read_only = _worker_source_read_only(explicit_source_root)
        worker_launcher = explicit_source_root / "benchmarks/ssp2v_landlock.py"
        worker_cwd = explicit_source_root / "benchmarks"
        include_repository_source = False
    captured_launch_options: dict[str, Any] = {}
    if explicit_source_root is not None:
        captured_launch_options = {
            "launcher_path": worker_launcher,
            "cwd": worker_cwd,
            "environment": _worker_environment(
                outdir, source_root=explicit_source_root
            ),
            "include_repository_source": include_repository_source,
            "system_read_only": worker_system_read_only,
        }
    for launch in schedule:
        record = arm_records[launch.arm]
        command = _resource_worker_command(
            artifact_root=outdir,
            blob_relative_path=arm_paths[launch.arm].relative_to(outdir).as_posix(),
            native_relative_path=native_path.relative_to(outdir).as_posix(),
            stream_record=record,
            native_record=native_record,
            worker_source=worker_source,
            measurement_session_sha256=session_sha256,
            launch=launch,
        )
        sample, attestation, launch_receipt = _run_json_worker(
            command,
            outdir=outdir,
            timeout=600,
            read_only=[
                *source_read_only,
                outdir / "runtime",
                arm_paths[launch.arm],
            ],
            worker_seal_field="canonical_worker_sha256",
            **captured_launch_options,
        )
        samples.append(sample)
        launches.append({"attestation": attestation, "launch": launch_receipt})
    return samples, launches


def _run_decode_schedule(
    *,
    outdir: Path,
    schedule: Sequence[decode_worker.LaunchSpec],
    arm_paths: Mapping[str, Path],
    arm_records: Mapping[str, Mapping[str, Any]],
    native_path: Path,
    native_record: Mapping[str, Any],
    session_sha256: str,
    image_index: int,
    tuple_index: int,
    worker_source_root: Path | None = None,
    worker_system_read_only: Sequence[Path] | None = None,
) -> dict[str, Any]:
    samples, launches = _run_decode_schedule_samples(
        outdir=outdir,
        schedule=schedule,
        arm_paths=arm_paths,
        arm_records=arm_records,
        native_path=native_path,
        native_record=native_record,
        session_sha256=session_sha256,
        worker_source_root=worker_source_root,
        worker_system_read_only=worker_system_read_only,
    )
    summary = decode_worker.summarize_schedule(
        samples,
        schedule,
        image_index=image_index,
        tuple_index=tuple_index,
        measurement_session_sha256=session_sha256,
        expected_blob_sha256_by_arm={
            arm: str(record["blob_sha256"]) for arm, record in arm_records.items()
        },
    )
    return {
        "schedule": [launch.to_dict() for launch in schedule],
        "samples": samples,
        "launches": launches,
        "summary": summary,
        "fresh_processes": True,
    }


def _benchmark_cell_relative(image_id: str, bits: Sequence[int]) -> Path:
    return Path("benchmark") / image_id / f"{_tuple_label(bits)}.json"


def _measurement_session_sha256(
    *,
    kind: str,
    image_id: str,
    bits: Sequence[int],
    quality_stage_sha256: str,
    candidate: str | None = None,
    q_name: str | None = None,
    replay_phase_sha256: str | None = None,
) -> str:
    if image_id not in actual.FROZEN_IMAGES or tuple(bits) not in actual.FROZEN_TUPLES:
        raise LifecycleError("resource measurement session has an unfrozen cell identity")
    _require_sha256(quality_stage_sha256, "resource quality-stage seal")
    value: dict[str, Any] = {
        "stage": "captured-replay" if replay_phase_sha256 is not None else "benchmark-dev",
        "kind": kind,
        "image_id": image_id,
        "bit_tuple": list(bits),
        "quality_stage_sha256": quality_stage_sha256,
    }
    if replay_phase_sha256 is not None:
        value["replay_phase_sha256"] = _require_sha256(
            replay_phase_sha256, "replay phase seal"
        )
    if kind == "primary":
        if candidate not in actual.VARIANT_ORDER or q_name not in actual.PRIMARY_ORDER:
            raise LifecycleError("primary measurement session has invalid arm identities")
        value.update({"candidate": candidate, "q": q_name})
    elif kind == "secondary":
        if candidate is not None or q_name is not None:
            raise LifecycleError("secondary measurement session contains primary arm identities")
    else:
        raise LifecycleError("unknown resource measurement-session kind")
    return _sha256_bytes(_canonical_json(value))


def _decoded_stream_record(execution_record: Mapping[str, Any], name: str) -> dict[str, Any]:
    if name in execute.EXACT_FORMAT_ORDER:
        stream = execution_record["exact_formats"][name]
        boundary = (
            stream["boundary"]["state_sha256"]
            if name in ("sspl1", "ssp2z")
            else stream["boundary_state_sha256"]
        )
    else:
        stream = execution_record["variants"][name]["stream"]
        boundary = stream["boundary"]["state_sha256"]
    return {
        "blob_bytes": stream["complete_bytes"],
        "blob_sha256": stream["blob_sha256"],
        "symbol_sha256": stream["symbol_sha256"],
        "boundary_state_sha256": boundary,
    }


def _actual_row(
    quality_cell: Mapping[str, Any],
    primary_resource: Mapping[str, Any] | None,
    secondary_resource: Mapping[str, Any],
) -> dict[str, Any]:
    return actual.seal_record(
        {
            "schema": actual.ROW_SCHEMA,
            "image_id": quality_cell["image_id"],
            "bit_tuple": quality_cell["bit_tuple"],
            "validity": "valid",
            "invalid_reason": None,
            "method_state": "complete",
            "lloyd_all_fixed": True,
            "formats": quality_cell["formats"],
            "primary_metrics": quality_cell["primary_metrics"],
            "variants": quality_cell["variants"],
            "primary_resource": dict(primary_resource) if primary_resource is not None else None,
            "secondary_resource": dict(secondary_resource),
        }
    )


def _resource_records(
    quality_cell: Mapping[str, Any],
    primary: Mapping[str, Any] | None,
    secondary: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    primary_record = None
    if primary is not None:
        summaries = primary["summary"]["summaries"]
        primary_record = {
            "candidate_label": quality_cell["selection"]["selected_label"],
            "q_name": quality_cell["q_name"],
            "candidate_median_ns": summaries["candidate"]["median_ns"],
            "q_median_ns": summaries["q"]["median_ns"],
            "candidate_peak_rss_bytes": summaries["candidate"]["maximum_peak_rss_bytes"],
            "q_peak_rss_bytes": summaries["q"]["maximum_peak_rss_bytes"],
            "fresh_comp011": True,
            "schedule_exact": True,
        }
    secondary_summaries = secondary["summary"]["summaries"]
    secondary_record = {
        "ssp2l_median_ns": secondary_summaries["ssp2l"]["median_ns"],
        "sspl1_median_ns": secondary_summaries["sspl1"]["median_ns"],
        "fresh_comp011": True,
        "schedule_exact": True,
    }
    return primary_record, secondary_record


def benchmark_development(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    _recover_interrupted_stage_completion(outdir, "benchmark-dev")
    existing = outdir / _STAGE_MANIFEST_PATHS["benchmark-dev"]
    if existing.exists():
        return verify_benchmark_stage(outdir)
    journal, guard, records = _stage_guard(outdir, "benchmark-dev")
    preflight_record = records["preflight"]
    candidate = verify_candidate_stage(outdir)
    quality_record = verify_quality_stage(outdir, verify_target_payloads=False)
    prior_seal = str(quality_record["quality_stage_sha256"])
    if quality_record.get("state") == "inapplicable_terminal_lloyd_failure":
        failure_cell = candidate["cells"][-1]
        early_row = actual.seal_record(
            {
                "schema": actual.ROW_SCHEMA,
                "image_id": failure_cell["image_id"],
                "bit_tuple": failure_cell["bit_tuple"],
                "validity": "valid",
                "invalid_reason": None,
                "method_state": "lloyd_failure",
                "lloyd_all_fixed": False,
                "formats": None,
                "primary_metrics": None,
                "variants": None,
                "primary_resource": None,
                "secondary_resource": None,
            }
        )
        rows_artifact = _journaled_immutable(
            journal,
            outdir,
            outdir / "actual_rows.jsonl",
            _canonical_json(early_row),
            stage="benchmark-dev",
            path_class="artifact_runtime",
            purpose="persist terminal Lloyd method-failure analysis row",
            prior_stage_seal=prior_seal,
        )
        return _finalize_stage(
            outdir,
            journal,
            stage="benchmark-dev",
            schema=BENCHMARK_STAGE_SCHEMA,
            prior_stage_seal=prior_seal,
            body={
                "state": "inapplicable_terminal_lloyd_failure",
                "quality_stage_sha256": prior_seal,
                "cells": [],
                "cell_count": 0,
                "actual_rows": rows_artifact,
                "actual_row_sha256": [early_row["record_sha256"]],
                "primary_schedules_applicable": False,
                "fresh_primary_process_count": 0,
                "fresh_secondary_process_count": 0,
                "old_timing_samples_reused": False,
                "target_or_pixel_payloads_opened": False,
            },
        )
    invalid_quality = next(
        (
            cell
            for cell in quality_record["cells"]
            if cell["selection"]["state"] == "invalid"
        ),
        None,
    )
    if invalid_quality is not None:
        invalid_row = actual.seal_record(
            {
                "schema": actual.ROW_SCHEMA,
                "image_id": invalid_quality["image_id"],
                "bit_tuple": invalid_quality["bit_tuple"],
                "validity": "invalid",
                "invalid_reason": str(invalid_quality["selection"]["invalid_reason"]),
                "method_state": "invalid",
                "lloyd_all_fixed": None,
                "formats": None,
                "primary_metrics": None,
                "variants": None,
                "primary_resource": None,
                "secondary_resource": None,
            }
        )
        rows_artifact = _journaled_immutable(
            journal,
            outdir,
            outdir / "actual_rows.jsonl",
            _canonical_json(invalid_row),
            stage="benchmark-dev",
            path_class="artifact_runtime",
            purpose="persist quality-invalid no-decision analysis row",
            prior_stage_seal=prior_seal,
        )
        return _finalize_stage(
            outdir,
            journal,
            stage="benchmark-dev",
            schema=BENCHMARK_STAGE_SCHEMA,
            prior_stage_seal=prior_seal,
            body={
                "state": "inapplicable_invalid_quality",
                "quality_stage_sha256": prior_seal,
                "cells": [],
                "cell_count": 0,
                "actual_rows": rows_artifact,
                "actual_row_sha256": [invalid_row["record_sha256"]],
                "primary_schedules_applicable": False,
                "fresh_primary_process_count": 0,
                "fresh_secondary_process_count": 0,
                "old_timing_samples_reused": False,
                "target_or_pixel_payloads_opened": False,
            },
        )
    benchmark_paths = [
        outdir
        / _benchmark_cell_relative(
            str(quality_cell["image_id"]), quality_cell["bit_tuple"]
        )
        for quality_cell in quality_record["cells"]
    ]
    benchmark_receipts = _verify_stage_cell_receipts(
        outdir,
        EventJournal.verify(journal.root),
        stage="benchmark-dev",
        prior_stage_seal=prior_seal,
        expected_paths=benchmark_paths,
        purpose="persist unified primary/secondary resource schedules",
    )
    native_path, native_record = _native_arithmetic_record(outdir, preflight_record)
    all_primary = all(
        cell["selection"]["state"] == "selected" for cell in quality_record["cells"]
    )
    benchmark_cells: list[dict[str, Any]] = []
    for cell_index, (candidate_cell, quality_cell) in enumerate(
        zip(candidate["cells"], quality_record["cells"], strict=True)
    ):
        image_id = quality_cell["image_id"]
        bits = tuple(quality_cell["bit_tuple"])
        destination = outdir / _benchmark_cell_relative(image_id, bits)
        destination_label = _artifact_label(outdir, destination)
        if destination.exists():
            checked = _verify_benchmark_cell(
                outdir,
                _read_canonical_object(destination),
                candidate_cell,
                quality_cell,
                all_primary=all_primary,
                preflight_record=preflight_record,
                quality_stage_sha256=prior_seal,
            )
            if destination_label not in benchmark_receipts:
                _journaled_immutable(
                    journal,
                    outdir,
                    destination,
                    _canonical_json(checked),
                    stage="benchmark-dev",
                    path_class="artifact_runtime",
                    purpose="persist unified primary/secondary resource schedules",
                    prior_stage_seal=prior_seal,
                )
            benchmark_cells.append(checked)
            continue
        image_index = actual.FROZEN_IMAGES.index(image_id)
        tuple_index = actual.FROZEN_TUPLES.index(bits)
        execution_record = candidate_cell["execution"]
        base = _artifact_path(
            outdir, candidate_cell["relative_path"], "candidate-cell path"
        )
        sspl1_path = base / "exact/sspl1.bin"
        ssp2l_path = base / "exact/ssp2l.bin"
        for path in (sspl1_path, ssp2l_path):
            guard.register(path, "artifact_stream")
            guard.read_bytes(
                path,
                stage="benchmark-dev",
                purpose="verify secondary schedule immutable stream",
                prior_stage_seal=prior_seal,
            )
        secondary_session = _measurement_session_sha256(
            kind="secondary",
            image_id=image_id,
            bits=bits,
            quality_stage_sha256=prior_seal,
        )
        secondary = _run_decode_schedule(
            outdir=outdir,
            schedule=decode_worker.secondary_schedule(image_index, tuple_index),
            arm_paths={"sspl1": sspl1_path, "ssp2l": ssp2l_path},
            arm_records={
                "sspl1": _decoded_stream_record(execution_record, "sspl1"),
                "ssp2l": _decoded_stream_record(execution_record, "ssp2l"),
            },
            native_path=native_path,
            native_record=native_record,
            session_sha256=secondary_session,
            image_index=image_index,
            tuple_index=tuple_index,
        )
        primary = None
        if all_primary:
            selected = str(quality_cell["selection"]["selected_label"])
            q_name = str(quality_cell["q_name"])
            candidate_path = base / f"variants/{selected}.ssp2v"
            q_path = base / f"exact/{q_name}.bin"
            for path in (candidate_path, q_path):
                guard.register(path, "artifact_stream")
                guard.read_bytes(
                    path,
                    stage="benchmark-dev",
                    purpose="verify primary schedule immutable stream",
                    prior_stage_seal=prior_seal,
                )
            primary_session = _measurement_session_sha256(
                kind="primary",
                image_id=image_id,
                bits=bits,
                quality_stage_sha256=prior_seal,
                candidate=selected,
                q_name=q_name,
            )
            primary = _run_decode_schedule(
                outdir=outdir,
                schedule=decode_worker.primary_schedule(image_index, tuple_index),
                arm_paths={"candidate": candidate_path, "q": q_path},
                arm_records={
                    "candidate": _decoded_stream_record(execution_record, selected),
                    "q": _decoded_stream_record(execution_record, q_name),
                },
                native_path=native_path,
                native_record=native_record,
                session_sha256=primary_session,
                image_index=image_index,
                tuple_index=tuple_index,
            )
        primary_resource, secondary_resource = _resource_records(
            quality_cell, primary, secondary
        )
        row = _actual_row(quality_cell, primary_resource, secondary_resource)
        record = actual.seal_record(
            {
                "schema": "structsplat.comp011.benchmark-cell.v1",
                "protocol": PROTOCOL,
                "task_sha256": TASK_SHA256,
                "image_id": image_id,
                "bit_tuple": list(bits),
                "quality_cell_sha256": quality_cell["quality_cell_sha256"],
                "primary_applicable": all_primary,
                "primary": primary,
                "secondary": secondary,
                "actual_row": row,
                "old_timing_samples_reused": False,
                "confirmation_payloads_accessed": False,
            },
            "benchmark_cell_sha256",
        )
        _journaled_immutable(
            journal,
            outdir,
            destination,
            _canonical_json(record),
            stage="benchmark-dev",
            path_class="artifact_runtime",
            purpose="persist unified primary/secondary resource schedules",
            prior_stage_seal=prior_seal,
        )
        benchmark_cells.append(
            _verify_benchmark_cell(
                outdir,
                record,
                candidate_cell,
                quality_cell,
                all_primary=all_primary,
                preflight_record=preflight_record,
                quality_stage_sha256=prior_seal,
            )
        )
    completed_receipts = _verify_stage_cell_receipts(
        outdir,
        EventJournal.verify(journal.root),
        stage="benchmark-dev",
        prior_stage_seal=prior_seal,
        expected_paths=benchmark_paths,
        purpose="persist unified primary/secondary resource schedules",
    )
    if len(completed_receipts) != len(benchmark_paths):
        raise LifecycleError("benchmark-dev cell receipt grid is incomplete")
    rows = [cell["actual_row"] for cell in benchmark_cells]
    rows_payload = b"".join(_canonical_json(row) for row in rows)
    rows_artifact = _journaled_immutable(
        journal,
        outdir,
        outdir / "actual_rows.jsonl",
        rows_payload,
        stage="benchmark-dev",
        path_class="artifact_runtime",
        purpose="persist exact scalar analysis rows",
        prior_stage_seal=prior_seal,
    )
    return _finalize_stage(
        outdir,
        journal,
        stage="benchmark-dev",
        schema=BENCHMARK_STAGE_SCHEMA,
        prior_stage_seal=prior_seal,
        body={
            "state": "complete_resource_grid",
            "quality_stage_sha256": prior_seal,
            "cells": benchmark_cells,
            "cell_count": len(benchmark_cells),
            "actual_rows": rows_artifact,
            "actual_row_sha256": [row["record_sha256"] for row in rows],
            "primary_schedules_applicable": all_primary,
            "fresh_primary_process_count": 320 if all_primary else 0,
            "fresh_secondary_process_count": 320,
            "old_timing_samples_reused": False,
            "target_or_pixel_payloads_opened": False,
        },
    )


def _verify_benchmark_cell(
    outdir: Path,
    record: Mapping[str, Any],
    candidate_cell: Mapping[str, Any],
    quality_cell: Mapping[str, Any],
    *,
    all_primary: bool,
    preflight_record: Mapping[str, Any],
    quality_stage_sha256: str,
) -> dict[str, Any]:
    actual.verify_record(record, "benchmark_cell_sha256")
    benchmark_identity = _identity(record, "benchmark cell")
    quality_identity = _identity(quality_cell, "quality cell")
    candidate_identity = _identity(candidate_cell, "candidate cell")
    if benchmark_identity != quality_identity or benchmark_identity != candidate_identity:
        raise LifecycleError("benchmark-cell identity is detached from quality/candidate identity")
    if (
        record.get("schema") != "structsplat.comp011.benchmark-cell.v1"
        or record.get("protocol") != PROTOCOL
        or record.get("task_sha256") != TASK_SHA256
        or record.get("quality_cell_sha256") != quality_cell.get("quality_cell_sha256")
        or record.get("primary_applicable") is not all_primary
        or record.get("old_timing_samples_reused") is not False
        or record.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("benchmark-cell schema/provenance/freshness drift")
    image_id = quality_cell["image_id"]
    bits = tuple(quality_cell["bit_tuple"])
    image_index = actual.FROZEN_IMAGES.index(image_id)
    tuple_index = actual.FROZEN_TUPLES.index(bits)
    execution_record = candidate_cell["execution"]
    worker = _decode_worker_source_record()
    native = preflight_record["native"]["arithmetic"]
    native_path = outdir / "runtime/native" / str(native["path"])
    candidate_base = _artifact_path(
        outdir, candidate_cell["relative_path"], "benchmark candidate-cell path"
    )
    secondary = record.get("secondary")
    if not isinstance(secondary, Mapping):
        raise LifecycleError("benchmark-cell secondary schedule is absent")
    frozen_secondary = decode_worker.secondary_schedule(image_index, tuple_index)
    if (
        secondary.get("schedule") != [launch.to_dict() for launch in frozen_secondary]
        or secondary.get("fresh_processes") is not True
        or not isinstance(secondary.get("samples"), list)
        or len(secondary["samples"]) != 20
        or not isinstance(secondary.get("launches"), list)
        or len(secondary["launches"]) != 20
    ):
        raise LifecycleError("secondary resource schedule/freshness record drift")
    for bundle, sample in zip(secondary["launches"], secondary["samples"], strict=True):
        _verify_launch_bundle(bundle, sample, "canonical_worker_sha256")
    expected_secondary_records = {
        "sspl1": _decoded_stream_record(execution_record, "sspl1"),
        "ssp2l": _decoded_stream_record(execution_record, "ssp2l"),
    }
    expected_secondary = {
        arm: stream["blob_sha256"]
        for arm, stream in expected_secondary_records.items()
    }
    secondary_session = _measurement_session_sha256(
        kind="secondary",
        image_id=image_id,
        bits=bits,
        quality_stage_sha256=quality_stage_sha256,
    )
    secondary_paths = {
        "sspl1": candidate_base / "exact/sspl1.bin",
        "ssp2l": candidate_base / "exact/ssp2l.bin",
    }
    for launch_spec, bundle in zip(
        frozen_secondary, secondary["launches"], strict=True
    ):
        blob_path = secondary_paths[launch_spec.arm]
        expected_command = _resource_worker_command(
            artifact_root=outdir,
            blob_relative_path=blob_path.relative_to(outdir).as_posix(),
            native_relative_path=native_path.relative_to(outdir).as_posix(),
            stream_record=expected_secondary_records[launch_spec.arm],
            native_record=native,
            worker_source=worker,
            measurement_session_sha256=secondary_session,
            launch=launch_spec,
        )
        _verify_ordinary_decode_worker_policy(
            bundle,
            outdir=outdir,
            expected_command=expected_command,
            blob_path=blob_path,
        )
    recomputed_secondary = decode_worker.summarize_schedule(
        secondary["samples"],
        frozen_secondary,
        image_index=image_index,
        tuple_index=tuple_index,
        measurement_session_sha256=secondary_session,
        expected_blob_sha256_by_arm=expected_secondary,
    )
    if secondary.get("summary") != recomputed_secondary:
        raise LifecycleError("secondary resource summary does not recompute")
    for launch, sample in zip(frozen_secondary, secondary["samples"], strict=True):
        expected = expected_secondary_records[launch.arm]
        exactness = sample.get("exactness")
        if (
            sample.get("blob_bytes") != expected["blob_bytes"]
            or sample.get("symbol_sha256") != expected["symbol_sha256"]
            or sample.get("boundary_state_sha256")
            != expected["boundary_state_sha256"]
            or not isinstance(exactness, Mapping)
            or exactness.get("expected_blob_bytes") != expected["blob_bytes"]
            or exactness.get("expected_symbol_sha256") != expected["symbol_sha256"]
            or exactness.get("expected_boundary_state_sha256")
            != expected["boundary_state_sha256"]
        ):
            raise LifecycleError("secondary resource sample authority drift")
    primary = record.get("primary")
    if all_primary:
        if not isinstance(primary, Mapping):
            raise LifecycleError("applicable primary resource schedule is absent")
        selected = quality_cell["selection"]["selected_label"]
        q_name = quality_cell["q_name"]
        frozen_primary = decode_worker.primary_schedule(image_index, tuple_index)
        if (
            primary.get("schedule") != [launch.to_dict() for launch in frozen_primary]
            or primary.get("fresh_processes") is not True
            or not isinstance(primary.get("samples"), list)
            or len(primary["samples"]) != 20
            or not isinstance(primary.get("launches"), list)
            or len(primary["launches"]) != 20
        ):
            raise LifecycleError("primary resource schedule/freshness record drift")
        for bundle, sample in zip(primary["launches"], primary["samples"], strict=True):
            _verify_launch_bundle(bundle, sample, "canonical_worker_sha256")
        expected_primary_records = {
            "candidate": _decoded_stream_record(execution_record, selected),
            "q": _decoded_stream_record(execution_record, q_name),
        }
        expected_primary = {
            arm: stream["blob_sha256"]
            for arm, stream in expected_primary_records.items()
        }
        primary_session = _measurement_session_sha256(
            kind="primary",
            image_id=image_id,
            bits=bits,
            quality_stage_sha256=quality_stage_sha256,
            candidate=selected,
            q_name=q_name,
        )
        primary_paths = {
            "candidate": candidate_base / f"variants/{selected}.ssp2v",
            "q": candidate_base / f"exact/{q_name}.bin",
        }
        for launch_spec, bundle in zip(
            frozen_primary, primary["launches"], strict=True
        ):
            blob_path = primary_paths[launch_spec.arm]
            expected_command = _resource_worker_command(
                artifact_root=outdir,
                blob_relative_path=blob_path.relative_to(outdir).as_posix(),
                native_relative_path=native_path.relative_to(outdir).as_posix(),
                stream_record=expected_primary_records[launch_spec.arm],
                native_record=native,
                worker_source=worker,
                measurement_session_sha256=primary_session,
                launch=launch_spec,
            )
            _verify_ordinary_decode_worker_policy(
                bundle,
                outdir=outdir,
                expected_command=expected_command,
                blob_path=blob_path,
            )
        recomputed_primary = decode_worker.summarize_schedule(
            primary["samples"],
            frozen_primary,
            image_index=image_index,
            tuple_index=tuple_index,
            measurement_session_sha256=primary_session,
            expected_blob_sha256_by_arm=expected_primary,
        )
        if primary.get("summary") != recomputed_primary:
            raise LifecycleError("primary resource summary does not recompute")
        for launch, sample in zip(frozen_primary, primary["samples"], strict=True):
            expected = expected_primary_records[launch.arm]
            exactness = sample.get("exactness")
            if (
                sample.get("blob_bytes") != expected["blob_bytes"]
                or sample.get("symbol_sha256") != expected["symbol_sha256"]
                or sample.get("boundary_state_sha256")
                != expected["boundary_state_sha256"]
                or not isinstance(exactness, Mapping)
                or exactness.get("expected_blob_bytes") != expected["blob_bytes"]
                or exactness.get("expected_symbol_sha256") != expected["symbol_sha256"]
                or exactness.get("expected_boundary_state_sha256")
                != expected["boundary_state_sha256"]
            ):
                raise LifecycleError("primary resource sample authority drift")
    elif primary is not None:
        raise LifecycleError("primary schedule ran without sixteen global selections")
    primary_resource, secondary_resource = _resource_records(
        quality_cell, primary if isinstance(primary, Mapping) else None, secondary
    )
    expected_row = _actual_row(quality_cell, primary_resource, secondary_resource)
    if record.get("actual_row") != expected_row:
        raise LifecycleError("benchmark actual row does not recompute")
    expected_bindings = {
        "worker_module": {
            "source_label": decode_worker.WORKER_MODULE_LABEL,
            "bytes": worker["bytes"],
            "sha256": worker["sha256"],
            "identity_verified_before_operation": True,
        },
        "native_library": {
            "artifact_relative_path": f"runtime/native/{native['path']}",
            "bytes": native["bytes"],
            "sha256": native["sha256"],
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
        "cdf_asset": {
            "source_label": "benchmarks/assets/ssp2e_cdf_v1.bin",
            "bytes": comp009_assets.CDF_ASSET_SIZE,
            "sha256": comp009_assets.CDF_ASSET_SHA256,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
    }
    schedules = [secondary, *([primary] if isinstance(primary, Mapping) else [])]
    launch_nonces = [
        bundle["launch"]["nonce"]
        for schedule_record in schedules
        for bundle in schedule_record["launches"]
    ]
    if len(launch_nonces) != len(set(launch_nonces)):
        raise LifecycleError("resource worker launch nonce was reused within a cell")
    if any(
        sample.get("source_bindings") != expected_bindings
        for schedule_record in schedules
        for sample in schedule_record["samples"]
    ):
        raise LifecycleError("resource worker source/native/CDF binding differs from preflight")
    return dict(record)


def verify_benchmark_stage(outdir: Path, *, captured_replay: bool = False) -> dict[str, Any]:
    outdir = _lexical(outdir)
    quality_record = verify_quality_stage(
        outdir,
        captured_replay=captured_replay,
        verify_target_payloads=False,
    )
    candidate = verify_candidate_stage(outdir, captured_replay=captured_replay)
    preflight_record = verify_preflight(outdir, captured_replay=captured_replay)
    record = _verify_stage_base(
        outdir,
        stage="benchmark-dev",
        schema=BENCHMARK_STAGE_SCHEMA,
        prior_record=quality_record,
    )
    cells = record.get("cells")
    if quality_record.get("state") == "inapplicable_terminal_lloyd_failure":
        failure_cell = candidate["cells"][-1]
        early_row = actual.seal_record(
            {
                "schema": actual.ROW_SCHEMA,
                "image_id": failure_cell["image_id"],
                "bit_tuple": failure_cell["bit_tuple"],
                "validity": "valid",
                "invalid_reason": None,
                "method_state": "lloyd_failure",
                "lloyd_all_fixed": False,
                "formats": None,
                "primary_metrics": None,
                "variants": None,
                "primary_resource": None,
                "secondary_resource": None,
            }
        )
        rows_record = record.get("actual_rows")
        if (
            record.get("state") != "inapplicable_terminal_lloyd_failure"
            or record.get("quality_stage_sha256")
            != quality_record.get("quality_stage_sha256")
            or cells != []
            or record.get("cell_count") != 0
            or not isinstance(rows_record, Mapping)
            or record.get("actual_row_sha256") != [early_row["record_sha256"]]
            or record.get("fresh_primary_process_count") != 0
            or record.get("fresh_secondary_process_count") != 0
            or record.get("primary_schedules_applicable") is not False
            or record.get("old_timing_samples_reused") is not False
            or record.get("target_or_pixel_payloads_opened") is not False
        ):
            raise LifecycleError("terminal Lloyd benchmark-skip manifest drift")
        rows_path = _artifact_path(outdir, rows_record["relative_path"], "rows path")
        _verify_regular_record(rows_path, rows_record)
        if _read_regular_no_follow(rows_path) != _canonical_json(
            early_row
        ):
            raise LifecycleError("terminal Lloyd scalar row artifact drift")
        return record
    invalid_quality = next(
        (
            cell
            for cell in quality_record["cells"]
            if cell["selection"]["state"] == "invalid"
        ),
        None,
    )
    if invalid_quality is not None:
        invalid_row = actual.seal_record(
            {
                "schema": actual.ROW_SCHEMA,
                "image_id": invalid_quality["image_id"],
                "bit_tuple": invalid_quality["bit_tuple"],
                "validity": "invalid",
                "invalid_reason": str(invalid_quality["selection"]["invalid_reason"]),
                "method_state": "invalid",
                "lloyd_all_fixed": None,
                "formats": None,
                "primary_metrics": None,
                "variants": None,
                "primary_resource": None,
                "secondary_resource": None,
            }
        )
        rows_record = record.get("actual_rows")
        if (
            record.get("state") != "inapplicable_invalid_quality"
            or record.get("quality_stage_sha256")
            != quality_record.get("quality_stage_sha256")
            or cells != []
            or record.get("cell_count") != 0
            or not isinstance(rows_record, Mapping)
            or record.get("actual_row_sha256") != [invalid_row["record_sha256"]]
            or record.get("fresh_primary_process_count") != 0
            or record.get("fresh_secondary_process_count") != 0
            or record.get("primary_schedules_applicable") is not False
            or record.get("old_timing_samples_reused") is not False
            or record.get("target_or_pixel_payloads_opened") is not False
        ):
            raise LifecycleError("invalid-quality benchmark-skip manifest drift")
        rows_path = _artifact_path(outdir, rows_record["relative_path"], "rows path")
        _verify_regular_record(rows_path, rows_record)
        if _read_regular_no_follow(rows_path) != _canonical_json(
            invalid_row
        ):
            raise LifecycleError("invalid-quality scalar row artifact drift")
        return record
    all_primary = all(
        cell["selection"]["state"] == "selected" for cell in quality_record["cells"]
    )
    if (
        not isinstance(cells, list)
        or len(cells) != 16
        or record.get("state") != "complete_resource_grid"
        or record.get("cell_count") != 16
        or record.get("quality_stage_sha256") != quality_record.get("quality_stage_sha256")
        or record.get("primary_schedules_applicable") is not all_primary
        or record.get("fresh_primary_process_count") != (320 if all_primary else 0)
        or record.get("fresh_secondary_process_count") != 320
        or record.get("old_timing_samples_reused") is not False
        or record.get("target_or_pixel_payloads_opened") is not False
    ):
        raise LifecycleError("benchmark-stage inventory/schedule/freshness drift")
    benchmark_identities = [_identity(cell, "benchmark cell") for cell in cells]
    candidate_identities = [
        _identity(cell, "candidate cell") for cell in candidate["cells"]
    ]
    quality_identities = [
        _identity(cell, "quality cell") for cell in quality_record["cells"]
    ]
    if not (
        benchmark_identities
        == candidate_identities
        == quality_identities
        == _expected_identities()
    ):
        raise LifecycleError("benchmark identities differ from the frozen quality/candidate grid")
    checked = [
        _verify_benchmark_cell(
            outdir,
            cell,
            candidate_cell,
            quality_cell,
            all_primary=all_primary,
            preflight_record=preflight_record,
            quality_stage_sha256=str(quality_record["quality_stage_sha256"]),
        )
        for cell, candidate_cell, quality_cell in zip(
            cells, candidate["cells"], quality_record["cells"], strict=True
        )
    ]
    for cell in checked:
        persisted = _read_canonical_object(
            outdir / _benchmark_cell_relative(cell["image_id"], cell["bit_tuple"])
        )
        if persisted != cell:
            raise LifecycleError("benchmark manifest cell differs from immutable per-cell file")
    benchmark_paths = [
        outdir / _benchmark_cell_relative(cell["image_id"], cell["bit_tuple"])
        for cell in checked
    ]
    receipts = _verify_stage_cell_receipts(
        outdir,
        EventJournal.verify(outdir / "journal/events"),
        stage="benchmark-dev",
        prior_stage_seal=str(quality_record["quality_stage_sha256"]),
        expected_paths=benchmark_paths,
        purpose="persist unified primary/secondary resource schedules",
    )
    if len(receipts) != len(benchmark_paths):
        raise LifecycleError("benchmark-stage cell receipt grid is incomplete")
    rows = [cell["actual_row"] for cell in checked]
    rows_payload = b"".join(_canonical_json(row) for row in rows)
    rows_record = record.get("actual_rows")
    if not isinstance(rows_record, Mapping):
        raise LifecycleError("benchmark scalar-row artifact record is absent")
    path = _artifact_path(outdir, rows_record["relative_path"], "rows path")
    _verify_regular_record(path, rows_record)
    if (
        _read_regular_no_follow(path) != rows_payload
        or record.get("actual_row_sha256") != [row["record_sha256"] for row in rows]
    ):
        raise LifecycleError("benchmark scalar-row file/seal vector drift")
    return record


def _read_actual_rows(outdir: Path, benchmark: Mapping[str, Any]) -> list[dict[str, Any]]:
    reference = benchmark.get("actual_rows")
    if not isinstance(reference, Mapping):
        raise LifecycleError("benchmark actual-row artifact reference is absent")
    payload = _read_regular_no_follow(
        _artifact_path(outdir, reference["relative_path"], "rows path")
    )
    if len(payload) != reference.get("bytes") or _sha256_bytes(payload) != reference.get("sha256"):
        raise LifecycleError("benchmark actual-row artifact identity drift")
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        if not line:
            raise LifecycleError("actual-row JSONL contains a blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LifecycleError("actual-row JSONL parse failed") from exc
        if not isinstance(row, dict) or _canonical_json(row).removesuffix(b"\n") != line:
            raise LifecycleError("actual-row JSONL row is not canonical")
        actual.verify_record(row)
        rows.append(row)
    if [row["record_sha256"] for row in rows] != benchmark.get("actual_row_sha256"):
        raise LifecycleError("actual-row JSONL seal vector drift")
    return rows


def analyze_stage(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    _recover_interrupted_stage_completion(outdir, "analyze")
    existing = outdir / _STAGE_MANIFEST_PATHS["analyze"]
    if existing.exists():
        return verify_analysis_stage(outdir)
    journal, _guard, records = _stage_guard(outdir, "analyze")
    benchmark = verify_benchmark_stage(outdir)
    prior_seal = str(benchmark["benchmark_stage_sha256"])
    rows = _read_actual_rows(outdir, benchmark)
    analysis = actual.analyze_rows(rows)
    analysis_artifact = _journaled_immutable(
        journal,
        outdir,
        outdir / "analysis.json",
        _canonical_json(analysis),
        stage="analyze",
        path_class="artifact_runtime",
        purpose="persist recomputed frozen COMP-011 decision",
        prior_stage_seal=prior_seal,
    )
    return _finalize_stage(
        outdir,
        journal,
        stage="analyze",
        schema=ANALYSIS_RECORD_SCHEMA,
        prior_stage_seal=prior_seal,
        body={
            "state": (
                "terminal_lloyd_failure_analysis"
                if benchmark.get("state") == "inapplicable_terminal_lloyd_failure"
                else "complete_analysis"
            ),
            "benchmark_stage_sha256": prior_seal,
            "actual_row_sha256": [row["record_sha256"] for row in rows],
            "row_count": len(rows),
            "analysis": analysis_artifact,
            "analysis_sha256": analysis["analysis_sha256"],
            "decision": analysis["decision"],
            "ssp2l_decision": analysis["ssp2l_decision"],
            "confirmation_access_authorized_in_this_task": False,
            "target_or_pixel_payloads_opened": False,
        },
    )


def verify_analysis_stage(outdir: Path, *, captured_replay: bool = False) -> dict[str, Any]:
    outdir = _lexical(outdir)
    benchmark = verify_benchmark_stage(outdir, captured_replay=captured_replay)
    record = _verify_stage_base(
        outdir,
        stage="analyze",
        schema=ANALYSIS_RECORD_SCHEMA,
        prior_record=benchmark,
    )
    rows = _read_actual_rows(outdir, benchmark)
    recomputed = actual.analyze_rows(rows)
    analysis_reference = record.get("analysis")
    if not isinstance(analysis_reference, Mapping):
        raise LifecycleError("analysis artifact reference is absent")
    analysis_path = _artifact_path(
        outdir, analysis_reference["relative_path"], "analysis path"
    )
    _verify_regular_record(analysis_path, analysis_reference)
    persisted = _read_canonical_object(analysis_path)
    actual.verify_record(persisted, "analysis_sha256")
    expected_state = (
        "terminal_lloyd_failure_analysis"
        if benchmark.get("state") == "inapplicable_terminal_lloyd_failure"
        else "complete_analysis"
    )
    if (
        persisted != recomputed
        or record.get("state") != expected_state
        or record.get("benchmark_stage_sha256") != benchmark.get("benchmark_stage_sha256")
        or record.get("actual_row_sha256") != [row["record_sha256"] for row in rows]
        or record.get("row_count") != len(rows)
        or record.get("analysis_sha256") != recomputed["analysis_sha256"]
        or record.get("decision") != recomputed["decision"]
        or record.get("ssp2l_decision") != recomputed["ssp2l_decision"]
        or record.get("confirmation_access_authorized_in_this_task") is not False
        or record.get("target_or_pixel_payloads_opened") is not False
    ):
        raise LifecycleError("analysis record/decision does not recompute from sealed rows")
    return record


def _replay_candidate_set(outdir: Path) -> dict[str, Any]:
    """Regenerate all deterministic codec evidence before any replay target-copy open."""

    preflight_record = verify_preflight(outdir, captured_replay=True)
    imported = verify_import_streams(outdir, captured_replay=True)
    candidate = verify_candidate_stage(outdir, captured_replay=True)
    authority = _authority_copy(outdir, preflight_record)
    authority_by_identity = {
        (cell["image_id"], tuple(cell["bit_tuple"])): cell
        for cell in authority["comp009"]["cells"]
    }
    imported_by_identity = {
        (cell["image_id"], tuple(cell["bit_tuple"])): cell for cell in imported["streams"]
    }
    native_path, native_record = _native_arithmetic_record(outdir, preflight_record)
    lloyd_record = preflight_record["native"]["lloyd"]
    lloyd_path = outdir / "runtime/native" / str(lloyd_record["path"])
    _verify_regular_record(lloyd_path, lloyd_record)
    archive = _artifact_path(
        outdir,
        authority["comp009"]["captured_source_archive"]["relative_path"],
        "captured COMP-009 archive path",
    )
    scratch_value = os.environ.get("COMP011_WORKER_SCRATCH")
    if not isinstance(scratch_value, str) or not scratch_value:
        raise LifecycleError("replay phase-A scratch authority is absent")
    scratch = Path(scratch_value)
    if not scratch.is_absolute() or _lexical(scratch) != scratch:
        raise LifecycleError("replay phase-A scratch path is not lexical absolute")
    _reject_symlink_components(scratch, require_regular=False)
    if not scratch.is_dir():
        raise LifecycleError("replay phase-A scratch authority is not a directory")
    regenerated_root = scratch / "regenerated-artifacts"
    regenerated_root.mkdir(mode=0o755, parents=False, exist_ok=False)
    scratch_native_path = regenerated_root / "runtime/native" / str(native_record["path"])
    _exclusive_write(scratch_native_path, _read_regular_no_follow(native_path))
    _verify_regular_record(scratch_native_path, native_record)

    regenerated_cells: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="comp011-replay-comp009-") as temporary:
        extracted = Path(temporary) / "source"
        safe_extract_source_archive(
            archive,
            extracted,
            expected_manifest=authority["comp009"]["captured_source_manifest"],
        )
        for original_cell in candidate["cells"]:
            identity = (original_cell["image_id"], tuple(original_cell["bit_tuple"]))
            source_record = imported_by_identity[identity]
            source_path = _artifact_path(
                outdir, source_record["relative_path"], "imported-stream path"
            )
            upstream_target_probe = _lexical(
                Path(str(preflight_record["operational_upstream_roots"]["comp008"]))
            ) / _safe_relative_path(
                authority["comp008"]["targets"][0]["relative_path"],
                "replay producing-worker denied target probe",
            )
            denied_probes = [upstream_target_probe]
            result, producing_diagnostics = _run_producing_cell_worker(
                outdir=outdir,
                image_id=str(identity[0]),
                bits=identity[1],
                source_path=source_path,
                source_record=source_record,
                captured_comp009_root=extracted,
                comp009_authority=authority_by_identity[identity],
                captured_source_manifest_sha256=str(
                    authority["comp009"]["source_manifest_sha256"]
                ),
                native_path=native_path,
                native_record=native_record,
                lloyd_path=lloyd_path,
                lloyd_record=lloyd_record,
                denied_read_probes=denied_probes,
            )
            execute.verify_record(result.record)
            if result.record != original_cell["execution"]:
                raise LifecycleError("replay deterministic execution record differs from original")
            files = _candidate_blob_files(result)
            regenerated_artifacts = _artifact_records_for_files(files)
            original_artifacts = {item["path"]: item for item in original_cell["artifacts"]}
            if set(files) != set(original_artifacts):
                raise LifecycleError("replay candidate blob inventory differs from original")
            cold_records: dict[str, dict[str, Any]] = {}
            for relative, blob in files.items():
                artifact = original_artifacts[relative]
                logical_relative = (
                    _safe_relative_path(
                        original_cell["relative_path"], "candidate-cell path"
                    )
                    / _safe_relative_path(relative, "candidate stream path")
                )
                persisted_blob_path = _artifact_path(
                    outdir,
                    logical_relative,
                    "candidate stream path",
                )
                if (
                    len(blob) != artifact["bytes"]
                    or _sha256_bytes(blob) != artifact["sha256"]
                    or blob != _read_regular_no_follow(persisted_blob_path)
                ):
                    raise LifecycleError("replay candidate complete stream is not byte-identical")
                regenerated_blob_path = regenerated_root / logical_relative
                _exclusive_write(regenerated_blob_path, blob)
                name = Path(relative).stem
                expected_symbol, expected_boundary = _cold_decode_expectations(result.record, name)
                components_sha256 = decode_worker.component_records_sha256(
                    decode_worker.stream_component_records(blob)
                )
                cold_bundle = _run_cold_decode_sample(
                    outdir=regenerated_root,
                    blob_path=regenerated_blob_path,
                    blob_sha256=str(artifact["sha256"]),
                    symbol_sha256=expected_symbol,
                    boundary_state_sha256=expected_boundary,
                    native_path=scratch_native_path,
                    native_record=native_record,
                    stream_components_sha256=components_sha256,
                )
                cold = cold_bundle["cold_seal"]
                original_cold = original_cell["cold_decodes"][relative]["cold_seal"]
                if cold["cold_decode_state_sha256"] != original_cold["cold_decode_state_sha256"]:
                    raise LifecycleError("replay cold-decode semantic state differs from original")
                cold_records[relative] = cold_bundle
            regenerated_cells.append(
                {
                    "image_id": identity[0],
                    "bit_tuple": list(identity[1]),
                    "state": original_cell["state"],
                    "relative_path": _candidate_cell_relative(
                        str(identity[0]), identity[1]
                    ).as_posix(),
                    "execution": result.record,
                    "artifacts": regenerated_artifacts,
                    "cold_decodes": cold_records,
                    "producing_diagnostics": producing_diagnostics,
                }
            )
            if isinstance(result, execute.LloydMethodFailure):
                break
    if candidate["state"] == "complete_candidate_set":
        evidence_cells = []
        for original, regenerated in zip(candidate["cells"], regenerated_cells, strict=True):
            evidence_cells.append(
                {
                    "execution": regenerated["execution"],
                    "artifacts": regenerated["artifacts"],
                    "cold_decodes": regenerated["cold_decodes"],
                }
            )
        evidence = _candidate_global_evidence(evidence_cells)
        seal = _sha256_bytes(_canonical_json(evidence))
        if seal != candidate["candidate_set_sha256"]:
            raise LifecycleError("replay global candidate-set seal differs from original")
    else:
        seal = None
        if (
            regenerated_cells[-1]["state"] != "terminal_lloyd_failure"
            or regenerated_cells[-1]["execution"] != candidate["cells"][-1]["execution"]
        ):
            raise LifecycleError("replay terminal Lloyd stopping condition drift")
    return {
        "state": candidate["state"],
        "candidate_set_sha256": seal,
        "regenerated_artifact_root": os.fspath(regenerated_root),
        "cells": regenerated_cells,
        "cell_count": len(regenerated_cells),
        "all_complete_streams_byte_identical": True,
        "all_cold_decode_states_identical": True,
        "targets_opened_before_candidate_match": False,
    }


def _metric_endpoints(values: Sequence[Mapping[str, Any]], operation: str) -> dict[str, float]:
    reducer = min if operation == "min" else max
    return {
        name: reducer(float(value[name]) for value in values) for name in quality.METRIC_NAMES
    }


def _verify_replay_quality_classification(
    original_primary: Sequence[Mapping[str, Any]],
    original_candidate: Sequence[Mapping[str, Any]],
    replay_primary: Sequence[Mapping[str, Any]],
    replay_candidate: Sequence[Mapping[str, Any]],
) -> None:
    """Reproduce every metric class and keep both runs outside all frozen bands."""

    original = actual.evaluate_quality(original_primary, original_candidate)
    replay = actual.evaluate_quality(replay_primary, replay_candidate)
    if replay.get("state") != original.get("state"):
        raise LifecycleError("replay decision-relevant quality class drift")
    original_classes = original.get("metric_classes")
    replay_classes = replay.get("metric_classes")
    if not isinstance(original_classes, Mapping) or not isinstance(
        replay_classes, Mapping
    ):
        raise LifecycleError("replay decision-relevant quality classification is absent")
    if any(
        value == "ambiguous"
        for value in (*original_classes.values(), *replay_classes.values())
    ):
        raise LifecycleError("replay decision-relevant metric entered an exclusion band")
    if replay_classes != original_classes:
        raise LifecycleError("replay decision-relevant per-metric class drift")


def _decision_relevant_variant_prefix(
    original: Mapping[str, Any], candidate_cell: Mapping[str, Any] | None = None
) -> list[str]:
    selection = original.get("selection")
    if not isinstance(selection, Mapping):
        raise LifecycleError("quality replay source selection is absent")
    state = selection.get("state")
    if candidate_cell is None:
        if state in {"method_failure", "invalid"}:
            return list(actual.VARIANT_ORDER)
        raise LifecycleError("selected quality replay requires regenerated candidate evidence")
    execution = candidate_cell.get("execution")
    if not isinstance(execution, Mapping):
        raise LifecycleError("quality replay regenerated execution is absent")
    ordered = sorted(
        actual.VARIANT_ORDER,
        key=lambda label: (
            int(execution["variants"][label]["stream"]["complete_bytes"]),
            *actual.VARIANT_SPECS[label],
            label.encode("ascii"),
        ),
    )
    if state == "selected":
        if selection.get("ordered_labels") != ordered:
            raise LifecycleError(
                "quality replay original ordering differs from regenerated bytes"
            )
        selected = selection.get("selected_label")
        if selected not in ordered:
            raise LifecycleError("quality replay selected variant is absent from its ordering")
        relevant = set(ordered[: ordered.index(selected) + 1])
    elif state in {"method_failure", "invalid"}:
        if selection.get("ordered_labels") not in ([], ordered):
            raise LifecycleError(
                "quality replay original ordering differs from regenerated bytes"
            )
        relevant = set(actual.VARIANT_ORDER)
    else:
        raise LifecycleError("quality replay source selection has an unknown state")
    return [label for label in actual.VARIANT_ORDER if label in relevant]


def _replay_phase_b_candidate_context(
    outdir: Path,
    phase_a_path: Path,
) -> dict[str, Any]:
    """Verify phase A and project only its regenerated candidate authority."""

    outdir = _lexical(outdir)
    phase_a = _read_canonical_object(phase_a_path)
    actual.verify_record(phase_a, "replay_phase_a_sha256")
    if (
        phase_a.get("schema") != REPLAY_PHASE_A_SCHEMA
        or phase_a.get("source_manifest_sha256")
        != source_manifest()["source_manifest_sha256"]
        or phase_a.get("targets_opened") is not False
        or phase_a.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("persisted replay phase-A authorization is invalid")
    candidate_replay = phase_a.get("candidate_replay")
    candidate = _read_canonical_object(
        outdir / _STAGE_MANIFEST_PATHS["fit-encode-cold-decode-dev"]
    )
    if not isinstance(candidate_replay, Mapping):
        raise LifecycleError("replay phase-A candidate authority is absent")
    if candidate_replay.get("state") != candidate.get("state") or (
        candidate.get("state") == "complete_candidate_set"
        and candidate_replay.get("candidate_set_sha256")
        != candidate.get("candidate_set_sha256")
    ):
        raise LifecycleError(
            "replay phase-A candidate authorization differs from candidate stage"
        )
    replayed_cells = candidate_replay.get("cells")
    original_cells = candidate.get("cells")
    if (
        not isinstance(replayed_cells, list)
        or not isinstance(original_cells, list)
        or len(replayed_cells) != len(original_cells)
    ):
        raise LifecycleError("replay phase-A candidate cell inventory is incomplete")
    replayed_candidate = {
        "state": candidate["state"],
        "candidate_set_sha256": candidate_replay.get("candidate_set_sha256"),
        "cells": [
            {
                "image_id": regenerated["image_id"],
                "bit_tuple": regenerated["bit_tuple"],
                "state": regenerated["state"],
                "relative_path": regenerated["relative_path"],
                "execution": regenerated["execution"],
                "artifacts": regenerated["artifacts"],
                "cold_decodes": regenerated["cold_decodes"],
            }
            for regenerated in replayed_cells
        ],
    }
    return {
        "outdir": outdir,
        "phase_a": phase_a,
        "candidate": candidate,
        "candidate_replay": candidate_replay,
        "replayed_candidate": replayed_candidate,
        "regenerated_root": _lexical(
            Path(str(candidate_replay["regenerated_artifact_root"]))
        ),
    }


def _replay_quality_plan_body(
    *,
    context: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Construct the complete logical quality schedule without executing it."""

    replayed_candidate = context["replayed_candidate"]
    if replayed_candidate.get("state") != "complete_candidate_set":
        raise LifecycleError("quality planning requires one complete candidate set")
    candidate_cells = replayed_candidate.get("cells")
    original_cells = original_quality.get("cells")
    target_by_image = {
        target["image_id"]: target for target in targets.get("targets", [])
    }
    if (
        not isinstance(candidate_cells, list)
        or not isinstance(original_cells, list)
        or len(candidate_cells) != 16
        or len(original_cells) != 16
        or set(target_by_image) != set(actual.FROZEN_IMAGES)
    ):
        raise LifecycleError("quality planning authority grid is incomplete")
    planned_cells: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for candidate_cell, original in zip(
        candidate_cells, original_cells, strict=True
    ):
        if _identity(candidate_cell, "quality plan candidate") != _identity(
            original, "quality plan original"
        ):
            raise LifecycleError("quality plan cell identity drift")
        prefix = _decision_relevant_variant_prefix(original, candidate_cell)
        schedules = (
            ("primary", *prefix),
            (*reversed(prefix), "primary"),
            ("primary", *prefix),
        )
        q_name = str(candidate_cell["execution"]["selected_q"]["name"])
        if original.get("q_name") != q_name:
            raise LifecycleError(
                "quality plan original Q differs from regenerated execution"
            )
        target = target_by_image[original["image_id"]]
        tasks: list[dict[str, Any]] = []
        for repetition, order in enumerate(schedules):
            for order_index, arm in enumerate(order):
                stream_name = q_name if arm == "primary" else arm
                stream_relative = (
                    f"exact/{stream_name}.bin"
                    if arm == "primary"
                    else f"variants/{stream_name}.ssp2v"
                )
                blob_relative = (
                    _safe_relative_path(
                        candidate_cell["relative_path"],
                        "quality plan candidate-cell path",
                    )
                    / _safe_relative_path(
                        stream_relative, "quality plan stream path"
                    )
                ).as_posix()
                expected_stream = _decoded_stream_record(
                    candidate_cell["execution"], stream_name
                )
                identity = {
                    "replay_phase_a_sha256": context["phase_a"][
                        "replay_phase_a_sha256"
                    ],
                    "image_id": original["image_id"],
                    "bit_tuple": original["bit_tuple"],
                    "repetition": repetition,
                    "order_index": order_index,
                    "arm": arm,
                    "stream_name": stream_name,
                    "blob_relative_path": blob_relative,
                }
                task_id = _sha256_bytes(_canonical_json(identity))
                task_ids.append(task_id)
                tasks.append(
                    {
                        "task_id": task_id,
                        "repetition": repetition,
                        "order_index": order_index,
                        "arm": arm,
                        "stream_name": stream_name,
                        "blob_relative_path": blob_relative,
                        "expected_stream": expected_stream,
                        "target": {
                            "image_id": target["image_id"],
                            "rgb_relative_path": target["rgb_copy"][
                                "relative_path"
                            ],
                            "rgb_sha256": target["rgb_copy"]["sha256"],
                            "dimensions_hw": target["rgb_dimensions_hw"],
                        },
                    }
                )
        planned_cells.append(
            {
                "image_id": original["image_id"],
                "bit_tuple": original["bit_tuple"],
                "q_name": q_name,
                "prefix": prefix,
                "schedule": [list(order) for order in schedules],
                "tasks": tasks,
            }
        )
    if len(task_ids) != len(set(task_ids)):
        raise LifecycleError("quality plan task identifier reuse")
    native = preflight_record["native"]["arithmetic"]
    renderer = preflight_record["renderer"]
    return {
        "schema": REPLAY_QUALITY_PLAN_SCHEMA,
        "protocol": PROTOCOL,
        "task_sha256": TASK_SHA256,
        "source_manifest_sha256": source_manifest()["source_manifest_sha256"],
        "replay_phase_a_sha256": context["phase_a"][
            "replay_phase_a_sha256"
        ],
        "candidate_set_sha256": context["candidate_replay"].get(
            "candidate_set_sha256"
        ),
        "target_copy_sha256": targets["target_copy_sha256"],
        "quality_stage_sha256": original_quality["quality_stage_sha256"],
        "native": dict(native),
        "renderer": dict(renderer),
        "expected_lpips_state_sha256": preflight_record["renderer_proof"][
            "same_worker_metric_repeat"
        ]["lpips_state_sha256"],
        "cells": planned_cells,
        "task_count": len(task_ids),
        "broker_executes_only_declared_tasks": True,
        "scientific_reduction_deferred": True,
        "targets_opened": True,
        "confirmation_payloads_accessed": False,
    }


def replay_phase_b_quality_plan_worker(
    outdir: Path,
    phase_a_path: Path,
) -> dict[str, Any]:
    """Captured-source pass that plans, but cannot execute, quality jobs."""

    context = _replay_phase_b_candidate_context(outdir, phase_a_path)
    if context["candidate_replay"].get("state") != "complete_candidate_set":
        raise LifecycleError("terminal replay cannot enter quality planning")
    targets = verify_target_copy(context["outdir"], captured_replay=True)
    original_quality = verify_quality_stage(
        context["outdir"], captured_replay=True
    )
    preflight_record = verify_preflight(
        context["outdir"], captured_replay=True
    )
    return actual.seal_record(
        _replay_quality_plan_body(
            context=context,
            targets=targets,
            original_quality=original_quality,
            preflight_record=preflight_record,
        ),
        "quality_plan_sha256",
    )


def _verify_replay_quality_plan(
    plan: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
) -> None:
    actual.verify_record(plan, "quality_plan_sha256")
    expected = actual.seal_record(
        _replay_quality_plan_body(
            context=context,
            targets=targets,
            original_quality=original_quality,
            preflight_record=preflight_record,
        ),
        "quality_plan_sha256",
    )
    if dict(plan) != expected:
        raise LifecycleError("replay quality plan differs from exact authority")


def _execute_replay_quality_plan(
    plan: Mapping[str, Any],
    *,
    outdir: Path,
    context: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    captured_source_root: Path,
    system_read_only: Sequence[Path],
    gpu_read_write: Sequence[Path],
    scratch_root: Path,
) -> dict[str, Any]:
    """Trusted host broker for exact quality jobs; it performs no reduction."""

    if os.environ.get(sandbox.SANDBOX_ATTESTATION_ENV) is not None:
        raise LifecycleError("quality broker must execute outside every sandbox")
    _verify_replay_quality_plan(
        plan,
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight_record,
    )
    outdir = _lexical(outdir)
    captured_source_root = _lexical(captured_source_root)
    regenerated_root = _lexical(context["regenerated_root"])
    target_by_image = {
        target["image_id"]: target for target in targets["targets"]
    }
    renderer_record_path = outdir / "runtime/renderer/record.json"
    native_path, native_record = _native_arithmetic_record(
        outdir, preflight_record
    )
    result_cells: list[dict[str, Any]] = []
    launch_nonces: list[str] = []
    worker_pids: list[int] = []
    for cell in plan["cells"]:
        task_results: list[dict[str, Any]] = []
        for planned in cell["tasks"]:
            blob_path = _artifact_path(
                regenerated_root,
                planned["blob_relative_path"],
                "broker quality blob path",
            )
            expected = planned["expected_stream"]
            task = _run_render_metric_task(
                outdir=outdir,
                renderer_record_path=renderer_record_path,
                renderer_record=preflight_record["renderer"],
                blob_path=blob_path,
                blob_sha256=str(expected["blob_sha256"]),
                symbol_sha256=str(expected["symbol_sha256"]),
                boundary_state_sha256=str(
                    expected["boundary_state_sha256"]
                ),
                native_path=native_path,
                native_record=native_record,
                target_record=target_by_image[cell["image_id"]],
                expected_lpips_state_sha256=str(
                    plan["expected_lpips_state_sha256"]
                ),
                worker_source_root=captured_source_root,
                worker_system_read_only=system_read_only,
                worker_gpu_read_write=gpu_read_write,
                scratch_root=scratch_root,
            )
            for name, worker, seal_field in (
                ("render", task["render"], "render_worker_sha256"),
                ("metric", task["metric"], "metric_worker_sha256"),
            ):
                _verify_launch_bundle(
                    {
                        "attestation": task[f"{name}_sandbox"],
                        "launch": task[f"{name}_launch"],
                    },
                    worker,
                    seal_field,
                    expected_parent_sandbox_attestation_sha256=None,
                )
                launch_nonces.append(task[f"{name}_launch"]["nonce"])
                worker_pids.append(int(worker["pid"]))
            task_results.append(
                {"task_id": planned["task_id"], "task": task}
            )
        result_cells.append(
            {
                "image_id": cell["image_id"],
                "bit_tuple": cell["bit_tuple"],
                "tasks": task_results,
            }
        )
    if (
        len(launch_nonces) != len(set(launch_nonces))
        or len(worker_pids) != len(set(worker_pids))
    ):
        raise LifecycleError("quality broker fresh-worker identity reuse")
    return actual.seal_record(
        {
            "schema": REPLAY_QUALITY_RESULTS_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "quality_plan_sha256": plan["quality_plan_sha256"],
            "cells": result_cells,
            "task_count": plan["task_count"],
            "worker_launch_count": len(launch_nonces),
            "all_workers_fresh_root": True,
            "scientific_reduction_performed": False,
            "confirmation_payloads_accessed": False,
        },
        "quality_results_sha256",
    )


def _reduce_replay_quality_results(
    plan: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    captured_source_root: Path,
    replay_source_authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Captured-source quality reduction over host-brokered fresh-root receipts."""

    actual.verify_record(results, "quality_results_sha256")
    if (
        set(results)
        != {
            "schema",
            "protocol",
            "task_sha256",
            "source_manifest_sha256",
            "quality_plan_sha256",
            "cells",
            "task_count",
            "worker_launch_count",
            "all_workers_fresh_root",
            "scientific_reduction_performed",
            "confirmation_payloads_accessed",
            "quality_results_sha256",
        }
        or results.get("schema") != REPLAY_QUALITY_RESULTS_SCHEMA
        or results.get("protocol") != PROTOCOL
        or results.get("task_sha256") != TASK_SHA256
        or results.get("source_manifest_sha256")
        != plan.get("source_manifest_sha256")
        or results.get("quality_plan_sha256")
        != plan.get("quality_plan_sha256")
        or results.get("task_count") != plan.get("task_count")
        or results.get("worker_launch_count")
        != 2 * int(plan.get("task_count", -1))
        or results.get("all_workers_fresh_root") is not True
        or results.get("scientific_reduction_performed") is not False
        or results.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("replay quality broker result authority drift")
    planned_cells = plan.get("cells")
    result_cells = results.get("cells")
    candidate_cells = context["replayed_candidate"].get("cells")
    original_cells = original_quality.get("cells")
    target_by_image = {
        target["image_id"]: target for target in targets.get("targets", [])
    }
    if not all(
        isinstance(values, list) and len(values) == 16
        for values in (
            planned_cells,
            result_cells,
            candidate_cells,
            original_cells,
        )
    ):
        raise LifecycleError("replay quality reduction grid is incomplete")
    replay_cells: list[dict[str, Any]] = []
    row_inputs: list[dict[str, Any]] = []
    all_nonces: list[str] = []
    for planned_cell, result_cell, candidate_cell, original in zip(
        planned_cells,
        result_cells,
        candidate_cells,
        original_cells,
        strict=True,
    ):
        identity = _identity(planned_cell, "quality plan cell")
        if (
            identity != _identity(result_cell, "quality result cell")
            or identity != _identity(candidate_cell, "quality candidate cell")
            or identity != _identity(original, "quality original cell")
        ):
            raise LifecycleError("replay quality plan/result cell identity drift")
        planned_tasks = planned_cell.get("tasks")
        result_tasks = result_cell.get("tasks")
        if (
            set(result_cell) != {"image_id", "bit_tuple", "tasks"}
            or not isinstance(planned_tasks, list)
            or not isinstance(result_tasks, list)
            or len(result_tasks) != len(planned_tasks)
        ):
            raise LifecycleError("replay quality result task grid drift")
        prefix = list(planned_cell["prefix"])
        metrics: dict[str, list[dict[str, float]]] = {
            "primary": [],
            **{label: [] for label in prefix},
        }
        assembled_tasks: list[dict[str, Any]] = []
        for planned, returned in zip(
            planned_tasks, result_tasks, strict=True
        ):
            if (
                not isinstance(returned, Mapping)
                or set(returned) != {"task_id", "task"}
                or returned.get("task_id") != planned.get("task_id")
                or not isinstance(returned.get("task"), Mapping)
            ):
                raise LifecycleError("replay quality task/result binding drift")
            task = returned["task"]
            if set(task) != {
                "render",
                "metric",
                "render_sandbox",
                "render_launch",
                "metric_sandbox",
                "metric_launch",
            }:
                raise LifecycleError("replay quality worker result key drift")
            actual.verify_record(task["render"], "render_worker_sha256")
            actual.verify_record(task["metric"], "metric_worker_sha256")
            _verify_launch_bundle(
                {
                    "attestation": task["render_sandbox"],
                    "launch": task["render_launch"],
                },
                task["render"],
                "render_worker_sha256",
                expected_parent_sandbox_attestation_sha256=None,
            )
            _verify_launch_bundle(
                {
                    "attestation": task["metric_sandbox"],
                    "launch": task["metric_launch"],
                },
                task["metric"],
                "metric_worker_sha256",
                expected_parent_sandbox_attestation_sha256=None,
            )
            all_nonces.extend(
                (
                    str(task["render_launch"]["nonce"]),
                    str(task["metric_launch"]["nonce"]),
                )
            )
            arm = str(planned["arm"])
            values = task["metric"].get("metrics")
            if arm not in metrics or not isinstance(values, Mapping):
                raise LifecycleError("replay quality metric arm/result drift")
            metrics[arm].append(
                {name: float(values[name]) for name in quality.METRIC_NAMES}
            )
            assembled_tasks.append(
                {
                    "repetition": planned["repetition"],
                    "order_index": planned["order_index"],
                    "arm": arm,
                    "stream_name": planned["stream_name"],
                    **task,
                }
            )
        q_name = str(planned_cell["q_name"])
        original_variants = {
            variant["label"]: variant for variant in original["variants"]
        }
        full_metrics: dict[str, Sequence[Mapping[str, Any]]] = {
            label: (
                metrics[label]
                if label in metrics
                else original_variants[label]["metrics"]
            )
            for label in actual.VARIANT_ORDER
        }
        formats, replay_variants = _quality_scalar_inputs(
            candidate_cell, full_metrics
        )
        for variant in replay_variants:
            label = variant["label"]
            if label not in metrics:
                continue
            original_metrics = original_variants[label]["metrics"]
            for endpoint in ("min", "max"):
                if not quality.replay_endpoint_match(
                    _metric_endpoints(original_metrics, endpoint),
                    _metric_endpoints(metrics[label], endpoint),
                ):
                    raise LifecycleError(
                        "replay quality endpoint exceeds frozen tolerance"
                    )
            _verify_replay_quality_classification(
                original["primary_metrics"],
                original_metrics,
                metrics["primary"],
                metrics[label],
            )
        for endpoint in ("min", "max"):
            if not quality.replay_endpoint_match(
                _metric_endpoints(original["primary_metrics"], endpoint),
                _metric_endpoints(metrics["primary"], endpoint),
            ):
                raise LifecycleError(
                    "replay primary quality endpoint exceeds frozen tolerance"
                )
        replay_selection = actual.select_variant(
            metrics["primary"], replay_variants
        )
        for field in (
            "state",
            "invalid_reason",
            "selected_label",
            "selected_bytes",
            "ordered_labels",
        ):
            if replay_selection.get(field) != original["selection"].get(field):
                raise LifecycleError(
                    "replay selected variant/state differs from original"
                )
        replay_cell = {
            "image_id": original["image_id"],
            "bit_tuple": original["bit_tuple"],
            "prefix": prefix,
            "schedule": planned_cell["schedule"],
            "metrics": metrics,
            "metric_envelopes": _quality_metric_envelopes(metrics),
            "nongating_later_metrics": {
                label: original_variants[label]["metrics"]
                for label in actual.VARIANT_ORDER
                if label not in metrics
            },
            "selection": replay_selection,
            "tasks": assembled_tasks,
            "all_endpoint_tolerances_pass": True,
            "all_qualification_classes_reproduced": True,
            "all_decision_exclusion_bands_clear": True,
        }
        reconstructed, nonces = _verify_replay_quality_cell_evidence(
            replay_cell,
            original,
            candidate_cell,
            target_by_image[original["image_id"]],
            preflight_record,
            outdir=context["outdir"],
            regenerated_artifact_root=context["regenerated_root"],
            captured_source_root=captured_source_root,
            replay_source_authority=replay_source_authority,
            phase_b_sandbox_attestation_sha256=None,
        )
        expected_reconstructed = {
            "image_id": candidate_cell["image_id"],
            "bit_tuple": candidate_cell["bit_tuple"],
            "q_name": q_name,
            "formats": formats,
            "primary_metrics": metrics["primary"],
            "variants": replay_variants,
            "selection": replay_selection,
        }
        if reconstructed != expected_reconstructed:
            raise LifecycleError(
                "replay quality captured reduction does not recompute"
            )
        replay_cells.append(replay_cell)
        row_inputs.append(
            {
                "quality_cell": reconstructed,
                "candidate_cell": candidate_cell,
            }
        )
        if nonces != all_nonces[-len(nonces) :]:
            raise LifecycleError(
                "replay quality verifier/broker launch order drift"
            )
    if len(all_nonces) != len(set(all_nonces)):
        raise LifecycleError("replay quality launch nonce reuse")
    return replay_cells, row_inputs


def _replay_resource_schedule_plan(
    *,
    kind: str,
    image_id: str,
    bits: Sequence[int],
    schedule: Sequence[decode_worker.LaunchSpec],
    session_sha256: str,
    arm_paths: Mapping[str, str],
    arm_records: Mapping[str, Mapping[str, Any]],
    quality_results_sha256: str,
) -> dict[str, Any]:
    if (
        {launch.arm for launch in schedule} != set(arm_paths)
        or set(arm_paths) != set(arm_records)
    ):
        raise LifecycleError("resource plan arm authority is incomplete")
    tasks: list[dict[str, Any]] = []
    for launch in schedule:
        identity = {
            "quality_results_sha256": quality_results_sha256,
            "kind": kind,
            "image_id": image_id,
            "bit_tuple": list(bits),
            "session_sha256": session_sha256,
            "launch": launch.to_dict(),
            "blob_relative_path": arm_paths[launch.arm],
        }
        tasks.append(
            {
                "task_id": _sha256_bytes(_canonical_json(identity)),
                "launch": launch.to_dict(),
                "arm": launch.arm,
                "blob_relative_path": arm_paths[launch.arm],
                "expected_stream": dict(arm_records[launch.arm]),
            }
        )
    return {
        "kind": kind,
        "measurement_session_sha256": session_sha256,
        "schedule": [launch.to_dict() for launch in schedule],
        "tasks": tasks,
    }


def _replay_resource_plan_body(
    *,
    context: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    row_inputs: Sequence[Mapping[str, Any]],
    quality_plan_sha256: str,
    quality_results_sha256: str,
) -> dict[str, Any]:
    invalid = next(
        (
            item["quality_cell"]
            for item in row_inputs
            if item["quality_cell"]["selection"]["state"] == "invalid"
        ),
        None,
    )
    if invalid is not None:
        cells: list[dict[str, Any]] = []
        state = "inapplicable_invalid_quality"
        all_primary = False
    else:
        state = "complete_resource_plan"
        all_primary = all(
            item["quality_cell"]["selection"]["state"] == "selected"
            for item in row_inputs
        )
        quality_stage_sha256 = str(
            original_quality["quality_stage_sha256"]
        )
        cells = []
        for item in row_inputs:
            quality_cell = item["quality_cell"]
            candidate_cell = item["candidate_cell"]
            image_id = str(quality_cell["image_id"])
            bits = tuple(quality_cell["bit_tuple"])
            image_index = actual.FROZEN_IMAGES.index(image_id)
            tuple_index = actual.FROZEN_TUPLES.index(bits)
            execution_record = candidate_cell["execution"]
            base = _safe_relative_path(
                candidate_cell["relative_path"],
                "resource plan candidate-cell path",
            )
            secondary_session = _measurement_session_sha256(
                kind="secondary",
                image_id=image_id,
                bits=bits,
                quality_stage_sha256=quality_stage_sha256,
                replay_phase_sha256=str(
                    context["phase_a"]["replay_phase_a_sha256"]
                ),
            )
            secondary = _replay_resource_schedule_plan(
                kind="secondary",
                image_id=image_id,
                bits=bits,
                schedule=decode_worker.secondary_schedule(
                    image_index, tuple_index
                ),
                session_sha256=secondary_session,
                arm_paths={
                    "sspl1": (base / "exact/sspl1.bin").as_posix(),
                    "ssp2l": (base / "exact/ssp2l.bin").as_posix(),
                },
                arm_records={
                    "sspl1": _decoded_stream_record(
                        execution_record, "sspl1"
                    ),
                    "ssp2l": _decoded_stream_record(
                        execution_record, "ssp2l"
                    ),
                },
                quality_results_sha256=quality_results_sha256,
            )
            primary = None
            if all_primary:
                selected = str(
                    quality_cell["selection"]["selected_label"]
                )
                q_name = str(quality_cell["q_name"])
                primary_session = _measurement_session_sha256(
                    kind="primary",
                    image_id=image_id,
                    bits=bits,
                    quality_stage_sha256=quality_stage_sha256,
                    candidate=selected,
                    q_name=q_name,
                    replay_phase_sha256=str(
                        context["phase_a"]["replay_phase_a_sha256"]
                    ),
                )
                primary = _replay_resource_schedule_plan(
                    kind="primary",
                    image_id=image_id,
                    bits=bits,
                    schedule=decode_worker.primary_schedule(
                        image_index, tuple_index
                    ),
                    session_sha256=primary_session,
                    arm_paths={
                        "candidate": (
                            base / f"variants/{selected}.ssp2v"
                        ).as_posix(),
                        "q": (base / f"exact/{q_name}.bin").as_posix(),
                    },
                    arm_records={
                        "candidate": _decoded_stream_record(
                            execution_record, selected
                        ),
                        "q": _decoded_stream_record(
                            execution_record, q_name
                        ),
                    },
                    quality_results_sha256=quality_results_sha256,
                )
            cells.append(
                {
                    "image_id": image_id,
                    "bit_tuple": list(bits),
                    "primary": primary,
                    "secondary": secondary,
                }
            )
    task_ids = [
        task["task_id"]
        for cell in cells
        for schedule in (cell["secondary"], cell["primary"])
        if schedule is not None
        for task in schedule["tasks"]
    ]
    if len(task_ids) != len(set(task_ids)):
        raise LifecycleError("resource plan task identifier reuse")
    return actual.seal_record(
        {
            "schema": "structsplat.comp011.replay-resource-plan.v1",
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": source_manifest()[
                "source_manifest_sha256"
            ],
            "replay_phase_a_sha256": context["phase_a"][
                "replay_phase_a_sha256"
            ],
            "quality_plan_sha256": quality_plan_sha256,
            "quality_results_sha256": quality_results_sha256,
            "state": state,
            "cells": cells,
            "cell_count": len(cells),
            "task_count": len(task_ids),
            "primary_schedules_applicable": all_primary,
            "scientific_summary_deferred": True,
            "confirmation_payloads_accessed": False,
        },
        "resource_plan_sha256",
    )


def replay_phase_b_quality_reduce_worker(
    outdir: Path,
    phase_a_path: Path,
    plan_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    """Captured-source pass that reduces quality and authorizes resources."""

    context = _replay_phase_b_candidate_context(outdir, phase_a_path)
    plan = _read_canonical_object(plan_path)
    results = _read_canonical_object(results_path)
    targets = verify_target_copy(context["outdir"], captured_replay=True)
    original_quality = verify_quality_stage(
        context["outdir"], captured_replay=True
    )
    preflight_record = verify_preflight(
        context["outdir"], captured_replay=True
    )
    _verify_replay_quality_plan(
        plan,
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight_record,
    )
    captured_source_root = _captured_replay_source_root()
    if captured_source_root is None:
        raise LifecycleError(
            "quality reduction lacks captured-source authority"
        )
    replay_source_authority = _derive_replay_captured_source_authority(
        context["outdir"], preflight_record, captured_source_root
    )
    quality_replay, row_inputs = _reduce_replay_quality_results(
        plan,
        results,
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight_record,
        captured_source_root=captured_source_root,
        replay_source_authority=replay_source_authority,
    )
    resource_plan = _replay_resource_plan_body(
        context=context,
        original_quality=original_quality,
        row_inputs=row_inputs,
        quality_plan_sha256=str(plan["quality_plan_sha256"]),
        quality_results_sha256=str(results["quality_results_sha256"]),
    )
    return actual.seal_record(
        {
            "schema": REPLAY_QUALITY_AUTHORIZATION_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": source_manifest()[
                "source_manifest_sha256"
            ],
            "replay_phase_a_sha256": context["phase_a"][
                "replay_phase_a_sha256"
            ],
            "quality_plan_sha256": plan["quality_plan_sha256"],
            "quality_results_sha256": results["quality_results_sha256"],
            "quality_replay": quality_replay,
            "resource_plan": resource_plan,
            "target_copy_sha256": targets["target_copy_sha256"],
            "quality_stage_sha256": original_quality[
                "quality_stage_sha256"
            ],
            "captured_source_authority": replay_source_authority,
            "targets_opened": True,
            "confirmation_payloads_accessed": False,
        },
        "quality_authorization_sha256",
    )


def _verify_replay_quality_authorization(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    results: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    captured_source_root: Path,
    replay_source_authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual.verify_record(authorization, "quality_authorization_sha256")
    _verify_replay_quality_plan(
        plan,
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight_record,
    )
    quality_replay, row_inputs = _reduce_replay_quality_results(
        plan,
        results,
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight_record,
        captured_source_root=captured_source_root,
        replay_source_authority=replay_source_authority,
    )
    resource_plan = _replay_resource_plan_body(
        context=context,
        original_quality=original_quality,
        row_inputs=row_inputs,
        quality_plan_sha256=str(plan["quality_plan_sha256"]),
        quality_results_sha256=str(results["quality_results_sha256"]),
    )
    expected = actual.seal_record(
        {
            "schema": REPLAY_QUALITY_AUTHORIZATION_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": source_manifest()[
                "source_manifest_sha256"
            ],
            "replay_phase_a_sha256": context["phase_a"][
                "replay_phase_a_sha256"
            ],
            "quality_plan_sha256": plan["quality_plan_sha256"],
            "quality_results_sha256": results["quality_results_sha256"],
            "quality_replay": quality_replay,
            "resource_plan": resource_plan,
            "target_copy_sha256": targets["target_copy_sha256"],
            "quality_stage_sha256": original_quality[
                "quality_stage_sha256"
            ],
            "captured_source_authority": dict(replay_source_authority),
            "targets_opened": True,
            "confirmation_payloads_accessed": False,
        },
        "quality_authorization_sha256",
    )
    if dict(authorization) != expected:
        raise LifecycleError(
            "replay quality authorization differs from captured reduction"
        )
    return quality_replay, row_inputs


def _resource_plan_arm_bindings(
    schedule_plan: Mapping[str, Any],
) -> tuple[
    tuple[decode_worker.LaunchSpec, ...],
    dict[str, str],
    dict[str, Mapping[str, Any]],
]:
    tasks = schedule_plan.get("tasks")
    schedule_values = schedule_plan.get("schedule")
    if (
        not isinstance(tasks, list)
        or not isinstance(schedule_values, list)
        or len(tasks) != len(schedule_values)
    ):
        raise LifecycleError("resource schedule plan is malformed")
    launches = tuple(
        decode_worker.LaunchSpec(**dict(value)).validated()
        for value in schedule_values
    )
    arm_paths: dict[str, str] = {}
    arm_records: dict[str, Mapping[str, Any]] = {}
    for task, launch in zip(tasks, launches, strict=True):
        if (
            not isinstance(task, Mapping)
            or task.get("launch") != launch.to_dict()
            or task.get("arm") != launch.arm
        ):
            raise LifecycleError("resource plan task/launch drift")
        path = str(task["blob_relative_path"])
        record = task["expected_stream"]
        if not isinstance(record, Mapping):
            raise LifecycleError("resource plan stream authority is absent")
        if launch.arm in arm_paths and (
            arm_paths[launch.arm] != path
            or arm_records[launch.arm] != record
        ):
            raise LifecycleError("resource plan arm authority changed by launch")
        arm_paths[launch.arm] = path
        arm_records[launch.arm] = record
    return launches, arm_paths, arm_records


def _execute_replay_resource_plan(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    quality_results: Mapping[str, Any],
    *,
    outdir: Path,
    context: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    captured_source_root: Path,
    replay_source_authority: Mapping[str, Any],
    system_read_only: Sequence[Path],
) -> dict[str, Any]:
    """Trusted host broker for resource samples; summaries remain captured."""

    if os.environ.get(sandbox.SANDBOX_ATTESTATION_ENV) is not None:
        raise LifecycleError("resource broker must execute outside every sandbox")
    _verify_replay_quality_authorization(
        authorization,
        plan,
        quality_results,
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight_record,
        captured_source_root=captured_source_root,
        replay_source_authority=replay_source_authority,
    )
    resource_plan = authorization["resource_plan"]
    actual.verify_record(resource_plan, "resource_plan_sha256")
    regenerated_root = _lexical(context["regenerated_root"])
    native_record = preflight_record["native"]["arithmetic"]
    native_path = (
        regenerated_root / "runtime/native" / str(native_record["path"])
    )
    if resource_plan["cells"]:
        _verify_regular_record(native_path, native_record)
    result_cells: list[dict[str, Any]] = []
    nonces: list[str] = []
    pids: list[int] = []
    for cell in resource_plan["cells"]:
        result_cell: dict[str, Any] = {
            "image_id": cell["image_id"],
            "bit_tuple": cell["bit_tuple"],
            "primary": None,
            "secondary": None,
        }
        for kind in ("secondary", "primary"):
            schedule_plan = cell[kind]
            if schedule_plan is None:
                continue
            launches, relative_paths, arm_records = (
                _resource_plan_arm_bindings(schedule_plan)
            )
            samples, launch_bundles = _run_decode_schedule_samples(
                outdir=regenerated_root,
                schedule=launches,
                arm_paths={
                    arm: _artifact_path(
                        regenerated_root,
                        relative,
                        "resource broker blob path",
                    )
                    for arm, relative in relative_paths.items()
                },
                arm_records=arm_records,
                native_path=native_path,
                native_record=native_record,
                session_sha256=str(
                    schedule_plan["measurement_session_sha256"]
                ),
                worker_source_root=captured_source_root,
                worker_system_read_only=system_read_only,
            )
            task_results: list[dict[str, Any]] = []
            for planned, sample, bundle in zip(
                schedule_plan["tasks"],
                samples,
                launch_bundles,
                strict=True,
            ):
                _verify_launch_bundle(
                    bundle,
                    sample,
                    "canonical_worker_sha256",
                    expected_parent_sandbox_attestation_sha256=None,
                )
                nonces.append(str(bundle["launch"]["nonce"]))
                pids.append(int(bundle["attestation"]["pid"]))
                task_results.append(
                    {
                        "task_id": planned["task_id"],
                        "sample": sample,
                        "launch_bundle": bundle,
                    }
                )
            result_cell[kind] = {
                "kind": kind,
                "measurement_session_sha256": schedule_plan[
                    "measurement_session_sha256"
                ],
                "tasks": task_results,
            }
        result_cells.append(result_cell)
    if len(nonces) != len(set(nonces)) or len(pids) != len(set(pids)):
        raise LifecycleError("resource broker fresh-worker identity reuse")
    return actual.seal_record(
        {
            "schema": REPLAY_RESOURCE_RESULTS_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "quality_authorization_sha256": authorization[
                "quality_authorization_sha256"
            ],
            "resource_plan_sha256": resource_plan[
                "resource_plan_sha256"
            ],
            "cells": result_cells,
            "cell_count": len(result_cells),
            "task_count": resource_plan["task_count"],
            "all_workers_fresh_root": True,
            "scientific_summary_performed": False,
            "confirmation_payloads_accessed": False,
        },
        "resource_results_sha256",
    )


def _reduce_replay_resource_results(
    resource_results: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    context: Mapping[str, Any],
    row_inputs: Sequence[Mapping[str, Any]],
    original_quality: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    captured_source_root: Path,
    replay_source_authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual.verify_record(resource_results, "resource_results_sha256")
    resource_plan = authorization["resource_plan"]
    if (
        set(resource_results)
        != {
            "schema",
            "protocol",
            "task_sha256",
            "source_manifest_sha256",
            "quality_authorization_sha256",
            "resource_plan_sha256",
            "cells",
            "cell_count",
            "task_count",
            "all_workers_fresh_root",
            "scientific_summary_performed",
            "confirmation_payloads_accessed",
            "resource_results_sha256",
        }
        or resource_results.get("schema") != REPLAY_RESOURCE_RESULTS_SCHEMA
        or resource_results.get("protocol") != PROTOCOL
        or resource_results.get("task_sha256") != TASK_SHA256
        or resource_results.get("source_manifest_sha256")
        != authorization.get("source_manifest_sha256")
        or resource_results.get("quality_authorization_sha256")
        != authorization.get("quality_authorization_sha256")
        or resource_results.get("resource_plan_sha256")
        != resource_plan.get("resource_plan_sha256")
        or resource_results.get("cell_count")
        != resource_plan.get("cell_count")
        or resource_results.get("task_count")
        != resource_plan.get("task_count")
        or resource_results.get("all_workers_fresh_root") is not True
        or resource_results.get("scientific_summary_performed") is not False
        or resource_results.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("replay resource broker result authority drift")
    plan_cells = resource_plan.get("cells")
    result_cells = resource_results.get("cells")
    if (
        not isinstance(plan_cells, list)
        or not isinstance(result_cells, list)
        or len(plan_cells) != len(result_cells)
    ):
        raise LifecycleError("replay resource result grid is incomplete")
    regenerated_root = _lexical(context["regenerated_root"])
    replay_rows: list[dict[str, Any]] = []
    resource_records: list[dict[str, Any]] = []
    all_nonces: list[str] = []
    for plan_cell, result_cell, row_input in zip(
        plan_cells, result_cells, row_inputs, strict=True
    ):
        identity = _identity(plan_cell, "resource plan cell")
        if (
            identity != _identity(result_cell, "resource result cell")
            or identity
            != _identity(row_input["quality_cell"], "resource quality cell")
            or set(result_cell)
            != {"image_id", "bit_tuple", "primary", "secondary"}
        ):
            raise LifecycleError("replay resource cell identity/key drift")
        image_id = identity[0]
        bits = identity[1]
        image_index = actual.FROZEN_IMAGES.index(image_id)
        tuple_index = actual.FROZEN_TUPLES.index(bits)
        complete_schedules: dict[str, Any] = {}
        for kind in ("secondary", "primary"):
            schedule_plan = plan_cell[kind]
            schedule_result = result_cell[kind]
            if schedule_plan is None:
                if schedule_result is not None:
                    raise LifecycleError(
                        "replay resource ran an inapplicable schedule"
                    )
                complete_schedules[kind] = None
                continue
            if (
                not isinstance(schedule_result, Mapping)
                or set(schedule_result)
                != {"kind", "measurement_session_sha256", "tasks"}
                or schedule_result.get("kind") != kind
                or schedule_result.get("measurement_session_sha256")
                != schedule_plan.get("measurement_session_sha256")
            ):
                raise LifecycleError(
                    "replay resource schedule result binding drift"
                )
            launches, relative_paths, arm_records = (
                _resource_plan_arm_bindings(schedule_plan)
            )
            planned_tasks = schedule_plan["tasks"]
            returned_tasks = schedule_result.get("tasks")
            if (
                not isinstance(returned_tasks, list)
                or len(returned_tasks) != len(planned_tasks)
            ):
                raise LifecycleError(
                    "replay resource task result grid drift"
                )
            samples: list[dict[str, Any]] = []
            launch_bundles: list[dict[str, Any]] = []
            for planned, returned in zip(
                planned_tasks, returned_tasks, strict=True
            ):
                if (
                    not isinstance(returned, Mapping)
                    or set(returned)
                    != {"task_id", "sample", "launch_bundle"}
                    or returned.get("task_id") != planned.get("task_id")
                    or not isinstance(returned.get("sample"), Mapping)
                    or not isinstance(returned.get("launch_bundle"), Mapping)
                ):
                    raise LifecycleError(
                        "replay resource task/result binding drift"
                    )
                samples.append(dict(returned["sample"]))
                launch_bundles.append(dict(returned["launch_bundle"]))
            summary = decode_worker.summarize_schedule(
                samples,
                launches,
                image_index=image_index,
                tuple_index=tuple_index,
                measurement_session_sha256=str(
                    schedule_plan["measurement_session_sha256"]
                ),
                expected_blob_sha256_by_arm={
                    arm: str(record["blob_sha256"])
                    for arm, record in arm_records.items()
                },
            )
            complete = {
                "schedule": schedule_plan["schedule"],
                "samples": samples,
                "launches": launch_bundles,
                "summary": summary,
                "fresh_processes": True,
            }
            verified_nonces = _verify_replay_resource_schedule(
                complete,
                frozen_schedule=launches,
                image_index=image_index,
                tuple_index=tuple_index,
                measurement_session_sha256=str(
                    schedule_plan["measurement_session_sha256"]
                ),
                expected_streams=arm_records,
                expected_blob_paths=relative_paths,
                artifact_root=regenerated_root,
                expected_source_root=captured_source_root,
                preflight_record=preflight_record,
                replay_source_authority=replay_source_authority,
                phase_b_sandbox_attestation_sha256=None,
            )
            all_nonces.extend(verified_nonces)
            complete_schedules[kind] = complete
        quality_cell = row_input["quality_cell"]
        primary_resource, secondary_resource = _resource_records(
            quality_cell,
            complete_schedules["primary"],
            complete_schedules["secondary"],
        )
        replay_rows.append(
            _actual_row(
                quality_cell, primary_resource, secondary_resource
            )
        )
        resource_records.append(
            {
                "image_id": image_id,
                "bit_tuple": list(bits),
                "primary": complete_schedules["primary"],
                "secondary": complete_schedules["secondary"],
            }
        )
    if len(all_nonces) != len(set(all_nonces)):
        raise LifecycleError("replay resource launch nonce reuse")
    return resource_records, replay_rows


def _replay_phase_b_final_record(
    *,
    context: Mapping[str, Any],
    plan: Mapping[str, Any],
    quality_results: Mapping[str, Any],
    authorization: Mapping[str, Any],
    resource_results: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
    original_benchmark: Mapping[str, Any],
    original_analysis_record: Mapping[str, Any],
    original_analysis: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    captured_source_root: Path,
    replay_source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically reduce every persisted phase-B intermediate."""

    quality_replay, row_inputs = _verify_replay_quality_authorization(
        authorization,
        plan,
        quality_results,
        context=context,
        targets=targets,
        original_quality=original_quality,
        preflight_record=preflight_record,
        captured_source_root=captured_source_root,
        replay_source_authority=replay_source_authority,
    )
    invalid = next(
        (
            item["quality_cell"]
            for item in row_inputs
            if item["quality_cell"]["selection"]["state"] == "invalid"
        ),
        None,
    )
    if invalid is not None:
        actual.verify_record(resource_results, "resource_results_sha256")
        if (
            set(resource_results)
            != {
                "schema",
                "protocol",
                "task_sha256",
                "source_manifest_sha256",
                "quality_authorization_sha256",
                "resource_plan_sha256",
                "cells",
                "cell_count",
                "task_count",
                "all_workers_fresh_root",
                "scientific_summary_performed",
                "confirmation_payloads_accessed",
                "resource_results_sha256",
            }
            or authorization["resource_plan"]["state"]
            != "inapplicable_invalid_quality"
            or resource_results.get("schema")
            != REPLAY_RESOURCE_RESULTS_SCHEMA
            or resource_results.get("protocol") != PROTOCOL
            or resource_results.get("task_sha256") != TASK_SHA256
            or resource_results.get("source_manifest_sha256")
            != authorization.get("source_manifest_sha256")
            or resource_results.get("quality_authorization_sha256")
            != authorization.get("quality_authorization_sha256")
            or resource_results.get("resource_plan_sha256")
            != authorization["resource_plan"].get("resource_plan_sha256")
            or resource_results.get("cells") != []
            or resource_results.get("cell_count") != 0
            or resource_results.get("task_count") != 0
            or resource_results.get("all_workers_fresh_root") is not True
            or resource_results.get("scientific_summary_performed")
            is not False
            or resource_results.get("confirmation_payloads_accessed")
            is not False
        ):
            raise LifecycleError(
                "invalid-quality replay contains forbidden resources"
            )
        resource_records: list[dict[str, Any]] = []
        replay_rows = [
            actual.seal_record(
                {
                    "schema": actual.ROW_SCHEMA,
                    "image_id": invalid["image_id"],
                    "bit_tuple": invalid["bit_tuple"],
                    "validity": "invalid",
                    "invalid_reason": str(
                        invalid["selection"]["invalid_reason"]
                    ),
                    "method_state": "invalid",
                    "lloyd_all_fixed": None,
                    "formats": None,
                    "primary_metrics": None,
                    "variants": None,
                    "primary_resource": None,
                    "secondary_resource": None,
                }
            )
        ]
    else:
        resource_records, replay_rows = _reduce_replay_resource_results(
            resource_results,
            authorization=authorization,
            context=context,
            row_inputs=row_inputs,
            original_quality=original_quality,
            preflight_record=preflight_record,
            captured_source_root=captured_source_root,
            replay_source_authority=replay_source_authority,
        )
    replay_analysis = actual.analyze_rows(replay_rows)
    replay_signature = _analysis_gate_signature(replay_analysis)
    if replay_signature != _analysis_gate_signature(original_analysis):
        raise LifecycleError(
            "captured replay frozen gate/decision signature drift"
        )
    return actual.seal_record(
        {
            "schema": REPLAY_PHASE_B_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": source_manifest()[
                "source_manifest_sha256"
            ],
            "candidate_replay": context["candidate_replay"],
            "replay_phase_a_sha256": context["phase_a"][
                "replay_phase_a_sha256"
            ],
            "quality_replay": quality_replay,
            "resource_replay": resource_records,
            "replay_rows": replay_rows,
            "replay_analysis": replay_analysis,
            "analysis_gate_signature": replay_signature,
            "analysis_record_sha256": original_analysis_record[
                "analysis_record_sha256"
            ],
            "benchmark_stage_sha256": original_benchmark[
                "benchmark_stage_sha256"
            ],
            "quality_plan_sha256": plan["quality_plan_sha256"],
            "quality_results_sha256": quality_results[
                "quality_results_sha256"
            ],
            "quality_authorization_sha256": authorization[
                "quality_authorization_sha256"
            ],
            "resource_plan_sha256": authorization["resource_plan"][
                "resource_plan_sha256"
            ],
            "resource_results_sha256": resource_results[
                "resource_results_sha256"
            ],
            "worker_dispatch": "host_derived_fresh_root_v1",
            "targets_opened": True,
            "confirmation_payloads_accessed": False,
        },
        "captured_replay_worker_sha256",
    )


def _verify_replay_phase_b_final_record(
    phase_b: Mapping[str, Any],
    **reduction_inputs: Any,
) -> None:
    expected = _replay_phase_b_final_record(**reduction_inputs)
    if dict(phase_b) != expected:
        raise LifecycleError(
            "replay phase-B differs from deterministic intermediate reduction"
        )


def replay_phase_b_final_reduce_worker(
    outdir: Path,
    phase_a_path: Path,
    plan_path: Path,
    quality_results_path: Path,
    authorization_path: Path,
    resource_results_path: Path,
) -> dict[str, Any]:
    """Captured-source final reduction; no worker launch is permitted here."""

    context = _replay_phase_b_candidate_context(outdir, phase_a_path)
    plan = _read_canonical_object(plan_path)
    quality_results = _read_canonical_object(quality_results_path)
    authorization = _read_canonical_object(authorization_path)
    resource_results = _read_canonical_object(resource_results_path)
    targets = verify_target_copy(context["outdir"], captured_replay=True)
    original_quality = verify_quality_stage(
        context["outdir"], captured_replay=True
    )
    original_benchmark = verify_benchmark_stage(
        context["outdir"], captured_replay=True
    )
    original_analysis_record = verify_analysis_stage(
        context["outdir"], captured_replay=True
    )
    original_analysis = _read_canonical_object(
        context["outdir"] / "analysis.json"
    )
    preflight_record = verify_preflight(
        context["outdir"], captured_replay=True
    )
    captured_source_root = _captured_replay_source_root()
    if captured_source_root is None:
        raise LifecycleError("final reduction lacks captured-source authority")
    replay_source_authority = _derive_replay_captured_source_authority(
        context["outdir"], preflight_record, captured_source_root
    )
    return _replay_phase_b_final_record(
        context=context,
        plan=plan,
        quality_results=quality_results,
        authorization=authorization,
        resource_results=resource_results,
        targets=targets,
        original_quality=original_quality,
        original_benchmark=original_benchmark,
        original_analysis_record=original_analysis_record,
        original_analysis=original_analysis,
        preflight_record=preflight_record,
        captured_source_root=captured_source_root,
        replay_source_authority=replay_source_authority,
    )


def _replay_quality_grid(
    outdir: Path,
    candidate_artifact_root: Path,
    preflight_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    targets: Mapping[str, Any],
    original_quality: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    native_path, native_record = _native_arithmetic_record(outdir, preflight_record)
    renderer_record = preflight_record["renderer"]
    renderer_record_path = outdir / "runtime/renderer/record.json"
    target_by_image = {target["image_id"]: target for target in targets["targets"]}
    expected_lpips = preflight_record["renderer_proof"]["same_worker_metric_repeat"][
        "lpips_state_sha256"
    ]
    replay_cells: list[dict[str, Any]] = []
    replay_rows_inputs: list[dict[str, Any]] = []
    for candidate_cell, original in zip(
        candidate["cells"], original_quality["cells"], strict=True
    ):
        selection = original["selection"]
        prefix = _decision_relevant_variant_prefix(original, candidate_cell)
        schedules = (
            ("primary", *prefix),
            (*reversed(prefix), "primary"),
            ("primary", *prefix),
        )
        metrics: dict[str, list[dict[str, float]]] = {
            "primary": [],
            **{label: [] for label in prefix},
        }
        tasks: list[dict[str, Any]] = []
        q_name = str(candidate_cell["execution"]["selected_q"]["name"])
        if original.get("q_name") != q_name:
            raise LifecycleError("quality replay original Q differs from regenerated execution")
        base = _artifact_path(
            candidate_artifact_root,
            candidate_cell["relative_path"],
            "regenerated candidate-cell path",
        )
        target = target_by_image[original["image_id"]]
        for repetition, order in enumerate(schedules):
            for order_index, arm in enumerate(order):
                name = q_name if arm == "primary" else arm
                relative = f"exact/{name}.bin" if arm == "primary" else f"variants/{name}.ssp2v"
                artifact = next(item for item in candidate_cell["artifacts"] if item["path"] == relative)
                symbol, boundary = _cold_decode_expectations(candidate_cell["execution"], name)
                task = _run_render_metric_task(
                    outdir=outdir,
                    renderer_record_path=renderer_record_path,
                    renderer_record=renderer_record,
                    blob_path=base / relative,
                    blob_sha256=artifact["sha256"],
                    symbol_sha256=symbol,
                    boundary_state_sha256=boundary,
                    native_path=native_path,
                    native_record=native_record,
                    target_record=target,
                    expected_lpips_state_sha256=expected_lpips,
                )
                value = dict(task["metric"]["metrics"])
                metrics[arm].append(value)
                tasks.append(
                    {
                        "repetition": repetition,
                        "order_index": order_index,
                        "arm": arm,
                        "stream_name": name,
                        **task,
                    }
                )
        original_variants = {
            variant["label"]: variant for variant in original["variants"]
        }
        full_metrics: dict[str, Sequence[Mapping[str, Any]]] = {
            label: (
                metrics[label]
                if label in metrics
                else original_variants[label]["metrics"]
            )
            for label in actual.VARIANT_ORDER
        }
        formats, replay_variants = _quality_scalar_inputs(candidate_cell, full_metrics)
        for variant in replay_variants:
            if variant["label"] in metrics:
                original_metrics = original_variants[variant["label"]]["metrics"]
                for endpoint in ("min", "max"):
                    if not quality.replay_endpoint_match(
                        _metric_endpoints(original_metrics, endpoint),
                        _metric_endpoints(metrics[variant["label"]], endpoint),
                    ):
                        raise LifecycleError("replay quality endpoint exceeds frozen tolerance")
                _verify_replay_quality_classification(
                    original["primary_metrics"],
                    original_metrics,
                    metrics["primary"],
                    metrics[variant["label"]],
                )
        for endpoint in ("min", "max"):
            if not quality.replay_endpoint_match(
                _metric_endpoints(original["primary_metrics"], endpoint),
                _metric_endpoints(metrics["primary"], endpoint),
            ):
                raise LifecycleError("replay primary quality endpoint exceeds frozen tolerance")
        replay_selection = actual.select_variant(metrics["primary"], replay_variants)
        for field in ("state", "invalid_reason", "selected_label", "selected_bytes", "ordered_labels"):
            if replay_selection.get(field) != selection.get(field):
                raise LifecycleError("replay selected variant/state differs from original")
        replay_cells.append(
            {
                "image_id": original["image_id"],
                "bit_tuple": original["bit_tuple"],
                "prefix": prefix,
                "schedule": [list(order) for order in schedules],
                "metrics": metrics,
                "metric_envelopes": _quality_metric_envelopes(metrics),
                "nongating_later_metrics": {
                    label: original_variants[label]["metrics"]
                    for label in actual.VARIANT_ORDER
                    if label not in metrics
                },
                "selection": replay_selection,
                "tasks": tasks,
                "all_endpoint_tolerances_pass": True,
                "all_qualification_classes_reproduced": True,
                "all_decision_exclusion_bands_clear": True,
            }
        )
        replay_rows_inputs.append(
            {
                "quality_cell": {
                    "image_id": candidate_cell["image_id"],
                    "bit_tuple": candidate_cell["bit_tuple"],
                    "q_name": q_name,
                    "formats": formats,
                    "primary_metrics": metrics["primary"],
                    "variants": replay_variants,
                    "selection": replay_selection,
                },
                "candidate_cell": candidate_cell,
            }
        )
    return replay_cells, replay_rows_inputs


def _analysis_gate_signature(analysis: Mapping[str, Any]) -> dict[str, Any]:
    def resource_signature(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        return {
            "state": value.get("state"),
            "invalid_reason": value.get("invalid_reason"),
            "gates": value.get("gates"),
            "pass": value.get("pass"),
        }

    return {
        "status": analysis.get("status"),
        "invalid_reason": analysis.get("invalid_reason"),
        "decision": analysis.get("decision"),
        "ssp2l_decision": analysis.get("ssp2l_decision"),
        "early_failure": analysis.get("early_failure"),
        "tuple_results": [
            {
                "state": result.get("state"),
                "invalid_reason": result.get("invalid_reason"),
                "bit_tuple": result.get("bit_tuple"),
                "actual_rate": (
                    {
                        "gates": result["actual_rate"].get("gates"),
                        "pass": result["actual_rate"].get("pass"),
                    }
                    if isinstance(result.get("actual_rate"), Mapping)
                    else None
                ),
                "operational_bundle": (
                    {
                        "gates": result["operational_bundle"].get("gates"),
                        "pass": result["operational_bundle"].get("pass"),
                    }
                    if isinstance(result.get("operational_bundle"), Mapping)
                    else None
                ),
                "quality": (
                    {
                        "gates": result["quality"].get("gates"),
                        "median_state": result["quality"].get("median_psnr", {}).get(
                            "state"
                        ),
                    }
                    if isinstance(result.get("quality"), Mapping)
                    else None
                ),
                "convergence": (
                    {"gates": result["convergence"].get("gates")}
                    if isinstance(result.get("convergence"), Mapping)
                    else None
                ),
                "all_gates_pass": result.get("all_gates_pass"),
                "resources": resource_signature(result.get("resources")),
            }
            for result in analysis.get("tuple_results", [])
        ],
        "secondary_tuple_results": [
            {
                "state": result.get("state"),
                "invalid_reason": result.get("invalid_reason"),
                "bit_tuple": result.get("bit_tuple"),
                "gates": result.get("gates"),
                "pass": result.get("pass"),
                "resources": resource_signature(result.get("resources")),
            }
            for result in analysis.get("secondary_tuple_results", [])
        ],
    }


def replay_phase_a_worker(outdir: Path) -> dict[str, Any]:
    """Captured-source, target-hidden candidate regeneration phase."""

    candidate_replay = _replay_candidate_set(_lexical(outdir))
    return actual.seal_record(
        {
            "schema": REPLAY_PHASE_A_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": source_manifest()["source_manifest_sha256"],
            "candidate_replay": candidate_replay,
            "targets_opened": False,
            "confirmation_payloads_accessed": False,
        },
        "replay_phase_a_sha256",
    )


def replay_phase_b_worker(outdir: Path, phase_a_path: Path) -> dict[str, Any]:
    """Captured-source target/quality/resource phase authorized by persisted phase A."""

    outdir = _lexical(outdir)
    phase_a = _read_canonical_object(phase_a_path)
    actual.verify_record(phase_a, "replay_phase_a_sha256")
    if (
        phase_a.get("schema") != REPLAY_PHASE_A_SCHEMA
        or phase_a.get("source_manifest_sha256")
        != source_manifest()["source_manifest_sha256"]
        or phase_a.get("targets_opened") is not False
        or phase_a.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("persisted replay phase-A authorization is invalid")
    candidate_replay = phase_a["candidate_replay"]
    candidate = _read_canonical_object(outdir / _STAGE_MANIFEST_PATHS["fit-encode-cold-decode-dev"])
    if candidate_replay.get("state") != candidate.get("state") or (
        candidate.get("state") == "complete_candidate_set"
        and candidate_replay.get("candidate_set_sha256")
        != candidate.get("candidate_set_sha256")
    ):
        raise LifecycleError("replay phase-A candidate authorization differs from candidate stage")
    replayed_cells = candidate_replay.get("cells")
    original_cells = candidate.get("cells")
    if (
        not isinstance(replayed_cells, list)
        or not isinstance(original_cells, list)
        or len(replayed_cells) != len(original_cells)
    ):
        raise LifecycleError("replay phase-A candidate cell inventory is incomplete")
    replayed_candidate = {
        "state": candidate["state"],
        "candidate_set_sha256": candidate_replay.get("candidate_set_sha256"),
        "cells": [
            {
                "image_id": regenerated["image_id"],
                "bit_tuple": regenerated["bit_tuple"],
                "state": regenerated["state"],
                "relative_path": regenerated["relative_path"],
                "execution": regenerated["execution"],
                "artifacts": regenerated["artifacts"],
                "cold_decodes": regenerated["cold_decodes"],
            }
            for regenerated in replayed_cells
        ],
    }
    regenerated_root = Path(str(candidate_replay["regenerated_artifact_root"]))
    if candidate_replay["state"] == "valid_terminal_lloyd_failure":
        original_analysis_record = _read_canonical_object(outdir / "analysis_record.json")
        original_analysis = _read_canonical_object(outdir / "analysis.json")
        original_benchmark = _read_canonical_object(outdir / "benchmark_stage.json")
        actual.verify_record(original_benchmark, "benchmark_stage_sha256")
        early_row = actual.seal_record(
            {
                "schema": actual.ROW_SCHEMA,
                "image_id": replayed_cells[-1]["image_id"],
                "bit_tuple": replayed_cells[-1]["bit_tuple"],
                "validity": "valid",
                "invalid_reason": None,
                "method_state": "lloyd_failure",
                "lloyd_all_fixed": False,
                "formats": None,
                "primary_metrics": None,
                "variants": None,
                "primary_resource": None,
                "secondary_resource": None,
            }
        )
        replay_analysis = actual.analyze_rows([early_row])
        if _analysis_gate_signature(replay_analysis) != _analysis_gate_signature(original_analysis):
            raise LifecycleError("terminal Lloyd replay decision drift")
        return actual.seal_record(
            {
                "schema": REPLAY_PHASE_B_SCHEMA,
                "protocol": PROTOCOL,
                "task_sha256": TASK_SHA256,
                "source_manifest_sha256": source_manifest()["source_manifest_sha256"],
                "candidate_replay": candidate_replay,
                "replay_phase_a_sha256": phase_a["replay_phase_a_sha256"],
                "quality_replay": None,
                "resource_replay": None,
                "replay_rows": [early_row],
                "replay_analysis": replay_analysis,
                "analysis_gate_signature": _analysis_gate_signature(replay_analysis),
                "analysis_record_sha256": original_analysis_record["analysis_record_sha256"],
                "benchmark_stage_sha256": original_benchmark["benchmark_stage_sha256"],
                "targets_opened": False,
                "confirmation_payloads_accessed": False,
            },
            "captured_replay_worker_sha256",
        )

    # First legal replay target-copy verification occurs only after exact candidate-set match.
    targets = verify_target_copy(outdir, captured_replay=True)
    original_quality = verify_quality_stage(outdir, captured_replay=True)
    original_benchmark = verify_benchmark_stage(outdir, captured_replay=True)
    original_analysis_record = verify_analysis_stage(outdir, captured_replay=True)
    original_analysis = _read_canonical_object(outdir / "analysis.json")
    preflight_record = verify_preflight(outdir, captured_replay=True)
    replay_quality, row_inputs = _replay_quality_grid(
        outdir,
        regenerated_root,
        preflight_record,
        replayed_candidate,
        targets,
        original_quality,
    )
    all_primary = all(
        item["quality_cell"]["selection"]["state"] == "selected" for item in row_inputs
    )
    invalid = next(
        (
            item["quality_cell"]
            for item in row_inputs
            if item["quality_cell"]["selection"]["state"] == "invalid"
        ),
        None,
    )
    resource_records: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    if invalid is not None:
        replay_rows = [
            actual.seal_record(
                {
                    "schema": actual.ROW_SCHEMA,
                    "image_id": invalid["image_id"],
                    "bit_tuple": invalid["bit_tuple"],
                    "validity": "invalid",
                    "invalid_reason": str(invalid["selection"]["invalid_reason"]),
                    "method_state": "invalid",
                    "lloyd_all_fixed": None,
                    "formats": None,
                    "primary_metrics": None,
                    "variants": None,
                    "primary_resource": None,
                    "secondary_resource": None,
                }
            )
        ]
    else:
        native_record = preflight_record["native"]["arithmetic"]
        native_path = regenerated_root / "runtime/native" / str(native_record["path"])
        _verify_regular_record(native_path, native_record)
        quality_stage_sha256 = str(original_quality["quality_stage_sha256"])
        for item in row_inputs:
            quality_cell = item["quality_cell"]
            candidate_cell = item["candidate_cell"]
            image_id = quality_cell["image_id"]
            bits = tuple(quality_cell["bit_tuple"])
            image_index = actual.FROZEN_IMAGES.index(image_id)
            tuple_index = actual.FROZEN_TUPLES.index(bits)
            execution_record = candidate_cell["execution"]
            base = _artifact_path(
                regenerated_root,
                candidate_cell["relative_path"],
                "regenerated candidate-cell path",
            )
            secondary_session = _measurement_session_sha256(
                kind="secondary",
                image_id=image_id,
                bits=bits,
                quality_stage_sha256=quality_stage_sha256,
                replay_phase_sha256=str(phase_a["replay_phase_a_sha256"]),
            )
            secondary = _run_decode_schedule(
                outdir=regenerated_root,
                schedule=decode_worker.secondary_schedule(image_index, tuple_index),
                arm_paths={
                    "sspl1": base / "exact/sspl1.bin",
                    "ssp2l": base / "exact/ssp2l.bin",
                },
                arm_records={
                    "sspl1": _decoded_stream_record(execution_record, "sspl1"),
                    "ssp2l": _decoded_stream_record(execution_record, "ssp2l"),
                },
                native_path=native_path,
                native_record=native_record,
                session_sha256=secondary_session,
                image_index=image_index,
                tuple_index=tuple_index,
            )
            primary = None
            if all_primary:
                selected = quality_cell["selection"]["selected_label"]
                q_name = quality_cell["q_name"]
                primary_session = _measurement_session_sha256(
                    kind="primary",
                    image_id=image_id,
                    bits=bits,
                    quality_stage_sha256=quality_stage_sha256,
                    candidate=selected,
                    q_name=q_name,
                    replay_phase_sha256=str(phase_a["replay_phase_a_sha256"]),
                )
                primary = _run_decode_schedule(
                    outdir=regenerated_root,
                    schedule=decode_worker.primary_schedule(image_index, tuple_index),
                    arm_paths={
                        "candidate": base / f"variants/{selected}.ssp2v",
                        "q": base / f"exact/{q_name}.bin",
                    },
                    arm_records={
                        "candidate": _decoded_stream_record(execution_record, selected),
                        "q": _decoded_stream_record(execution_record, q_name),
                    },
                    native_path=native_path,
                    native_record=native_record,
                    session_sha256=primary_session,
                    image_index=image_index,
                    tuple_index=tuple_index,
                )
            primary_resource, secondary_resource = _resource_records(
                quality_cell, primary, secondary
            )
            replay_rows.append(_actual_row(quality_cell, primary_resource, secondary_resource))
            resource_records.append(
                {
                    "image_id": image_id,
                    "bit_tuple": list(bits),
                    "primary": primary,
                    "secondary": secondary,
                }
            )
    replay_analysis = actual.analyze_rows(replay_rows)
    replay_signature = _analysis_gate_signature(replay_analysis)
    original_signature = _analysis_gate_signature(original_analysis)
    if replay_signature != original_signature:
        raise LifecycleError("captured replay frozen gate/decision signature drift")
    return actual.seal_record(
        {
            "schema": REPLAY_PHASE_B_SCHEMA,
            "protocol": PROTOCOL,
            "task_sha256": TASK_SHA256,
            "source_manifest_sha256": source_manifest()["source_manifest_sha256"],
            "candidate_replay": candidate_replay,
            "replay_phase_a_sha256": phase_a["replay_phase_a_sha256"],
            "quality_replay": replay_quality,
            "resource_replay": resource_records,
            "replay_rows": replay_rows,
            "replay_analysis": replay_analysis,
            "analysis_gate_signature": replay_signature,
            "analysis_record_sha256": original_analysis_record["analysis_record_sha256"],
            "benchmark_stage_sha256": original_benchmark["benchmark_stage_sha256"],
            "targets_opened": True,
            "confirmation_payloads_accessed": False,
        },
        "captured_replay_worker_sha256",
    )


_CAPTURED_REPLAY_SCRIPT = r"""
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
os.environ["COMP011_CAPTURED_REPLAY_SOURCE_ROOT"] = str(root)
sys.path[:0] = [str(root), str(root / "src")]
from benchmarks import ssp2v_actual_run as runner

pathlib.Path(runner.__file__).resolve(strict=True).relative_to(root)
mode = sys.argv[2]
if mode == "phase-a":
    record = runner.replay_phase_a_worker(pathlib.Path(sys.argv[3]))
elif mode == "phase-b":
    record = runner.replay_phase_b_worker(pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
elif mode == "phase-b-quality-plan":
    record = runner.replay_phase_b_quality_plan_worker(
        pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
    )
elif mode == "phase-b-quality-reduce":
    record = runner.replay_phase_b_quality_reduce_worker(
        pathlib.Path(sys.argv[3]),
        pathlib.Path(sys.argv[4]),
        pathlib.Path(sys.argv[5]),
        pathlib.Path(sys.argv[6]),
    )
elif mode == "phase-b-final-reduce":
    record = runner.replay_phase_b_final_reduce_worker(
        pathlib.Path(sys.argv[3]),
        pathlib.Path(sys.argv[4]),
        pathlib.Path(sys.argv[5]),
        pathlib.Path(sys.argv[6]),
        pathlib.Path(sys.argv[7]),
        pathlib.Path(sys.argv[8]),
    )
else:
    raise RuntimeError("unknown captured replay phase")
sys.stdout.buffer.write(runner._canonical_json(record))
sys.stdout.buffer.flush()
"""


def _derive_replay_captured_source_authority(
    outdir: Path,
    preflight_record: Mapping[str, Any],
    captured_source_root: Path,
) -> dict[str, Any]:
    """Bind a transient relocated root to durable preflight source receipts."""

    outdir = _lexical(outdir)
    source_root = _lexical(captured_source_root)
    if not source_root.is_absolute():
        raise LifecycleError("replay captured source root is not absolute")
    manifest = _read_canonical_object(outdir / "source_manifest.json")
    actual.verify_record(manifest, "source_manifest_sha256")
    transient_manifest = _verify_runtime_source_tree(
        source_root, expected_manifest=manifest
    )
    runtime_source = preflight_record.get("runtime_source")
    if not isinstance(runtime_source, Mapping):
        raise LifecycleError("replay durable runtime-source authority is absent")
    actual.verify_record(runtime_source, "runtime_source_sha256")
    archive_sha256 = _sha256_file(outdir / "source_snapshot.tar")
    durable_root = _verify_runtime_source_record(
        outdir,
        runtime_source,
        expected_manifest=manifest,
        expected_archive_sha256=archive_sha256,
    )
    launcher = runtime_source.get("launcher")
    manifest_launcher = next(
        (
            item
            for item in manifest.get("files", [])
            if item.get("path") == "benchmarks/ssp2v_landlock.py"
        ),
        None,
    )
    if (
        manifest.get("source_manifest_sha256")
        != preflight_record.get("source_manifest_sha256")
        or archive_sha256 != preflight_record.get("source_archive_sha256")
        or runtime_source.get("source_manifest_sha256")
        != manifest.get("source_manifest_sha256")
        or runtime_source.get("source_archive_sha256") != archive_sha256
        or transient_manifest != manifest
        or durable_root != outdir / _RUNTIME_SOURCE_RELATIVE
        or not isinstance(launcher, Mapping)
        or not isinstance(manifest_launcher, Mapping)
        or launcher.get("path") != "benchmarks/ssp2v_landlock.py"
        or launcher.get("bytes") != manifest_launcher.get("bytes")
        or launcher.get("sha256") != manifest_launcher.get("sha256")
    ):
        raise LifecycleError("replay captured source authority drift")
    return actual.seal_record(
        {
            "schema": "structsplat.comp011.replay-captured-source-authority.v1",
            "source_root": os.fspath(source_root),
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "source_archive_sha256": archive_sha256,
            "runtime_source_sha256": runtime_source["runtime_source_sha256"],
            "launcher": dict(launcher),
            "sealed_root_read_only": True,
            "transient_tree_reverified": True,
        },
        "replay_source_authority_sha256",
    )


def _verify_replay_captured_source_authority(
    authority: Mapping[str, Any],
    *,
    preflight_record: Mapping[str, Any],
    expected_source_root: Path,
) -> Path:
    """Verify a derived replay source authority without statting its transient root."""

    actual.verify_record(authority, "replay_source_authority_sha256")
    runtime_source = preflight_record.get("runtime_source")
    if isinstance(runtime_source, Mapping):
        actual.verify_record(runtime_source, "runtime_source_sha256")
    launcher = runtime_source.get("launcher") if isinstance(runtime_source, Mapping) else None
    root = _lexical(Path(str(authority.get("source_root"))))
    if (
        set(authority)
        != {
            "schema",
            "source_root",
            "source_manifest_sha256",
            "source_archive_sha256",
            "runtime_source_sha256",
            "launcher",
            "sealed_root_read_only",
            "transient_tree_reverified",
            "replay_source_authority_sha256",
        }
        or authority.get("schema")
        != "structsplat.comp011.replay-captured-source-authority.v1"
        or root != _lexical(expected_source_root)
        or authority.get("source_manifest_sha256")
        != preflight_record.get("source_manifest_sha256")
        or authority.get("source_archive_sha256")
        != preflight_record.get("source_archive_sha256")
        or not isinstance(runtime_source, Mapping)
        or authority.get("runtime_source_sha256")
        != runtime_source.get("runtime_source_sha256")
        or not isinstance(launcher, Mapping)
        or authority.get("launcher") != launcher
        or authority.get("sealed_root_read_only") is not True
        or authority.get("transient_tree_reverified") is not True
    ):
        raise LifecycleError("replay captured source authority binding drift")
    return root


def _verify_replay_phase_a(
    outdir: Path,
    phase_a: Mapping[str, Any],
    candidate: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    producer_source_root: Path | None = None,
    replay_source_authority: Mapping[str, Any] | None = None,
    phase_a_sandbox_attestation_sha256: str | None = None,
) -> None:
    phase_a_parent = _require_sha256(
        phase_a_sandbox_attestation_sha256,
        "replay phase-A sandbox attestation",
    )
    actual.verify_record(phase_a, "replay_phase_a_sha256")
    replay = phase_a.get("candidate_replay")
    if (
        phase_a.get("schema") != REPLAY_PHASE_A_SCHEMA
        or phase_a.get("protocol") != PROTOCOL
        or phase_a.get("task_sha256") != TASK_SHA256
        or phase_a.get("source_manifest_sha256")
        != preflight_record.get("source_manifest_sha256")
        or phase_a.get("targets_opened") is not False
        or phase_a.get("confirmation_payloads_accessed") is not False
        or not isinstance(replay, Mapping)
        or replay.get("state") != candidate.get("state")
        or replay.get("targets_opened_before_candidate_match") is not False
        or replay.get("all_complete_streams_byte_identical") is not True
        or replay.get("all_cold_decode_states_identical") is not True
    ):
        raise LifecycleError("captured replay phase-A schema/candidate/access proof drift")
    replay_cells = replay.get("cells")
    candidate_cells = candidate.get("cells")
    regenerated_root_value = replay.get("regenerated_artifact_root")
    if (
        not isinstance(replay_cells, list)
        or not isinstance(candidate_cells, list)
        or len(replay_cells) != replay.get("cell_count")
        or len(replay_cells) != len(candidate_cells)
        or not isinstance(regenerated_root_value, str)
        or not Path(regenerated_root_value).is_absolute()
        or _lexical(Path(regenerated_root_value)) != Path(regenerated_root_value)
        or Path(regenerated_root_value).name != "regenerated-artifacts"
        or (
            producer_source_root is not None
            and Path(regenerated_root_value)
            != _lexical(producer_source_root).parent
            / "phase-a-scratch/regenerated-artifacts"
        )
    ):
        raise LifecycleError("captured replay phase-A cell inventory drift")
    replay_nonces: list[str] = []
    cold_source_roots: set[Path] = set()
    regenerated_root = Path(regenerated_root_value)
    native_record = preflight_record["native"]["arithmetic"]
    native_relative_path = f"runtime/native/{native_record['path']}"
    worker_source = _decode_worker_source_record()
    for regenerated, original in zip(replay_cells, candidate_cells, strict=True):
        expected_relative = _candidate_cell_relative(
            str(regenerated.get("image_id")), regenerated.get("bit_tuple", [])
        ).as_posix()
        if (
            _identity(regenerated, "replay candidate cell")
            != _identity(original, "candidate cell")
            or regenerated.get("state") != original.get("state")
            or regenerated.get("relative_path") != expected_relative
            or regenerated.get("execution") != original.get("execution")
            or regenerated.get("artifacts") != original.get("artifacts")
        ):
            raise LifecycleError("captured replay phase-A deterministic cell drift")
        diagnostics = regenerated.get("producing_diagnostics")
        producing = diagnostics.get("producing_worker") if isinstance(diagnostics, Mapping) else None
        if not isinstance(diagnostics, Mapping) or not isinstance(producing, Mapping):
            raise LifecycleError("captured replay phase-A producing evidence is absent")
        cold = regenerated.get("cold_decodes")
        original_cold = original.get("cold_decodes")
        if not isinstance(cold, Mapping) or set(cold) != set(original_cold):
            raise LifecycleError("captured replay phase-A cold inventory drift")
        artifact_by_path = {
            item["path"]: item
            for item in regenerated["artifacts"]
            if isinstance(item, Mapping)
        }
        if len(artifact_by_path) != len(regenerated["artifacts"]):
            raise LifecycleError("captured replay phase-A artifact inventory drift")
        for path, bundle in cold.items():
            if not isinstance(bundle, Mapping) or set(bundle) != {
                "cold_seal",
                "sandbox_attestation",
                "launch",
            }:
                raise LifecycleError("captured replay phase-A cold launch bundle drift")
            cold_seal = bundle["cold_seal"]
            artifact = artifact_by_path.get(path)
            if not isinstance(cold_seal, Mapping) or not isinstance(artifact, Mapping):
                raise LifecycleError("captured replay phase-A cold artifact binding is absent")
            name = Path(path).stem
            symbol_sha256, boundary_sha256 = _cold_decode_expectations(
                regenerated["execution"], name
            )
            original_blob_path = _artifact_path(
                outdir,
                _safe_relative_path(expected_relative, "replay candidate-cell path")
                / _safe_relative_path(path, "replay candidate stream path"),
                "replay candidate stream path",
            )
            original_blob = _read_regular_no_follow(original_blob_path)
            components_sha256 = decode_worker.component_records_sha256(
                decode_worker.stream_component_records(original_blob)
            )
            decode_worker.verify_candidate_cold_seal(
                cold_seal,
                expected_blob_sha256=str(artifact["sha256"]),
                expected_blob_bytes=int(artifact["bytes"]),
                expected_symbol_sha256=symbol_sha256,
                expected_boundary_state_sha256=boundary_sha256,
                expected_stream_components_sha256=components_sha256,
            )
            _verify_launch_bundle(
                {
                    "attestation": bundle["sandbox_attestation"],
                    "launch": bundle["launch"],
                },
                cold_seal,
                "cold_seal_sha256",
                expected_parent_sandbox_attestation_sha256=phase_a_parent,
            )
            expected_blob_relative = f"{expected_relative}/{path}"
            expected_command = _candidate_cold_worker_command(
                artifact_root=regenerated_root,
                blob_relative_path=expected_blob_relative,
                native_relative_path=native_relative_path,
                blob_sha256=str(artifact["sha256"]),
                blob_bytes=int(artifact["bytes"]),
                native_record=native_record,
                worker_source=worker_source,
                symbol_sha256=symbol_sha256,
                boundary_state_sha256=boundary_sha256,
                stream_components_sha256=components_sha256,
            )
            if not isinstance(replay_source_authority, Mapping):
                raise LifecycleError(
                    "captured replay phase-A source authority is absent"
                )
            cold_source_roots.add(
                _verify_captured_decode_worker_policy(
                    bundle,
                    expected_command=expected_command,
                    artifact_root=regenerated_root,
                    blob_relative_path=expected_blob_relative,
                    native_relative_path=native_relative_path,
                    expected_source_root=producer_source_root,
                    preflight_record=preflight_record,
                    replay_source_authority=replay_source_authority,
                )
            )
            original_cold_seal = original_cold[path]["cold_seal"]
            if (
                cold_seal.get("blob", {}).get("artifact_relative_path")
                != f"{expected_relative}/{path}"
                or cold_seal.get("source_bindings")
                != original_cold_seal.get("source_bindings")
                or cold_seal.get("cold_decode_state_sha256")
                != original_cold_seal.get("cold_decode_state_sha256")
            ):
                raise LifecycleError("captured replay phase-A cold state drift")
            replay_nonces.append(str(bundle["launch"]["nonce"]))
        config = producing.get("config")
        worker = producing.get("worker")
        attestation = producing.get("sandbox_attestation")
        launch = producing.get("launch")
        denied_paths = producing.get("denied_read_probe_paths")
        if not all(
            isinstance(value, Mapping)
            for value in (config, worker, attestation, launch)
        ) or not isinstance(denied_paths, list):
            raise LifecycleError("captured replay phase-A producing launch is malformed")
        replay_authority = _authority_copy(outdir, preflight_record)
        comp009_authority = next(
            cell
            for cell in replay_authority["comp009"]["cells"]
            if _identity(cell, "COMP-009 authority")
            == _identity(regenerated, "replay candidate cell")
        )
        expected_upstream_probe = _lexical(
            Path(str(preflight_record["operational_upstream_roots"]["comp008"]))
        ) / _safe_relative_path(
            replay_authority["comp008"]["targets"][0]["relative_path"],
            "replay producing-worker denied target probe",
        )
        if not isinstance(replay_source_authority, Mapping):
            raise LifecycleError("captured replay phase-A source authority is absent")
        _verify_producer_launch_policy(
            outdir=outdir,
            image_id=str(regenerated["image_id"]),
            bits=regenerated["bit_tuple"],
            config=config,
            worker=worker,
            attestation=attestation,
            launch=launch,
            denied_read_probe_paths=denied_paths,
            expected_source_root=producer_source_root,
            expected_native_record=preflight_record["native"]["arithmetic"],
            expected_lloyd_record=preflight_record["native"]["lloyd"],
            expected_comp009_authority=comp009_authority,
            expected_captured_source_manifest_sha256=str(
                replay_authority["comp009"]["source_manifest_sha256"]
            ),
            expected_denied_read_probe_paths=[expected_upstream_probe],
            preflight_record=preflight_record,
            replay_source_authority=replay_source_authority,
            expected_parent_sandbox_attestation_sha256=phase_a_parent,
        )
        captured_provenance = diagnostics.get("captured_comp009_reproduction")
        producer_policy = config.get("producer_policy")
        if not isinstance(captured_provenance, Mapping) or not isinstance(
            producer_policy, Mapping
        ):
            raise LifecycleError(
                "captured replay phase-A nested COMP-009 evidence is absent"
            )
        _verify_captured_comp009_reproduction_policy(
            captured_provenance,
            producing_config=config,
            producing_policy=producer_policy,
            expected_authority=comp009_authority,
            producer_sandbox_attestation_sha256=str(
                attestation["attestation_sha256"]
            ),
        )
        _verify_launch_bundle(
            {"attestation": attestation, "launch": launch},
            worker,
            "producing_worker_sha256",
            expected_parent_sandbox_attestation_sha256=phase_a_parent,
        )
        actual.verify_record(worker, "producing_worker_sha256")
        if (
            worker.get("execution") != regenerated.get("execution")
            or worker.get("artifacts") != regenerated.get("artifacts")
            or worker.get("diagnostics")
            != {
                key: value
                for key, value in diagnostics.items()
                if key != "producing_worker"
            }
            or worker.get("target_or_pixel_payloads_opened") is not False
            or worker.get("confirmation_payloads_accessed") is not False
        ):
            raise LifecycleError("captured replay phase-A producing worker detached from cell")
    if len(replay_nonces) != len(set(replay_nonces)):
        raise LifecycleError("captured replay phase-A launch nonce reuse")
    if len(cold_source_roots) != 1:
        raise LifecycleError("captured replay phase-A cold source roots differ")
    if candidate.get("state") == "complete_candidate_set" and (
        replay.get("candidate_set_sha256") != candidate.get("candidate_set_sha256")
    ):
        raise LifecycleError("captured replay phase-A global candidate seal drift")


def _captured_worker_environment_semantics(
    environment: Mapping[str, Any],
    *,
    outdir: Path,
    captured_source_root: Path,
    tmpdir: Path | None,
    worker_scratch: Path | None = None,
    preflight_record: Mapping[str, Any] | None = None,
    require_replay_preflight_binding: bool = False,
) -> None:
    inherited_names = {
        "PATH",
        "LANG",
        "LC_ALL",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST",
        "CXX",
        "CC",
    }
    required: dict[str, str] = {
        **quality.FROZEN_THREAD_ENVIRONMENT,
        "LD_PRELOAD": quality.REQUIRED_RENDERER_PRELOAD,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "NO_PROXY": "*",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HOME": "/nonexistent-comp011-home",
        "TORCH_HOME": os.fspath(outdir / "runtime/metrics/torch"),
        "PYTHONPATH": os.pathsep.join(
            (
                os.fspath(outdir / "runtime/metrics/site"),
                os.fspath(captured_source_root),
                os.fspath(captured_source_root / "src"),
            )
        ),
    }
    durable_source_manifest_sha256 = (
        preflight_record.get("source_manifest_sha256")
        if isinstance(preflight_record, Mapping)
        else None
    )
    if isinstance(durable_source_manifest_sha256, str):
        required["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = _require_sha256(
            durable_source_manifest_sha256,
            "captured worker durable source manifest SHA-256",
        )
    elif (captured_source_root / "SOURCE_MANIFEST.json").is_file():
        required["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
            _verify_runtime_source_tree(captured_source_root)[
                "source_manifest_sha256"
            ]
        )
    if require_replay_preflight_binding:
        if not isinstance(preflight_record, Mapping):
            raise LifecycleError("replay preflight environment authority is absent")
        required[_REPLAY_PREFLIGHT_BINDING_ENV] = _require_sha256(
            preflight_record.get("preflight_binding_sha256"),
            "replay outer preflight binding",
        )
        required.update(_replay_path_inventory_environment())
    if tmpdir is not None:
        required["TMPDIR"] = os.fspath(tmpdir)
    if worker_scratch is not None:
        required["COMP011_WORKER_SCRATCH"] = os.fspath(worker_scratch)
    exact_base: dict[str, Any] | None = None
    if isinstance(preflight_record, Mapping):
        workers = preflight_record.get("sandboxed_workers")
        proof = workers.get("renderer_proof") if isinstance(workers, Mapping) else None
        launch = proof.get("launch") if isinstance(proof, Mapping) else None
        policy = launch.get("worker_policy") if isinstance(launch, Mapping) else None
        base = policy.get("environment") if isinstance(policy, Mapping) else None
        if isinstance(base, Mapping):
            exact_base = dict(base)
            exact_base.pop("TMPDIR", None)
            exact_base.pop("COMP011_WORKER_SCRATCH", None)
            exact_base.update(required)
        else:
            runtime = preflight_record.get("environment")
            renderer = preflight_record.get("renderer")
            abi = renderer.get("abi") if isinstance(renderer, Mapping) else None
            loader = abi.get("loader_environment") if isinstance(abi, Mapping) else None
            if not isinstance(runtime, Mapping) or not isinstance(loader, Mapping):
                raise LifecycleError(
                    "captured worker preflight environment authority is absent"
                )
            cuda_visible = runtime.get("cuda_visible_devices")
            loader_path = loader.get("LD_LIBRARY_PATH")
            if cuda_visible is None:
                if "CUDA_VISIBLE_DEVICES" in environment:
                    raise LifecycleError("captured worker CUDA device authority drift")
            elif environment.get("CUDA_VISIBLE_DEVICES") != cuda_visible:
                raise LifecycleError("captured worker CUDA device authority drift")
            if loader_path is None:
                if "LD_LIBRARY_PATH" in environment:
                    raise LifecycleError("captured worker loader environment authority drift")
            elif environment.get("LD_LIBRARY_PATH") != loader_path:
                raise LifecycleError("captured worker loader environment authority drift")
    if (
        not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items())
        or (exact_base is not None and dict(environment) != exact_base)
        or any(environment.get(key) != value for key, value in required.items())
        or set(environment) - set(required) - inherited_names
        or (tmpdir is None and "TMPDIR" in environment)
        or (worker_scratch is None and "COMP011_WORKER_SCRATCH" in environment)
    ):
        raise LifecycleError("captured nested worker environment semantics drift")


def _safe_system_read_only_policy_path(path: Path) -> bool:
    lexical = _lexical(path)
    fixed_candidates = (
        Path(sys.prefix),
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path("/etc/ld.so.cache"),
        Path("/etc/localtime"),
        Path("/etc/nvcc.profile"),
        Path("/dev/null"),
        Path("/dev/urandom"),
        Path("/proc/self/maps"),
        Path("/proc/self/status"),
        Path("/proc/self/stat"),
        Path("/proc/self/statm"),
        Path("/proc/self/cmdline"),
        Path("/proc/cpuinfo"),
        Path("/proc/meminfo"),
        Path("/proc/devices"),
        Path("/proc/sys/vm/mmap_min_addr"),
        Path("/proc/driver/nvidia"),
        Path("/sys/devices/system/cpu/online"),
        Path("/sys/devices/system/cpu/possible"),
        Path("/sys/devices/system/node"),
        Path("/sys/devices/system/memory/block_size_bytes"),
    )
    fixed = {
        _lexical(
            candidate
            if candidate.is_relative_to(Path("/proc/self"))
            else candidate.resolve(strict=False)
        )
        for candidate in fixed_candidates
    }
    if lexical in fixed:
        return True
    value = lexical.as_posix()
    return (
        re.fullmatch(r"/sys/devices/system/cpu/cpu(?:0|[1-9][0-9]*)/online", value)
        is not None
        or re.fullmatch(r"/sys/module/nvidia[0-9A-Za-z_]*", value) is not None
        or re.fullmatch(
            r"/sys/bus/pci/devices/[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:"
            r"[0-9A-Fa-f]{2}\.[0-7]/numa_node",
            value,
        )
        is not None
        or re.fullmatch(
            r"/sys/devices/pci[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}"
            r"(?:/[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7])+"
            r"/numa_node",
            value,
        )
        is not None
    )


def _verify_nested_policy_path_inventory(
    policy: Mapping[str, Any],
    *,
    exact_read_only: set[Path],
    exact_workspace_read_only: set[Path],
    exact_read_write: set[Path],
    captured_source_root: Path,
    outdir: Path,
    preflight_record: Mapping[str, Any],
    allow_gpu_read_write: bool,
) -> None:
    read_only = policy.get("read_only_request")
    read_write = policy.get("read_write_request")
    denied = policy.get("denied_read_probe_paths")
    if not isinstance(read_only, list) or not isinstance(read_write, list) or denied != []:
        raise LifecycleError("captured nested worker path policy is malformed")
    ro_paths = {_lexical(Path(str(value))) for value in read_only}
    rw_paths = {_lexical(Path(str(value))) for value in read_write}
    exact_ro = {_lexical(path) for path in exact_read_only}
    exact_workspace_ro = {_lexical(path) for path in exact_workspace_read_only}
    exact_rw = {_lexical(path) for path in exact_read_write}
    workspace = _lexical(captured_source_root).parent
    operational = preflight_record.get("operational_upstream_roots")
    if not isinstance(operational, Mapping):
        raise LifecycleError("captured nested worker upstream authority is absent")
    upstream_roots = {
        _lexical(Path(str(value))) for value in operational.values()
    }
    if (
        not exact_ro.issubset(ro_paths)
        or not exact_rw.issubset(rw_paths)
        or {path for path in ro_paths if path.is_relative_to(_lexical(outdir))}
        != {path for path in exact_ro if path.is_relative_to(_lexical(outdir))}
        or {path for path in ro_paths if path.is_relative_to(workspace)}
        != exact_workspace_ro
        or {path for path in rw_paths if path.is_relative_to(workspace)} != exact_rw
        or any(
            path == root or path.is_relative_to(root)
            for path in (*ro_paths, *rw_paths)
            for root in upstream_roots
        )
    ):
        raise LifecycleError("captured nested worker protected path capability drift")
    allowed_ro = exact_ro | exact_workspace_ro
    if any(
        path not in allowed_ro and not _safe_system_read_only_policy_path(path)
        for path in ro_paths
    ):
        raise LifecycleError("captured nested worker gained an unexpected read capability")
    allowed_gpu = {
        Path("/proc/self/task"),
        Path("/dev/nvidia0"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
    }
    if any(
        path not in exact_rw and (not allow_gpu_read_write or path not in allowed_gpu)
        for path in rw_paths
    ):
        raise LifecycleError("captured nested worker gained an unexpected write capability")


def _verify_replay_render_metric_policy(
    task: Mapping[str, Any],
    *,
    outdir: Path,
    captured_source_root: Path,
    regenerated_artifact_root: Path,
    candidate_cell: Mapping[str, Any],
    stream_name: str,
    expected_stream: Mapping[str, Any],
    render: Mapping[str, Any],
    target_record: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    replay_source_authority: Mapping[str, Any],
    expected_lpips_state_sha256: str,
    fresh_root_dispatch: bool = False,
) -> None:
    authority_root = _verify_replay_captured_source_authority(
        replay_source_authority,
        preflight_record=preflight_record,
        expected_source_root=captured_source_root,
    )
    render_launch = task.get("render_launch")
    metric_launch = task.get("metric_launch")
    if not isinstance(render_launch, Mapping) or not isinstance(metric_launch, Mapping):
        raise LifecycleError("replay quality launch receipts are absent")
    render_policy = render_launch.get("worker_policy")
    metric_policy = metric_launch.get("worker_policy")
    if not isinstance(render_policy, Mapping) or not isinstance(metric_policy, Mapping):
        raise LifecycleError("replay quality persisted worker policies are absent")
    relative_stream = (
        f"exact/{stream_name}.bin"
        if stream_name == candidate_cell["execution"]["selected_q"]["name"]
        else f"variants/{stream_name}.ssp2v"
    )
    blob_path = _lexical(regenerated_artifact_root) / _safe_relative_path(
        candidate_cell["relative_path"], "replay regenerated candidate-cell path"
    ) / _safe_relative_path(relative_stream, "replay regenerated stream path")
    native = preflight_record["native"]["arithmetic"]
    native_path = outdir / "runtime/native" / str(native["path"])
    renderer_path = outdir / "runtime/renderer/record.json"
    render_command = render_policy.get("command")
    if not isinstance(render_command, list) or len(render_command) != 24:
        raise LifecycleError("replay render command vector is malformed")
    render_path = Path(str(render_command[-1]))
    render_temp = render_path.parent
    expected_render_temp_parent = (
        captured_source_root.parent
        / "phase-b-scratch"
        / "quality-broker"
        if fresh_root_dispatch
        else captured_source_root.parent / "phase-b-scratch"
    )
    expected_render_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_render-arm",
        "--artifact-root",
        os.fspath(outdir),
        "--renderer-record",
        os.fspath(renderer_path),
        "--blob",
        os.fspath(blob_path),
        "--native",
        os.fspath(native_path),
        "--expected-blob-sha256",
        str(expected_stream["blob_sha256"]),
        "--expected-native-sha256",
        str(native["sha256"]),
        "--expected-native-bytes",
        str(native["bytes"]),
        "--expected-symbol-sha256",
        str(expected_stream["symbol_sha256"]),
        "--expected-boundary-state-sha256",
        str(expected_stream["boundary_state_sha256"]),
        "--output",
        os.fspath(render_path),
    ]
    if (
        render_command != expected_render_command
        or render_path.name != "render.npy"
        or render_temp.parent != expected_render_temp_parent
        or not render_temp.name.startswith("comp011-render-metric-")
        or render_policy.get("cwd") != os.fspath(captured_source_root / "benchmarks")
        or render_policy.get("timeout_seconds") != 600.0
    ):
        raise LifecycleError("replay render launch differs from exact semantic policy")
    render_environment = render_policy.get("environment")
    if not isinstance(render_environment, Mapping):
        raise LifecycleError("replay render environment is absent")
    _captured_worker_environment_semantics(
        render_environment,
        outdir=outdir,
        captured_source_root=captured_source_root,
        tmpdir=render_temp,
        preflight_record=preflight_record,
    )
    if (
        render_environment.get("STRUCTSPLAT_SOURCE_MANIFEST_SHA256")
        != replay_source_authority.get("source_manifest_sha256")
    ):
        raise LifecycleError("replay render source manifest environment drift")
    _verify_nested_policy_path_inventory(
        render_policy,
        exact_read_only={outdir / "runtime", blob_path},
        exact_workspace_read_only={
            authority_root,
            blob_path,
        },
        exact_read_write={render_temp},
        captured_source_root=captured_source_root,
        outdir=outdir,
        preflight_record=preflight_record,
        allow_gpu_read_write=True,
    )
    target_path = _artifact_path(
        outdir, target_record["rgb_copy"]["relative_path"], "replay target RGB path"
    )
    metric_command = metric_policy.get("command")
    expected_metric_command = [
        sys.executable,
        "-m",
        "benchmarks.ssp2v_actual_run",
        "_metric-worker",
        "--render",
        os.fspath(render_path),
        "--target-rgb",
        os.fspath(target_path),
        "--height",
        str(target_record["rgb_dimensions_hw"][0]),
        "--width",
        str(target_record["rgb_dimensions_hw"][1]),
        "--expected-render-npy-sha256",
        str(render["render_npy"]["sha256"]),
        "--expected-render-tensor-sha256",
        str(render["render_tensor"]["sha256"]),
        "--expected-target-rgb-sha256",
        str(target_record["rgb_copy"]["sha256"]),
        "--expected-lpips-state-sha256",
        expected_lpips_state_sha256,
    ]
    if (
        metric_command != expected_metric_command
        or metric_policy.get("cwd") != os.fspath(captured_source_root / "benchmarks")
        or metric_policy.get("timeout_seconds") != 600.0
    ):
        raise LifecycleError("replay metric launch differs from exact semantic policy")
    metric_environment = metric_policy.get("environment")
    if not isinstance(metric_environment, Mapping):
        raise LifecycleError("replay metric environment is absent")
    _captured_worker_environment_semantics(
        metric_environment,
        outdir=outdir,
        captured_source_root=captured_source_root,
        tmpdir=render_temp / "metric-scratch",
        preflight_record=preflight_record,
    )
    if (
        metric_environment.get("STRUCTSPLAT_SOURCE_MANIFEST_SHA256")
        != replay_source_authority.get("source_manifest_sha256")
    ):
        raise LifecycleError("replay metric source manifest environment drift")
    _verify_nested_policy_path_inventory(
        metric_policy,
        exact_read_only={outdir / "runtime", render_path, target_path},
        exact_workspace_read_only={
            authority_root,
            render_path,
        },
        exact_read_write={render_temp / "metric-scratch"},
        captured_source_root=captured_source_root,
        outdir=outdir,
        preflight_record=preflight_record,
        allow_gpu_read_write=False,
    )
    launcher = replay_source_authority.get("launcher")
    expected_launcher = captured_source_root / "benchmarks/ssp2v_landlock.py"
    if (
        not isinstance(launcher, Mapping)
        or any(
            policy.get("launcher_path") != os.fspath(expected_launcher)
            or policy.get("launcher_sha256") != launcher.get("sha256")
            for policy in (render_policy, metric_policy)
        )
    ):
        raise LifecycleError("replay nested launcher authority drift")


def _verify_replay_quality_cell_evidence(
    cell: Mapping[str, Any],
    original: Mapping[str, Any],
    candidate_cell: Mapping[str, Any],
    target_record: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    *,
    outdir: Path,
    regenerated_artifact_root: Path,
    captured_source_root: Path,
    replay_source_authority: Mapping[str, Any],
    phase_b_sandbox_attestation_sha256: str | None,
) -> tuple[dict[str, Any], list[str]]:
    """Recompute one replay quality prefix from its sealed worker receipts."""

    if set(cell) != {
        "image_id",
        "bit_tuple",
        "prefix",
        "schedule",
        "metrics",
        "metric_envelopes",
        "nongating_later_metrics",
        "selection",
        "tasks",
        "all_endpoint_tolerances_pass",
        "all_qualification_classes_reproduced",
        "all_decision_exclusion_bands_clear",
    }:
        raise LifecycleError("replay phase-B quality cell key set drift")
    identity = _identity(cell, "replay quality cell")
    if (
        identity != _identity(original, "original quality cell")
        or identity != _identity(candidate_cell, "replay candidate cell")
        or target_record.get("image_id") != identity[0]
    ):
        raise LifecycleError("replay phase-B quality identity drift")
    prefix = _decision_relevant_variant_prefix(original, candidate_cell)
    schedules = (
        ("primary", *prefix),
        (*reversed(prefix), "primary"),
        ("primary", *prefix),
    )
    expected_schedule = [list(order) for order in schedules]
    tasks = cell.get("tasks")
    if (
        cell.get("prefix") != prefix
        or cell.get("schedule") != expected_schedule
        or cell.get("all_endpoint_tolerances_pass") is not True
        or cell.get("all_qualification_classes_reproduced") is not True
        or cell.get("all_decision_exclusion_bands_clear") is not True
        or not isinstance(tasks, list)
        or len(tasks) != sum(len(order) for order in schedules)
    ):
        raise LifecycleError("replay phase-B quality prefix/schedule drift")
    expected_tasks = [
        (repetition, order_index, arm)
        for repetition, order in enumerate(schedules)
        for order_index, arm in enumerate(order)
    ]
    derived: dict[str, list[dict[str, float]]] = {
        "primary": [],
        **{label: [] for label in prefix},
    }
    execution_record = candidate_cell["execution"]
    q_name = str(execution_record["selected_q"]["name"])
    if original.get("q_name") != q_name:
        raise LifecycleError("replay phase-B original Q differs from regenerated execution")
    renderer = preflight_record["renderer"]
    native = preflight_record["native"]["arithmetic"]
    worker_source = _decode_worker_source_record()
    lpips_state = preflight_record["renderer_proof"]["same_worker_metric_repeat"][
        "lpips_state_sha256"
    ]
    target_height, target_width = target_record["rgb_dimensions_hw"]
    render_pids: list[int] = []
    metric_pids: list[int] = []
    nonces: list[str] = []
    for task, (repetition, order_index, arm) in zip(tasks, expected_tasks, strict=True):
        if not isinstance(task, Mapping):
            raise LifecycleError("replay phase-B quality task is malformed")
        stream_name = q_name if arm == "primary" else arm
        if (
            set(task)
            != {
                "repetition",
                "order_index",
                "arm",
                "stream_name",
                "render",
                "metric",
                "render_sandbox",
                "render_launch",
                "metric_sandbox",
                "metric_launch",
            }
            or task.get("repetition") != repetition
            or task.get("order_index") != order_index
            or task.get("arm") != arm
            or task.get("stream_name") != stream_name
        ):
            raise LifecycleError("replay phase-B quality task vector drift")
        render = task.get("render")
        metric = task.get("metric")
        if not isinstance(render, Mapping) or not isinstance(metric, Mapping):
            raise LifecycleError("replay phase-B quality worker receipt absent")
        actual.verify_record(render, "render_worker_sha256")
        actual.verify_record(metric, "metric_worker_sha256")
        _verify_launch_bundle(
            {"attestation": task["render_sandbox"], "launch": task["render_launch"]},
            render,
            "render_worker_sha256",
            expected_parent_sandbox_attestation_sha256=(
                phase_b_sandbox_attestation_sha256
            ),
        )
        _verify_launch_bundle(
            {"attestation": task["metric_sandbox"], "launch": task["metric_launch"]},
            metric,
            "metric_worker_sha256",
            expected_parent_sandbox_attestation_sha256=(
                phase_b_sandbox_attestation_sha256
            ),
        )
        expected_stream = _decoded_stream_record(execution_record, stream_name)
        _verify_replay_render_metric_policy(
            task,
            outdir=outdir,
            captured_source_root=captured_source_root,
            regenerated_artifact_root=regenerated_artifact_root,
            candidate_cell=candidate_cell,
            stream_name=stream_name,
            expected_stream=expected_stream,
            render=render,
            target_record=target_record,
            preflight_record=preflight_record,
            replay_source_authority=replay_source_authority,
            expected_lpips_state_sha256=str(lpips_state),
            fresh_root_dispatch=(
                phase_b_sandbox_attestation_sha256 is None
            ),
        )
        render_tensor = render.get("render_tensor")
        render_pid = render.get("pid")
        metric_pid = metric.get("pid")
        thread_state = metric.get("thread_state")
        values = metric.get("metrics")
        if (
            render.get("schema") != "structsplat.comp011.render-arm-worker.v1"
            or render.get("protocol") != PROTOCOL
            or render.get("task_sha256") != TASK_SHA256
            or render.get("blob_sha256") != expected_stream["blob_sha256"]
            or render.get("symbol_sha256") != expected_stream["symbol_sha256"]
            or render.get("boundary_state_sha256")
            != expected_stream["boundary_state_sha256"]
            or render.get("renderer_relative_path") != renderer["relative_path"]
            or render.get("renderer_sha256") != renderer["sha256"]
            or render.get("native_library")
            != {
                "relative_path": f"runtime/native/{native['path']}",
                "bytes": native["bytes"],
                "sha256": native["sha256"],
            }
            or render.get("decode_worker") != worker_source
            or render.get("cuda_synchronized_before_and_after") is not True
            or render.get("strict_complete_stream_decode") is not True
            or metric.get("schema") != "structsplat.comp011.metric-worker.v1"
            or metric.get("protocol") != PROTOCOL
            or metric.get("task_sha256") != TASK_SHA256
            or metric.get("render_tensor") != render_tensor
            or metric.get("target_tensor") != target_record["strict_target_tensor"]
            or metric.get("lpips_state_sha256") != lpips_state
            or metric.get("call_order") != ["psnr", "ms_ssim", "lpips"]
            or metric.get("fresh_single_thread_cpu_worker") is not True
            or isinstance(render_pid, bool)
            or not isinstance(render_pid, int)
            or render_pid <= 0
            or isinstance(metric_pid, bool)
            or not isinstance(metric_pid, int)
            or metric_pid <= 0
            or not isinstance(render_tensor, Mapping)
            or render_tensor.get("dtype") != "torch.float32"
            or render_tensor.get("shape") != [1, 3, target_height, target_width]
            or render_tensor.get("bytes") != 4 * 3 * target_height * target_width
            or not isinstance(thread_state, Mapping)
            or set(thread_state)
            != {
                "torch_num_threads",
                "torch_num_interop_threads",
                "deterministic_algorithms",
                "manual_seed",
                "thread_environment",
            }
            or thread_state.get("torch_num_threads") != 1
            or thread_state.get("torch_num_interop_threads") != 1
            or thread_state.get("deterministic_algorithms") is not True
            or thread_state.get("manual_seed") != 0
            or thread_state.get("thread_environment")
            != quality.FROZEN_THREAD_ENVIRONMENT
            or not isinstance(values, Mapping)
            or set(values) != set(quality.METRIC_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                for value in values.values()
            )
        ):
            raise LifecycleError("replay phase-B quality worker authority drift")
        derived[arm].append(
            {name: float(values[name]) for name in quality.METRIC_NAMES}
        )
        render_pids.append(render_pid)
        metric_pids.append(metric_pid)
        nonces.extend(
            (str(task["render_launch"]["nonce"]), str(task["metric_launch"]["nonce"]))
        )
    if (
        len(set(render_pids)) != len(render_pids)
        or len(set(metric_pids)) != len(metric_pids)
        or len(set(nonces)) != len(nonces)
        or any(len(values) != 3 for values in derived.values())
        or cell.get("metrics") != derived
        or cell.get("metric_envelopes") != _quality_metric_envelopes(derived)
    ):
        raise LifecycleError("replay phase-B quality metrics/freshness drift")

    original_variants = {
        variant["label"]: variant for variant in original["variants"]
    }
    for arm in ("primary", *prefix):
        original_metrics = (
            original["primary_metrics"]
            if arm == "primary"
            else original_variants[arm]["metrics"]
        )
        for endpoint in ("min", "max"):
            if not quality.replay_endpoint_match(
                _metric_endpoints(original_metrics, endpoint),
                _metric_endpoints(derived[arm], endpoint),
            ):
                raise LifecycleError("replay phase-B quality endpoint tolerance drift")
        if arm != "primary":
            _verify_replay_quality_classification(
                original["primary_metrics"],
                original_metrics,
                derived["primary"],
                derived[arm],
            )
    nongating_later = cell.get("nongating_later_metrics")
    expected_later = {
        label: original_variants[label]["metrics"]
        for label in actual.VARIANT_ORDER
        if label not in derived
    }
    if nongating_later != expected_later:
        raise LifecycleError("replay phase-B nongating later metrics provenance drift")
    full_metrics: dict[str, Sequence[Mapping[str, Any]]] = {
        "primary": derived["primary"],
        **{
            label: (
                derived[label]
                if label in derived
                else nongating_later[label]
            )
            for label in actual.VARIANT_ORDER
        },
    }
    formats, variants = _quality_scalar_inputs(candidate_cell, full_metrics)
    selection = actual.select_variant(derived["primary"], variants)
    if cell.get("selection") != selection or any(
        selection.get(field) != original["selection"].get(field)
        for field in (
            "state",
            "invalid_reason",
            "selected_label",
            "selected_bytes",
            "ordered_labels",
        )
    ):
        raise LifecycleError("replay phase-B selected variant/state drift")
    reconstructed = {
        "image_id": candidate_cell["image_id"],
        "bit_tuple": candidate_cell["bit_tuple"],
        "q_name": q_name,
        "formats": formats,
        "primary_metrics": list(derived["primary"]),
        "variants": variants,
        "selection": selection,
    }
    return reconstructed, nonces


def _verify_replay_resource_schedule(
    schedule_record: Mapping[str, Any],
    *,
    frozen_schedule: Sequence[decode_worker.LaunchSpec],
    image_index: int,
    tuple_index: int,
    measurement_session_sha256: str,
    expected_streams: Mapping[str, Mapping[str, Any]],
    expected_blob_paths: Mapping[str, str],
    artifact_root: Path,
    expected_source_root: Path,
    preflight_record: Mapping[str, Any],
    replay_source_authority: Mapping[str, Any],
    phase_b_sandbox_attestation_sha256: str | None,
) -> list[str]:
    if (
        set(schedule_record)
        != {"schedule", "samples", "launches", "summary", "fresh_processes"}
        or schedule_record.get("schedule")
        != [launch.to_dict() for launch in frozen_schedule]
        or schedule_record.get("fresh_processes") is not True
        or not isinstance(schedule_record.get("samples"), list)
        or len(schedule_record["samples"]) != len(frozen_schedule)
        or not isinstance(schedule_record.get("launches"), list)
        or len(schedule_record["launches"]) != len(frozen_schedule)
    ):
        raise LifecycleError("replay phase-B resource schedule/freshness drift")
    if set(expected_blob_paths) != set(expected_streams):
        raise LifecycleError("replay phase-B resource blob-path inventory drift")
    worker = _decode_worker_source_record()
    native = preflight_record["native"]["arithmetic"]
    native_relative_path = f"runtime/native/{native['path']}"
    source_roots: set[Path] = set()
    for launch_spec, bundle, sample in zip(
        frozen_schedule,
        schedule_record["launches"],
        schedule_record["samples"],
        strict=True,
    ):
        _verify_launch_bundle(
            bundle,
            sample,
            "canonical_worker_sha256",
            expected_parent_sandbox_attestation_sha256=(
                phase_b_sandbox_attestation_sha256
            ),
        )
        expected = expected_streams[launch_spec.arm]
        expected_blob_path = _safe_relative_path(
            expected_blob_paths[launch_spec.arm], "replay resource blob path"
        ).as_posix()
        acquisition = sample.get("blob_acquisition")
        exactness = sample.get("exactness")
        if (
            sample.get("blob_bytes") != expected["blob_bytes"]
            or sample.get("blob_sha256") != expected["blob_sha256"]
            or sample.get("symbol_sha256") != expected["symbol_sha256"]
            or sample.get("boundary_state_sha256")
            != expected["boundary_state_sha256"]
            or not isinstance(exactness, Mapping)
            or exactness.get("expected_blob_bytes") != expected["blob_bytes"]
            or exactness.get("expected_symbol_sha256") != expected["symbol_sha256"]
            or exactness.get("expected_boundary_state_sha256")
            != expected["boundary_state_sha256"]
            or not isinstance(acquisition, Mapping)
            or acquisition.get("artifact_relative_path") != expected_blob_path
        ):
            raise LifecycleError("replay phase-B resource sample authority drift")
        expected_command = _resource_worker_command(
            artifact_root=artifact_root,
            blob_relative_path=expected_blob_path,
            native_relative_path=native_relative_path,
            stream_record=expected,
            native_record=native,
            worker_source=worker,
            measurement_session_sha256=measurement_session_sha256,
            launch=launch_spec,
        )
        source_roots.add(
            _verify_captured_decode_worker_policy(
                bundle,
                expected_command=expected_command,
                artifact_root=artifact_root,
                blob_relative_path=expected_blob_path,
                native_relative_path=native_relative_path,
                expected_source_root=expected_source_root,
                preflight_record=preflight_record,
                replay_source_authority=replay_source_authority,
            )
        )
    expected_blob_sha256 = {
        arm: str(record["blob_sha256"]) for arm, record in expected_streams.items()
    }
    recomputed = decode_worker.summarize_schedule(
        schedule_record["samples"],
        frozen_schedule,
        image_index=image_index,
        tuple_index=tuple_index,
        measurement_session_sha256=measurement_session_sha256,
        expected_blob_sha256_by_arm=expected_blob_sha256,
    )
    if schedule_record.get("summary") != recomputed:
        raise LifecycleError("replay phase-B resource summary does not recompute")
    expected_bindings = {
        "worker_module": {
            "source_label": decode_worker.WORKER_MODULE_LABEL,
            "bytes": worker["bytes"],
            "sha256": worker["sha256"],
            "identity_verified_before_operation": True,
        },
        "native_library": {
            "artifact_relative_path": f"runtime/native/{native['path']}",
            "bytes": native["bytes"],
            "sha256": native["sha256"],
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
        "cdf_asset": {
            "source_label": "benchmarks/assets/ssp2e_cdf_v1.bin",
            "bytes": comp009_assets.CDF_ASSET_SIZE,
            "sha256": comp009_assets.CDF_ASSET_SHA256,
            "identity_verified_from_same_opened_bytes_before_load": True,
            "loaded_from_sealed_verified_bytes": True,
        },
    }
    if any(
        sample.get("source_bindings") != expected_bindings
        for sample in schedule_record["samples"]
    ):
        raise LifecycleError("replay phase-B resource source binding drift")
    if source_roots != {_lexical(expected_source_root)}:
        raise LifecycleError("replay phase-B resource source roots differ")
    return [str(bundle["launch"]["nonce"]) for bundle in schedule_record["launches"]]


def _verify_replay_phase_b(
    outdir: Path,
    phase_b: Mapping[str, Any],
    phase_a: Mapping[str, Any],
    candidate: Mapping[str, Any],
    targets: Mapping[str, Any],
    quality_record: Mapping[str, Any],
    benchmark_record: Mapping[str, Any],
    analysis_record: Mapping[str, Any],
    analysis: Mapping[str, Any],
    preflight_record: Mapping[str, Any],
    captured_source_root: Path | None = None,
    replay_source_authority: Mapping[str, Any] | None = None,
    phase_b_sandbox_attestation_sha256: str | None = None,
) -> None:
    actual.verify_record(phase_b, "captured_replay_worker_sha256")
    if (
        phase_b.get("schema") != REPLAY_PHASE_B_SCHEMA
        or phase_b.get("protocol") != PROTOCOL
        or phase_b.get("task_sha256") != TASK_SHA256
        or phase_b.get("source_manifest_sha256")
        != preflight_record.get("source_manifest_sha256")
        or phase_b.get("replay_phase_a_sha256")
        != phase_a.get("replay_phase_a_sha256")
        or phase_b.get("candidate_replay") != phase_a.get("candidate_replay")
        or phase_b.get("analysis_record_sha256")
        != analysis_record.get("analysis_record_sha256")
        or phase_b.get("benchmark_stage_sha256")
        != benchmark_record.get("benchmark_stage_sha256")
        or phase_b.get("confirmation_payloads_accessed") is not False
    ):
        raise LifecycleError("captured replay phase-B source/authorization/decision drift")
    quality_replay = phase_b.get("quality_replay")
    resource_replay = phase_b.get("resource_replay")
    regenerated = phase_a["candidate_replay"].get("cells")
    original_candidate_cells = candidate.get("cells")
    if (
        not isinstance(regenerated, list)
        or not isinstance(original_candidate_cells, list)
        or len(regenerated) != len(original_candidate_cells)
    ):
        raise LifecycleError("replay phase-B regenerated candidate inventory drift")
    candidate_cells = [
        {
            "image_id": replayed["image_id"],
            "bit_tuple": replayed["bit_tuple"],
            "state": replayed["state"],
            "relative_path": replayed["relative_path"],
            "execution": replayed["execution"],
            "artifacts": replayed["artifacts"],
            "cold_decodes": replayed["cold_decodes"],
        }
        for replayed in regenerated
    ]
    if candidate.get("state") == "valid_terminal_lloyd_failure":
        failure = candidate_cells[-1]
        early_row = actual.seal_record(
            {
                "schema": actual.ROW_SCHEMA,
                "image_id": failure["image_id"],
                "bit_tuple": failure["bit_tuple"],
                "validity": "valid",
                "invalid_reason": None,
                "method_state": "lloyd_failure",
                "lloyd_all_fixed": False,
                "formats": None,
                "primary_metrics": None,
                "variants": None,
                "primary_resource": None,
                "secondary_resource": None,
            }
        )
        replay_analysis = actual.analyze_rows([early_row])
        if (
            phase_b.get("targets_opened") is not False
            or quality_replay is not None
            or resource_replay is not None
            or targets.get("state") != "inapplicable_terminal_lloyd_failure"
            or quality_record.get("state") != "inapplicable_terminal_lloyd_failure"
            or benchmark_record.get("state") != "inapplicable_terminal_lloyd_failure"
            or phase_b.get("analysis_gate_signature")
            != _analysis_gate_signature(replay_analysis)
            or phase_b.get("analysis_gate_signature") != _analysis_gate_signature(analysis)
            or phase_b.get("replay_rows") != [early_row]
            or phase_b.get("replay_analysis") != replay_analysis
        ):
            raise LifecycleError("target-free replay phase-B contains target-derived work")
        return
    if (
        candidate.get("state") != "complete_candidate_set"
        or targets.get("state") != "complete_target_copy"
        or quality_record.get("state") != "complete_quality_grid"
        or phase_b.get("targets_opened") is not True
        or not isinstance(quality_replay, list)
        or len(quality_replay) != 16
        or not isinstance(quality_record.get("cells"), list)
        or len(quality_record["cells"]) != 16
        or len(candidate_cells) != 16
    ):
        raise LifecycleError("replay phase-B target/quality evidence is incomplete")
    if captured_source_root is None:
        raise LifecycleError("replay phase-B captured source-root binding is absent")
    if not isinstance(replay_source_authority, Mapping):
        raise LifecycleError("replay phase-B captured source authority is absent")
    if (
        phase_b.get("worker_dispatch") != "host_derived_fresh_root_v1"
        or phase_b_sandbox_attestation_sha256 is not None
        or any(
            not isinstance(phase_b.get(field), str)
            or len(str(phase_b[field])) != 64
            for field in (
                "quality_plan_sha256",
                "quality_results_sha256",
                "quality_authorization_sha256",
                "resource_plan_sha256",
                "resource_results_sha256",
            )
        )
    ):
        raise LifecycleError(
            "replay phase-B fresh-root dispatch authority drift"
        )
    phase_b_parent = None
    _verify_replay_captured_source_authority(
        replay_source_authority,
        preflight_record=preflight_record,
        expected_source_root=captured_source_root,
    )
    target_by_image = {
        target["image_id"]: target for target in targets.get("targets", [])
    }
    if set(target_by_image) != set(actual.FROZEN_IMAGES):
        raise LifecycleError("replay phase-B target inventory drift")
    reconstructed_quality: list[dict[str, Any]] = []
    launch_nonces: list[str] = []
    for cell, original, candidate_cell in zip(
        quality_replay, quality_record["cells"], candidate_cells, strict=True
    ):
        if (
            not isinstance(cell, Mapping)
            or cell.get("image_id") not in target_by_image
        ):
            raise LifecycleError("replay phase-B quality cell evidence drift")
        checked, nonces = _verify_replay_quality_cell_evidence(
            cell,
            original,
            candidate_cell,
            target_by_image[cell["image_id"]],
            preflight_record,
            outdir=outdir,
            regenerated_artifact_root=Path(
                str(phase_a["candidate_replay"]["regenerated_artifact_root"])
            ),
            captured_source_root=captured_source_root,
            replay_source_authority=replay_source_authority,
            phase_b_sandbox_attestation_sha256=phase_b_parent,
        )
        reconstructed_quality.append(checked)
        launch_nonces.extend(nonces)

    invalid = next(
        (
            cell
            for cell in reconstructed_quality
            if cell["selection"]["state"] == "invalid"
        ),
        None,
    )
    replay_rows: list[dict[str, Any]] = []
    if invalid is not None:
        if resource_replay != []:
            raise LifecycleError("invalid-quality replay ran forbidden resource schedules")
        replay_rows = [
            actual.seal_record(
                {
                    "schema": actual.ROW_SCHEMA,
                    "image_id": invalid["image_id"],
                    "bit_tuple": invalid["bit_tuple"],
                    "validity": "invalid",
                    "invalid_reason": str(invalid["selection"]["invalid_reason"]),
                    "method_state": "invalid",
                    "lloyd_all_fixed": None,
                    "formats": None,
                    "primary_metrics": None,
                    "variants": None,
                    "primary_resource": None,
                    "secondary_resource": None,
                }
            )
        ]
    else:
        if not isinstance(resource_replay, list) or len(resource_replay) != 16:
            raise LifecycleError("replay phase-B resource grid is incomplete")
        all_primary = all(
            cell["selection"]["state"] == "selected"
            for cell in reconstructed_quality
        )
        quality_stage_sha256 = str(quality_record["quality_stage_sha256"])
        regenerated_artifact_root = Path(
            str(phase_a["candidate_replay"]["regenerated_artifact_root"])
        )
        for resource_cell, quality_cell, candidate_cell in zip(
            resource_replay, reconstructed_quality, candidate_cells, strict=True
        ):
            if (
                not isinstance(resource_cell, Mapping)
                or set(resource_cell) != {"image_id", "bit_tuple", "primary", "secondary"}
                or _identity(resource_cell, "replay resource cell")
                != _identity(quality_cell, "replay quality cell")
            ):
                raise LifecycleError("replay phase-B resource identity/key drift")
            image_id = quality_cell["image_id"]
            bits = tuple(quality_cell["bit_tuple"])
            image_index = actual.FROZEN_IMAGES.index(image_id)
            tuple_index = actual.FROZEN_TUPLES.index(bits)
            execution_record = candidate_cell["execution"]
            candidate_relative = _safe_relative_path(
                str(candidate_cell["relative_path"]),
                "replay resource candidate-cell path",
            )
            secondary = resource_cell.get("secondary")
            if not isinstance(secondary, Mapping):
                raise LifecycleError("replay phase-B secondary resource schedule absent")
            secondary_session = _measurement_session_sha256(
                kind="secondary",
                image_id=image_id,
                bits=bits,
                quality_stage_sha256=quality_stage_sha256,
                replay_phase_sha256=str(phase_a["replay_phase_a_sha256"]),
            )
            launch_nonces.extend(
                _verify_replay_resource_schedule(
                    secondary,
                    frozen_schedule=decode_worker.secondary_schedule(
                        image_index, tuple_index
                    ),
                    image_index=image_index,
                    tuple_index=tuple_index,
                    measurement_session_sha256=secondary_session,
                    expected_streams={
                        "sspl1": _decoded_stream_record(execution_record, "sspl1"),
                        "ssp2l": _decoded_stream_record(execution_record, "ssp2l"),
                    },
                    expected_blob_paths={
                        "sspl1": (candidate_relative / "exact/sspl1.bin").as_posix(),
                        "ssp2l": (candidate_relative / "exact/ssp2l.bin").as_posix(),
                    },
                    artifact_root=regenerated_artifact_root,
                    expected_source_root=captured_source_root,
                    preflight_record=preflight_record,
                    replay_source_authority=replay_source_authority,
                    phase_b_sandbox_attestation_sha256=phase_b_parent,
                )
            )
            primary = resource_cell.get("primary")
            if all_primary:
                if not isinstance(primary, Mapping):
                    raise LifecycleError("replay phase-B primary resource schedule absent")
                selected = str(quality_cell["selection"]["selected_label"])
                q_name = str(quality_cell["q_name"])
                primary_session = _measurement_session_sha256(
                    kind="primary",
                    image_id=image_id,
                    bits=bits,
                    quality_stage_sha256=quality_stage_sha256,
                    candidate=selected,
                    q_name=q_name,
                    replay_phase_sha256=str(phase_a["replay_phase_a_sha256"]),
                )
                launch_nonces.extend(
                    _verify_replay_resource_schedule(
                        primary,
                        frozen_schedule=decode_worker.primary_schedule(
                            image_index, tuple_index
                        ),
                        image_index=image_index,
                        tuple_index=tuple_index,
                        measurement_session_sha256=primary_session,
                        expected_streams={
                            "candidate": _decoded_stream_record(
                                execution_record, selected
                            ),
                            "q": _decoded_stream_record(execution_record, q_name),
                        },
                        expected_blob_paths={
                            "candidate": (
                                candidate_relative / f"variants/{selected}.ssp2v"
                            ).as_posix(),
                            "q": (
                                candidate_relative / f"exact/{q_name}.bin"
                            ).as_posix(),
                        },
                        artifact_root=regenerated_artifact_root,
                        expected_source_root=captured_source_root,
                        preflight_record=preflight_record,
                        replay_source_authority=replay_source_authority,
                        phase_b_sandbox_attestation_sha256=phase_b_parent,
                    )
                )
            elif primary is not None:
                raise LifecycleError("replay phase-B ran globally inapplicable primary schedule")
            primary_resource, secondary_resource = _resource_records(
                quality_cell,
                primary if isinstance(primary, Mapping) else None,
                secondary,
            )
            replay_rows.append(
                _actual_row(quality_cell, primary_resource, secondary_resource)
            )
    if len(launch_nonces) != len(set(launch_nonces)):
        raise LifecycleError("replay phase-B worker launch nonce reuse")
    replay_analysis = actual.analyze_rows(replay_rows)
    replay_signature = _analysis_gate_signature(replay_analysis)
    if (
        phase_b.get("replay_rows") != replay_rows
        or phase_b.get("replay_analysis") != replay_analysis
        or phase_b.get("analysis_gate_signature") != replay_signature
        or replay_signature != _analysis_gate_signature(analysis)
    ):
        raise LifecycleError("replay phase-B analysis gates do not recompute")


def _attested_allowlist_path(path: Path, child_pid: int) -> str:
    if path.is_relative_to(Path("/proc/self")):
        return os.fspath(
            Path("/proc") / str(child_pid) / path.relative_to("/proc/self")
        )
    # Replay workspaces are deliberately destroyed after the launch.  Their
    # persisted attestation remains verifiable from the lexical absolute path.
    return os.fspath(path.resolve(strict=False))


def _verify_replay_outer_launch_policy(
    *,
    bundle: Mapping[str, Any],
    worker: Mapping[str, Any],
    worker_seal_field: str,
    phase: str,
    outdir: Path,
    phase_a_path: Path | None,
    preflight_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expected_captured_source_root: Path | None = None,
    phase_input_paths: Sequence[Path] = (),
) -> None:
    attestation = bundle.get("sandbox_attestation")
    launch = bundle.get("launch")
    if not isinstance(attestation, Mapping) or not isinstance(launch, Mapping):
        raise LifecycleError("replay outer launch policy bundle is malformed")
    _verify_launch_bundle(
        {"attestation": attestation, "launch": launch}, worker, worker_seal_field
    )
    command = attestation.get("command")
    supported_phases = {
        "phase-a",
        "phase-b",
        "phase-b-quality-plan",
        "phase-b-quality-reduce",
        "phase-b-final-reduce",
    }
    if phase not in supported_phases:
        raise LifecycleError("replay outer phase is unknown")
    expected_length = 8 if phase == "phase-a" else 9 + len(phase_input_paths)
    if not isinstance(command, list) or len(command) != expected_length:
        raise LifecycleError("replay outer command vector is malformed")
    extracted = Path(str(command[5]))
    if expected_captured_source_root is None:
        expected_captured_source_root = extracted
    if extracted != expected_captured_source_root:
        raise LifecycleError("replay outer captured source-root binding drift")
    scratch_relative = {
        "phase-a": Path("phase-a-scratch"),
        "phase-b": Path("phase-b-scratch"),
        "phase-b-quality-plan": Path("phase-b-scratch/quality-plan"),
        "phase-b-quality-reduce": Path("phase-b-scratch/quality-reduce"),
        "phase-b-final-reduce": Path("phase-b-scratch/final-reduce"),
    }[phase]
    scratch = extracted.parent / scratch_relative
    expected_command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        _CAPTURED_REPLAY_SCRIPT,
        os.fspath(extracted),
        phase,
        os.fspath(outdir),
    ]
    if phase_a_path is not None:
        expected_command.append(os.fspath(phase_a_path))
    expected_command.extend(os.fspath(path) for path in phase_input_paths)
    policy = launch.get("worker_policy")
    if not isinstance(policy, Mapping):
        raise LifecycleError("replay outer persisted launch policy is absent")
    source = preflight_record.get("source_manifest")
    if not isinstance(source, Mapping):
        source = _read_canonical_object(outdir / "source_manifest.json")
    launcher_record = next(
        (
            item
            for item in source.get("files", [])
            if item.get("path") == "benchmarks/ssp2v_landlock.py"
        ),
        None,
    )
    if not isinstance(launcher_record, Mapping):
        raise LifecycleError("captured replay launcher source binding is absent")
    if phase == "phase-a":
        explicit_readonly = [
            extracted,
            outdir / "runtime",
            outdir / "upstream",
            outdir / "streams",
            outdir / "candidates",
            outdir / "journal/events",
            outdir / "preflight.json",
            outdir / "source_manifest.json",
            outdir / "source_snapshot.tar",
            outdir / "import_streams.json",
            outdir / "candidate_stage.json",
        ]
    else:
        if phase_a_path is None:
            raise LifecycleError("replay phase-B launch lacks its phase-A path")
        terminal = (
            phase == "phase-b"
            and candidate.get("state") == "valid_terminal_lloyd_failure"
        )
        if terminal:
            # Do not stat or grant even directory-level access to target-derived
            # artifacts on the preregistered target-forbidden branch.
            explicit_readonly = [
                extracted,
                outdir / "candidate_stage.json",
                outdir / "benchmark_stage.json",
                outdir / "analysis.json",
                outdir / "analysis_record.json",
                phase_a_path,
            ]
        else:
            replay_scratch = expected_captured_source_root.parent / "phase-a-scratch"
            expected_regenerated_root = replay_scratch / "regenerated-artifacts"
            worker_candidate = worker.get("candidate_replay")
            if phase in {"phase-b", "phase-b-final-reduce"} and (
                not isinstance(worker_candidate, Mapping)
                or worker_candidate.get("regenerated_artifact_root")
                != os.fspath(expected_regenerated_root)
            ):
                raise LifecycleError(
                    "replay phase-B regenerated artifact root is detached from phase A"
                )
            if phase in {
                "phase-b-quality-plan",
                "phase-b-quality-reduce",
            }:
                persisted_phase_a = _read_canonical_object(phase_a_path)
                actual.verify_record(
                    persisted_phase_a, "replay_phase_a_sha256"
                )
                if worker.get(
                    "replay_phase_a_sha256"
                ) != persisted_phase_a.get("replay_phase_a_sha256"):
                    raise LifecycleError(
                        "replay phase-B pass is detached from phase-A authorization"
                    )
            explicit_readonly = [
                extracted,
                outdir / "runtime",
                outdir / "upstream",
                outdir / "streams",
                outdir / "candidates",
                outdir / "targets",
                outdir / "quality",
                outdir / "benchmark",
                outdir / "journal/events",
                outdir / "preflight.json",
                outdir / "source_manifest.json",
                outdir / "source_snapshot.tar",
                outdir / "import_streams.json",
                outdir / "candidate_stage.json",
                outdir / "target_copy.json",
                outdir / "quality_stage.json",
                outdir / "benchmark_stage.json",
                outdir / "actual_rows.jsonl",
                outdir / "analysis.json",
                outdir / "analysis_record.json",
                phase_a_path,
                replay_scratch,
                *phase_input_paths,
            ]
    authority = _authority_copy(outdir, preflight_record)
    upstream_target = _lexical(
        Path(str(preflight_record["operational_upstream_roots"]["comp008"]))
    ) / _safe_relative_path(
        authority["comp008"]["targets"][0]["relative_path"],
        "replay outer target probe",
    )
    denied_paths = [os.fspath(upstream_target)]
    if phase == "phase-a" and candidate.get("state") == "complete_candidate_set":
        denied_paths.extend(
            os.fspath(path)
            for path in (
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.png",
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.rgb",
            )
        )
    if (
        phase == "phase-b"
        and candidate.get("state") == "valid_terminal_lloyd_failure"
    ):
        denied_paths.extend(
            os.fspath(path)
            for path in (
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.png",
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.rgb",
            )
        )
    environment = policy.get("environment")
    read_only = policy.get("read_only_request")
    read_write = policy.get("read_write_request")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(read_only, list)
        or not isinstance(read_write, list)
        or any(
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or _lexical(Path(value)).as_posix() != value
            for value in (*read_only, *read_write)
        )
    ):
        raise LifecycleError("replay outer persisted policy inventories are malformed")
    if read_only != sorted(set(read_only)) or read_write != sorted(set(read_write)):
        raise LifecycleError("replay outer persisted policy inventories are malformed")
    _captured_worker_environment_semantics(
        environment,
        outdir=outdir,
        captured_source_root=extracted,
        tmpdir=scratch,
        worker_scratch=scratch,
        preflight_record=preflight_record,
        require_replay_preflight_binding=True,
    )
    frozen_system_ro = {
        _lexical(Path(value))
        for value in json.loads(str(environment[_REPLAY_SYSTEM_READ_ONLY_ENV]))
    }
    frozen_gpu_rw = {
        _lexical(Path(value))
        for value in json.loads(str(environment[_REPLAY_GPU_READ_WRITE_ENV]))
    }
    ro_paths = {_lexical(Path(str(value))) for value in read_only}
    rw_paths = {_lexical(Path(str(value))) for value in read_write}
    exact_ro = {_lexical(path) for path in explicit_readonly}
    if phase in {
        "phase-b-quality-plan",
        "phase-b-quality-reduce",
        "phase-b-final-reduce",
    }:
        exact_rw = {scratch, Path("/proc/self/task")}
    else:
        exact_rw = {scratch} | frozen_gpu_rw
    operational = preflight_record.get("operational_upstream_roots")
    if not isinstance(operational, Mapping):
        raise LifecycleError("replay outer upstream authority is absent")
    upstream_roots = {_lexical(Path(str(value))) for value in operational.values()}
    if (
        command != expected_command
        or policy.get("command") != expected_command
        or policy.get("cwd") != os.fspath(extracted)
        or policy.get("timeout_seconds") != 24.0 * 3600.0
        or policy.get("launcher_path")
        != os.fspath(extracted / "benchmarks/ssp2v_landlock.py")
        or policy.get("launcher_sha256") != launcher_record.get("sha256")
        or policy.get("denied_read_probe_paths") != sorted(denied_paths)
        or ro_paths != exact_ro | frozen_system_ro
        or rw_paths != exact_rw
        or any(
            path == root or path.is_relative_to(root)
            for path in (*ro_paths, *rw_paths)
            for root in upstream_roots
        )
    ):
        raise LifecycleError("replay outer launch differs from its exact frozen policy")


def replay_stage(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    _recover_interrupted_stage_completion(outdir, "replay")
    existing = outdir / _STAGE_MANIFEST_PATHS["replay"]
    if existing.exists():
        return verify_replay_stage(outdir)
    # Phase A verifies only the target-free prefix.  No target/quality/analysis verifier runs yet.
    preflight_record = verify_preflight(outdir)
    verify_import_streams(outdir)
    candidate = verify_candidate_stage(outdir)
    authority = _authority_copy(outdir, preflight_record)
    archive = outdir / "source_snapshot.tar"
    with tempfile.TemporaryDirectory(prefix="comp011-captured-replay-") as temporary:
        workspace = Path(temporary)
        extracted = workspace / f"randomized-{os.urandom(8).hex()}"
        scratch_a = workspace / "phase-a-scratch"
        scratch_b = workspace / "phase-b-scratch"
        scratch_a.mkdir()
        scratch_b.mkdir()
        replay_system_read_only = _sandbox_system_read_only()
        replay_gpu_read_write = _sandbox_gpu_read_write()
        replay_inventory_environment = _replay_path_inventory_environment(
            system_read_only=replay_system_read_only,
            gpu_read_write=replay_gpu_read_write,
        )
        safe_extract_source_archive(
            archive,
            extracted,
            expected_manifest=_read_canonical_object(outdir / "source_manifest.json"),
        )
        phase_a_command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _CAPTURED_REPLAY_SCRIPT,
            os.fspath(extracted),
            "phase-a",
            os.fspath(outdir),
        ]
        phase_a_environment = _worker_environment(outdir)
        phase_a_environment["PYTHONPATH"] = os.pathsep.join(
            (
                os.fspath(outdir / "runtime/metrics/site"),
                os.fspath(extracted),
                os.fspath(extracted / "src"),
            )
        )
        phase_a_environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
            _verify_runtime_source_tree(extracted)["source_manifest_sha256"]
        )
        phase_a_environment[_REPLAY_PREFLIGHT_BINDING_ENV] = str(
            preflight_record["preflight_binding_sha256"]
        )
        phase_a_environment.update(replay_inventory_environment)
        phase_a_environment["TMPDIR"] = os.fspath(scratch_a)
        phase_a_environment["COMP011_WORKER_SCRATCH"] = os.fspath(scratch_a)
        upstream_target_probe = _lexical(
            Path(str(preflight_record["operational_upstream_roots"]["comp008"]))
        ) / _safe_relative_path(
            authority["comp008"]["targets"][0]["relative_path"],
            "replay upstream target probe",
        )
        artifact_target_probes = (
            [
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.png",
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.rgb",
            ]
            if candidate.get("state") == "complete_candidate_set"
            else []
        )
        phase_a, phase_a_sandbox, phase_a_launch = _run_json_worker(
            phase_a_command,
            outdir=outdir,
            timeout=24 * 3600,
            read_only=[
                extracted,
                outdir / "runtime",
                outdir / "upstream",
                outdir / "streams",
                outdir / "candidates",
                outdir / "journal/events",
                outdir / "preflight.json",
                outdir / "source_manifest.json",
                outdir / "source_snapshot.tar",
                outdir / "import_streams.json",
                outdir / "candidate_stage.json",
            ],
            read_write=[scratch_a, *replay_gpu_read_write],
            denied_read_probes=[upstream_target_probe, *artifact_target_probes],
            worker_seal_field="replay_phase_a_sha256",
            launcher_path=extracted / "benchmarks/ssp2v_landlock.py",
            cwd=extracted,
            environment=phase_a_environment,
            include_repository_source=False,
            system_read_only=replay_system_read_only,
        )
        phase_a_source_authority = _derive_replay_captured_source_authority(
            outdir, preflight_record, extracted
        )
        _verify_replay_phase_a(
            outdir,
            phase_a,
            candidate,
            preflight_record,
            extracted,
            phase_a_source_authority,
            phase_a_sandbox.get("attestation_sha256"),
        )
        if verify_preflight(outdir) != preflight_record:
            raise LifecycleError("outer preflight authority drifted after replay phase A")

        # Only the exact persisted phase-A candidate seal authorizes target-derived verification.
        journal, _guard, records = _stage_guard(outdir, "replay")
        analysis_record = records["analyze"]
        prior_seal = str(analysis_record["analysis_record_sha256"])
        phase_a_reference = _journaled_immutable(
            journal,
            outdir,
            outdir
            / (
                "replay/phase_a-"
                f"{phase_a.get('replay_phase_a_sha256', _sha256_bytes(_canonical_json(phase_a)))}.json"
            ),
            _canonical_json(phase_a),
            stage="replay",
            path_class="artifact_runtime",
            purpose="persist target-hidden captured replay phase-A authorization",
            prior_stage_seal=prior_seal,
        )
        phase_a_path = _artifact_path(
            outdir, phase_a_reference["relative_path"], "replay phase-A path"
        )
        terminal_replay = candidate.get("state") == "valid_terminal_lloyd_failure"
        if terminal_replay:
            phase_b_command = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _CAPTURED_REPLAY_SCRIPT,
                os.fspath(extracted),
                "phase-b",
                os.fspath(outdir),
                os.fspath(phase_a_path),
            ]
            phase_b_environment = _worker_environment(outdir)
            phase_b_environment["PYTHONPATH"] = os.pathsep.join(
                (
                    os.fspath(outdir / "runtime/metrics/site"),
                    os.fspath(extracted),
                    os.fspath(extracted / "src"),
                )
            )
            phase_b_environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
                _verify_runtime_source_tree(extracted)[
                    "source_manifest_sha256"
                ]
            )
            phase_b_environment[_REPLAY_PREFLIGHT_BINDING_ENV] = str(
                preflight_record["preflight_binding_sha256"]
            )
            phase_b_environment.update(replay_inventory_environment)
            phase_b_environment["TMPDIR"] = os.fspath(scratch_b)
            phase_b_environment["COMP011_WORKER_SCRATCH"] = os.fspath(
                scratch_b
            )
            phase_b_readonly = [
                extracted,
                outdir / "candidate_stage.json",
                outdir / "benchmark_stage.json",
                outdir / "analysis.json",
                outdir / "analysis_record.json",
                phase_a_path,
            ]
            phase_b_denied_probes = [
                upstream_target_probe,
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.png",
                outdir / "targets" / actual.FROZEN_IMAGES[0] / "target.rgb",
            ]
            phase_b, phase_b_sandbox, phase_b_launch = _run_json_worker(
                phase_b_command,
                outdir=outdir,
                timeout=24 * 3600,
                read_only=phase_b_readonly,
                read_write=[scratch_b, *replay_gpu_read_write],
                denied_read_probes=phase_b_denied_probes,
                worker_seal_field="captured_replay_worker_sha256",
                launcher_path=extracted
                / "benchmarks/ssp2v_landlock.py",
                cwd=extracted,
                environment=phase_b_environment,
                include_repository_source=False,
                system_read_only=replay_system_read_only,
            )
            quality_plan = None
            quality_results = None
            quality_authorization = None
            resource_results = None
            quality_plan_reference = None
            quality_results_reference = None
            quality_authorization_reference = None
            resource_results_reference = None
            phase_b_plan_bundle = None
            phase_b_quality_reduce_bundle = None
        else:
            pass_scratch = {
                name: scratch_b / name
                for name in (
                    "quality-plan",
                    "quality-broker",
                    "quality-reduce",
                    "resource-broker",
                    "final-reduce",
                )
            }
            for path in pass_scratch.values():
                path.mkdir()
            phase_b_readonly = [
                extracted,
                outdir / "runtime",
                outdir / "upstream",
                outdir / "streams",
                outdir / "candidates",
                outdir / "targets",
                outdir / "quality",
                outdir / "benchmark",
                outdir / "journal/events",
                outdir / "preflight.json",
                outdir / "source_manifest.json",
                outdir / "source_snapshot.tar",
                outdir / "import_streams.json",
                outdir / "candidate_stage.json",
                outdir / "target_copy.json",
                outdir / "quality_stage.json",
                outdir / "benchmark_stage.json",
                outdir / "actual_rows.jsonl",
                outdir / "analysis.json",
                outdir / "analysis_record.json",
                phase_a_path,
                scratch_a,
            ]

            def phase_b_pass_environment(pass_name: str) -> dict[str, str]:
                environment = _worker_environment(outdir)
                environment["PYTHONPATH"] = os.pathsep.join(
                    (
                        os.fspath(outdir / "runtime/metrics/site"),
                        os.fspath(extracted),
                        os.fspath(extracted / "src"),
                    )
                )
                environment["STRUCTSPLAT_SOURCE_MANIFEST_SHA256"] = str(
                    _verify_runtime_source_tree(extracted)[
                        "source_manifest_sha256"
                    ]
                )
                environment[_REPLAY_PREFLIGHT_BINDING_ENV] = str(
                    preflight_record["preflight_binding_sha256"]
                )
                environment.update(replay_inventory_environment)
                environment["TMPDIR"] = os.fspath(
                    pass_scratch[pass_name]
                )
                environment["COMP011_WORKER_SCRATCH"] = os.fspath(
                    pass_scratch[pass_name]
                )
                return environment

            quality_plan_command = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _CAPTURED_REPLAY_SCRIPT,
                os.fspath(extracted),
                "phase-b-quality-plan",
                os.fspath(outdir),
                os.fspath(phase_a_path),
            ]
            quality_plan, plan_sandbox, plan_launch = _run_json_worker(
                quality_plan_command,
                outdir=outdir,
                timeout=24 * 3600,
                read_only=phase_b_readonly,
                read_write=[
                    pass_scratch["quality-plan"],
                    *_sandbox_process_read_write(),
                ],
                denied_read_probes=[upstream_target_probe],
                worker_seal_field="quality_plan_sha256",
                launcher_path=extracted
                / "benchmarks/ssp2v_landlock.py",
                cwd=extracted,
                environment=phase_b_pass_environment("quality-plan"),
                include_repository_source=False,
                system_read_only=replay_system_read_only,
            )
            phase_b_plan_bundle = {
                "sandbox_attestation": plan_sandbox,
                "launch": plan_launch,
            }
            quality_plan_path = scratch_b / "quality-plan.json"
            _exclusive_write(quality_plan_path, _canonical_json(quality_plan))
            replay_context = _replay_phase_b_candidate_context(
                outdir, phase_a_path
            )
            targets = verify_target_copy(outdir)
            original_quality = verify_quality_stage(outdir)
            _verify_replay_quality_plan(
                quality_plan,
                context=replay_context,
                targets=targets,
                original_quality=original_quality,
                preflight_record=preflight_record,
            )
            quality_results = _execute_replay_quality_plan(
                quality_plan,
                outdir=outdir,
                context=replay_context,
                targets=targets,
                original_quality=original_quality,
                preflight_record=preflight_record,
                captured_source_root=extracted,
                system_read_only=replay_system_read_only,
                gpu_read_write=replay_gpu_read_write,
                scratch_root=pass_scratch["quality-broker"],
            )
            quality_results_path = scratch_b / "quality-results.json"
            _exclusive_write(
                quality_results_path, _canonical_json(quality_results)
            )
            quality_reduce_command = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _CAPTURED_REPLAY_SCRIPT,
                os.fspath(extracted),
                "phase-b-quality-reduce",
                os.fspath(outdir),
                os.fspath(phase_a_path),
                os.fspath(quality_plan_path),
                os.fspath(quality_results_path),
            ]
            (
                quality_authorization,
                quality_reduce_sandbox,
                quality_reduce_launch,
            ) = _run_json_worker(
                quality_reduce_command,
                outdir=outdir,
                timeout=24 * 3600,
                read_only=[
                    *phase_b_readonly,
                    quality_plan_path,
                    quality_results_path,
                ],
                read_write=[
                    pass_scratch["quality-reduce"],
                    *_sandbox_process_read_write(),
                ],
                denied_read_probes=[upstream_target_probe],
                worker_seal_field="quality_authorization_sha256",
                launcher_path=extracted
                / "benchmarks/ssp2v_landlock.py",
                cwd=extracted,
                environment=phase_b_pass_environment("quality-reduce"),
                include_repository_source=False,
                system_read_only=replay_system_read_only,
            )
            phase_b_quality_reduce_bundle = {
                "sandbox_attestation": quality_reduce_sandbox,
                "launch": quality_reduce_launch,
            }
            quality_authorization_path = (
                scratch_b / "quality-authorization.json"
            )
            _exclusive_write(
                quality_authorization_path,
                _canonical_json(quality_authorization),
            )
            replay_source_authority = _derive_replay_captured_source_authority(
                outdir, preflight_record, extracted
            )
            _verify_replay_quality_authorization(
                quality_authorization,
                quality_plan,
                quality_results,
                context=replay_context,
                targets=targets,
                original_quality=original_quality,
                preflight_record=preflight_record,
                captured_source_root=extracted,
                replay_source_authority=replay_source_authority,
            )
            resource_results = _execute_replay_resource_plan(
                quality_authorization,
                quality_plan,
                quality_results,
                outdir=outdir,
                context=replay_context,
                targets=targets,
                original_quality=original_quality,
                preflight_record=preflight_record,
                captured_source_root=extracted,
                replay_source_authority=replay_source_authority,
                system_read_only=replay_system_read_only,
            )
            resource_results_path = scratch_b / "resource-results.json"
            _exclusive_write(
                resource_results_path, _canonical_json(resource_results)
            )
            final_reduce_command = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                _CAPTURED_REPLAY_SCRIPT,
                os.fspath(extracted),
                "phase-b-final-reduce",
                os.fspath(outdir),
                os.fspath(phase_a_path),
                os.fspath(quality_plan_path),
                os.fspath(quality_results_path),
                os.fspath(quality_authorization_path),
                os.fspath(resource_results_path),
            ]
            phase_b, phase_b_sandbox, phase_b_launch = _run_json_worker(
                final_reduce_command,
                outdir=outdir,
                timeout=24 * 3600,
                read_only=[
                    *phase_b_readonly,
                    quality_plan_path,
                    quality_results_path,
                    quality_authorization_path,
                    resource_results_path,
                ],
                read_write=[
                    pass_scratch["final-reduce"],
                    *_sandbox_process_read_write(),
                ],
                denied_read_probes=[upstream_target_probe],
                worker_seal_field="captured_replay_worker_sha256",
                launcher_path=extracted
                / "benchmarks/ssp2v_landlock.py",
                cwd=extracted,
                environment=phase_b_pass_environment("final-reduce"),
                include_repository_source=False,
                system_read_only=replay_system_read_only,
            )
        replay_source_authority = _derive_replay_captured_source_authority(
            outdir, preflight_record, extracted
        )
        if replay_source_authority != phase_a_source_authority:
            raise LifecycleError(
                "captured replay source authority changed between phase workers"
            )
        original_analysis = _read_canonical_object(outdir / "analysis.json")
        if not terminal_replay:
            _verify_replay_phase_b_final_record(
                phase_b,
                context=replay_context,
                plan=quality_plan,
                quality_results=quality_results,
                authorization=quality_authorization,
                resource_results=resource_results,
                targets=targets,
                original_quality=original_quality,
                original_benchmark=records["benchmark-dev"],
                original_analysis_record=analysis_record,
                original_analysis=original_analysis,
                preflight_record=preflight_record,
                captured_source_root=extracted,
                replay_source_authority=replay_source_authority,
            )
        _verify_replay_phase_b(
            outdir,
            phase_b,
            phase_a,
            candidate,
            records["import-targets"],
            records["quality-select-dev"],
            records["benchmark-dev"],
            analysis_record,
            original_analysis,
            preflight_record,
            extracted,
            replay_source_authority,
            (
                phase_b_sandbox.get("attestation_sha256")
                if terminal_replay
                else None
            ),
        )
        if verify_preflight(outdir) != preflight_record:
            raise LifecycleError("outer preflight authority drifted after replay phase B")
        if not terminal_replay:
            quality_plan_reference = _journaled_immutable(
                journal,
                outdir,
                outdir
                / (
                    "replay/quality_plan-"
                    f"{quality_plan['quality_plan_sha256']}.json"
                ),
                _canonical_json(quality_plan),
                stage="replay",
                path_class="artifact_runtime",
                purpose="persist captured replay quality execution plan",
                prior_stage_seal=prior_seal,
            )
            quality_results_reference = _journaled_immutable(
                journal,
                outdir,
                outdir
                / (
                    "replay/quality_results-"
                    f"{quality_results['quality_results_sha256']}.json"
                ),
                _canonical_json(quality_results),
                stage="replay",
                path_class="artifact_runtime",
                purpose="persist host-brokered fresh-root quality receipts",
                prior_stage_seal=prior_seal,
            )
            quality_authorization_reference = _journaled_immutable(
                journal,
                outdir,
                outdir
                / (
                    "replay/quality_authorization-"
                    f"{quality_authorization['quality_authorization_sha256']}.json"
                ),
                _canonical_json(quality_authorization),
                stage="replay",
                path_class="artifact_runtime",
                purpose="persist captured quality reduction and resource authorization",
                prior_stage_seal=prior_seal,
            )
            resource_results_reference = _journaled_immutable(
                journal,
                outdir,
                outdir
                / (
                    "replay/resource_results-"
                    f"{resource_results['resource_results_sha256']}.json"
                ),
                _canonical_json(resource_results),
                stage="replay",
                path_class="artifact_runtime",
                purpose="persist host-brokered fresh-root resource receipts",
                prior_stage_seal=prior_seal,
            )
    return _finalize_stage(
        outdir,
        journal,
        stage="replay",
        schema=REPLAY_SCHEMA,
        prior_stage_seal=prior_seal,
        body={
            "state": "complete_replay",
            "analysis_record_sha256": prior_seal,
            "source_archive_sha256": preflight_record["source_archive_sha256"],
            "captured_source_authority": replay_source_authority,
            "phase_a": phase_a_reference,
            "captured_phase_a": phase_a,
            "captured_phase_b": phase_b,
            "phase_a_launch": {
                "sandbox_attestation": phase_a_sandbox,
                "launch": phase_a_launch,
            },
            "phase_b_launch": {
                "sandbox_attestation": phase_b_sandbox,
                "launch": phase_b_launch,
            },
            "phase_b_plan_launch": phase_b_plan_bundle,
            "phase_b_quality_reduce_launch": phase_b_quality_reduce_bundle,
            "quality_plan": quality_plan_reference,
            "quality_results": quality_results_reference,
            "quality_authorization": quality_authorization_reference,
            "resource_results": resource_results_reference,
            "captured_worker_command": {
                "isolated_python_flags": ["-I", "-B"],
                "script_sha256": _sha256_bytes(_CAPTURED_REPLAY_SCRIPT.encode("utf-8")),
                "randomized_root": True,
                "split_target_hidden_phase_a": True,
                "phase_b_worker_dispatch": (
                    "terminal_target_free_single_pass"
                    if terminal_replay
                    else "host_derived_fresh_root_v1"
                ),
            },
            "decision_signature_reproduced": True,
            "confirmation_payloads_accessed": False,
            "target_or_pixel_payloads_opened": phase_b.get("targets_opened"),
        },
    )


def verify_replay_stage(outdir: Path) -> dict[str, Any]:
    outdir = _lexical(outdir)
    analysis_record = verify_analysis_stage(outdir)
    record = _verify_stage_base(
        outdir,
        stage="replay",
        schema=REPLAY_SCHEMA,
        prior_record=analysis_record,
    )
    phase_a = record.get("captured_phase_a")
    phase_b = record.get("captured_phase_b")
    if not isinstance(phase_a, Mapping) or not isinstance(phase_b, Mapping):
        raise LifecycleError("captured replay phase records are absent")
    preflight_record = verify_preflight(outdir)
    candidate = verify_candidate_stage(outdir)
    targets = verify_target_copy(outdir)
    quality_record = verify_quality_stage(outdir)
    benchmark_record = verify_benchmark_stage(outdir)
    phase_a_reference = record.get("phase_a")
    if not isinstance(phase_a_reference, Mapping):
        raise LifecycleError("persisted replay phase-A reference is absent")
    phase_a_path = _artifact_path(
        outdir, phase_a_reference.get("relative_path"), "replay phase-A path"
    )
    _verify_regular_record(phase_a_path, phase_a_reference)
    if _read_canonical_object(phase_a_path) != phase_a:
        raise LifecycleError("persisted replay phase-A artifact differs from manifest")
    if phase_a_path.name != (
        f"phase_a-{phase_a.get('replay_phase_a_sha256')}.json"
    ):
        raise LifecycleError("persisted replay phase-A content address drift")
    phase_a_outer = record.get("phase_a_launch")
    phase_b_outer = record.get("phase_b_launch")
    phase_a_attestation = (
        phase_a_outer.get("sandbox_attestation")
        if isinstance(phase_a_outer, Mapping)
        else None
    )
    phase_b_attestation = (
        phase_b_outer.get("sandbox_attestation")
        if isinstance(phase_b_outer, Mapping)
        else None
    )
    phase_a_command = (
        phase_a_attestation.get("command")
        if isinstance(phase_a_attestation, Mapping)
        else None
    )
    phase_b_command = (
        phase_b_attestation.get("command")
        if isinstance(phase_b_attestation, Mapping)
        else None
    )
    terminal_replay = (
        candidate.get("state") == "valid_terminal_lloyd_failure"
    )
    expected_phase_b_length = 9 if terminal_replay else 13
    if (
        not isinstance(phase_a_command, list)
        or len(phase_a_command) != 8
        or not isinstance(phase_b_command, list)
        or len(phase_b_command) != expected_phase_b_length
        or phase_a_command[5] != phase_b_command[5]
    ):
        raise LifecycleError("replay outer phases do not share one captured source root")
    captured_source_root = Path(str(phase_a_command[5]))
    if (
        not captured_source_root.is_absolute()
        or _lexical(captured_source_root) != captured_source_root
    ):
        raise LifecycleError("replay captured source root is not canonical")
    replay_source_authority = record.get("captured_source_authority")
    if not isinstance(replay_source_authority, Mapping):
        raise LifecycleError("persisted replay captured source authority is absent")
    _verify_replay_captured_source_authority(
        replay_source_authority,
        preflight_record=preflight_record,
        expected_source_root=captured_source_root,
    )
    _verify_replay_phase_a(
        outdir,
        phase_a,
        candidate,
        preflight_record,
        captured_source_root,
        replay_source_authority,
        phase_a_attestation.get("attestation_sha256"),
    )
    quality_plan = None
    quality_results = None
    quality_authorization = None
    resource_results = None
    quality_plan_path = None
    quality_results_path = None
    quality_authorization_path = None
    resource_results_path = None
    original_analysis = _read_canonical_object(outdir / "analysis.json")
    if terminal_replay:
        if any(
            record.get(name) is not None
            for name in (
                "phase_b_plan_launch",
                "phase_b_quality_reduce_launch",
                "quality_plan",
                "quality_results",
                "quality_authorization",
                "resource_results",
            )
        ):
            raise LifecycleError(
                "terminal replay contains target-derived broker records"
            )
    else:
        loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
        intermediate_seals = {
            "quality_plan": "quality_plan_sha256",
            "quality_results": "quality_results_sha256",
            "quality_authorization": "quality_authorization_sha256",
            "resource_results": "resource_results_sha256",
        }
        for name, seal_field in intermediate_seals.items():
            reference = record.get(name)
            if not isinstance(reference, Mapping):
                raise LifecycleError(
                    f"replay {name} immutable reference is absent"
                )
            path = _artifact_path(
                outdir, reference.get("relative_path"), f"replay {name} path"
            )
            _verify_regular_record(path, reference)
            value = _read_canonical_object(path)
            seal = value.get(seal_field)
            if (
                not isinstance(seal, str)
                or path.name != f"{name}-{seal}.json"
            ):
                raise LifecycleError(
                    f"replay {name} content-addressed path drift"
                )
            loaded[name] = (path, value)
        quality_plan_path, quality_plan = loaded["quality_plan"]
        quality_results_path, quality_results = loaded["quality_results"]
        quality_authorization_path, quality_authorization = loaded[
            "quality_authorization"
        ]
        resource_results_path, resource_results = loaded["resource_results"]
        replay_context = _replay_phase_b_candidate_context(
            outdir, phase_a_path
        )
        _verify_replay_quality_authorization(
            quality_authorization,
            quality_plan,
            quality_results,
            context=replay_context,
            targets=targets,
            original_quality=quality_record,
            preflight_record=preflight_record,
            captured_source_root=captured_source_root,
            replay_source_authority=replay_source_authority,
        )
        actual.verify_record(resource_results, "resource_results_sha256")
        if (
            phase_b.get("quality_plan_sha256")
            != quality_plan.get("quality_plan_sha256")
            or phase_b.get("quality_results_sha256")
            != quality_results.get("quality_results_sha256")
            or phase_b.get("quality_authorization_sha256")
            != quality_authorization.get("quality_authorization_sha256")
            or phase_b.get("resource_plan_sha256")
            != quality_authorization.get("resource_plan", {}).get(
                "resource_plan_sha256"
            )
            or phase_b.get("resource_results_sha256")
            != resource_results.get("resource_results_sha256")
        ):
            raise LifecycleError(
                "replay phase-B intermediate seal binding drift"
            )
        _verify_replay_phase_b_final_record(
            phase_b,
            context=replay_context,
            plan=quality_plan,
            quality_results=quality_results,
            authorization=quality_authorization,
            resource_results=resource_results,
            targets=targets,
            original_quality=quality_record,
            original_benchmark=benchmark_record,
            original_analysis_record=analysis_record,
            original_analysis=original_analysis,
            preflight_record=preflight_record,
            captured_source_root=captured_source_root,
            replay_source_authority=replay_source_authority,
        )
    _verify_replay_phase_b(
        outdir,
        phase_b,
        phase_a,
        candidate,
        targets,
        quality_record,
        benchmark_record,
        analysis_record,
        original_analysis,
        preflight_record,
        captured_source_root,
        replay_source_authority,
        (
            phase_b_attestation.get("attestation_sha256")
            if terminal_replay
            else None
        ),
    )
    outer_passes: list[
        tuple[
            str,
            Mapping[str, Any],
            str,
            str,
            Sequence[Path],
        ]
    ] = [
        (
            "phase_a_launch",
            phase_a,
            "replay_phase_a_sha256",
            "phase-a",
            (),
        )
    ]
    if terminal_replay:
        outer_passes.append(
            (
                "phase_b_launch",
                phase_b,
                "captured_replay_worker_sha256",
                "phase-b",
                (),
            )
        )
    else:
        assert all(
            value is not None
            for value in (
                quality_plan,
                quality_results,
                quality_authorization,
                resource_results,
                quality_plan_path,
                quality_results_path,
                quality_authorization_path,
                resource_results_path,
            )
        )
        replay_scratch = captured_source_root.parent / "phase-b-scratch"
        broker_quality_plan_path = replay_scratch / "quality-plan.json"
        broker_quality_results_path = replay_scratch / "quality-results.json"
        broker_quality_authorization_path = (
            replay_scratch / "quality-authorization.json"
        )
        broker_resource_results_path = (
            replay_scratch / "resource-results.json"
        )
        outer_passes.extend(
            (
                (
                    "phase_b_plan_launch",
                    quality_plan,
                    "quality_plan_sha256",
                    "phase-b-quality-plan",
                    (),
                ),
                (
                    "phase_b_quality_reduce_launch",
                    quality_authorization,
                    "quality_authorization_sha256",
                    "phase-b-quality-reduce",
                    (
                        broker_quality_plan_path,
                        broker_quality_results_path,
                    ),
                ),
                (
                    "phase_b_launch",
                    phase_b,
                    "captured_replay_worker_sha256",
                    "phase-b-final-reduce",
                    (
                        broker_quality_plan_path,
                        broker_quality_results_path,
                        broker_quality_authorization_path,
                        broker_resource_results_path,
                    ),
                ),
            )
        )
    for name, worker, seal_field, phase, phase_inputs in outer_passes:
        bundle = record.get(name)
        if not isinstance(bundle, Mapping):
            raise LifecycleError(f"replay {name} bundle is absent")
        _verify_replay_outer_launch_policy(
            bundle=bundle,
            worker=worker,
            worker_seal_field=seal_field,
            phase=phase,
            outdir=outdir,
            phase_a_path=(None if name == "phase_a_launch" else phase_a_path),
            preflight_record=preflight_record,
            candidate=candidate,
            expected_captured_source_root=captured_source_root,
            phase_input_paths=phase_inputs,
        )
    if (
        record.get("state") != "complete_replay"
        or record.get("analysis_record_sha256") != analysis_record["analysis_record_sha256"]
        or record.get("source_archive_sha256") != preflight_record["source_archive_sha256"]
        or phase_b.get("source_manifest_sha256") != preflight_record["source_manifest_sha256"]
        or phase_b.get("analysis_record_sha256") != analysis_record["analysis_record_sha256"]
        or record.get("decision_signature_reproduced") is not True
        or record.get("confirmation_payloads_accessed") is not False
        or record.get("target_or_pixel_payloads_opened")
        is not phase_b.get("targets_opened")
        or record.get("captured_worker_command")
        != {
            "isolated_python_flags": ["-I", "-B"],
            "script_sha256": _sha256_bytes(
                _CAPTURED_REPLAY_SCRIPT.encode("utf-8")
            ),
            "randomized_root": True,
            "split_target_hidden_phase_a": True,
            "phase_b_worker_dispatch": (
                "terminal_target_free_single_pass"
                if terminal_replay
                else "host_derived_fresh_root_v1"
            ),
        }
    ):
        raise LifecycleError("replay manifest/source/decision binding drift")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("preflight")
    child.add_argument("--outdir", type=Path, required=True)
    child.add_argument("--comp008", type=Path, default=DEFAULT_COMP008)
    child.add_argument("--comp009", type=Path, default=DEFAULT_COMP009)
    child.add_argument("--comp010", type=Path, default=DEFAULT_COMP010)
    for name in STAGES[1:]:
        stage = subparsers.add_parser(name)
        stage.add_argument("--outdir", type=Path, required=True)
    worker = subparsers.add_parser("_lloyd-proof")
    worker.add_argument("--native", type=Path, required=True)
    subparsers.add_parser("_build-renderer")
    worker = subparsers.add_parser("_renderer-proof")
    worker.add_argument("--artifact-root", type=Path, required=True)
    worker.add_argument("--renderer-record", type=Path, required=True)
    worker = subparsers.add_parser("_produce-cell")
    worker.add_argument("--artifact-root", type=Path, required=True)
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--captured-comp009-root", type=Path, required=True)
    worker.add_argument("--native", type=Path, required=True)
    worker.add_argument("--lloyd", type=Path, required=True)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker = subparsers.add_parser("_import-stream")
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--config", type=Path, required=True)
    worker = subparsers.add_parser("_render-arm")
    worker.add_argument("--artifact-root", type=Path, required=True)
    worker.add_argument("--renderer-record", type=Path, required=True)
    worker.add_argument("--blob", type=Path, required=True)
    worker.add_argument("--native", type=Path, required=True)
    worker.add_argument("--expected-blob-sha256", required=True)
    worker.add_argument("--expected-native-sha256", required=True)
    worker.add_argument("--expected-native-bytes", type=int, required=True)
    worker.add_argument("--expected-symbol-sha256", required=True)
    worker.add_argument("--expected-boundary-state-sha256", required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker = subparsers.add_parser("_metric-worker")
    worker.add_argument("--render", type=Path, required=True)
    worker.add_argument("--target-rgb", type=Path, required=True)
    worker.add_argument("--height", type=int, required=True)
    worker.add_argument("--width", type=int, required=True)
    worker.add_argument("--expected-render-npy-sha256", required=True)
    worker.add_argument("--expected-render-tensor-sha256", required=True)
    worker.add_argument("--expected-target-rgb-sha256", required=True)
    worker.add_argument("--expected-lpips-state-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        record = preflight(args.outdir, args.comp008, args.comp009, args.comp010)
        print(record["preflight_binding_sha256"])
    elif args.command == "_lloyd-proof":
        sys.stdout.buffer.write(_canonical_json(lloyd_complexity_worker(args.native)))
    elif args.command == "_build-renderer":
        sys.stdout.buffer.write(_canonical_json(renderer_build_worker()))
    elif args.command == "_renderer-proof":
        renderer_record = _read_canonical_object(args.renderer_record)
        sys.stdout.buffer.write(
            _canonical_json(renderer_proof_worker(args.artifact_root, renderer_record))
        )
    elif args.command == "_produce-cell":
        sys.stdout.buffer.write(
            _canonical_json(
                produce_candidate_cell_worker(
                    artifact_root=args.artifact_root,
                    source_path=args.source,
                    captured_comp009_root=args.captured_comp009_root,
                    native_path=args.native,
                    lloyd_path=args.lloyd,
                    config_path=args.config,
                    output_root=args.output_root,
                )
            )
        )
    elif args.command == "_import-stream":
        sys.stdout.buffer.write(
            _canonical_json(import_stream_worker(args.source, args.config))
        )
    elif args.command == "_render-arm":
        renderer_record = _read_canonical_object(args.renderer_record)
        sys.stdout.buffer.write(
            _canonical_json(
                render_arm_worker(
                    args.artifact_root,
                    renderer_record,
                    args.blob,
                    args.native,
                    expected_blob_sha256=args.expected_blob_sha256,
                    expected_native_sha256=args.expected_native_sha256,
                    expected_native_bytes=args.expected_native_bytes,
                    expected_symbol_sha256=args.expected_symbol_sha256,
                    expected_boundary_state_sha256=args.expected_boundary_state_sha256,
                    output_path=args.output,
                )
            )
        )
    elif args.command == "_metric-worker":
        sys.stdout.buffer.write(
            _canonical_json(
                metric_worker(
                    args.render,
                    args.target_rgb,
                    height=args.height,
                    width=args.width,
                    expected_render_npy_sha256=args.expected_render_npy_sha256,
                    expected_render_tensor_sha256=args.expected_render_tensor_sha256,
                    expected_target_rgb_sha256=args.expected_target_rgb_sha256,
                    expected_lpips_state_sha256=args.expected_lpips_state_sha256,
                )
            )
        )
    elif args.command == "import-streams":
        print(import_streams(args.outdir)["import_streams_sha256"])
    elif args.command == "fit-encode-cold-decode-dev":
        print(fit_encode_cold_decode(args.outdir)["fit_stage_sha256"])
    elif args.command == "import-targets":
        print(import_targets(args.outdir)["target_copy_sha256"])
    elif args.command == "quality-select-dev":
        print(quality_select(args.outdir)["quality_stage_sha256"])
    elif args.command == "benchmark-dev":
        print(benchmark_development(args.outdir)["benchmark_stage_sha256"])
    elif args.command == "analyze":
        print(analyze_stage(args.outdir)["analysis_record_sha256"])
    elif args.command == "replay":
        print(replay_stage(args.outdir)["replay_sha256"])
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
