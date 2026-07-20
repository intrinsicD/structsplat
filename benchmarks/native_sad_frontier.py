"""BENCH-016: source-pinned native SAD versus frozen StructSplat controls.

The lifecycle is deliberately split.  ``preflight`` writes an immutable source/protocol binding
without opening a DIV2K image.  ``prepare-targets`` may only run against that live binding and
persists every source/derived hash before either scientific runner is enabled.  The runners append
one row per completed attempt; ``analyze`` and ``replay`` consume only persisted artifacts.

This adapter implements protocol v6 from ``tasks/BENCH-016-native-sad-frontier.md``.  Every arm
has three same-seed fresh-process execution repeats.  Both StructSplat controls use the maximum
terminal site count across the three native repeats.  SAD TXT sizes are diagnostics, never a
complete-stream or compression claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image


PROTOCOL = "bench016-native-sad-frontier-v6"
SCHEMA_VERSION = 6
UPSTREAM_COMMIT = "0eeb7e1e72b81e550f90db8ebbf432cdc57383ed"
TARGET_IDS = ("0057", "0171", "0285", "0399", "0513", "0627", "0741", "0785")
RATES = (0.5, 2.0)
NATIVE_REPEATS = (0, 1, 2)
MAX_SIDE = 768
ITERATIONS = 4_000
LOG_FREQUENCY = 20
RENDER_WARMUPS = 2
RENDER_SAMPLES = 51
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
SAD_BYTES_PER_SITE = 16
SAD_SCALARS_PER_SITE = 10
SS_SCALARS_PER_GAUSSIAN = 8
FLOAT32_BYTES = 4
SS_METHOD = "structsplat_best_checkpoint"
SS_ARMS = ("ss_same_n", "ss_equal_terminal_coordinates")
SS_SEED = 0
SS_START_FRACTION = 0.5
SS_GROWTH_WAVES = 5
SS_FEATURE_CAP = 12.0
SS_FEATURE_CAP_REFERENCE_SIDE = 160.0
CENTRAL_METRIC_NAMES = (
    "central_psnr",
    "central_ssim",
    "central_proxy_ms_ssim",
    "central_lpips_alex",
)

ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "tasks" / "BENCH-016-native-sad-frontier.md"
DEFAULT_SAD_REPO = ROOT / "results" / "native_envs" / "sad_official"
V4_ARTIFACT_DIR = ROOT / "results" / "bench016_native_sad_frontier_v4_2026-07-16"
V4_INVALIDITY_SHA256 = "a3b9c696ed0fc0d574f4e177353ffec6eb5ba2600aafca640425a33a35d2f447"
V4_BINDING_FILE_SHA256 = "473f458ec5d3729829dd34b371ab7e6b82f5d87f2e0d57e140afb1b69c65ce71"
V4_INVALIDITY_BUNDLE_MANIFEST_SHA256 = (
    "338220bc9b1cf8a96508bb5be7d23a8c6f2217b1dd3aeab3b9ae0ff7bdf9f1f0"
)
V4_INVALIDITY_BUNDLE_FILE_COUNT = 45
V4_INVALIDITY_BUNDLE_TOTAL_BYTES = 6_950_227
V5_ARTIFACT_DIR = ROOT / "results" / "bench016_native_sad_frontier_v5_2026-07-16"
V5_BINDING_FILE_SHA256 = "a87aeed30594bf12f7fe711df332ef794c13e589c38fe7ff103fa90fe8b1b6f3"
V5_INVALIDITY_SHA256 = "333bd6eeb13bc4931e46808f3dbb10e4fa63ddeafe03481378002b07abce39b4"
V5_INVALIDITY_BUNDLE_MANIFEST_SHA256 = (
    "8eac6e6210de179c1a39251051f38102238d9a5634489a160709b106a5e664c7"
)
V5_INVALIDITY_BUNDLE_FILE_COUNT = 165
V5_INVALIDITY_BUNDLE_TOTAL_BYTES = 31_140_605
REQUIRED_LD_PRELOAD = Path("/lib/x86_64-linux-gnu/libstdc++.so.6").resolve()
LIFECYCLE_FILES = frozenset({"artifact_manifest.json", "replay.json", "completion.json"})

_TRACE_RE = re.compile(
    r"^Iter\s+(?P<iteration>\d+)\s+\|\s+PSNR:\s+"
    r"(?P<psnr>[-+0-9.eE]+)\s+dB(?:\s+\|\s+SSIM:\s+(?P<ssim>[-+0-9.eE]+))?"
    r"\s+\|\s+Active:\s+(?P<active>\d+)/(?P<capacity>\d+)"
    r"\s+\|\s+(?P<speed>[-+0-9.eE]+)\s+it/s\s+\|\s+"
    r"(?P<elapsed>[-+0-9.eE]+)s$"
)
_TRAINING_TIME_RE = re.compile(r"^Training time:\s+([-+0-9.eE]+)\s+s$")
_TOTAL_TIME_RE = re.compile(r"^Total time:\s+([-+0-9.eE]+)\s+s$")
_RENDER_TIME_RE = re.compile(
    r"^Candidate build:\s+([-+0-9.eE]+)\s+ms\s+\|\s+Render:\s+"
    r"([-+0-9.eE]+)\s+ms$"
)
_RENDER_LOADED_RE = re.compile(r"^Loaded\s+(\d+)\s+sites$")
_RENDER_SIZE_RE = re.compile(r"^Render size:\s+(\d+)x(\d+)$")
_RENDER_SAVED_RE = re.compile(r"^Saved:\s+(.+)$")
COMMON_AUC_START = 1.0 / ITERATIONS


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(pixels: np.ndarray) -> str:
    """Hash shape and contiguous RGB8 pixels using the repository benchmark convention."""
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable artifact already exists with different bytes: {path}")


def _write_json_once(path: Path, payload: Any) -> None:
    _write_once(path, _canonical_json(payload))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read strict JSON artifact {path}") from exc
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise RuntimeError(f"truncated JSONL row {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed JSONL row {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row {path}:{line_number}")
            rows.append(row)
    return rows


def _checked_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        list(command), cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_state(repo: Path) -> dict[str, Any]:
    commit = _checked_output(("git", "rev-parse", "HEAD"), cwd=repo)
    tree = _checked_output(("git", "rev-parse", "HEAD^{tree}"), cwd=repo)
    status = _checked_output(
        ("git", "status", "--porcelain", "--untracked-files=normal"), cwd=repo
    )
    diff = subprocess.check_output(("git", "diff", "--binary", "HEAD"), cwd=repo)
    origin = subprocess.run(
        ("git", "config", "--get", "remote.origin.url"),
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return {
        "commit": commit,
        "tree": tree,
        "clean": not bool(status),
        "status": status.splitlines(),
        "diff_sha256": _sha256_bytes(diff),
        "origin": origin,
    }


def _source_files() -> dict[str, Path]:
    return {
        "adapter": Path(__file__).resolve(),
        "adapter_tests": (ROOT / "tests" / "test_native_sad_frontier.py").resolve(),
        "task": TASK_PATH.resolve(),
        "benchmarks_package_init": (ROOT / "benchmarks" / "__init__.py").resolve(),
        "benchmark_util": (ROOT / "benchmarks" / "_util.py").resolve(),
        "benchmark_common": (ROOT / "benchmarks" / "common.py").resolve(),
        "fair_density": (ROOT / "benchmarks" / "fair_density_control_compare.py").resolve(),
        "storage_budget": (ROOT / "benchmarks" / "storage_budget.py").resolve(),
        "div2k_subset_acquire": (ROOT / "benchmarks" / "div2k_subset_acquire.py").resolve(),
        "structsplat_package_init": (ROOT / "src" / "structsplat" / "__init__.py").resolve(),
        "codec": (ROOT / "src" / "structsplat" / "codec.py").resolve(),
        "config": (ROOT / "src" / "structsplat" / "config.py").resolve(),
        "cuda_render": (ROOT / "src" / "structsplat" / "cuda_render.py").resolve(),
        "cuda_render_cpp": (ROOT / "src" / "structsplat" / "cuda" / "render_ext.cpp").resolve(),
        "cuda_render_cu": (ROOT / "src" / "structsplat" / "cuda" / "render_ext.cu").resolve(),
        "fit": (ROOT / "src" / "structsplat" / "fit.py").resolve(),
        "gaussians": (ROOT / "src" / "structsplat" / "gaussians.py").resolve(),
        "init": (ROOT / "src" / "structsplat" / "init.py").resolve(),
        "density": (ROOT / "src" / "structsplat" / "density.py").resolve(),
        "metrics": (ROOT / "src" / "structsplat" / "metrics.py").resolve(),
        "render": (ROOT / "src" / "structsplat" / "render.py").resolve(),
        "sampling": (ROOT / "src" / "structsplat" / "sampling.py").resolve(),
        "structure_tensor": (
            ROOT / "src" / "structsplat" / "structure_tensor.py"
        ).resolve(),
    }


def _source_hashes() -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for name, path in _source_files().items()
    }


def _command_version(command: Sequence[str]) -> str | None:
    try:
        return _checked_output(command).splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment() -> dict[str, Any]:
    from benchmarks._util import package_versions, repository_state

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": package_versions(),
        "monitoring_packages": {
            name: importlib.metadata.version(name)
            for name in ("psutil", "nvidia-ml-py")
        },
        "repository": repository_state(),
        "nvcc": _command_version(("nvcc", "--version")),
        "cmake": _command_version(("cmake", "--version")),
        "cxx": _command_version(("c++", "--version")),
        "nvidia_smi": _command_version(("nvidia-smi",)),
    }


def _protocol_spec(sad_repo: Path, source_dir: Path | None) -> dict[str, Any]:
    from structsplat.codec import CodecConfig
    from structsplat.config import FitConfig

    codec = CodecConfig()
    fit = FitConfig(
        iters=ITERATIONS,
        renderer="cuda",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=LOG_FREQUENCY,
        checkpoint_policy="terminal",  # fair-density applies the selected method override
        compute_lpips=False,
        color_basis="constant",
        qat_mode="off",
        adaptive_count=False,
    )
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "decision_scope": "development_external_validity_not_sota",
        "preregistration_scope": (
            "v6 outcome-responsive integrity repair frozen after v4 and v5 DIV2K access; "
            "data, methods, rates, repeats, horizon, quality estimator, aggregation, and "
            "scientific gates remain unchanged"
        ),
        "fresh_execution": {
            "required_native_rows": 48,
            "required_structsplat_rows": 96,
            "required_total_rows": 144,
            "v4_scientific_row_imports_allowed": False,
            "v5_scientific_row_imports_allowed": False,
        },
        "further_structurally_distinct_integrity_failure_action": (
            "terminate BENCH-016 as invalid/no-decision; no v7 repair"
        ),
        "target_source_url": (
            "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
        ),
        "target_ids": list(TARGET_IDS),
        "sealed_validation_ids": "0801--0900",
        "source_dir_declared": None if source_dir is None else str(source_dir.resolve()),
        "resize": {
            "library": "Pillow",
            "mode": "RGB",
            "resampling": "Image.Resampling.LANCZOS",
            "max_side": MAX_SIDE,
            "dimension_rule": "round(native_dimension * 768 / max(native_dimensions))",
            "pixel_hash_rule": "sha256(shape<int64-le> || contiguous RGB8 bytes)",
        },
        "sad": {
            "repo": str(sad_repo.resolve()),
            "commit": UPSTREAM_COMMIT,
            "backend": "cuda",
            "config": "training_config.json",
            "rates_bpp": list(RATES),
            "repeats": list(NATIVE_REPEATS),
            "iterations": ITERATIONS,
            "log_frequency": LOG_FREQUENCY,
            "bytes_per_site_nominal": SAD_BYTES_PER_SITE,
            "trainable_scalars_per_site": SAD_SCALARS_PER_SITE,
            "compressed_txt_diagnostic": "gzip level=9 mtime=0",
            "quality_reconstruction": (
                "role-fixed first fresh official saved-TXT replay, invoked once without retry "
                "or selection; training-process and 53 timing-replay PNGs are diagnostic"
            ),
            "recipient_replay_scope": (
                "saved TXT plus pinned official decoder/config; recipient-replayable only, "
                "not deployable, self-contained, or compressed"
            ),
        },
        "structsplat": {
            "method": SS_METHOD,
            "seed": SS_SEED,
            "execution_repeats": list(NATIVE_REPEATS),
            "start_fraction": SS_START_FRACTION,
            "growth_waves": SS_GROWTH_WAVES,
            "start_count_rule": "max(16, round(0.5 * final_count))",
            "split_every": ITERATIONS // (SS_GROWTH_WAVES + 1),
            "split_count_rule": "ceil((final_count - start_count) / 5)",
            "split_schedule_stops_at_max": False,
            "growth_contract": (
                "five nonempty additions reach cap; scheduled opportunity at update 3996 is "
                "a cap no-op"
            ),
            "renderer": "cuda",
            "pixel_loss": "l1",
            "ssim_weight": 0.3,
            "feature_cap": SS_FEATURE_CAP,
            "feature_cap_reference_side": SS_FEATURE_CAP_REFERENCE_SIDE,
            "checkpoint_policy": "best_psnr_final_count",
            "same_n": "max(final_sad_sites over repeats)",
            "equal_terminal_coordinates": "ceil(10 * max(final_sad_sites) / 8)",
            "trainable_scalars_per_gaussian": SS_SCALARS_PER_GAUSSIAN,
            "base_fit": asdict(fit),
            "codec_config": asdict(codec),
        },
        "timing": {
            "render_warmups": RENDER_WARMUPS,
            "render_samples": RENDER_SAMPLES,
            "summary": "median",
            "sad_prepared_sample": "upstream CUDA event Render field, candidate build excluded",
            "sad_primary_replay_non_kernel_residual_upper_bound": (
                "max(0, fresh process wall - Candidate build CUDA event - Render CUDA event); "
                "includes process startup, configuration, CUDA context, TXT parse/upload, and PNG output"
            ),
            "sad_output_semantics": (
                "all 53 timing invocations retain ordered unique outputs and exact command, "
                "site-count, dimensions, saved-path, process-boundary, byte/pixel-hash, and "
                "stdout semantics; pixel variation from the role-fixed primary replay is "
                "threshold-free diagnostic evidence"
            ),
            "structsplat_prepared_sample": "CUDA event exact-render field",
            "structsplat_cold": "persisted SSPL1 read + decode + exact render wall clock",
        },
        "analysis": {
            "all_arm_repeat_scalar_aggregation": "arithmetic mean",
            "all_arm_trace_aggregation": "linear interpolation on union progress grid then mean",
            "all_arm_time_memory_aggregation": "median",
            "native_progress_rule": "stdout k is post-update (k+1)/4000",
            "structsplat_progress_rule": (
                "history k is pre-update k/4000; separately recorded trajectory-terminal at 1"
            ),
            "common_auc_interval": [1.0 / ITERATIONS, 1.0],
            "auc_scope": "internal_training_state_normalized_iteration",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_rng": "numpy.PCG64",
        },
        "claims": {
            "compression_tested": False,
            "sad_recipient_replayable": True,
            "sad_deployable": False,
            "sad_self_contained": False,
            "sad_compressed_representation": False,
            "production_authorized": False,
            "sota_claim_authorized": False,
        },
    }


def _sad_paths(sad_repo: Path) -> dict[str, Path]:
    return {
        "config": sad_repo / "training_config.json",
        "binary": sad_repo / "backends" / "cuda" / "bin" / "sad_cuda",
        "run_script": sad_repo / "run.sh",
    }


def _prebuild_structsplat_cuda() -> dict[str, Any]:
    """Build/load the exact renderer before any fresh-process timing boundary begins."""
    from structsplat.cuda_render import _load_extension

    extension = _load_extension()
    extension_path = Path(extension.__file__).resolve()
    if not extension_path.is_file():
        raise RuntimeError("StructSplat CUDA extension loaded without a persistent module file")
    return {
        "path": str(extension_path),
        "sha256": _sha256_file(extension_path),
        "bytes": extension_path.stat().st_size,
    }


def _preload_binding() -> dict[str, Any]:
    raw = os.environ.get("LD_PRELOAD")
    if not raw:
        raise RuntimeError(
            f"BENCH-016 requires LD_PRELOAD={REQUIRED_LD_PRELOAD} before preflight"
        )
    parts = [Path(part).resolve() for part in raw.split(":") if part]
    if parts != [REQUIRED_LD_PRELOAD]:
        raise RuntimeError(
            f"BENCH-016 requires the exact sole LD_PRELOAD {REQUIRED_LD_PRELOAD}, got {parts}"
        )
    if not REQUIRED_LD_PRELOAD.is_file():
        raise FileNotFoundError(REQUIRED_LD_PRELOAD)
    strings = subprocess.check_output(("strings", str(REQUIRED_LD_PRELOAD)), text=True)
    cxxabi = sorted(
        {line.strip() for line in strings.splitlines() if line.startswith("CXXABI_")}
    )
    return {
        "path": str(REQUIRED_LD_PRELOAD),
        "sha256": _sha256_file(REQUIRED_LD_PRELOAD),
        "bytes": REQUIRED_LD_PRELOAD.stat().st_size,
        "cxxabi_symbols": cxxabi,
    }


def _metric_weight_binding() -> dict[str, Any]:
    alexnet = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
    spec = importlib.util.find_spec("lpips")
    if spec is None or spec.origin is None:
        raise RuntimeError("BENCH-016 requires the installed lpips package")
    lpips_alex = Path(spec.origin).resolve().parent / "weights" / "v0.1" / "alex.pth"
    assets = {"torchvision_alexnet": alexnet.resolve(), "lpips_alex_v0_1": lpips_alex.resolve()}
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"BENCH-016 metric weights must be cached before binding; missing {missing}"
        )
    result = {
        name: {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for name, path in assets.items()
    }
    if not result["torchvision_alexnet"]["sha256"].startswith("7be5be79"):
        raise RuntimeError("cached torchvision AlexNet weight identity drift")
    if not result["lpips_alex_v0_1"]["sha256"].startswith("df73285e"):
        raise RuntimeError("installed LPIPS AlexNet calibration weight identity drift")
    return result


def _prior_invalidity_binding(
    *,
    artifact_dir: Path,
    protocol: str,
    binding_sha256: str,
    invalidity_sha256: str,
    bundle_sha256: str,
    bundle_file_count: int,
    bundle_total_bytes: int,
) -> dict[str, Any]:
    """Validate one sealed no-decision predecessor without importing scientific rows."""
    paths = {
        "binding": artifact_dir / "binding.json",
        "invalidity": artifact_dir / "invalidity.json",
        "invalidity_bundle_manifest": artifact_dir / "invalidity_bundle_manifest.json",
    }
    expected_hashes = {
        "binding": binding_sha256,
        "invalidity": invalidity_sha256,
        "invalidity_bundle_manifest": bundle_sha256,
    }
    for name, path in paths.items():
        if not path.is_file() or _sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"sealed BENCH-016 {protocol} predecessor {name} drift")

    invalidity = _load_json(paths["invalidity"])
    old_binding = _load_json(paths["binding"])
    bundle = _load_json(paths["invalidity_bundle_manifest"])
    raw_entries = bundle.get("files", [])
    if not isinstance(raw_entries, list):
        raise RuntimeError(f"sealed BENCH-016 {protocol} bundle inventory drift")
    bundle_entries = {str(record["path"]): record for record in raw_entries}
    if (
        len(bundle_entries) != len(raw_entries)
        or invalidity.get("protocol") != protocol
        or invalidity.get("status") != "invalid_no_decision"
        or invalidity.get("binding_file_sha256") != binding_sha256
        or invalidity.get("scientific_gate_evaluation_performed") is not False
        or old_binding.get("protocol") != protocol
        or bundle_entries.get("binding.json", {}).get("sha256")
        != binding_sha256
        or bundle_entries.get("invalidity.json", {}).get("sha256")
        != invalidity_sha256
    ):
        raise RuntimeError(f"sealed BENCH-016 {protocol} predecessor semantics drift")

    bundle_total = 0
    for relative, record in bundle_entries.items():
        candidate = (artifact_dir / relative).resolve()
        try:
            candidate.relative_to(artifact_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"{protocol} invalidity bundle path escapes its artifact") from exc
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(record["bytes"])
            or _sha256_file(candidate) != record["sha256"]
        ):
            raise RuntimeError(f"sealed BENCH-016 {protocol} bundle member drift: {relative}")
        bundle_total += int(record["bytes"])
    if (
        int(bundle.get("file_count", -1)) != len(bundle_entries)
        or int(bundle.get("file_count", -1)) != bundle_file_count
        or int(bundle.get("total_bytes", -1)) != bundle_total
        or int(bundle.get("total_bytes", -1)) != bundle_total_bytes
        or bundle.get("exclusions") != ["invalidity_bundle_manifest.json"]
    ):
        raise RuntimeError(f"sealed BENCH-016 {protocol} invalidity bundle inventory drift")

    return {
        name: {
            "path": str(path.resolve()),
            "sha256": expected_hashes[name],
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    } | {
        "protocol": protocol,
        "status": "invalid_no_decision",
        "scientific_target_access_occurred": True,
        "scientific_gate_evaluation_performed": False,
        "fresh_v6_rows_imported": 0,
        "bundle_file_count": int(bundle["file_count"]),
        "bundle_total_bytes": int(bundle["total_bytes"]),
    }


def _prior_v4_invalidity_binding() -> dict[str, Any]:
    return _prior_invalidity_binding(
        artifact_dir=V4_ARTIFACT_DIR,
        protocol="bench016-native-sad-frontier-v4",
        binding_sha256=V4_BINDING_FILE_SHA256,
        invalidity_sha256=V4_INVALIDITY_SHA256,
        bundle_sha256=V4_INVALIDITY_BUNDLE_MANIFEST_SHA256,
        bundle_file_count=V4_INVALIDITY_BUNDLE_FILE_COUNT,
        bundle_total_bytes=V4_INVALIDITY_BUNDLE_TOTAL_BYTES,
    )


def _prior_v5_invalidity_binding() -> dict[str, Any]:
    return _prior_invalidity_binding(
        artifact_dir=V5_ARTIFACT_DIR,
        protocol="bench016-native-sad-frontier-v5",
        binding_sha256=V5_BINDING_FILE_SHA256,
        invalidity_sha256=V5_INVALIDITY_SHA256,
        bundle_sha256=V5_INVALIDITY_BUNDLE_MANIFEST_SHA256,
        bundle_file_count=V5_INVALIDITY_BUNDLE_FILE_COUNT,
        bundle_total_bytes=V5_INVALIDITY_BUNDLE_TOTAL_BYTES,
    )


def _binding_payload(sad_repo: Path, source_dir: Path | None) -> dict[str, Any]:
    paths = _sad_paths(sad_repo)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned SAD files: {missing}")
    state = _git_state(sad_repo)
    if state["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"SAD commit mismatch: expected {UPSTREAM_COMMIT}, got {state['commit']}"
        )
    if not state["clean"]:
        raise RuntimeError(f"pinned SAD worktree is dirty: {state['status']}")
    config = _load_json(paths["config"])
    if int(config.get("DEFAULT_ITERS", -1)) != ITERATIONS:
        raise RuntimeError("upstream config DEFAULT_ITERS is not the frozen 4000")
    help_result = subprocess.run(
        (str(paths["binary"]), "--help"),
        cwd=sad_repo,
        env={**os.environ, "CONFIG_PATH": str(paths["config"].resolve())},
        text=True,
        capture_output=True,
        check=False,
    )
    help_text = help_result.stdout + help_result.stderr
    if help_result.returncode != 0 or "--target-bpp" not in help_text or "--render" not in help_text:
        raise RuntimeError("pinned SAD CUDA binary failed its target-free CLI check")
    preload = _preload_binding()
    metric_weights = _metric_weight_binding()
    prebuilt_cuda_extension = _prebuild_structsplat_cuda()
    predecessor_v4 = _prior_v4_invalidity_binding()
    predecessor_v5 = _prior_v5_invalidity_binding()
    protocol = _protocol_spec(sad_repo, source_dir)
    sources = _source_hashes()
    upstream = {
        "git": state,
        "config_sha256": _sha256_file(paths["config"]),
        "config_bytes": paths["config"].stat().st_size,
        "binary_sha256": _sha256_file(paths["binary"]),
        "binary_bytes": paths["binary"].stat().st_size,
        "run_script_sha256": _sha256_file(paths["run_script"]),
        "resolved_config": config,
    }
    binding_core = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "created_before_v6_scientific_execution": True,
        "created_before_any_scientific_target_access": False,
        "prior_scientific_target_access_disclosed": True,
        "v6_scientific_target_files_opened": False,
        "predecessor_v4_invalidity": predecessor_v4,
        "predecessor_v5_invalidity": predecessor_v5,
        "protocol_spec": protocol,
        "executed_source_files": sources,
        "upstream": upstream,
        "prebuilt_structsplat_cuda_extension": prebuilt_cuda_extension,
        "required_ld_preload": preload,
        "metric_weight_assets": metric_weights,
        "environment": _environment(),
    }
    binding_core["binding_sha256"] = _sha256_bytes(_canonical_json(binding_core))
    return binding_core


def _archive_binding_sources(outdir: Path, binding: dict[str, Any], sad_repo: Path) -> None:
    archive = outdir / "source_archive"
    for name, record in binding["executed_source_files"].items():
        source = Path(record["path"])
        target = archive / "structsplat" / f"{name}{source.suffix}"
        _write_once(target, source.read_bytes())
        if _sha256_file(target) != record["sha256"]:
            raise RuntimeError(f"source archive hash mismatch for {name}")
    for name, source in _sad_paths(sad_repo).items():
        if name == "binary":
            continue
        target = archive / "sad" / source.name
        _write_once(target, source.read_bytes())
    for version in ("v4", "v5"):
        predecessor = binding[f"predecessor_{version}_invalidity"]
        for name in ("binding", "invalidity", "invalidity_bundle_manifest"):
            record = predecessor[name]
            source = Path(record["path"])
            target = archive / f"prior_{version}" / source.name
            _write_once(target, source.read_bytes())
            if _sha256_file(target) != record["sha256"]:
                raise RuntimeError(f"archived {version} predecessor hash mismatch for {name}")


def run_preflight(outdir: Path, sad_repo: Path, source_dir: Path | None = None) -> dict[str, Any]:
    """Bind sources/protocol without listing, opening, or decoding the DIV2K directory."""
    from benchmarks import div2k_subset_acquire as acquisition

    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError("BENCH-016 preflight requires an empty output directory")
    outdir.mkdir(parents=True, exist_ok=True)
    binding_path = outdir / "binding.json"
    candidate = _binding_payload(sad_repo.resolve(), source_dir)
    _write_json_once(binding_path, candidate)
    binding = _load_json(binding_path)
    _archive_binding_sources(outdir, binding, sad_repo.resolve())
    checks = {
        "protocol_v6": binding["protocol"] == PROTOCOL,
        "target_ids_frozen": binding["protocol_spec"]["target_ids"] == list(TARGET_IDS),
        "three_native_repeats": binding["protocol_spec"]["sad"]["repeats"] == [0, 1, 2],
        "upstream_commit": binding["upstream"]["git"]["commit"] == UPSTREAM_COMMIT,
        "upstream_clean": binding["upstream"]["git"]["clean"] is True,
        "cuda_binary_bound": len(binding["upstream"]["binary_sha256"]) == 64,
        "config_unchanged": binding["upstream"]["config_sha256"] == _sha256_file(
            sad_repo / "training_config.json"
        ),
        "adapter_bound": binding["executed_source_files"]["adapter"]["sha256"]
        == _sha256_file(__file__),
        "acquisition_helper_contract": (
            acquisition.FROZEN_IDS == TARGET_IDS
            and acquisition.FROZEN_URL
            == binding["protocol_spec"]["target_source_url"]
            and binding["executed_source_files"]["div2k_subset_acquire"]["sha256"]
            == _sha256_file(acquisition.__file__)
        ),
        "task_bound": binding["executed_source_files"]["task"]["sha256"]
        == _sha256_file(TASK_PATH),
        "prior_scientific_access_disclosed": (
            binding["created_before_any_scientific_target_access"] is False
            and binding["prior_scientific_target_access_disclosed"] is True
            and binding["predecessor_v4_invalidity"]["scientific_target_access_occurred"]
            is True
            and binding["predecessor_v5_invalidity"]["scientific_target_access_occurred"]
            is True
        ),
        "no_v6_scientific_target_access": binding["v6_scientific_target_files_opened"] is False,
        "v4_invalidity_bound": (
            binding["predecessor_v4_invalidity"]["invalidity"]["sha256"]
            == V4_INVALIDITY_SHA256
            and binding["predecessor_v4_invalidity"]["binding"]["sha256"]
            == V4_BINDING_FILE_SHA256
            and binding["predecessor_v4_invalidity"]["invalidity_bundle_manifest"]["sha256"]
            == V4_INVALIDITY_BUNDLE_MANIFEST_SHA256
        ),
        "v5_invalidity_bound": (
            binding["predecessor_v5_invalidity"]["invalidity"]["sha256"]
            == V5_INVALIDITY_SHA256
            and binding["predecessor_v5_invalidity"]["binding"]["sha256"]
            == V5_BINDING_FILE_SHA256
            and binding["predecessor_v5_invalidity"]["invalidity_bundle_manifest"]["sha256"]
            == V5_INVALIDITY_BUNDLE_MANIFEST_SHA256
            and binding["predecessor_v5_invalidity"]["bundle_file_count"]
            == V5_INVALIDITY_BUNDLE_FILE_COUNT
            and binding["predecessor_v5_invalidity"]["bundle_total_bytes"]
            == V5_INVALIDITY_BUNDLE_TOTAL_BYTES
        ),
        "fresh_v6_restart": (
            binding["predecessor_v4_invalidity"]["fresh_v6_rows_imported"] == 0
            and binding["predecessor_v5_invalidity"]["fresh_v6_rows_imported"] == 0
        ),
        "compression_claim_false": binding["protocol_spec"]["claims"]["compression_tested"]
        is False,
    }
    result = {
        "protocol": PROTOCOL,
        "binding_sha256": _sha256_file(binding_path),
        "checks": checks,
        "preflight_pass": all(checks.values()),
        "v6_scientific_cells_present": False,
        "prior_v4_and_v5_scientific_cells_present": True,
        "development_run_authorized": all(checks.values()),
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }
    _write_json_once(outdir / "preflight.json", result)
    if not result["preflight_pass"]:
        raise RuntimeError(f"BENCH-016 preflight failed: {checks}")
    return result


def _validate_binding(outdir: Path) -> dict[str, Any]:
    binding_path = outdir / "binding.json"
    preflight_path = outdir / "preflight.json"
    binding = _load_json(binding_path)
    preflight = _load_json(preflight_path)
    if binding.get("protocol") != PROTOCOL or preflight.get("preflight_pass") is not True:
        raise RuntimeError("BENCH-016 binding/preflight is absent or not protocol-v6 passing")
    core = dict(binding)
    recorded_hash = core.pop("binding_sha256", None)
    if recorded_hash != _sha256_bytes(_canonical_json(core)):
        raise RuntimeError("binding self-hash mismatch")
    for name, record in binding["executed_source_files"].items():
        path = Path(record["path"])
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"live executed source drift: {name}")
    sad_repo = Path(binding["protocol_spec"]["sad"]["repo"])
    state = _git_state(sad_repo)
    if state["commit"] != UPSTREAM_COMMIT or not state["clean"]:
        raise RuntimeError("live pinned SAD repository drift")
    sad_paths = _sad_paths(sad_repo)
    if _sha256_file(sad_paths["config"]) != binding["upstream"]["config_sha256"]:
        raise RuntimeError("live SAD config drift")
    if _sha256_file(sad_paths["binary"]) != binding["upstream"]["binary_sha256"]:
        raise RuntimeError("live SAD binary drift")
    extension = binding.get("prebuilt_structsplat_cuda_extension") or {}
    extension_path = Path(str(extension.get("path", "")))
    if not extension_path.is_file() or _sha256_file(extension_path) != extension.get("sha256"):
        raise RuntimeError("prebuilt StructSplat CUDA extension drift")
    preload = binding.get("required_ld_preload") or {}
    live_preload = _preload_binding()
    if live_preload != preload:
        raise RuntimeError("bound LD_PRELOAD drift")
    if _metric_weight_binding() != binding.get("metric_weight_assets"):
        raise RuntimeError("bound central-metric weight drift")
    if _prior_v4_invalidity_binding() != binding.get("predecessor_v4_invalidity"):
        raise RuntimeError("bound v4 predecessor invalidity history drift")
    if _prior_v5_invalidity_binding() != binding.get("predecessor_v5_invalidity"):
        raise RuntimeError("bound v5 predecessor invalidity history drift")
    return binding


def _orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


def _resolve_div2k_source(source_dir: Path, image_id: str) -> Path:
    candidates = (
        source_dir / f"{image_id}.png",
        source_dir / "DIV2K_train_HR" / f"{image_id}.png",
    )
    found = [path.resolve() for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected exactly one extracted DIV2K source for {image_id}, found {found}"
        )
    return found[0]


def _validate_acquisition_source(source_dir: Path) -> dict[str, Any]:
    from benchmarks import div2k_subset_acquire as acquisition

    manifest_path = source_dir / acquisition.MANIFEST_NAME
    sidecar_path = source_dir / acquisition.MANIFEST_HASH_NAME
    manifest = _load_json(manifest_path)
    file_hash = _sha256_file(manifest_path)
    if sidecar_path.read_text(encoding="ascii") != (
        f"{file_hash}  {acquisition.MANIFEST_NAME}\n"
    ):
        raise RuntimeError("DIV2K subset acquisition manifest sidecar mismatch")
    core = dict(manifest)
    recorded_payload_hash = core.pop("manifest_payload_sha256", None)
    if recorded_payload_hash != _sha256_bytes(acquisition.canonical_json(core)):
        raise RuntimeError("DIV2K subset acquisition manifest self-hash mismatch")
    if (
        manifest.get("schema") != acquisition.MANIFEST_SCHEMA
        or manifest.get("frozen_ids") != list(TARGET_IDS)
        or manifest.get("remotezip_version") != acquisition.REMOTEZIP_VERSION
        or not manifest.get("range_requests")
    ):
        raise RuntimeError("DIV2K subset acquisition protocol drift")
    members = manifest.get("members")
    if not isinstance(members, list) or [member.get("id") for member in members] != list(TARGET_IDS):
        raise RuntimeError("DIV2K subset acquisition member matrix drift")
    for member in members:
        path = source_dir / str(member["local_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(member["byte_size"])
            or _sha256_file(path) != member["sha256"]
        ):
            raise RuntimeError(f"acquired DIV2K source drift for {member['id']}")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_hash,
        "manifest_payload_sha256": recorded_payload_hash,
        "manifest_hash_sidecar_sha256": _sha256_file(sidecar_path),
        "remotezip_version": manifest["remotezip_version"],
        "range_request_count": len(manifest["range_requests"]),
    }


def _prepare_one_target(source: Path, target: Path, image_id: str) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    with Image.open(source) as opened:
        opened.load()
        native_width, native_height = opened.size
        rgb = opened.convert("RGB")
        source_pixels = np.asarray(rgb, dtype=np.uint8)
    scale = MAX_SIDE / max(native_width, native_height)
    derived_width = round(native_width * scale)
    derived_height = round(native_height * scale)
    if max(derived_width, derived_height) != MAX_SIDE:
        raise RuntimeError(f"resize rule failed to produce a {MAX_SIDE}-pixel side for {image_id}")
    resized = rgb.resize((derived_width, derived_height), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    resized.save(target, format="PNG")
    with Image.open(target) as persisted:
        persisted.load()
        target_pixels = np.asarray(persisted.convert("RGB"), dtype=np.uint8)
    return {
        "image_id": image_id,
        "source_path": str(source),
        "source_byte_sha256": _sha256_bytes(source_bytes),
        "source_bytes": len(source_bytes),
        "source_pixel_sha256": _pixel_sha256(source_pixels),
        "source_height": native_height,
        "source_width": native_width,
        "source_orientation": _orientation(native_width, native_height),
        "derived_path": str(target.resolve()),
        "derived_byte_sha256": _sha256_file(target),
        "derived_bytes": target.stat().st_size,
        "derived_pixel_sha256": _pixel_sha256(target_pixels),
        "derived_height": derived_height,
        "derived_width": derived_width,
        "derived_orientation": _orientation(derived_width, derived_height),
        "pixel_count": derived_height * derived_width,
        "scale": scale,
    }


def prepare_targets(outdir: Path, source_dir: Path) -> dict[str, Any]:
    binding = _validate_binding(outdir)
    declared = binding["protocol_spec"].get("source_dir_declared")
    if declared is not None and Path(declared).resolve() != source_dir.resolve():
        raise RuntimeError("source directory differs from the preflight declaration")
    manifest_path = outdir / "targets.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        _validate_targets(outdir, manifest)
        return manifest
    target_dir = outdir / "targets"
    if target_dir.exists() and any(target_dir.iterdir()):
        raise RuntimeError("unbound target artifacts already exist; use a fresh output directory")
    acquisition = _validate_acquisition_source(source_dir.resolve())
    records = []
    for image_id in TARGET_IDS:
        source = _resolve_div2k_source(source_dir.resolve(), image_id)
        records.append(_prepare_one_target(source, target_dir / f"{image_id}.png", image_id))
    manifest = {
        "protocol": PROTOCOL,
        "binding_sha256": _sha256_file(outdir / "binding.json"),
        "target_ids": list(TARGET_IDS),
        "records": records,
        "acquisition": acquisition,
        "all_derived_before_training": True,
        "scientific_outcomes_present_at_write": False,
    }
    _write_json_once(manifest_path, manifest)
    _validate_targets(outdir, manifest)
    return manifest


def _validate_targets(
    outdir: Path,
    manifest: dict[str, Any] | None = None,
    *,
    validate_acquisition: bool = False,
) -> dict[str, Any]:
    manifest = manifest or _load_json(outdir / "targets.json")
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("binding_sha256") != _sha256_file(outdir / "binding.json")
        or manifest.get("target_ids") != list(TARGET_IDS)
        or manifest.get("all_derived_before_training") is not True
        or manifest.get("scientific_outcomes_present_at_write") is not False
    ):
        raise RuntimeError("target manifest protocol/ID drift")
    if validate_acquisition:
        acquisition_record = manifest.get("acquisition") or {}
        acquisition_path = Path(str(acquisition_record.get("manifest_path", "")))
        live_acquisition = _validate_acquisition_source(acquisition_path.parent)
        if live_acquisition != acquisition_record:
            raise RuntimeError("DIV2K acquisition provenance drift")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(TARGET_IDS):
        raise RuntimeError("target manifest is incomplete")
    for expected, record in zip(TARGET_IDS, records):
        if record.get("image_id") != expected:
            raise RuntimeError("target manifest order drift")
        path = Path(record["derived_path"])
        if not path.is_file() or _sha256_file(path) != record["derived_byte_sha256"]:
            raise RuntimeError(f"derived target byte drift for {expected}")
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if list(pixels.shape[:2]) != [record["derived_height"], record["derived_width"]]:
            raise RuntimeError(f"derived target shape drift for {expected}")
        if _pixel_sha256(pixels) != record["derived_pixel_sha256"]:
            raise RuntimeError(f"derived target pixel drift for {expected}")
        source_height = int(record["source_height"])
        source_width = int(record["source_width"])
        expected_scale = MAX_SIDE / max(source_height, source_width)
        if (
            max(int(record["derived_width"]), int(record["derived_height"])) != MAX_SIDE
            or int(record["pixel_count"])
            != int(record["derived_width"]) * int(record["derived_height"])
            or int(record["derived_width"]) != round(source_width * expected_scale)
            or int(record["derived_height"]) != round(source_height * expected_scale)
            or not math.isclose(float(record["scale"]), expected_scale)
            or _orientation(source_width, source_height) != record["source_orientation"]
            or _orientation(record["derived_width"], record["derived_height"])
            != record["derived_orientation"]
            or record["derived_orientation"] != record["source_orientation"]
        ):
            raise RuntimeError(f"derived target orientation drift for {expected}")
        if validate_acquisition:
            source_path = Path(record["source_path"])
            if (
                not source_path.is_file()
                or source_path.stat().st_size != int(record["source_bytes"])
                or _sha256_file(source_path) != record["source_byte_sha256"]
            ):
                raise RuntimeError(f"DIV2K source byte drift for {expected}")
            with Image.open(source_path) as source_image:
                source_image.load()
                source_pixels = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
            if (
                source_pixels.shape != (source_height, source_width, 3)
                or _pixel_sha256(source_pixels) != record["source_pixel_sha256"]
            ):
                raise RuntimeError(f"DIV2K source pixel drift for {expected}")
    return manifest


def _target_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["image_id"]: record for record in manifest["records"]}


def parse_sad_trace(text: str) -> list[dict[str, Any]]:
    trace = []
    for raw in text.splitlines():
        match = _TRACE_RE.match(raw.strip())
        if not match:
            continue
        groups = match.groupdict()
        trace.append(
            {
                "iteration": int(groups["iteration"]),
                "psnr": float(groups["psnr"]),
                "ssim": None if groups["ssim"] is None else float(groups["ssim"]),
                "active_sites": int(groups["active"]),
                "capacity_sites": int(groups["capacity"]),
                "iterations_per_second": float(groups["speed"]),
                "elapsed_seconds": float(groups["elapsed"]),
            }
        )
    if not trace:
        raise RuntimeError("native SAD stdout contains no convergence rows")
    iterations = [row["iteration"] for row in trace]
    expected = [*range(0, ITERATIONS, LOG_FREQUENCY), ITERATIONS - 1]
    if iterations != expected:
        raise RuntimeError(f"native SAD trace is incomplete or non-monotone: {iterations[:2]}...{iterations[-2:]}")
    return trace


def _parse_named_time(text: str, regex: re.Pattern[str], name: str) -> float:
    matches = [float(match.group(1)) for line in text.splitlines() if (match := regex.match(line.strip()))]
    if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0.0:
        raise RuntimeError(f"expected exactly one finite positive {name} in native stdout")
    return matches[0]


def _parse_render_time(text: str) -> tuple[float, float]:
    matches = [_RENDER_TIME_RE.match(line.strip()) for line in text.splitlines()]
    values = [(float(match.group(1)), float(match.group(2))) for match in matches if match]
    if len(values) != 1 or not all(math.isfinite(value) and value >= 0.0 for value in values[0]):
        raise RuntimeError("expected one finite native candidate/render timing row")
    return values[0]


def _parse_render_stdout_semantics(
    text: str,
    *,
    expected_site_count: int,
    expected_width: int,
    expected_height: int,
    expected_saved_path: Path,
) -> dict[str, Any]:
    """Fail closed unless an official replay proves the exact requested output semantics."""
    stripped = [line.strip() for line in text.splitlines()]
    loaded = [int(match.group(1)) for line in stripped if (match := _RENDER_LOADED_RE.match(line))]
    sizes = [
        (int(match.group(1)), int(match.group(2)))
        for line in stripped
        if (match := _RENDER_SIZE_RE.match(line))
    ]
    saved = [match.group(1) for line in stripped if (match := _RENDER_SAVED_RE.match(line))]
    candidate_ms, render_ms = _parse_render_time(text)
    if loaded != [expected_site_count]:
        raise RuntimeError("native replay stdout site-count semantics drift")
    if sizes != [(expected_width, expected_height)]:
        raise RuntimeError("native replay stdout dimension semantics drift")
    if len(saved) != 1 or Path(saved[0]).resolve() != expected_saved_path.resolve():
        raise RuntimeError("native replay stdout saved-path semantics drift")
    return {
        "loaded_site_count": loaded[0],
        "render_width": sizes[0][0],
        "render_height": sizes[0][1],
        "saved_path": str(expected_saved_path.resolve()),
        "candidate_build_ms": candidate_ms,
        "render_ms": render_ms,
    }


def _command_sha256(command: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_json(list(command)))


def _process_tree_pids(root_pid: int) -> set[int]:
    pending = [int(root_pid)]
    found: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    return found


def _process_tree_rss_bytes(root_pid: int) -> tuple[int | None, list[int]]:
    pids = sorted(_process_tree_pids(root_pid))
    total = 0
    seen = False
    for pid in pids:
        status = Path(f"/proc/{pid}/status")
        try:
            lines = status.read_text().splitlines()
        except (FileNotFoundError, PermissionError):
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                total += int(line.split()[1]) * 1024
                seen = True
                break
    return (total if seen else None), pids


def _nvml_process_bytes(pynvml: Any, pids: set[int]) -> int | None:
    try:
        values = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in processes:
                used = getattr(process, "usedGpuMemory", None)
                if int(process.pid) in pids and used is not None and int(used) >= 0:
                    values.append(int(used))
    except Exception:
        return None
    return sum(values) if values else 0


class _ProcessResourceSampler:
    def __init__(self, pid: int, interval_seconds: float = 0.01):
        self.pid = int(pid)
        self.interval_seconds = interval_seconds
        self.started_ns = time.monotonic_ns()
        self.rss_samples: list[dict[str, Any]] = []
        self.gpu_samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> _ProcessResourceSampler:
        def sample_rss() -> None:
            while not self._stop.is_set():
                value, pids = _process_tree_rss_bytes(self.pid)
                if value is not None:
                    self.rss_samples.append(
                        {
                            "elapsed_ns": time.monotonic_ns() - self.started_ns,
                            "bytes": value,
                            "pids": pids,
                        }
                    )
                self._stop.wait(self.interval_seconds)

        def sample_gpu() -> None:
            try:
                import pynvml

                pynvml.nvmlInit()
                while not self._stop.is_set():
                    pids = _process_tree_pids(self.pid)
                    value = _nvml_process_bytes(pynvml, pids)
                    if value is not None:
                        self.gpu_samples.append(
                            {
                                "elapsed_ns": time.monotonic_ns() - self.started_ns,
                                "bytes": value,
                                "pids": sorted(pids),
                            }
                        )
                    self._stop.wait(self.interval_seconds)
            except Exception:
                return
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

        self._threads = [
            threading.Thread(target=sample_rss, daemon=True),
            threading.Thread(target=sample_gpu, daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)

    @property
    def peak_rss(self) -> int | None:
        return max((sample["bytes"] for sample in self.rss_samples), default=None)

    @property
    def peak_gpu(self) -> int | None:
        positive = [sample["bytes"] for sample in self.gpu_samples if sample["bytes"] > 0]
        return max(positive, default=None)

    def payload(self) -> dict[str, Any]:
        def cadence(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
            elapsed = np.asarray([sample["elapsed_ns"] for sample in samples], dtype=np.int64)
            if len(elapsed) < 2:
                return {"samples": len(elapsed), "median_interval_ns": None, "max_interval_ns": None}
            intervals = np.diff(elapsed)
            return {
                "samples": len(elapsed),
                "median_interval_ns": int(np.median(intervals)),
                "max_interval_ns": int(intervals.max()),
            }

        return {
            "sampling_interval_seconds_requested": self.interval_seconds,
            "clock": "time.monotonic_ns",
            "rss_source": "/proc process-tree VmRSS sum",
            "gpu_source": "NVML compute-app process-tree usedGpuMemory sum",
            "rss_samples": self.rss_samples,
            "gpu_samples": self.gpu_samples,
            "peak_rss_bytes": self.peak_rss,
            "peak_gpu_device_bytes": self.peak_gpu,
            "rss_observed_cadence": cadence(self.rss_samples),
            "gpu_observed_cadence": cadence(self.gpu_samples),
        }


def _timed_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    resource_path: Path,
) -> dict[str, Any]:
    if stdout_path.exists() or resource_path.exists():
        raise RuntimeError(f"append-only command artifacts already exist: {stdout_path}")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.monotonic_ns()
    with stdout_path.open("xb") as stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        with _ProcessResourceSampler(process.pid) as sampler:
            returncode = process.wait()
            ended_ns = time.monotonic_ns()
    elapsed = (ended_ns - started_ns) / 1e9
    resource_payload = sampler.payload()
    resource_payload.update(
        {
            "process_started_monotonic_ns": started_ns,
            "process_ended_monotonic_ns": ended_ns,
            "process_wall_seconds": elapsed,
        }
    )
    _write_json_once(resource_path, resource_payload)
    return {
        "returncode": returncode,
        "wall_seconds": elapsed,
        "peak_gpu_device_bytes": sampler.peak_gpu,
        "peak_rss_bytes": sampler.peak_rss,
        "resource_samples_path": str(resource_path.resolve()),
        "stdout_sha256": _sha256_file(stdout_path),
        "resource_sha256": _sha256_file(resource_path),
    }


def _wall_command(
    command: Sequence[str], *, cwd: Path, env: dict[str, str], stdout_path: Path
) -> dict[str, Any]:
    """Launch-to-exit wall timing without resource polling (cold/render diagnostics only)."""
    if stdout_path.exists():
        raise RuntimeError(f"append-only command stdout already exists: {stdout_path}")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.monotonic_ns()
    with stdout_path.open("xb") as stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        returncode = process.wait()
        ended_ns = time.monotonic_ns()
    return {
        "returncode": returncode,
        "wall_seconds": (ended_ns - started_ns) / 1e9,
        "process_started_monotonic_ns": started_ns,
        "process_ended_monotonic_ns": ended_ns,
        "stdout_sha256": _sha256_file(stdout_path),
    }


def _rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _strict_rgb8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB":
            raise RuntimeError(f"native replay output is not an RGB PNG: {path}")
        array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"native replay output has invalid RGB shape: {array.shape}")
    return array


def _central_metrics(
    target_path: Path,
    reconstruction_path: Path,
    *,
    require_lpips: bool = True,
    lpips_function: Callable[[Any, Any], float | None] | None = None,
) -> dict[str, float | None]:
    import torch
    from structsplat import metrics as metrics

    target_u8 = _rgb8(target_path)
    reconstruction_u8 = _rgb8(reconstruction_path)
    if target_u8.shape != reconstruction_u8.shape:
        raise RuntimeError(
            f"central metric shape mismatch: {target_u8.shape} versus {reconstruction_u8.shape}"
        )
    # Central scoring is deliberately CPU-only.  The long-lived orchestration process must never
    # retain a CUDA context or LPIPS allocation that competes with a later fresh-process fit.
    device = "cpu"
    target = torch.as_tensor(target_u8 / 255.0, dtype=torch.float32, device=device)
    reconstruction = torch.as_tensor(
        reconstruction_u8 / 255.0, dtype=torch.float32, device=device
    )
    lpips_function = lpips_function or metrics.LPIPS.distance
    lpips = lpips_function(reconstruction, target)
    result = {
        "central_psnr": metrics.psnr(reconstruction, target),
        "central_ssim": float(metrics.ssim(reconstruction, target)),
        "central_proxy_ms_ssim": metrics.ms_ssim(reconstruction, target),
        "central_lpips_alex": None if lpips is None else float(lpips),
    }
    for name, value in result.items():
        if value is None:
            if require_lpips and name == "central_lpips_alex":
                raise RuntimeError("AlexNet LPIPS is required by BENCH-016 but unavailable")
            continue
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite central metric {name}")
    return result


def _reconstruction_variation_diagnostics(
    reference_u8: np.ndarray,
    sample_u8: np.ndarray,
    reference_metrics: dict[str, float | None],
    sample_metrics: dict[str, float | None],
    *,
    metric_delta_name: str,
) -> dict[str, Any]:
    """Describe replay variation completely without imposing a numeric validity tolerance."""
    if reference_u8.shape != sample_u8.shape:
        raise RuntimeError("native replay reconstruction shape drift")
    difference = np.abs(reference_u8.astype(np.int16) - sample_u8.astype(np.int16))
    differing_pixels = np.any(difference != 0, axis=2)
    mse = float(np.mean(np.square(difference.astype(np.float64) / 255.0)))
    metric_delta: dict[str, float | None] = {}
    for name in CENTRAL_METRIC_NAMES:
        reference_value = reference_metrics.get(name)
        sample_value = sample_metrics.get(name)
        metric_delta[name] = (
            None
            if reference_value is None or sample_value is None
            else float(sample_value) - float(reference_value)
        )
    differing_pixel_count = int(np.count_nonzero(differing_pixels))
    differing_code_count = int(np.count_nonzero(difference))
    return {
        "scope": "diagnostic_only_never_a_validity_or_scientific_decision_gate",
        "shape_equal": True,
        "exact_pixel_match": differing_code_count == 0,
        "max_uint8_difference": int(difference.max()),
        "per_channel_max_uint8_difference": {
            channel: int(difference[..., index].max())
            for index, channel in enumerate(("r", "g", "b"))
        },
        "differing_pixel_count": differing_pixel_count,
        "differing_pixel_fraction": float(np.mean(differing_pixels)),
        "differing_channel_code_count": differing_code_count,
        "differing_channel_code_fraction": float(np.mean(difference != 0)),
        "difference_psnr_db": float(10.0 * math.log10(1.0 / max(mse, 1e-12))),
        "difference_psnr_mse_floor": 1e-12,
        metric_delta_name: metric_delta,
    }


def _native_training_replay_diagnostics(
    training_u8: np.ndarray,
    primary_u8: np.ndarray,
    training_metrics: dict[str, float | None],
    primary_metrics: dict[str, float | None],
) -> dict[str, Any]:
    """Describe training-PNG drift without turning it into a validity or decision gate."""
    return _reconstruction_variation_diagnostics(
        training_u8,
        primary_u8,
        training_metrics,
        primary_metrics,
        metric_delta_name="primary_minus_training_metric_delta",
    )


def _prepared_replay_aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all role-fixed timing replays; every field remains threshold-free."""
    diagnostics = [record["primary_replay_variation_diagnostics"] for record in records]
    if len(diagnostics) != RENDER_WARMUPS + RENDER_SAMPLES:
        raise RuntimeError("native prepared replay diagnostic count drift")

    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "min": float(min(values)),
            "median": float(statistics.median(values)),
            "max": float(max(values)),
        }

    metric_delta_summary = {}
    for metric in CENTRAL_METRIC_NAMES:
        values = [
            float(item["sample_minus_primary_metric_delta"][metric]) for item in diagnostics
        ]
        metric_delta_summary[metric] = summary(values)
    pixel_hashes = [str(record["reconstruction_pixel_sha256"]) for record in records]
    return {
        "scope": "threshold_free_diagnostic_never_a_validity_or_scientific_decision_gate",
        "replay_count": len(records),
        "exact_primary_pixel_match_count": sum(
            bool(item["exact_pixel_match"]) for item in diagnostics
        ),
        "unique_pixel_hash_count": len(set(pixel_hashes)),
        "max_uint8_difference": max(int(item["max_uint8_difference"]) for item in diagnostics),
        "max_differing_pixel_fraction": max(
            float(item["differing_pixel_fraction"]) for item in diagnostics
        ),
        "max_differing_channel_code_fraction": max(
            float(item["differing_channel_code_fraction"]) for item in diagnostics
        ),
        "difference_psnr_db": summary(
            [float(item["difference_psnr_db"]) for item in diagnostics]
        ),
        "sample_minus_primary_metric_delta": metric_delta_summary,
    }


def _count_sites(path: Path) -> tuple[int, int | None]:
    count = 0
    declared = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("# Total sites:"):
                declared = int(line.split(":", 1)[1].strip())
            elif line.strip() and not line.startswith("#"):
                values = line.split()
                if len(values) != SAD_SCALARS_PER_SITE:
                    raise RuntimeError(f"malformed SAD site row with {len(values)} scalars")
                if not all(math.isfinite(float(value)) for value in values):
                    raise RuntimeError("non-finite SAD site scalar")
                count += 1
    if count <= 0 or (declared is not None and declared != count):
        raise RuntimeError(f"SAD site count mismatch: parsed={count}, declared={declared}")
    return count, declared


def _jfa_seed_collision_diagnostics(
    path: Path, width: int, height: int, cand_downscale: int
) -> dict[str, Any]:
    """Count non-atomic JFA home-pixel collisions using upstream CUDA cast/clamp semantics."""
    if width <= 0 or height <= 0 or cand_downscale <= 0:
        raise ValueError("JFA collision diagnostics require positive dimensions/downscale")
    cand_width = max(1, (width + cand_downscale - 1) // cand_downscale)
    cand_height = max(1, (height + cand_downscale - 1) // cand_downscale)
    occupancy: dict[tuple[int, int], int] = {}
    site_count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.split()
            if len(values) != SAD_SCALARS_PER_SITE:
                raise RuntimeError("malformed SAD site row during JFA collision audit")
            position_x, position_y = float(values[0]), float(values[1])
            if not math.isfinite(position_x) or not math.isfinite(position_y):
                raise RuntimeError("non-finite SAD position during JFA collision audit")
            x = min(max(int(position_x / cand_downscale), 0), cand_width - 1)
            y = min(max(int(position_y / cand_downscale), 0), cand_height - 1)
            occupancy[(x, y)] = occupancy.get((x, y), 0) + 1
            site_count += 1
    multiplicities = [count for count in occupancy.values() if count > 1]
    return {
        "scope": "diagnostic_only_never_a_validity_or_scientific_decision_gate",
        "upstream_mechanism": "non_atomic_jfa_home_pixel_seed_overwrite",
        "candidate_downscale": cand_downscale,
        "candidate_width": cand_width,
        "candidate_height": cand_height,
        "site_count": site_count,
        "occupied_home_pixels": len(occupancy),
        "collision_home_pixel_count": len(multiplicities),
        "sites_in_collision_home_pixels": sum(multiplicities),
        "excess_colliding_site_writes": sum(count - 1 for count in multiplicities),
        "max_home_pixel_multiplicity": max(multiplicities, default=1),
    }


def native_progress_trace(trace: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    points = [
        {
            "progress": float(int(row["iteration"]) + 1) / ITERATIONS,
            "psnr": float(row["psnr"]),
        }
        for row in trace
    ]
    if not points or points[0]["progress"] != COMMON_AUC_START or points[-1]["progress"] != 1.0:
        raise RuntimeError("native convergence trace does not cover [1/4000,1]")
    return points


def structsplat_progress_trace(
    history_trace: Sequence[dict[str, Any]], trajectory_terminal_psnr: float
) -> list[dict[str, float]]:
    points = [
        {
            "progress": float(int(row["iteration"])) / ITERATIONS,
            "psnr": float(row["psnr"]),
        }
        for row in history_trace
    ]
    if not points or points[0]["progress"] != 0.0 or points[-1]["progress"] >= 1.0:
        raise RuntimeError("StructSplat history trace does not have pre-update semantics")
    points.append({"progress": 1.0, "psnr": float(trajectory_terminal_psnr)})
    return points


def _trace_on_common_domain(trace: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    progress = np.asarray([row["progress"] for row in trace], dtype=np.float64)
    psnr = np.asarray([row["psnr"] for row in trace], dtype=np.float64)
    if (
        len(progress) < 2
        or np.any(np.diff(progress) <= 0.0)
        or progress[0] > COMMON_AUC_START
        or progress[-1] < 1.0
        or not np.all(np.isfinite(psnr))
    ):
        raise ValueError("trace cannot cover the frozen common AUC interval")
    common_progress = [COMMON_AUC_START]
    common_progress.extend(
        float(value) for value in progress if COMMON_AUC_START < value < 1.0
    )
    common_progress.append(1.0)
    values = np.interp(common_progress, progress, psnr)
    return [
        {"progress": point, "psnr": float(value)}
        for point, value in zip(common_progress, values)
    ]


def normalized_psnr_auc(trace: Sequence[dict[str, float]]) -> float:
    common = _trace_on_common_domain(trace)
    progress = np.asarray([row["progress"] for row in common], dtype=np.float64)
    psnr = np.asarray([row["psnr"] for row in common], dtype=np.float64)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(psnr, progress) / (1.0 - COMMON_AUC_START))


def _sad_command(binary: Path, target: Path, rate: float, cell_dir: Path) -> list[str]:
    return [
        str(binary.resolve()),
        str(target.resolve()),
        "--iters",
        str(ITERATIONS),
        "--target-bpp",
        f"{rate:.1f}",
        "--log-freq",
        str(LOG_FREQUENCY),
        "--init-gradient",
        "--out-dir",
        str(cell_dir.resolve()),
    ]


def _cold_sad_command(
    binary: Path, sites: Path, reconstruction: Path, width: int, height: int
) -> list[str]:
    return [
        str(binary.resolve()),
        "--render",
        str(sites.resolve()),
        "--width",
        str(width),
        "--height",
        str(height),
        "--out",
        str(reconstruction.resolve()),
    ]


def _record_command(
    outdir: Path, stage: str, key: dict[str, Any], command: Sequence[str], cwd: Path
) -> None:
    _append_jsonl(
        outdir / "commands.jsonl",
        {
            "protocol": PROTOCOL,
            "stage": stage,
            "key": key,
            "cwd": str(cwd.resolve()),
            "argv": list(command),
            "environment_overrides": {
                "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
            },
        },
    )


def _existing_success_keys(
    path: Path,
    fields: Sequence[str],
    *,
    binding_file_sha256: str,
    targets_file_sha256: str,
) -> set[tuple[Any, ...]]:
    """Return only rows belonging to the exact live protocol/binding/target identity."""
    return {
        tuple(row.get(field) for field in fields)
        for row in _load_jsonl(path)
        if (
            row.get("status") == "ok"
            and row.get("protocol") == PROTOCOL
            and row.get("binding_file_sha256") == binding_file_sha256
            and row.get("targets_file_sha256") == targets_file_sha256
        )
    }


def run_sad(
    outdir: Path,
    *,
    image_ids: Sequence[str] | None = None,
    rates: Sequence[float] | None = None,
    repeats: Sequence[int] | None = None,
    max_new_rows: int | None = None,
) -> list[dict[str, Any]]:
    binding = _validate_binding(outdir)
    targets = _validate_targets(outdir)
    target_map = _target_by_id(targets)
    sad_repo = Path(binding["protocol_spec"]["sad"]["repo"])
    sad_paths = _sad_paths(sad_repo)
    env = {**os.environ, "CONFIG_PATH": str(sad_paths["config"].resolve())}
    selected_ids = tuple(image_ids or TARGET_IDS)
    selected_rates = tuple(float(rate) for rate in (rates or RATES))
    selected_repeats = tuple(int(repeat) for repeat in (repeats or NATIVE_REPEATS))
    if not set(selected_ids).issubset(TARGET_IDS):
        raise ValueError("run-sad image selection is outside the frozen target IDs")
    if not set(selected_rates).issubset(RATES):
        raise ValueError("run-sad rate selection is outside the frozen rates")
    if not set(selected_repeats).issubset(NATIVE_REPEATS):
        raise ValueError("run-sad repeat selection is outside {0,1,2}")
    ledger = outdir / "sad_rows.jsonl"
    binding_file_sha256 = _sha256_file(outdir / "binding.json")
    targets_file_sha256 = _sha256_file(outdir / "targets.json")
    done = _existing_success_keys(
        ledger,
        ("image_id", "target_bpp", "repeat"),
        binding_file_sha256=binding_file_sha256,
        targets_file_sha256=targets_file_sha256,
    )
    new_rows: list[dict[str, Any]] = []
    for image_id in selected_ids:
        target_record = target_map[image_id]
        target = Path(target_record["derived_path"])
        for rate in selected_rates:
            for repeat in selected_repeats:
                key_tuple = (image_id, rate, repeat)
                if key_tuple in done:
                    continue
                key = {"image_id": image_id, "target_bpp": rate, "repeat": repeat}
                cell_dir = outdir / "sad" / image_id / f"bpp_{rate:.1f}" / f"repeat_{repeat}"
                if cell_dir.exists():
                    raise RuntimeError(
                        f"incomplete append-only SAD cell already exists; preserve it and use a "
                        f"fresh output directory or a new protocol: {cell_dir}"
                    )
                cell_dir.mkdir(parents=True)
                command = _sad_command(sad_paths["binary"], target, rate, cell_dir)
                _record_command(outdir, "sad_train", key, command, sad_repo)
                command_record = {
                    "protocol": PROTOCOL,
                    "binding_sha256": _sha256_file(outdir / "binding.json"),
                    "target_manifest_sha256": _sha256_file(outdir / "targets.json"),
                    "key": key,
                    "cwd": str(sad_repo),
                    "argv": command,
                    "environment_overrides": {"CONFIG_PATH": env["CONFIG_PATH"]},
                }
                _write_json_once(cell_dir / "command.json", command_record)
                timing = _timed_command(
                    command,
                    cwd=sad_repo,
                    env=env,
                    stdout_path=cell_dir / "train_stdout.txt",
                    resource_path=cell_dir / "train_resources.json",
                )
                if timing["peak_gpu_device_bytes"] is None or timing["peak_rss_bytes"] is None:
                    raise RuntimeError(f"mandatory fresh-process memory samples missing for {key}")
                row_base = {
                    "protocol": PROTOCOL,
                    "binding_file_sha256": binding_file_sha256,
                    "targets_file_sha256": targets_file_sha256,
                    "arm": "sad_native",
                    **key,
                    "target_path": str(target.resolve()),
                    "target_byte_sha256": target_record["derived_byte_sha256"],
                    "target_pixel_sha256": target_record["derived_pixel_sha256"],
                    "height": target_record["derived_height"],
                    "width": target_record["derived_width"],
                    "pixel_count": target_record["pixel_count"],
                    "command_path": str((cell_dir / "command.json").resolve()),
                    "command_sha256": _sha256_file(cell_dir / "command.json"),
                    "raw_stdout_path": str((cell_dir / "train_stdout.txt").resolve()),
                    **timing,
                }
                if timing["returncode"] != 0:
                    row = {**row_base, "status": "error", "error": "native SAD process failed"}
                    _append_jsonl(ledger, row)
                    raise RuntimeError(f"native SAD process failed for {key}")
                stdout = (cell_dir / "train_stdout.txt").read_text(
                    encoding="utf-8", errors="replace"
                )
                if "Training | backend=cuda" not in stdout:
                    raise RuntimeError(f"native SAD backend fallback for {key}")
                trace = parse_sad_trace(stdout)
                train_seconds = _parse_named_time(stdout, _TRAINING_TIME_RE, "training time")
                total_seconds = _parse_named_time(stdout, _TOTAL_TIME_RE, "total time")
                training_reconstruction = cell_dir / f"{target.stem}.png"
                sites = cell_dir / f"{target.stem}_sites.txt"
                if not training_reconstruction.is_file() or not sites.is_file():
                    raise RuntimeError(f"native SAD omitted reconstruction/sites for {key}")
                final_sites, declared_sites = _count_sites(sites)
                diagnostic_training_metrics = _central_metrics(target, training_reconstruction)
                normalized_trace = native_progress_trace(trace)
                auc = normalized_psnr_auc(normalized_trace)
                target_sites = int(
                    math.floor(rate * target_record["pixel_count"] / 128.0 + 0.5)
                )
                target_tolerance = max(1, int(math.ceil(0.01 * target_sites)))
                target_error = abs(final_sites - target_sites)
                if target_error > target_tolerance:
                    raise RuntimeError(
                        f"native SAD target-attainment failure for {key}: N={final_sites}, "
                        f"target={target_sites}, tolerance={target_tolerance}"
                    )

                jfa_collision_diagnostics = _jfa_seed_collision_diagnostics(
                    sites,
                    int(target_record["derived_width"]),
                    int(target_record["derived_height"]),
                    int(binding["upstream"]["resolved_config"]["CAND_DOWNSCALE"]),
                )
                if int(jfa_collision_diagnostics["site_count"]) != final_sites:
                    raise RuntimeError(f"native SAD JFA collision site-count drift for {key}")

                # Protocol v6 fixes this exact invocation role before observing its pixels.  It is
                # launched once and is never retried, replaced, or selected from the timing ensemble.
                primary_dir = cell_dir / "primary_replay"
                primary_dir.mkdir()
                primary_reconstruction = primary_dir / "reconstruction.png"
                primary_command = _cold_sad_command(
                    sad_paths["binary"],
                    sites,
                    primary_reconstruction,
                    target_record["derived_width"],
                    target_record["derived_height"],
                )
                _record_command(
                    outdir, "sad_role_fixed_primary_saved_txt_replay", key, primary_command, sad_repo
                )
                primary_timing = _wall_command(
                    primary_command,
                    cwd=sad_repo,
                    env=env,
                    stdout_path=primary_dir / "stdout.txt",
                )
                if primary_timing["returncode"] != 0 or not primary_reconstruction.is_file():
                    raise RuntimeError(f"role-fixed primary saved-TXT replay failed for {key}")
                primary_stdout = (primary_dir / "stdout.txt").read_text(
                    encoding="utf-8", errors="replace"
                )
                primary_semantics = _parse_render_stdout_semantics(
                    primary_stdout,
                    expected_site_count=final_sites,
                    expected_width=int(target_record["derived_width"]),
                    expected_height=int(target_record["derived_height"]),
                    expected_saved_path=primary_reconstruction,
                )
                primary_candidate_ms = float(primary_semantics["candidate_build_ms"])
                primary_render_ms = float(primary_semantics["render_ms"])
                primary_metrics = _central_metrics(target, primary_reconstruction)
                training_u8 = _strict_rgb8(training_reconstruction)
                primary_u8 = _strict_rgb8(primary_reconstruction)
                expected_shape = (
                    int(target_record["derived_height"]),
                    int(target_record["derived_width"]),
                    3,
                )
                if training_u8.shape != expected_shape or primary_u8.shape != expected_shape:
                    raise RuntimeError(f"native SAD reconstruction shape/orientation drift for {key}")
                training_replay_diagnostics = _native_training_replay_diagnostics(
                    training_u8,
                    primary_u8,
                    diagnostic_training_metrics,
                    primary_metrics,
                )

                timing_dir = cell_dir / "prepared_timing"
                timing_dir.mkdir()
                render_samples: list[float] = []
                candidate_samples: list[float] = []
                prepared_render_records: list[dict[str, Any]] = []
                primary_pixel_sha256 = _pixel_sha256(primary_u8)
                for sample_index in range(RENDER_WARMUPS + RENDER_SAMPLES):
                    sample_reconstruction = timing_dir / f"sample_{sample_index:02d}.png"
                    sample_stdout_path = timing_dir / f"sample_{sample_index:02d}_stdout.txt"
                    sample_command = _cold_sad_command(
                        sad_paths["binary"],
                        sites,
                        sample_reconstruction,
                        target_record["derived_width"],
                        target_record["derived_height"],
                    )
                    sample_key = {**key, "sample_index": sample_index}
                    _record_command(
                        outdir, "sad_prepared_render_timing", sample_key, sample_command, sad_repo
                    )
                    sample_timing = _wall_command(
                        sample_command,
                        cwd=sad_repo,
                        env=env,
                        stdout_path=sample_stdout_path,
                    )
                    if sample_timing["returncode"] != 0 or not sample_reconstruction.is_file():
                        raise RuntimeError(f"native SAD render timing sample failed for {sample_key}")
                    sample_stdout = sample_stdout_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    sample_semantics = _parse_render_stdout_semantics(
                        sample_stdout,
                        expected_site_count=final_sites,
                        expected_width=int(target_record["derived_width"]),
                        expected_height=int(target_record["derived_height"]),
                        expected_saved_path=sample_reconstruction,
                    )
                    candidate_ms = float(sample_semantics["candidate_build_ms"])
                    render_ms = float(sample_semantics["render_ms"])
                    sample_u8 = _strict_rgb8(sample_reconstruction)
                    sample_pixel_sha256 = _pixel_sha256(sample_u8)
                    if sample_u8.shape != primary_u8.shape:
                        raise RuntimeError(
                            f"native SAD prepared replay shape drift for {sample_key}"
                        )
                    sample_metrics = _central_metrics(target, sample_reconstruction)
                    variation = _reconstruction_variation_diagnostics(
                        primary_u8,
                        sample_u8,
                        primary_metrics,
                        sample_metrics,
                        metric_delta_name="sample_minus_primary_metric_delta",
                    )
                    process_wall_ms = (
                        int(sample_timing["process_ended_monotonic_ns"])
                        - int(sample_timing["process_started_monotonic_ns"])
                    ) / 1e6
                    prepared_render_records.append(
                        {
                            "sample_index": sample_index,
                            "invocation_ordinal_after_primary": sample_index + 1,
                            "sample_role": (
                                "discarded_warmup"
                                if sample_index < RENDER_WARMUPS
                                else "retained_timing_sample"
                            ),
                            "reconstruction_path": str(sample_reconstruction.resolve()),
                            "reconstruction_sha256": _sha256_file(sample_reconstruction),
                            "reconstruction_pixel_sha256": sample_pixel_sha256,
                            "primary_replay_pixel_sha256": primary_pixel_sha256,
                            "command_argv": sample_command,
                            "command_sha256": _command_sha256(sample_command),
                            "cwd": str(sad_repo.resolve()),
                            "config_path": env["CONFIG_PATH"],
                            "sites_sha256": _sha256_file(sites),
                            "stdout_path": str(sample_stdout_path.resolve()),
                            "stdout_sha256": sample_timing["stdout_sha256"],
                            "stdout_semantics": sample_semantics,
                            "returncode": int(sample_timing["returncode"]),
                            "process_started_monotonic_ns": sample_timing[
                                "process_started_monotonic_ns"
                            ],
                            "process_ended_monotonic_ns": sample_timing[
                                "process_ended_monotonic_ns"
                            ],
                            "full_process_wall_ms": process_wall_ms,
                            "candidate_build_ms": candidate_ms,
                            "render_ms": render_ms,
                            "central_metrics": sample_metrics,
                            "primary_replay_variation_diagnostics": variation,
                        }
                    )
                    if sample_index >= RENDER_WARMUPS:
                        candidate_samples.append(candidate_ms)
                        render_samples.append(render_ms)
                if (
                    len(render_samples) != RENDER_SAMPLES
                    or len(prepared_render_records) != RENDER_WARMUPS + RENDER_SAMPLES
                ):
                    raise RuntimeError("native SAD prepared timing sample count drift")
                prepared_replay_aggregate = _prepared_replay_aggregate(prepared_render_records)

                compressed = gzip.compress(sites.read_bytes(), compresslevel=9, mtime=0)
                compressed_path = cell_dir / f"{target.stem}_sites.txt.gz"
                _write_once(compressed_path, compressed)
                nominal_state_bytes = final_sites * SAD_SCALARS_PER_SITE * FLOAT32_BYTES
                nominal_bpp = final_sites * SAD_BYTES_PER_SITE * 8.0 / target_record["pixel_count"]
                row = {
                    **row_base,
                    "status": "ok",
                    "error": "",
                    "iterations": ITERATIONS,
                    "log_frequency": LOG_FREQUENCY,
                    "convergence_trace": trace,
                    "internal_training_state_normalized_iteration_trace": normalized_trace,
                    "internal_training_state_normalized_iteration_psnr_auc_db": auc,
                    "auc_scope": "internal_training_state_normalized_iteration",
                    "upstream_terminal_psnr_db": trace[-1]["psnr"],
                    "upstream_terminal_active_sites": trace[-1]["active_sites"],
                    "native_training_seconds": train_seconds,
                    "native_total_seconds": total_seconds,
                    "training_wall_seconds": timing["wall_seconds"],
                    "process_wall_seconds": timing["wall_seconds"],
                    "primitive_count": final_sites,
                    "trainable_scalar_count": final_sites * SAD_SCALARS_PER_SITE,
                    "sad_final_active_sites": final_sites,
                    "sad_declared_active_sites": declared_sites,
                    "sad_initial_active_sites": int(
                        binding["upstream"]["resolved_config"]["DEFAULT_SITES"]
                    ),
                    "sad_logged_active_site_trajectory": [
                        int(point["active_sites"]) for point in trace
                    ],
                    "sad_logged_capacity_site_trajectory": [
                        int(point["capacity_sites"]) for point in trace
                    ],
                    "sad_allocated_capacity_sites_exposed": max(
                        int(point["capacity_sites"]) for point in trace
                    ),
                    "sad_target_bpp": rate,
                    "sad_target_sites": target_sites,
                    "sad_target_site_tolerance": target_tolerance,
                    "sad_target_site_error": target_error,
                    "sad_target_attainment_pass": True,
                    "sad_nominal_bytes_per_site": SAD_BYTES_PER_SITE,
                    "sad_nominal_state_bytes": final_sites * SAD_BYTES_PER_SITE,
                    "sad_nominal_bpp": nominal_bpp,
                    "sad_raw_float32_state_bytes": nominal_state_bytes,
                    "sad_exported_txt_bytes": sites.stat().st_size,
                    "sad_exported_txt_gzip_bytes": len(compressed),
                    "sad_representation_scope": (
                        "recipient_replayable_saved_txt_plus_pinned_official_decoder"
                    ),
                    "sad_recipient_replayable": True,
                    "sad_deployable": False,
                    "sad_self_contained": False,
                    "sad_compressed_representation": False,
                    "sad_complete_stream_available": False,
                    "compression_tested": False,
                    "scientific_artifact_origin": "fresh_v6_execution",
                    "sites_path": str(sites.resolve()),
                    "sites_sha256": _sha256_file(sites),
                    "sites_gzip_path": str(compressed_path.resolve()),
                    "sites_gzip_sha256": _sha256_file(compressed_path),
                    "training_reconstruction_path": str(training_reconstruction.resolve()),
                    "training_reconstruction_sha256": _sha256_file(training_reconstruction),
                    "training_reconstruction_pixel_sha256": _pixel_sha256(training_u8),
                    "jfa_seed_collision_diagnostics": jfa_collision_diagnostics,
                    "primary_replay_invocation_ordinal": 0,
                    "primary_replay_retry_count": 0,
                    "primary_replay_selection_policy": (
                        "role_fixed_first_fresh_official_saved_txt_replay_never_retried_or_selected"
                    ),
                    "primary_replay_command_argv": primary_command,
                    "primary_replay_command_sha256": _command_sha256(primary_command),
                    "primary_replay_cwd": str(sad_repo.resolve()),
                    "primary_replay_config_path": env["CONFIG_PATH"],
                    "primary_replay_returncode": int(primary_timing["returncode"]),
                    "primary_replay_stdout_semantics": primary_semantics,
                    "primary_replay_reconstruction_path": str(
                        primary_reconstruction.resolve()
                    ),
                    "primary_replay_reconstruction_sha256": _sha256_file(
                        primary_reconstruction
                    ),
                    "primary_replay_reconstruction_pixel_sha256": primary_pixel_sha256,
                    "primary_replay_candidate_build_ms": primary_candidate_ms,
                    "primary_replay_render_ms": primary_render_ms,
                    "primary_replay_full_process_wall_ms": (
                        primary_timing["wall_seconds"] * 1_000.0
                    ),
                    "primary_replay_process_started_monotonic_ns": primary_timing[
                        "process_started_monotonic_ns"
                    ],
                    "primary_replay_process_ended_monotonic_ns": primary_timing[
                        "process_ended_monotonic_ns"
                    ],
                    "primary_replay_stdout_path": str(
                        (primary_dir / "stdout.txt").resolve()
                    ),
                    "primary_replay_stdout_sha256": primary_timing["stdout_sha256"],
                    "primary_replay_non_kernel_residual_upper_bound_ms": max(
                        0.0,
                        primary_timing["wall_seconds"] * 1_000.0
                        - primary_candidate_ms
                        - primary_render_ms,
                    ),
                    "primary_replay_non_kernel_residual_includes": [
                        "process_startup",
                        "configuration_loading",
                        "cuda_context_work",
                        "txt_parse_and_host_to_device_upload",
                        "png_output",
                    ],
                    "native_quality_reconstruction_scope": (
                        "role_fixed_first_fresh_official_saved_txt_replay_only"
                    ),
                    "diagnostic_training_metrics": diagnostic_training_metrics,
                    "training_vs_primary_replay_diagnostics": training_replay_diagnostics,
                    "render_warmups": RENDER_WARMUPS,
                    "render_sample_count": RENDER_SAMPLES,
                    "prepared_candidate_build_ms_samples": candidate_samples,
                    "prepared_render_ms_samples": render_samples,
                    "prepared_render_records": prepared_render_records,
                    "prepared_replay_variation_aggregate": prepared_replay_aggregate,
                    "prepared_replay_pixel_variation_is_threshold_free_diagnostic": True,
                    "prepared_candidate_build_ms_median": statistics.median(candidate_samples),
                    "prepared_render_ms_median": statistics.median(render_samples),
                    **primary_metrics,
                }
                _append_jsonl(ledger, row)
                new_rows.append(row)
                done.add(key_tuple)
                if max_new_rows is not None and len(new_rows) >= max_new_rows:
                    return new_rows
    return new_rows


def conservative_control_budgets(native_rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    if len(native_rows) != len(NATIVE_REPEATS):
        raise ValueError("control scheduling requires exactly three native repeats")
    repeats = sorted(int(row["repeat"]) for row in native_rows)
    if repeats != list(NATIVE_REPEATS) or any(row.get("status") != "ok" for row in native_rows):
        raise ValueError("control scheduling requires valid native repeats {0,1,2}")
    counts = [int(row["sad_final_active_sites"]) for row in native_rows]
    if any(count <= 0 for count in counts):
        raise ValueError("native site counts must be positive")
    same_n = max(counts)
    return {
        "native_repeat_min_n": min(counts),
        "native_repeat_max_n": same_n,
        "ss_same_n": same_n,
        "ss_equal_terminal_coordinates": int(
            math.ceil(SAD_SCALARS_PER_SITE * same_n / SS_SCALARS_PER_GAUSSIAN)
        ),
    }


def _control_worker_command(
    contract_path: Path,
    cell_dir: Path,
) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "benchmarks.native_sad_frontier",
        "_control-worker",
        "--contract",
        str(contract_path.resolve()),
        "--cell-dir",
        str(cell_dir.resolve()),
    ]


def _prepared_structsplat_timing(field: Any, fit_config: Any, height: int, width: int) -> list[float]:
    import torch
    from structsplat.render import render_field

    def render() -> Any:
        return render_field(
            field.means,
            field.conics(fit_config.aa_dilation),
            field.colors,
            field.radii(fit_config.sigma_cutoff, fit_config.aa_dilation),
            height,
            width,
            fit_config.render_chunk,
            fit_config.renderer,
            field.opacity_values(),
            scales=field.scales(),
            rotations=field.rotations,
            support_fade=fit_config.support_fade,
            sigma_cutoff=fit_config.sigma_cutoff,
        )

    with torch.no_grad():
        for _ in range(RENDER_WARMUPS):
            render()
        torch.cuda.synchronize()
        samples = []
        for _ in range(RENDER_SAMPLES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            render()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return samples


def _run_control_worker(
    contract_path: Path,
    cell_dir: Path,
) -> dict[str, Any]:
    """Fresh fit process ending immediately after synchronized selected-field export."""
    import torch
    from benchmarks import fair_density_control_compare as fair
    from structsplat.config import FitConfig, StructureTensorConfig
    from structsplat.fit import fit

    contract = _load_json(contract_path)
    if contract.get("protocol") != PROTOCOL:
        raise RuntimeError("invalid StructSplat single-cell contract")
    image_id = str(contract["image_id"])
    rate = float(contract["target_bpp"])
    arm = str(contract["arm"])
    repeat = int(contract["repeat"])
    final_budget = int(contract["final_budget"])
    if (
        arm not in SS_ARMS
        or repeat not in NATIVE_REPEATS
        or contract.get("method") != SS_METHOD
        or int(contract.get("seed", -1)) != SS_SEED
        or len(str(contract.get("binding_file_sha256", ""))) != 64
        or len(str(contract.get("targets_file_sha256", ""))) != 64
    ):
        raise ValueError(f"unknown StructSplat control arm {arm}")
    target_record = contract["target_record"]
    target_path = Path(target_record["derived_path"])
    target_u8 = _rgb8(target_path)
    if _pixel_sha256(target_u8) != target_record["derived_pixel_sha256"]:
        raise RuntimeError("StructSplat single-cell target pixel drift")
    image = target_u8.astype(np.float32) / 255.0
    target = torch.as_tensor(image, dtype=torch.float32, device="cuda")
    start_budget = fair._start_budget(final_budget, SS_START_FRACTION)  # noqa: SLF001
    difference = final_budget - start_budget
    if difference < 21:
        raise RuntimeError(f"frozen growth precondition D>=21 failed: D={difference}")
    base_fit = FitConfig(
        iters=ITERATIONS,
        renderer="cuda",
        pixel_loss="l1",
        ssim_weight=0.3,
        log_every=LOG_FREQUENCY,
        compute_lpips=False,
        color_basis="constant",
        qat_mode="off",
        adaptive_count=False,
        support_fade=False,
    )
    field, fit_config, init_seconds, actual_start, method_meta = fair._build_method(  # noqa: SLF001
        SS_METHOD,
        image,
        target_path,
        final_budget,
        start_budget,
        SS_SEED,
        base_fit,
        StructureTensorConfig(),
        SS_GROWTH_WAVES,
        "cuda",
        feature_cap=SS_FEATURE_CAP,
        feature_cap_reference_side=SS_FEATURE_CAP_REFERENCE_SIDE,
    )
    required = {
        "iters": ITERATIONS,
        "renderer": "cuda",
        "pixel_loss": "l1",
        "ssim_weight": 0.3,
        "compute_lpips": False,
        "target_psnr": None,
        "target_psnrs": [],
        "early_stop_patience": None,
        "checkpoint_policy": "best_psnr_final_count",
        "color_basis": "constant",
        "qat_mode": "off",
        "adaptive_count": False,
        "log_every": LOG_FREQUENCY,
        "split_mode": "residual_tensor_add",
        "refine_site": "residual_tensor",
        "refine_primitive": "sampled_add",
        "split_schedule_stops_at_max": False,
    }
    drift = {
        key: (getattr(fit_config, key), value)
        for key, value in required.items()
        if getattr(fit_config, key) != value
    }
    if drift or field.opacities is not None or field.color_grads is not None:
        raise RuntimeError(f"frozen StructSplat control semantics drift: {drift}")
    expected_feature_cap_px = SS_FEATURE_CAP * max(image.shape[:2]) / SS_FEATURE_CAP_REFERENCE_SIDE
    init_config = method_meta.get("init_config") or {}
    if (
        method_meta.get("growth_rule") != "residual_tensor_add"
        or method_meta.get("configured_growth_waves") != SS_GROWTH_WAVES
        or method_meta.get("scale_cap_rule") != "feature"
        or method_meta.get("scale_cap_input") != SS_FEATURE_CAP
        or method_meta.get("scale_cap_reference_side") != SS_FEATURE_CAP_REFERENCE_SIDE
        or not math.isclose(
            float(method_meta.get("scale_cap_max", float("nan"))),
            expected_feature_cap_px,
        )
        or init_config.get("strategy") != "aniso_onedge"
        or init_config.get("sampling_mode") != "wse"
        or init_config.get("opacity_mode") != "none"
        or int(init_config.get("num_gaussians", -1)) != start_budget
        or int(init_config.get("seed", -1)) != SS_SEED
    ):
        raise RuntimeError("frozen StructSplat tensor-growth/feature-cap metadata drift")
    expected_split_count = int(math.ceil(difference / SS_GROWTH_WAVES))
    if (
        fit_config.max_gaussians != final_budget
        or fit_config.split_every != 666
        or fit_config.split_count != expected_split_count
    ):
        raise RuntimeError("frozen five-wave final-budget schedule drift")
    if not cell_dir.is_dir() or contract_path.parent != cell_dir:
        raise RuntimeError("StructSplat cell contract/output directory mismatch")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fit_started = time.perf_counter()
    output = fit(field, target, fit_config, verbose=False)
    torch.cuda.synchronize()
    fit_wall_seconds = time.perf_counter() - fit_started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    selected_field = output["field"]
    if (
        selected_field.n != final_budget
        or int(output.get("selected_n_gaussians", -1)) != final_budget
        or int(output.get("trajectory_terminal_n_gaussians", -1)) != final_budget
    ):
        raise RuntimeError(
            "StructSplat control failed selected/terminal exact final-count invariant"
        )
    history = output.get("history") or {}
    split_events = list(history.get("split_events") or [])
    expected_split_iters = [665, 1331, 1997, 2663, 3329]
    if (
        len(split_events) != 5
        or [int(event.get("iter", -1)) for event in split_events] != expected_split_iters
        or any(int(event.get("added", 0)) <= 0 for event in split_events)
        or sum(int(event["added"]) for event in split_events) != difference
    ):
        raise RuntimeError(f"frozen five-positive-growth-event invariant failed: {split_events}")
    render = output["render"].detach().clamp(0.0, 1.0)
    reconstruction = cell_dir / "reconstruction.png"
    Image.fromarray(
        np.clip(render.cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGB"
    ).save(reconstruction)
    npz_path = cell_dir / "field.npz"
    selected_field.save(str(npz_path))
    torch.cuda.synchronize()
    trace = [
        {"iteration": int(iteration), "psnr": float(psnr)}
        for iteration, psnr in zip(history.get("iter", []), history.get("psnr", []))
    ]
    if not trace or trace[0]["iteration"] != 0:
        raise RuntimeError("StructSplat convergence trace is missing progress zero")
    expected_trace_iterations = [*range(0, ITERATIONS, LOG_FREQUENCY), ITERATIONS - 1]
    if [row["iteration"] for row in trace] != expected_trace_iterations:
        raise RuntimeError("StructSplat convergence trace cadence drift")
    terminal_psnr = float(output["trajectory_terminal_psnr"])
    if (
        int(output.get("iterations_run", -1)) != ITERATIONS
        or int(output.get("trajectory_terminal_iter", -1)) != ITERATIONS
    ):
        raise RuntimeError("StructSplat control failed exact 4,000-update horizon")
    normalized_trace = structsplat_progress_trace(trace, terminal_psnr)
    result = {
        "protocol": PROTOCOL,
        "binding_file_sha256": contract["binding_file_sha256"],
        "targets_file_sha256": contract["targets_file_sha256"],
        "image_id": image_id,
        "target_bpp": rate,
        "arm": arm,
        "repeat": repeat,
        "seed": SS_SEED,
        "method": SS_METHOD,
        "final_budget": final_budget,
        "start_budget": start_budget,
        "actual_start_gaussians": actual_start,
        "growth_difference": difference,
        "split_events": split_events,
        "split_event_iterations_zero_based": expected_split_iters,
        "split_event_additions_sum": sum(int(event["added"]) for event in split_events),
        "sixth_scheduled_opportunity_zero_based": 3995,
        "sixth_scheduled_completed_step": 3996,
        "sixth_scheduled_opportunity_was_cap_noop": True,
        "final_gaussians": int(selected_field.n),
        "selected_n_gaussians": int(output["selected_n_gaussians"]),
        "trajectory_terminal_n_gaussians": int(output["trajectory_terminal_n_gaussians"]),
        "primitive_count": int(selected_field.n),
        "trainable_scalar_count": int(selected_field.n) * SS_SCALARS_PER_GAUSSIAN,
        "structsplat_nominal_float32_state_bytes": int(selected_field.n)
        * SS_SCALARS_PER_GAUSSIAN
        * FLOAT32_BYTES,
        "fit_config": asdict(fit_config),
        "method_metadata": method_meta,
        "init_seconds": float(init_seconds),
        "reported_fit_seconds": float(output["fit_seconds"]),
        "training_wall_seconds": fit_wall_seconds,
        "torch_peak_allocated_bytes": peak_allocated,
        "torch_peak_reserved_bytes": peak_reserved,
        "iterations_run": int(output.get("iterations_run", ITERATIONS)),
        "selected_iter": output.get("selected_iter"),
        "selected_from_checkpoint": bool(output.get("selected_from_checkpoint", False)),
        "selected_checkpoint_psnr": float(output["selected_checkpoint_psnr"]),
        "trajectory_terminal_iter": int(output["trajectory_terminal_iter"]),
        "trajectory_terminal_psnr": terminal_psnr,
        "convergence_trace": trace,
        "internal_training_state_normalized_iteration_trace": normalized_trace,
        "internal_training_state_normalized_iteration_psnr_auc_db": normalized_psnr_auc(
            normalized_trace
        ),
        "auc_scope": "internal_training_state_normalized_iteration",
        "reconstruction_path": str(reconstruction.resolve()),
        "reconstruction_sha256": _sha256_file(reconstruction),
        "reconstruction_pixel_sha256": _pixel_sha256(_rgb8(reconstruction)),
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": _sha256_file(npz_path),
    }
    _write_json_once(cell_dir / "worker_result.json", result)
    return result


def _control_artifact_worker_command(outdir: Path, cell_dir: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "benchmarks.native_sad_frontier",
        "_control-artifact-worker",
        "--outdir",
        str(outdir.resolve()),
        "--cell-dir",
        str(cell_dir.resolve()),
    ]


def _run_control_artifact_worker(outdir: Path, cell_dir: Path) -> dict[str, Any]:
    """Disposable post-fit process for SSPL1 encoding and prepared render timing."""
    from structsplat import codec
    from structsplat.config import FitConfig
    from structsplat.gaussians import GaussianField

    _validate_binding(outdir)
    fit_result = _load_json(cell_dir / "worker_result.json")
    targets = _validate_targets(outdir)
    target_record = _target_by_id(targets)[fit_result["image_id"]]
    field = GaussianField.load(fit_result["npz_path"], device="cuda")
    fit_config = FitConfig(**fit_result["fit_config"])
    codec_config = codec.CodecConfig()
    blob = codec.encode(
        field,
        target_record["derived_height"],
        target_record["derived_width"],
        codec_config,
        fit_config,
    )
    stream_path = cell_dir / "field.sspl"
    _write_once(stream_path, blob)
    components = codec.blob_components(blob)
    if components["total_bytes"] != len(blob):
        raise RuntimeError("SSPL1 component accounting mismatch")
    prepared_samples = _prepared_structsplat_timing(
        field,
        fit_config,
        target_record["derived_height"],
        target_record["derived_width"],
    )
    result = {
        "protocol": PROTOCOL,
        "render_warmups": RENDER_WARMUPS,
        "render_sample_count": RENDER_SAMPLES,
        "prepared_render_ms_samples": prepared_samples,
        "prepared_render_ms_median": statistics.median(prepared_samples),
        "sspl_path": str(stream_path.resolve()),
        "sspl_sha256": _sha256_file(stream_path),
        "sspl_bytes": len(blob),
        "sspl_actual_bpp": len(blob) * 8.0 / target_record["pixel_count"],
        "sspl_components": components,
        "sspl_codec_config": asdict(codec_config),
    }
    _write_json_once(cell_dir / "artifact_result.json", result)
    return result


def _control_cold_worker_command(contract_path: Path) -> list[str]:
    return [
        str(Path(sys.executable).resolve()),
        "-m",
        "benchmarks.native_sad_frontier",
        "_control-cold-worker",
        "--contract",
        str(contract_path.resolve()),
    ]


def _run_control_cold_worker(contract_path: Path) -> dict[str, Any]:
    """Fresh process whose measured region is SSPL1 read, decode, exact render, and sync."""
    import torch
    from structsplat import codec

    contract = _load_json(contract_path)
    if contract.get("protocol") != PROTOCOL:
        raise RuntimeError("invalid StructSplat cold-decode contract")
    stream_path = Path(contract["sspl_path"])
    reconstruction_path = Path(contract["cold_reconstruction_path"])

    # Imports and the tiny contract read precede the metric; no CUDA or codec operation does.
    started = time.perf_counter()
    blob = stream_path.read_bytes()
    if _sha256_bytes(blob) != contract["sspl_sha256"]:
        raise RuntimeError("SSPL1 stream drift before fresh cold decode")
    render = codec.decode_and_render(blob, device="cuda").clamp(0.0, 1.0)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - started

    Image.fromarray(
        np.clip(render.cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8),
        mode="RGB",
    ).save(reconstruction_path)
    result = {
        "protocol": PROTOCOL,
        "sspl_cold_decode_render_seconds": cold_seconds,
        "sspl_cold_metric_boundary": "stream_read_through_synchronized_exact_render",
        "sspl_cold_metric_excludes_process_startup": True,
        "cold_reconstruction_path": str(reconstruction_path.resolve()),
        "cold_reconstruction_sha256": _sha256_file(reconstruction_path),
        "cold_reconstruction_pixel_sha256": _pixel_sha256(_rgb8(reconstruction_path)),
    }
    _write_json_once(contract_path.parent / "cold_worker_result.json", result)
    return result


def run_structsplat(
    outdir: Path,
    *,
    image_ids: Sequence[str] | None = None,
    rates: Sequence[float] | None = None,
    max_new_rows: int | None = None,
) -> list[dict[str, Any]]:
    binding = _validate_binding(outdir)
    targets = _validate_targets(outdir)
    target_map = _target_by_id(targets)
    binding_file_sha256 = _sha256_file(outdir / "binding.json")
    targets_file_sha256 = _sha256_file(outdir / "targets.json")
    sad_rows = [
        row
        for row in _load_jsonl(outdir / "sad_rows.jsonl")
        if (
            row.get("status") == "ok"
            and row.get("protocol") == PROTOCOL
            and row.get("binding_file_sha256") == binding_file_sha256
            and row.get("targets_file_sha256") == targets_file_sha256
        )
    ]
    selected_ids = tuple(image_ids or TARGET_IDS)
    selected_rates = tuple(float(rate) for rate in (rates or RATES))
    if not set(selected_ids).issubset(TARGET_IDS) or not set(selected_rates).issubset(RATES):
        raise ValueError("StructSplat selection is outside the frozen matrix")
    ledger = outdir / "structsplat_rows.jsonl"
    worker_env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "LD_PRELOAD": binding["required_ld_preload"]["path"],
    }
    done = _existing_success_keys(
        ledger,
        ("image_id", "target_bpp", "arm", "repeat"),
        binding_file_sha256=binding_file_sha256,
        targets_file_sha256=targets_file_sha256,
    )
    new_rows: list[dict[str, Any]] = []
    for image_id in selected_ids:
        for rate in selected_rates:
            native = [
                row
                for row in sad_rows
                if row["image_id"] == image_id and float(row["target_bpp"]) == rate
            ]
            budgets = conservative_control_budgets(native)
            for arm in SS_ARMS:
                final_budget = budgets[arm]
                for repeat in NATIVE_REPEATS:
                    key_tuple = (image_id, rate, arm, repeat)
                    if key_tuple in done:
                        continue
                    key = {
                        "image_id": image_id,
                        "target_bpp": rate,
                        "arm": arm,
                        "repeat": repeat,
                    }
                    cell_dir = (
                        outdir
                        / "structsplat"
                        / image_id
                        / f"bpp_{rate:.1f}"
                        / arm
                        / f"repeat_{repeat}"
                    )
                    if cell_dir.exists():
                        raise RuntimeError(
                            f"incomplete append-only StructSplat cell exists: {cell_dir}"
                        )
                    cell_dir.mkdir(parents=True)
                    target_record = target_map[image_id]
                    contract_path = cell_dir / "cell_contract.json"
                    contract = {
                        "protocol": PROTOCOL,
                        "binding_file_sha256": binding_file_sha256,
                        "targets_file_sha256": targets_file_sha256,
                        "image_id": image_id,
                        "target_bpp": rate,
                        "arm": arm,
                        "repeat": repeat,
                        "final_budget": final_budget,
                        "target_record": target_record,
                        "method": SS_METHOD,
                        "seed": SS_SEED,
                        "required_ld_preload": binding["required_ld_preload"],
                    }
                    _write_json_once(contract_path, contract)
                    command = _control_worker_command(contract_path, cell_dir)
                    _record_command(outdir, "structsplat_fit", key, command, ROOT)
                    log_dir = (
                        outdir
                        / "structsplat_worker_logs"
                        / image_id
                        / f"bpp_{rate:.1f}"
                        / arm
                    )
                    worker_stdout = log_dir / f"repeat_{repeat}_fit_stdout.txt"
                    worker_resource = log_dir / f"repeat_{repeat}_fit_resources.json"
                    timing = _timed_command(
                        command,
                        cwd=ROOT,
                        env=worker_env,
                        stdout_path=worker_stdout,
                        resource_path=worker_resource,
                    )
                    if timing["returncode"] != 0:
                        row = {
                            "protocol": PROTOCOL,
                            **key,
                            "status": "error",
                            "error": "StructSplat fit worker failed",
                            **timing,
                        }
                        _append_jsonl(ledger, row)
                        raise RuntimeError(f"StructSplat fit worker failed for {key}")
                    if timing["peak_gpu_device_bytes"] is None or timing["peak_rss_bytes"] is None:
                        raise RuntimeError(f"mandatory fresh-process memory samples missing for {key}")
                    worker = _load_json(cell_dir / "worker_result.json")

                    artifact_command = _control_artifact_worker_command(outdir, cell_dir)
                    _record_command(
                        outdir, "structsplat_postfit_artifacts", key, artifact_command, ROOT
                    )
                    artifact_stdout = log_dir / f"repeat_{repeat}_artifact_stdout.txt"
                    artifact_timing = _wall_command(
                        artifact_command,
                        cwd=ROOT,
                        env=worker_env,
                        stdout_path=artifact_stdout,
                    )
                    if artifact_timing["returncode"] != 0:
                        raise RuntimeError(f"StructSplat post-fit artifact worker failed for {key}")
                    artifacts = _load_json(cell_dir / "artifact_result.json")
                    cold_contract_path = cell_dir / "cold_contract.json"
                    cold_reconstruction = cell_dir / "cold_reconstruction.png"
                    cold_contract = {
                        "protocol": PROTOCOL,
                        "binding_file_sha256": binding_file_sha256,
                        "targets_file_sha256": targets_file_sha256,
                        "image_id": image_id,
                        "target_bpp": rate,
                        "arm": arm,
                        "repeat": repeat,
                        "sspl_path": artifacts["sspl_path"],
                        "sspl_sha256": artifacts["sspl_sha256"],
                        "cold_reconstruction_path": str(cold_reconstruction.resolve()),
                    }
                    _write_json_once(cold_contract_path, cold_contract)
                    cold_command = _control_cold_worker_command(cold_contract_path)
                    _record_command(
                        outdir, "structsplat_cold_decode_render", key, cold_command, ROOT
                    )
                    cold_stdout = log_dir / f"repeat_{repeat}_cold_stdout.txt"
                    cold_timing = _wall_command(
                        cold_command,
                        cwd=ROOT,
                        env=worker_env,
                        stdout_path=cold_stdout,
                    )
                    if cold_timing["returncode"] != 0:
                        raise RuntimeError(f"StructSplat cold-decode worker failed for {key}")
                    cold = _load_json(cell_dir / "cold_worker_result.json")
                    reconstruction = Path(worker["reconstruction_path"])
                    central = _central_metrics(
                        Path(target_record["derived_path"]), reconstruction
                    )
                    cold_central = _central_metrics(
                        Path(target_record["derived_path"]),
                        Path(cold["cold_reconstruction_path"]),
                    )
                    from structsplat import codec

                    stream = Path(artifacts["sspl_path"])
                    try:
                        components = codec.blob_components(stream.read_bytes())
                    except Exception as exc:
                        raise RuntimeError(f"malformed persisted SSPL1 for {key}") from exc
                    if components != artifacts["sspl_components"]:
                        raise RuntimeError(f"persisted SSPL1 component drift for {key}")
                    coordinate_slack = (
                        SS_SCALARS_PER_GAUSSIAN * final_budget
                        - SAD_SCALARS_PER_SITE * budgets["native_repeat_max_n"]
                        if arm == "ss_equal_terminal_coordinates"
                        else None
                    )
                    if coordinate_slack is not None and coordinate_slack not in range(8):
                        raise RuntimeError("primary terminal-coordinate rounding slack drift")
                    row = {
                        **worker,
                        **artifacts,
                        **cold,
                        "binding_file_sha256": binding_file_sha256,
                        "targets_file_sha256": targets_file_sha256,
                        "status": "ok",
                        "error": "",
                        "scientific_artifact_origin": "fresh_v6_execution",
                        "target_path": target_record["derived_path"],
                        "target_byte_sha256": target_record["derived_byte_sha256"],
                        "target_pixel_sha256": target_record["derived_pixel_sha256"],
                        "height": target_record["derived_height"],
                        "width": target_record["derived_width"],
                        "pixel_count": target_record["pixel_count"],
                        "native_repeat_counts": [
                            int(native_row["sad_final_active_sites"])
                            for native_row in sorted(native, key=lambda row: row["repeat"])
                        ],
                        "native_count_rule": "maximum_across_three_repeats",
                        "native_count_min": budgets["native_repeat_min_n"],
                        "native_count_max": budgets["native_repeat_max_n"],
                        "terminal_coordinate_rounding_slack": coordinate_slack,
                        "training_wall_seconds": timing["wall_seconds"],
                        "process_wall_seconds": timing["wall_seconds"],
                        "peak_gpu_device_bytes": timing["peak_gpu_device_bytes"],
                        "peak_rss_bytes": timing["peak_rss_bytes"],
                        "resource_samples_path": timing["resource_samples_path"],
                        "resource_samples_sha256": timing["resource_sha256"],
                        "raw_stdout_path": str(worker_stdout.resolve()),
                        "raw_stdout_sha256": timing["stdout_sha256"],
                        "worker_result_path": str((cell_dir / "worker_result.json").resolve()),
                        "worker_result_sha256": _sha256_file(cell_dir / "worker_result.json"),
                        "cell_contract_path": str(contract_path.resolve()),
                        "cell_contract_sha256": _sha256_file(contract_path),
                        "postfit_artifact_process_wall_seconds": artifact_timing["wall_seconds"],
                        "postfit_artifact_stdout_path": str(artifact_stdout.resolve()),
                        "postfit_artifact_stdout_sha256": artifact_timing["stdout_sha256"],
                        "artifact_result_path": str((cell_dir / "artifact_result.json").resolve()),
                        "artifact_result_sha256": _sha256_file(cell_dir / "artifact_result.json"),
                        "cold_contract_path": str(cold_contract_path.resolve()),
                        "cold_contract_sha256": _sha256_file(cold_contract_path),
                        "cold_worker_result_path": str(
                            (cell_dir / "cold_worker_result.json").resolve()
                        ),
                        "cold_worker_result_sha256": _sha256_file(
                            cell_dir / "cold_worker_result.json"
                        ),
                        "sspl_cold_process_wall_seconds": cold_timing["wall_seconds"],
                        "sspl_cold_process_started_monotonic_ns": cold_timing[
                            "process_started_monotonic_ns"
                        ],
                        "sspl_cold_process_ended_monotonic_ns": cold_timing[
                            "process_ended_monotonic_ns"
                        ],
                        "cold_worker_stdout_path": str(cold_stdout.resolve()),
                        "cold_worker_stdout_sha256": cold_timing["stdout_sha256"],
                        "fit_worker_internal_wall_seconds": worker["training_wall_seconds"],
                        **central,
                        "cold_metrics": cold_central,
                        "compression_tested": False,
                        "sspl_scope": "descriptive StructSplat complete-stream diagnostic only",
                    }
                    _append_jsonl(ledger, row)
                    new_rows.append(row)
                    done.add(key_tuple)
                    if max_new_rows is not None and len(new_rows) >= max_new_rows:
                        return new_rows
    return new_rows


def aggregate_traces(traces: Sequence[Sequence[dict[str, float]]]) -> list[dict[str, float]]:
    if not traces:
        raise ValueError("cannot aggregate zero convergence traces")
    common_traces = [_trace_on_common_domain(trace) for trace in traces]
    progress = sorted(
        {float(point["progress"]) for trace in common_traces for point in trace}
    )
    if progress[0] != COMMON_AUC_START or progress[-1] != 1.0:
        raise ValueError("all aggregate traces must cover [1/4000,1]")
    interpolated = []
    for trace in common_traces:
        x = np.asarray([point["progress"] for point in trace], dtype=np.float64)
        y = np.asarray([point["psnr"] for point in trace], dtype=np.float64)
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("aggregate trace progress must be strictly increasing")
        interpolated.append(np.interp(progress, x, y))
    means = np.mean(np.stack(interpolated, axis=0), axis=0)
    return [
        {"progress": float(point), "psnr": float(value)}
        for point, value in zip(progress, means)
    ]


def _bootstrap_effects(
    values: Sequence[float], *, seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(TARGET_IDS),) or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap requires exactly eight finite image-level effects")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(array), size=(replicates, len(array)), dtype=np.int16)
    samples = array[indices]
    mean_samples = samples.mean(axis=1)
    median_samples = np.median(samples, axis=1)
    return {
        "replicates": replicates,
        "seed": seed,
        "rng": "numpy.PCG64",
        "indices_sha256": _sha256_bytes(indices.astype("<i2", copy=False).tobytes()),
        "mean_observed": float(array.mean()),
        "mean_ci95": [
            float(value)
            for value in np.quantile(mean_samples, [0.025, 0.975], method="linear")
        ],
        "median_observed": float(np.median(array)),
        "median_ci95": [
            float(value)
            for value in np.quantile(median_samples, [0.025, 0.975], method="linear")
        ],
    }


def _median_required(values: Iterable[Any], name: str) -> float:
    parsed = []
    for value in values:
        if value is None:
            raise RuntimeError(f"missing required value for {name}")
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"non-finite required value for {name}")
        parsed.append(number)
    if not parsed:
        raise RuntimeError(f"empty required values for {name}")
    return float(statistics.median(parsed))


def _validate_file_hash(row: dict[str, Any], path_key: str, hash_key: str) -> None:
    path = Path(row[path_key])
    if not path.is_file() or _sha256_file(path) != row[hash_key]:
        raise RuntimeError(f"artifact drift for {path_key}: {path}")


def _require_close(actual: Any, expected: Any, name: str, *, tolerance: float = 1e-8) -> None:
    if actual is None or expected is None:
        if actual is expected:
            return
        raise RuntimeError(f"{name} missing-value drift: {actual!r} != {expected!r}")
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{name} derivation drift: {actual!r} != {expected!r}")


def _validate_metric_derivation(
    row_metrics: dict[str, Any], recomputed: dict[str, Any], prefix: str
) -> None:
    for key in (
        "central_psnr",
        "central_ssim",
        "central_proxy_ms_ssim",
        "central_lpips_alex",
    ):
        _require_close(row_metrics.get(key), recomputed.get(key), f"{prefix}.{key}", tolerance=1e-7)


def _validate_resource_derivation(row: dict[str, Any]) -> None:
    resource = _load_json(Path(row["resource_samples_path"]))
    rss_samples = [int(sample["bytes"]) for sample in resource.get("rss_samples", [])]
    gpu_samples = [
        int(sample["bytes"])
        for sample in resource.get("gpu_samples", [])
        if int(sample["bytes"]) > 0
    ]
    if not rss_samples or not gpu_samples:
        raise RuntimeError("resource sample ledger lacks mandatory RSS/positive VRAM samples")
    if max(rss_samples) != int(row["peak_rss_bytes"]):
        raise RuntimeError("peak RSS is not derived from raw samples")
    if max(gpu_samples) != int(row["peak_gpu_device_bytes"]):
        raise RuntimeError("peak VRAM is not derived from raw samples")
    if int(resource["peak_rss_bytes"]) != int(row["peak_rss_bytes"]):
        raise RuntimeError("resource-ledger peak RSS drift")
    if int(resource["peak_gpu_device_bytes"]) != int(row["peak_gpu_device_bytes"]):
        raise RuntimeError("resource-ledger peak VRAM drift")
    _require_close(
        resource.get("process_wall_seconds"),
        row.get("training_wall_seconds"),
        "fresh_process_wall_seconds",
        tolerance=1e-10,
    )
    start = int(resource["process_started_monotonic_ns"])
    end = int(resource["process_ended_monotonic_ns"])
    _require_close(
        (end - start) / 1e9,
        row.get("training_wall_seconds"),
        "fresh_process_monotonic_boundary",
        tolerance=1e-10,
    )


def _validate_prepared_timing(row: dict[str, Any], prefix: str) -> None:
    samples = [float(value) for value in row.get("prepared_render_ms_samples", [])]
    if len(samples) != RENDER_SAMPLES or not all(
        math.isfinite(value) and value >= 0.0 for value in samples
    ):
        raise RuntimeError(f"{prefix} prepared timing sample drift")
    _require_close(
        statistics.median(samples),
        row.get("prepared_render_ms_median"),
        f"{prefix}.prepared_render_ms_median",
        tolerance=1e-10,
    )


def _validate_native_prepared_replays(
    row: dict[str, Any],
    primary_u8: np.ndarray,
    primary_metrics: dict[str, float | None],
    *,
    target_path: Path,
    binary: Path,
    cwd: Path,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Reopen and semantically rederive all 53 outputs without a pixel-variation gate."""
    records = row.get("prepared_render_records") or []
    if [int(record.get("sample_index", -1)) for record in records] != list(
        range(RENDER_WARMUPS + RENDER_SAMPLES)
    ):
        raise RuntimeError("native prepared timing raw-record cadence drift")
    reconstruction_paths = [Path(record["reconstruction_path"]).resolve() for record in records]
    stdout_paths = [Path(record["stdout_path"]).resolve() for record in records]
    primary_paths = {
        Path(row["primary_replay_reconstruction_path"]).resolve(),
        Path(row["primary_replay_stdout_path"]).resolve(),
    }
    expected_count = RENDER_WARMUPS + RENDER_SAMPLES
    if (
        len(set(reconstruction_paths)) != expected_count
        or len(set(stdout_paths)) != expected_count
        or not set(reconstruction_paths).isdisjoint(stdout_paths)
        or not set(reconstruction_paths + stdout_paths).isdisjoint(primary_paths)
    ):
        raise RuntimeError("native prepared replay paths must be unique for all 53 invocations")
    candidate_samples: list[float] = []
    render_samples: list[float] = []
    sites_path = Path(row["sites_path"])
    sites_sha256 = _sha256_file(sites_path)
    primary_pixel_sha256 = _pixel_sha256(primary_u8)
    site_count = int(row["sad_final_active_sites"])
    width = int(row["width"])
    height = int(row["height"])
    for record in records:
        sample_index = int(record["sample_index"])
        expected_role = (
            "discarded_warmup"
            if sample_index < RENDER_WARMUPS
            else "retained_timing_sample"
        )
        reconstruction_path = Path(record["reconstruction_path"])
        stdout_path = Path(record["stdout_path"])
        expected_command = _cold_sad_command(
            binary, sites_path, reconstruction_path, width, height
        )
        if (
            record.get("sample_role") != expected_role
            or int(record.get("invocation_ordinal_after_primary", -1)) != sample_index + 1
            or not reconstruction_path.is_file()
            or _sha256_file(reconstruction_path) != record.get("reconstruction_sha256")
            or not stdout_path.is_file()
            or _sha256_file(stdout_path) != record.get("stdout_sha256")
            or record.get("command_argv") != expected_command
            or record.get("command_sha256") != _command_sha256(expected_command)
            or record.get("cwd") != str(cwd.resolve())
            or record.get("config_path") != str((cwd / "training_config.json").resolve())
            or record.get("sites_sha256") != sites_sha256
            or record.get("primary_replay_pixel_sha256") != primary_pixel_sha256
            or int(record.get("returncode", -1)) != 0
        ):
            raise RuntimeError("native prepared replay identity/file/hash drift")
        started_ns = int(record.get("process_started_monotonic_ns", -1))
        ended_ns = int(record.get("process_ended_monotonic_ns", -1))
        if started_ns < 0 or ended_ns <= started_ns:
            raise RuntimeError("native prepared replay process-boundary drift")
        wall_ms = (ended_ns - started_ns) / 1e6
        _require_close(
            record.get("full_process_wall_ms"), wall_ms, "prepared replay process wall"
        )
        sample_u8 = _strict_rgb8(reconstruction_path)
        sample_pixel_sha256 = _pixel_sha256(sample_u8)
        if (
            sample_u8.shape != primary_u8.shape
            or record.get("reconstruction_pixel_sha256") != sample_pixel_sha256
        ):
            raise RuntimeError("native prepared replay shape/pixel-hash drift")
        semantics = _parse_render_stdout_semantics(
            stdout_path.read_text(encoding="utf-8", errors="replace"),
            expected_site_count=site_count,
            expected_width=width,
            expected_height=height,
            expected_saved_path=reconstruction_path,
        )
        _require_json_equal(record.get("stdout_semantics"), semantics, "prepared stdout semantics")
        candidate = float(semantics["candidate_build_ms"])
        render = float(semantics["render_ms"])
        _require_close(
            record.get("candidate_build_ms"), candidate, "prepared candidate event"
        )
        _require_close(record.get("render_ms"), render, "prepared render event")
        sample_metrics = _central_metrics(target_path, reconstruction_path)
        _validate_metric_derivation(
            record.get("central_metrics") or {}, sample_metrics, "prepared replay metrics"
        )
        variation = _reconstruction_variation_diagnostics(
            primary_u8,
            sample_u8,
            primary_metrics,
            sample_metrics,
            metric_delta_name="sample_minus_primary_metric_delta",
        )
        _require_json_equal(
            record.get("primary_replay_variation_diagnostics"),
            variation,
            "prepared replay threshold-free pixel/metric diagnostics",
        )
        if sample_index >= RENDER_WARMUPS:
            candidate_samples.append(candidate)
            render_samples.append(render)
    return candidate_samples, render_samples, _prepared_replay_aggregate(records)


def _require_json_equal(actual: Any, expected: Any, name: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise RuntimeError(f"{name} persisted derivation drift")


def _validate_payload_projection(
    row: dict[str, Any], payload: dict[str, Any], prefix: str, *, skip: frozenset[str] = frozenset()
) -> None:
    for key, expected in payload.items():
        if key not in skip:
            _require_json_equal(row.get(key), expected, f"{prefix}.{key}")


def _validate_npz_field(row: dict[str, Any]) -> None:
    count = int(row["final_budget"])
    with np.load(row["npz_path"], allow_pickle=False) as arrays:
        required_shapes = {
            "means": (count, 2),
            "log_scales": (count, 2),
            "rotations": (count,),
            "colors": (count, 3),
        }
        for name, shape in required_shapes.items():
            if name not in arrays.files or arrays[name].shape != shape:
                raise RuntimeError(f"StructSplat NPZ {name} shape drift")
            if not np.all(np.isfinite(arrays[name])):
                raise RuntimeError(f"StructSplat NPZ {name} contains non-finite values")
        if "opacities" in arrays.files or "color_grads" in arrays.files:
            raise RuntimeError("StructSplat ordinary constant-RGB field contract drift")


def _validate_native_initial_capacity_state(
    row: dict[str, Any], binding: dict[str, Any], trace: Sequence[dict[str, Any]]
) -> None:
    expected_initial = int(binding["upstream"]["resolved_config"]["DEFAULT_SITES"])
    expected_capacity = max(int(point["capacity_sites"]) for point in trace)
    if (
        int(row.get("sad_initial_active_sites", -1)) != expected_initial
        or int(row.get("sad_allocated_capacity_sites_exposed", -1)) != expected_capacity
    ):
        raise RuntimeError("native SAD initial/capacity state drift")


def _validate_rows(outdir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from structsplat import codec
    from structsplat.config import FitConfig

    targets = _validate_targets(outdir, validate_acquisition=True)
    target_map = _target_by_id(targets)
    binding = _load_json(outdir / "binding.json")
    binding_file_sha256 = _sha256_file(outdir / "binding.json")
    targets_file_sha256 = _sha256_file(outdir / "targets.json")
    sad = [row for row in _load_jsonl(outdir / "sad_rows.jsonl") if row.get("status") == "ok"]
    controls = [
        row for row in _load_jsonl(outdir / "structsplat_rows.jsonl") if row.get("status") == "ok"
    ]
    for row in [*sad, *controls]:
        if (
            row.get("protocol") != PROTOCOL
            or row.get("binding_file_sha256") != binding_file_sha256
            or row.get("targets_file_sha256") != targets_file_sha256
            or row.get("scientific_artifact_origin") != "fresh_v6_execution"
        ):
            raise RuntimeError("scientific row protocol/binding/target/fresh-origin identity drift")
    expected_sad = {
        (image_id, rate, repeat)
        for image_id in TARGET_IDS
        for rate in RATES
        for repeat in NATIVE_REPEATS
    }
    actual_sad = {(row["image_id"], float(row["target_bpp"]), int(row["repeat"])) for row in sad}
    if len(sad) != len(expected_sad) or actual_sad != expected_sad:
        raise RuntimeError(f"native SAD matrix incomplete/duplicated: {len(sad)}/48")
    expected_controls = {
        (image_id, rate, arm, repeat)
        for image_id in TARGET_IDS
        for rate in RATES
        for arm in SS_ARMS
        for repeat in NATIVE_REPEATS
    }
    actual_controls = {
        (row["image_id"], float(row["target_bpp"]), row["arm"], int(row["repeat"]))
        for row in controls
    }
    if len(controls) != len(expected_controls) or actual_controls != expected_controls:
        raise RuntimeError(f"StructSplat control matrix incomplete/duplicated: {len(controls)}/96")

    sad_repo = Path(binding["protocol_spec"]["sad"]["repo"])
    sad_paths = _sad_paths(sad_repo)
    for row in sad:
        target = target_map[row["image_id"]]
        rate = float(row["target_bpp"])
        repeat = int(row["repeat"])
        if (
            row.get("arm") != "sad_native"
            or rate not in RATES
            or repeat not in NATIVE_REPEATS
            or row.get("target_path") != target["derived_path"]
            or row.get("target_byte_sha256") != target["derived_byte_sha256"]
            or row.get("target_pixel_sha256") != target["derived_pixel_sha256"]
            or int(row.get("height", -1)) != int(target["derived_height"])
            or int(row.get("width", -1)) != int(target["derived_width"])
            or int(row.get("pixel_count", -1)) != int(target["pixel_count"])
        ):
            raise RuntimeError("native target hash mismatch")
        for path_key, hash_key in (
            ("command_path", "command_sha256"),
            ("raw_stdout_path", "stdout_sha256"),
            ("resource_samples_path", "resource_sha256"),
            ("sites_path", "sites_sha256"),
            ("sites_gzip_path", "sites_gzip_sha256"),
            ("training_reconstruction_path", "training_reconstruction_sha256"),
            (
                "primary_replay_reconstruction_path",
                "primary_replay_reconstruction_sha256",
            ),
            ("primary_replay_stdout_path", "primary_replay_stdout_sha256"),
        ):
            _validate_file_hash(row, path_key, hash_key)

        command_record = _load_json(Path(row["command_path"]))
        expected_key = {"image_id": row["image_id"], "target_bpp": rate, "repeat": repeat}
        expected_command = _sad_command(
            sad_paths["binary"], Path(target["derived_path"]), rate, Path(row["sites_path"]).parent
        )
        if (
            command_record.get("protocol") != PROTOCOL
            or command_record.get("binding_sha256") != binding_file_sha256
            or command_record.get("target_manifest_sha256") != targets_file_sha256
            or command_record.get("key") != expected_key
            or command_record.get("argv") != expected_command
        ):
            raise RuntimeError("native SAD command identity/config drift")

        stdout = Path(row["raw_stdout_path"]).read_text(encoding="utf-8", errors="replace")
        if "Training | backend=cuda" not in stdout:
            raise RuntimeError("native SAD persisted stdout lacks CUDA backend proof")
        trace = parse_sad_trace(stdout)
        _require_json_equal(row.get("convergence_trace"), trace, "native convergence trace")
        normalized_trace = native_progress_trace(trace)
        _require_json_equal(
            row.get("internal_training_state_normalized_iteration_trace"),
            normalized_trace,
            "native internal-training-state normalized-iteration trace",
        )
        _require_close(
            row.get("internal_training_state_normalized_iteration_psnr_auc_db"),
            normalized_psnr_auc(normalized_trace),
            "native internal-training-state normalized-iteration PSNR AUC",
        )
        if row.get("auc_scope") != "internal_training_state_normalized_iteration":
            raise RuntimeError("native AUC scope drift")
        _require_close(
            row.get("native_training_seconds"),
            _parse_named_time(stdout, _TRAINING_TIME_RE, "training time"),
            "native upstream training time",
        )
        _require_close(
            row.get("native_total_seconds"),
            _parse_named_time(stdout, _TOTAL_TIME_RE, "total time"),
            "native upstream total time",
        )

        final_sites, declared_sites = _count_sites(Path(row["sites_path"]))
        if (
            int(row.get("primitive_count", -1)) != final_sites
            or int(row.get("sad_final_active_sites", -1)) != final_sites
            or row.get("sad_declared_active_sites") != declared_sites
            or int(trace[-1]["active_sites"]) != final_sites
            or int(row.get("upstream_terminal_active_sites", -1)) != final_sites
            or int(row.get("trainable_scalar_count", -1))
            != final_sites * SAD_SCALARS_PER_SITE
        ):
            raise RuntimeError("native SAD persisted site/state count drift")
        _require_close(
            row.get("upstream_terminal_psnr_db"), trace[-1]["psnr"], "native terminal PSNR"
        )
        if row.get("sad_logged_active_site_trajectory") != [
            int(point["active_sites"]) for point in trace
        ] or row.get("sad_logged_capacity_site_trajectory") != [
            int(point["capacity_sites"]) for point in trace
        ]:
            raise RuntimeError("native SAD persisted count trajectory drift")
        _validate_native_initial_capacity_state(row, binding, trace)
        jfa_collision_diagnostics = _jfa_seed_collision_diagnostics(
            Path(row["sites_path"]),
            int(target["derived_width"]),
            int(target["derived_height"]),
            int(binding["upstream"]["resolved_config"]["CAND_DOWNSCALE"]),
        )
        _require_json_equal(
            row.get("jfa_seed_collision_diagnostics"),
            jfa_collision_diagnostics,
            "native JFA seed-collision diagnostics",
        )

        target_sites = int(math.floor(rate * target["pixel_count"] / 128.0 + 0.5))
        target_tolerance = max(1, int(math.ceil(0.01 * target_sites)))
        target_error = abs(final_sites - target_sites)
        achieved_bpp = 128.0 * final_sites / target["pixel_count"]
        if (
            int(row.get("sad_target_sites", -1)) != target_sites
            or int(row.get("sad_target_site_tolerance", -1)) != target_tolerance
            or int(row.get("sad_target_site_error", -1)) != target_error
            or row.get("sad_target_attainment_pass") is not True
            or target_error > target_tolerance
        ):
            raise RuntimeError("native SAD target-attainment derivation drift")
        for name, actual, expected in (
            ("sad_target_bpp", row.get("sad_target_bpp"), rate),
            ("sad_nominal_bpp", row.get("sad_nominal_bpp"), achieved_bpp),
        ):
            _require_close(actual, expected, name)
        expected_integers = {
            "sad_nominal_bytes_per_site": SAD_BYTES_PER_SITE,
            "sad_nominal_state_bytes": final_sites * SAD_BYTES_PER_SITE,
            "sad_raw_float32_state_bytes": final_sites * SAD_SCALARS_PER_SITE * FLOAT32_BYTES,
            "sad_exported_txt_bytes": Path(row["sites_path"]).stat().st_size,
        }
        for name, expected in expected_integers.items():
            if int(row.get(name, -1)) != expected:
                raise RuntimeError(f"native SAD {name} derivation drift")
        expected_gzip = gzip.compress(
            Path(row["sites_path"]).read_bytes(), compresslevel=9, mtime=0
        )
        if (
            Path(row["sites_gzip_path"]).read_bytes() != expected_gzip
            or gzip.decompress(expected_gzip) != Path(row["sites_path"]).read_bytes()
            or int(row.get("sad_exported_txt_gzip_bytes", -1)) != len(expected_gzip)
        ):
            raise RuntimeError("native SAD gzip diagnostic derivation drift")

        training_path = Path(row["training_reconstruction_path"])
        primary_path = Path(row["primary_replay_reconstruction_path"])
        training_u8 = _strict_rgb8(training_path)
        primary_u8 = _strict_rgb8(primary_path)
        expected_shape = (int(target["derived_height"]), int(target["derived_width"]), 3)
        if training_u8.shape != expected_shape or primary_u8.shape != expected_shape:
            raise RuntimeError("native SAD reconstruction shape/orientation drift")
        if (
            row.get("training_reconstruction_pixel_sha256") != _pixel_sha256(training_u8)
            or row.get("primary_replay_reconstruction_pixel_sha256")
            != _pixel_sha256(primary_u8)
        ):
            raise RuntimeError("native SAD reconstruction pixel hash drift")
        diagnostic_training_metrics = _central_metrics(
            Path(target["derived_path"]), training_path
        )
        primary_metrics = _central_metrics(Path(target["derived_path"]), primary_path)
        _validate_metric_derivation(
            row.get("diagnostic_training_metrics") or {},
            diagnostic_training_metrics,
            "native diagnostic training reconstruction",
        )
        _validate_metric_derivation(row, primary_metrics, "native role-fixed primary TXT replay")
        diagnostics = _native_training_replay_diagnostics(
            training_u8,
            primary_u8,
            diagnostic_training_metrics,
            primary_metrics,
        )
        _require_json_equal(
            row.get("training_vs_primary_replay_diagnostics"),
            diagnostics,
            "native training-versus-primary diagnostic",
        )
        if row.get("native_quality_reconstruction_scope") != (
            "role_fixed_first_fresh_official_saved_txt_replay_only"
        ):
            raise RuntimeError("native quality reconstruction scope drift")

        expected_primary_command = _cold_sad_command(
            sad_paths["binary"],
            Path(row["sites_path"]),
            primary_path,
            int(target["derived_width"]),
            int(target["derived_height"]),
        )
        if (
            int(row.get("primary_replay_invocation_ordinal", -1)) != 0
            or int(row.get("primary_replay_retry_count", -1)) != 0
            or row.get("primary_replay_selection_policy")
            != "role_fixed_first_fresh_official_saved_txt_replay_never_retried_or_selected"
            or row.get("primary_replay_command_argv") != expected_primary_command
            or row.get("primary_replay_command_sha256")
            != _command_sha256(expected_primary_command)
            or row.get("primary_replay_cwd") != str(sad_repo.resolve())
            or row.get("primary_replay_config_path")
            != str(sad_paths["config"].resolve())
            or int(row.get("primary_replay_returncode", -1)) != 0
        ):
            raise RuntimeError("native role-fixed primary replay identity/selection drift")
        primary_stdout = Path(row["primary_replay_stdout_path"]).read_text(
            encoding="utf-8", errors="replace"
        )
        primary_semantics = _parse_render_stdout_semantics(
            primary_stdout,
            expected_site_count=final_sites,
            expected_width=int(target["derived_width"]),
            expected_height=int(target["derived_height"]),
            expected_saved_path=primary_path,
        )
        _require_json_equal(
            row.get("primary_replay_stdout_semantics"),
            primary_semantics,
            "native primary replay stdout semantics",
        )
        candidate_ms = float(primary_semantics["candidate_build_ms"])
        render_ms = float(primary_semantics["render_ms"])
        primary_wall_ms = (
            int(row["primary_replay_process_ended_monotonic_ns"])
            - int(row["primary_replay_process_started_monotonic_ns"])
        ) / 1e6
        if primary_wall_ms <= 0.0:
            raise RuntimeError("native primary replay process-boundary drift")
        _require_close(
            row.get("primary_replay_candidate_build_ms"),
            candidate_ms,
            "primary replay candidate event",
        )
        _require_close(
            row.get("primary_replay_render_ms"),
            render_ms,
            "primary replay render event",
        )
        _require_close(
            row.get("primary_replay_full_process_wall_ms"),
            primary_wall_ms,
            "primary replay process wall",
        )
        _require_close(
            row.get("primary_replay_non_kernel_residual_upper_bound_ms"),
            max(0.0, primary_wall_ms - candidate_ms - render_ms),
            "primary replay non-kernel residual upper bound",
        )
        if row.get("primary_replay_non_kernel_residual_includes") != [
            "process_startup",
            "configuration_loading",
            "cuda_context_work",
            "txt_parse_and_host_to_device_upload",
            "png_output",
        ]:
            raise RuntimeError("native primary replay residual interpretation drift")

        candidate_samples, render_samples, variation_aggregate = (
            _validate_native_prepared_replays(
                row,
                primary_u8,
                primary_metrics,
                target_path=Path(target["derived_path"]),
                binary=sad_paths["binary"],
                cwd=sad_repo,
            )
        )
        _require_json_equal(
            row.get("prepared_candidate_build_ms_samples"),
            candidate_samples,
            "native prepared candidate samples",
        )
        _require_json_equal(
            row.get("prepared_render_ms_samples"), render_samples, "native prepared render samples"
        )
        _require_close(
            row.get("prepared_candidate_build_ms_median"),
            statistics.median(candidate_samples),
            "native prepared candidate median",
        )
        _validate_prepared_timing(row, "native")
        _require_json_equal(
            row.get("prepared_replay_variation_aggregate"),
            variation_aggregate,
            "native prepared replay variation aggregate",
        )
        if row.get("prepared_replay_pixel_variation_is_threshold_free_diagnostic") is not True:
            raise RuntimeError("native prepared replay threshold-free scope drift")
        if (
            row.get("compression_tested") is not False
            or row.get("sad_complete_stream_available") is not False
            or row.get("sad_recipient_replayable") is not True
            or row.get("sad_deployable") is not False
            or row.get("sad_self_contained") is not False
            or row.get("sad_compressed_representation") is not False
            or row.get("sad_representation_scope")
            != "recipient_replayable_saved_txt_plus_pinned_official_decoder"
        ):
            raise RuntimeError("native SAD recipient-replay/representation claim boundary drift")
        if row.get("peak_gpu_device_bytes") is None or row.get("peak_rss_bytes") is None:
            raise RuntimeError("native SAD memory measurement missing")
        _validate_resource_derivation(row)

    sad_by_cell = {
        (image_id, rate): sorted(
            [
                row
                for row in sad
                if row["image_id"] == image_id and float(row["target_bpp"]) == rate
            ],
            key=lambda item: int(item["repeat"]),
        )
        for image_id in TARGET_IDS
        for rate in RATES
    }
    for row in controls:
        target = target_map[row["image_id"]]
        rate = float(row["target_bpp"])
        arm = str(row["arm"])
        repeat = int(row["repeat"])
        native = sad_by_cell[(row["image_id"], rate)]
        budgets = conservative_control_budgets(native)
        expected_budget = budgets[arm]
        if (
            arm not in SS_ARMS
            or repeat not in NATIVE_REPEATS
            or row.get("method") != SS_METHOD
            or int(row.get("seed", -1)) != SS_SEED
            or row.get("target_path") != target["derived_path"]
            or row.get("target_byte_sha256") != target["derived_byte_sha256"]
            or row.get("target_pixel_sha256") != target["derived_pixel_sha256"]
            or int(row.get("height", -1)) != int(target["derived_height"])
            or int(row.get("width", -1)) != int(target["derived_width"])
            or int(row.get("pixel_count", -1)) != int(target["pixel_count"])
        ):
            raise RuntimeError("StructSplat target hash mismatch")
        for path_key, hash_key in (
            ("cell_contract_path", "cell_contract_sha256"),
            ("worker_result_path", "worker_result_sha256"),
            ("artifact_result_path", "artifact_result_sha256"),
            ("cold_contract_path", "cold_contract_sha256"),
            ("cold_worker_result_path", "cold_worker_result_sha256"),
            ("raw_stdout_path", "raw_stdout_sha256"),
            ("postfit_artifact_stdout_path", "postfit_artifact_stdout_sha256"),
            ("cold_worker_stdout_path", "cold_worker_stdout_sha256"),
            ("resource_samples_path", "resource_samples_sha256"),
            ("reconstruction_path", "reconstruction_sha256"),
            ("npz_path", "npz_sha256"),
            ("sspl_path", "sspl_sha256"),
            ("cold_reconstruction_path", "cold_reconstruction_sha256"),
        ):
            _validate_file_hash(row, path_key, hash_key)

        contract = _load_json(Path(row["cell_contract_path"]))
        cold_contract = _load_json(Path(row["cold_contract_path"]))
        worker = _load_json(Path(row["worker_result_path"]))
        artifacts = _load_json(Path(row["artifact_result_path"]))
        cold = _load_json(Path(row["cold_worker_result_path"]))
        expected_identity = {
            "protocol": PROTOCOL,
            "binding_file_sha256": binding_file_sha256,
            "targets_file_sha256": targets_file_sha256,
            "image_id": row["image_id"],
            "target_bpp": rate,
            "arm": arm,
            "repeat": repeat,
        }
        for key, expected in expected_identity.items():
            if contract.get(key) != expected:
                raise RuntimeError(f"StructSplat cell contract {key} drift")
        if (
            contract.get("method") != SS_METHOD
            or int(contract.get("seed", -1)) != SS_SEED
            or int(contract.get("final_budget", -1)) != expected_budget
            or contract.get("target_record") != target
            or contract.get("required_ld_preload") != binding["required_ld_preload"]
        ):
            raise RuntimeError("StructSplat cell contract method/config drift")
        if (
            cold_contract.get("protocol") != PROTOCOL
            or cold_contract.get("binding_file_sha256") != binding_file_sha256
            or cold_contract.get("targets_file_sha256") != targets_file_sha256
            or cold_contract.get("sspl_sha256") != row["sspl_sha256"]
            or cold_contract.get("cold_reconstruction_path")
            != row["cold_reconstruction_path"]
        ):
            raise RuntimeError("StructSplat fresh-cold contract drift")
        _validate_payload_projection(
            row, worker, "StructSplat fit worker", skip=frozenset({"training_wall_seconds"})
        )
        _validate_payload_projection(row, artifacts, "StructSplat artifact worker")
        _validate_payload_projection(row, cold, "StructSplat cold worker")
        _require_close(
            row.get("fit_worker_internal_wall_seconds"),
            worker.get("training_wall_seconds"),
            "StructSplat internal fit wall",
        )

        fit_config = FitConfig(**row["fit_config"])
        required_fit = {
            "iters": ITERATIONS,
            "renderer": "cuda",
            "pixel_loss": "l1",
            "ssim_weight": 0.3,
            "compute_lpips": False,
            "target_psnr": None,
            "target_psnrs": [],
            "early_stop_patience": None,
            "checkpoint_policy": "best_psnr_final_count",
            "color_basis": "constant",
            "qat_mode": "off",
            "adaptive_count": False,
            "support_fade": False,
            "log_every": LOG_FREQUENCY,
            "split_mode": "residual_tensor_add",
            "refine_site": "residual_tensor",
            "refine_primitive": "sampled_add",
            "split_schedule_stops_at_max": False,
            "max_gaussians": expected_budget,
            "split_every": 666,
        }
        drift = {
            key: (getattr(fit_config, key), expected)
            for key, expected in required_fit.items()
            if getattr(fit_config, key) != expected
        }
        start_budget = max(16, round(SS_START_FRACTION * expected_budget))
        difference = expected_budget - start_budget
        method_metadata = row.get("method_metadata") or {}
        init_config = method_metadata.get("init_config") or {}
        expected_feature_cap_px = (
            SS_FEATURE_CAP
            * max(int(target["derived_height"]), int(target["derived_width"]))
            / SS_FEATURE_CAP_REFERENCE_SIDE
        )
        metadata_drift = (
            method_metadata.get("growth_rule") != "residual_tensor_add"
            or method_metadata.get("configured_growth_waves") != SS_GROWTH_WAVES
            or method_metadata.get("scale_cap_rule") != "feature"
            or method_metadata.get("scale_cap_input") != SS_FEATURE_CAP
            or method_metadata.get("scale_cap_reference_side")
            != SS_FEATURE_CAP_REFERENCE_SIDE
            or not math.isclose(
                float(method_metadata.get("scale_cap_max", float("nan"))),
                expected_feature_cap_px,
            )
            or init_config.get("strategy") != "aniso_onedge"
            or init_config.get("sampling_mode") != "wse"
            or init_config.get("opacity_mode") != "none"
            or int(init_config.get("num_gaussians", -1)) != start_budget
            or int(init_config.get("seed", -1)) != SS_SEED
        )
        if (
            drift
            or metadata_drift
            or difference < 21
            or int(row.get("final_budget", -1)) != expected_budget
            or int(row.get("start_budget", -1)) != start_budget
            or int(row.get("actual_start_gaussians", -1)) != start_budget
            or int(row.get("growth_difference", -1)) != difference
            or fit_config.split_count != int(math.ceil(difference / SS_GROWTH_WAVES))
            or int(row.get("iterations_run", -1)) != ITERATIONS
            or int(row.get("trajectory_terminal_iter", -1)) != ITERATIONS
        ):
            raise RuntimeError(f"StructSplat frozen fit/horizon config drift: {drift}")
        trace = row.get("convergence_trace") or []
        expected_iterations = [*range(0, ITERATIONS, LOG_FREQUENCY), ITERATIONS - 1]
        if [int(point.get("iteration", -1)) for point in trace] != expected_iterations:
            raise RuntimeError("StructSplat exact history cadence drift")
        normalized_trace = structsplat_progress_trace(trace, row["trajectory_terminal_psnr"])
        _require_json_equal(
            row.get("internal_training_state_normalized_iteration_trace"),
            normalized_trace,
            "StructSplat internal-training-state normalized-iteration trace",
        )
        _require_close(
            row.get("internal_training_state_normalized_iteration_psnr_auc_db"),
            normalized_psnr_auc(normalized_trace),
            "StructSplat internal-training-state normalized-iteration PSNR AUC",
        )
        if row.get("auc_scope") != "internal_training_state_normalized_iteration":
            raise RuntimeError("StructSplat AUC scope drift")
        split_events = row.get("split_events") or []
        expected_split_iterations = [665, 1331, 1997, 2663, 3329]
        if (
            len(split_events) != SS_GROWTH_WAVES
            or [int(event.get("iter", -1)) for event in split_events]
            != expected_split_iterations
            or any(int(event.get("added", 0)) <= 0 for event in split_events)
            or sum(int(event["added"]) for event in split_events) != difference
            or row.get("split_event_iterations_zero_based") != expected_split_iterations
            or int(row.get("split_event_additions_sum", -1)) != difference
            or row.get("sixth_scheduled_opportunity_was_cap_noop") is not True
            or int(row.get("sixth_scheduled_completed_step", -1)) != 3996
        ):
            raise RuntimeError("StructSplat exact five-growth-event contract drift")
        if any(
            int(row.get(name, -1)) != expected_budget
            for name in (
                "final_gaussians",
                "selected_n_gaussians",
                "trajectory_terminal_n_gaussians",
                "primitive_count",
            )
        ):
            raise RuntimeError("StructSplat selected/terminal count drift")
        if (
            int(row.get("trainable_scalar_count", -1))
            != expected_budget * SS_SCALARS_PER_GAUSSIAN
            or int(row.get("structsplat_nominal_float32_state_bytes", -1))
            != expected_budget * SS_SCALARS_PER_GAUSSIAN * FLOAT32_BYTES
        ):
            raise RuntimeError("StructSplat nominal state accounting drift")
        expected_native_counts = [int(item["sad_final_active_sites"]) for item in native]
        expected_slack = (
            SS_SCALARS_PER_GAUSSIAN * expected_budget
            - SAD_SCALARS_PER_SITE * budgets["native_repeat_max_n"]
            if arm == "ss_equal_terminal_coordinates"
            else None
        )
        if (
            row.get("native_repeat_counts") != expected_native_counts
            or int(row.get("native_count_min", -1)) != budgets["native_repeat_min_n"]
            or int(row.get("native_count_max", -1)) != budgets["native_repeat_max_n"]
            or row.get("native_count_rule") != "maximum_across_three_repeats"
            or row.get("terminal_coordinate_rounding_slack") != expected_slack
            or (expected_slack is not None and expected_slack not in range(8))
        ):
            raise RuntimeError("StructSplat conservative coordinate budget drift")

        _validate_npz_field(row)
        stream = Path(row["sspl_path"])
        blob = stream.read_bytes()
        try:
            components = codec.blob_components(blob)
            header = codec.blob_header(blob)
        except Exception as exc:
            raise RuntimeError("malformed persisted SSPL1") from exc
        if (
            components != row.get("sspl_components")
            or int(row.get("sspl_bytes", -1)) != len(blob)
            or components["total_bytes"] != len(blob)
            or int(header.get("n", -1)) != expected_budget
            or int(header.get("H", -1)) != int(target["derived_height"])
            or int(header.get("W", -1)) != int(target["derived_width"])
            or row.get("sspl_codec_config") != asdict(codec.CodecConfig())
        ):
            raise RuntimeError("SSPL1 byte accounting drift")
        _require_close(
            row.get("sspl_actual_bpp"),
            len(blob) * 8.0 / target["pixel_count"],
            "SSPL1 actual BPP",
        )

        reconstruction = Path(row["reconstruction_path"])
        cold_reconstruction = Path(row["cold_reconstruction_path"])
        selected_u8 = _rgb8(reconstruction)
        cold_u8 = _rgb8(cold_reconstruction)
        expected_shape = (int(target["derived_height"]), int(target["derived_width"]), 3)
        if selected_u8.shape != expected_shape or cold_u8.shape != expected_shape:
            raise RuntimeError("StructSplat reconstruction shape/orientation drift")
        if (
            row.get("reconstruction_pixel_sha256") != _pixel_sha256(selected_u8)
            or row.get("cold_reconstruction_pixel_sha256") != _pixel_sha256(cold_u8)
        ):
            raise RuntimeError("StructSplat reconstruction pixel hash drift")
        central = _central_metrics(Path(target["derived_path"]), reconstruction)
        cold_central = _central_metrics(Path(target["derived_path"]), cold_reconstruction)
        _validate_metric_derivation(row, central, "StructSplat selected reconstruction")
        _validate_metric_derivation(
            row.get("cold_metrics") or {}, cold_central, "StructSplat decoded reconstruction"
        )

        _validate_prepared_timing(row, "StructSplat")
        cold_wall = (
            int(row["sspl_cold_process_ended_monotonic_ns"])
            - int(row["sspl_cold_process_started_monotonic_ns"])
        ) / 1e9
        _require_close(
            row.get("sspl_cold_process_wall_seconds"), cold_wall, "SSPL1 cold process wall"
        )
        cold_metric = float(row.get("sspl_cold_decode_render_seconds", float("nan")))
        if (
            not math.isfinite(cold_metric)
            or cold_metric <= 0.0
            or cold_metric > cold_wall
            or row.get("sspl_cold_metric_excludes_process_startup") is not True
            or row.get("sspl_cold_metric_boundary")
            != "stream_read_through_synchronized_exact_render"
        ):
            raise RuntimeError("SSPL1 fresh-cold metric boundary drift")
        if row.get("compression_tested") is not False:
            raise RuntimeError("global compression claim boundary drift")
        if row.get("peak_gpu_device_bytes") is None or row.get("peak_rss_bytes") is None:
            raise RuntimeError("StructSplat memory measurement missing")
        _validate_resource_derivation(row)
    return sad, controls


def _analyze_persisted(outdir: Path) -> dict[str, Any]:
    sad, controls = _validate_rows(outdir)
    cells = []
    rate_summaries = []
    scalar_metrics = (
        "central_psnr",
        "central_ssim",
        "central_proxy_ms_ssim",
        "central_lpips_alex",
    )
    resource_metrics = (
        "training_wall_seconds",
        "prepared_render_ms_median",
        "peak_gpu_device_bytes",
        "peak_rss_bytes",
    )

    def aggregate_arm(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if len(rows) != len(NATIVE_REPEATS):
            raise RuntimeError("arm aggregation requires exactly three execution repeats")
        trace = aggregate_traces(
            [row["internal_training_state_normalized_iteration_trace"] for row in rows]
        )
        averages = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in scalar_metrics
        }
        averages["internal_training_state_normalized_iteration_psnr_auc_db"] = (
            normalized_psnr_auc(trace)
        )
        medians = {
            metric: _median_required((row.get(metric) for row in rows), metric)
            for metric in resource_metrics
        }
        return {
            "repeat_count": len(rows),
            "averages": averages,
            "medians": medians,
            "aggregate_trace": trace,
            "final_primitive_counts": [int(row["primitive_count"]) for row in rows],
            "sspl_bytes": [row.get("sspl_bytes") for row in rows],
            "sspl_actual_bpp": [row.get("sspl_actual_bpp") for row in rows],
        }

    for rate in RATES:
        rate_cells = []
        for image_id in TARGET_IDS:
            native = sorted(
                [row for row in sad if row["image_id"] == image_id and float(row["target_bpp"]) == rate],
                key=lambda row: row["repeat"],
            )
            equal_rows = sorted(
                [
                    row
                    for row in controls
                    if row["image_id"] == image_id
                    and float(row["target_bpp"]) == rate
                    and row["arm"] == "ss_equal_terminal_coordinates"
                ],
                key=lambda row: row["repeat"],
            )
            same_rows = sorted(
                [
                    row
                    for row in controls
                    if row["image_id"] == image_id
                    and float(row["target_bpp"]) == rate
                    and row["arm"] == "ss_same_n"
                ],
                key=lambda row: row["repeat"],
            )
            native_summary = aggregate_arm(native)
            equal_summary = aggregate_arm(equal_rows)
            same_summary = aggregate_arm(same_rows)
            cell = {
                "image_id": image_id,
                "target_bpp": rate,
                "native_site_counts": [row["sad_final_active_sites"] for row in native],
                "native_achieved_nominal_bpp": [row["sad_nominal_bpp"] for row in native],
                "native_sites_hash_agreement": len({row["sites_sha256"] for row in native}) == 1,
                "native_primary_replay_hash_agreement": len(
                    {row["primary_replay_reconstruction_sha256"] for row in native}
                )
                == 1,
                "sad_native": native_summary,
                "ss_equal_terminal_coordinates": equal_summary,
                "ss_same_n": same_summary,
                "terminal_coordinate_rounding_slack": equal_rows[0][
                    "terminal_coordinate_rounding_slack"
                ],
            }
            native_average = native_summary["averages"]
            native_medians = native_summary["medians"]
            equal_average = equal_summary["averages"]
            equal_medians = equal_summary["medians"]
            cell["effects_vs_equal_terminal_coordinates"] = {
                "psnr_gain_db": native_average["central_psnr"]
                - equal_average["central_psnr"],
                "lpips_delta": native_average["central_lpips_alex"]
                - equal_average["central_lpips_alex"],
                "internal_training_state_auc_gain_db": native_average[
                    "internal_training_state_normalized_iteration_psnr_auc_db"
                ]
                - equal_average[
                    "internal_training_state_normalized_iteration_psnr_auc_db"
                ],
                "training_time_ratio": native_medians["training_wall_seconds"]
                / equal_medians["training_wall_seconds"],
                "prepared_render_time_ratio": native_medians["prepared_render_ms_median"]
                / equal_medians["prepared_render_ms_median"],
                "peak_vram_ratio": native_medians["peak_gpu_device_bytes"]
                / equal_medians["peak_gpu_device_bytes"],
                "peak_rss_ratio": native_medians["peak_rss_bytes"]
                / equal_medians["peak_rss_bytes"],
            }
            rate_cells.append(cell)
            cells.append(cell)
        effects = [cell["effects_vs_equal_terminal_coordinates"] for cell in rate_cells]
        psnr = [effect["psnr_gain_db"] for effect in effects]
        lpips = [effect["lpips_delta"] for effect in effects]
        auc = [effect["internal_training_state_auc_gain_db"] for effect in effects]
        training = [effect["training_time_ratio"] for effect in effects]
        render = [effect["prepared_render_time_ratio"] for effect in effects]
        vram = [effect["peak_vram_ratio"] for effect in effects]
        rss = [effect["peak_rss_ratio"] for effect in effects]
        bootstraps = {
            name: _bootstrap_effects(values, seed=BOOTSTRAP_SEED)
            for name, values in (
                ("psnr_gain_db", psnr),
                ("lpips_delta", lpips),
                ("internal_training_state_auc_gain_db", auc),
                ("training_time_ratio", training),
                ("prepared_render_time_ratio", render),
                ("peak_vram_ratio", vram),
                ("peak_rss_ratio", rss),
            )
        }
        bootstrap = bootstraps["psnr_gain_db"]
        gates = {
            "median_psnr_gain_at_least_1db": float(np.median(psnr)) >= 1.0,
            "at_least_7_of_8_positive_psnr": int(np.sum(np.asarray(psnr) > 0.0)) >= 7,
            "worst_psnr_gain_at_least_minus_0p25db": min(psnr) >= -0.25,
            "bootstrap_mean_psnr_lower_strictly_positive": bootstrap["mean_ci95"][0] > 0.0,
            "median_lpips_nonworse": float(np.median(lpips)) <= 0.0,
            "median_internal_training_state_auc_gain_at_least_0p5db": (
                float(np.median(auc)) >= 0.5
            ),
            "median_training_time_ratio_at_most_0p50": float(np.median(training)) <= 0.50,
            "median_prepared_render_ratio_at_most_1p10": float(np.median(render)) <= 1.10,
            "median_peak_vram_ratio_at_most_1p50": float(np.median(vram)) <= 1.50,
            "median_peak_rss_ratio_at_most_1p50": float(np.median(rss)) <= 1.50,
        }
        rate_summaries.append(
            {
                "target_bpp": rate,
                "images": len(rate_cells),
                "psnr_gain_db": {
                    "median": float(np.median(psnr)),
                    "mean": float(np.mean(psnr)),
                    "worst": min(psnr),
                    "positive_images": int(np.sum(np.asarray(psnr) > 0.0)),
                    "bootstrap": bootstrap,
                },
                "median_lpips_delta": float(np.median(lpips)),
                "median_internal_training_state_auc_gain_db": float(np.median(auc)),
                "median_training_time_ratio": float(np.median(training)),
                "median_prepared_render_time_ratio": float(np.median(render)),
                "median_peak_vram_ratio": float(np.median(vram)),
                "median_peak_rss_ratio": float(np.median(rss)),
                "paired_image_bootstraps": bootstraps,
                "gates": gates,
                "rate_pass": all(gates.values()),
            }
        )
    development_pass = all(summary["rate_pass"] for summary in rate_summaries)
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "matrix": {
            "independent_images": len(TARGET_IDS),
            "image_rate_cells": len(TARGET_IDS) * len(RATES),
            "native_repeats": len(TARGET_IDS) * len(RATES) * len(NATIVE_REPEATS),
            "structsplat_repetitions": (
                len(TARGET_IDS) * len(RATES) * len(SS_ARMS) * len(NATIVE_REPEATS)
            ),
            "all_valid": True,
            "fresh_v6_rows": 144,
            "v4_scientific_rows_imported": 0,
            "v5_scientific_rows_imported": 0,
        },
        "native_quality_reconstruction_scope": (
            "role_fixed_first_fresh_official_saved_txt_replay_only"
        ),
        "auc_scope": "internal_training_state_normalized_iteration",
        "cells": cells,
        "rate_summaries": rate_summaries,
        "development_screen_pass": development_pass,
        "decision": (
            "implement one controlled operator bridge" if development_pass else "abandon SAD reuse"
        ),
        "stage1_or_production_authorized": False,
        "compression_tested": False,
        "sad_recipient_replayable_only": True,
        "sad_deployable": False,
        "sad_self_contained": False,
        "sad_compressed_representation": False,
        "claim_boundary": {
            "quality": "eight-image downsampled DIV2K training development screen only",
            "convergence": (
                "internal-training-state normalized-iteration whole-system trajectory; "
                "mechanism and wall-time convergence not attributed"
            ),
            "performance": "single bound GPU/software environment only",
            "compression": (
                "not tested for SAD; TXT plus decoder is recipient-replayable only and is "
                "not deployable, self-contained, or a compressed representation"
            ),
            "expressiveness": (
                "at-least-matched terminal coordinates only; not equal training state or a rank theorem"
            ),
            "preregistration": (
                "v6 outcome-responsive integrity repair was frozen after v4 and v5 development "
                "access; all scientific choices were retained, but design-after-access risk remains"
            ),
        },
    }


def _artifact_manifest(outdir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(path for path in outdir.rglob("*") if path.is_file()):
        relative = path.relative_to(outdir).as_posix()
        if relative in LIFECYCLE_FILES:
            continue
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        )
    return {
        "protocol": PROTOCOL,
        "exclusions": sorted(LIFECYCLE_FILES),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }


def _validate_artifact_manifest(outdir: Path, manifest: dict[str, Any]) -> bool:
    if manifest.get("protocol") != PROTOCOL:
        return False
    return manifest == _artifact_manifest(outdir)


def analyze(outdir: Path) -> dict[str, Any]:
    _validate_binding(outdir)
    analysis = _analyze_persisted(outdir)
    _write_json_once(outdir / "analysis.json", analysis)
    manifest = _artifact_manifest(outdir)
    _write_json_once(outdir / "artifact_manifest.json", manifest)
    return analysis


def replay(outdir: Path) -> dict[str, Any]:
    _validate_binding(outdir)
    manifest = _load_json(outdir / "artifact_manifest.json")
    checks: dict[str, bool] = {
        "artifact_manifest": _validate_artifact_manifest(outdir, manifest),
    }
    recorded = _load_json(outdir / "analysis.json")
    recomputed = _analyze_persisted(outdir)
    checks["analysis_exact"] = _canonical_json(recorded) == _canonical_json(recomputed)
    result = {
        "protocol": PROTOCOL,
        "checks": checks,
        "check_count": len(checks),
        "pass_count": sum(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "artifact_audit_passed": all(checks.values()),
        "outcome_ledgers_interpreted": all(checks.values()),
    }
    _write_json_once(outdir / "replay.json", result)
    if not result["artifact_audit_passed"]:
        raise RuntimeError(f"BENCH-016 replay audit failed: {result['failed_checks']}")
    completion = {
        "protocol": PROTOCOL,
        "analysis_sha256": _sha256_file(outdir / "analysis.json"),
        "artifact_manifest_sha256": _sha256_file(outdir / "artifact_manifest.json"),
        "replay_sha256": _sha256_file(outdir / "replay.json"),
        "development_screen_pass": recorded["development_screen_pass"],
        "decision": recorded["decision"],
        "replay_audit_passed": True,
    }
    _write_json_once(outdir / "completion.json", completion)
    return result


def _add_selection(parser: argparse.ArgumentParser, *, repeats: bool = False) -> None:
    parser.add_argument("--image-id", action="append", choices=TARGET_IDS, dest="image_ids")
    parser.add_argument("--rate", action="append", type=float, choices=RATES, dest="rates")
    if repeats:
        parser.add_argument(
            "--repeat", action="append", type=int, choices=NATIVE_REPEATS, dest="repeats"
        )
    parser.add_argument("--max-new-rows", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BENCH-016 pinned native SAD frontier adapter")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--outdir", type=Path, required=True)
    preflight.add_argument("--sad-repo", type=Path, default=DEFAULT_SAD_REPO)
    preflight.add_argument("--source-dir", type=Path, default=None)
    prepare = subparsers.add_parser("prepare-targets")
    prepare.add_argument("--outdir", type=Path, required=True)
    prepare.add_argument("--source-dir", type=Path, required=True)
    sad = subparsers.add_parser("run-sad")
    sad.add_argument("--outdir", type=Path, required=True)
    _add_selection(sad, repeats=True)
    control = subparsers.add_parser("run-structsplat")
    control.add_argument("--outdir", type=Path, required=True)
    _add_selection(control)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--outdir", type=Path, required=True)
    audit = subparsers.add_parser("replay")
    audit.add_argument("--outdir", type=Path, required=True)
    worker = subparsers.add_parser("_control-worker")
    worker.add_argument("--contract", type=Path, required=True)
    worker.add_argument("--cell-dir", type=Path, required=True)
    artifact_worker = subparsers.add_parser("_control-artifact-worker")
    artifact_worker.add_argument("--outdir", type=Path, required=True)
    artifact_worker.add_argument("--cell-dir", type=Path, required=True)
    cold_worker = subparsers.add_parser("_control-cold-worker")
    cold_worker.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "preflight":
        result = run_preflight(args.outdir, args.sad_repo, args.source_dir)
    elif args.stage == "prepare-targets":
        result = prepare_targets(args.outdir, args.source_dir)
    elif args.stage == "run-sad":
        result = run_sad(
            args.outdir,
            image_ids=args.image_ids,
            rates=args.rates,
            repeats=args.repeats,
            max_new_rows=args.max_new_rows,
        )
    elif args.stage == "run-structsplat":
        result = run_structsplat(
            args.outdir,
            image_ids=args.image_ids,
            rates=args.rates,
            max_new_rows=args.max_new_rows,
        )
    elif args.stage == "analyze":
        result = analyze(args.outdir)
    elif args.stage == "replay":
        result = replay(args.outdir)
    elif args.stage == "_control-artifact-worker":
        result = _run_control_artifact_worker(args.outdir, args.cell_dir)
    elif args.stage == "_control-cold-worker":
        result = _run_control_cold_worker(args.contract)
    else:
        result = _run_control_worker(args.contract, args.cell_dir)
    if not args.stage.startswith("_"):
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
