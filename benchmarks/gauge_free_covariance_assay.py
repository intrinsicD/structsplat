"""Source-bound runner for COMP-007's gauge-free covariance codec assay.

The runner fits one shared float field per target/count, encodes the complete frozen candidate
matrix, cold-decodes every selected stream, and persists exact blobs plus a row ledger.  Scientific
reduction lives in :mod:`benchmarks.gauge_free_covariance_analysis`; production SSPL1 is untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping, Sequence
import zlib

import numpy as np
from PIL import Image
import torch

from benchmarks import gauge_free_covariance_analysis as analysis
from benchmarks import gauge_free_covariance_codec as experimental_codec
from benchmarks import gauge_free_covariance_core as core
from benchmarks.common import load_image
from structsplat import cuda_render as cuda_renderer
from structsplat import metrics as metrics
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import fit
from structsplat.gaussians import GaussianField
from structsplat.init import build_field
from structsplat.render import render_field


PROTOCOL = "comp007-gauge-free-covariance-codec-v3"
SCHEMA = "comp007-config-v3"
TARGET_MAX_SIDE = 160
COUNTS = (640, 2560)
FIT_STEPS = 500
FIT_SEED = 0
CHARTS = ("current_rs", "canonical_rs", "log_spd")
CODERS = ("zlib9", "zstd9")
PREDICTORS = ("absolute", "modular_delta")
TIMING_REPEATS = 5
BOOTSTRAP_SEED = 20_260_716
BOOTSTRAP_REPLICATES = 20_000
DEVELOPMENT_IDS = tuple(range(2, 25, 2))
CONFIRMATION_IDS = tuple(range(1, 24, 2))
SYSTEM_LIBSTDCXX = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
GEOMETRY_SYNTAX_BYTES = 29  # 6 float32 endpoints + chart/predictor + 3 bit-depth tags
QUALITY_REPLAY_TOLERANCES = {
    "mse": 1e-8,
    "psnr": 1e-5,
    "ssim": 1e-6,
    "ms_ssim": 1e-6,
}

SOURCE_PATHS = (
    "benchmarks/__init__.py",
    "benchmarks/_util.py",
    "benchmarks/gauge_free_covariance_assay.py",
    "benchmarks/gauge_free_covariance_analysis.py",
    "benchmarks/gauge_free_covariance_codec.py",
    "benchmarks/gauge_free_covariance_core.py",
    "benchmarks/common.py",
    "src/structsplat/__init__.py",
    "src/structsplat/cli.py",
    "src/structsplat/codec.py",
    "src/structsplat/config.py",
    "src/structsplat/density.py",
    "src/structsplat/fit.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/generate.py",
    "src/structsplat/init.py",
    "src/structsplat/metrics.py",
    "src/structsplat/predictor.py",
    "src/structsplat/pyramid.py",
    "src/structsplat/render.py",
    "src/structsplat/sampling.py",
    "src/structsplat/structural_controls.py",
    "src/structsplat/structure_tensor.py",
    "src/structsplat/visualize.py",
    "src/structsplat/cuda_render.py",
    "src/structsplat/cuda/__init__.py",
    "src/structsplat/cuda/render_ext.cpp",
    "src/structsplat/cuda/render_ext.cu",
    "pyproject.toml",
)
ARCHIVE_PATHS = SOURCE_PATHS + (
    "tasks/COMP-007-gauge-free-covariance-codec.md",
    "tests/test_gauge_free_covariance_analysis.py",
    "tests/test_gauge_free_covariance_assay.py",
    "tests/test_gauge_free_covariance_codec.py",
    "tests/test_gauge_free_covariance_core.py",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha256_bytes(
        _canonical_json(
            {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "payload_sha256": _sha256_bytes(array.tobytes(order="C")),
            }
        )
    )


def _field_sha256(field: GaussianField) -> str:
    arrays = {
        "means": field.means.detach().cpu().contiguous().numpy(),
        "effective_log_scales": field.effective_log_scales().detach().cpu().contiguous().numpy(),
        "rotations": field.rotations.detach().cpu().contiguous().numpy(),
        "colors": field.colors.detach().cpu().contiguous().numpy(),
    }
    return _sha256_bytes(
        _canonical_json({name: _array_sha256(array) for name, array in arrays.items()})
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as handle:
        handle.write(_canonical_json(dict(row)) + b"\n")
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
            "tracked_diff_sha256": _sha256_bytes(tracked_diff),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _environment(root: Path, device: str) -> dict[str, Any]:
    cuda: dict[str, Any] | None = None
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.device(device))
        extension = cuda_renderer._load_extension()  # noqa: SLF001 - bind the executed binary
        extension_path = Path(extension.__file__).resolve()
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            driver = None
        cuda = {
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "cuda_runtime": torch.version.cuda,
            "driver": driver,
            "renderer_extension": {
                "filename": extension_path.name,
                "size_bytes": extension_path.stat().st_size,
                "sha256": _sha256_file(extension_path),
            },
        }
    try:
        zstandard_version = importlib.metadata.version("zstandard")
    except importlib.metadata.PackageNotFoundError:
        zstandard_version = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pillow": Image.__version__,
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "zstandard": zstandard_version,
        "device": device,
        "cuda": cuda,
        "torch_num_threads": torch.get_num_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "git": _git_metadata(root),
    }


def _environment_signature(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Return the execution-relevant, resume-stable portion of an environment record."""
    keys = (
        "python",
        "platform",
        "numpy",
        "torch",
        "pillow",
        "zlib_compile",
        "zlib_runtime",
        "zstandard",
        "device",
        "cuda",
        "torch_num_threads",
        "cublas_workspace_config",
        "ld_preload",
    )
    return {key: environment.get(key) for key in keys}


def _binding_sha256(science: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "science": dict(science),
                "environment": _environment_signature(environment),
            }
        )
    )


def _source_manifest(root: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing COMP-007 source dependency: {relative}")
        manifest[relative] = _sha256_file(path)
    return manifest


def _write_source_archive(root: Path, outdir: Path) -> dict[str, Any]:
    archive = outdir / "executed_sources_v3.tar"
    with tarfile.open(archive, mode="w", format=tarfile.PAX_FORMAT) as handle:
        for relative in ARCHIVE_PATHS:
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"missing COMP-007 archive dependency: {relative}")
            info = handle.gettarinfo(str(path), arcname=relative)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as source:
                handle.addfile(info, source)
    return {
        "path": archive.name,
        "sha256": _sha256_file(archive),
        "members": {relative: _sha256_file(root / relative) for relative in ARCHIVE_PATHS},
    }


def allocations() -> tuple[tuple[int, int, int], ...]:
    return analysis.expected_allocations()


def _target_path(root: Path, image_id: int) -> Path:
    return root / "results" / "datasets" / "abl004" / "kodak24" / f"kodim{image_id:02d}.png"


def _target_manifest(root: Path) -> list[dict[str, Any]]:
    entries = []
    for image_id in range(1, 25):
        path = _target_path(root, image_id)
        if not path.is_file():
            raise RuntimeError(f"missing frozen Kodak source image: {path}")
        target = load_image(path, max_side=TARGET_MAX_SIDE).astype(np.float32, copy=False)
        entries.append(
            {
                "image_id": f"{image_id:02d}",
                "split": "development" if image_id in DEVELOPMENT_IDS else "confirmation",
                "relative_path": path.relative_to(root).as_posix(),
                "source_png_sha256": _sha256_file(path),
                "source_size_bytes": path.stat().st_size,
                "target_shape": list(target.shape),
                "target_float32_sha256": _array_sha256(target),
            }
        )
    return entries


def _fit_config() -> FitConfig:
    return FitConfig(
        iters=FIT_STEPS,
        renderer="cuda_tiled",
        color_basis="constant",
        ssim_backend="builtin",
        log_every=FIT_STEPS,
        checkpoint_policy="terminal",
    )


def _science_config(
    root: Path,
    split: str,
    confirmation_parent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in {"development", "confirmation"}:
        raise ValueError("split must be development or confirmation")
    task_path = root / "tasks" / "COMP-007-gauge-free-covariance-codec.md"
    science = {
        "protocol": PROTOCOL,
        "split": split,
        "target_max_side": TARGET_MAX_SIDE,
        "development_ids": list(DEVELOPMENT_IDS),
        "confirmation_ids": list(CONFIRMATION_IDS),
        "target_manifest": _target_manifest(root),
        "counts": list(COUNTS),
        "fit_seed": FIT_SEED,
        "fit_config": asdict(_fit_config()),
        "init_config": asdict(
            InitConfig(
                strategy="quadtree_wse",
                num_gaussians=COUNTS[0],
                opacity_mode="none",
                seed=FIT_SEED,
            )
        ),
        "structure_tensor_config": asdict(StructureTensorConfig()),
        "charts": list(CHARTS),
        "coders": list(CODERS),
        "predictors": list(PREDICTORS),
        "allocations": [list(value) for value in allocations()],
        "timing_repeats": TIMING_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "independent_unit": "source_image",
        "task_protocol_sha256": _sha256_file(task_path),
        "source_manifest": _source_manifest(root),
    }
    if split == "confirmation":
        if confirmation_parent is None:
            raise ValueError("confirmation science requires a verified development parent")
        science["confirmation_parent"] = dict(confirmation_parent)
    elif confirmation_parent is not None:
        raise ValueError("development science cannot carry a confirmation parent")
    return science


def prepare_run(
    outdir: Path,
    root: Path,
    split: str,
    device: str,
    confirmation_parent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    science = _science_config(root, split, confirmation_parent)
    environment = _environment(root, device)
    binding_sha256 = _binding_sha256(science, environment)
    config_path = outdir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema") != SCHEMA or config.get("binding_sha256") != binding_sha256:
            raise RuntimeError("COMP-007 config/source/data binding drift; use a new outdir")
        return config
    outdir.mkdir(parents=True, exist_ok=False)
    archive = _write_source_archive(root, outdir)
    config = {
        "schema": SCHEMA,
        "binding_sha256": binding_sha256,
        "science": science,
        "environment": environment,
        "executed_source_archive": archive,
        "command": list(sys.argv),
    }
    _atomic_json(config_path, config)
    return config


def _split_ids(split: str) -> tuple[int, ...]:
    return DEVELOPMENT_IDS if split == "development" else CONFIRMATION_IDS


def _confirmation_authorized(
    path: Path | None,
    *,
    root: Path | None = None,
    device: str = "cuda",
) -> bool:
    if path is None or not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        outdir = path.parent
        config_path = outdir / "config.json"
        audit_path = outdir / "artifact_audit.json"
        if not config_path.is_file() or not audit_path.is_file():
            return False
        config = json.loads(config_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if root is None:
            root = Path(__file__).resolve().parents[1]
        expected_science = _science_config(root, "development")
        current_environment = _environment(root, device)
        authorization = summary.get("confirmation_authorization", {})
        payload = _confirmation_authorization_payload(summary, outdir)
        recomputed_analysis = analysis.analyze_candidates(
            _read_jsonl(outdir / "candidates.jsonl"),
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_replicates=BOOTSTRAP_REPLICATES,
        )
        return bool(
            summary.get("schema") == "comp007-analysis-v2"
            and summary.get("split") == "development"
            and summary.get("analysis", {}).get("decision") == "pass_authorize_confirmation"
            and summary.get("analysis", {}).get("confirmation_authorized") is True
            and summary.get("analysis") == recomputed_analysis
            and summary.get("checks", {}).get("artifact_audit_passed") is True
            and config.get("schema") == SCHEMA
            and config.get("science") == expected_science
            and config.get("science", {}).get("split") == "development"
            and config.get("binding_sha256")
            == _binding_sha256(config["science"], config["environment"])
            and _environment_signature(config["environment"])
            == _environment_signature(current_environment)
            and summary.get("binding_sha256") == config.get("binding_sha256")
            and summary.get("config_sha256") == _sha256_file(config_path)
            and summary.get("artifact_audit_sha256") == _sha256_file(audit_path)
            and audit.get("passes") is True
            and audit.get("rerendered") is True
            and audit.get("decision_ready") is True
            and authorization.get("schema") == "comp007-confirmation-authorization-v1"
            and authorization.get("payload") == payload
            and authorization.get("sha256") == _sha256_bytes(_canonical_json(payload))
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _target_for_entry(root: Path, entry: Mapping[str, Any]) -> np.ndarray:
    path = root / str(entry["relative_path"])
    target = load_image(path, max_side=TARGET_MAX_SIDE).astype(np.float32, copy=False)
    if list(target.shape) != entry["target_shape"] or _array_sha256(target) != entry[
        "target_float32_sha256"
    ]:
        raise RuntimeError(f"target binding drift for image {entry['image_id']}")
    return target


def _field_to_device(field: GaussianField, device: str) -> GaussianField:
    def move(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.detach().to(device=device).contiguous()

    return GaussianField(
        move(field.means),
        move(field.log_scales),
        move(field.rotations),
        move(field.colors),
        move(field.opacities),
        move(field.scale_max),
        move(field.color_grads),
        move(field.background_mask),
        move(field.filter_variance),
    )


@torch.no_grad()
def _render(field: GaussianField, height: int, width: int, device: str) -> torch.Tensor:
    active = _field_to_device(field, device)
    mode = "cuda_tiled" if torch.device(device).type == "cuda" else "normalized"
    return render_field(
        active.means,
        active.conics(0.0),
        active.colors,
        active.radii(3.0, 0.0),
        height,
        width,
        512,
        mode,
        active.opacity_values(),
        scales=active.scales(),
        rotations=active.rotations,
        support_fade=False,
        sigma_cutoff=3.0,
    )


def _covariance_error(reference: GaussianField, decoded: GaussianField, height: int, width: int) -> dict:
    means = reference.means.detach().cpu().numpy().astype(np.float64)
    order, _, _, _ = experimental_codec.decoder_synchronized_morton_order(means, height, width)
    reference_entries = core.covariance_entries(
        reference.effective_log_scales().detach().cpu().numpy().astype(np.float64)[order],
        reference.rotations.detach().cpu().numpy().astype(np.float64)[order],
    )
    decoded_entries = core.covariance_entries(
        decoded.log_scales.detach().cpu().numpy().astype(np.float64),
        decoded.rotations.detach().cpu().numpy().astype(np.float64),
    )
    difference = decoded_entries - reference_entries
    squared = difference[:, 0] ** 2 + 2.0 * difference[:, 1] ** 2 + difference[:, 2] ** 2
    reference_squared = (
        reference_entries[:, 0] ** 2
        + 2.0 * reference_entries[:, 1] ** 2
        + reference_entries[:, 2] ** 2
    )
    per_row_relative = np.sqrt(squared / np.maximum(reference_squared, 1e-24))
    return {
        "relative_frobenius_rms": float(
            math.sqrt(float(squared.sum()) / max(float(reference_squared.sum()), 1e-24))
        ),
        "relative_frobenius_max_row": float(per_row_relative.max()),
    }


def _unquantized_controls(field: GaussianField, height: int, width: int, device: str) -> dict:
    log_scales = field.effective_log_scales().detach().cpu().numpy().astype(np.float64)
    rotations = field.rotations.detach().cpu().numpy().astype(np.float64)
    reference_covariance = core.covariance_entries(log_scales, rotations)
    canonical_scales, canonical_rotations = core.canonicalize_rs(log_scales, rotations)
    log_scales_roundtrip, log_rotations_roundtrip = core.log_spd_to_rs(
        core.rs_to_log_spd(log_scales, rotations)
    )
    candidate_covariance = core.covariance_entries(log_scales_roundtrip, log_rotations_roundtrip)
    difference = candidate_covariance - reference_covariance
    numerator = difference[:, 0] ** 2 + 2.0 * difference[:, 1] ** 2 + difference[:, 2] ** 2
    denominator = (
        reference_covariance[:, 0] ** 2
        + 2.0 * reference_covariance[:, 1] ** 2
        + reference_covariance[:, 2] ** 2
    )
    relative = float(np.sqrt(numerator / np.maximum(denominator, 1e-24)).max())
    if relative > 1e-12:
        raise RuntimeError(f"unquantized covariance round-trip failed: {relative}")

    def replaced(scales: np.ndarray, angles: np.ndarray) -> GaussianField:
        return GaussianField(
            field.means,
            torch.as_tensor(np.ascontiguousarray(scales), dtype=torch.float32),
            torch.as_tensor(np.ascontiguousarray(angles), dtype=torch.float32),
            field.colors,
        )

    current = _render(field, height, width, device)
    canonical = _render(replaced(canonical_scales, canonical_rotations), height, width, device)
    log_spd = _render(
        replaced(log_scales_roundtrip, log_rotations_roundtrip), height, width, device
    )
    current_canonical = float((current - canonical).abs().max().detach().cpu())
    current_log_spd = float((current - log_spd).abs().max().detach().cpu())
    if max(current_canonical, current_log_spd) > 2e-5:
        raise RuntimeError(
            "unquantized render parity failed: "
            f"canonical={current_canonical}, log_spd={current_log_spd}"
        )
    return {
        "covariance_roundtrip_max_relative": relative,
        "render_current_vs_canonical_max_abs": current_canonical,
        "render_current_vs_log_spd_max_abs": current_log_spd,
    }


def _ensure_field(
    *,
    root: Path,
    outdir: Path,
    entry: Mapping[str, Any],
    count: int,
    binding_sha256: str,
    device: str,
    existing: Mapping[str, Mapping[str, Any]],
) -> tuple[GaussianField, np.ndarray, dict[str, Any]]:
    image_id = str(entry["image_id"])
    cell_id = f"kodim{image_id}:n{count}"
    target = _target_for_entry(root, entry)
    record = existing.get(cell_id)
    if record is not None:
        if record.get("binding_sha256") != binding_sha256:
            raise RuntimeError(f"field binding drift for {cell_id}")
        field_path = outdir / str(record["field_path"])
        if _sha256_file(field_path) != record["field_file_sha256"]:
            raise RuntimeError(f"field file drift for {cell_id}")
        field = GaussianField.load(str(field_path), device="cpu")
        if _field_sha256(field) != record["field_sha256"] or field.n != count:
            raise RuntimeError(f"loaded field drift for {cell_id}")
        return field, target, dict(record)

    np.random.seed(FIT_SEED)
    torch.manual_seed(FIT_SEED)
    torch.cuda.manual_seed_all(FIT_SEED)
    init = build_field(
        target,
        InitConfig(
            strategy="quadtree_wse",
            num_gaussians=count,
            opacity_mode="none",
            seed=FIT_SEED,
        ),
        StructureTensorConfig(),
        device=device,
    )
    initial_field = _field_to_device(init, "cpu")
    initial_hash = _field_sha256(initial_field)
    initial_field_path = outdir / "initial_fields" / f"kodim{image_id}_n{count}.npz"
    initial_field_path.parent.mkdir(parents=True, exist_ok=True)
    initial_field.save(str(initial_field_path))
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = fit(init, target_tensor, _fit_config(), verbose=False)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - started
    field = _field_to_device(result["field"], "cpu")
    if field.n != count or field.opacities is not None or field.color_grads is not None:
        raise RuntimeError(f"fitted field schema drift for {cell_id}")
    controls = _unquantized_controls(field, target.shape[0], target.shape[1], device)
    field_path = outdir / "fields" / f"kodim{image_id}_n{count}.npz"
    field_path.parent.mkdir(parents=True, exist_ok=True)
    field.save(str(field_path))
    target_path = outdir / "targets" / f"kodim{image_id}.npy"
    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, target, allow_pickle=False)
    record = {
        "schema": "comp007-field-v1",
        "binding_sha256": binding_sha256,
        "cell_id": cell_id,
        "image_id": image_id,
        "n": count,
        "target_float32_sha256": entry["target_float32_sha256"],
        "target_path": target_path.relative_to(outdir).as_posix(),
        "initial_field_sha256": initial_hash,
        "initial_field_path": initial_field_path.relative_to(outdir).as_posix(),
        "initial_field_file_sha256": _sha256_file(initial_field_path),
        "field_path": field_path.relative_to(outdir).as_posix(),
        "field_file_sha256": _sha256_file(field_path),
        "field_sha256": _field_sha256(field),
        "fit_seconds": float(fit_seconds),
        "fit_terminal_psnr": float(result["psnr"]),
        "unquantized_controls": controls,
    }
    _append_jsonl(outdir / "fields.jsonl", record)
    return field, target, record


def _timed_encode(
    field: GaussianField,
    height: int,
    width: int,
    config: experimental_codec.CodecConfig,
) -> tuple[bytes, float, list[float]]:
    samples = []
    blob = b""
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter_ns()
        candidate = experimental_codec.encode(field, height, width, config)
        samples.append((time.perf_counter_ns() - started) * 1e-9)
        if blob and candidate != blob:
            raise RuntimeError("candidate encoder is not byte-deterministic")
        blob = candidate
    return blob, float(np.median(samples)), samples


def _timed_decode(blob: bytes) -> tuple[GaussianField, float, list[float]]:
    samples = []
    decoded: GaussianField | None = None
    decoded_hash: str | None = None
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter_ns()
        candidate = experimental_codec.decode(blob, device="cpu")
        samples.append((time.perf_counter_ns() - started) * 1e-9)
        candidate_hash = _field_sha256(candidate)
        if decoded_hash is not None and candidate_hash != decoded_hash:
            raise RuntimeError("candidate decoder is not deterministic")
        decoded, decoded_hash = candidate, candidate_hash
    assert decoded is not None
    return decoded, float(np.median(samples)), samples


@torch.no_grad()
def _quality(
    decoded: GaussianField, target: np.ndarray, device: str
) -> tuple[dict[str, float], str]:
    height, width = target.shape[:2]
    rendered = _render(decoded, height, width, device).clamp(0.0, 1.0)
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
    values = {
        "mse": float(metrics.mse(rendered, target_tensor).detach().cpu()),
        "psnr": metrics.psnr(rendered, target_tensor),
        "ssim": float(metrics.ssim(rendered, target_tensor).detach().cpu()),
        "ms_ssim": metrics.ms_ssim(rendered, target_tensor),
    }
    render_hash = _array_sha256(rendered.detach().cpu().contiguous().numpy())
    return values, render_hash


def _row_key(cell_id: str, chart: str, coder: str, allocation: Sequence[int]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "cell_id": cell_id,
                "chart": chart,
                "coder": coder,
                "allocation": list(allocation),
            }
        )
    )[:24]


def _run_field_candidates(
    *,
    outdir: Path,
    binding_sha256: str,
    field: GaussianField,
    target: np.ndarray,
    field_record: Mapping[str, Any],
    device: str,
    existing: Mapping[str, Mapping[str, Any]],
) -> None:
    height, width = target.shape[:2]
    cell_id = str(field_record["cell_id"])
    image_id = str(field_record["image_id"])
    count = int(field_record["n"])
    quality_cache: dict[tuple[str, tuple[int, int, int]], tuple[dict, str, dict, str]] = {}
    for chart in CHARTS:
        for allocation in allocations():
            cache_key = (chart, allocation)
            for coder in CODERS:
                key = _row_key(cell_id, chart, coder, allocation)
                old = existing.get(key)
                if old is not None:
                    if old.get("binding_sha256") != binding_sha256:
                        raise RuntimeError(f"candidate binding drift for {key}")
                    blob_path = outdir / str(old["stream_path"])
                    if _sha256_file(blob_path) != old["stream_sha256"]:
                        raise RuntimeError(f"candidate stream drift for {key}")
                    payload = blob_path.read_bytes()
                    if experimental_codec.canonical_reencode(payload) != payload:
                        raise RuntimeError(f"noncanonical saved entropy stream for {key}")
                    if experimental_codec.decoded_reencode(payload) != payload:
                        raise RuntimeError(f"decoded-field replay differs for {key}")
                    continue

                predictor_options: list[tuple[int, int, str, bytes, float, list[float]]] = []
                for predictor_index, predictor in enumerate(PREDICTORS):
                    config = experimental_codec.CodecConfig(
                        chart=chart,  # type: ignore[arg-type]
                        geometry_bits=allocation,
                        predictor=predictor,  # type: ignore[arg-type]
                        coder=coder,  # type: ignore[arg-type]
                    )
                    blob, encode_seconds, encode_samples = _timed_encode(
                        field, height, width, config
                    )
                    predictor_options.append(
                        (
                            len(blob),
                            predictor_index,
                            predictor,
                            blob,
                            encode_seconds,
                            encode_samples,
                        )
                    )
                (
                    _,
                    _,
                    predictor,
                    blob,
                    selected_encode_seconds,
                    selected_encode_samples,
                ) = min(predictor_options)
                encode_samples = [
                    float(sum(option[5][repeat] for option in predictor_options))
                    for repeat in range(TIMING_REPEATS)
                ]
                encode_seconds = float(np.median(encode_samples))
                components = experimental_codec.blob_components(blob)
                if int(components["total_bytes"]) != len(blob):
                    raise RuntimeError("complete container byte accounting failed")
                if experimental_codec.canonical_reencode(blob) != blob:
                    raise RuntimeError("canonical cold stream replay failed")
                decoded, decode_seconds, decode_samples = _timed_decode(blob)
                if experimental_codec.decoded_reencode(blob) != blob:
                    raise RuntimeError("decoded-field cold stream replay failed")

                if cache_key not in quality_cache:
                    quality, render_hash = _quality(decoded, target, device)
                    covariance = _covariance_error(field, decoded, height, width)
                    quality_cache[cache_key] = (
                        quality,
                        render_hash,
                        covariance,
                        _field_sha256(decoded),
                    )
                quality, render_hash, covariance, decoded_hash = quality_cache[cache_key]
                if _field_sha256(decoded) != decoded_hash:
                    raise RuntimeError("coder/predictor changed decoded candidate symbols")

                relative_path = (
                    Path("streams")
                    / cell_id.replace(":", "_")
                    / chart
                    / coder
                    / f"b{allocation[0]}-{allocation[1]}-{allocation[2]}_{predictor}.gfc"
                )
                blob_path = outdir / relative_path
                _atomic_bytes(blob_path, blob)
                geometry = components["streams"]["geometry"]
                row = {
                    "schema": "comp007-candidate-v1",
                    "binding_sha256": binding_sha256,
                    "row_key": key,
                    "cell_id": cell_id,
                    "image_id": image_id,
                    "n": count,
                    "chart": chart,
                    "coder": "zlib" if coder == "zlib9" else "zstd",
                    "coder_container_name": coder,
                    "predictor": predictor,
                    "predictor_option_bytes": {
                        item[2]: int(item[0]) for item in predictor_options
                    },
                    "allocation": list(allocation),
                    "total_bits": int(sum(allocation)),
                    "total_bytes": len(blob),
                    "geometry_framed_bytes": int(geometry["framed_bytes"]),
                    "geometry_attributed_bytes": int(geometry["framed_bytes"])
                    + GEOMETRY_SYNTAX_BYTES,
                    "components": components,
                    "header": experimental_codec.blob_header(blob),
                    **quality,
                    "render_sha256": render_hash,
                    "covariance_error": covariance,
                    "encode_seconds": encode_seconds,
                    "encode_seconds_samples": encode_samples,
                    "selected_encode_seconds": selected_encode_seconds,
                    "selected_encode_seconds_samples": selected_encode_samples,
                    "cold_decode_seconds": decode_seconds,
                    "cold_decode_seconds_samples": decode_samples,
                    "timing_repeats": TIMING_REPEATS,
                    "decoded_field_sha256": decoded_hash,
                    "parent_field_sha256": field_record["field_sha256"],
                    "target_float32_sha256": field_record["target_float32_sha256"],
                    "stream_path": relative_path.as_posix(),
                    "stream_sha256": _sha256_file(blob_path),
                    "cold_reencode_byte_identical": True,
                }
                _append_jsonl(outdir / "candidates.jsonl", row)


def run_split(
    outdir: Path,
    root: Path,
    split: str,
    device: str,
    development_summary: Path | None,
) -> None:
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen COMP-007 fitting protocol requires a CUDA device")
    confirmation_parent = None
    if split == "confirmation" and not _confirmation_authorized(
        development_summary, root=root, device=device
    ):
        raise RuntimeError("confirmation is sealed until a fully passing audited development run")
    if split == "confirmation":
        assert development_summary is not None
        confirmation_parent = _confirmation_parent_record(development_summary)
    torch.set_num_threads(1)
    np.random.seed(FIT_SEED)
    torch.manual_seed(FIT_SEED)
    torch.cuda.manual_seed_all(FIT_SEED)
    config = prepare_run(outdir, root, split, device, confirmation_parent)
    binding = str(config["binding_sha256"])
    manifest = {
        entry["image_id"]: entry
        for entry in config["science"]["target_manifest"]
        if entry["split"] == split
    }
    existing_fields = {row["cell_id"]: row for row in _read_jsonl(outdir / "fields.jsonl")}
    existing_rows = {row["row_key"]: row for row in _read_jsonl(outdir / "candidates.jsonl")}
    for numeric_id in _split_ids(split):
        image_id = f"{numeric_id:02d}"
        entry = manifest[image_id]
        for count in COUNTS:
            field, target, record = _ensure_field(
                root=root,
                outdir=outdir,
                entry=entry,
                count=count,
                binding_sha256=binding,
                device=device,
                existing=existing_fields,
            )
            _run_field_candidates(
                outdir=outdir,
                binding_sha256=binding,
                field=field,
                target=target,
                field_record=record,
                device=device,
                existing=existing_rows,
            )
            existing_fields[str(record["cell_id"])] = record
            existing_rows = {
                row["row_key"]: row for row in _read_jsonl(outdir / "candidates.jsonl")
            }


def _audit_archive(outdir: Path, config: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    record = config.get("executed_source_archive", {})
    archive = outdir / str(record.get("path", ""))
    observed_members: dict[str, str] = {}
    expected_members = dict(record.get("members", {}))
    if not archive.is_file():
        errors.append("executed-source archive is missing")
        return {"path": str(archive), "member_count": 0}
    archive_sha256 = _sha256_file(archive)
    if archive_sha256 != record.get("sha256"):
        errors.append("executed-source archive hash differs")
    try:
        with tarfile.open(archive, mode="r") as handle:
            file_members = [member for member in handle.getmembers() if member.isfile()]
            if {member.name for member in file_members} != set(expected_members):
                errors.append("executed-source archive member set differs")
            for member in file_members:
                source = handle.extractfile(member)
                if source is None:
                    errors.append(f"archive member cannot be read: {member.name}")
                    continue
                observed_members[member.name] = _sha256_bytes(source.read())
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"executed-source archive cannot be read: {exc}")
    for name, expected in expected_members.items():
        if observed_members.get(name) != expected:
            errors.append(f"archive member hash differs: {name}")
    science = config.get("science", {})
    for name, expected in science.get("source_manifest", {}).items():
        if observed_members.get(name) != expected:
            errors.append(f"source manifest/archive mismatch: {name}")
    task_name = "tasks/COMP-007-gauge-free-covariance-codec.md"
    if observed_members.get(task_name) != science.get("task_protocol_sha256"):
        errors.append("task protocol/archive mismatch")
    return {
        "path": archive.name,
        "sha256": archive_sha256,
        "member_count": len(observed_members),
        "members_match": observed_members == expected_members,
    }


def _samples_match_median(row: Mapping[str, Any], stem: str) -> bool:
    samples = np.asarray(row.get(f"{stem}_seconds_samples", []), dtype=np.float64)
    return bool(
        samples.shape == (TIMING_REPEATS,)
        and np.isfinite(samples).all()
        and np.all(samples > 0.0)
        and float(row.get(f"{stem}_seconds", -1.0)) == float(np.median(samples))
    )


def _confirmation_authorization_payload(
    summary: Mapping[str, Any], outdir: Path
) -> dict[str, Any]:
    return {
        "binding_sha256": summary.get("binding_sha256"),
        "config_sha256": _sha256_file(outdir / "config.json"),
        "artifact_audit_sha256": _sha256_file(outdir / "artifact_audit.json"),
        "fields_sha256": _sha256_file(outdir / "fields.jsonl"),
        "candidates_sha256": _sha256_file(outdir / "candidates.jsonl"),
        "executed_source_archive_sha256": summary.get("executed_source_archive_sha256"),
        "decision": summary.get("analysis", {}).get("decision"),
        "split": summary.get("split"),
    }


def _confirmation_parent_record(path: Path) -> dict[str, Any]:
    summary_path = path.resolve()
    outdir = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "analysis_path": str(summary_path),
        "analysis_sha256": _sha256_file(summary_path),
        "binding_sha256": summary.get("binding_sha256"),
        "config_sha256": _sha256_file(outdir / "config.json"),
        "artifact_audit_sha256": _sha256_file(outdir / "artifact_audit.json"),
        "fields_sha256": _sha256_file(outdir / "fields.jsonl"),
        "candidates_sha256": _sha256_file(outdir / "candidates.jsonl"),
        "executed_source_archive_sha256": summary.get("executed_source_archive_sha256"),
        "confirmation_authorization": summary.get("confirmation_authorization"),
    }


def audit_artifacts(outdir: Path, *, rerender: bool, device: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    config_path = outdir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    science = config.get("science", {})
    binding = str(config.get("binding_sha256", ""))
    fields = _read_jsonl(outdir / "fields.jsonl")
    rows = _read_jsonl(outdir / "candidates.jsonl")
    errors: list[str] = []

    if config.get("schema") != SCHEMA:
        errors.append("config schema differs")
    if science.get("protocol") != PROTOCOL:
        errors.append("science protocol differs")
    if binding != _binding_sha256(science, config.get("environment", {})):
        errors.append("config binding does not recompute")
    current_environment = _environment(root, device)
    if _environment_signature(config.get("environment", {})) != _environment_signature(
        current_environment
    ):
        errors.append("audit environment differs from executed environment")
    for relative, expected in science.get("source_manifest", {}).items():
        path = root / relative
        if not path.is_file() or _sha256_file(path) != expected:
            errors.append(f"current source differs from executed source: {relative}")
    task_path = root / "tasks" / "COMP-007-gauge-free-covariance-codec.md"
    if not task_path.is_file() or _sha256_file(task_path) != science.get("task_protocol_sha256"):
        errors.append("current task protocol differs from executed task")
    archive_audit = _audit_archive(outdir, config, errors)

    split = str(science.get("split", ""))
    if split == "confirmation":
        parent = science.get("confirmation_parent")
        try:
            parent_path = Path(str(parent["analysis_path"]))
            if not _confirmation_authorized(parent_path, root=root, device=device):
                errors.append("confirmation parent is no longer authorized")
            elif _confirmation_parent_record(parent_path) != parent:
                errors.append("confirmation parent binding differs")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            errors.append("confirmation parent record is invalid")
    elif science.get("confirmation_parent") is not None:
        errors.append("development config unexpectedly contains a confirmation parent")
    expected_ids = {f"{value:02d}" for value in _split_ids(split)} if split in {
        "development",
        "confirmation",
    } else set()
    expected_fields = len(expected_ids) * len(COUNTS)
    expected_rows = expected_fields * len(CHARTS) * len(CODERS) * len(allocations())
    if len(fields) != expected_fields:
        errors.append(f"expected {expected_fields} fields, found {len(fields)}")
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} candidates, found {len(rows)}")
    if len({record.get("cell_id") for record in fields}) != len(fields):
        errors.append("field cell IDs are not unique")
    if len({row.get("row_key") for row in rows}) != len(rows):
        errors.append("candidate row keys are not unique")
    if len({row.get("stream_path") for row in rows}) != len(rows):
        errors.append("candidate stream paths are not unique")

    manifest = {
        str(entry["image_id"]): entry
        for entry in science.get("target_manifest", [])
        if entry.get("split") == split
    }
    target_cache: dict[str, np.ndarray] = {}
    for image_id in expected_ids:
        entry = manifest.get(image_id)
        if entry is None:
            errors.append(f"target manifest is missing image {image_id}")
            continue
        source_path = root / str(entry["relative_path"])
        if not source_path.is_file() or _sha256_file(source_path) != entry["source_png_sha256"]:
            errors.append(f"source target hash differs: {image_id}")
            continue
        try:
            target_cache[image_id] = _target_for_entry(root, entry)
        except RuntimeError as exc:
            errors.append(str(exc))

    field_lookup: dict[str, dict[str, Any]] = {}
    loaded_fields: dict[str, GaussianField] = {}
    for record in fields:
        cell_id = str(record.get("cell_id"))
        image_id = str(record.get("image_id"))
        count = int(record.get("n", -1))
        if (
            record.get("binding_sha256") != binding
            or image_id not in expected_ids
            or count not in COUNTS
            or cell_id != f"kodim{image_id}:n{count}"
        ):
            errors.append(f"field semantic binding differs: {cell_id}")
            continue
        field_path = outdir / str(record.get("field_path"))
        if not field_path.is_file() or _sha256_file(field_path) != record.get("field_file_sha256"):
            errors.append(f"field file differs: {cell_id}")
            continue
        try:
            field = GaussianField.load(str(field_path), device="cpu")
        except Exception as exc:  # audit collects independent failures
            errors.append(f"field cannot be loaded {cell_id}: {type(exc).__name__}: {exc}")
            continue
        if field.n != count or _field_sha256(field) != record.get("field_sha256"):
            errors.append(f"field semantic hash differs: {cell_id}")
        initial_path = outdir / str(record.get("initial_field_path"))
        initial_field: GaussianField | None = None
        if (
            not initial_path.is_file()
            or _sha256_file(initial_path) != record.get("initial_field_file_sha256")
        ):
            errors.append(f"initial field file differs: {cell_id}")
        else:
            try:
                initial_field = GaussianField.load(str(initial_path), device="cpu")
                if (
                    initial_field.n != count
                    or _field_sha256(initial_field) != record.get("initial_field_sha256")
                ):
                    errors.append(f"initial field semantic hash differs: {cell_id}")
            except Exception as exc:
                errors.append(
                    f"initial field cannot be loaded {cell_id}: {type(exc).__name__}: {exc}"
                )
        entry = manifest.get(image_id)
        target_path = outdir / str(record.get("target_path"))
        if not target_path.is_file():
            errors.append(f"saved target is missing: {cell_id}")
        else:
            saved_target = np.load(target_path, allow_pickle=False)
            expected_target_hash = None if entry is None else entry["target_float32_sha256"]
            if (
                _array_sha256(saved_target) != expected_target_hash
                or record.get("target_float32_sha256") != expected_target_hash
            ):
                errors.append(f"saved target semantic hash differs: {cell_id}")
        if rerender and image_id in target_cache:
            try:
                np.random.seed(FIT_SEED)
                torch.manual_seed(FIT_SEED)
                torch.cuda.manual_seed_all(FIT_SEED)
                rebuilt_initial = build_field(
                    target_cache[image_id],
                    InitConfig(
                        strategy="quadtree_wse",
                        num_gaussians=count,
                        opacity_mode="none",
                        seed=FIT_SEED,
                    ),
                    StructureTensorConfig(),
                    device=device,
                )
                if _field_sha256(rebuilt_initial) != record.get("initial_field_sha256"):
                    errors.append(f"initial field provenance replay differs: {cell_id}")
                controls = _unquantized_controls(
                    field,
                    target_cache[image_id].shape[0],
                    target_cache[image_id].shape[1],
                    device,
                )
                if (
                    controls["covariance_roundtrip_max_relative"] > 1e-12
                    or max(
                        controls["render_current_vs_canonical_max_abs"],
                        controls["render_current_vs_log_spd_max_abs"],
                    )
                    > 2e-5
                ):
                    errors.append(f"unquantized controls fail on replay: {cell_id}")
            except Exception as exc:
                errors.append(f"unquantized replay failed {cell_id}: {type(exc).__name__}: {exc}")
        field_lookup[cell_id] = record
        loaded_fields[cell_id] = field

    expected_row_keys = {
        _row_key(f"kodim{image_id}:n{count}", chart, coder, allocation)
        for image_id in expected_ids
        for count in COUNTS
        for chart in CHARTS
        for coder in CODERS
        for allocation in allocations()
    }
    observed_row_keys = {str(row.get("row_key")) for row in rows}
    if observed_row_keys != expected_row_keys:
        errors.append("candidate row-key grid differs from the frozen matrix")

    quality_cache: dict[tuple[str, str], tuple[dict[str, float], str]] = {}
    render_hash_matches = 0
    render_hash_differences = 0
    for index, row in enumerate(rows):
        prefix = f"row {index}/{row.get('row_key')}"
        cell_id = str(row.get("cell_id"))
        image_id = str(row.get("image_id"))
        count = int(row.get("n", -1))
        chart = str(row.get("chart"))
        coder_name = str(row.get("coder_container_name"))
        coder_label = "zlib" if coder_name == "zlib9" else "zstd" if coder_name == "zstd9" else ""
        try:
            allocation = tuple(int(value) for value in row.get("allocation", []))
        except (TypeError, ValueError):
            allocation = ()
        expected_key = (
            _row_key(cell_id, chart, coder_name, allocation)
            if len(allocation) == 3 and coder_name in CODERS
            else ""
        )
        if (
            row.get("binding_sha256") != binding
            or cell_id != f"kodim{image_id}:n{count}"
            or image_id not in expected_ids
            or count not in COUNTS
            or chart not in CHARTS
            or coder_name not in CODERS
            or row.get("coder") != coder_label
            or allocation not in allocations()
            or int(row.get("total_bits", -1)) != sum(allocation)
            or row.get("row_key") != expected_key
        ):
            errors.append(f"{prefix}: row semantics differ")
            continue
        stream_path = outdir / str(row.get("stream_path"))
        if not stream_path.is_file() or _sha256_file(stream_path) != row.get("stream_sha256"):
            errors.append(f"{prefix}: stream hash differs")
            continue
        blob = stream_path.read_bytes()
        try:
            components = experimental_codec.blob_components(blob)
            header = experimental_codec.blob_header(blob)
            if experimental_codec.canonical_reencode(blob) != blob:
                errors.append(f"{prefix}: entropy replay differs")
            if experimental_codec.decoded_reencode(blob) != blob:
                errors.append(f"{prefix}: decoded-field replay differs")
            if components != row.get("components") or len(blob) != row.get("total_bytes"):
                errors.append(f"{prefix}: component accounting differs")
            geometry_framed = int(components["streams"]["geometry"]["framed_bytes"])
            if (
                row.get("geometry_framed_bytes") != geometry_framed
                or row.get("geometry_attributed_bytes")
                != geometry_framed + GEOMETRY_SYNTAX_BYTES
            ):
                errors.append(f"{prefix}: covariance-byte attribution differs")
            if header != row.get("header") or any(
                (
                    header["chart"] != chart,
                    header["coder"] != coder_name,
                    tuple(header["geometry_bits"]) != allocation,
                    header["predictor"] != row.get("predictor"),
                    int(header["n"]) != count,
                )
            ):
                errors.append(f"{prefix}: header semantics differ")
            if (
                not _samples_match_median(row, "encode")
                or not _samples_match_median(row, "selected_encode")
                or not _samples_match_median(row, "cold_decode")
            ):
                errors.append(f"{prefix}: raw timing ledger differs")

            parent = loaded_fields[cell_id]
            options: dict[str, bytes] = {}
            for predictor in PREDICTORS:
                codec_config = experimental_codec.CodecConfig(
                    chart=chart,  # type: ignore[arg-type]
                    geometry_bits=allocation,  # type: ignore[arg-type]
                    predictor=predictor,  # type: ignore[arg-type]
                    coder=coder_name,  # type: ignore[arg-type]
                )
                options[predictor] = experimental_codec.encode(
                    parent, int(header["height"]), int(header["width"]), codec_config
                )
            option_bytes = {name: len(value) for name, value in options.items()}
            selected = min(PREDICTORS, key=lambda name: (option_bytes[name], PREDICTORS.index(name)))
            if row.get("predictor_option_bytes") != option_bytes or row.get("predictor") != selected:
                errors.append(f"{prefix}: predictor selection differs")
            if options[selected] != blob:
                errors.append(f"{prefix}: parent-field re-encode differs")

            decoded = experimental_codec.decode(blob)
            decoded_hash = _field_sha256(decoded)
            if decoded_hash != row.get("decoded_field_sha256"):
                errors.append(f"{prefix}: decoded field differs")
            if row.get("parent_field_sha256") != field_lookup[cell_id].get("field_sha256"):
                errors.append(f"{prefix}: parent field link differs")
            if row.get("target_float32_sha256") != field_lookup[cell_id].get(
                "target_float32_sha256"
            ):
                errors.append(f"{prefix}: target link differs")
            covariance = _covariance_error(
                parent, decoded, int(header["height"]), int(header["width"])
            )
            if any(
                abs(float(covariance[name]) - float(row.get("covariance_error", {}).get(name, math.inf)))
                > 1e-12
                for name in covariance
            ):
                errors.append(f"{prefix}: covariance diagnostic differs")

            if rerender:
                cache_key = (image_id, decoded_hash)
                if cache_key not in quality_cache:
                    quality_cache[cache_key] = _quality(decoded, target_cache[image_id], device)
                quality, render_hash = quality_cache[cache_key]
                for name, tolerance in QUALITY_REPLAY_TOLERANCES.items():
                    if abs(float(quality[name]) - float(row.get(name, math.inf))) > tolerance:
                        errors.append(f"{prefix}: {name} replay differs")
                if render_hash == row.get("render_sha256"):
                    render_hash_matches += 1
                else:
                    render_hash_differences += 1
        except Exception as exc:  # audit must collect rather than hide later errors
            errors.append(f"{prefix}: {type(exc).__name__}: {exc}")

    result = {
        "schema": "comp007-audit-v2",
        "binding_sha256": binding,
        "split": split,
        "field_count": len(fields),
        "candidate_count": len(rows),
        "rerendered": bool(rerender),
        "quality_replay_tolerances": QUALITY_REPLAY_TOLERANCES,
        "render_hash_matches": render_hash_matches,
        "render_hash_differences_diagnostic_only": render_hash_differences,
        "errors": errors,
        "passes": not errors,
        "decision_ready": bool(rerender) and not errors,
        "fields_sha256": _sha256_file(outdir / "fields.jsonl"),
        "candidates_sha256": _sha256_file(outdir / "candidates.jsonl"),
        "config_sha256": _sha256_file(config_path),
        "environment_signature": _environment_signature(current_environment),
        "executed_source_archive": archive_audit,
    }
    _atomic_json(outdir / "artifact_audit.json", result)
    return result


def analyze_run(outdir: Path, split: str, device: str) -> dict[str, Any]:
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    configured_split = config.get("science", {}).get("split")
    if configured_split != split:
        raise RuntimeError(
            f"analysis split {split!r} differs from bound config split {configured_split!r}"
        )
    audit = audit_artifacts(outdir, rerender=True, device=device)
    rows = _read_jsonl(outdir / "candidates.jsonl")
    reduced = analysis.analyze_candidates(
        rows,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )
    if split == "confirmation" and reduced.get("status") == "available":
        required = {
            name: gate
            for name, gate in reduced.get("gates", {}).items()
            if not name.startswith("gate_8_")
        }
        confirmation_pass = bool(required) and all(gate["passes"] for gate in required.values())
        reduced["confirmation_decision"] = "pass" if confirmation_pass else "fail"
    result = {
        "schema": "comp007-analysis-v2",
        "split": split,
        "binding_sha256": config["binding_sha256"],
        "analysis": reduced,
        "checks": {
            "artifact_audit_passed": bool(audit["passes"] and audit["decision_ready"]),
        },
        "artifact_audit": audit,
        "config_sha256": _sha256_file(outdir / "config.json"),
        "artifact_audit_sha256": _sha256_file(outdir / "artifact_audit.json"),
        "executed_source_archive_sha256": config["executed_source_archive"]["sha256"],
    }
    if not audit["passes"] or not audit["decision_ready"]:
        result["analysis"]["decision"] = "unavailable"
        result["analysis"]["confirmation_authorized"] = False
    if (
        split == "development"
        and result["analysis"].get("decision") == "pass_authorize_confirmation"
        and result["checks"]["artifact_audit_passed"]
    ):
        payload = _confirmation_authorization_payload(result, outdir)
        result["confirmation_authorization"] = {
            "schema": "comp007-confirmation-authorization-v1",
            "payload": payload,
            "sha256": _sha256_bytes(_canonical_json(payload)),
        }
    _atomic_json(outdir / "analysis.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "analyze", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument("--outdir", type=Path, required=True)
        child.add_argument("--device", default="cuda")
        child.add_argument("--split", choices=("development", "confirmation"), default="development")
        if command == "run":
            child.add_argument("--development-summary", type=Path)
        if command == "audit":
            child.add_argument("--rerender", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    outdir = arguments.outdir.resolve()
    if arguments.command == "run":
        run_split(
            outdir,
            root,
            arguments.split,
            arguments.device,
            arguments.development_summary,
        )
    elif arguments.command == "analyze":
        result = analyze_run(outdir, arguments.split, arguments.device)
        compact = {
            "status": result["analysis"].get("status"),
            "decision": result["analysis"].get("decision"),
            "errors": result["analysis"].get("errors"),
            "per_coder": result["analysis"].get("per_coder"),
            "gates": result["analysis"].get("gates"),
        }
        for value in compact.get("per_coder", {}).values():
            value.get("paired_image_bootstrap", {}).pop("samples", None)
        print(json.dumps(compact, indent=2, sort_keys=True))
    else:
        result = audit_artifacts(outdir, rerender=arguments.rerender, device=arguments.device)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
