"""Parent-bound BENCH-013 Stage-0 downstream assay.

This runner consumes the immutable, audited early-screen artifact rather than regenerating its
fields.  It evaluates the complete frozen forward and permutation matrices and, when its selected
cell passes forward gates, the preregistered gradient audit.  Production renderer, fitter, codec,
and defaults remain untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping

import numpy as np
import torch

from benchmarks import local_linear_reproducing_assay as early
from benchmarks import local_linear_reproducing_downstream_core as core


PROTOCOL = "bench013-local-linear-reproducing-stage0-full-v3"
SCHEMA = "bench013-stage0-full-config-v3"
PARENT_BINDING = "394323650251ba129eab8b5e03ac2139ab2e348e1571fc4863646b3c777c13ab"
PARENT_TASK_SHA256 = "cc5506f1536771d77a8ade9e8c076520d772d984287124db70d0c5c3ae8aeb11"
PREVIOUS_CLARIFIED_TASK_SHA256 = (
    "9d74511c95786592b5071b5f9b61434983b95bb0a28a97377903b45327cebb09"
)
TASK_SHA256 = "b40b9075262c8c2dc07212490a1178b0168b6da9e433cb4ca23a10957bb1d0ad"
PARENT_HASHES = {
    "analysis.json": "5207065843161bccbf4512b6a385848717259a44151e83a180830cea53ae149d",
    "replay.json": "00708bc516c7f6bdf0d1062809ff5f9d2d19e4f490827a55da69b1a276a178c8",
    "completion.json": "70ee49fbe38a1c2700aabca8c60d26e68b5553a8a47d0146629b4749fb492bd8",
    "artifact_manifest.json": "f3c5363f085566e129b595421e93eba8e23c8d95ea2fa7edfd101a3da09f26a0",
    "executed_sources.tar": "861cfcf6cacdf263afe6cd362a29843b1e3b913a07eb441addb1e4e9f273f5aa",
}
HEIGHT = early.HEIGHT
WIDTH = early.WIDTH
PIXELS = HEIGHT * WIDTH
FORWARD_CELLS = early.EXPECTED_CELLS
LOGICAL_FIELDS = early.EXPECTED_FIELDS
PERMUTATION_SEED = 20_260_716_013
GRADIENT_SEED = 20_260_716_014
GRADIENT_STEP = 2.0**-12
PERMUTATION_ORDERS = ("identity", "reverse", "pcg64_0", "pcg64_1")
SHORT_CIRCUIT = early.SHORT_CIRCUIT
REPLAY_FLOAT32_BACKWARD_ERROR_MULTIPLIER = 128.0
RECOVERY_PREDECESSOR = {
    "path": "results/bench013_local_linear_stage0_full_2026-07-16",
    "binding_sha256": "6817380d2599ff2d5d8a0a22d244b36fb5e9cb76ae4b220777d40d87b6a5e7d4",
    "inventory_sha256": "111c0680df375e753ef6578847e1b80769e4e5f1a9bed09ebb9e5c9f05231ab9",
    "inventory_algorithm": (
        "python_path_component_order_relative_posix_tab_size_tab_sha256_lf_no_trailing_lf"
    ),
}
RECOVERY_EXPECTED_NPZ_COUNT = 344
RECOVERY_JSON_PATHS = ("controls.json", "gradient.json")
RECOVERY_JSONL_PATHS = (
    "input_fields.jsonl",
    "forward_cells.jsonl",
    "effective_fields.jsonl",
    "permutations.jsonl",
    "gradient.jsonl",
)
RECOVERY_CONFIG_DROP_KEYS = frozenset(
    {
        "schema",
        "protocol",
        "clarified_task_sha256",
        "source_manifest",
        "binding_sha256",
        "infrastructure_recovery",
        "replay_validation",
    }
)

GATES = {
    "float64_constant_max_abs": 1e-12,
    "float64_affine_max_abs": 1e-10,
    "float32_affine_max_abs": 2e-5,
    "float32_same_state_max_abs": 2e-5,
    "float32_same_state_rmse": 2e-6,
    "float64_partition_and_moment": 1e-10,
    "float32_partition_and_moment": 2e-5,
    "a1_p99": 1.5,
    "a1_max": 4.0,
    "step_excursion": 2.0 / 255.0,
    "permutation_float64_max_abs": 1e-12,
    "permutation_float32_max_abs": 2e-5,
    "gradient_fd_abs": 1e-8,
    "gradient_fd_rel": 1e-4,
    "gradient_cross_abs": 2e-5,
    "gradient_cross_rel": 2e-3,
    "gradient_full_abs": 2e-5,
    "gradient_full_rel": 2e-3,
}

SOURCE_PATHS = (
    "benchmarks/__init__.py",
    "benchmarks/local_linear_reproducing_assay.py",
    "benchmarks/local_linear_reproducing_core.py",
    "benchmarks/local_linear_reproducing_downstream_core.py",
    "benchmarks/local_linear_reproducing_full_assay.py",
    "src/structsplat/__init__.py",
    "src/structsplat/config.py",
    "src/structsplat/density.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/init.py",
    "src/structsplat/render.py",
    "src/structsplat/sampling.py",
    "src/structsplat/structural_controls.py",
    "src/structsplat/structure_tensor.py",
    "tasks/BENCH-013-local-linear-reproducing-compositor.md",
    "pyproject.toml",
)
ARCHIVE_PATHS = SOURCE_PATHS + (
    "tests/test_local_linear_reproducing_assay.py",
    "tests/test_local_linear_reproducing_core.py",
    "tests/test_local_linear_reproducing_downstream_core.py",
    "tests/test_local_linear_reproducing_full_assay.py",
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


def _directory_inventory(directory: Path) -> dict[str, Any]:
    """Hash regular files in the predecessor's frozen Python-Path component order."""

    directory = directory.resolve()
    if not directory.is_dir():
        raise RuntimeError(f"recovery predecessor is not a directory: {directory}")
    records: list[str] = []
    total_bytes = 0
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        records.append(
            f"{path.relative_to(directory).as_posix()}\t{size}\t{_sha256_file(path)}"
        )
    payload = "\n".join(records).encode("utf-8")
    return {
        "algorithm": RECOVERY_PREDECESSOR["inventory_algorithm"],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "payload_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def verify_recovery_predecessor(root: Path) -> dict[str, Any]:
    """Verify the immutable lifecycle-invalid v2 attempt before creating a v3 outdir."""

    predecessor = (root / str(RECOVERY_PREDECESSOR["path"])).resolve()
    inventory = _directory_inventory(predecessor)
    config_path = predecessor / "config.json"
    environment_path = predecessor / "environment.json"
    replay_path = predecessor / "replay.json"
    parent_path = predecessor / "parent_binding.json"
    if not (
        config_path.is_file()
        and environment_path.is_file()
        and replay_path.is_file()
        and parent_path.is_file()
    ):
        raise RuntimeError("recovery predecessor lacks required lifecycle records")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    parent_binding = json.loads(parent_path.read_text(encoding="utf-8"))
    predecessor_science = dict(config)
    predecessor_science.pop("binding_sha256", None)
    recomputed_binding = _binding(predecessor_science, environment)
    replay_checks = replay.get("checks", {})
    lifecycle = {
        "config_schema": config.get("schema"),
        "config_protocol": config.get("protocol"),
        "config_binding_sha256": config.get("binding_sha256"),
        "config_recomputed_binding_sha256": recomputed_binding,
        "config_task_sha256": config.get("clarified_task_sha256"),
        "replay_schema": replay.get("schema"),
        "replay_binding_sha256": replay.get("binding_sha256"),
        "replay_artifact_audit_passed": replay.get("artifact_audit_passed"),
        "replay_check_count": len(replay_checks),
        "replay_pass_count": sum(value is True for value in replay_checks.values()),
        "replay_failed_checks": sorted(
            key for key, value in replay_checks.items() if value is not True
        ),
        "parent_binding_sha256": parent_binding.get("binding_sha256"),
        "analysis_absent": not (predecessor / "analysis.json").exists(),
        "completion_absent": not (predecessor / "completion.json").exists(),
        "artifact_manifest_absent": not (predecessor / "artifact_manifest.json").exists(),
    }
    valid = (
        inventory["sha256"] == RECOVERY_PREDECESSOR["inventory_sha256"]
        and lifecycle
        == {
            "config_schema": "bench013-stage0-full-config-v2",
            "config_protocol": "bench013-local-linear-reproducing-stage0-full-v2",
            "config_binding_sha256": RECOVERY_PREDECESSOR["binding_sha256"],
            "config_recomputed_binding_sha256": RECOVERY_PREDECESSOR[
                "binding_sha256"
            ],
            "config_task_sha256": PREVIOUS_CLARIFIED_TASK_SHA256,
            "replay_schema": "bench013-stage0-full-replay-v2",
            "replay_binding_sha256": RECOVERY_PREDECESSOR["binding_sha256"],
            "replay_artifact_audit_passed": False,
            "replay_check_count": 14,
            "replay_pass_count": 13,
            "replay_failed_checks": ["forward_raw_replay"],
            "parent_binding_sha256": PARENT_BINDING,
            "analysis_absent": True,
            "completion_absent": True,
            "artifact_manifest_absent": True,
        }
    )
    record = {
        "schema": "bench013-stage0-recovery-predecessor-verification-v1",
        "path": str(RECOVERY_PREDECESSOR["path"]),
        "expected_binding_sha256": RECOVERY_PREDECESSOR["binding_sha256"],
        "expected_inventory_sha256": RECOVERY_PREDECESSOR["inventory_sha256"],
        "inventory": inventory,
        "lifecycle": lifecycle,
        "verified": valid,
    }
    if not valid:
        raise RuntimeError("recovery predecessor inventory/lifecycle verification failed")
    return record


def _recursive_npz_manifest(directory: Path) -> dict[str, Any]:
    records = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(directory.rglob("*.npz"))
        if path.is_file()
    ]
    return {
        "file_count": len(records),
        "record_sha256": _sha256_bytes(_canonical_json(records)),
        "files": records,
    }


def _binding_normalized_record(
    path: Path, binding: str, *, jsonl: bool
) -> dict[str, Any]:
    try:
        if jsonl:
            values: list[Any] = _read_jsonl(path)
        else:
            values = [json.loads(path.read_text(encoding="utf-8"))]
        normalized: list[Any] = []
        bindings_valid = True
        for value in values:
            if not isinstance(value, Mapping):
                bindings_valid = False
                normalized.append(value)
                continue
            item = dict(value)
            bindings_valid &= item.pop("binding_sha256", None) == binding
            normalized.append(item)
        payload_value: Any = normalized if jsonl else normalized[0]
        payload = _canonical_json(payload_value)
        return {
            "path": path.name,
            "row_count": len(values),
            "binding_fields_valid": bindings_valid,
            "payload_bytes": len(payload),
            "normalized_sha256": _sha256_bytes(payload),
        }
    except Exception as error:  # pragma: no cover - corrupt artifacts are diagnostic-total
        return {
            "path": path.name,
            "row_count": 0,
            "binding_fields_valid": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _normalized_science_config_record(path: Path, binding: str) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        binding_valid = config.get("binding_sha256") == binding
        normalized = {
            key: value
            for key, value in config.items()
            if key not in RECOVERY_CONFIG_DROP_KEYS
        }
        payload = _canonical_json(normalized)
        return {
            "binding_field_valid": binding_valid,
            "dropped_keys": sorted(RECOVERY_CONFIG_DROP_KEYS),
            "payload_bytes": len(payload),
            "normalized_sha256": _sha256_bytes(payload),
        }
    except Exception as error:  # pragma: no cover - corrupt artifacts are diagnostic-total
        return {
            "binding_field_valid": False,
            "dropped_keys": sorted(RECOVERY_CONFIG_DROP_KEYS),
            "error": f"{type(error).__name__}: {error}",
        }


def _recovery_equivalence(
    predecessor: Path,
    current: Path,
    current_binding: str,
    *,
    expected_npz_count: int = RECOVERY_EXPECTED_NPZ_COUNT,
) -> dict[str, Any]:
    """Compare v3 science payloads to v2 after only authorized binding/config normalization."""

    predecessor_binding = str(RECOVERY_PREDECESSOR["binding_sha256"])
    predecessor_npz = _recursive_npz_manifest(predecessor)
    current_npz = _recursive_npz_manifest(current)
    npz_equal = (
        predecessor_npz["file_count"] == expected_npz_count
        and current_npz["file_count"] == expected_npz_count
        and predecessor_npz["files"] == current_npz["files"]
    )

    ledger_records: dict[str, Any] = {}
    for relative in RECOVERY_JSON_PATHS + RECOVERY_JSONL_PATHS:
        predecessor_record = _binding_normalized_record(
            predecessor / relative,
            predecessor_binding,
            jsonl=relative.endswith(".jsonl"),
        )
        current_record = _binding_normalized_record(
            current / relative,
            current_binding,
            jsonl=relative.endswith(".jsonl"),
        )
        equivalent = (
            predecessor_record.get("binding_fields_valid") is True
            and current_record.get("binding_fields_valid") is True
            and predecessor_record.get("row_count") == current_record.get("row_count")
            and predecessor_record.get("normalized_sha256")
            == current_record.get("normalized_sha256")
        )
        ledger_records[relative] = {
            "predecessor": predecessor_record,
            "current": current_record,
            "equivalent": equivalent,
        }

    predecessor_parent_hash = _sha256_file(predecessor / "parent_binding.json")
    current_parent_hash = _sha256_file(current / "parent_binding.json")
    parent_equal = predecessor_parent_hash == current_parent_hash
    predecessor_config = _normalized_science_config_record(
        predecessor / "config.json", predecessor_binding
    )
    current_config = _normalized_science_config_record(
        current / "config.json", current_binding
    )
    config_equal = (
        predecessor_config.get("binding_field_valid") is True
        and current_config.get("binding_field_valid") is True
        and predecessor_config.get("normalized_sha256")
        == current_config.get("normalized_sha256")
    )
    equivalence_pass = (
        npz_equal
        and all(record["equivalent"] for record in ledger_records.values())
        and parent_equal
        and config_equal
    )
    return {
        "schema": "bench013-stage0-recovery-equivalence-v1",
        "predecessor": {
            "path": str(RECOVERY_PREDECESSOR["path"]),
            "binding_sha256": predecessor_binding,
            "inventory_sha256": RECOVERY_PREDECESSOR["inventory_sha256"],
        },
        "current_binding_sha256": current_binding,
        "npz": {
            "expected_file_count": expected_npz_count,
            "predecessor": predecessor_npz,
            "current": current_npz,
            "equivalent": npz_equal,
        },
        "binding_normalized_ledgers": ledger_records,
        "parent_binding": {
            "predecessor_sha256": predecessor_parent_hash,
            "current_sha256": current_parent_hash,
            "equivalent": parent_equal,
        },
        "normalized_science_config": {
            "predecessor": predecessor_config,
            "current": current_config,
            "equivalent": config_equal,
        },
        "equivalence_pass": equivalence_pass,
    }


def _array_record(value: np.ndarray) -> dict[str, Any]:
    return early._array_record(np.ascontiguousarray(value))  # noqa: SLF001


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        return {"nonfinite": "positive_infinity" if value > 0 else "negative_infinity"}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_number(value: Any) -> float:
    if isinstance(value, Mapping) and "nonfinite" in value:
        state = value["nonfinite"]
        if state == "nan":
            return float("nan")
        return float("inf") if state == "positive_infinity" else float("-inf")
    return float(value)


def _atomic_json(path: Path, value: Any) -> None:
    early._atomic_json(path, _json_safe(value))  # noqa: SLF001


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    early._deterministic_npz(path, arrays)  # noqa: SLF001


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(_canonical_json(_json_safe(dict(value))) + b"\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _source_manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"missing BENCH-013 full-assay source: {relative}")
        result[relative] = _sha256_file(path)
    return result


def _environment(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        git_record: dict[str, Any] = {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "dirty": bool(status),
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
        }
    except (OSError, subprocess.CalledProcessError):
        git_record = {"commit": None, "branch": None, "dirty": None}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "git": git_record,
    }


def _binding(science: Mapping[str, Any], environment: Mapping[str, Any]) -> str:
    stable_keys = (
        "python",
        "platform",
        "numpy",
        "torch",
        "device",
        "torch_num_threads",
        "torch_num_interop_threads",
        "torch_deterministic_algorithms",
        "omp_num_threads",
        "mkl_num_threads",
    )
    stable_environment = {key: environment[key] for key in stable_keys}
    return _sha256_bytes(
        _canonical_json({"science": dict(science), "environment": stable_environment})
    )


def verify_parent(parent: Path) -> dict[str, Any]:
    """Verify the immutable early artifact and every file in its own manifest."""

    for name, expected in PARENT_HASHES.items():
        actual = _sha256_file(parent / name)
        if actual != expected:
            raise RuntimeError(f"parent {name} hash mismatch: {actual} != {expected}")
    manifest = json.loads((parent / "artifact_manifest.json").read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = parent / relative
        if not path.is_file() or path.stat().st_size != int(record["size_bytes"]):
            raise RuntimeError(f"parent artifact file missing/size mismatch: {relative}")
        if _sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"parent artifact file hash mismatch: {relative}")
    analysis = json.loads((parent / "analysis.json").read_text(encoding="utf-8"))
    replay = json.loads((parent / "replay.json").read_text(encoding="utf-8"))
    completion = json.loads((parent / "completion.json").read_text(encoding="utf-8"))
    config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    if analysis["binding_sha256"] != PARENT_BINDING or config["binding_sha256"] != PARENT_BINDING:
        raise RuntimeError("parent binding mismatch")
    if not analysis["complete"] or analysis["decision"] != "continue":
        raise RuntimeError("parent did not authorize the downstream phase")
    if not replay["artifact_audit_passed"] or not completion["complete"]:
        raise RuntimeError("parent replay/completion is not valid")
    if (
        config["source_manifest"]["tasks/BENCH-013-local-linear-reproducing-compositor.md"]
        != PARENT_TASK_SHA256
    ):
        raise RuntimeError("parent task hash mismatch")
    return {
        "binding_sha256": PARENT_BINDING,
        "parent_task_sha256": PARENT_TASK_SHA256,
        "downstream_task_sha256": PREVIOUS_CLARIFIED_TASK_SHA256,
        "critical_hashes": dict(PARENT_HASHES),
        "manifest_file_count": int(manifest["file_count"]),
        "manifest_payload_sha256": _sha256_bytes(_canonical_json(manifest)),
    }


def _load_field(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(dtype=dtype)


def _evaluate(
    arrays: Mapping[str, np.ndarray], target: str, permutation: np.ndarray
) -> tuple[core.ReferenceResult, core.ReferenceResult, core.CandidateResult, np.ndarray, np.ndarray]:
    permutation = np.ascontiguousarray(permutation, dtype=np.int64)
    means32 = _tensor(arrays["means"][permutation], torch.float32)
    logs32 = _tensor(arrays["log_scales"][permutation], torch.float32)
    rotations32 = _tensor(arrays["rotations"][permutation], torch.float32)
    conics32 = _tensor(arrays["conics_float32"][permutation], torch.float32)
    radii = torch.from_numpy(np.ascontiguousarray(arrays["radii"][permutation])).to(
        dtype=torch.int64
    )
    exact_colors = early.target_values(
        target, arrays["means"], dtype=np.float64
    )[permutation]
    colors32 = np.ascontiguousarray(exact_colors.astype(np.float32))
    means64 = means32.to(torch.float64)
    mathematical_conics = core.conics_from_parameters(
        logs32.to(torch.float64), rotations32.to(torch.float64)
    )
    mathematical = core.reference_qr(
        means64,
        mathematical_conics,
        _tensor(exact_colors, torch.float64),
        radii,
        HEIGHT,
        WIDTH,
        retain_effective_weights=False,
    )
    same_state = core.reference_qr(
        means64,
        conics32.to(torch.float64),
        _tensor(colors32, torch.float64),
        radii,
        HEIGHT,
        WIDTH,
        retain_effective_weights=False,
    )
    candidate = core.candidate_cholesky(
        means32,
        conics32,
        _tensor(colors32, torch.float32),
        radii,
        HEIGHT,
        WIDTH,
    )
    return mathematical, same_state, candidate, exact_colors, colors32


def _max_abs(value: torch.Tensor) -> float:
    return float(value.abs().max())


def _worst_pixel(values: np.ndarray, *, maximum: bool) -> dict[str, Any]:
    flat = np.asarray(values).reshape(-1)
    index = int(np.argmax(flat) if maximum else np.argmin(flat))
    value = float(flat[index])
    return {
        "x": index % WIDTH,
        "y": index // WIDTH,
        "value": value,
        "finite": math.isfinite(value),
    }


def _pixelwise_max_abs(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return np.max(np.abs(array.reshape(PIXELS, -1)), axis=1)


def _first_failure(mask: np.ndarray) -> dict[str, Any]:
    failures = np.flatnonzero(~np.asarray(mask, dtype=bool).reshape(-1))
    if failures.size == 0:
        return {"failure_count": 0, "first": None}
    index = int(failures[0])
    return {
        "failure_count": int(failures.size),
        "first": {"x": index % WIDTH, "y": index // WIDTH},
    }


def _worst_pixels(raw: Mapping[str, np.ndarray], target: str) -> dict[str, Any]:
    margin = raw["mathematical_singular_values"][:, 2] - raw[
        "mathematical_rank_thresholds"
    ]
    result = {
        "math64_minimum_mass": _worst_pixel(raw["mathematical_mass"], maximum=False),
        "math64_minimum_active": _worst_pixel(
            raw["mathematical_active_counts"], maximum=False
        ),
        "math64_minimum_rank_margin": _worst_pixel(margin, maximum=False),
        "math64_maximum_condition": _worst_pixel(
            raw["mathematical_condition_numbers"], maximum=True
        ),
        "same64_maximum_condition": _worst_pixel(
            raw["same_state_condition_numbers"], maximum=True
        ),
        "cand32_minimum_mass": _worst_pixel(raw["candidate_mass"], maximum=False),
        "cand32_minimum_active": _worst_pixel(
            raw["candidate_active_counts"], maximum=False
        ),
        "cand32_maximum_condition_diagnostic": _worst_pixel(
            raw["candidate_condition_numbers"], maximum=True
        ),
        "math64_maximum_a1": _worst_pixel(raw["mathematical_a1"], maximum=True),
        "cand32_maximum_a1": _worst_pixel(raw["candidate_a1"], maximum=True),
        "math64_maximum_partition_residual": _worst_pixel(
            np.abs(raw["mathematical_partition"] - 1.0), maximum=True
        ),
        "cand32_maximum_partition_residual": _worst_pixel(
            np.abs(raw["candidate_partition"] - np.float32(1.0)), maximum=True
        ),
        "math64_maximum_first_moment_residual": _worst_pixel(
            _pixelwise_max_abs(raw["mathematical_first_moment"]), maximum=True
        ),
        "cand32_maximum_first_moment_residual": _worst_pixel(
            _pixelwise_max_abs(raw["candidate_first_moment"]), maximum=True
        ),
        "same_state_parity_error": _worst_pixel(
            _pixelwise_max_abs(
                raw["candidate_output"].astype(np.float64) - raw["same_state_output"]
            ),
            maximum=True,
        ),
        "cand32_solve": _first_failure(raw["candidate_solve_pass"]),
    }
    if target in ("constant", "affine_xy"):
        truth = early.target_image(target, dtype=np.float64)
        result["math64_reproduction_error"] = _worst_pixel(
            _pixelwise_max_abs(raw["mathematical_output"] - truth), maximum=True
        )
        if target == "affine_xy":
            result["cand32_affine_error"] = _worst_pixel(
                _pixelwise_max_abs(raw["candidate_output"].astype(np.float64) - truth),
                maximum=True,
            )
    return result


def _range_excursion_diagnostic(
    candidate_output: np.ndarray, exact_colors: np.ndarray
) -> dict[str, Any]:
    output = np.asarray(candidate_output, dtype=np.float64)
    low = np.min(exact_colors, axis=0)
    high = np.max(exact_colors, axis=0)
    below = np.maximum(low[None, None, :] - output, 0.0)
    above = np.maximum(output - high[None, None, :], 0.0)
    stacked = np.stack((below, above), axis=0)
    flat_index = int(np.argmax(stacked))
    side, y, x, channel = np.unravel_index(flat_index, stacked.shape)
    return {
        "sample_min_per_channel": [float(value) for value in low],
        "sample_max_per_channel": [float(value) for value in high],
        "maximum": float(stacked[side, y, x, channel]),
        "worst": {
            "x": int(x),
            "y": int(y),
            "channel": int(channel),
            "side": "below" if int(side) == 0 else "above",
        },
    }


def downstream_controls() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the exact 9x9 positive and negative plumbing controls through the new core."""

    def geometry(means: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(means.shape[0])
        return (
            _tensor(means, torch.float64),
            torch.tensor((1.0 / 64.0, 0.0, 1.0 / 64.0), dtype=torch.float64).repeat(
                count, 1
            ),
            torch.full((count, 2), 8, dtype=torch.int64),
        )

    positive_means = np.array(((0, 0), (8, 0), (0, 8), (8, 8)), dtype=np.float64)
    collinear_means = np.array(((1, 4), (4, 4), (7, 4)), dtype=np.float64)
    one_means = np.array(((4, 4),), dtype=np.float64)
    truth = early.target_image("affine_xy", height=9, width=9, dtype=np.float64)

    means, conics, radii = geometry(positive_means)
    colors = early.target_values(
        "affine_xy", positive_means, height=9, width=9, dtype=np.float64
    )
    positive = core.reference_qr(means, conics, _tensor(colors, torch.float64), radii, 9, 9)
    candidate = core.candidate_cholesky(
        means.float(), conics.float(), _tensor(colors.astype(np.float32), torch.float32), radii, 9, 9
    )
    positive_error = float(
        np.max(np.abs(positive.output.detach().numpy() - truth))
    )
    candidate_error = float(
        np.max(np.abs(candidate.output.detach().numpy().astype(np.float64) - truth))
    )

    col_means, col_conics, col_radii = geometry(collinear_means)
    col_colors = early.target_values(
        "affine_xy", collinear_means, height=9, width=9, dtype=np.float64
    )
    collinear = core.reference_qr(
        col_means, col_conics, _tensor(col_colors, torch.float64), col_radii, 9, 9
    )
    collinear_candidate = core.candidate_cholesky(
        col_means.float(),
        col_conics.float(),
        _tensor(col_colors.astype(np.float32), torch.float32),
        col_radii,
        9,
        9,
    )

    one_mean, one_conic, one_radii = geometry(one_means)
    one_colors = early.target_values(
        "affine_xy", one_means, height=9, width=9, dtype=np.float64
    )
    one = core.reference_qr(
        one_mean, one_conic, _tensor(one_colors, torch.float64), one_radii, 9, 9
    )
    one_candidate = core.candidate_cholesky(
        one_mean.float(),
        one_conic.float(),
        _tensor(one_colors.astype(np.float32), torch.float32),
        one_radii,
        9,
        9,
    )

    positive_accepted = (
        bool((positive.mass >= early.MINIMUM_MASS).all())
        and bool((positive.active_counts >= early.MINIMUM_CONTRIBUTORS).all())
        and bool((positive.ranks == 3).all())
        and bool((positive.condition_numbers <= early.MAXIMUM_CONDITION).all())
        and bool(positive.finite_pass.all())
        and positive_error <= 1e-12
    )
    candidate_accepted = (
        bool(candidate.mass_pass.all())
        and bool(candidate.contributor_pass.all())
        and bool(candidate.solve_pass.all())
        and candidate_error <= GATES["float32_affine_max_abs"]
    )
    collinear_rejected = bool((collinear.ranks < 3).all())
    one_rejected = (
        bool((one.ranks < 3).all())
        and bool((one.active_counts < 3).all())
    )
    summary = {
        "schema": "bench013-stage0-full-controls-v2",
        "positive": {
            "math64_affine_max_abs": positive_error,
            "math64_mass_min": float(positive.mass.min()),
            "math64_active_min": int(positive.active_counts.min()),
            "math64_rank_min": int(positive.ranks.min()),
            "math64_condition_max": float(positive.condition_numbers.max()),
            "math64_finite_failure_count": int((~positive.finite_pass).sum()),
            "math64_accepted": positive_accepted,
            "cand32_affine_max_abs": candidate_error,
            "cand32_solve_failure_count": int((~candidate.solve_pass).sum()),
            "cand32_accepted": candidate_accepted,
        },
        "negative_collinear": {
            "math64_rank_max": int(collinear.ranks.max()),
            "cand32_solve_failure_count": int((~collinear_candidate.solve_pass).sum()),
            "rejected": collinear_rejected,
        },
        "negative_one_node": {
            "math64_rank_max": int(one.ranks.max()),
            "active_max": int(one.active_counts.max()),
            "cand32_solve_failure_count": int((~one_candidate.solve_pass).sum()),
            "rejected": one_rejected,
        },
        # Both positive paths are assay-plumbing controls.  The registered negative controls
        # remain the float64 rank/contributor rejections; candidate negative results are
        # diagnostic because finite-precision Cholesky can occasionally accept a rank-2 pixel.
        "valid": (
            positive_accepted
            and candidate_accepted
            and collinear_rejected
            and one_rejected
        ),
    }
    raw = {
        "positive_math64_output": positive.output.detach().numpy(),
        "positive_math64_ranks": positive.ranks.numpy(),
        "positive_math64_condition": positive.condition_numbers.numpy(),
        "positive_cand32_output": candidate.output.detach().numpy(),
        "positive_cand32_solve_pass": candidate.solve_pass.numpy(),
        "collinear_math64_ranks": collinear.ranks.numpy(),
        "collinear_cand32_solve_pass": collinear_candidate.solve_pass.numpy(),
        "one_math64_ranks": one.ranks.numpy(),
        "one_active_counts": one.active_counts.numpy(),
        "one_cand32_solve_pass": one_candidate.solve_pass.numpy(),
    }
    return summary, {name: np.ascontiguousarray(value) for name, value in raw.items()}


def _metrics_and_gates(
    target: str,
    mathematical: core.ReferenceResult,
    same_state: core.ReferenceResult,
    candidate: core.CandidateResult,
    exact_colors: np.ndarray,
    colors32: np.ndarray,
) -> tuple[dict[str, Any], dict[str, bool]]:
    truth64 = _tensor(early.target_image(target, dtype=np.float64), torch.float64)
    candidate64 = candidate.output.to(torch.float64)
    difference = candidate64 - same_state.output
    metrics: dict[str, Any] = {
        "mathematical_finite_failure_count": int((~mathematical.finite_pass).sum()),
        "mathematical_rank_failure_count": int((mathematical.ranks != 3).sum()),
        "mathematical_mass_min": float(mathematical.mass.min()),
        "mathematical_active_min": int(mathematical.active_counts.min()),
        "mathematical_condition_max": float(mathematical.condition_numbers.max()),
        "same_state_finite_failure_count": int((~same_state.finite_pass).sum()),
        "same_state_rank_failure_count": int((same_state.ranks != 3).sum()),
        "same_state_mass_min": float(same_state.mass.min()),
        "same_state_active_min": int(same_state.active_counts.min()),
        "same_state_condition_max": float(same_state.condition_numbers.max()),
        "candidate_mass_failure_count": int((~candidate.mass_pass).sum()),
        "candidate_contributor_failure_count": int((~candidate.contributor_pass).sum()),
        "candidate_mass_min": float(candidate.mass.min()),
        "candidate_active_min": int(candidate.active_counts.min()),
        "candidate_condition_max_diagnostic": float(candidate.condition_numbers.max()),
        "candidate_info_failure_count": int((~candidate.cholesky_info_pass).sum()),
        "candidate_diagonal_failure_count": int((~candidate.cholesky_diagonal_pass).sum()),
        "candidate_finite_failure_count": int((~candidate.finite_pass).sum()),
        "candidate_solve_failure_count": int((~candidate.solve_pass).sum()),
        "same_state_max_abs": _max_abs(difference),
        "same_state_rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "float64_partition_max_abs": _max_abs(mathematical.partition - 1.0),
        "float64_first_moment_max_abs": _max_abs(mathematical.first_moment),
        "float32_partition_max_abs": _max_abs(candidate.partition - 1.0),
        "float32_first_moment_max_abs": _max_abs(candidate.first_moment),
        "float64_a1_p99": core.quantile_linear(mathematical.amplification_l1, 0.99),
        "float64_a1_max": float(mathematical.amplification_l1.max()),
        "float32_a1_p99": core.quantile_linear(candidate.amplification_l1, 0.99),
        "float32_a1_max": float(candidate.amplification_l1.max()),
    }
    gates: dict[str, bool] = {
        "mathematical_geometry": (
            metrics["mathematical_finite_failure_count"] == 0
            and metrics["mathematical_rank_failure_count"] == 0
            and metrics["mathematical_mass_min"] >= early.MINIMUM_MASS
            and metrics["mathematical_active_min"] >= early.MINIMUM_CONTRIBUTORS
            and metrics["mathematical_condition_max"] <= early.MAXIMUM_CONDITION
        ),
        "same_state_oracle": (
            metrics["same_state_finite_failure_count"] == 0
            and metrics["same_state_rank_failure_count"] == 0
        ),
        "float32_mass_contributor": (
            metrics["candidate_mass_failure_count"] == 0
            and metrics["candidate_contributor_failure_count"] == 0
        ),
        "float32_cholesky": metrics["candidate_solve_failure_count"] == 0,
        "same_state_parity": (
            metrics["same_state_max_abs"] <= GATES["float32_same_state_max_abs"]
            and metrics["same_state_rmse"] <= GATES["float32_same_state_rmse"]
        ),
        "float64_partition_moment": (
            metrics["float64_partition_max_abs"] <= GATES["float64_partition_and_moment"]
            and metrics["float64_first_moment_max_abs"]
            <= GATES["float64_partition_and_moment"]
        ),
        "float32_partition_moment": (
            metrics["float32_partition_max_abs"] <= GATES["float32_partition_and_moment"]
            and metrics["float32_first_moment_max_abs"]
            <= GATES["float32_partition_and_moment"]
        ),
        "float64_a1": (
            metrics["float64_a1_p99"] <= GATES["a1_p99"]
            and metrics["float64_a1_max"] <= GATES["a1_max"]
        ),
        "float32_a1": (
            metrics["float32_a1_p99"] <= GATES["a1_p99"]
            and metrics["float32_a1_max"] <= GATES["a1_max"]
        ),
    }
    if target == "constant":
        metrics["float64_constant_max_abs"] = _max_abs(mathematical.output - truth64)
        gates["float64_constant_reproduction"] = (
            metrics["float64_constant_max_abs"] <= GATES["float64_constant_max_abs"]
        )
    if target == "affine_xy":
        metrics["float64_affine_max_abs"] = _max_abs(mathematical.output - truth64)
        metrics["float32_affine_max_abs"] = _max_abs(candidate64 - truth64)
        gates["float64_affine_reproduction"] = (
            metrics["float64_affine_max_abs"] <= GATES["float64_affine_max_abs"]
        )
        gates["float32_affine_reproduction"] = (
            metrics["float32_affine_max_abs"] <= GATES["float32_affine_max_abs"]
        )
    if target in ("vertical_step", "diagonal_step", "checker8"):
        excursion = _range_excursion_diagnostic(
            candidate.output.detach().numpy(), exact_colors
        )
        metrics["float32_range_excursion"] = excursion["maximum"]
        metrics["float32_range_excursion_diagnostic"] = excursion
        gates["float32_range_excursion"] = (
            metrics["float32_range_excursion"] <= GATES["step_excursion"]
        )
    return metrics, gates


def _forward_arrays(
    mathematical: core.ReferenceResult,
    same_state: core.ReferenceResult,
    candidate: core.CandidateResult,
) -> dict[str, np.ndarray]:
    def array(value: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(value.detach().cpu().numpy())

    return {
        "mathematical_output": array(mathematical.output),
        "mathematical_mass": array(mathematical.mass),
        "mathematical_active_counts": array(mathematical.active_counts),
        "mathematical_singular_values": array(mathematical.singular_values),
        "mathematical_rank_thresholds": array(mathematical.rank_thresholds),
        "mathematical_ranks": array(mathematical.ranks),
        "mathematical_covariance_eigenvalues": array(
            mathematical.covariance_eigenvalues
        ),
        "mathematical_condition_numbers": array(mathematical.condition_numbers),
        "mathematical_partition": array(mathematical.partition),
        "mathematical_first_moment": array(mathematical.first_moment),
        "mathematical_a1": array(mathematical.amplification_l1),
        "mathematical_finite_pass": array(mathematical.finite_pass),
        "same_state_output": array(same_state.output),
        "same_state_singular_values": array(same_state.singular_values),
        "same_state_rank_thresholds": array(same_state.rank_thresholds),
        "same_state_ranks": array(same_state.ranks),
        "same_state_mass": array(same_state.mass),
        "same_state_active_counts": array(same_state.active_counts),
        "same_state_covariance_eigenvalues": array(same_state.covariance_eigenvalues),
        "same_state_condition_numbers": array(same_state.condition_numbers),
        "same_state_finite_pass": array(same_state.finite_pass),
        "same_state_partition": array(same_state.partition),
        "same_state_first_moment": array(same_state.first_moment),
        "same_state_a1": array(same_state.amplification_l1),
        "candidate_output": array(candidate.output),
        "candidate_accumulators": array(candidate.accumulators),
        "candidate_mass": array(candidate.mass),
        "candidate_active_counts": array(candidate.active_counts),
        "candidate_mass_pass": array(candidate.mass_pass),
        "candidate_contributor_pass": array(candidate.contributor_pass),
        "candidate_offset_mean": array(candidate.offset_mean),
        "candidate_color_mean": array(candidate.color_mean),
        "candidate_covariance": array(candidate.centered_covariance),
        "candidate_covariance_eigenvalues": array(candidate.covariance_eigenvalues),
        "candidate_condition_numbers": array(candidate.condition_numbers),
        "candidate_cross_covariance": array(candidate.centered_cross_covariance),
        "candidate_cholesky": array(candidate.cholesky),
        "candidate_solved_slopes": array(candidate.solved_slopes),
        "candidate_inverse_mean": array(candidate.inverse_mean),
        "candidate_cholesky_info": array(candidate.cholesky_info),
        "candidate_info_pass": array(candidate.cholesky_info_pass),
        "candidate_diagonal_pass": array(candidate.cholesky_diagonal_pass),
        "candidate_finite_pass": array(candidate.finite_pass),
        "candidate_solve_pass": array(candidate.solve_pass),
        "candidate_partition": array(candidate.partition),
        "candidate_first_moment": array(candidate.first_moment),
        "candidate_a1": array(candidate.amplification_l1),
    }


def _expected_forward_shapes() -> dict[str, tuple[int, ...]]:
    scalar = (PIXELS,)
    vector2 = (PIXELS, 2)
    matrix2 = (PIXELS, 2, 2)
    result = {
        "mathematical_output": (HEIGHT, WIDTH, 3),
        "mathematical_mass": scalar,
        "mathematical_active_counts": scalar,
        "mathematical_singular_values": (PIXELS, 3),
        "mathematical_rank_thresholds": scalar,
        "mathematical_ranks": scalar,
        "mathematical_covariance_eigenvalues": vector2,
        "mathematical_condition_numbers": scalar,
        "mathematical_partition": scalar,
        "mathematical_first_moment": vector2,
        "mathematical_a1": scalar,
        "mathematical_finite_pass": scalar,
        "same_state_output": (HEIGHT, WIDTH, 3),
        "same_state_singular_values": (PIXELS, 3),
        "same_state_rank_thresholds": scalar,
        "same_state_ranks": scalar,
        "same_state_mass": scalar,
        "same_state_active_counts": scalar,
        "same_state_covariance_eigenvalues": vector2,
        "same_state_condition_numbers": scalar,
        "same_state_finite_pass": scalar,
        "same_state_partition": scalar,
        "same_state_first_moment": vector2,
        "same_state_a1": scalar,
        "candidate_output": (HEIGHT, WIDTH, 3),
        "candidate_accumulators": (PIXELS, core.CHANNEL_COUNT),
        "candidate_mass": scalar,
        "candidate_active_counts": scalar,
        "candidate_mass_pass": scalar,
        "candidate_contributor_pass": scalar,
        "candidate_offset_mean": vector2,
        "candidate_color_mean": (PIXELS, 3),
        "candidate_covariance": matrix2,
        "candidate_covariance_eigenvalues": vector2,
        "candidate_condition_numbers": scalar,
        "candidate_cross_covariance": (PIXELS, 2, 3),
        "candidate_cholesky": matrix2,
        "candidate_solved_slopes": (PIXELS, 2, 3),
        "candidate_inverse_mean": vector2,
        "candidate_cholesky_info": scalar,
        "candidate_info_pass": scalar,
        "candidate_diagonal_pass": scalar,
        "candidate_finite_pass": scalar,
        "candidate_solve_pass": scalar,
        "candidate_partition": scalar,
        "candidate_first_moment": vector2,
        "candidate_a1": scalar,
    }
    return result


GRADIENT_BLOCKS = ("means_over_l", "log_scales", "rotations_over_pi", "colors")


def _candidate_mask_contract(arrays: Mapping[str, np.ndarray]) -> bool:
    """Check persisted candidate masks against their defining raw diagnostics."""

    cholesky = np.asarray(arrays["candidate_cholesky"])
    diagonal = np.all(np.diagonal(cholesky, axis1=1, axis2=2) > 0.0, axis=1)
    return bool(
        np.array_equal(
            arrays["candidate_mass"], arrays["candidate_accumulators"][:, 0]
        )
        and np.array_equal(
            arrays["candidate_info_pass"], arrays["candidate_cholesky_info"] == 0
        )
        and np.array_equal(
            arrays["candidate_mass_pass"],
            arrays["candidate_mass"] >= early.MINIMUM_MASS,
        )
        and np.array_equal(
            arrays["candidate_contributor_pass"],
            arrays["candidate_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
        )
        and np.array_equal(arrays["candidate_diagonal_pass"], diagonal)
        and np.array_equal(
            arrays["candidate_solve_pass"],
            arrays["candidate_info_pass"]
            & arrays["candidate_diagonal_pass"]
            & arrays["candidate_finite_pass"],
        )
    )


def _candidate_eigenvalue_contract(arrays: Mapping[str, np.ndarray]) -> bool:
    """Bind eigenvalues to persisted covariance and condition to those bound eigenvalues."""

    covariance_eigenvalues = np.asarray(arrays["candidate_covariance_eigenvalues"])
    covariance_derived = np.linalg.eigvalsh(
        np.asarray(arrays["candidate_covariance"])
    )
    condition_derived = np.where(
        covariance_eigenvalues[:, 0] > 0.0,
        covariance_eigenvalues[:, 1] / covariance_eigenvalues[:, 0],
        np.asarray(np.inf, dtype=covariance_eigenvalues.dtype),
    )
    return bool(
        np.allclose(
            covariance_eigenvalues,
            covariance_derived,
            rtol=2e-6,
            atol=2e-8,
            equal_nan=True,
        )
        and np.array_equal(
            arrays["candidate_condition_numbers"],
            condition_derived,
            equal_nan=True,
        )
    )


def _componentwise_backward_error_contract(
    matrix: np.ndarray,
    solution: np.ndarray,
    right_hand_side: np.ndarray,
) -> bool:
    """Validate a persisted float32 solve with a fixed componentwise backward bound."""

    matrix64 = np.asarray(matrix, dtype=np.float64)
    solution64 = np.asarray(solution, dtype=np.float64)
    right64 = np.asarray(right_hand_side, dtype=np.float64)
    if solution64.ndim == 2:
        solution64 = solution64[:, :, None]
    if right64.ndim == 2:
        right64 = right64[:, :, None]
    if (
        matrix64.ndim != 3
        or matrix64.shape[1:] != (2, 2)
        or solution64.ndim != 3
        or right64.shape != solution64.shape
        or matrix64.shape[0] != solution64.shape[0]
        or matrix64.shape[2] != solution64.shape[1]
    ):
        return False
    if not (
        np.isfinite(matrix64).all()
        and np.isfinite(solution64).all()
        and np.isfinite(right64).all()
    ):
        return False
    product = np.einsum("pij,pjk->pik", matrix64, solution64)
    residual = np.abs(product - right64)
    component_scale = (
        np.einsum("pij,pjk->pik", np.abs(matrix64), np.abs(solution64))
        + np.abs(right64)
    )
    tolerance = (
        REPLAY_FLOAT32_BACKWARD_ERROR_MULTIPLIER
        * np.finfo(np.float32).eps
        * component_scale
    )
    return bool(np.all(residual <= tolerance))


def _effective_arrays_match(
    arrays: Mapping[str, np.ndarray],
    representative: Mapping[str, np.ndarray],
    keys: set[str],
) -> bool:
    """Bind an effective-field table to the first lexical forward representative."""

    return set(arrays) == keys and all(
        np.array_equal(arrays[name], representative[name], equal_nan=True)
        for name in keys
    )


def _gradient_raw_shape_contract(arrays: Mapping[str, np.ndarray], count: int) -> bool:
    """Validate the complete frozen gradient NPZ key and shape contract."""

    block_shapes = {
        "means_over_l": (count, 2),
        "log_scales": (count, 2),
        "rotations_over_pi": (count,),
        "colors": (count, 3),
    }
    expected_keys = {
        "cotangent_float64",
        "cotangent_float32",
        "contributors_gid",
        "contributors_px",
        "contributors_py",
        "frozen_radii",
        "base_means_float32",
        "base_log_scales_float32",
        "base_rotations_float32",
        "base_colors_float32",
        "directional_loss_plus64",
        "directional_loss_minus64",
    }
    for block in GRADIENT_BLOCKS:
        expected_keys.update(
            {
                f"gradient64__{block}",
                f"gradient32__{block}",
                f"raw_gradient64__{block}",
                f"raw_gradient32__{block}",
                f"directions__{block}",
            }
        )
    if set(arrays) != expected_keys:
        return False
    contributors = np.asarray(arrays["contributors_gid"])
    if contributors.ndim != 1:
        return False
    shapes = {
        "cotangent_float64": (HEIGHT, WIDTH, 3),
        "cotangent_float32": (HEIGHT, WIDTH, 3),
        "contributors_gid": contributors.shape,
        "contributors_px": contributors.shape,
        "contributors_py": contributors.shape,
        "frozen_radii": (count, 2),
        "base_means_float32": (count, 2),
        "base_log_scales_float32": (count, 2),
        "base_rotations_float32": (count,),
        "base_colors_float32": (count, 3),
        "directional_loss_plus64": (8,),
        "directional_loss_minus64": (8,),
    }
    for block, block_shape in block_shapes.items():
        shapes[f"directions__{block}"] = (2, *block_shape)
        for prefix in (
            "gradient64__",
            "gradient32__",
            "raw_gradient64__",
            "raw_gradient32__",
        ):
            shapes[f"{prefix}{block}"] = block_shape
    return all(np.asarray(arrays[name]).shape == shape for name, shape in shapes.items())


def _permutation_row_identity_contract(
    row: Mapping[str, Any],
    index: int,
    cell: Mapping[str, Any],
    binding: str,
) -> bool:
    """Bind a sorted permutation ledger row to its exact registered order slot."""

    return bool(
        row["binding_sha256"] == binding
        and all(row[key] == value for key, value in cell.items())
        and int(row["order_index"]) == index
        and row["order"] == PERMUTATION_ORDERS[index]
    )


def _gradient_jsonl_metadata_contract(
    rows: list[dict[str, Any]],
    gradient: Mapping[str, Any],
    binding: str,
    selected: str,
) -> bool:
    """Bind gradient JSONL row schemas and status row to the JSON summary."""

    if not rows or rows[-1].get("row_type") != "status":
        return False
    expected_schemas = {
        "direction": "bench013-stage0-gradient-direction-row-v2",
        "block": "bench013-stage0-gradient-block-row-v2",
        "status": "bench013-stage0-gradient-status-row-v2",
    }
    common = all(
        row.get("row_type") in expected_schemas
        and row.get("binding_sha256") == binding
        and row.get("cell_id") == selected
        and row.get("schema") == expected_schemas[str(row["row_type"])]
        and row.get("status") == gradient["status"]
        for row in rows
    )
    return common and bool(rows[-1].get("gradient_pass")) == bool(
        gradient["gradient_pass"]
    )


def _dimensionless_gradients(
    gradients: tuple[torch.Tensor, ...], height: int, width: int
) -> dict[str, torch.Tensor]:
    coordinate_scale = float(max(int(height) - 1, int(width) - 1, 1))
    return {
        "means_over_l": gradients[0] * coordinate_scale,
        "log_scales": gradients[1],
        "rotations_over_pi": gradients[2] * math.pi,
        "colors": gradients[3],
    }


def _gradient_block_metrics(
    block: str, value64: torch.Tensor, value32: torch.Tensor
) -> dict[str, Any]:
    """Return the frozen mixed-norm block gate and its relative diagnostic."""

    value32_as64 = value32.to(torch.float64)
    norm64 = float(torch.linalg.vector_norm(value64))
    norm32 = float(torch.linalg.vector_norm(value32_as64))
    absolute_error = float(torch.linalg.vector_norm(value32_as64 - value64))
    relative = absolute_error / max(norm64, 1e-12)
    tolerance = GATES["gradient_full_abs"] + GATES["gradient_full_rel"] * norm64
    finite = bool(torch.isfinite(value64).all() and torch.isfinite(value32_as64).all())
    return {
        "block": block,
        "gradient64_l2": norm64,
        "gradient32_l2": norm32,
        "absolute_l2_error": absolute_error,
        "mixed_l2_tolerance": tolerance,
        "relative_l2_error": relative,
        "finite": finite,
        "pass": finite and absolute_error <= tolerance,
    }


def _gradient_loss(
    parameters: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    radii: torch.Tensor,
    contributors: core.FrozenContributors,
    cotangent: torch.Tensor,
    *,
    candidate: bool,
) -> torch.Tensor:
    means, log_scales, rotations, colors = parameters
    if candidate:
        output = core.candidate_render_frozen(
            means, log_scales, rotations, colors, radii, contributors
        )
    else:
        output = core.reference_render_frozen(
            means, log_scales, rotations, colors, radii, contributors
        )
    return torch.sum(output * cotangent)


def _gradient_audit(
    arrays: Mapping[str, np.ndarray], target: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if target != "affine_xy":
        raise ValueError("the frozen gradient cell is affine_xy")
    count = int(arrays["means"].shape[0])
    radii = torch.from_numpy(np.ascontiguousarray(arrays["radii"])).to(torch.int64)
    means32_base = _tensor(arrays["means"], torch.float32)
    logs32_base = _tensor(arrays["log_scales"], torch.float32)
    rotations32_base = _tensor(arrays["rotations"], torch.float32)
    colors_np = early.target_values(target, arrays["means"], dtype=np.float64).astype(np.float32)
    colors32_base = _tensor(colors_np, torch.float32)
    contributors = core.enumerate_contributors(means32_base, radii, HEIGHT, WIDTH)

    rng = np.random.Generator(np.random.PCG64(GRADIENT_SEED))
    cotangent_np = rng.standard_normal((HEIGHT, WIDTH, 3))
    cotangent_np /= np.linalg.norm(cotangent_np.reshape(-1))
    cotangent32_np = np.ascontiguousarray(cotangent_np.astype(np.float32))
    direction_shapes = {
        "means_over_l": (count, 2),
        "log_scales": (count, 2),
        "rotations_over_pi": (count,),
        "colors": (count, 3),
    }
    directions: dict[str, list[np.ndarray]] = {}
    for block in GRADIENT_BLOCKS:
        values: list[np.ndarray] = []
        for _ in range(2):
            direction = rng.standard_normal(direction_shapes[block])
            direction /= np.linalg.norm(direction.reshape(-1))
            values.append(np.ascontiguousarray(direction))
        directions[block] = values

    base64 = (
        means32_base.to(torch.float64).detach().requires_grad_(True),
        logs32_base.to(torch.float64).detach().requires_grad_(True),
        rotations32_base.to(torch.float64).detach().requires_grad_(True),
        colors32_base.to(torch.float64).detach().requires_grad_(True),
    )
    loss64 = _gradient_loss(
        base64,
        radii,
        contributors,
        torch.from_numpy(cotangent_np).to(torch.float64),
        candidate=False,
    )
    raw64 = torch.autograd.grad(loss64, base64)
    gradient64 = _dimensionless_gradients(raw64, HEIGHT, WIDTH)

    base32 = (
        means32_base.detach().requires_grad_(True),
        logs32_base.detach().requires_grad_(True),
        rotations32_base.detach().requires_grad_(True),
        colors32_base.detach().requires_grad_(True),
    )
    loss32 = _gradient_loss(
        base32,
        radii,
        contributors,
        torch.from_numpy(cotangent32_np),
        candidate=True,
    )
    raw32 = torch.autograd.grad(loss32, base32)
    gradient32 = _dimensionless_gradients(raw32, HEIGHT, WIDTH)

    parameter_index = {
        "means_over_l": 0,
        "log_scales": 1,
        "rotations_over_pi": 2,
        "colors": 3,
    }
    raw_scale = {
        "means_over_l": float(max(HEIGHT - 1, WIDTH - 1, 1)),
        "log_scales": 1.0,
        "rotations_over_pi": math.pi,
        "colors": 1.0,
    }
    directional_rows: list[dict[str, Any]] = []
    directional_pass = True
    for block in GRADIENT_BLOCKS:
        block_index = parameter_index[block]
        for direction_index, direction_np in enumerate(directions[block]):
            direction64 = torch.from_numpy(direction_np).to(torch.float64)
            plus = [value.detach().clone() for value in base64]
            minus = [value.detach().clone() for value in base64]
            perturbation = GRADIENT_STEP * raw_scale[block] * direction64
            plus[block_index] += perturbation
            minus[block_index] -= perturbation
            with torch.no_grad():
                plus_loss = _gradient_loss(
                    tuple(plus),
                    radii,
                    contributors,
                    torch.from_numpy(cotangent_np).to(torch.float64),
                    candidate=False,
                )
                minus_loss = _gradient_loss(
                    tuple(minus),
                    radii,
                    contributors,
                    torch.from_numpy(cotangent_np).to(torch.float64),
                    candidate=False,
                )
            finite_difference = float((plus_loss - minus_loss) / (2.0 * GRADIENT_STEP))
            ad64 = float(torch.sum(gradient64[block] * direction64))
            ad32 = float(
                torch.sum(gradient32[block].to(torch.float64) * direction64)
            )
            fd_error = abs(ad64 - finite_difference)
            fd_tolerance = GATES["gradient_fd_abs"] + GATES["gradient_fd_rel"] * max(
                abs(ad64), abs(finite_difference)
            )
            cross_error = abs(ad32 - ad64)
            cross_tolerance = GATES["gradient_cross_abs"] + GATES["gradient_cross_rel"] * abs(
                ad64
            )
            row_pass = (
                math.isfinite(ad64)
                and math.isfinite(ad32)
                and math.isfinite(finite_difference)
                and fd_error <= fd_tolerance
                and cross_error <= cross_tolerance
            )
            directional_pass &= row_pass
            directional_rows.append(
                {
                    "block": block,
                    "direction": direction_index,
                    "ad64": ad64,
                    "fd64": finite_difference,
                    "ad32": ad32,
                    "loss_plus64": float(plus_loss),
                    "loss_minus64": float(minus_loss),
                    "ad64_fd64_abs_error": fd_error,
                    "ad64_fd64_tolerance": fd_tolerance,
                    "ad32_ad64_abs_error": cross_error,
                    "ad32_ad64_tolerance": cross_tolerance,
                    "pass": row_pass,
                }
            )

    block_rows: list[dict[str, Any]] = []
    block_pass = True
    for block in GRADIENT_BLOCKS:
        row = _gradient_block_metrics(block, gradient64[block], gradient32[block])
        block_pass &= bool(row["pass"])
        block_rows.append(row)

    arrays_out: dict[str, np.ndarray] = {
        "cotangent_float64": np.ascontiguousarray(cotangent_np),
        "cotangent_float32": cotangent32_np,
        "contributors_gid": contributors.gid.numpy(),
        "contributors_px": contributors.px.numpy(),
        "contributors_py": contributors.py.numpy(),
        "frozen_radii": radii.numpy(),
        "base_means_float32": means32_base.numpy(),
        "base_log_scales_float32": logs32_base.numpy(),
        "base_rotations_float32": rotations32_base.numpy(),
        "base_colors_float32": colors32_base.numpy(),
        "directional_loss_plus64": np.asarray(
            [row["loss_plus64"] for row in directional_rows], dtype=np.float64
        ),
        "directional_loss_minus64": np.asarray(
            [row["loss_minus64"] for row in directional_rows], dtype=np.float64
        ),
    }
    for block in GRADIENT_BLOCKS:
        arrays_out[f"gradient64__{block}"] = np.ascontiguousarray(
            gradient64[block].detach().numpy()
        )
        arrays_out[f"gradient32__{block}"] = np.ascontiguousarray(
            gradient32[block].detach().numpy()
        )
        raw_index = {
            "means_over_l": 0,
            "log_scales": 1,
            "rotations_over_pi": 2,
            "colors": 3,
        }[block]
        arrays_out[f"raw_gradient64__{block}"] = np.ascontiguousarray(
            raw64[raw_index].detach().numpy()
        )
        arrays_out[f"raw_gradient32__{block}"] = np.ascontiguousarray(
            raw32[raw_index].detach().numpy()
        )
        arrays_out[f"directions__{block}"] = np.stack(directions[block], axis=0)
    summary = {
        "schema": "bench013-stage0-gradient-v2",
        "seed": GRADIENT_SEED,
        "step": GRADIENT_STEP,
        "coordinate_blocks": list(GRADIENT_BLOCKS),
        "cotangent_l2_float64": float(np.linalg.norm(cotangent_np.reshape(-1))),
        "directions": directional_rows,
        "blocks": block_rows,
        "directional_pass": directional_pass,
        "block_pass": block_pass,
        "gradient_pass": directional_pass and block_pass,
    }
    return summary, arrays_out


def _raw_metrics_and_gates(
    target: str, arrays: Mapping[str, np.ndarray], exact_colors: np.ndarray
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Replay forward metrics using only persisted arrays and NumPy."""

    truth = early.target_image(target, dtype=np.float64)
    mathematical_output = np.asarray(arrays["mathematical_output"], dtype=np.float64)
    same_output = np.asarray(arrays["same_state_output"], dtype=np.float64)
    candidate_output = np.asarray(arrays["candidate_output"], dtype=np.float32)
    difference = candidate_output.astype(np.float64) - same_output
    condition = np.asarray(arrays["mathematical_condition_numbers"])
    metrics: dict[str, Any] = {
        "mathematical_finite_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["mathematical_finite_pass"], dtype=bool))
        ),
        "mathematical_rank_failure_count": int(
            np.count_nonzero(np.asarray(arrays["mathematical_ranks"]) != 3)
        ),
        "mathematical_mass_min": float(np.min(arrays["mathematical_mass"])),
        "mathematical_active_min": int(np.min(arrays["mathematical_active_counts"])),
        "mathematical_condition_max": float(np.max(condition)),
        "same_state_finite_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["same_state_finite_pass"], dtype=bool))
        ),
        "same_state_rank_failure_count": int(
            np.count_nonzero(np.asarray(arrays["same_state_ranks"]) != 3)
        ),
        "same_state_mass_min": float(np.min(arrays["same_state_mass"])),
        "same_state_active_min": int(np.min(arrays["same_state_active_counts"])),
        "same_state_condition_max": float(np.max(arrays["same_state_condition_numbers"])),
        "candidate_mass_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["candidate_mass_pass"], dtype=bool))
        ),
        "candidate_contributor_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["candidate_contributor_pass"], dtype=bool))
        ),
        "candidate_mass_min": float(np.min(arrays["candidate_mass"])),
        "candidate_active_min": int(np.min(arrays["candidate_active_counts"])),
        "candidate_condition_max_diagnostic": float(
            np.max(arrays["candidate_condition_numbers"])
        ),
        "candidate_info_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["candidate_info_pass"], dtype=bool))
        ),
        "candidate_diagonal_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["candidate_diagonal_pass"], dtype=bool))
        ),
        "candidate_finite_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["candidate_finite_pass"], dtype=bool))
        ),
        "candidate_solve_failure_count": int(
            np.count_nonzero(~np.asarray(arrays["candidate_solve_pass"], dtype=bool))
        ),
        "same_state_max_abs": float(np.max(np.abs(difference))),
        "same_state_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "float64_partition_max_abs": float(
            np.max(np.abs(np.asarray(arrays["mathematical_partition"]) - 1.0))
        ),
        "float64_first_moment_max_abs": float(
            np.max(np.abs(arrays["mathematical_first_moment"]))
        ),
        "float32_partition_max_abs": float(
            np.max(np.abs(np.asarray(arrays["candidate_partition"]) - np.float32(1.0)))
        ),
        "float32_first_moment_max_abs": float(
            np.max(np.abs(arrays["candidate_first_moment"]))
        ),
        "float64_a1_p99": float(
            np.quantile(arrays["mathematical_a1"], 0.99, method="linear")
        ),
        "float64_a1_max": float(np.max(arrays["mathematical_a1"])),
        "float32_a1_p99": float(
            np.quantile(arrays["candidate_a1"], 0.99, method="linear")
        ),
        "float32_a1_max": float(np.max(arrays["candidate_a1"])),
    }
    gates: dict[str, bool] = {
        "mathematical_geometry": (
            metrics["mathematical_finite_failure_count"] == 0
            and metrics["mathematical_rank_failure_count"] == 0
            and metrics["mathematical_mass_min"] >= early.MINIMUM_MASS
            and metrics["mathematical_active_min"] >= early.MINIMUM_CONTRIBUTORS
            and metrics["mathematical_condition_max"] <= early.MAXIMUM_CONDITION
        ),
        "same_state_oracle": (
            metrics["same_state_finite_failure_count"] == 0
            and metrics["same_state_rank_failure_count"] == 0
        ),
        "float32_mass_contributor": (
            metrics["candidate_mass_failure_count"] == 0
            and metrics["candidate_contributor_failure_count"] == 0
        ),
        "float32_cholesky": metrics["candidate_solve_failure_count"] == 0,
        "same_state_parity": (
            metrics["same_state_max_abs"] <= GATES["float32_same_state_max_abs"]
            and metrics["same_state_rmse"] <= GATES["float32_same_state_rmse"]
        ),
        "float64_partition_moment": (
            metrics["float64_partition_max_abs"] <= GATES["float64_partition_and_moment"]
            and metrics["float64_first_moment_max_abs"]
            <= GATES["float64_partition_and_moment"]
        ),
        "float32_partition_moment": (
            metrics["float32_partition_max_abs"] <= GATES["float32_partition_and_moment"]
            and metrics["float32_first_moment_max_abs"]
            <= GATES["float32_partition_and_moment"]
        ),
        "float64_a1": (
            metrics["float64_a1_p99"] <= GATES["a1_p99"]
            and metrics["float64_a1_max"] <= GATES["a1_max"]
        ),
        "float32_a1": (
            metrics["float32_a1_p99"] <= GATES["a1_p99"]
            and metrics["float32_a1_max"] <= GATES["a1_max"]
        ),
    }
    if target == "constant":
        metrics["float64_constant_max_abs"] = float(
            np.max(np.abs(mathematical_output - truth))
        )
        gates["float64_constant_reproduction"] = (
            metrics["float64_constant_max_abs"] <= GATES["float64_constant_max_abs"]
        )
    if target == "affine_xy":
        metrics["float64_affine_max_abs"] = float(
            np.max(np.abs(mathematical_output - truth))
        )
        metrics["float32_affine_max_abs"] = float(
            np.max(np.abs(candidate_output.astype(np.float64) - truth))
        )
        gates["float64_affine_reproduction"] = (
            metrics["float64_affine_max_abs"] <= GATES["float64_affine_max_abs"]
        )
        gates["float32_affine_reproduction"] = (
            metrics["float32_affine_max_abs"] <= GATES["float32_affine_max_abs"]
        )
    if target in ("vertical_step", "diagonal_step", "checker8"):
        excursion = _range_excursion_diagnostic(candidate_output, exact_colors)
        metrics["float32_range_excursion"] = excursion["maximum"]
        metrics["float32_range_excursion_diagnostic"] = excursion
        gates["float32_range_excursion"] = (
            metrics["float32_range_excursion"] <= GATES["step_excursion"]
        )
    return metrics, gates


def _nested_close(first: Any, second: Any) -> bool:
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return set(first) == set(second) and all(
            _nested_close(first[key], second[key]) for key in first
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _nested_close(left, right) for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, float) or isinstance(second, float):
        left = float(first)
        right = float(second)
        if not math.isfinite(left) or not math.isfinite(right):
            return (math.isnan(left) and math.isnan(right)) or left == right
        return math.isclose(left, right, rel_tol=2e-12, abs_tol=1e-15)
    return first == second


def _write_source_archive(root: Path, outdir: Path) -> dict[str, Any]:
    path = outdir / "executed_sources.tar"
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative in ARCHIVE_PATHS:
            source = root / relative
            if not source.is_file():
                raise RuntimeError(f"missing BENCH-013 full archive dependency: {relative}")
            info = archive.gettarinfo(str(source), arcname=relative)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "members": {relative: _sha256_file(root / relative) for relative in ARCHIVE_PATHS},
    }


def _artifact_manifest(outdir: Path) -> dict[str, Any]:
    files = {
        path.relative_to(outdir).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(outdir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    return {
        "schema": "bench013-stage0-full-artifact-manifest-v3",
        "file_count": len(files),
        "files": files,
    }


def _outcome_free_unavailable_analysis(
    *,
    binding: str,
    replay: Mapping[str, Any],
    controls_valid: bool,
    source_archive_record: Mapping[str, Any],
    timing: Mapping[str, Any],
    recovery_equivalence: Mapping[str, Any],
    recovery_equivalence_file_sha256: str,
) -> dict[str, Any]:
    """Build a lifecycle failure record without interpreting any scientific ledger."""

    return {
        "schema": "bench013-stage0-full-analysis-v3",
        "binding_sha256": binding,
        "parent_binding_sha256": PARENT_BINDING,
        "parent_task_sha256": PARENT_TASK_SHA256,
        "clarified_task_sha256": TASK_SHA256,
        "complete": False,
        "artifact_audit_passed": bool(replay.get("artifact_audit_passed", False)),
        "controls_valid": controls_valid,
        "recovery_equivalence": {
            "path": "recovery_equivalence.json",
            "file_sha256": recovery_equivalence_file_sha256,
            "equivalence_pass": bool(
                recovery_equivalence.get("equivalence_pass", False)
            ),
        },
        "decision": "unavailable",
        "classification": "invalid_lifecycle_or_recovery_equivalence",
        "outcome_ledgers_interpreted": False,
        "stage1_authorized": False,
        "claim_boundary": {
            "quality": False,
            "convergence": False,
            "performance": False,
            "compression": False,
            "expressiveness": False,
            "method_novelty": False,
        },
        "source_archive": dict(source_archive_record),
        "timing": dict(timing),
    }


def run(outdir: Path, root: Path, parent: Path) -> dict[str, Any]:
    if outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {outdir}")
    if _sha256_file(root / "tasks/BENCH-013-local-linear-reproducing-compositor.md") != TASK_SHA256:
        raise RuntimeError("clarified downstream task hash mismatch")
    predecessor_verification = verify_recovery_predecessor(root)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    outdir.mkdir(parents=True)
    parent_record = verify_parent(parent)
    environment = _environment(root)
    source_manifest = _source_manifest(root)

    target_arrays: dict[str, np.ndarray] = {}
    for target in early.TARGETS:
        target_arrays[f"{target}__float64"] = early.target_image(target, dtype=np.float64)
        target_arrays[f"{target}__float32"] = early.target_image(target, dtype=np.float32)
    _deterministic_npz(outdir / "targets.npz", target_arrays)
    target_record = {
        name: _array_record(value) for name, value in sorted(target_arrays.items())
    }
    parent_config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    target_formulas = {
        target: {
            "formula": early.TARGET_FORMULAS[target],
            "formula_sha256": _sha256_bytes(early.TARGET_FORMULAS[target].encode("utf-8")),
            "parent_formula_sha256": parent_config["targets"][target]["formula_sha256"],
        }
        for target in early.TARGETS
    }
    if any(
        record["formula_sha256"] != record["parent_formula_sha256"]
        for record in target_formulas.values()
    ):
        raise RuntimeError("parent analytic formula binding mismatch")

    science = {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "clarified_task_sha256": TASK_SHA256,
        "parent": parent_record,
        "infrastructure_recovery": {
            "predecessor": dict(RECOVERY_PREDECESSOR),
            "preflight_verification": predecessor_verification,
            "previous_clarified_task_sha256": PREVIOUS_CLARIFIED_TASK_SHA256,
            "lifecycle_only_changes": [
                "source_archive_metadata_scope",
                "candidate_spectral_replay_binding",
                "candidate_solve_backward_error_replay",
                "top_level_lifecycle_schema_v3",
                "predecessor_equivalence_gate",
            ],
            "scientific_protocol_unchanged": True,
            "scientific_metric_and_gate_ledgers_uninterpreted_before_repair": True,
            "incidental_status_exposure": (
                "not_reached_preregistered_base_forward_failure"
            ),
        },
        "dimensions": {"height": HEIGHT, "width": WIDTH, "all_pixels_scored": True},
        "counts": list(early.COUNTS),
        "seeds": list(early.SEEDS),
        "targets": list(early.TARGETS),
        "cohorts": list(early.COHORTS),
        "paths": {
            "math64": "promoted_parameters_recomputed_conics_exact_colors_active_qr",
            "same64": "promoted_persisted_conics_and_stored_target_colors_active_qr",
            "cand32": "stored_state_15_channel_index_add_centered_cholesky",
        },
        "per_pixel_diagnostics": (
            "one deterministic NPZ table per cell, incrementally ledgered in "
            "forward_cells.jsonl"
        ),
        "gates": dict(GATES),
        "replay_validation": {
            "candidate_eigen_rtol": 2e-6,
            "candidate_eigen_atol": 2e-8,
            "candidate_condition_from_persisted_eigenvalues": "exact_float32",
            "candidate_solve_componentwise_backward_error_eps32_multiplier": (
                REPLAY_FLOAT32_BACKWARD_ERROR_MULTIPLIER
            ),
            "recovery_equivalence": {
                "expected_npz_count": RECOVERY_EXPECTED_NPZ_COUNT,
                "npz_comparison": "recursive_relative_path_size_sha256_exact",
                "binding_normalized_json": list(RECOVERY_JSON_PATHS),
                "binding_normalized_jsonl": list(RECOVERY_JSONL_PATHS),
                "binding_normalization": "drop_top_level_binding_sha256_only",
                "parent_binding": "exact_bytes",
                "science_config_drop_keys": sorted(RECOVERY_CONFIG_DROP_KEYS),
            },
        },
        "permutation": {
            "seed": PERMUTATION_SEED,
            "orders": list(PERMUTATION_ORDERS),
            "global_stream": True,
            "lexical_cells": True,
            "permuted_before_enumeration": True,
        },
        "gradient": {
            "seed": GRADIENT_SEED,
            "step": GRADIENT_STEP,
            "cell_id": early._cell_id("target_conditioned", "affine_xy", 56, 0),  # noqa: SLF001
            "blocks": list(GRADIENT_BLOCKS),
        },
        "plumbing_controls": {
            "height": 9,
            "width": 9,
            "positive": "four corners affine reproduction",
            "negatives": ["three collinear", "one node"],
            "new_downstream_core": True,
        },
        "expected": {
            "logical_fields": LOGICAL_FIELDS,
            "forward_cells": FORWARD_CELLS,
            "forward_pixels": FORWARD_CELLS * PIXELS,
            "permutation_rows": FORWARD_CELLS * len(PERMUTATION_ORDERS),
            "gradient_cells": 1,
            "gradient_rows": "13_if_completed_else_1",
        },
        "target_arrays": target_record,
        "target_formulas": target_formulas,
        "source_manifest": source_manifest,
    }
    binding = _binding(science, environment)
    _atomic_json(outdir / "config.json", {**science, "binding_sha256": binding})
    _atomic_json(outdir / "environment.json", environment)
    _atomic_json(outdir / "parent_binding.json", parent_record)
    source_archive_record = _write_source_archive(root, outdir)
    _atomic_json(outdir / "source_archive.json", source_archive_record)
    controls, control_arrays = downstream_controls()
    _deterministic_npz(outdir / "controls_raw.npz", control_arrays)
    controls.update(
        {
            "binding_sha256": binding,
            "path": "controls_raw.npz",
            "file_sha256": _sha256_file(outdir / "controls_raw.npz"),
        }
    )
    _atomic_json(outdir / "controls.json", controls)
    if not controls["valid"]:
        replayed_controls, replayed_arrays = downstream_controls()
        control_base = {
            key: value
            for key, value in controls.items()
            if key not in ("binding_sha256", "path", "file_sha256")
        }
        controls_replayed = _nested_close(replayed_controls, control_base) and all(
            np.array_equal(control_arrays[name], replayed_arrays[name], equal_nan=True)
            for name in control_arrays
        )
        unavailable = {
            "schema": "bench013-stage0-full-analysis-v3",
            "binding_sha256": binding,
            "parent_binding_sha256": PARENT_BINDING,
            "parent_task_sha256": PARENT_TASK_SHA256,
            "clarified_task_sha256": TASK_SHA256,
            "complete": False,
            "artifact_audit_passed": False,
            "controls_valid": False,
            "controls_replayed": controls_replayed,
            "decision": "unavailable",
            "classification": "invalid_plumbing_controls_before_science",
            "scientific_rows_written": 0,
            "stage1_authorized": False,
            "source_archive": source_archive_record,
        }
        _atomic_json(outdir / "analysis.json", unavailable)
        _atomic_json(outdir / "artifact_manifest.json", _artifact_manifest(outdir))
        return unavailable

    parent_fields = _read_jsonl(parent / "fields.jsonl")
    if {row["field_id"] for row in parent_fields} != {
        row["field_id"] for row in early.field_specs()
    }:
        raise RuntimeError("parent field-id set mismatch")
    parent_by_field = {str(row["field_id"]): row for row in parent_fields}
    local_field_rows: list[dict[str, Any]] = []
    arrays_by_field: dict[str, dict[str, np.ndarray]] = {}
    (outdir / "input_fields").mkdir()
    with (outdir / "input_fields.jsonl").open("wb") as handle:
        for spec in early.field_specs():
            field_id = str(spec["field_id"])
            parent_row = parent_by_field[field_id]
            source = parent / parent_row["path"]
            destination = outdir / "input_fields" / f"{field_id}.npz"
            shutil.copyfile(source, destination)
            if _sha256_file(destination) != parent_row["file_sha256"]:
                raise RuntimeError(f"copied field hash mismatch: {field_id}")
            arrays = _load_field(destination)
            arrays_by_field[field_id] = arrays
            color_hashes: dict[str, Any] = {}
            for target in spec["evaluation_targets"]:
                exact = early.target_values(str(target), arrays["means"], dtype=np.float64)
                stored = np.ascontiguousarray(exact.astype(np.float32))
                if _array_record(stored) != parent_row["target_color_hashes_float32"][target]:
                    raise RuntimeError(f"target-color parent hash mismatch: {field_id}/{target}")
                color_hashes[str(target)] = {
                    "exact_float64": _array_record(exact),
                    "stored_float32": _array_record(stored),
                }
            record = {
                "schema": "bench013-stage0-full-input-field-v2",
                "binding_sha256": binding,
                **dict(spec),
                "path": destination.relative_to(outdir).as_posix(),
                "file_sha256": _sha256_file(destination),
                "parent_file_sha256": parent_row["file_sha256"],
                "geometry_sha256": parent_row["geometry_sha256"],
                "target_colors": color_hashes,
            }
            _append_jsonl(handle, record)
            local_field_rows.append(record)
        handle.flush()
        os.fsync(handle.fileno())

    cells = early.cell_specs()
    parent_cells = {
        row["cell_id"]: row for row in _read_jsonl(parent / "cells.jsonl")
    }
    if set(parent_cells) != {row["cell_id"] for row in cells}:
        raise RuntimeError("parent cell-id set mismatch")
    forward_rows: list[dict[str, Any]] = []
    forward_by_cell: dict[str, dict[str, Any]] = {}
    effective_rows: list[dict[str, Any]] = []
    effective_fields: set[str] = set()
    (outdir / "forward").mkdir()
    (outdir / "effective").mkdir()
    start_forward = time.perf_counter()
    with (outdir / "forward_cells.jsonl").open("wb") as forward_handle, (
        outdir / "effective_fields.jsonl"
    ).open("wb") as effective_handle:
        for cell in cells:
            field_id = str(cell["field_id"])
            arrays = arrays_by_field[field_id]
            identity = np.arange(int(cell["count"]), dtype=np.int64)
            mathematical, same_state, candidate, exact_colors, colors32 = _evaluate(
                arrays, str(cell["target"]), identity
            )
            raw = _forward_arrays(mathematical, same_state, candidate)
            metrics, gates = _metrics_and_gates(
                str(cell["target"]), mathematical, same_state, candidate, exact_colors, colors32
            )
            replay_metrics, replay_gates = _raw_metrics_and_gates(
                str(cell["target"]), raw, exact_colors
            )
            if gates != replay_gates or not _nested_close(metrics, replay_metrics):
                raise RuntimeError(f"in-memory forward replay mismatch: {cell['cell_id']}")
            relative = Path("forward") / f"{cell['cell_id']}.npz"
            _deterministic_npz(outdir / relative, raw)
            parent_cell = parent_cells[str(cell["cell_id"])]
            if parent_cell["analytic_colors_float64"] != _array_record(exact_colors):
                raise RuntimeError(f"parent exact-color hash mismatch: {cell['cell_id']}")
            if parent_cell["analytic_colors_float32"] != _array_record(colors32):
                raise RuntimeError(f"parent stored-color hash mismatch: {cell['cell_id']}")
            record = {
                "schema": "bench013-stage0-full-forward-cell-v2",
                "binding_sha256": binding,
                **dict(cell),
                "path": relative.as_posix(),
                "file_sha256": _sha256_file(outdir / relative),
                "pixel_count": PIXELS,
                "exact_colors_float64": _array_record(exact_colors),
                "stored_colors_float32": _array_record(colors32),
                "parent_cell_color_binding": {
                    "analytic_colors_float64": parent_cell["analytic_colors_float64"],
                    "analytic_colors_float32": parent_cell["analytic_colors_float32"],
                },
                "metrics": metrics,
                "gates": gates,
                "worst_pixels": _worst_pixels(raw, str(cell["target"])),
                "forward_pass": all(gates.values()),
            }
            _append_jsonl(forward_handle, record)
            forward_rows.append(record)
            forward_by_cell[str(cell["cell_id"])] = record

            if field_id not in effective_fields:
                effective_raw = {
                    name: raw[name]
                    for name in (
                        "mathematical_partition",
                        "mathematical_first_moment",
                        "mathematical_a1",
                        "same_state_partition",
                        "same_state_first_moment",
                        "same_state_a1",
                        "candidate_partition",
                        "candidate_first_moment",
                        "candidate_a1",
                    )
                }
                effective_relative = Path("effective") / f"{field_id}.npz"
                _deterministic_npz(outdir / effective_relative, effective_raw)
                effective_record = {
                    "schema": "bench013-stage0-full-effective-field-v2",
                    "binding_sha256": binding,
                    "field_id": field_id,
                    "representative_cell_id": cell["cell_id"],
                    "path": effective_relative.as_posix(),
                    "file_sha256": _sha256_file(outdir / effective_relative),
                    "math64": {
                        "partition_max_abs": metrics["float64_partition_max_abs"],
                        "first_moment_max_abs": metrics["float64_first_moment_max_abs"],
                        "a1_p99": metrics["float64_a1_p99"],
                        "a1_max": metrics["float64_a1_max"],
                        "moment_pass": gates["float64_partition_moment"],
                        "a1_pass": gates["float64_a1"],
                        "pass": gates["float64_partition_moment"] and gates["float64_a1"],
                    },
                    "same64_diagnostic": {
                        "partition_max_abs": float((same_state.partition - 1.0).abs().max()),
                        "first_moment_max_abs": float(same_state.first_moment.abs().max()),
                        "a1_p99": core.quantile_linear(same_state.amplification_l1, 0.99),
                        "a1_max": float(same_state.amplification_l1.max()),
                    },
                    "cand32": {
                        "partition_max_abs": metrics["float32_partition_max_abs"],
                        "first_moment_max_abs": metrics["float32_first_moment_max_abs"],
                        "a1_p99": metrics["float32_a1_p99"],
                        "a1_max": metrics["float32_a1_max"],
                        "moment_pass": gates["float32_partition_moment"],
                        "a1_pass": gates["float32_a1"],
                        "pass": gates["float32_partition_moment"] and gates["float32_a1"],
                    },
                    "effective_pass": (
                        gates["float64_partition_moment"]
                        and gates["float64_a1"]
                        and gates["float32_partition_moment"]
                        and gates["float32_a1"]
                    ),
                }
                _append_jsonl(effective_handle, effective_record)
                effective_rows.append(effective_record)
                effective_fields.add(field_id)
        forward_handle.flush()
        effective_handle.flush()
        os.fsync(forward_handle.fileno())
        os.fsync(effective_handle.fileno())
    forward_seconds = time.perf_counter() - start_forward

    permutation_rows: list[dict[str, Any]] = []
    permutation_rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED))
    (outdir / "permutations").mkdir()
    start_permutation = time.perf_counter()
    with (outdir / "permutations.jsonl").open("wb") as handle:
        for cell in cells:
            cell_id = str(cell["cell_id"])
            field_id = str(cell["field_id"])
            arrays = arrays_by_field[field_id]
            count = int(cell["count"])
            permutations = (
                np.arange(count, dtype=np.int64),
                np.arange(count - 1, -1, -1, dtype=np.int64),
                permutation_rng.permutation(count).astype(np.int64),
                permutation_rng.permutation(count).astype(np.int64),
            )
            baseline_record = forward_by_cell[cell_id]
            with np.load(
                outdir / baseline_record["path"], allow_pickle=False
            ) as baseline_npz:
                baseline_raw = {name: baseline_npz[name] for name in baseline_npz.files}
            baseline_gates = dict(baseline_record["gates"])
            order_raw: list[dict[str, np.ndarray]] = []
            order_records: list[dict[str, Any]] = []
            for order_index, (order_name, permutation) in enumerate(
                zip(PERMUTATION_ORDERS, permutations, strict=True)
            ):
                if order_index == 0:
                    raw = baseline_raw
                    metrics = dict(baseline_record["metrics"])
                    gates = baseline_gates
                else:
                    mathematical, same_state, candidate, exact_colors, colors32 = _evaluate(
                        arrays, str(cell["target"]), permutation
                    )
                    raw = _forward_arrays(mathematical, same_state, candidate)
                    metrics, gates = _metrics_and_gates(
                        str(cell["target"]),
                        mathematical,
                        same_state,
                        candidate,
                        exact_colors,
                        colors32,
                    )
                order_raw.append(raw)
                math_difference = float(
                    np.max(
                        np.abs(
                            raw["mathematical_output"]
                            - baseline_raw["mathematical_output"]
                        )
                    )
                )
                same_difference = float(
                    np.max(
                        np.abs(raw["same_state_output"] - baseline_raw["same_state_output"])
                    )
                )
                candidate_difference = float(
                    np.max(
                        np.abs(
                            raw["candidate_output"].astype(np.float64)
                            - baseline_raw["candidate_output"].astype(np.float64)
                        )
                    )
                )
                decisions_match = (
                    np.array_equal(
                        raw["mathematical_ranks"], baseline_raw["mathematical_ranks"]
                    )
                    and np.array_equal(
                        raw["mathematical_finite_pass"],
                        baseline_raw["mathematical_finite_pass"],
                    )
                    and np.array_equal(
                        raw["mathematical_mass"] >= early.MINIMUM_MASS,
                        baseline_raw["mathematical_mass"] >= early.MINIMUM_MASS,
                    )
                    and np.array_equal(
                        raw["mathematical_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
                        baseline_raw["mathematical_active_counts"]
                        >= early.MINIMUM_CONTRIBUTORS,
                    )
                    and np.array_equal(
                        raw["mathematical_condition_numbers"] <= early.MAXIMUM_CONDITION,
                        baseline_raw["mathematical_condition_numbers"]
                        <= early.MAXIMUM_CONDITION,
                    )
                    and np.array_equal(
                        raw["same_state_ranks"], baseline_raw["same_state_ranks"]
                    )
                    and np.array_equal(
                        raw["same_state_finite_pass"], baseline_raw["same_state_finite_pass"]
                    )
                    and np.array_equal(
                        raw["same_state_mass"] >= early.MINIMUM_MASS,
                        baseline_raw["same_state_mass"] >= early.MINIMUM_MASS,
                    )
                    and np.array_equal(
                        raw["same_state_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
                        baseline_raw["same_state_active_counts"]
                        >= early.MINIMUM_CONTRIBUTORS,
                    )
                    and np.array_equal(
                        raw["same_state_condition_numbers"] <= early.MAXIMUM_CONDITION,
                        baseline_raw["same_state_condition_numbers"]
                        <= early.MAXIMUM_CONDITION,
                    )
                    and np.array_equal(
                        raw["candidate_mass_pass"], baseline_raw["candidate_mass_pass"]
                    )
                    and np.array_equal(
                        raw["candidate_contributor_pass"],
                        baseline_raw["candidate_contributor_pass"],
                    )
                    and np.array_equal(
                        raw["candidate_info_pass"], baseline_raw["candidate_info_pass"]
                    )
                    and np.array_equal(
                        raw["candidate_diagonal_pass"],
                        baseline_raw["candidate_diagonal_pass"],
                    )
                    and np.array_equal(
                        raw["candidate_finite_pass"], baseline_raw["candidate_finite_pass"]
                    )
                    and np.array_equal(
                        raw["candidate_solve_pass"], baseline_raw["candidate_solve_pass"]
                    )
                    and gates == baseline_gates
                )
                permutation_pass = (
                    math_difference <= GATES["permutation_float64_max_abs"]
                    and same_difference <= GATES["permutation_float64_max_abs"]
                    and candidate_difference <= GATES["permutation_float32_max_abs"]
                    and decisions_match
                )
                row = {
                    "schema": "bench013-stage0-full-permutation-row-v2",
                    "binding_sha256": binding,
                    **dict(cell),
                    "order": order_name,
                    "order_index": order_index,
                    "permutation_sha256": _array_record(permutation)["record_sha256"],
                    "math64_max_abs_from_identity": math_difference,
                    "same64_max_abs_from_identity": same_difference,
                    "cand32_max_abs_from_identity": candidate_difference,
                    "forward_metrics": metrics,
                    "forward_gates": gates,
                    "decisions_match_identity": decisions_match,
                    "permutation_pass": permutation_pass,
                }
                order_records.append(row)
            permutation_arrays: dict[str, np.ndarray] = {
                "permutations": np.stack(permutations, axis=0)
            }
            for name in order_raw[0]:
                permutation_arrays[name] = np.stack(
                    [values[name] for values in order_raw], axis=0
                )
            relative = Path("permutations") / f"{cell_id}.npz"
            _deterministic_npz(outdir / relative, permutation_arrays)
            file_sha = _sha256_file(outdir / relative)
            for row in order_records:
                row["path"] = relative.as_posix()
                row["file_sha256"] = file_sha
                _append_jsonl(handle, row)
                permutation_rows.append(row)
        handle.flush()
        os.fsync(handle.fileno())
    permutation_seconds = time.perf_counter() - start_permutation

    gradient_cell_id = early._cell_id(  # noqa: SLF001
        "target_conditioned", "affine_xy", 56, 0
    )
    gradient_base = forward_by_cell[gradient_cell_id]
    if gradient_base["forward_pass"]:
        gradient_field = arrays_by_field[str(gradient_base["field_id"])]
        start_gradient = time.perf_counter()
        gradient, gradient_arrays = _gradient_audit(gradient_field, "affine_xy")
        gradient_seconds = time.perf_counter() - start_gradient
        gradient.update(
            {
                "binding_sha256": binding,
                "cell_id": gradient_cell_id,
                "field_id": gradient_base["field_id"],
                "status": "completed",
            }
        )
        _deterministic_npz(outdir / "gradient_raw.npz", gradient_arrays)
        gradient["path"] = "gradient_raw.npz"
        gradient["file_sha256"] = _sha256_file(outdir / "gradient_raw.npz")
    else:
        gradient_seconds = 0.0
        gradient = {
            "schema": "bench013-stage0-gradient-v2",
            "binding_sha256": binding,
            "cell_id": gradient_cell_id,
            "field_id": gradient_base["field_id"],
            "status": "not_reached_preregistered_base_forward_failure",
            "gradient_pass": False,
        }
    _atomic_json(outdir / "gradient.json", gradient)
    gradient_row_count = 0
    with (outdir / "gradient.jsonl").open("wb") as handle:
        if gradient["status"] == "completed":
            for row in gradient["directions"]:
                _append_jsonl(
                    handle,
                    {
                        "schema": "bench013-stage0-gradient-direction-row-v2",
                        "binding_sha256": binding,
                        "cell_id": gradient_cell_id,
                        "status": "completed",
                        "row_type": "direction",
                        **row,
                    },
                )
                gradient_row_count += 1
            for row in gradient["blocks"]:
                _append_jsonl(
                    handle,
                    {
                        "schema": "bench013-stage0-gradient-block-row-v2",
                        "binding_sha256": binding,
                        "cell_id": gradient_cell_id,
                        "status": "completed",
                        "row_type": "block",
                        **row,
                    },
                )
                gradient_row_count += 1
        _append_jsonl(
            handle,
            {
                "schema": "bench013-stage0-gradient-status-row-v2",
                "binding_sha256": binding,
                "cell_id": gradient_cell_id,
                "status": gradient["status"],
                "row_type": "status",
                "gradient_pass": bool(gradient["gradient_pass"]),
            },
        )
        gradient_row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    timing = {
        "diagnostic_only": True,
        "forward_seconds": forward_seconds,
        "permutation_seconds": permutation_seconds,
        "gradient_seconds": gradient_seconds,
        "gradient_jsonl_rows": gradient_row_count,
    }
    _atomic_json(outdir / "timing.json", timing)

    predecessor_path = (root / str(RECOVERY_PREDECESSOR["path"])).resolve()
    recovery_equivalence = _recovery_equivalence(
        predecessor_path, outdir, binding
    )
    recovery_equivalence_path = outdir / "recovery_equivalence.json"
    _atomic_json(recovery_equivalence_path, recovery_equivalence)
    recovery_equivalence_file_sha256 = _sha256_file(recovery_equivalence_path)

    replay = replay_full_artifacts(outdir, root, parent)
    replay["binding_sha256"] = binding
    _atomic_json(outdir / "replay.json", replay)
    complete = bool(
        replay["artifact_audit_passed"]
        and recovery_equivalence["equivalence_pass"]
    )
    if complete:
        _atomic_json(
            outdir / "completion.json",
            {
                "schema": "bench013-stage0-full-completion-v3",
                "binding_sha256": binding,
                "complete": True,
                "written_after_independent_replay": True,
                "recovery_equivalence": {
                    "path": recovery_equivalence_path.name,
                    "file_sha256": recovery_equivalence_file_sha256,
                    "equivalence_pass": True,
                },
                "counts": replay["counts"],
            },
        )

    controls_valid = bool(controls["valid"])
    if not complete:
        analysis = _outcome_free_unavailable_analysis(
            binding=binding,
            replay=replay,
            controls_valid=controls_valid,
            source_archive_record=source_archive_record,
            timing=timing,
            recovery_equivalence=recovery_equivalence,
            recovery_equivalence_file_sha256=recovery_equivalence_file_sha256,
        )
        _atomic_json(outdir / "analysis.json", analysis)
        _atomic_json(outdir / "artifact_manifest.json", _artifact_manifest(outdir))
        return analysis

    forward_pass = all(row["forward_pass"] for row in forward_rows)
    effective_pass = all(row["effective_pass"] for row in effective_rows)
    permutation_pass = all(row["permutation_pass"] for row in permutation_rows)
    gradient_pass = bool(gradient["gradient_pass"])
    scientific_pass = forward_pass and effective_pass and permutation_pass and gradient_pass
    if not controls_valid:
        decision = "unavailable"
        classification = "invalid_or_incomplete_assay"
    elif scientific_pass:
        decision = "pass_authorize_stage1"
        classification = "stage0_pass"
    else:
        decision = "kill"
        classification = "stage0_kill"
    analysis = {
        "schema": "bench013-stage0-full-analysis-v3",
        "binding_sha256": binding,
        "parent_binding_sha256": PARENT_BINDING,
        "parent_task_sha256": PARENT_TASK_SHA256,
        "clarified_task_sha256": TASK_SHA256,
        "complete": complete,
        "artifact_audit_passed": bool(replay["artifact_audit_passed"]),
        "controls_valid": controls_valid,
        "recovery_equivalence": {
            "path": recovery_equivalence_path.name,
            "file_sha256": recovery_equivalence_file_sha256,
            "equivalence_pass": True,
        },
        "outcome_ledgers_interpreted": True,
        "decision": decision,
        "classification": classification,
        "forward": {
            "pass": forward_pass,
            "failure_cells": sum(not row["forward_pass"] for row in forward_rows),
        },
        "effective_weights": {
            "pass": effective_pass,
            "failure_fields": sum(not row["effective_pass"] for row in effective_rows),
        },
        "permutation": {
            "pass": permutation_pass,
            "failure_rows": sum(not row["permutation_pass"] for row in permutation_rows),
        },
        "gradient": gradient,
        "stage1_authorized": complete and controls_valid and scientific_pass,
        "claim_boundary": {
            "quality": False,
            "convergence": False,
            "performance": False,
            "compression": False,
            "expressiveness": False,
            "method_novelty": False,
        },
        "source_archive": source_archive_record,
        "timing": timing,
    }
    _atomic_json(outdir / "analysis.json", analysis)
    _atomic_json(outdir / "artifact_manifest.json", _artifact_manifest(outdir))
    return analysis


def replay_full_artifacts(outdir: Path, root: Path, parent: Path) -> dict[str, Any]:
    """Reconstruct every downstream gate from raw arrays and verify all bindings."""

    checks: dict[str, bool] = {}
    config = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((outdir / "environment.json").read_text(encoding="utf-8"))
    binding = str(config["binding_sha256"])
    science = dict(config)
    science.pop("binding_sha256")
    try:
        predecessor_verification = verify_recovery_predecessor(root)
    except RuntimeError as error:
        predecessor_verification = {
            "schema": "bench013-stage0-recovery-predecessor-verification-v1",
            "verified": False,
            "error": str(error),
        }
    checks["config_binding"] = _binding(science, environment) == binding
    checks["current_source_hashes"] = _source_manifest(root) == config["source_manifest"]
    checks["clarified_task_hash"] = (
        _sha256_file(root / "tasks/BENCH-013-local-linear-reproducing-compositor.md")
        == TASK_SHA256
        == config["clarified_task_sha256"]
    )
    checks["lifecycle_schema"] = (
        config["schema"] == SCHEMA and config["protocol"] == PROTOCOL
    )
    checks["infrastructure_recovery"] = config.get("infrastructure_recovery") == {
        "predecessor": dict(RECOVERY_PREDECESSOR),
        "preflight_verification": predecessor_verification,
        "previous_clarified_task_sha256": PREVIOUS_CLARIFIED_TASK_SHA256,
        "lifecycle_only_changes": [
            "source_archive_metadata_scope",
            "candidate_spectral_replay_binding",
            "candidate_solve_backward_error_replay",
            "top_level_lifecycle_schema_v3",
            "predecessor_equivalence_gate",
        ],
        "scientific_protocol_unchanged": True,
        "scientific_metric_and_gate_ledgers_uninterpreted_before_repair": True,
        "incidental_status_exposure": "not_reached_preregistered_base_forward_failure",
    }
    checks["replay_validation"] = config.get("replay_validation") == {
        "candidate_eigen_rtol": 2e-6,
        "candidate_eigen_atol": 2e-8,
        "candidate_condition_from_persisted_eigenvalues": "exact_float32",
        "candidate_solve_componentwise_backward_error_eps32_multiplier": (
            REPLAY_FLOAT32_BACKWARD_ERROR_MULTIPLIER
        ),
        "recovery_equivalence": {
            "expected_npz_count": RECOVERY_EXPECTED_NPZ_COUNT,
            "npz_comparison": "recursive_relative_path_size_sha256_exact",
            "binding_normalized_json": list(RECOVERY_JSON_PATHS),
            "binding_normalized_jsonl": list(RECOVERY_JSONL_PATHS),
            "binding_normalization": "drop_top_level_binding_sha256_only",
            "parent_binding": "exact_bytes",
            "science_config_drop_keys": sorted(RECOVERY_CONFIG_DROP_KEYS),
        },
    }
    checks["predecessor_verified_in_replay"] = bool(
        predecessor_verification.get("verified")
    )
    stored_recovery_equivalence = json.loads(
        (outdir / "recovery_equivalence.json").read_text(encoding="utf-8")
    )
    replayed_recovery_equivalence = _recovery_equivalence(
        (root / str(RECOVERY_PREDECESSOR["path"])).resolve(),
        outdir,
        binding,
    )
    recovery_equivalence_audit = {
        "path": "recovery_equivalence.json",
        "file_sha256": _sha256_file(outdir / "recovery_equivalence.json"),
        "recomputed_record_sha256": _sha256_bytes(
            _canonical_json(replayed_recovery_equivalence)
        ),
        "equivalence_pass": bool(
            replayed_recovery_equivalence.get("equivalence_pass", False)
        ),
    }
    checks["recovery_equivalence"] = bool(
        _nested_close(stored_recovery_equivalence, replayed_recovery_equivalence)
        and replayed_recovery_equivalence["equivalence_pass"]
    )
    checks["parent_binding"] = verify_parent(parent) == config["parent"]
    if not all(checks.values()):
        return {
            "schema": "bench013-stage0-full-replay-v3",
            "checks": checks,
            "artifact_audit_passed": False,
            "outcome_ledgers_interpreted": False,
            "recovery_equivalence": recovery_equivalence_audit,
        }

    stored_controls = json.loads((outdir / "controls.json").read_text(encoding="utf-8"))
    replayed_controls, replayed_control_arrays = downstream_controls()
    control_path = outdir / stored_controls["path"]
    control_ok = _sha256_file(control_path) == stored_controls["file_sha256"]
    with np.load(control_path, allow_pickle=False) as archive:
        stored_control_arrays = {name: archive[name] for name in archive.files}
    control_ok &= stored_control_arrays.keys() == replayed_control_arrays.keys()
    control_ok &= all(
        np.array_equal(stored_control_arrays[name], replayed_control_arrays[name], equal_nan=True)
        for name in replayed_control_arrays
    )
    control_payload = dict(stored_controls)
    control_payload.pop("binding_sha256")
    control_payload.pop("path")
    control_payload.pop("file_sha256")
    checks["plumbing_controls"] = bool(
        control_ok
        and stored_controls["binding_sha256"] == binding
        and _nested_close(control_payload, _json_safe(replayed_controls))
        and bool(stored_controls["valid"])
    )

    with np.load(outdir / "targets.npz", allow_pickle=False) as archive:
        target_arrays = {name: archive[name] for name in archive.files}
    expected_targets: dict[str, np.ndarray] = {}
    for target in early.TARGETS:
        expected_targets[f"{target}__float64"] = early.target_image(target, dtype=np.float64)
        expected_targets[f"{target}__float32"] = early.target_image(target, dtype=np.float32)
    checks["target_arrays"] = (
        expected_targets.keys() == target_arrays.keys()
        and all(np.array_equal(expected_targets[name], target_arrays[name]) for name in target_arrays)
        and {
            name: _array_record(value) for name, value in sorted(target_arrays.items())
        }
        == config["target_arrays"]
    )
    parent_config = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    expected_formulas = {
        target: {
            "formula": early.TARGET_FORMULAS[target],
            "formula_sha256": _sha256_bytes(early.TARGET_FORMULAS[target].encode("utf-8")),
            "parent_formula_sha256": parent_config["targets"][target]["formula_sha256"],
        }
        for target in early.TARGETS
    }
    checks["target_formulas"] = (
        config["target_formulas"] == expected_formulas
        and all(
            record["formula_sha256"] == record["parent_formula_sha256"]
            for record in expected_formulas.values()
        )
    )

    parent_fields = {row["field_id"]: row for row in _read_jsonl(parent / "fields.jsonl")}
    input_rows = _read_jsonl(outdir / "input_fields.jsonl")
    expected_fields = {row["field_id"]: row for row in early.field_specs()}
    input_ok = len(input_rows) == LOGICAL_FIELDS
    input_arrays: dict[str, dict[str, np.ndarray]] = {}
    for row in input_rows:
        field_id = str(row["field_id"])
        expected = expected_fields.get(field_id)
        parent_row = parent_fields.get(field_id)
        input_ok &= expected is not None and parent_row is not None
        input_ok &= row["binding_sha256"] == binding
        if expected is not None:
            input_ok &= all(row[key] == value for key, value in expected.items())
        path = outdir / row["path"]
        input_ok &= _sha256_file(path) == row["file_sha256"]
        if parent_row is not None:
            input_ok &= row["file_sha256"] == parent_row["file_sha256"]
            input_ok &= row["geometry_sha256"] == parent_row["geometry_sha256"]
        arrays = _load_field(path)
        input_arrays[field_id] = arrays
        if expected is not None:
            input_ok &= set(row["target_colors"]) == set(expected["evaluation_targets"])
            for target in expected["evaluation_targets"]:
                exact = early.target_values(str(target), arrays["means"], dtype=np.float64)
                stored = exact.astype(np.float32)
                input_ok &= row["target_colors"][target] == {
                    "exact_float64": _array_record(exact),
                    "stored_float32": _array_record(stored),
                }
    checks["input_fields"] = bool(
        input_ok and {row["field_id"] for row in input_rows} == set(expected_fields)
    )

    expected_cells = {row["cell_id"]: row for row in early.cell_specs()}
    parent_cells = {row["cell_id"]: row for row in _read_jsonl(parent / "cells.jsonl")}
    forward_rows = _read_jsonl(outdir / "forward_cells.jsonl")
    forward_ok = len(forward_rows) == FORWARD_CELLS
    replay_forward: dict[str, tuple[dict[str, Any], dict[str, bool], dict[str, np.ndarray]]] = {}
    for row in forward_rows:
        cell_id = str(row["cell_id"])
        expected = expected_cells.get(cell_id)
        forward_ok &= expected is not None and row["binding_sha256"] == binding
        if expected is not None:
            forward_ok &= all(row[key] == value for key, value in expected.items())
        path = outdir / row["path"]
        forward_ok &= _sha256_file(path) == row["file_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        expected_shapes = _expected_forward_shapes()
        forward_ok &= set(raw) == set(expected_shapes)
        forward_ok &= all(
            raw[name].shape == shape for name, shape in expected_shapes.items()
        )
        arrays = input_arrays[str(row["field_id"])]
        exact_colors = early.target_values(str(row["target"]), arrays["means"], dtype=np.float64)
        stored_colors = exact_colors.astype(np.float32)
        forward_ok &= row["exact_colors_float64"] == _array_record(exact_colors)
        forward_ok &= row["stored_colors_float32"] == _array_record(stored_colors)
        forward_ok &= row["parent_cell_color_binding"] == {
            "analytic_colors_float64": parent_cells[cell_id]["analytic_colors_float64"],
            "analytic_colors_float32": parent_cells[cell_id]["analytic_colors_float32"],
        }
        forward_ok &= parent_cells[cell_id]["analytic_colors_float64"] == _array_record(
            exact_colors
        )
        forward_ok &= parent_cells[cell_id]["analytic_colors_float32"] == _array_record(
            stored_colors
        )
        metrics, gates = _raw_metrics_and_gates(str(row["target"]), raw, exact_colors)
        forward_ok &= _nested_close(row["metrics"], _json_safe(metrics))
        forward_ok &= row["gates"] == gates
        forward_ok &= _nested_close(
            row["worst_pixels"], _json_safe(_worst_pixels(raw, str(row["target"])))
        )
        forward_ok &= bool(row["forward_pass"]) == all(gates.values())
        forward_ok &= _candidate_mask_contract(raw)
        forward_ok &= _candidate_eigenvalue_contract(raw)
        singular = raw["mathematical_singular_values"]
        active = raw["mathematical_active_counts"]
        thresholds = (
            np.maximum(active, 3) * np.finfo(np.float64).eps * singular[:, 0]
        )
        forward_ok &= np.array_equal(raw["mathematical_ranks"], (singular > thresholds[:, None]).sum(1))
        forward_ok &= np.allclose(
            raw["mathematical_rank_thresholds"], thresholds, rtol=2e-15, atol=0.0
        )
        same_singular = raw["same_state_singular_values"]
        same_thresholds = (
            np.maximum(active, 3) * np.finfo(np.float64).eps * same_singular[:, 0]
        )
        forward_ok &= np.array_equal(
            raw["same_state_ranks"], (same_singular > same_thresholds[:, None]).sum(1)
        )
        forward_ok &= np.allclose(
            raw["same_state_rank_thresholds"], same_thresholds, rtol=2e-15, atol=0.0
        )
        forward_ok &= np.array_equal(
            raw["candidate_info_pass"], raw["candidate_cholesky_info"] == 0
        )
        forward_ok &= np.array_equal(raw["candidate_mass"], raw["candidate_accumulators"][:, 0])
        accumulators = raw["candidate_accumulators"]
        mass = accumulators[:, 0]
        offset_mean = accumulators[:, 1:3] / mass[:, None]
        second = np.stack(
            (
                np.stack((accumulators[:, 3], accumulators[:, 4]), axis=1),
                np.stack((accumulators[:, 4], accumulators[:, 5]), axis=1),
            ),
            axis=1,
        ) / mass[:, None, None]
        covariance = second - offset_mean[:, :, None] * offset_mean[:, None, :]
        covariance = 0.5 * (covariance + np.swapaxes(covariance, 1, 2))
        color_mean = accumulators[:, 6:9] / mass[:, None]
        cross = np.stack((accumulators[:, 9:12], accumulators[:, 12:15]), axis=1)
        cross = cross / mass[:, None, None] - offset_mean[:, :, None] * color_mean[:, None, :]
        forward_ok &= np.allclose(raw["candidate_offset_mean"], offset_mean, rtol=1e-6, atol=1e-8)
        forward_ok &= np.allclose(raw["candidate_color_mean"], color_mean, rtol=1e-6, atol=1e-8)
        forward_ok &= np.allclose(raw["candidate_covariance"], covariance, rtol=1e-6, atol=1e-8)
        forward_ok &= np.allclose(raw["candidate_cross_covariance"], cross, rtol=1e-6, atol=1e-8)
        forward_ok &= np.array_equal(
            raw["candidate_mass_pass"], raw["candidate_mass"] >= early.MINIMUM_MASS
        )
        forward_ok &= np.array_equal(
            raw["candidate_contributor_pass"],
            raw["candidate_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
        )
        forward_ok &= np.array_equal(
            raw["mathematical_active_counts"], raw["same_state_active_counts"]
        ) and np.array_equal(
            raw["mathematical_active_counts"], raw["candidate_active_counts"]
        )
        for prefix in ("mathematical", "same_state", "candidate"):
            eigenvalues = raw[f"{prefix}_covariance_eigenvalues"]
            expected_condition = np.where(
                eigenvalues[:, 0] > 0.0,
                eigenvalues[:, 1] / eigenvalues[:, 0],
                np.asarray(np.inf, dtype=eigenvalues.dtype),
            )
            if prefix == "candidate":
                forward_ok &= np.array_equal(
                    raw["candidate_condition_numbers"],
                    expected_condition,
                    equal_nan=True,
                )
            else:
                forward_ok &= np.allclose(
                    raw[f"{prefix}_condition_numbers"],
                    expected_condition,
                    rtol=2e-14,
                    atol=0.0,
                    equal_nan=True,
                )
        forward_ok &= np.array_equal(
            raw["candidate_diagonal_pass"],
            np.all(np.diagonal(raw["candidate_cholesky"], axis1=1, axis2=2) > 0.0, axis=1),
        )
        forward_ok &= np.array_equal(
            raw["candidate_solve_pass"],
            raw["candidate_info_pass"]
            & raw["candidate_diagonal_pass"]
            & raw["candidate_finite_pass"],
        )
        solved = np.flatnonzero(raw["candidate_solve_pass"])
        if solved.size:
            factor = raw["candidate_cholesky"][solved]
            solved_covariance = raw["candidate_covariance"][solved]
            solved_slopes = raw["candidate_solved_slopes"][solved]
            solved_inverse_mean = raw["candidate_inverse_mean"][solved]
            forward_ok &= np.allclose(
                factor @ np.swapaxes(factor, 1, 2),
                solved_covariance,
                rtol=2e-5,
                atol=2e-8,
            )
            forward_ok &= _componentwise_backward_error_contract(
                solved_covariance,
                solved_slopes,
                raw["candidate_cross_covariance"][solved],
            )
            forward_ok &= _componentwise_backward_error_contract(
                solved_covariance,
                solved_inverse_mean,
                raw["candidate_offset_mean"][solved],
            )
            reconstructed_output = raw["candidate_color_mean"][solved] - np.einsum(
                "pi,pic->pc",
                raw["candidate_offset_mean"][solved],
                solved_slopes,
            )
            forward_ok &= np.allclose(
                raw["candidate_output"].reshape(PIXELS, 3)[solved],
                reconstructed_output,
                rtol=2e-5,
                atol=2e-7,
            )
        mathematical_finite = (
            np.isfinite(raw["mathematical_output"]).reshape(PIXELS, -1).all(axis=1)
            & np.isfinite(raw["mathematical_mass"])
            & np.isfinite(raw["mathematical_singular_values"]).all(axis=1)
            & np.isfinite(raw["mathematical_covariance_eigenvalues"]).all(axis=1)
            & np.isfinite(raw["mathematical_partition"])
            & np.isfinite(raw["mathematical_first_moment"]).all(axis=1)
            & np.isfinite(raw["mathematical_a1"])
        )
        same_finite = (
            np.isfinite(raw["same_state_output"]).reshape(PIXELS, -1).all(axis=1)
            & np.isfinite(raw["same_state_mass"])
            & np.isfinite(raw["same_state_singular_values"]).all(axis=1)
            & np.isfinite(raw["same_state_covariance_eigenvalues"]).all(axis=1)
            & np.isfinite(raw["same_state_partition"])
            & np.isfinite(raw["same_state_first_moment"]).all(axis=1)
            & np.isfinite(raw["same_state_a1"])
        )
        candidate_finite = (
            np.isfinite(raw["candidate_accumulators"]).all(axis=1)
            & np.isfinite(raw["candidate_offset_mean"]).all(axis=1)
            & np.isfinite(raw["candidate_covariance"]).all(axis=(1, 2))
            & np.isfinite(raw["candidate_color_mean"]).all(axis=1)
            & np.isfinite(raw["candidate_cross_covariance"]).all(axis=(1, 2))
            & np.isfinite(raw["candidate_cholesky"]).all(axis=(1, 2))
            & np.isfinite(raw["candidate_solved_slopes"]).all(axis=(1, 2))
            & np.isfinite(raw["candidate_inverse_mean"]).all(axis=1)
            & np.isfinite(raw["candidate_output"]).reshape(PIXELS, -1).all(axis=1)
            & np.isfinite(raw["candidate_partition"])
            & np.isfinite(raw["candidate_first_moment"]).all(axis=1)
            & np.isfinite(raw["candidate_a1"])
        )
        forward_ok &= np.array_equal(
            raw["mathematical_finite_pass"], mathematical_finite
        )
        forward_ok &= np.array_equal(raw["same_state_finite_pass"], same_finite)
        forward_ok &= np.array_equal(raw["candidate_finite_pass"], candidate_finite)
        forward_ok &= int(row["pixel_count"]) == PIXELS
        replay_forward[cell_id] = (metrics, gates, raw)
    checks["forward_raw_replay"] = bool(
        forward_ok and {row["cell_id"] for row in forward_rows} == set(expected_cells)
    )

    effective_rows = _read_jsonl(outdir / "effective_fields.jsonl")
    effective_ok = len(effective_rows) == LOGICAL_FIELDS
    expected_representatives: dict[str, str] = {}
    for cell in early.cell_specs():
        expected_representatives.setdefault(str(cell["field_id"]), str(cell["cell_id"]))
    effective_keys = {
        "mathematical_partition",
        "mathematical_first_moment",
        "mathematical_a1",
        "same_state_partition",
        "same_state_first_moment",
        "same_state_a1",
        "candidate_partition",
        "candidate_first_moment",
        "candidate_a1",
    }
    for row in effective_rows:
        field_id = str(row["field_id"])
        effective_ok &= field_id in expected_fields and row["binding_sha256"] == binding
        path = outdir / row["path"]
        effective_ok &= _sha256_file(path) == row["file_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        effective_ok &= row["representative_cell_id"] == expected_representatives[field_id]
        representative_raw = replay_forward[str(row["representative_cell_id"])][2]
        effective_ok &= _effective_arrays_match(raw, representative_raw, effective_keys)
        math_record = {
            "partition_max_abs": float(np.max(np.abs(raw["mathematical_partition"] - 1.0))),
            "first_moment_max_abs": float(np.max(np.abs(raw["mathematical_first_moment"]))),
            "a1_p99": float(np.quantile(raw["mathematical_a1"], 0.99, method="linear")),
            "a1_max": float(np.max(raw["mathematical_a1"])),
        }
        math_record["moment_pass"] = (
            math_record["partition_max_abs"] <= GATES["float64_partition_and_moment"]
            and math_record["first_moment_max_abs"] <= GATES["float64_partition_and_moment"]
        )
        math_record["a1_pass"] = (
            math_record["a1_p99"] <= GATES["a1_p99"]
            and math_record["a1_max"] <= GATES["a1_max"]
        )
        math_record["pass"] = math_record["moment_pass"] and math_record["a1_pass"]
        same_record = {
            "partition_max_abs": float(np.max(np.abs(raw["same_state_partition"] - 1.0))),
            "first_moment_max_abs": float(np.max(np.abs(raw["same_state_first_moment"]))),
            "a1_p99": float(np.quantile(raw["same_state_a1"], 0.99, method="linear")),
            "a1_max": float(np.max(raw["same_state_a1"])),
        }
        cand_record = {
            "partition_max_abs": float(
                np.max(np.abs(raw["candidate_partition"] - np.float32(1.0)))
            ),
            "first_moment_max_abs": float(np.max(np.abs(raw["candidate_first_moment"]))),
            "a1_p99": float(np.quantile(raw["candidate_a1"], 0.99, method="linear")),
            "a1_max": float(np.max(raw["candidate_a1"])),
        }
        cand_record["moment_pass"] = (
            cand_record["partition_max_abs"] <= GATES["float32_partition_and_moment"]
            and cand_record["first_moment_max_abs"] <= GATES["float32_partition_and_moment"]
        )
        cand_record["a1_pass"] = (
            cand_record["a1_p99"] <= GATES["a1_p99"]
            and cand_record["a1_max"] <= GATES["a1_max"]
        )
        cand_record["pass"] = cand_record["moment_pass"] and cand_record["a1_pass"]
        effective_ok &= _nested_close(row["math64"], _json_safe(math_record))
        effective_ok &= _nested_close(row["same64_diagnostic"], _json_safe(same_record))
        effective_ok &= _nested_close(row["cand32"], _json_safe(cand_record))
        effective_ok &= bool(row["effective_pass"]) == (
            math_record["pass"] and cand_record["pass"]
        )
    checks["effective_raw_replay"] = bool(
        effective_ok and {row["field_id"] for row in effective_rows} == set(expected_fields)
    )

    permutation_rows = _read_jsonl(outdir / "permutations.jsonl")
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in permutation_rows:
        rows_by_cell.setdefault(str(row["cell_id"]), []).append(row)
    rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED))
    permutation_ok = len(permutation_rows) == FORWARD_CELLS * len(PERMUTATION_ORDERS)
    for cell_id in sorted(expected_cells):
        cell = expected_cells[cell_id]
        count = int(cell["count"])
        expected_permutations = (
            np.arange(count, dtype=np.int64),
            np.arange(count - 1, -1, -1, dtype=np.int64),
            rng.permutation(count).astype(np.int64),
            rng.permutation(count).astype(np.int64),
        )
        rows = sorted(rows_by_cell.get(cell_id, []), key=lambda value: value["order_index"])
        permutation_ok &= len(rows) == len(PERMUTATION_ORDERS)
        if len(rows) != len(PERMUTATION_ORDERS):
            continue
        path = outdir / rows[0]["path"]
        file_sha = _sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            stacked = {name: archive[name] for name in archive.files}
        expected_stacked_shapes = {
            name: (len(PERMUTATION_ORDERS), *shape)
            for name, shape in _expected_forward_shapes().items()
        }
        expected_stacked_shapes["permutations"] = (len(PERMUTATION_ORDERS), count)
        permutation_ok &= set(stacked) == set(expected_stacked_shapes)
        permutation_ok &= all(
            stacked[name].shape == shape
            for name, shape in expected_stacked_shapes.items()
        )
        permutation_ok &= np.array_equal(
            stacked["permutations"], np.stack(expected_permutations, axis=0)
        )
        baseline = replay_forward[cell_id][2]
        exact_colors = early.target_values(
            str(cell["target"]), input_arrays[str(cell["field_id"])]["means"], dtype=np.float64
        )
        for index, row in enumerate(rows):
            permutation_ok &= _permutation_row_identity_contract(
                row, index, cell, binding
            )
            permutation_ok &= row["path"] == rows[0]["path"]
            permutation_ok &= row["file_sha256"] == file_sha
            permutation_ok &= row["permutation_sha256"] == _array_record(
                expected_permutations[index]
            )["record_sha256"]
            raw = {
                name: value[index] for name, value in stacked.items() if name != "permutations"
            }
            metrics, gates = _raw_metrics_and_gates(str(cell["target"]), raw, exact_colors)
            permutation_ok &= _nested_close(row["forward_metrics"], _json_safe(metrics))
            permutation_ok &= row["forward_gates"] == gates
            math_difference = float(
                np.max(np.abs(raw["mathematical_output"] - baseline["mathematical_output"]))
            )
            same_difference = float(
                np.max(np.abs(raw["same_state_output"] - baseline["same_state_output"]))
            )
            candidate_difference = float(
                np.max(
                    np.abs(
                        raw["candidate_output"].astype(np.float64)
                        - baseline["candidate_output"].astype(np.float64)
                    )
                )
            )
            decisions_match = (
                np.array_equal(raw["mathematical_ranks"], baseline["mathematical_ranks"])
                and np.array_equal(
                    raw["mathematical_finite_pass"], baseline["mathematical_finite_pass"]
                )
                and np.array_equal(
                    raw["mathematical_mass"] >= early.MINIMUM_MASS,
                    baseline["mathematical_mass"] >= early.MINIMUM_MASS,
                )
                and np.array_equal(
                    raw["mathematical_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
                    baseline["mathematical_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
                )
                and np.array_equal(
                    raw["mathematical_condition_numbers"] <= early.MAXIMUM_CONDITION,
                    baseline["mathematical_condition_numbers"] <= early.MAXIMUM_CONDITION,
                )
                and np.array_equal(raw["same_state_ranks"], baseline["same_state_ranks"])
                and np.array_equal(
                    raw["same_state_finite_pass"], baseline["same_state_finite_pass"]
                )
                and np.array_equal(
                    raw["same_state_mass"] >= early.MINIMUM_MASS,
                    baseline["same_state_mass"] >= early.MINIMUM_MASS,
                )
                and np.array_equal(
                    raw["same_state_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
                    baseline["same_state_active_counts"] >= early.MINIMUM_CONTRIBUTORS,
                )
                and np.array_equal(
                    raw["same_state_condition_numbers"] <= early.MAXIMUM_CONDITION,
                    baseline["same_state_condition_numbers"] <= early.MAXIMUM_CONDITION,
                )
                and np.array_equal(raw["candidate_mass_pass"], baseline["candidate_mass_pass"])
                and np.array_equal(
                    raw["candidate_contributor_pass"], baseline["candidate_contributor_pass"]
                )
                and np.array_equal(raw["candidate_info_pass"], baseline["candidate_info_pass"])
                and np.array_equal(
                    raw["candidate_diagonal_pass"], baseline["candidate_diagonal_pass"]
                )
                and np.array_equal(
                    raw["candidate_finite_pass"], baseline["candidate_finite_pass"]
                )
                and np.array_equal(raw["candidate_solve_pass"], baseline["candidate_solve_pass"])
                and gates == replay_forward[cell_id][1]
            )
            passed = (
                math_difference <= GATES["permutation_float64_max_abs"]
                and same_difference <= GATES["permutation_float64_max_abs"]
                and candidate_difference <= GATES["permutation_float32_max_abs"]
                and decisions_match
            )
            permutation_ok &= _nested_close(
                row["math64_max_abs_from_identity"], _json_safe(math_difference)
            )
            permutation_ok &= _nested_close(
                row["same64_max_abs_from_identity"], _json_safe(same_difference)
            )
            permutation_ok &= _nested_close(
                row["cand32_max_abs_from_identity"], _json_safe(candidate_difference)
            )
            permutation_ok &= bool(row["decisions_match_identity"]) == decisions_match
            permutation_ok &= bool(row["permutation_pass"]) == passed
    checks["permutation_raw_replay"] = bool(
        permutation_ok and set(rows_by_cell) == set(expected_cells)
    )

    gradient = json.loads((outdir / "gradient.json").read_text(encoding="utf-8"))
    selected = early._cell_id("target_conditioned", "affine_xy", 56, 0)  # noqa: SLF001
    selected_field_id = str(expected_cells[selected]["field_id"])
    gradient_ok = (
        gradient["cell_id"] == selected
        and gradient["field_id"] == selected_field_id
        and gradient["binding_sha256"] == binding
    )
    if replay_forward[selected][1] and all(replay_forward[selected][1].values()):
        gradient_ok &= gradient["status"] == "completed"
        path = outdir / gradient["path"]
        gradient_ok &= _sha256_file(path) == gradient["file_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            raw_gradient = {name: archive[name] for name in archive.files}
        gradient_ok &= _gradient_raw_shape_contract(raw_gradient, 56)
        expected_gradient_shapes = {
            "means_over_l": (56, 2),
            "log_scales": (56, 2),
            "rotations_over_pi": (56,),
            "colors": (56, 3),
        }
        gradient_ok &= raw_gradient["directional_loss_plus64"].shape == (8,)
        gradient_ok &= raw_gradient["directional_loss_minus64"].shape == (8,)
        for block, block_shape in expected_gradient_shapes.items():
            gradient_ok &= raw_gradient[f"directions__{block}"].shape == (2, *block_shape)
            gradient_ok &= raw_gradient[f"gradient64__{block}"].shape == block_shape
            gradient_ok &= raw_gradient[f"gradient32__{block}"].shape == block_shape
            gradient_ok &= raw_gradient[f"raw_gradient64__{block}"].shape == block_shape
            gradient_ok &= raw_gradient[f"raw_gradient32__{block}"].shape == block_shape
        gradient_rng = np.random.Generator(np.random.PCG64(GRADIENT_SEED))
        cotangent = gradient_rng.standard_normal((HEIGHT, WIDTH, 3))
        cotangent /= np.linalg.norm(cotangent.reshape(-1))
        gradient_ok &= np.array_equal(raw_gradient["cotangent_float64"], cotangent)
        gradient_ok &= np.array_equal(
            raw_gradient["cotangent_float32"], cotangent.astype(np.float32)
        )
        gradient_ok &= int(gradient["seed"]) == GRADIENT_SEED
        gradient_ok &= _nested_close(gradient["step"], GRADIENT_STEP)
        gradient_ok &= gradient["coordinate_blocks"] == list(GRADIENT_BLOCKS)
        gradient_ok &= _nested_close(
            gradient["cotangent_l2_float64"], float(np.linalg.norm(cotangent.reshape(-1)))
        )
        selected_arrays = input_arrays[selected_field_id]
        expected_colors = early.target_values(
            "affine_xy", selected_arrays["means"], dtype=np.float64
        ).astype(np.float32)
        gradient_ok &= np.array_equal(raw_gradient["base_means_float32"], selected_arrays["means"])
        gradient_ok &= np.array_equal(
            raw_gradient["base_log_scales_float32"], selected_arrays["log_scales"]
        )
        gradient_ok &= np.array_equal(
            raw_gradient["base_rotations_float32"], selected_arrays["rotations"]
        )
        gradient_ok &= np.array_equal(raw_gradient["base_colors_float32"], expected_colors)
        gradient_ok &= np.array_equal(raw_gradient["frozen_radii"], selected_arrays["radii"])
        reconstructed = core.enumerate_contributors(
            _tensor(selected_arrays["means"], torch.float32),
            torch.from_numpy(selected_arrays["radii"]).to(torch.int64),
            HEIGHT,
            WIDTH,
        )
        gradient_ok &= np.array_equal(raw_gradient["contributors_gid"], reconstructed.gid.numpy())
        gradient_ok &= np.array_equal(raw_gradient["contributors_px"], reconstructed.px.numpy())
        gradient_ok &= np.array_equal(raw_gradient["contributors_py"], reconstructed.py.numpy())
        gradient_ok &= len(gradient["blocks"]) == 4
        gradient_ok &= {row["block"] for row in gradient["blocks"]} == set(GRADIENT_BLOCKS)
        gradient_ok &= len(gradient["directions"]) == 8
        gradient_ok &= [
            (row["block"], int(row["direction"])) for row in gradient["directions"]
        ] == [(block, index) for block in GRADIENT_BLOCKS for index in range(2)]
        block_pass = True
        expected_direction_arrays: dict[str, np.ndarray] = {}
        for block in GRADIENT_BLOCKS:
            expected_directions = []
            for _ in range(2):
                direction = gradient_rng.standard_normal(expected_gradient_shapes[block])
                direction /= np.linalg.norm(direction.reshape(-1))
                expected_directions.append(direction)
            gradient_ok &= np.array_equal(
                raw_gradient[f"directions__{block}"], np.stack(expected_directions)
            )
            expected_direction_arrays[block] = np.stack(expected_directions)
            raw64 = raw_gradient[f"raw_gradient64__{block}"]
            raw32 = raw_gradient[f"raw_gradient32__{block}"]
            g64 = raw_gradient[f"gradient64__{block}"]
            g32 = raw_gradient[f"gradient32__{block}"].astype(np.float64)
            scale = (
                float(max(HEIGHT - 1, WIDTH - 1, 1))
                if block == "means_over_l"
                else math.pi
                if block == "rotations_over_pi"
                else 1.0
            )
            gradient_ok &= np.array_equal(g64, raw64 * scale)
            gradient_ok &= np.array_equal(
                raw_gradient[f"gradient32__{block}"], raw32 * np.float32(scale)
            )
            norm64 = float(np.linalg.norm(g64))
            absolute_error = float(np.linalg.norm(g32 - g64))
            relative = absolute_error / max(norm64, 1e-12)
            tolerance = GATES["gradient_full_abs"] + GATES["gradient_full_rel"] * norm64
            row = next(value for value in gradient["blocks"] if value["block"] == block)
            passed = (
                np.isfinite(g64).all()
                and np.isfinite(g32).all()
                and absolute_error <= tolerance
            )
            block_pass &= passed
            gradient_ok &= _nested_close(row["gradient64_l2"], norm64)
            gradient_ok &= _nested_close(row["gradient32_l2"], float(np.linalg.norm(g32)))
            gradient_ok &= _nested_close(row["absolute_l2_error"], absolute_error)
            gradient_ok &= _nested_close(row["mixed_l2_tolerance"], tolerance)
            gradient_ok &= _nested_close(row["relative_l2_error"], relative)
            gradient_ok &= bool(row["finite"]) == bool(
                np.isfinite(g64).all() and np.isfinite(g32).all()
            )
            gradient_ok &= bool(row["pass"]) == passed
        directional_pass = True
        for row_index, row in enumerate(gradient["directions"]):
            block = str(row["block"])
            direction_index = int(row["direction"])
            direction = expected_direction_arrays[block][direction_index]
            ad64 = float(np.sum(raw_gradient[f"gradient64__{block}"] * direction))
            ad32 = float(
                np.sum(
                    raw_gradient[f"gradient32__{block}"].astype(np.float64) * direction
                )
            )
            plus_loss = float(raw_gradient["directional_loss_plus64"][row_index])
            minus_loss = float(raw_gradient["directional_loss_minus64"][row_index])
            fd64 = (plus_loss - minus_loss) / (2.0 * GRADIENT_STEP)
            gradient_ok &= _nested_close(row["ad64"], _json_safe(ad64))
            gradient_ok &= _nested_close(row["ad32"], _json_safe(ad32))
            gradient_ok &= _nested_close(row["fd64"], _json_safe(fd64))
            gradient_ok &= _nested_close(row["loss_plus64"], _json_safe(plus_loss))
            gradient_ok &= _nested_close(row["loss_minus64"], _json_safe(minus_loss))
            fd_error = abs(ad64 - fd64)
            fd_tolerance = GATES["gradient_fd_abs"] + GATES["gradient_fd_rel"] * max(
                abs(ad64), abs(fd64)
            )
            cross_error = abs(ad32 - ad64)
            cross_tolerance = GATES["gradient_cross_abs"] + GATES["gradient_cross_rel"] * abs(ad64)
            passed = (
                all(math.isfinite(value) for value in (ad64, fd64, ad32))
                and fd_error <= fd_tolerance
                and cross_error <= cross_tolerance
            )
            directional_pass &= passed
            gradient_ok &= _nested_close(row["ad64_fd64_abs_error"], fd_error)
            gradient_ok &= _nested_close(row["ad64_fd64_tolerance"], fd_tolerance)
            gradient_ok &= _nested_close(row["ad32_ad64_abs_error"], cross_error)
            gradient_ok &= _nested_close(row["ad32_ad64_tolerance"], cross_tolerance)
            gradient_ok &= bool(row["pass"]) == passed
        gradient_ok &= bool(gradient["gradient_pass"]) == (
            block_pass and directional_pass
        )
        gradient_ok &= bool(gradient["block_pass"]) == block_pass
        gradient_ok &= bool(gradient["directional_pass"]) == directional_pass
    else:
        gradient_ok &= (
            gradient["status"] == "not_reached_preregistered_base_forward_failure"
        )
    gradient_jsonl = _read_jsonl(outdir / "gradient.jsonl")
    if gradient["status"] == "completed":
        gradient_ok &= len(gradient_jsonl) == 13
        gradient_ok &= [row["row_type"] for row in gradient_jsonl] == (
            ["direction"] * 8 + ["block"] * 4 + ["status"]
        )
        for index, row in enumerate(gradient_jsonl[:8]):
            for key, value in gradient["directions"][index].items():
                gradient_ok &= _nested_close(row[key], value)
        for index, row in enumerate(gradient_jsonl[8:12]):
            for key, value in gradient["blocks"][index].items():
                gradient_ok &= _nested_close(row[key], value)
    else:
        gradient_ok &= len(gradient_jsonl) == 1
        gradient_ok &= gradient_jsonl[0]["row_type"] == "status"
    gradient_ok &= _gradient_jsonl_metadata_contract(
        gradient_jsonl, gradient, binding, selected
    )
    checks["gradient_raw_replay"] = bool(gradient_ok)

    source = json.loads((outdir / "source_archive.json").read_text(encoding="utf-8"))
    archive_path = outdir / source["path"]
    archived_hashes: dict[str, str] = {}
    with tarfile.open(archive_path, mode="r") as archive:
        for member in archive.getmembers():
            extracted = archive.extractfile(member)
            if extracted is not None:
                archived_hashes[member.name] = _sha256_bytes(extracted.read())
    checks["executed_source_archive"] = (
        _sha256_file(archive_path) == source["sha256"]
        and archived_hashes == source["members"]
        and all(
            archived_hashes.get(relative) == digest
            for relative, digest in config["source_manifest"].items()
        )
    )
    counts = {
        "logical_fields": len(input_rows),
        "forward_cells": len(forward_rows),
        "forward_pixels": sum(int(row["pixel_count"]) for row in forward_rows),
        "effective_fields": len(effective_rows),
        "permutation_rows": len(permutation_rows),
        "gradient_cells": 1,
        "gradient_rows": len(gradient_jsonl),
    }
    checks["exact_counts"] = counts == {
        "logical_fields": LOGICAL_FIELDS,
        "forward_cells": FORWARD_CELLS,
        "forward_pixels": FORWARD_CELLS * PIXELS,
        "effective_fields": LOGICAL_FIELDS,
        "permutation_rows": FORWARD_CELLS * len(PERMUTATION_ORDERS),
        "gradient_cells": 1,
        "gradient_rows": 13 if gradient["status"] == "completed" else 1,
    }
    return {
        "schema": "bench013-stage0-full-replay-v3",
        "checks": checks,
        "artifact_audit_passed": all(checks.values()),
        "outcome_ledgers_interpreted": True,
        "recovery_equivalence": recovery_equivalence_audit,
        "counts": counts,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--parent-outdir",
        type=Path,
        default=Path("results/bench013_local_linear_stage0_early_2026-07-16"),
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv)
    analysis = run(
        arguments.outdir.resolve(),
        arguments.root.resolve(),
        arguments.parent_outdir.resolve(),
    )
    print(json.dumps(_json_safe(analysis), indent=2, sort_keys=True, allow_nan=False))
    if analysis["decision"] == "unavailable":
        return 3
    if analysis["decision"] == "kill":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
