"""Resumable parent and candidate-selection scaffold for BENCH-009 Stage 1.

This runner stops before tangent-range construction, the frozen action grid, recovery, gates,
or scientific interpretation.  ``parents`` creates exact no-opacity N=64 checkpoints only after
validating a supplied Stage-1 preflight artifact.  ``candidate-smoke`` exercises discovery-only
affine, carrier, and exact two-birth selectors on saved parents.  Neither phase authorizes a
development result, method promotion, rate claim, or compression claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from benchmarks import residual_tangent_auction_stage1 as manifest_module
from benchmarks.residual_tangent_auction_stage1_core import (
    BirthGeometry,
    SixCoefficientEvaluation,
    build_two_birth_packet,
    checkerboard_fold_masks,
    evaluate_six_coefficient_action,
    normalized_responsibilities,
    render_normalized,
)
from structsplat import metrics as M
from structsplat.config import FitConfig, InitConfig, StructureTensorConfig
from structsplat.fit import _make_optimizer, _render
from structsplat.gaussians import GaussianField
from structsplat.init import build_field


PROTOCOL = "BENCH-009 Stage-1 parent/candidate runner scaffold v1"
SCOPE = "parent_checkpoints_and_candidate_smoke_not_scientific_cells"
PRICE_SCOPE = "provisional_fixed_width_selector_fields_not_actual_rate_or_compression"
SIZE = manifest_module.SIZE
PARENT_COUNT = manifest_module.PARENT_COUNT
SEEDS = manifest_module.SEEDS
PARENT_HORIZONS = manifest_module.PARENT_HORIZONS
SIGMA_CUTOFF = manifest_module.SIGMA_CUTOFF
RENDER_CHUNK = 512
NUMERICAL_REPEATABILITY_TOLERANCE = 1e-12

SMALL_CARRIER_MAGNITUDES = (0.18, 0.32)
SMALL_CARRIER_ORIENTATIONS_DEG = (0.0, 45.0, 90.0, 135.0)
FULL_CARRIER_MAGNITUDES = (0.12, 0.20, 0.30, 0.42)
FULL_CARRIER_ORIENTATIONS_DEG = tuple(22.5 * index for index in range(8))
CARRIER_COEFFICIENT_ORDER = (
    "sin_r",
    "sin_g",
    "sin_b",
    "cos_r",
    "cos_g",
    "cos_b",
)
CARRIER_SELECTOR_FIELDS = (
    {"name": "action_type", "type": "uint8", "width_bytes": 1},
    {"name": "parent_row", "type": "uint16", "width_bytes": 2},
    {"name": "carrier_bank", "type": "uint8", "width_bytes": 1},
    {"name": "frequency_index", "type": "uint16", "width_bytes": 2},
    {"name": "sin_cos_rgb", "type": "float32[6]", "width_bytes": 24},
)
AFFINE_SELECTOR_FIELDS = (
    {"name": "action_type", "type": "uint8", "width_bytes": 1},
    {"name": "parent_row", "type": "uint16", "width_bytes": 2},
    {"name": "xy_rgb", "type": "float32[6]", "width_bytes": 24},
)
BIRTH_SELECTOR_FIELDS = (
    {"name": "action_type", "type": "uint8", "width_bytes": 1},
    {"name": "geometry_pair", "type": "fixed_geometry[2]", "width_bytes": 40},
    {"name": "birth_rgb", "type": "float32[6]", "width_bytes": 24},
)
TANGENT_SELECTOR_FIELDS = (
    {"name": "action_type", "type": "uint8", "width_bytes": 1},
    {"name": "parameter_indices", "type": "uint16[6]", "width_bytes": 12},
    {"name": "coefficients", "type": "float32[6]", "width_bytes": 24},
)
BIRTH_GEOMETRY_SITE_COUNT = 8
BIRTH_GEOMETRY_MIN_SPACING_PIXELS = 6.0

SOURCE_PATHS = (
    "benchmarks/residual_tangent_auction_stage1_runner.py",
    "benchmarks/residual_tangent_auction_stage1.py",
    "benchmarks/residual_tangent_auction_stage1_core.py",
    "benchmarks/residual_tangent_auction_stage1_science.py",
    "benchmarks/residual_tangent_auction_stage1_analysis.py",
    "benchmarks/residual_tangent_auction_stage1_recovery.py",
    "benchmarks/residual_tangent_auction_matrixfree.py",
    "src/structsplat/config.py",
    "src/structsplat/fit.py",
    "src/structsplat/gaussians.py",
    "src/structsplat/init.py",
    "src/structsplat/metrics.py",
    "src/structsplat/render.py",
    "tasks/BENCH-009-residual-tangent-auction.md",
    "tests/test_residual_tangent_auction_stage1.py",
    "tests/test_residual_tangent_auction_stage1_core.py",
    "tests/test_residual_tangent_auction_stage1_runner.py",
    "tests/test_residual_tangent_auction_stage1_science.py",
    "tests/test_residual_tangent_auction_stage1_analysis.py",
    "tests/test_residual_tangent_auction_stage1_recovery.py",
    "tests/test_residual_tangent_auction_matrixfree.py",
    "pyproject.toml",
)


@dataclass(frozen=True)
class CarrierFrequency:
    index: int
    magnitude: float
    orientation_degrees: float
    frequency_xy: tuple[float, float]

    def as_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "magnitude_cycles_per_local_sigma": self.magnitude,
            "orientation_degrees": self.orientation_degrees,
            "frequency_xy": list(self.frequency_xy),
        }


@dataclass(frozen=True)
class CandidateSelection:
    kind: str
    identity: dict[str, Any]
    coefficients: tuple[float, ...]
    discovery_rank: int
    discovery_condition: float | None
    discovery_base_mse: float
    discovery_post_mse: float
    heldout_base_mse: float
    heldout_post_mse: float
    heldout_mse_gain: float
    all_pixel_oracle_coefficients: tuple[float, ...]
    all_pixel_oracle_post_mse: float
    candidate_count: int
    optimized_dofs: int
    provisional_selector_fields: tuple[dict[str, Any], ...]
    provisional_packet_bytes: int
    discovery_input_sha256: str
    numerical_repeatability_tolerance: float = NUMERICAL_REPEATABILITY_TOLERANCE
    price_scope: str = PRICE_SCOPE

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.identity))

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["identity_sha256"] = self.identity_sha256
        record["coefficients"] = list(self.coefficients)
        record["all_pixel_oracle_coefficients"] = list(
            self.all_pixel_oracle_coefficients
        )
        record["provisional_selector_fields"] = list(self.provisional_selector_fields)
        return record


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    array = np.ascontiguousarray(
        value.detach().cpu().numpy() if torch.is_tensor(value) else value
    )
    header = _canonical_json({"dtype": str(array.dtype), "shape": list(array.shape)})
    return _sha256_bytes(header + b"\0" + array.tobytes(order="C"))


def field_sha256(field: GaussianField) -> str:
    """Hash exact tensor content, including the absence of optional state."""

    digest = hashlib.sha256()
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
        value = getattr(field, name)
        digest.update(name.encode("ascii") + b"\0")
        if value is None:
            digest.update(b"NONE\n")
        else:
            digest.update(_tensor_sha256(value).encode("ascii") + b"\n")
    return digest.hexdigest()


def _strict_load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-strict JSON constant {value} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-strict JSON constant {value}")
            ))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict) or key not in record:
            raise ValueError(f"JSONL {path}:{line_number} lacks key {key!r}")
        identifier = str(record[key])
        if identifier in records:
            raise ValueError(f"duplicate {key}={identifier} in {path}")
        records[identifier] = record
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _carrier_bank(
    magnitudes: Sequence[float],
    orientations_degrees: Sequence[float],
) -> tuple[CarrierFrequency, ...]:
    values: list[CarrierFrequency] = []
    for magnitude in magnitudes:
        for angle_degrees in orientations_degrees:
            angle = math.radians(float(angle_degrees))
            values.append(
                CarrierFrequency(
                    index=len(values),
                    magnitude=float(magnitude),
                    orientation_degrees=float(angle_degrees),
                    frequency_xy=(
                        float(magnitude) * math.cos(angle),
                        float(magnitude) * math.sin(angle),
                    ),
                )
            )
    return tuple(values)


SMALL_CARRIER_BANK = _carrier_bank(
    SMALL_CARRIER_MAGNITUDES, SMALL_CARRIER_ORIENTATIONS_DEG
)
FULL_CARRIER_BANK = _carrier_bank(
    FULL_CARRIER_MAGNITUDES, FULL_CARRIER_ORIENTATIONS_DEG
)
CARRIER_BANKS = {
    "small": SMALL_CARRIER_BANK,
    "full": FULL_CARRIER_BANK,
}


def carrier_bank_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, bank in CARRIER_BANKS.items():
        entries = [item.as_record() for item in bank]
        result[name] = {
            "entries": entries,
            "entry_count": len(entries),
            "coefficient_basis": "sin_cos_rgb",
            "coefficient_order": list(CARRIER_COEFFICIENT_ORDER),
            "optimized_dofs": 6,
            "provisional_selector_fields": list(CARRIER_SELECTOR_FIELDS),
            "provisional_packet_bytes": sum(
                int(item["width_bytes"]) for item in CARRIER_SELECTOR_FIELDS
            ),
            "price_scope": PRICE_SCOPE,
            "bank_sha256": _sha256_bytes(_canonical_json(entries)),
        }
    return result


def resolved_parent_configs(
    seed: int,
    iterations: int = PARENT_HORIZONS[-1][1],
) -> tuple[InitConfig, FitConfig]:
    if seed not in SEEDS:
        raise ValueError(f"parent seed must be one of {SEEDS}")
    if iterations != PARENT_HORIZONS[-1][1]:
        raise ValueError("parent trajectory must run exactly 160 optimizer steps")
    init_config = InitConfig(
        strategy="quadtree_wse",
        num_gaussians=PARENT_COUNT,
        seed=int(seed),
        opacity_mode="none",
    )
    fit_config = FitConfig(
        iters=int(iterations),
        renderer="normalized",
        render_chunk=RENDER_CHUNK,
        log_every=int(iterations),
        checkpoint_policy="terminal",
        lr_schedule="none",
        color_basis="constant",
        color_solve_every=None,
        color_solve_schedule="none",
        qat_mode="off",
        lambda_rate=0.0,
        prune_every=None,
        split_every=None,
        split_count=0,
        refine_site="none",
        relocate_every=None,
        relocate_count=0,
        max_gaussians=PARENT_COUNT,
        adaptive_count=False,
        support_fade=False,
        covariance_filter_mode="none",
        aa_dilation=0.0,
    )
    return init_config, fit_config


def _assert_parent_configs(init_config: InitConfig, fit_config: FitConfig) -> None:
    if (
        init_config.strategy != "quadtree_wse"
        or init_config.num_gaussians != PARENT_COUNT
        or init_config.opacity_mode != "none"
    ):
        raise RuntimeError("parent initialization drifted from the frozen protocol")
    forbidden = (
        fit_config.renderer != "normalized",
        fit_config.checkpoint_policy != "terminal",
        fit_config.color_basis != "constant",
        fit_config.color_solve_every is not None,
        fit_config.color_solve_schedule != "none",
        fit_config.qat_mode != "off",
        fit_config.lambda_rate != 0.0,
        fit_config.prune_every is not None,
        fit_config.split_every is not None,
        fit_config.split_count != 0,
        fit_config.refine_site != "none",
        fit_config.relocate_every is not None,
        fit_config.relocate_count != 0,
        fit_config.adaptive_count,
        fit_config.support_fade,
        fit_config.covariance_filter_mode != "none",
    )
    if any(forbidden):
        raise RuntimeError("parent fitter enables a forbidden topology/rate/appearance path")


def validate_preflight_artifact(directory: str | Path) -> dict[str, Any]:
    """Validate the supplied preflight while preserving its fail-closed development flag."""

    root = Path(directory)
    paths = {name: root / name for name in ("config.json", "manifest.json", "preflight.json")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage-1 preflight artifacts: {missing}")
    config = _strict_load_json(paths["config.json"])
    manifest = _strict_load_json(paths["manifest.json"])
    preflight = _strict_load_json(paths["preflight.json"])
    checks = {
        "preflight_pass": preflight.get("preflight_pass") is True,
        "preflight_development_false": preflight.get("development_run_authorized") is False,
        "preflight_scientific_false": preflight.get("scientific_cells_present") is False,
        "manifest_development_false": manifest.get("development_run_authorized") is False,
        "manifest_scientific_false": manifest.get("scientific_cells_present") is False,
        "method_promotion_false": preflight.get("method_promotion_authorized") is False,
        "compression_false": preflight.get("compression_claim_authorized") is False,
        "config_hash": preflight.get("config_sha256") == _sha256_file(paths["config.json"]),
        "manifest_hash": (
            preflight.get("manifest_sha256") == _sha256_file(paths["manifest.json"])
        ),
        "protocol_match": (
            config.get("protocol") == manifest.get("protocol") == preflight.get("protocol")
        ),
    }
    repo_root = Path(__file__).resolve().parents[1]
    provenance = config.get("provenance")
    critical_sources = (
        provenance.get("critical_sources") if isinstance(provenance, dict) else None
    )
    required_paths = tuple(manifest_module.CRITICAL_SOURCE_PATHS)
    missing_critical: list[str] = []
    changed_critical: list[str] = []
    if not isinstance(critical_sources, dict):
        missing_critical.extend(required_paths)
    else:
        for relative in required_paths:
            record = critical_sources.get(relative)
            if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                missing_critical.append(relative)
                continue
            live_path = repo_root / relative
            if not live_path.is_file() or _sha256_file(live_path) != record["sha256"]:
                changed_critical.append(relative)
    checks["critical_source_paths_complete"] = not missing_critical
    checks["critical_source_hashes_live"] = not changed_critical
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        details = {
            "failed": failed,
            "missing_critical": missing_critical,
            "changed_critical": changed_critical,
        }
        raise RuntimeError(
            f"Stage-1 preflight artifact is not parent-safe: {details}"
        )
    return {
        "directory": str(root.resolve()),
        "config_sha256": _sha256_file(paths["config.json"]),
        "manifest_sha256": _sha256_file(paths["manifest.json"]),
        "preflight_sha256": _sha256_file(paths["preflight.json"]),
        "target_manifest_sha256": manifest.get("target_manifest_sha256"),
        "critical_source_paths": list(required_paths),
        "critical_source_hashes_live": True,
        "preflight_pass": True,
        "development_run_authorized": False,
        "scientific_cells_present": False,
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {relative: _sha256_file(root / relative) for relative in SOURCE_PATHS}


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_sha256": None, "diff_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": _sha256_bytes(status),
        "diff_sha256": _sha256_bytes(diff),
    }


def runner_binding(preflight: dict[str, Any]) -> dict[str, Any]:
    init_config, fit160 = resolved_parent_configs(SEEDS[0])
    return {
        "protocol": PROTOCOL,
        "scope": SCOPE,
        "preflight": preflight,
        "source_sha256": _source_hashes(),
        "axes": {
            "size": SIZE,
            "parent_count": PARENT_COUNT,
            "seeds": list(SEEDS),
            "parent_horizons": [list(item) for item in PARENT_HORIZONS],
            "sigma_cutoff": SIGMA_CUTOFF,
            "render_chunk": RENDER_CHUNK,
        },
        "parent_protocol": {
            "independent_run_per_horizon": False,
            "single_trajectory_with_post_step_checkpoints": [24, 160],
            "trajectory_count": len(manifest_module.TARGET_FAMILIES) * len(SEEDS),
            "checkpoint_count": (
                len(manifest_module.TARGET_FAMILIES)
                * len(SEEDS)
                * len(PARENT_HORIZONS)
            ),
            "init_seed_policy": "cell seed",
            "init_config_seed101": asdict(init_config),
            "fit_config_trajectory": asdict(fit160),
            "optimizer_state_across_checkpoints": "continuous_adam_state",
            "topology_events": "disabled",
            "qat": "disabled",
            "color_solve": "disabled",
        },
        "carrier_banks": carrier_bank_manifest(),
        "selector_fields": {
            "tangent": list(TANGENT_SELECTOR_FIELDS),
            "affine": list(AFFINE_SELECTOR_FIELDS),
            "carrier": list(CARRIER_SELECTOR_FIELDS),
            "two_birth": list(BIRTH_SELECTOR_FIELDS),
            "price_scope": PRICE_SCOPE,
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "git": _git_state(),
        "development_run_authorized": False,
        "scientific_cells_present": False,
        "recovery_runner_present": False,
        "method_promotion_authorized": False,
        "compression_claim_authorized": False,
    }


def initialize_runner_output(outdir: Path, preflight: dict[str, Any]) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    binding = runner_binding(preflight)
    record = {
        "binding": binding,
        "binding_sha256": _sha256_bytes(_canonical_json(binding)),
    }
    path = outdir / "runner_config.json"
    if path.exists():
        existing = _strict_load_json(path)
        if existing != record:
            raise RuntimeError("runner source/protocol binding changed; use a fresh output directory")
    else:
        _write_json(path, record)
    return record


def _parent_id(target_record: dict[str, Any], seed: int, name: str, iterations: int) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "protocol": PROTOCOL,
                "target_id": target_record["target_id"],
                "target_pixel_sha256": target_record["pixel_sha256"],
                "seed": int(seed),
                "horizon_name": name,
                "iterations": int(iterations),
                "parent_count": PARENT_COUNT,
            }
        )
    )


def _parent_trajectory_id(target_record: dict[str, Any], seed: int) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "protocol": PROTOCOL,
                "target_id": target_record["target_id"],
                "target_pixel_sha256": target_record["pixel_sha256"],
                "seed": int(seed),
                "steps": 160,
                "checkpoint_steps": [24, 160],
            }
        )
    )


def run_parent_training_trajectory(
    initial: GaussianField,
    target: torch.Tensor,
    fit_config: FitConfig,
    *,
    checkpoint_steps: tuple[int, ...] = (24, 160),
) -> tuple[dict[int, GaussianField], dict[int, float]]:
    """Run the frozen Adam loop once and snapshot exact post-step 24/160 states."""

    if fit_config.iters != 160 or checkpoint_steps != (24, 160):
        raise ValueError("parent trajectory requires exact post-step checkpoints (24, 160)")
    if (
        fit_config.optimizer != "adam"
        or fit_config.pixel_loss != "l1"
        or fit_config.loss_weighting != "none"
        or fit_config.loss_warmup_iters != 0
        or fit_config.loss_target_downsample != 1
        or fit_config.ssim_weight != 0.3
        or fit_config.geometry_loss_weight != 0.0
        or fit_config.lr_schedule != "none"
    ):
        raise RuntimeError("parent objective drifted from default L1/SSIM Adam")
    if initial.n != PARENT_COUNT or initial.opacities is not None:
        raise ValueError("parent trajectory requires no-opacity N=64 initialization")
    height, width = int(target.shape[0]), int(target.shape[1])
    field = initial.detached()
    field.trainable()
    optimizer = _make_optimizer(field, fit_config)
    lo, hi = math.log(0.35), math.log(max(height, width))
    snapshots: dict[int, GaussianField] = {}
    elapsed: dict[int, float] = {}
    start = time.perf_counter()
    for iteration in range(fit_config.iters):
        image = _render(field, fit_config, height, width)
        pixel = (image - target).abs().mean()
        similarity = M.ssim(image, target, backend=fit_config.ssim_backend)
        loss = (
            (1.0 - fit_config.ssim_weight) * pixel
            + fit_config.ssim_weight * (1.0 - similarity)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            field.log_scales.clamp_(lo, hi)
            if field.scale_max is not None:
                cap = torch.log(torch.clamp(field.scale_max, min=1e-3))
                torch.minimum(field.log_scales, cap, out=field.log_scales)
        post_step = iteration + 1
        if post_step in checkpoint_steps:
            snapshots[post_step] = field.detached()
            elapsed[post_step] = time.perf_counter() - start
    if tuple(snapshots) != checkpoint_steps:
        raise RuntimeError("parent trajectory did not produce both frozen checkpoints")
    return snapshots, elapsed


def fit_parent_checkpoint(
    target_record: dict[str, Any],
    *,
    seed: int,
    horizon_name: str,
    iterations: int,
    field_path: Path,
    binding_sha256: str,
) -> dict[str, Any]:
    """Removed unsafe API: independent horizon fits violate the frozen protocol."""

    raise RuntimeError(
        "independent horizon fits are forbidden; use fit_parent_trajectory"
    )


def fit_parent_trajectory(
    target_record: dict[str, Any],
    *,
    seed: int,
    field_paths: dict[int, Path],
    binding_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one target/seed trajectory and persist its post-step 24/160 states."""

    torch.set_num_threads(1)
    if set(field_paths) != {24, 160}:
        raise ValueError("field paths must cover exact checkpoint steps 24 and 160")
    image = np.ascontiguousarray(target_record["image"], dtype=np.float32)
    if image.shape != (SIZE, SIZE, 3):
        raise ValueError("parent target must be exact 64x64 RGB")
    if _tensor_sha256(image) != target_record["pixel_sha256"]:
        raise ValueError("parent target hash does not match its record")
    init_config, fit_config = resolved_parent_configs(seed)
    _assert_parent_configs(init_config, fit_config)
    target = torch.as_tensor(image, dtype=torch.float32, device="cpu")
    torch.manual_seed(int(seed))
    start = time.perf_counter()
    initial = build_field(
        image,
        init_config,
        StructureTensorConfig(),
        device="cpu",
    )
    initialization_seconds = time.perf_counter() - start
    if initial.n != PARENT_COUNT or initial.opacities is not None:
        raise RuntimeError("parent initializer violated N=64/no-opacity")
    initial_hash = field_sha256(initial)
    snapshots, elapsed = run_parent_training_trajectory(initial, target, fit_config)
    trajectory_id = _parent_trajectory_id(target_record, seed)
    names_by_step = {value: name for name, value in PARENT_HORIZONS}
    parent_records: list[dict[str, Any]] = []
    for checkpoint_step, field in snapshots.items():
        if field.n != PARENT_COUNT or field.opacities is not None or field.color_grads is not None:
            raise RuntimeError("parent trajectory violated topology/appearance invariants")
        render = render_normalized(
            field,
            SIZE,
            SIZE,
            sigma_cutoff=SIGMA_CUTOFF,
            render_chunk=RENDER_CHUNK,
        )
        if not bool(torch.isfinite(render).all()):
            raise RuntimeError("parent render contains non-finite values")
        field_path = field_paths[checkpoint_step]
        field_path.parent.mkdir(parents=True, exist_ok=True)
        field.save(str(field_path))
        exact_hash = field_sha256(field)
        reloaded = GaussianField.load(str(field_path), device="cpu")
        if field_sha256(reloaded) != exact_hash:
            raise RuntimeError("saved parent field does not round-trip exactly")
        mse = float(M.mse(render, target))
        parent_records.append(
            {
                "parent_id": _parent_id(
                    target_record,
                    seed,
                    names_by_step[checkpoint_step],
                    checkpoint_step,
                ),
                "trajectory_id": trajectory_id,
                "protocol": PROTOCOL,
                "scope": SCOPE,
                "binding_sha256": binding_sha256,
                "target_id": target_record["target_id"],
                "target_family": target_record.get("family"),
                "target_pixel_sha256": target_record["pixel_sha256"],
                "seed": int(seed),
                "horizon_name": names_by_step[checkpoint_step],
                "iterations": checkpoint_step,
                "checkpoint_step": checkpoint_step,
                "checkpoint_state": "exact_post_optimizer_step",
                "independent_horizon_run": False,
                "optimizer_state_from_step1_continuous": True,
                "field_path": str(field_path),
                "field_file_sha256": _sha256_file(field_path),
                "field_sha256": exact_hash,
                "initial_field_sha256": initial_hash,
                "n_gaussians": field.n,
                "opacity_present": False,
                "color_grads_present": False,
                "mse": mse,
                "psnr": float(M.psnr_from_mse(torch.tensor(mse))),
                "ssim": float(M.ssim(render, target)),
                "ms_ssim": float(M.ms_ssim(render, target)),
                "initialization_seconds": float(initialization_seconds),
                "trajectory_elapsed_seconds": float(elapsed[checkpoint_step]),
                "init_config": asdict(init_config),
                "fit_config": asdict(fit_config),
                "topology_events": "disabled",
                "qat": "disabled",
                "color_solve": "disabled",
                "development_run_authorized": False,
                "scientific_cell": False,
                "method_promotion_authorized": False,
                "compression_claim_authorized": False,
            }
        )
    trajectory_record = {
        "trajectory_id": trajectory_id,
        "protocol": PROTOCOL,
        "binding_sha256": binding_sha256,
        "target_id": target_record["target_id"],
        "target_pixel_sha256": target_record["pixel_sha256"],
        "seed": int(seed),
        "optimizer": "Adam",
        "optimizer_state": "continuous_steps_1_through_160",
        "checkpoint_steps": [24, 160],
        "checkpoint_parent_ids": [item["parent_id"] for item in parent_records],
        "initial_field_sha256": initial_hash,
        "initialization_seconds": float(initialization_seconds),
        "trajectory_seconds": float(elapsed[160]),
        "development_run_authorized": False,
    }
    return trajectory_record, parent_records


def _rgb_columns(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    height, width = first.shape
    for basis in (first, second):
        for channel in range(3):
            image = basis.new_zeros((height, width, 3))
            image[..., channel] = basis
            columns.append(image.reshape(-1))
    return torch.stack(columns, dim=1)


def _cached_local_bases(
    field: GaussianField,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache normalized ownership and local x/y coordinates for every parent row."""

    responsibility, _denominator = normalized_responsibilities(
        field,
        height,
        width,
        sigma_cutoff=SIGMA_CUTOFF,
    )
    device, dtype = field.means.device, field.means.dtype
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dx = xx.unsqueeze(0) - field.means[:, 0, None, None]
    dy = yy.unsqueeze(0) - field.means[:, 1, None, None]
    cosine = torch.cos(field.rotations)[:, None, None]
    sine = torch.sin(field.rotations)[:, None, None]
    scales = field.effective_scales().clamp_min(1e-6)
    local_x = (cosine * dx + sine * dy) / scales[:, 0, None, None]
    local_y = (-sine * dx + cosine * dy) / scales[:, 1, None, None]
    return responsibility, local_x, local_y


def _mask_mse(value: torch.Tensor, mask: torch.Tensor) -> float:
    return float(value.reshape(-1)[mask].square().mean())


def _selection_from_evaluation(
    *,
    kind: str,
    identity: dict[str, Any],
    evaluation: SixCoefficientEvaluation,
    candidate_count: int,
    selector_fields: tuple[dict[str, Any], ...],
    residual: torch.Tensor,
    discovery: torch.Tensor,
    heldout: torch.Tensor,
    design: torch.Tensor,
    discovery_input_sha256: str,
    offset: torch.Tensor | None = None,
) -> CandidateSelection:
    flat_offset = residual.new_zeros(residual.numel()) if offset is None else offset.reshape(-1)
    coefficients = evaluation.crossfit.coefficients
    remaining = residual.reshape(-1) - flat_offset - design @ coefficients
    oracle_coefficients = evaluation.all_pixel_oracle.coefficients
    oracle_remaining = residual.reshape(-1) - flat_offset - design @ oracle_coefficients
    return CandidateSelection(
        kind=kind,
        identity=identity,
        coefficients=tuple(float(value) for value in coefficients),
        discovery_rank=evaluation.crossfit.discovery_rank,
        discovery_condition=evaluation.crossfit.discovery_condition,
        discovery_base_mse=_mask_mse(residual, discovery),
        discovery_post_mse=_mask_mse(remaining, discovery),
        heldout_base_mse=_mask_mse(residual, heldout),
        heldout_post_mse=_mask_mse(remaining, heldout),
        heldout_mse_gain=_mask_mse(residual, heldout) - _mask_mse(remaining, heldout),
        all_pixel_oracle_coefficients=tuple(float(value) for value in oracle_coefficients),
        all_pixel_oracle_post_mse=float(oracle_remaining.square().mean()),
        candidate_count=int(candidate_count),
        optimized_dofs=6,
        provisional_selector_fields=selector_fields,
        provisional_packet_bytes=sum(int(item["width_bytes"]) for item in selector_fields),
        discovery_input_sha256=discovery_input_sha256,
    )


def _discovery_input_sha256(
    field: GaussianField,
    residual: torch.Tensor,
    discovery: torch.Tensor,
    selector_specification: dict[str, Any],
) -> str:
    """Hash exact selector-visible bytes separately from numerical repeatability."""

    selected = discovery.to(device=residual.device, dtype=torch.bool).reshape(-1)
    payload = {
        "field_sha256": field_sha256(field),
        "selector_specification": selector_specification,
        "discovery_mask_sha256": _tensor_sha256(selected),
        "discovery_residual_sha256": _tensor_sha256(residual.reshape(-1)[selected]),
    }
    return _sha256_bytes(_canonical_json(payload))


def _selection_inputs(
    field: GaussianField,
    target: torch.Tensor | np.ndarray,
    fold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if field.opacities is not None or field.color_grads is not None:
        raise ValueError("candidate selection requires a constant-color no-opacity parent")
    target_tensor = torch.as_tensor(
        target, device=field.means.device, dtype=field.means.dtype
    )
    if target_tensor.ndim != 3 or target_tensor.shape[-1] != 3:
        raise ValueError("candidate target must be HxWx3")
    height, width = int(target_tensor.shape[0]), int(target_tensor.shape[1])
    base = render_normalized(
        field,
        height,
        width,
        sigma_cutoff=SIGMA_CUTOFF,
        render_chunk=RENDER_CHUNK,
    )
    residual = (target_tensor - base).reshape(-1)
    discovery, heldout = checkerboard_fold_masks(height, width, fold)
    return residual, discovery.to(residual.device), heldout.to(residual.device), base


def _verify_heldout_invariance(
    selector: Callable[[torch.Tensor | np.ndarray, bool], CandidateSelection],
    target: torch.Tensor | np.ndarray,
    heldout: torch.Tensor,
    original: CandidateSelection,
) -> None:
    target_tensor = torch.as_tensor(target).clone()
    flat = target_tensor.reshape(-1)
    indices = torch.arange(int(heldout.sum()), device=flat.device, dtype=flat.dtype)
    flat[heldout.to(flat.device)] += 0.731 * torch.sin(indices + 0.5)
    perturbed = selector(target_tensor, False)
    discovery_hash_changed = (
        perturbed.discovery_input_sha256 != original.discovery_input_sha256
    )
    identity_changed = perturbed.identity_sha256 != original.identity_sha256
    original_coefficients = torch.tensor(original.coefficients, dtype=torch.float64)
    perturbed_coefficients = torch.tensor(perturbed.coefficients, dtype=torch.float64)
    coefficient_change = float((original_coefficients - perturbed_coefficients).abs().max())
    if (
        discovery_hash_changed
        or identity_changed
        or coefficient_change > NUMERICAL_REPEATABILITY_TOLERANCE
    ):
        raise RuntimeError(
            "held-out perturbation changed discovery-only candidate selection: "
            f"discovery_hash_changed={discovery_hash_changed}, "
            f"identity_changed={identity_changed}, coefficient_change={coefficient_change}, "
            f"numerical_tolerance={NUMERICAL_REPEATABILITY_TOLERANCE}"
        )


def select_affine_candidate(
    field: GaussianField,
    target: torch.Tensor | np.ndarray,
    fold: int,
    *,
    rcond: float = manifest_module.PRIMARY_RCOND,
    verify_leakage: bool = True,
) -> CandidateSelection:
    """Select one of all parent rows using discovery scores only."""

    residual, discovery, heldout, _base = _selection_inputs(field, target, fold)
    discovery_hash = _discovery_input_sha256(
        field,
        residual,
        discovery,
        {"kind": "affine", "rcond": float(rcond), "candidate_rows": field.n},
    )
    height, width = torch.as_tensor(target).shape[:2]
    responsibility, local_x, local_y = _cached_local_bases(
        field, int(height), int(width)
    )
    best: tuple[tuple[float, int], int, torch.Tensor, SixCoefficientEvaluation] | None = None
    for gid in range(field.n):
        design = _rgb_columns(
            responsibility[gid] * local_x[gid],
            responsibility[gid] * local_y[gid],
        )
        evaluation = evaluate_six_coefficient_action(
            design, residual, discovery, heldout, rcond=rcond
        )
        key = (float(evaluation.crossfit.discovery_post_mse), gid)
        if best is None or key < best[0]:
            best = (key, gid, design, evaluation)
    if best is None:
        raise RuntimeError("affine candidate bank is empty")
    _key, gid, design, evaluation = best
    selection = _selection_from_evaluation(
        kind="affine",
        identity={"parent_row": gid},
        evaluation=evaluation,
        candidate_count=field.n,
        selector_fields=AFFINE_SELECTOR_FIELDS,
        residual=residual,
        discovery=discovery,
        heldout=heldout,
        design=design,
        discovery_input_sha256=discovery_hash,
    )
    if verify_leakage:
        _verify_heldout_invariance(
            lambda changed, verify: select_affine_candidate(
                field, changed, fold, rcond=rcond, verify_leakage=verify
            ),
            target,
            heldout,
            selection,
        )
    return selection


def select_carrier_candidate(
    field: GaussianField,
    target: torch.Tensor | np.ndarray,
    fold: int,
    *,
    bank_name: str = "full",
    rcond: float = manifest_module.PRIMARY_RCOND,
    verify_leakage: bool = True,
) -> CandidateSelection:
    """Select a parent and frozen local frequency using discovery scores only."""

    try:
        bank = CARRIER_BANKS[bank_name]
    except KeyError as exc:
        raise ValueError(f"unknown carrier bank {bank_name!r}") from exc
    residual, discovery, heldout, _base = _selection_inputs(field, target, fold)
    bank_records = [item.as_record() for item in bank]
    discovery_hash = _discovery_input_sha256(
        field,
        residual,
        discovery,
        {
            "kind": "carrier",
            "rcond": float(rcond),
            "bank_name": bank_name,
            "bank_sha256": _sha256_bytes(_canonical_json(bank_records)),
        },
    )
    height, width = torch.as_tensor(target).shape[:2]
    responsibility, local_x, local_y = _cached_local_bases(
        field, int(height), int(width)
    )
    best: tuple[
        tuple[float, int, int], int, CarrierFrequency, torch.Tensor, SixCoefficientEvaluation
    ] | None = None
    for gid in range(field.n):
        for frequency in bank:
            phase = 2.0 * math.pi * (
                frequency.frequency_xy[0] * local_x[gid]
                + frequency.frequency_xy[1] * local_y[gid]
            )
            design = _rgb_columns(
                responsibility[gid] * torch.sin(phase),
                responsibility[gid] * torch.cos(phase),
            )
            evaluation = evaluate_six_coefficient_action(
                design, residual, discovery, heldout, rcond=rcond
            )
            key = (float(evaluation.crossfit.discovery_post_mse), gid, frequency.index)
            if best is None or key < best[0]:
                best = (key, gid, frequency, design, evaluation)
    if best is None:
        raise RuntimeError("carrier candidate bank is empty")
    _key, gid, frequency, design, evaluation = best
    selection = _selection_from_evaluation(
        kind="carrier",
        identity={
            "parent_row": gid,
            "bank_name": bank_name,
            "frequency_index": frequency.index,
            "frequency_xy": list(frequency.frequency_xy),
        },
        evaluation=evaluation,
        candidate_count=field.n * len(bank),
        selector_fields=CARRIER_SELECTOR_FIELDS,
        residual=residual,
        discovery=discovery,
        heldout=heldout,
        design=design,
        discovery_input_sha256=discovery_hash,
    )
    if verify_leakage:
        _verify_heldout_invariance(
            lambda changed, verify: select_carrier_candidate(
                field,
                changed,
                fold,
                bank_name=bank_name,
                rcond=rcond,
                verify_leakage=verify,
            ),
            target,
            heldout,
            selection,
        )
    return selection


def discovery_birth_geometry_bank(
    residual: torch.Tensor,
    discovery: torch.Tensor,
    height: int,
    width: int,
    *,
    parent_count: int,
    site_count: int = BIRTH_GEOMETRY_SITE_COUNT,
    minimum_spacing: float = BIRTH_GEOMETRY_MIN_SPACING_PIXELS,
) -> tuple[BirthGeometry, ...]:
    """Create fixed isotropic sites from discovery residual only, with deterministic NMS."""

    if residual.shape != (height * width * 3,):
        raise ValueError("birth geometry residual has the wrong shape")
    discovery_pixels = discovery.reshape(height, width, 3).all(dim=2)
    score = residual.reshape(height, width, 3).square().mean(dim=2)
    score = torch.where(discovery_pixels, score, torch.full_like(score, -torch.inf))
    order = torch.argsort(score.reshape(-1), descending=True, stable=True)
    selected: list[tuple[int, int]] = []
    for flat_index in order.tolist():
        y, x = divmod(int(flat_index), width)
        if not bool(discovery_pixels[y, x]) or not math.isfinite(float(score[y, x])):
            continue
        if all(math.hypot(x - old_x, y - old_y) >= minimum_spacing for old_y, old_x in selected):
            selected.append((y, x))
        if len(selected) == site_count:
            break
    if len(selected) != site_count:
        raise RuntimeError("discovery-only birth geometry bank could not fill its frozen size")
    base_scale = max(1.25, 0.45 * math.sqrt(height * width / max(parent_count, 1)))
    return tuple(
        BirthGeometry(
            mean_xy=(float(x), float(y)),
            scale_xy=(float(base_scale), float(base_scale)),
            angle_radians=0.0,
        )
        for y, x in selected
    )


def select_two_birth_candidate(
    field: GaussianField,
    target: torch.Tensor | np.ndarray,
    fold: int,
    *,
    rcond: float = manifest_module.PRIMARY_RCOND,
    verify_leakage: bool = True,
) -> CandidateSelection:
    """Select an unordered exact finite pair from a discovery-only geometry bank."""

    residual, discovery, heldout, _base = _selection_inputs(field, target, fold)
    discovery_hash = _discovery_input_sha256(
        field,
        residual,
        discovery,
        {
            "kind": "two_birth",
            "rcond": float(rcond),
            "geometry_site_count": BIRTH_GEOMETRY_SITE_COUNT,
            "geometry_min_spacing_pixels": BIRTH_GEOMETRY_MIN_SPACING_PIXELS,
            "pair_order": "unordered_lexicographic",
        },
    )
    height, width = torch.as_tensor(target).shape[:2]
    geometries = discovery_birth_geometry_bank(
        residual,
        discovery,
        int(height),
        int(width),
        parent_count=field.n,
    )
    pairs = tuple(itertools.combinations(range(len(geometries)), 2))
    best: tuple[
        tuple[float, int, int], tuple[int, int], Any, SixCoefficientEvaluation
    ] | None = None
    for first, second in pairs:
        packet = build_two_birth_packet(
            field,
            (geometries[first], geometries[second]),
            int(height),
            int(width),
            sigma_cutoff=SIGMA_CUTOFF,
            render_chunk=RENDER_CHUNK,
        )
        shifted = residual - packet.offset
        evaluation = evaluate_six_coefficient_action(
            packet.design, shifted, discovery, heldout, rcond=rcond
        )
        key = (float(evaluation.crossfit.discovery_post_mse), first, second)
        if best is None or key < best[0]:
            best = (key, (first, second), packet, evaluation)
    if best is None:
        raise RuntimeError("two-birth pair bank is empty")
    _key, pair, packet, evaluation = best
    geometry_records = [
        {
            "site_index": index,
            "mean_xy": list(geometries[index].mean_xy),
            "scale_xy": list(geometries[index].scale_xy),
            "angle_radians": geometries[index].angle_radians,
        }
        for index in pair
    ]
    selection = _selection_from_evaluation(
        kind="two_birth",
        identity={
            "unordered_site_pair": list(pair),
            "geometries": geometry_records,
            "geometry_bank_site_count": len(geometries),
            "geometry_source": "discovery_rgb_squared_residual_nms",
        },
        evaluation=evaluation,
        candidate_count=len(pairs),
        selector_fields=BIRTH_SELECTOR_FIELDS,
        residual=residual,
        discovery=discovery,
        heldout=heldout,
        design=packet.design,
        discovery_input_sha256=discovery_hash,
        offset=packet.offset,
    )
    if verify_leakage:
        _verify_heldout_invariance(
            lambda changed, verify: select_two_birth_candidate(
                field, changed, fold, rcond=rcond, verify_leakage=verify
            ),
            target,
            heldout,
            selection,
        )
    return selection


def _parent_specs(targets: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for family in manifest_module.TARGET_FAMILIES:
        target = targets[family]
        for seed in SEEDS:
            yield {
                "target": target,
                "seed": seed,
                "trajectory_id": _parent_trajectory_id(target, seed),
                "checkpoints": [
                    {
                        "horizon_name": name,
                        "iterations": iterations,
                        "parent_id": _parent_id(target, seed, name, iterations),
                    }
                    for name, iterations in PARENT_HORIZONS
                ],
            }


def run_parent_phase(
    outdir: Path,
    binding: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    *,
    max_parents: int | None,
) -> dict[str, Any]:
    path = outdir / "parents.jsonl"
    trajectory_path = outdir / "parent_trajectories.jsonl"
    existing = _load_jsonl(path, "parent_id")
    existing_trajectories = _load_jsonl(trajectory_path, "trajectory_id")
    missing = [
        spec
        for spec in _parent_specs(targets)
        if spec["trajectory_id"] not in existing_trajectories
        or any(item["parent_id"] not in existing for item in spec["checkpoints"])
    ]
    selected = missing if max_parents is None else missing[:max_parents]
    completed_trajectories = 0
    completed_checkpoints = 0
    for spec in selected:
        field_paths = {
            int(item["iterations"]): (
                outdir / "parents" / f"{item['parent_id']}.npz"
            ).resolve()
            for item in spec["checkpoints"]
        }
        trajectory_record, parent_records = fit_parent_trajectory(
            spec["target"],
            seed=int(spec["seed"]),
            field_paths=field_paths,
            binding_sha256=str(binding["binding_sha256"]),
        )
        for record in parent_records:
            parent_id = str(record["parent_id"])
            if parent_id in existing:
                if existing[parent_id].get("field_sha256") != record.get("field_sha256"):
                    raise RuntimeError("resumed trajectory changed an existing checkpoint")
                continue
            _append_jsonl(path, record)
            existing[parent_id] = record
            completed_checkpoints += 1
        trajectory_id = str(trajectory_record["trajectory_id"])
        if trajectory_id not in existing_trajectories:
            _append_jsonl(trajectory_path, trajectory_record)
            existing_trajectories[trajectory_id] = trajectory_record
        completed_trajectories += 1
    total = len(_load_jsonl(path, "parent_id"))
    trajectory_total = len(_load_jsonl(trajectory_path, "trajectory_id"))
    return {
        "phase": "parents",
        "completed_this_invocation": completed_trajectories,
        "trajectories_completed_this_invocation": completed_trajectories,
        "checkpoint_records_completed_this_invocation": completed_checkpoints,
        "parent_trajectories_total": trajectory_total,
        "parent_trajectories_expected": len(manifest_module.TARGET_FAMILIES) * len(SEEDS),
        "parent_records_total": total,
        "parent_records_expected": len(manifest_module.TARGET_FAMILIES)
        * len(SEEDS)
        * len(PARENT_HORIZONS),
        "all_parents_complete": (
            trajectory_total == len(manifest_module.TARGET_FAMILIES) * len(SEEDS)
            and total
            == len(manifest_module.TARGET_FAMILIES) * len(SEEDS) * len(PARENT_HORIZONS)
        ),
        "max_parents_semantics": "maximum_target_seed_trajectories",
        "development_run_authorized": False,
    }


def _load_bound_parent(record: dict[str, Any], binding_sha256: str) -> GaussianField:
    if record.get("binding_sha256") != binding_sha256:
        raise RuntimeError("parent record belongs to another runner binding")
    path = Path(str(record["field_path"]))
    if not path.is_file() or _sha256_file(path) != record.get("field_file_sha256"):
        raise RuntimeError("saved parent file is missing or changed")
    field = GaussianField.load(str(path), device="cpu")
    if field_sha256(field) != record.get("field_sha256"):
        raise RuntimeError("saved parent tensors changed")
    return field


def run_candidate_smoke_phase(
    outdir: Path,
    binding: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    *,
    max_parents: int | None,
) -> dict[str, Any]:
    parent_records = list(_load_jsonl(outdir / "parents.jsonl", "parent_id").values())
    if not parent_records:
        raise RuntimeError("candidate-smoke requires at least one saved parent")
    parent_records.sort(key=lambda item: str(item["parent_id"]))
    if max_parents is not None:
        parent_records = parent_records[:max_parents]
    target_by_id = {item["target_id"]: item for item in targets.values()}
    identities_path = outdir / "candidate_identities.jsonl"
    existing = _load_jsonl(identities_path, "identity_record_id")
    completed = 0
    for parent_record in parent_records:
        parent_id = str(parent_record["parent_id"])
        target_record = target_by_id[str(parent_record["target_id"])]
        field = _load_bound_parent(parent_record, str(binding["binding_sha256"]))
        target = target_record["image"]
        selectors: tuple[tuple[str, Callable[[int], CandidateSelection]], ...] = (
            ("affine", lambda fold: select_affine_candidate(field, target, fold)),
            (
                "carrier_small",
                lambda fold: select_carrier_candidate(
                    field, target, fold, bank_name="small"
                ),
            ),
            ("two_birth", lambda fold: select_two_birth_candidate(field, target, fold)),
        )
        for fold in manifest_module.FOLDS:
            for selector_name, selector in selectors:
                identity_record_id = _sha256_bytes(
                    _canonical_json(
                        {
                            "protocol": PROTOCOL,
                            "parent_id": parent_id,
                            "fold": fold,
                            "selector": selector_name,
                        }
                    )
                )
                if identity_record_id in existing:
                    continue
                start = time.perf_counter()
                selection = selector(fold)
                record = {
                    "identity_record_id": identity_record_id,
                    "protocol": PROTOCOL,
                    "scope": "candidate_smoke_non_scientific",
                    "binding_sha256": binding["binding_sha256"],
                    "parent_id": parent_id,
                    "target_id": target_record["target_id"],
                    "target_pixel_sha256": target_record["pixel_sha256"],
                    "heldout_fold": fold,
                    "selector": selector_name,
                    "selection": selection.as_record(),
                    "selection_seconds": time.perf_counter() - start,
                    "heldout_leakage_check": "passed",
                    "development_run_authorized": False,
                    "scientific_cell": False,
                    "method_promotion_authorized": False,
                    "compression_claim_authorized": False,
                }
                _append_jsonl(identities_path, record)
                existing[identity_record_id] = record
                completed += 1
    return {
        "phase": "candidate-smoke",
        "completed_this_invocation": completed,
        "candidate_identity_records_total": len(existing),
        "development_run_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("parents", "candidate-smoke"), required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument(
        "--outdir", default="results/bench009_tangent_auction_stage1_runner"
    )
    parser.add_argument("--max-parents", type=int, default=None)
    args = parser.parse_args(argv)
    if args.max_parents is not None and args.max_parents <= 0:
        parser.error("--max-parents must be positive")
    torch.set_num_threads(1)
    preflight = validate_preflight_artifact(args.preflight_dir)
    torch.use_deterministic_algorithms(True)
    outdir = Path(args.outdir)
    binding = initialize_runner_output(outdir, preflight)
    targets = manifest_module.target_suite()
    invocation = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, "-m", __name__, *(argv or sys.argv[1:])]),
        "phase": args.phase,
        "max_parents": args.max_parents,
        "binding_sha256": binding["binding_sha256"],
    }
    _append_jsonl(outdir / "invocations.jsonl", invocation)
    if args.phase == "parents":
        result = run_parent_phase(
            outdir,
            binding,
            targets,
            max_parents=args.max_parents,
        )
    else:
        result = run_candidate_smoke_phase(
            outdir,
            binding,
            targets,
            max_parents=args.max_parents,
        )
    result.update(
        {
            "outdir": str(outdir),
            "binding_sha256": binding["binding_sha256"],
            "scientific_cells_present": False,
            "recovery_runner_present": False,
            "method_promotion_authorized": False,
            "compression_claim_authorized": False,
        }
    )
    _write_json(outdir / "run_state.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
