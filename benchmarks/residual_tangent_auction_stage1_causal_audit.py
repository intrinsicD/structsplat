"""Post-run causal-ledger audit for BENCH-009 Stage 1.

This file is intentionally *not* part of the BENCH-009 experiment source binding.  It is a
standalone, fail-closed consumer of the frozen manifest and the completed immediate/recovery
ledgers.  The audit never repairs, resumes, or rewrites experiment artifacts.

The strongest interpretation this audit can authorize is development-set RGB-MSE-local capacity
outside the discovered truncated current-tangent range.  It cannot establish a deficit under the
parent optimizer's ``0.7 L1 + 0.3 (1 - SSIM)`` objective, and it cannot establish compression.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from benchmarks import residual_tangent_auction_stage1 as manifest_module
from benchmarks import residual_tangent_auction_stage1_analysis as analysis_module
from benchmarks import residual_tangent_auction_stage1_recovery as recovery_module
from benchmarks import residual_tangent_auction_stage1_science as science_module
from benchmarks.residual_tangent_auction_matrixfree import tensor_sha256
from benchmarks.residual_tangent_auction_stage1_core import checkerboard_fold_masks


SCHEMA_ID = "bench009.causal-ledger-audit.v1"
SCOPE = "development_rgb_mse_local_diagnostic"
AUDIT_RELATIVE_PATH = "benchmarks/residual_tangent_auction_stage1_causal_audit.py"
BOOTSTRAP_SEED = 20_260_715
BOOTSTRAP_REPLICATES = 100_000
FAMILIES = tuple(manifest_module.TARGET_FAMILIES)
SEEDS = tuple(manifest_module.SEEDS)
FOLDS = tuple(manifest_module.FOLDS)
RCONDS = tuple(manifest_module.RCONDS)
RADII = tuple(manifest_module.TRUST_RADII)
PRIMARY_DAMPING = float(manifest_module.PRIMARY_DAMPING)
PRIMARY_RCOND = float(manifest_module.PRIMARY_RCOND)
NEAR_PLATEAU = 160
CAUSAL_ACTIONS = tuple(manifest_module.CAUSAL_ACTIONS)
MATCHED_ACTIONS = tuple(manifest_module.MATCHED_ACTIONS)
ALL_ACTIONS = CAUSAL_ACTIONS + MATCHED_ACTIONS
CANDIDATES = {
    "affine6": ("j_plus_affine6", "affine6"),
    "carrier6": ("j_plus_carrier6", "carrier6"),
}
CAUSAL_CONTROLS = ("full_j", "j_plus_birth2_rgb6")


def _rcond_label(rcond: float) -> str:
    if rcond == 1e-5:
        return "rcond=1e-5"
    if rcond == 1e-4:
        return "rcond=1e-4"
    return f"rcond={rcond:.17g}"


EXPECTED_COUNTS = {
    "immediate_cells": 4_608,
    "crossfit_immediate": 3_072,
    "oracle_immediate": 1_536,
    "crossfit_immediate_per_ledger": 1_536,
    "oracle_immediate_per_ledger": 768,
    "crossfit_immediate_per_action": 384,
    "oracle_immediate_per_action": 192,
    "diagnostic_scores": 1_152,
    "crossfit_scores": 768,
    "oracle_scores": 384,
    "nonbirth_linear_scores": 864,
    "finite_birth_scores": 288,
    "crossfit_scores_per_action": 96,
    "oracle_scores_per_action": 48,
    "matched_immediate_evidence": 2_304,
    "matched_crossfit_immediate_evidence": 1_536,
    "matched_oracle_evidence": 768,
    "matched_recovery_evidence": 768,
    "matched_evidence_union": 3_072,
    "recovery_sources": 816,
    "recovery_trajectories": 816,
    "recovery_checkpoints": 2_448,
    "checkpoint_0": 816,
    "checkpoint_20": 816,
    "checkpoint_100": 816,
    "new_recovery_renders": 1_632,
    "causal_action_checkpoints": 1_152,
    "matched_action_checkpoints": 1_152,
    "baseline_checkpoints": 144,
    "checkpoints_per_nonbaseline_action": 288,
    "action_step0_immediate_links": 768,
    "baseline_step0_manifest_links": 48,
}


class DuplicateJsonKey(ValueError):
    """Raised when JSON object-pair parsing observes a duplicate key."""


@dataclass(frozen=True)
class LoadedLedger:
    records: tuple[dict[str, Any], ...]
    by_id: dict[str, dict[str, Any]]
    paths: tuple[str, ...]
    errors: tuple[str, ...]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_prefixed_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return prefix + hashlib.sha256(_canonical_json(dict(payload))).hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _finite_tree(value: Any, location: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non-finite number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{location}.{key}")
        return
    raise ValueError(f"unsupported JSON value {type(value).__name__} at {location}")


def _decode_json(payload: str, *, label: str) -> Any:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
        _finite_tree(value)
        return value
    except Exception as exc:
        raise ValueError(f"{label}: {type(exc).__name__}: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = _decode_json(path.read_text(encoding="utf-8"), label=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level value is not an object")
    return value


def _load_jsonl(paths: Sequence[Path], id_field: str) -> LoadedLedger:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                errors.append(f"{path}:{line_number}: blank JSONL line")
                continue
            try:
                value = _decode_json(line, label=f"{path}:{line_number}")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if not isinstance(value, dict):
                errors.append(f"{path}:{line_number}: row is not an object")
                continue
            identifier = value.get(id_field)
            if not isinstance(identifier, str) or not identifier:
                errors.append(f"{path}:{line_number}: missing string {id_field}")
                continue
            if identifier in by_id:
                errors.append(f"duplicate {id_field}={identifier} across strict ledger inputs")
                continue
            records.append(value)
            by_id[identifier] = value
    return LoadedLedger(
        records=tuple(records),
        by_id=by_id,
        paths=tuple(str(path.resolve()) for path in paths),
        errors=tuple(errors),
    )


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _set_errors(errors: list[str], observed: set[str], expected: set[str], label: str) -> None:
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        errors.append(
            f"{label} ID set mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )


def _binding_config(path: Path, *, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        config = _load_json(path)
    except Exception as exc:
        errors.append(f"{label} config load failed: {type(exc).__name__}: {exc}")
        return None
    binding = config.get("binding")
    if not isinstance(binding, dict):
        errors.append(f"{label} config lacks an object binding")
        return config
    if config.get("binding_sha256") != _sha256_json(binding):
        errors.append(f"{label} binding_sha256 is not canonical")
    return config


def _validate_live_sources(binding: Mapping[str, Any], *, label: str, errors: list[str]) -> None:
    sources = binding.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        errors.append(f"{label} binding lacks source_sha256")
        return
    root = Path(__file__).resolve().parents[1]
    for relative, expected in sources.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            errors.append(f"{label} binding has a malformed source entry")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"{label} bound source is missing: {relative}")
        elif _sha256_file(path) != expected:
            errors.append(f"{label} bound source is stale: {relative}")


def source_binding_isolation() -> dict[str, Any]:
    memberships = {
        "manifest_critical_sources": AUDIT_RELATIVE_PATH
        in set(manifest_module.CRITICAL_SOURCE_PATHS),
        "science_sources": AUDIT_RELATIVE_PATH in set(science_module.SOURCE_PATHS),
        "recovery_sources": AUDIT_RELATIVE_PATH in set(recovery_module.SOURCE_PATHS),
    }
    runner_sources: tuple[str, ...] = ()
    try:
        from benchmarks import residual_tangent_auction_stage1_runner as runner_module

        runner_sources = tuple(getattr(runner_module, "SOURCE_PATHS", ()))
    except Exception:
        runner_sources = ()
    memberships["runner_sources"] = AUDIT_RELATIVE_PATH in set(runner_sources)
    return {
        "audit_path": AUDIT_RELATIVE_PATH,
        "included_in_any_experiment_source_binding": any(memberships.values()),
        "memberships": memberships,
        "pass": not any(memberships.values()),
    }


def _bootstrap_indices() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(
        0,
        len(FAMILIES),
        size=(BOOTSTRAP_REPLICATES, len(FAMILIES)),
        endpoint=False,
    )


def _summarize_family_values(
    values: Mapping[str, float], bootstrap_indices: np.ndarray
) -> dict[str, Any]:
    missing = [family for family in FAMILIES if family not in values]
    if missing:
        return {
            "status": "unavailable",
            "missing_families": missing,
            "family_means": dict(values),
        }
    family_array = np.asarray([values[family] for family in FAMILIES], dtype=np.float64)
    samples = np.mean(family_array[bootstrap_indices], axis=1)
    return {
        "status": "available",
        "mean": float(np.mean(family_array)),
        "bootstrap_95_lower": float(np.percentile(samples, 2.5)),
        "positive_family_count": int(np.count_nonzero(family_array > 0.0)),
        "family_means": {family: float(values[family]) for family in FAMILIES},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_rng": "numpy.PCG64",
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_rank -= np.mean(x_rank)
    y_rank -= np.mean(y_rank)
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(x_rank, y_rank) / denominator)


def _calibration_cell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    for row in rows:
        floor = max(1e-12, float(row["base_mse"]) * 1e-5)
        target = eligible if float(row["predicted_mse_gain"]) > floor else excluded
        target.append(row)
    predicted = np.asarray([float(row["predicted_mse_gain"]) for row in eligible], dtype=np.float64)
    realized = np.asarray([float(row["realized_mse_gain"]) for row in eligible], dtype=np.float64)
    enough = len(eligible) >= 3 and len(np.unique(predicted)) >= 3 and len(np.unique(realized)) >= 3
    result: dict[str, Any] = {
        "raw_row_count": len(rows),
        "eligible_row_count": len(eligible),
        "at_or_below_floor_count": len(excluded),
        "at_or_below_floor_cell_ids": [row["cell_id"] for row in excluded],
        "enough_non_tied_eligible_rows": enough,
    }
    if not enough:
        result.update(
            spearman=None,
            median_realized_over_predicted=None,
            gate_pass=False,
            failure_reason="fewer than three eligible non-tied rows",
        )
        return result
    spearman = _spearman(predicted, realized)
    ratio = float(np.median(realized / predicted))
    result.update(
        spearman=spearman,
        median_realized_over_predicted=ratio,
        spearman_minimum=0.8,
        gain_ratio_bounds=[0.5, 2.0],
        gate_pass=(math.isfinite(spearman) and spearman >= 0.8 and 0.5 <= ratio <= 2.0),
    )
    return result


def all_eight_action_calibration(immediate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in immediate if row.get("selection_scope") == "crossfit"]
    errors: list[str] = []
    if len(rows) != EXPECTED_COUNTS["crossfit_immediate"]:
        errors.append(f"all-action calibration has {len(rows)} rows, expected 3072")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("family"),
            row.get("seed"),
            row.get("parent_iters"),
            row.get("heldout_fold"),
            row.get("damping"),
            row.get("rcond"),
            row.get("trust_radius"),
        )
        grouped[key].append(row)
    for key, siblings in grouped.items():
        if {str(row.get("action")) for row in siblings} != set(ALL_ACTIONS):
            errors.append(f"all-action calibration sibling set drifted at {key}")
        if len({float(row["base_mse"]) for row in siblings}) != 1:
            errors.append(f"all-action calibration base-MSE mismatch at {key}")
    aggregate = _calibration_cell(rows)
    strata: dict[str, Any] = {}
    for action in ALL_ACTIONS:
        for name, iterations in manifest_module.PARENT_HORIZONS:
            selected = [
                row
                for row in rows
                if row.get("action") == action and int(row.get("parent_iters", -1)) == iterations
            ]
            key = f"action={action}|parent_horizon={name}"
            strata[key] = _calibration_cell(selected)
            if len(selected) != 192:
                errors.append(f"calibration stratum {key} has {len(selected)} rows")
    aggregate_pass = bool(aggregate["gate_pass"])
    strata_pass = len(strata) == 16 and all(value["gate_pass"] for value in strata.values())
    return {
        "status": "available" if not errors else "unavailable",
        **aggregate,
        "strata": strata,
        "aggregate_gate_pass": aggregate_pass,
        "all_action_horizon_strata_pass": strata_pass,
        "gate_pass": not errors and aggregate_pass and strata_pass,
        "errors": errors,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_science_ledgers(
    science_dirs: Sequence[Path], errors: list[str]
) -> tuple[list[dict[str, Any]], LoadedLedger, LoadedLedger, LoadedLedger]:
    configs: list[dict[str, Any]] = []
    immediate_paths: list[Path] = []
    score_paths: list[Path] = []
    evidence_paths: list[Path] = []
    for directory in science_dirs:
        config = _binding_config(
            directory / "science_config.json",
            label=f"science[{directory}]",
            errors=errors,
        )
        if config is not None:
            configs.append(config)
            binding = config.get("binding")
            if isinstance(binding, dict):
                _validate_live_sources(binding, label=f"science[{directory}]", errors=errors)
        immediate_paths.append(directory / "immediate_cells.jsonl")
        score_paths.append(directory / "diagnostic_scores.jsonl")
        evidence_paths.append(directory / "matched_evidence.jsonl")
    immediate = _load_jsonl(immediate_paths, "cell_id")
    scores = _load_jsonl(score_paths, "score_record_id")
    evidence = _load_jsonl(evidence_paths, "cell_id")
    errors.extend(immediate.errors)
    errors.extend(scores.errors)
    errors.extend(evidence.errors)
    bindings = {
        str(config.get("binding_sha256"))
        for config in configs
        if isinstance(config.get("binding_sha256"), str)
    }
    if len(bindings) != 1:
        errors.append(f"science shards do not declare exactly one binding: {sorted(bindings)}")
    if configs:
        reference_binding = configs[0].get("binding")
        if any(config.get("binding") != reference_binding for config in configs[1:]):
            errors.append("science shard binding payloads are not identical")
    return configs, immediate, scores, evidence


def _validate_runner_and_manifest(
    runner_dir: Path,
    science_configs: Sequence[Mapping[str, Any]],
    live_manifest: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any] | None, LoadedLedger, dict[str, Any] | None]:
    runner_path = runner_dir / "runner_config.json"
    runner = _binding_config(runner_path, label="runner", errors=errors)
    parents = _load_jsonl((runner_dir / "parents.jsonl",), "parent_id")
    errors.extend(parents.errors)
    persisted_manifest: dict[str, Any] | None = None
    if runner is None:
        return None, parents, None
    binding = runner.get("binding")
    if isinstance(binding, dict):
        _validate_live_sources(binding, label="runner", errors=errors)
    else:
        return runner, parents, None
    runner_sha = runner.get("binding_sha256")
    runner_file_sha = _sha256_file(runner_path) if runner_path.is_file() else None
    for index, science in enumerate(science_configs):
        science_binding = science.get("binding")
        if not isinstance(science_binding, dict):
            continue
        if science_binding.get("runner_binding_sha256") != runner_sha:
            errors.append(f"science[{index}] is bound to a different runner")
        if science_binding.get("runner_config_sha256") != runner_file_sha:
            errors.append(f"science[{index}] runner_config_sha256 mismatch")
    preflight = binding.get("preflight")
    if not isinstance(preflight, dict):
        errors.append("runner binding lacks preflight provenance")
    else:
        directory_value = preflight.get("directory")
        if not isinstance(directory_value, str):
            errors.append("runner preflight directory is missing")
        else:
            preflight_dir = Path(directory_value)
            manifest_path = preflight_dir / "manifest.json"
            config_path = preflight_dir / "config.json"
            preflight_path = preflight_dir / "preflight.json"
            for path, key in (
                (manifest_path, "manifest_sha256"),
                (config_path, "config_sha256"),
                (preflight_path, "preflight_sha256"),
            ):
                if not path.is_file():
                    errors.append(f"bound preflight artifact is missing: {path}")
                elif _sha256_file(path) != preflight.get(key):
                    errors.append(f"bound preflight artifact hash mismatch: {path}")
            if manifest_path.is_file():
                try:
                    persisted_manifest = _load_json(manifest_path)
                except Exception as exc:
                    errors.append(f"persisted manifest parse failed: {type(exc).__name__}: {exc}")
                else:
                    if persisted_manifest != live_manifest:
                        errors.append(
                            "persisted source-bound manifest differs from live build_manifest()"
                        )
    if len(parents.records) != 24:
        errors.append(f"parent ledger has {len(parents.records)} rows, expected 24")
    for row in parents.records:
        if row.get("binding_sha256") != runner_sha:
            errors.append(f"parent {row.get('parent_id')} has a stale runner binding")
        field_path = row.get("field_path")
        expected_file_sha = row.get("field_file_sha256")
        if not isinstance(field_path, str) or not isinstance(expected_file_sha, str):
            errors.append(f"parent {row.get('parent_id')} lacks field provenance")
            continue
        path = Path(field_path)
        if not path.is_file() or _sha256_file(path) != expected_file_sha:
            errors.append(f"parent field file is missing or stale: {path}")
    return runner, parents, persisted_manifest


def _validate_immediate(
    ledger: LoadedLedger,
    science_binding: str | None,
    live_manifest: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    expected_rows = {
        str(row["cell_id"]): row
        for row in live_manifest["action_cells"]
        if row["cell_kind"] == "core_immediate"
    }
    _set_errors(errors, set(ledger.by_id), set(expected_rows), "immediate")
    for identifier, row in ledger.by_id.items():
        expected = expected_rows.get(identifier)
        if expected is None:
            continue
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(f"immediate {identifier} drifted manifest field {key}")
        if row.get("status") != "ok" or row.get("error") is not None:
            errors.append(f"immediate {identifier} is a terminal error row")
        if row.get("binding_sha256") != science_binding:
            errors.append(f"immediate {identifier} has a stale science binding")
        if row.get("source_cell_id") != identifier:
            errors.append(f"immediate {identifier} source_cell_id is not self-linked")
        eligible = (
            row.get("selection_scope") == "crossfit"
            and row.get("damping") == PRIMARY_DAMPING
            and row.get("rcond") == PRIMARY_RCOND
        )
        if row.get("recovery_source_eligible") is not eligible:
            errors.append(f"immediate {identifier} recovery eligibility drifted")
        hashes = row.get("source_hashes")
        if eligible:
            if not isinstance(hashes, dict) or set(hashes) != {
                "state_sha256",
                "packet_sha256",
                "native_render_sha256",
                "native_support_sha256",
                "metric_sha256",
            }:
                errors.append(f"immediate {identifier} lacks exact recovery source hashes")
        elif hashes is not None:
            errors.append(f"immediate {identifier} has unexpected recovery source hashes")
    rows = ledger.records
    crossfit = [row for row in rows if row.get("selection_scope") == "crossfit"]
    oracle = [
        row
        for row in rows
        if row.get("selection_scope") == "all_pixel_identity_and_coefficient_oracle"
    ]
    observed = {
        "immediate_cells": len(rows),
        "crossfit_immediate": len(crossfit),
        "oracle_immediate": len(oracle),
        "crossfit_by_ledger": dict(Counter(str(row.get("ledger")) for row in crossfit)),
        "oracle_by_ledger": dict(Counter(str(row.get("ledger")) for row in oracle)),
        "crossfit_by_action": dict(Counter(str(row.get("action")) for row in crossfit)),
        "oracle_by_action": dict(Counter(str(row.get("action")) for row in oracle)),
    }
    _expect(errors, len(rows) == 4_608, f"immediate count is {len(rows)}, expected 4608")
    _expect(errors, len(crossfit) == 3_072, "crossfit immediate count is not 3072")
    _expect(errors, len(oracle) == 1_536, "oracle immediate count is not 1536")
    for ledger_name in ("causal", "matched"):
        _expect(
            errors,
            observed["crossfit_by_ledger"].get(ledger_name) == 1_536,
            f"crossfit {ledger_name} ledger count is not 1536",
        )
        _expect(
            errors,
            observed["oracle_by_ledger"].get(ledger_name) == 768,
            f"oracle {ledger_name} ledger count is not 768",
        )
    for action in ALL_ACTIONS:
        _expect(
            errors,
            observed["crossfit_by_action"].get(action) == 384,
            f"crossfit action {action} count is not 384",
        )
        _expect(
            errors,
            observed["oracle_by_action"].get(action) == 192,
            f"oracle action {action} count is not 192",
        )
    return observed


def _score_coordinate(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("target_id"),
        row.get("family"),
        row.get("seed"),
        row.get("parent_horizon"),
        row.get("selection_scope"),
        row.get("heldout_fold"),
        row.get("ledger"),
        row.get("action"),
        row.get("rcond"),
    )


def _validate_scores(
    ledger: LoadedLedger,
    science_binding: str | None,
    live_manifest: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    expected_coordinates = {_score_coordinate(row) for row in live_manifest["score_plan"]}
    observed_coordinates: set[tuple[Any, ...]] = set()
    for identifier, row in ledger.by_id.items():
        coordinate = _score_coordinate(row)
        if coordinate in observed_coordinates:
            errors.append(f"duplicate diagnostic score coordinate {coordinate}")
        observed_coordinates.add(coordinate)
        if row.get("binding_sha256") != science_binding:
            errors.append(f"score {identifier} has a stale science binding")
        payload = dict(row)
        payload.pop("score_record_id", None)
        if identifier != _sha256_json(payload):
            errors.append(f"score {identifier} is not its canonical payload hash")
        action = row.get("action")
        statistic = row.get("statistic")
        expected_statistic = {
            "full_j": "unregularized_current_tangent_projector_energy",
            "tangent6": "unregularized_six_direction_projector_energy",
            "j_plus_affine6": "unregularized_projector_energy",
            "affine6": "unregularized_projector_energy",
            "j_plus_carrier6": "unregularized_projector_energy",
            "carrier6": "unregularized_projector_energy",
            "j_plus_birth2_rgb6": "finite_birth_affine_action_gain_not_projector",
            "birth2_rgb6": "finite_birth_affine_action_gain_not_projector",
        }.get(str(action))
        if statistic != expected_statistic:
            errors.append(f"score {identifier} has wrong statistic {statistic!r}")
        rank = row.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            errors.append(f"score {identifier} has invalid rank")
    missing = expected_coordinates - observed_coordinates
    unexpected = observed_coordinates - expected_coordinates
    if missing or unexpected:
        errors.append(
            "diagnostic score coordinates differ from score_plan: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    rows = ledger.records
    crossfit = [row for row in rows if row.get("selection_scope") == "crossfit"]
    oracle = [
        row
        for row in rows
        if row.get("selection_scope") == "all_pixel_identity_and_coefficient_oracle"
    ]
    births = [row for row in rows if "birth" in str(row.get("action"))]
    nonbirth = [row for row in rows if "birth" not in str(row.get("action"))]
    observed = {
        "diagnostic_scores": len(rows),
        "crossfit_scores": len(crossfit),
        "oracle_scores": len(oracle),
        "nonbirth_linear_scores": len(nonbirth),
        "finite_birth_scores": len(births),
        "crossfit_by_action": dict(Counter(str(row.get("action")) for row in crossfit)),
        "oracle_by_action": dict(Counter(str(row.get("action")) for row in oracle)),
    }
    for name, expected in (
        ("diagnostic_scores", 1_152),
        ("crossfit_scores", 768),
        ("oracle_scores", 384),
        ("nonbirth_linear_scores", 864),
        ("finite_birth_scores", 288),
    ):
        _expect(
            errors,
            observed[name] == expected,
            f"{name} count is {observed[name]}, expected {expected}",
        )
    for action in ALL_ACTIONS:
        _expect(
            errors,
            observed["crossfit_by_action"].get(action) == 96,
            f"crossfit score count for {action} is not 96",
        )
        _expect(
            errors,
            observed["oracle_by_action"].get(action) == 48,
            f"oracle score count for {action} is not 48",
        )
    return observed


def _scoring_mask_hash(scope: str, fold: int | None) -> tuple[str, int]:
    if scope == "crossfit":
        if fold is None:
            raise ValueError("crossfit scoring hash requires a fold")
        _discovery, scoring = checkerboard_fold_masks(64, 64, int(fold))
    elif scope == "all_pixel_identity_and_coefficient_oracle":
        scoring = torch.ones(64 * 64 * 3, dtype=torch.bool)
    else:
        raise ValueError(f"unknown selection scope {scope!r}")
    return tensor_sha256(scoring), int(scoring.sum())


def _expected_immediate_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    scope = str(row["selection_scope"])
    fold = None if row["heldout_fold"] is None else int(row["heldout_fold"])
    mask_sha, scoring_count = _scoring_mask_hash(scope, fold)
    record: dict[str, Any] = {
        "schema": analysis_module.SCHEMA_ID,
        "record_kind": analysis_module.RECORD_KIND,
        "component": "crossfit_immediate" if scope == "crossfit" else "all_pixel_oracle",
        "selection_scope": scope,
        "ledger": analysis_module.MATCHED_LEDGER,
        "family": row["family"],
        "seed": int(row["seed"]),
        "parent_checkpoint": int(row["parent_iters"]),
        "fold": -1 if fold is None else fold,
        "action": row["action"],
        "damping": float(row["damping"]),
        "rcond": float(row["rcond"]),
        "trust_radius": float(row["trust_radius"]),
        "checkpoint": 0,
        "status": row["status"],
        "error": row.get("error"),
        "binding_sha256": row["binding_sha256"],
    }
    record["cell_id"] = analysis_module.stable_cell_id(record)
    if row["status"] == "ok":
        record.update(
            scoring_mask_sha256=mask_sha,
            parent_mse=row["base_mse"],
            predicted_gain=row["predicted_mse_gain"],
            realized_gain=row["realized_mse_gain"],
            residual_energy_reduction=row["predicted_mse_gain"] * scoring_count,
            psnr_db=row["realized_psnr"],
            metadata={
                "rich_cell_id": row["cell_id"],
                "identity_sha256": row["identity_sha256"],
                "support": row["support"],
            },
        )
    return record


def _validate_immediate_evidence(
    immediate: LoadedLedger, evidence: LoadedLedger, errors: list[str]
) -> dict[str, Any]:
    expected: dict[str, dict[str, Any]] = {}
    for row in immediate.records:
        if row.get("ledger") != "matched":
            continue
        projection = _expected_immediate_evidence(row)
        identifier = str(projection["cell_id"])
        if identifier in expected:
            errors.append(f"duplicate expected immediate evidence ID {identifier}")
        expected[identifier] = projection
    _set_errors(errors, set(evidence.by_id), set(expected), "matched immediate evidence")
    for identifier, observed in evidence.by_id.items():
        wanted = expected.get(identifier)
        if wanted is not None and observed != wanted:
            errors.append(f"matched immediate evidence {identifier} is not an exact projection")
    crossfit = sum(row.get("component") == "crossfit_immediate" for row in evidence.records)
    oracle = sum(row.get("component") == "all_pixel_oracle" for row in evidence.records)
    _expect(errors, len(evidence.records) == 2_304, "matched immediate evidence is not 2304 rows")
    _expect(errors, crossfit == 1_536, "crossfit matched immediate evidence is not 1536")
    _expect(errors, oracle == 768, "oracle matched evidence is not 768")
    return {
        "matched_immediate_evidence": len(evidence.records),
        "matched_crossfit_immediate_evidence": crossfit,
        "matched_oracle_evidence": oracle,
        "expected_ids": set(expected),
    }


def _load_recovery_ledgers(
    recovery_dir: Path, errors: list[str]
) -> tuple[
    dict[str, Any] | None,
    LoadedLedger,
    LoadedLedger,
    LoadedLedger,
    LoadedLedger,
]:
    config = _binding_config(recovery_dir / "recovery_config.json", label="recovery", errors=errors)
    sources = _load_jsonl((recovery_dir / "recovery_sources.jsonl",), "trajectory_id")
    trajectories = _load_jsonl((recovery_dir / "recovery_trajectories.jsonl",), "trajectory_id")
    checkpoints = _load_jsonl((recovery_dir / "recovery_checkpoints.jsonl",), "checkpoint_id")
    evidence = _load_jsonl((recovery_dir / "matched_recovery_evidence.jsonl",), "cell_id")
    for ledger in (sources, trajectories, checkpoints, evidence):
        errors.extend(ledger.errors)
    if config is not None:
        binding = config.get("binding")
        if isinstance(binding, dict):
            _validate_live_sources(binding, label="recovery", errors=errors)
        source_path = recovery_dir / "recovery_sources.jsonl"
        if source_path.is_file() and config.get("recovery_sources_sha256") != _sha256_file(
            source_path
        ):
            errors.append("recovery_sources.jsonl hash differs from recovery_config")
        if config.get("prepared_source_count") != 816:
            errors.append("recovery_config prepared_source_count is not 816")
    return config, sources, trajectories, checkpoints, evidence


def _validate_recovery_config_links(
    config: Mapping[str, Any] | None,
    runner_dir: Path,
    science_dirs: Sequence[Path],
    runner_config: Mapping[str, Any] | None,
    science_configs: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> tuple[str | None, str | None]:
    if config is None:
        return None, None
    binding = config.get("binding")
    if not isinstance(binding, dict):
        return None, None
    recovery_sha = config.get("binding_sha256")
    science_sha = binding.get("science_binding_sha256")
    runner_sha = binding.get("runner_binding_sha256")
    declared_science = {item.get("binding_sha256") for item in science_configs if item is not None}
    if science_sha not in declared_science:
        errors.append("recovery binding references a different science binding")
    if runner_config is None or runner_sha != runner_config.get("binding_sha256"):
        errors.append("recovery binding references a different runner binding")
    declared_runner_dir = binding.get("runner_dir")
    if (
        not isinstance(declared_runner_dir, str)
        or Path(declared_runner_dir).resolve() != runner_dir
    ):
        errors.append("recovery binding runner_dir differs from the supplied runner")
    declared_science_dir = binding.get("science_dir")
    science_path: Path | None = None
    if isinstance(declared_science_dir, str):
        science_path = Path(declared_science_dir).resolve()
        if science_path not in {path.resolve() for path in science_dirs}:
            errors.append("recovery binding science_dir is not among supplied science inputs")
    else:
        errors.append("recovery binding lacks science_dir")
    linked_files: list[tuple[Path | None, str]] = [
        (
            None if science_path is None else science_path / "science_config.json",
            "science_config_sha256",
        ),
        (
            None if science_path is None else science_path / "immediate_cells.jsonl",
            "immediate_cells_sha256",
        ),
        (runner_dir / "runner_config.json", "runner_config_sha256"),
        (runner_dir / "parents.jsonl", "parents_jsonl_sha256"),
    ]
    for path, key in linked_files:
        if path is None or not path.is_file():
            errors.append(f"recovery bound input is missing for {key}")
        elif _sha256_file(path) != binding.get(key):
            errors.append(f"recovery bound input hash mismatch for {key}")
    for key, expected in (
        ("trajectory_count", 816),
        ("logical_checkpoint_count", 2_448),
        ("new_recovery_render_count", 1_632),
    ):
        if binding.get(key) != expected:
            errors.append(f"recovery binding {key} is not {expected}")
    return (
        str(science_sha) if isinstance(science_sha, str) else None,
        str(recovery_sha) if isinstance(recovery_sha, str) else None,
    )


def _parent_by_id(parents: LoadedLedger) -> dict[str, dict[str, Any]]:
    return {
        str(row["parent_id"]): row
        for row in parents.records
        if isinstance(row.get("parent_id"), str)
    }


def _baseline_manifest_index(
    live_manifest: Mapping[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in live_manifest["action_cells"]:
        if row.get("cell_kind") != "no_action_baseline" or row.get("selection_scope") != "crossfit":
            continue
        key = (
            row["family"],
            row["seed"],
            row["parent_iters"],
            row["heldout_fold"],
        )
        result[key] = row
    return result


def _expected_recovery_evidence(
    raw: Mapping[str, Any], immediate: Mapping[str, Any]
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": analysis_module.SCHEMA_ID,
        "record_kind": analysis_module.RECORD_KIND,
        "component": "crossfit_recovery",
        "status": "ok",
        "error": None,
        "binding_sha256": raw["binding_sha256"],
        "selection_scope": analysis_module.CROSS_FIT_SCOPE,
        "ledger": analysis_module.MATCHED_LEDGER,
        "family": raw["family"],
        "seed": raw["seed"],
        "parent_checkpoint": raw["parent_checkpoint"],
        "fold": raw["fold"],
        "action": raw["action"],
        "damping": PRIMARY_DAMPING,
        "rcond": PRIMARY_RCOND,
        "trust_radius": raw["trust_radius"],
        "checkpoint": raw["logical_checkpoint"],
        "scoring_mask_sha256": raw["scoring_mask_sha256"],
        "parent_mse": raw["parent_heldout_mse"],
        "predicted_gain": immediate["predicted_mse_gain"],
        "realized_gain": raw["heldout_realized_gain"],
        "residual_energy_reduction": immediate["residual_energy_reduction"],
        "psnr_db": raw["heldout_psnr_db"],
        "metadata": {
            "trajectory_id": raw["trajectory_id"],
            "checkpoint_id": raw["checkpoint_id"],
            "step0_source_cell_id": raw["step0_source_cell_id"],
            "state_sha256": raw["state_sha256"],
            "packet_sha256": raw["packet_sha256"],
            "native_render_sha256": raw["native_render_sha256"],
            "native_support_sha256": raw["native_support_sha256"],
            "recovery_binding_sha256": raw["recovery_binding_sha256"],
            "source_binding_sha256": raw["source_binding_sha256"],
            "manifest_recovery_cell_id": raw["manifest_recovery_cell_id"],
        },
    }
    record["cell_id"] = analysis_module.stable_cell_id(record)
    return record


def _validate_recovery(
    *,
    config: Mapping[str, Any] | None,
    sources: LoadedLedger,
    trajectories: LoadedLedger,
    checkpoints: LoadedLedger,
    evidence: LoadedLedger,
    immediate: LoadedLedger,
    parents: LoadedLedger,
    live_manifest: Mapping[str, Any],
    science_binding: str | None,
    recovery_binding: str | None,
    recovery_dir: Path,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_recovery = {str(row["cell_id"]): row for row in live_manifest["recovery_plan"]}
    baseline_index = _baseline_manifest_index(live_manifest)
    parent_index = _parent_by_id(parents)
    observed_manifest_ids: set[str] = set()
    expected_checkpoint_ids: set[str] = set()
    expected_evidence: dict[str, dict[str, Any]] = {}
    action_step0_links: set[str] = set()
    baseline_step0_links: set[str] = set()
    for trajectory_id, source in sources.by_id.items():
        plan = source.get("plan")
        if not isinstance(plan, dict):
            errors.append(f"recovery source {trajectory_id} lacks plan")
            continue
        if trajectory_id != _stable_prefixed_id("bench009-recovery-plan:", plan):
            errors.append(f"recovery source {trajectory_id} has a noncanonical plan ID")
        if source.get("binding_sha256") != recovery_binding:
            errors.append(f"recovery source {trajectory_id} has a stale binding")
        source_binding = source.get("source_binding")
        if not isinstance(source_binding, dict):
            errors.append(f"recovery source {trajectory_id} lacks source_binding")
            continue
        if source.get("source_binding_sha256") != _sha256_json(source_binding):
            errors.append(f"recovery source {trajectory_id} binding hash is noncanonical")
        manifest_id = plan.get("manifest_recovery_cell_id")
        if not isinstance(manifest_id, str):
            errors.append(f"recovery source {trajectory_id} lacks manifest link")
            continue
        observed_manifest_ids.add(manifest_id)
        manifest_row = manifest_recovery.get(manifest_id)
        if manifest_row is None:
            errors.append(f"recovery source {trajectory_id} has unknown manifest link")
            continue
        parent = parent_index.get(str(plan.get("parent_id")))
        if parent is None:
            errors.append(f"recovery source {trajectory_id} has unknown parent")
            continue
        expected_axes = {
            "family": manifest_row["family"],
            "seed": manifest_row["seed"],
            "parent_checkpoint": manifest_row["parent_iters"],
            "fold": manifest_row["heldout_fold"],
            "action": manifest_row["action"],
            "trust_radius": manifest_row["trust_radius"],
        }
        for key, value in expected_axes.items():
            if plan.get(key) != value:
                errors.append(f"recovery source {trajectory_id} drifted plan field {key}")
        for key, value in (
            ("family", parent.get("target_family")),
            ("seed", parent.get("seed")),
            ("parent_checkpoint", parent.get("iterations")),
        ):
            if plan.get(key) != value:
                errors.append(f"recovery source {trajectory_id} disagrees with parent {key}")
        if source_binding.get("manifest_recovery_cell_id") != manifest_id:
            errors.append(f"recovery source {trajectory_id} source manifest link drifted")
        if source_binding.get("manifest_recovery_row_sha256") != _sha256_json(manifest_row):
            errors.append(f"recovery source {trajectory_id} manifest row hash drifted")
        if source_binding.get("upstream_cell_id") != plan.get("step0_source_cell_id"):
            errors.append(f"recovery source {trajectory_id} upstream cell link drifted")
        action = str(plan.get("action"))
        step0_source_id = str(plan.get("step0_source_cell_id"))
        immediate_source: Mapping[str, Any] | None = None
        if action == "no_action":
            expected_baseline = baseline_index.get(
                (
                    plan.get("family"),
                    plan.get("seed"),
                    plan.get("parent_checkpoint"),
                    plan.get("fold"),
                )
            )
            if expected_baseline is None or expected_baseline.get("cell_id") != step0_source_id:
                errors.append(
                    f"baseline recovery source {trajectory_id} link is not manifest-exact"
                )
            baseline_step0_links.add(step0_source_id)
        else:
            immediate_source = immediate.by_id.get(step0_source_id)
            if immediate_source is None:
                errors.append(f"action recovery source {trajectory_id} lacks immediate row")
            else:
                wanted = {
                    "family": plan.get("family"),
                    "seed": plan.get("seed"),
                    "parent_iters": plan.get("parent_checkpoint"),
                    "heldout_fold": plan.get("fold"),
                    "action": action,
                    "trust_radius": plan.get("trust_radius"),
                    "selection_scope": "crossfit",
                    "damping": PRIMARY_DAMPING,
                    "rcond": PRIMARY_RCOND,
                }
                drift = [key for key, value in wanted.items() if immediate_source.get(key) != value]
                if drift:
                    errors.append(f"action recovery source {trajectory_id} axes drifted: {drift}")
            action_step0_links.add(step0_source_id)
        terminal = trajectories.by_id.get(trajectory_id)
        if terminal is None:
            errors.append(f"recovery source {trajectory_id} lacks terminal trajectory")
            continue
        stable_terminal = {
            "status": "ok",
            "error": None,
            "binding_sha256": science_binding,
            "recovery_binding_sha256": recovery_binding,
            "source_binding_sha256": source.get("source_binding_sha256"),
            "manifest_recovery_cell_id": manifest_id,
            "step0_source_cell_id": step0_source_id,
            "family": plan.get("family"),
            "seed": plan.get("seed"),
            "parent_checkpoint": plan.get("parent_checkpoint"),
            "parent_id": plan.get("parent_id"),
            "fold": plan.get("fold"),
            "action": action,
            "trust_radius": plan.get("trust_radius"),
        }
        for key, value in stable_terminal.items():
            if terminal.get(key) != value:
                errors.append(f"terminal {trajectory_id} drifted {key}")
        wanted_checkpoint_ids = [
            _stable_prefixed_id(
                "bench009-recovery-checkpoint:",
                {"trajectory_id": trajectory_id, "checkpoint": checkpoint},
            )
            for checkpoint in (0, 20, 100)
        ]
        expected_checkpoint_ids.update(wanted_checkpoint_ids)
        if terminal.get("logical_checkpoint_ids") != wanted_checkpoint_ids:
            errors.append(f"terminal {trajectory_id} checkpoint links are not exact")
        if terminal.get("logical_checkpoint_count") != 3:
            errors.append(f"terminal {trajectory_id} logical checkpoint count is not 3")
        if terminal.get("new_render_checkpoint_count") != 2:
            errors.append(f"terminal {trajectory_id} new-render count is not 2")
        canonical_hashes = source_binding.get("canonical_source_hashes")
        if not isinstance(canonical_hashes, dict):
            errors.append(f"recovery source {trajectory_id} lacks canonical hashes")
            canonical_hashes = {}
        trajectory_expected_evidence: list[str] = []
        for checkpoint, checkpoint_id in zip((0, 20, 100), wanted_checkpoint_ids, strict=True):
            raw = checkpoints.by_id.get(checkpoint_id)
            if raw is None:
                errors.append(f"missing checkpoint {checkpoint_id}")
                continue
            expected_fields = {
                "schema": recovery_module.RAW_CHECKPOINT_SCHEMA,
                "record_kind": recovery_module.RAW_CHECKPOINT_KIND,
                "trajectory_id": trajectory_id,
                "status": "ok",
                "error": None,
                "binding_sha256": science_binding,
                "recovery_binding_sha256": recovery_binding,
                "source_binding_sha256": source.get("source_binding_sha256"),
                "manifest_recovery_cell_id": manifest_id,
                "step0_source_cell_id": step0_source_id,
                "family": plan.get("family"),
                "seed": plan.get("seed"),
                "parent_checkpoint": plan.get("parent_checkpoint"),
                "parent_id": plan.get("parent_id"),
                "fold": plan.get("fold"),
                "action": action,
                "trust_radius": plan.get("trust_radius"),
                "logical_checkpoint": checkpoint,
                "optimizer_steps_completed": checkpoint,
                "newly_rendered_recovery_checkpoint": checkpoint in (20, 100),
                "native_dynamic_support_every_step": True,
                "topology_events": "disabled",
                "step0_link_verified": True,
            }
            for key, value in expected_fields.items():
                if raw.get(key) != value:
                    errors.append(f"checkpoint {checkpoint_id} drifted {key}")
            for key, canonical_key in (
                ("source_state_sha256", "state_sha256"),
                ("source_packet_sha256", "packet_sha256"),
                ("source_native_render_sha256", "native_render_sha256"),
                ("source_native_support_sha256", "native_support_sha256"),
                ("source_metric_sha256", "metric_sha256"),
            ):
                if raw.get(key) != canonical_hashes.get(canonical_key):
                    errors.append(f"checkpoint {checkpoint_id} source hash {key} drifted")
            if checkpoint == 0:
                for key, source_key in (
                    ("state_sha256", "source_state_sha256"),
                    ("packet_sha256", "source_packet_sha256"),
                    ("native_render_sha256", "source_native_render_sha256"),
                    ("native_support_sha256", "source_native_support_sha256"),
                ):
                    if raw.get(key) != raw.get(source_key):
                        errors.append(f"checkpoint {checkpoint_id} is not source-exact for {key}")
                if action == "no_action":
                    if raw.get("heldout_mse") != raw.get("parent_heldout_mse"):
                        errors.append(f"baseline checkpoint {checkpoint_id} changed heldout MSE")
                    if raw.get("heldout_realized_gain") != 0.0:
                        errors.append(f"baseline checkpoint {checkpoint_id} gain is not zero")
                elif immediate_source is not None:
                    exact_metrics = {
                        "heldout_mse": immediate_source.get("realized_mse"),
                        "parent_heldout_mse": immediate_source.get("base_mse"),
                        "heldout_realized_gain": immediate_source.get("realized_mse_gain"),
                    }
                    for key, value in exact_metrics.items():
                        if raw.get(key) != value:
                            errors.append(f"action checkpoint {checkpoint_id} drifted {key}")
                    source_hashes = immediate_source.get("source_hashes")
                    if isinstance(source_hashes, dict):
                        for key, value in source_hashes.items():
                            if canonical_hashes.get(key) != value:
                                errors.append(
                                    f"action checkpoint {checkpoint_id} immediate source hash {key} drifted"
                                )
                        if raw.get("source_state_sha256") != source_hashes.get("state_sha256"):
                            errors.append(
                                f"action checkpoint {checkpoint_id} source state differs from immediate"
                            )
                        if raw.get("source_packet_sha256") != source_hashes.get("packet_sha256"):
                            errors.append(
                                f"action checkpoint {checkpoint_id} source packet differs from immediate"
                            )
                        if raw.get("source_native_render_sha256") != immediate_source.get(
                            "native_render_sha256"
                        ):
                            errors.append(
                                f"action checkpoint {checkpoint_id} source render differs from immediate"
                            )
            if action in MATCHED_ACTIONS and checkpoint in (20, 100):
                if immediate_source is None:
                    errors.append(f"matched checkpoint {checkpoint_id} lacks immediate source")
                else:
                    projection = _expected_recovery_evidence(raw, immediate_source)
                    expected_evidence[str(projection["cell_id"])] = projection
                    trajectory_expected_evidence.append(str(projection["cell_id"]))
        if terminal.get("evidence_cell_ids") != trajectory_expected_evidence:
            errors.append(f"terminal {trajectory_id} evidence links are not exact")
    _set_errors(
        errors,
        observed_manifest_ids,
        set(manifest_recovery),
        "recovery manifest links",
    )
    _set_errors(errors, set(trajectories.by_id), set(sources.by_id), "terminal trajectories")
    _set_errors(errors, set(checkpoints.by_id), expected_checkpoint_ids, "recovery checkpoints")
    _set_errors(errors, set(evidence.by_id), set(expected_evidence), "matched recovery evidence")
    for identifier, observed in evidence.by_id.items():
        wanted = expected_evidence.get(identifier)
        if wanted is not None and observed != wanted:
            errors.append(f"matched recovery evidence {identifier} is not an exact projection")
    # Reuse the bound recovery validator as an independent internal cross-check.
    try:
        native_validation = recovery_module.validate_final_recovery_ledger(recovery_dir)
    except Exception as exc:
        native_validation = {
            "component_valid": False,
            "errors": [f"validator raised {type(exc).__name__}: {exc}"],
        }
    if native_validation.get("component_valid") is not True:
        errors.append("native recovery validator did not accept the complete ledger")
        errors.extend(
            f"native recovery validator: {item}"
            for item in list(native_validation.get("errors", []))[:20]
        )
    checkpoint_rows = checkpoints.records
    causal_rows = [row for row in checkpoint_rows if row.get("action") in CAUSAL_ACTIONS]
    matched_rows = [row for row in checkpoint_rows if row.get("action") in MATCHED_ACTIONS]
    baseline_rows = [row for row in checkpoint_rows if row.get("action") == "no_action"]
    by_checkpoint = Counter(int(row.get("logical_checkpoint", -1)) for row in checkpoint_rows)
    by_action = Counter(str(row.get("action")) for row in checkpoint_rows)
    new_renders = sum(
        row.get("newly_rendered_recovery_checkpoint") is True for row in checkpoint_rows
    )
    counts = {
        "recovery_sources": len(sources.records),
        "recovery_trajectories": len(trajectories.records),
        "recovery_checkpoints": len(checkpoint_rows),
        "by_checkpoint": dict(sorted(by_checkpoint.items())),
        "new_recovery_renders": new_renders,
        "causal_action_checkpoints": len(causal_rows),
        "matched_action_checkpoints": len(matched_rows),
        "baseline_checkpoints": len(baseline_rows),
        "by_action": dict(by_action),
        "action_step0_immediate_links": len(action_step0_links),
        "baseline_step0_manifest_links": len(baseline_step0_links),
        "matched_recovery_evidence": len(evidence.records),
        "native_validator": native_validation,
    }
    for name, actual, expected in (
        ("recovery_sources", len(sources.records), 816),
        ("recovery_trajectories", len(trajectories.records), 816),
        ("recovery_checkpoints", len(checkpoint_rows), 2_448),
        ("checkpoint 0", by_checkpoint.get(0, 0), 816),
        ("checkpoint 20", by_checkpoint.get(20, 0), 816),
        ("checkpoint 100", by_checkpoint.get(100, 0), 816),
        ("new recovery renders", new_renders, 1_632),
        ("causal checkpoint rows", len(causal_rows), 1_152),
        ("matched checkpoint rows", len(matched_rows), 1_152),
        ("baseline checkpoint rows", len(baseline_rows), 144),
        ("action step0 links", len(action_step0_links), 768),
        ("baseline step0 links", len(baseline_step0_links), 48),
        ("matched recovery evidence", len(evidence.records), 768),
    ):
        _expect(errors, actual == expected, f"{name} count is {actual}, expected {expected}")
    for action in ALL_ACTIONS:
        _expect(
            errors,
            by_action.get(action, 0) == 288,
            f"checkpoint count for {action} is not 288",
        )
    return counts, expected_evidence


def _score_index(scores: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in scores:
        key = (
            row.get("selection_scope"),
            row.get("family"),
            row.get("seed"),
            row.get("parent_horizon"),
            row.get("heldout_fold"),
            row.get("ledger"),
            row.get("action"),
            row.get("rcond"),
        )
        result[key] = row
    return result


def _projector_candidate(
    *,
    label: str,
    causal_action: str,
    matched_action: str,
    scope: str,
    score_index: Mapping[tuple[Any, ...], Mapping[str, Any]],
    bootstrap_indices: np.ndarray,
    errors: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    folds: tuple[int | None, ...] = FOLDS if scope == "crossfit" else (None,)
    for rcond in RCONDS:
        units: list[dict[str, Any]] = []
        family_deltas: dict[str, list[float]] = defaultdict(list)
        sum_j = 0.0
        sum_joint = 0.0
        sum_delta = 0.0
        rank_histogram: Counter[int] = Counter()
        for family in FAMILIES:
            for seed in SEEDS:
                for fold in folds:
                    base_key = (
                        scope,
                        family,
                        seed,
                        "near_plateau",
                        fold,
                    )
                    j = score_index.get(base_key + ("causal", "full_j", rcond))
                    candidate = score_index.get(base_key + ("causal", causal_action, rcond))
                    matched = score_index.get(base_key + ("matched", matched_action, rcond))
                    unit_label = f"{family}/{seed}/{fold}/rcond={rcond:.17g}"
                    if j is None or candidate is None or matched is None:
                        errors.append(f"missing projector unit {label}/{unit_label}")
                        continue
                    if j.get("statistic") != "unregularized_current_tangent_projector_energy":
                        errors.append(f"full-J projector statistic drifted at {unit_label}")
                    if candidate.get("statistic") != "unregularized_projector_energy":
                        errors.append(
                            f"candidate projector statistic drifted at {label}/{unit_label}"
                        )
                    p_j = float(j["joint_or_action_energy"])
                    delta = float(candidate["incremental_energy"])
                    joint = float(candidate["joint_or_action_energy"])
                    rank_gain = int(candidate["rank"]) - int(j["rank"])
                    if (
                        j.get("tangent_projector_energy") != p_j
                        or j.get("incremental_energy") != p_j
                    ):
                        errors.append(f"full-J projector fields disagree at {unit_label}")
                    if candidate.get("tangent_projector_energy") != p_j:
                        errors.append(
                            f"candidate tangent projector differs from full J at {label}/{unit_label}"
                        )
                    if delta != joint - p_j:
                        errors.append(
                            f"candidate incremental energy is not exact subtraction at {label}/{unit_label}"
                        )
                    if delta < 0.0:
                        errors.append(
                            f"candidate incremental energy is negative at {label}/{unit_label}"
                        )
                    if rank_gain < 0:
                        errors.append(
                            f"candidate joint rank is below full-J rank at {label}/{unit_label}"
                        )
                    duplicate_fields = (
                        "statistic",
                        "tangent_projector_energy",
                        "incremental_energy",
                        "joint_or_action_energy",
                        "rank",
                    )
                    if any(
                        candidate.get(field) != matched.get(field) for field in duplicate_fields
                    ):
                        errors.append(
                            f"causal/matched projector duplicate differs at {label}/{unit_label}"
                        )
                    sum_j += p_j
                    sum_delta += delta
                    sum_joint += joint
                    family_deltas[family].append(delta)
                    rank_histogram[rank_gain] += 1
                    units.append(
                        {
                            "family": family,
                            "seed": seed,
                            "heldout_fold": fold,
                            "p_j": p_j,
                            "delta_extension": delta,
                            "p_joint": joint,
                            "full_j_rank": int(j["rank"]),
                            "joint_rank": int(candidate["rank"]),
                            "rank_gain": rank_gain,
                            "full_j_score_record_id": j["score_record_id"],
                            "candidate_score_record_id": candidate["score_record_id"],
                            "matched_duplicate_score_record_id": matched["score_record_id"],
                        }
                    )
        expected_units = 24 if scope == "crossfit" else 12
        if len(units) != expected_units:
            errors.append(
                f"{label}/{scope}/rcond={rcond:.17g} has {len(units)} units, expected {expected_units}"
            )
        family_means = {
            family: float(np.mean(family_deltas[family]))
            for family in FAMILIES
            if len(family_deltas[family]) == len(SEEDS) * len(folds)
        }
        if sum_joint <= 0.0 or sum_j <= 0.0:
            errors.append(
                f"{label}/{scope}/rcond={rcond:.17g} has nonpositive projector denominator"
            )
        cell: dict[str, Any] = {
            "unit_count": len(units),
            "units": units,
            "sum_p_j": sum_j,
            "sum_delta_extension": sum_delta,
            "sum_p_joint": sum_joint,
            "outside_fraction": None if sum_joint <= 0.0 else sum_delta / sum_joint,
            "outside_vs_j": None if sum_j <= 0.0 else sum_delta / sum_j,
            "rank_gain_histogram": {
                str(key): value for key, value in sorted(rank_histogram.items())
            },
            "positive_rank_gain_unit_count": sum(
                count for gain, count in rank_histogram.items() if gain > 0
            ),
            "all_rank_gains_nonnegative": all(gain >= 0 for gain in rank_histogram),
            "negative_incremental_energy_unit_count": sum(
                unit["delta_extension"] < 0.0 for unit in units
            ),
            "minimum_incremental_energy": (
                None if not units else min(unit["delta_extension"] for unit in units)
            ),
            "minimum_negative_incremental_energy": (
                None
                if not any(unit["delta_extension"] < 0.0 for unit in units)
                else min(unit["delta_extension"] for unit in units if unit["delta_extension"] < 0.0)
            ),
        }
        if scope == "crossfit":
            bootstrap = _summarize_family_values(family_means, bootstrap_indices)
            cell["delta_family_bootstrap"] = bootstrap
            cell["rank_gain_compatible_with_added_truncated_direction"] = bool(
                cell["all_rank_gains_nonnegative"] and cell["positive_rank_gain_unit_count"] > 0
            )
            cell["strict_projector_condition_pass"] = (
                _strict_projector_condition_pass(cell, bootstrap)
            )
        else:
            cell["descriptive_family_means"] = family_means
        result[_rcond_label(rcond)] = cell
    if scope == "crossfit":
        result["passes_both_rconds"] = all(
            result[_rcond_label(rcond)]["strict_projector_condition_pass"] for rcond in RCONDS
        )
    return result


def _strict_projector_condition_pass(
    cell: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> bool:
    """Require the literal per-unit projector invariant as well as aggregate evidence.

    A positive family bootstrap cannot rescue even one negative incremental projector energy.
    Such a row proves that the independently truncated base and joint retained subspaces were not
    nested, so the purported literal decomposition is invalid before any aggregate statistic is
    interpreted.
    """

    return bool(
        bootstrap.get("status") == "available"
        and bootstrap.get("bootstrap_95_lower", 0.0) > 0.0
        and cell.get("rank_gain_compatible_with_added_truncated_direction") is True
        and cell.get("negative_incremental_energy_unit_count") == 0
    )


def projector_analysis(
    scores: Sequence[Mapping[str, Any]],
    bootstrap_indices: np.ndarray,
    errors: list[str],
) -> dict[str, Any]:
    index = _score_index(scores)
    result: dict[str, Any] = {}
    for label, (causal_action, matched_action) in CANDIDATES.items():
        result[label] = {
            "crossfit_near_plateau": _projector_candidate(
                label=label,
                causal_action=causal_action,
                matched_action=matched_action,
                scope="crossfit",
                score_index=index,
                bootstrap_indices=bootstrap_indices,
                errors=errors,
            ),
            "all_pixel_oracle_descriptive": _projector_candidate(
                label=label,
                causal_action=causal_action,
                matched_action=matched_action,
                scope="all_pixel_identity_and_coefficient_oracle",
                score_index=index,
                bootstrap_indices=bootstrap_indices,
                errors=errors,
            ),
        }
    birth_rows = [row for row in scores if "birth" in str(row.get("action"))]
    birth_excluded = len(birth_rows) == 288 and all(
        row.get("statistic") == "finite_birth_affine_action_gain_not_projector"
        for row in birth_rows
    )
    if not birth_excluded:
        errors.append("finite-birth scores were not exactly excluded as non-projectors")
    result["birth_excluded_as_nonprojector"] = birth_excluded
    result["birth_score_row_count"] = len(birth_rows)
    result["negative_delta_protocol_diagnosis"] = {
        "relative_rcond_joint_refactorization_can_break_projector_nesting": True,
        "rank_gain_nonnegative_does_not_imply_subspace_nesting": True,
        "mechanism": (
            "the joint [J,A] compact matrix is re-SVDed and thresholded relative to its own "
            "largest singular value; adding A can rotate or discard a base-retained mode even "
            "when the retained joint rank is no smaller"
        ),
        "consequence": (
            "a negative saved joint-minus-base energy is mathematically possible under this "
            "relative-rcond construction and invalidates its interpretation as a literal "
            "outside-J projector increment"
        ),
        "audit_policy": "fail_closed_without_tolerance_clamp_or_post_hoc_rescue",
    }
    return result


def _immediate_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row.get("selection_scope"),
            row.get("family"),
            row.get("seed"),
            row.get("parent_iters"),
            row.get("heldout_fold"),
            row.get("damping"),
            row.get("rcond"),
            row.get("trust_radius"),
            row.get("action"),
        )
        result[key] = row
    return result


def _checkpoint_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row.get("family"),
            row.get("seed"),
            row.get("parent_checkpoint"),
            row.get("fold"),
            row.get("trust_radius"),
            row.get("action"),
            row.get("logical_checkpoint"),
        )
        result[key] = row
    return result


def _family_means_from_units(
    unit_values: Mapping[str, Sequence[float]],
    *,
    expected_per_family: int,
    label: str,
    errors: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for family in FAMILIES:
        values = list(unit_values.get(family, ()))
        if len(values) != expected_per_family:
            errors.append(
                f"{label}/{family} has {len(values)} pairs, expected {expected_per_family}"
            )
            continue
        result[family] = float(np.mean(np.asarray(values, dtype=np.float64)))
    return result


def finite_causal_analysis(
    immediate_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    bootstrap_indices: np.ndarray,
    errors: list[str],
) -> dict[str, Any]:
    immediate_index = _immediate_index(immediate_rows)
    recovery_index = _checkpoint_index(checkpoint_rows)
    result: dict[str, Any] = {}
    for label, (candidate_action, _matched_action) in CANDIDATES.items():
        candidate_result: dict[str, Any] = {"radii": {}}
        for radius in RADII:
            effect_units: dict[str, dict[str, list[float]]] = {
                "immediate_vs_stronger_control_db": defaultdict(list),
                "predicted_gain_vs_full_j_mse": defaultdict(list),
                "realized_vs_full_j_db": defaultdict(list),
            }
            selected_control_counts: Counter[str] = Counter()
            unit_details: list[dict[str, Any]] = []
            for family in FAMILIES:
                for seed in SEEDS:
                    for fold in FOLDS:
                        base = (
                            "crossfit",
                            family,
                            seed,
                            NEAR_PLATEAU,
                            fold,
                            PRIMARY_DAMPING,
                            PRIMARY_RCOND,
                            radius,
                        )
                        candidate = immediate_index.get(base + (candidate_action,))
                        full_j = immediate_index.get(base + ("full_j",))
                        birth = immediate_index.get(base + ("j_plus_birth2_rgb6",))
                        if candidate is None or full_j is None or birth is None:
                            errors.append(
                                f"missing finite causal immediate pair {label}/{family}/{seed}/{fold}/{radius}"
                            )
                            continue
                        controls = [full_j, birth]
                        stronger = max(controls, key=lambda row: float(row["realized_psnr"]))
                        selected_control_counts[str(stronger["action"])] += 1
                        effects = {
                            "immediate_vs_stronger_control_db": (
                                float(candidate["realized_psnr"]) - float(stronger["realized_psnr"])
                            ),
                            "predicted_gain_vs_full_j_mse": (
                                float(candidate["predicted_mse_gain"])
                                - float(full_j["predicted_mse_gain"])
                            ),
                            "realized_vs_full_j_db": (
                                float(candidate["realized_psnr"]) - float(full_j["realized_psnr"])
                            ),
                        }
                        for name, value in effects.items():
                            effect_units[name][family].append(value)
                        unit_details.append(
                            {
                                "family": family,
                                "seed": seed,
                                "fold": fold,
                                "stronger_control": stronger["action"],
                                **effects,
                            }
                        )
            effect_summaries: dict[str, Any] = {}
            for name, values in effect_units.items():
                family_means = _family_means_from_units(
                    values,
                    expected_per_family=4,
                    label=f"{label}/radius={radius}/{name}",
                    errors=errors,
                )
                effect_summaries[name] = _summarize_family_values(family_means, bootstrap_indices)
            recovery: dict[str, Any] = {}
            for checkpoint in (20, 100):
                recovery_units: dict[str, list[float]] = defaultdict(list)
                recovery_controls: Counter[str] = Counter()
                for family in FAMILIES:
                    for seed in SEEDS:
                        for fold in FOLDS:
                            fixed = (family, seed, NEAR_PLATEAU, fold)
                            candidate = recovery_index.get(
                                fixed + (radius, candidate_action, checkpoint)
                            )
                            full_j = recovery_index.get(fixed + (radius, "full_j", checkpoint))
                            birth = recovery_index.get(
                                fixed + (radius, "j_plus_birth2_rgb6", checkpoint)
                            )
                            if candidate is None or full_j is None or birth is None:
                                errors.append(
                                    f"missing finite causal recovery pair {label}/{family}/{seed}/{fold}/{radius}/{checkpoint}"
                                )
                                continue
                            stronger = max(
                                (full_j, birth),
                                key=lambda row: float(row["heldout_psnr_db"]),
                            )
                            recovery_controls[str(stronger["action"])] += 1
                            recovery_units[family].append(
                                float(candidate["heldout_psnr_db"])
                                - float(stronger["heldout_psnr_db"])
                            )
                family_means = _family_means_from_units(
                    recovery_units,
                    expected_per_family=4,
                    label=f"{label}/radius={radius}/recovery={checkpoint}",
                    errors=errors,
                )
                recovery[str(checkpoint)] = {
                    **_summarize_family_values(family_means, bootstrap_indices),
                    "selected_stronger_control_counts": dict(recovery_controls),
                }
            immediate_primary = effect_summaries["immediate_vs_stronger_control_db"]
            checks = {
                "immediate_mean_at_least_frozen_0.15_db": (
                    immediate_primary.get("status") == "available"
                    and immediate_primary.get("mean", -math.inf) >= manifest_module.PSNR_SURVIVAL_DB
                ),
                "immediate_bootstrap_lower_above_zero": (
                    immediate_primary.get("status") == "available"
                    and immediate_primary.get("bootstrap_95_lower", -math.inf) > 0.0
                ),
                "immediate_positive_in_at_least_four_families": (
                    immediate_primary.get("status") == "available"
                    and immediate_primary.get("positive_family_count", 0)
                    >= manifest_module.MINIMUM_POSITIVE_FAMILIES
                ),
                "recovery_mean_positive": {
                    checkpoint: recovery[str(checkpoint)].get("status") == "available"
                    and recovery[str(checkpoint)].get("mean", -math.inf) > 0.0
                    for checkpoint in (20, 100)
                },
            }
            radius_pass = bool(
                all(value for key, value in checks.items() if key != "recovery_mean_positive")
                and all(checks["recovery_mean_positive"].values())
            )
            candidate_result["radii"][str(radius)] = {
                "trust_radius": radius,
                "unit_count": len(unit_details),
                "units": unit_details,
                "effects": effect_summaries,
                "selected_stronger_control_counts": dict(selected_control_counts),
                "recovery": recovery,
                "checks": checks,
                "finite_realization_and_recovery_pass": radius_pass,
            }
        candidate_result["passes_both_radii"] = all(
            value["finite_realization_and_recovery_pass"]
            for value in candidate_result["radii"].values()
        )
        result[label] = candidate_result
    return result


def run_matched_analyzer(
    immediate_evidence: LoadedLedger,
    recovery_evidence: LoadedLedger,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    combined = list(immediate_evidence.records) + list(recovery_evidence.records)
    expected_ids = {str(row["cell_id"]) for row in combined}
    components = Counter(str(row.get("component")) for row in combined)
    counts = {
        "crossfit_immediate": components.get("crossfit_immediate", 0),
        "all_pixel_oracle": components.get("all_pixel_oracle", 0),
        "crossfit_recovery": components.get("crossfit_recovery", 0),
        "combined": len(combined),
    }
    for name, expected in (
        ("crossfit_immediate", 1_536),
        ("all_pixel_oracle", 768),
        ("crossfit_recovery", 768),
        ("combined", 3_072),
    ):
        if counts[name] != expected:
            errors.append(
                f"matched evidence component {name} has {counts[name]} rows, expected {expected}"
            )
    bindings = {str(row.get("binding_sha256")) for row in combined}
    expected_binding = next(iter(bindings)) if len(bindings) == 1 else None
    if expected_binding is None:
        errors.append("matched evidence union does not have exactly one binding")
        return {
            "status": "unavailable",
            "errors": ["mixed evidence bindings"],
        }, counts
    with tempfile.TemporaryDirectory(prefix="bench009-causal-audit-") as temporary:
        path = Path(temporary) / "matched_evidence_union.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                for row in combined
            ),
            encoding="utf-8",
        )
        loaded = analysis_module.load_jsonl(
            path,
            expected_ids=expected_ids,
            expected_binding_sha256=expected_binding,
        )
        result = analysis_module.analyze_evidence(loaded)
    if result.get("status") != "available":
        errors.append("frozen matched analyzer returned unavailable")
        errors.extend(f"matched analyzer: {item}" for item in list(result.get("errors", []))[:20])
    return result, counts


def _identity_key(label: str, row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    identity = row.get("identity")
    if not isinstance(identity, dict):
        return None
    if label == "affine6":
        return (identity.get("parent_row"),)
    return (identity.get("parent_row"), identity.get("frequency_index"))


def locality_transport_analysis(
    immediate_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    bootstrap_indices: np.ndarray,
    errors: list[str],
) -> dict[str, Any]:
    immediate = _immediate_index(immediate_rows)
    scores = _score_index(score_rows)
    result: dict[str, Any] = {}
    for label, (candidate_action, _matched_action) in CANDIDATES.items():
        discovery_units: dict[str, list[float]] = defaultdict(list)
        for family in FAMILIES:
            for seed in SEEDS:
                for fold in FOLDS:
                    key = (
                        "crossfit",
                        family,
                        seed,
                        "near_plateau",
                        fold,
                    )
                    candidate = scores.get(key + ("causal", candidate_action, PRIMARY_RCOND))
                    if candidate is None:
                        errors.append(
                            f"missing locality discovery score {label}/{family}/{seed}/{fold}"
                        )
                    else:
                        discovery_units[family].append(float(candidate["incremental_energy"]))
        discovery_means = _family_means_from_units(
            discovery_units,
            expected_per_family=4,
            label=f"{label}/locality/discovery",
            errors=errors,
        )
        discovery_summary = _summarize_family_values(discovery_means, bootstrap_indices)
        radii: dict[str, Any] = {}
        all_negative_transport = True
        for radius in RADII:
            heldout_units: dict[str, list[float]] = defaultdict(list)
            optimism_units: dict[str, list[float]] = defaultdict(list)
            nonlinear_failure_count = 0
            pair_count = 0
            for family in FAMILIES:
                for seed in SEEDS:
                    fold_effects: list[float] = []
                    for fold in FOLDS:
                        fixed = (
                            "crossfit",
                            family,
                            seed,
                            NEAR_PLATEAU,
                            fold,
                            PRIMARY_DAMPING,
                            PRIMARY_RCOND,
                            radius,
                        )
                        candidate = immediate.get(fixed + (candidate_action,))
                        control = immediate.get(fixed + ("full_j",))
                        if candidate is None or control is None:
                            errors.append(
                                f"missing locality crossfit pair {label}/{family}/{seed}/{fold}/{radius}"
                            )
                            continue
                        predicted = float(candidate["predicted_mse_gain"]) - float(
                            control["predicted_mse_gain"]
                        )
                        realized = float(candidate["realized_psnr"]) - float(
                            control["realized_psnr"]
                        )
                        heldout_units[family].append(predicted)
                        fold_effects.append(predicted)
                        pair_count += 1
                        nonlinear_failure_count += int(predicted > 0.0 and realized <= 0.0)
                    oracle_fixed = (
                        "all_pixel_identity_and_coefficient_oracle",
                        family,
                        seed,
                        NEAR_PLATEAU,
                        None,
                        PRIMARY_DAMPING,
                        PRIMARY_RCOND,
                        radius,
                    )
                    oracle_candidate = immediate.get(oracle_fixed + (candidate_action,))
                    oracle_control = immediate.get(oracle_fixed + ("full_j",))
                    if oracle_candidate is None or oracle_control is None or len(fold_effects) != 2:
                        errors.append(
                            f"missing locality oracle pair {label}/{family}/{seed}/{radius}"
                        )
                    else:
                        oracle_effect = float(oracle_candidate["predicted_mse_gain"]) - float(
                            oracle_control["predicted_mse_gain"]
                        )
                        optimism_units[family].append(oracle_effect - float(np.mean(fold_effects)))
            heldout_means = _family_means_from_units(
                heldout_units,
                expected_per_family=4,
                label=f"{label}/locality/heldout/{radius}",
                errors=errors,
            )
            heldout_summary = _summarize_family_values(heldout_means, bootstrap_indices)
            negated_summary = _summarize_family_values(
                {family: -value for family, value in heldout_means.items()},
                bootstrap_indices,
            )
            optimism_means = _family_means_from_units(
                optimism_units,
                expected_per_family=2,
                label=f"{label}/locality/optimism/{radius}",
                errors=errors,
            )
            optimism_summary = _summarize_family_values(optimism_means, bootstrap_indices)
            negative_transport = bool(
                negated_summary.get("status") == "available"
                and negated_summary.get("bootstrap_95_lower", -math.inf) > 0.0
            )
            all_negative_transport = all_negative_transport and negative_transport
            radii[str(radius)] = {
                "heldout_predicted_delta_vs_full_j": heldout_summary,
                "bootstrap_of_negated_heldout_delta": negated_summary,
                "oracle_optimism_gap": optimism_summary,
                "prediction_positive_but_realization_nonpositive_count": nonlinear_failure_count,
                "paired_unit_count": pair_count,
                "transported_prediction_strictly_negative": negative_transport,
            }
        agreement_units: list[dict[str, Any]] = []
        agreement_count = 0
        for family in FAMILIES:
            for seed in SEEDS:
                identities: dict[str, tuple[Any, ...] | None] = {}
                for scope, fold, name in (
                    ("crossfit", 0, "fold0"),
                    ("crossfit", 1, "fold1"),
                    ("all_pixel_identity_and_coefficient_oracle", None, "oracle"),
                ):
                    key = (
                        scope,
                        family,
                        seed,
                        NEAR_PLATEAU,
                        fold,
                        PRIMARY_DAMPING,
                        PRIMARY_RCOND,
                        RADII[0],
                        candidate_action,
                    )
                    row = immediate.get(key)
                    identities[name] = None if row is None else _identity_key(label, row)
                agrees = (
                    identities["fold0"] is not None
                    and identities["fold0"] == identities["fold1"] == identities["oracle"]
                )
                agreement_count += int(agrees)
                agreement_units.append(
                    {
                        "family": family,
                        "seed": seed,
                        "fold0": identities["fold0"],
                        "fold1": identities["fold1"],
                        "oracle": identities["oracle"],
                        "all_three_agree": agrees,
                    }
                )
        discovery_positive = bool(
            discovery_summary.get("status") == "available"
            and discovery_summary.get("bootstrap_95_lower", -math.inf) > 0.0
        )
        result[label] = {
            "discovery_incremental_energy": discovery_summary,
            "radii": radii,
            "identity_agreement": {
                "agreement_count": agreement_count,
                "unit_count": len(agreement_units),
                "units": agreement_units,
            },
            "locality_split_failure_supported": discovery_positive and all_negative_transport,
            "oracle_is_descriptive_only": True,
        }
    return result


def current_grammar_headroom_analysis(
    immediate_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    bootstrap_indices: np.ndarray,
    errors: list[str],
) -> dict[str, Any]:
    immediate = _immediate_index(immediate_rows)
    recovery = _checkpoint_index(checkpoint_rows)
    radii: dict[str, Any] = {}
    for radius in RADII:
        immediate_units: dict[str, list[float]] = defaultdict(list)
        for family in FAMILIES:
            for seed in SEEDS:
                for fold in FOLDS:
                    key = (
                        "crossfit",
                        family,
                        seed,
                        NEAR_PLATEAU,
                        fold,
                        PRIMARY_DAMPING,
                        PRIMARY_RCOND,
                        radius,
                        "full_j",
                    )
                    row = immediate.get(key)
                    if row is None:
                        errors.append(
                            f"missing full-J immediate headroom row {family}/{seed}/{fold}/{radius}"
                        )
                    else:
                        immediate_units[family].append(float(row["realized_psnr_gain_db"]))
        immediate_means = _family_means_from_units(
            immediate_units,
            expected_per_family=4,
            label=f"full-J headroom/{radius}/immediate",
            errors=errors,
        )
        immediate_summary = _summarize_family_values(immediate_means, bootstrap_indices)
        recovery_results: dict[str, Any] = {}
        for checkpoint in (20, 100):
            recovery_units: dict[str, list[float]] = defaultdict(list)
            for family in FAMILIES:
                for seed in SEEDS:
                    for fold in FOLDS:
                        fixed = (family, seed, NEAR_PLATEAU, fold)
                        full_j = recovery.get(fixed + (radius, "full_j", checkpoint))
                        baseline = recovery.get(fixed + (None, "no_action", checkpoint))
                        if full_j is None or baseline is None:
                            errors.append(
                                f"missing full-J/no-action recovery pair {family}/{seed}/{fold}/{radius}/{checkpoint}"
                            )
                        else:
                            recovery_units[family].append(
                                float(full_j["heldout_psnr_db"])
                                - float(baseline["heldout_psnr_db"])
                            )
            family_means = _family_means_from_units(
                recovery_units,
                expected_per_family=4,
                label=f"full-J headroom/{radius}/recovery/{checkpoint}",
                errors=errors,
            )
            recovery_results[str(checkpoint)] = _summarize_family_values(
                family_means, bootstrap_indices
            )
        checks = {
            "immediate_bootstrap_lower_above_zero": (
                immediate_summary.get("status") == "available"
                and immediate_summary.get("bootstrap_95_lower", -math.inf) > 0.0
            ),
            "recovery_mean_positive": {
                checkpoint: recovery_results[str(checkpoint)].get("status") == "available"
                and recovery_results[str(checkpoint)].get("mean", -math.inf) > 0.0
                for checkpoint in (20, 100)
            },
        }
        radii[str(radius)] = {
            "immediate_full_j_vs_parent_db": immediate_summary,
            "recovery_full_j_vs_no_action_db": recovery_results,
            "checks": checks,
            "passes": checks["immediate_bootstrap_lower_above_zero"]
            and all(checks["recovery_mean_positive"].values()),
        }
    supported = all(value["passes"] for value in radii.values())
    caught_by_100 = any(
        value["recovery_full_j_vs_no_action_db"]["100"].get("mean", math.inf) <= 0.0
        for value in radii.values()
    )
    return {
        "radii": radii,
        "supported": supported,
        "trajectory_interpretation": (
            "convergence_acceleration_only_if_valid"
            if caught_by_100
            else "conditioning_or_trajectory_headroom_if_valid"
        ),
        "parent_objective_deficit_not_established": True,
    }


def _validity_pass(matched: Mapping[str, Any], all_eight: Mapping[str, Any]) -> bool:
    gates = matched.get("gates")
    return bool(
        matched.get("status") == "available"
        and isinstance(gates, dict)
        and gates.get("validity_pass") is True
        and all_eight.get("gate_pass") is True
    )


def _matched_candidate_survives(matched: Mapping[str, Any], matched_action: str) -> bool:
    survival = matched.get("matched_survival")
    if not isinstance(survival, dict):
        return False
    actions = survival.get("action_survival")
    return bool(
        isinstance(actions, dict)
        and isinstance(actions.get(matched_action), dict)
        and actions[matched_action].get("survives") is True
    )


def _classifications(
    *,
    integrity_pass: bool,
    validity_pass: bool,
    matched: Mapping[str, Any],
    projector: Mapping[str, Any],
    finite: Mapping[str, Any],
    locality: Mapping[str, Any],
    headroom: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_details: dict[str, Any] = {}
    any_matched_survives = False
    any_outside = False
    all_zero_rank_when_matched = True
    for label, (_causal_action, matched_action) in CANDIDATES.items():
        matched_survives = _matched_candidate_survives(matched, matched_action)
        any_matched_survives = any_matched_survives or matched_survives
        projector_crossfit = projector.get(label, {}).get("crossfit_near_plateau", {})
        projector_pass = bool(projector_crossfit.get("passes_both_rconds"))
        finite_pass = bool(finite.get(label, {}).get("passes_both_radii"))
        strict = matched_survives and projector_pass and finite_pass
        any_outside = any_outside or strict
        zero_ranks = all(
            cell.get("positive_rank_gain_unit_count") == 0
            for key, cell in projector_crossfit.items()
            if key.startswith("rcond=") and isinstance(cell, dict)
        )
        if matched_survives:
            all_zero_rank_when_matched = all_zero_rank_when_matched and zero_ranks
        candidate_details[label] = {
            "matched_utility_survives_both_radii": matched_survives,
            "literal_projector_passes_both_rconds": projector_pass,
            "finite_causal_and_recovery_passes_both_radii": finite_pass,
            "strict_post_run_outside_j_diagnostic_pass": strict,
            "all_rank_gains_zero_at_both_rconds": zero_ranks,
            "requires_disjoint_objective_aligned_replication": True,
        }
    available = integrity_pass and validity_pass
    if not available:
        classification = {
            "inside_j_reparameterization": "unavailable",
            "outside_j_capacity": "unavailable",
            "current_grammar_mse_headroom": "unavailable",
            "locality_split_failure": "unavailable",
            "candidate_details": candidate_details,
        }
    else:
        if any_outside:
            inside = "not_supported"
        elif any_matched_survives:
            # Even all-zero truncated-rank gain does not prove membership in exact span(J).
            inside = "consistent"
        else:
            inside = "not_supported"
        classification = {
            "inside_j_reparameterization": inside,
            "outside_j_capacity": "supported" if any_outside else "not_supported",
            "current_grammar_mse_headroom": (
                "supported" if headroom.get("supported") is True else "not_supported"
            ),
            "locality_split_failure": (
                "supported"
                if any(
                    value.get("locality_split_failure_supported") is True
                    for value in locality.values()
                    if isinstance(value, dict)
                )
                else "not_supported"
            ),
            "candidate_details": candidate_details,
            "inside_j_wording": (
                "consistent_with_the_seeded_truncated_span_j"
                if inside == "consistent" and all_zero_rank_when_matched
                else "useful_six_dof_basis_without_outside_j_establishment"
                if inside == "consistent"
                else None
            ),
        }
    if not available:
        next_experiment = "complete_or_repair_the_frozen_ledgers_before_interpretation"
    elif any_outside:
        next_experiment = "disjoint_objective_aligned_replication_before_codec_authorization"
    elif classification["locality_split_failure"] == "supported":
        next_experiment = "locality_balanced_capacity_assay_as_a_new_experiment"
    elif any_matched_survives:
        next_experiment = "transient_tangent_distillation_or_reduced_manifold_fitting"
    elif classification["current_grammar_mse_headroom"] == "supported":
        next_experiment = "objective_aligned_reduced_space_lm_control"
    else:
        next_experiment = "stop_richer_local_syntax_on_this_screen"
    authorization = {
        "compression_claim": False,
        "parent_objective_optimizer_deficit_claim": False,
        "scientific_interpretation": False,
        "codec_native_followup_without_replication": False,
        "next_experiment": next_experiment,
        "reason": (
            "post-run causal diagnostics were not preregistered as a publishable gate and the "
            "parent objective differs from the RGB-MSE auction/recovery objective"
        ),
    }
    return classification, authorization


def run_audit(
    *,
    runner_dir: str | Path,
    science_dirs: Sequence[str | Path],
    recovery_dir: str | Path,
) -> dict[str, Any]:
    errors: list[str] = []
    isolation = source_binding_isolation()
    if not isolation["pass"]:
        errors.append("causal audit file is included in an experiment source binding")
    try:
        live_manifest = manifest_module.build_manifest()
    except Exception as exc:
        return {
            "schema": SCHEMA_ID,
            "status": "unavailable",
            "scope": SCOPE,
            "integrity": {
                "counts": {},
                "bindings_match": False,
                "manifest_exact": False,
                "step0_links_exact": False,
                "source_binding_isolation": isolation,
                "error_count": 1,
                "errors": [f"live manifest construction failed: {type(exc).__name__}: {exc}"],
            },
            "classification": {
                "inside_j_reparameterization": "unavailable",
                "outside_j_capacity": "unavailable",
                "current_grammar_mse_headroom": "unavailable",
                "locality_split_failure": "unavailable",
            },
            "authorization": {
                "compression_claim": False,
                "parent_objective_optimizer_deficit_claim": False,
                "scientific_interpretation": False,
                "next_experiment": "restore_the_source_bound_manifest",
            },
        }
    runner_path = Path(runner_dir).resolve()
    science_paths = tuple(Path(path).resolve() for path in science_dirs)
    recovery_path = Path(recovery_dir).resolve()
    science_configs, immediate, scores, immediate_evidence = _load_science_ledgers(
        science_paths, errors
    )
    runner_config, parents, _persisted_manifest = _validate_runner_and_manifest(
        runner_path, science_configs, live_manifest, errors
    )
    science_binding = (
        str(science_configs[0].get("binding_sha256"))
        if science_configs and isinstance(science_configs[0].get("binding_sha256"), str)
        else None
    )
    immediate_counts = _validate_immediate(immediate, science_binding, live_manifest, errors)
    score_counts = _validate_scores(scores, science_binding, live_manifest, errors)
    immediate_evidence_counts = _validate_immediate_evidence(immediate, immediate_evidence, errors)
    (
        recovery_config,
        recovery_sources,
        trajectories,
        checkpoints,
        recovery_evidence,
    ) = _load_recovery_ledgers(recovery_path, errors)
    recovery_science_binding, recovery_binding = _validate_recovery_config_links(
        recovery_config,
        runner_path,
        science_paths,
        runner_config,
        science_configs,
        errors,
    )
    if recovery_science_binding is not None and recovery_science_binding != science_binding:
        errors.append("recovery and science bindings differ")
    recovery_counts, _expected_recovery_evidence = _validate_recovery(
        config=recovery_config,
        sources=recovery_sources,
        trajectories=trajectories,
        checkpoints=checkpoints,
        evidence=recovery_evidence,
        immediate=immediate,
        parents=parents,
        live_manifest=live_manifest,
        science_binding=science_binding,
        recovery_binding=recovery_binding,
        recovery_dir=recovery_path,
        errors=errors,
    )
    bootstrap_indices = _bootstrap_indices()
    projector = projector_analysis(scores.records, bootstrap_indices, errors)
    finite = finite_causal_analysis(
        immediate.records, checkpoints.records, bootstrap_indices, errors
    )
    locality = locality_transport_analysis(
        immediate.records, scores.records, bootstrap_indices, errors
    )
    headroom = current_grammar_headroom_analysis(
        immediate.records, checkpoints.records, bootstrap_indices, errors
    )
    matched_analyzer, evidence_union_counts = run_matched_analyzer(
        immediate_evidence, recovery_evidence, errors
    )
    all_eight = all_eight_action_calibration(immediate.records)
    validity_pass = _validity_pass(matched_analyzer, all_eight)
    integrity_pass = not errors
    classification, authorization = _classifications(
        integrity_pass=integrity_pass,
        validity_pass=validity_pass,
        matched=matched_analyzer,
        projector=projector,
        finite=finite,
        locality=locality,
        headroom=headroom,
    )
    error_count = len(errors)
    bindings_match = not any(
        any(token in error.lower() for token in ("binding", "bound source", "config_sha256"))
        for error in errors
    )
    manifest_exact = not any("manifest" in error.lower() for error in errors)
    step0_links_exact = (
        recovery_counts.get("action_step0_immediate_links") == 768
        and recovery_counts.get("baseline_step0_manifest_links") == 48
        and not any("checkpoint" in error.lower() and "source" in error.lower() for error in errors)
    )
    counts = {
        "expected": EXPECTED_COUNTS,
        "immediate": immediate_counts,
        "diagnostic_scores": score_counts,
        "immediate_matched_evidence": {
            key: value for key, value in immediate_evidence_counts.items() if key != "expected_ids"
        },
        "recovery": recovery_counts,
        "matched_evidence_union": evidence_union_counts,
    }
    result = {
        "schema": SCHEMA_ID,
        "status": "available" if integrity_pass and validity_pass else "unavailable",
        "scope": SCOPE,
        "interpretation_boundary": {
            "parent_training_objective": "0.7_l1_plus_0.3_one_minus_ssim",
            "auction_and_recovery_objective": "rgb_mse",
            "parent_optimizer_deficit_claim_available": False,
            "strongest_permitted_headroom_wording": (
                "current_grammar_rgb_mse_headroom_after_an_l1_ssim_trained_parent"
            ),
            "oracle_rows_are_descriptive_only": True,
            "finite_birth_is_not_a_projector": True,
        },
        "integrity": {
            "counts": counts,
            "bindings_match": bindings_match,
            "manifest_exact": manifest_exact,
            "step0_links_exact": step0_links_exact,
            "source_binding_isolation": isolation,
            "error_count": error_count,
            "errors": errors[:300],
            "errors_truncated": error_count > 300,
        },
        "validity": {
            "matched_analyzer": matched_analyzer,
            "all_eight_action_calibration": all_eight,
            "gate_pass": validity_pass,
            "classification_policy": (
                "global calibration, all 16 action-by-horizon strata, and frozen matched "
                "winner stability must pass before causal classification"
            ),
        },
        "causal_projector": projector,
        "finite_causal": finite,
        "matched_utility": matched_analyzer.get("matched_survival"),
        "recovery": {
            "ledger_counts": recovery_counts,
            "current_grammar_headroom": headroom,
        },
        "locality_transport": locality,
        "classification": classification,
        "authorization": authorization,
        "post_run_diagnostic": True,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "rng": "numpy.PCG64",
            "seed": BOOTSTRAP_SEED,
            "shared_index_matrix_shape": list(bootstrap_indices.shape),
            "shared_across_all_family_statistics": True,
        },
    }
    _canonical_json(result)
    return result


def run_self_test(toy_artifact: str | Path | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    isolation = source_binding_isolation()
    checks["source_binding_isolation"] = isolation["pass"]
    indices_a = _bootstrap_indices()
    indices_b = _bootstrap_indices()
    checks["bootstrap_indices_deterministic"] = bool(np.array_equal(indices_a, indices_b))
    values = {family: float(index + 1) for index, family in enumerate(FAMILIES)}
    first = _summarize_family_values(values, indices_a)
    second = _summarize_family_values(values, indices_b)
    checks["family_bootstrap_deterministic"] = first == second
    with tempfile.TemporaryDirectory(prefix="bench009-causal-audit-selftest-") as temporary:
        duplicate_path = Path(temporary) / "duplicate.jsonl"
        duplicate_path.write_text('{"id":"a","x":1,"x":2}\n', encoding="utf-8")
        duplicate = _load_jsonl((duplicate_path,), "id")
        checks["duplicate_json_keys_rejected"] = bool(duplicate.errors)
        nonfinite_path = Path(temporary) / "nonfinite.jsonl"
        nonfinite_path.write_text('{"id":"a","x":1e309}\n', encoding="utf-8")
        nonfinite = _load_jsonl((nonfinite_path,), "id")
        checks["nonfinite_numbers_rejected"] = bool(nonfinite.errors)
    toy_validation: dict[str, Any] | None = None
    if toy_artifact is not None:
        try:
            toy_validation = science_module.validate_toy_preflight(toy_artifact)
            checks["source_bound_science_toy_valid"] = bool(
                toy_validation.get("preflight_pass") is True
                and toy_validation.get("development_targets_accessed") is False
            )
        except Exception as exc:
            toy_validation = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            checks["source_bound_science_toy_valid"] = False
    return {
        "schema": f"{SCHEMA_ID}.self-test",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "bootstrap_reference": first,
        "toy_validation": toy_validation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-dir")
    parser.add_argument(
        "--science-dir",
        action="append",
        default=[],
        help="science artifact directory; repeat only for disjoint shards",
    )
    parser.add_argument("--recovery-dir")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--toy-artifact", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        result = run_self_test(args.toy_artifact)
        if args.output is not None:
            _write_json_atomic(args.output, result)
        else:
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "pass" else 2
    missing = [
        name
        for name, value in (
            ("--runner-dir", args.runner_dir),
            ("--science-dir", args.science_dir),
            ("--recovery-dir", args.recovery_dir),
            ("--output", args.output),
        )
        if not value
    ]
    if missing:
        parser.error(f"full audit requires {', '.join(missing)}")
    result = run_audit(
        runner_dir=args.runner_dir,
        science_dirs=args.science_dir,
        recovery_dir=args.recovery_dir,
    )
    assert args.output is not None
    _write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": result["status"],
                "integrity_errors": result["integrity"]["error_count"],
                "validity_gate_pass": result["validity"]["gate_pass"],
                "classification": result["classification"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if result["status"] == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_RELATIVE_PATH",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EXPECTED_COUNTS",
    "SCHEMA_ID",
    "all_eight_action_calibration",
    "finite_causal_analysis",
    "locality_transport_analysis",
    "main",
    "projector_analysis",
    "run_audit",
    "run_self_test",
    "source_binding_isolation",
]
