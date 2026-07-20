"""Fail-closed evidence analysis for BENCH-009 Stage 1.

The scientific runner may emit richer artifacts, but the analyzer consumes one compact JSONL
projection.  Each line is a terminal record with this schema (``metadata`` is the only optional
extension point)::

    {
      "schema": "bench009.stage1.matched-evidence.v1",
      "record_kind": "matched_action_evidence",
      "component": "crossfit_immediate" | "crossfit_recovery" |
                   "all_pixel_oracle",
      "status": "ok" | "error", "error": null | "...",
      "cell_id": "bench009-evidence:<sha256 of the stable key>",
      "binding_sha256": "<64 lowercase hex>",
      "selection_scope": "crossfit" |
                         "all_pixel_identity_and_coefficient_oracle",
      "ledger": "matched_six_dof",
      "family": "...", "seed": 101, "parent_checkpoint": 160, "fold": 0,
      "action": "tangent6" | "affine6" | "carrier6" | "birth2_rgb6",
      "damping": 0.001, "rcond": 0.00001, "trust_radius": 0.25,
      "checkpoint": 0 | 20 | 100,
      "scoring_mask_sha256": "<64 lowercase hex>",
      "parent_mse": 0.01, "predicted_gain": 0.001,
      "realized_gain": 0.0009, "residual_energy_reduction": 1.2,
      "psnr_db": 31.2, "metadata": {}
    }

The stable key comprises every field from ``schema`` through ``checkpoint`` except status,
binding, and metrics.  Error records retain the stable key but may omit metric fields; their
presence invalidates the component.  Callers supply the expected stable cell IDs and the frozen
binding hash.  Missing, duplicate, unexpected, stale, nonfinite, hash-mismatched, and terminal
error rows return ``status='unavailable'`` rather than a partial statistic.

Only cross-fit rows enter gates.  Oracle rows are parsed and counted but are intentionally ignored
by every scientific statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_ID = "bench009.stage1.matched-evidence.v1"
RECORD_KIND = "matched_action_evidence"
MATCHED_LEDGER = "matched_six_dof"
CROSS_FIT_SCOPE = "crossfit"
ORACLE_SCOPE = "all_pixel_identity_and_coefficient_oracle"
COMPONENTS = ("crossfit_immediate", "crossfit_recovery", "all_pixel_oracle")
ACTIONS = ("tangent6", "affine6", "carrier6", "birth2_rgb6")
CANDIDATE_ACTIONS = ("affine6", "carrier6")
CONTROL_ACTIONS = ("tangent6", "birth2_rgb6")
TARGET_FAMILIES = (
    "ramp_shading",
    "step_junction",
    "thin_curve",
    "stationary_sinusoid",
    "chirp",
    "natural_texture",
)
SEEDS = (101, 211)
PARENT_CHECKPOINTS = (24, 160)
FOLDS = (0, 1)
DAMPINGS = (1e-3, 1e-2)
RCONDS = (1e-5, 1e-4)
TRUST_RADII = (0.25, 0.75)
RECOVERY_CHECKPOINTS = (20, 100)
PRIMARY_DAMPING = 1e-3
PRIMARY_RCOND = 1e-5
NEAR_PLATEAU_CHECKPOINT = 160
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_715
GAIN_RATIO_BOUNDS = (0.5, 2.0)
SPEARMAN_MINIMUM = 0.8
ENERGY_SURVIVAL_RATIO = 1.25
PSNR_SURVIVAL_DB = 0.15
MINIMUM_POSITIVE_FAMILIES = 4

STABLE_KEY_FIELDS = (
    "schema",
    "record_kind",
    "component",
    "selection_scope",
    "ledger",
    "family",
    "seed",
    "parent_checkpoint",
    "fold",
    "action",
    "damping",
    "rcond",
    "trust_radius",
    "checkpoint",
)
BASE_FIELDS = frozenset(
    (*STABLE_KEY_FIELDS, "status", "error", "cell_id", "binding_sha256")
)
METRIC_FIELDS = frozenset(
    (
        "scoring_mask_sha256",
        "parent_mse",
        "predicted_gain",
        "realized_gain",
        "residual_energy_reduction",
        "psnr_db",
    )
)
ALLOWED_FIELDS = BASE_FIELDS | METRIC_FIELDS | {"metadata"}
HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class AnalysisAxes:
    """Axes used by the frozen scientific analysis.

    Overrides exist for unit tests only.  The module-level ``FROZEN_AXES`` is the BENCH-009
    contract and enforces exactly four seed-by-fold pairs per target family.
    """

    families: tuple[str, ...] = TARGET_FAMILIES
    seeds: tuple[int, ...] = SEEDS
    parent_checkpoints: tuple[int, ...] = PARENT_CHECKPOINTS
    folds: tuple[int, ...] = FOLDS
    dampings: tuple[float, ...] = DAMPINGS
    rconds: tuple[float, ...] = RCONDS
    trust_radii: tuple[float, ...] = TRUST_RADII
    actions: tuple[str, ...] = ACTIONS
    near_plateau_checkpoint: int = NEAR_PLATEAU_CHECKPOINT
    primary_damping: float = PRIMARY_DAMPING
    primary_rcond: float = PRIMARY_RCOND
    recovery_checkpoints: tuple[int, ...] = RECOVERY_CHECKPOINTS


FROZEN_AXES = AnalysisAxes()


@dataclass(frozen=True)
class EvidenceLoad:
    """Result of strict JSONL parsing and component-completeness validation."""

    status: str
    records: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    expected_count: int | None
    observed_count: int

    @property
    def available(self) -> bool:
        return self.status == "available"


class _DuplicateJsonKey(ValueError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def stable_cell_id(record: Mapping[str, Any]) -> str:
    """Return the content-derived ID for a record's declared stable key."""

    missing = [field for field in STABLE_KEY_FIELDS if field not in record]
    if missing:
        raise ValueError(f"stable key missing fields: {missing}")
    key = {field: record[field] for field in STABLE_KEY_FIELDS}
    return "bench009-evidence:" + hashlib.sha256(_canonical_json(key)).hexdigest()


def expected_cell_ids(keys: Iterable[Mapping[str, Any]]) -> frozenset[str]:
    """Convert manifest-like stable keys into the IDs accepted by :func:`load_jsonl`."""

    ids = [stable_cell_id(key) for key in keys]
    if len(ids) != len(set(ids)):
        raise ValueError("expected stable keys contain duplicates")
    return frozenset(ids)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS)
    )


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value}")


def _validate_record(record: Any, line_number: int) -> list[str]:
    prefix = f"line {line_number}"
    if not isinstance(record, dict):
        return [f"{prefix}: top-level JSON value is not an object"]
    errors: list[str] = []
    unknown = sorted(set(record) - ALLOWED_FIELDS)
    if unknown:
        errors.append(f"{prefix}: unknown fields {unknown}")
    missing = sorted(BASE_FIELDS - set(record))
    if missing:
        errors.append(f"{prefix}: missing base fields {missing}")
        return errors

    if record["schema"] != SCHEMA_ID:
        errors.append(f"{prefix}: wrong schema {record['schema']!r}")
    if record["record_kind"] != RECORD_KIND:
        errors.append(f"{prefix}: wrong record_kind {record['record_kind']!r}")
    if record["component"] not in COMPONENTS:
        errors.append(f"{prefix}: unknown component {record['component']!r}")
    if record["selection_scope"] not in (CROSS_FIT_SCOPE, ORACLE_SCOPE):
        errors.append(f"{prefix}: unknown selection_scope {record['selection_scope']!r}")
    if record["ledger"] != MATCHED_LEDGER:
        errors.append(f"{prefix}: wrong ledger {record['ledger']!r}")
    if not isinstance(record["family"], str) or not record["family"]:
        errors.append(f"{prefix}: family must be a non-empty string")
    for field in ("seed", "parent_checkpoint", "fold", "checkpoint"):
        if not isinstance(record[field], int) or isinstance(record[field], bool):
            errors.append(f"{prefix}: {field} must be an integer")
    if record["action"] not in ACTIONS:
        errors.append(f"{prefix}: unknown action {record['action']!r}")
    for field in ("damping", "rcond", "trust_radius"):
        if not _finite_number(record[field]) or float(record[field]) <= 0.0:
            errors.append(f"{prefix}: {field} must be finite and positive")
    if not _is_sha256(record["binding_sha256"]):
        errors.append(f"{prefix}: invalid binding_sha256")
    if record["status"] not in ("ok", "error"):
        errors.append(f"{prefix}: status must be 'ok' or 'error'")
    if record["status"] == "error":
        if not isinstance(record["error"], str) or not record["error"]:
            errors.append(f"{prefix}: terminal error record needs non-empty error text")
    elif record["error"] is not None:
        errors.append(f"{prefix}: ok record must set error to null")

    try:
        wanted_cell_id = stable_cell_id(record)
    except (TypeError, ValueError) as exc:
        errors.append(f"{prefix}: invalid stable key: {exc}")
    else:
        if record["cell_id"] != wanted_cell_id:
            errors.append(f"{prefix}: cell_id does not hash the stable key")

    if record["component"] == "crossfit_immediate":
        if record["selection_scope"] != CROSS_FIT_SCOPE or record["checkpoint"] != 0:
            errors.append(f"{prefix}: crossfit_immediate needs crossfit scope/checkpoint 0")
    elif record["component"] == "crossfit_recovery":
        if (
            record["selection_scope"] != CROSS_FIT_SCOPE
            or record["checkpoint"] not in RECOVERY_CHECKPOINTS
        ):
            errors.append(
                f"{prefix}: crossfit_recovery needs crossfit scope/checkpoint 20 or 100"
            )
    elif record["component"] == "all_pixel_oracle":
        if record["selection_scope"] != ORACLE_SCOPE:
            errors.append(f"{prefix}: oracle component needs oracle selection scope")

    if record["status"] == "ok":
        metric_missing = sorted(METRIC_FIELDS - set(record))
        if metric_missing:
            errors.append(f"{prefix}: ok record missing metric fields {metric_missing}")
        else:
            if not _is_sha256(record["scoring_mask_sha256"]):
                errors.append(f"{prefix}: invalid scoring_mask_sha256")
            for field in METRIC_FIELDS - {"scoring_mask_sha256"}:
                if not _finite_number(record[field]):
                    errors.append(f"{prefix}: {field} must be finite")
            if _finite_number(record["parent_mse"]) and record["parent_mse"] < 0.0:
                errors.append(f"{prefix}: parent_mse must be nonnegative")
    if "metadata" in record and not isinstance(record["metadata"], dict):
        errors.append(f"{prefix}: metadata must be an object")
    return errors


def load_jsonl(
    path: str | Path,
    *,
    expected_ids: Collection[str] | None,
    expected_binding_sha256: str | None,
) -> EvidenceLoad:
    """Strictly parse a terminal component and fail closed on any integrity problem."""

    source = Path(path)
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    if expected_binding_sha256 is not None and not _is_sha256(expected_binding_sha256):
        errors.append("expected binding is not a lowercase SHA-256")
    if not source.is_file():
        return EvidenceLoad(
            status="unavailable",
            records=(),
            errors=(f"JSONL source is missing: {source}",),
            expected_count=None if expected_ids is None else len(expected_ids),
            observed_count=0,
        )
    try:
        lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        return EvidenceLoad(
            status="unavailable",
            records=(),
            errors=(f"cannot read JSONL source: {exc}",),
            expected_count=None if expected_ids is None else len(expected_ids),
            observed_count=0,
        )
    if not lines:
        errors.append("JSONL source is empty")
    seen_ids: dict[str, int] = {}
    bindings: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            errors.append(f"line {line_number}: blank JSONL line")
            continue
        try:
            record = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"line {line_number}: invalid strict JSON: {exc}")
            continue
        validation_errors = _validate_record(record, line_number)
        errors.extend(validation_errors)
        if not isinstance(record, dict):
            continue
        cell_id = record.get("cell_id")
        if isinstance(cell_id, str):
            if cell_id in seen_ids:
                errors.append(
                    f"line {line_number}: duplicate cell_id first seen on line "
                    f"{seen_ids[cell_id]}"
                )
            else:
                seen_ids[cell_id] = line_number
        binding = record.get("binding_sha256")
        if isinstance(binding, str):
            bindings.add(binding)
            if (
                expected_binding_sha256 is not None
                and binding != expected_binding_sha256
            ):
                errors.append(f"line {line_number}: stale binding_sha256")
        if record.get("status") == "error":
            errors.append(f"line {line_number}: terminal error row")
        records.append(record)

    if len(bindings) > 1:
        errors.append("component mixes multiple binding hashes")
    observed_ids = set(seen_ids)
    if expected_ids is not None:
        expected = set(expected_ids)
        invalid_expected = sorted(cell for cell in expected if not isinstance(cell, str))
        if invalid_expected:
            errors.append("expected cell IDs contain non-string values")
        missing = sorted(expected - observed_ids)
        unexpected = sorted(observed_ids - expected)
        if missing:
            errors.append(f"missing expected cells ({len(missing)}): {missing[:3]}")
        if unexpected:
            errors.append(f"unexpected cells ({len(unexpected)}): {unexpected[:3]}")

    status = "available" if not errors else "unavailable"
    return EvidenceLoad(
        status=status,
        records=tuple(records),
        errors=tuple(errors),
        expected_count=None if expected_ids is None else len(expected_ids),
        observed_count=len(records),
    )


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
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denominator)


def _crossfit(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [record for record in records if record["selection_scope"] == CROSS_FIT_SCOPE]


def _same_mask_errors(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Ensure every action sibling uses the same scoring mask and parent MSE."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    sibling_fields = (
        "component",
        "family",
        "seed",
        "parent_checkpoint",
        "fold",
        "damping",
        "rcond",
        "trust_radius",
        "checkpoint",
    )
    for record in records:
        key = tuple(record[field] for field in sibling_fields)
        grouped.setdefault(key, []).append(record)
    errors: list[str] = []
    for key, siblings in grouped.items():
        mask_hashes = {record["scoring_mask_sha256"] for record in siblings}
        parent_mses = {float(record["parent_mse"]) for record in siblings}
        if len(mask_hashes) != 1:
            errors.append(f"scoring-mask mismatch for sibling cell {key}")
        if len(parent_mses) != 1:
            errors.append(f"parent-MSE mismatch for sibling cell {key}")
    return errors


def calibration_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute floor-aware calibration globally and per action/parent horizon.

    The per-stratum requirement prevents exactly linear packet actions from masking a failed
    nonlinear tangent realization when rows are pooled.
    """

    rows = [
        record
        for record in _crossfit(records)
        if record["component"] == "crossfit_immediate" and record["checkpoint"] == 0
    ]
    mask_errors = _same_mask_errors(rows)
    def summarize(selected_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        eligible: list[Mapping[str, Any]] = []
        excluded: list[Mapping[str, Any]] = []
        for record in selected_rows:
            floor = max(1e-12, float(record["parent_mse"]) * 1e-5)
            (eligible if float(record["predicted_gain"]) > floor else excluded).append(
                record
            )
        cell: dict[str, Any] = {
            "raw_row_count": len(selected_rows),
            "eligible_row_count": len(eligible),
            "at_or_below_floor_count": len(excluded),
            "at_or_below_floor_cell_ids": [record["cell_id"] for record in excluded],
        }
        predicted = np.asarray(
            [record["predicted_gain"] for record in eligible], dtype=np.float64
        )
        realized = np.asarray(
            [record["realized_gain"] for record in eligible], dtype=np.float64
        )
        enough_non_tied = (
            len(eligible) >= 3
            and len(np.unique(predicted)) >= 3
            and len(np.unique(realized)) >= 3
        )
        if not enough_non_tied:
            cell.update(
                spearman=None,
                median_realized_over_predicted=None,
                enough_non_tied_eligible_rows=False,
                gate_pass=False,
                failure_reason="fewer than three eligible non-tied rows",
            )
            return cell
        spearman = _spearman(predicted, realized)
        median_ratio = float(np.median(realized / predicted))
        cell.update(
            spearman=spearman,
            median_realized_over_predicted=median_ratio,
            enough_non_tied_eligible_rows=True,
            spearman_minimum=SPEARMAN_MINIMUM,
            gain_ratio_bounds=list(GAIN_RATIO_BOUNDS),
            gate_pass=(
                math.isfinite(spearman)
                and spearman >= SPEARMAN_MINIMUM
                and GAIN_RATIO_BOUNDS[0] <= median_ratio <= GAIN_RATIO_BOUNDS[1]
            ),
        )
        return cell

    aggregate = summarize(rows)
    result: dict[str, Any] = {
        "status": "available",
        **aggregate,
        "same_scoring_mask": not mask_errors,
        "errors": mask_errors,
    }
    if mask_errors:
        result.update(status="unavailable", gate_pass=False)
        return result
    strata: dict[str, Any] = {}
    checkpoints = sorted({int(record["parent_checkpoint"]) for record in rows})
    for action in ACTIONS:
        for checkpoint in checkpoints:
            key = f"action={action}|parent_checkpoint={checkpoint}"
            strata[key] = summarize(
                [
                    record
                    for record in rows
                    if record["action"] == action
                    and int(record["parent_checkpoint"]) == checkpoint
                ]
            )
    result["aggregate_gate_pass"] = bool(aggregate["gate_pass"])
    result["strata"] = strata
    result["all_action_horizon_strata_pass"] = bool(strata) and all(
        cell["gate_pass"] for cell in strata.values()
    )
    result["gate_pass"] = bool(
        result["aggregate_gate_pass"] and result["all_action_horizon_strata_pass"]
    )
    return result


def _record_index(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[Any, ...], Mapping[str, Any]], list[str]]:
    fields = (
        "component",
        "family",
        "seed",
        "parent_checkpoint",
        "fold",
        "action",
        "damping",
        "rcond",
        "trust_radius",
        "checkpoint",
    )
    index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    errors: list[str] = []
    for record in _crossfit(records):
        key = tuple(record[field] for field in fields)
        if key in index:
            errors.append(f"duplicate scientific coordinate {key}")
        index[key] = record
    return index, errors


def _lookup_key(
    *,
    component: str,
    family: str,
    seed: int,
    parent_checkpoint: int,
    fold: int,
    action: str,
    damping: float,
    rcond: float,
    trust_radius: float,
    checkpoint: int,
) -> tuple[Any, ...]:
    return (
        component,
        family,
        seed,
        parent_checkpoint,
        fold,
        action,
        damping,
        rcond,
        trust_radius,
        checkpoint,
    )


def winner_stability(
    records: Sequence[Mapping[str, Any]], axes: AnalysisAxes = FROZEN_AXES
) -> dict[str, Any]:
    """Audit numerical-grid stability separately at each frozen parent checkpoint.

    Pooling underfit and near-plateau parents can hide an instability when their changes point in
    opposite directions.  Conversely, requiring the same winner *between* checkpoints would erase
    the deliberately preregistered optimizer-state contrast.  We therefore vary damping, rcond,
    and radius within each ``family x parent_checkpoint`` stratum, and mark a target family as
    changed when either checkpoint has more than one grid winner.
    """

    index, errors = _record_index(records)
    winners_by_stratum: dict[str, dict[str, dict[str, str]]] = {}
    for family in axes.families:
        horizon_winners: dict[str, dict[str, str]] = {}
        for parent_checkpoint in axes.parent_checkpoints:
            grid_winners: dict[str, str] = {}
            for damping in axes.dampings:
                for rcond in axes.rconds:
                    for radius in axes.trust_radii:
                        means: dict[str, float] = {}
                        for action in axes.actions:
                            values: list[float] = []
                            for seed in axes.seeds:
                                for fold in axes.folds:
                                    key = _lookup_key(
                                        component="crossfit_immediate",
                                        family=family,
                                        seed=seed,
                                        parent_checkpoint=parent_checkpoint,
                                        fold=fold,
                                        action=action,
                                        damping=damping,
                                        rcond=rcond,
                                        trust_radius=radius,
                                        checkpoint=0,
                                    )
                                    record = index.get(key)
                                    if record is None:
                                        errors.append(f"missing winner-stability row {key}")
                                    else:
                                        values.append(float(record["psnr_db"]))
                            if values:
                                means[action] = float(np.mean(values))
                        grid_label = f"d={damping:.17g}|r={rcond:.17g}|t={radius:.17g}"
                        if len(means) != len(axes.actions):
                            continue
                        maximum = max(means.values())
                        tied = [
                            action
                            for action, value in means.items()
                            if math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12)
                        ]
                        if len(tied) != 1:
                            errors.append(
                                "ambiguous tied winner for "
                                f"{family}/checkpoint={parent_checkpoint} {grid_label}: {tied}"
                            )
                        else:
                            grid_winners[grid_label] = tied[0]
            horizon_winners[str(parent_checkpoint)] = grid_winners
        winners_by_stratum[family] = horizon_winners
    if errors:
        return {
            "status": "unavailable",
            "errors": errors,
            "gate_pass": False,
            "winners_by_family_and_parent_checkpoint": winners_by_stratum,
        }
    unique_winners = {
        family: {
            checkpoint: sorted(set(grid.values()))
            for checkpoint, grid in horizons.items()
        }
        for family, horizons in winners_by_stratum.items()
    }
    changed_horizons = {
        family: [
            checkpoint
            for checkpoint, winners in horizons.items()
            if len(winners) > 1
        ]
        for family, horizons in unique_winners.items()
    }
    changed_families = [
        family for family, checkpoints in changed_horizons.items() if checkpoints
    ]
    return {
        "status": "available",
        "errors": [],
        "winners_by_family_and_parent_checkpoint": winners_by_stratum,
        "unique_winners_by_family_and_parent_checkpoint": unique_winners,
        "changed_parent_checkpoints_by_family": changed_horizons,
        "changed_families": changed_families,
        "changed_family_count": len(changed_families),
        "maximum_changed_families": 1,
        "gate_pass": len(changed_families) <= 1,
    }


def _bootstrap_family_means(
    family_means: np.ndarray, bootstrap_indices: np.ndarray
) -> tuple[float, float]:
    samples = np.mean(family_means[bootstrap_indices], axis=1)
    return float(np.mean(family_means)), float(np.percentile(samples, 2.5))


def matched_survival_statistics(
    records: Sequence[Mapping[str, Any]], axes: AnalysisAxes = FROZEN_AXES
) -> dict[str, Any]:
    """Compute paired near-plateau survival evidence, separately by radius/checkpoint."""

    if len(axes.families) != 6:
        return {
            "status": "unavailable",
            "errors": ["paired family bootstrap requires exactly six target families"],
            "gate_pass": False,
        }
    if len(axes.seeds) * len(axes.folds) != 4:
        return {
            "status": "unavailable",
            "errors": ["family means require exactly four seed-by-fold pairs"],
            "gate_pass": False,
        }
    index, errors = _record_index(records)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_indices = rng.integers(
        0,
        len(axes.families),
        size=(BOOTSTRAP_REPLICATES, len(axes.families)),
        endpoint=False,
    )
    cells: dict[str, dict[str, dict[str, Any]]] = {
        action: {} for action in CANDIDATE_ACTIONS
    }
    checkpoints = (0, *axes.recovery_checkpoints)
    for action in CANDIDATE_ACTIONS:
        for radius in axes.trust_radii:
            for checkpoint in checkpoints:
                component = "crossfit_immediate" if checkpoint == 0 else "crossfit_recovery"
                family_psnr_means: list[float] = []
                family_candidate_energy: list[float] = []
                family_control_energy: list[float] = []
                family_detail: dict[str, Any] = {}
                selected_control_counts = {control: 0 for control in CONTROL_ACTIONS}
                for family in axes.families:
                    psnr_deltas: list[float] = []
                    candidate_energy: list[float] = []
                    control_energy: list[float] = []
                    for seed in axes.seeds:
                        for fold in axes.folds:
                            fixed = {
                                "component": component,
                                "family": family,
                                "seed": seed,
                                "parent_checkpoint": axes.near_plateau_checkpoint,
                                "fold": fold,
                                "damping": axes.primary_damping,
                                "rcond": axes.primary_rcond,
                                "trust_radius": radius,
                                "checkpoint": checkpoint,
                            }
                            candidate_key = _lookup_key(action=action, **fixed)
                            candidate = index.get(candidate_key)
                            controls = [
                                index.get(_lookup_key(action=control, **fixed))
                                for control in CONTROL_ACTIONS
                            ]
                            if candidate is None:
                                errors.append(f"missing paired candidate row {candidate_key}")
                                continue
                            if any(control is None for control in controls):
                                errors.append(
                                    "missing paired control row for "
                                    f"{family}/{seed}/{fold}/{radius}/{checkpoint}"
                                )
                                continue
                            concrete_controls = [control for control in controls if control is not None]
                            strongest_psnr = max(
                                concrete_controls, key=lambda row: float(row["psnr_db"])
                            )
                            selected_control_counts[str(strongest_psnr["action"])] += 1
                            psnr_deltas.append(
                                float(candidate["psnr_db"])
                                - float(strongest_psnr["psnr_db"])
                            )
                            if checkpoint == 0:
                                strongest_energy = max(
                                    float(control["residual_energy_reduction"])
                                    for control in concrete_controls
                                )
                                candidate_energy.append(
                                    float(candidate["residual_energy_reduction"])
                                )
                                control_energy.append(strongest_energy)
                    if len(psnr_deltas) != 4:
                        errors.append(
                            f"expected four paired rows for {action}/{family}/{radius}/{checkpoint}"
                        )
                        continue
                    psnr_mean = float(np.mean(psnr_deltas))
                    family_psnr_means.append(psnr_mean)
                    detail: dict[str, Any] = {
                        "paired_count": len(psnr_deltas),
                        "mean_psnr_delta_db": psnr_mean,
                    }
                    if checkpoint == 0:
                        if len(candidate_energy) != 4 or len(control_energy) != 4:
                            errors.append(
                                f"expected four energy pairs for {action}/{family}/{radius}"
                            )
                        else:
                            candidate_mean = float(np.mean(candidate_energy))
                            control_mean = float(np.mean(control_energy))
                            family_candidate_energy.append(candidate_mean)
                            family_control_energy.append(control_mean)
                            detail.update(
                                mean_candidate_residual_energy_reduction=candidate_mean,
                                mean_stronger_control_residual_energy_reduction=control_mean,
                            )
                    family_detail[family] = detail
                label = f"radius={radius:.17g}|checkpoint={checkpoint}"
                if len(family_psnr_means) != len(axes.families):
                    continue
                family_array = np.asarray(family_psnr_means, dtype=np.float64)
                mean_delta, lower = _bootstrap_family_means(
                    family_array, bootstrap_indices
                )
                positive_count = int(np.count_nonzero(family_array > 0.0))
                cell: dict[str, Any] = {
                    "trust_radius": radius,
                    "checkpoint": checkpoint,
                    "family_means": family_detail,
                    "mean_psnr_delta_db": mean_delta,
                    "bootstrap_95_lower_db": lower,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_rng": "numpy.PCG64",
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "positive_family_count": positive_count,
                    "selected_psnr_control_counts": selected_control_counts,
                }
                if checkpoint == 0:
                    control_total = float(np.sum(family_control_energy))
                    if control_total <= 0.0:
                        errors.append(
                            f"nonpositive stronger-control energy for {action}/{radius}"
                        )
                        energy_ratio = None
                    else:
                        # Equal family weight: each sum contains one mean per family.
                        energy_ratio = float(
                            np.sum(family_candidate_energy) / control_total
                        )
                    cell["residual_energy_ratio_vs_stronger_control"] = energy_ratio
                cells[action][label] = cell

    if errors:
        return {
            "status": "unavailable",
            "errors": errors,
            "cells": cells,
            "gate_pass": False,
        }

    action_survival: dict[str, Any] = {}
    for action in CANDIDATE_ACTIONS:
        radius_checks: dict[str, Any] = {}
        for radius in axes.trust_radii:
            immediate = cells[action][f"radius={radius:.17g}|checkpoint=0"]
            recovery_positive = {
                str(checkpoint): (
                    cells[action][
                        f"radius={radius:.17g}|checkpoint={checkpoint}"
                    ]["mean_psnr_delta_db"]
                    > 0.0
                )
                for checkpoint in axes.recovery_checkpoints
            }
            checks = {
                "energy_ratio_at_least_1.25": (
                    immediate["residual_energy_ratio_vs_stronger_control"]
                    >= ENERGY_SURVIVAL_RATIO
                ),
                "immediate_psnr_delta_at_least_0.15_db": (
                    immediate["mean_psnr_delta_db"] >= PSNR_SURVIVAL_DB
                ),
                "immediate_bootstrap_lower_above_zero": (
                    immediate["bootstrap_95_lower_db"] > 0.0
                ),
                "immediate_positive_in_at_least_four_families": (
                    immediate["positive_family_count"] >= MINIMUM_POSITIVE_FAMILIES
                ),
                "recovery_mean_positive": recovery_positive,
            }
            radius_pass = (
                all(value for key, value in checks.items() if key != "recovery_mean_positive")
                and all(recovery_positive.values())
            )
            radius_checks[str(radius)] = {"checks": checks, "survives": radius_pass}
        action_survival[action] = {
            "radii": radius_checks,
            "survives": all(item["survives"] for item in radius_checks.values()),
        }
    return {
        "status": "available",
        "errors": [],
        "cells": cells,
        "action_survival": action_survival,
        "gate_pass": any(item["survives"] for item in action_survival.values()),
    }


def analyze_evidence(
    evidence: EvidenceLoad, axes: AnalysisAxes = FROZEN_AXES
) -> dict[str, Any]:
    """Run all evidence gates, propagating unavailable input without partial claims."""

    if not evidence.available:
        return {
            "status": "unavailable",
            "errors": list(evidence.errors),
            "input_expected_count": evidence.expected_count,
            "input_observed_count": evidence.observed_count,
            "gates": None,
        }
    records = evidence.records
    same_mask_errors = _same_mask_errors(_crossfit(records))
    if same_mask_errors:
        return {
            "status": "unavailable",
            "errors": same_mask_errors,
            "input_expected_count": evidence.expected_count,
            "input_observed_count": evidence.observed_count,
            "gates": None,
        }
    calibration = calibration_statistics(records)
    stability = winner_stability(records, axes)
    survival = matched_survival_statistics(records, axes)
    unavailable = [
        name
        for name, result in (
            ("calibration", calibration),
            ("winner_stability", stability),
            ("matched_survival", survival),
        )
        if result["status"] == "unavailable"
    ]
    if unavailable:
        return {
            "status": "unavailable",
            "errors": [f"unavailable analysis component: {name}" for name in unavailable],
            "input_expected_count": evidence.expected_count,
            "input_observed_count": evidence.observed_count,
            "oracle_rows_excluded": sum(
                record["selection_scope"] == ORACLE_SCOPE for record in records
            ),
            "calibration": calibration,
            "winner_stability": stability,
            "matched_survival": survival,
            "gates": None,
        }
    validity_pass = bool(calibration["gate_pass"] and stability["gate_pass"])
    return {
        "status": "available",
        "errors": [],
        "input_expected_count": evidence.expected_count,
        "input_observed_count": evidence.observed_count,
        "oracle_rows_excluded": sum(
            record["selection_scope"] == ORACLE_SCOPE for record in records
        ),
        "calibration": calibration,
        "winner_stability": stability,
        "matched_survival": survival,
        "gates": {
            "validity_pass": validity_pass,
            "representation_survival_pass": bool(survival["gate_pass"]),
            "matched_six_dof_utility_screen_pass": bool(
                validity_pass and survival["gate_pass"]
            ),
            # A six-column extension can lie wholly in span(J) yet be a better conditioned or
            # better aligned six-dimensional basis than six selected coordinate columns.  The
            # matched screen therefore cannot, by itself, establish new expressiveness.  The
            # separately persisted causal projector ledger and an objective-alignment audit are
            # required before any representation interpretation or implementation promotion.
            "causal_outside_j_audit_present": False,
            "objective_alignment_audit_present": False,
            "scientific_interpretation_authorized": False,
        },
        "interpretation_boundary": {
            "matched_survival_is_not_outside_j_evidence": True,
            "parent_objective": "0.7_l1_plus_0.3_one_minus_ssim",
            "auction_and_recovery_objective": "rgb_mse",
            "required_supplements": [
                "causal_projector_and_joint_action_audit",
                "objective_aligned_parent_or_optimizer_diagnostic",
            ],
        },
    }


__all__ = [
    "ACTIONS",
    "AnalysisAxes",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EvidenceLoad",
    "FROZEN_AXES",
    "SCHEMA_ID",
    "analyze_evidence",
    "calibration_statistics",
    "expected_cell_ids",
    "load_jsonl",
    "matched_survival_statistics",
    "stable_cell_id",
    "winner_stability",
]
