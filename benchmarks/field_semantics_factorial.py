"""BENCH-020 sealed field-semantics and alpha-policy factorial.

The module is deliberately an experiment controller, not a second conversion pipeline.  It
freezes source/prepared artifacts, clean implementations, ordered initial-geometry banks, exact
semantic arms, raw-byte ledgers, work/loss/gate contracts, metrics, and killing rules.  External
or repository-native executors consume the generated cell plan and return schema-bound rows.

There are three outcome-separated phases:

1. fixed-geometry coefficient/DC elimination;
2. matched development factorial after a deterministic domain lock; and
3. sealed confirmation after a distinct review of the development decision.

The confirmation root must still be empty when the lock is created.  No routine in this module
can relabel a normalized field as additive or invent an independently supervised mass target.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_SCHEMA = "structsplat.bench020.protocol.v1"
REVIEW_SCHEMA = "structsplat.bench020.protocol_review.v1"
ROW_SCHEMA = "structsplat.bench020.result.v1"
FIELD_MANIFEST_SCHEMA = "structsplat.bench020.field_manifest.v1"
METRICS_SCHEMA = "structsplat.bench020.metrics.v1"
HISTORY_SCHEMA = "structsplat.bench020.history.v1"
DOMAIN_LOCK_SCHEMA = "structsplat.bench020.domain_lock.v1"
DEVELOPMENT_ANALYSIS_SCHEMA = "structsplat.bench020.development_analysis.v1"
DEVELOPMENT_REVIEW_SCHEMA = "structsplat.bench020.development_review.v1"
CONFIRMATION_LOCK_SCHEMA = "structsplat.bench020.confirmation_lock.v1"
CONFIRMATION_ANALYSIS_SCHEMA = "structsplat.bench020.confirmation_analysis.v1"
RESULTS_AUDIT_SCHEMA = "structsplat.bench020.results_audit.v1"
REPORT_SCHEMA = "structsplat.bench020.report.v1"

PHASES = ("coefficient_screen", "development", "confirmation")
ADDITIVE_FAMILIES = frozenset({"incumbent_factorized_additive", "direct_additive", "dual_additive"})
REQUIRED_FAMILIES = frozenset(
    {
        "incumbent_factorized_additive",
        "direct_additive",
        "normalized_plain",
        "normalized_maintained",
    }
)
COEFFICIENT_VARIANTS = frozenset(
    {"zero_dc_nonnegative", "counted_dc_signed_bounded", "not_applicable"}
)
ALPHA_POLICIES = frozenset({"alpha_gated", "hard_contained", "not_applicable"})
EXECUTION_KINDS = frozenset({"matched", "maintained_reference", "native_authentic"})
METRIC_DIRECTIONS = frozenset({"higher", "lower"})
REQUIRED_REPORT_ARTIFACTS = (
    "field_manifest",
    "field_payload",
    "metrics_json",
    "history_json",
    "raw_render",
    "evaluated_render",
)


class ProtocolError(ValueError):
    """Fail-closed protocol, row, lock, or analysis validation error."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        raise ProtocolError(f"{label} must be a non-empty whitespace-free identifier")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProtocolError(f"{label} must be a list")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{label} must be >= {minimum}")
    return value


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ProtocolError(f"{label} must be >= {minimum}")
    return result


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def _git_oid(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ProtocolError(f"{label} must be a lowercase Git object ID")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProtocolError(
            f"{label} fields mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _artifact_path(record: Mapping[str, Any], base: Path) -> Path:
    raw = Path(_string(record.get("path"), "artifact.path")).expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def _seal_artifact(record: object, base: Path, label: str) -> dict[str, Any]:
    value = _mapping(record, label)
    if set(value) not in ({"path"}, {"path", "sha256", "bytes"}):
        raise ProtocolError(f"{label} must contain path, optionally with sha256/bytes")
    path = _artifact_path(value, base)
    if not path.is_file():
        raise ProtocolError(f"{label} is not a file: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_artifact(record: object, base: Path, label: str) -> Path:
    value = _mapping(record, label)
    _exact_keys(value, {"path", "sha256", "bytes"}, label)
    path = _artifact_path(value, base)
    if not path.is_file():
        raise ProtocolError(f"{label} is not a file: {path}")
    expected_bytes = _integer(value["bytes"], f"{label}.bytes", minimum=0)
    expected_sha = _sha256(value["sha256"], f"{label}.sha256")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise ProtocolError(f"{label} no longer matches its seal")
    return path


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"git {' '.join(arguments)} failed in {root}") from exc
    return result.stdout.strip()


def _seal_repository(record: object, base: Path, index: int) -> dict[str, Any]:
    value = _mapping(record, f"repositories[{index}]")
    if set(value) not in (
        {"name", "root", "environment"},
        {"name", "root", "commit", "tree", "environment"},
    ):
        raise ProtocolError(f"repositories[{index}] has unexpected or missing fields")
    name = _identifier(value["name"], f"repositories[{index}].name")
    raw_root = Path(_string(value["root"], f"repositories[{index}].root")).expanduser()
    root = raw_root.resolve() if raw_root.is_absolute() else (base / raw_root).resolve()
    if not root.is_dir():
        raise ProtocolError(f"repository root does not exist: {root}")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ProtocolError(f"repository {name} is dirty; formal freezing requires clean source")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    return {
        "name": name,
        "root": str(root),
        "commit": _git_oid(commit, f"repository {name} commit"),
        "tree": _git_oid(tree, f"repository {name} tree"),
        "environment": _seal_artifact(value["environment"], base, f"{name}.environment"),
    }


def _validate_repository(record: object, base: Path, index: int) -> None:
    value = _mapping(record, f"repositories[{index}]")
    _exact_keys(
        value,
        {"name", "root", "commit", "tree", "environment"},
        f"repositories[{index}]",
    )
    name = _identifier(value["name"], f"repositories[{index}].name")
    root = Path(_string(value["root"], f"repositories[{index}].root"))
    if not root.is_absolute() or not root.is_dir():
        raise ProtocolError(f"repository {name} root must be an existing absolute directory")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ProtocolError(f"repository {name} became dirty")
    if _git(root, "rev-parse", "HEAD") != _git_oid(value["commit"], f"{name}.commit"):
        raise ProtocolError(f"repository {name} commit changed")
    if _git(root, "rev-parse", "HEAD^{tree}") != _git_oid(value["tree"], f"{name}.tree"):
        raise ProtocolError(f"repository {name} tree changed")
    _validate_artifact(value["environment"], base, f"{name}.environment")


def _protocol_without_seals(protocol: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(protocol))
    for name in ("state", "design_sha256", "protocol_sha256", "review"):
        value.pop(name, None)
    return value


def design_digest(protocol: Mapping[str, Any]) -> str:
    return _digest(_protocol_without_seals(protocol))


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(protocol))
    value.pop("protocol_sha256", None)
    return _digest(value)


def _iter_artifact_slots(protocol: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for index, repository in enumerate(protocol["repositories"]):
        yield f"repositories[{index}].environment", repository["environment"]
    for split_name in ("development", "confirmation"):
        for index, unit in enumerate(protocol["datasets"][split_name]):
            for name in ("pixels", "mask", "camera", "prepared_target"):
                yield f"datasets.{split_name}[{index}].{name}", unit[name]
    for index, entry in enumerate(protocol["initial_geometry"]):
        yield f"initial_geometry[{index}].bank", entry["bank"]
    target = protocol["structural_target"]
    if target["definition"] is not None:
        yield "structural_target.definition", target["definition"]
    for index, arm in enumerate(protocol["arms"]):
        for name in ("profile", "loss_contract", "gate_contract"):
            yield f"arms[{index}].{name}", arm[name]
    for phase in PHASES:
        yield f"phases.{phase}.work_contract", protocol["phases"][phase]["work_contract"]
    for name in ("environment", "downstream_protocol"):
        yield f"execution.{name}", protocol["execution"][name]


def _seal_protocol_artifacts(protocol: dict[str, Any], base: Path) -> None:
    protocol["repositories"] = [
        _seal_repository(record, base, index)
        for index, record in enumerate(protocol["repositories"])
    ]
    for split_name in ("development", "confirmation"):
        for index, unit in enumerate(protocol["datasets"][split_name]):
            for name in ("pixels", "mask", "camera", "prepared_target"):
                unit[name] = _seal_artifact(
                    unit[name], base, f"datasets.{split_name}[{index}].{name}"
                )
    for index, entry in enumerate(protocol["initial_geometry"]):
        entry["bank"] = _seal_artifact(entry["bank"], base, f"initial_geometry[{index}].bank")
    target = protocol["structural_target"]
    if target.get("definition") is not None:
        target["definition"] = _seal_artifact(
            target["definition"], base, "structural_target.definition"
        )
    for index, arm in enumerate(protocol["arms"]):
        for name in ("profile", "loss_contract", "gate_contract"):
            arm[name] = _seal_artifact(arm[name], base, f"arms[{index}].{name}")
    for phase in PHASES:
        protocol["phases"][phase]["work_contract"] = _seal_artifact(
            protocol["phases"][phase]["work_contract"],
            base,
            f"phases.{phase}.work_contract",
        )
    for name in ("environment", "downstream_protocol"):
        protocol["execution"][name] = _seal_artifact(
            protocol["execution"][name], base, f"execution.{name}"
        )


def _validate_unit(
    unit: Mapping[str, Any], split_name: str, index: int, base: Path
) -> tuple[str, str, str, set[str]]:
    label = f"datasets.{split_name}[{index}]"
    _exact_keys(
        unit,
        {
            "id",
            "capture_group",
            "frame_id",
            "metadata_selector",
            "pixels",
            "mask",
            "camera",
            "prepared_target",
            "views",
        },
        label,
    )
    unit_id = _identifier(unit["id"], f"{label}.id")
    capture = _identifier(unit["capture_group"], f"{label}.capture_group")
    frame = _identifier(unit["frame_id"], f"{label}.frame_id")
    selector = _mapping(unit["metadata_selector"], f"{label}.metadata_selector")
    if not selector:
        raise ProtocolError(f"{label}.metadata_selector must not be empty")
    canonical_json(selector)
    for name in ("pixels", "mask", "camera", "prepared_target"):
        _validate_artifact(unit[name], base, f"{label}.{name}")
    views = [
        _identifier(value, f"{label}.views") for value in _list(unit["views"], f"{label}.views")
    ]
    if not views or len(views) != len(set(views)):
        raise ProtocolError(f"{label}.views must be unique and non-empty")
    return unit_id, capture, frame, set(views)


def _validate_raw_ledger(
    ledger: Mapping[str, Any], unit_ids: set[str], label: str
) -> dict[str, Mapping[str, Any]]:
    _exact_keys(ledger, {"bytes_per_row", "fixed_bytes_by_unit", "components"}, label)
    bytes_per_row = _integer(ledger["bytes_per_row"], f"{label}.bytes_per_row", minimum=1)
    fixed = _mapping(ledger["fixed_bytes_by_unit"], f"{label}.fixed_bytes_by_unit")
    if set(fixed) != unit_ids:
        raise ProtocolError(f"{label}.fixed_bytes_by_unit must cover every dataset unit")
    fixed_values = {
        unit: _integer(value, f"{label}.fixed_bytes_by_unit.{unit}", minimum=0)
        for unit, value in fixed.items()
    }
    components = _list(ledger["components"], f"{label}.components")
    if not components:
        raise ProtocolError(f"{label}.components must not be empty")
    names: set[str] = set()
    row_sum = 0
    fixed_sum = {unit: 0 for unit in unit_ids}
    for index, raw_component in enumerate(components):
        component = _mapping(raw_component, f"{label}.components[{index}]")
        _exact_keys(
            component,
            {"name", "bytes_per_row", "fixed_bytes_by_unit"},
            f"{label}.components[{index}]",
        )
        name = _identifier(component["name"], f"{label}.components[{index}].name")
        if name in names:
            raise ProtocolError(f"{label} has duplicate component {name}")
        names.add(name)
        row_sum += _integer(component["bytes_per_row"], f"{label}.{name}.bytes_per_row", minimum=0)
        component_fixed = _mapping(
            component["fixed_bytes_by_unit"], f"{label}.{name}.fixed_bytes_by_unit"
        )
        if set(component_fixed) != unit_ids:
            raise ProtocolError(f"{label}.{name} fixed bytes do not cover all units")
        for unit, value in component_fixed.items():
            fixed_sum[unit] += _integer(value, f"{label}.{name}.{unit}", minimum=0)
    if row_sum != bytes_per_row or fixed_sum != fixed_values:
        raise ProtocolError(f"{label} component ledger does not sum to its declared totals")
    if not {"geometry", "appearance"}.issubset(names):
        raise ProtocolError(f"{label} must count geometry and appearance")
    return {component["name"]: component for component in components}


def _validate_policy_contracts(arm: Mapping[str, Any], label: str) -> None:
    policies = arm["alpha_policies"]
    contracts = _mapping(arm["policy_contracts"], f"{label}.policy_contracts")
    if set(contracts) != set(policies):
        raise ProtocolError(f"{label}.policy_contracts must match alpha_policies")
    for policy, raw_contract in contracts.items():
        contract = _mapping(raw_contract, f"{label}.policy_contracts.{policy}")
        _exact_keys(
            contract,
            {"target_space", "loss_scope", "gate_scope", "profile_scope"},
            f"{label}.policy_contracts.{policy}",
        )
        scopes = [
            _identifier(contract[name], f"{label}.{policy}.{name}")
            for name in ("loss_scope", "gate_scope", "profile_scope")
        ]
        if len(set(scopes)) != 1:
            raise ProtocolError(
                f"{label}.{policy} loss/gate/profile scopes must agree; mixed no-boundary "
                "contracts are forbidden"
            )
        target_space = _identifier(contract["target_space"], f"{label}.{policy}.target_space")
        if policy == "alpha_gated" and target_space != "alpha_matted_rgb":
            raise ProtocolError(f"{label} alpha_gated target_space must be alpha_matted_rgb")
        if policy == "hard_contained" and target_space != "foreground_rgb_zero_outside":
            raise ProtocolError(
                f"{label} hard_contained target_space must be foreground_rgb_zero_outside"
            )


def _validate_semantic_record(arm: Mapping[str, Any], family: str, label: str) -> None:
    semantics = _mapping(arm["semantics"], f"{label}.semantics")
    _exact_keys(
        semantics,
        {
            "contract",
            "schema_version",
            "renderer_equation",
            "coefficient_authority",
            "structural_mass",
            "payload_format",
        },
        f"{label}.semantics",
    )
    for name in semantics:
        _identifier(semantics[name], f"{label}.semantics.{name}")
    expected = {
        "incumbent_factorized_additive": {
            "contract": "legacy_gaussian_field",
            "schema_version": "1",
            "renderer_equation": "additive_rgb_peak_one_v1",
            "coefficient_authority": "factorized_color_times_opacity",
            "structural_mass": "absent_not_derived",
            "payload_format": "legacy_gaussian_field_npz_v1",
        },
        "direct_additive": {
            "contract": "observation_field_v2",
            "schema_version": "2.0.0",
            "renderer_equation": "additive_rgb_peak_one_v1",
            "coefficient_authority": "direct_rgb_coeff",
            "structural_mass": "absent_not_derived",
            "payload_format": "observation_field_v2_lossless_npz_v1",
        },
        "dual_additive": {
            "contract": "observation_field_v2",
            "schema_version": "2.0.0",
            "renderer_equation": "additive_rgb_peak_one_v1",
            "coefficient_authority": "direct_rgb_coeff",
            "structural_mass": "independently_supervised",
            "payload_format": "observation_field_v2_lossless_npz_v1",
        },
        "normalized_plain": {
            "contract": "legacy_gaussian_field",
            "schema_version": "1",
            "renderer_equation": "normalized_weighted_sum_v1",
            "coefficient_authority": "normalized_color",
            "structural_mass": "normalizer_not_mass",
            "payload_format": "legacy_gaussian_field_npz_v1",
        },
        "normalized_maintained": {
            "contract": "legacy_gaussian_field",
            "schema_version": "1",
            "renderer_equation": "normalized_weighted_sum_v1",
            "coefficient_authority": "normalized_color",
            "structural_mass": "normalizer_not_mass",
            "payload_format": "legacy_gaussian_field_npz_v1",
        },
    }
    if family in expected:
        for name, value in expected[family].items():
            if semantics[name] != value:
                raise ProtocolError(
                    f"{label} family {family} requires {name}={value!r}; semantic relabelling "
                    "is forbidden"
                )
    if arm["semantic_sha256"] != _digest(semantics):
        raise ProtocolError(f"{label}.semantic_sha256 does not bind its semantic record")


def _validate_arm(
    arm: Mapping[str, Any], index: int, unit_ids: set[str], base: Path
) -> tuple[str, str]:
    label = f"arms[{index}]"
    _exact_keys(
        arm,
        {
            "id",
            "family",
            "execution_kind",
            "semantics",
            "semantic_sha256",
            "coefficient_variants",
            "alpha_policies",
            "policy_contracts",
            "profile",
            "loss_contract",
            "gate_contract",
            "raw_byte_ledgers",
        },
        label,
    )
    arm_id = _identifier(arm["id"], f"{label}.id")
    family = _identifier(arm["family"], f"{label}.family")
    if family not in REQUIRED_FAMILIES | {"dual_additive", "native_authentic"}:
        raise ProtocolError(f"{label}.family is unknown")
    if arm["execution_kind"] not in EXECUTION_KINDS:
        raise ProtocolError(f"{label}.execution_kind is invalid")
    _sha256(arm["semantic_sha256"], f"{label}.semantic_sha256")
    _validate_semantic_record(arm, family, label)
    variants = [
        _identifier(value, f"{label}.coefficient_variants")
        for value in _list(arm["coefficient_variants"], f"{label}.coefficient_variants")
    ]
    if (
        not variants
        or len(variants) != len(set(variants))
        or not set(variants) <= COEFFICIENT_VARIANTS
    ):
        raise ProtocolError(f"{label}.coefficient_variants are invalid")
    if family in {"direct_additive", "dual_additive"}:
        if set(variants) == {"not_applicable"}:
            raise ProtocolError(f"{label} additive direct/dual arm needs coefficient variants")
    elif variants != ["not_applicable"]:
        raise ProtocolError(f"{label} non-direct arm must use only not_applicable")
    policies = [
        _identifier(value, f"{label}.alpha_policies")
        for value in _list(arm["alpha_policies"], f"{label}.alpha_policies")
    ]
    if not policies or len(policies) != len(set(policies)) or not set(policies) <= ALPHA_POLICIES:
        raise ProtocolError(f"{label}.alpha_policies are invalid")
    _validate_policy_contracts(arm, label)
    for name in ("profile", "loss_contract", "gate_contract"):
        _validate_artifact(arm[name], base, f"{label}.{name}")
    ledgers = _mapping(arm["raw_byte_ledgers"], f"{label}.raw_byte_ledgers")
    if set(ledgers) != set(variants):
        raise ProtocolError(f"{label}.raw_byte_ledgers must match coefficient_variants")
    for variant, raw_policies in ledgers.items():
        policy_ledgers = _mapping(raw_policies, f"{label}.raw_byte_ledgers.{variant}")
        if set(policy_ledgers) != set(policies):
            raise ProtocolError(f"{label}.raw_byte_ledgers.{variant} must match alpha_policies")
        for policy, raw_ledger in policy_ledgers.items():
            components = _validate_raw_ledger(
                _mapping(raw_ledger, f"{label}.raw_byte_ledgers.{variant}.{policy}"),
                unit_ids,
                f"{label}.raw_byte_ledgers.{variant}.{policy}",
            )
            required_components = {
                "geometry",
                "appearance",
                "structural_mass",
                "factorized_opacity",
                "background",
                "packed_alpha",
                "metadata",
            }
            if set(components) != required_components:
                raise ProtocolError(
                    f"{label} byte ledger must expose {sorted(required_components)} separately"
                )
            row_bytes = {name: int(record["bytes_per_row"]) for name, record in components.items()}
            if family == "dual_additive" and row_bytes["structural_mass"] <= 0:
                raise ProtocolError(f"{label} dual_additive must count structural_mass bytes")
            if family != "dual_additive" and row_bytes["structural_mass"] != 0:
                raise ProtocolError(f"{label} non-dual arm cannot count structural_mass rows")
            if family == "incumbent_factorized_additive" and row_bytes["factorized_opacity"] <= 0:
                raise ProtocolError(f"{label} incumbent additive must count factorized opacity")
            if (
                family in {"direct_additive", "dual_additive"}
                and row_bytes["factorized_opacity"] != 0
            ):
                raise ProtocolError(f"{label} direct coefficients cannot count factorized opacity")
            background_values = set(components["background"]["fixed_bytes_by_unit"].values())
            if variant == "counted_dc_signed_bounded" and (
                not background_values or min(background_values) <= 0
            ):
                raise ProtocolError(f"{label} counted-DC variant must count background bytes")
            if variant != "counted_dc_signed_bounded" and any(background_values):
                raise ProtocolError(f"{label} zero/native DC variant cannot count background bytes")
            alpha_values = set(components["packed_alpha"]["fixed_bytes_by_unit"].values())
            if policy == "alpha_gated" and (not alpha_values or min(alpha_values) <= 0):
                raise ProtocolError(f"{label} alpha_gated policy must count packed alpha bytes")
    return arm_id, family


def _metric_specs(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = _list(protocol["metrics"], "metrics")
    specs: dict[str, Mapping[str, Any]] = {}
    for index, raw_spec in enumerate(values):
        spec = _mapping(raw_spec, f"metrics[{index}]")
        _exact_keys(
            spec,
            {"name", "direction", "role", "phases", "availability"},
            f"metrics[{index}]",
        )
        name = _identifier(spec["name"], f"metrics[{index}].name")
        if name in specs:
            raise ProtocolError(f"duplicate metric {name}")
        if spec["direction"] not in METRIC_DIRECTIONS:
            raise ProtocolError(f"metric {name} has invalid direction")
        if spec["role"] not in {"quality", "boundary", "downstream", "guard"}:
            raise ProtocolError(f"metric {name} has invalid role")
        phases = _list(spec["phases"], f"metric {name}.phases")
        if not phases or not set(phases) <= set(PHASES) or len(phases) != len(set(phases)):
            raise ProtocolError(f"metric {name} has invalid phases")
        if spec["availability"] not in {"required", "unavailable"}:
            raise ProtocolError(f"metric {name} has invalid availability")
        specs[name] = spec
    mandatory = {
        "foreground_psnr",
        "boundary_psnr",
        "ms_ssim",
        "lpips",
        "alpha_mae",
        "outside_rgb_mae",
        "stage1_objective",
        "downstream_response",
    }
    if not mandatory.issubset(specs):
        raise ProtocolError(f"metrics are missing {sorted(mandatory - set(specs))}")
    if specs["downstream_response"]["availability"] != "required":
        raise ProtocolError("downstream_response must be required")
    return specs


def _validate_convergence(
    convergence: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]
) -> None:
    _exact_keys(
        convergence,
        {
            "metric",
            "targets",
            "primary_target",
            "target_rule",
            "unreached_policy",
            "auc_axis",
            "auc_horizon",
            "auc_interpolation",
        },
        "convergence",
    )
    metric = _identifier(convergence["metric"], "convergence.metric")
    if metric not in specs or specs[metric]["availability"] != "required":
        raise ProtocolError("convergence.metric must be a required metric")
    if metric != "foreground_psnr":
        raise ProtocolError("BENCH-020 convergence.metric must be foreground_psnr")
    if set(specs[metric]["phases"]) != set(PHASES):
        raise ProtocolError("convergence.metric must be required in every phase")
    targets = [
        _finite(value, f"convergence.targets[{index}]")
        for index, value in enumerate(_list(convergence["targets"], "convergence.targets"))
    ]
    if not targets or len(targets) != len(set(targets)) or targets != sorted(targets):
        raise ProtocolError("convergence.targets must be unique, non-empty, and increasing")
    primary = _finite(convergence["primary_target"], "convergence.primary_target")
    if primary not in targets:
        raise ProtocolError("convergence.primary_target must be one of convergence.targets")
    expected = {
        "target_rule": "first_observed_at_or_above",
        "unreached_policy": "null_with_right_censor_horizon",
        "auc_axis": "wall_seconds",
        "auc_horizon": "frozen_phase_wall_seconds_hold_last",
        "auc_interpolation": "linear_between_observations",
    }
    for name, value in expected.items():
        if convergence[name] != value:
            raise ProtocolError(f"convergence.{name} must be {value!r}")


def _validate_phase(phase: Mapping[str, Any], name: str, arm_ids: set[str], base: Path) -> None:
    common = {
        "work_contract",
        "iterations",
        "wall_seconds",
        "renderer_call_cap",
        "evaluation_every",
        "checkpoint_rule",
    }
    extra = {
        "coefficient_screen": {"screen_arm_id", "lane_id", "alpha_policy"},
        "development": set(),
        "confirmation": set(),
    }[name]
    _exact_keys(phase, common | extra, f"phases.{name}")
    _validate_artifact(phase["work_contract"], base, f"phases.{name}.work_contract")
    _integer(phase["iterations"], f"phases.{name}.iterations", minimum=1)
    _finite(phase["wall_seconds"], f"phases.{name}.wall_seconds", minimum=0.0)
    if phase["wall_seconds"] <= 0:
        raise ProtocolError(f"phases.{name}.wall_seconds must be > 0")
    _integer(phase["renderer_call_cap"], f"phases.{name}.renderer_call_cap", minimum=1)
    _integer(phase["evaluation_every"], f"phases.{name}.evaluation_every", minimum=1)
    _identifier(phase["checkpoint_rule"], f"phases.{name}.checkpoint_rule")
    if name == "coefficient_screen":
        if phase["screen_arm_id"] not in arm_ids:
            raise ProtocolError("coefficient screen_arm_id is unknown")
        _identifier(phase["lane_id"], "coefficient screen lane_id")
        if phase["alpha_policy"] not in ALPHA_POLICIES:
            raise ProtocolError("coefficient screen alpha_policy is invalid")


def _validate_gate_metric(
    record: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    _exact_keys(record, {"metric", "margin"}, label)
    metric = _identifier(record["metric"], f"{label}.metric")
    if metric not in specs or specs[metric]["availability"] != "required":
        raise ProtocolError(f"{label} references unavailable metric {metric}")
    _finite(record["margin"], f"{label}.margin", minimum=0.0)


def _validate_gates(
    gates: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    arm_ids: set[str],
    families: Mapping[str, str],
    lane_ids: set[str],
    capture_count: int,
    claim_scope: str,
) -> None:
    _exact_keys(gates, {"missing_policy", "domain", "killing"}, "gates")
    if gates["missing_policy"] != "fail_closed":
        raise ProtocolError("gates.missing_policy must be fail_closed")
    domain = _mapping(gates["domain"], "gates.domain")
    _exact_keys(
        domain,
        {
            "quality_metric",
            "max_mean_degradation",
            "feasibility_guards",
            "priority",
            "max_finalists",
        },
        "gates.domain",
    )
    quality_metric = _identifier(domain["quality_metric"], "gates.domain.quality_metric")
    if quality_metric not in specs or specs[quality_metric]["availability"] != "required":
        raise ProtocolError("domain quality metric is unavailable")
    _finite(domain["max_mean_degradation"], "max_mean_degradation", minimum=0.0)
    max_finalists = _integer(domain["max_finalists"], "max_finalists", minimum=1)
    if max_finalists > 2:
        raise ProtocolError("domain max_finalists must be <= 2")
    for index, raw_guard in enumerate(_list(domain["feasibility_guards"], "feasibility_guards")):
        guard = _mapping(raw_guard, f"feasibility_guards[{index}]")
        _exact_keys(guard, {"metric", "op", "threshold"}, f"feasibility_guards[{index}]")
        metric = _identifier(guard["metric"], "feasibility metric")
        if metric not in specs or specs[metric]["availability"] != "required":
            raise ProtocolError(f"feasibility guard metric {metric} is unavailable")
        if guard["op"] not in {"<=", ">="}:
            raise ProtocolError("feasibility guard op must be <= or >=")
        _finite(guard["threshold"], "feasibility threshold")
    priority = _list(domain["priority"], "gates.domain.priority")
    if not priority:
        raise ProtocolError("domain priority must not be empty")
    priority_names: set[str] = set()
    for index, raw_priority in enumerate(priority):
        item = _mapping(raw_priority, f"gates.domain.priority[{index}]")
        _exact_keys(item, {"metric", "direction"}, f"gates.domain.priority[{index}]")
        metric = _identifier(item["metric"], "domain priority metric")
        if metric in priority_names or metric not in specs:
            raise ProtocolError("domain priority metrics must be unique and declared")
        priority_names.add(metric)
        if item["direction"] != specs[metric]["direction"]:
            raise ProtocolError("domain priority direction disagrees with metric declaration")

    killing = _mapping(gates["killing"], "gates.killing")
    _exact_keys(
        killing,
        {
            "controls",
            "candidate_families",
            "lane_ids",
            "quality_noninferiority",
            "downstream_favorable",
            "absolute_guards",
            "pareto_metrics",
            "bootstrap_replicates",
            "bootstrap_seed",
            "minimum_capture_groups",
            "selection_rule",
        },
        "gates.killing",
    )
    controls = [_identifier(v, "killing control") for v in _list(killing["controls"], "controls")]
    if len(controls) < 2 or len(set(controls)) != len(controls) or not set(controls) <= arm_ids:
        raise ProtocolError("killing controls must contain at least two unique known arms")
    control_families = {families[arm] for arm in controls}
    if not {"incumbent_factorized_additive", "normalized_plain"}.issubset(control_families):
        raise ProtocolError("killing controls must include incumbent additive and normalized plain")
    candidates = {
        _identifier(v, "candidate family")
        for v in _list(killing["candidate_families"], "candidate_families")
    }
    if not candidates or not candidates <= {"direct_additive", "dual_additive"}:
        raise ProtocolError("candidate_families must be direct and/or dual additive")
    gate_lanes = {
        _identifier(v, "gate lane") for v in _list(killing["lane_ids"], "killing.lane_ids")
    }
    if not gate_lanes or not gate_lanes <= lane_ids:
        raise ProtocolError("killing lane_ids must be declared lanes")
    quality = _list(killing["quality_noninferiority"], "quality_noninferiority")
    if not quality:
        raise ProtocolError("quality_noninferiority must not be empty")
    for index, raw_metric in enumerate(quality):
        _validate_gate_metric(
            _mapping(raw_metric, f"quality_noninferiority[{index}]"),
            specs,
            f"quality_noninferiority[{index}]",
        )
    downstream = _mapping(killing["downstream_favorable"], "downstream_favorable")
    _validate_gate_metric(downstream, specs, "downstream_favorable")
    if specs[downstream["metric"]]["role"] != "downstream":
        raise ProtocolError("downstream_favorable must reference a downstream metric")
    for index, raw_guard in enumerate(_list(killing["absolute_guards"], "absolute_guards")):
        guard = _mapping(raw_guard, f"absolute_guards[{index}]")
        _exact_keys(guard, {"metric", "op", "threshold"}, f"absolute_guards[{index}]")
        metric = _identifier(guard["metric"], "absolute guard metric")
        if metric not in specs or specs[metric]["availability"] != "required":
            raise ProtocolError("absolute guard references unavailable metric")
        if guard["op"] not in {"<=", ">="}:
            raise ProtocolError("absolute guard op must be <= or >=")
        _finite(guard["threshold"], "absolute guard threshold")
    pareto = [_identifier(v, "pareto metric") for v in _list(killing["pareto_metrics"], "pareto")]
    if len(pareto) < 2 or len(pareto) != len(set(pareto)) or not set(pareto) <= set(specs):
        raise ProtocolError("pareto_metrics must contain at least two unique declared metrics")
    _integer(killing["bootstrap_replicates"], "bootstrap_replicates", minimum=1000)
    _integer(killing["bootstrap_seed"], "bootstrap_seed")
    minimum_capture_groups = _integer(
        killing["minimum_capture_groups"], "minimum_capture_groups", minimum=1
    )
    if claim_scope == "general" and minimum_capture_groups < 3:
        raise ProtocolError("general BENCH-020 claims require at least three capture groups")
    if capture_count < minimum_capture_groups:
        raise ProtocolError("development data do not meet the frozen minimum capture groups")
    if killing["selection_rule"] != "single_nondominated_survivor":
        raise ProtocolError("selection_rule must be single_nondominated_survivor")


def validate_protocol(
    protocol: Mapping[str, Any], *, base: str | Path = ".", require_frozen: bool = False
) -> None:
    base_path = Path(base).resolve()
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ProtocolError(f"protocol schema must be {PROTOCOL_SCHEMA}")
    state = protocol.get("state")
    if state not in {"review", "frozen"}:
        raise ProtocolError("validated protocol state must be review or frozen")
    if require_frozen and state != "frozen":
        raise ProtocolError("formal execution requires a frozen protocol")
    expected = {
        "schema",
        "task_id",
        "state",
        "driver",
        "claim_scope",
        "repositories",
        "datasets",
        "seeds",
        "seed_stream_sha256",
        "initial_geometry",
        "structural_target",
        "arms",
        "lanes",
        "phases",
        "metrics",
        "convergence",
        "gates",
        "aa_replay",
        "execution",
        "design_sha256",
    }
    if state == "frozen":
        expected |= {"review", "protocol_sha256"}
    _exact_keys(protocol, expected, "protocol")
    if protocol["task_id"] != "BENCH-020":
        raise ProtocolError("task_id must be BENCH-020")
    driver = _identifier(protocol["driver"], "driver")
    if protocol["claim_scope"] not in {"general", "workload_specific"}:
        raise ProtocolError("claim_scope must be general or workload_specific")

    repositories = _list(protocol["repositories"], "repositories")
    if not repositories:
        raise ProtocolError("repositories must not be empty")
    repository_names = []
    for index, repository in enumerate(repositories):
        _validate_repository(repository, base_path, index)
        repository_names.append(repository["name"])
    if "structsplat" not in repository_names or len(repository_names) != len(set(repository_names)):
        raise ProtocolError("repositories must uniquely include structsplat")

    datasets = _mapping(protocol["datasets"], "datasets")
    _exact_keys(datasets, {"development", "confirmation"}, "datasets")
    split_ids: dict[str, set[str]] = {}
    all_units: dict[str, Mapping[str, Any]] = {}
    all_frames: dict[str, str] = {}
    protected_source_digests: dict[str, set[str]] = {}
    development_captures: set[str] = set()
    confirmation_captures: set[str] = set()
    for split_name in ("development", "confirmation"):
        values = _list(datasets[split_name], f"datasets.{split_name}")
        if not values:
            raise ProtocolError(f"datasets.{split_name} must not be empty")
        split_ids[split_name] = set()
        protected_source_digests[split_name] = set()
        for index, raw_unit in enumerate(values):
            unit = _mapping(raw_unit, f"datasets.{split_name}[{index}]")
            unit_id, capture, frame, _views = _validate_unit(unit, split_name, index, base_path)
            if unit_id in all_units:
                raise ProtocolError("dataset unit IDs must be globally unique")
            if frame in all_frames:
                raise ProtocolError("frame IDs must be disjoint across all dataset units")
            all_units[unit_id] = unit
            all_frames[frame] = split_name
            split_ids[split_name].add(unit_id)
            protected_source_digests[split_name].update(
                unit[name]["sha256"] for name in ("pixels", "prepared_target")
            )
            if split_name == "development":
                development_captures.add(capture)
            else:
                confirmation_captures.add(capture)
    if protected_source_digests["development"] & protected_source_digests["confirmation"]:
        raise ProtocolError("development and confirmation source targets must be hash-disjoint")

    seeds = _list(protocol["seeds"], "seeds")
    if (
        len(seeds) < 3
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ProtocolError("seeds must contain at least three unique integers")
    seed_streams = _mapping(protocol["seed_stream_sha256"], "seed_stream_sha256")
    if set(seed_streams) != {str(seed) for seed in seeds}:
        raise ProtocolError("seed_stream_sha256 must cover every seed exactly")
    for seed, digest in seed_streams.items():
        _sha256(digest, f"seed_stream_sha256.{seed}")

    geometry = _list(protocol["initial_geometry"], "initial_geometry")
    geometry_keys: set[tuple[str, int]] = set()
    geometry_prefixes: dict[tuple[str, int], set[int]] = {}
    for index, raw_entry in enumerate(geometry):
        entry = _mapping(raw_entry, f"initial_geometry[{index}]")
        _exact_keys(entry, {"unit_id", "seed", "bank", "prefixes"}, f"initial_geometry[{index}]")
        unit_id = _identifier(entry["unit_id"], "geometry unit_id")
        seed = _integer(entry["seed"], "geometry seed")
        key = (unit_id, seed)
        if unit_id not in all_units or seed not in seeds or key in geometry_keys:
            raise ProtocolError(
                "initial_geometry keys must uniquely cover declared unit/seed pairs"
            )
        geometry_keys.add(key)
        _validate_artifact(entry["bank"], base_path, f"initial_geometry[{index}].bank")
        prefix_counts: set[int] = set()
        prefixes = _list(entry["prefixes"], f"initial_geometry[{index}].prefixes")
        if not prefixes:
            raise ProtocolError("each geometry bank needs at least one frozen prefix")
        for prefix_index, raw_prefix in enumerate(prefixes):
            prefix = _mapping(raw_prefix, f"initial_geometry[{index}].prefixes[{prefix_index}]")
            _exact_keys(prefix, {"row_count", "sha256"}, "geometry prefix")
            count = _integer(prefix["row_count"], "geometry prefix row_count", minimum=1)
            if count in prefix_counts:
                raise ProtocolError("geometry prefix row counts must be unique")
            prefix_counts.add(count)
            _sha256(prefix["sha256"], "geometry prefix sha256")
        geometry_prefixes[key] = prefix_counts
    expected_geometry = {(unit, seed) for unit in all_units for seed in seeds}
    if geometry_keys != expected_geometry:
        raise ProtocolError("initial_geometry must cover every dataset unit and seed")

    target = _mapping(protocol["structural_target"], "structural_target")
    _exact_keys(target, {"status", "metric", "direction", "definition"}, "structural_target")
    if target["status"] not in {"validated", "unavailable"}:
        raise ProtocolError("structural_target.status must be validated or unavailable")
    if target["status"] == "validated":
        _identifier(target["metric"], "structural_target.metric")
        if target["direction"] not in METRIC_DIRECTIONS or target["definition"] is None:
            raise ProtocolError("validated structural target needs direction and definition")
        _validate_artifact(target["definition"], base_path, "structural_target.definition")
    elif any(target[name] is not None for name in ("metric", "direction", "definition")):
        raise ProtocolError(
            "unavailable structural target must use null metric/direction/definition"
        )

    arms = _list(protocol["arms"], "arms")
    arm_ids: set[str] = set()
    families: dict[str, str] = {}
    family_set: set[str] = set()
    arm_index: dict[str, Mapping[str, Any]] = {}
    for index, raw_arm in enumerate(arms):
        arm = _mapping(raw_arm, f"arms[{index}]")
        arm_id, family = _validate_arm(arm, index, set(all_units), base_path)
        if arm_id in arm_ids:
            raise ProtocolError("arm IDs must be unique")
        arm_ids.add(arm_id)
        families[arm_id] = family
        family_set.add(family)
        arm_index[arm_id] = arm
    if not REQUIRED_FAMILIES.issubset(family_set):
        raise ProtocolError(
            f"arms are missing required families {sorted(REQUIRED_FAMILIES - family_set)}"
        )
    if "dual_additive" in family_set and target["status"] != "validated":
        raise ProtocolError("dual_additive is forbidden without a validated structural target")
    family_counts = {family: list(families.values()).count(family) for family in family_set}
    if any(family_counts[family] != 1 for family in REQUIRED_FAMILIES):
        raise ProtocolError("each required semantic family must have exactly one arm")
    for arm_id, family in families.items():
        execution_kind = arm_index[arm_id]["execution_kind"]
        if (
            family
            in {
                "incumbent_factorized_additive",
                "direct_additive",
                "dual_additive",
                "normalized_plain",
            }
            and execution_kind != "matched"
        ):
            raise ProtocolError(f"semantic arm {arm_id} must be execution_kind='matched'")
        if family == "normalized_maintained" and execution_kind != "maintained_reference":
            raise ProtocolError("normalized_maintained must be a maintained_reference")
        if family == "native_authentic" and execution_kind != "native_authentic":
            raise ProtocolError("native_authentic family must use native_authentic execution")
        if execution_kind == "matched" and set(arm_index[arm_id]["alpha_policies"]) != {
            "alpha_gated",
            "hard_contained",
        }:
            raise ProtocolError("matched masked semantic arms must expose both alpha policies")

    lanes = _list(protocol["lanes"], "lanes")
    lane_ids: set[str] = set()
    lane_kinds: set[str] = set()
    lane_index: dict[str, Mapping[str, Any]] = {}
    for index, raw_lane in enumerate(lanes):
        lane = _mapping(raw_lane, f"lanes[{index}]")
        _exact_keys(lane, {"id", "kind", "value"}, f"lanes[{index}]")
        lane_id = _identifier(lane["id"], f"lanes[{index}].id")
        if lane_id in lane_ids:
            raise ProtocolError("lane IDs must be unique")
        if lane["kind"] not in {"fixed_rows", "equal_canonical_raw_bytes"}:
            raise ProtocolError("lane kind is invalid")
        _integer(lane["value"], f"lanes[{index}].value", minimum=1)
        lane_ids.add(lane_id)
        lane_kinds.add(lane["kind"])
        lane_index[lane_id] = lane
    if len(lanes) != 2 or lane_kinds != {"fixed_rows", "equal_canonical_raw_bytes"}:
        raise ProtocolError("protocol requires exactly one fixed-row and one equal-raw-byte lane")

    phases = _mapping(protocol["phases"], "phases")
    _exact_keys(phases, set(PHASES), "phases")
    for phase_name in PHASES:
        _validate_phase(
            _mapping(phases[phase_name], f"phases.{phase_name}"), phase_name, arm_ids, base_path
        )
    screen_arm = arm_index[phases["coefficient_screen"]["screen_arm_id"]]
    if screen_arm["family"] != "direct_additive":
        raise ProtocolError("coefficient screen must use a direct_additive arm")
    if phases["coefficient_screen"]["lane_id"] not in lane_ids:
        raise ProtocolError("coefficient screen lane is unknown")
    if phases["coefficient_screen"]["alpha_policy"] not in screen_arm["alpha_policies"]:
        raise ProtocolError("coefficient screen alpha policy is unsupported by its arm")

    # Every planned row count must have a frozen prefix in the common ordered geometry bank.
    for unit_id in all_units:
        for arm in arms:
            if arm["execution_kind"] != "matched":
                continue
            for lane in lanes:
                for variant in arm["coefficient_variants"]:
                    for alpha_policy in arm["alpha_policies"]:
                        count, _raw_bytes = _row_budget(arm, lane, unit_id, variant, alpha_policy)
                        for seed in seeds:
                            if count not in geometry_prefixes[(unit_id, seed)]:
                                raise ProtocolError(
                                    f"geometry bank {unit_id}/{seed} lacks planned prefix {count}"
                                )

    specs = _metric_specs(protocol)
    _validate_convergence(_mapping(protocol["convergence"], "convergence"), specs)
    if target["status"] == "validated" and specs["stage1_objective"]["availability"] != "required":
        raise ProtocolError("validated structural target requires the Stage-1 objective metric")
    _validate_gates(
        _mapping(protocol["gates"], "gates"),
        specs,
        arm_ids,
        families,
        lane_ids,
        len(development_captures),
        protocol["claim_scope"],
    )
    minimum_capture_groups = int(protocol["gates"]["killing"]["minimum_capture_groups"])
    if len(confirmation_captures) < minimum_capture_groups:
        raise ProtocolError("confirmation data do not meet the frozen minimum capture groups")
    candidate_policies = {
        policy
        for arm in arms
        if arm["family"] in protocol["gates"]["killing"]["candidate_families"]
        for policy in arm["alpha_policies"]
    }
    for control_id in protocol["gates"]["killing"]["controls"]:
        if not candidate_policies <= set(arm_index[control_id]["alpha_policies"]):
            raise ProtocolError("every killing control must support every candidate alpha policy")
        if arm_index[control_id]["execution_kind"] != "matched":
            raise ProtocolError("killing controls must be matched arms")
    aa = _mapping(protocol["aa_replay"], "aa_replay")
    _exact_keys(
        aa,
        {
            "unit_id",
            "arm_id",
            "coefficient_variant",
            "alpha_policy",
            "lane_id",
            "seed",
            "metric_abs_tolerance",
        },
        "aa_replay",
    )
    if aa["unit_id"] not in split_ids["development"] or aa["arm_id"] not in arm_ids:
        raise ProtocolError("A/A replay unit or arm is unknown")
    if aa["coefficient_variant"] not in arm_index[aa["arm_id"]]["coefficient_variants"]:
        raise ProtocolError("A/A coefficient variant is unsupported")
    if aa["alpha_policy"] not in arm_index[aa["arm_id"]]["alpha_policies"]:
        raise ProtocolError("A/A alpha policy is unsupported")
    if aa["lane_id"] not in lane_ids or aa["seed"] not in seeds:
        raise ProtocolError("A/A lane or seed is unknown")
    coefficient_phase = phases["coefficient_screen"]
    if (
        aa["arm_id"] != coefficient_phase["screen_arm_id"]
        or aa["lane_id"] != coefficient_phase["lane_id"]
        or aa["alpha_policy"] != coefficient_phase["alpha_policy"]
        or aa["coefficient_variant"] == "not_applicable"
    ):
        raise ProtocolError("A/A replay must name an actual coefficient-screen cell")
    tolerances = _mapping(aa["metric_abs_tolerance"], "aa_replay.metric_abs_tolerance")
    if not tolerances:
        raise ProtocolError("A/A replay must freeze metric tolerances")
    for metric, tolerance in tolerances.items():
        if metric not in specs or specs[metric]["availability"] != "required":
            raise ProtocolError("A/A tolerance references unavailable metric")
        _finite(tolerance, f"A/A tolerance {metric}", minimum=0.0)

    execution = _mapping(protocol["execution"], "execution")
    _exact_keys(
        execution,
        {"environment", "downstream_protocol", "working_directory", "commands", "outcome_roots"},
        "execution",
    )
    for name in ("environment", "downstream_protocol"):
        _validate_artifact(execution[name], base_path, f"execution.{name}")
    working = Path(_string(execution["working_directory"], "working_directory"))
    if not working.is_absolute() or not working.is_dir():
        raise ProtocolError("execution working_directory must be an existing absolute path")
    commands = _mapping(execution["commands"], "execution.commands")
    _exact_keys(commands, set(PHASES), "execution.commands")
    for phase_name, raw_command in commands.items():
        command = _list(raw_command, f"execution.commands.{phase_name}")
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ProtocolError(f"execution command {phase_name} must be a non-empty argv list")
    roots = _mapping(execution["outcome_roots"], "execution.outcome_roots")
    _exact_keys(roots, set(PHASES), "execution.outcome_roots")
    root_paths: list[Path] = []
    for phase_name, raw_root in roots.items():
        root = Path(_string(raw_root, f"outcome root {phase_name}"))
        if not root.is_absolute():
            raise ProtocolError("all outcome roots must be absolute")
        root_paths.append(root)
    if len(set(root_paths)) != len(root_paths):
        raise ProtocolError("phase outcome roots must be distinct")

    if protocol["design_sha256"] != design_digest(protocol):
        raise ProtocolError("protocol design SHA-256 is missing or mismatched")
    if state == "frozen":
        review = _mapping(protocol["review"], "review")
        _exact_keys(
            review,
            {"driver", "reviewer", "verdict", "design_sha256", "artifact"},
            "review",
        )
        reviewer = _identifier(review["reviewer"], "review.reviewer")
        if (
            review["driver"] != driver
            or reviewer.casefold() == driver.casefold()
            or review["verdict"] != "approved"
            or review["design_sha256"] != protocol["design_sha256"]
        ):
            raise ProtocolError("frozen protocol lacks a matching distinct approval")
        _validate_artifact(review["artifact"], base_path, "review.artifact")
        if protocol["protocol_sha256"] != protocol_digest(protocol):
            raise ProtocolError("protocol SHA-256 is missing or mismatched")


def prepare_review(draft: Mapping[str, Any], *, base: str | Path = ".") -> dict[str, Any]:
    protocol = copy.deepcopy(dict(draft))
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("state") != "draft":
        raise ProtocolError("prepare-review requires a BENCH-020 draft")
    for name in ("design_sha256", "protocol_sha256", "review"):
        protocol.pop(name, None)
    base_path = Path(base).resolve()
    _seal_protocol_artifacts(protocol, base_path)
    working = Path(
        _string(protocol["execution"]["working_directory"], "working_directory")
    ).expanduser()
    working = working.resolve() if working.is_absolute() else (base_path / working).resolve()
    if not working.is_dir():
        raise ProtocolError("execution working_directory does not exist")
    protocol["execution"]["working_directory"] = str(working)
    roots = protocol["execution"]["outcome_roots"]
    resolved_roots: dict[str, str] = {}
    for phase_name in PHASES:
        raw = Path(_string(roots[phase_name], f"outcome root {phase_name}")).expanduser()
        root = raw.resolve() if raw.is_absolute() else (base_path / raw).resolve()
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ProtocolError(f"{phase_name} outcome root is already non-empty")
        resolved_roots[phase_name] = str(root)
    if len(set(resolved_roots.values())) != len(PHASES):
        raise ProtocolError("phase outcome roots must be distinct")
    protocol["execution"]["outcome_roots"] = resolved_roots
    protocol["state"] = "review"
    protocol["design_sha256"] = design_digest(protocol)
    validate_protocol(protocol, base=base_path)
    return protocol


def review_template(protocol: Mapping[str, Any], *, base: str | Path = ".") -> dict[str, Any]:
    validate_protocol(protocol, base=base)
    if protocol["state"] != "review":
        raise ProtocolError("review-template requires a review-ready protocol")
    return {
        "schema": REVIEW_SCHEMA,
        "driver": protocol["driver"],
        "reviewer": "replace-with-distinct-reviewer",
        "verdict": "approved-or-rejected",
        "design_sha256": protocol["design_sha256"],
        "outcomes_accessed": False,
        "notes": (
            "Review source/prepared disjointness, geometry prefixes, arm semantics, raw-byte "
            "ledgers, work/loss/gate consistency, metrics, phase isolation, and killing rules."
        ),
    }


def finalize_protocol(
    reviewed: Mapping[str, Any], review_path: str | Path, *, base: str | Path = "."
) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(reviewed))
    base_path = Path(base).resolve()
    validate_protocol(protocol, base=base_path)
    if protocol["state"] != "review":
        raise ProtocolError("finalize requires a review-ready protocol")
    review_file = Path(review_path).expanduser().resolve()
    try:
        record = json.loads(review_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("cannot read protocol review") from exc
    review = _mapping(record, "protocol review")
    _exact_keys(
        review,
        {
            "schema",
            "driver",
            "reviewer",
            "verdict",
            "design_sha256",
            "outcomes_accessed",
            "notes",
        },
        "protocol review",
    )
    driver = _identifier(review["driver"], "review.driver")
    reviewer = _identifier(review["reviewer"], "review.reviewer")
    if (
        review["schema"] != REVIEW_SCHEMA
        or driver != protocol["driver"]
        or reviewer.casefold() == driver.casefold()
        or review["verdict"] != "approved"
        or review["outcomes_accessed"] is not False
        or review["design_sha256"] != protocol["design_sha256"]
    ):
        raise ProtocolError("protocol review is not a matching outcome-unseen distinct approval")
    for phase_name, raw_root in protocol["execution"]["outcome_roots"].items():
        root = Path(raw_root)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ProtocolError(f"{phase_name} outcome root became non-empty before finalization")
    protocol["review"] = {
        "driver": driver,
        "reviewer": reviewer,
        "verdict": "approved",
        "design_sha256": protocol["design_sha256"],
        "artifact": _seal_artifact({"path": str(review_file)}, base_path, "review artifact"),
    }
    protocol["state"] = "frozen"
    protocol["protocol_sha256"] = protocol_digest(protocol)
    validate_protocol(protocol, base=base_path, require_frozen=True)
    return protocol


def _unit_index(protocol: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    result: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for split_name in ("development", "confirmation"):
        for unit in protocol["datasets"][split_name]:
            result[unit["id"]] = (split_name, unit)
    return result


def _arm_index(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {arm["id"]: arm for arm in protocol["arms"]}


def _lane_index(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {lane["id"]: lane for lane in protocol["lanes"]}


def _geometry_index(protocol: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {(entry["unit_id"], entry["seed"]): entry for entry in protocol["initial_geometry"]}


def _row_budget(
    arm: Mapping[str, Any],
    lane: Mapping[str, Any],
    unit_id: str,
    coefficient_variant: str,
    alpha_policy: str,
) -> tuple[int, int]:
    try:
        ledger = arm["raw_byte_ledgers"][coefficient_variant][alpha_policy]
    except KeyError as exc:
        raise ProtocolError(
            f"arm {arm['id']} has no raw-byte ledger for {coefficient_variant}/{alpha_policy}"
        ) from exc
    per_row = int(ledger["bytes_per_row"])
    fixed = int(ledger["fixed_bytes_by_unit"][unit_id])
    if lane["kind"] == "fixed_rows":
        count = int(lane["value"])
        return count, fixed + count * per_row
    target = int(lane["value"])
    count = (target - fixed) // per_row
    if count < 1:
        raise ProtocolError(
            f"equal-byte lane {lane['id']} cannot fund one {arm['id']} row for {unit_id}"
        )
    raw_bytes = fixed + count * per_row
    if raw_bytes > target or target - raw_bytes >= per_row:
        raise ProtocolError("equal-byte lane floor calculation is inconsistent")
    return count, raw_bytes


def _geometry_prefix(entry: Mapping[str, Any], row_count: int) -> str:
    for prefix in entry["prefixes"]:
        if prefix["row_count"] == row_count:
            return prefix["sha256"]
    raise ProtocolError(
        f"geometry bank {entry['unit_id']}/{entry['seed']} lacks prefix {row_count}"
    )


def _work_record(protocol: Mapping[str, Any], phase: str) -> dict[str, Any]:
    record = protocol["phases"][phase]
    return {
        "work_contract_sha256": record["work_contract"]["sha256"],
        "iterations": record["iterations"],
        "wall_seconds": record["wall_seconds"],
        "renderer_call_cap": record["renderer_call_cap"],
        "evaluation_every": record["evaluation_every"],
        "checkpoint_rule": record["checkpoint_rule"],
    }


def _cell(
    protocol: Mapping[str, Any],
    *,
    phase: str,
    unit: Mapping[str, Any],
    arm: Mapping[str, Any],
    coefficient_variant: str,
    alpha_policy: str,
    lane: Mapping[str, Any],
    seed: int,
    replicate: str,
) -> dict[str, Any]:
    row_count, raw_bytes = _row_budget(arm, lane, unit["id"], coefficient_variant, alpha_policy)
    geometry = _geometry_index(protocol)[(unit["id"], seed)]
    geometry_prefix = (
        _geometry_prefix(geometry, row_count) if arm["execution_kind"] == "matched" else None
    )
    work = _work_record(protocol, phase)
    identity = {
        "protocol_sha256": protocol["protocol_sha256"],
        "phase": phase,
        "unit_id": unit["id"],
        "capture_group": unit["capture_group"],
        "frame_id": unit["frame_id"],
        "arm_id": arm["id"],
        "family": arm["family"],
        "execution_kind": arm["execution_kind"],
        "coefficient_variant": coefficient_variant,
        "alpha_policy": alpha_policy,
        "lane_id": lane["id"],
        "seed": seed,
        "row_count": row_count,
        "canonical_raw_bytes": raw_bytes,
        "bindings": {
            "pixels_sha256": unit["pixels"]["sha256"],
            "mask_sha256": unit["mask"]["sha256"],
            "camera_sha256": unit["camera"]["sha256"],
            "prepared_target_sha256": unit["prepared_target"]["sha256"],
            "metadata_selector_sha256": _digest(unit["metadata_selector"]),
            "geometry_bank_sha256": (
                geometry["bank"]["sha256"] if arm["execution_kind"] == "matched" else None
            ),
            "initial_geometry_sha256": geometry_prefix,
            "seed_stream_sha256": protocol["seed_stream_sha256"][str(seed)],
            "semantic_sha256": arm["semantic_sha256"],
            "profile_sha256": arm["profile"]["sha256"],
            "loss_contract_sha256": arm["loss_contract"]["sha256"],
            "gate_contract_sha256": arm["gate_contract"]["sha256"],
            "requested_work_sha256": _digest(work),
            "environment_sha256": protocol["execution"]["environment"]["sha256"],
            "downstream_protocol_sha256": protocol["execution"]["downstream_protocol"]["sha256"],
        },
        "requested_work": work,
    }
    result = {**identity, "identity_sha256": _digest(identity), "replicate": replicate}
    result["cell_id"] = _digest(result)
    return result


def _validate_lock_digest(lock: Mapping[str, Any], schema: str, label: str) -> None:
    if lock.get("schema") != schema:
        raise ProtocolError(f"{label} schema must be {schema}")
    digest_name = {
        DOMAIN_LOCK_SCHEMA: "domain_lock_sha256",
        CONFIRMATION_LOCK_SCHEMA: "confirmation_lock_sha256",
    }[schema]
    recorded = lock.get(digest_name)
    value = copy.deepcopy(dict(lock))
    value.pop(digest_name, None)
    if recorded != _digest(value):
        raise ProtocolError(f"{label} digest is missing or mismatched")


def _domain_finalists(
    protocol: Mapping[str, Any], domain_lock: Mapping[str, Any] | None
) -> list[str]:
    screen_arm = _arm_index(protocol)[protocol["phases"]["coefficient_screen"]["screen_arm_id"]]
    if domain_lock is None:
        raise ProtocolError("development planning requires a domain lock")
    _validate_lock_digest(domain_lock, DOMAIN_LOCK_SCHEMA, "domain lock")
    if domain_lock.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ProtocolError("domain lock binds a different protocol")
    finalists = domain_lock.get("finalists")
    if not isinstance(finalists, list) or not finalists:
        raise ProtocolError("domain lock has no finalists")
    if (
        len(finalists) != len(set(finalists))
        or not set(finalists) <= set(screen_arm["coefficient_variants"])
        or len(finalists) > protocol["gates"]["domain"]["max_finalists"]
    ):
        raise ProtocolError("domain lock finalists violate the frozen candidate set")
    return finalists


def _selection_from_confirmation_lock(
    protocol: Mapping[str, Any], confirmation_lock: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    if confirmation_lock is None:
        raise ProtocolError("confirmation planning requires a confirmation lock")
    _validate_lock_digest(confirmation_lock, CONFIRMATION_LOCK_SCHEMA, "confirmation lock")
    if confirmation_lock.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ProtocolError("confirmation lock binds a different protocol")
    selected = _mapping(confirmation_lock.get("selected"), "confirmation selected")
    _exact_keys(
        selected,
        {"arm_id", "coefficient_variant", "alpha_policy"},
        "confirmation selected",
    )
    return selected


def expected_cells(
    protocol: Mapping[str, Any],
    phase: str,
    *,
    domain_lock: Mapping[str, Any] | None = None,
    confirmation_lock: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validate_protocol(protocol, require_frozen=True)
    if phase not in PHASES:
        raise ProtocolError(f"unknown phase {phase!r}")
    arms = _arm_index(protocol)
    lanes = _lane_index(protocol)
    units = _unit_index(protocol)
    specifications: list[tuple[Mapping[str, Any], str, str, Mapping[str, Any]]] = []
    split_name = "development" if phase != "confirmation" else "confirmation"
    phase_units = [unit for split, unit in units.values() if split == split_name]

    if phase == "coefficient_screen":
        phase_record = protocol["phases"][phase]
        arm = arms[phase_record["screen_arm_id"]]
        lane = lanes[phase_record["lane_id"]]
        for variant in arm["coefficient_variants"]:
            if variant != "not_applicable":
                specifications.append((arm, variant, phase_record["alpha_policy"], lane))
    elif phase == "development":
        finalists = _domain_finalists(protocol, domain_lock)
        for arm in protocol["arms"]:
            variants = (
                finalists
                if arm["family"] in {"direct_additive", "dual_additive"}
                else ["not_applicable"]
            )
            for variant in variants:
                if variant not in arm["coefficient_variants"]:
                    continue
                for alpha_policy in arm["alpha_policies"]:
                    for lane in protocol["lanes"]:
                        specifications.append((arm, variant, alpha_policy, lane))
    else:
        selected = _selection_from_confirmation_lock(protocol, confirmation_lock)
        selected_arm = arms[selected["arm_id"]]
        controls = [arms[arm_id] for arm_id in protocol["gates"]["killing"]["controls"]]
        chosen = [
            (
                selected_arm,
                selected["coefficient_variant"],
                selected["alpha_policy"],
            )
        ]
        for control in controls:
            policy = (
                selected["alpha_policy"]
                if selected["alpha_policy"] in control["alpha_policies"]
                else control["alpha_policies"][0]
            )
            chosen.append((control, "not_applicable", policy))
        seen: set[tuple[str, str, str]] = set()
        for arm, variant, alpha_policy in chosen:
            key = (arm["id"], variant, alpha_policy)
            if key in seen:
                continue
            seen.add(key)
            for lane in protocol["lanes"]:
                specifications.append((arm, variant, alpha_policy, lane))

    aa = protocol["aa_replay"]
    cells: list[dict[str, Any]] = []
    for unit in phase_units:
        for arm, variant, alpha_policy, lane in specifications:
            for seed in protocol["seeds"]:
                replicates = ["primary"]
                if (
                    phase == "coefficient_screen"
                    and unit["id"] == aa["unit_id"]
                    and arm["id"] == aa["arm_id"]
                    and variant == aa["coefficient_variant"]
                    and alpha_policy == aa["alpha_policy"]
                    and lane["id"] == aa["lane_id"]
                    and seed == aa["seed"]
                ):
                    replicates.append("aa")
                for replicate in replicates:
                    cells.append(
                        _cell(
                            protocol,
                            phase=phase,
                            unit=unit,
                            arm=arm,
                            coefficient_variant=variant,
                            alpha_policy=alpha_policy,
                            lane=lane,
                            seed=seed,
                            replicate=replicate,
                        )
                    )
    return sorted(cells, key=lambda cell: cell["cell_id"])


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        if source.suffix == ".jsonl":
            values = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            values = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read rows from {source}") from exc
    if not isinstance(values, list) or not all(isinstance(row, dict) for row in values):
        raise ProtocolError("rows must be a JSON list/JSONL stream of objects")
    return values


def _validate_result_artifact(record: object, base: Path, label: str) -> dict[str, Any]:
    path = _validate_artifact(record, base, label)
    value = dict(_mapping(record, label))
    value["path"] = str(path)
    return value


def _load_json_artifact(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"{label} is not valid JSON") from exc
    return _mapping(value, label)


def _phase_outcome_root(protocol: Mapping[str, Any], phase: str) -> Path:
    return Path(protocol["execution"]["outcome_roots"][phase]).expanduser().resolve()


def _require_later_outcomes_empty(protocol: Mapping[str, Any], phase: str) -> None:
    try:
        phase_index = PHASES.index(phase)
    except ValueError as exc:
        raise ProtocolError(f"unknown phase {phase!r}") from exc
    for later_phase in PHASES[phase_index + 1 :]:
        root = _phase_outcome_root(protocol, later_phase)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ProtocolError(f"{later_phase} outcome root is non-empty before {phase} analysis")


def _require_artifact_in_phase_root(
    protocol: Mapping[str, Any], phase: str, path: Path, label: str
) -> None:
    root = _phase_outcome_root(protocol, phase)
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProtocolError(
            f"{label} must be contained in the frozen {phase} outcome root"
        ) from exc


def summarize_convergence(
    protocol: Mapping[str, Any], phase: str, points: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Replay the frozen first-hit and normalized wall-time AUC conventions."""
    if phase not in PHASES:
        raise ProtocolError(f"unknown phase {phase!r}")
    if not points:
        raise ProtocolError("convergence history must contain at least one observation")
    normalized: list[tuple[int, float, float]] = []
    for index, raw_point in enumerate(points):
        point = _mapping(raw_point, f"history.points[{index}]")
        _exact_keys(point, {"iteration", "wall_seconds", "value"}, f"history.points[{index}]")
        iteration = _integer(point["iteration"], f"history.points[{index}].iteration", minimum=0)
        wall_seconds = _finite(
            point["wall_seconds"], f"history.points[{index}].wall_seconds", minimum=0.0
        )
        value = _finite(point["value"], f"history.points[{index}].value")
        if normalized and (iteration <= normalized[-1][0] or wall_seconds < normalized[-1][1]):
            raise ProtocolError(
                "convergence history iterations must increase and wall time must not decrease"
            )
        normalized.append((iteration, wall_seconds, value))
    if normalized[0][0] != 0 or normalized[0][1] != 0.0:
        raise ProtocolError("convergence history must begin at iteration 0 and wall_seconds 0")
    horizon = float(protocol["phases"][phase]["wall_seconds"])
    if normalized[-1][1] > horizon:
        raise ProtocolError("convergence history exceeds the frozen wall-time horizon")
    area = 0.0
    for left, right in zip(normalized, normalized[1:]):
        area += (right[1] - left[1]) * (left[2] + right[2]) * 0.5
    area += (horizon - normalized[-1][1]) * normalized[-1][2]
    primary_target = float(protocol["convergence"]["primary_target"])
    first_hit = next((point for point in normalized if point[2] >= primary_target), None)
    return {
        "target_reached": first_hit is not None,
        "iterations_to_target": None if first_hit is None else first_hit[0],
        "wall_seconds_to_target": None if first_hit is None else first_hit[1],
        "psnr_time_auc": area / horizon,
    }


def _required_metrics(protocol: Mapping[str, Any], phase: str) -> set[str]:
    return {
        name
        for name, spec in _metric_specs(protocol).items()
        if spec["availability"] == "required" and phase in spec["phases"]
    }


def validate_result_rows(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    *,
    base: str | Path = ".",
    domain_lock: Mapping[str, Any] | None = None,
    confirmation_lock: Mapping[str, Any] | None = None,
    require_complete: bool = True,
    validate_artifacts: bool = True,
) -> dict[str, Any]:
    plans = expected_cells(
        protocol,
        phase,
        domain_lock=domain_lock,
        confirmation_lock=confirmation_lock,
    )
    expected = {cell["cell_id"]: cell for cell in plans}
    observed: dict[str, Mapping[str, Any]] = {}
    base_path = Path(base).resolve()
    required_metrics = _required_metrics(protocol, phase)
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"rows[{index}]")
        _exact_keys(
            row,
            {"schema", "cell", "status", "error", "telemetry", "metrics", "artifacts"},
            f"rows[{index}]",
        )
        if row["schema"] != ROW_SCHEMA:
            raise ProtocolError(f"rows[{index}] has the wrong schema")
        cell = _mapping(row["cell"], f"rows[{index}].cell")
        cell_id = _string(cell.get("cell_id"), f"rows[{index}].cell.cell_id")
        if cell_id not in expected or dict(cell) != expected[cell_id]:
            raise ProtocolError(f"rows[{index}] cell does not match the frozen plan")
        if cell_id in observed:
            raise ProtocolError(f"duplicate result row for cell {cell_id}")
        observed[cell_id] = row
        status = row["status"]
        if status not in {"ok", "error"}:
            raise ProtocolError(f"rows[{index}].status must be ok or error")
        metrics = _mapping(row["metrics"], f"rows[{index}].metrics")
        telemetry = _mapping(row["telemetry"], f"rows[{index}].telemetry")
        artifacts = _mapping(row["artifacts"], f"rows[{index}].artifacts")
        if status == "error":
            if not isinstance(row["error"], str) or not row["error"].strip():
                raise ProtocolError("error rows require a non-empty error string")
            if metrics or telemetry or artifacts:
                raise ProtocolError("error rows retain the cell and error only")
            continue
        if row["error"] is not None:
            raise ProtocolError("ok rows require error=null")
        if set(metrics) != required_metrics:
            raise ProtocolError(
                f"ok row metrics mismatch: expected={sorted(required_metrics)}, "
                f"actual={sorted(metrics)}"
            )
        for metric, value in metrics.items():
            _finite(value, f"rows[{index}].metrics.{metric}")
        telemetry_keys = {
            "row_count",
            "canonical_raw_bytes",
            "iterations_requested",
            "iterations_executed",
            "renderer_calls_requested",
            "renderer_calls_executed",
            "wall_seconds",
            "peak_memory_mb",
            "checkpoint_id",
            "authoritative_preclamp_sha256",
            "evaluation_clip_policy",
            "target_reached",
            "iterations_to_target",
            "wall_seconds_to_target",
            "psnr_time_auc",
        }
        _exact_keys(telemetry, telemetry_keys, f"rows[{index}].telemetry")
        if telemetry["row_count"] != cell["row_count"]:
            raise ProtocolError("telemetry row_count does not match the frozen cell")
        if telemetry["canonical_raw_bytes"] != cell["canonical_raw_bytes"]:
            raise ProtocolError("telemetry canonical_raw_bytes does not match the frozen ledger")
        if telemetry["iterations_requested"] != cell["requested_work"]["iterations"]:
            raise ProtocolError("telemetry iterations_requested drifted")
        if telemetry["renderer_calls_requested"] != cell["requested_work"]["renderer_call_cap"]:
            raise ProtocolError("telemetry renderer_calls_requested drifted")
        _integer(telemetry["iterations_executed"], "iterations_executed", minimum=0)
        _integer(telemetry["renderer_calls_executed"], "renderer_calls_executed", minimum=0)
        if telemetry["iterations_executed"] > telemetry["iterations_requested"]:
            raise ProtocolError("iterations_executed exceeds the requested horizon")
        if telemetry["renderer_calls_executed"] > telemetry["renderer_calls_requested"]:
            raise ProtocolError("renderer_calls_executed exceeds the frozen cap")
        _finite(telemetry["wall_seconds"], "wall_seconds", minimum=0.0)
        if telemetry["wall_seconds"] > cell["requested_work"]["wall_seconds"]:
            raise ProtocolError("wall_seconds exceeds the frozen phase horizon")
        _finite(telemetry["peak_memory_mb"], "peak_memory_mb", minimum=0.0)
        _identifier(telemetry["checkpoint_id"], "checkpoint_id")
        _sha256(telemetry["authoritative_preclamp_sha256"], "authoritative_preclamp_sha256")
        if telemetry["evaluation_clip_policy"] != "clip_0_1_for_metrics_only":
            raise ProtocolError("evaluation clipping policy drifted")
        if not isinstance(telemetry["target_reached"], bool):
            raise ProtocolError("target_reached must be boolean")
        if telemetry["target_reached"]:
            _integer(telemetry["iterations_to_target"], "iterations_to_target", minimum=0)
            _finite(
                telemetry["wall_seconds_to_target"],
                "wall_seconds_to_target",
                minimum=0.0,
            )
            if telemetry["iterations_to_target"] > telemetry["iterations_executed"]:
                raise ProtocolError("iterations_to_target exceeds iterations_executed")
            if telemetry["wall_seconds_to_target"] > telemetry["wall_seconds"]:
                raise ProtocolError("wall_seconds_to_target exceeds wall_seconds")
        elif (
            telemetry["iterations_to_target"] is not None
            or telemetry["wall_seconds_to_target"] is not None
        ):
            raise ProtocolError("unreached convergence target requires null first-hit telemetry")
        _finite(telemetry["psnr_time_auc"], "psnr_time_auc")
        if set(artifacts) != set(REQUIRED_REPORT_ARTIFACTS):
            raise ProtocolError("ok rows have unexpected or missing artifacts")
        if validate_artifacts:
            artifact_paths: dict[str, Path] = {}
            normalized_artifacts = {}
            for name, value in artifacts.items():
                normalized_artifacts[name] = _validate_result_artifact(
                    value, base_path, f"rows[{index}].{name}"
                )
                artifact_paths[name] = _artifact_path(value, base_path)
                _require_artifact_in_phase_root(
                    protocol,
                    phase,
                    artifact_paths[name],
                    f"rows[{index}].{name}",
                )
            if (
                telemetry["authoritative_preclamp_sha256"]
                != normalized_artifacts["raw_render"]["sha256"]
            ):
                raise ProtocolError("pre-clamp digest must bind the raw_render artifact")
            field_manifest = _load_json_artifact(
                artifact_paths["field_manifest"], f"rows[{index}].field_manifest"
            )
            _exact_keys(
                field_manifest,
                {
                    "schema",
                    "cell_id",
                    "identity_sha256",
                    "semantic_sha256",
                    "renderer_equation",
                    "coefficient_variant",
                    "alpha_policy",
                    "row_count",
                    "canonical_raw_bytes",
                    "authoritative_preclamp_sha256",
                    "payload_format",
                    "payload_sha256",
                    "payload_bytes",
                },
                f"rows[{index}].field_manifest",
            )
            expected_manifest = {
                "schema": FIELD_MANIFEST_SCHEMA,
                "cell_id": cell["cell_id"],
                "identity_sha256": cell["identity_sha256"],
                "semantic_sha256": cell["bindings"]["semantic_sha256"],
                "renderer_equation": _arm_index(protocol)[cell["arm_id"]]["semantics"][
                    "renderer_equation"
                ],
                "coefficient_variant": cell["coefficient_variant"],
                "alpha_policy": cell["alpha_policy"],
                "row_count": cell["row_count"],
                "canonical_raw_bytes": cell["canonical_raw_bytes"],
                "authoritative_preclamp_sha256": telemetry["authoritative_preclamp_sha256"],
                "payload_format": _arm_index(protocol)[cell["arm_id"]]["semantics"][
                    "payload_format"
                ],
                "payload_sha256": normalized_artifacts["field_payload"]["sha256"],
                "payload_bytes": normalized_artifacts["field_payload"]["bytes"],
            }
            if dict(field_manifest) != expected_manifest:
                raise ProtocolError("field_manifest does not match the frozen cell semantics")
            metrics_record = _load_json_artifact(
                artifact_paths["metrics_json"], f"rows[{index}].metrics_json"
            )
            _exact_keys(
                metrics_record,
                {"schema", "cell_id", "metrics", "telemetry_sha256"},
                f"rows[{index}].metrics_json",
            )
            if dict(metrics_record) != {
                "schema": METRICS_SCHEMA,
                "cell_id": cell["cell_id"],
                "metrics": dict(metrics),
                "telemetry_sha256": _digest(telemetry),
            }:
                raise ProtocolError("metrics_json does not match row metrics/telemetry")
            history_record = _load_json_artifact(
                artifact_paths["history_json"], f"rows[{index}].history_json"
            )
            _exact_keys(
                history_record,
                {"schema", "cell_id", "metric", "points"},
                f"rows[{index}].history_json",
            )
            if (
                history_record["schema"] != HISTORY_SCHEMA
                or history_record["cell_id"] != cell["cell_id"]
                or history_record["metric"] != protocol["convergence"]["metric"]
            ):
                raise ProtocolError("history_json does not bind the frozen cell/metric")
            points = _list(history_record["points"], f"rows[{index}].history_json.points")
            replayed = summarize_convergence(protocol, phase, points)
            if points[-1]["iteration"] != telemetry["iterations_executed"]:
                raise ProtocolError("history terminal iteration differs from telemetry")
            if points[-1]["wall_seconds"] > telemetry["wall_seconds"]:
                raise ProtocolError("history terminal wall time exceeds telemetry")
            for name, replayed_value in replayed.items():
                observed_value = telemetry[name]
                if isinstance(replayed_value, float):
                    if not isinstance(observed_value, (int, float)) or not math.isclose(
                        float(observed_value), replayed_value, rel_tol=1e-9, abs_tol=1e-9
                    ):
                        raise ProtocolError(f"history replay disagrees with telemetry.{name}")
                elif observed_value != replayed_value:
                    raise ProtocolError(f"history replay disagrees with telemetry.{name}")
    missing = sorted(set(expected) - set(observed))
    if require_complete and missing:
        raise ProtocolError(f"missing {len(missing)} frozen result cells")
    errors = sorted(cell_id for cell_id, row in observed.items() if row["status"] == "error")
    return {
        "expected": len(expected),
        "observed": len(observed),
        "missing": missing,
        "errors": errors,
        "complete": not missing and not errors,
    }


def aa_replay_result(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    aa_rows = [
        row
        for row in rows
        if row.get("cell", {}).get("phase") == "coefficient_screen"
        and row.get("cell", {}).get("replicate") in {"primary", "aa"}
        and row.get("cell", {}).get("unit_id") == protocol["aa_replay"]["unit_id"]
        and row.get("cell", {}).get("arm_id") == protocol["aa_replay"]["arm_id"]
        and row.get("cell", {}).get("coefficient_variant")
        == protocol["aa_replay"]["coefficient_variant"]
        and row.get("cell", {}).get("alpha_policy") == protocol["aa_replay"]["alpha_policy"]
        and row.get("cell", {}).get("lane_id") == protocol["aa_replay"]["lane_id"]
        and row.get("cell", {}).get("seed") == protocol["aa_replay"]["seed"]
    ]
    by_replicate = {row["cell"]["replicate"]: row for row in aa_rows}
    if set(by_replicate) != {"primary", "aa"}:
        return {"pass": False, "reason": "A/A rows are missing", "metric_deltas": {}}
    primary, replay = by_replicate["primary"], by_replicate["aa"]
    if primary["status"] != "ok" or replay["status"] != "ok":
        return {"pass": False, "reason": "A/A row failed", "metric_deltas": {}}
    primary_identity = dict(primary["cell"])
    replay_identity = dict(replay["cell"])
    for value in (primary_identity, replay_identity):
        value.pop("replicate", None)
        value.pop("cell_id", None)
    identity_equal = primary_identity == replay_identity
    artifact_equal = all(
        primary["artifacts"][name]["sha256"] == replay["artifacts"][name]["sha256"]
        for name in ("field_payload", "raw_render", "evaluated_render")
    )
    metric_deltas = {
        name: abs(float(primary["metrics"][name]) - float(replay["metrics"][name]))
        for name in protocol["aa_replay"]["metric_abs_tolerance"]
    }
    metric_pass = all(
        metric_deltas[name] <= float(tolerance)
        for name, tolerance in protocol["aa_replay"]["metric_abs_tolerance"].items()
    )
    passed = identity_equal and artifact_equal and metric_pass
    return {
        "pass": passed,
        "reason": None if passed else "identity, artifact, or metric replay drift",
        "identity_equal": identity_equal,
        "artifact_equal": artifact_equal,
        "metric_deltas": metric_deltas,
    }


def matched_input_invariants(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], phase: str
) -> dict[str, Any]:
    matched_ids = {arm["id"] for arm in protocol["arms"] if arm["execution_kind"] == "matched"}
    groups: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        cell = row.get("cell", {})
        if (
            row.get("status") != "ok"
            or cell.get("phase") != phase
            or cell.get("arm_id") not in matched_ids
            or cell.get("replicate") != "primary"
        ):
            continue
        key = (cell["unit_id"], cell["lane_id"], cell["seed"], cell["alpha_policy"])
        groups.setdefault(key, []).append(row)
    failures: list[dict[str, Any]] = []
    invariant_names = (
        "pixels_sha256",
        "mask_sha256",
        "camera_sha256",
        "prepared_target_sha256",
        "metadata_selector_sha256",
        "seed_stream_sha256",
        "geometry_bank_sha256",
        "requested_work_sha256",
        "environment_sha256",
        "downstream_protocol_sha256",
    )
    for key, group in groups.items():
        if len({row["cell"]["arm_id"] for row in group}) < 2:
            continue
        for name in invariant_names:
            values = {row["cell"]["bindings"][name] for row in group}
            if len(values) != 1:
                failures.append({"group": list(key), "binding": name, "values": sorted(values)})
        row_counts = {row["cell"]["row_count"] for row in group}
        if len(row_counts) == 1:
            geometry = {row["cell"]["bindings"]["initial_geometry_sha256"] for row in group}
            if len(geometry) != 1:
                failures.append(
                    {
                        "group": list(key),
                        "binding": "initial_geometry_sha256",
                        "values": sorted(geometry),
                    }
                )
    return {"pass": not failures, "checked_groups": len(groups), "failures": failures}


def _ok_primary(rows: Sequence[Mapping[str, Any]], phase: str) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("cell", {}).get("phase") == phase
        and row.get("cell", {}).get("replicate") == "primary"
    ]


def _rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash scientific row content without binding machine-local artifact path spellings."""
    normalized = copy.deepcopy(list(rows))
    for row in normalized:
        artifacts = row.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        for record in artifacts.values():
            if isinstance(record, dict) and "path" in record:
                record["path"] = "<content-addressed-artifact>"
    return _digest(normalized)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ProtocolError("cannot average an empty sequence")
    return sum(values) / len(values)


def _direction(protocol: Mapping[str, Any], metric: str) -> float:
    return 1.0 if _metric_specs(protocol)[metric]["direction"] == "higher" else -1.0


def _guard_pass(value: float, op: str, threshold: float) -> bool:
    return value <= threshold if op == "<=" else value >= threshold


def analyze_coefficient_screen(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, base: str | Path = "."
) -> dict[str, Any]:
    _require_later_outcomes_empty(protocol, "coefficient_screen")
    status = validate_result_rows(
        protocol,
        rows,
        "coefficient_screen",
        base=base,
        require_complete=False,
    )
    aa = aa_replay_result(protocol, rows)
    primary = _ok_primary(rows, "coefficient_screen")
    domain_gate = protocol["gates"]["domain"]
    quality_metric = domain_gate["quality_metric"]
    screen_arm = protocol["phases"]["coefficient_screen"]["screen_arm_id"]
    variants = [
        variant
        for variant in _arm_index(protocol)[screen_arm]["coefficient_variants"]
        if variant != "not_applicable"
    ]
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_rows = [row for row in primary if row["cell"]["coefficient_variant"] == variant]
        metrics = (
            {
                name: _mean([float(row["metrics"][name]) for row in variant_rows])
                for name in _required_metrics(protocol, "coefficient_screen")
            }
            if variant_rows
            else {}
        )
        feasible = bool(variant_rows)
        guard_results = []
        for guard in domain_gate["feasibility_guards"]:
            values = [float(row["metrics"][guard["metric"]]) for row in variant_rows]
            passed = bool(values) and all(
                _guard_pass(value, guard["op"], float(guard["threshold"])) for value in values
            )
            feasible = feasible and passed
            guard_results.append({**guard, "pass": passed})
        summaries.append(
            {
                "coefficient_variant": variant,
                "rows": len(variant_rows),
                "metrics": metrics,
                "feasibility": guard_results,
                "feasible": feasible,
            }
        )
    complete = status["complete"] and aa["pass"]
    feasible = [summary for summary in summaries if summary["feasible"]]
    finalists: list[str] = []
    if complete and feasible:
        oriented_best = max(
            _direction(protocol, quality_metric) * summary["metrics"][quality_metric]
            for summary in feasible
        )
        margin = float(domain_gate["max_mean_degradation"])
        survivors = [
            summary
            for summary in feasible
            if _direction(protocol, quality_metric) * summary["metrics"][quality_metric]
            >= oriented_best - margin
        ]

        def priority_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
            return tuple(
                _direction(protocol, item["metric"]) * summary["metrics"][item["metric"]]
                for item in domain_gate["priority"]
            )

        survivors.sort(
            key=lambda summary: (priority_key(summary), summary["coefficient_variant"]),
            reverse=True,
        )
        finalists = [
            summary["coefficient_variant"]
            for summary in survivors[: int(domain_gate["max_finalists"])]
        ]
    core = {
        "schema": DOMAIN_LOCK_SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "coefficient_rows_sha256": _rows_digest(rows),
        "complete": complete,
        "status": status,
        "aa_replay": aa,
        "summaries": summaries,
        "finalists": finalists,
        "decision": "advance" if finalists else "stop_no_valid_domain",
    }
    return {**core, "domain_lock_sha256": _digest(core)}


def _unit_seed_averages(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[tuple[str, str, str, str, str], float]:
    groups: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in rows:
        cell = row["cell"]
        key = (
            cell["unit_id"],
            cell["arm_id"],
            cell["coefficient_variant"],
            cell["alpha_policy"],
            cell["lane_id"],
        )
        groups.setdefault(key, []).append(float(row["metrics"][metric]))
    return {key: _mean(values) for key, values in groups.items()}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ProtocolError("cannot take quantile of empty values")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_interval(
    differences: Mapping[str, Sequence[float]], replicates: int, seed: int
) -> dict[str, float]:
    clusters = sorted(differences)
    cluster_means = {cluster: _mean(differences[cluster]) for cluster in clusters}
    point = _mean(list(cluster_means.values()))
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = [cluster_means[rng.choice(clusters)] for _ in clusters]
        draws.append(_mean(sample))
    return {"mean": point, "lower": _quantile(draws, 0.025), "upper": _quantile(draws, 0.975)}


def _candidate_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    cell = row["cell"]
    return (
        cell["arm_id"],
        cell["coefficient_variant"],
        cell["alpha_policy"],
        cell["lane_id"],
    )


def _candidate_records(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    families = {arm["id"]: arm["family"] for arm in protocol["arms"]}
    allowed = set(protocol["gates"]["killing"]["candidate_families"])
    keys = {
        _candidate_key(row)[:3]
        for row in rows
        if families[row["cell"]["arm_id"]] in allowed
        and row["cell"]["lane_id"] in protocol["gates"]["killing"]["lane_ids"]
    }
    return [
        {
            "arm_id": arm_id,
            "coefficient_variant": variant,
            "alpha_policy": alpha_policy,
        }
        for arm_id, variant, alpha_policy in sorted(keys)
    ]


def _comparison_rows(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, str]
) -> list[dict[str, Any]]:
    units = _unit_index(protocol)
    specs = _metric_specs(protocol)
    killing = protocol["gates"]["killing"]
    comparisons: list[dict[str, Any]] = []
    metrics = [*killing["quality_noninferiority"], killing["downstream_favorable"]]
    for control_id in killing["controls"]:
        for lane_id in killing["lane_ids"]:
            for metric_gate in metrics:
                metric = metric_gate["metric"]
                averages = _unit_seed_averages(rows, metric)
                differences: dict[str, list[float]] = {}
                paired_units = 0
                for unit_id, (split, unit) in units.items():
                    del split
                    candidate_key = (
                        unit_id,
                        candidate["arm_id"],
                        candidate["coefficient_variant"],
                        candidate["alpha_policy"],
                        lane_id,
                    )
                    control_candidates = [
                        key
                        for key in averages
                        if key[0] == unit_id
                        and key[1] == control_id
                        and key[4] == lane_id
                        and key[3] == candidate["alpha_policy"]
                    ]
                    if candidate_key not in averages or len(control_candidates) != 1:
                        continue
                    control_key = control_candidates[0]
                    direction = 1.0 if specs[metric]["direction"] == "higher" else -1.0
                    difference = direction * (averages[candidate_key] - averages[control_key])
                    differences.setdefault(unit["capture_group"], []).append(difference)
                    paired_units += 1
                if not differences:
                    interval = None
                    passed = False
                else:
                    interval = _cluster_interval(
                        differences,
                        int(killing["bootstrap_replicates"]),
                        int(killing["bootstrap_seed"]) + len(comparisons),
                    )
                    passed = interval["lower"] >= -float(metric_gate["margin"])
                comparisons.append(
                    {
                        "control_id": control_id,
                        "lane_id": lane_id,
                        "metric": metric,
                        "direction": specs[metric]["direction"],
                        "margin": float(metric_gate["margin"]),
                        "paired_units": paired_units,
                        "capture_groups": len(differences),
                        "interval": interval,
                        "pass": passed,
                    }
                )
    return comparisons


def _candidate_guard_rows(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, str]
) -> list[dict[str, Any]]:
    candidate_identity = (
        candidate["arm_id"],
        candidate["coefficient_variant"],
        candidate["alpha_policy"],
    )
    candidate_rows = [
        row
        for row in rows
        if _candidate_key(row)[:3] == candidate_identity
        and row["cell"]["lane_id"] in protocol["gates"]["killing"]["lane_ids"]
    ]
    results = []
    for guard in protocol["gates"]["killing"]["absolute_guards"]:
        values = [float(row["metrics"][guard["metric"]]) for row in candidate_rows]
        mean = _mean(values) if values else None
        passed = mean is not None and _guard_pass(mean, guard["op"], float(guard["threshold"]))
        results.append({**guard, "mean": mean, "pass": passed})
    return results


def _candidate_means(
    protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, str]
) -> dict[str, float]:
    candidate_identity = (
        candidate["arm_id"],
        candidate["coefficient_variant"],
        candidate["alpha_policy"],
    )
    candidate_rows = [
        row
        for row in rows
        if _candidate_key(row)[:3] == candidate_identity
        and row["cell"]["lane_id"] in protocol["gates"]["killing"]["lane_ids"]
    ]
    return {
        metric: _mean([float(row["metrics"][metric]) for row in candidate_rows])
        for metric in protocol["gates"]["killing"]["pareto_metrics"]
    }


def _dominates(
    protocol: Mapping[str, Any], left: Mapping[str, float], right: Mapping[str, float]
) -> bool:
    metrics = protocol["gates"]["killing"]["pareto_metrics"]
    no_worse = True
    strictly_better = False
    for metric in metrics:
        direction = _direction(protocol, metric)
        left_value = direction * left[metric]
        right_value = direction * right[metric]
        no_worse = no_worse and left_value >= right_value
        strictly_better = strictly_better or left_value > right_value
    return no_worse and strictly_better


def analyze_factorial(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    *,
    base: str | Path = ".",
    domain_lock: Mapping[str, Any],
    confirmation_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require_later_outcomes_empty(protocol, phase)
    if phase not in {"development", "confirmation"}:
        raise ProtocolError("factorial analysis phase must be development or confirmation")
    status = validate_result_rows(
        protocol,
        rows,
        phase,
        base=base,
        domain_lock=domain_lock,
        confirmation_lock=confirmation_lock,
        require_complete=False,
    )
    invariant = matched_input_invariants(protocol, rows, phase)
    primary = _ok_primary(rows, phase)
    candidates = _candidate_records(protocol, primary)
    evaluated = []
    for candidate in candidates:
        comparisons = _comparison_rows(protocol, primary, candidate)
        guards = _candidate_guard_rows(protocol, primary, candidate)
        means = _candidate_means(protocol, primary, candidate)
        passed = (
            bool(comparisons)
            and all(item["pass"] for item in comparisons)
            and all(item["pass"] for item in guards)
        )
        evaluated.append(
            {
                **candidate,
                "comparisons": comparisons,
                "absolute_guards": guards,
                "pareto_means": means,
                "passes_killing_screen": passed,
            }
        )
    survivors = [candidate for candidate in evaluated if candidate["passes_killing_screen"]]
    nondominated = [
        candidate
        for candidate in survivors
        if not any(
            other is not candidate
            and _dominates(protocol, other["pareto_means"], candidate["pareto_means"])
            for other in survivors
        )
    ]
    selected = None
    if status["complete"] and invariant["pass"] and len(nondominated) == 1:
        selected = {
            name: nondominated[0][name]
            for name in ("arm_id", "coefficient_variant", "alpha_policy")
        }
    if not status["complete"]:
        decision = "incomplete"
    elif not invariant["pass"]:
        decision = "invalid_matched_inputs"
    elif not survivors:
        decision = "stop_no_additive_survivor"
    elif len(nondominated) != 1:
        decision = "heterogeneous_tradeoff"
    else:
        decision = "advance_one" if phase == "development" else "confirm_one"
    schema = DEVELOPMENT_ANALYSIS_SCHEMA if phase == "development" else CONFIRMATION_ANALYSIS_SCHEMA
    core = {
        "schema": schema,
        "protocol_sha256": protocol["protocol_sha256"],
        "phase": phase,
        "rows_sha256": _rows_digest(rows),
        "domain_lock_sha256": domain_lock["domain_lock_sha256"],
        "confirmation_lock_sha256": (
            None if confirmation_lock is None else confirmation_lock["confirmation_lock_sha256"]
        ),
        "status": status,
        "matched_input_invariants": invariant,
        "candidates": evaluated,
        "nondominated": [
            {name: candidate[name] for name in ("arm_id", "coefficient_variant", "alpha_policy")}
            for candidate in nondominated
        ],
        "selected": selected,
        "decision": decision,
    }
    core["analysis_sha256"] = _digest(core)
    return core


def development_review_template(
    protocol: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    if analysis.get("schema") != DEVELOPMENT_ANALYSIS_SCHEMA:
        raise ProtocolError("development review requires a development analysis")
    return {
        "schema": DEVELOPMENT_REVIEW_SCHEMA,
        "driver": protocol["driver"],
        "reviewer": "replace-with-distinct-results-reviewer",
        "verdict": "approved-or-rejected",
        "protocol_sha256": protocol["protocol_sha256"],
        "development_analysis_sha256": analysis["analysis_sha256"],
        "outcomes_accessed": True,
        "notes": "Audit A/A, missing/error retention, gates, bootstrap units, and selection.",
    }


def lock_confirmation(
    protocol: Mapping[str, Any],
    analysis: Mapping[str, Any],
    review_path: str | Path,
    *,
    base: str | Path = ".",
) -> dict[str, Any]:
    validate_protocol(protocol, base=base, require_frozen=True)
    if analysis.get("schema") != DEVELOPMENT_ANALYSIS_SCHEMA:
        raise ProtocolError("confirmation lock requires development analysis")
    analysis_core = copy.deepcopy(dict(analysis))
    recorded_analysis_digest = analysis_core.pop("analysis_sha256", None)
    if recorded_analysis_digest != _digest(analysis_core):
        raise ProtocolError("development analysis digest is mismatched")
    if analysis.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ProtocolError("development analysis binds a different protocol")
    if analysis.get("decision") != "advance_one" or not isinstance(
        analysis.get("selected"), Mapping
    ):
        raise ProtocolError("development analysis did not select exactly one candidate")
    confirmation_root = Path(protocol["execution"]["outcome_roots"]["confirmation"])
    if confirmation_root.exists() and (
        not confirmation_root.is_dir() or any(confirmation_root.iterdir())
    ):
        raise ProtocolError("confirmation outcome root is non-empty; locking is too late")
    review_file = Path(review_path).expanduser().resolve()
    try:
        review = json.loads(review_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("cannot read development review") from exc
    record = _mapping(review, "development review")
    _exact_keys(
        record,
        {
            "schema",
            "driver",
            "reviewer",
            "verdict",
            "protocol_sha256",
            "development_analysis_sha256",
            "outcomes_accessed",
            "notes",
        },
        "development review",
    )
    reviewer = _identifier(record["reviewer"], "development reviewer")
    if (
        record["schema"] != DEVELOPMENT_REVIEW_SCHEMA
        or record["driver"] != protocol["driver"]
        or reviewer.casefold() == protocol["driver"].casefold()
        or record["verdict"] != "approved"
        or record["protocol_sha256"] != protocol["protocol_sha256"]
        or record["development_analysis_sha256"] != analysis["analysis_sha256"]
        or record["outcomes_accessed"] is not True
    ):
        raise ProtocolError("development review is not a matching distinct approval")
    selected = {
        name: analysis["selected"][name]
        for name in ("arm_id", "coefficient_variant", "alpha_policy")
    }
    core = {
        "schema": CONFIRMATION_LOCK_SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "development_analysis_sha256": analysis["analysis_sha256"],
        "selected": selected,
        "review": {
            "reviewer": reviewer,
            "verdict": "approved",
            "artifact": _seal_artifact(
                {"path": str(review_file)}, Path(base).resolve(), "development review"
            ),
        },
        "confirmation_outcome_root": str(confirmation_root),
    }
    return {**core, "confirmation_lock_sha256": _digest(core)}


def results_audit_template(
    protocol: Mapping[str, Any], confirmation_analysis: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": RESULTS_AUDIT_SCHEMA,
        "driver": protocol["driver"],
        "reviewer": "replace-with-distinct-results-auditor",
        "verdict": "approved-or-rejected",
        "protocol_sha256": protocol["protocol_sha256"],
        "confirmation_analysis_sha256": confirmation_analysis.get("analysis_sha256"),
        "outcomes_accessed": True,
        "notes": "Audit raw rows, report projections, gates, uncertainty, and final disposition.",
    }


def _validate_results_audit(
    protocol: Mapping[str, Any],
    confirmation_analysis: Mapping[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    audit_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("cannot read results audit") from exc
    audit = _mapping(value, "results audit")
    _exact_keys(
        audit,
        {
            "schema",
            "driver",
            "reviewer",
            "verdict",
            "protocol_sha256",
            "confirmation_analysis_sha256",
            "outcomes_accessed",
            "notes",
        },
        "results audit",
    )
    reviewer = _identifier(audit["reviewer"], "results audit reviewer")
    if (
        audit["schema"] != RESULTS_AUDIT_SCHEMA
        or audit["driver"] != protocol["driver"]
        or reviewer.casefold() == protocol["driver"].casefold()
        or audit["verdict"] != "approved"
        or audit["protocol_sha256"] != protocol["protocol_sha256"]
        or audit["confirmation_analysis_sha256"] != confirmation_analysis.get("analysis_sha256")
        or audit["outcomes_accessed"] is not True
    ):
        raise ProtocolError("results audit is not a matching distinct approval")
    return {
        "reviewer": reviewer,
        "verdict": "approved",
        "artifact": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "bytes": audit_path.stat().st_size,
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row).decode("utf-8") + "\n")


def _csv_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return canonical_json(value).decode("utf-8")


def _flat_row(row: Mapping[str, Any]) -> dict[str, object]:
    cell = row["cell"]
    flattened: dict[str, object] = {
        "schema": row["schema"],
        "cell_id": cell["cell_id"],
        "phase": cell["phase"],
        "unit_id": cell["unit_id"],
        "capture_group": cell["capture_group"],
        "frame_id": cell["frame_id"],
        "arm_id": cell["arm_id"],
        "family": cell["family"],
        "coefficient_variant": cell["coefficient_variant"],
        "alpha_policy": cell["alpha_policy"],
        "lane_id": cell["lane_id"],
        "seed": cell["seed"],
        "replicate": cell["replicate"],
        "status": row["status"],
        "error": row["error"],
    }
    for prefix in ("telemetry", "metrics"):
        for name, value in row[prefix].items():
            flattened[f"{prefix}.{name}"] = _csv_value(value)
    for name, value in row["artifacts"].items():
        flattened[f"artifact.{name}"] = value["path"]
        flattened[f"artifact.{name}.sha256"] = value["sha256"]
    return flattened


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened = [_flat_row(row) for row in rows]
    fields = sorted({name for row in flattened for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def _validate_csv_projection(
    path: Path, rows: Sequence[Mapping[str, Any]], phase: str
) -> list[str]:
    problems: list[str] = []
    flattened = [_flat_row(row) for row in rows]
    expected_fields = sorted({name for row in flattened for name in row})
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            projected = list(reader)
    except (OSError, csv.Error) as exc:
        return [f"{phase} CSV is invalid: {exc}"]
    if fields != expected_fields:
        problems.append(f"{phase} CSV columns disagree with JSON rows")
        return problems
    if len(projected) != len(flattened):
        problems.append(f"{phase} CSV row count disagrees with JSON rows")
        return problems
    for row_index, (csv_row, expected_row) in enumerate(zip(projected, flattened)):
        for field in expected_fields:
            value = expected_row.get(field)
            expected = "" if value is None else str(value)
            if csv_row.get(field) != expected:
                problems.append(f"{phase} CSV row {row_index} field {field!r} disagrees with JSON")
    return problems


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _copy_binding(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _materialize_protocol_bindings(protocol: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    used: set[str] = set()
    records = list(_iter_artifact_slots(protocol))
    if protocol.get("review"):
        records.append(("review.artifact", protocol["review"]["artifact"]))
    for label, record in records:
        source = Path(record["path"])
        stem = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_") or "binding"
        suffix = "".join(source.suffixes) or ".bin"
        name = f"{stem}-{record['sha256'][:12]}{suffix}"
        if name in used:
            name = f"{stem}-{record['sha256']}{suffix}"
        used.add(name)
        destination = out / "bindings" / name
        _copy_binding(source, destination)
        copied = _file_record(destination, out)
        if copied["sha256"] != record["sha256"] or copied["bytes"] != record["bytes"]:
            raise ProtocolError(f"copied binding {label} changed content")
        bindings.append({"label": label, **copied})
    return bindings


def _portable_rows(
    rows: Sequence[Mapping[str, Any]], out: Path, phase: str, base: Path
) -> list[dict[str, Any]]:
    result = copy.deepcopy(list(rows))
    for row in result:
        if row["status"] != "ok":
            continue
        cell_id = row["cell"]["cell_id"]
        for name, record in row["artifacts"].items():
            source = _artifact_path(record, base)
            _validate_artifact(record, base, f"{phase}.{cell_id}.{name}")
            suffix = "".join(source.suffixes) or ".bin"
            destination = out / "artifacts" / phase / cell_id / f"{name}{suffix}"
            _copy_binding(source, destination)
            row["artifacts"][name] = _file_record(destination, out)
    if _rows_digest(result) != _rows_digest(rows):
        raise ProtocolError("portable row projection changed scientific content")
    return result


def _analysis_digest_valid(analysis: Mapping[str, Any]) -> bool:
    value = copy.deepcopy(dict(analysis))
    recorded = value.pop("analysis_sha256", None)
    return isinstance(recorded, str) and recorded == _digest(value)


def write_report(
    protocol: Mapping[str, Any],
    coefficient_rows: Sequence[Mapping[str, Any]],
    domain_lock: Mapping[str, Any],
    development_rows: Sequence[Mapping[str, Any]],
    development_analysis: Mapping[str, Any],
    outdir: str | Path,
    *,
    row_base: str | Path = ".",
    confirmation_lock: Mapping[str, Any] | None = None,
    confirmation_rows: Sequence[Mapping[str, Any]] | None = None,
    confirmation_analysis: Mapping[str, Any] | None = None,
    results_audit_path: str | Path | None = None,
    command: str = "",
) -> dict[str, Any]:
    validate_protocol(protocol, require_frozen=True)
    _validate_lock_digest(domain_lock, DOMAIN_LOCK_SCHEMA, "domain lock")
    if domain_lock["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ProtocolError("domain lock binds a different protocol")
    validate_result_rows(
        protocol,
        coefficient_rows,
        "coefficient_screen",
        base=row_base,
        require_complete=False,
    )
    validate_result_rows(
        protocol,
        development_rows,
        "development",
        base=row_base,
        domain_lock=domain_lock,
        require_complete=False,
    )
    if (
        development_analysis.get("schema") != DEVELOPMENT_ANALYSIS_SCHEMA
        or not _analysis_digest_valid(development_analysis)
        or development_analysis.get("rows_sha256") != _rows_digest(development_rows)
    ):
        raise ProtocolError("development analysis is invalid or does not bind its rows")
    if development_analysis.get("domain_lock_sha256") != domain_lock["domain_lock_sha256"]:
        raise ProtocolError("development analysis binds a different domain lock")

    confirmation_rows = list(confirmation_rows or [])
    if (confirmation_lock is None) != (confirmation_analysis is None):
        raise ProtocolError("confirmation lock and analysis must be provided together")
    if confirmation_lock is not None:
        _validate_lock_digest(confirmation_lock, CONFIRMATION_LOCK_SCHEMA, "confirmation lock")
        validate_result_rows(
            protocol,
            confirmation_rows,
            "confirmation",
            base=row_base,
            domain_lock=domain_lock,
            confirmation_lock=confirmation_lock,
            require_complete=False,
        )
        if (
            confirmation_analysis.get("schema") != CONFIRMATION_ANALYSIS_SCHEMA
            or not _analysis_digest_valid(confirmation_analysis)
            or confirmation_analysis.get("rows_sha256") != _rows_digest(confirmation_rows)
            or confirmation_analysis.get("confirmation_lock_sha256")
            != confirmation_lock["confirmation_lock_sha256"]
        ):
            raise ProtocolError("confirmation analysis is invalid or mismatched")
    elif confirmation_rows:
        raise ProtocolError("confirmation rows require a lock and analysis")

    audit = None
    if results_audit_path is not None:
        if confirmation_analysis is None:
            raise ProtocolError("results audit requires confirmation analysis")
        audit = _validate_results_audit(protocol, confirmation_analysis, results_audit_path)

    out = Path(outdir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ProtocolError("report output directory must be absent or empty")
    out.mkdir(parents=True, exist_ok=True)
    row_base_path = Path(row_base).resolve()
    portable_coefficient = _portable_rows(
        coefficient_rows, out, "coefficient_screen", row_base_path
    )
    portable_development = _portable_rows(development_rows, out, "development", row_base_path)
    portable_confirmation = _portable_rows(confirmation_rows, out, "confirmation", row_base_path)

    protocol_path = out / "protocol" / "frozen.json"
    domain_path = out / "analysis" / "domain_lock.json"
    development_path = out / "analysis" / "development.json"
    _write_json(protocol_path, protocol)
    _write_json(domain_path, domain_lock)
    _write_json(development_path, development_analysis)
    files: dict[str, dict[str, Any]] = {
        "protocol": _file_record(protocol_path, out),
        "domain_lock": _file_record(domain_path, out),
        "development_analysis": _file_record(development_path, out),
    }
    if confirmation_lock is not None and confirmation_analysis is not None:
        lock_path = out / "analysis" / "confirmation_lock.json"
        confirmation_path = out / "analysis" / "confirmation.json"
        _write_json(lock_path, confirmation_lock)
        _write_json(confirmation_path, confirmation_analysis)
        files["confirmation_lock"] = _file_record(lock_path, out)
        files["confirmation_analysis"] = _file_record(confirmation_path, out)
    if audit is not None:
        source = Path(audit["artifact"]["path"])
        audit_path = out / "analysis" / "results_audit.json"
        _copy_binding(source, audit_path)
        audit["artifact"] = _file_record(audit_path, out)
        files["results_audit"] = audit["artifact"]

    row_sets = {
        "coefficient_screen": portable_coefficient,
        "development": portable_development,
        "confirmation": portable_confirmation,
    }
    for phase, phase_rows in row_sets.items():
        json_path = out / "rows" / f"{phase}.json"
        jsonl_path = out / "rows" / f"{phase}.jsonl"
        csv_path = out / "rows" / f"{phase}.csv"
        _write_json(json_path, phase_rows)
        _write_jsonl(jsonl_path, phase_rows)
        _write_csv(csv_path, phase_rows)
        files[f"{phase}_json"] = _file_record(json_path, out)
        files[f"{phase}_jsonl"] = _file_record(jsonl_path, out)
        files[f"{phase}_csv"] = _file_record(csv_path, out)

    bindings = _materialize_protocol_bindings(protocol, out)
    confirmation_decision = (
        None if confirmation_analysis is None else confirmation_analysis.get("decision")
    )
    selected = None if confirmation_analysis is None else confirmation_analysis.get("selected")
    structurally_complete = bool(
        domain_lock.get("decision") == "advance"
        and development_analysis.get("decision") == "advance_one"
        and confirmation_decision == "confirm_one"
        and selected is not None
    )
    claim_ready = structurally_complete and audit is not None
    if claim_ready:
        family = _arm_index(protocol)[selected["arm_id"]]["family"]
        decision = {
            "outcome": family,
            "selected": selected,
            "interpretation": "reviewed sealed confirmation selected one Field V2 semantic",
        }
    elif confirmation_decision in {"stop_no_additive_survivor", "heterogeneous_tradeoff"}:
        decision = {
            "outcome": "no_new_contract",
            "selected": None,
            "interpretation": confirmation_decision,
        }
    else:
        decision = {
            "outcome": "diagnostic_incomplete",
            "selected": None,
            "interpretation": "no audited sealed confirmation decision",
        }
    manifest = {
        "schema": REPORT_SCHEMA,
        "task_id": "BENCH-020",
        "protocol_sha256": protocol["protocol_sha256"],
        "claim_scope": protocol["claim_scope"],
        "command": command,
        "files": files,
        "protocol_bindings": bindings,
        "row_counts": {phase: len(rows) for phase, rows in row_sets.items()},
        "row_digests": {phase: _rows_digest(rows) for phase, rows in row_sets.items()},
        "structurally_complete": structurally_complete,
        "claim_ready": claim_ready,
        "decision": decision,
    }
    manifest_path = out / "manifest.json"
    _write_json(manifest_path, manifest)
    links = [record["path"] for record in files.values()]
    links.extend(record["path"] for record in bindings)
    link_html = "\n".join(f'<li><a href="{path}">{path}</a></li>' for path in sorted(links))
    index = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BENCH-020 field semantics</title></head>
<body><h1>BENCH-020 field semantics and alpha-policy factorial</h1>
<p>Decision: <code>{decision["outcome"]}</code>; claim-ready: <code>{str(claim_ready).lower()}</code>.</p>
<p>Raw appearance is authoritative. Evaluation clipping, alpha matting, structural mass, and
canonical raw bytes remain separately bound by the frozen protocol.</p>
<p>First-hit convergence and PSNR-time AUC replay from per-cell histories under the frozen
wall-time horizon and target contract.</p>
<h2>Portable files</h2><ul>{link_html}</ul></body></html>\n"""
    (out / "index.html").write_text(index, encoding="utf-8")
    return manifest


def _contained_file(root: Path, record: object, label: str) -> tuple[Path | None, list[str]]:
    problems: list[str] = []
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "bytes"}:
        return None, [f"{label} is not a complete file record"]
    raw = Path(str(record["path"]))
    if raw.is_absolute() or ".." in raw.parts:
        return None, [f"{label} path escapes the report"]
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None, [f"{label} path escapes the report"]
    if not path.is_file():
        return None, [f"{label} is missing"]
    try:
        expected_bytes = int(record["bytes"])
    except (TypeError, ValueError):
        return None, [f"{label} byte count is invalid"]
    if path.stat().st_size != expected_bytes or sha256_file(path) != record["sha256"]:
        problems.append(f"{label} hash/size mismatch")
    return path, problems


def validate_report_bundle(root: str | Path) -> list[str]:
    report = Path(root).resolve()
    problems: list[str] = []
    manifest_path = report / "manifest.json"
    index_path = report / "index.html"
    if not manifest_path.is_file() or not index_path.is_file():
        return ["BENCH-020 report requires manifest.json and index.html"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["BENCH-020 manifest is invalid JSON"]
    if not isinstance(manifest, Mapping) or manifest.get("schema") != REPORT_SCHEMA:
        return [f"BENCH-020 manifest schema must be {REPORT_SCHEMA}"]
    required_manifest = {
        "schema",
        "task_id",
        "protocol_sha256",
        "claim_scope",
        "command",
        "files",
        "protocol_bindings",
        "row_counts",
        "row_digests",
        "structurally_complete",
        "claim_ready",
        "decision",
    }
    if set(manifest) != required_manifest:
        problems.append("BENCH-020 manifest fields mismatch")
    if manifest.get("task_id") != "BENCH-020":
        problems.append("BENCH-020 manifest task_id is invalid")
    if manifest.get("claim_scope") not in {"general", "workload_specific"}:
        problems.append("BENCH-020 manifest claim_scope is invalid")
    if not isinstance(manifest.get("command"), str) or not manifest["command"].strip():
        problems.append("BENCH-020 manifest has no executed command")
    files = manifest.get("files")
    loaded: dict[str, object] = {}
    resolved_files: dict[str, Path] = {}
    if not isinstance(files, Mapping):
        problems.append("BENCH-020 files must be an object")
        files = {}
    required_files = {
        "protocol",
        "domain_lock",
        "development_analysis",
        *(f"{phase}_{suffix}" for phase in PHASES for suffix in ("json", "jsonl", "csv")),
    }
    optional_files = {"confirmation_lock", "confirmation_analysis", "results_audit"}
    if not required_files <= set(files) or not set(files) <= required_files | optional_files:
        problems.append("BENCH-020 files are missing required entries or contain unknown entries")
    if ("confirmation_lock" in files) != ("confirmation_analysis" in files):
        problems.append("confirmation lock and analysis files must appear together")
    for name, record in files.items():
        path, file_problems = _contained_file(report, record, f"files.{name}")
        problems.extend(file_problems)
        if path is not None:
            resolved_files[name] = path
        if path is not None and path.suffix in {".json", ".jsonl"}:
            try:
                if path.suffix == ".jsonl":
                    loaded[name] = [
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                else:
                    loaded[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                problems.append(f"files.{name} is invalid JSON")
    bindings = manifest.get("protocol_bindings")
    if not isinstance(bindings, list) or not bindings:
        problems.append("BENCH-020 report has no protocol bindings")
        bindings = []
    for index, record in enumerate(bindings):
        if not isinstance(record, Mapping) or set(record) != {"label", "path", "sha256", "bytes"}:
            problems.append(f"protocol_bindings[{index}] is malformed")
            continue
        _path, binding_problems = _contained_file(
            report,
            {name: record[name] for name in ("path", "sha256", "bytes")},
            f"protocol_bindings[{index}]",
        )
        problems.extend(binding_problems)

    protocol = loaded.get("protocol")
    domain = loaded.get("domain_lock")
    development = loaded.get("development_analysis")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        problems.append("report protocol is missing or has wrong schema")
    else:
        if protocol.get("state") != "frozen":
            problems.append("report protocol is not frozen")
        if protocol.get("protocol_sha256") != protocol_digest(protocol):
            problems.append("report protocol self-digest mismatch")
        if protocol.get("protocol_sha256") != manifest.get("protocol_sha256"):
            problems.append("manifest/protocol digest mismatch")
    if not isinstance(domain, Mapping):
        problems.append("domain lock is missing")
    else:
        try:
            _validate_lock_digest(domain, DOMAIN_LOCK_SCHEMA, "domain lock")
        except ProtocolError as exc:
            problems.append(str(exc))
        if domain.get("protocol_sha256") != manifest.get("protocol_sha256"):
            problems.append("domain lock binds a different protocol")
    if not isinstance(development, Mapping) or not _analysis_digest_valid(development):
        problems.append("development analysis is missing or invalid")
    elif (
        development.get("protocol_sha256") != manifest.get("protocol_sha256")
        or not isinstance(domain, Mapping)
        or development.get("domain_lock_sha256") != domain.get("domain_lock_sha256")
    ):
        problems.append("development analysis binds a different protocol/domain lock")

    if isinstance(protocol, Mapping):
        expected_bindings = dict(_iter_artifact_slots(protocol))
        review = protocol.get("review")
        if isinstance(review, Mapping) and isinstance(review.get("artifact"), Mapping):
            expected_bindings["review.artifact"] = review["artifact"]
        observed_bindings = {
            record.get("label"): record
            for record in bindings
            if isinstance(record, Mapping) and isinstance(record.get("label"), str)
        }
        if len(observed_bindings) != len(bindings) or set(observed_bindings) != set(
            expected_bindings
        ):
            problems.append("portable protocol bindings do not exactly cover the frozen protocol")
        for label in sorted(set(observed_bindings) & set(expected_bindings)):
            observed = observed_bindings[label]
            expected = expected_bindings[label]
            if observed.get("sha256") != expected.get("sha256") or observed.get(
                "bytes"
            ) != expected.get("bytes"):
                problems.append(f"portable protocol binding {label} differs from its source seal")

    row_counts = manifest.get("row_counts")
    row_digests = manifest.get("row_digests")
    if not isinstance(row_counts, Mapping) or set(row_counts) != set(PHASES):
        problems.append("manifest row_counts must cover every phase exactly")
        row_counts = {}
    if not isinstance(row_digests, Mapping) or set(row_digests) != set(PHASES):
        problems.append("manifest row_digests must cover every phase exactly")
        row_digests = {}

    for phase in PHASES:
        json_rows = loaded.get(f"{phase}_json")
        jsonl_rows = loaded.get(f"{phase}_jsonl")
        if not isinstance(json_rows, list) or not isinstance(jsonl_rows, list):
            problems.append(f"{phase} JSON/JSONL rows are missing")
            continue
        if canonical_json(json_rows) != canonical_json(jsonl_rows):
            problems.append(f"{phase} JSON and JSONL projections disagree")
        if row_counts.get(phase) != len(json_rows):
            problems.append(f"{phase} row count disagrees with manifest")
        if row_digests.get(phase) != _rows_digest(json_rows):
            problems.append(f"{phase} row digest disagrees with manifest")
        csv_path = resolved_files.get(f"{phase}_csv")
        if csv_path is None:
            problems.append(f"{phase} CSV projection is missing")
        else:
            problems.extend(_validate_csv_projection(csv_path, json_rows, phase))
        for row_index, row in enumerate(json_rows):
            label = f"{phase} row {row_index}"
            if not isinstance(row, Mapping) or row.get("schema") != ROW_SCHEMA:
                problems.append(f"{phase} row {row_index} has wrong schema")
                continue
            if set(row) != {
                "schema",
                "cell",
                "status",
                "error",
                "telemetry",
                "metrics",
                "artifacts",
            }:
                problems.append(f"{label} fields mismatch")
                continue
            artifacts = row.get("artifacts", {})
            if row.get("status") == "error":
                if (
                    not isinstance(row.get("error"), str)
                    or not row["error"].strip()
                    or row.get("telemetry")
                    or row.get("metrics")
                    or artifacts
                ):
                    problems.append(f"{label} error contract is malformed")
                continue
            if row.get("status") != "ok" or row.get("error") is not None:
                problems.append(f"{label} status/error contract is malformed")
                continue
            if not isinstance(artifacts, Mapping) or set(artifacts) != set(
                REQUIRED_REPORT_ARTIFACTS
            ):
                problems.append(f"{phase} row {row_index} artifacts mismatch")
                continue
            artifact_paths: dict[str, Path] = {}
            for name, record in artifacts.items():
                path, artifact_problems = _contained_file(
                    report, record, f"{phase} row {row_index}.{name}"
                )
                problems.extend(artifact_problems)
                if path is not None:
                    artifact_paths[name] = path
            if not isinstance(protocol, Mapping) or len(artifact_paths) != len(
                REQUIRED_REPORT_ARTIFACTS
            ):
                continue
            try:
                cell = _mapping(row.get("cell"), f"{label}.cell")
                metrics = _mapping(row.get("metrics"), f"{label}.metrics")
                telemetry = _mapping(row.get("telemetry"), f"{label}.telemetry")
                field_manifest = _load_json_artifact(
                    artifact_paths["field_manifest"], f"{label}.field_manifest"
                )
                arm = next(arm for arm in protocol["arms"] if arm["id"] == cell.get("arm_id"))
                expected_field = {
                    "schema": FIELD_MANIFEST_SCHEMA,
                    "cell_id": cell["cell_id"],
                    "identity_sha256": cell["identity_sha256"],
                    "semantic_sha256": cell["bindings"]["semantic_sha256"],
                    "renderer_equation": arm["semantics"]["renderer_equation"],
                    "coefficient_variant": cell["coefficient_variant"],
                    "alpha_policy": cell["alpha_policy"],
                    "row_count": cell["row_count"],
                    "canonical_raw_bytes": cell["canonical_raw_bytes"],
                    "authoritative_preclamp_sha256": telemetry["authoritative_preclamp_sha256"],
                    "payload_format": arm["semantics"]["payload_format"],
                    "payload_sha256": artifacts["field_payload"]["sha256"],
                    "payload_bytes": artifacts["field_payload"]["bytes"],
                }
                if dict(field_manifest) != expected_field:
                    problems.append(f"{label} field manifest disagrees with its row")
                metrics_record = _load_json_artifact(
                    artifact_paths["metrics_json"], f"{label}.metrics_json"
                )
                if dict(metrics_record) != {
                    "schema": METRICS_SCHEMA,
                    "cell_id": cell["cell_id"],
                    "metrics": dict(metrics),
                    "telemetry_sha256": _digest(telemetry),
                }:
                    problems.append(f"{label} metrics artifact disagrees with its row")
                history = _load_json_artifact(
                    artifact_paths["history_json"], f"{label}.history_json"
                )
                if (
                    history.get("schema") != HISTORY_SCHEMA
                    or history.get("cell_id") != cell["cell_id"]
                    or history.get("metric") != protocol["convergence"]["metric"]
                ):
                    problems.append(f"{label} history artifact has the wrong identity")
                else:
                    replayed = summarize_convergence(protocol, phase, history.get("points", []))
                    for name, expected_value in replayed.items():
                        observed_value = telemetry.get(name)
                        if isinstance(expected_value, float):
                            if not isinstance(observed_value, (int, float)) or not math.isclose(
                                float(observed_value),
                                expected_value,
                                rel_tol=1e-9,
                                abs_tol=1e-9,
                            ):
                                problems.append(f"{label} history disagrees with telemetry.{name}")
                        elif observed_value != expected_value:
                            problems.append(f"{label} history disagrees with telemetry.{name}")
            except (
                KeyError,
                StopIteration,
                TypeError,
                ProtocolError,
                OSError,
                json.JSONDecodeError,
            ):
                problems.append(f"{label} portable artifact contract is malformed")

    coefficient_rows = loaded.get("coefficient_screen_json")
    development_rows = loaded.get("development_json")
    confirmation_rows = loaded.get("confirmation_json")
    if isinstance(domain, Mapping) and isinstance(coefficient_rows, list):
        if domain.get("coefficient_rows_sha256") != _rows_digest(coefficient_rows):
            problems.append("domain lock does not bind the coefficient rows")
    if isinstance(development, Mapping) and isinstance(development_rows, list):
        if development.get("rows_sha256") != _rows_digest(development_rows):
            problems.append("development analysis does not bind the development rows")

    confirmation_lock = loaded.get("confirmation_lock")
    confirmation = loaded.get("confirmation_analysis")
    if confirmation_lock is not None:
        try:
            _validate_lock_digest(
                _mapping(confirmation_lock, "confirmation lock"),
                CONFIRMATION_LOCK_SCHEMA,
                "confirmation lock",
            )
        except ProtocolError as exc:
            problems.append(str(exc))
        if confirmation_lock.get("protocol_sha256") != manifest.get("protocol_sha256"):
            problems.append("confirmation lock binds a different protocol")
    if confirmation is not None and (
        not isinstance(confirmation, Mapping) or not _analysis_digest_valid(confirmation)
    ):
        problems.append("confirmation analysis is invalid")
    elif isinstance(confirmation, Mapping):
        if (
            confirmation.get("protocol_sha256") != manifest.get("protocol_sha256")
            or not isinstance(confirmation_lock, Mapping)
            or confirmation.get("confirmation_lock_sha256")
            != confirmation_lock.get("confirmation_lock_sha256")
        ):
            problems.append("confirmation analysis binds a different protocol/confirmation lock")
        if isinstance(confirmation_rows, list) and confirmation.get("rows_sha256") != _rows_digest(
            confirmation_rows
        ):
            problems.append("confirmation analysis does not bind the confirmation rows")

    audit = loaded.get("results_audit")
    audit_valid = False
    if audit is not None:
        if not isinstance(audit, Mapping) or set(audit) != {
            "schema",
            "driver",
            "reviewer",
            "verdict",
            "protocol_sha256",
            "confirmation_analysis_sha256",
            "outcomes_accessed",
            "notes",
        }:
            problems.append("results audit is malformed")
        elif not isinstance(protocol, Mapping) or not isinstance(confirmation, Mapping):
            problems.append("results audit has no protocol/confirmation to audit")
        else:
            audit_valid = bool(
                audit.get("schema") == RESULTS_AUDIT_SCHEMA
                and audit.get("driver") == protocol.get("driver")
                and isinstance(audit.get("reviewer"), str)
                and audit["reviewer"].strip()
                and audit["reviewer"].casefold() != str(protocol.get("driver")).casefold()
                and audit.get("verdict") == "approved"
                and audit.get("protocol_sha256") == protocol.get("protocol_sha256")
                and audit.get("confirmation_analysis_sha256") == confirmation.get("analysis_sha256")
                and audit.get("outcomes_accessed") is True
            )
            if not audit_valid:
                problems.append("results audit is not a matching distinct approval")

    structurally_complete = bool(
        isinstance(domain, Mapping)
        and domain.get("decision") == "advance"
        and isinstance(development, Mapping)
        and development.get("decision") == "advance_one"
        and isinstance(confirmation, Mapping)
        and confirmation.get("decision") == "confirm_one"
        and confirmation.get("selected") is not None
    )
    if manifest.get("structurally_complete") is not structurally_complete:
        problems.append("manifest structurally_complete disagrees with sealed analyses")
    expected_claim_ready = structurally_complete and audit_valid
    if manifest.get("claim_ready") is not expected_claim_ready:
        problems.append("manifest claim_ready disagrees with confirmation/audit state")
    decision = manifest.get("decision")
    if not isinstance(decision, Mapping) or set(decision) != {
        "outcome",
        "selected",
        "interpretation",
    }:
        problems.append("BENCH-020 decision is malformed")
    expected_decision: dict[str, Any]
    if expected_claim_ready and isinstance(protocol, Mapping) and isinstance(confirmation, Mapping):
        selected = confirmation["selected"]
        arms = {arm.get("id"): arm for arm in protocol.get("arms", []) if isinstance(arm, Mapping)}
        selected_arm = arms.get(selected.get("arm_id")) if isinstance(selected, Mapping) else None
        expected_decision = {
            "outcome": None if selected_arm is None else selected_arm.get("family"),
            "selected": selected,
            "interpretation": "reviewed sealed confirmation selected one Field V2 semantic",
        }
    elif isinstance(confirmation, Mapping) and confirmation.get("decision") in {
        "stop_no_additive_survivor",
        "heterogeneous_tradeoff",
    }:
        expected_decision = {
            "outcome": "no_new_contract",
            "selected": None,
            "interpretation": confirmation["decision"],
        }
    else:
        expected_decision = {
            "outcome": "diagnostic_incomplete",
            "selected": None,
            "interpretation": "no audited sealed confirmation decision",
        }
    if isinstance(decision, Mapping) and dict(decision) != expected_decision:
        problems.append("manifest decision disagrees with the sealed analysis/audit state")
    html = index_path.read_text(encoding="utf-8", errors="replace")
    for record in list(files.values()) + [
        {name: binding[name] for name in ("path", "sha256", "bytes")}
        for binding in bindings
        if isinstance(binding, Mapping) and "path" in binding
    ]:
        if isinstance(record, Mapping) and str(record.get("path")) not in html:
            problems.append(f"index.html does not link {record.get('path')}")
    return problems


def _template_policy_contract(policy: str) -> dict[str, str]:
    target = "alpha_matted_rgb" if policy == "alpha_gated" else "foreground_rgb_zero_outside"
    return {
        "target_space": target,
        "loss_scope": "foreground_matted",
        "gate_scope": "foreground_matted",
        "profile_scope": "foreground_matted",
    }


def _template_ledger(
    unit_ids: Sequence[str],
    *,
    geometry: int,
    appearance: int,
    mass: int = 0,
    opacity: int = 0,
    background: int = 0,
    alpha: int = 0,
    metadata: int = 96,
) -> dict[str, Any]:
    fixed = {unit_id: background + alpha + metadata for unit_id in unit_ids}
    components = [
        {
            "name": "geometry",
            "bytes_per_row": geometry,
            "fixed_bytes_by_unit": {unit: 0 for unit in unit_ids},
        },
        {
            "name": "appearance",
            "bytes_per_row": appearance,
            "fixed_bytes_by_unit": {unit: 0 for unit in unit_ids},
        },
        {
            "name": "structural_mass",
            "bytes_per_row": mass,
            "fixed_bytes_by_unit": {unit: 0 for unit in unit_ids},
        },
        {
            "name": "factorized_opacity",
            "bytes_per_row": opacity,
            "fixed_bytes_by_unit": {unit: 0 for unit in unit_ids},
        },
        {
            "name": "background",
            "bytes_per_row": 0,
            "fixed_bytes_by_unit": {unit: background for unit in unit_ids},
        },
        {
            "name": "packed_alpha",
            "bytes_per_row": 0,
            "fixed_bytes_by_unit": {unit: alpha for unit in unit_ids},
        },
        {
            "name": "metadata",
            "bytes_per_row": 0,
            "fixed_bytes_by_unit": {unit: metadata for unit in unit_ids},
        },
    ]
    return {
        "bytes_per_row": geometry + appearance + mass + opacity,
        "fixed_bytes_by_unit": fixed,
        "components": components,
    }


def protocol_template() -> dict[str, Any]:
    def artifact() -> dict[str, str]:
        return {"path": "REPLACE_WITH_FILE"}

    dev_specs = [
        ("dev_capture_a", "capture_a", "frame_a"),
        ("dev_capture_b", "capture_b", "frame_b"),
        ("dev_capture_c", "capture_c", "frame_c"),
    ]
    confirmation_specs = [
        ("confirmation_capture_d", "capture_d", "frame_d"),
        ("confirmation_capture_e", "capture_e", "frame_e"),
        ("confirmation_capture_f", "capture_f", "frame_f"),
    ]

    def unit(unit_id: str, capture: str, frame: str) -> dict[str, Any]:
        return {
            "id": unit_id,
            "capture_group": capture,
            "frame_id": frame,
            "metadata_selector": {"rule": "REPLACE_WITH_METADATA_SELECTION", "value": frame},
            "pixels": artifact(),
            "mask": artifact(),
            "camera": artifact(),
            "prepared_target": artifact(),
            "views": [f"{frame}_view"],
        }

    development = [unit(*spec) for spec in dev_specs]
    confirmation = [unit(*spec) for spec in confirmation_specs]
    unit_ids = [value["id"] for value in development + confirmation]
    policies = ["alpha_gated", "hard_contained"]
    policy_contracts = {policy: _template_policy_contract(policy) for policy in policies}

    def ledgers(
        variants: Sequence[str], *, appearance: int, mass: int = 0, opacity: int = 0
    ) -> dict[str, Any]:
        result = {}
        for variant in variants:
            result[variant] = {}
            for policy in policies:
                result[variant][policy] = _template_ledger(
                    unit_ids,
                    geometry=20,
                    appearance=appearance,
                    mass=mass,
                    opacity=opacity,
                    background=12 if variant == "counted_dc_signed_bounded" else 0,
                    alpha=512 if policy == "alpha_gated" else 0,
                )
        return result

    def semantics(family: str) -> dict[str, str]:
        family_values = {
            "incumbent_factorized_additive": (
                "legacy_gaussian_field",
                "1",
                "additive_rgb_peak_one_v1",
                "factorized_color_times_opacity",
                "absent_not_derived",
                "legacy_gaussian_field_npz_v1",
            ),
            "direct_additive": (
                "observation_field_v2",
                "2.0.0",
                "additive_rgb_peak_one_v1",
                "direct_rgb_coeff",
                "absent_not_derived",
                "observation_field_v2_lossless_npz_v1",
            ),
            "normalized_plain": (
                "legacy_gaussian_field",
                "1",
                "normalized_weighted_sum_v1",
                "normalized_color",
                "normalizer_not_mass",
                "legacy_gaussian_field_npz_v1",
            ),
            "normalized_maintained": (
                "legacy_gaussian_field",
                "1",
                "normalized_weighted_sum_v1",
                "normalized_color",
                "normalizer_not_mass",
                "legacy_gaussian_field_npz_v1",
            ),
        }
        contract, version, equation, authority, mass_value, payload_format = family_values[family]
        return {
            "contract": contract,
            "schema_version": version,
            "renderer_equation": equation,
            "coefficient_authority": authority,
            "structural_mass": mass_value,
            "payload_format": payload_format,
        }

    additive_variants = ["zero_dc_nonnegative", "counted_dc_signed_bounded"]
    arms = [
        {
            "id": "incumbent_additive",
            "family": "incumbent_factorized_additive",
            "execution_kind": "matched",
            "semantics": semantics("incumbent_factorized_additive"),
            "coefficient_variants": ["not_applicable"],
            "alpha_policies": list(policies),
            "policy_contracts": copy.deepcopy(policy_contracts),
            "profile": artifact(),
            "loss_contract": artifact(),
            "gate_contract": artifact(),
            "raw_byte_ledgers": ledgers(["not_applicable"], appearance=12, opacity=4),
        },
        {
            "id": "direct_additive",
            "family": "direct_additive",
            "execution_kind": "matched",
            "semantics": semantics("direct_additive"),
            "coefficient_variants": additive_variants,
            "alpha_policies": list(policies),
            "policy_contracts": copy.deepcopy(policy_contracts),
            "profile": artifact(),
            "loss_contract": artifact(),
            "gate_contract": artifact(),
            "raw_byte_ledgers": ledgers(additive_variants, appearance=12),
        },
        {
            "id": "normalized_plain",
            "family": "normalized_plain",
            "execution_kind": "matched",
            "semantics": semantics("normalized_plain"),
            "coefficient_variants": ["not_applicable"],
            "alpha_policies": list(policies),
            "policy_contracts": copy.deepcopy(policy_contracts),
            "profile": artifact(),
            "loss_contract": artifact(),
            "gate_contract": artifact(),
            "raw_byte_ledgers": ledgers(["not_applicable"], appearance=12, opacity=4),
        },
        {
            "id": "normalized_maintained",
            "family": "normalized_maintained",
            "execution_kind": "maintained_reference",
            "semantics": semantics("normalized_maintained"),
            "coefficient_variants": ["not_applicable"],
            "alpha_policies": list(policies),
            "policy_contracts": copy.deepcopy(policy_contracts),
            "profile": artifact(),
            "loss_contract": artifact(),
            "gate_contract": artifact(),
            "raw_byte_ledgers": ledgers(["not_applicable"], appearance=12, opacity=4),
        },
    ]
    for arm in arms:
        arm["semantic_sha256"] = _digest(arm["semantics"])
    lanes = [
        {"id": "fixed_rows", "kind": "fixed_rows", "value": 1024},
        {"id": "equal_raw_bytes", "kind": "equal_canonical_raw_bytes", "value": 50000},
    ]
    seeds = [20001, 20002, 20003]
    prefixes_by_unit: dict[str, set[int]] = {unit_id: set() for unit_id in unit_ids}
    for arm in arms:
        if arm["execution_kind"] != "matched":
            continue
        for variant in arm["coefficient_variants"]:
            for policy in arm["alpha_policies"]:
                for lane in lanes:
                    for unit_id in unit_ids:
                        count, _ = _row_budget(arm, lane, unit_id, variant, policy)
                        prefixes_by_unit[unit_id].add(count)
    initial_geometry = [
        {
            "unit_id": unit_id,
            "seed": seed,
            "bank": artifact(),
            "prefixes": [
                {
                    "row_count": count,
                    "sha256": hashlib.sha256(f"{unit_id}:{seed}:{count}".encode()).hexdigest(),
                }
                for count in sorted(prefixes_by_unit[unit_id])
            ],
        }
        for unit_id in unit_ids
        for seed in seeds
    ]
    metrics = [
        {
            "name": "foreground_psnr",
            "direction": "higher",
            "role": "quality",
            "phases": list(PHASES),
            "availability": "required",
        },
        {
            "name": "boundary_psnr",
            "direction": "higher",
            "role": "boundary",
            "phases": list(PHASES),
            "availability": "required",
        },
        {
            "name": "ms_ssim",
            "direction": "higher",
            "role": "quality",
            "phases": list(PHASES),
            "availability": "required",
        },
        {
            "name": "lpips",
            "direction": "lower",
            "role": "quality",
            "phases": list(PHASES),
            "availability": "required",
        },
        {
            "name": "alpha_mae",
            "direction": "lower",
            "role": "guard",
            "phases": list(PHASES),
            "availability": "required",
        },
        {
            "name": "outside_rgb_mae",
            "direction": "lower",
            "role": "guard",
            "phases": list(PHASES),
            "availability": "required",
        },
        {
            "name": "stage1_objective",
            "direction": "higher",
            "role": "downstream",
            "phases": ["development", "confirmation"],
            "availability": "unavailable",
        },
        {
            "name": "downstream_response",
            "direction": "higher",
            "role": "downstream",
            "phases": ["development", "confirmation"],
            "availability": "required",
        },
    ]
    return {
        "schema": PROTOCOL_SCHEMA,
        "task_id": "BENCH-020",
        "state": "draft",
        "driver": "replace-driver",
        "claim_scope": "general",
        "repositories": [{"name": "structsplat", "root": ".", "environment": artifact()}],
        "datasets": {"development": development, "confirmation": confirmation},
        "seeds": seeds,
        "seed_stream_sha256": {
            str(seed): hashlib.sha256(f"seed-stream:{seed}".encode()).hexdigest() for seed in seeds
        },
        "initial_geometry": initial_geometry,
        "structural_target": {
            "status": "unavailable",
            "metric": None,
            "direction": None,
            "definition": None,
        },
        "arms": arms,
        "lanes": lanes,
        "phases": {
            "coefficient_screen": {
                "work_contract": artifact(),
                "iterations": 250,
                "wall_seconds": 300.0,
                "renderer_call_cap": 600,
                "evaluation_every": 25,
                "checkpoint_rule": "best_foreground_psnr_then_terminal",
                "screen_arm_id": "direct_additive",
                "lane_id": "fixed_rows",
                "alpha_policy": "alpha_gated",
            },
            "development": {
                "work_contract": artifact(),
                "iterations": 1000,
                "wall_seconds": 1200.0,
                "renderer_call_cap": 2400,
                "evaluation_every": 50,
                "checkpoint_rule": "best_foreground_psnr_at_terminal_rows",
            },
            "confirmation": {
                "work_contract": artifact(),
                "iterations": 1000,
                "wall_seconds": 1200.0,
                "renderer_call_cap": 2400,
                "evaluation_every": 50,
                "checkpoint_rule": "best_foreground_psnr_at_terminal_rows",
            },
        },
        "metrics": metrics,
        "convergence": {
            "metric": "foreground_psnr",
            "targets": [30.0, 35.0],
            "primary_target": 35.0,
            "target_rule": "first_observed_at_or_above",
            "unreached_policy": "null_with_right_censor_horizon",
            "auc_axis": "wall_seconds",
            "auc_horizon": "frozen_phase_wall_seconds_hold_last",
            "auc_interpolation": "linear_between_observations",
        },
        "gates": {
            "missing_policy": "fail_closed",
            "domain": {
                "quality_metric": "foreground_psnr",
                "max_mean_degradation": 0.05,
                "feasibility_guards": [{"metric": "alpha_mae", "op": "<=", "threshold": 0.01}],
                "priority": [
                    {"metric": "foreground_psnr", "direction": "higher"},
                    {"metric": "lpips", "direction": "lower"},
                ],
                "max_finalists": 2,
            },
            "killing": {
                "controls": ["incumbent_additive", "normalized_plain"],
                "candidate_families": ["direct_additive"],
                "lane_ids": ["fixed_rows", "equal_raw_bytes"],
                "quality_noninferiority": [
                    {"metric": "foreground_psnr", "margin": 0.2},
                    {"metric": "lpips", "margin": 0.01},
                ],
                "downstream_favorable": {"metric": "downstream_response", "margin": 0.0},
                "absolute_guards": [
                    {"metric": "alpha_mae", "op": "<=", "threshold": 0.01},
                    {"metric": "outside_rgb_mae", "op": "<=", "threshold": 0.01},
                ],
                "pareto_metrics": ["foreground_psnr", "lpips", "downstream_response"],
                "bootstrap_replicates": 10000,
                "bootstrap_seed": 20020,
                "minimum_capture_groups": 3,
                "selection_rule": "single_nondominated_survivor",
            },
        },
        "aa_replay": {
            "unit_id": "dev_capture_a",
            "arm_id": "direct_additive",
            "coefficient_variant": "zero_dc_nonnegative",
            "alpha_policy": "alpha_gated",
            "lane_id": "fixed_rows",
            "seed": seeds[0],
            "metric_abs_tolerance": {
                "foreground_psnr": 0.0,
                "boundary_psnr": 0.0,
                "alpha_mae": 0.0,
            },
        },
        "execution": {
            "environment": artifact(),
            "downstream_protocol": artifact(),
            "working_directory": "/REPLACE_WITH_CLEAN_STRUCTSPLAT_ROOT",
            "commands": {
                phase: ["python", "REPLACE_WITH_EXECUTOR", phase, "--cell", "{cell_json}"]
                for phase in PHASES
            },
            "outcome_roots": {
                phase: f"/REPLACE_WITH_EMPTY_{phase.upper()}_ROOT" for phase in PHASES
            },
        },
    }


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read {label} from {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _stdout(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("template")
    template.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-review")
    prepare.add_argument("--draft", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    review = subparsers.add_parser("review-template")
    review.add_argument("--protocol", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--reviewed", type=Path, required=True)
    finalize.add_argument("--review", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--require-frozen", action="store_true")

    plan = subparsers.add_parser("plan")
    plan.add_argument("--protocol", type=Path, required=True)
    plan.add_argument("--phase", choices=PHASES, required=True)
    plan.add_argument("--domain-lock", type=Path)
    plan.add_argument("--confirmation-lock", type=Path)
    plan.add_argument("--output", type=Path)

    coefficient = subparsers.add_parser("analyze-coefficient")
    coefficient.add_argument("--protocol", type=Path, required=True)
    coefficient.add_argument("--rows", type=Path, required=True)
    coefficient.add_argument("--output", type=Path, required=True)
    development = subparsers.add_parser("analyze-development")
    development.add_argument("--protocol", type=Path, required=True)
    development.add_argument("--domain-lock", type=Path, required=True)
    development.add_argument("--rows", type=Path, required=True)
    development.add_argument("--output", type=Path, required=True)
    development_review = subparsers.add_parser("development-review-template")
    development_review.add_argument("--protocol", type=Path, required=True)
    development_review.add_argument("--analysis", type=Path, required=True)
    development_review.add_argument("--output", type=Path, required=True)
    lock = subparsers.add_parser("lock-confirmation")
    lock.add_argument("--protocol", type=Path, required=True)
    lock.add_argument("--analysis", type=Path, required=True)
    lock.add_argument("--review", type=Path, required=True)
    lock.add_argument("--output", type=Path, required=True)
    confirmation = subparsers.add_parser("analyze-confirmation")
    confirmation.add_argument("--protocol", type=Path, required=True)
    confirmation.add_argument("--domain-lock", type=Path, required=True)
    confirmation.add_argument("--confirmation-lock", type=Path, required=True)
    confirmation.add_argument("--rows", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("results-audit-template")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--analysis", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    report = subparsers.add_parser("report")
    report.add_argument("--protocol", type=Path, required=True)
    report.add_argument("--coefficient-rows", type=Path, required=True)
    report.add_argument("--domain-lock", type=Path, required=True)
    report.add_argument("--development-rows", type=Path, required=True)
    report.add_argument("--development-analysis", type=Path, required=True)
    report.add_argument("--confirmation-lock", type=Path)
    report.add_argument("--confirmation-rows", type=Path)
    report.add_argument("--confirmation-analysis", type=Path)
    report.add_argument("--results-audit", type=Path)
    report.add_argument("--outdir", type=Path, required=True)
    check = subparsers.add_parser("check-report")
    check.add_argument("report", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "template":
        _write_json(args.output, protocol_template())
        return 0
    if args.command == "prepare-review":
        value = prepare_review(_load_object(args.draft, "draft"), base=args.draft.parent)
        _write_json(args.output, value)
        return 0
    if args.command == "review-template":
        protocol = _load_object(args.protocol, "protocol")
        _write_json(args.output, review_template(protocol, base=args.protocol.parent))
        return 0
    if args.command == "finalize":
        value = finalize_protocol(
            _load_object(args.reviewed, "review-ready protocol"),
            args.review,
            base=args.reviewed.parent,
        )
        _write_json(args.output, value)
        return 0
    if args.command == "validate":
        protocol = _load_object(args.protocol, "protocol")
        validate_protocol(protocol, base=args.protocol.parent, require_frozen=args.require_frozen)
        print(
            f"BENCH-020 protocol OK: {protocol.get('protocol_sha256', protocol['design_sha256'])}"
        )
        return 0
    if args.command == "plan":
        protocol = _load_object(args.protocol, "protocol")
        domain = None if args.domain_lock is None else _load_object(args.domain_lock, "domain lock")
        lock = (
            None
            if args.confirmation_lock is None
            else _load_object(args.confirmation_lock, "confirmation lock")
        )
        cells = expected_cells(protocol, args.phase, domain_lock=domain, confirmation_lock=lock)
        payload = {
            "protocol_sha256": protocol["protocol_sha256"],
            "phase": args.phase,
            "outcome_root": protocol["execution"]["outcome_roots"][args.phase],
            "command": protocol["execution"]["commands"][args.phase],
            "cells": cells,
        }
        if args.output is None:
            _stdout(payload)
        else:
            _write_json(args.output, payload)
        return 0
    if args.command == "analyze-coefficient":
        protocol = _load_object(args.protocol, "protocol")
        rows = load_rows(args.rows)
        value = analyze_coefficient_screen(protocol, rows, base=args.rows.parent)
        _write_json(args.output, value)
        return 0
    if args.command == "analyze-development":
        protocol = _load_object(args.protocol, "protocol")
        domain = _load_object(args.domain_lock, "domain lock")
        rows = load_rows(args.rows)
        value = analyze_factorial(
            protocol,
            rows,
            "development",
            base=args.rows.parent,
            domain_lock=domain,
        )
        _write_json(args.output, value)
        return 0
    if args.command == "development-review-template":
        protocol = _load_object(args.protocol, "protocol")
        analysis = _load_object(args.analysis, "development analysis")
        _write_json(args.output, development_review_template(protocol, analysis))
        return 0
    if args.command == "lock-confirmation":
        protocol = _load_object(args.protocol, "protocol")
        analysis = _load_object(args.analysis, "development analysis")
        value = lock_confirmation(protocol, analysis, args.review, base=args.protocol.parent)
        _write_json(args.output, value)
        return 0
    if args.command == "analyze-confirmation":
        protocol = _load_object(args.protocol, "protocol")
        domain = _load_object(args.domain_lock, "domain lock")
        lock = _load_object(args.confirmation_lock, "confirmation lock")
        rows = load_rows(args.rows)
        value = analyze_factorial(
            protocol,
            rows,
            "confirmation",
            base=args.rows.parent,
            domain_lock=domain,
            confirmation_lock=lock,
        )
        _write_json(args.output, value)
        return 0
    if args.command == "results-audit-template":
        protocol = _load_object(args.protocol, "protocol")
        analysis = _load_object(args.analysis, "confirmation analysis")
        _write_json(args.output, results_audit_template(protocol, analysis))
        return 0
    if args.command == "report":
        protocol = _load_object(args.protocol, "protocol")
        coefficient_rows = load_rows(args.coefficient_rows)
        domain = _load_object(args.domain_lock, "domain lock")
        development_rows = load_rows(args.development_rows)
        development_analysis = _load_object(args.development_analysis, "development analysis")
        lock = (
            None
            if args.confirmation_lock is None
            else _load_object(args.confirmation_lock, "confirmation lock")
        )
        confirmation_rows = (
            [] if args.confirmation_rows is None else load_rows(args.confirmation_rows)
        )
        confirmation_analysis = (
            None
            if args.confirmation_analysis is None
            else _load_object(args.confirmation_analysis, "confirmation analysis")
        )
        manifest = write_report(
            protocol,
            coefficient_rows,
            domain,
            development_rows,
            development_analysis,
            args.outdir,
            row_base=args.coefficient_rows.parent,
            confirmation_lock=lock,
            confirmation_rows=confirmation_rows,
            confirmation_analysis=confirmation_analysis,
            results_audit_path=args.results_audit,
            command=" ".join(sys.argv),
        )
        _stdout(
            {
                "report": str(args.outdir.resolve()),
                "decision": manifest["decision"],
                "claim_ready": manifest["claim_ready"],
            }
        )
        return 0
    if args.command == "check-report":
        problems = validate_report_bundle(args.report)
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print("BENCH-020 report: OK")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_ANALYSIS_SCHEMA",
    "CONFIRMATION_LOCK_SCHEMA",
    "DEVELOPMENT_ANALYSIS_SCHEMA",
    "DOMAIN_LOCK_SCHEMA",
    "FIELD_MANIFEST_SCHEMA",
    "HISTORY_SCHEMA",
    "METRICS_SCHEMA",
    "PHASES",
    "PROTOCOL_SCHEMA",
    "ProtocolError",
    "REPORT_SCHEMA",
    "RESULTS_AUDIT_SCHEMA",
    "ROW_SCHEMA",
    "aa_replay_result",
    "analyze_coefficient_screen",
    "analyze_factorial",
    "canonical_json",
    "development_review_template",
    "expected_cells",
    "finalize_protocol",
    "load_rows",
    "lock_confirmation",
    "main",
    "matched_input_invariants",
    "prepare_review",
    "protocol_template",
    "results_audit_template",
    "review_template",
    "sha256_file",
    "summarize_convergence",
    "validate_protocol",
    "validate_report_bundle",
    "validate_result_rows",
    "write_report",
]
